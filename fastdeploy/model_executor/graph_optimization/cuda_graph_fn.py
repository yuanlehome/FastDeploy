"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
函数级 CUDA graph 装饰器。

设计思路参见 cuda_graph_fn_design.md：
- FastDeploy 现有的 CudaGraphPiecewiseBackend 面向 paddle.nn.Layer，
  依赖 FDConfig / ForwardMeta 等重量级对象。
- 本模块提供 @cuda_graph_fn 装饰器，适用于任意独立函数，
  只要函数的所有操作均为 GPU kernel，即可透明地完成
  warmup → capture → replay 三段式 CUDA graph 加速。

关键约束（来自对 cudagraph_piecewise_backend.py 的研究）：
1. CUDA graph 只能捕获 GPU kernel，禁止在 capture 区间内执行
   .cpu() / .numpy() / .item() 等同步 D2H 操作。
2. capture 与 replay 时输入 Tensor 地址必须完全一致——
   装饰器通过维护 input_buffers（固定地址 pre-allocated Tensor）
   并在每次调用前将真实输入 paddle.assign() 进去来保证这一点。
3. capture_begin 前须调用 paddle.device.synchronize()。
4. 通过 _share_buffer_to 将输出 Tensor 地址绑定到持久 output_buffer，
   replay 结果直接从 output_buffer 读取。
5. batch_size 通过向上 padding 到预定义的 capture_sizes 来命中缓存，
   超出最大 capture_size 的调用 fallback 到 eager 执行。
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import paddle

from fastdeploy.utils import get_logger

logger = get_logger("cuda_graph_fn", "cuda_graph_fn.log")

# ---------------------------------------------------------------------------
# 平台检测：非 CUDA 平台时装饰器退化为 passthrough
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = paddle.is_compiled_with_cuda()

if _CUDA_AVAILABLE:
    from paddle.device.cuda import graphs as _cuda_graphs

    try:
        from paddle.base.core import CUDAGraph as _CoreCUDAGraph

        _HAS_POOL_API = True
    except Exception:
        _HAS_POOL_API = False
else:
    _cuda_graphs = None
    _HAS_POOL_API = False


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------


@dataclass
class _FnSizeEntry:
    """
    记录单个 capture_size 的捕获状态。

    与 ConcreteSizeEntry（cudagraph_piecewise_backend.py:38）对应，
    但面向普通函数而非 paddle.nn.Layer.forward。
    """

    real_shape: int
    # warmup 完成次数
    num_finished_warmup: int = 0
    # 是否已完成 capture
    captured: bool = False
    # capture 得到的 CUDAGraph 对象
    cuda_graph: Optional[Any] = None
    # 固定地址输入 buffer（按 real_shape padding 后的尺寸）
    input_buffers: Optional[List[paddle.Tensor]] = None
    # 与 capture 输出地址绑定的持久 buffer
    output_buffers: List[Optional[paddle.Tensor]] = field(default_factory=list)
    # 记录输出是否为单个 Tensor（用于还原返回格式）
    output_is_single: bool = False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_real_to_padded(capture_sizes: List[int]) -> Dict[int, int]:
    """
    构建 real_batch_size → padded_capture_size 的映射。

    算法与 config.py init_with_cudagrpah_size() 一致：
    对每个 real_size，找最小的 capture_size >= real_size。
    空列表时返回空映射（所有调用均 fallback 到 eager）。
    """
    if not capture_sizes:
        return {}
    sorted_sizes = sorted(capture_sizes)
    mapping: Dict[int, int] = {}
    for real in range(1, sorted_sizes[-1] + 1):
        for s in sorted_sizes:
            if s >= real:
                mapping[real] = s
                break
    return mapping


def _get_tensor_args(args: Tuple, kwargs: Dict) -> List[paddle.Tensor]:
    """提取所有 paddle.Tensor 类型的参数（保持顺序）。"""
    tensors = []
    for a in args:
        if isinstance(a, paddle.Tensor):
            tensors.append(a)
    for v in kwargs.values():
        if isinstance(v, paddle.Tensor):
            tensors.append(v)
    return tensors


def _get_batch_size(args: Tuple, kwargs: Dict, size_from: Optional[int]) -> int:
    """
    从调用参数中提取 batch_size。
    size_from：指定用第几个 Tensor 参数的 shape[0]；默认使用第一个 Tensor。
    """
    tensors = _get_tensor_args(args, kwargs)
    if not tensors:
        raise ValueError("[cuda_graph_fn] No paddle.Tensor found in function arguments.")
    idx = size_from if size_from is not None else 0
    return tensors[idx].shape[0]


def _alloc_input_buffers(args: Tuple, kwargs: Dict, padded_size: int, size_from: Optional[int]) -> List[paddle.Tensor]:
    """
    为每个 Tensor 参数分配固定地址 input buffer。

    所有 Tensor 参数的第0维一律 padding 到 padded_size。
    这样无论 real_batch_size 如何变化，buffer 形状始终固定，
    与 FastDeploy padding_cudagraph_inputs() 的策略一致。

    非 Tensor 参数不需要 buffer（直接在调用时传入原始值）。
    """
    buffers: List[paddle.Tensor] = []

    for a in args:
        if isinstance(a, paddle.Tensor):
            padded_shape = list(a.shape)
            padded_shape[0] = padded_size
            buf = paddle.zeros(padded_shape, dtype=a.dtype)
            buffers.append(buf)

    for v in kwargs.values():
        if isinstance(v, paddle.Tensor):
            padded_shape = list(v.shape)
            padded_shape[0] = padded_size
            buf = paddle.zeros(padded_shape, dtype=v.dtype)
            buffers.append(buf)

    return buffers


def _copy_inputs_to_buffers(
    args: Tuple,
    kwargs: Dict,
    buffers: List[paddle.Tensor],
    real_batch_size: int,
    size_from: Optional[int],
) -> None:
    """
    将真实输入 Tensor 拷贝到固定地址 input buffer。

    所有 Tensor 参数的 buffer 第0维均为 padded_size，
    只拷贝前 real_batch_size 行，其余行保持为 0（padding 区间不影响输出结果，
    因为外层函数仅读取 [:real_bsz] 的输出）。

    使用 paddle.assign() 而非 copy_()，与 FastDeploy 内部惯例一致。
    """
    all_tensors = [a for a in args if isinstance(a, paddle.Tensor)] + [
        v for v in kwargs.values() if isinstance(v, paddle.Tensor)
    ]

    for i, real_t in enumerate(all_tensors):
        buf = buffers[i]
        real_rows = real_t.shape[0]
        if buf.shape[0] > real_rows:
            # 只更新真实行，padding 行保持为 0
            paddle.assign(real_t, buf[:real_rows])
        else:
            paddle.assign(real_t, buf)


def _rebuild_args_with_buffers(
    args: Tuple,
    kwargs: Dict,
    buffers: List[paddle.Tensor],
) -> Tuple[Tuple, Dict]:
    """
    用 input_buffers 替换原始 args/kwargs 中的 Tensor，保留非 Tensor 原样。
    返回新的 (args, kwargs)。
    """
    buf_iter = iter(buffers)
    new_args = tuple(next(buf_iter) if isinstance(a, paddle.Tensor) else a for a in args)
    new_kwargs = {k: next(buf_iter) if isinstance(v, paddle.Tensor) else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def _input_buffers_compatible(args, kwargs, buffers):
    """
    Check whether existing input_buffers match the current call's non-batch dimensions.

    Buffer shape[0] (padded_size) is allowed to differ from real tensor shape[0].
    All other dimensions (vocab_size, hidden_dim, etc.) and dtype must be identical;
    otherwise the captured graph would operate on the wrong memory layout.
    """
    all_tensors = [a for a in args if isinstance(a, paddle.Tensor)] + [
        v for v in kwargs.values() if isinstance(v, paddle.Tensor)
    ]
    if len(all_tensors) != len(buffers):
        return False
    for real_t, buf in zip(all_tensors, buffers):
        if real_t.dtype != buf.dtype:
            return False
        if list(real_t.shape[1:]) != list(buf.shape[1:]):
            return False
    return True


# ---------------------------------------------------------------------------
# 核心后端类
# ---------------------------------------------------------------------------


class CudaGraphFnBackend:
    """
    函数级 CUDA graph 后端。

    生命周期：
        warmup(N次) → capture(1次) → replay(所有后续调用)

    与 CudaGraphPiecewiseBackend 的区别：
    - 不依赖 FDConfig / ForwardMeta
    - 支持任意签名的普通函数（只要返回值为 Tensor 或 List[Tensor]）
    - batch_size 从函数的第 size_from 个 Tensor 参数的 shape[0] 推断
    """

    def __init__(
        self,
        fn: Callable,
        capture_sizes: List[int],
        num_warmups: int = 2,
        size_from: Optional[int] = None,
        use_unique_memory_pool: bool = True,
    ):
        self.fn = fn
        self.capture_sizes = sorted(capture_sizes)
        self.num_warmups = num_warmups
        self.size_from = size_from

        # real_batch_size → padded_capture_size 映射
        self.real_to_padded = _build_real_to_padded(capture_sizes)

        # 每个 capture_size 对应一个 _FnSizeEntry
        self.entries: Dict[int, _FnSizeEntry] = {s: _FnSizeEntry(real_shape=s) for s in self.capture_sizes}

        # 共享 memory pool（与 CudaGraphPiecewiseBackend 相同策略）
        self.pool_id = None
        if _CUDA_AVAILABLE and use_unique_memory_pool and _HAS_POOL_API:
            try:
                self.pool_id = _CoreCUDAGraph.gen_new_memory_pool_id()
            except Exception as e:
                logger.warning(f"[cuda_graph_fn] Failed to create memory pool: {e}")

        logger.info(
            f"[cuda_graph_fn] Initialized for '{fn.__name__}', "
            f"capture_sizes={self.capture_sizes}, num_warmups={num_warmups}"
        )

    def __call__(self, *args, **kwargs):
        # --- 非 CUDA 平台：直接 passthrough ---
        if not _CUDA_AVAILABLE:
            return self.fn(*args, **kwargs)

        # --- 获取 batch_size 并查找对应的 padded_size ---
        try:
            batch_size = _get_batch_size(args, kwargs, self.size_from)
        except ValueError:
            return self.fn(*args, **kwargs)

        padded_size = self.real_to_padded.get(batch_size)
        if padded_size is None:
            # 超出 capture 范围（或 capture_sizes 为空），fallback 到 eager
            return self.fn(*args, **kwargs)

        entry = self.entries[padded_size]

        # --- 分配 input_buffers（首次调用时或非 batch 维度变化时重新分配）---
        need_alloc = entry.input_buffers is None
        if not need_alloc and not _input_buffers_compatible(args, kwargs, entry.input_buffers):
            need_alloc = True
            if entry.captured:
                entry.captured = False
                entry.cuda_graph = None
                entry.output_buffers = []
                entry.num_finished_warmup = 0
                logger.warning(
                    f"[cuda_graph_fn] Input shapes changed for padded_size={padded_size}, "
                    f"fn='{self.fn.__name__}' — invalidating capture and re-allocating buffers."
                )

        if need_alloc:
            entry.input_buffers = _alloc_input_buffers(args, kwargs, padded_size, self.size_from)
            logger.debug(
                f"[cuda_graph_fn] Allocated input_buffers for padded_size={padded_size}, " f"fn='{self.fn.__name__}'"
            )

        # --- 将真实输入拷入固定地址 buffer ---
        _copy_inputs_to_buffers(args, kwargs, entry.input_buffers, batch_size, self.size_from)
        buf_args, buf_kwargs = _rebuild_args_with_buffers(args, kwargs, entry.input_buffers)

        # --- Warmup 阶段 ---
        if not entry.captured and entry.num_finished_warmup < self.num_warmups:
            entry.num_finished_warmup += 1
            logger.info(
                f"[cuda_graph_fn] Warmup {entry.num_finished_warmup}/{self.num_warmups} "
                f"for padded_size={padded_size}, fn='{self.fn.__name__}'"
            )
            return self.fn(*buf_args, **buf_kwargs)

        # --- Capture 阶段 ---
        if not entry.captured:
            paddle.device.synchronize()

            # 第一步：在 capture 前执行一次 eager 调用，获取输出形状和 dtype，
            # 预分配 output_buffers（普通 GPU 内存，不来自 pool）。
            # 这样 capture 时可以用 paddle.assign 直接写入 output_buffers，
            # 避免 pool 内存别名（aliasing）导致 replay 结果错误。
            pre_outputs = self.fn(*buf_args, **buf_kwargs)
            paddle.device.synchronize()

            single_output = isinstance(pre_outputs, paddle.Tensor)
            output_list = [pre_outputs] if single_output else list(pre_outputs)

            entry.output_buffers = []
            for out in output_list:
                if out is not None and isinstance(out, paddle.Tensor):
                    entry.output_buffers.append(paddle.zeros_like(out))
                else:
                    entry.output_buffers.append(out)

            entry.output_is_single = single_output

            # 第二步：正式 capture。函数内部的中间 Tensor 使用 pool 内存，
            # 最终通过 paddle.assign 写入 output_buffers（普通 GPU 内存）。
            # assign 操作被记录在 CUDA graph 中，replay 时直接写入固定地址。
            new_graph = _cuda_graphs.CUDAGraph(pool_id=self.pool_id)
            new_graph.capture_begin()
            cap_outputs = self.fn(*buf_args, **buf_kwargs)
            cap_list = [cap_outputs] if single_output else list(cap_outputs)
            for cap_out, out_buf in zip(cap_list, entry.output_buffers):
                if cap_out is not None and isinstance(cap_out, paddle.Tensor) and out_buf is not None:
                    paddle.assign(cap_out, out_buf)
            new_graph.capture_end()

            entry.cuda_graph = new_graph
            entry.captured = True

            paddle.device.synchronize()

            logger.info(
                f"[cuda_graph_fn] CUDAGraph captured for padded_size={padded_size}, " f"fn='{self.fn.__name__}'"
            )

        # --- Replay 阶段 ---
        entry.cuda_graph.replay()
        logger.debug(f"[cuda_graph_fn] CUDAGraph replayed for padded_size={padded_size}, " f"fn='{self.fn.__name__}'")
        if entry.output_is_single:
            return entry.output_buffers[0]
        return entry.output_buffers


# ---------------------------------------------------------------------------
# 公共装饰器 API
# ---------------------------------------------------------------------------


def cuda_graph_fn(
    capture_sizes: List[int],
    num_warmups: int = 2,
    size_from: Optional[int] = None,
    use_unique_memory_pool: bool = True,
):
    """
    函数级 CUDA graph 装饰器。

    将任意只含 GPU kernel 操作的函数透明地转为
    "warmup → capture → replay" 三段式 CUDA graph 加速。

    Args:
        capture_sizes: 预定义的 batch size 列表（如 [1,2,4,8,16,32]）。
            运行时 batch_size 会向上对齐到最近的 capture_size。
            超出最大 capture_size 的调用会 fallback 到 eager 执行。
        num_warmups: 每个 capture_size 在正式 capture 前执行的热身次数（默认 2）。
        size_from: 从哪个 Tensor 参数（按位置，从 0 计数）的 shape[0] 读取 batch_size。
            默认为 None，使用第一个 Tensor 参数。
        use_unique_memory_pool: 是否为该函数分配独立 CUDA memory pool（默认 True）。

    Example::

        @cuda_graph_fn(capture_sizes=[1, 2, 4, 8, 16, 32], num_warmups=2)
        def my_gpu_kernel(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
            return paddle.matmul(a, b)

    Important:
        函数内部不得包含 D2H 操作（.cpu()/.numpy()/.item()）。
        如果原函数混合了 GPU 计算和 D2H 操作，请先将其拆分：
        GPU 计算部分用本装饰器，D2H/CPU 后处理保持 eager 执行。

    Non-CUDA platforms:
        在非 CUDA 平台上，装饰器自动退化为直接调用原函数（passthrough）。
    """

    def decorator(fn: Callable) -> Callable:
        backend = CudaGraphFnBackend(
            fn=fn,
            capture_sizes=capture_sizes,
            num_warmups=num_warmups,
            size_from=size_from,
            use_unique_memory_pool=use_unique_memory_pool,
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return backend(*args, **kwargs)

        # 暴露 backend 供外部检查/测试
        wrapper._cuda_graph_backend = backend
        return wrapper

    return decorator
