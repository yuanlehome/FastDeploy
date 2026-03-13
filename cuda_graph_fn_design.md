# CUDA Graph Function Decorator 设计文档

> 作者：研发 Agent
> 日期：2026-03-13
> 目标：实现一个函数级别的 CUDA graph 装饰器 `@cuda_graph_fn`，让任意独立函数（非 `paddle.nn.Layer`）也能享受 CUDA graph 加速，并首先应用在 `_compute_sampling_mask` 上。

---

## 1. 背景与动机

### 1.1 FastDeploy 现有 CUDA graph 架构

FastDeploy 的 CUDA graph 支持围绕 **模型级别** 封装，核心链路为：

```
@support_graph_optimization (decorator.py)
    └── GraphOptWrapper
        └── GraphOptBackend (graph_optimization_backend.py)
            └── CudaGraphPiecewiseBackend (cudagraph_piecewise_backend.py)
                └── paddle.device.cuda.graphs.CUDAGraph
```

**关键设计模式（从源码提炼）：**

| 模式 | 来源文件 | 说明 |
|---|---|---|
| `ConcreteSizeEntry` | `cudagraph_piecewise_backend.py:38` | 每个 batch size 对应一个捕获状态记录，含 warmup 计数、`CUDAGraph` 对象、output buffer |
| warmup → capture → replay | `cudagraph_piecewise_backend.py:168` | 3段式流程：先跑 N 次热身，再 `capture_begin/end`，后续调用 `replay()` |
| `_share_buffer_to` | `cudagraph_piecewise_backend.py:200` | 将捕获时的输出张量地址与持久 buffer 绑定，replay 写入该 buffer |
| `paddle.device.synchronize()` | `cudagraph_piecewise_backend.py:184` | capture_begin 前必须同步，防止异步操作进入图 |
| `real_shape → padded_shape` | `cudagraph_piecewise_backend.py:150` | batch size 向上 padding 到预定义的捕获尺寸列表 |

### 1.2 问题：现有装饰器不适用于普通函数

`@support_graph_optimization` 的前提是：
- 被装饰的 **必须** 是 `paddle.nn.Layer` 的子类
- `GraphOptBackend` 依赖 `FDConfig`、`ForwardMeta` 等重型对象
- 调用约定强绑定于 `kwargs["forward_meta"].ids_remove_padding` 提供 real_shape

`_compute_sampling_mask` 是一个 **普通模块级函数**：
```python
def _compute_sampling_mask(probs: paddle.Tensor, top_p: paddle.Tensor) -> List[np.ndarray]:
```
- 接收 `(probs, top_p)` 两个 Tensor，返回 `List[np.ndarray]`（**含 D2H + CPU 操作**）
- 无 `FDConfig`、无 `ForwardMeta`、无 Layer 继承

---

## 2. 关键约束发现

### 2.1 CUDA graph 的根本限制

**CUDA graph 只能捕获 GPU kernel**。凡是在 `capture_begin` / `capture_end` 之间执行的：
- ✅ GPU kernel（paddle op、argsort、cumsum 等）
- ✅ cudaMemcpyAsync (GPU→GPU)
- ❌ CPU 操作、Python 逻辑
- ❌ `.cpu()` / `.numpy()` / `.item()` — 触发同步 D2H 拷贝
- ❌ 动态 Python 分支（`if`、`for`）

**结论：`_compute_sampling_mask` 的 GPU 部分可以进图，D2H 部分不能进图。**

### 2.2 输入 Tensor 必须是"静态地址"

CUDA graph replay 时，输入 Tensor 地址必须与 capture 时完全一致，才能保证正确性。FastDeploy 通过 `padding_cudagraph_inputs()` 在 runner 层提前分配固定 buffer，并始终写入这些 buffer。

对于函数级装饰器，需在装饰器内部维护每个 `capture_shape` 对应的 **input buffer**（形状固定的 pre-allocated tensor），在每次调用时把真实输入 `copy_` 进去。

### 2.3 输出 Tensor 地址的绑定

FastDeploy 用 `_share_buffer_to` 将 capture 时的输出 tensor 的底层地址绑定到一个持久 `output_buffer`，replay 后直接读 `output_buffer`。
装饰器版本需同样处理：创建 `output_buffer = paddle.zeros_like(output)` → `output._share_buffer_to(output_buffer)`。

### 2.4 `_compute_sampling_mask` 的特殊性

该函数同时含有：
1. **GPU 部分**：`argsort`, `take_along_axis`, `cumsum`, `where`, `sum`, `max` → 可进图
2. **D2H 部分**：`k_per_row.max().item()`, `sorted_indices[:, :max_k].cpu().numpy()`, `k_per_row.numpy()` → 不可进图

**分离策略**：将函数拆分为：
- `_compute_sampling_mask_gpu(probs, top_p)` → 只做 GPU 计算，返回 `(sorted_indices, k_per_row, max_k_tensor)` 三个 GPU tensor
- `_compute_sampling_mask_cpu(sorted_indices_gpu, k_per_row_gpu, max_k: int)` → 只做 D2H + Python 切片，返回 `List[np.ndarray]`

用 `@cuda_graph_fn` 装饰 GPU 部分，D2H 部分保持 eager 执行。

---

## 3. 装饰器设计

### 3.1 接口设计

```python
@cuda_graph_fn(
    capture_sizes=[1, 2, 4, 8, 16, 32],  # 预定义的 batch sizes
    num_warmups=2,                         # 每个 size 的 warmup 次数
    size_from=None,                        # 从哪个参数获取 batch size（默认第一个 Tensor 的 shape[0]）
    use_unique_memory_pool=True,           # 是否共享 memory pool
)
def my_gpu_func(tensor_a, tensor_b):
    ...
```

### 3.2 核心数据结构

```python
@dataclass
class FnConcreteSizeEntry:
    real_shape: int
    captured: bool = False
    num_finished_warmup: int = 0
    cuda_graph: Optional[CUDAGraph] = None
    input_buffers: List[paddle.Tensor]   # 固定地址输入 buffer
    output_buffers: List[paddle.Tensor]  # 固定地址输出 buffer
```

### 3.3 调用流程

```
__call__(args, kwargs)
    │
    ├─ 提取 batch_size（从第一个 Tensor 的 shape[0]）
    ├─ batch_size → padded_size（找最小够用的 capture_size）
    ├─ 找到 entry = entries[padded_size]
    │
    ├─ [未捕获 && warmup 未完成]
    │     for n in range(warmup):
    │         copy 真实输入 → input_buffers（首次 warmup 时分配 buffers）
    │         runnable(*input_buffers)
    │
    ├─ [未捕获 && warmup 完成]
    │     synchronize()
    │     capture_begin()
    │         outputs = runnable(*input_buffers)
    │         _share_buffer_to(output_buffers)
    │     capture_end()
    │     entry.captured = True
    │
    └─ [已捕获]
          copy 真实输入 → input_buffers
          cuda_graph.replay()
          return output_buffers
```

### 3.4 输入拷贝策略

- 首次 warmup：`paddle.zeros_like(real_input)` 分配 input_buffers，`paddle.assign(real_input, input_buffers[i])` 拷贝数据
- 后续调用：直接 `input_buffers[i].copy_(real_input)` 或 `paddle.assign(real_input, input_buffers[i])`

---

## 4. 应用到 `_compute_sampling_mask`

### 4.1 调用点分析

`_compute_sampling_mask` 在两处被调用：
- `sampler.py:587`（标准采样路径）：`probs.shape = [B, V]`, `top_p.shape = [B, 1]`，B 由调度器动态决定
- `sampler.py:928`（spec decode 路径）：`probs.shape = [total_accepted_tokens, V]`，同样动态

`capture_sizes` 需要覆盖常见 batch size 范围，可沿用 `cudagraph_capture_sizes`；但函数装饰器此时不依赖 FDConfig，改为提供默认覆盖列表或运行时动态扩展。

### 4.2 实现选择

**方案A（推荐）**：将 `_compute_sampling_mask` 内部拆为 GPU 子函数 + CPU 后处理，GPU 子函数加装饰器。
**方案B**：在装饰器内部自动检测 Tensor 输出，仅对 Tensor 返回值进行图化，非 Tensor 操作（`.numpy()`）自动排在 replay 之后。

选择**方案A**：更清晰、装饰器无需黑魔法、与 FastDeploy 现有拆分惯例一致。

---

## 5. 实现细节验证

### 5.1 `paddle.device.cuda.graphs` API

从源码 `cudagraph_piecewise_backend.py` 确认：
```python
from paddle.device.cuda import graphs
new_graph = graphs.CUDAGraph(pool_id=pool_id)  # pool_id 可为 None
new_graph.capture_begin()
...
new_graph.capture_end()
new_graph.replay()
```
`pool_id` 通过 `paddle.base.core.CUDAGraph.gen_new_memory_pool_id()` 生成（需 `paddle.is_compiled_with_cuda()`）。

### 5.2 `_share_buffer_to` 可用性

`output._share_buffer_to(output_buffer)` 是 PaddlePaddle 内部 API，在 `cudagraph_piecewise_backend.py:200` 中使用。装饰器中同样使用。

### 5.3 `padding_size` 计算

从 `config.py` 的 `init_with_cudagrpah_size()` 学习，预先构建 `real → padded` 映射：
```python
real_to_padded = {}
sorted_sizes = sorted(capture_sizes)
for real in range(1, max(sorted_sizes) + 1):
    for s in sorted_sizes:
        if s >= real:
            real_to_padded[real] = s
            break
```
超出最大 capture_size 的 batch → fallback 到 eager 执行。

---

## 6. 文件布局

```
fastdeploy/model_executor/graph_optimization/
    cuda_graph_fn.py          ← 新建：函数级 CUDA graph 装饰器
    decorator.py              ← 已有：类级别装饰器（不改动）
    cudagraph_piecewise_backend.py  ← 已有（不改动）
    ...

fastdeploy/model_executor/layers/sample/
    sampler.py                ← 修改：引入并应用 @cuda_graph_fn
```

---

## 7. 关键实现代码（草稿）

（详见 `cuda_graph_fn.py` 实现）

核心 `CudaGraphFnBackend.__call__` 伪代码：

```python
def __call__(self, *args, **kwargs):
    batch_size = _get_batch_size(args, kwargs, self.size_from)
    padded_size = self.real_to_padded.get(batch_size)
    if padded_size is None:
        return self.fn(*args, **kwargs)  # fallback

    entry = self.entries[padded_size]

    # 首次：分配 input_buffers（按 padded_size padding 第0维）
    if entry.input_buffers is None:
        entry.input_buffers = _alloc_padded_buffers(args, kwargs, padded_size, self.size_from)

    # 拷贝真实输入到固定地址 buffer
    _copy_inputs(args, kwargs, entry.input_buffers, batch_size)

    # Warmup 阶段
    if not entry.captured and entry.num_finished_warmup < self.num_warmups:
        entry.num_finished_warmup += 1
        return self.fn(*entry.input_buffers)  # warmup 直接返回

    # Capture 阶段
    if not entry.captured:
        paddle.device.synchronize()
        entry.cuda_graph = graphs.CUDAGraph(pool_id=self.pool_id)
        entry.cuda_graph.capture_begin()
        outputs = self.fn(*entry.input_buffers)
        entry.cuda_graph.capture_end()
        # 绑定输出 buffer
        entry.output_buffers = []
        for out in _to_list(outputs):
            buf = paddle.zeros_like(out)
            out._share_buffer_to(buf)
            entry.output_buffers.append(buf)
        entry.captured = True
        paddle.device.synchronize()
        return _from_list(entry.output_buffers, outputs)

    # Replay 阶段
    entry.cuda_graph.replay()
    return _from_list(entry.output_buffers, ...)
```

---

## 8. 已知风险与 TODO

| 风险 | 缓解措施 |
|---|---|
| `top_p` 第0维不是 batch_size | `_compute_sampling_mask_gpu` 内 `top_p = top_p[:real_bsz]` 要在拷贝前完成 |
| vocab_size 维度变化 | vocab_size 在整个服务生命周期固定，安全 |
| Non-CUDA 平台 | 在 `cuda_graph_fn` 初始化时检测 `paddle.is_compiled_with_cuda()`，否则 passthrough |
| Tensor 以外的参数（int、bool）传递 | 跳过 buffer 化，直接传入；但需确保它们在 capture 与 replay 时相同 |
| 多线程并发调用 | 不支持并发 capture，调用方需保证单线程捕获（与 FastDeploy 现行约定一致）|

---

*本文档随实现过程持续更新。*

---

## 9. 实现验证记录

### 9.1 文件布局（已落地）

```
fastdeploy/model_executor/graph_optimization/
    cuda_graph_fn.py          ✅ 新建：函数级 CUDA graph 装饰器
    __init__.py               ✅ 更新：导出 cuda_graph_fn, CudaGraphFnBackend
    decorator.py              ✅ 未改动
    cudagraph_piecewise_backend.py  ✅ 未改动

fastdeploy/model_executor/layers/sample/
    sampler.py                ✅ 修改：拆分 + 应用 @cuda_graph_fn
```

### 9.2 sampler.py 改动说明

**原函数结构（单体，GPU+D2H 混合）：**
```
_compute_sampling_mask(probs, top_p) -> List[np.ndarray]
    GPU: argsort, take_along_axis, cumsum, where, sum
    D2H: .max().item(), .cpu().numpy(), .numpy()
```

**拆分后结构：**
```
_compute_sampling_mask_gpu(probs, top_p) -> [sorted_indices, k_per_row]  ← @cuda_graph_fn
    GPU: argsort, take_along_axis, cumsum, where, sum
    返回两个 GPU Tensor

_compute_sampling_mask(probs, top_p) -> List[np.ndarray]  ← 保持原始接口
    调用 _compute_sampling_mask_gpu (CUDA graph 加速)
    D2H: .max().item(), .cpu().numpy(), .numpy()
    CPU: Python list comprehension
```

**调用点：** `sampler.py:588` 和 `:931` 均调用 `_compute_sampling_mask`，接口不变，无需修改调用点。

### 9.3 返回值类型验证

`_compute_sampling_mask_gpu` 返回 `List[paddle.Tensor]`（含两个元素），符合装饰器对 `List[Tensor]` 的处理路径（`output_is_single = False`，`entry.output_buffers` 是长度为2的列表）。

`_compute_sampling_mask` 解包 `sorted_indices, k_per_row = _compute_sampling_mask_gpu(...)` 正确对应两个 buffer。

### 9.4 padding 行为验证

**输入 padding：**
- `probs` 的 shape 为 `[real_bsz, vocab_size]`，装饰器按 `padded_size` padding 第0维，多余行填0
- `sorted_indices` 和 `k_per_row` 对应 padding 行的结果：`k_per_row[i] = 0`（全0的 probs，cumsum=0 < top_p）
  - 实际上0行概率的 mask_cum 全为 True（0-0=0 < top_p>0），k_per_row 会是 vocab_size
  - **修正**：这是 padding 语义问题，需要在 `_compute_sampling_mask` 中确保只使用 `[:real_bsz]` 的结果——已通过 `k_per_row_cpu = k_per_row.numpy()` 后再 `for i in range(real_bsz)` 来天然截断，padding 行结果被忽略 ✅

**top_p 的 padding：**
- `top_p` 的 shape 为 `[real_bsz, 1]`，在 `_compute_sampling_mask` 内 `top_p = top_p[:real_bsz]` 已做截断
- 装饰器按 padded_size padding 第0维（若 top_p 也是 size_from=0 对应参数，则被 padding）
- **实际上 top_p 是第二个 Tensor 参数（size_from=0 指向 probs），所以 top_p 的 shape[0] 不参与 padding 计算**
- 但 top_p 也是 input_buffers 之一，其 shape 按 `paddle.zeros_like(top_p)` 分配（不做 padding），形状固定为 `[real_bsz, 1]`
- **问题**：如果 `real_bsz` 在不同调用间变化，而 top_p 的 buffer 形状是第一次调用时分配的，后续 real_bsz 不同时会 shape mismatch ❌

### 9.5 top_p shape mismatch 问题及修复

**根因：** 非 size_from 的 Tensor 参数（top_p）也需要随 padded_size 一起扩展第0维，否则 `paddle.assign(real_t, buf)` 在 real_bsz 变化时 shape 不匹配。

**修复方案：** `_alloc_input_buffers` 中对所有 Tensor 参数统一按 padded_size padding 第0维（而非仅 size_from 对应的 Tensor）。这样所有 Tensor 的 buffer 第0维都固定为 padded_size，拷贝时只写入前 real_bsz 行，与 FastDeploy 的 `padding_cudagraph_inputs()` 策略一致。

**已修复**：见 `cuda_graph_fn.py` 中 `_alloc_input_buffers` — 所有 Tensor 参数一律 padding 第0维到 padded_size，移除了原来区分 size_tensor_idx 的逻辑。

### 9.6 非 CUDA 平台 passthrough 验证

`_CUDA_AVAILABLE = paddle.is_compiled_with_cuda()` 在模块加载时检测一次。
`CudaGraphFnBackend.__call__` 首行 `if not _CUDA_AVAILABLE: return self.fn(*args, **kwargs)` 保证 passthrough。

### 9.7 capture_sizes 选择说明

`capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]` 覆盖了常见解码批量大小。
- batch_size > 256 时 fallback 到 eager（与 FastDeploy `real_shape > cudagraph_switch_threshold` 的逻辑一致）
- 可根据实际 `max_num_seqs` 调整上限

### 9.9 paddle.argsort / cumsum CUDA graph 兼容性（已解决）

**历史问题：** paddle 3.4.0.dev20260221 中，`paddle.argsort` 和 `paddle.cumsum` 使用 thrust `DeviceRadixSort` / `CUB` 内部会调用 `cudaStreamSynchronize`，在 stream capture 模式下触发 `cudaErrorStreamCaptureUnsupported`，导致 capture 阶段 crash。

**当前状态（已修复）：** paddle 已修复上述问题（argsort / cumsum 改为 graph-safe 实现）。

`_compute_sampling_mask_gpu` 现包含完整 GPU 计算链路：
```
argsort → take_along_axis → cumsum → mask_cum → where → sum
```
全部运行于 `capture_begin()` / `capture_end()` 区间内，正常享受 CUDA graph 加速。

**`capture_sizes` 配置：** `[1, 2, 4, 8, 16, 32, 64, 128, 256]`，覆盖常见解码批量大小；超出 256 的请求 fallback 到 eager。

两者独立工作，互不干扰：
- `CudaGraphPiecewiseBackend` 捕获整个模型的 `forward`（包括 attention、FFN 等）
- `@cuda_graph_fn` 捕获 `_compute_sampling_mask_gpu`（采样器内的 GPU 计算片段）
- 当整个模型已在 CUDA graph 中运行时，`_compute_sampling_mask_gpu` 会作为 graph 的一部分被捕获，`@cuda_graph_fn` 的 replay 路径不会被触发（因为此时采样器本身不在模型 graph 内）
- 当模型以 eager 模式运行（prefill、batch_size > threshold 等）时，`@cuda_graph_fn` 为采样 GPU 计算独立提供 graph 加速

### 9.10 测试运行结果（最终）

```
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A1_warmup_runs_correctly         PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A2_capture_triggers_after_warmup PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A3_replay_returns_correct_result PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A4_independent_entries_per_size  PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A5_fallback_for_oversized_batch  PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnDecorator::test_A6_multi_tensor_args_non_tensor_passthrough PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestComputeSamplingMask::test_B1_correctness_standard_top_p    PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestComputeSamplingMask::test_B2_top_p_one_keeps_all_tokens     PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestComputeSamplingMask::test_B3_top_p_very_small_keeps_one_token PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestComputeSamplingMask::test_B4_multi_size_warmup_capture_replay_consistency PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestComputeSamplingMask::test_B5_spec_decode_scenario           PASSED
tests/graph_optimization/test_cuda_graph_fn.py::TestCudaGraphFnStateIsolation::test_C1_two_decorated_functions_are_independent PASSED

12 passed, 0 failed  ·  16.87s
```
