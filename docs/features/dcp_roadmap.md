# DCP (Decode Context Parallel) 后续工作规划

> 文档状态：2026-03-13
> 当前分支：`dcp/basic_support`

---

## 一、背景与现状

DCP（Decode Context Parallel）将一个 decode 请求的 KV cache 按 interleaved round-robin 方式分散到多个 GPU（TP 组内切分），让每张 GPU 只存储并计算 `1/dcp_size` 的历史 token，从而缓解长序列 decode 的显存和计算瓶颈。

### 1.1 已完成的基础工作

| 模块 | 已实现内容 |
|------|-----------|
| `args_utils.py` | CLI 参数 `-dcp` / `--cp-kv-cache-interleave-size` |
| `config.py` | `ParallelConfig` 字段 + DCP 通信组创建（从 TP group 内切分子组） |
| `input_batch.py` | `dcp_context_kv_lens` 静态 buffer + block_table 列宽缩减 |
| `forward_meta.py` | `dcp_context_kv_lens` / `max_dcp_context_kv_len` 字段 |
| `gpu_model_runner.py` | KV cache 大小调整 + 每步 local seq_lens 计算（`copy_` 写入静态 buffer） |
| `dcp_comm.py` | `get_dcp_local_seq_lens`、Triton LSE 修正 kernel、`cp_lse_ag_out_rs`、`merge_attn_states` |

### 1.2 核心缺口

**AppendAttentionBackend 尚未接入 DCP。**
`dcp_context_kv_lens` 已被计算并挂入 `forward_meta`，但 `append_attn_backend.py` 完全未消费该字段，导致每个 rank 目前只做了本地 KV 的 attention，跨 rank 的 LSE 合并尚未执行——即**当前 DCP 在 decode 阶段不产生正确结果**。

---

## 二、后续工作清单

### T1 — AppendAttentionBackend 接入 DCP【最高优先级】

**文件**：`fastdeploy/model_executor/layers/attention/append_attn_backend.py`

DCP 的 decode context attention 分为五个步骤（参考 vLLM `_forward_with_dcp()`）：

```
Step 1: AllGather query（跨 DCP 组，将 GQA 的本地 head 还原为全量 head）
        query_all = dcp_group.all_gather(query, dim=head_dim)   # [B, H/N*N, D]

Step 2: 本地 context attention（对本 rank 的 KV 分片，non-causal）
        context_out, context_lse = append_attn(
            q=query_all,
            kv_cache=local_kv_cache,
            seq_lens=dcp_context_kv_lens,   # 本 rank 负责的历史 token 数
            max_seq_len=max_dcp_context_kv_len,
            return_lse=True,                # ← append_attn 需要支持返回 LSE
        )

Step 3: LSE 加权合并（AllGather LSE + Triton kernel 修正 output）
        context_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_out, context_lse, dcp_group, return_lse=True
        )

Step 4: 当前 query 的因果 self-attention（本地 KV，causal=True）
        query_out, query_lse = append_attn(
            q=query, kv=query_kv, causal=True, return_lse=True
        )

Step 5: 合并 context + query attention
        merge_attn_states(output, context_out_cor, context_lse_cor,
                          query_out, query_lse)
```

**需要改动的子任务：**

- **T1.1** `append_attention` / `append_attention_with_output` 自定义算子支持 `return_lse=True`
  LSE shape：`[num_heads, num_tokens]`（与 flash_attn 返回约定一致）。需确认 custom op 是否已有该输出，若无则修改 C++/CUDA kernel 接口。

- **T1.2** 在 `AppendAttentionBackend.forward()` 中插入 DCP 分支
  当 `forward_meta.dcp_context_kv_lens is not None` 时走上述五步流程；否则走原有路径（零改动风险）。

- **T1.3** `init_attention_metadata()` 中传入 `dcp_context_kv_lens`
  目前 `forward_meta.dcp_context_kv_lens` 只被 `gpu_model_runner` 设置，attention backend 未消费。需在 `init_attention_metadata` 中读取并缓存。

- **T1.4** AllGather query 的 head 拼接与还原
  GQA 场景下各 rank 持有不同的 Q head 子集，AllGather 后 head 顺序需与 KV cache 排布对齐；ReduceScatter 之后需正确 slice 回本 rank 的 head 子集。

---

### T2 — CUDA Graph 兼容

**文件**：`gpu_model_runner.py`、`dcp_comm.py`、`input_batch.py`

当前问题：`get_dcp_local_seq_lens` 使用 Python + Paddle ops 在 CPU 端做标量运算，再 `copy_` 到 GPU。这本身不破坏 CG（因为 copy 目标地址固定），但若未来引入动态 shape（如 paged KV pointer 更新）则会出问题。

**子任务：**

- **T2.1** 将 `get_dcp_local_seq_lens` 改写为 Triton kernel
  在 GPU 上原地计算 local seq lens，无 CPU→GPU 同步。参考 vLLM `vllm/v1/worker/gpu/cp_utils.py:prepare_dcp_local_seq_lens()`。
  函数签名：
  ```python
  def prepare_dcp_local_seq_lens_triton(
      out: paddle.Tensor,          # [max_num_seqs], int32, 预分配静态 buffer
      seq_lens: paddle.Tensor,     # [max_num_seqs], int32
      num_reqs: int,
      dcp_size: int,
      dcp_rank: int,
      interleave: int,
  ) -> None: ...
  ```

- **T2.2** AllGather query 与 CUDA Graph
  `T1` 中的 `all_gather` 在 CG capture 时需保证 buffer 地址固定。需为 AllGather 的输出预分配静态 buffer `[max_num_seqs * max_chunk_tokens, num_heads, head_dim]` 并放入 `InputBatch`。

- **T2.3** `merge_attn_states` 输出 buffer 静态化
  Triton kernel 的输出 tensor 需在 CG 内复用同一地址，应在 `InputBatch` 中预分配 `dcp_merge_out_buf`。

- **T2.4** 验证 CG capture & replay
  用 `use_cudagraph=True` + `decode_context_parallel_size=2` 跑端到端测试，确认无 `CUDAGraphCaptureError`。

---

### T3 — Prefix Cache 兼容 ✅（基本完成，待测试）

**文件**：`fastdeploy/cache_manager/prefix_cache_manager.py`、`fastdeploy/worker/input_batch.py`

DCP 下 KV cache 的物理分配粒度发生变化，prefix cache 的 block 匹配必须在**全局粒度**（`block_size * dcp_world_size`）对齐。

**已完成：**

- **T3.1 ✅** Block table 列宽对齐（`input_batch.py:229-241`）
  `block_tables` 的列宽按 `virtual_block_size = block_size * dcp_world_size` 计算，每个 rank 只存 `ceil(max_model_len / virtual_block_size)` 个 block entry：
  ```python
  virtual_block_size = self.cache_config.block_size * dcp_world_size
  pre_max_block_num = (max_model_len + virtual_block_size - 1) // virtual_block_size + enc_dec_block_num
  self.block_tables = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")
  ```
  `gpu_model_runner.py` 中 `max_block_num_per_seq` 也已按 `effective_block_size_for_max` 缩减。

- **T3.2 ✅** Prefix cache hit 的 virtual-block 对齐（`prefix_cache_manager.py:80-89, 983-1029`）
  `PrefixCacheManager.__init__` 读取 `dcp_world_size`；在 prefix cache 匹配后，若命中块数不是 `dcp_world_size` 的整数倍，则从末尾截断到对齐边界，确保所有 DCP rank 的缓存视图一致。截断时正确回退 CPU→GPU swap 状态，并重新遍历 radix tree 定位对齐节点。

**待完成：**

- **T3.3** Prefix cache + DCP 场景下的端到端测试
  启用 `enable_prefix_cache=True` + `decode_context_parallel_size=2`，验证输出正确性和显存使用。

---

### T4 — Chunked Prefill 兼容

**文件**：`append_attn_backend.py`、`gpu_model_runner.py`

DCP 仅加速 decode 阶段的 **context attention**（历史 KV），对 prefill/extend（新 query token）不做分片。混合批次（prefill + decode）需要正确区分两种请求。

**子任务：**

- **T4.1** 混合批次中区分 prefill vs decode 请求
  在 `T1` 的 DCP 分支中，`dcp_context_kv_lens` 对 prefill 请求应为 0（不走 DCP context path），仅对 decode 请求有效：
  ```python
  # context_kv_lens = seq_lens_decoder（不含新 token 的历史长度）
  # prefill 请求的 seq_lens_decoder == 0，自然跳过
  ```

- **T4.2** Chunked prefill 分块边界与 DCP block 粒度对齐
  当 `chunk_size < block_size * dcp_world_size` 时，某个 chunk 可能跨越一个 DCP block 边界，需要确认 KV 写入路径正确。

- **T4.3** 混合批次端到端测试
  启用 `enable_chunked_prefill=True` + `decode_context_parallel_size=2`，验证输出一致性。

---

### T5 — Flash Attention 3 / MLA 原生 CP 支持

**文件**：`fastdeploy/model_executor/layers/attention/`

vLLM 中 FA3 原生支持通过 `cp_world_size`/`cp_rank`/`cp_tot_seqused_k` 参数在 kernel 内部处理 AllGather，无需手动通信。FastDeploy 若未来引入 FA3 backend，应利用这一特性。

**子任务：**

- **T5.1** 评估 FastDeploy FA3 backend 集成路径
  确认 FastDeploy 是否已有/计划引入 FA3 backend；若有，参考 vLLM `backends/mla/flashattn_mla.py` 传入 `cp_*` 参数。

- **T5.2** MLA 模型（DeepSeek 系列）DCP 验证
  MLA 的 KV head 压缩特性与 DCP 的 head 分发逻辑需要单独验证（GQA head 数可能为 1，AllGather query 的 head 维度处理不同）。

---

### T6 — MTP（Multi-Token Prediction）兼容

**文件**：`gpu_model_runner.py`、`append_attn_backend.py`

MTP/Speculative Decoding 在 decode 时会对同一序列产生多个 draft token，每个 draft token 都有独立的 attention 请求，`dcp_context_kv_lens` 的计算需要感知 draft token 数量。

**子任务：**

- **T6.1** `cp_kv_cache_interleave_size=1` 时的 MTP 兼容性验证
  interleave=1 时 DCP 分配为逐 token round-robin，MTP 的多个 draft token 与之无冲突；需端到端验证。

- **T6.2** `cp_kv_cache_interleave_size>1` 时的 MTP 兼容性
  interleave>1 时 block-level 分配，MTP draft token 若落在边界块内需特殊处理；先明确是否支持，若不支持则在参数校验中报错：
  ```python
  if speculative_config is not None and cp_kv_cache_interleave_size > 1:
      raise ValueError("MTP + DCP with interleave_size > 1 is not supported yet.")
  ```

- **T6.3** Speculative decoding + DCP 端到端测试

---

### T7 — A2A 通信后端

**文件**：`dcp_comm.py`

当前 `cp_lse_ag_out_rs` 实现为：AllGather LSE（1次 NCCL）+ Triton correct（计算）+ AllReduce output + slice（模拟 ReduceScatter，实际通信量是真 RS 的 `dcp_size` 倍）。共 **2 次 NCCL** 但通信量偏高。

vLLM 新增的 A2A 后端（[arxiv:2507.07120](https://arxiv.org/abs/2507.07120)）改为：output A2A（1次）+ LSE A2A（1次），共 **2 次 NCCL**，通信量与数据规模无关地减少 `(dcp_size-1)/dcp_size`，适合长序列 + 大 dcp_size 场景。

**子任务：**

- **T7.1** 实现 `cp_lse_a2a_reduce()` 函数
  参考 vLLM `vllm/v1/attention/ops/dcp_alltoall.py`，在 `dcp_comm.py` 中实现 A2A backend。

- **T7.2** 添加 `--dcp-comm-backend` CLI 参数
  支持 `"ag_rs"`（默认，向后兼容）和 `"a2a"` 两种后端。

- **T7.3** 性能对比测试
  长序列（16k-128k）下对比 AG+RS vs A2A 的吞吐和延迟。

---

### T8 — 兼容性门控与参数校验

**文件**：`config.py`、`engine/args_utils.py`

**子任务：**

- **T8.1** 添加不兼容特性的启动报错
  ```python
  # config.py 或 args_utils.py
  if dcp_size > 1:
      if sliding_window is not None:
          raise ValueError("DCP does not support sliding window attention.")
      # Mamba / linear attention 等同理
  ```

- **T8.2** 添加 `assert tp_size % dcp_size == 0` 的友好报错提示

- **T8.3** `cp_kv_cache_interleave_size` 与 `block_size` 的整除校验
  若 `interleave_size > 1`，需确认 `block_size % interleave_size == 0`（避免最后一个 block 边界处的 token 分配歧义）。

---

### T9 — 测试与验证

- **T9.1** 单元测试：`get_dcp_local_seq_lens`（Python vs Triton kernel 对比）
- **T9.2** 单元测试：`cp_lse_ag_out_rs` / `merge_attn_states` 精度验证（与单卡全量 attention 结果对比）
- **T9.3** 集成测试：`dcp=2` + `tp=8`，Llama-3-70B，512/4096/16384 token context
- **T9.4** 正确性测试：与 `dcp=1`（基准）输出逐 token 一致性对比（允许浮点误差 `1e-3`）
- **T9.5** 性能基准：decode throughput（tokens/s/GPU）vs context length，`dcp=1/2/4`
- **T9.6** 显存验证：`dcp=N` 时每卡 KV cache 显存是否接近 `1/N`

---

## 三、任务优先级与依赖关系

```
T1 (AppendAttn DCP)          ← 核心，阻塞所有验证
├── T1.1 (append_attn return LSE)
├── T1.2 (DCP forward branch)
├── T1.3 (init_attention_metadata)
└── T1.4 (head allgather/slice)
        ↓
T2 (CUDA Graph)              ← 依赖 T1 完成后验证
T3 (Prefix Cache) ✅          ← T3.1/T3.2 已完成；T3.3（测试）依赖 T1
T4 (Chunked Prefill)         ← 依赖 T1，T4.1 基本不需额外改动
T5 (FA3/MLA)                 ← 独立，低优先级
T6 (MTP)                     ← 依赖 T1，T6.2 先加报错
T7 (A2A backend)             ← 独立，性能优化
T8 (参数校验)                 ← 独立，可早期合入
T9 (测试)                    ← 依赖 T1-T4
```

**推荐执行顺序：**
1. **T8** — 先合入参数校验（无功能风险）
2. **T1.1** — 修改 append_attn 算子接口
3. **T1.2 ~ T1.4** — 完成 DCP forward 路径
4. **T9.4** — 正确性验证（最重要的里程碑）
5. **T2** — CUDA Graph 支持（线上部署必须）
6. **T3 / T4** — prefix cache 与 chunked prefill
7. **T7** — A2A 通信优化
8. **T5 / T6** — FA3 / MTP 扩展

---

## 四、关键参考

- vLLM DCP 实现：`vllm/v1/attention/backends/flash_attn.py`（`_forward_with_dcp`）
- vLLM A2A 后端：`vllm/v1/attention/ops/dcp_alltoall.py`
- vLLM CP Triton kernel：`vllm/v1/worker/gpu/cp_utils.py`
- FastDeploy DCP 通信库：`fastdeploy/distributed/dcp_comm.py`
- Ring Attention 论文（LSE 合并原理）：https://arxiv.org/pdf/2501.01005
- A2A 通信后端论文：https://arxiv.org/abs/2507.07120
