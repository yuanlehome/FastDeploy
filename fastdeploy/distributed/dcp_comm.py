# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
Decode Context Parallel (DCP) communication and computation utilities.
"""

from __future__ import annotations

import paddle
import paddle.distributed as dist
import triton
import triton.language as tl


def get_dcp_local_seq_lens(
    seq_lens: paddle.Tensor,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> paddle.Tensor:
    """
    Calculate the local KV sequence lengths for a given DCP rank.

    While using DCP, KV cache size stored on each rank may be different.
    This function computes the local sequence lengths for the specified rank.

    Args:
        seq_lens: [num_reqs] - total sequence lengths per request
        dcp_size: DCP world size
        dcp_rank: rank within DCP group
        cp_kv_cache_interleave_size: interleave granularity (I)

    Returns:
        local_seq_lens: [num_reqs] - local KV lengths for this rank
    """
    I = cp_kv_cache_interleave_size
    seq_lens_i32 = seq_lens.cast("int32")
    # Complete rounds: each rank gets base tokens
    base = seq_lens_i32 // I // dcp_size * I
    # Remainder tokens after complete rounds
    remainder = seq_lens_i32 - base * dcp_size
    # This rank's share of remainder, clipped to [0, I]
    remainder = paddle.clip(remainder - dcp_rank * I, min=0, max=I)
    return base + remainder


class CPTritonContext:
    """Cache for Triton JIT kernels to avoid recompilation."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """
    Triton kernel to correct attention output using all-gathered LSEs.

    Args:
        outputs_ptr: Pointer to input tensor of shape [B, H, D]
        new_output_ptr: Pointer to output tensor of shape [B, H, D]
        lses_ptr: Pointer to input tensor of shape [N, B, H]
        vlse_ptr: Pointer to output tensor of shape [B, H]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, PADDED_HEAD_DIM)
    d_mask = d_offsets < HEAD_DIM
    num_n_offsets = tl.arange(0, N_ROUNDED)

    # shape = [N]
    lse_offsets = num_n_offsets * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H

    # calc final lse
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    # shape = [D]
    output_offsets = batch_idx * outputs_stride_B + head_idx * outputs_stride_H + d_offsets * outputs_stride_D

    # correct output
    lse_offset = lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets, mask=d_mask)
    output = output * factor

    tl.store(new_output_ptr + output_offsets, output, mask=d_mask)


def correct_attn_out(
    out: paddle.Tensor,
    lses: paddle.Tensor,
    cp_rank: int,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e: bool = True,
) -> tuple:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [B, H, D]
        lses: Tensor of shape [N, B, H]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation
        is_lse_base_on_e: Whether LSE is base-e (True) or base-2 (False)

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, f"expected lses [N,B,H] (optionally with a 1-sized extra dim), " f"got {tuple(lses.shape)}"

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.strides
    l_sN, l_sB, l_sH = lses.strides

    lse = paddle.empty([B, H], dtype=lses.dtype)

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
    )
    const_args = {
        "HEAD_DIM": D,
        "PADDED_HEAD_DIM": triton.next_power_of_2(D),
        "N_ROUNDED": N,
        "IS_BASE_E": is_lse_base_on_e,
    }
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def _cp_lse_common(
    cp_attn_out: paddle.Tensor,
    cp_attn_lse: paddle.Tensor,
    dcp_group,
    dcp_world_size: int,
    dcp_rank: int,
    ctx: CPTritonContext | None = None,
    is_lse_base_on_e: bool = True,
):
    """
    Common logic for DCP LSE correction: AllGather LSE + Triton correct output.

    Args:
        cp_attn_out: [B, H, D]
        cp_attn_lse: [B, H]
        dcp_group: paddle distributed group
        dcp_world_size: DCP world size
        dcp_rank: rank in DCP group
        ctx: Triton context to avoid recompilation
        is_lse_base_on_e: Whether LSE is base-e (True) or base-2 (False)
    """
    if dcp_world_size == 1:
        return cp_attn_out, cp_attn_lse

    if ctx is None:
        ctx = CPTritonContext()

    cp_attn_lse = cp_attn_lse.contiguous()
    lse_list = []
    dist.all_gather(lse_list, cp_attn_lse, group=dcp_group)
    lses = paddle.stack(lse_list, axis=0)  # [N, B, H]

    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        dcp_rank,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: paddle.Tensor,
    cp_attn_lse: paddle.Tensor,
    dcp_group,
    dcp_world_size: int,
    dcp_rank: int,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
):
    """
    DCP combine using AllGather LSE + correct output + ReduceScatter.

    Args:
        cp_attn_out: [B, H_global, D] - attention output
        cp_attn_lse: [B, H_global] - attention LSE
        dcp_group: paddle distributed group
        dcp_world_size: DCP world size
        dcp_rank: rank in DCP group
        ctx: Triton context to avoid recompilation
        is_lse_base_on_e: Whether LSE is base-e (True) or base-2 (False)
        return_lse: whether to return corrected LSE

    Returns:
        out: [B, H_local, D] or (out, lse) if return_lse
    """
    out, lse = _cp_lse_common(
        cp_attn_out,
        cp_attn_lse,
        dcp_group,
        dcp_world_size,
        dcp_rank,
        ctx=ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )

    # ReduceScatter output along head dimension
    B, H_global, D = out.shape
    H_local = H_global // dcp_world_size

    # all_reduce + slice as reduce_scatter along dim=1
    dist.all_reduce(out, group=dcp_group)
    out = out[:, H_local * dcp_rank : H_local * (dcp_rank + 1), :]  # [B, H_local, D]

    if return_lse:
        lse = lse[:, H_local * dcp_rank : H_local * (dcp_rank + 1)]  # [B, H_local]
        return out, lse
    return out


@triton.jit
def _merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS] or nullptr
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    """
    Triton kernel that merges two partial attention results using LSE-weighted
    combination. Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
    """
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 returns inf for 0-len seqlens, FA3 returns -inf. Normalize to -inf.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output + token_idx * num_heads * prefix_head_stride + head_idx * prefix_head_stride + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output + token_idx * num_heads * prefix_head_stride + head_idx * prefix_head_stride + head_arange,
        mask=head_mask,
    )

    # Compute scale first, then multiply (numerical stability)
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * output_head_stride + head_idx * output_head_stride + head_arange,
        out,
        mask=head_mask,
    )


def merge_attn_states(
    output: paddle.Tensor,
    prefix_output: paddle.Tensor,
    prefix_lse: paddle.Tensor,
    suffix_output: paddle.Tensor,
    suffix_lse: paddle.Tensor,
    output_lse: paddle.Tensor | None = None,
) -> None:
    """
    Merge two partial attention results using LSE-weighted combination (Triton).
    Used to combine context attention (from DCP) with query self-attention.

    All tensors:
        output:        [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_output: [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_lse:    [NUM_HEADS, NUM_TOKENS]
        suffix_output: [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        suffix_lse:    [NUM_HEADS, NUM_TOKENS]
        output_lse:    [NUM_HEADS, NUM_TOKENS] (optional)

    Result is written in-place to `output` (and optionally `output_lse`).
    """
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    prefix_head_stride = prefix_output.strides[1]
    output_head_stride = output.strides[1]

    _merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        head_size,
        padded_head_size,
        output_lse is not None,
    )
