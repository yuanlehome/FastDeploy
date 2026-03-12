#!/usr/bin/env python
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
Worker script for cp_lse_ag_out_rs multi-process tests.

Each rank simulates holding a partial attention result (out, lse) over its
local KV shard, calls cp_lse_ag_out_rs, and saves its result to a per-rank
.npz file so the pytest entry script can collect and verify correctness.

Run via paddle.distributed.launch (see test_cp_lse_ag_out_rs.py).

Environment variables consumed
--------------------------------
TEST_CASE     : name of the test case to run (required)
OUTPUT_DIR    : directory where per-rank .npz files are written (required)
"""

import os
import sys

import numpy as np
import paddle
import paddle.distributed as dist

# ---------------------------------------------------------------------------
# Bootstrap: PYTHONPATH & distributed init
# ---------------------------------------------------------------------------
FD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if FD_ROOT not in sys.path:
    sys.path.insert(0, FD_ROOT)

from fastdeploy.distributed.dcp_comm import cp_lse_ag_out_rs  # noqa: E402

dist.init_parallel_env()
rank = dist.get_rank()
world_size = dist.get_world_size()
dcp_group = dist.new_group(list(range(world_size)))

output_dir = os.environ["OUTPUT_DIR"]
test_case = os.environ["TEST_CASE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_cp_lse_ag_out_rs(all_outs, all_lses, dcp_rank, dcp_world_size):
    """
    Pure-numpy reference for cp_lse_ag_out_rs (base-e LSE).

    all_outs  : list of np.ndarray [B, H, D], one per rank
    all_lses  : list of np.ndarray [B, H],    one per rank
    dcp_rank  : the rank whose output slice we return
    Returns (out_local [B, H_local, D], lse_local [B, H_local])
    """
    N = dcp_world_size
    B, H, D = all_outs[0].shape

    # Stack LSEs: [N, B, H]
    lses = np.stack(all_lses, axis=0)

    # ----- correct each rank's output -----
    # global LSE = log(sum_n exp(lse_n))
    lse_max = np.max(lses, axis=0)  # [B, H]
    lse_shifted = lses - lse_max[None]  # [N, B, H]
    lse_shifted = np.where(np.isnan(lse_shifted) | np.isinf(lse_shifted), -np.inf, lse_shifted)
    exp_sum = np.sum(np.exp(lse_shifted), axis=0)  # [B, H]
    global_lse = np.log(exp_sum) + lse_max  # [B, H]

    corrected_outs = []
    for n in range(N):
        factor = np.exp(all_lses[n] - global_lse)  # [B, H]
        factor = np.where(np.isnan(factor) | np.isinf(factor), 0.0, factor)
        out_n = all_outs[n] * factor[:, :, None]  # [B, H, D]
        corrected_outs.append(out_n)

    # sum-reduce across ranks → all_reduce result [B, H, D]
    full_out = np.sum(corrected_outs, axis=0)  # [B, H, D]
    full_lse = global_lse  # [B, H]

    # slice head dim for this rank
    H_local = H // dcp_world_size
    out_local = full_out[:, H_local * dcp_rank : H_local * (dcp_rank + 1), :]
    lse_local = full_lse[:, H_local * dcp_rank : H_local * (dcp_rank + 1)]
    return out_local, lse_local


def _save(name, rank, **arrays):
    path = os.path.join(output_dir, f"{name}_rank{rank}.npz")
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def run_basic(B=4, H=8, D=64, seed=42):
    """
    Each rank holds an independent partial attention out+lse.
    After cp_lse_ag_out_rs the result should match the numpy reference.
    """
    rng = np.random.default_rng(seed + rank)
    out_np = rng.standard_normal((B, H, D)).astype(np.float32)
    lse_np = rng.standard_normal((B, H)).astype(np.float32)

    # Gather all rank tensors for reference (via dist.all_gather)
    out_t = paddle.to_tensor(out_np)
    lse_t = paddle.to_tensor(lse_np)

    all_outs_list = []
    all_lses_list = []
    dist.all_gather(all_outs_list, out_t, group=dcp_group)
    dist.all_gather(all_lses_list, lse_t, group=dcp_group)

    all_outs_np = [x.numpy() for x in all_outs_list]
    all_lses_np = [x.numpy() for x in all_lses_list]

    # Compute reference
    ref_out, ref_lse = _reference_cp_lse_ag_out_rs(all_outs_np, all_lses_np, rank, world_size)

    # Run function under test
    result = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=True,
    )
    result_out, result_lse = result

    _save(
        "basic", rank, result_out=result_out.numpy(), result_lse=result_lse.numpy(), ref_out=ref_out, ref_lse=ref_lse
    )


def run_return_lse_false(B=4, H=8, D=64, seed=10):
    """
    When return_lse=False only a tensor (not a tuple) is returned.
    """
    rng = np.random.default_rng(seed + rank)
    out_np = rng.standard_normal((B, H, D)).astype(np.float32)
    lse_np = rng.standard_normal((B, H)).astype(np.float32)

    out_t = paddle.to_tensor(out_np)
    lse_t = paddle.to_tensor(lse_np)

    result = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=False,
    )

    # result must be a Tensor, not a tuple
    assert isinstance(result, paddle.Tensor), f"rank {rank}: expected Tensor, got {type(result)}"
    H_local = H // world_size
    assert list(result.shape) == [B, H_local, D], f"rank {rank}: wrong shape {list(result.shape)}"
    _save("return_lse_false", rank, out=result.numpy())


def run_output_shape(B=6, H=16, D=32, seed=77):
    """
    Verify output tensor shapes.
    out:  [B, H//N, D]
    lse:  [B, H//N]
    """
    rng = np.random.default_rng(seed + rank)
    out_t = paddle.to_tensor(rng.standard_normal((B, H, D)).astype(np.float32))
    lse_t = paddle.to_tensor(rng.standard_normal((B, H)).astype(np.float32))

    out_r, lse_r = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=True,
    )
    H_local = H // world_size
    assert list(out_r.shape) == [B, H_local, D], f"rank {rank}: out shape {list(out_r.shape)}"
    assert list(lse_r.shape) == [B, H_local], f"rank {rank}: lse shape {list(lse_r.shape)}"
    _save("output_shape", rank, ok=np.array([1]))


def run_inf_lse(B=4, H=8, D=64, seed=99):
    """
    Inject inf/nan LSE values.  Result must be finite (no nan propagation).
    """
    rng = np.random.default_rng(seed + rank)
    out_np = rng.standard_normal((B, H, D)).astype(np.float32)
    lse_np = rng.standard_normal((B, H)).astype(np.float32)

    if rank == 0:
        lse_np[0, 0] = float("inf")
        lse_np[1, 2] = float("nan")

    out_t = paddle.to_tensor(out_np)
    lse_t = paddle.to_tensor(lse_np)

    out_r, lse_r = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=True,
    )
    _save("inf_lse", rank, out=out_r.numpy(), lse=lse_r.numpy())


def run_single_batch(H=8, D=64, seed=5):
    """B=1 edge case."""
    B = 1
    rng = np.random.default_rng(seed + rank)
    out_t = paddle.to_tensor(rng.standard_normal((B, H, D)).astype(np.float32))
    lse_t = paddle.to_tensor(rng.standard_normal((B, H)).astype(np.float32))

    out_r, lse_r = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=True,
    )
    H_local = H // world_size
    assert list(out_r.shape) == [B, H_local, D]
    assert list(lse_r.shape) == [B, H_local]
    _save("single_batch", rank, ok=np.array([1]))


def run_non_pow2_head_dim(B=4, H=8, D=96, seed=33):
    """Head dimension that is not a power of 2."""
    rng = np.random.default_rng(seed + rank)
    out_t = paddle.to_tensor(rng.standard_normal((B, H, D)).astype(np.float32))
    lse_t = paddle.to_tensor(rng.standard_normal((B, H)).astype(np.float32))

    out_r, lse_r = cp_lse_ag_out_rs(
        out_t,
        lse_t,
        dcp_group=dcp_group,
        dcp_world_size=world_size,
        dcp_rank=rank,
        return_lse=True,
    )
    H_local = H // world_size
    assert list(out_r.shape) == [B, H_local, D]
    _save("non_pow2_head_dim", rank, ok=np.array([1]))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
CASES = {
    "basic": run_basic,
    "return_lse_false": run_return_lse_false,
    "output_shape": run_output_shape,
    "inf_lse": run_inf_lse,
    "single_batch": run_single_batch,
    "non_pow2_head_dim": run_non_pow2_head_dim,
}

if test_case not in CASES:
    raise ValueError(f"Unknown TEST_CASE={test_case!r}. Available: {list(CASES)}")

CASES[test_case]()
print(f"[rank {rank}] {test_case} OK")
