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
Unit tests for get_dcp_local_seq_lens in fastdeploy/distributed/dcp_comm.py.

Usage:
    pytest tests/distributed/test_get_dcp_local_seq_lens.py -v
"""

import paddle
import pytest

from fastdeploy.distributed.dcp_comm import get_dcp_local_seq_lens


def _local_seq_lens(seq_lens, dcp_size, dcp_rank, interleave=1):
    """Pure-Python reference implementation of get_dcp_local_seq_lens."""
    I = interleave
    result = []
    for s in seq_lens:
        base = s // I // dcp_size * I
        remainder = s - base * dcp_size
        rank_remainder = max(0, min(remainder - dcp_rank * I, I))
        result.append(base + rank_remainder)
    return result


def _run(seq_lens, dcp_size, dcp_rank, interleave=1):
    """Helper: run get_dcp_local_seq_lens and return a Python list of ints."""
    t = paddle.to_tensor(seq_lens, dtype="int32")
    out = get_dcp_local_seq_lens(t, dcp_size=dcp_size, dcp_rank=dcp_rank, cp_kv_cache_interleave_size=interleave)
    return out.tolist()


# ---------------------------------------------------------------------------
# Basic correctness: dcp_size=1 → identity
# ---------------------------------------------------------------------------


class TestDcpSize1:
    def test_single_seq(self):
        assert _run([16], dcp_size=1, dcp_rank=0) == [16]

    def test_multiple_seqs(self):
        seq_lens = [4, 8, 12, 100]
        assert _run(seq_lens, dcp_size=1, dcp_rank=0) == seq_lens

    def test_zero_len(self):
        assert _run([0], dcp_size=1, dcp_rank=0) == [0]

    def test_one_token(self):
        assert _run([1], dcp_size=1, dcp_rank=0) == [1]


# ---------------------------------------------------------------------------
# dcp_size=2, interleave=1 (default)
# ---------------------------------------------------------------------------


class TestDcpSize2Interleave1:
    """
    With I=1, dcp_size=2:
      base     = s // 2            (same for every rank)
      remainder = s - base*2 = s % 2
      rank0 gets: base + clip(remainder - 0, 0, 1) = base + min(remainder, 1)
      rank1 gets: base + clip(remainder - 1, 0, 1) = base + max(remainder-1, 0)
    """

    def test_even_seq(self):
        # s=8: base=4, remainder=0 → both ranks get 4
        assert _run([8], dcp_size=2, dcp_rank=0) == [4]
        assert _run([8], dcp_size=2, dcp_rank=1) == [4]

    def test_odd_seq(self):
        # s=9: base=4, remainder=1 → rank0=5, rank1=4
        assert _run([9], dcp_size=2, dcp_rank=0) == [5]
        assert _run([9], dcp_size=2, dcp_rank=1) == [4]

    def test_sum_equals_total(self):
        for s in range(0, 33):
            r0 = _run([s], dcp_size=2, dcp_rank=0)[0]
            r1 = _run([s], dcp_size=2, dcp_rank=1)[0]
            assert r0 + r1 == s, f"s={s}: r0={r0} + r1={r1} != {s}"

    def test_matches_reference(self):
        seq_lens = list(range(0, 20))
        for rank in range(2):
            expected = _local_seq_lens(seq_lens, dcp_size=2, dcp_rank=rank, interleave=1)
            got = _run(seq_lens, dcp_size=2, dcp_rank=rank)
            assert got == expected, f"rank={rank}: got={got}, expected={expected}"


# ---------------------------------------------------------------------------
# dcp_size=4, interleave=1
# ---------------------------------------------------------------------------


class TestDcpSize4Interleave1:
    def test_sum_equals_total(self):
        for s in range(0, 25):
            total = sum(_run([s], dcp_size=4, dcp_rank=r)[0] for r in range(4))
            assert total == s, f"s={s}: sum={total} != {s}"

    def test_matches_reference(self):
        seq_lens = [0, 1, 3, 4, 5, 7, 8, 11, 16, 17, 100]
        for rank in range(4):
            expected = _local_seq_lens(seq_lens, dcp_size=4, dcp_rank=rank, interleave=1)
            got = _run(seq_lens, dcp_size=4, dcp_rank=rank)
            assert got == expected, f"rank={rank}: got={got}, expected={expected}"


# ---------------------------------------------------------------------------
# Interleave (I > 1)
# ---------------------------------------------------------------------------


class TestInterleave:
    """
    With I=4, dcp_size=2:
      A "complete round" distributes I tokens to each rank → 2*I per round.
      base     = s // (I * dcp_size) * I
      remainder = s - base * dcp_size
      rank_share = clip(remainder - rank*I, 0, I)
    """

    @pytest.mark.parametrize("I", [2, 4, 8])
    def test_sum_equals_total(self, I):
        for s in range(0, 3 * I * 2 + I + 1):
            total = sum(_run([s], dcp_size=2, dcp_rank=r, interleave=I)[0] for r in range(2))
            assert total == s, f"I={I}, s={s}: sum={total} != {s}"

    @pytest.mark.parametrize("I", [1, 2, 4, 8])
    def test_matches_reference_dcp2(self, I):
        seq_lens = list(range(0, 4 * I * 2 + 1))
        for rank in range(2):
            expected = _local_seq_lens(seq_lens, dcp_size=2, dcp_rank=rank, interleave=I)
            got = _run(seq_lens, dcp_size=2, dcp_rank=rank, interleave=I)
            assert got == expected, f"I={I}, rank={rank}"

    @pytest.mark.parametrize("I", [1, 2, 4])
    def test_matches_reference_dcp4(self, I):
        seq_lens = list(range(0, 4 * I * 4 + 1))
        for rank in range(4):
            expected = _local_seq_lens(seq_lens, dcp_size=4, dcp_rank=rank, interleave=I)
            got = _run(seq_lens, dcp_size=4, dcp_rank=rank, interleave=I)
            assert got == expected, f"I={I}, rank={rank}"

    def test_exactly_one_full_round_per_rank(self):
        # s = I * dcp_size → each rank gets exactly I tokens
        I, dcp_size = 4, 3
        s = I * dcp_size  # 12
        for rank in range(dcp_size):
            assert _run([s], dcp_size=dcp_size, dcp_rank=rank, interleave=I) == [I]

    def test_partial_last_rank(self):
        # s = I * (dcp_size - 1) + I//2: last rank gets I//2, others get I
        I, dcp_size = 4, 3
        partial = I // 2
        s = I * (dcp_size - 1) + partial
        for rank in range(dcp_size - 1):
            assert _run([s], dcp_size=dcp_size, dcp_rank=rank, interleave=I) == [I], f"rank={rank} should get I={I}"
        assert _run([s], dcp_size=dcp_size, dcp_rank=dcp_size - 1, interleave=I) == [
            partial
        ], f"last rank should get {partial}"


# ---------------------------------------------------------------------------
# Non-divisible interleave lengths
# ---------------------------------------------------------------------------


class TestNonDivisible:
    def test_seq_not_multiple_of_interleave(self):
        # s=5, I=4, dcp_size=2:
        #   base = 5//4//2*4 = 0
        #   remainder = 5
        #   rank0 gets clip(5-0, 0, 4) = 4
        #   rank1 gets clip(5-4, 0, 4) = 1
        assert _run([5], dcp_size=2, dcp_rank=0, interleave=4) == [4]
        assert _run([5], dcp_size=2, dcp_rank=1, interleave=4) == [1]

    def test_seq_smaller_than_interleave(self):
        # s=3, I=4, dcp_size=2:
        #   base = 3//4//2*4 = 0
        #   remainder = 3
        #   rank0 gets clip(3, 0, 4) = 3
        #   rank1 gets clip(3-4, 0, 4) = 0
        assert _run([3], dcp_size=2, dcp_rank=0, interleave=4) == [3]
        assert _run([3], dcp_size=2, dcp_rank=1, interleave=4) == [0]


# ---------------------------------------------------------------------------
# Output dtype and shape
# ---------------------------------------------------------------------------


class TestOutputProperties:
    def test_output_dtype_is_int32(self):
        t = paddle.to_tensor([10, 20, 30], dtype="int32")
        out = get_dcp_local_seq_lens(t, dcp_size=2, dcp_rank=0)
        assert out.dtype == paddle.int32

    def test_output_shape_matches_input(self):
        seq_lens = [1, 5, 10, 100]
        t = paddle.to_tensor(seq_lens, dtype="int32")
        out = get_dcp_local_seq_lens(t, dcp_size=2, dcp_rank=0)
        assert list(out.shape) == [len(seq_lens)]

    def test_empty_input(self):
        t = paddle.to_tensor([], dtype="int32")
        out = get_dcp_local_seq_lens(t, dcp_size=2, dcp_rank=0)
        assert list(out.shape) == [0]

    def test_int64_input_cast(self):
        # Input may be int64; output should still be int32
        t = paddle.to_tensor([8, 9], dtype="int64")
        out = get_dcp_local_seq_lens(t, dcp_size=2, dcp_rank=0)
        assert out.dtype == paddle.int32

    def test_non_negative(self):
        seq_lens = list(range(0, 50))
        t = paddle.to_tensor(seq_lens, dtype="int32")
        for rank in range(4):
            out = get_dcp_local_seq_lens(t, dcp_size=4, dcp_rank=rank)
            assert (out >= 0).all().item(), f"rank={rank} produced negative values"

    def test_local_le_total(self):
        seq_lens = list(range(0, 50))
        t = paddle.to_tensor(seq_lens, dtype="int32")
        for rank in range(4):
            out = get_dcp_local_seq_lens(t, dcp_size=4, dcp_rank=rank)
            assert (out <= t).all().item(), f"rank={rank}: local > total"


# ---------------------------------------------------------------------------
# Batch consistency: per-element independence
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    def test_batch_vs_single(self):
        """Batch result should equal element-wise single calls."""
        seq_lens = [0, 1, 7, 8, 9, 15, 16, 17, 100]
        for dcp_size in [2, 4]:
            for rank in range(dcp_size):
                batch_result = _run(seq_lens, dcp_size=dcp_size, dcp_rank=rank)
                single_results = [_run([s], dcp_size=dcp_size, dcp_rank=rank)[0] for s in seq_lens]
                assert batch_result == single_results, f"dcp_size={dcp_size}, rank={rank}"
