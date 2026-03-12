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
Unit tests for merge_attn_states in fastdeploy/distributed/dcp_comm.py.

merge_attn_states merges two partial attention results (prefix / suffix)
via LSE-weighted combination:

    max_lse  = max(p_lse, s_lse)
    p_se     = exp(p_lse - max_lse)
    s_se     = exp(s_lse - max_lse)
    out_se   = p_se + s_se
    output   = (p_se * prefix_output + s_se * suffix_output) / out_se
    out_lse  = log(out_se) + max_lse       (if output_lse is provided)

Edge cases:
  * p_lse or s_lse == +inf  → treated as -inf (FA2 convention)
  * p_lse == -inf           → output == suffix_output, lse == s_lse
  * s_lse == -inf           → output == prefix_output, lse == p_lse
  * both -inf               → output is 0 * prefix + 0 * suffix = 0, lse = -inf

Usage:
    pytest tests/distributed/dcp/test_merge_attn_states.py -v
"""

import numpy as np
import paddle
import pytest

from fastdeploy.distributed.dcp_comm import merge_attn_states

# ---------------------------------------------------------------------------
# Numpy reference implementation
# ---------------------------------------------------------------------------


def _ref_merge(prefix_output, prefix_lse, suffix_output, suffix_lse):
    """
    Pure-numpy reference for merge_attn_states.

    Args:
        prefix_output: [T, H, D]  float32
        prefix_lse:    [H, T]     float32
        suffix_output: [T, H, D]  float32
        suffix_lse:    [H, T]     float32

    Returns:
        output:     [T, H, D]
        output_lse: [H, T]
    """
    T, H, D = prefix_output.shape

    # +inf → -inf (FA2 convention)
    p_lse = np.where(prefix_lse == np.inf, -np.inf, prefix_lse)  # [H, T]
    s_lse = np.where(suffix_lse == np.inf, -np.inf, suffix_lse)  # [H, T]

    max_lse = np.maximum(p_lse, s_lse)  # [H, T]
    p_se = np.exp(p_lse - max_lse)  # [H, T]
    s_se = np.exp(s_lse - max_lse)  # [H, T]
    out_se = p_se + s_se  # [H, T]

    # output_lse [H, T]
    output_lse = np.log(out_se) + max_lse

    # scale [H, T] → broadcast to [T, H, D]
    p_scale = (p_se / out_se).T[:, :, None]  # [T, H, 1]
    s_scale = (s_se / out_se).T[:, :, None]  # [T, H, 1]
    output = p_scale * prefix_output + s_scale * suffix_output

    return output.astype(np.float32), output_lse.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(prefix_output_np, prefix_lse_np, suffix_output_np, suffix_lse_np, with_output_lse=True):
    """Convert numpy → paddle, call merge_attn_states, return numpy results."""
    T, H, D = prefix_output_np.shape

    output = paddle.empty([T, H, D], dtype="float32")
    output_lse = paddle.empty([H, T], dtype="float32") if with_output_lse else None

    merge_attn_states(
        output,
        paddle.to_tensor(prefix_output_np),
        paddle.to_tensor(prefix_lse_np),
        paddle.to_tensor(suffix_output_np),
        paddle.to_tensor(suffix_lse_np),
        output_lse=output_lse,
    )

    if with_output_lse:
        return output.numpy(), output_lse.numpy()
    return output.numpy(), None


def _assert_close(actual, expected, *, rtol=1e-4, atol=1e-4, label=""):
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{label}: mismatch" if label else "mismatch",
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    """Output matches numpy reference for normal inputs."""

    @pytest.mark.parametrize(
        "T,H,D",
        [
            (32, 8, 128),
            (1, 8, 128),  # single token
            (16, 1, 128),  # single head
            (64, 16, 64),
        ],
    )
    def test_matches_reference(self, T, H, D):
        rng = np.random.default_rng(42)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, ref_out, label="output")
        _assert_close(got_lse, ref_lse, label="output_lse")

    def test_output_lse_none(self):
        """When output_lse=None, function must still produce correct output."""
        rng = np.random.default_rng(7)
        T, H, D = 16, 4, 64
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        ref_out, _ = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse, with_output_lse=False)

        _assert_close(got_out, ref_out, label="output (no output_lse)")
        assert got_lse is None


# ---------------------------------------------------------------------------
# Non-power-of-2 head dimension
# ---------------------------------------------------------------------------


class TestNonPow2HeadDim:
    """merge_attn_states uses PADDED_HEAD_SIZE = next_power_of_2(D), so D
    need not be a power of 2."""

    @pytest.mark.parametrize("D", [48, 80, 96, 112, 160, 192])
    def test_non_pow2_head_dim(self, D):
        T, H = 16, 8
        rng = np.random.default_rng(D)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, ref_out, label=f"output D={D}")
        _assert_close(got_lse, ref_lse, label=f"output_lse D={D}")


# ---------------------------------------------------------------------------
# Degenerate LSE cases (one side dominates)
# ---------------------------------------------------------------------------


class TestDegenerateLse:
    """
    When one LSE is -inf the corresponding partial result has zero weight,
    so the output must equal the other side exactly.
    """

    def test_prefix_lse_neginf(self):
        """All prefix LSEs = -inf → output == suffix_output."""
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(1)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = np.full((H, T), -np.inf, dtype=np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, s_out, label="output should equal suffix_output")
        _assert_close(got_lse, s_lse, label="lse should equal suffix_lse")

    def test_suffix_lse_neginf(self):
        """All suffix LSEs = -inf → output == prefix_output."""
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(2)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = np.full((H, T), -np.inf, dtype=np.float32)

        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, p_out, label="output should equal prefix_output")
        _assert_close(got_lse, p_lse, label="lse should equal prefix_lse")

    def test_equal_lse_half_weight(self):
        """
        When p_lse == s_lse, weights are equal (0.5/0.5).
        output = 0.5 * prefix_output + 0.5 * suffix_output
        output_lse = p_lse + log(2)
        """
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(3)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        lse = rng.standard_normal((H, T)).astype(np.float32)

        got_out, got_lse = _run(p_out, lse, s_out, lse)

        expected_out = 0.5 * p_out + 0.5 * s_out
        expected_lse = lse + np.log(2).astype(np.float32)

        _assert_close(got_out, expected_out, rtol=1e-4, atol=1e-4, label="output (equal lse)")
        _assert_close(got_lse, expected_lse, rtol=1e-4, atol=1e-4, label="output_lse (equal lse)")


# ---------------------------------------------------------------------------
# FA2 +inf convention
# ---------------------------------------------------------------------------


class TestInfLse:
    """+inf LSE (FA2 zero-length sequence convention) must be treated as -inf."""

    def test_prefix_inf_treated_as_neginf(self):
        """
        +inf prefix LSE  ≡  -inf prefix LSE  →  output == suffix_output.
        """
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(10)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = np.full((H, T), np.inf, dtype=np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, s_out, label="output should equal suffix_output")
        _assert_close(got_lse, s_lse, label="lse should equal suffix_lse")

    def test_suffix_inf_treated_as_neginf(self):
        """
        +inf suffix LSE  ≡  -inf suffix LSE  →  output == prefix_output.
        """
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(11)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = np.full((H, T), np.inf, dtype=np.float32)

        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, p_out, label="output should equal prefix_output")
        _assert_close(got_lse, p_lse, label="lse should equal prefix_lse")

    def test_mixed_inf_and_normal(self):
        """
        Scattered +inf values among normal LSEs — matches numpy reference
        where +inf is replaced by -inf.
        """
        T, H, D = 16, 8, 64
        rng = np.random.default_rng(12)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        # Inject +inf at a few positions
        p_lse[0, 0] = np.inf
        p_lse[2, 3] = np.inf
        s_lse[1, 5] = np.inf

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, ref_out, label="output (mixed inf)")
        _assert_close(got_lse, ref_lse, label="output_lse (mixed inf)")


# ---------------------------------------------------------------------------
# In-place semantics: output must be fully overwritten
# ---------------------------------------------------------------------------


class TestInPlace:
    def test_output_fully_overwritten(self):
        """The initial content of `output` must not bleed into the result."""
        T, H, D = 16, 4, 64
        rng = np.random.default_rng(20)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        # Pre-fill output with large garbage values
        garbage = paddle.full([T, H, D], 1e9, dtype="float32")
        output_lse = paddle.empty([H, T], dtype="float32")
        merge_attn_states(
            garbage,
            paddle.to_tensor(p_out),
            paddle.to_tensor(p_lse),
            paddle.to_tensor(s_out),
            paddle.to_tensor(s_lse),
            output_lse=output_lse,
        )

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        _assert_close(garbage.numpy(), ref_out, label="output (garbage init)")
        _assert_close(output_lse.numpy(), ref_lse, label="output_lse (garbage init)")


# ---------------------------------------------------------------------------
# Output shape & dtype
# ---------------------------------------------------------------------------


class TestOutputProperties:
    def test_output_shape(self):
        T, H, D = 10, 6, 128
        rng = np.random.default_rng(30)
        p_out = paddle.to_tensor(rng.standard_normal((T, H, D)).astype(np.float32))
        s_out = paddle.to_tensor(rng.standard_normal((T, H, D)).astype(np.float32))
        p_lse = paddle.to_tensor(rng.standard_normal((H, T)).astype(np.float32))
        s_lse = paddle.to_tensor(rng.standard_normal((H, T)).astype(np.float32))

        output = paddle.empty([T, H, D], dtype="float32")
        output_lse = paddle.empty([H, T], dtype="float32")

        merge_attn_states(output, p_out, p_lse, s_out, s_lse, output_lse=output_lse)

        assert list(output.shape) == [T, H, D]
        assert list(output_lse.shape) == [H, T]

    def test_output_dtype_float32(self):
        T, H, D = 4, 2, 32
        rng = np.random.default_rng(31)
        p_out = paddle.to_tensor(rng.standard_normal((T, H, D)).astype(np.float32))
        s_out = paddle.to_tensor(rng.standard_normal((T, H, D)).astype(np.float32))
        p_lse = paddle.to_tensor(rng.standard_normal((H, T)).astype(np.float32))
        s_lse = paddle.to_tensor(rng.standard_normal((H, T)).astype(np.float32))

        output = paddle.empty([T, H, D], dtype="float32")
        output_lse = paddle.empty([H, T], dtype="float32")

        merge_attn_states(output, p_out, p_lse, s_out, s_lse, output_lse=output_lse)

        assert output.dtype == paddle.float32
        assert output_lse.dtype == paddle.float32

    def test_single_token_single_head(self):
        T, H, D = 1, 1, 64
        rng = np.random.default_rng(32)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        _assert_close(got_out, ref_out, label="T=H=1 output")
        _assert_close(got_lse, ref_lse, label="T=H=1 output_lse")


# ---------------------------------------------------------------------------
# Numerical stability: large LSE values
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_large_lse_values(self):
        """Large positive LSE values should not produce inf/nan in output."""
        T, H, D = 8, 4, 64
        rng = np.random.default_rng(40)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.random((H, T)).astype(np.float32) * 200 - 100  # [-100, 100]
        s_lse = rng.random((H, T)).astype(np.float32) * 200 - 100

        ref_out, ref_lse = _ref_merge(p_out, p_lse, s_out, s_lse)
        got_out, got_lse = _run(p_out, p_lse, s_out, s_lse)

        assert np.isfinite(got_out).all(), "output contains NaN/Inf"
        assert np.isfinite(got_lse).all(), "output_lse contains NaN/Inf"
        _assert_close(got_out, ref_out, rtol=1e-3, atol=1e-3, label="large lse output")

    def test_output_lse_associativity(self):
        """
        Merging (A, B) then merging the result with C must equal merging A with
        the prior merge of (B, C).  Tests log-sum-exp associativity through the
        kernel.
        """
        T, H, D = 12, 4, 64
        rng = np.random.default_rng(50)
        a_out = rng.standard_normal((T, H, D)).astype(np.float32)
        b_out = rng.standard_normal((T, H, D)).astype(np.float32)
        c_out = rng.standard_normal((T, H, D)).astype(np.float32)
        a_lse = rng.standard_normal((H, T)).astype(np.float32)
        b_lse = rng.standard_normal((H, T)).astype(np.float32)
        c_lse = rng.standard_normal((H, T)).astype(np.float32)

        # (A ⊕ B) ⊕ C
        ab_out, ab_lse = _run(a_out, a_lse, b_out, b_lse)
        abc_left_out, abc_left_lse = _run(ab_out, ab_lse, c_out, c_lse)

        # A ⊕ (B ⊕ C)
        bc_out, bc_lse = _run(b_out, b_lse, c_out, c_lse)
        abc_right_out, abc_right_lse = _run(a_out, a_lse, bc_out, bc_lse)

        _assert_close(abc_left_out, abc_right_out, rtol=1e-4, atol=1e-4, label="associativity output")
        _assert_close(abc_left_lse, abc_right_lse, rtol=1e-4, atol=1e-4, label="associativity lse")

    def test_output_lse_commutativity(self):
        """
        The combined LSE must be commutative: merge(A, B) == merge(B, A)
        in terms of the output LSE and the magnitude of the output.
        (Output values differ by sign due to scale symmetry but their
        weighted magnitudes must match.)
        """
        T, H, D = 8, 4, 32
        rng = np.random.default_rng(55)
        p_out = rng.standard_normal((T, H, D)).astype(np.float32)
        s_out = rng.standard_normal((T, H, D)).astype(np.float32)
        p_lse = rng.standard_normal((H, T)).astype(np.float32)
        s_lse = rng.standard_normal((H, T)).astype(np.float32)

        out_ps, lse_ps = _run(p_out, p_lse, s_out, s_lse)
        out_sp, lse_sp = _run(s_out, s_lse, p_out, p_lse)

        # LSE must be identical regardless of order
        _assert_close(lse_ps, lse_sp, rtol=1e-5, atol=1e-5, label="lse commutativity")
        # Output must be identical (weighted sum is commutative)
        _assert_close(out_ps, out_sp, rtol=1e-4, atol=1e-4, label="output commutativity")
