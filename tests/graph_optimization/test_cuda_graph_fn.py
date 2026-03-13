"""
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

"""
Tests for cuda_graph_fn decorator and its application on _compute_sampling_mask.

Test coverage:
  A. Decorator internals (CudaGraphFnBackend)
     A1. warmup phase: 正确执行 N 次热身，每次结果与 eager 一致
     A2. capture phase: warmup 结束后下一次调用触发 capture，结果与 eager 一致
     A3. replay phase:  capture 后所有后续调用走 replay，结果与 eager 一致
     A4. multi-size:    不同 batch_size 分别触发独立的 warmup→capture→replay
     A5. fallback:      batch_size 超出最大 capture_size 时 fallback 到 eager
     A6. non-tensor passthrough: 非 Tensor 参数正常透传

  B. _compute_sampling_mask correctness (GPU kernel + D2H)
     B1. 基准正确性：标准 top_p 截断，结果与参考实现匹配
     B2. top_p = 1.0：保留所有 token
     B3. top_p 极小：每行只保留概率最大的 token
     B4. 多 batch_size：多次调用（warmup + capture + replay）结果一致
     B5. spec decode 场景：total_accepted_tokens 维度，accept_top_p 正确展开
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

# ---------------------------------------------------------------------------
# Skip if no GPU
# ---------------------------------------------------------------------------
SKIP_NO_CUDA = not paddle.is_compiled_with_cuda()
SKIP_REASON = "Requires CUDA"


# ---------------------------------------------------------------------------
# Helper: reference (pure-eager, no CUDA graph) implementation
# ---------------------------------------------------------------------------


def _reference_compute_sampling_mask(probs: paddle.Tensor, top_p: paddle.Tensor):
    """
    Reference eager implementation of _compute_sampling_mask (no CUDA graph).
    Directly mirrors the original algorithm for golden-value comparison.
    """
    real_bsz = probs.shape[0]
    top_p = top_p[:real_bsz]
    sorted_indices = paddle.argsort(probs, axis=-1, descending=True)
    sorted_probs = paddle.take_along_axis(probs, sorted_indices, axis=-1)
    cum_probs = paddle.cumsum(sorted_probs, axis=-1)
    mask_cum = (cum_probs - sorted_probs) < top_p
    full_mask = (top_p >= 1.0).expand_as(mask_cum)
    mask_cum = paddle.where(full_mask, paddle.ones_like(mask_cum), mask_cum)
    k_per_row = mask_cum.astype("int32").sum(axis=-1)
    max_k = int(k_per_row.max().item())
    sorted_indices_cpu = sorted_indices[:, :max_k].cpu().numpy()
    k_per_row_cpu = k_per_row.numpy()
    return [sorted_indices_cpu[i, : k_per_row_cpu[i]] for i in range(real_bsz)]


def _arrays_equal(a, b):
    """Compare two List[np.ndarray] element-wise (sorted, for set equality)."""
    if len(a) != len(b):
        return False
    for ai, bi in zip(a, b):
        if not np.array_equal(np.sort(ai), np.sort(bi)):
            return False
    return True


# ---------------------------------------------------------------------------
# Section A: Decorator internals
# ---------------------------------------------------------------------------


@unittest.skipIf(SKIP_NO_CUDA, SKIP_REASON)
class TestCudaGraphFnDecorator(unittest.TestCase):
    """Tests for CudaGraphFnBackend / @cuda_graph_fn decorator mechanics."""

    def setUp(self):
        from fastdeploy.model_executor.graph_optimization.cuda_graph_fn import (
            cuda_graph_fn,
        )

        # A simple single-Tensor-in / single-Tensor-out GPU function
        self.call_counter = 0
        outer_self = self

        @cuda_graph_fn(capture_sizes=[4, 8], num_warmups=2, size_from=0)
        def gpu_add_one(x: paddle.Tensor) -> paddle.Tensor:
            outer_self.call_counter += 1
            return x + 1.0

        self.fn = gpu_add_one
        self.backend = gpu_add_one._cuda_graph_backend

    def _make_input(self, bsz, fill=0.0):
        return paddle.full([bsz, 16], fill_value=fill, dtype="float32")

    # ------------------------------------------------------------------ A1
    def test_A1_warmup_runs_correctly(self):
        """Warmup phase executes eagerly and returns correct results."""
        x = self._make_input(4, fill=2.0)
        entry = self.backend.entries[4]

        # First call: warmup #1
        out = self.fn(x)
        self.assertEqual(entry.num_finished_warmup, 1)
        self.assertFalse(entry.captured)
        np.testing.assert_allclose(out.numpy(), np.full([4, 16], 3.0, dtype="float32"))

        # Second call: warmup #2
        out = self.fn(x)
        self.assertEqual(entry.num_finished_warmup, 2)
        self.assertFalse(entry.captured)
        np.testing.assert_allclose(out.numpy(), np.full([4, 16], 3.0, dtype="float32"))

    # ------------------------------------------------------------------ A2
    def test_A2_capture_triggers_after_warmup(self):
        """Third call triggers capture; result must still match eager."""
        x = self._make_input(4, fill=5.0)
        entry = self.backend.entries[4]

        # two warmups
        self.fn(x)
        self.fn(x)
        self.assertFalse(entry.captured)

        # third call: capture
        out = self.fn(x)
        self.assertTrue(entry.captured)
        self.assertIsNotNone(entry.cuda_graph)
        np.testing.assert_allclose(out.numpy(), np.full([4, 16], 6.0, dtype="float32"))

    # ------------------------------------------------------------------ A3
    def test_A3_replay_returns_correct_result(self):
        """After capture, replay returns correct values for varying inputs."""
        entry = self.backend.entries[4]

        # force through warmup + capture
        x_warmup = self._make_input(4, fill=1.0)
        self.fn(x_warmup)
        self.fn(x_warmup)
        self.fn(x_warmup)
        self.assertTrue(entry.captured)

        # replay with new input values
        for fill_val in [10.0, 20.0, 99.0]:
            x = self._make_input(4, fill=fill_val)
            out = self.fn(x)
            expected = np.full([4, 16], fill_val + 1.0, dtype="float32")
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, err_msg=f"Replay mismatch at fill={fill_val}")

    # ------------------------------------------------------------------ A4
    def test_A4_independent_entries_per_size(self):
        """Different batch sizes use independent capture entries."""
        # drive bsz=4 through full lifecycle
        x4 = self._make_input(4, fill=1.0)
        for _ in range(3):
            self.fn(x4)

        # bsz=8 should still be at warmup #0
        entry8 = self.backend.entries[8]
        self.assertEqual(entry8.num_finished_warmup, 0)
        self.assertFalse(entry8.captured)

        # drive bsz=8 through warmup → capture
        x8 = self._make_input(8, fill=2.0)
        for _ in range(3):
            out8 = self.fn(x8)

        self.assertTrue(entry8.captured)
        np.testing.assert_allclose(out8.numpy(), np.full([8, 16], 3.0, dtype="float32"))

    # ------------------------------------------------------------------ A5
    def test_A5_fallback_for_oversized_batch(self):
        """batch_size > max capture_size falls back to eager execution."""
        x = self._make_input(32, fill=7.0)  # 32 > max(8)
        out = self.fn(x)
        np.testing.assert_allclose(out.numpy(), np.full([32, 16], 8.0, dtype="float32"))

    # ------------------------------------------------------------------ A6
    def test_A6_multi_tensor_args_non_tensor_passthrough(self):
        """Non-Tensor args are passed through unchanged; second Tensor is also buffered."""
        from fastdeploy.model_executor.graph_optimization.cuda_graph_fn import (
            cuda_graph_fn,
        )

        @cuda_graph_fn(capture_sizes=[4], num_warmups=1, size_from=0)
        def fn_with_scale(x: paddle.Tensor, scale: float, y: paddle.Tensor) -> paddle.Tensor:
            return x * scale + y

        x = paddle.ones([4, 8], dtype="float32")
        y = paddle.full([4, 8], 10.0, dtype="float32")

        # warmup
        out = fn_with_scale(x, 2.0, y)
        # capture
        out = fn_with_scale(x, 2.0, y)
        # replay
        out = fn_with_scale(x, 2.0, y)
        np.testing.assert_allclose(out.numpy(), np.full([4, 8], 12.0, dtype="float32"))


# ---------------------------------------------------------------------------
# Section B: _compute_sampling_mask correctness
# ---------------------------------------------------------------------------


@unittest.skipIf(SKIP_NO_CUDA, SKIP_REASON)
class TestComputeSamplingMask(unittest.TestCase):
    """
    Tests for the full _compute_sampling_mask function including CUDA graph
    lifecycle (warmup → capture → replay) with correctness checks against
    the reference eager implementation.
    """

    # Import once, avoid re-importing per test
    @classmethod
    def setUpClass(cls):
        from fastdeploy.model_executor.layers.sample.sampler import (
            _compute_sampling_mask,
        )

        cls.fn = staticmethod(_compute_sampling_mask)

    def _make_probs(self, bsz, vocab_size, seed=42):
        """Return softmax-normalised random probs [bsz, vocab_size]."""
        paddle.seed(seed)
        logits = paddle.randn([bsz, vocab_size])
        return F.softmax(logits, axis=-1)

    def _make_top_p(self, bsz, value):
        return paddle.full([bsz, 1], fill_value=value, dtype="float32")

    # ------------------------------------------------------------------ B1
    def test_B1_correctness_standard_top_p(self):
        """Results match the reference eager implementation for top_p=0.9."""
        VOCAB = 1000
        for bsz in [1, 3, 7]:
            with self.subTest(bsz=bsz):
                probs = self._make_probs(bsz, VOCAB, seed=bsz)
                top_p = self._make_top_p(bsz, 0.9)
                got = self.fn(probs, top_p)
                ref = _reference_compute_sampling_mask(probs, top_p)
                self.assertTrue(_arrays_equal(got, ref), f"bsz={bsz}: got={got}, ref={ref}")

    # ------------------------------------------------------------------ B2
    def test_B2_top_p_one_keeps_all_tokens(self):
        """top_p = 1.0 must retain the full vocabulary for every request."""
        VOCAB = 200
        BSZ = 2
        probs = self._make_probs(BSZ, VOCAB)
        top_p = self._make_top_p(BSZ, 1.0)
        result = self.fn(probs, top_p)
        for i, row in enumerate(result):
            self.assertEqual(len(row), VOCAB, f"row {i}: expected {VOCAB} tokens, got {len(row)}")

    # ------------------------------------------------------------------ B3
    def test_B3_top_p_very_small_keeps_one_token(self):
        """top_p close to 0 should retain only the highest-probability token."""
        VOCAB = 500
        BSZ = 4
        probs = self._make_probs(BSZ, VOCAB, seed=7)
        # Use a tiny top_p so only the top-1 token satisfies the threshold.
        # The condition is  (cum_p - p_j) < top_p  which for j=0 is  0 < top_p.
        # With top_p = 1e-9 only the first sorted token passes.
        top_p = self._make_top_p(BSZ, 1e-9)
        result = self.fn(probs, top_p)
        for i, row in enumerate(result):
            self.assertEqual(len(row), 1, f"row {i}: expected 1 token with tiny top_p, got {len(row)}")

    # ------------------------------------------------------------------ B4
    def test_B4_multi_size_warmup_capture_replay_consistency(self):
        """
        Verify that _compute_sampling_mask returns correct results across
        multiple batch sizes and multiple calls (warmup → capture → replay).

        _compute_sampling_mask_gpu is decorated with @cuda_graph_fn, so the
        first num_warmups calls per batch size run eagerly, the next call
        triggers capture, and all subsequent calls execute via graph replay.
        This test drives each batch size through the full lifecycle and checks
        that the returned values always match the reference eager implementation.
        """
        VOCAB = 300
        TOTAL_CALLS = 8

        for bsz in [1, 2, 4, 8]:
            with self.subTest(bsz=bsz):
                for call_idx in range(TOTAL_CALLS):
                    paddle.seed(call_idx * 17 + bsz * 3)
                    probs = F.softmax(paddle.randn([bsz, VOCAB]), axis=-1)
                    top_p = paddle.full([bsz, 1], fill_value=0.85, dtype="float32")

                    got = self.fn(probs, top_p)
                    ref = _reference_compute_sampling_mask(probs, top_p)

                    self.assertTrue(_arrays_equal(got, ref), f"bsz={bsz} call={call_idx}: result mismatch")

    # ------------------------------------------------------------------ B5
    def test_B5_spec_decode_scenario(self):
        """
        Simulate the speculative-decode call site in sampler.py:931.

        accept_top_p = sampling_metadata.top_p[:real_bsz]
                         .squeeze(1)
                         .repeat_interleave(accept_nums)
                         .unsqueeze(1)

        The resulting accept_top_p has shape [total_accepted, 1] which is the
        effective bsz passed to _compute_sampling_mask.
        """
        VOCAB = 400
        real_bsz = 4
        # Suppose each request accepted 2 tokens on average
        accept_nums = paddle.to_tensor([2, 1, 3, 2], dtype="int64")
        total_accepted = int(accept_nums.sum().item())  # 8

        paddle.seed(99)
        base_top_p = paddle.full([real_bsz, 1], fill_value=0.9, dtype="float32")
        accept_top_p = base_top_p[:real_bsz].squeeze(1).repeat_interleave(accept_nums).unsqueeze(1)
        # accept_top_p.shape == [8, 1]
        self.assertEqual(accept_top_p.shape, [total_accepted, 1])

        target_probs = F.softmax(paddle.randn([total_accepted, VOCAB]), axis=-1)

        got = self.fn(target_probs, accept_top_p)
        ref = _reference_compute_sampling_mask(target_probs, accept_top_p)

        self.assertEqual(len(got), total_accepted)
        self.assertTrue(_arrays_equal(got, ref), "spec-decode scenario: CUDA-graph vs eager mismatch")


# ---------------------------------------------------------------------------
# Section C: State isolation between test instances
# ---------------------------------------------------------------------------


@unittest.skipIf(SKIP_NO_CUDA, SKIP_REASON)
class TestCudaGraphFnStateIsolation(unittest.TestCase):
    """
    Verify that two independently decorated functions do not share CUDA graph
    state (separate backends, separate entries, separate memory pools).
    """

    def test_C1_two_decorated_functions_are_independent(self):
        from fastdeploy.model_executor.graph_optimization.cuda_graph_fn import (
            cuda_graph_fn,
        )

        @cuda_graph_fn(capture_sizes=[4], num_warmups=1)
        def fn_a(x: paddle.Tensor) -> paddle.Tensor:
            return x * 2.0

        @cuda_graph_fn(capture_sizes=[4], num_warmups=1)
        def fn_b(x: paddle.Tensor) -> paddle.Tensor:
            return x * 3.0

        x = paddle.ones([4, 8], dtype="float32")

        # warmup both
        fn_a(x)
        fn_b(x)
        # capture both
        out_a = fn_a(x)
        out_b = fn_b(x)

        np.testing.assert_allclose(out_a.numpy(), np.full([4, 8], 2.0, dtype="float32"))
        np.testing.assert_allclose(out_b.numpy(), np.full([4, 8], 3.0, dtype="float32"))

        # verify completely separate backends
        self.assertIsNot(fn_a._cuda_graph_backend, fn_b._cuda_graph_backend)
        self.assertIsNot(
            fn_a._cuda_graph_backend.entries[4].cuda_graph,
            fn_b._cuda_graph_backend.entries[4].cuda_graph,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
