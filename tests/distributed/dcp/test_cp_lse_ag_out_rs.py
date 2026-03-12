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
Unit tests for cp_lse_ag_out_rs in fastdeploy/distributed/dcp_comm.py.

Each test launches cp_lse_ag_out_rs_worker.py via paddle.distributed.launch
with N GPUs, collects per-rank .npz output, and asserts correctness.

Usage:
    # Run from FastDeploy root (requires ≥2 GPUs):
    pytest tests/distributed/dcp/test_cp_lse_ag_out_rs.py -v

    # Specify GPU count explicitly via env:
    DCP_TEST_GPUS=0,1,2,3 pytest tests/distributed/dcp/test_cp_lse_ag_out_rs.py -v
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "cp_lse_ag_out_rs_worker.py")

# GPUs to use, default "0,1".  Override via DCP_TEST_GPUS env var.
_DEFAULT_GPUS = "0,1"
GPUS = os.environ.get("DCP_TEST_GPUS", _DEFAULT_GPUS)
N_GPUS = len(GPUS.split(","))


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------


def _launch_worker(test_case: str, output_dir: str, timeout: int = 120) -> None:
    """
    Launch cp_lse_ag_out_rs_worker.py via paddle.distributed.launch.
    Raises AssertionError on non-zero exit.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = FD_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["TEST_CASE"] = test_case
    env["OUTPUT_DIR"] = output_dir

    command = [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        "--gpus",
        GPUS,
        WORKER_SCRIPT,
    ]

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(f"[{test_case}] Worker timed out after {timeout}s")

    if proc.returncode != 0:
        raise AssertionError(
            f"[{test_case}] Worker failed (rc={proc.returncode})\n" f"STDOUT:\n{stdout}\nSTDERR:\n{stderr[-2000:]}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCpLseAgOutRs:
    """Tests for cp_lse_ag_out_rs (AllGather-LSE + correct + ReduceScatter)."""

    def test_basic_correctness(self):
        """
        Output must match pure-numpy reference for each rank's head slice.
        Validates both the attention output correction and the head-dim slice.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("basic", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"basic_rank{r}.npz"))
                np.testing.assert_allclose(
                    data["result_out"],
                    data["ref_out"],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"rank {r}: output mismatch vs numpy reference",
                )
                np.testing.assert_allclose(
                    data["result_lse"],
                    data["ref_lse"],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"rank {r}: LSE mismatch vs numpy reference",
                )

    def test_return_lse_false(self):
        """
        When return_lse=False the return value is a plain Tensor with shape
        [B, H//N, D].  Worker asserts the type; we check the file was written.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("return_lse_false", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"return_lse_false_rank{r}.npz"))
                assert "out" in data, f"rank {r}: missing output tensor"

    def test_output_shape(self):
        """
        For B=6, H=16, D=32 with N ranks:
          out  shape should be [6, 16//N, 32]
          lse  shape should be [6, 16//N]
        Worker verifies shapes internally.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("output_shape", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"output_shape_rank{r}.npz"))
                assert data["ok"][0] == 1, f"rank {r}: shape check failed"

    def test_inf_nan_lse(self):
        """
        Inject inf/nan into rank-0 LSE.  Output must be finite (no NaN propagation).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("inf_lse", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"inf_lse_rank{r}.npz"))
                assert np.isfinite(data["out"]).all(), f"rank {r}: output contains NaN/Inf"
                assert np.isfinite(data["lse"]).all(), f"rank {r}: LSE contains NaN/Inf"

    def test_single_batch(self):
        """B=1 edge case — shape and execution should succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("single_batch", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"single_batch_rank{r}.npz"))
                assert data["ok"][0] == 1, f"rank {r}: single-batch check failed"

    def test_non_pow2_head_dim(self):
        """Head dim D=96 (non power-of-2) should work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("non_pow2_head_dim", tmpdir)

            for r in range(N_GPUS):
                data = np.load(os.path.join(tmpdir, f"non_pow2_head_dim_rank{r}.npz"))
                assert data["ok"][0] == 1, f"rank {r}: non-pow2 head dim check failed"

    def test_head_partition_non_overlap(self):
        """
        All ranks together must cover every head exactly once.
        Verified by collecting result_out from all ranks and reconstructing
        the full head tensor (using the reference), then checking each
        rank's slice equals the corresponding columns of the global output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            _launch_worker("basic", tmpdir)

            # Load per-rank result_out: [B, H_local, D]
            rank_outs = [np.load(os.path.join(tmpdir, f"basic_rank{r}.npz"))["result_out"] for r in range(N_GPUS)]
            # All slices must have the same shape
            shapes = [o.shape for o in rank_outs]
            assert len(set(shapes)) == 1, f"rank outputs have different shapes: {shapes}"

            # Concatenate along head dim → full [B, H, D]
            full_out = np.concatenate(rank_outs, axis=1)

            # Each rank's ref_out should equal the corresponding slice of full_out
            H_local = rank_outs[0].shape[1]
            for r in range(N_GPUS):
                ref = np.load(os.path.join(tmpdir, f"basic_rank{r}.npz"))["ref_out"]
                expected = full_out[:, H_local * r : H_local * (r + 1), :]
                np.testing.assert_allclose(
                    ref,
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"rank {r}: head slice does not match global output",
                )


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(f"cp_lse_ag_out_rs tests  |  GPUs: {GPUS}  |  N={N_GPUS}")
    print("=" * 70)

    test_fns = [
        ("basic correctness", TestCpLseAgOutRs().test_basic_correctness),
        ("return_lse=False", TestCpLseAgOutRs().test_return_lse_false),
        ("output shape", TestCpLseAgOutRs().test_output_shape),
        ("inf/NaN LSE", TestCpLseAgOutRs().test_inf_nan_lse),
        ("single batch (B=1)", TestCpLseAgOutRs().test_single_batch),
        ("non-pow2 head dim (D=96)", TestCpLseAgOutRs().test_non_pow2_head_dim),
        ("head partition non-overlap", TestCpLseAgOutRs().test_head_partition_non_overlap),
    ]

    passed = failed = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    sys.exit(1 if failed else 0)
