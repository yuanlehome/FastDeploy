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
Lightweight NVTX instrumentation helpers for Nsight Systems profiling.

Usage:
    from fastdeploy.metrics.nvtx import nvtx_range, nvtx_mark

    with nvtx_range("decode.step", bs=32):
        ...

    nvtx_mark("kv_cache.alloc")

Enable via environment variable:
    FD_ENABLE_NVTX=1
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Lazy, one-shot initialisation — resolved on first call, cached forever.
# No lock needed: worst case two threads both evaluate to the same bool.
# ---------------------------------------------------------------------------
_enabled: bool | None = None
_nvtx_mod = None  # paddle.device.cuda.nvtx (or None)


def _init():
    global _enabled, _nvtx_mod
    _enabled = os.environ.get("FD_ENABLE_NVTX", "0") == "1"
    if _enabled:
        try:
            import paddle.device.cuda  # noqa: F401

            # Paddle ≥ 2.6 exposes nvtx helpers
            _nvtx_mod = paddle.device.cuda
            # Quick sanity check that the API is actually available
            if not hasattr(_nvtx_mod, "nvtx_range_push"):
                _nvtx_mod = None
                _enabled = False
        except Exception:
            _nvtx_mod = None
            _enabled = False


def _is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _init()
    return _enabled  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def nvtx_range(name: str, **kwargs):
    """Context-manager that wraps a code region in an NVTX range.

    Extra *kwargs* are appended as ``key=value`` pairs to the range name
    so they show up in Nsight Systems hover text (e.g. ``bs=64``).

    When NVTX is disabled the overhead is a single boolean check.
    """
    if not _is_enabled():
        yield
        return

    if kwargs:
        suffix = " ".join(f"{k}={v}" for k, v in kwargs.items())
        label = f"{name} [{suffix}]"
    else:
        label = name

    _nvtx_mod.nvtx_range_push(label)
    try:
        yield
    finally:
        _nvtx_mod.nvtx_range_pop()


def nvtx_mark(name: str):
    """Emit a single NVTX marker (instant event)."""
    if not _is_enabled():
        return
    # Paddle does not expose nvtx_mark, emulate with zero-width push/pop
    _nvtx_mod.nvtx_range_push(name)
    _nvtx_mod.nvtx_range_pop()
