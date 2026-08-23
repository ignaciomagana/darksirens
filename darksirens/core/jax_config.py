"""JAX runtime configuration helpers."""

from __future__ import annotations

import os

#: Default device allocator.  ``"default"`` is XLA's BFC (caching) allocator;
#: ``"platform"`` is the raw ``cudaMalloc``/``cudaFree`` one.  Named so callers
#: and tests can assert the production default without hard-coding the string.
DEFAULT_XLA_ALLOCATOR = "default"

#: Default preallocation setting.  OFF; see :func:`configure_jax_runtime`.
DEFAULT_XLA_PREALLOCATE = "false"


def configure_jax_runtime() -> None:
    """Configure JAX memory and precision defaults used by the inference CLI.

    Environment variables are set before importing JAX in callers that invoke
    this helper at module startup.  ``darksirens.gw.utils`` sets the same two
    knobs to the same values at import time (``setdefault``, so whichever runs
    first wins) because it must not enable x64 itself; keep the two in sync.
    ``setdefault`` also means an EXPLICIT environment override always wins: a
    user who exports ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` still gets the
    platform allocator.

    Preallocation stays OFF deliberately: the block-size planner sizes the
    selection/PE blocks against the device's *free* memory
    (``likelihood/block_sizing.probe_device_memory_bytes``), which a
    preallocating allocator would report as fully consumed.  There used to be a
    ``mem_fraction`` argument setting ``XLA_PYTHON_CLIENT_MEM_FRACTION``; that
    variable only takes effect on the PREALLOCATING path, so it capped nothing
    and is gone rather than advertising a control that does nothing.

    The allocator is BFC (``"default"``), NOT ``"platform"`` (2026-08-23).  The
    platform allocator hands every transient buffer straight to
    ``cudaMalloc``/``cudaFree``, which on the shipped real spectral likelihood
    (1,067,946 injections, 259 events, Dynesty value-only, off:off blocking, 20
    warm repetitions, clean process on an H100 NVL) measured **23.0 ms per call
    against BFC's 13.7 ms** — 1.68x, i.e. BFC removes ~40% of per-call wall time,
    compounded over 1e5-1e6 sampler evaluations.  It also reports NO memory
    statistics: ``device.memory_stats()`` returns ``None`` under it, so the
    auto block-size planner lost the allocator-used/limit view its constants
    were calibrated against and fell back to ``nvidia-smi`` (which remains the
    fallback, for shared boxes and for an explicit ``platform`` override).
    Every published peak-memory constant in ``likelihood/block_sizing`` was
    measured under BFC/preallocate=false, so this default is also the one the
    calibration ran under.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_XLA_ALLOCATOR)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
