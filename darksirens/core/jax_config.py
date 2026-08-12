"""JAX runtime configuration helpers."""

from __future__ import annotations

import os


def configure_jax_runtime() -> None:
    """Configure JAX memory and precision defaults used by the inference CLI.

    Environment variables are set before importing JAX in callers that invoke
    this helper at module startup.  ``darksirens.gw.utils`` sets the same two
    knobs to the same values at import time (``setdefault``, so whichever runs
    first wins) because it must not enable x64 itself; keep the two in sync.

    Preallocation stays OFF deliberately: the block-size planner sizes the
    selection/PE blocks against the device's *free* memory
    (``likelihood/block_sizing.probe_device_memory_bytes``), which a
    preallocating allocator would report as fully consumed.  There used to be a
    ``mem_fraction`` argument setting ``XLA_PYTHON_CLIENT_MEM_FRACTION``; that
    variable only takes effect on the PREALLOCATING path, so it capped nothing
    and is gone rather than advertising a control that does nothing.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
