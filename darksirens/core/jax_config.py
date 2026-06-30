"""JAX runtime configuration helpers."""

from __future__ import annotations

import os


def configure_jax_runtime(mem_fraction: str = "0.95") -> None:
    """Configure JAX memory and precision defaults used by the inference CLI.

    Environment variables are set before importing JAX in callers that invoke
    this helper at module startup.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", mem_fraction)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
