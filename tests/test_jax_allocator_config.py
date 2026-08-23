"""The production JAX allocator defaults (review finding JAX-02).

``core.jax_config`` used to default to ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``,
the raw ``cudaMalloc``/``cudaFree`` allocator.  Measured on an H100 NVL against
the shipped real spectral likelihood (1,067,946 injections, 259 events, Dynesty
value-only, ``off:off`` blocking, 20 warm repetitions, clean process): 23.0
ms/call under ``platform`` versus 13.7 ms/call under BFC (``default``), i.e.
1.68x, and ``platform`` reports NO memory statistics at all — so the auto
block-size planner also lost the allocator view every published peak-memory
constant was calibrated against.

The default is now BFC with preallocation still OFF (the block-size planner
sizes against *free* memory, which a preallocating allocator would report as
fully consumed).  An EXPLICIT environment override must still win, in both
directions, and is what these tests pin.

The subprocess cases are the ones that matter: the env has to be set before the
first JAX import, so only a cold process can attest the real ordering.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from darksirens.core.jax_config import (
    DEFAULT_XLA_ALLOCATOR,
    DEFAULT_XLA_PREALLOCATE,
    configure_jax_runtime,
)

_ALLOC = "XLA_PYTHON_CLIENT_ALLOCATOR"
_PREALLOC = "XLA_PYTHON_CLIENT_PREALLOCATE"


def test_default_allocator_is_bfc_not_platform():
    """BFC: 1.68x faster per call, and the only allocator that reports stats."""
    assert DEFAULT_XLA_ALLOCATOR == "default"
    assert DEFAULT_XLA_PREALLOCATE == "false"


@pytest.fixture()
def _restored_precision():
    """Keep ``configure_jax_runtime``'s global JAX updates out of other tests."""
    import jax

    previous = jax.config.jax_default_matmul_precision
    yield
    jax.config.update("jax_default_matmul_precision", previous)


def test_configure_sets_the_bfc_defaults(monkeypatch, _restored_precision):
    monkeypatch.delenv(_ALLOC, raising=False)
    monkeypatch.delenv(_PREALLOC, raising=False)
    configure_jax_runtime()
    assert os.environ[_ALLOC] == "default"
    assert os.environ[_PREALLOC] == "false"


@pytest.mark.parametrize("allocator", ["platform", "bfc", "cuda_async"])
def test_explicit_allocator_override_is_honored(monkeypatch, _restored_precision,
                                                allocator):
    """A user who exports an allocator keeps it; the default only fills a gap."""
    monkeypatch.setenv(_ALLOC, allocator)
    monkeypatch.setenv(_PREALLOC, "true")
    configure_jax_runtime()
    assert os.environ[_ALLOC] == allocator
    assert os.environ[_PREALLOC] == "true"


def _cold_env(**overrides):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    for key in (_ALLOC, _PREALLOC):
        env.pop(key, None)
    env.update(overrides)
    return env


_PROBE = (
    "import os, sys;"
    " mod = sys.argv[1];"
    " __import__(mod);"
    " print(os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR'),"
    "       os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))"
)


def _cold_probe(module, **env_overrides):
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        check=True, capture_output=True, text=True, env=_cold_env(**env_overrides),
    ).stdout
    return out.strip().splitlines()[-1].split()


@pytest.mark.parametrize("module", ["darksirens.gw.utils"])
def test_cold_import_sets_bfc(module):
    """``gw.utils`` sets the env at import time and must agree with jax_config."""
    assert _cold_probe(module) == ["default", "false"]


@pytest.mark.parametrize("module", ["darksirens.gw.utils"])
def test_cold_import_honors_explicit_platform(module):
    assert _cold_probe(module, XLA_PYTHON_CLIENT_ALLOCATOR="platform") == \
        ["platform", "false"]


def test_jax_config_is_importable_without_importing_jax():
    """The env must be settable BEFORE JAX loads, so this module cannot pull it in."""
    code = (
        "import sys;"
        " import darksirens.core.jax_config;"
        " print('jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
        env=_cold_env(),
    ).stdout
    assert out.strip().splitlines()[-1] == "False"
