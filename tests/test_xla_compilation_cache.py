"""The opt-in XLA persistent compilation cache (perf candidate C20).

A production dark-siren startup performs ~350 XLA compilations totalling ~19 s
of its ~38 s of setup, and JAX already has a persistent cache for exactly that
— but in the pinned environment the cache is DEAD: ``jax._src.path`` probes for
``etils.epath`` with ``importlib.util.find_spec`` (which does not execute the
module), the probe says yes, and the import then fails on
``etils/epath/resource_utils.py``'s ``import importlib_resources`` (absent
here).  Every cache read and write raises, JAX warns once and compiles anyway,
and the cache directory stays empty.

:func:`darksirens.core.jax_config._install_local_path_shim` pre-sets the module
globals so the failing ``__getattr__`` never runs; measured on an H100 NVL with
the 259-event production configuration, that turns 44.14 s of process wall into
24.54 s warm (-19.60 s) with the log-likelihood at all 8 benchmark draws
byte-equal.  These tests pin the two things that make it safe: the feature is
OFF unless ``DARKSIRENS_XLA_CACHE`` is set, and a cache HIT returns a
bit-identical value.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from darksirens.core.jax_config import (
    DEFAULT_XLA_CACHE_MAX_SIZE_BYTES,
    XLA_CACHE_ENV,
    _install_local_path_shim,
    configure_jax_runtime,
    enable_persistent_compilation_cache,
    resolve_xla_cache_dir,
)


@pytest.fixture()
def _restored_jax_config():
    """Keep the global JAX updates these tests make out of every other test."""
    import jax

    keys = (
        "jax_default_matmul_precision",
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_compilation_cache_max_size",
    )
    previous = {k: getattr(jax.config, k) for k in keys}
    yield
    for key, value in previous.items():
        jax.config.update(key, value)


def test_cache_is_off_when_the_env_var_is_unset(monkeypatch, _restored_jax_config):
    """The opt-in is the whole safety argument: no env var, no behaviour change."""
    import jax

    jax.config.update("jax_compilation_cache_dir", None)
    monkeypatch.delenv(XLA_CACHE_ENV, raising=False)
    configure_jax_runtime()
    assert jax.config.jax_compilation_cache_dir is None


def test_cache_is_off_when_the_env_var_is_blank(monkeypatch, _restored_jax_config):
    import jax

    jax.config.update("jax_compilation_cache_dir", None)
    monkeypatch.setenv(XLA_CACHE_ENV, "   ")
    configure_jax_runtime()
    assert jax.config.jax_compilation_cache_dir is None


def test_configure_enables_the_cache_and_keeps_the_x64_contract(
    monkeypatch, tmp_path, _restored_jax_config
):
    """The shim must not disturb the precision configuration it runs beside."""
    import jax

    monkeypatch.setenv(XLA_CACHE_ENV, str(tmp_path / "xla"))
    configure_jax_runtime()
    assert jax.config.jax_compilation_cache_dir == resolve_xla_cache_dir(
        str(tmp_path / "xla")
    )
    assert os.path.isdir(jax.config.jax_compilation_cache_dir)
    # min_compile_time 0.0, NOT jax's 1.0 s default: the ~340 small eager
    # build-time compiles are each well under a second and are exactly the ones
    # the default discards (measured 11.66 s of saving at 1.0 s vs 19.60 s).
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0
    # Any max_size other than -1 is what puts a filelock around LRUCache.put;
    # with -1 the entry is written with a bare write_bytes and two concurrent
    # cold starts can leave a torn entry that put() never repairs.
    assert jax.config.jax_compilation_cache_max_size == DEFAULT_XLA_CACHE_MAX_SIZE_BYTES
    assert DEFAULT_XLA_CACHE_MAX_SIZE_BYTES != -1
    assert jax.config.jax_enable_x64 is True
    assert jax.config.jax_default_matmul_precision == "highest"


def test_shim_leaves_a_usable_local_path_type():
    """``jax._src.path.Path`` must be callable, and consistent with the flag
    ``lru_cache`` uses to pick ``stat().st_size`` over epath's ``stat().length``."""
    import pathlib

    import jax._src.path as jax_path

    assert _install_local_path_shim() is True
    assert jax_path.Path(os.sep).exists()
    if jax_path.Path is pathlib.Path:
        assert jax_path.epath_installed is False


def test_shim_is_idempotent():
    assert _install_local_path_shim() is True
    assert _install_local_path_shim() is True


def test_enable_degrades_to_off_on_an_unusable_root(tmp_path, _restored_jax_config):
    """A cache fault must cost a compile, never the run."""
    import jax

    jax.config.update("jax_compilation_cache_dir", None)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    assert enable_persistent_compilation_cache(str(blocker)) is None
    assert jax.config.jax_compilation_cache_dir is None


def test_cache_dir_is_split_per_host_and_jaxlib(tmp_path):
    """One scratch root can be shared: the writers are kept apart by directory."""
    import platform

    import jaxlib

    resolved = resolve_xla_cache_dir(str(tmp_path))
    assert os.path.dirname(resolved) == str(tmp_path)
    leaf = os.path.basename(resolved)
    assert leaf.startswith(platform.node().split(".")[0])
    assert leaf.endswith(f"jaxlib{jaxlib.version.__version__}")


_HIT_PROBE = """
import os, sys
os.environ["DARKSIRENS_XLA_CACHE"] = sys.argv[1]
from darksirens.core.jax_config import configure_jax_runtime, resolve_xla_cache_dir
configure_jax_runtime()
import jax, jax.numpy as jnp
import jax._src.compiler as compiler
n = [0]
orig = compiler.backend_compile
def counting(*a, **k):
    n[0] += 1
    return orig(*a, **k)
compiler.backend_compile = counting
f = jax.jit(lambda x: jnp.log(jnp.sum(jnp.exp(x * 1.0000001))))
v = f(jnp.arange(5.0))
print("RESULT %d %r %d" % (n[0], float(v),
                           len(os.listdir(resolve_xla_cache_dir(sys.argv[1])))))
"""


def _hit_probe(cache_root):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-c", _HIT_PROBE, str(cache_root)],
        check=True, capture_output=True, text=True, env=env,
    ).stdout
    line = [x for x in out.splitlines() if x.startswith("RESULT ")][-1].split()
    return int(line[1]), float(line[2]), int(line[3])


def test_a_cache_hit_is_bit_identical(tmp_path):
    """Cold writes entries, warm compiles NOTHING and returns the same bits.

    This is the accuracy argument in executable form: the cache stores and
    returns the SAME serialized executable ``backend_compile`` produced, keyed
    on the HLO module plus the jaxlib version, backend, compile options and
    device kind, so a mismatched entry cannot be served.
    """
    root = tmp_path / "xla"
    cold_compiles, cold_value, cold_entries = _hit_probe(root)
    assert cold_compiles > 0
    # Nonzero entries is itself the proof the shim works: without it JAX warns
    # and leaves the directory completely empty.
    assert cold_entries > 0
    warm_compiles, warm_value, _ = _hit_probe(root)
    assert warm_compiles == 0
    assert warm_value == cold_value
