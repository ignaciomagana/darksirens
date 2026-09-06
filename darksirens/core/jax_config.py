"""JAX runtime configuration helpers."""

from __future__ import annotations

import os

#: Environment variable naming the root of the on-disk XLA persistent
#: compilation cache.  Unset or empty (the default) leaves the cache OFF, so
#: nothing about a production run changes unless the user opts in.
XLA_CACHE_ENV = "DARKSIRENS_XLA_CACHE"

#: Disk cap for the persistent compilation cache, in bytes.  Any value other
#: than ``-1`` also switches ``jax._src.lru_cache.LRUCache`` onto its
#: ``filelock`` path, which is the ONLY thing that serialises concurrent
#: writers: with the ``-1`` default the entry is written with a bare
#: ``write_bytes`` (no temp+rename) and two processes cold-starting the same
#: configuration at once can leave a torn file that ``put()`` never repairs
#: (it returns early when the path exists), so that key would compile forever.
#: A torn entry can only cost a compile, never a wrong executable, but the cap
#: is what keeps the benefit.  4.3 MB covers the production dark-siren
#: configuration and 1.8 MB the spectral one, so this is a very loose bound.
DEFAULT_XLA_CACHE_MAX_SIZE_BYTES = 2 * 1024**3

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

    cache_root = os.environ.get(XLA_CACHE_ENV, "").strip()
    if cache_root:
        _install_local_path_shim()

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    if cache_root:
        enable_persistent_compilation_cache(cache_root)


def _install_local_path_shim() -> bool:
    """Make JAX's persistent compilation cache usable in the pinned environment.

    ``jax._src.path`` decides whether ``etils.epath`` is available with
    ``importlib.util.find_spec("etils.epath")``, which does NOT execute the
    module.  In the pinned environment that probe says yes and the import then
    fails: ``etils/epath/resource_utils.py`` does ``import importlib_resources``
    and that package is absent.  ``jax._src.path.__getattr__("Path")`` therefore
    raises on every cache init/get/put, ``jax._src.compiler`` warns once
    ("Error reading persistent compilation cache entry ... No module named
    'importlib_resources'") and silently compiles, leaving the cache directory
    empty — measured: 358 compiles / 19.6 s both cold AND warm, 0 entries
    written.

    Pre-setting the module globals so the failing ``__getattr__`` never runs is
    enough for a LOCAL cache directory (``LRUCache`` only demands ``epath`` for
    a non-local path).  It touches no third-party namespace, unlike aliasing
    ``importlib.resources`` under the name ``importlib_resources``, and both
    forms were measured to produce the SAME cache keys.

    Returns whether the shim is in place.  Failure is not an error: the cache
    then stays dead exactly as it is today.
    """
    try:
        import jax._src.path as _jax_path
    except Exception:
        return False
    try:
        _jax_path.Path  # noqa: B018 - triggers the epath import that may fail
        return True
    except Exception:
        pass
    try:
        import pathlib

        _jax_path.Path = pathlib.Path
        # ``lru_cache._evict_if_needed`` branches on this flag to choose between
        # ``stat().st_size`` (pathlib) and ``stat().length`` (epath); leaving it
        # True with a pathlib ``Path`` makes every cache WRITE raise
        # "AttributeError: 'os.stat_result' object has no attribute 'length'".
        _jax_path.epath_installed = False
        return True
    except Exception:
        return False


def resolve_xla_cache_dir(root: str) -> str:
    """The per-host, per-``jaxlib`` subdirectory of ``root`` to cache into.

    The cache key already covers the HLO module, the jaxlib version, the
    backend, the compile options and the device kind, so a mismatched entry can
    never be served.  The split is about WRITERS, not readers: this scratch
    root can be shared between boxes and between environments, and a directory
    per host and jaxlib build keeps unrelated jobs out of each other's entries
    on top of the ``filelock`` locking the size cap turns on.
    """
    import platform

    try:
        import jaxlib

        version = jaxlib.version.__version__
    except Exception:  # pragma: no cover - jaxlib always exposes this
        version = "unknown"
    host = platform.node().split(".")[0] or "unknown"
    return os.path.join(os.path.expanduser(root), f"{host}-jaxlib{version}")


def enable_persistent_compilation_cache(
    root: str, max_size_bytes: int = DEFAULT_XLA_CACHE_MAX_SIZE_BYTES
) -> str | None:
    """Point JAX's persistent compilation cache at ``root`` and return the dir.

    A production dark-siren startup performs ~350 XLA compilations totalling
    ~19 s: ~100 building module-level constants at import, ~200 in the eager
    build-time pin/KDE steps inside ``make_likelihood``, and ~33 on the first
    likelihood call.  Cached, all of them are served from disk.  Measured on an
    H100 NVL with the 259-event production configuration (PROD_ARGS,
    ``--n-calls 20``): process wall 44.14 s with no cache versus 24.54 s warm
    (-19.60 s, -44%), first call 9.95 s -> 1.78 s, cold-run penalty +0.73 s for
    676 entries / 4.3 MB.  Spectral: 12.92 s -> 7.49 s (-5.43 s), 312 entries /
    1.8 MB.  Per-call time is unchanged (59.6 -> 59.6 ms production, 13.8 ->
    13.9 ms spectral, both inside the run-to-run spread) and the log-likelihood
    at all 8 benchmark draws is byte-equal: the cache returns the SAME
    serialized executable ``backend_compile`` produced.

    ``jax_persistent_cache_min_compile_time_secs`` MUST be 0.0 rather than
    JAX's 1.0 s default: the ~340 small eager build-time compiles are each well
    under a second and are exactly the ones the default throws away — measured
    11.66 s of saving at 1.0 s against 19.60 s at 0.0.

    Cache errors are left non-fatal (``jax_raise_persistent_cache_errors``
    stays at its default ``False``), so any fault degrades to a plain compile.
    """
    import jax

    cache_dir = resolve_xla_cache_dir(root)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_compilation_cache_max_size", int(max_size_bytes))
    except Exception:
        return None
    return cache_dir
