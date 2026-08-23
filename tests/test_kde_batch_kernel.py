"""The shared, fixed-shape KDE batch kernel (review finding JAX-07).

``build_pixel_kde_cache``, ``build_field_normalization_inputs`` and
``compute_lss_overdensity`` each built a local ``jit(vmap(_kde_dndz_obs, ...))``
inside the call.  Two costs followed: the compiled executable was discarded when
the function returned, so every call recompiled from scratch; and the short
final batch is a SECOND shape, so each freshly created wrapper compiled twice
(measured at 513 rows with ``batch_size=512``: cache size 1 after the 512-row
call, 2 after the 1-row tail).

The kernel is now a module-level singleton and :func:`_kde_rows` gives it one
shape per build by splitting evenly and padding the tail — the pattern
``cli/analyze.batched_map`` already uses.  These tests pin BOTH halves: that the
batching is numerically inert (bit-identical to one un-batched call, so no
padded row leaks into the output), and that a build compiles one shape, not two.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import jit, vmap

import darksirens.redshift.completion as completion
from darksirens.redshift.completion import (
    _batched_kde_dndz_obs,
    _kde_dndz_obs,
    _kde_rows,
    build_pixel_kde_cache,
)


def _catalog(n_pix, n_gal=6, seed=0):
    rng = np.random.default_rng(seed)
    zgals = jnp.asarray(rng.uniform(0.01, 0.4, size=(n_pix, n_gal)))
    ngals = jnp.asarray(rng.integers(1, n_gal + 1, size=n_pix).astype(np.int32))
    return zgals, ngals


def _reference(pix_idx, zgals, ngals):
    """One un-batched ``jit(vmap(...))`` call — what a full-length batch used to do.

    Deliberately jitted: an op-by-op ``vmap`` fuses differently and lands ~3e-13
    away, which would make a bit-identity assertion measure XLA, not batching.
    """
    return np.asarray(jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))(
        jnp.asarray(pix_idx, dtype=jnp.int32), zgals, None, ngals))


@pytest.mark.parametrize("n_pix", [1, 5, 511, 513, 1024])
@pytest.mark.parametrize("batch_size", [512, 64, 1])
def test_batched_rows_are_bit_identical_to_one_unbatched_call(n_pix, batch_size):
    """Rows reduce only over the galaxy axis, so any split must be inert — and the
    padded tail rows must be trimmed, not written."""
    zgals, ngals = _catalog(n_pix)
    pix_idx = np.arange(n_pix, dtype=np.int32)
    got = _kde_rows(pix_idx, zgals, None, ngals, batch_size)
    assert got.shape == (n_pix, int(completion.zgrid.size))
    assert np.array_equal(got, _reference(pix_idx, zgals, ngals))


def test_non_contiguous_pixel_order_is_preserved():
    """The builders pass arbitrary unique-pixel lists, not ``arange``."""
    n_pix = 600
    zgals, ngals = _catalog(n_pix, seed=3)
    pix_idx = np.array([7, 500, 1, 599, 42, 42, 0], dtype=np.int32)
    got = _kde_rows(pix_idx, zgals, None, ngals, batch_size=3)
    assert np.array_equal(got, _reference(pix_idx, zgals, ngals))


def test_build_pixel_kde_cache_matches_the_unbatched_reference():
    n_pix = 513
    zgals, ngals = _catalog(n_pix, seed=1)
    pix_idx = np.arange(n_pix, dtype=np.int32)
    cache, lookup = build_pixel_kde_cache(
        pix_idx, zgals, n_pix, ngals=ngals, batch_size=512)
    assert np.array_equal(np.asarray(cache), _reference(pix_idx, zgals, ngals))
    assert np.array_equal(np.asarray(lookup), pix_idx)


def test_build_is_deterministic_across_batch_sizes():
    n_pix = 513
    zgals, ngals = _catalog(n_pix, seed=2)
    pix_idx = np.arange(n_pix, dtype=np.int32)
    a, _ = build_pixel_kde_cache(pix_idx, zgals, n_pix, ngals=ngals, batch_size=512)
    b, _ = build_pixel_kde_cache(pix_idx, zgals, n_pix, ngals=ngals, batch_size=97)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def _cache_size():
    if not hasattr(_batched_kde_dndz_obs, "_cache_size"):
        pytest.skip("this JAX build exposes no per-function compilation cache size")
    return _batched_kde_dndz_obs._cache_size()


def test_a_build_compiles_one_shape_and_a_repeat_compiles_none():
    """The finding itself: two specializations per call, on a new wrapper each time.

    A distinct galaxy-axis width gives this build its own cache entries, so the
    deltas are independent of whatever else has run in the session.
    """
    n_pix, n_gal = 513, 7
    zgals, ngals = _catalog(n_pix, n_gal=n_gal, seed=4)
    pix_idx = np.arange(n_pix, dtype=np.int32)

    before = _cache_size()
    build_pixel_kde_cache(pix_idx, zgals, n_pix, ngals=ngals, batch_size=512)
    after_first = _cache_size()
    build_pixel_kde_cache(pix_idx, zgals, n_pix, ngals=ngals, batch_size=512)
    after_second = _cache_size()

    assert after_first - before == 1, "a build must compile ONE batch shape"
    assert after_second == after_first, "a repeat build must reuse the executable"


def test_the_kernel_is_a_module_level_singleton():
    """A per-call ``jit(vmap(...))`` would throw the compilation away on return."""
    assert isinstance(_batched_kde_dndz_obs, jax.stages.Wrapped)
