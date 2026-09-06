"""``z_of_dL_precomputed`` spells out ``jnp.interp`` -- and must stay bit-identical.

The inversion runs twice per production likelihood call (the PE seam and the
selection seam) over ~1e6 samples each.  ``jnp.interp`` hard-codes
``searchsorted(side='right')`` at the module-default ``method='scan'``, which
lowers to a ``while`` op XLA cannot fuse through; ``method='scan_unrolled'``
lowers the same binary search to straight-line code that fuses into the
surrounding memory-bound kernel.  The method changes only HOW the insertion
index is found, never which index, so these tests pin (a) index equality
against the default method over the nodes, their neighbouring ulps, the
out-of-range tails and the NaN sentinel, (b) bit-identity of the interpolated
value against ``jnp.interp`` under a shared ``jit`` (``jnp.interp`` jits its own
body, so an EAGER comparison compares a fused graph against an unfused one and
is not the right reference), and (c) the ``DARKSIRENS_INTERP_SCAN=1`` escape
hatch.
"""
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.utils import cosmology as cosmo_mod
from darksirens.utils.cosmology import dL_of_z, z_of_dL_precomputed, zgrid

H0, OM0, W0, WA = 70.0, 0.3, -1.0, 0.0


def _grid(h0=H0):
    return dL_of_z(zgrid, h0, OM0, W0, WA)


def _probe_dL(grid):
    """Bulk draws, every node, every node +/- k ulp, both tails, and NaN/inf."""
    g = np.asarray(grid)
    rng = np.random.default_rng(1)
    parts = [
        rng.uniform(float(g[0]), float(g[-1]), 20_000),
        np.exp(rng.uniform(np.log(max(float(g[1]), 1e-6)), np.log(float(g[-1])),
                           20_000)),
        g,
        0.5 * (g[:-1] + g[1:]),
    ]
    for k in (1, 2, 4):
        parts += [g + k * np.spacing(g), g - k * np.spacing(g)]
    parts.append(np.array([0.0, -0.0, -1.0, -1e30, 1e-320, float(g[-1]) * 1.5,
                           1e30, np.inf, -np.inf, np.nan]))
    return np.concatenate(parts)


def test_scan_unrolled_index_matches_the_default_method():
    """The insertion index is the same one the shipped ``method='scan'`` returns."""
    grid = _grid()
    x = jnp.asarray(_probe_dL(grid))
    n = int(zgrid.shape[0])

    def idx(method):
        return jax.jit(lambda v: jnp.clip(
            jnp.searchsorted(grid, v, side="right", method=method), 1, n - 1))(x)

    want = np.asarray(idx("scan"))
    got = np.asarray(idx("scan_unrolled"))
    assert np.array_equal(got, want)
    # The probe really does exercise the whole index range, endpoints included.
    assert want.min() == 1 and want.max() == n - 1


@pytest.mark.parametrize("h0", [20.0, 70.0, 140.0])
def test_z_of_dL_precomputed_is_bit_identical_to_jnp_interp(h0):
    """Same float64 bits as the pre-change body, at both ends of the H0 prior."""
    grid = _grid(h0)
    x = jnp.asarray(_probe_dL(grid))

    def old_path(v):
        in_grid = (v >= grid[0]) & (v <= grid[-1])
        return jnp.where(in_grid, jnp.interp(v, grid, zgrid), jnp.nan)

    want = np.asarray(jax.jit(old_path)(x))
    got = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    assert np.array_equal(got[ok], want[ok])


def test_out_of_grid_and_nan_still_return_nan():
    """The ``in_grid`` mask is unchanged: no silent extrapolation."""
    grid = _grid()
    x = jnp.asarray([float(np.asarray(grid)[0]) - 1.0,
                     float(np.asarray(grid)[-1]) + 1.0,
                     np.nan, np.inf, -np.inf])
    got = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    assert np.all(np.isnan(got))


def test_dx0_guard_uses_spacing_of_eps():
    """``np.spacing(eps)``, not ``eps``: jax's own zero-width-cell guard."""
    want = float(np.spacing(np.finfo(np.asarray(zgrid).dtype).eps))
    assert cosmo_mod._INTERP_DX0_EPS == want


def test_escape_hatch_restores_the_stock_interp_bit_for_bit(monkeypatch):
    grid = _grid()
    x = jnp.asarray(_probe_dL(grid))

    def old_path(v):
        in_grid = (v >= grid[0]) & (v <= grid[-1])
        return jnp.where(in_grid, jnp.interp(v, grid, zgrid), jnp.nan)

    want = np.asarray(jax.jit(old_path)(x))
    monkeypatch.setattr(cosmo_mod, "_USE_INTERP_SCAN", True)
    hatched = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    monkeypatch.setattr(cosmo_mod, "_USE_INTERP_SCAN", False)
    fast = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    for arr in (hatched, fast):
        assert np.array_equal(np.isnan(arr), np.isnan(want))
        ok = ~np.isnan(want)
        assert np.array_equal(arr[ok], want[ok])


def test_escape_hatch_is_wired_to_the_environment():
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    code = ("import darksirens.utils.cosmology as c;"
            "print(c._USE_INTERP_SCAN)")
    on = subprocess.run([sys.executable, "-c", code],
                        env=dict(env, DARKSIRENS_INTERP_SCAN="1"),
                        capture_output=True, text=True, check=True)
    env.pop("DARKSIRENS_INTERP_SCAN", None)
    off = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    assert on.stdout.strip() == "True"
    assert off.stdout.strip() == "False"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_promotion_matches_jnp_interp(dtype):
    grid = _grid()
    x = jnp.asarray(np.asarray(_probe_dL(grid)[:1000], dtype=dtype))
    got = jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x)
    want = jax.jit(lambda v: jnp.interp(v, grid, zgrid))(x)
    assert got.dtype == want.dtype
