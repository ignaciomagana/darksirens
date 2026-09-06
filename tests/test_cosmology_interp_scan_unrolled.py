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
is not the right reference), including on doctored grids that fire the
zero-width-cell guard and the out-of-range fills, (c) the mechanism itself --
the compiled HLO carries no ``while`` op unless the escape hatch is on -- and
(d) the ``DARKSIRENS_INTERP_SCAN=1`` escape hatch and the shape fail-fast.
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
from darksirens.utils.cosmology import (
    _interp_dx0_eps,
    _interp_unrolled,
    dL_of_z,
    z_of_dL_precomputed,
    zgrid,
)

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


def _bit_equal(got, want):
    """Equal NaN patterns and equal bits everywhere else."""
    got, want = np.asarray(got), np.asarray(want)
    if not np.array_equal(np.isnan(got), np.isnan(want)):
        return False
    ok = ~np.isnan(want)
    return np.array_equal(got[ok], want[ok])


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
    assert _bit_equal(got, want)


def test_out_of_grid_and_nan_still_return_nan():
    """The ``in_grid`` mask is unchanged: no silent extrapolation."""
    grid = _grid()
    x = jnp.asarray([float(np.asarray(grid)[0]) - 1.0,
                     float(np.asarray(grid)[-1]) + 1.0,
                     np.nan, np.inf, -np.inf])
    got = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    assert np.all(np.isnan(got))


# --------------------------------------------------------------------------
# The bare body, where the branches the ``in_grid`` mask hides are observable.
# --------------------------------------------------------------------------

def _doctored_grids():
    """Grids that fire the branches a well-formed ``dL_grid`` never reaches."""
    base = np.asarray(_grid()).copy()
    out = {}
    g = base.copy(); g[1] = g[0]                       # zero-width FIRST cell
    out["zero_width_first_cell"] = g
    g = base.copy(); g[-2] = g[-1]                     # zero-width LAST cell
    out["zero_width_last_cell"] = g
    g = base.copy(); g[100] = g[99]; g[300] = g[299] = g[298]
    out["interior_duplicates"] = g                     # duplicate + 3-node plateau
    g = base.copy(); g[200] = np.nextafter(g[199], np.inf)
    out["one_ulp_cell"] = g
    return out


def _doctored_probe(g):
    """Nodes, their neighbourhoods, cell midpoints and both out-of-range tails."""
    parts = [g, g + np.spacing(g), g - np.spacing(g), 0.5 * (g[:-1] + g[1:]),
             np.array([float(g[0]) - 1.0, float(g[0]) - 1e-9, -1.0, 0.0,
                       float(g[-1]) + 1e-9, float(g[-1]) + 1.0, 1e30, np.nan])]
    return np.concatenate(parts)


@pytest.mark.parametrize("name", sorted(_doctored_grids()))
def test_interp_unrolled_matches_jnp_interp_on_doctored_grids(name):
    """The bare body equals ``jnp.interp`` bit for bit, mask or no mask.

    Unlike :func:`z_of_dL_precomputed`, this compares the branches the
    ``in_grid`` mask would otherwise hide: ``side='right'`` at exact and
    duplicated nodes, the ``dx0`` zero-width-cell guard (reachable only where a
    clipped index lands on a duplicated end cell), and the two out-of-range
    fills.
    """
    g = _doctored_grids()[name]
    grid = jnp.asarray(g)
    x = jnp.asarray(_doctored_probe(g))
    got = np.asarray(jax.jit(lambda v: _interp_unrolled(v, grid, zgrid))(x))
    want = np.asarray(jax.jit(lambda v: jnp.interp(v, grid, zgrid))(x))
    assert _bit_equal(got, want)


def test_the_doctored_grids_actually_fire_the_dx0_guard():
    """Guard against the test above going vacuous: ``dx0`` must be True somewhere."""
    fired = {}
    for name in ("zero_width_first_cell", "zero_width_last_cell"):
        g = _doctored_grids()[name]
        grid = jnp.asarray(g)
        x = jnp.asarray(_doctored_probe(g))
        n = grid.shape[0]
        i = jnp.clip(jnp.searchsorted(grid, x, side="right",
                                      method="scan_unrolled"), 1, n - 1)
        dx = np.asarray(grid[i] - grid[i - 1])
        fired[name] = bool(np.any(np.abs(dx) <= _interp_dx0_eps(x, grid)))
    assert all(fired.values()), fired


def test_dx0_guard_follows_the_promoted_dtype():
    """``np.spacing(eps)`` of the PROMOTED xp dtype -- jax's own rule."""
    f64 = np.zeros(3, np.float64)
    f32 = np.zeros(3, np.float32)
    assert _interp_dx0_eps(zgrid) == float(np.spacing(np.finfo(np.float64).eps))
    assert _interp_dx0_eps(f32, f32) == float(np.spacing(np.finfo(np.float32).eps))
    # Mixed operands promote, exactly as ``promote_dtypes_inexact`` does.
    assert _interp_dx0_eps(f32, f64) == _interp_dx0_eps(f64, f64)
    assert _interp_dx0_eps(f32, f32) != _interp_dx0_eps(f64, f64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float32_grid_matches_jnp_interp(dtype):
    """A float32 grid promotes -- and picks the float32 epsilon -- like jax's."""
    g = np.asarray(_grid(), dtype=dtype)
    grid = jnp.asarray(g)
    fp = jnp.asarray(np.asarray(zgrid), dtype=dtype)
    x = jnp.asarray(np.asarray(_doctored_probe(g), dtype=dtype))
    got = np.asarray(jax.jit(lambda v: _interp_unrolled(v, grid, fp))(x))
    want = np.asarray(jax.jit(lambda v: jnp.interp(v, grid, fp))(x))
    assert got.dtype == want.dtype
    assert _bit_equal(got, want)


# --------------------------------------------------------------------------
# The mechanism, the fail-fast and the escape hatch.
# --------------------------------------------------------------------------

def _while_ops(fn, x):
    """Number of ``while`` ops in the COMPILED HLO of ``fn``."""
    text = jax.jit(fn).lower(x).compile().as_text()
    return sum(" while(" in line for line in text.splitlines())


def test_the_unrolled_search_emits_no_while_op():
    """The point of the change: no ``while`` for XLA to refuse to fuse through.

    Reverting ``method='scan_unrolled'`` to the module default puts the loop
    back, so this is what stops the one token this change is about from being
    undone silently.
    """
    grid = _grid()
    x = jnp.asarray(np.linspace(0.0, float(np.asarray(grid)[-1]), 1024))
    assert _while_ops(lambda v: z_of_dL_precomputed(v, grid), x) == 0
    assert _while_ops(lambda v: jnp.interp(v, grid, zgrid), x) == 1


def test_a_grid_that_does_not_match_zgrid_fails_fast():
    """``jnp.interp``'s ``shape(xp) != shape(fp)`` guard, kept by hand.

    Without it the clipped index and jax's clamped gathers would absorb the
    mismatch and return silently wrong redshifts.
    """
    grid = _grid()
    n = int(zgrid.shape[0])
    with pytest.raises(ValueError, match=f"{n} nodes"):
        z_of_dL_precomputed(jnp.asarray([100.0]), grid[: n // 2])
    with pytest.raises(ValueError, match="one-dimensional"):
        z_of_dL_precomputed(jnp.asarray([100.0]), jnp.stack([grid, grid]))
    # The stock path fails on the same input, so the hatch is a pure A/B.
    with pytest.raises(ValueError):
        jnp.interp(jnp.asarray([100.0]), grid[: n // 2], zgrid)


def test_a_batch_of_grids_still_maps():
    """The guard sees the per-example shape, so ``vmap`` over grids is unaffected."""
    grid = _grid()
    x = jnp.asarray([100.0, 1000.0])
    stacked = jnp.stack([grid, _grid(60.0)])
    got = np.asarray(jax.vmap(lambda g: z_of_dL_precomputed(x, g))(stacked))
    want = np.stack([np.asarray(z_of_dL_precomputed(x, stacked[0])),
                     np.asarray(z_of_dL_precomputed(x, stacked[1]))])
    assert np.array_equal(got, want)


def test_escape_hatch_restores_the_stock_interp_bit_for_bit(monkeypatch):
    grid = _grid()
    x = jnp.asarray(_probe_dL(grid))

    def old_path(v):
        in_grid = (v >= grid[0]) & (v <= grid[-1])
        return jnp.where(in_grid, jnp.interp(v, grid, zgrid), jnp.nan)

    want = np.asarray(jax.jit(old_path)(x))
    monkeypatch.setattr(cosmo_mod, "_USE_INTERP_SCAN", True)
    hatched = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    assert _while_ops(lambda v: z_of_dL_precomputed(v, grid),
                      jnp.asarray(np.linspace(0.0, 1.0, 8))) == 1
    monkeypatch.setattr(cosmo_mod, "_USE_INTERP_SCAN", False)
    fast = np.asarray(jax.jit(lambda v: z_of_dL_precomputed(v, grid))(x))
    for arr in (hatched, fast):
        assert _bit_equal(arr, want)


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
    # Values too, not only the dtype: promotion must not move a single bit.
    in_grid = np.asarray((x >= grid[0]) & (x <= grid[-1]))
    assert np.array_equal(np.asarray(got)[in_grid], np.asarray(want)[in_grid])
