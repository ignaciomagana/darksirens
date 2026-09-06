"""``log_interp_zgrid`` interpolates without a binary search -- and must land on
exactly the node ``searchsorted`` would have found.

``zgrid`` is ``expm1`` of a UNIFORM grid in log(1+z), so the bracketing node of
any z is closed-form ``floor(log1p(z)/dlog)``; ``jnp.interp`` instead runs a
serial ceil(log2(N))-level ``lax.scan`` binary search with a dependent gather
per level, on the hottest primitive in the redshift stack.  These tests pin the
closed-form index to ``searchsorted(side='right')`` POINTWISE (which makes the
isolated interpolation bit-identical, including the pinned last node and the
one-ulp disagreements between libm's ``log1p`` and the tabulated nodes), and
pin the ``DARKSIRENS_INTERP_SEARCHSORTED=1`` escape hatch to reproduce the old
``jnp.interp`` spelling bit-for-bit -- fused into a larger graph the two paths
differ in the last ulp (XLA contracts the surrounding arithmetic differently
once the scan leaves the graph), so the hatch is what makes the A/B possible in
one process.
"""
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift import grid as grid_mod
from darksirens.redshift.completion import log_galaxy_measure_grid
from darksirens.redshift.grid import (_interp_zgrid, log_interp_zgrid, zMax,
                                      zgrid, zgrid_upper_index)

H0, OM0, W0, WA = 70.0, 0.3, -1.0, 0.0
_ZG = np.asarray(zgrid)


def _cosmo():
    return CosmoParams(H0=H0, Om0=OM0, w0=W0, wa=WA)


def _survey(sigma_kde=1e-3):
    return SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                        alpha_miss=1.0, sigma_kde=sigma_kde)


def _fp(seed=0):
    """A generic monotone log-valued grid (values, not physics, are the point)."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.normal(size=_ZG.size)) * 3.0 - 20.0)


def _probe_z():
    rng = np.random.default_rng(1)
    parts = [
        rng.uniform(0.0, zMax, 20_000),                       # bulk
        np.expm1(rng.uniform(0.0, np.log1p(zMax), 20_000)),   # log-uniform
        _ZG,                                                  # exact nodes
        0.5 * (_ZG[:-1] + _ZG[1:]),                           # midpoints
    ]
    for k in (1, 2, 4):                                       # +/- k ulp of nodes
        parts += [_ZG + k * np.spacing(_ZG), _ZG - k * np.spacing(_ZG)]
    parts.append(np.array([0.0, -0.0, 1e-320, 1e-12, zMax, zMax + 1.0, 1e30,
                           -1.0]))
    return np.concatenate(parts)


def test_closed_form_index_matches_searchsorted():
    """The index itself -- not just the interpolated value -- is the same one."""
    z = jnp.asarray(_probe_z())
    n = _ZG.size
    want = jnp.clip(jnp.searchsorted(zgrid, z, side="right"), 1, n - 1)

    def closed_form(x):
        i = jnp.clip(
            jnp.floor(jnp.log1p(jnp.maximum(x, 0.0)) / grid_mod._DLOG)
            .astype(jnp.int32) + 1, 1, n - 1)
        i = i - ((i > 1) & (x < zgrid[i - 1]))
        return i + ((i < n - 1) & (x >= zgrid[i]))

    assert np.array_equal(np.asarray(jax.jit(closed_form)(z)), np.asarray(want))


def test_interp_zgrid_is_bit_identical_to_jnp_interp():
    """Pointwise, in isolation, there is no numerics change at all."""
    fp = _fp()
    z = jnp.asarray(_probe_z())
    got = np.asarray(jax.jit(lambda x: _interp_zgrid(x, fp))(z))
    want = np.asarray(jax.jit(lambda x: jnp.interp(x, zgrid, fp))(z))
    assert np.array_equal(got, want)


def test_interp_zgrid_handles_nan_like_jnp_interp():
    fp = _fp()
    z = jnp.asarray([np.nan, np.inf, -np.inf])
    got = np.asarray(jax.jit(lambda x: _interp_zgrid(x, fp))(z))
    want = np.asarray(jax.jit(lambda x: jnp.interp(x, zgrid, fp))(z))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    assert np.array_equal(got[ok], want[ok])


def test_escape_hatch_restores_the_searchsorted_path_bit_for_bit(monkeypatch):
    """A/B: with the hatch on, ``log_interp_zgrid`` IS the pre-change spelling.

    Compared against the old body written out verbatim, not against itself.
    """
    fp = _fp(seed=3)
    z = jnp.asarray(_probe_z())

    def old_path(x):
        z1 = zgrid[1]
        below = fp[1] + 2.0 * jnp.log(
            jnp.maximum(x, jnp.finfo(zgrid.dtype).tiny) / z1)
        return jnp.where(x < z1, below, jnp.interp(x, zgrid, fp))

    want = np.asarray(jax.jit(old_path)(z))

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", True)
    hatched = np.asarray(jax.jit(lambda x: log_interp_zgrid(x, fp))(z))
    assert np.array_equal(hatched, want)

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", False)
    fast = np.asarray(jax.jit(lambda x: log_interp_zgrid(x, fp))(z))
    # Isolated, the fast path is bit-identical too (the ~1 ulp drift only shows
    # up once the call is fused into a larger reduction).
    assert np.array_equal(fast, want)


def test_escape_hatch_is_wired_to_the_environment():
    """The module flag comes from ``DARKSIRENS_INTERP_SEARCHSORTED``."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    code = ("import darksirens.redshift.grid as g;"
            "print(g._USE_SEARCHSORTED)")
    on = subprocess.run([sys.executable, "-c", code],
                        env=dict(env, DARKSIRENS_INTERP_SEARCHSORTED="1"),
                        capture_output=True, text=True, check=True)
    env.pop("DARKSIRENS_INTERP_SEARCHSORTED", None)
    off = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    assert on.stdout.strip() == "True"
    assert off.stdout.strip() == "False"


def test_kernel_state_agrees_with_the_searchsorted_path_to_one_ulp(monkeypatch):
    """In situ the change is a deliberate ~1 ulp one: bound it, and check the
    hatch still reproduces the old numbers exactly through the real code path."""
    from darksirens.redshift.catalog import catalog_kernel_state

    rng = np.random.default_rng(7)
    n_rows, n_max = 64, 16
    cat = EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(rng.uniform(1e-3, 1.5, size=(n_rows, n_max))),
        dzgals=jnp.asarray(rng.uniform(1e-3, 5e-2, size=(n_rows, n_max))),
        wgals=jnp.asarray(rng.uniform(0.1, 1.0, size=(n_rows, n_max))),
        ngals=jnp.asarray(rng.integers(1, n_max + 1, size=(n_rows,)),
                          dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((n_rows, int(zgrid.size))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    log_g = log_galaxy_measure_grid(_cosmo(), _survey())

    def run():
        st = catalog_kernel_state(_cosmo(), _survey(), cat, log_g_grid=log_g)
        return np.asarray(st.log_kw)

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", True)
    old = run()
    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", False)
    new = run()

    fin = np.isfinite(old)
    assert np.array_equal(fin, np.isfinite(new))
    rel = np.abs(new[fin] - old[fin]) / np.maximum(np.abs(old[fin]), 1e-300)
    assert rel.max() < 1e-13

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", True)
    assert np.array_equal(run(), old)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_promotion_matches_jnp_interp(dtype):
    fp = _fp(seed=5)
    z = jnp.asarray(np.asarray(_probe_z()[:1000], dtype=dtype))
    got = jax.jit(lambda x: _interp_zgrid(x, fp))(z)
    want = jax.jit(lambda x: jnp.interp(x, zgrid, fp))(z)
    assert got.dtype == want.dtype


# ---------------------------------------------------------------------------
# ``prior._grid_bracket``: the SAME closed-form index, the other convention
# ---------------------------------------------------------------------------

def _bracket_probe_z():
    """``_probe_z`` plus the out-of-range values ``_grid_bracket`` must survive."""
    return np.concatenate([
        _probe_z(),
        np.array([np.inf, -np.inf, -1e-300, 5e-324, zMax - np.spacing(zMax),
                  zMax + np.spacing(zMax), 2.0 * zMax, 1e300]),
    ])


def _old_grid_bracket(z):
    """``_grid_bracket``'s body before the closed form, written out verbatim."""
    idx = jnp.clip(jnp.searchsorted(zgrid, z, side="right") - 1, 0, zgrid.size - 2)
    t = (z - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx])
    return idx, jnp.clip(t, 0.0, 1.0)


def test_zgrid_upper_index_matches_searchsorted():
    """The shared helper returns the integer ``searchsorted`` returns."""
    z = jnp.asarray(_bracket_probe_z())
    n = _ZG.size
    want = jnp.clip(jnp.searchsorted(zgrid, z, side="right"), 1, n - 1)
    got = jax.jit(lambda x: zgrid_upper_index(x))(z)
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_grid_bracket_is_bit_identical_to_the_searchsorted_path():
    """Index AND clipped weight, exactly -- not to a tolerance."""
    from darksirens.redshift.prior import _grid_bracket

    z = jnp.asarray(_bracket_probe_z())
    idx, t = jax.jit(lambda x: _grid_bracket(x))(z)
    idx_w, t_w = jax.jit(lambda x: _old_grid_bracket(x))(z)
    assert np.array_equal(np.asarray(idx), np.asarray(idx_w))
    assert np.array_equal(np.asarray(t), np.asarray(t_w))


def test_grid_bracket_nan_weight_is_nan_on_both_paths():
    """NaN z lands on a DIFFERENT index than ``searchsorted`` -- and it is inert.

    The index is only ever consumed through ``_interp_row(lo, hi, t)``, whose
    NaN ``t`` makes ``miss`` NaN and ``where(miss > 0, ...)`` -inf on both
    paths, so only the weight has to agree.  (Production z cannot be NaN: dL is
    clipped to the distance grid before ``z_of_dL_precomputed``.)
    """
    from darksirens.redshift.prior import _grid_bracket

    z = jnp.asarray([np.nan])
    n = _ZG.size
    idx, t = jax.jit(lambda x: _grid_bracket(x))(z)
    idx_w, t_w = jax.jit(lambda x: _old_grid_bracket(x))(z)
    assert np.isnan(np.asarray(t)).all() and np.isnan(np.asarray(t_w)).all()
    # Both indices stay in range, so the two-node gather is safe either way.
    for got in (np.asarray(idx), np.asarray(idx_w)):
        assert (got >= 0).all() and (got <= n - 2).all()


def test_grid_bracket_escape_hatch_restores_searchsorted(monkeypatch):
    """``DARKSIRENS_INTERP_SEARCHSORTED=1`` puts the binary search back.

    The flag is read inside the helper body, so a monkeypatch before the trace
    switches the path (that is what makes the A/B possible in one process).
    Each arm is traced through a FRESH lambda: jax caches a trace on the
    function object, so re-tracing ``_grid_bracket`` itself would silently
    replay the first arm's jaxpr.
    """
    from darksirens.redshift.prior import _grid_bracket

    z = jnp.asarray(_bracket_probe_z())
    want_idx, want_t = jax.jit(lambda x: _old_grid_bracket(x))(z)

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", True)
    idx, t = jax.jit(lambda x: _grid_bracket(x))(z)
    assert np.array_equal(np.asarray(idx), np.asarray(want_idx))
    assert np.array_equal(np.asarray(t), np.asarray(want_t))
    # The hatch really is the binary search: searchsorted's default
    # method='scan' is the ceil(log2(N))-level lax.scan the closed form removes.
    assert "scan[" in str(jax.make_jaxpr(lambda x: _grid_bracket(x))(z))

    monkeypatch.setattr(grid_mod, "_USE_SEARCHSORTED", False)
    assert "scan[" not in str(jax.make_jaxpr(lambda x: _grid_bracket(x))(z))
