"""
Model-contract tests for the dark-siren redshift prior.

These pin the *physics* of the corrected implementation:

  1. p(z|pix) integrates to 1 per pixel (occupied, sparse, empty) —
     additive-density mixture, not a pointwise mixture of normalised PDFs.
  2. The completeness estimator is calibrated: galaxies drawn from
     dN = n0*apix*dV*C_true(z) with the TRUE n0 recover C_true (no
     sqrt(2*pi)*sigma factor, no double-counted roll-off), boundary
     included.
  3. Catalog:missing odds equal the count odds N_obs : N_miss.
  4. Each catalog kernel (volumetric tilt included) carries unit mass —
     a galaxy is not up-weighted for being far away.
  5. NaN inputs map to -inf (never probability 1); sigma_eff is floored
     so spectroscopic dz -> 0 cannot silently drop galaxies.
  6. Compact-fallback KDE caches are keyed by global pixel
     (regression for the silent clamp-to-wrong-row bug).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest
from scipy.special import expit

from darksirens.redshift import zgrid
from darksirens.redshift.catalog import log_catalog_prior_vmap
from darksirens.redshift.completion import (
    _S_EXP,
    _precompute_grids,
    build_pixel_kde_cache,
    completion_curves,
)
from darksirens.redshift.prior import (
    _log_prior_dark_sirens,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.likelihood.catalog_views import _global_cache_lookup
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.utils.cosmology import dV_of_z

ZG = np.asarray(zgrid)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
N0_TRUE, APIX = 0.002, 8.0e-4
SURVEY = SurveyParams(
    n0=N0_TRUE, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0, sigma_kde=0.0
)


def _draw(C_true, seed):
    r = np.random.default_rng(seed)
    dV = np.asarray(dV_of_z(zgrid, 67.74, 0.3075, -1.0, 0.0))
    f = N0_TRUE * APIX * dV * C_true
    cdf = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ZG))])
    n_tot = cdf[-1]
    cdf /= n_tot
    n = r.poisson(n_tot)
    return np.interp(r.uniform(size=n), cdf, ZG)


def _catalog(z_rows, unique_pixels=None):
    n_rows = len(z_rows)
    nmax = max(max((len(r) for r in z_rows), default=1), 1)
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    for i, r in enumerate(z_rows):
        zg[i, : len(r)] = r
        dz[i, : len(r)] = 0.003
        w[i, : len(r)] = 1.0
        ng[i] = len(r)
    zg, dz, w, ng = jnp.asarray(zg), jnp.asarray(dz), jnp.asarray(w), jnp.asarray(ng)
    keys = (
        np.arange(n_rows, dtype=np.int32)
        if unique_pixels is None
        else np.asarray(unique_pixels, np.int32)
    )
    npc = n_rows if unique_pixels is None else int(keys.max()) + 1
    kde, idx = build_pixel_kde_cache(keys, zg, npc, ngals=ng)
    if unique_pixels is not None:
        # production fallback path: cache rows are compact, lookup re-keyed
        kde, ident = build_pixel_kde_cache(
            np.arange(n_rows, dtype=np.int32), zg, n_rows, ngals=ng
        )
        idx = jnp.asarray(_global_cache_lookup(keys, n_rows, ident))
    return EMCatalog(
        apix=APIX,
        zgals=zg,
        dzgals=dz,
        wgals=w,
        ngals=ng,
        delta_g_pix_z=jnp.zeros((1, ZG.size)),
        dN_obs_kde=kde,
        pixel_to_cache_idx=idx,
        unique_pixels=None
        if unique_pixels is None
        else jnp.asarray(unique_pixels, jnp.int32),
    )


@pytest.fixture(scope="module")
def sigmoid_setup():
    C_true = expit((0.5 - ZG) / 0.1)
    cat = _catalog([_draw(C_true, 101), _draw(0.3 * C_true, 202), np.array([])])
    state = prepare_redshift_prior_state("dark_sirens", COSMO, SURVEY, cat)
    curves = completion_curves(COSMO, SURVEY, cat)
    return C_true, cat, state, curves


def test_prior_normalises_per_pixel(sigmoid_setup):
    _, cat, state, _ = sigmoid_setup
    for row in (0, 1, 2):
        lp = np.asarray(
            eval_redshift_prior_with_state(
                "dark_sirens",
                state,
                zgrid,
                jnp.full(ZG.size, row, jnp.int32),
                COSMO,
                SURVEY,
                cat,
            )
        )
        assert abs(_trapezoid(np.exp(lp), ZG) - 1.0) < 5e-3


def test_completeness_is_calibrated(sigmoid_setup):
    C_true, cat, _, curves = sigmoid_setup
    grids = _precompute_grids(COSMO, SURVEY, cat)
    dN_exp = np.asarray(grids.dN_exp)
    C_rec = 1.0 - np.asarray(curves.dN_miss[0]) / np.maximum(dN_exp, 1e-300)
    truth = np.asarray(_S_EXP @ jnp.asarray(C_true * dN_exp)) / np.maximum(
        np.asarray(_S_EXP @ jnp.asarray(dN_exp)), 1e-300
    )
    band = (ZG > 0.15) & (ZG < 1.0)
    calib = np.mean(C_rec[band] / truth[band])
    assert 0.9 < calib < 1.1  # old KDE was off by sqrt(2*pi)*sigma ~ 7.98x
    i50 = np.argmin(np.abs(ZG - 0.5))
    assert 0.35 < C_rec[i50] < 0.65  # old roll-off double-count gave C_true^2


def test_catalog_missing_odds_are_count_odds(sigmoid_setup):
    _, cat, state, curves = sigmoid_setup
    for row in (0, 1):
        n_obs = float(cat.ngals[row])
        n_miss = float(curves.N_miss[row])
        z_norm = n_obs + n_miss
        miss_frac = _trapezoid(np.asarray(state.dN_miss[row]) / z_norm, ZG)
        np.testing.assert_allclose(1.0 - miss_frac, n_obs / z_norm, atol=1e-9)


def test_empty_pixel_is_pure_missing_density(sigmoid_setup):
    _, cat, state, curves = sigmoid_setup
    lp = np.asarray(
        eval_redshift_prior_with_state(
            "dark_sirens",
            state,
            zgrid,
            jnp.full(ZG.size, 2, jnp.int32),
            COSMO,
            SURVEY,
            cat,
        )
    )
    p_miss = np.asarray(state.dN_miss[2]) / float(curves.N_miss[2])
    np.testing.assert_allclose(np.exp(lp), p_miss, atol=1e-12)


def test_tilted_kernels_carry_unit_mass():
    cat = _catalog([np.array([0.05, 0.4, 1.2])])
    cat = cat._replace(dzgals=jnp.full_like(cat.dzgals, 0.05))  # broad photo-z
    survey = SURVEY._replace(delta=0.7)
    lp = np.asarray(
        log_catalog_prior_vmap(
            zgrid, jnp.zeros(ZG.size, jnp.int32), COSMO, survey, cat
        )
    )
    assert abs(_trapezoid(np.exp(lp), ZG) - 1.0) < 1e-3


def test_nan_maps_to_neg_inf_and_sigma_floor_keeps_galaxies():
    cat = _catalog([np.array([0.30])])
    out = np.asarray(
        _log_prior_dark_sirens(
            jnp.array([jnp.nan, 0.30]),
            jnp.zeros(2, jnp.int32),
            COSMO,
            SURVEY,
            cat,
        )
    )
    assert np.isneginf(out[0]) and np.isfinite(out[1])

    spec = cat._replace(dzgals=jnp.zeros_like(cat.dzgals))  # dz = 0
    s0 = SURVEY._replace(sigma_kde=0.0)
    with_gal = float(
        _log_prior_dark_sirens(
            jnp.array([0.30]), jnp.zeros(1, jnp.int32), COSMO, s0, spec
        )[0]
    )
    no_gal = spec._replace(ngals=jnp.zeros_like(spec.ngals), wgals=jnp.zeros_like(spec.wgals))
    without_gal = float(
        _log_prior_dark_sirens(
            jnp.array([0.30]), jnp.zeros(1, jnp.int32), COSMO, s0, no_gal
        )[0]
    )
    assert np.isfinite(with_gal)
    assert with_gal > without_gal  # the spectroscopic galaxy still contributes


def test_fallback_cache_lookup_is_global_pixel_keyed():
    uniq = np.array([7, 3, 42], dtype=np.int32)
    cat = _catalog(
        [np.array([0.10]), np.array([0.20]), np.array([0.40])], unique_pixels=uniq
    )
    for row, z_pk in [(0, 0.10), (1, 0.20), (2, 0.40)]:
        kde = np.asarray(cat.dN_obs_kde[int(cat.pixel_to_cache_idx[int(uniq[row])])])
        assert abs(ZG[np.argmax(kde)] - z_pk) < 0.01
