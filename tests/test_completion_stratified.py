"""Stratified selection consumption inside the likelihood (PR-3 core).

SurveyParams.selection_strata stacks one C_sel curve per stratum,
pixel_stratum_map routes each pixel to its curve, and the global field
normalizer decomposes the empty-pixel budget per stratum.  Pinned here:

  (1) degenerate strata (identical offsets) reproduce the single-stratum
      curve and normalizer EXACTLY -- the S->1 consistency limit;
  (2) _row_C returns each pixel's own stratum curve;
  (3) the stratified V_empty equals the hand-computed
      Sum_s (1 - Cbar_s) n_empty_s (the normalizer's subtle piece);
  (4) selection_strata=None stays bit-identical to the pre-PR path;
  (5) the H0 firewall survives stratification.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.core.types import (  # noqa: E402
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.redshift.completion import (  # noqa: E402
    _field_missing_curve,
    _precompute_grids,
    _row_C,
    build_field_normalization_inputs,
)
from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import c_sel_gaussian  # noqa: E402

THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)
# Two strata: stratum 0 is the reference; stratum 1 is 0.4 mag shallower with
# 10% wider scatter (a north/south-style photometric-system offset).
STRATA = ((24.0, 0.0, 1.0), (23.6, 0.05, 1.1))


def _em(n_pix=12, max_gals=3, seed=5, stratum_map=None):
    rng = np.random.default_rng(seed)
    zg = np.full((n_pix, max_gals), 100.0)
    dz = np.ones((n_pix, max_gals))
    w = np.zeros((n_pix, max_gals))
    ng = np.zeros(n_pix, dtype=int)
    for p in range(n_pix):
        n = int(rng.integers(0, max_gals))
        zs = np.sort(rng.uniform(0.02, 0.4, n))
        zg[p, :n] = zs
        dz[p, :n] = 1e-3
        w[p, :n] = 1.0
        ng[p] = n
    import healpy as hp

    apix = hp.nside2pixarea(1)
    field = build_field_normalization_inputs(
        jnp.asarray(zg), jnp.asarray(w), jnp.asarray(ng))
    extra = {}
    if stratum_map is not None:
        stratum_map = np.asarray(stratum_map, dtype=np.int32)
        occupied = ng > 0
        counts = np.array([
            np.sum((stratum_map == s) & ~occupied)
            for s in range(int(stratum_map.max()) + 1)
        ], dtype=float)
        extra = dict(pixel_stratum_map=jnp.asarray(stratum_map),
                     empty_stratum_counts=jnp.asarray(counts))
    return EMCatalog(
        apix=float(apix), zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=field.dN_obs_s, field_n_empty=field.n_empty,
        field_N_obs_total=field.N_obs_total,
        field_occupied_pixels=field.occupied_pixels, **extra)


def _survey(**kw):
    base = dict(n0=1e-4, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=0.0, c_mode=C_MODE_SELECTION_STRUCT)
    base.update(THETA)
    base.update(kw)
    return SurveyParams(**base)


def _cosmo(h0=67.74):
    return CosmoParams(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


def _alternating_map(n_pix=12):
    return np.arange(n_pix) % 2


def test_degenerate_strata_reproduce_the_single_curve_exactly():
    em = _em(stratum_map=_alternating_map())
    degenerate = ((THETA["m_lim"], 0.0, 1.0), (THETA["m_lim"], 0.0, 1.0))
    g1 = _precompute_grids(_cosmo(), _survey(), em)
    g2 = _precompute_grids(_cosmo(), _survey(selection_strata=degenerate), em)
    assert g2.C_bar_raw.shape == (2, zgrid.size)
    for s in range(2):
        np.testing.assert_array_equal(np.asarray(g2.C_bar_raw[s]),
                                      np.asarray(g1.C_bar_raw))
    v1, _ = _field_missing_curve(_cosmo(), _survey(), em)
    v2, _ = _field_missing_curve(_cosmo(), _survey(selection_strata=degenerate), em)
    np.testing.assert_allclose(np.asarray(v2), np.asarray(v1),
                               rtol=0, atol=1e-12)


def test_row_c_returns_each_pixels_stratum_curve():
    em = _em(stratum_map=_alternating_map())
    sv = _survey(selection_strata=STRATA)
    grids = _precompute_grids(_cosmo(), sv, em)
    for pix in (0, 1, 5):
        C, gp = _row_C(pix, grids, em)
        s = pix % 2
        ref = c_sel_gaussian(zgrid, STRATA[s][0],
                             THETA["M0hat"] + STRATA[s][1],
                             THETA["sigma_M"] * STRATA[s][2],
                             67.74, 0.3075, -1.0, 0.0)
        np.testing.assert_allclose(np.asarray(C),
                                   np.clip(np.asarray(ref), 0, 1),
                                   rtol=0, atol=1e-14)


def test_stratified_v_empty_matches_hand_sum():
    stratum_map = _alternating_map()
    em = _em(stratum_map=stratum_map)
    sv = _survey(selection_strata=STRATA)
    V_total, _ = _field_missing_curve(_cosmo(), sv, em)

    grids = _precompute_grids(_cosmo(), sv, em)
    C = np.clip(np.asarray(grids.C_bar_raw), 0.0, 1.0)      # (2, N_grid)
    ng = np.asarray(em.ngals)
    occ_pix = np.asarray(em.field_occupied_pixels)
    V_occ = np.zeros(zgrid.size)
    for p in occ_pix:
        V_occ += 1.0 - C[stratum_map[p]]
    n_empty_s = np.array([np.sum((stratum_map == s) & (ng == 0))
                          for s in (0, 1)], dtype=float)
    V_empty = np.sum((1.0 - C) * n_empty_s[:, None], axis=0)
    np.testing.assert_allclose(np.asarray(V_total), V_occ + V_empty,
                               rtol=0, atol=1e-10)


def test_stratified_missing_map_requires_stratum_inputs():
    em = _em()          # no stratum map
    sv = _survey(selection_strata=STRATA)
    with pytest.raises(ValueError, match="pixel_stratum_map"):
        _field_missing_curve(_cosmo(), sv, em)


def test_h0_firewall_survives_stratification():
    em = _em(stratum_map=_alternating_map())
    a = _precompute_grids(_cosmo(50.0), _survey(selection_strata=STRATA), em)
    b = _precompute_grids(_cosmo(95.0), _survey(selection_strata=STRATA), em)
    np.testing.assert_allclose(np.asarray(a.C_bar_raw),
                               np.asarray(b.C_bar_raw), rtol=0, atol=1e-12)


def test_none_strata_is_bit_identical_legacy():
    em = _em(stratum_map=_alternating_map())
    g = _precompute_grids(_cosmo(), _survey(), em)
    assert g.C_bar_raw.ndim == 1
    ref = c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                         THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0)
    np.testing.assert_array_equal(np.asarray(g.C_bar_raw), np.asarray(ref))
