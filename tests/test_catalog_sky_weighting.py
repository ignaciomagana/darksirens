"""FIELD-convention catalog sky weighting (``--catalog_sky_weighting field``).

The field convention replaces the per-pixel catalog-prior normalizer ``Z[pix] =
N_obs + N_miss`` with a survey-GLOBAL ``Z(theta) = Sum_all-pixels[N_obs +
N_miss]``, so a K>=2 mixture weight ``fcat_k`` measures the HOST FRACTION
(number-density / sky-clustering contrast) rather than the per-pixel z-shape
preference.  Empty pixels then carry weight proportional to their n0-implied
missing count, and ``log10n0`` becomes genuinely informative -- the estimand fix
motivated by the gws-agn campaign.

Design under test:
  * ``darksirens.redshift.completion.build_field_normalization_inputs`` /
    ``field_global_log_Z`` -- the survey-global normalizer, term-consistent with
    ``_assemble_curves``'s per-pixel completeness ``C``.
  * ``darksirens.redshift.prior`` -- ``DarkSirenPriorState.log_Z_global`` and the
    static ``catalog_sky_weighting`` branch in ``_eval_dark_scalar``.
  * ``darksirens.likelihood.{core,factory}`` -- static ``catalog_sky_weighting``
    threading, K=1/K>=2 wiring, and the scope gates.

Conventions mirror ``tests/test_completion_depth.py`` (x64 via conftest) and
``tests/test_multitracer_likelihood.py`` (tiny in-memory bundle fixtures; ``==``
for bit-identity, tight tolerances for the mixture identities).
"""
import jax
jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    _kde_dndz_obs,
    _precompute_grids,
    build_field_normalization_inputs,
    field_global_log_Z,
)
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(n0=1e-2, z_depth=None):
    return SurveyParams(
        n0=n0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
        z_depth=z_depth,
    )


def _synthetic_full_sky(npix=12, maxg=3, seed=0):
    """A tiny full-sky catalog: a few occupied pixels, the rest empty."""
    zgals = np.zeros((npix, maxg))
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    occ = {1: [0.10], 3: [0.20, 0.25], 4: [0.15], 7: [0.30, 0.32, 0.28]}
    for p, zs in occ.items():
        for j, z in enumerate(zs):
            zgals[p, j] = z
            wgals[p, j] = 1.0
        ngals[p] = len(zs)
    dzgals = np.full((npix, maxg), 0.02)
    return zgals, dzgals, wgals, ngals


def _catalog_with_field(zgals, wgals, ngals, apix=1.0):
    """Build an EMCatalog carrying the field-normalization inputs (on-the-fly
    KDE fallback, ``dN_obs_kde=None`` -- the same recipe field_global_log_Z
    reproduces)."""
    fobs, n_empty, N_obs_total, _occ = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    return EMCatalog(
        apix=apix,
        zgals=jnp.asarray(zgals),
        dzgals=jnp.asarray(np.full_like(zgals, 0.02)),
        wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=fobs,
        field_n_empty=jnp.asarray(float(n_empty)),
        field_N_obs_total=jnp.asarray(float(N_obs_total)),
    )


def _brute_force_Z(cosmo, survey, zgals, ngals, apix=1.0, z_depth=None):
    """Pure-NumPy global normalizer Z from first principles: per pixel
    ``N_obs + trapz((1 - C) dN_exp * depthmask)`` with ``C`` computed
    independently via the matched KDE / smoothed expected-count ratio."""
    npix = zgals.shape[0]
    cat = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(zgals),
        wgals=None, ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, cat)
    dN_exp = np.asarray(grids.dN_exp)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth)
    dN_exp_safe = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    zg = np.asarray(zgrid)
    mask = np.ones_like(zg) if z_depth is None else (zg <= z_depth).astype(float)

    Z = 0.0
    for p in range(npix):
        nobs = int(ngals[p])
        Z += nobs
        if nobs > 0:
            obs = np.asarray(
                _kde_dndz_obs(p, jnp.asarray(zgals), ngals=jnp.asarray(ngals))
            )
        else:
            obs = np.zeros_like(zg)
        C = np.clip(obs / dN_exp_safe, 0.0, 1.0)
        Z += np.trapezoid((1.0 - C) * dN_exp * mask, zg)
    return Z


# ---------------------------------------------------------------------------
# Completion-level: global Z vs brute force
# ---------------------------------------------------------------------------

def test_field_global_Z_matches_bruteforce():
    cosmo = _cosmo()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()
    apix = hp.nside2pixarea(1)

    for z_depth in (None, float(np.asarray(zgrid)[len(zgrid) // 3])):
        survey = _survey(n0=1e-2, z_depth=z_depth)
        cat = _catalog_with_field(zgals, wgals, ngals, apix=apix)
        Z = float(np.exp(np.asarray(field_global_log_Z(cosmo, survey, cat))))
        Z_bf = _brute_force_Z(cosmo, survey, zgals, ngals, apix=apix, z_depth=z_depth)
        rel = abs(Z - Z_bf) / Z_bf
        # f32 field table -> the achieved tolerance is far tighter than the 1e-6
        # budget (the empty-pixel + total-count terms are exact f64), but assert
        # against the documented ceiling.
        assert rel <= 1e-6, (z_depth, Z, Z_bf, rel)


# ---------------------------------------------------------------------------
# Prior-state level: field changes the normalizer; PE/selection share Z
# ---------------------------------------------------------------------------

def test_field_prior_uses_global_not_per_pixel_normalizer():
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()
    cat = _catalog_with_field(zgals, wgals, ngals)

    st_c = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="conditional"
    )
    st_f = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="field"
    )
    z = jnp.array([0.20, 0.30])
    pix = jnp.array([3, 7], dtype=jnp.int32)  # occupied rows
    lp_c = np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", st_c, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="conditional",
    ))
    lp_f = np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", st_f, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="field",
    ))
    assert np.all(np.isfinite(lp_c)) and np.all(np.isfinite(lp_f))
    # Field vs conditional differ: the numerator is identical, only the
    # denominator swaps Z[pix] -> Z_global, so the shift is exactly the
    # per-pixel log-normalizer difference.
    assert np.all(np.abs(lp_c - lp_f) > 1e-6)
    diff = lp_c - lp_f
    expected = np.asarray(st_f.log_Z_global) - np.asarray(st_c.log_Z)[np.asarray(pix)]
    np.testing.assert_allclose(diff, expected, rtol=1e-12, atol=1e-12)


def test_field_pe_selection_share_same_global_Z():
    """PE and selection states built from the SAME catalog + theta must carry
    the SAME log_Z_global (the constant cancels structurally between the PE and
    selection terms)."""
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()
    cat_pe = _catalog_with_field(zgals, wgals, ngals)
    cat_sel = _catalog_with_field(zgals, wgals, ngals)  # same arrays -> same Z

    st_pe = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat_pe, catalog_sky_weighting="field"
    )
    st_sel = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat_sel, catalog_sky_weighting="field"
    )
    assert float(st_pe.log_Z_global) == float(st_sel.log_Z_global)


def test_field_empty_pixel_alive_and_n0_scaling():
    """In field mode an empty pixel's prior is finite and MONOTONE in n0 (it
    decreases as log10n0 -> very low), whereas in conditional mode it is exactly
    n0-independent (n0 cancels between the numerator and the per-pixel Z).  This
    pins the estimand difference the field convention was built for."""
    cosmo = _cosmo()
    # Row 0 empty, rows 1/2 occupied.
    zgals = np.array([[0.0, 0.0], [0.15, 0.0], [0.25, 0.30]])
    wgals = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    ngals = np.array([0, 1, 2], dtype=np.int32)
    z = jnp.array([0.20])
    empty_pix = jnp.array([0], dtype=jnp.int32)

    def _prior(mode, n0):
        survey = _survey(n0=n0)
        cat = _catalog_with_field(zgals, wgals, ngals)
        st = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat, catalog_sky_weighting=mode
        )
        return float(eval_redshift_prior_with_state(
            "dark_sirens", st, z, empty_pix, cosmo, survey, cat,
            catalog_sky_weighting=mode,
        )[0])

    n0s = [1e-1, 1e-3, 1e-6]

    # Conditional: n0-invariant empty-pixel prior (the estimand blind spot).
    cond_vals = [_prior("conditional", n0) for n0 in n0s]
    assert np.all(np.isfinite(cond_vals))
    np.testing.assert_allclose(cond_vals, cond_vals[0], rtol=1e-9, atol=1e-9)

    # Field: finite at high n0, strictly decreasing as n0 -> low (log10n0 informative).
    field_vals = [_prior("field", n0) for n0 in n0s]
    assert np.isfinite(field_vals[0])
    assert field_vals[0] > field_vals[1] > field_vals[2]


# ---------------------------------------------------------------------------
# Likelihood level: full-sky single-catalog data path
# ---------------------------------------------------------------------------

def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled}
    return pop_lower, pop_upper, pop_labels, pop_fid, sampled, fixed


def _mid_pop():
    pop_lower, pop_upper, *_ = _pop_bits()
    return 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))


def _full_sky_data(nsamp=2, n_sel=8):
    """test_selection_prior_model-style full-sky single-catalog data dict (the
    union/full-catalog path, so make_likelihood builds field inputs itself)."""
    nside = 1
    n_pix = hp.nside2npix(nside)
    zgals = np.full((n_pix, 1), 0.10, dtype=float)
    zgals[2, 0] = 0.28
    dzgals = np.full((n_pix, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix, 1), dtype=float)
    ngals = np.ones(n_pix, dtype=np.int32)
    return {
        "nEvents": 1, "nsamp": nsamp, "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside), "nside": nside, "n_pix_catalog": n_pix,
        "zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals_catalog": ngals,
        "zgals_catalog": zgals, "dzgals_catalog": dzgals, "wgals_catalog": wgals,
        # Compact (1, N_grid) dummy overdensity, matching what the production
        # loaders always emit for non-LSS runs (the field-mode gate rejects
        # expanded per-pixel grids by static shape).
        "delta_g_pix_z": jnp.zeros((1, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0]), "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]), "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp), "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel), "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=jnp.int32),
    }


def _base_opts(**overrides):
    pop_lower, pop_upper, _labels, _fid, sampled, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _single_ll_value(opts, data):
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return ll, float(ll(jnp.asarray([_mid_pop()])))


def test_conditional_default_bit_identical_k1():
    """K=1: no flag == catalog_sky_weighting='conditional' (float ==)."""
    data = _full_sky_data()
    _ll_a, v_default = _single_ll_value(_base_opts(), dict(data))
    _ll_b, v_explicit = _single_ll_value(
        _base_opts(catalog_sky_weighting="conditional"), dict(data)
    )
    assert np.isfinite(v_default)
    assert v_default == v_explicit


def test_field_mode_changes_normalizer_and_grad_finite():
    """Field != conditional for a multi-pixel catalog, and jax.grad of the
    field-mode likelihood w.r.t. the sampler coord is finite."""
    data = _full_sky_data()
    _ll_c, v_cond = _single_ll_value(_base_opts(), dict(data))
    ll_field, v_field = _single_ll_value(
        _base_opts(catalog_sky_weighting="field"), dict(data)
    )
    assert np.isfinite(v_cond) and np.isfinite(v_field)
    assert v_field != v_cond

    grad = jax.grad(lambda c: ll_field(c))(jnp.asarray([_mid_pop()]))
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# Mixture (K=2) field identities -- mirror test_multitracer_likelihood.py
# ---------------------------------------------------------------------------

APIX1 = hp.nside2pixarea(1)
Z_A = 0.10
Z_B = 0.30


def _shared_physics(nsamp=2, n_sel=8):
    return dict(
        nEvents=1, nsamp=nsamp, Ndraw=float(n_sel),
        m1det=jnp.array([36.0, 38.0]), m2det=jnp.array([28.8, 30.4]),
        dL=jnp.array([460.0, 500.0]), chieff=jnp.array([0.0, 0.02]),
        p_pe=jnp.ones(nsamp),
        m1detsels=jnp.linspace(34.0, 40.0, n_sel),
        m2detsels=0.8 * jnp.linspace(34.0, 40.0, n_sel),
        dLsels=jnp.linspace(430.0, 530.0, n_sel),
        chieffsels=jnp.zeros(n_sel), p_draw=jnp.ones(n_sel),
    )


def _field_bundle(apix, z, dz=0.02, nsamp=2, n_sel=8, n_empty=3):
    """A single-occupied-row compact bundle PLUS survey-global field inputs (one
    observed galaxy, ``n_empty`` empty survey pixels).  Same compact-view
    contract as load_multitracer_catalog_bundles, extended with the precomputed
    field-normalization inputs so the mixture builder can read them directly."""
    bundle = dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=np.array([[z]]), dzgals_pe=np.array([[dz]]),
        wgals_pe=np.array([[1.0]]), ngals_pe=np.array([1], dtype=np.int32),
        unique_pixels_pe=np.array([0], dtype=np.int32),
        sample_to_unique_pe=np.zeros(nsamp, dtype=np.int32),
        zgals_sel=np.array([[z]]), dzgals_sel=np.array([[dz]]),
        wgals_sel=np.array([[1.0]]), ngals_sel=np.array([1], dtype=np.int32),
        unique_pixels_sel=np.array([0], dtype=np.int32),
        sample_to_unique_sel=np.zeros(n_sel, dtype=np.int32),
    )
    fobs, _ne, nobs, _occ = build_field_normalization_inputs(
        jnp.asarray([[z]]), None, jnp.asarray([1], dtype=jnp.int32)
    )
    bundle["field_dN_obs_s"] = fobs
    bundle["field_n_empty"] = float(n_empty)
    bundle["field_N_obs_total"] = float(nobs)
    return bundle


def _k1_field_value(z):
    """K=1 field likelihood for catalog ``z`` (single-catalog path reads the
    field inputs supplied in ``data``)."""
    data = dict(_shared_physics())
    data.update(_field_bundle(APIX1, z))
    opts = _base_opts(catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return float(ll(jnp.asarray([_mid_pop()])))


def _k2_field_likelihood(z_a, z_b):
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_field_bundle(APIX1, z_a), _field_bundle(APIX1, z_b)]
    opts = _base_opts(n_catalogs=2, catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_conditional_default_bit_identical_k2():
    """K=2: no flag == catalog_sky_weighting='conditional' (float ==)."""
    def _conditional_bundle(z):
        b = _field_bundle(APIX1, z)
        for key in ("field_dN_obs_s", "field_n_empty", "field_N_obs_total"):
            b.pop(key)
        return b

    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_conditional_bundle(Z_A), _conditional_bundle(Z_B)]

    ll_default = make_likelihood(_base_opts(n_catalogs=2), dict(data),
                                 pop_fid, fixed_parameter_values=fixed)
    ll_explicit = make_likelihood(
        _base_opts(n_catalogs=2, catalog_sky_weighting="conditional"),
        dict(data), pop_fid, fixed_parameter_values=fixed,
    )
    coord = jnp.asarray([_mid_pop(), 0.37])
    v_default = float(ll_default(coord))
    v_explicit = float(ll_explicit(coord))
    assert np.isfinite(v_default)
    assert v_default == v_explicit


def test_k2_field_duplicated_equals_k1():
    """K=2 with catalog B == catalog A equals K=1(A) at ANY fcat_2 (weights sum
    to 1), in field mode -- tolerance <=1e-12."""
    val_k1 = _k1_field_value(Z_A)
    ll_k2 = _k2_field_likelihood(Z_A, Z_A)
    val_k2 = float(ll_k2(jnp.asarray([_mid_pop(), 0.37])))
    assert np.isfinite(val_k1)
    assert abs(val_k2 - val_k1) <= 1e-12


def test_k2_field_fcat_endpoints():
    """K=2 field: fcat_2 -> 0 collapses onto catalog A, fcat_2 -> 1 onto catalog
    B -- tolerance <=1e-9."""
    val_a = _k1_field_value(Z_A)
    val_b = _k1_field_value(Z_B)
    assert val_a != val_b
    ll_k2 = _k2_field_likelihood(Z_A, Z_B)
    val_at_0 = float(ll_k2(jnp.asarray([_mid_pop(), 0.0])))
    val_at_1 = float(ll_k2(jnp.asarray([_mid_pop(), 1.0])))
    assert abs(val_at_0 - val_a) <= 1e-9
    assert abs(val_at_1 - val_b) <= 1e-9


# ---------------------------------------------------------------------------
# Scope gates
# ---------------------------------------------------------------------------

def test_field_gate_rejects_marks_ensemble_and_inconsistent_budgets():
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()

    # Marked-host model: rejected (until the marked global normalizer lands).
    cat_marks = _catalog_with_field(zgals, wgals, ngals)
    with pytest.raises(NotImplementedError):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_marks,
            mark_model="loglinear", mark_names=("logmstar",),
            catalog_sky_weighting="field",
        )

    # Q ENSEMBLE (lss_marginalize): rejected (per-member Z_global not built yet).
    cat_members = cat_marks._replace(
        lss_completion_logq_members=jnp.zeros((2, zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(NotImplementedError):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_members, catalog_sky_weighting="field"
        )

    # Deterministic Q table WITHOUT the survey-global Q rows: the numerator
    # and normalizer would carry different budgets -> rejected with a build hint.
    cat_q_inconsistent = cat_marks._replace(
        lss_completion_logq=jnp.zeros((zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(ValueError, match="field_lss_q"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_q_inconsistent,
            catalog_sky_weighting="field",
        )

    # Per-pixel delta_g WITHOUT the survey-global delta_g rows: rejected.
    cat_dg_inconsistent = cat_marks._replace(
        delta_g_pix_z=jnp.zeros((zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(ValueError, match="field_delta_g"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_dg_inconsistent,
            catalog_sky_weighting="field",
        )

    # Missing field-normalization inputs: rejected (needs the full-sky catalog).
    cat_nofield = EMCatalog(
        apix=1.0, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(zgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    with pytest.raises(ValueError):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_nofield, catalog_sky_weighting="field"
        )
