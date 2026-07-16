"""Modulated FIELD-convention global normalizer (PR-3 of the multitracer
unification): ``field_global_log_Z`` accumulates the (N_grid,) field missing
curve ``V(z;theta) = Sum_occ (1 - C_p)·lss_p(z) + Sum_empty lss_p(z)`` and
integrates once, so the survey-global Z carries the SAME per-pixel budget
modulation as the numerator:

    lss_p = 1                                  (legacy)
    lss_p = Q_p(z)                             (deterministic Q_LSS table)
    lss_p = max(1 + b_eff·delta_g_p(z), 0)     (local overdensity)

Empty pixels: delta_g == 0 by construction (lss = 1); the Q empty-pixel sum is
a data constant (``field_lss_q_empty_sum``).

Fixtures mirror tests/test_catalog_sky_weighting.py (tiny synthetic full sky,
brute-force NumPy normalizer from first principles, x64 via conftest).
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
    build_field_delta_g_inputs,
    build_field_lss_q_inputs,
    field_global_log_Z,
)
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

NG = len(zgrid)


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(n0=1e-2, b_miss=1.0, alpha_miss=1.0, z_depth=None):
    return SurveyParams(
        n0=n0, z50=1.0, w=0.5, delta=0.0, b_miss=b_miss, alpha_miss=alpha_miss,
        z_depth=z_depth,
    )


def _synthetic_full_sky(npix=12, maxg=3):
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


def _logq_table(npix=12):
    """Global (npix, NG) log Q with distinct per-pixel structure."""
    base = np.linspace(-0.4, 0.4, NG)
    return np.array([base * np.cos(0.7 * p) for p in range(npix)])


def _delta_g_table(npix=12, amp=0.6):
    """Global (npix, NG) overdensity; empty pixels MUST carry delta_g = 0 (the
    production compute_lss_overdensity convention) -- enforced by the caller."""
    base = amp * np.sin(np.linspace(0.0, 4.0, NG))
    return np.array([base * ((p % 5) - 2) / 2.0 for p in range(npix)])


def _catalog(zgals, dzgals, wgals, ngals, apix=1.0, logq=None, delta_g=None):
    """EMCatalog with field-normalization inputs (+ optional modulations)."""
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    occupied = np.asarray(field.occupied_pixels)
    kwargs = dict(
        apix=apix,
        zgals=jnp.asarray(zgals),
        dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=jnp.asarray(float(field.n_empty)),
        field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
        field_occupied_pixels=jnp.asarray(occupied, dtype=jnp.int32),
    )
    if logq is not None:
        q_occ, q_empty_sum = build_field_lss_q_inputs(
            jnp.asarray(logq), occupied, int(zgals.shape[0])
        )
        kwargs["lss_completion_logq"] = jnp.asarray(logq)
        kwargs["lss_completion_indexing"] = 2
        kwargs["field_lss_q"] = q_occ
        kwargs["field_lss_q_empty_sum"] = q_empty_sum
    if delta_g is not None:
        dg = np.array(delta_g)
        empty = np.ones(zgals.shape[0], dtype=bool)
        empty[occupied] = False
        dg[empty] = 0.0  # production convention: empty pixels carry delta_g = 0
        kwargs["delta_g_pix_z"] = jnp.asarray(dg)
        kwargs["field_delta_g"] = build_field_delta_g_inputs(
            jnp.asarray(dg), occupied
        )
    return EMCatalog(**kwargs)


def _brute_force_Z(cosmo, survey, zgals, ngals, lss_fn, apix=1.0, z_depth=None):
    """First-principles NumPy Z: per pixel N_obs + trapz((1-C)·dN_exp·lss·mask)."""
    npix = zgals.shape[0]
    cat = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(zgals),
        wgals=None, ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=None,
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
        Z += np.trapezoid((1.0 - C) * dN_exp * lss_fn(p) * mask, zg)
    return Z


# ---------------------------------------------------------------------------
# Builder contracts
# ---------------------------------------------------------------------------

def test_field_inputs_carry_occupied_pixels():
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    assert list(np.asarray(field.occupied_pixels)) == [1, 3, 4, 7]
    assert int(field.n_empty) == 8
    assert float(field.N_obs_total) == 7.0
    assert field.dN_obs_s.shape == (4, NG)


def test_q_empty_sum_is_the_empty_pixel_Q_sum():
    zgals, _, wgals, ngals = _synthetic_full_sky()
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    logq = _logq_table()
    q_occ, q_empty_sum = build_field_lss_q_inputs(
        jnp.asarray(logq), np.asarray(field.occupied_pixels), 12
    )
    occupied = set(np.asarray(field.occupied_pixels).tolist())
    expected = np.sum(
        [np.exp(logq[p]) for p in range(12) if p not in occupied], axis=0
    )
    np.testing.assert_allclose(np.asarray(q_empty_sum), expected, rtol=1e-6)
    assert q_occ.shape == (4, NG)


# ---------------------------------------------------------------------------
# Brute-force parity per modulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z_depth", [None, 0.8])
def test_field_Z_unmodulated_matches_bruteforce(z_depth):
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cosmo, survey = _cosmo(), _survey(z_depth=z_depth)
    cat = _catalog(zgals, dzgals, wgals, ngals)
    Z_brute = _brute_force_Z(cosmo, survey, zgals, ngals, lambda p: 1.0,
                             z_depth=z_depth)
    logZ = float(field_global_log_Z(cosmo, survey, cat))
    np.testing.assert_allclose(logZ, np.log(Z_brute), rtol=1e-6)


@pytest.mark.parametrize("z_depth", [None, 0.8])
def test_field_Z_qdet_matches_bruteforce(z_depth):
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cosmo, survey = _cosmo(), _survey(z_depth=z_depth)
    logq = _logq_table()
    cat = _catalog(zgals, dzgals, wgals, ngals, logq=logq)
    Z_brute = _brute_force_Z(
        cosmo, survey, zgals, ngals, lambda p: np.exp(logq[p]), z_depth=z_depth
    )
    logZ = float(field_global_log_Z(cosmo, survey, cat))
    np.testing.assert_allclose(logZ, np.log(Z_brute), rtol=1e-5)


def test_field_Z_delta_g_matches_bruteforce_including_clamp():
    """b_eff large enough that (1 + b_eff·delta_g) goes NEGATIVE for part of
    the grid on some pixels -- the max(., 0) clamp must match brute force."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cosmo, survey = _cosmo(), _survey(b_miss=2.0, alpha_miss=1.2)
    dg = _delta_g_table()
    occupied = [1, 3, 4, 7]
    dg_eff = np.array(dg)
    empty = np.ones(12, dtype=bool)
    empty[occupied] = False
    dg_eff[empty] = 0.0
    b_eff = survey.alpha_miss * survey.b_miss
    assert (1.0 + b_eff * dg_eff).min() < 0.0  # the clamp genuinely engages
    cat = _catalog(zgals, dzgals, wgals, ngals, delta_g=dg)
    Z_brute = _brute_force_Z(
        cosmo, survey, zgals, ngals,
        lambda p: np.maximum(1.0 + b_eff * dg_eff[p], 0.0),
    )
    logZ = float(field_global_log_Z(cosmo, survey, cat))
    np.testing.assert_allclose(logZ, np.log(Z_brute), rtol=1e-5)


# ---------------------------------------------------------------------------
# Prior-level: field - conditional shift identity under each modulation
# ---------------------------------------------------------------------------

def _prior_shift(cat_field, cat_cond, cosmo, survey):
    """(field lp - conditional lp) and (log_Z[pix] - log_Z_global) per sample."""
    z = jnp.asarray([0.10, 0.22, 0.31])
    pix = jnp.asarray([1, 3, 7], dtype=jnp.int32)
    lp = {}
    states = {}
    for name, cat, weighting in (
        ("field", cat_field, "field"), ("cond", cat_cond, "conditional"),
    ):
        state = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat,
            catalog_sky_weighting=weighting,
        )
        states[name] = state
        lp[name] = eval_redshift_prior_with_state(
            "dark_sirens", state, z, pix, cosmo, survey, cat,
            catalog_sky_weighting=weighting,
        )
    shift = np.asarray(lp["field"] - lp["cond"])
    expected = np.asarray(
        states["cond"].log_Z[pix] - states["field"].log_Z_global
    )
    return shift, expected


# ---------------------------------------------------------------------------
# Likelihood-level: K>=2 mixture with per-catalog Q under the field normalizer
# (fills the zero-coverage gap flagged in the unification review)
# ---------------------------------------------------------------------------

from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {
        lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled
    }
    overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    mid = 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))
    return pop_fid, overrides, fixed, mid


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


def _field_bundle(logq=None, nsamp=2, n_sel=8):
    """One-catalog bundle over the 12-pixel synthetic sky with field inputs
    (+ optional global Q table mirrored into the field normalizer)."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    pe_pix = np.array([7, 7], dtype=np.int32)[:nsamp]
    sel_pix = np.array([1, 3, 4, 7, 1, 3, 4, 7], dtype=np.int32)[:n_sel]
    up_pe, s2u_pe = np.unique(pe_pix, return_inverse=True)
    up_se, s2u_se = np.unique(sel_pix, return_inverse=True)
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    bundle = dict(
        nside=1, apix=hp.nside2pixarea(1), n_pix_catalog=12,
        delta_g_pix_z=jnp.zeros((1, NG)),
        zgals_pe=zgals[up_pe], dzgals_pe=dzgals[up_pe], wgals_pe=wgals[up_pe],
        ngals_pe=ngals[up_pe],
        unique_pixels_pe=up_pe.astype(np.int32),
        sample_to_unique_pe=s2u_pe.astype(np.int32),
        zgals_sel=zgals[up_se], dzgals_sel=dzgals[up_se], wgals_sel=wgals[up_se],
        ngals_sel=ngals[up_se],
        unique_pixels_sel=up_se.astype(np.int32),
        sample_to_unique_sel=s2u_se.astype(np.int32),
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=field.n_empty,
        field_N_obs_total=field.N_obs_total,
        field_occupied_pixels=field.occupied_pixels,
    )
    if logq is not None:
        q_occ, q_empty_sum = build_field_lss_q_inputs(
            jnp.asarray(logq), field.occupied_pixels, 12
        )
        bundle["lss_completion_logq"] = jnp.asarray(logq)
        bundle["lss_completion_indexing"] = 2
        bundle["field_lss_q"] = q_occ
        bundle["field_lss_q_empty_sum"] = q_empty_sum
    return bundle


def _bundle_likelihood(bundles, fixed, pop_fid, overrides):
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = list(bundles)
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=False,
        fix_survey=True,
        prior_overrides=overrides,
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field",
        n_catalogs=len(bundles),
    )
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_k2_duplicated_catalog_with_q_equals_k1_with_q_under_field():
    """K=2 (A+Q, A+Q) at any fcat_2 == K=1 (A+Q): the Q-modulated field
    normalizer must preserve the duplicated-catalog mixture identity."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq = _logq_table()

    ll_k1 = _bundle_likelihood([_field_bundle(logq=logq)], fixed, pop_fid, overrides)
    val_k1 = float(ll_k1(jnp.asarray([mid])))
    assert np.isfinite(val_k1)

    ll_k2 = _bundle_likelihood(
        [_field_bundle(logq=logq), _field_bundle(logq=logq)],
        fixed, pop_fid, overrides,
    )
    val_k2 = float(ll_k2(jnp.asarray([mid, 0.37])))
    assert abs(val_k2 - val_k1) <= 1e-12


def test_per_catalog_q_is_live_and_asymmetric_at_k2():
    """Q on catalog 2 only changes the K=2 likelihood relative to no-Q, and
    relative to Q-on-both: the per-catalog budgets are genuinely independent."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq = _logq_table()
    coord = jnp.asarray([mid, 0.37])

    val_no_q = float(_bundle_likelihood(
        [_field_bundle(), _field_bundle()], fixed, pop_fid, overrides)(coord))
    val_q_on_2 = float(_bundle_likelihood(
        [_field_bundle(), _field_bundle(logq=logq)], fixed, pop_fid, overrides)(coord))
    val_q_on_both = float(_bundle_likelihood(
        [_field_bundle(logq=logq), _field_bundle(logq=logq)],
        fixed, pop_fid, overrides)(coord))

    assert np.isfinite(val_no_q) and np.isfinite(val_q_on_2)
    assert val_q_on_2 != val_no_q
    assert val_q_on_2 != val_q_on_both


@pytest.mark.parametrize("modulation", ["qdet", "delta_g"])
def test_field_conditional_shift_identity_under_modulation(modulation):
    """field lp - conditional lp == log_Z[pix] - log_Z_global with the SAME
    modulated numerator: the modulation must enter numerator and normalizer
    consistently, leaving only the normalizer swap."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cosmo, survey = _cosmo(), _survey()
    if modulation == "qdet":
        logq = _logq_table()
        cat_field = _catalog(zgals, dzgals, wgals, ngals, logq=logq)
        cat_cond = _catalog(zgals, dzgals, wgals, ngals, logq=logq)
    else:
        dg = _delta_g_table(amp=0.3)
        cat_field = _catalog(zgals, dzgals, wgals, ngals, delta_g=dg)
        cat_cond = _catalog(zgals, dzgals, wgals, ngals, delta_g=dg)
    shift, expected = _prior_shift(cat_field, cat_cond, cosmo, survey)
    np.testing.assert_allclose(shift, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Q-ensemble marginalization: per-member field Z + shared member index at K>=2
# ---------------------------------------------------------------------------

from darksirens.redshift.completion import (
    build_field_lss_q_member_inputs,
    field_global_log_Z_members,
)


def _logq_members_table(m=3, npix=12):
    base = _logq_table(npix)
    return np.stack([base * s for s in np.linspace(0.6, 1.4, m)], axis=0)


def test_field_Z_members_matches_per_member_scalar():
    """field_global_log_Z_members[m] == field_global_log_Z with member m's Q
    rows installed as THE deterministic table -- one normalizer per member."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cosmo, survey = _cosmo(), _survey()
    logq_m = _logq_members_table()
    cat = _catalog(zgals, dzgals, wgals, ngals, logq=logq_m[0])
    occupied = np.asarray([1, 3, 4, 7])
    qm_occ, qm_empty = build_field_lss_q_member_inputs(logq_m, occupied, 12)
    cat = cat._replace(
        field_lss_q_members=qm_occ, field_lss_q_empty_sum_members=qm_empty,
        lss_completion_logq_members=jnp.asarray(logq_m),
    )
    per_member = np.asarray(field_global_log_Z_members(cosmo, survey, cat))
    assert per_member.shape == (3,)
    for m in range(3):
        cat_m = _catalog(zgals, dzgals, wgals, ngals, logq=logq_m[m])
        expected = float(field_global_log_Z(cosmo, survey, cat_m))
        np.testing.assert_allclose(per_member[m], expected, rtol=1e-12)


def _ensemble_bundle(logq_members, nsamp=2, n_sel=8):
    """Bundle carrying a Q ensemble (mean table = member mean) + field inputs
    incl. the per-member normalizer rows."""
    logq_members = np.asarray(logq_members)
    bundle = _field_bundle(logq=logq_members.mean(axis=0), nsamp=nsamp, n_sel=n_sel)
    occ = np.asarray(bundle["field_occupied_pixels"])
    qm_occ, qm_empty = build_field_lss_q_member_inputs(logq_members, occ, 12)
    bundle["lss_completion_logq_members"] = jnp.asarray(logq_members)
    bundle["field_lss_q_members"] = qm_occ
    bundle["field_lss_q_empty_sum_members"] = qm_empty
    return bundle


def test_k2_duplicated_ensemble_equals_k1_ensemble_under_field():
    """K=2 (A+ens, A+ens) with lss_marginalize == K=1 (A+ens) with
    lss_marginalize at any fcat_2 (the duplicated-catalog identity must
    commute with the shared member index)."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()

    data1 = dict(_shared_physics())
    data1["apix"] = hp.nside2pixarea(1)
    data1["catalogs"] = [_ensemble_bundle(logq_m)]
    opts1 = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides,
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=1, lss_marginalize=True,
    )
    from darksirens.likelihood.factory import make_likelihood as _mk
    val_k1 = float(_mk(opts1, data1, pop_fid, fixed_parameter_values=fixed)(
        jnp.asarray([mid])))
    assert np.isfinite(val_k1)

    data2 = dict(_shared_physics())
    data2["apix"] = hp.nside2pixarea(1)
    data2["catalogs"] = [_ensemble_bundle(logq_m), _ensemble_bundle(logq_m)]
    opts2 = SimpleNamespace(**{**vars(opts1), "n_catalogs": 2})
    val_k2 = float(_mk(opts2, data2, pop_fid, fixed_parameter_values=fixed)(
        jnp.asarray([mid, 0.37])))
    assert abs(val_k2 - val_k1) <= 1e-12


def test_k2_marginalized_equals_logmeanexp_of_member_runs():
    """K=2 lss_marginalize == logsumexp_m[ K=2 deterministic with member m's Q
    as THE table (numerator AND field normalizer) ] - log M: the shared-member
    construction is exactly the member average of coherent per-member runs."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()
    coord = jnp.asarray([mid, 0.37])

    # _bundle_likelihood does not set lss_marginalize; build opts explicitly.
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = [_ensemble_bundle(logq_m), _ensemble_bundle(logq_m)]
    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides,
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=2, lss_marginalize=True,
    )
    from darksirens.likelihood.factory import make_likelihood as _mk
    marg = float(_mk(opts, data, pop_fid, fixed_parameter_values=fixed)(coord))

    member_vals = []
    for m in range(logq_m.shape[0]):
        member_vals.append(_bundle_likelihood(
            [_field_bundle(logq=logq_m[m]), _field_bundle(logq=logq_m[m])],
            fixed, pop_fid, overrides,
        )(coord))
    member_vals = np.asarray([float(v) for v in member_vals])
    expected = float(
        np.log(np.mean(np.exp(member_vals - member_vals.max())))
        + member_vals.max()
    )
    np.testing.assert_allclose(marg, expected, atol=1e-9, rtol=0)


def test_unequal_member_counts_raise_at_k2():
    pop_fid, overrides, fixed, mid = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = [
        _ensemble_bundle(_logq_members_table(m=2)),
        _ensemble_bundle(_logq_members_table(m=3)),
    ]
    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides,
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=2, lss_marginalize=True,
    )
    from darksirens.likelihood.factory import make_likelihood as _mk
    ll = _mk(opts, data, pop_fid, fixed_parameter_values=fixed)
    with pytest.raises(ValueError, match="same M"):
        ll(jnp.asarray([mid, 0.37]))


# ---------------------------------------------------------------------------
# Marked-host field normalizer (field_global_log_Z_marked)
# ---------------------------------------------------------------------------

from darksirens.redshift.completion import (
    build_field_mark_inputs,
    field_global_log_Z_marked,
)
from darksirens.redshift.prior import _mu_miss_from_flat


def _mark_table(npix=12, maxg=3):
    """(npix, maxg) z-centred logmstar-like marks, deterministic pattern."""
    vals = 0.3 * np.sin(0.9 * np.arange(npix))[:, None] + 0.1 * np.arange(maxg)[None, :]
    return vals


def _marked_catalog(eta_active=True):
    """Full-catalog EMCatalog (rows == global pixels, so view == full sky)
    carrying view marks AND the flat full-sky field mark inputs."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    marks = _mark_table()
    cat = _catalog(zgals, dzgals, wgals, ngals)
    fz, fw, fvals = build_field_mark_inputs(
        zgals, wgals, ngals, {"logmstar": marks}, ("logmstar",)
    )
    cat = cat._replace(
        mark_logmstar=jnp.asarray(marks),
        field_mark_z=fz, field_mark_w=fw, field_mark_values=fvals,
    )
    return cat, zgals, wgals, ngals, marks


def test_marked_field_Z_eta0_reduces_to_unmarked():
    """eta = 0 (h == 1) with unit weights: the marked global Z equals the
    unmarked one exactly (S_obs == N_obs_total; mu_miss == 1)."""
    cosmo, survey = _cosmo(), _survey()
    cat, *_ = _marked_catalog()
    n_marks = 1
    log_h_flat = jnp.zeros(cat.field_mark_z.shape[0])
    mu_miss = _mu_miss_from_flat(
        jnp.asarray(cat.field_mark_z), jnp.exp(log_h_flat),
        jnp.ones_like(log_h_flat),
    )
    logZ_marked = float(field_global_log_Z_marked(
        cosmo, survey, cat, mu_miss, log_h_flat
    ))
    logZ_plain = float(field_global_log_Z(cosmo, survey, cat))
    assert logZ_marked == logZ_plain


def test_marked_field_Z_matches_bruteforce_assembly():
    """Z_marked == Sum w_i h_i + integral mu_miss dN_exp V dz, with V from the
    per-pixel NumPy loop and the SAME mu_miss curve (semi-brute force: the
    binned mu estimator is the library's, the Z assembly is independent)."""
    cosmo, survey = _cosmo(), _survey()
    cat, zgals, wgals, ngals, marks = _marked_catalog()
    eta = jnp.asarray([1.3])
    log_h_flat = jnp.clip(cat.field_mark_values @ eta, -7.0, 7.0)
    mu_miss = _mu_miss_from_flat(
        jnp.asarray(cat.field_mark_z), jnp.exp(log_h_flat),
        jnp.ones_like(log_h_flat),
    )
    logZ = float(field_global_log_Z_marked(cosmo, survey, cat, mu_miss, log_h_flat))

    S_obs = float(jnp.sum(jnp.asarray(cat.field_mark_w) * jnp.exp(log_h_flat)))
    # Independent per-pixel V assembly (lss = 1 mode).
    grids = _precompute_grids(cosmo, survey, cat)
    dN_exp = np.asarray(grids.dN_exp)
    dN_exp_safe = np.where(np.asarray(grids.dN_exp_smooth) > 0.0,
                           np.asarray(grids.dN_exp_smooth), 1.0)
    zg = np.asarray(zgrid)
    V = np.zeros_like(zg)
    for p in range(zgals.shape[0]):
        if int(ngals[p]) > 0:
            obs = np.asarray(
                _kde_dndz_obs(p, jnp.asarray(zgals), ngals=jnp.asarray(ngals))
            )
            V += 1.0 - np.clip(obs / dN_exp_safe, 0.0, 1.0)
        else:
            V += 1.0
    Z_bf = S_obs + np.trapezoid(np.asarray(mu_miss) * dN_exp * V, zg)
    np.testing.assert_allclose(logZ, np.log(Z_bf), rtol=1e-5)


def test_marked_field_conditional_shift_identity():
    """The marked field lp differs from the per-pixel-normalized lp of the
    SAME state by exactly log_Z[pix] - log_Z_global: only the normalizer
    swaps, the marked numerator is shared.  (Evaluating both conventions on
    one state sidesteps the f32 flat-marks vs f64 view-marks mu_miss noise a
    cross-state comparison would carry.)"""
    cosmo, survey = _cosmo(), _survey()
    cat, *_ = _marked_catalog()
    z = jnp.asarray([0.10, 0.22, 0.31])
    pix = jnp.asarray([1, 3, 7], dtype=jnp.int32)
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat,
        mark_model="loglinear", mark_params=jnp.asarray([0.9]),
        mark_names=("logmstar",),
        catalog_sky_weighting="field",
    )
    lp_field = eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="field",
    )
    lp_cond = eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="conditional",
    )
    shift = np.asarray(lp_field - lp_cond)
    expected = np.asarray(state.log_Z[pix] - state.log_Z_global)
    np.testing.assert_allclose(shift, expected, rtol=1e-10)


def test_marked_field_pe_and_sel_views_share_global_Z():
    """Two DIFFERENT compact views of the same survey must produce the SAME
    marked global Z (it is built from the view-independent flat full-sky
    inputs), so the constants cancel structurally between the PE and
    selection terms."""
    cosmo, survey = _cosmo(), _survey()
    cat_full, zgals, wgals, ngals, marks = _marked_catalog()
    # A restricted "selection view": only rows {1, 3} (compact, with
    # unique_pixels mapping) -- different kernels, same field inputs.
    rows = np.array([1, 3], dtype=np.int32)
    cat_view = cat_full._replace(
        zgals=jnp.asarray(np.asarray(zgals)[rows]),
        dzgals=jnp.asarray(np.full_like(np.asarray(zgals)[rows], 0.02)),
        wgals=jnp.asarray(np.asarray(wgals)[rows]),
        ngals=jnp.asarray(np.asarray(ngals)[rows]),
        mark_logmstar=jnp.asarray(_mark_table()[rows]),
        unique_pixels=jnp.asarray(rows),
    )
    eta = jnp.asarray([0.7])
    kw = dict(mark_model="loglinear", mark_params=eta,
              mark_names=("logmstar",), catalog_sky_weighting="field")
    st_full = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat_full, **kw)
    st_view = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat_view, **kw)
    assert float(st_full.log_Z_global) == float(st_view.log_Z_global)


def test_marked_field_eta_gradient_finite_and_nonzero():
    cosmo, survey = _cosmo(), _survey()
    cat, *_ = _marked_catalog()
    z = jnp.asarray([0.10, 0.22, 0.31])
    pix = jnp.asarray([1, 3, 7], dtype=jnp.int32)

    def _sum_lp(eta):
        # materialize_state=False: the optimization barriers are for the
        # pre-jit data path; differentiating through prepare outside jit
        # (as NUTS effectively does inside the jitted likelihood) uses the
        # unmaterialized state.
        state = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat,
            mark_model="loglinear", mark_params=eta, mark_names=("logmstar",),
            catalog_sky_weighting="field", materialize_state=False,
        )
        return jnp.sum(eval_redshift_prior_with_state(
            "dark_sirens", state, z, pix, cosmo, survey, cat,
            catalog_sky_weighting="field",
        ))

    g = jax.grad(_sum_lp)(jnp.asarray([0.4]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g[0])) > 0.0


def test_marked_field_requires_flat_mark_inputs():
    """Marks under field WITHOUT the flat full-sky inputs: rejected with the
    build hint (the numerator and normalizer would disagree)."""
    cosmo, survey = _cosmo(), _survey()
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    cat = _catalog(zgals, dzgals, wgals, ngals)._replace(
        mark_logmstar=jnp.asarray(_mark_table())
    )
    with pytest.raises(ValueError, match="field_mark"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat,
            mark_model="loglinear", mark_params=jnp.asarray([0.5]),
            mark_names=("logmstar",), catalog_sky_weighting="field",
        )
