"""
test_marks_lss_marginalize.py
-----------------------------
Composition of the marked-host model (``--mark_model loglinear``: galaxies
reweighted by h(mark|eta), missing budget scaled by mu_miss(z|eta)) with the
LSS-completion ENSEMBLE marginalisation (``--lss_marginalize``:
logL = logsumexp_m logL(Q_m) - log M).

Before this fix the marked branch of ``prepare_redshift_prior_state`` returned a
scalar ``DarkSirenPriorState`` unconditionally and silently dropped the ensemble
member arrays, so a marked + ensemble + ``lss_marginalize`` run raised a
misleading "requires an LSS-completion ENSEMBLE on EVERY PE catalog" error.  The
marked branch now ALSO builds per-member marked missing densities / normalizers
(a ``DarkSirenEnsemblePriorState``) whenever the catalog carries an ensemble,
mirroring the unmarked ensemble design.

The decisive checks are self-consistency at the likelihood level: the marginalised
value equals the log-mean-exp of the per-member likelihoods, where each per-member
value is the *deterministic* marked likelihood run with that member's Q table.
Covered in BOTH conditional and field weighting; plus the K=2 duplicated-catalog
identity, the eta=0 reduction, K=2 per-catalog isolation / zero-weight gradients,
and the posterior-mean regression that pins the state-type change to a no-op when
``lss_marginalize`` is OFF.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    build_pixel_kde_cache,
    completion_curves,
)
from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    _mu_miss_grid,
    DarkSirenEnsemblePriorState,
    DarkSirenPriorState,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.gw.populations import get_fixed_population_params
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.marks import mark_model_parser

NG = int(zgrid.size)
_Z = np.asarray(zgrid)
_LOGQ_CLIP = 7.0
_LOG_H_CLIP = 7.0
COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
# b_miss = 0 so the legacy local-overdensity factor is 1 and the missing branch is
# driven purely by Q_LSS (what the members vary) times mu_miss (what eta varies).
# Low z50 + n0=1 keeps the catalog INcomplete over the event range so both the
# missing branch (hence Q) and the mark reweighting actually move the likelihood.
SURVEY = SurveyParams(n0=1.0, z50=0.15, w=0.08, delta=0.0, b_miss=0.0, alpha_miss=1.0)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))

# z-TILTED members: a constant Q cancels in the normalised prior, so the members
# carry distinct slopes logQ_m(z) = slope_m (z - 0.2).
_SLOPES = np.array([-5.0, -1.5, 1.5, 5.0])
_M = len(_SLOPES)

# Per-galaxy marks (2 rows x 3 slots) with z-varying structure so mu_miss(z|eta)
# is genuinely z-dependent and eta reweights the observed hosts non-trivially.
_MARKS = np.array([[1.0, -0.6, 0.4], [0.7, -0.3, 0.0]])
_MARK_NAMES = ("logmstar",)


def _member_logq(m):
    """(NG,) tilted log Q for member ``m`` over the package zgrid."""
    return _SLOPES[m] * (_Z - 0.2)


def _members_table():
    """(M, n_rows=2, NG) ensemble of member logQ tables."""
    return np.stack([np.broadcast_to(_member_logq(m), (2, NG)) for m in range(_M)])


def _q_mean_table():
    """(n_rows=2, NG) posterior-MEAN Q = mean_m exp(clip(logQ_m)), as a logq."""
    q_mean = np.mean(
        [np.exp(np.clip(_member_logq(m), -_LOGQ_CLIP, _LOGQ_CLIP)) for m in range(_M)],
        axis=0,
    )
    return np.broadcast_to(np.log(q_mean), (2, NG))


def _dark_catalog(*, logq=None, logq_members=None, marks=_MARKS, unit_only=False):
    """Two-row dark-siren catalog (KDE cache, compact rows) carrying either a
    deterministic Q table or an ensemble, plus optional per-galaxy marks."""
    rows = [np.array([0.10, 0.12, 0.15]), np.array([0.28, 0.32])]
    n_rows, nmax = 2, 3
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    for i, r in enumerate(rows):
        zg[i, : len(r)] = r
        dz[i, : len(r)] = 0.003
        w[i, : len(r)] = 1.0
        ng[i] = len(r)
    zg, dz, w, ng = (jnp.asarray(a) for a in (zg, dz, w, ng))
    kde, idx = build_pixel_kde_cache(np.arange(n_rows, dtype=np.int32), zg, n_rows, ngals=ng)
    fields = dict(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((n_rows, NG)), dN_obs_kde=kde, pixel_to_cache_idx=idx,
        unique_pixels=None,
        lss_completion_logq=(None if logq is None else jnp.asarray(logq)),
        lss_completion_logq_members=(
            None if logq_members is None else jnp.asarray(logq_members)
        ),
    )
    if marks is not None:
        fields["mark_logmstar"] = jnp.asarray(marks)
    return EMCatalog(**fields)


def _gw(n_events, n_samp, seed, n_rows=2):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(420.0, 1500.0, total))   # z ~ 0.09-0.30
    chieff = jnp.asarray(rng.uniform(-0.2, 0.2, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.asarray(rng.integers(0, n_rows, total), dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=m2det / m1det, valid=valid)


_N_EV, _N_SAMP, _N_SEL = 4, 64, 300
_GW_PE = _gw(_N_EV, _N_SAMP, seed=0)
_GW_SEL = _gw(_N_SEL, 1, seed=10)


def _ll(cat, *, marginalize=False, mark_model="loglinear", eta=(1.3,)):
    return darksiren_log_likelihood(
        COSMO, SURVEY, POP, _GW_PE, cat, _GW_SEL, cat,
        _N_EV, _N_SAMP, float(_N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=marginalize,
        mark_model=mark_model,
        mark_params=(None if eta is None else jnp.asarray(eta)),
        mark_names=(() if mark_model == "none" else _MARK_NAMES),
    )


# ---------------------------------------------------------------------------
# 1. K=1 conditional identity
# ---------------------------------------------------------------------------

def test_conditional_marginalize_equals_logmeanexp_of_marked_members():
    """marked + ensemble + lss_marginalize == logsumexp_m[ marked deterministic
    run with member m's Q ] - log M (conditional weighting)."""
    eta = (1.3,)
    ll_marg = float(
        _ll(_dark_catalog(logq_members=_members_table()), marginalize=True, eta=eta)
    )
    per_member = jnp.asarray([
        _ll(
            _dark_catalog(logq=np.broadcast_to(_member_logq(m), (2, NG))),
            marginalize=False, eta=eta,
        )
        for m in range(_M)
    ])
    expected = float(logsumexp(per_member) - jnp.log(_M))
    assert np.isfinite(ll_marg)
    np.testing.assert_allclose(ll_marg, expected, rtol=1e-12, atol=0.0)
    # Members genuinely differ (otherwise the identity is vacuous).
    assert float(per_member.max() - per_member.min()) > 1e-3


def test_conditional_single_member_reduces_to_deterministic():
    """One marked member -> marginalisation is that member's deterministic ll."""
    one = np.broadcast_to(_member_logq(2), (1, 2, NG))
    ll_marg = float(_ll(_dark_catalog(logq_members=one), marginalize=True))
    ll_det = float(_ll(_dark_catalog(logq=np.broadcast_to(_member_logq(2), (2, NG)))))
    np.testing.assert_allclose(ll_marg, ll_det, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 3. eta = 0 with unit weights: marked-marginalized == unmarked-marginalized
# ---------------------------------------------------------------------------

def test_eta_zero_marked_marginalize_equals_unmarked_marginalize():
    """loglinear with eta=0 + unit weights: mu_miss(h==1) == 1 exactly and the
    observed marked mass reduces to the count, so the marked marginalized ll
    equals the unmarked marginalized ll."""
    cat = _dark_catalog(logq_members=_members_table())
    ll_marked = float(_ll(cat, marginalize=True, mark_model="loglinear", eta=(0.0,)))
    ll_unmarked = float(_ll(cat, marginalize=True, mark_model="none", eta=None))
    if ll_marked != ll_unmarked:
        # Bitwise equality can miss by ULPs: log_N_host (log-sum-exp of the marked
        # kernel masses) and log_Nobs (direct count log) take different summation
        # paths even though both equal log N_obs for h==1, unit weights.
        np.testing.assert_allclose(ll_marked, ll_unmarked, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 4. K=2 duplicated-catalog identity (conditional)
# ---------------------------------------------------------------------------

def _ll_k2(cat_a, cat_b, log_w, *, marginalize, eta_a=(1.3,), eta_b=(1.3,)):
    """Two-catalog marked marginalized ll via a direct core call (conditional)."""
    pix2_pe = jnp.stack([_GW_PE.pixels, _GW_PE.pixels], axis=1)
    pix2_sel = jnp.stack([_GW_SEL.pixels, _GW_SEL.pixels], axis=1)
    gw_pe = _GW_PE._replace(pixels=pix2_pe)
    gw_sel = _GW_SEL._replace(pixels=pix2_sel)
    return darksiren_log_likelihood(
        COSMO, SURVEY, POP, gw_pe, cat_a, gw_sel, cat_a,
        _N_EV, _N_SAMP, float(_N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=marginalize,
        mark_model="loglinear",
        mark_params_all=(jnp.asarray(eta_a), jnp.asarray(eta_b)),
        mark_names_all=(_MARK_NAMES, _MARK_NAMES),
        n_catalogs=2,
        mixture_surveys=(SURVEY,),
        mixture_em_catalogs_pe=(cat_b,),
        mixture_em_catalogs_sel=(cat_b,),
        mixture_log_weights=jnp.asarray(log_w),
    )


def test_k2_duplicated_marked_marginalize_equals_k1_conditional():
    """K=2 (A+marks+ens, A+marks+ens) with tied etas and weights summing to 1
    == K=1 (A+marks+ens), both marginalized (conditional weighting)."""
    eta = (0.9,)
    cat = _dark_catalog(logq_members=_members_table())
    val_k1 = float(_ll(cat, marginalize=True, eta=eta))
    log_w = np.log(np.array([0.6, 0.4]))
    val_k2 = float(_ll_k2(cat, cat, log_w, marginalize=True, eta_a=eta, eta_b=eta))
    assert np.isfinite(val_k1)
    np.testing.assert_allclose(val_k2, val_k1, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 7. Posterior-mean regression: state-type change is a no-op with marginalize OFF
# ---------------------------------------------------------------------------

def test_marginalize_off_matches_posterior_mean_scalar_path():
    """marks + ensemble + lss_marginalize OFF == marks + a Q input carrying ONLY
    the posterior-mean table (no members): the scalar path is unmoved by the
    state-type change (scalar DarkSirenPriorState -> DarkSirenEnsemblePriorState
    whose scalar fields are identical)."""
    ll_ens = float(_ll(_dark_catalog(logq_members=_members_table()), marginalize=False))
    ll_mean = float(_ll(_dark_catalog(logq=_q_mean_table()), marginalize=False))
    assert np.isfinite(ll_ens)
    np.testing.assert_allclose(ll_ens, ll_mean, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 8. State-shape unit test: marked member density == curves members * mu_miss
# ---------------------------------------------------------------------------

def test_marked_members_state_shape_and_broadcast():
    """prepare_redshift_prior_state with marks+members returns a
    DarkSirenEnsemblePriorState whose member-INDEPENDENT base curve is
    curves.base_miss * mu_miss[None,:] (the (M,N_rows,N_grid) cube is no longer
    materialised; each member's marked density is base_miss * Q_eff_m,
    reconstructed at the query brackets)."""
    cat = _dark_catalog(logq_members=_members_table())
    eta = jnp.asarray([1.1])
    state = prepare_redshift_prior_state(
        "dark_sirens", COSMO, SURVEY, cat,
        mark_model="loglinear", mark_params=eta, mark_names=_MARK_NAMES,
        materialize_state=False,
    )
    assert isinstance(state, DarkSirenEnsemblePriorState)
    curves = completion_curves(COSMO, SURVEY, cat)
    assert curves.base_miss is not None
    log_h = jnp.clip(
        mark_model_parser("loglinear", _MARK_NAMES)(cat, eta), -_LOG_H_CLIP, _LOG_H_CLIP
    )
    mu_miss = _mu_miss_grid(cat, log_h)
    # The marked base curve folds mu_miss into the member-independent base.
    expected_base = np.asarray(curves.base_miss) * np.asarray(mu_miss)[None, :]
    np.testing.assert_allclose(np.asarray(state.base_miss), expected_base, rtol=0, atol=0)
    assert state.base_miss.shape == (2, NG)
    # The compact per-member missing mass has the member axis; scalar-compat
    # field is the posterior-mean compose.
    assert state.log_Z_members.shape == (_M, 2)
    np.testing.assert_allclose(
        np.asarray(state.dN_miss),
        np.asarray(curves.dN_miss) * np.asarray(mu_miss)[None, :],
        rtol=0, atol=0,
    )


def test_marked_no_members_stays_scalar_state():
    """With marks but NO ensemble the state is still the scalar
    DarkSirenPriorState (only the ensemble case changes type)."""
    cat = _dark_catalog(logq=np.zeros((2, NG)))
    state = prepare_redshift_prior_state(
        "dark_sirens", COSMO, SURVEY, cat,
        mark_model="loglinear", mark_params=jnp.asarray([0.5]),
        mark_names=_MARK_NAMES, materialize_state=False,
    )
    assert isinstance(state, DarkSirenPriorState)
    assert not isinstance(state, DarkSirenEnsemblePriorState)


# ===========================================================================
# FIELD weighting (bundle source via make_likelihood)
# ===========================================================================
import healpy as hp
from types import SimpleNamespace

from darksirens.redshift.completion import (
    build_field_normalization_inputs,
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
    build_field_mark_inputs,
)
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params as _get_fixed
from darksirens.likelihood.factory import make_likelihood

_NGf = NG


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
    base = np.linspace(-0.4, 0.4, NG)
    return np.array([base * np.cos(0.7 * p) for p in range(npix)])


def _logq_members_table(m=3, npix=12):
    base = _logq_table(npix)
    return np.stack([base * s for s in np.linspace(0.6, 1.4, m)], axis=0)


def _mark_table(npix=12, maxg=3, scale=1.0):
    return scale * (
        0.3 * np.sin(0.9 * np.arange(npix))[:, None] + 0.1 * np.arange(maxg)[None, :]
    )


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


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = _get_fixed("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled}
    overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    mid = 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))
    return pop_fid, overrides, fixed, mid


def _marked_field_bundle(logq=None, logq_members=None, mark_scale=1.0, nsamp=2, n_sel=8):
    """One-catalog bundle over the 12-pixel synthetic sky with field inputs,
    optionally a deterministic Q or a Q ensemble, and optionally marks."""
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
    occ = np.asarray(field.occupied_pixels)
    if logq_members is not None:
        logq_members = np.asarray(logq_members)
        mean_logq = logq_members.mean(axis=0)  # posterior-mean scalar table
        q_occ, q_empty = build_field_lss_q_inputs(jnp.asarray(mean_logq), occ, 12)
        qm_occ, qm_empty = build_field_lss_q_member_inputs(logq_members, occ, 12)
        bundle["lss_completion_logq"] = jnp.asarray(mean_logq)
        bundle["lss_completion_logq_members"] = jnp.asarray(logq_members)
        bundle["lss_completion_indexing"] = 2
        bundle["field_lss_q"] = q_occ
        bundle["field_lss_q_empty_sum"] = q_empty
        bundle["field_lss_q_members"] = qm_occ
        bundle["field_lss_q_empty_sum_members"] = qm_empty
    elif logq is not None:
        q_occ, q_empty = build_field_lss_q_inputs(jnp.asarray(logq), occ, 12)
        bundle["lss_completion_logq"] = jnp.asarray(logq)
        bundle["lss_completion_indexing"] = 2
        bundle["field_lss_q"] = q_occ
        bundle["field_lss_q_empty_sum"] = q_empty
    if mark_scale is not None:
        marks = _mark_table(scale=mark_scale)
        bundle["mark_logmstar"] = jnp.asarray(marks)
        fz, fw, fvals = build_field_mark_inputs(
            zgals, wgals, ngals, {"logmstar": marks}, ("logmstar",)
        )
        bundle["field_mark_z"] = fz
        bundle["field_mark_w"] = fw
        bundle["field_mark_values"] = fvals
    return bundle


def _field_opts(n_catalogs, overrides, fixed, mark_names_by_catalog,
                *, lss_marginalize=False, barrier="auto"):
    mark_names = tuple(mark_names_by_catalog[0]) if mark_names_by_catalog else ()
    return SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides, fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume", bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=n_catalogs,
        lss_marginalize=lss_marginalize,
        mark_model="loglinear", mark_names=mark_names,
        mark_names_by_catalog=tuple(tuple(n) for n in mark_names_by_catalog),
        redshift_prior_barrier=barrier,
    )


def _field_likelihood(bundles, mark_names_by_catalog, fixed, pop_fid, overrides,
                      *, lss_marginalize=False, barrier="auto"):
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = list(bundles)
    opts = _field_opts(len(bundles), overrides, fixed, mark_names_by_catalog,
                       lss_marginalize=lss_marginalize, barrier=barrier)
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


# ---------------------------------------------------------------------------
# 2. K=1 field identity
# ---------------------------------------------------------------------------

def test_field_marginalize_equals_logmeanexp_of_marked_member_runs():
    """K=1 field: marked + ensemble + lss_marginalize == logsumexp_m[ K=1 field
    deterministic marked run with member m's Q as THE table ] - log M."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()
    # K=1 marked run samples [pop_sampled, eta_logmstar].
    coord = jnp.asarray([mid, 0.7])

    marg = float(_field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0)],
        (("logmstar",),), fixed, pop_fid, overrides, lss_marginalize=True,
    )(coord))
    assert np.isfinite(marg)

    member_vals = []
    for m in range(logq_m.shape[0]):
        member_vals.append(float(_field_likelihood(
            [_marked_field_bundle(logq=logq_m[m], mark_scale=1.0)],
            (("logmstar",),), fixed, pop_fid, overrides, lss_marginalize=False,
        )(coord)))
    member_vals = np.asarray(member_vals)
    expected = float(
        np.log(np.mean(np.exp(member_vals - member_vals.max()))) + member_vals.max()
    )
    np.testing.assert_allclose(marg, expected, rtol=1e-11, atol=1e-9)
    assert float(member_vals.max() - member_vals.min()) > 1e-4


# ---------------------------------------------------------------------------
# 4 (field). K=2 duplicated-catalog identity under field weighting
# ---------------------------------------------------------------------------

def test_k2_duplicated_marked_ensemble_equals_k1_under_field():
    """K=2 (A+marks+ens, A+marks+ens) with tied etas and lss_marginalize ==
    K=1 (A+marks+ens) marginalized at any fcat_2 (field weighting)."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()
    eta_val = 0.8

    val_k1 = float(_field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0)],
        (("logmstar",),), fixed, pop_fid, overrides, lss_marginalize=True,
    )(jnp.asarray([mid, eta_val])))
    assert np.isfinite(val_k1)

    fixed_k2 = dict(fixed)
    fixed_k2["eta_logmstar_c2"] = eta_val
    val_k2 = float(_field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         _marked_field_bundle(logq_members=logq_m, mark_scale=1.0)],
        (("logmstar",), ("logmstar",)), fixed_k2, pop_fid, overrides,
        lss_marginalize=True,
    )(jnp.asarray([mid, 0.37, eta_val])))
    np.testing.assert_allclose(val_k2, val_k1, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. K=2 one marked + one unmarked, both ensembles, marginalize: finite grads,
#    per-catalog eta isolation
# ---------------------------------------------------------------------------

def test_k2_mixed_marks_ensemble_marginalize_finite_and_isolated():
    """K=2 both-with-ensembles + marginalize: finite ll and finite grads; the
    eta_logmstar_c2 label exists (and its grad is live) only when catalog 2 is
    marked, mirroring the per-catalog isolation of the mark blocks."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()

    # Catalog 2 UNMARKED: no eta_c2 coordinate -> coord is [pop, fcat_2, eta_c1].
    ll_c1only = _field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         _marked_field_bundle(logq_members=logq_m, mark_scale=None)],
        (("logmstar",), ()), fixed, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    coord3 = jnp.asarray([mid, 0.5, 0.6])
    v3 = float(ll_c1only(coord3))
    g3 = np.asarray(jax.grad(lambda c: ll_c1only(c))(coord3))
    assert np.isfinite(v3)
    assert np.all(np.isfinite(g3))
    assert g3.shape == (3,)  # no eta_c2 label

    # Catalog 2 MARKED: eta_c2 present -> coord is [pop, fcat_2, eta_c1, eta_c2].
    ll_both = _field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         _marked_field_bundle(logq_members=logq_m, mark_scale=1.4)],
        (("logmstar",), ("logmstar",)), fixed, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    coord4 = jnp.asarray([mid, 0.5, 0.6, 0.3])
    v4 = float(ll_both(coord4))
    g4 = np.asarray(jax.grad(lambda c: ll_both(c))(coord4))
    assert np.isfinite(v4)
    assert np.all(np.isfinite(g4))
    assert g4.shape == (4,)
    assert abs(float(g4[3])) > 0.0  # eta_c2 is live at interior fcat_2


# ---------------------------------------------------------------------------
# 6. Zero-weight catalog: grad wrt eta_c2 == 0 exactly at fcat_2 = 0
# ---------------------------------------------------------------------------

def test_k2_zero_weight_catalog_eta_grad_is_zero():
    """At fcat_2 = 0 the second catalog carries zero mixture weight (log_w =
    -inf), so it drops from the logsumexp and grad wrt eta_logmstar_c2 == 0
    exactly, even under marginalization."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    logq_m = _logq_members_table()
    # FIX the fcat_2 stick at the boundary 0 (not sampled): the stick transform
    # has an infinite derivative there, so sampling fcat_2 = 0 would make the
    # fcat_2 gradient itself NaN -- unrelated to the eta_c2 isolation under test.
    # With w_2 = 0 the sampled coord is [pop, eta_c1, eta_c2] and eta_c2 (index 2)
    # must have exactly-zero gradient.
    fixed_w0 = dict(fixed)
    fixed_w0["fcat_2"] = 0.0
    ll = _field_likelihood(
        [_marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         _marked_field_bundle(logq_members=logq_m, mark_scale=1.4)],
        (("logmstar",), ("logmstar",)), fixed_w0, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    coord = jnp.asarray([mid, 0.6, 0.3])  # [pop, eta_c1, eta_c2]; w_2 = 0
    g = np.asarray(jax.grad(lambda c: ll(c))(coord))
    assert np.all(np.isfinite(g))
    assert float(g[2]) == 0.0  # eta_c2 exactly inert for a zero-weight catalog
