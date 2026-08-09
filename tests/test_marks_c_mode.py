"""Marked-host model x completeness estimator mode (the untested cell of the
compatibility matrix).

``field_global_log_Z_marked`` never looks at ``survey.c_mode``: it reuses
``_field_missing_curve``, which resolves the per-pixel / aggregate / selection
branch from ``grids.C_bar_raw``.  So the marked normalizer inherits whichever
budget the unmarked one carries, and the marked modulation ``mu_miss(z|eta)``
rides ON TOP of that ``(1 - C_bar)`` budget.  Pinned here for c_mode
``aggregate`` and ``selection``:

  (1) the missing curve inside the marked normalizer IS the plain
      ``_field_missing_curve`` -- one formation site, hand-checked against
      ``(1 - C_bar) * N_pix_total`` with the beyond-depth relaxation;
  (2) ``mu_miss`` multiplies that budget and nothing else (``mu == 1`` returns
      the plain missing mass);
  (3) ``eta = 0`` reduces the marked normalizer to ``field_global_log_Z``
      EXACTLY, with and without a biting ``z_depth``;
  (4) the c_mode really reaches the marked normalizer (per-pixel / aggregate /
      selection give three different budgets; ``M0hat`` moves the selection one);
  (5) the prior state, the jitted/differentiated state, the assembled
      likelihood and the sampled parameter space all survive the combination.
"""
import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from darksirens.core.types import (  # noqa: E402
    C_MODE_AGGREGATE_STRUCT,
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.gw.populations import pop_model_prior_parser  # noqa: E402
from darksirens.gw.populations.registry import (  # noqa: E402
    get_fixed_population_params,
)
from darksirens.inference.parameters import build_parameter_decoder  # noqa: E402
from darksirens.inference.prior import build_parameter_space  # noqa: E402
from darksirens.likelihood.factory import make_likelihood  # noqa: E402
from darksirens.marks import mark_model_flat_parser  # noqa: E402
from darksirens.redshift import zgrid  # noqa: E402
from darksirens.redshift.completion import (  # noqa: E402
    _field_missing_curve,
    _precompute_grids,
    build_field_depth_inputs,
    build_field_mark_inputs,
    build_field_normalization_inputs,
    field_global_log_Z,
    field_global_log_Z_marked,
    field_marked_observed_global_total,
    field_observed_global_total,
)
from darksirens.redshift.prior import (  # noqa: E402
    _mu_miss_from_flat,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.redshift.selection import c_sel_gaussian  # noqa: E402

NG = int(zgrid.size)
N_PIX = 12                      # nside=1 full sky: rows ARE global pixels
APIX = hp.nside2pixarea(1)
#: n0 chosen so the AGGREGATE Cbar is genuinely in (0, 1) on this 7-galaxy sky
#: (peak 0.42, no clip engages) -- at the usual 1e-2 the sky-aggregate ratio
#: underflows to ~1e-7 and the aggregate branch is indistinguishable from
#: per_pixel, which would make every assertion below vacuous.
N0 = 3e-9
#: Selection theta at the SurveyParams fiducials (core/constants.py), so the
#: same numbers reappear in the decoder/likelihood tests below.
THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)
#: Both c_modes carried as the leaf-less STRUCTURAL sentinels (never ints):
#: an int leaf hard-errors at a jit boundary by design.
C_MODES_UNDER_TEST = [
    pytest.param(C_MODE_AGGREGATE_STRUCT, id="aggregate"),
    pytest.param(C_MODE_SELECTION_STRUCT, id="selection"),
]


def _cosmo(H0=67.74):
    return CosmoParams(H0=H0, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(c_mode=None, z_depth=None, **kw):
    base = dict(n0=N0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
                sigma_kde=0.0, z_depth=z_depth)
    if c_mode is not None:
        base["c_mode"] = c_mode
    if c_mode is C_MODE_SELECTION_STRUCT:
        base.update(THETA)
    base.update(kw)
    return SurveyParams(**base)


def _synthetic_full_sky(npix=N_PIX, maxg=3):
    """4 occupied of 12 pixels, 7 galaxies -- the tests/test_field_normalizer_
    modulations.py sky, reused so the two files share one geometry."""
    zgals = np.zeros((npix, maxg))
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    occ = {1: [0.10], 3: [0.20, 0.25], 4: [0.15], 7: [0.30, 0.32, 0.28]}
    for p, zs in occ.items():
        for j, z in enumerate(zs):
            zgals[p, j] = z
            wgals[p, j] = 1.0
        ngals[p] = len(zs)
    return zgals, np.full((npix, maxg), 0.02), wgals, ngals


def _mark_table(npix=N_PIX, maxg=3):
    """(npix, maxg) z-centred logmstar-like marks, deterministic pattern."""
    return (0.3 * np.sin(0.9 * np.arange(npix))[:, None]
            + 0.1 * np.arange(maxg)[None, :])


def _marked_catalog():
    """Full-sky EMCatalog with the field normalization inputs, the flat FULL-SKY
    depth inputs (read only when z_depth is set) and both the view marks and the
    flat full-sky mark inputs the marked field normalizer requires."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals))
    depth = build_field_depth_inputs(
        jnp.asarray(zgals), jnp.asarray(dzgals), jnp.asarray(wgals),
        jnp.asarray(ngals))
    marks = _mark_table()
    fz, fw, fvals = build_field_mark_inputs(
        zgals, wgals, ngals, {"logmstar": marks}, ("logmstar",))
    return EMCatalog(
        apix=APIX, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=jnp.asarray(float(field.n_empty)),
        field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
        field_occupied_pixels=jnp.asarray(np.asarray(field.occupied_pixels),
                                          dtype=jnp.int32),
        field_depth_z=depth.z, field_depth_dz=depth.dz, field_depth_c=depth.c,
        mark_logmstar=jnp.asarray(marks),
        field_mark_z=fz, field_mark_w=fw, field_mark_values=fvals,
    )


def _flat_marks(cat, eta):
    """``(log_h_flat, mu_miss)`` built through the LIBRARY parser, so the pair
    is op-for-op what prepare_redshift_prior_state builds (the parser casts eta
    to the f32 mark dtype; a hand ``values @ eta`` promotes to f64 and drifts by
    ~1e-10 in log Z)."""
    log_h_flat = jnp.clip(
        mark_model_flat_parser("loglinear", ("logmstar",))(
            cat.field_mark_values, jnp.asarray([eta])),
        -7.0, 7.0)
    mu_miss = _mu_miss_from_flat(
        jnp.asarray(cat.field_mark_z), jnp.exp(log_h_flat),
        jnp.ones_like(log_h_flat))
    return log_h_flat, mu_miss


def _hand_budget(cosmo, survey, cat, c_mode):
    """Independent NumPy ``(C_bar, V_total)`` for the shared-curve modes.

    ``C_bar`` is rebuilt from the mode's own definition -- the sky-aggregate
    ratio over ``N_pix_total = round(4 pi / apix)`` for aggregate, the analytic
    Phi curve for selection -- and the budget is ``(1 - C_bar)`` times the
    OCCUPIED + EMPTY pixel count (spelled from the catalog's own rows, not from
    N_PIX, so a fixture change cannot silently conflate the two counts), with
    the beyond-depth relaxation to the bare pixel count applied last.
    """
    grids = _precompute_grids(cosmo, survey, cat)
    smooth = np.asarray(grids.dN_exp_smooth)
    safe = np.where(smooth > 0.0, smooth, 1.0)
    if c_mode is C_MODE_AGGREGATE_STRUCT:
        obs_sum = np.sum(np.asarray(cat.field_dN_obs_s, dtype=np.float64), axis=0)
        n_pix_total = round(4.0 * np.pi / APIX)
        C_bar = np.clip(obs_sum / (n_pix_total * safe), 0.0, 1.0)
    else:
        C_bar = np.clip(np.asarray(c_sel_gaussian(
            zgrid, THETA["m_lim"], THETA["M0hat"], THETA["sigma_M"],
            cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)), 0.0, 1.0)
    n_budget = float(np.asarray(cat.field_dN_obs_s).shape[0]) + float(
        np.asarray(cat.field_n_empty))
    V = (1.0 - C_bar) * n_budget
    if survey.z_depth is not None:
        V = np.where(np.asarray(zgrid) <= survey.z_depth, V, n_budget)
    return C_bar, V, np.asarray(grids.dN_exp)


# ------------------------------------------------------------------
# The normalizer itself
# ------------------------------------------------------------------

@pytest.mark.parametrize("z_depth", [None, 0.22])
@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_marked_missing_curve_is_the_plain_field_curve(c_mode, z_depth):
    """The curve pinned here is the SAME ``_field_missing_curve`` that
    ``field_global_log_Z_marked`` calls, so hand-checking it against
    ``(1 - C_bar) * (n_occ + n_empty)`` pins the missing curve INSIDE the
    marked normalizer for both shared-curve c_modes."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode, z_depth=z_depth)

    V_lib, dN_exp = _field_missing_curve(cosmo, survey, cat)
    C_bar, V_hand, dN_hand = _hand_budget(cosmo, survey, cat, c_mode)
    np.testing.assert_allclose(np.asarray(V_lib), V_hand, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(dN_exp), dN_hand)
    # the curve is non-degenerate: the shared budget is neither saturated nor dead
    assert 0.0 < float(C_bar.max()) and float(C_bar.min()) < 1.0


@pytest.mark.parametrize("z_depth", [None, 0.22])
@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_marked_field_Z_equals_hand_budget_with_mu_on_top(c_mode, z_depth):
    """The whole marked ``Z`` reassembled from an independently written
    ``(1 - C_bar) * N_pix`` budget with ``mu_miss`` inserted before the
    quadrature: the marked modulation multiplies the c_mode budget and the
    observed marked mass is added untouched."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode, z_depth=z_depth)

    log_h_flat, mu_miss = _flat_marks(cat, 1.3)
    _C, V_hand, dN_exp = _hand_budget(cosmo, survey, cat, c_mode)
    S_obs = field_marked_observed_global_total(cosmo, survey, cat, log_h_flat)
    Z_hand = S_obs + jnp.trapezoid(
        jnp.asarray(mu_miss) * jnp.asarray(dN_exp) * jnp.asarray(V_hand), zgrid)
    logZ = float(field_global_log_Z_marked(cosmo, survey, cat, mu_miss, log_h_flat))
    np.testing.assert_allclose(logZ, float(jnp.log(Z_hand)), rtol=1e-12)


@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_mu_one_returns_the_plain_missing_mass(c_mode):
    """``mu_miss`` applies ON TOP of the ``(1 - C_bar)`` budget, in isolation:
    with the modulation switched off (but a NONZERO ``log_h_flat``, so the
    observed term genuinely differs) the marked missing mass IS the unmarked
    one."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode)

    log_h_flat, _mu = _flat_marks(cat, 1.3)
    S_obs = float(field_marked_observed_global_total(cosmo, survey, cat, log_h_flat))
    logZ_mu1 = float(field_global_log_Z_marked(
        cosmo, survey, cat, jnp.ones(NG), log_h_flat))
    logZ_plain = float(field_global_log_Z(cosmo, survey, cat))
    N_obs = float(field_observed_global_total(cosmo, survey, cat))
    np.testing.assert_allclose(np.exp(logZ_mu1) - S_obs,
                               np.exp(logZ_plain) - N_obs, rtol=1e-9)


@pytest.mark.parametrize("z_depth", [None, 0.22])
@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_marked_field_Z_eta0_reduces_to_unmarked_exactly(c_mode, z_depth):
    """``eta = 0`` is an EXACT (bit-identical) reduction to the unmarked
    normalizer under both shared-curve c_modes, with and without a biting
    depth (0.22 cuts the 0.25/0.28/0.30/0.32 galaxies)."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode, z_depth=z_depth)

    log_h_flat, mu_miss = _flat_marks(cat, 0.0)
    # h == 1 makes the binned estimator EXACTLY 1 on every grid node (sums of
    # 1.0 / counts, interpolated between equal values) -- assert it, so a change
    # to _mu_miss_from_flat that breaks the reduction is attributed here, not to
    # the normalizer.
    np.testing.assert_array_equal(np.asarray(mu_miss), np.ones(NG))
    assert float(field_global_log_Z_marked(cosmo, survey, cat, mu_miss, log_h_flat)) \
        == float(field_global_log_Z(cosmo, survey, cat))


def test_c_mode_reaches_the_marked_normalizer():
    """Non-inertness pin: per_pixel / aggregate / selection deliver three
    genuinely different budgets (and different marked ``Z``) to
    ``field_global_log_Z_marked``, and the selection theta steers it -- so the
    reductions above cannot pass vacuously."""
    cat, cosmo = _marked_catalog(), _cosmo()
    log_h_flat, mu_miss = _flat_marks(cat, 1.3)
    V = {}
    Z = {}
    for key, cm in (("per_pixel", None), ("aggregate", C_MODE_AGGREGATE_STRUCT),
                    ("selection", C_MODE_SELECTION_STRUCT)):
        sv = _survey(cm)
        V[key] = np.asarray(_field_missing_curve(cosmo, sv, cat)[0])
        Z[key] = float(field_global_log_Z_marked(cosmo, sv, cat, mu_miss, log_h_flat))
    assert np.max(np.abs(V["aggregate"] - V["per_pixel"])) > 1.0
    assert np.max(np.abs(V["selection"] - V["per_pixel"])) > 1.0
    assert abs(Z["aggregate"] - Z["per_pixel"]) > 1e-6
    assert abs(Z["selection"] - Z["per_pixel"]) > 1e-6
    # and the SELECTION theta steers the marked normalizer (the parametric channel)
    sv_sel = _survey(C_MODE_SELECTION_STRUCT)
    brighter = float(field_global_log_Z_marked(
        cosmo, sv_sel._replace(M0hat=THETA["M0hat"] - 1.0), cat, mu_miss, log_h_flat))
    assert abs(brighter - Z["selection"]) > 1e-3


# ------------------------------------------------------------------
# The composed prior state
# ------------------------------------------------------------------

@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_marked_state_carries_the_c_mode_normalizer(c_mode):
    """The state's ``log_Z_global`` IS the marked normalizer under the c_mode,
    and the field/conditional evaluations still differ by exactly
    ``log_Z[pix] - log_Z_global`` (the numerator/normalizer identity)."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode)
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat,
        mark_model="loglinear", mark_params=jnp.asarray([0.9]),
        mark_names=("logmstar",), catalog_sky_weighting="field")
    log_h_flat, mu_miss = _flat_marks(cat, 0.9)
    assert float(state.log_Z_global) == float(
        field_global_log_Z_marked(cosmo, survey, cat, mu_miss, log_h_flat))

    z = jnp.asarray([0.10, 0.22, 0.31])
    pix = jnp.asarray([1, 3, 7], dtype=jnp.int32)
    lp_field = eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="field")
    lp_cond = eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="conditional")
    np.testing.assert_allclose(
        np.asarray(lp_field - lp_cond),
        np.asarray(state.log_Z[pix] - state.log_Z_global), rtol=1e-10)


@pytest.mark.parametrize("c_mode", C_MODES_UNDER_TEST)
def test_marked_state_is_jit_and_grad_compatible(c_mode):
    """The marked state traces and differentiates under both shared-curve
    c_modes -- in eta, and (selection only) in the parametric M0hat."""
    cat, cosmo = _marked_catalog(), _cosmo()
    survey = _survey(c_mode)
    z = jnp.asarray([0.10, 0.22, 0.31])
    pix = jnp.asarray([1, 3, 7], dtype=jnp.int32)

    def _sum_lp(eta, sv=survey):
        # materialize_state=False: the optimization barriers on the materialized
        # state have no differentiation rule.
        st = prepare_redshift_prior_state(
            "dark_sirens", cosmo, sv, cat, mark_model="loglinear",
            mark_params=eta, mark_names=("logmstar",),
            catalog_sky_weighting="field", materialize_state=False)
        return jnp.sum(eval_redshift_prior_with_state(
            "dark_sirens", st, z, pix, cosmo, sv, cat,
            catalog_sky_weighting="field"))

    assert np.isfinite(float(jax.jit(_sum_lp)(jnp.asarray([0.4]))))
    g = jax.grad(_sum_lp)(jnp.asarray([0.4]))
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g[0])) > 0.0
    if c_mode is C_MODE_SELECTION_STRUCT:
        gm = jax.grad(lambda m0: _sum_lp(jnp.asarray([0.4]),
                                         survey._replace(M0hat=m0)))(THETA["M0hat"])
        assert np.isfinite(float(gm)) and abs(float(gm)) > 0.0


# ------------------------------------------------------------------
# Likelihood-level smoke and parameter-space pin
# ------------------------------------------------------------------

def _shared_physics(nsamp=2, n_sel=64):
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


def _marked_bundle(nsamp=2, n_sel=64):
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    pe_pix = np.array([7, 7], dtype=np.int32)[:nsamp]
    sel_pix = np.resize(np.array([1, 3, 4, 7], dtype=np.int32), n_sel)
    up_pe, s2u_pe = np.unique(pe_pix, return_inverse=True)
    up_se, s2u_se = np.unique(sel_pix, return_inverse=True)
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals))
    marks = _mark_table()
    fz, fw, fvals = build_field_mark_inputs(
        zgals, wgals, ngals, {"logmstar": marks}, ("logmstar",))
    return dict(
        nside=1, apix=APIX, n_pix_catalog=N_PIX,
        delta_g_pix_z=jnp.zeros((1, NG)),
        zgals_pe=zgals[up_pe], dzgals_pe=dzgals[up_pe], wgals_pe=wgals[up_pe],
        ngals_pe=ngals[up_pe], unique_pixels_pe=up_pe.astype(np.int32),
        sample_to_unique_pe=s2u_pe.astype(np.int32),
        zgals_sel=zgals[up_se], dzgals_sel=dzgals[up_se],
        wgals_sel=wgals[up_se], ngals_sel=ngals[up_se],
        unique_pixels_sel=up_se.astype(np.int32),
        sample_to_unique_sel=s2u_se.astype(np.int32),
        field_dN_obs_s=field.dN_obs_s, field_n_empty=field.n_empty,
        field_N_obs_total=field.N_obs_total,
        field_occupied_pixels=field.occupied_pixels,
        mark_logmstar=jnp.asarray(marks),
        field_mark_z=fz, field_mark_w=fw, field_mark_values=fvals,
    )


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i])
             for i, lbl in enumerate(pop_labels) if lbl != sampled}
    overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    mid = 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))
    return pop_fid, overrides, fixed, mid


@pytest.mark.parametrize("c_mode, extra_fixed",
                         [("aggregate", {}), ("selection", {"m_lim": 20.0})])
def test_marked_likelihood_finite_and_eta_live(c_mode, extra_fixed):
    """The assembled likelihood survives marks x c_mode and eta stays live.

    Two fixture constraints, both forced by the selection cell:

    * ``n_sel = 64`` rather than the 8 the sibling marks tests use -- the
      parametric budget concentrates the redshift prior far harder than the
      counts estimators, so at 8 injections Neff falls under the selection-term
      variance guard and the likelihood is exactly ``-inf`` (correct guard
      behaviour, unusable as a smoke test).
    * ``m_lim = 20.0`` pinned through ``fixed_parameter_values`` (a
      ``_PINNED_SURVEY_PARAMS`` entry with ``fixed_ok=True``, so this is the
      endorsed path): at the fiducial 24.0 the survey is near-complete over the
      injection redshifts and the run sits inside the guard's soft wall.

    No ``jax.grad`` here -- ``make_likelihood``'s jitted core contains
    ``optimization_barrier``, which has no differentiation rule; T7 covers
    differentiability instead.
    """
    pop_fid, overrides, fixed, mid = _pop_bits()
    fixed = dict(fixed, **extra_fixed)
    data = dict(_shared_physics())
    data["apix"] = APIX
    data["catalogs"] = [_marked_bundle()]
    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides, fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume", bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=1,
        mark_model="loglinear", mark_names=("logmstar",),
        mark_names_by_catalog=(("logmstar",),),
        c_mode=c_mode, use_LSS=False,
    )
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    # sampled coords: [pop, eta_logmstar]  (fix_survey drops the survey block)
    vals = [float(ll(jnp.asarray([mid, e]))) for e in (-1.2, 0.0, 1.2)]
    assert np.all(np.isfinite(vals))
    assert len(set(vals)) == 3          # eta is genuinely live under this c_mode


def test_selection_and_mark_blocks_sample_without_collision():
    """The selection block and the eta block occupy disjoint, ordered slots:
    the c_mode adds exactly (M0hat, sigma_M) ahead of the marks, the
    magnitude-fit prior flips only those two, and the decoder routes each
    coordinate to its own destination (an off-by-one would feed eta into
    sigma_M)."""
    kw = dict(universe_model="dark_sirens", use_lss=False,
              mark_model="loglinear", mark_names=("logmstar", "logssfr"))
    labels = build_parameter_space("powerlaw+peak", True, True, False,
                                   c_mode="selection", **kw)[0]
    assert labels == ["log10n0", "delta", "sigma_kde",
                      "M0hat", "sigma_M", "eta_logmstar", "eta_logssfr"]
    assert len(set(labels)) == len(labels)
    # the selection pair is exactly what the c_mode adds -- nothing else moves
    agg = build_parameter_space("powerlaw+peak", True, True, False,
                                c_mode="aggregate", **kw)[0]
    assert agg == ["log10n0", "delta", "sigma_kde", "eta_logmstar", "eta_logssfr"]

    # the magnitude-fit prior flips ONLY the selection labels; the eta block
    # stays flat
    kinds = dict(zip(*[build_parameter_space(
        "powerlaw+peak", True, True, False, c_mode="selection",
        selection_prior={"M0hat": (-20.19, 0.02), "sigma_M": (1.0, 0.015)},
        **kw)[i] for i in (0, 11)]))
    assert kinds["M0hat"] == ("normal", -20.19, 0.02)
    assert kinds["sigma_M"] == ("normal", 1.0, 0.015)
    assert kinds["eta_logmstar"] == ("uniform", None, None)

    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        fix_population=True, fix_cosmology=True, fix_survey=False,
        prior_overrides=None, fixed_parameter_values={},
        mark_model="loglinear", mark_names=("logmstar", "logssfr"),
        n_catalogs=1, complete_empty_pixel_policy="volume",
        c_mode="selection", use_LSS=False)
    dec = build_parameter_decoder(opts, get_fixed_population_params("powerlaw+peak"))
    assert list(dec.sampled_labels) == labels
    _c, sv, _p, _sky, mark_params = dec.decode(
        jnp.asarray([-2.5, 0.3, 0.01, -20.7, 1.4, 0.8, -0.6]))
    assert float(sv.M0hat) == -20.7 and float(sv.sigma_M) == 1.4
    assert float(sv.m_lim) == THETA["m_lim"]        # pinned datum, never sampled
    np.testing.assert_allclose(np.asarray(mark_params), [0.8, -0.6])
    np.testing.assert_allclose(float(sv.n0), 10.0 ** -2.5, rtol=1e-12)
