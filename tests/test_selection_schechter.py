"""Schechter luminosity-function selection family, end to end.

The load-bearing properties pinned here are (a) EXACT H0-invariance of the
consumed schechter curve -- ``Mstar_hat`` is h-scaled exactly like ``M0hat``
and ``M_faint_offset`` is a magnitude DIFFERENCE, so no h survives anywhere in
``C_sel``; (b) recovery of the h-scaled truth from a DAG-consistent truncated
sample (draw the population, then apply the same cut the likelihood normalizes
against), INCLUDING the ``alpha ~ -1.0 to -1.3`` faint-end slopes real
catalogs measure and the deep regime where the survey reaches past the
modelled faint cutoff; (c) the family firewalls -- the ``alpha > -2``
spelling domain, the ``M_faint_offset`` protocol constant and the diagnostics
that say so out loud, no K(z), single stratum, and the Q-table-vs-run family
check; and (d) that every gaussian path is bit-identical to the pre-schechter
code, including the emitted label ordering.
"""

import json

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from scipy.special import gammaincc, gammainccinv  # noqa: E402

from darksirens.core.constants import (  # noqa: E402
    SELECTION_FAMILIES as CONST_SELECTION_FAMILIES,
    SURVEY_PARAMS_FID_BY_NAME,
)
from darksirens.core.types import (  # noqa: E402
    C_MODE_SELECTION_STRUCT,
    SELECTION_FAMILY_SCHECHTER_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.inference.prior import build_parameter_space  # noqa: E402
from darksirens.redshift.completion import (  # noqa: E402
    _precompute_grids,
    completion_curves,
)
from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import (  # noqa: E402
    H0_REF,
    M_FAINT_OFFSET_DEFAULT,
    SELECTION_FAMILIES,
    SelectionFit,
    c_sel_schechter,
    fit_selection_from_mags,
    load_selection_fit_json,
    load_selection_fit_strata,
)
from darksirens.utils.cosmology import distance_modulus  # noqa: E402

MSTAR_TRUE = -20.5
ALPHA_TRUE = -0.7
M_LIM = 21.5
OFFSET = 5.0

SCHECHTER_THETA = dict(m_lim=22.0, Mstar_hat=-20.5, alpha=-0.7,
                       M_faint_offset=5.0)
GAUSSIAN_THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)


def _schechter_truncated_sample(rng, n, h0_true, Mstar_true=MSTAR_TRUE,
                                alpha_true=ALPHA_TRUE, m_lim=M_LIM,
                                offset=OFFSET, zlo=0.08, zhi=0.5):
    """DAG-consistent mock: draw the modelled population, THEN cut at m_lim.

    The population is the Schechter LF truncated at ``M <= Mstar + offset``,
    sampled through the exact inverse-CDF of the truncated Gamma in
    ``x = L/L*`` for ``alpha > -1`` and by rejection off the pure power law
    below it (``gammainccinv`` is the regularized spelling and stops at
    ``a > 0``; the LF does not).  The default ``zlo`` keeps ``m_lim - DM(z)``
    bright-ward of the faint cutoff at every redshift; ``zlo=0.004`` reaches
    past it, which is the DEEP configuration a real low-z catalog is always in.
    """
    a = alpha_true + 1.0
    x_faint = 10.0 ** (-0.4 * offset)
    N = 40 * n
    z = rng.uniform(zlo ** 1.5, zhi ** 1.5, size=N) ** (1.0 / 1.5)
    if a > 0.0:
        x = gammainccinv(a, rng.uniform(size=N) * gammaincc(a, x_faint))
    else:
        x_hi, chunks, got = 40.0, [], 0
        while got < N:
            v = rng.uniform(size=2 * N)
            xx = (x_faint ** a + v * (x_hi ** a - x_faint ** a)) ** (1.0 / a)
            xx = xx[rng.uniform(size=xx.size) < np.exp(-xx)]
            chunks.append(xx)
            got += xx.size
        x = np.concatenate(chunks)[:N]
    M = Mstar_true - 2.5 * np.log10(x)
    m = M + np.asarray(distance_modulus(jnp.asarray(z), h0_true))
    keep = m <= m_lim
    return m[keep][:n], z[keep][:n]


def _em(n_pix=12, max_gals=3, seed=5):
    rng = np.random.default_rng(seed)
    zg = np.full((n_pix, max_gals), 100.0)
    dz = np.ones((n_pix, max_gals))
    w = np.zeros((n_pix, max_gals))
    ng = np.zeros(n_pix, dtype=int)
    for p in range(n_pix):
        k = int(rng.integers(0, max_gals))
        zg[p, :k] = np.sort(rng.uniform(0.02, 0.4, k))
        dz[p, :k] = 1e-3
        w[p, :k] = 1.0
        ng[p] = k
    import healpy as hp
    return EMCatalog(
        apix=float(hp.nside2pixarea(1)), zgals=jnp.asarray(zg),
        dzgals=jnp.asarray(dz), wgals=jnp.asarray(w), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
        pixel_to_cache_idx=None)


def _survey(**kw):
    base = dict(n0=1e-4, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=0.0)
    base.update(kw)
    return SurveyParams(**base)


def _cosmo(h0=67.74):
    return CosmoParams(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


# ---------------------------------------------------------------------------
# The offline fit
# ---------------------------------------------------------------------------

def test_schechter_fit_recovers_h_scaled_truth_independent_of_true_h0():
    """theta_hat measures (M* - 5 log10 h, alpha) whatever the true H0 is."""
    for h0_true in (60.0, 80.0):
        rng = np.random.default_rng(7)
        m, z = _schechter_truncated_sample(rng, 4000, h0_true)
        fit = fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                      M_faint_offset=OFFSET)
        sd = np.sqrt(np.diag(fit.cov))
        mstar_true = MSTAR_TRUE - 5.0 * np.log10(h0_true / H0_REF)
        assert fit.family == "schechter"
        assert fit.M_faint_offset == OFFSET
        assert fit.M0hat is None and fit.sigma_M is None
        assert abs(fit.Mstar_hat - mstar_true) < 4.0 * sd[0], (
            f"H0_true={h0_true}: {fit.Mstar_hat} vs {mstar_true} (sd {sd[0]})")
        assert abs(fit.alpha - ALPHA_TRUE) < 4.0 * sd[1]
        cov = np.asarray(fit.cov)
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_schechter_fit_recovers_the_faint_end_slopes_real_catalogs_measure():
    """alpha ~ -1.0 to -1.3 -- 2MASS K, SDSS r, GLADE B -- must FIT.

    Every real galaxy LF is steeper than -1, so a Schechter channel that
    cannot reach there cannot fit a real catalog.  The curve is the ratio of
    UNREGULARIZED upper incomplete gammas, finite for any alpha because both
    arguments are positive; only the regularized spelling stops at alpha = -1.
    """
    for alpha_true in (-1.02, -1.1, -1.25):
        rng = np.random.default_rng(7)
        m, z = _schechter_truncated_sample(rng, 4000, 70.0,
                                           alpha_true=alpha_true)
        fit = fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                      M_faint_offset=OFFSET)
        sd = np.sqrt(np.diag(fit.cov))
        mstar_true = MSTAR_TRUE - 5.0 * np.log10(70.0 / H0_REF)
        assert abs(fit.alpha - alpha_true) < 4.0 * sd[1], (
            f"alpha_true={alpha_true}: {fit.alpha} +/- {sd[1]}")
        assert abs(fit.Mstar_hat - mstar_true) < 4.0 * sd[0]


def test_schechter_fit_truncation_is_doing_work():
    """Normalizing against a limit no galaxy reaches (the untruncated LF)
    is biased by many sd -- the per-galaxy truncation carries real signal."""
    rng = np.random.default_rng(7)
    m, z = _schechter_truncated_sample(rng, 4000, 70.0)
    fit = fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                  M_faint_offset=OFFSET)
    sd = np.sqrt(np.diag(fit.cov))
    # A limit far fainter than any galaxy makes x_lim -> 0 for every galaxy,
    # so the per-galaxy support collapses to the untruncated one.
    naive = fit_selection_from_mags(m, z, 99.0, family="schechter",
                                    M_faint_offset=OFFSET)
    assert abs(naive.Mstar_hat - fit.Mstar_hat) > 5.0 * sd[0]
    assert abs(naive.alpha - fit.alpha) > 5.0 * sd[1]


def test_schechter_fit_survives_a_survey_that_reaches_the_faint_cutoff():
    """The DEEP regime -- ``m_lim - DM(z)`` fainter than ``Mstar + offset`` --
    is the one every low-z catalog is in, and it must FIT.

    ``M_faint = Mstar_hat + M_faint_offset`` is a parameter-dependent support
    edge whose MLE would be an order statistic, so it is kept out of the fit
    likelihood entirely; the only support edges are the per-galaxy detection
    limit and the optional parameter-free ``m_faint_cut``.
    """
    rng = np.random.default_rng(7)
    m, z = _schechter_truncated_sample(rng, 4000, 70.0, alpha_true=-1.25,
                                       zlo=0.004)
    mstar_true = MSTAR_TRUE - 5.0 * np.log10(70.0 / H0_REF)
    fit = fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                  M_faint_offset=OFFSET,
                                  m_faint_cut=mstar_true + OFFSET)
    sd = np.sqrt(np.diag(fit.cov))
    assert fit.meta["frac_complete_at_m_faint"] > 0.05, (
        "the fixture no longer exercises the deep regime")
    assert abs(fit.alpha - (-1.25)) < 4.0 * sd[1]
    assert abs(fit.Mstar_hat - mstar_true) < 4.0 * sd[0]
    assert fit.meta["m_faint_cut"] == mstar_true + OFFSET

    # A cutoff the sample reaches far past no longer refuses the fit: it is a
    # denominator convention, so it is reported, not enforced -- but a deep
    # sample with no declared cut is a modelling gap and says so.
    with pytest.warns(RuntimeWarning, match="m_faint_cut"):
        tight = fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                        M_faint_offset=1.5)
    assert tight.meta["n_gal_faintward_of_m_faint"] > 0
    assert tight.meta["m_faint_cut"] is None


def test_m_faint_offset_leaves_the_fit_alone_and_says_so():
    """The protocol constant cannot be fitted from magnitudes, so the fit is
    bit-identical across offsets -- and stamps the budget it silently buys."""
    rng = np.random.default_rng(7)
    m, z = _schechter_truncated_sample(rng, 2000, 70.0)
    fits = [fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                    M_faint_offset=off)
            for off in (4.0, 5.0, 6.0)]
    for f_ in fits[1:]:
        assert f_.Mstar_hat == fits[0].Mstar_hat
        assert f_.alpha == fits[0].alpha
        assert f_.meta["nll"] == fits[0].meta["nll"]
        assert f_.meta["m_faint_offset_constrained"] is False

    # ... while the missing-galaxy budget it multiplies moves a lot.
    budget = [f_.meta["missing_budget_vs_offset"][f"{off:.2f}"]
              for f_, off in zip(fits, (4.0, 5.0, 6.0))]
    assert budget[0] < budget[1] < budget[2]
    assert budget[2] - budget[0] > 0.05, budget


def test_schechter_fit_refuses_k_corr_template():
    rng = np.random.default_rng(7)
    m, z = _schechter_truncated_sample(rng, 200, 70.0)
    with pytest.raises(NotImplementedError, match="c_sel_schechter"):
        fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                k_corr_coeffs=(0.4,))
    with pytest.raises(ValueError, match="M_faint_offset"):
        fit_selection_from_mags(m, z, M_LIM, family="schechter",
                                M_faint_offset=-1.0)
    with pytest.raises(NotImplementedError, match="unknown selection family"):
        fit_selection_from_mags(m, z, M_LIM, family="lognormal")
    with pytest.raises(NotImplementedError, match="m_faint_cut"):
        fit_selection_from_mags(m, z, M_LIM, m_faint_cut=-18.0)


def test_upper_gamma_ratio_matches_quadrature_below_alpha_minus_one():
    """The curve is the UNREGULARIZED ratio, so alpha <= -1 is ordinary.

    Gamma(a, x_lim)/Gamma(a, x_faint) is finite for any real a because both
    arguments are strictly positive; the regularized spelling's Gamma(a)
    cancels exactly.  Quadrature is the referee.
    """
    from scipy.integrate import quad

    from darksirens.redshift.selection import _upper_gamma_scaled

    x_faint = 0.01
    for alpha in (-1.02, -1.1, -1.25, -1.5, -1.9, -0.7):
        a = alpha + 1.0
        den = quad(lambda t: t ** (a - 1.0) * np.exp(-t), x_faint, 80.0,
                   limit=400)[0]
        for x_lim in (0.02, 0.1, 0.3, 1.0):
            num = quad(lambda t: t ** (a - 1.0) * np.exp(-t), x_lim, 80.0,
                       limit=400)[0]
            got = (float(_upper_gamma_scaled(a, x_lim))
                   / float(_upper_gamma_scaled(a, x_faint)))
            assert abs(got - num / den) < 1e-8 * max(num / den, 1e-3), (
                f"alpha={alpha} x_lim={x_lim}: {got} vs {num / den}")

    # alpha = -1 exactly is a REMOVABLE singularity, not a wall: the nudge
    # keeps the curve continuous and finite through it.
    at_wall = float(_upper_gamma_scaled(0.0, 0.1)) / float(
        _upper_gamma_scaled(0.0, x_faint))
    just_off = float(_upper_gamma_scaled(-1e-3, 0.1)) / float(
        _upper_gamma_scaled(-1e-3, x_faint))
    assert np.isfinite(at_wall) and abs(at_wall - just_off) < 1e-3


def test_alpha_domain_is_refused_everywhere(tmp_path):
    """alpha <= -2 is refused at JSON validation, at construction, and by the
    prior bounds -- that is where the recurrence's one lift runs out."""
    from darksirens.redshift.selection import _validate_stratum

    bad = {"family": "schechter", "m_lim": 21.5, "Mstar_hat": -20.0,
           "alpha": -2.5, "M_faint_offset": 5.0, "cov": np.eye(2).tolist()}
    with pytest.raises(ValueError, match="alpha"):
        _validate_stratum(tmp_path / "f.json", bad)
    with pytest.raises(ValueError, match="alpha"):
        SelectionFit(family="schechter", m_lim=21.5, Mstar_hat=-20.0,
                     alpha=-2.5, M_faint_offset=5.0, cov=np.eye(2))
    with pytest.raises(ValueError, match="outside the sampled bounds"):
        build_parameter_space(
            "powerlaw+peak", True, True, False, universe_model="dark_sirens",
            use_lss=False, c_mode="selection", selection_family="schechter",
            selection_prior={"alpha": (-2.5, 0.01)})

    # A real-catalog slope loads, constructs and priors cleanly.
    ok = dict(bad, alpha=-1.21)
    assert _validate_stratum(tmp_path / "f.json", ok)["alpha"] == -1.21
    build_parameter_space(
        "powerlaw+peak", True, True, False, universe_model="dark_sirens",
        use_lss=False, c_mode="selection", selection_family="schechter",
        selection_prior={"alpha": (-1.21, 0.03)})


def _schechter_space(**kw):
    return build_parameter_space(
        "powerlaw+peak", True, True, False, universe_model="dark_sirens",
        use_lss=False, c_mode="selection", selection_family="schechter", **kw)


def test_alpha_domain_survives_prior_overrides_and_fixed_values():
    """The alpha > -2 wall is not a taste bound: --prior_overrides and
    --fixed_parameter_values both route around the registry floor, and past
    the wall the recurrence has no more lifts to give."""
    _schechter_space(prior_overrides={"alpha": [-1.95, -0.2]})     # inside
    _schechter_space(fixed_parameter_values={"alpha": -1.21})      # GLADE B

    with pytest.raises(ValueError, match="not widenable"):
        _schechter_space(prior_overrides={"alpha": [-2.5, -0.2]})
    with pytest.raises(ValueError, match="not widenable"):
        _schechter_space(fixed_parameter_values={"alpha": -2.2})

    # The wall the bounds protect announces itself rather than reading as a
    # perfectly complete survey; INSIDE it, every real faint-end slope works.
    cosmo = _cosmo()
    bad = np.asarray(c_sel_schechter(zgrid, 21.5, -20.31, -2.2, 5.0, cosmo.H0,
                                     cosmo.Om0, cosmo.w0, cosmo.wa))
    assert np.all(np.isnan(bad))
    for alpha in (-1.02, -1.05, -1.21, -0.68):
        good = np.asarray(c_sel_schechter(zgrid, 21.5, -20.31, alpha, 5.0,
                                          cosmo.H0, cosmo.Om0, cosmo.w0,
                                          cosmo.wa))
        assert np.all(np.isfinite(good)), alpha
        assert float(good.min()) < 1.0 and float(good.max()) <= 1.0


def test_fixed_faint_offset_must_be_positive():
    """M_faint_offset is pinned, so it has no bounds to validate against; the
    faint cutoff must still lie faint-ward of M*."""
    _schechter_space(fixed_parameter_values={"M_faint_offset": 4.0})
    with pytest.raises(ValueError, match="M_faint_offset must be positive"):
        _schechter_space(fixed_parameter_values={"M_faint_offset": -1.0})


def test_selection_fit_json_roundtrip_schechter(tmp_path):
    fit = SelectionFit(family="schechter", m_lim=21.5, Mstar_hat=-20.31,
                       alpha=-0.68, M_faint_offset=5.0,
                       cov=np.eye(2) * 1e-4, n_gal=4000)
    payload = {"format_version": "darksirens-selection-fit-1.0",
               "strata": [fit.to_jsonable()]}
    p = tmp_path / "fit.json"
    p.write_text(json.dumps(payload))
    s = load_selection_fit_json(p)
    assert s["family"] == "schechter"
    assert s["Mstar_hat"] == -20.31 and s["alpha"] == -0.68
    assert s["M_faint_offset"] == 5.0
    assert s["k_corr_coeffs"] == ()

    missing = dict(payload["strata"][0])
    del missing["alpha"]
    p.write_text(json.dumps({**payload, "strata": [missing]}))
    with pytest.raises(ValueError, match="'alpha'"):
        load_selection_fit_json(p)

    multi = {"format_version": "darksirens-selection-fit-1.1",
             "strata": [fit.to_jsonable(),
                        SelectionFit(family="schechter", m_lim=20.5,
                                     Mstar_hat=-20.4, alpha=-0.7,
                                     M_faint_offset=5.0, cov=np.eye(2) * 1e-4,
                                     n_gal=900, stratum="1").to_jsonable()]}
    p.write_text(json.dumps(multi))
    with pytest.raises(NotImplementedError, match="gaussian-only"):
        load_selection_fit_strata(p)


def test_gaussian_jsonable_key_order_is_unchanged():
    """Payload compatibility: the gaussian key order is frozen."""
    fit = SelectionFit(family="gaussian", m_lim=21.0, M0hat=-20.9,
                       sigma_M=0.9, cov=np.eye(2) * 1e-4, n_gal=1000)
    assert tuple(fit.to_jsonable()) == (
        "family", "m_lim", "M0hat", "sigma_M", "cov", "n_gal", "stratum",
        "k_corr_coeffs", "meta")


# ---------------------------------------------------------------------------
# The consumed curve
# ---------------------------------------------------------------------------

def test_precompute_grids_carries_c_sel_schechter_exactly():
    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT,
                 selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                 **SCHECHTER_THETA)
    grids = _precompute_grids(_cosmo(), sv, _em())
    ref = c_sel_schechter(zgrid, SCHECHTER_THETA["m_lim"],
                          SCHECHTER_THETA["Mstar_hat"],
                          SCHECHTER_THETA["alpha"],
                          SCHECHTER_THETA["M_faint_offset"],
                          67.74, 0.3075, -1.0, 0.0)
    np.testing.assert_allclose(np.asarray(grids.C_bar_raw), np.asarray(ref),
                               rtol=0, atol=1e-14)


def test_schechter_c_bar_exactly_h0_invariant_through_survey_path():
    """THE firewall, end to end through _precompute_grids."""
    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT,
                 selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                 **SCHECHTER_THETA)
    em = _em()
    a = _precompute_grids(_cosmo(50.0), sv, em).C_bar_raw
    b = _precompute_grids(_cosmo(95.0), sv, em).C_bar_raw
    np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                               rtol=0, atol=1e-12)


def test_family_via_eager_string_and_sentinel_agree():
    em = _em()
    a = _precompute_grids(
        _cosmo(), _survey(c_mode=C_MODE_SELECTION_STRUCT,
                          selection_family="schechter", **SCHECHTER_THETA), em
    ).C_bar_raw
    b = _precompute_grids(
        _cosmo(), _survey(c_mode=C_MODE_SELECTION_STRUCT,
                          selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                          **SCHECHTER_THETA), em
    ).C_bar_raw
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_traced_family_leaf_hard_errors():
    em = _em()

    def f(sv):
        return completion_curves(_cosmo(), sv, em).dN_miss

    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT,
                 selection_family=jnp.asarray(1.0), **SCHECHTER_THETA)
    with pytest.raises((ValueError, jax.errors.JaxRuntimeError),
                       match="selection_family"):
        jax.jit(f)(sv)


def test_schechter_refuses_kcorr_and_strata_in_likelihood():
    em = _em()
    base = dict(c_mode=C_MODE_SELECTION_STRUCT,
                selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                **SCHECHTER_THETA)
    with pytest.raises(NotImplementedError, match="K\\(z\\)"):
        _precompute_grids(_cosmo(), _survey(k_corr_coeffs=(0.4,), **base), em)
    with pytest.raises(NotImplementedError, match="gaussian-only"):
        _precompute_grids(
            _cosmo(), _survey(selection_strata=((21.5, 0.0, 1.0),), **base), em)


def test_schechter_theta_moves_the_missing_budget():
    """A brighter M* means a more complete survey: strictly less missing."""
    em = _em()
    base = dict(c_mode=C_MODE_SELECTION_STRUCT,
                selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                **SCHECHTER_THETA)
    n0 = float(np.sum(np.asarray(completion_curves(
        _cosmo(), _survey(**base), em).N_miss)))
    brighter = dict(base, Mstar_hat=SCHECHTER_THETA["Mstar_hat"] - 1.5)
    n1 = float(np.sum(np.asarray(completion_curves(
        _cosmo(), _survey(**brighter), em).N_miss)))
    assert n1 < n0


def test_gaussian_paths_are_bit_identical():
    """Nothing about the gaussian channel moves: curves, grids, or labels."""
    em = _em()
    none_family = _precompute_grids(
        _cosmo(), _survey(c_mode=C_MODE_SELECTION_STRUCT, **GAUSSIAN_THETA),
        em)
    named = _precompute_grids(
        _cosmo(), _survey(c_mode=C_MODE_SELECTION_STRUCT,
                          selection_family="gaussian", **GAUSSIAN_THETA), em)
    assert np.array_equal(np.asarray(none_family.C_bar_raw),
                          np.asarray(named.C_bar_raw))
    for a, b in zip(none_family, named):
        if a is None:
            assert b is None
            continue
        assert np.array_equal(np.asarray(a), np.asarray(b))

    kw = dict(universe_model="dark_sirens", use_lss=False)
    on = build_parameter_space("powerlaw+peak", True, True, False,
                               c_mode="selection", **kw)[0]
    assert [lbl for lbl in on if lbl in SURVEY_PARAMS_FID_BY_NAME] == [
        "log10n0", "delta", "sigma_kde", "M0hat", "sigma_M"]
    off = build_parameter_space("powerlaw+peak", True, True, False,
                                c_mode="per_pixel", **kw)[0]
    assert not ({"M0hat", "sigma_M", "Mstar_hat", "alpha"} & set(off))


def test_schechter_labels_sample_iff_family_schechter():
    kw = dict(universe_model="dark_sirens", use_lss=False, c_mode="selection")
    gauss = build_parameter_space("powerlaw+peak", True, True, False, **kw)[0]
    schech = build_parameter_space("powerlaw+peak", True, True, False,
                                   selection_family="schechter", **kw)[0]
    assert {"Mstar_hat", "alpha"} <= set(schech)
    assert not ({"M0hat", "sigma_M"} & set(schech))
    assert not ({"m_lim", "M_faint_offset"} & set(schech))
    assert not ({"m_lim", "M_faint_offset"} & set(gauss))
    # Index stability: every label BEFORE the survey block, and the block
    # length itself, are untouched by the family swap.
    assert len(gauss) == len(schech)
    i = gauss.index("log10n0")
    assert gauss[:i] == schech[:i]
    assert gauss[i:i + 3] == schech[i:i + 3]        # log10n0, delta, sigma_kde
    assert gauss[i + 5:] == schech[i + 5:]          # everything after the block


def test_fixed_gaussian_param_under_schechter_family_is_fatal():
    kw = dict(universe_model="dark_sirens", use_lss=False, c_mode="selection")
    with pytest.raises(ValueError, match="'schechter' selection family"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              selection_family="schechter",
                              fixed_parameter_values={"sigma_M": 0.9}, **kw)
    with pytest.raises(ValueError, match="'gaussian' selection family"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              fixed_parameter_values={"alpha": -0.7}, **kw)


def test_selection_prior_flips_kinds_for_schechter():
    kw = dict(universe_model="dark_sirens", use_lss=False, c_mode="selection",
              selection_family="schechter")
    res = build_parameter_space(
        "powerlaw+peak", True, True, False,
        selection_prior={"Mstar_hat": (-20.31, 0.04), "alpha": (-0.68, 0.03)},
        **kw)
    kinds = dict(zip(res[0], res[11]))
    assert kinds["Mstar_hat"] == ("normal", -20.31, 0.04)
    assert kinds["alpha"] == ("normal", -0.68, 0.03)
    with pytest.raises(ValueError, match="'gaussian' family"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              selection_prior={"M0hat": (-20.19, 0.02)}, **kw)


# ---------------------------------------------------------------------------
# Provenance: the builder stamp and the inference-side firewall
# ---------------------------------------------------------------------------

def test_qtable_family_mismatch_is_fatal():
    from darksirens.cli import inference as cli

    gauss_fid = {"path": "qt.h5", "selection_family": "gaussian",
                 "selection_m_lim": 21.0, "selection_M0hat": -20.9,
                 "selection_sigma_M": 0.9}
    schech_fid = {"path": "qt.h5", "selection_family": "schechter",
                  "selection_m_lim": 21.5, "selection_Mstar_hat": -20.31,
                  "selection_alpha": -0.68, "selection_M_faint_offset": 5.0}

    class Opts:
        selection_family = "schechter"
        selection_fit_theta = {"m_lim": 21.5, "Mstar_hat": -20.31,
                               "alpha": -0.68, "M_faint_offset": 5.0}
        selection_fit_kcorr = ()

    with pytest.raises(SystemExit):                 # gaussian table, schechter run
        cli._check_selection_qtable_theta([gauss_fid], Opts())
    cli._check_selection_qtable_theta([schech_fid], Opts())     # exact match

    with pytest.raises(SystemExit):                 # theta mismatch
        cli._check_selection_qtable_theta(
            [{**schech_fid, "selection_alpha": -0.5}], Opts())

    stripped = {k: v for k, v in schech_fid.items()
                if k != "selection_M_faint_offset"}
    with pytest.raises(SystemExit):                 # missing protocol stamp
        cli._check_selection_qtable_theta([stripped], Opts())

    # A pre-stamp gaussian table still verifies against a gaussian run.
    class GOpts:
        selection_family = "gaussian"
        selection_fit_theta = {"m_lim": 21.0, "M0hat": -20.9, "sigma_M": 0.9}
        selection_fit_kcorr = ()

    cli._check_selection_qtable_theta(
        [{k: v for k, v in gauss_fid.items() if k != "selection_family"}],
        GOpts())


def test_family_mismatch_beats_the_no_fit_ablation_escape_hatch():
    """Without --selection_fit the run's family can only be gaussian, so a
    schechter-stamped table has no consistent configuration: the wide-open
    ablation widens theta WITHIN a family, it does not change the LF."""
    from darksirens.cli import inference as cli

    class NoFit:                       # the sanctioned wide-open ablation
        selection_family = "gaussian"
        selection_fit_theta = None
        selection_fit_kcorr = ()

    schech_fid = {"path": "qt.h5", "selection_family": "schechter",
                  "selection_m_lim": 21.5, "selection_Mstar_hat": -20.31,
                  "selection_alpha": -0.68, "selection_M_faint_offset": 5.0}
    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta([schech_fid], NoFit())

    # Same table with no family stamp at all: the family is INFERRED from the
    # theta keys, so the pre-tag tables are walled too.
    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta(
            [{k: v for k, v in schech_fid.items() if k != "selection_family"}],
            NoFit())

    # A gaussian table with no fit stays a warning: same family, wider theta.
    gauss_fid = {"path": "qt.h5", "selection_family": "gaussian",
                 "selection_m_lim": 21.0, "selection_M0hat": -20.9,
                 "selection_sigma_M": 0.9}
    cli._check_selection_qtable_theta([gauss_fid], NoFit())


def test_selection_fit_pins_protocol_constant_over_explicit_fixed_value():
    """m_lim is a per-run datum the fit defaults; M_faint_offset is a protocol
    constant an explicit --fixed_parameter_values may only restate."""
    from darksirens.cli import inference as cli

    sel = {"family": "schechter", "m_lim": 21.5, "Mstar_hat": -20.31,
           "alpha": -0.68, "M_faint_offset": 5.0}

    fixed = {}
    assert cli._resolve_selection_fit_pins(sel, "schechter", fixed) == {
        "m_lim": 21.5, "Mstar_hat": -20.31, "alpha": -0.68,
        "M_faint_offset": 5.0}
    assert fixed == {"m_lim": 21.5, "M_faint_offset": 5.0}

    # A restated offset is fine; a different one changes the LF the likelihood
    # integrates while every provenance stamp still agrees.
    cli._resolve_selection_fit_pins(sel, "schechter", {"M_faint_offset": 5.0})
    with pytest.raises(SystemExit):
        cli._resolve_selection_fit_pins(sel, "schechter",
                                        {"M_faint_offset": 4.0})

    # An overridden m_lim IS allowed, and the returned theta is the EFFECTIVE
    # one, so the Q-table check sees the value the likelihood will use.
    theta = cli._resolve_selection_fit_pins(sel, "schechter", {"m_lim": 21.0})
    assert theta["m_lim"] == 21.0

    class Opts:
        selection_family = "schechter"
        selection_fit_theta = theta
        selection_fit_kcorr = ()

    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta(
            [{"path": "qt.h5", "selection_family": "schechter",
              "selection_m_lim": 21.5, "selection_Mstar_hat": -20.31,
              "selection_alpha": -0.68, "selection_M_faint_offset": 5.0}],
            Opts())


def test_builder_stamp_and_cbar_dispatch_by_family():
    from darksirens.cli.build_lognormal_completion import (
        _selection_cbar_fine, _selection_theta_stamp,
    )

    gauss = {"family": "gaussian", "m_lim": 21.0, "M0hat": -20.9,
             "sigma_M": 0.9}
    assert _selection_theta_stamp(gauss) == {
        "selection_family": "gaussian", "selection_m_lim": 21.0,
        "selection_M0hat": -20.9, "selection_sigma_M": 0.9}

    schech = {"family": "schechter", "m_lim": 21.5, "Mstar_hat": -20.31,
              "alpha": -0.68, "M_faint_offset": 5.0}
    assert _selection_theta_stamp(schech) == {
        "selection_family": "schechter", "selection_m_lim": 21.5,
        "selection_Mstar_hat": -20.31, "selection_alpha": -0.68,
        "selection_M_faint_offset": 5.0}

    cosmo = _cosmo()
    ref = np.asarray(c_sel_schechter(
        zgrid, 21.5, -20.31, -0.68, 5.0, cosmo.H0, cosmo.Om0, cosmo.w0,
        cosmo.wa), dtype=float)
    np.testing.assert_allclose(_selection_cbar_fine(schech, cosmo), ref,
                               rtol=0, atol=1e-14)


def test_decoder_threads_family_and_pins_the_offset():
    from darksirens.gw.populations import pop_model_prior_parser
    from darksirens.inference.parameters import build_parameter_decoder

    class Opts:
        pop_model = "powerlaw+peak"
        fix_population = True
        fix_cosmology = True
        fix_survey = False
        fix_de = False
        universe_model = "dark_sirens"
        use_LSS = False
        c_mode = "selection"
        selection_family = "schechter"
        selection_prior = None

    fixed = {"m_lim": 21.5, "M_faint_offset": 4.0}
    pop_fid = (0.0,) * len(pop_model_prior_parser("powerlaw+peak")[2])
    dec = build_parameter_decoder(Opts(), pop_fid,
                                  fixed_parameter_values=fixed)
    assert dec.selection_family == "schechter"
    assert {"Mstar_hat", "alpha"} <= set(dec.sampled_labels)
    coord = jnp.asarray([
        {"Mstar_hat": -20.4, "alpha": -0.6}.get(lbl, 0.01)
        for lbl in dec.sampled_labels])
    _, survey, *_ = dec.decode(coord)
    assert isinstance(survey.selection_family,
                      type(SELECTION_FAMILY_SCHECHTER_STRUCT))
    assert float(survey.m_lim) == 21.5
    assert float(survey.M_faint_offset) == 4.0
    assert float(survey.Mstar_hat) == -20.4
    assert float(survey.alpha) == -0.6


def test_fiducial_and_module_default_offsets_agree():
    assert SURVEY_PARAMS_FID_BY_NAME["M_faint_offset"] == M_FAINT_OFFSET_DEFAULT
    assert CONST_SELECTION_FAMILIES == SELECTION_FAMILIES
