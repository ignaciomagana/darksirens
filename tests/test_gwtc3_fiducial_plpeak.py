"""The GWTC-3 fiducial POWER LAW + PEAK preset (arXiv:2111.03634).

The registry already carried a ``powerlaw+peak`` composition, but it is a
CURATED test model: 90% of its fiducial population sits in the 35 Msun
Gaussian (GWTC-3 measures lambda_peak = 0.038) and that Gaussian is untapered
on primary mass, where Eq. B4 of the paper multiplies the WHOLE mixture by the
low-mass taper.  The second difference is structural -- no choice of weights
turns one density into the other -- so reading a `powerlaw+peak` result as
"the LVK model" is a category error rather than a small bias.

``gwtc3_fiducial_plpeak`` is the published model instead: Table VI's priors
verbatim and Eqs. B4-B7 term for term.  These tests pin what makes it that
model rather than a lookalike -- the prior box, the tapered peak, the single
minimum mass shared by the primary-mass and mass-ratio tapers -- and pin the
provenance of the fiducial vector, including the two entries that are NOT
measurements.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

#: numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and removed the old
#: spelling; this file is checked on both majors (tests/fast_subset.txt
#: rejects files that pin one of them).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from darksirens.gw.populations.registry import (
    get_fixed_population_params,
    get_model,
    pop_model_prior_parser,
)


MODEL = "gwtc3_fiducial_plpeak"


# ---------------------------------------------------------------------------
# Table VI: the model definition and its prior box
# ---------------------------------------------------------------------------

#: arXiv:2111.03634 Table VI ("Summary of Power Law + Peak model parameters"),
#: every row, in the paper's own order.  beta_q is the pairing slope and is
#: therefore sampled after the mass block here; the rest keep Table VI's order.
TABLE_VI_PRIORS = {
    "alpha": (-4.0, 12.0),
    "beta_q": (-2.0, 7.0),
    "m_min": (2.0, 10.0),
    "m_max": (30.0, 100.0),
    "lambda_peak": (0.0, 1.0),
    "mu_m": (20.0, 50.0),
    "sigma_m": (1.0, 10.0),
    "delta_m": (0.0, 10.0),
}


def test_table_vi_priors_and_parameter_order():
    lower, upper, labels, _, latex = pop_model_prior_parser(MODEL)

    expected = [
        (r"$\alpha$", *TABLE_VI_PRIORS["alpha"]),
        (r"$m_{\min}$", *TABLE_VI_PRIORS["m_min"]),
        (r"$m_{\max}$", *TABLE_VI_PRIORS["m_max"]),
        (r"$\lambda_{\rm peak}$", *TABLE_VI_PRIORS["lambda_peak"]),
        (r"$\mu_m$", *TABLE_VI_PRIORS["mu_m"]),
        (r"$\sigma_m$", *TABLE_VI_PRIORS["sigma_m"]),
        (r"$\delta_m$", *TABLE_VI_PRIORS["delta_m"]),
        (r"$\beta_q$", *TABLE_VI_PRIORS["beta_q"]),
    ]
    assert list(zip(labels[:8], lower[:8], upper[:8])) == expected
    assert latex == r"\text{GWTC-3 Fiducial PL+G}"

    # Spin and the rate index are NOT in Table VI (the paper's Default spin
    # model is a Beta magnitude plus a cos-tilt mixture, which this chi_eff
    # likelihood cannot represent).  They are the repo's own block, and the
    # test says so rather than letting the count silently drift.
    assert labels[8:] == [r"$\mu_\chi$", r"$\sigma_\chi$", r"$\gamma$"]
    assert len(labels) == 11


@pytest.mark.parametrize(
    "alias", ["gwtc3_powerlaw+peak", "gwtc3_fiducial_powerlaw+peak"]
)
def test_alias_spellings_resolve_with_a_deprecation_warning(alias):
    """The grammar-shaped spelling must not fall through to the CURATED
    ``powerlaw+peak`` composition, which is a different density."""
    with pytest.warns(DeprecationWarning, match=MODEL):
        _, _, labels, _, latex = pop_model_prior_parser(alias)
    assert latex == r"\text{GWTC-3 Fiducial PL+G}"
    assert labels[0] == r"$\alpha$"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.testing.assert_array_equal(
            np.asarray(get_fixed_population_params(alias)),
            np.asarray(get_fixed_population_params(MODEL)),
        )


# ---------------------------------------------------------------------------
# The fiducial vector: what is measured, and what is not
# ---------------------------------------------------------------------------

#: Values arXiv:2111.03634 actually reports, each with the sentence it comes
#: from (Sec. V B for the mass block, Fig. 13 for the rate index).
PAPER_REPORTED_MEDIANS = {
    r"$\alpha$": 3.5,                 # alpha = 3.5 +0.6 -0.56
    r"$m_{\min}$": 5.0,               # mmin = 5.0 +0.86 -1.7 Msun
    r"$\lambda_{\rm peak}$": 0.038,   # lambda = 0.038 +0.058 -0.026
    r"$\mu_m$": 34.0,                 # Gaussian peak at 34 +2.6 -4.0 Msun
    r"$\delta_m$": 4.9,               # delta_m = 4.9 +3.4 -3.2 Msun
    r"$\beta_q$": 1.1,                # beta_q = 1.1 +1.7 -1.3
    r"$\gamma$": 2.9,                 # kappa = 2.9 +1.7 -1.8, R(z) ~ (1+z)^kappa
}

#: The paper reports NO posterior for these two: it quotes the derived
#: m_99% = 44 +9.2 -5.1 Msun instead of m_max, and says only that the Gaussian's
#: mean and width are "consistent with previous inferences".  They are prior
#: MIDPOINTS -- placeholders, not GWTC-3 numbers -- and this test exists so
#: that stays visible rather than being read off the vector as a measurement.
UNREPORTED_PLACEHOLDERS = {
    r"$m_{\max}$": 65.0,
    r"$\sigma_m$": 5.5,
}


def test_fiducial_uses_the_paper_medians_where_the_paper_reports_one():
    _, _, labels, _, _ = pop_model_prior_parser(MODEL)
    theta = np.asarray(get_fixed_population_params(MODEL))
    by_label = dict(zip(labels, theta))

    for label, expected in PAPER_REPORTED_MEDIANS.items():
        assert by_label[label] == pytest.approx(expected), label

    # gamma is the paper's kappa with NO offset: this model's redshift term is
    # (1+z)^(gamma-1), i.e. R(z) ~ (1+z)^gamma, matching R(z) ~ (1+z)^kappa.
    assert by_label[r"$\gamma$"] == pytest.approx(2.9)


def test_unreported_parameters_are_flagged_prior_midpoints():
    lower, upper, labels, _, _ = pop_model_prior_parser(MODEL)
    theta = np.asarray(get_fixed_population_params(MODEL))

    for label, expected in UNREPORTED_PLACEHOLDERS.items():
        i = labels.index(label)
        midpoint = 0.5 * (float(lower[i]) + float(upper[i]))
        assert theta[i] == pytest.approx(expected)
        assert theta[i] == pytest.approx(midpoint), (
            f"{label} is not a GWTC-3 measurement; it must stay the neutral "
            "midpoint of its own Table VI prior so nobody quotes it as one"
        )


def test_every_fiducial_lies_strictly_inside_its_table_vi_prior():
    lower, upper, labels, _, _ = pop_model_prior_parser(MODEL)
    theta = np.asarray(get_fixed_population_params(MODEL))
    for label, lo, hi, value in zip(labels, lower, upper, theta):
        assert lo < float(value) < hi, f"{label} = {value} not inside ({lo}, {hi})"


# ---------------------------------------------------------------------------
# Eq. B4: the taper multiplies the WHOLE mixture
# ---------------------------------------------------------------------------

def _mass_component():
    return get_model(MODEL).mass_component


def _mass_theta(**over):
    base = dict(alpha=3.5, m_min=5.0, m_max=65.0, lambda_peak=0.038,
                mu_m=34.0, sigma_m=5.5, delta_m=4.9)
    base.update(over)
    return jnp.asarray([base["alpha"], base["m_min"], base["m_max"],
                        base["lambda_peak"], base["mu_m"], base["sigma_m"],
                        base["delta_m"]])


def test_the_gaussian_peak_is_tapered_unlike_the_curated_composition():
    """Eq. B4's S(m1 | m_min, delta_m) multiplies the bracket, peak included.

    The curated ``powerlaw+peak`` mixture instead applies each component's own
    smoothing, leaving its Gaussian with support all the way down to the
    global normalisation floor.  Contrasted here directly, because this is the
    difference that no weight setting can absorb.
    """
    from darksirens.gw.populations.parametric import Gaussian
    from darksirens.gw.populations.base import ParamSpec

    mass = _mass_component()
    # A pure-peak draw (lambda_peak = 1): every bit of density is Gaussian.
    theta = _mass_theta(lambda_peak=1.0, m_min=5.0, delta_m=4.9)
    below = np.asarray(mass._eval_unnorm(jnp.asarray([1.0, 3.0, 4.999]), theta))
    assert np.all(below == 0.0), "the peak is not tapered at the low-mass edge"

    # Inside the taper window the density rises smoothly from 0 to the plateau.
    window = np.asarray(mass._eval_unnorm(
        jnp.asarray([5.5, 7.0, 9.0, 9.9]), theta))
    assert np.all(np.diff(window) > 0.0)

    # The curated composition's Gaussian, for contrast: untapered, so it keeps
    # finite density far below any m_min.
    curated_peak = Gaussian(ParamSpec(r"$\mu$", 20.0, 50.0),
                            ParamSpec(r"$\sigma$", 1.0, 10.0))
    assert float(curated_peak._eval_unnorm(3.0, jnp.asarray([35.0, 5.0]))) > 0.0


def test_taper_is_the_paper_s_S_function_term_for_term():
    """Eq. B5/B6: S = 0 below m_min, 1 at and above m_min + delta_m."""
    from darksirens.gw.populations.utils import sfilter_low

    mass = _mass_component()
    theta = _mass_theta(m_min=5.0, delta_m=4.9)
    m = jnp.asarray([2.0, 4.9, 5.0, 6.0, 8.0, 9.9, 12.0, 30.0])
    got = np.asarray(mass._eval_unnorm(m, theta))
    pretaper = np.asarray(mass._mixture_pretaper(m, theta))
    S = np.asarray(sfilter_low(m, 5.0, 4.9))
    np.testing.assert_allclose(got, pretaper * S, rtol=0, atol=0)
    assert S[0] == 0.0 and S[1] == 0.0        # m <= m_min
    assert S[-1] == 1.0 and S[-2] == 1.0      # m >= m_min + delta_m


def test_analytic_normaliser_matches_dense_quadrature():
    """``_norm`` is closed-form apart from the taper-window deficit, and the
    Gaussian's mass BELOW m_min (which the taper removes entirely, since the
    peak is not truncated) is part of that deficit.  Dropping it would
    over-normalise every draw whose peak sits near the low-mass edge, so the
    corner cases below include one."""
    mass = _mass_component()
    grid = np.linspace(0.5, 400.0, 2_000_001)
    cases = [
        _mass_theta(),                                     # the fiducial
        _mass_theta(lambda_peak=1.0, mu_m=20.0, sigma_m=10.0, m_min=10.0),
        _mass_theta(alpha=-4.0, m_min=2.0, m_max=30.0, delta_m=0.0,
                    lambda_peak=0.9, mu_m=20.0, sigma_m=1.0),
        _mass_theta(alpha=12.0, m_min=10.0, m_max=100.0, delta_m=10.0,
                    lambda_peak=0.5, mu_m=50.0, sigma_m=10.0),
    ]
    for theta in cases:
        analytic = float(mass._norm(theta))
        quad = _trapezoid(np.asarray(mass._eval_unnorm(jnp.asarray(grid), theta)),
                        grid)
        assert analytic == pytest.approx(quad, rel=2.0e-4), np.asarray(theta)


def test_the_model_density_integrates_to_one():
    mass = _mass_component()
    grid = np.linspace(0.5, 400.0, 2_000_001)
    theta = _mass_theta()
    dens = np.asarray(mass(jnp.asarray(grid), theta))
    assert _trapezoid(dens, grid) == pytest.approx(1.0, rel=2.0e-4)


# ---------------------------------------------------------------------------
# Eq. B7 and the single minimum mass (why no cube map applies here)
# ---------------------------------------------------------------------------

def test_one_minimum_mass_shared_by_both_tapers():
    """"A single minimum mass is imposed upon all BH" (Sec. V B): Eq. B7's
    mass-ratio taper uses the SAME m_min and delta_m as Eq. B4.

    This is the reason the model declares no joint prior constraint.  The
    GWTC-5 fiducial model has two low-mass edges whose Table 5 prior is
    conditional, which is what the ``conditional_upper`` cube map exists for
    (F-115); inventing a second edge here so a cube map would have something
    to act on would be a different model from the published one.
    """
    lower, upper, labels, _, _ = pop_model_prior_parser(MODEL)
    low_mass_labels = [lab for lab in labels if "min" in lab or "low" in lab]
    assert low_mass_labels == [r"$m_{\min}$"]

    model = get_model(MODEL)
    assert model.constraint_groups == ()

    from darksirens.inference.prior import resolve_joint_prior_constraints

    assert resolve_joint_prior_constraints(MODEL, labels, lower, upper) == []


def test_mass_ratio_support_edge_follows_m_min():
    """p(q | m1) must vanish once m2 = q m1 drops below m_min, for the same
    m_min the primary-mass taper uses."""
    model = get_model(MODEL)
    theta = get_fixed_population_params(MODEL)
    m1 = 20.0
    m_min = float(np.asarray(theta)[1])
    q_below = 0.5 * m_min / m1
    q_above = 0.9
    lp = np.asarray(model.log_p_pop(
        jnp.asarray([m1, m1]), jnp.asarray([q_below, q_above]),
        0.2, 0.0, theta))
    assert np.isneginf(lp[0])
    assert np.isfinite(lp[1])


# ---------------------------------------------------------------------------
# Sampler-facing health
# ---------------------------------------------------------------------------

def test_log_p_pop_is_finite_across_the_support():
    model = get_model(MODEL)
    theta = get_fixed_population_params(MODEL)
    lp = model.log_p_pop(
        jnp.asarray([10.0, 20.0, 35.0, 60.0]),
        jnp.asarray([0.8, 0.9, 0.7, 0.95]),
        jnp.asarray([0.1, 0.2, 0.3, 0.4]),
        jnp.asarray([0.0, 0.05, -0.05, 0.1]),
        theta,
    )
    assert np.all(np.isfinite(np.asarray(lp)))


def test_gradients_are_finite_at_the_prior_corners():
    """The taper's essential singularity and the delta_m = 0 / sigma_m floor
    corners run inside the jitted, differentiated likelihood; a NaN there
    walls the sampler out of a whole face of the prior box."""
    model = get_model(MODEL)
    base = np.asarray(get_fixed_population_params(MODEL), dtype=float)

    def logp(theta):
        return jnp.sum(model.log_p_pop(
            jnp.asarray([12.0, 35.0]), jnp.asarray([0.8, 0.9]),
            jnp.asarray([0.2, 0.3]), jnp.asarray([0.0, 0.0]), theta))

    corners = []
    for delta_m in (0.0, 10.0):
        for sigma_m in (1.0, 10.0):
            theta = base.copy()
            theta[5] = sigma_m
            theta[6] = delta_m
            corners.append(theta)
    for theta in corners:
        g = np.asarray(jax.grad(logp)(jnp.asarray(theta)))
        assert np.all(np.isfinite(g)), theta
