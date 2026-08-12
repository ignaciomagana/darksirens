import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.gw.populations.parametric import (
    broken_powerlaw_norm,
    truncated_gaussian_norm,
)
from darksirens.gw.populations.registry import (
    get_fixed_population_params,
    get_model,
    pop_model_prior_parser,
)
from darksirens.gw.populations.utils import sfilter_low


MODEL = "gwtc5_fiducial_brokenpowerlaw+2peaks"


def test_gwtc5_fiducial_bpl2peaks_table5_priors_and_fiducial_order():
    lower, upper, labels, _, latex = pop_model_prior_parser(MODEL)

    expected = [
        (r"$\alpha_1$", -4.0, 12.0),
        (r"$\alpha_2$", -4.0, 12.0),
        (r"$m_{\rm break}$", 20.0, 50.0),
        (r"$\mu_1$", 5.0, 20.0),
        (r"$\sigma_1$", 0.0, 10.0),
        (r"$\mu_2$", 25.0, 60.0),
        (r"$\sigma_2$", 0.0, 10.0),
        (r"$m_{1,{\rm low}}$", 3.0, 10.0),
        (r"$\delta m_1$", 0.0, 10.0),
        (r"$\lambda_0$", 0.0, 1.0),
        (r"$\lambda_1$", 0.0, 1.0),
        (r"$\beta_q$", -2.0, 7.0),
        (r"$m_{2,{\rm low}}$", 3.0, 10.0),
        (r"$\delta m_2$", 0.0, 10.0),
    ]

    assert list(zip(labels[:14], lower[:14], upper[:14])) == expected
    assert latex == r"\text{GWTC-5 Fiducial BPL+2G}"
    assert len(get_fixed_population_params("gwtc5_brokenpowerlaw+2peaks")) == len(labels)
    assert len(get_fixed_population_params("gwtc5_fiducial_bpl2peaks")) == len(labels)

    theta = get_fixed_population_params(MODEL)
    assert len(theta) == len(labels)
    assert labels[8] == r"$\delta m_1$"
    assert labels[13] == r"$\delta m_2$"
    assert theta[8] == pytest.approx(3.5302)
    assert theta[13] == pytest.approx(4.8128)


#: Marginal posterior medians from the LVK GWTC-5.0 data release, run
#: ``gwtc5_updated_default_mmax`` with the ``PowerLawRedshift`` arm (the one
#: whose rate law matches this model's ``(1+z)^(gamma-1)``).  Ordered as
#: ``param_specs``.  See the ``fiducial=`` comment in ``registry.py``.
LVK_GWTC5_MEDIANS = (
    1.4816, 5.4187, 37.451, 9.9109, 0.7841, 32.3273, 5.7263,
    4.4856, 3.5302, 0.4004, 0.5457, 1.0438, 3.4633,
    4.8128, 0.0633, 0.3654, 2.5439,
)


def test_gwtc5_fiducial_matches_lvk_release_medians():
    """The fiducial vector is the published posterior, not round placeholders.

    Guards the specific regression this replaced: ``gamma = 0.0`` asserted a
    non-evolving comoving merger rate against a 3.6-sigma measurement, which
    biases a dark-siren H0 through the source redshift distribution.
    """
    _, _, labels, _, _ = pop_model_prior_parser(MODEL)
    theta = get_fixed_population_params(MODEL)

    assert len(theta) == len(LVK_GWTC5_MEDIANS) == len(labels)
    for label, value, expected in zip(labels, theta, LVK_GWTC5_MEDIANS):
        assert value == pytest.approx(expected), label

    # gamma is the release's ``lamb``; R(z) ~ (1+z)^gamma with no offset.
    assert labels[16] == r"$\gamma$"
    assert theta[16] == pytest.approx(2.5439)

    # The mixture weights must stay on the simplex, and the 35 Msun peak must
    # NOT carry the old equal-thirds weight -- the release puts ~5% there.
    lambda0, lambda1 = float(theta[9]), float(theta[10])
    second_peak = 1.0 - lambda0 - lambda1
    assert 0.0 < second_peak < 1.0
    assert second_peak == pytest.approx(0.0539, abs=5.0e-4)


def test_gwtc5_fiducial_lies_within_the_release_priors():
    """Every fiducial sits strictly inside its own Table 5 prior box.

    ``m_2,low`` and ``mu_chi`` previously sat exactly ON their prior floors,
    which is a degenerate starting point for a fixed-population run.
    """
    lower, upper, labels, _, _ = pop_model_prior_parser(MODEL)
    theta = get_fixed_population_params(MODEL)

    for label, lo, hi, value in zip(labels, lower, upper, theta):
        assert lo < float(value) < hi, f"{label} = {value} not inside ({lo}, {hi})"

    # Table 5 constrains m_2,low <= m_1,low; the medians must respect it.
    assert float(theta[12]) <= float(theta[7])


def test_gwtc5_fiducial_bpl2peaks_log_population_is_finite_and_enforces_constraints():
    model = get_model(MODEL)
    theta = get_fixed_population_params(MODEL)

    logp = model.log_p_pop(
        jnp.array([20.0, 40.0]),
        jnp.array([0.8, 0.9]),
        jnp.array([0.1, 0.2]),
        jnp.array([0.0, 0.05]),
        theta,
    )
    assert np.all(np.isfinite(np.asarray(logp)))

    # Table 5 has m2_low ~ U(3, m1_low) and lambda_0, lambda_1 on a simplex.  The
    # implemented m2_low/m1_low prior is the uniform ORDERED TRIANGLE, not that
    # conditional (see GWTC5FiducialBPL2PeaksPopulationModel.constraint_groups);
    # only the support statement asserted here is shared by both.
    bad_m2_low = theta.at[12].set(theta[7] + 0.5)
    assert np.all(np.isneginf(np.asarray(model.log_p_pop(20.0, 0.8, 0.1, 0.0, bad_m2_low))))

    bad_simplex = theta.at[9].set(0.8).at[10].set(0.4)
    assert np.all(np.isneginf(np.asarray(model.log_p_pop(20.0, 0.8, 0.1, 0.0, bad_simplex))))


# ===========================================================================
# Component normalisers: analytic, exact, differentiable in the truncation edge
# ===========================================================================
# The three components used to be normalised by a trapezoid over the fixed
# 500-node grid linspace(1, 300, 500) (h = 0.5992 Msun) of a HARD-masked
# integrand, ``(m >= m1_low) & (m < m_high)``.  Two defects followed, both
# measured on master at the registered fiducial vector:
#
# (a) narrow peaks were unresolvable.  Both width priors start at sigma = 0, and
#     trapezoid error for a Gaussian grows like 2 exp(-2 pi^2 sigma^2 / h^2), so
#     with m1_low = 5 the peak normaliser / analytic truth ran
#         sigma = 0.3: 1.0141  0.9982  0.9859  1.0019
#         sigma = 0.2: 1.2203  0.9714  0.7804  1.0290
#         sigma = 0.1: 2.3732  0.6435  0.0571  0.9330
#     for mu = 10.00, 10.15, 10.30, 10.45 -- a 41x swing in the component's
#     effective amplitude from moving mu by 0.45 Msun.  The same coarse grid
#     normalised the TOTAL mixture, so the model's own p(m1) integrated to
#     0.7883 at sigma_1 = 0.1 and to 7.0434 at (mu_1, sigma_1) = (10.3, 0.1)
#     instead of 1.
#
# (b) the normalisers were step functions of the SAMPLED edge m1_low, jumping
#     whenever the edge crossed a node: N_bpl = 226.603 for m1_low in
#     [5.00, 5.19] then 199.399 from 5.20 (truth: 221.648 -> 212.225, smooth), so
#     log p(m1=30, q=0.7) dropped by 1.96e-2 discontinuously at every node
#     (5.193, 5.792, ...).  And because a boolean mask has zero derivative,
#     autodiff lost the whole d/dm1_low channel: jax.grad gave +0.0656 / +0.0653
#     / +0.0708 / +0.0730 / +0.0812 / +0.0905 at m1_low = 4.6 ... 6.4 against a
#     quadrature reference of +0.0211 / +0.0267 / +0.0312 / +0.0358 / +0.0422 /
#     +0.0486 -- a factor 1.9-3.1 too large, which biases every NUTS trajectory.
#
# Both are fixed by closed-form component normalisers (erf / power-law
# antiderivative) plus a window-relative quadrature for the taper deficit.

M_HIGH = 300.0
_SIGMAS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)
# mu swept ACROSS one old mass-grid cell (h = 0.5992 Msun): the pre-fix error
# oscillated with the offset of mu from the nodes.
_MUS = (5.0, 10.0, 10.15, 10.3, 10.45, 20.0, 25.0, 35.0, 60.0)


def _quad_trunc_gauss(mu, sigma, lo, hi=M_HIGH):
    """scipy.integrate.quad reference, restricted to where the integrand is
    representable.

    Clipping to ``[mu - 40 sigma, mu + 40 sigma]`` changes the integral by less
    than 1e-300 relative and is what lets the adaptive rule SEE a narrow peak
    inside a 300-Msun window (unclipped, quad silently returns half the integral
    -- or 1e-52 -- for sigma <~ 0.05).  Returns None when the clipped range is
    empty, i.e. the peak is more than 40 sigma outside the window and quadrature
    is not the right reference (``_cephes_trunc_gauss`` covers those)."""
    from scipy.integrate import quad

    a = max(lo, mu - 40.0 * sigma)
    b = min(hi, mu + 40.0 * sigma)
    if not b > a:
        return None
    f = lambda m: np.exp(-0.5 * ((m - mu) / sigma) ** 2)
    pts = [mu] if a < mu < b else None
    return quad(f, a, b, points=pts, limit=800, epsabs=0.0, epsrel=1e-13)[0]


def _cephes_trunc_gauss(mu, sigma, lo, hi=M_HIGH):
    """Closed form via scipy/Cephes ``ndtr`` -- an implementation independent of
    the jax erf/erfc the model uses, and accurate deep into the tail."""
    from scipy.stats import norm as sp_norm

    return sigma * np.sqrt(2.0 * np.pi) * (
        sp_norm.sf((lo - mu) / sigma) - sp_norm.sf((hi - mu) / sigma))


def _quad_broken_powerlaw(a1, a2, mb, lo, hi=M_HIGH):
    from scipy.integrate import quad

    out = 0.0
    if min(mb, hi) > lo:
        out += quad(lambda m: (m / mb) ** (-a1), lo, min(mb, hi),
                    limit=500, epsabs=0.0, epsrel=1e-13)[0]
    if hi > max(mb, lo):
        out += quad(lambda m: (m / mb) ** (-a2), max(mb, lo), hi,
                    limit=500, epsabs=0.0, epsrel=1e-13)[0]
    return out


def test_peak_normaliser_is_exact_across_the_whole_sigma_prior():
    """Fix for (a).  The truncated-Gaussian normaliser matches scipy for every
    width in the prior [0, 10] -- including the narrow-peak corner the 0.6-Msun
    mass grid cannot resolve at all -- and for mu swept across a grid cell."""
    pytest.importorskip("scipy")
    worst_q = worst_c = 0.0
    n_q = n_c = 0
    for lo in (3.0, 5.0, 10.0):
        for sigma in _SIGMAS:
            for mu in _MUS:
                got = float(truncated_gaussian_norm(mu, sigma, lo, M_HIGH))
                ref_q = _quad_trunc_gauss(mu, sigma, lo)
                ref_c = _cephes_trunc_gauss(mu, sigma, lo)
                if ref_q is not None and ref_q > 1e-290:
                    worst_q = max(worst_q, abs(got / ref_q - 1.0)); n_q += 1
                if ref_c > 1e-290:
                    worst_c = max(worst_c, abs(got / ref_c - 1.0)); n_c += 1
    assert n_q > 250 and n_c > 250, (n_q, n_c)
    # Measured 6.6e-13 (quad) and 1.1e-13 (Cephes); pre-fix the same sweep hit
    # ratios of 2.37 and 0.057.
    assert worst_q < 1e-9, worst_q
    assert worst_c < 1e-9, worst_c


def test_peak_normaliser_at_the_sigma_zero_prior_edge():
    """sigma = 0 is inside the prior: the normaliser must follow the SAME floor
    the density applies (1e-12), so the pair stays consistent and the component
    degenerates to a vanishing spike instead of 0/0."""
    for sigma in (0.0, 1e-12, 1e-6):
        got = float(truncated_gaussian_norm(10.0, sigma, 5.0, M_HIGH))
        assert np.isclose(got, max(sigma, 1e-12) * np.sqrt(2.0 * np.pi), rtol=1e-12)


def test_broken_powerlaw_normaliser_is_exact():
    """The other closed form, including alpha = 1 (where the antiderivative is a
    log, not a power) and a break outside the truncation range."""
    pytest.importorskip("scipy")
    worst = 0.0
    for a1 in (-4.0, 0.0, 1.0, 1.0 + 1e-9, 2.0, 6.0, 12.0):
        for a2 in (-4.0, 1.0, 4.0, 12.0):
            for mb in (20.0, 35.0, 50.0):
                for lo in (3.0, 5.0, 5.193, 7.5, 10.0):
                    got = float(broken_powerlaw_norm(a1, a2, mb, lo, M_HIGH))
                    worst = max(worst, abs(got / _quad_broken_powerlaw(a1, a2, mb, lo) - 1.0))
    assert worst < 1e-9, worst          # measured 2.9e-15
    # Break below / above the range: the formula must degenerate to one slope.
    assert np.isclose(float(broken_powerlaw_norm(2.0, 4.0, 2.0, 5.0, M_HIGH)),
                      _quad_broken_powerlaw(2.0, 4.0, 2.0, 5.0), rtol=1e-12)
    assert np.isclose(float(broken_powerlaw_norm(2.0, 4.0, 400.0, 5.0, M_HIGH)),
                      _quad_broken_powerlaw(2.0, 4.0, 400.0, 5.0), rtol=1e-12)


def _logp_at(theta, m1=30.0, q=0.7, z=0.1, chi=0.0):
    model = get_model(MODEL)
    return float(model.log_p_pop(jnp.asarray([m1]), jnp.asarray([q]),
                                 jnp.asarray([z]), jnp.asarray([chi]),
                                 jnp.asarray(theta))[0])


def test_density_is_continuous_in_m1_low_at_the_old_grid_nodes():
    """Fix for (b), part 1.  The normalisers no longer step when m1_low crosses a
    mass-grid node, so the likelihood surface has no spurious 2% cliffs at
    m1_low = 1 + k*0.5992."""
    theta = np.asarray(get_fixed_population_params(MODEL), dtype=float)
    for node in (5.193, 5.792, 6.391):
        lo = np.array(theta); lo[7] = node - 1.0e-6
        hi = np.array(theta); hi[7] = node + 1.0e-6
        jump = _logp_at(hi) - _logp_at(lo)
        # Pre-fix: -1.96e-2 and -1.88e-2 at the first two nodes.  Post-fix the
        # difference is just the smooth slope over 2e-6 in m1_low.
        assert abs(jump) < 1e-6, (node, jump)
    # And the sweep is smooth: no cliff anywhere in the prior.
    #
    # Table 5 constrains m_2,low <= m_1,low, so the model is correctly -inf
    # below the fiducial m_2,low; start the sweep just inside the support
    # rather than at a constant that happens to clear the old fiducial.
    #
    # The test is a REFINEMENT check rather than an absolute bound on the step.
    # A bound like ``steps.max() < 5e-3`` conflates a cliff with a steep smooth
    # slope: it depends on both the grid spacing and the fiducial.  Under the
    # LVK-median fiducial the 10 Msun peak is narrow (sigma_1 = 0.78 against the
    # old placeholder 2.0), so the taper crossing it gives a genuine smooth
    # slope of 0.384 in log p per unit m1_low -- which trips a 5e-3 absolute
    # bound at 300 points while being perfectly continuous.  Halving the spacing
    # halves a smooth step and leaves a true discontinuity untouched, so the
    # ratio is the discriminating quantity and is fiducial-independent.
    lo_edge = float(theta[12]) + 1.0e-3

    def _max_step(n):
        mls = np.linspace(lo_edge, 9.95, n)
        vals = np.array([_logp_at(np.concatenate([theta[:7], [ml], theta[8:]]))
                         for ml in mls])
        steps = np.abs(np.diff(vals))
        assert np.all(np.isfinite(vals)), mls[~np.isfinite(vals)][:5]
        return steps.max(), mls[np.argmax(steps)]

    coarse, where = _max_step(300)
    fine, _ = _max_step(600)
    assert coarse / fine == pytest.approx(2.0, abs=0.3), (coarse, fine, where)


def test_m1_low_gradient_matches_a_quadrature_reference():
    """Fix for (b), part 2.  The truncation edge now carries its full derivative
    channel: d log N_i/d m1_low = -p_i(m1_low)/N_i, which the hard mask zeroed."""
    pytest.importorskip("scipy")
    from scipy.integrate import quad

    theta = np.asarray(get_fixed_population_params(MODEL), dtype=float)
    a1, a2, mb, mu1, s1, mu2, s2, _, dm1, l0, l1 = theta[:11]

    def ref_logp(ml, m1=30.0):
        """log p(m1) with every normaliser done by adaptive quadrature."""
        def dens(m):
            m = np.asarray(m, dtype=float)
            inside = (m >= ml) & (m < M_HIGH)
            bpl = np.where(inside, np.where(m < mb, (m / mb) ** (-a1),
                                            (m / mb) ** (-a2)), 0.0)
            g1 = np.where(inside, np.exp(-0.5 * ((m - mu1) / s1) ** 2), 0.0)
            g2 = np.where(inside, np.exp(-0.5 * ((m - mu2) / s2) ** 2), 0.0)
            mix = (l0 * bpl / _quad_broken_powerlaw(a1, a2, mb, ml)
                   + l1 * g1 / _quad_trunc_gauss(mu1, s1, ml)
                   + (1.0 - l0 - l1) * g2 / _quad_trunc_gauss(mu2, s2, ml))
            return mix * np.asarray(sfilter_low(jnp.asarray(m), ml, dm1))
        norm = quad(lambda m: float(dens(m)), ml, M_HIGH, limit=800,
                    points=[ml + dm1, mu1, mu2])[0]
        return np.log(float(dens(m1)) / norm)

    model = get_model(MODEL)
    grad_fn = jax.grad(lambda th: model.log_p_pop(
        jnp.asarray([30.0]), jnp.asarray([0.7]), jnp.asarray([0.1]),
        jnp.asarray([0.0]), th)[0])
    for ml in (4.6, 5.0, 5.3, 5.6, 6.0, 6.4):
        th = jnp.asarray(theta).at[7].set(ml)
        got = float(np.asarray(grad_fn(th))[7])
        h = 1.0e-5
        ref = (ref_logp(ml + h) - ref_logp(ml - h)) / (2.0 * h)
        # Pre-fix these ratios were 1.86-3.11.
        assert abs(got / ref - 1.0) < 1e-3, (ml, got, ref)


def _reference_pm1_integral(theta):
    """int p(m1) dm1 on a grid refined around both peaks and the taper window."""
    m1_low, dm1 = theta[7], theta[8]
    nodes = [np.linspace(1.0, M_HIGH, 30001)]
    for mu, sig in ((theta[3], theta[4]), (theta[5], theta[6])):
        w = max(8.0 * sig, 1e-6)
        nodes.append(np.linspace(max(mu - w, 1.0), min(mu + w, M_HIGH), 20001))
    nodes.append(np.linspace(m1_low, min(m1_low + max(dm1, 1e-6), M_HIGH), 20001))
    m = np.unique(np.concatenate(nodes))
    v = np.asarray(get_model(MODEL).mass_component(jnp.asarray(m),
                                                   jnp.asarray(theta[:11])))
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(m)))


def test_mass_density_is_normalised_including_narrow_peaks():
    """The mixture the model returns integrates to 1 for every width in the
    prior, narrow peaks included.  Pre-fix (coarse total-norm trapezoid):
    0.99998 at the fiducial but 0.7883 at sigma_1 = 0.1, 7.0434 at
    (mu_1, sigma_1) = (10.3, 0.1) and 0.6657 at sigma_1 = 0.01."""
    theta = np.asarray(get_fixed_population_params(MODEL), dtype=float)
    cases = {"fiducial": theta}
    for label, upd in (("sigma1=0.1", {4: 0.1}),
                       ("mu1=10.3,sigma1=0.1", {3: 10.3, 4: 0.1}),
                       ("sigma1=0.01", {4: 0.01}),
                       ("sigma2=0.15", {6: 0.15}),
                       ("peak inside taper window", {3: 6.0, 4: 0.5, 8: 8.0})):
        th = np.array(theta)
        for i, v in upd.items():
            th[i] = v
        cases[label] = th
    for label, th in cases.items():
        got = _reference_pm1_integral(th)
        assert abs(got - 1.0) < 2e-4, (label, got)
