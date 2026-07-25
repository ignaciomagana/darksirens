"""The N_obs(N_obs+3)/(2 N_eff) Monte-Carlo correction in selection.py.

This coefficient looks like a typo and has been "corrected" to
N_obs(N_obs+1)/(2 N_eff) by readers before. It is not a typo: the two forms
are two different PRIOR CONVENTIONS on the selection integral mu, and
N_obs(N_obs+3) is the one Farr (2019) eq. 11 states and the one the reference
implementation (``gwpopulation.vt.ResamplingVT.vt_factor``) uses.

Marginalising the rate with the scale-free prior p(R) ∝ 1/R leaves
L ∝ mu^{-N}; Monte-Carlo error makes mu uncertain, mu ~ N(mu_hat, sigma^2)
with sigma = mu_hat/sqrt(N_eff). Then

    flat prior on mu      ->  <mu^-N>/mu_hat^-N = 1 + N(N+1)/(2 N_eff)
    Jeffreys p(mu) ∝ 1/mu ->  <mu^-N>/mu_hat^-N = 1 + N(N+3)/(2 N_eff)

Both branches are integrated numerically below so the distinction is
demonstrated, not asserted. A reviewer who integrates only the flat branch
measures N(N+1) and concludes the code is wrong; these tests exist to make
that dead end obvious.
"""

import numpy as np
import pytest
from scipy.integrate import quad

import jax.numpy as jnp

from darksirens.likelihood.selection import selection_log_correction


def _marginalised_log_correction(N, Neff, jeffreys):
    """log <mu^-N> - N log mu_hat by direct quadrature, mu_hat = 1."""
    mu_hat, sig = 1.0, 1.0 / np.sqrt(Neff)
    lo, hi = mu_hat - 12.0 * sig, mu_hat + 12.0 * sig
    gauss = lambda m: np.exp(-0.5 * ((m - mu_hat) / sig) ** 2)
    weight = (lambda m: gauss(m) / m) if jeffreys else gauss
    num = quad(lambda m: m ** (-N) * weight(m), lo, hi, limit=400)[0]
    den = quad(weight, lo, hi, limit=400)[0]
    return float(np.log(num / den))


CASES = [(30, 30_000), (50, 50_000), (100, 100_000)]


# ============================================================================
# Which convention the closed form implements
# ============================================================================

@pytest.mark.parametrize("N,Neff", CASES)
def test_code_coefficient_is_the_jeffreys_branch(N, Neff):
    """N(N+3)/(2 Neff) reproduces the scale-invariant-prior quadrature."""
    coeff = N * (N + 3.0) / (2.0 * Neff)
    assert coeff == pytest.approx(_marginalised_log_correction(N, Neff, jeffreys=True),
                                  rel=2e-3)


@pytest.mark.parametrize("N,Neff", CASES)
def test_flat_prior_branch_gives_the_other_coefficient(N, Neff):
    """The flat-prior quadrature gives N(N+1)/2 — the value a reviewer
    integrating only that branch would 'measure', and would then report as a
    discrepancy against the code."""
    flat = _marginalised_log_correction(N, Neff, jeffreys=False)
    assert N * (N + 1.0) / (2.0 * Neff) == pytest.approx(flat, rel=2e-3)
    # The two conventions really are distinguishable at this N_eff.
    assert abs(flat - N * (N + 3.0) / (2.0 * Neff)) > 5e-4


@pytest.mark.parametrize("N,Neff", CASES)
def test_matches_gwpopulation_closed_form(N, Neff):
    """gwpopulation: vt_factor = mu / exp((3 + N) / (2 Neff)), and the
    likelihood carries -N log vt_factor."""
    mu = 0.37
    gwpop_vt_factor = mu / np.exp((3.0 + N) / (2.0 * Neff))
    want = -N * np.log(gwpop_vt_factor)

    got = float(selection_log_correction(
        jnp.asarray(np.log(mu)), jnp.asarray(float(Neff)), N,
        max_likelihood_variance=1e9,   # keep the guard out of the way here
    ))
    assert got == pytest.approx(want, rel=1e-12)


# ============================================================================
# The implementation, end to end
# ============================================================================

@pytest.mark.parametrize("soft_guard", [False, True])
def test_selection_log_correction_uses_n_times_n_plus_three(soft_guard):
    """Both the hard- and soft-guard branches carry the same coefficient."""
    N, Neff, mu = 40, 4.0e6, 0.5
    got = float(selection_log_correction(
        jnp.asarray(np.log(mu)), jnp.asarray(Neff), N,
        soft_guard=soft_guard, max_likelihood_variance=1e9,
    ))
    want = -N * np.log(mu) + N * (N + 3.0) / (2.0 * Neff)
    assert got == pytest.approx(want, rel=1e-10)
    # And is NOT the flat-prior form.
    assert got != pytest.approx(-N * np.log(mu) + N * (N + 1.0) / (2.0 * Neff), rel=1e-10)


@pytest.mark.parametrize("N", [30, 50, 100])
def test_convention_choice_is_subdominant_at_the_guard_boundary(N):
    """How much the prior convention could possibly matter.

    At the total-variance boundary N_eff = N_obs^2 / max_var the correction
    itself is NOT negligible — it tends to max_var/2 nats (~0.5 at the default
    max_var = 1). But the two conventions differ by only

        [N(N+3) - N(N+1)] / (2 N_eff) = N / N_eff = max_var / N_obs

    i.e. 1/N_obs nats, shrinking as the catalog grows. So the choice between
    them cannot drive the inference in the region the guard admits, whereas
    the guard boundary itself can.
    """
    max_var = 1.0
    Neff_boundary = N * N / max_var
    jeffreys = N * (N + 3.0) / (2.0 * Neff_boundary)
    flat = N * (N + 1.0) / (2.0 * Neff_boundary)

    assert jeffreys == pytest.approx((N + 3.0) * max_var / (2.0 * N), rel=1e-12)
    assert jeffreys - flat == pytest.approx(max_var / N, rel=1e-12)
    assert jeffreys - flat < 0.05
