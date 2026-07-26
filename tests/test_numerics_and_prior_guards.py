"""Three independent statistics/numerics defects.

* truncnorm_sample's truncation normaliser was a difference of two erf-based
  CDFs; past ~8.3 sigma both round to the same float, the difference underflows
  to 0 and the tiny floor replaces the true span -- making the returned proposal
  log-density ~655 nats too large and killing every draw for that event.
* Cosmology prior overrides / fixed values were never checked against the
  tabulated distance grid, so an out-of-grid bound silently truncated the prior
  (logL = -inf) while the run reported the requested range.
* b_miss was sampled as a fully unidentified flat dimension whenever --use_lss
  was off, because delta_g is then the all-zero dummy and b_miss enters ONLY
  through max(1 + b_eff*delta_g, 0).
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.stats import norm
from scipy.special import logsumexp

from darksirens.gw.populations.sampling import truncnorm_sample
from darksirens.inference.prior import build_parameter_space


# ---------------------------------------------------------------------------
# truncnorm span
# ---------------------------------------------------------------------------

def _exact_log_span(mu, sigma, lo, hi):
    """log(Phi(b) - Phi(a)) computed in log space (no catastrophic underflow)."""
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    if b < 0:
        return logsumexp([norm.logcdf(b), norm.logcdf(a)], b=[1, -1])
    if a > 0:
        return logsumexp([norm.logcdf(-a), norm.logcdf(-b)], b=[1, -1])
    return float(np.log(norm.cdf(b) - norm.cdf(a)))


def _exact_log_s(mu, sigma, lo, hi, x):
    return (-0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)
            - 0.5 * np.log(2 * np.pi) - _exact_log_span(mu, sigma, lo, hi))


@pytest.mark.parametrize("mu,sigma,lo,hi", [
    (0.0, 0.30, -1.0, 1.0),      # centred
    (0.5, 0.20, -1.0, 1.0),
    (1.5, 0.20, -1.0, 1.0),      # 2.5 sigma outside
    (2.0, 0.10, -1.0, 1.0),      # 10 sigma  -- erf saturates here
    (3.0, 0.20, -1.0, 1.0),
    (0.9, 0.01, 0.1, 0.2),       # 70 sigma, narrow per-event box
    (-0.95, 0.02, 0.3, 0.5),     # far on the other side
])
def test_truncnorm_log_density_is_exact_far_outside_the_window(mu, sigma, lo, hi):
    out = truncnorm_sample(jnp.asarray([0.5]), jnp.asarray(mu), jnp.asarray(sigma), lo, hi)
    x = float(np.asarray(out.x).ravel()[0])
    got = float(np.asarray(out.log_s).ravel()[0])
    assert np.isfinite(got)
    assert got == pytest.approx(_exact_log_s(mu, sigma, lo, hi, x), abs=1e-8), (
        "log_s wrong far from the window -- the erf-difference span underflowed"
    )


def test_truncnorm_span_does_not_saturate_at_the_tiny_floor():
    """The specific regression: -log(span) must not pin to 708.4 (= -log(tiny))."""
    out = truncnorm_sample(jnp.asarray([0.5]), jnp.asarray(2.0), jnp.asarray(0.1), -1.0, 1.0)
    x = float(np.asarray(out.x).ravel()[0])
    got = float(np.asarray(out.log_s).ravel()[0])
    naive_floor_logs = (-0.5 * ((x - 2.0) / 0.1) ** 2 - np.log(0.1)
                        - 0.5 * np.log(2 * np.pi) + 708.3964185322641)
    assert abs(got - naive_floor_logs) > 100.0, "still using the saturated floor"


def test_truncnorm_degenerate_window_stays_finite():
    out = truncnorm_sample(jnp.asarray([0.5]), jnp.asarray(0.0), jnp.asarray(0.1), 0.3, 0.3)
    assert np.all(np.isfinite(np.asarray(out.log_s)))


def test_truncnorm_gradient_is_finite_far_outside():
    """The proposal density is differentiated w.r.t. the population parameters."""
    def f(mu):
        return jnp.sum(truncnorm_sample(
            jnp.asarray([0.5]), mu, jnp.asarray(0.01), 0.1, 0.2).log_s)
    g = float(jax.grad(f)(jnp.asarray(0.9)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# cosmology prior vs interpolation grid
# ---------------------------------------------------------------------------

def _space(**kw):
    return build_parameter_space("powerlaw+peak", False, False, False, **kw)


def test_default_cosmology_prior_is_inside_the_grid():
    _space()


@pytest.mark.parametrize("overrides", [
    {"w0": [-3.0, -0.5]},
    {"wa": [-4.0, 4.0]},
    {"Om0": [0.05, 0.9]},
])
def test_out_of_grid_prior_override_is_rejected(overrides):
    with pytest.raises(ValueError, match="interpolation grid"):
        _space(prior_overrides=overrides)


def test_in_grid_prior_override_is_accepted():
    _space(prior_overrides={"w0": [-2.0, -0.5], "wa": [-2.0, 2.0]})


def test_out_of_grid_fixed_value_is_rejected():
    """Previously unreachable: validate_fixed_parameter_overrides only inspects
    labels that are BOTH fixed and overridden."""
    with pytest.raises(ValueError, match="interpolation grid"):
        _space(fixed_parameter_values={"Om0": 0.5})


def test_in_grid_fixed_value_is_accepted():
    _space(fixed_parameter_values={"Om0": 0.31, "w0": -1.0, "wa": 0.0})


def test_h0_is_not_grid_constrained():
    """H0 rescales the table analytically; it is not an interpolation axis."""
    _space(prior_overrides={"H0": [10.0, 200.0]})


# ---------------------------------------------------------------------------
# phantom b_miss
# ---------------------------------------------------------------------------

def _has_b_miss(**kw):
    return any("b_miss" in lbl for lbl in _space(universe_model="dark_sirens", **kw)[0])


def test_b_miss_not_sampled_when_use_lss_is_off():
    """delta_g is the all-zero dummy, so b_miss has exactly zero effect."""
    assert not _has_b_miss(use_lss=False)


def test_b_miss_sampled_when_use_lss_is_on():
    assert _has_b_miss(use_lss=True)


def test_b_miss_still_dropped_when_q_table_is_active():
    """Q replaces the overdensity factor (pre-existing behaviour, unchanged)."""
    assert not _has_b_miss(use_lss=True, lss_completion_active=True)
    assert not _has_b_miss(use_lss=False, lss_completion_active=True)


def test_dropping_phantom_b_miss_reduces_the_dimension_by_one():
    on = len(_space(universe_model="dark_sirens", use_lss=True)[0])
    off = len(_space(universe_model="dark_sirens", use_lss=False)[0])
    assert on - off == 1
