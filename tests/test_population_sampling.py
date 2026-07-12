"""
test_population_sampling.py
---------------------------
Grid inverse-CDF samplers for the flow-surrogate path
(darksirens/gw/populations/sampling.py).

The load-bearing property is exactness of the proposal density: every
sampler returns log_s such that E_s[t/s] equals the integral of the target
t, at any grid resolution. That identity is tested directly, along with
empirical marginals, the analytic truncated normal, and common-random-number
continuity in the hyperparameters.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.gw.populations.sampling import (
    cell_centers,
    make_mass_q_edges,
    resolve_mass_grid_bounds,
    sample_histogram_1d,
    sample_mass_q,
    sample_z_from_grid,
    truncnorm_sample,
)

RNG = np.random.default_rng(20260712)
J = 200_000
U = jnp.asarray(RNG.uniform(size=(J, 4)))


def _log_t_cells(model, theta, m1_edges, q_edges):
    tm = model.mixture_theta(theta)
    m1c = cell_centers(m1_edges)
    qc = cell_centers(q_edges)
    M1, Q = jnp.meshgrid(m1c, qc, indexing="ij")
    p = model.mixture.mass_q_density(M1.ravel(), Q.ravel(), tm).reshape(M1.shape)
    return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)


@pytest.fixture(scope="module")
def plp():
    model = get_model("powerlaw+peak")
    theta = jnp.asarray(np.asarray(get_fixed_population_params("powerlaw+peak")))
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges, q_edges = make_mass_q_edges(m1_lo, m1_hi, n_m1=256, n_q=128)
    log_t = _log_t_cells(model, theta, m1_edges, q_edges)
    return model, theta, m1_edges, q_edges, log_t


def test_histogram_1d_exactness_and_support():
    # Target: unnormalised linear ramp on [0, 2]; exact integral = 2.
    edges = jnp.linspace(0.0, 2.0, 41)
    dens = cell_centers(edges)  # t(x) = x at cell centers
    out = sample_histogram_1d(U[:, 0], edges, jnp.log(dens))
    assert out.x.shape == (J,)
    assert float(out.x.min()) >= 0.0 and float(out.x.max()) <= 2.0
    # E_s[t/s] with t the piecewise-constant cell density itself == norm.
    t_at = jnp.log(dens)[out.cell]
    est = jnp.exp(jax.scipy.special.logsumexp(t_at - out.log_s) - jnp.log(J))
    np.testing.assert_allclose(float(est), float(jnp.exp(out.log_norm)), rtol=1e-12)
    # Empirical CDF vs analytic CDF of the ramp: x^2/4.
    xs = np.sort(np.asarray(out.x))
    ref = xs**2 / 4.0
    emp = np.arange(1, J + 1) / J
    assert np.abs(emp - ref).max() < 5e-3


def test_mass_q_exact_reweight_recovers_target_integral(plp):
    model, theta, m1_edges, q_edges, log_t = plp
    out = sample_mass_q(U[:, 0], U[:, 1], m1_edges, q_edges, log_t)

    # Evaluate the TRUE target at the sampled points and reweight by the
    # exact proposal density: the mean of t/s estimates ∬ t dm1 dq.
    tm = model.mixture_theta(theta)
    p = model.mixture.mass_q_density(out.m1, out.q, tm)
    log_p = jnp.where(p > 0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
    est = float(jnp.exp(
        jax.scipy.special.logsumexp(log_p - out.log_s) - jnp.log(J)
    ))

    # Reference: dense trapezoid integral of the true target.
    m1g = jnp.geomspace(m1_edges[0], m1_edges[-1], 1024)
    qg = jnp.linspace(q_edges[0], q_edges[-1], 512)
    M1, Q = jnp.meshgrid(m1g, qg, indexing="ij")
    P = model.mixture.mass_q_density(M1.ravel(), Q.ravel(), tm).reshape(M1.shape)
    ref = float(jnp.trapezoid(jnp.trapezoid(P, qg, axis=1), m1g))

    assert est == pytest.approx(ref, rel=2e-2)


def test_mass_q_marginals_match_brute_force(plp):
    model, theta, m1_edges, q_edges, log_t = plp
    out = sample_mass_q(U[:, 0], U[:, 1], m1_edges, q_edges, log_t)
    m1 = np.asarray(out.m1)
    assert (m1 >= float(m1_edges[0])).all() and (m1 <= float(m1_edges[-1])).all()

    # Binned m1 marginal of the draws vs the (normalised) grid marginal.
    dm = np.diff(np.asarray(m1_edges))
    dq = np.diff(np.asarray(q_edges))
    mass = np.exp(np.asarray(log_t)) * dm[:, None] * dq[None, :]
    marg = mass.sum(axis=1)
    marg = marg / marg.sum()
    counts, _ = np.histogram(m1, bins=np.asarray(m1_edges))
    emp = counts / counts.sum()
    assert np.abs(emp - marg).sum() < 0.02  # total-variation distance

    q = np.asarray(out.q)
    margq = mass.sum(axis=0)
    margq = margq / margq.sum()
    countsq, _ = np.histogram(q, bins=np.asarray(q_edges))
    empq = countsq / countsq.sum()
    assert np.abs(empq - margq).sum() < 0.02


def test_truncnorm_against_scipy():
    from scipy import stats

    mu, sigma, lo, hi = 0.06, 0.12, -1.0, 1.0
    u = U[: 50_000, 2]
    out = truncnorm_sample(u, mu, sigma, lo, hi)
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    ref_x = stats.truncnorm.ppf(np.asarray(u), a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(np.asarray(out.x), ref_x, atol=1e-9)
    ref_logpdf = stats.truncnorm.logpdf(ref_x, a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(np.asarray(out.log_s), ref_logpdf, atol=1e-9)


def test_z_sampler_matches_analytic_cdf():
    # Toy target resembling a volumetric prior: t(z) ∝ z^2 on [0, 3].
    zgrid = jnp.linspace(0.0, 3.0, 400)
    log_t = jnp.where(zgrid > 0, 2.0 * jnp.log(jnp.maximum(zgrid, 1e-300)), -jnp.inf)
    out = sample_z_from_grid(U[:, 3], zgrid, log_t)
    zs = np.sort(np.asarray(out.x))
    ref = (zs / 3.0) ** 3
    emp = np.arange(1, J + 1) / J
    assert np.abs(emp - ref).max() < 5e-3


def test_common_random_numbers_continuity(plp):
    model, theta, m1_edges, q_edges, _ = plp
    theta2 = jnp.asarray(np.asarray(theta).copy())
    # Perturb alpha (index 1: [v1, alpha, m_min, m_max, dm_min, dm_max, ...]).
    theta2 = theta2.at[1].add(1e-4)

    u1, u2 = U[:4096, 0], U[:4096, 1]
    a = sample_mass_q(u1, u2, m1_edges, q_edges,
                      _log_t_cells(model, theta, m1_edges, q_edges))
    b = sample_mass_q(u1, u2, m1_edges, q_edges,
                      _log_t_cells(model, theta2, m1_edges, q_edges))
    # Same base uniforms + tiny hyperparameter step -> tiny coordinate step.
    assert float(jnp.median(jnp.abs(a.m1 - b.m1))) < 0.05
    assert float(jnp.median(jnp.abs(a.q - b.q))) < 0.01


def test_samplers_are_jittable(plp):
    model, theta, m1_edges, q_edges, log_t = plp

    @jax.jit
    def draw(u1, u2, log_t):
        out = sample_mass_q(u1, u2, m1_edges, q_edges, log_t)
        return out.m1, out.q, out.log_s

    m1, q, log_s = draw(U[:1024, 0], U[:1024, 1], log_t)
    assert np.isfinite(np.asarray(log_s)).all()
