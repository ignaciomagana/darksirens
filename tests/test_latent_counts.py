"""PR-3 pins for darksirens.redshift.latent_counts (PLAN §6.3 P4, P5, P6).

P4  ``shell_multinomial_logl`` vs scipy.stats.multinomial (coefficient
    restored) at 1e-10.
P5  ``hessian_separable`` (eq. 3, WITH the rank-1 term) vs the dense
    ``jax.hessian`` of the same (1') objective at 1e-10.
P6  ``count_map_solve`` vs a dense scipy Newton on the same objective at
    1e-8, plus ``grad_inf(xi_hat) < 1e-8``.

Plus: gradient vs jax.grad (analytic separable form); the IFT sensitivity
column against a finite-difference re-solve in ``b_gal``; antithetic draw
structure; integer-count round-trip against np.histogram.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift.latent_counts import (
    CountOperator,
    TracerCounts,
    count_map_solve,
    counts_from_catalog,
    dgrad_db,
    gradient,
    hessian_separable,
    laplace_draws,
    laplace_evidence,
    make_count_operator,
    objective,
    sensitivity,
    shell_multinomial_logl,
)
from darksirens.redshift.latent_field import build_latent_basis, shell_response

M_SPH, M_Z, G_S, N_FIT = 12, 3, 5, 40
Z_HI = 0.3


def _fixture(seed=0, bias=1.3, full=False):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_fine = np.linspace(1e-3, Z_HI, 120)
    basis = build_latent_basis(
        v, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.6, ls_z=0.15,
        zeta_fine=np.log1p(z_fine))
    edges = np.linspace(0.02, Z_HI, G_S + 1)
    W = shell_response(edges, z_fine,
                       lambda z: 0.02 * np.ones_like(z),
                       lambda z: z ** 2 + 1e-6)
    counts = rng.poisson(30.0, size=(N_FIT, G_S)).astype(float)
    f_p = rng.uniform(0.5, 1.0, size=N_FIT)
    tracer = TracerCounts(pix=np.arange(N_FIT), counts=counts,
                          completeness=f_p, bias=bias)
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, tracer)
    xi = rng.normal(size=M_SPH * M_Z) * 0.5
    if full:
        return op, xi, basis, tracer, z_fine, edges
    return op, xi


def _dense_pi(op, xi):
    Xi = np.reshape(np.asarray(xi), (op.m_sph, op.m_z))
    eta = op.bias * np.asarray(op.proj_sph) @ Xi @ np.asarray(op.phi_shell).T
    a = np.exp(np.asarray(op.log_fp))[:, None] * np.exp(eta)
    return a / a.sum(axis=0, keepdims=True)


def test_p4_logl_vs_scipy():
    from scipy.stats import multinomial
    from scipy.special import gammaln

    op, xi = _fixture()
    pi = _dense_pi(op, xi)
    N = np.asarray(op.counts)
    T = N.sum(axis=0)
    ref = 0.0
    for g in range(G_S):
        lp = multinomial(int(T[g]), pi[:, g]).logpmf(N[:, g])
        coeff = gammaln(T[g] + 1.0) - gammaln(N[:, g] + 1.0).sum()
        ref += lp - coeff
    got = float(shell_multinomial_logl(jnp.asarray(xi), op))
    assert abs(got - ref) < 1e-10 * max(1.0, abs(ref))


def test_gradient_vs_autodiff():
    op, xi = _fixture(seed=1)
    g_ana = np.asarray(gradient(jnp.asarray(xi), op))
    g_ad = np.asarray(jax.grad(lambda x: objective(x, op))(jnp.asarray(xi)))
    np.testing.assert_allclose(g_ana, g_ad, rtol=0, atol=1e-11)


def test_p5_hessian_vs_dense_autodiff():
    op, xi = _fixture(seed=2)
    H_sep = np.asarray(hessian_separable(jnp.asarray(xi), op))
    H_ad = np.asarray(jax.hessian(lambda x: objective(x, op))(jnp.asarray(xi)))
    assert np.max(np.abs(H_sep - H_ad)) < 1e-10
    # H >= I (Fisher + prior): smallest eigenvalue >= 1
    w = np.linalg.eigvalsh(H_sep)
    assert w.min() >= 1.0 - 1e-10


def test_p6_solve_vs_scipy_newton():
    from scipy.optimize import minimize

    op, _ = _fixture(seed=3)
    sol = count_map_solve(op)
    assert float(sol["grad_inf"]) < 1e-8
    res = minimize(
        lambda x: float(objective(jnp.asarray(x), op)),
        np.zeros(op.rank),
        jac=lambda x: np.asarray(gradient(jnp.asarray(x), op)),
        method="trust-ncg",
        hess=lambda x: np.asarray(hessian_separable(jnp.asarray(x), op)),
        options=dict(gtol=1e-12, maxiter=200))
    np.testing.assert_allclose(np.asarray(sol["xi_hat"]), res.x,
                               rtol=0, atol=1e-8)


def test_theta_is_an_argument_not_a_rebuild():
    """The same solver machinery runs at a different W(theta) — the anchor is
    the special case, not the only case (PLAN §7 PR-3)."""
    op, _, basis, tracer, z_fine, edges = _fixture(seed=4, full=True)
    # a different within-shell base — a proxy for theta != theta_ref
    W2 = shell_response(edges, z_fine,
                        lambda z: 0.02 * np.ones_like(z),
                        lambda z: (z + 0.05) ** 1.5)
    # rebuild only phi_shell (theta flows through W), reuse everything else
    op2 = make_count_operator(basis.phi_sph, basis.phi_z_fine, W2, tracer)
    # solve both; different W -> different xi_hat, same machinery converges
    s1 = count_map_solve(op)
    s2 = count_map_solve(op2)
    assert float(s2["grad_inf"]) < 1e-8
    assert not np.allclose(np.asarray(s1["xi_hat"]), np.asarray(s2["xi_hat"]))


def test_sensitivity_b_column_linear_response():
    op, _ = _fixture(seed=5, bias=1.0)
    sol = count_map_solve(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    col = np.asarray(sensitivity(
        xi_hat, L, dgrad_db(xi_hat, op)[:, None]))[:, 0]
    db = 1e-4
    op_db = CountOperator(op.proj_sph, op.phi_shell, op.counts,
                          op.log_fp, op.bias + db)
    xi_db = np.asarray(count_map_solve(op_db, xi0=xi_hat)["xi_hat"])
    fd = (xi_db - np.asarray(xi_hat)) / db
    np.testing.assert_allclose(col, fd, rtol=2e-3, atol=1e-6)


def test_laplace_draws_antithetic_and_covariance():
    """Antithetic PAIRING and the draw covariance — both discriminating.

    This test used to assert

        q  = resid^T (L L^T) resid
        g2 = L^T resid
        assert q == (g2**2).sum(1)

    which is the identity ``x^T L L^T x == |L^T x|^2``: true to 4e-16 for a
    uniform-random ``resid`` that is not a draw from anything at all.  It could
    not fail, so it pinned nothing.  Replaced with the two properties that
    actually characterise ``laplace_draws``.
    """
    op, _ = _fixture(seed=6)
    sol = count_map_solve(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    xh, Ln = np.asarray(xi_hat), np.asarray(L)
    n_draw = 64
    draws = np.asarray(laplace_draws(xi_hat, L, n_draw, jax.random.PRNGKey(0)))

    # ``steps = g L^{-1}``, so ``resid @ L`` recovers the whitened draws g.
    resid = draws - xh[None, :]
    white = resid @ Ln

    # (1) ANTITHETIC PAIRING, member by member: partner of k is k + n/2, and
    # the pair is an exact sign flip. A shuffled or unpaired stream still has
    # mean ~0 but fails here -- this is the property PR-5b's balanced member
    # ordering depends on, and the one the old assertion could not see.
    half = n_draw // 2
    np.testing.assert_allclose(white[:half], -white[half:], rtol=0, atol=1e-12)

    # ... and the pairing makes the mean EXACTLY xi_hat, not merely close.
    np.testing.assert_allclose(draws.mean(0), xh, rtol=0, atol=1e-12)
    np.testing.assert_allclose(white.mean(0), 0.0, rtol=0, atol=1e-12)

    # (2) The whitened draws must be standard normal, i.e. the member
    # covariance approaches H^{-1} = (L L^T)^{-1}. Independent of (1): a stream
    # with the right pairing but a wrong scale, or one that forgot the
    # triangular solve, passes (1) and fails this.
    n_big = 20_000
    big = np.asarray(laplace_draws(xi_hat, L, n_big, jax.random.PRNGKey(1)))
    w_big = (big - xh[None, :]) @ Ln
    M = xh.size
    cov_w = (w_big.T @ w_big) / n_big
    # sd of an empirical variance from n samples is sqrt(2/n) ~ 0.010 here;
    # 5 sigma with a seeded RNG.
    assert np.abs(np.diag(cov_w) - 1.0).max() < 0.05, np.diag(cov_w)
    off = cov_w - np.diag(np.diag(cov_w))
    assert np.abs(off).max() < 0.05, np.abs(off).max()

    # The same statement in the unwhitened basis: Cov(xi_m) ~ H^{-1}.
    H_inv = np.linalg.inv(Ln @ Ln.T)
    cov_x = np.cov(big.T, bias=True)
    scale = np.sqrt(np.outer(np.diag(H_inv), np.diag(H_inv)))
    assert np.abs((cov_x - H_inv) / scale).max() < 0.06

    # ... and it is really M-dimensional, not a collapsed rank-deficient cloud.
    assert np.linalg.matrix_rank(cov_x, tol=1e-8 * np.trace(cov_x) / M) == M

    with pytest.raises(ValueError):
        laplace_draws(xi_hat, L, 7, jax.random.PRNGKey(0))


def test_laplace_evidence_occam_term_present():
    op, _ = _fixture(seed=7)
    sol = count_map_solve(op)
    ev = float(laplace_evidence(op, sol["xi_hat"], sol["H_chol"]))
    l_only = float(shell_multinomial_logl(sol["xi_hat"], op))
    # the evidence is the count term MINUS a strictly positive Occam charge
    logdet = 2.0 * float(np.sum(np.log(np.diag(np.asarray(sol["H_chol"])))))
    assert logdet > 0.0  # H >= I
    assert ev < l_only


def test_counts_round_trip_histogram():
    rng = np.random.default_rng(8)
    n_pix, max_g = 6, 20
    zg = np.full((n_pix, max_g), 100.0)
    ng = rng.integers(0, max_g, size=n_pix)
    for p in range(n_pix):
        zg[p, :ng[p]] = rng.uniform(0.0, Z_HI, ng[p])
    edges = np.linspace(0.0, Z_HI, 7)
    got = counts_from_catalog(zg, ng, np.arange(n_pix), edges)
    for p in range(n_pix):
        ref, _ = np.histogram(zg[p, :ng[p]], bins=edges)
        assert np.array_equal(got[p], ref)
    assert got.sum() == ng.sum() - sum(
        (zg[p, :ng[p]] > Z_HI).sum() for p in range(n_pix))
