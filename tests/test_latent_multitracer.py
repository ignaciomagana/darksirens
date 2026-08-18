"""PR-7 pins: the K-tracer stack, and K = 1 bit-identical through it.

The ladder's standing invariant is that K = 1 does not move.  PR-7 adds a
stacked objective over K tracers sharing ONE ``xi``, and the way it is written
-- the single-tracer data terms extracted verbatim into ``_grad_data`` /
``_hess_data`` and summed -- is supposed to make the K = 1 case evaluate the
pre-PR-7 expression tree rather than an arithmetically equivalent one.  The
first block of tests is that claim, asserted with ``array_equal`` and not
``allclose``, on the objective, the gradient, the Hessian, the whole damped
solve (including its line-search trace) and the Laplace evidence.

The rest pins the K >= 2 content:

* the stacked gradient and Hessian against ``jax.grad`` / ``jax.hessian`` of
  the stacked objective, which is the K >= 2 twin of P4/P5;
* ``counts_from_catalog_by_tracer`` as a DISJOINT partition -- the K arrays sum
  to ``counts_from_catalog`` exactly (OWNER DECISION 9 discharged structurally),
  and ``check_disjoint_tracers`` refuses an overlapping set;
* ``dgrad_db_by_tracer`` column ``k`` against a finite difference of the
  stacked gradient in ``b_k``, and ``sensitivity`` on the K-column block
  against K separate one-column solves -- "K >= 2 adds columns, not code";
* the profile curvature's OFF-DIAGONAL, which is the shared field and is
  exactly zero in the decoupled model;
* the log-bias reparametrization identity;
* ``sky_moments_by_tracer`` against ``sky_moments`` per tracer;
* ``laplace_draws_multitracer``'s covariance ``H^-1 + V C V^T``.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift.latent_counts import (
    MultiTracerCountOperator,
    TracerCounts,
    bias_profile,
    bias_profile_curvature_log,
    bias_profile_grad,
    bias_profile_hessian,
    bias_ratio_from_profile,
    check_disjoint_tracers,
    count_map_solve,
    counts_from_catalog,
    counts_from_catalog_by_tracer,
    decoupled_bias_profile,
    decoupled_objective,
    dgrad_db,
    dgrad_db_by_tracer,
    gradient,
    hessian_separable,
    laplace_draws,
    laplace_draws_multitracer,
    laplace_evidence,
    log_bias_prior_potential,
    make_count_operator,
    make_multi_count_operator,
    multi_gradient,
    multi_hessian_separable,
    multi_objective,
    multi_shell_multinomial_logl,
    objective,
    sensitivity,
    shell_multinomial_logl,
    with_biases,
)
from darksirens.redshift.latent_field import (
    build_latent_basis,
    shell_response,
    sky_constant_coeffs,
    sky_moments,
    sky_moments_by_tracer,
)

M_SPH, M_Z, G_S, N_FIT = 10, 3, 4, 48
Z_HI = 0.3
M = M_SPH * M_Z


def _world(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_fine = np.linspace(1e-3, Z_HI, 100)
    basis = build_latent_basis(
        v, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.8, ls_z=0.2, zeta_fine=np.log1p(z_fine))
    edges = np.linspace(0.02, Z_HI, G_S + 1)
    W = shell_response(edges, z_fine, lambda z: 0.02 * np.ones_like(z),
                       lambda z: z ** 2 + 1e-9)
    return basis, W, rng


def _tracer(rng, bias, label, lam=40.0):
    return TracerCounts(pix=np.arange(N_FIT),
                        counts=rng.poisson(lam, size=(N_FIT, G_S)).astype(float),
                        completeness=rng.uniform(0.4, 1.0, size=N_FIT),
                        bias=bias, label=label)


def _stack(seed=0, biases=(1.3, 2.1)):
    basis, W, rng = _world(seed)
    tr = [_tracer(rng, b, f"t{k}") for k, b in enumerate(biases)]
    mop = make_multi_count_operator(basis.phi_sph, basis.phi_z_fine, W, tr)
    xi = rng.normal(size=M) * 0.4
    return basis, W, mop, tr, xi


# ---------------------------------------------------------- K = 1 identity

def test_k1_logl_objective_bit_identical():
    basis, W, rng = _world(1)
    t = _tracer(rng, 1.7, "solo")
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, t)
    mop = MultiTracerCountOperator((op,))
    xi = rng.normal(size=M) * 0.5
    assert float(multi_shell_multinomial_logl(xi, mop)) == float(
        shell_multinomial_logl(xi, op))
    assert float(multi_objective(xi, mop)) == float(objective(xi, op))


def test_k1_gradient_hessian_bit_identical():
    basis, W, rng = _world(2)
    t = _tracer(rng, 0.9, "solo")
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, t)
    mop = MultiTracerCountOperator((op,))
    xi = rng.normal(size=M) * 0.5
    assert np.array_equal(np.asarray(gradient(xi, op)),
                          np.asarray(multi_gradient(xi, mop)))
    assert np.array_equal(np.asarray(hessian_separable(xi, op)),
                          np.asarray(multi_hessian_separable(xi, mop)))


def test_k1_solve_bit_identical_including_line_search():
    """The whole damped solve, trace included.

    ``alpha`` and ``n_backtrack`` are compared too: a stack that reproduced
    ``xi_hat`` through a different sequence of accepted steps would be a
    different solve that happened to land in the same place, and the ladder's
    invariant is the former, not the latter.
    """
    basis, W, rng = _world(3)
    t = _tracer(rng, 1.1, "solo")
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, t)
    mop = MultiTracerCountOperator((op,))
    a = count_map_solve(op)
    b = count_map_solve(mop)
    for key in ("xi_hat", "H_chol", "grad_inf", "J", "alpha", "n_backtrack"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert np.array_equal(
        np.asarray(laplace_evidence(op, a["xi_hat"], a["H_chol"])),
        np.asarray(laplace_evidence(mop, b["xi_hat"], b["H_chol"])))


def test_k1_sensitivity_column_bit_identical():
    """``dgrad_db_by_tracer`` at K = 1 is ``dgrad_db``, and ``sensitivity``
    needs no change at all -- the PR-7 claim about §3.4's interface."""
    basis, W, rng = _world(4)
    t = _tracer(rng, 1.4, "solo")
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, t)
    mop = MultiTracerCountOperator((op,))
    xi = rng.normal(size=M) * 0.3
    one = np.asarray(dgrad_db(xi, op)).reshape(-1)
    col = np.asarray(dgrad_db_by_tracer(xi, mop))
    assert col.shape == (M, 1)
    assert np.array_equal(one, col[:, 0])


# ------------------------------------------------------ K >= 2 correctness

def test_stacked_gradient_vs_autodiff():
    _, _, mop, _, xi = _stack(5)
    g = np.asarray(multi_gradient(xi, mop))
    g_ad = np.asarray(jax.grad(lambda x: multi_objective(x, mop))(jnp.asarray(xi)))
    assert np.max(np.abs(g - g_ad)) < 1e-9


def test_stacked_hessian_vs_autodiff():
    """P5's K >= 2 twin.  ``multi_hessian_separable`` is the exact Fisher
    Hessian of the LINEARIZED (1') objective, which for the canonical
    multinomial logit equals the observed Hessian, so autodiff of the stacked
    objective must reproduce it -- rank-1 subtraction and all."""
    _, _, mop, _, xi = _stack(6)
    H = np.asarray(multi_hessian_separable(xi, mop))
    H_ad = np.asarray(jax.hessian(lambda x: multi_objective(x, mop))(jnp.asarray(xi)))
    assert np.max(np.abs(H - H_ad)) / np.max(np.abs(H_ad)) < 1e-9


def test_stack_is_the_sum_and_the_ridge_is_counted_once():
    """``J_stack = sum_k J_k - (K-1) * 0.5||xi||^2``.

    The correction term is the whole difference between the shared model and
    the decoupled one evaluated at a common ``xi``: K ridges versus one.  If it
    were absent the stack would over-regularize by a factor K and the draws
    would be under-dispersed by the same amount.
    """
    _, _, mop, _, xi = _stack(7)
    per = sum(float(objective(xi, op)) for op in mop.ops)
    ridge = 0.5 * float(np.dot(xi, xi))
    assert abs(float(multi_objective(xi, mop))
               - (per - (mop.n_tracer - 1) * ridge)) < 1e-9


def test_stacked_solve_converges():
    _, _, mop, _, _ = _stack(8)
    sol = count_map_solve(mop)
    assert float(sol["grad_inf"]) < 1e-8
    H = np.asarray(multi_hessian_separable(sol["xi_hat"], mop))
    assert np.min(np.linalg.eigvalsh(H)) >= 1.0 - 1e-9   # H >= I at any K


def test_rank_mismatch_refused():
    basis, W, rng = _world(9)
    t = _tracer(rng, 1.0, "a")
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, t)
    small = build_latent_basis(
        np.asarray(basis.phi_sph)[:, :1] * 0 + np.eye(N_FIT, 3),
        np.log1p(np.linspace(1e-3, Z_HI, 100)), n_inducing_sphere=M_SPH - 2,
        n_inducing_z=M_Z, z_node_hi=Z_HI, ls_sph=0.8, ls_z=0.2,
        zeta_fine=np.log1p(np.linspace(1e-3, Z_HI, 100)))
    op2 = make_count_operator(small.phi_sph, small.phi_z_fine, W, t)
    with pytest.raises(ValueError, match="share one field"):
        MultiTracerCountOperator((op, op2))


# --------------------------------------------------- OWNER DECISION 9 pins

def test_by_tracer_counts_are_a_disjoint_partition():
    rng = np.random.default_rng(11)
    n_pix, n_max = 30, 25
    ng = rng.integers(0, n_max + 1, size=n_pix)
    zg = rng.uniform(0.0, 0.3, size=(n_pix, n_max))
    lab = rng.integers(0, 3, size=(n_pix, n_max))
    edges = np.linspace(0.0, 0.3, 5)
    pix = np.arange(n_pix)
    parent = counts_from_catalog(zg, ng, pix, edges)
    parts = counts_from_catalog_by_tracer(zg, ng, pix, edges, lab, 3)
    assert np.array_equal(sum(parts), parent)     # exact, not approximate
    info = check_disjoint_tracers(
        [TracerCounts(pix, c, np.ones(n_pix), 1.0, label=f"t{k}")
         for k, c in enumerate(parts)], parent_counts=parent)
    assert info["verified"] and info["max_excess"] <= 0.0


def test_check_disjoint_refuses_an_overlapping_subset():
    """R14's failure mode: an AGN sample that is a SUBSET of the galaxies."""
    rng = np.random.default_rng(12)
    parent = rng.poisson(20.0, size=(15, 4)).astype(float)
    sub = np.minimum(parent, 5.0)                 # a subset, not a block
    tr = [TracerCounts(np.arange(15), parent, np.ones(15), 1.0, label="gal"),
          TracerCounts(np.arange(15), sub, np.ones(15), 1.0, label="agn")]
    with pytest.raises(ValueError, match="over-count"):
        check_disjoint_tracers(tr, parent_counts=parent)


def test_check_disjoint_reports_unverified_without_a_parent():
    rng = np.random.default_rng(13)
    tr = [TracerCounts(np.arange(9), rng.poisson(3.0, (9, 2)).astype(float),
                       np.ones(9), 1.0, label=f"t{k}") for k in range(2)]
    assert check_disjoint_tracers(tr)["verified"] is False
    tr2 = [TracerCounts(np.arange(9), tr[0].counts, np.ones(9), 1.0,
                        label="same") for _ in range(2)]
    with pytest.raises(ValueError, match="distinct"):
        check_disjoint_tracers(tr2)


# ------------------------------------------- K >= 2 ADDS COLUMNS (PLAN 3.4)

def test_dgrad_db_by_tracer_vs_finite_difference():
    _, _, mop, _, xi = _stack(14)
    cols = np.asarray(dgrad_db_by_tracer(xi, mop))
    assert cols.shape == (M, mop.n_tracer)
    h = 1e-5
    for k in range(mop.n_tracer):
        b = np.array([op.bias for op in mop.ops])
        bp, bm = b.copy(), b.copy()
        bp[k] += h
        bm[k] -= h
        fd = (np.asarray(multi_gradient(xi, with_biases(mop, bp)))
              - np.asarray(multi_gradient(xi, with_biases(mop, bm)))) / (2 * h)
        assert np.max(np.abs(cols[:, k] - fd)) < 1e-6 * max(
            1.0, np.max(np.abs(fd)))


def test_sensitivity_block_equals_k_separate_solves():
    """``sensitivity`` is linear in the stacked block and never inspects it,
    so K columns from K operators are admissible in the SHIPPED signature.

    Compared at 1e-12 relative and not bit-for-bit, deliberately and for a
    measured reason: ``cho_solve`` on a ``(M, 2)`` right-hand side and on two
    ``(M, 1)`` ones are the same linear algebra but not the same BLAS call, and
    the blocked triangular solve reorders the accumulation.  Measured on this
    fixture: ``6.66e-16`` and ``1.61e-15`` absolute on columns whose largest
    entries are ``0.163`` and ``0.218``, i.e. ``4.1e-15`` and ``7.4e-15``
    relative — a few ulp.  That is a statement about the library, not about the
    stacking, and pinning it at bit level would pin the library.  The tolerance
    is set three orders above the measurement.  The K = 1 identity, which IS
    bit-level
    (``test_k1_sensitivity_column_bit_identical``), is unaffected: there the
    block has one column either way.
    """
    _, _, mop, _, _ = _stack(15)
    sol = count_map_solve(mop)
    dg = dgrad_db_by_tracer(sol["xi_hat"], mop)
    S = np.asarray(sensitivity(sol["xi_hat"], sol["H_chol"], dg))
    for k in range(mop.n_tracer):
        one = np.asarray(sensitivity(sol["xi_hat"], sol["H_chol"],
                                     dg[:, k:k + 1]))[:, 0]
        assert np.max(np.abs(S[:, k] - one)) < 1e-12 * np.max(np.abs(one))


# ------------------------------------------------------- the coupling itself

def test_profile_curvature_off_diagonal_is_the_shared_field():
    """The (K, K) profile curvature's off-diagonal is
    ``(dgrad/db_j)^T H^-1 (dgrad/db_k)`` and NOTHING else: tracer j's
    likelihood does not contain ``b_k``, so ``J_bb`` is block diagonal."""
    _, _, mop, _, _ = _stack(16)
    sol = count_map_solve(mop)
    xi = sol["xi_hat"]
    Hb = np.asarray(bias_profile_hessian(xi, mop, H_chol=sol["H_chol"]))
    dg = dgrad_db_by_tracer(xi, mop)
    V = np.asarray(sensitivity(xi, sol["H_chol"], dg))
    coupling = np.asarray(dg).T @ V
    assert abs(Hb[0, 1] - coupling[0, 1]) < 1e-8 * abs(coupling[0, 1])
    assert abs(Hb[0, 1]) > 0.0


def test_decoupled_objective_separates_and_kills_the_coupling():
    """The decoupled model is a sum of K independent problems, so its bias
    covariance is block diagonal by construction -- the independent-fields
    product prior of ``inference/loaders.py:352-395``."""
    _, _, mop, _, xi = _stack(17)
    xs = jnp.stack([jnp.asarray(xi), jnp.asarray(xi) * 0.7])
    total = sum(float(objective(np.asarray(xs[k]), op))
                for k, op in enumerate(mop.ops))
    assert abs(float(decoupled_objective(xs, mop)) - total) < 1e-9
    de = decoupled_bias_profile(mop, n_outer=3, log_b_prior=(0.0, 1.0))
    C = np.asarray(de["cov_log_b"])
    assert C[0, 1] == 0.0 and C[1, 0] == 0.0


def test_log_bias_curvature_reparametrization_identity():
    """``d2P/du^2 = diag(b) H_b diag(b) + diag(b dP/db)`` — checked against a
    direct autodiff of ``P`` in ``u`` at fixed ``xi``, which is the only part
    of ``P`` the transform can be wrong about."""
    _, _, mop, _, xi = _stack(18)
    prior = (0.0, 0.8)
    b = jnp.asarray([op.bias for op in mop.ops])
    g_b = bias_profile_grad(xi, mop, log_b_prior=prior)
    H_b_fixed = jax.hessian(
        lambda bb: multi_objective(xi, with_biases(mop, bb))
        + log_bias_prior_potential(bb, prior))(b)
    got = np.asarray(bias_profile_curvature_log(g_b, H_b_fixed, b))
    want = np.asarray(jax.hessian(
        lambda u: multi_objective(xi, with_biases(mop, jnp.exp(u)))
        + log_bias_prior_potential(jnp.exp(u), prior))(jnp.log(b)))
    assert np.max(np.abs(got - want)) < 1e-6 * np.max(np.abs(want))


def test_bias_ratio_variance_is_the_exact_log_contraction():
    C = np.array([[4.0, 3.6], [3.6, 9.0]])
    out = bias_ratio_from_profile(np.array([1.0, 2.0]), C)
    assert abs(out["sigma_log"] ** 2 - (4.0 + 9.0 - 2 * 3.6)) < 1e-12
    assert abs(out["ratio"] - 2.0) < 1e-12
    assert abs(out["sigma"] - 2.0 * out["sigma_log"]) < 1e-12


# ----------------------------------------------- per-tracer moment tables

def test_sky_moments_by_tracer_matches_per_tracer_sky_moments():
    basis, W, rng = _world(19)
    xi_m = rng.normal(size=(3, M))
    b_nodes = np.linspace(0.0, 2.0, 5)
    f1 = rng.uniform(0.3, 1.0, N_FIT)
    f2 = rng.uniform(0.2, 0.9, N_FIT)
    A, B, P, F = sky_moments_by_tracer(basis, xi_m, b_nodes, [f1, f2])
    for k, f in enumerate((f1, f2)):
        a, b = sky_moments(basis, xi_m, b_nodes, f)
        assert np.array_equal(np.asarray(A[k]), np.asarray(a))
        assert np.array_equal(np.asarray(B[k]), np.asarray(b))
        pf, ff = sky_constant_coeffs(f)
        assert P[k] == pf and F[k] == ff
    # The tables differ BETWEEN tracers: that is the whole point of storing K
    # of them (the Z_k non-cancellation, PLAN §2.2).
    assert not np.allclose(np.asarray(B[0]), np.asarray(B[1]))


# ------------------------------------------------- rank-K draw covariance

def test_laplace_draws_multitracer_covariance_and_antithetic():
    rng = np.random.default_rng(20)
    m = 6
    Araw = rng.normal(size=(m, m))
    H = Araw @ Araw.T + m * np.eye(m)
    L = np.linalg.cholesky(H)
    V = rng.normal(size=(m, 2))
    Craw = rng.normal(size=(2, 2))
    C = Craw @ Craw.T + 0.5 * np.eye(2)
    xi_hat = rng.normal(size=m)
    n = 40000
    d, g, eps = laplace_draws_multitracer(
        jnp.asarray(xi_hat), jnp.asarray(L), n, jax.random.PRNGKey(0),
        cov_b=jnp.asarray(C), V_b=jnp.asarray(V), return_g=True,
        return_eps=True)
    d = np.asarray(d)
    # antithetic in BOTH streams, exactly
    assert np.array_equal(np.asarray(g)[: n // 2], -np.asarray(g)[n // 2:])
    assert np.array_equal(np.asarray(eps)[: n // 2], -np.asarray(eps)[n // 2:])
    assert np.max(np.abs(d.mean(axis=0) - xi_hat)) < 1e-10   # exact by pairing
    want = np.linalg.inv(H) + V @ C @ V.T
    got = np.cov(d.T, bias=True)
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 0.05


def test_laplace_draws_multitracer_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match=r"V_b must be"):
        laplace_draws_multitracer(
            jnp.zeros(5), jnp.eye(5), 4, jax.random.PRNGKey(0),
            cov_b=jnp.eye(2), V_b=jnp.zeros((5, 3)))


# ------------------------------------------------------ end-to-end recovery

def test_bias_ratio_recovered_on_a_drawn_two_tracer_world():
    """A small end-to-end: draw counts from the model at ``b = (1, 2)`` and
    recover the ratio.  The full Tier E campaign lives in
    ``experiments/field_level_plan/pr7``; this is the pin that the machinery
    the campaign uses is wired correctly at all."""
    basis, W, rng = _world(21)
    phi_shell = np.asarray(jnp.asarray(W) @ basis.phi_z_fine)
    xi_true = rng.normal(size=M)
    f_true = (np.asarray(basis.proj_sph) @ xi_true.reshape(M_SPH, M_Z)) \
        @ phi_shell.T
    fps = [rng.uniform(0.5, 1.0, N_FIT), rng.uniform(0.4, 0.9, N_FIT)]
    tr = []
    for k, b in enumerate((1.0, 2.0)):
        a = np.log(fps[k])[:, None] + b * f_true
        p = np.exp(a - a.max(0, keepdims=True))
        p /= p.sum(0, keepdims=True)
        c = np.stack([rng.multinomial(40000, p[:, g]) for g in range(G_S)],
                     axis=1).astype(float)
        tr.append(TracerCounts(np.arange(N_FIT), c, fps[k], 1.0, label=f"t{k}"))
    mop = make_multi_count_operator(basis.phi_sph, basis.phi_z_fine, W, tr)
    sh = bias_profile(mop, log_b_prior=(0.0, 1.0), n_outer=12)
    out = bias_ratio_from_profile(sh["b_hat"], sh["cov_log_b"])
    assert abs(out["ratio"] - 2.0) < 3.0 * out["sigma"]
    assert out["corr"] > 0.9                      # the shared field
    de = decoupled_bias_profile(mop, log_b_prior=(0.0, 1.0), n_outer=12)
    out_de = bias_ratio_from_profile(de["b_hat"], de["cov_log_b"])
    assert out_de["sigma"] > 5.0 * out["sigma"]   # the coupling, thrown away
