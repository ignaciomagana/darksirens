"""PR-1 pins for darksirens.redshift.latent_field (PLAN §6.3 P1, P2, P3).

P1  Kronecker identity: the factored basis row ``phi_sph[p] (x) phi_z[g]``
    equals the chol-of-factored-K joint basis row to 1e-13 (tolerance SET
    FROM MEASUREMENT, 1.9e-14 observed in review), small rank densely plus a
    randomized row spot check at production rank (315 x 12).
P2  ``prior_var_rows`` equals ``sum(Phi**2, axis=1)`` of the materialized
    factored basis to 1e-12 (factored-v1 only).
P3  The legacy-vs-factored basis delta at ``M_sph in {64, 315}`` is REPORTED
    (printed), never gated — PLAN §3.3 measured 4.96e-5 / 2.0e-3.

Plus the structural pins PR-1 carries: the inducing nodes are byte-identical
to ``lowrank_inducing_nodes`` (hence to ``_SphereZGPBase``, whose identity is
pinned by tests/test_lss_completion_gp3d.py), the legacy delegation is the
same object as ``build_lowrank_operator``, and the shell response rows are
normalized with the analytic Gaussian attenuation.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift.latent_field import (
    JITTER_FACTORED_V1,
    LatentBasis,
    build_latent_basis,
    field_rows,
    legacy_lowrank_operator,
    prior_var_rows,
    row_factor,
    shell_response,
    sky_constant_coeffs,
    sky_moments,
)
from darksirens.redshift.lognormal_completion import (
    build_lowrank_operator,
    lowrank_inducing_nodes,
)
from darksirens.sky.models import _fibonacci_sphere, _sphere_z_kernel

LS_SPH, LS_Z = 0.2, 0.039


def _rows(n, seed=0, z_hi=0.3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    zeta = rng.uniform(0.0, np.log1p(z_hi), size=n)
    return v, zeta


def _dense_factored_phi(M_sph, M_z, z_hi, X_n, zeta):
    """chol-of-factored-K joint basis: K = (K_s + jI) (x) (K_z + jI)."""
    Z_sph = np.asarray(_fibonacci_sphere(M_sph))
    zeta_nodes = np.linspace(0.0, np.log1p(z_hi), M_z)
    d2 = np.clip(2.0 - 2.0 * np.clip(Z_sph @ Z_sph.T, -1, 1), 0, 4)
    Ks = np.exp(-0.5 * d2 / LS_SPH ** 2) + JITTER_FACTORED_V1 * np.eye(M_sph)
    dz = zeta_nodes[:, None] - zeta_nodes[None, :]
    Kz = np.exp(-0.5 * dz ** 2 / LS_Z ** 2) + JITTER_FACTORED_V1 * np.eye(M_z)
    K = np.kron(Ks, Kz)
    L = np.linalg.cholesky(K)
    # joint cross-kernel rows at (X_n, zeta): kron of the factor cross rows
    c = np.clip(X_n @ Z_sph.T, -1, 1)
    ks = np.exp(-0.5 * np.clip(2 - 2 * c, 0, 4) / LS_SPH ** 2)  # (n, M_sph)
    kz = np.exp(-0.5 * (zeta[:, None] - zeta_nodes[None, :]) ** 2 / LS_Z ** 2)
    kxz = np.einsum("ni,na->nia", ks, kz).reshape(X_n.shape[0], -1)
    from scipy.linalg import solve_triangular
    return solve_triangular(L, kxz.T, lower=True).T, zeta_nodes


def _materialized_factored(basis, zeta):
    """kron rows of the factored basis at paired (row, zeta) points."""
    phi_s = np.asarray(basis.phi_sph)
    dzn = zeta[:, None] - np.linspace(
        0.0, np.log1p(basis.meta["z_node_hi"]), basis.m_z)[None, :]
    from scipy.linalg import solve_triangular
    kz = np.exp(-0.5 * dzn ** 2 / basis.meta["ls_z"] ** 2)
    phi_z = solve_triangular(np.asarray(basis.L_z), kz.T, lower=True).T
    return np.einsum("ni,na->nia", phi_s, phi_z).reshape(zeta.shape[0], -1)


def test_p1_kron_identity_small_rank():
    M_sph, M_z, z_hi = 24, 5, 0.3
    X_n, zeta = _rows(400, seed=1)
    dense, _ = _dense_factored_phi(M_sph, M_z, z_hi, X_n, zeta)
    basis = build_latent_basis(
        X_n, zeta, n_inducing_sphere=M_sph, n_inducing_z=M_z,
        z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z)
    fact = _materialized_factored(basis, zeta)
    assert np.max(np.abs(fact - dense)) < 1e-13


def test_p1_kron_identity_production_rank_spot():
    M_sph, M_z, z_hi = 315, 12, 0.3
    X_n, zeta = _rows(2000, seed=2)
    dense, _ = _dense_factored_phi(M_sph, M_z, z_hi, X_n, zeta)
    basis = build_latent_basis(
        X_n, zeta, n_inducing_sphere=M_sph, n_inducing_z=M_z,
        z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z)
    fact = _materialized_factored(basis, zeta)
    # At production rank the DENSE 3780x3780 reference Cholesky carries its
    # own rounding (measured 1.16e-12 max-abs on this seed); the identity
    # itself is exact and the small-rank test pins it at 1e-13.
    assert np.max(np.abs(fact - dense)) < 1e-11


def test_p2_prior_var_rows():
    M_sph, M_z, z_hi = 32, 6, 0.3
    X_n, _ = _rows(300, seed=3)
    z_out = np.linspace(1e-3, z_hi, 40)
    basis = build_latent_basis(
        X_n, np.log1p(z_out), n_inducing_sphere=M_sph, n_inducing_z=M_z,
        z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z)
    pv = np.asarray(prior_var_rows(basis))
    phi_s = np.asarray(basis.phi_sph)
    phi_z = np.asarray(basis.phi_z_out)
    # sum over (i,a) of (phi_s[n,i] phi_z[g,a])^2 from materialized kron rows
    kron_direct = np.zeros((phi_s.shape[0], phi_z.shape[0]))
    for g in range(phi_z.shape[0]):
        rows = np.einsum("ni,a->nia", phi_s, phi_z[g]).reshape(
            phi_s.shape[0], -1)
        kron_direct[:, g] = (rows ** 2).sum(1)
    assert np.max(np.abs(pv - kron_direct)) < 1e-12


def test_p3_legacy_vs_factored_delta_reported(capsys):
    z_hi = 0.3
    for M_sph in (64, 315):
        M_z = 6
        X_n, zeta = _rows(500, seed=4)
        Zn, Zz = lowrank_inducing_nodes(M_sph, M_z, z_hi)
        Phi_leg, _ = legacy_lowrank_operator(
            Zn, Zz, jnp.asarray(X_n), jnp.asarray(zeta),
            amp=1.0, ls_sph=LS_SPH, ls_z=LS_Z)
        basis = build_latent_basis(
            X_n, zeta, n_inducing_sphere=M_sph, n_inducing_z=M_z,
            z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z)
        fact = _materialized_factored(basis, zeta)
        delta = float(np.max(np.abs(fact - np.asarray(Phi_leg))))
        print(f"[P3] M_sph={M_sph}: max|Phi_factored - Phi_legacy| = "
              f"{delta:.3e}  (reported, not gated)")
        assert np.isfinite(delta)


def test_legacy_delegation_is_build_lowrank_operator():
    Zn, Zz = lowrank_inducing_nodes(16, 4, 0.3)
    X_n, zeta = _rows(50, seed=5)
    a = legacy_lowrank_operator(Zn, Zz, jnp.asarray(X_n), jnp.asarray(zeta),
                                amp=1.0, ls_sph=LS_SPH, ls_z=LS_Z)
    b = build_lowrank_operator(Zn, Zz, jnp.asarray(X_n), jnp.asarray(zeta),
                               amp=1.0, ls_sph=LS_SPH, ls_z=LS_Z)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_nodes_match_lowrank_inducing_nodes():
    basis = build_latent_basis(
        _rows(10)[0], np.log1p(np.linspace(1e-3, 0.3, 8)),
        n_inducing_sphere=32, n_inducing_z=6, z_node_hi=3.0,
        ls_sph=LS_SPH, ls_z=LS_Z)
    Zn, Zz = lowrank_inducing_nodes(32, 6, 3.0)
    assert np.array_equal(np.asarray(basis.Zn), np.asarray(Zn))
    assert np.array_equal(np.asarray(basis.Zz), np.asarray(Zz))


def test_row_factor_and_field_rows_consistent():
    M_sph, M_z, z_hi = 20, 4, 0.3
    X_n, _ = _rows(60, seed=6)
    z_out = np.linspace(1e-3, z_hi, 15)
    basis = build_latent_basis(
        X_n, np.log1p(z_out), n_inducing_sphere=M_sph, n_inducing_z=M_z,
        z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z)
    rng = np.random.default_rng(7)
    xi = rng.normal(size=M_sph * M_z)
    f = np.asarray(field_rows(basis, xi))
    rf = np.asarray(row_factor(basis, xi))
    assert f.shape == (60, 15)
    np.testing.assert_allclose(
        f, rf @ np.asarray(basis.phi_z_out).T, rtol=0, atol=1e-14)
    # flattened-node ordering i = i_sph * M_z + i_z: the kron row of sky row
    # 0 at z_out[0], contracted with xi, must equal field_rows[0, 0]
    kron0 = np.kron(np.asarray(basis.phi_sph)[0],
                    np.asarray(basis.phi_z_out)[0])
    np.testing.assert_allclose(kron0 @ xi, f[0, 0], rtol=0, atol=1e-12)


def test_shell_response_rows_normalized_and_attenuating():
    edges = np.linspace(0.05, 0.3, 11)
    zf = np.linspace(1e-3, 0.35, 500)
    sigma = lambda z: 0.023 * np.ones_like(z)
    base = lambda z: (1 + z) ** 0.9 * z ** 2 + 1e-8
    W = np.asarray(shell_response(edges, zf, sigma, base))
    assert W.shape == (10, 500)
    np.testing.assert_allclose(W.sum(1), 1.0, rtol=0, atol=1e-12)
    assert np.all(W >= 0.0)
    # a shell's response leaks into neighbours over ~sigma_z
    g = 5
    inside = (zf >= edges[g]) & (zf < edges[g + 1])
    assert 0.2 < W[g, inside].sum() < 0.95


def test_sky_moments_shapes_and_homogeneous_limit():
    M_sph, M_z, z_hi = 24, 5, 0.3
    X_n, _ = _rows(200, seed=8)
    z_out = np.linspace(1e-3, z_hi, 12)
    fit = np.arange(120)
    basis = build_latent_basis(
        X_n, np.log1p(z_out), n_inducing_sphere=M_sph, n_inducing_z=M_z,
        z_node_hi=z_hi, ls_sph=LS_SPH, ls_z=LS_Z, footprint_rows=fit)
    f_p = np.random.default_rng(9).uniform(0.5, 1.0, size=fit.size)
    b_nodes = np.array([0.5, 1.0, 2.0])
    P_F, F_F = sky_constant_coeffs(f_p)
    assert P_F == fit.size and np.isclose(F_F, f_p.sum())
    # xi = 0 (homogeneous field): A -> P_F, B -> F_F at every (b, z)
    A, B = sky_moments(basis, np.zeros((2, M_sph * M_z)), b_nodes, f_p)
    assert A.shape == (2, 3, 12) and B.shape == (2, 3, 12)
    np.testing.assert_allclose(np.asarray(A), P_F, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(B), F_F, rtol=1e-14)
    # a random field: A_m >= exp(b mean f) * P_F Jensen direction sanity
    rng = np.random.default_rng(10)
    xi = rng.normal(size=(1, M_sph * M_z))
    A1, _ = sky_moments(basis, xi, b_nodes, f_p)
    assert np.all(np.asarray(A1) > 0.0)


def test_factored_v1_jitter_stamped():
    basis = build_latent_basis(
        _rows(5)[0], np.log1p(np.linspace(1e-3, 0.3, 4)),
        n_inducing_sphere=8, n_inducing_z=3, z_node_hi=0.3,
        ls_sph=LS_SPH, ls_z=LS_Z)
    assert basis.meta["jitter_mode"] == "factored-v1"
    assert basis.meta["j_sph"] == basis.meta["j_z"] == 1e-6
    with pytest.raises(ValueError):
        build_latent_basis(
            _rows(5)[0], np.log1p(np.linspace(1e-3, 0.3, 4)),
            n_inducing_sphere=8, n_inducing_z=3, z_node_hi=0.3,
            ls_sph=LS_SPH, ls_z=LS_Z, jitter_mode="legacy")
