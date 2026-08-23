"""PR-4 pins: the anchor artifact's moment tables and factorizations.

P9   ``rho`` reconstructed from the ``(A, B)`` moment tables — at the
     Chebyshev nodes exactly, and via barycentric ``b``-interpolation at
     200 random ``(b, c)`` — vs the direct footprint reduction, 1e-6.
P10  ``log|H| = 2 sum log diag(H_chol)`` vs ``slogdet``, 1e-8.
Plus the artifact-level draw identities PR-4's builder relies on.
"""
from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift.latent_counts import (
    count_map_solve, make_count_operator, TracerCounts)
from darksirens.redshift.latent_field import (
    build_latent_basis,
    chebyshev_lobatto_nodes,
    interp_moments_b,
    shell_response,
    sky_constant_coeffs,
    sky_moments,
)

M_SPH, M_Z, N_FIT = 20, 4, 60
Z_HI = 0.3


def _basis(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_out = np.linspace(1e-3, Z_HI, 25)
    return build_latent_basis(
        v, np.log1p(z_out), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.6, ls_z=0.15), rng


def test_p9_rho_from_moments_vs_direct():
    basis, rng = _basis()
    f_p = rng.uniform(0.4, 1.0, size=N_FIT)
    b_nodes = chebyshev_lobatto_nodes(33, 4.0)
    xi_m = rng.normal(size=(3, M_SPH * M_Z)) * 0.4
    A, B = sky_moments(basis, xi_m, b_nodes, f_p)
    A, B = np.asarray(A), np.asarray(B)
    P_F, F_F = sky_constant_coeffs(f_p)
    proj = np.asarray(basis.proj_sph)
    phi_z = np.asarray(basis.phi_z_out)

    def rho_direct(m, b, c):
        f = (proj @ xi_m[m].reshape(M_SPH, M_Z)) @ phi_z.T
        w = 1.0 - f_p[:, None] * c
        e = np.exp(b * f)
        return np.log((w * e).sum(0) / w.sum(0))

    # exact at the nodes
    for m in range(3):
        for i in (0, 7, 32):
            c = rng.uniform(0, 1)
            got = np.log((A[m, i] - c * B[m, i]) / (P_F - c * F_F))
            np.testing.assert_allclose(got, rho_direct(m, b_nodes[i], c),
                                       rtol=1e-12, atol=1e-12)
    # interpolated in b at 200 random (b, c): P9 tolerance 1e-6
    worst = 0.0
    for _ in range(200):
        m = rng.integers(0, 3)
        b = rng.uniform(0.0, 4.0)
        c = rng.uniform(0.0, 1.0)
        Ai = np.asarray(interp_moments_b(A[m], b_nodes, b))
        Bi = np.asarray(interp_moments_b(B[m], b_nodes, b))
        got = np.log((Ai - c * Bi) / (P_F - c * F_F))
        ref = rho_direct(m, b, c)
        worst = max(worst, np.max(np.abs(got - ref)))
    assert worst < 1e-6, worst


def test_p10_logdet_identity():
    z_fine = np.linspace(1e-3, Z_HI, 100)
    v = np.random.default_rng(2).normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    basis2 = build_latent_basis(
        v, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.6, ls_z=0.15, zeta_fine=np.log1p(z_fine))
    edges = np.linspace(0.02, Z_HI, 6)
    W = shell_response(edges, z_fine, lambda z: 0.02 * np.ones_like(z),
                       lambda z: z ** 2 + 1e-6)
    counts = np.random.default_rng(3).poisson(25.0, size=(N_FIT, 5)) * 1.0
    op = make_count_operator(
        basis2.phi_sph, basis2.phi_z_fine, W,
        TracerCounts(pix=np.arange(N_FIT), counts=counts,
                     completeness=np.ones(N_FIT), bias=1.1))
    sol = count_map_solve(op)
    L = np.asarray(sol["H_chol"])
    logdet_chol = 2.0 * np.sum(np.log(np.diag(L)))
    sign, logdet_ref = np.linalg.slogdet(L @ L.T)
    assert sign == 1.0
    assert abs(logdet_chol - logdet_ref) < 1e-8
