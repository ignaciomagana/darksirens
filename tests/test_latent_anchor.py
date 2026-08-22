"""PR-4 pins: the anchor artifact's moment tables and factorizations.

P9   ``rho`` reconstructed from the ``(A, B)`` moment tables — at the
     Chebyshev nodes exactly, and via ``b``-interpolation at 200 random
     ``(b, c)`` — vs the direct footprint reduction, 1e-6.  The interpolated
     arm goes through ``likelihood.latent_q.rho_from_moments``, the function
     the seam actually calls: a pin on a private copy of the interpolant is a
     pin on nothing.
P10  ``log|H| = 2 sum log diag(H_chol)`` vs ``slogdet``, 1e-8.
Plus the artifact-level draw identities PR-4's builder relies on.
"""
from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.likelihood.latent_q import rho_from_moments
from darksirens.redshift.latent_counts import (
    count_map_solve, make_count_operator, TracerCounts)
from darksirens.redshift.latent_field import (
    build_latent_basis,
    chebyshev_lobatto_nodes,
    shell_response,
    sky_constant_coeffs,
    sky_moments,
)

M_SPH, M_Z, N_FIT = 20, 4, 60
Z_HI = 0.3
#: The anchor's field LEVEL and pixel-to-pixel spread, back-solved from the two
#: numbers measured on ``latent_anchor_v2a.h5``: ``A(b = 0) = 2.97e4 = |F|`` and
#: ``A(b = 4) = 2.37e19``, i.e. ``A`` runs over 14.9 orders across the shipped
#: b-node range.  The field is therefore COHERENT across the footprint -- one
#: level, a small spread -- and it is that, not the per-mode draw amplitude,
#: that the ``b``-interpolant sees.  A raw ``0.4``-amplitude draw (what this pin
#: used to carry) puts ``A`` over 0.8 orders, twelve orders too flat to
#: discriminate anything; a raw ``2.46`` draw through this small basis gives a
#: WIDE zero-mean field whose range swings between 6 and 10 orders with the
#: seed, so neither the range nor the pin's verdict would be reproducible.
F_LEVEL_PROD, F_SPREAD_PROD = 7.57, 0.6


def _basis(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_out = np.linspace(1e-3, Z_HI, 25)
    return build_latent_basis(
        v, np.log1p(z_out), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.6, ls_z=0.15), rng


def _production_level_xi(rng, basis, n_draw=3):
    """Draws whose FIELD sits where the anchor's does: a constant mode at
    ``F_LEVEL_PROD`` plus a 2.46-amplitude draw rescaled to a pixel spread of
    ``F_SPREAD_PROD``.  The constant mode is the basis's own least-squares
    representation of ``f == 1`` (it reproduces it to 1.3%)."""
    proj = np.asarray(basis.proj_sph)
    phi_z = np.asarray(basis.phi_z_out)
    unit = np.outer(
        np.linalg.lstsq(proj, np.ones(proj.shape[0]), rcond=None)[0],
        np.linalg.lstsq(phi_z, np.ones(phi_z.shape[0]), rcond=None)[0]).ravel()
    xi = rng.normal(size=(n_draw, M_SPH * M_Z)) * 2.46
    fields = [(proj @ x.reshape(M_SPH, M_Z)) @ phi_z.T for x in xi]
    return xi * (F_SPREAD_PROD / max(f.std() for f in fields)) \
        + F_LEVEL_PROD * unit[None, :]


def test_p9_rho_from_moments_vs_direct():
    """P9 through the SHIPPED consumer, at the anchor's dynamic range.

    Both halves of this pin used to be inert.  It called
    ``latent_field.interp_moments_b`` -- a second, private copy of the
    barycentric interpolant that nothing in the likelihood imports, so the
    log-space fix to ``latent_q.rho_from_moments`` left it untouched and the pin
    stopped tracking the shipped path.  And it built its moments from a
    ``0.4``-amplitude draw, over which ``A`` spans 0.8 orders and every
    interpolant on earth is exact to 3.5e-15.

    Now it calls ``rho_from_moments`` itself, over a field at the anchor's own
    level, where ``A`` spans 14.9 orders.
    """
    basis, rng = _basis()
    f_p = rng.uniform(0.4, 1.0, size=N_FIT)
    b_nodes = chebyshev_lobatto_nodes(33, 4.0)
    xi_m = _production_level_xi(rng, basis)
    A, B = sky_moments(basis, xi_m, b_nodes, f_p)
    A, B = np.asarray(A), np.asarray(B)
    P_F, F_F = sky_constant_coeffs(f_p)
    proj = np.asarray(basis.proj_sph)
    phi_z = np.asarray(basis.phi_z_out)
    # 14.90 dex, against the anchor's 14.9.  Below ~13 the linear-space
    # interpolant stops going negative and this pin stops discriminating.
    assert np.log10(A[:, -1, :] / A[:, 0, :]).max() > 14.0, "range lost"

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
    # interpolated in b at 200 random (b, c): P9 tolerance 1e-6, measured 4.6e-14
    # in log space and 7.0e+02 (the 1e-300 floor) in linear.
    worst = 0.0
    for _ in range(200):
        m = int(rng.integers(0, 3))
        b = rng.uniform(0.0, 4.0)
        c = rng.uniform(0.0, 1.0)
        got = np.asarray(rho_from_moments(
            jnp.asarray(A[m]), jnp.asarray(B[m]), c, b, jnp.asarray(b_nodes),
            P_F, F_F))
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
