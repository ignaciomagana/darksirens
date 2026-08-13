"""Factored (sphere x z) latent-field basis for the field-level likelihood.

One construction, used offline (the anchor builder, PR-4) and online (the
latent seam, PR-5+), so the migration pins of ``experiments/field_level_plan``
compare like against like.  The field is

    f(x) = (Phi_s (x) Phi_z) @ xi,     xi ~ N(0, I_M),  M = M_sph * M_z

with ``Phi_s = k_sph(X_n, Z_sph) L_sph^{-T}`` and ``Phi_z = k_z(zeta, Z_z)
L_z^{-T}`` — the Kronecker factorization of the whitened finite-rank GP of
:mod:`darksirens.sky.models` (chordal-RBF on n-hat x RBF on zeta = log1p z).
The kernel and the inducing-node geometry are REUSED UNMODIFIED
(:func:`darksirens.sky.models._sphere_z_kernel`,
:func:`darksirens.redshift.lognormal_completion.lowrank_inducing_nodes`), so
the node-for-node identity pinned by ``tests/test_lss_completion_gp3d.py``
holds by construction.

Jitter conventions (PLAN §3.3, OWNER DECISION 2):

* ``"factored-v1"`` — the named latent-mode convention:
  ``L_sph = chol(K_sph + j_sph I)``, ``L_z = chol(K_z + j_z I)`` with
  ``j_sph = j_z = 1e-6`` (absolute, amp-independent; ``amp == 1`` by PLAN
  §4.3).  Because the jitter is applied PER FACTOR, the joint kernel is
  exactly ``(K_sph + j I) (x) (K_z + j I)`` and its Cholesky factor is
  exactly ``L_sph (x) L_z`` — the Kronecker identity is exact (pin P1),
  never an approximation.
* ``"legacy"`` — delegates to
  :func:`darksirens.redshift.lognormal_completion.build_lowrank_operator`
  (joint kernel + ``jitter_rel * amp**2 + jitter_abs`` on the flattened
  nodes), byte-identical to the shipped gp3d/joint builders.  The
  factored-vs-legacy basis delta (~2.0e-3 at ``M_sph = 315``) is a REPORTED
  diagnostic, not a gate (pin P3).

The shell response ``W`` (PLAN §1.4) convolves the MODEL, not the data:
``W[g, n] = (int_{shell g} N(z; z_n, sigma_z(z_n)) dz) * base(z_n) * dz_n``,
rows normalized to 1, so integer shell counts stay multinomial-exact while
photo-z attenuation lives in the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from darksirens.sky.models import _sphere_z_kernel, _fibonacci_sphere
from darksirens.redshift.lognormal_completion import (
    lowrank_inducing_nodes,
    build_lowrank_operator,
)

#: PLAN §3.3 / OWNER DECISION 2 — absolute per-factor jitter of the named
#: ``factored-v1`` convention.  Chosen so ``cond(K + jI) < 1e8`` in f64 at
#: ``M_sph = 315`` (measured ``cond = 4.3e4`` at legacy jitter).
JITTER_FACTORED_V1: float = 1e-6

_VALID_JITTER_MODES = ("factored-v1", "legacy")


def _chol_factor(K: jnp.ndarray, jitter: float) -> jnp.ndarray:
    return jnp.linalg.cholesky(K + jitter * jnp.eye(K.shape[0], dtype=K.dtype))


def _whiten_rows(k_xz: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    """``k(X, Z) @ L^{-T}`` via one triangular solve (rows of a factor basis)."""
    return jsl.solve_triangular(L, k_xz.T, lower=True).T


def _factor_kernels(Z_sph, zeta_nodes, *, amp, ls_sph, ls_z):
    """The two kernel factors at the inducing nodes.

    ``amp`` multiplies the SPHERE factor only, so the product of the two
    factors equals :func:`_sphere_z_kernel`'s ``amp**2 k_sph k_z`` exactly.
    """
    cos = jnp.clip(Z_sph @ Z_sph.T, -1.0, 1.0)
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    K_sph = (amp ** 2) * jnp.exp(-0.5 * d2 / ls_sph ** 2)
    dz = zeta_nodes[:, None] - zeta_nodes[None, :]
    K_z = jnp.exp(-0.5 * dz ** 2 / ls_z ** 2)
    return K_sph, K_z


@dataclass(frozen=True)
class LatentBasis:
    """The factored basis plus the frozen shell response.

    All arrays are f64 JAX arrays.  ``phi_sph`` rows are the caller's sky
    rows (typically all ``n_pix`` HEALPix centers); ``proj_sph`` rows are the
    FITTED-FOOTPRINT subset ``F`` used by the count channel and the sky
    moments (PLAN eq. 2 sums run over ``F``, never the full sky).
    """
    phi_sph: Any          # (N_rows, M_sph)
    phi_z_out: Any        # (N_z_out, M_z)   consumption-grid factor rows
    phi_z_fine: Any       # (N_fine, M_z)    fine-grid factor rows (W's grid)
    shell_response: Any   # (G_s, N_fine)    W, rows normalized to 1, or None
    proj_sph: Any         # (n_fit, M_sph)   footprint rows of phi_sph
    L_sph: Any            # (M_sph, M_sph)   lower Cholesky of K_sph + j I
    L_z: Any              # (M_z, M_z)       lower Cholesky of K_z + j I
    Zn: Any               # (M, 3)   flattened inducing nodes (sphere)
    Zz: Any               # (M,)     flattened inducing nodes (zeta)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def m_sph(self) -> int:
        return int(self.phi_sph.shape[1])

    @property
    def m_z(self) -> int:
        return int(self.phi_z_out.shape[1])

    @property
    def rank(self) -> int:
        return self.m_sph * self.m_z


def build_latent_basis(
    X_n,
    zeta_out,
    *,
    n_inducing_sphere: int,
    n_inducing_z: int,
    z_node_hi: float,
    amp: float = 1.0,
    ls_sph: float,
    ls_z: float,
    jitter_mode: str = "factored-v1",
    zeta_fine=None,
    shell_edges_z=None,
    sigma_z_fn: Callable | None = None,
    base_fn: Callable | None = None,
    footprint_rows=None,
) -> LatentBasis:
    """Build the factored basis (and optionally ``W``) at the given rows.

    Parameters
    ----------
    X_n : (N_rows, 3) unit vectors of the sky rows (e.g. HEALPix centers).
    zeta_out : (N_z_out,) ``log1p(z)`` of the consumption grid.
    n_inducing_sphere, n_inducing_z, z_node_hi :
        Node geometry, EXACTLY :func:`lowrank_inducing_nodes` /
        ``_SphereZGPBase``.
    jitter_mode : "factored-v1" (latent mode) — per-factor absolute jitter
        ``1e-6``.  ("legacy" basis construction lives in
        :func:`legacy_lowrank_operator`; this builder is factored-only.)
    zeta_fine : (N_fine,) optional fine grid for the shell response.
    shell_edges_z, sigma_z_fn, base_fn :
        If all given (with ``zeta_fine``), the shell response ``W`` is built
        via :func:`shell_response` and stored.
    footprint_rows : optional integer index array selecting the fitted
        footprint ``F`` inside ``X_n``; ``proj_sph`` is that row subset
        (defaults to all rows).
    """
    if jitter_mode != "factored-v1":
        raise ValueError(
            f"build_latent_basis builds the factored basis; jitter_mode must "
            f"be 'factored-v1' (got {jitter_mode!r}). The legacy joint-kernel "
            f"path is legacy_lowrank_operator().")
    M_sph = int(n_inducing_sphere)
    M_z = int(n_inducing_z)
    Zn, Zz = lowrank_inducing_nodes(M_sph, M_z, z_node_hi)
    Z_sph = _fibonacci_sphere(M_sph)                        # (M_sph, 3)
    zeta_nodes = jnp.linspace(0.0, float(jnp.log1p(z_node_hi)), M_z)

    K_sph, K_z = _factor_kernels(Z_sph, zeta_nodes, amp=amp,
                                 ls_sph=ls_sph, ls_z=ls_z)
    L_sph = _chol_factor(K_sph, JITTER_FACTORED_V1)
    L_z = _chol_factor(K_z, JITTER_FACTORED_V1)

    X_n = jnp.asarray(X_n, dtype=jnp.float64)
    zeta_out = jnp.asarray(zeta_out, dtype=jnp.float64)

    cos = jnp.clip(X_n @ Z_sph.T, -1.0, 1.0)
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    k_sph = (amp ** 2) * jnp.exp(-0.5 * d2 / ls_sph ** 2)   # (N_rows, M_sph)
    phi_sph = _whiten_rows(k_sph, L_sph)

    def _phi_z(zeta):
        dz = zeta[:, None] - zeta_nodes[None, :]
        return _whiten_rows(jnp.exp(-0.5 * dz ** 2 / ls_z ** 2), L_z)

    phi_z_out = _phi_z(zeta_out)
    phi_z_fine = _phi_z(jnp.asarray(zeta_fine, dtype=jnp.float64)) \
        if zeta_fine is not None else None

    W = None
    if (shell_edges_z is not None and zeta_fine is not None
            and sigma_z_fn is not None and base_fn is not None):
        W = shell_response(np.asarray(shell_edges_z, dtype=float),
                           np.expm1(np.asarray(zeta_fine, dtype=float)),
                           sigma_z_fn, base_fn)

    proj = phi_sph if footprint_rows is None \
        else phi_sph[jnp.asarray(footprint_rows)]
    meta = dict(jitter_mode="factored-v1",
                j_sph=JITTER_FACTORED_V1, j_z=JITTER_FACTORED_V1,
                amp=float(amp), ls_sph=float(ls_sph), ls_z=float(ls_z),
                M_sph=M_sph, M_z=M_z, z_node_hi=float(z_node_hi))
    return LatentBasis(phi_sph=phi_sph, phi_z_out=phi_z_out,
                       phi_z_fine=phi_z_fine, shell_response=W,
                       proj_sph=proj, L_sph=L_sph, L_z=L_z,
                       Zn=Zn, Zz=Zz, meta=meta)


def legacy_lowrank_operator(Zn, Zz, X_n, X_z, *, amp, ls_sph, ls_z,
                            jitter_rel: float = 1e-4,
                            jitter_abs: float = 1e-9):
    """The legacy joint-kernel basis — BYTE-IDENTICAL to the shipped builders.

    Thin delegation to
    :func:`darksirens.redshift.lognormal_completion.build_lowrank_operator`
    so the gp3d and joint builders can route through this module without any
    numerical change (their outputs are pinned byte-identical by the existing
    gp3d test suite).  The factored-v1 basis differs from this one at the
    ~2.0e-3 level at ``M_sph = 315`` (PLAN §3.3) — that delta is reported by
    the PR-1 diagnostics, never gated.
    """
    return build_lowrank_operator(Zn, Zz, X_n, X_z, amp=amp,
                                  ls_sph=ls_sph, ls_z=ls_z,
                                  jitter_rel=jitter_rel,
                                  jitter_abs=jitter_abs)


# ---------------------------------------------------------------- evaluation

def row_factor(basis: LatentBasis, xi) -> jnp.ndarray:
    """``row_fac = Phi_s @ reshape(xi, (M_sph, M_z))`` — the ``(N_rows, M_z)``
    member row factor of the seam (PLAN §3.5).  Flattened-node ordering is
    ``i = i_sph * M_z + i_z`` (:func:`lowrank_inducing_nodes`), so the
    reshape is exactly ``(M_sph, M_z)`` row-major."""
    Xi = jnp.reshape(jnp.asarray(xi), (basis.m_sph, basis.m_z))
    return basis.phi_sph @ Xi


def field_rows(basis: LatentBasis, xi, *, z_axis: str = "out") -> jnp.ndarray:
    """The field ``f(p, z) = (Phi_s (x) Phi_z) xi`` on the row x z grid,
    shape ``(N_rows, N_z)``.  Never materializes the Kronecker basis."""
    phi_z = basis.phi_z_out if z_axis == "out" else basis.phi_z_fine
    return row_factor(basis, xi) @ phi_z.T


def at_nodes(basis: LatentBasis, xi) -> jnp.ndarray:
    """The field at the flattened inducing nodes ``(Zn, Zz)`` — the
    diagnostic view; equals ``(L_sph (x) L_z)^T``-whitened kernel rows."""
    zeta_nodes = jnp.unique(basis.Zz, size=basis.m_z)
    Xi = jnp.reshape(jnp.asarray(xi), (basis.m_sph, basis.m_z))
    # phi at the nodes: k(Z,Z) L^{-T} per factor
    K_sph = basis.L_sph @ basis.L_sph.T - JITTER_FACTORED_V1 * jnp.eye(basis.m_sph)
    K_z = basis.L_z @ basis.L_z.T - JITTER_FACTORED_V1 * jnp.eye(basis.m_z)
    phi_s = _whiten_rows(K_sph, basis.L_sph)
    phi_z = _whiten_rows(K_z, basis.L_z)
    return (phi_s @ Xi @ phi_z.T).reshape(-1)


def prior_var_rows(basis: LatentBasis, *, z_axis: str = "out") -> jnp.ndarray:
    """Per-(row, z) Nystrom prior variance ``sum_i Phi[v, i]**2``.

    Exactly separable for the Kronecker basis:
    ``sum_{i,a} (phi_s[p,i] phi_z[g,a])**2
      = (sum_i phi_s[p,i]**2)(sum_a phi_z[g,a]**2)`` —
    the quantity the per-voxel lognormal mean-one shift consumes
    (``factored-v1`` only; pin P2)."""
    phi_z = basis.phi_z_out if z_axis == "out" else basis.phi_z_fine
    vs = jnp.sum(basis.phi_sph ** 2, axis=1)      # (N_rows,)
    vz = jnp.sum(phi_z ** 2, axis=1)              # (N_z,)
    return vs[:, None] * vz[None, :]


# ------------------------------------------------------------- shell response

def shell_response(shell_edges_z: np.ndarray, z_fine: np.ndarray,
                   sigma_z_fn: Callable, base_fn: Callable) -> jnp.ndarray:
    """The frozen shell response ``W`` (PLAN §1.4) — convolve the MODEL.

    ``W[g, n] = (int_{shell g} dz N(z; z_n, sigma_z(z_n))) * base(z_n) * dz_n``
    with rows normalized to 1.  ``sigma_z_fn(z)`` is the population-average
    photo-z kernel width (per-galaxy kernels are the P8-gated upgrade);
    ``base_fn(z)`` is ``base(z; theta_ref)`` — at rung 1 the theta-dependent
    base moves INSIDE the shell integral (PR-3) and this frozen ``W`` is the
    ``theta = theta_ref`` special case.
    """
    from scipy.special import erf

    edges = np.asarray(shell_edges_z, dtype=float)          # (G_s + 1,)
    zn = np.asarray(z_fine, dtype=float)                    # (N_fine,)
    dz_n = np.gradient(zn)
    sig = np.maximum(np.asarray(sigma_z_fn(zn), dtype=float), 1e-12)
    lo = (edges[:-1, None] - zn[None, :]) / (np.sqrt(2.0) * sig[None, :])
    hi = (edges[1:, None] - zn[None, :]) / (np.sqrt(2.0) * sig[None, :])
    mass = 0.5 * (erf(hi) - erf(lo))                        # (G_s, N_fine)
    W = mass * (np.asarray(base_fn(zn), dtype=float) * dz_n)[None, :]
    norm = W.sum(axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError(
            "shell_response: a shell received zero model mass — the fine "
            "grid does not cover the shells (extend z_fine).")
    return jnp.asarray(W / norm)


# ---------------------------------------------------------------- sky moments

def sky_constant_coeffs(f_p_footprint) -> tuple[float, float]:
    """``(P_F, F_F)`` of PLAN eq. (2): the footprint pixel count and the sum
    of per-pixel selection fractions ``f_p`` OVER THE FITTED FOOTPRINT."""
    f = jnp.asarray(f_p_footprint)
    return float(f.shape[0]), float(jnp.sum(f))


def sky_moments(basis: LatentBasis, xi_members, b_nodes,
                f_p_footprint) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The theta-free moment tables ``(A_m, B_m)`` of PLAN eq. (2).

    ``A_m(z; b) = sum_{p in F} e^{b f_m(p, z)}``,
    ``B_m(z; b) = sum_{p in F} f_p e^{b f_m(p, z)}``,
    shapes ``(M_draw, n_b, N_z_out)``.  The sums run over the FITTED
    FOOTPRINT ``F`` (``basis.proj_sph`` rows) — never the full sky — so the
    off-footprint block of the budget identity is conserved trivially and
    the seam's off-footprint rows return bit-zero ``logQ`` (pin P13b).

    Online, the budget normalizer is then closed-form in the scalar
    ``c = C(z; theta)``:
    ``rho_m(z; c, b) = log[(A_m - c B_m) / (P_F - c F_F)]``.
    """
    xi_members = jnp.atleast_2d(jnp.asarray(xi_members))    # (M_draw, M)
    b_nodes = jnp.asarray(b_nodes, dtype=jnp.float64)       # (n_b,)
    f_p = jnp.asarray(f_p_footprint, dtype=jnp.float64)     # (n_fit,)

    def _one(xi):
        Xi = jnp.reshape(xi, (basis.m_sph, basis.m_z))
        f = (basis.proj_sph @ Xi) @ basis.phi_z_out.T       # (n_fit, N_z)
        e = jnp.exp(b_nodes[:, None, None] * f[None, :, :]) # (n_b, n_fit, N_z)
        A = jnp.sum(e, axis=1)                              # (n_b, N_z)
        B = jnp.einsum("p,bpz->bz", f_p, e)                 # (n_b, N_z)
        return A, B

    A, B = jax.vmap(_one)(xi_members)
    return A, B


def rho_from_moments(A, B, c, b_index) -> jnp.ndarray:
    """Closed-form budget normalizer ``rho_m(z; c)`` at one ``b`` node:
    ``log[(A - c B) / (P_F - c F_F)]`` with ``(P_F, F_F)`` folded in by the
    caller via ``A, B`` normalization or passed explicitly at the seam
    (PR-5); here the bare ``log(A - c B)`` building block."""
    return jnp.log(A[..., b_index, :] - c * B[..., b_index, :])
