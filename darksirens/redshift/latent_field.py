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

import dataclasses
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

#: PR-8 amplitude profiles.  ``"step"`` is what the sensitivity table quotes
#: (one assumed number for the whole unconstrained region); ``"growth"`` is the
#: same number carried by linear growth ``D(z)/D(z_depth)``, which DECAYS with
#: z and is therefore the conservative shape at fixed ``amp_hi``.
_VALID_AMP_KINDS = ("step", "growth")


def growth_factor(z, Om0: float) -> np.ndarray:
    """Linear growth factor ``D(z)``, flat LCDM, normalized to ``D(0) = 1``.

    ``D(a) ∝ H(a) int_0^a da' / (a' H(a'))^3`` — the exact quadrature rather
    than a fitting formula, because it costs nothing offline and the profile it
    feeds is the whole content of PR-8's table.  numpy only: this is builder-
    and loader-side, never traced (PLAN §3.7).
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    E = lambda x: np.sqrt(Om0 * x ** -3 + (1.0 - Om0))      # noqa: E731
    # Integrate on a fixed fine grid in a and interpolate, so the profile is a
    # single deterministic function of (z, Om0) and not of the caller's grid.
    ag = np.linspace(1e-6, 1.0, 200001)
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (1.0 / (ag[1:] * E(ag[1:])) ** 3
               + 1.0 / (ag[:-1] * E(ag[:-1])) ** 3) * np.diff(ag))])
    D_un = E(ag) * integ
    D = np.interp(a, ag, D_un) / float(np.interp(1.0, ag, D_un))
    return D


def amp_profile(z, *, z_depth, amp_hi, kind: str = "step",
                Om0: float = 0.3075) -> np.ndarray:
    """The PR-8 amplitude profile ``amp(z)`` — ONE below the depth, assumed above.

    PLAN §4.3 is the constraint this function is written around: ``(b, xi)`` and
    ``(amp, xi)`` enter the model only through ``b*xi`` and ``b*amp``, so where
    the counts constrain the field there is exactly one clustering amplitude and
    it is ``b_gal``.  ``amp`` is therefore **pinned at 1 for every ``z <=
    z_depth``, bit-exactly** — the profile returns literal ``1.0`` there, and a
    literal ``1.0`` multiplying the basis rows is the identity in IEEE754, so
    the fitted region is untouched rather than merely unchanged to rounding.
    Anything else would re-open the degeneracy §4.3 closes.

    ABOVE the depth there are no counts at all (R1: 99.994% of the missing
    budget lies above ``z = 0.30`` and the measured in-support fraction is
    6e-5), so ``amp(z > z_depth)`` is an ASSUMPTION, not a fitted quantity.
    That is why PR-8 ships a sensitivity scan and never a marginalized
    posterior (OWNER DECISION 7): the width above the depth is a pure function
    of the number chosen here, and quoting it as a measurement would be quoting
    a prior.

    Parameters
    ----------
    z : array
        Redshifts of the consumption rows.
    z_depth : float
        The FITTED depth — the edge of the count channel's support, not the
        node range.  ``amp == 1`` at or below it.
    amp_hi : float
        The assumed amplitude above the depth, relative to the fitted region.
        ``0.0`` reproduces the shipped "``Q == 1`` above ``z_depth``" convention
        (PLAN §4.2's stated under-dispersion) EXACTLY: the basis rows are
        multiplied by a literal zero, so the field, and with it ``logQ``, is
        bit-zero there.  ``1.0`` would be the full prior variance — a factor of
        ``e`` over the 38% of sky §4.2 names.
    kind : {"step", "growth"}
        ``"step"``: ``amp_hi`` everywhere above the depth (what the PR-8 table
        quotes).  ``"growth"``: ``amp_hi * D(z)/D(z_depth)``, the same assumed
        amplitude carried by linear growth.
    Om0 : float
        Matter density for the ``"growth"`` quadrature; ignored by ``"step"``.
    """
    z = np.asarray(z, dtype=float)
    if kind not in _VALID_AMP_KINDS:
        raise ValueError(
            f"amp_profile kind must be one of {_VALID_AMP_KINDS}, got {kind!r}.")
    amp_hi = float(amp_hi)
    if amp_hi < 0.0:
        raise ValueError(
            f"amp_hi must be >= 0 (it is an amplitude, and it enters the field "
            f"as amp*xi with xi ~ N(0, I)); got {amp_hi}.")
    above = z > float(z_depth)
    if kind == "step":
        hi = np.full(z.shape, amp_hi)
    else:
        D = growth_factor(z, Om0)
        D_depth = float(growth_factor(np.asarray([float(z_depth)]), Om0)[0])
        hi = amp_hi * D / D_depth
    # ``np.where`` and not an in-place write: the below-depth branch must be
    # the literal 1.0 the docstring promises, on every element.
    return np.where(above, hi, 1.0)


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
    amp_hi: float | None = None,
    amp_kind: str = "step",
    amp_z_depth: float | None = None,
    amp_Om0: float = 0.3075,
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
    amp_hi, amp_kind, amp_z_depth, amp_Om0 :
        PR-8's ``amp(z)`` profile (:func:`amp_profile`), applied to the
        REDSHIFT FACTOR ROWS: ``phi_z -> amp(z) phi_z``, which scales the field
        ``f(p, z) = row_fac[p] . phi_z[z]`` by ``amp(z)`` exactly, at every
        ``p``, with no change whatsoever to the node geometry, the kernels or
        their Cholesky factors.  That placement is deliberate — it is the only
        one under which ``amp`` is a pure per-``z`` amplitude and the
        ``amp == 1`` case is the IDENTITY (``x * 1.0 == x`` in IEEE754), so
        ``amp_hi=None`` and an all-ones profile produce BIT-IDENTICAL bases
        (pin P19a).  Scaling ``K_sph`` instead — the scalar ``amp`` argument
        above — does not have that property, because the per-factor jitter is
        absolute and does not scale with it.

        ``amp_hi is None`` (the default) is the legacy path: no profile is
        evaluated and no multiplication is performed.  Otherwise ``amp_z_depth``
        must be given (the FITTED depth, below which PLAN §4.3 pins
        ``amp == 1``) and the profile is applied to BOTH ``phi_z_out`` and
        ``phi_z_fine``.  ``phi_z_fine`` is the count operator's grid and lives
        entirely below the depth in every shipped configuration, where the
        profile is exactly ``1.0``; applying it there anyway is what keeps the
        fitted and consumed fields the same object by construction instead of
        by an argument about grid ranges.
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

    # ---- PR-8: the amp(z) profile, applied to the redshift factor rows ----
    if amp_hi is not None:
        if amp_z_depth is None:
            raise ValueError(
                "build_latent_basis: amp_hi requires amp_z_depth (the FITTED "
                "depth). PLAN §4.3 pins amp = 1 wherever the counts constrain "
                "the field, so the profile is meaningless without the edge of "
                "that region; passing amp_hi alone would scale the fitted "
                "region too and re-open the (b_gal, amp) degeneracy.")
        amp_out = amp_profile(np.expm1(np.asarray(zeta_out, dtype=float)),
                              z_depth=amp_z_depth, amp_hi=amp_hi,
                              kind=amp_kind, Om0=amp_Om0)
        phi_z_out = phi_z_out * jnp.asarray(amp_out)[:, None]
        if phi_z_fine is not None:
            amp_fine = amp_profile(
                np.expm1(np.asarray(zeta_fine, dtype=float)),
                z_depth=amp_z_depth, amp_hi=amp_hi, kind=amp_kind,
                Om0=amp_Om0)
            phi_z_fine = phi_z_fine * jnp.asarray(amp_fine)[:, None]

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
    if amp_hi is not None:
        # Written into ``basis_meta`` ONLY when a profile was applied, so an
        # artifact built before PR-8 and one built after with no profile carry
        # byte-identical metadata and the loader's ``.get("amp_hi")`` is the
        # single test for "does this artifact model the field above the
        # depth?".  These four numbers are all the loader needs to REBUILD the
        # profile (it already rebuilds ``phi_z`` from ``basis_meta``), so no
        # new dataset is stored and the two sides cannot drift.
        meta.update(amp_hi=float(amp_hi), amp_kind=str(amp_kind),
                    amp_z_depth=float(amp_z_depth), amp_Om0=float(amp_Om0))
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
                f_p_footprint, *, row_fac=None) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    ``row_fac`` — **pass the artifact's own ``(M_draw, n_fit, M_z)`` row
    factors whenever they will be STORED at reduced precision.**  The budget
    identity that ``rho`` enforces is exact only if the moments and the seam
    evaluate the SAME field: building ``A``/``B`` from the f64 ``xi`` while the
    seam consumes an f32 ``row_fac`` leaves a residual ``~ b |f| eps_f32``,
    measured at **2.7e-7** relative at the production corner (``b_GW = 4``,
    per-mode amplitude 2.46) against **2e-15** when the two agree.  That is far
    below the +55% Jensen inflation the identity removes, but eq. (4) is stated
    as an identity, so the builder closes it rather than carrying a tolerance.
    ``None`` recomputes the field from ``xi_members`` in f64 (the reference
    path, used by the tests that pin this function against a dense rebuild).
    """
    xi_members = jnp.atleast_2d(jnp.asarray(xi_members))    # (M_draw, M)
    b_nodes = jnp.asarray(b_nodes, dtype=jnp.float64)       # (n_b,)
    f_p = jnp.asarray(f_p_footprint, dtype=jnp.float64)     # (n_fit,)
    if row_fac is not None:
        row_fac = jnp.asarray(row_fac, dtype=jnp.float64)   # (M_draw, n_fit, M_z)

    def _one(xi, rf):
        if rf is None:
            Xi = jnp.reshape(xi, (basis.m_sph, basis.m_z))
            rf = basis.proj_sph @ Xi
        f = rf @ basis.phi_z_out.T                          # (n_fit, N_z)

        # One b node at a time: the (n_b, n_fit, N_z) exponential cube is
        # ~4 GB at production scale, so it is never materialized.
        def _at_b(b):
            e = jnp.exp(b * f)                              # (n_fit, N_z)
            return jnp.sum(e, axis=0), f_p @ e              # (N_z,), (N_z,)

        A, B = jax.lax.map(_at_b, b_nodes)
        return A, B

    if row_fac is None:
        A, B = jax.vmap(lambda xi: _one(xi, None))(xi_members)
    else:
        A, B = jax.vmap(lambda rf: _one(None, rf))(row_fac)
    return A, B


def sky_moments_by_tracer(basis: LatentBasis, xi_members, b_nodes,
                          f_p_by_tracer, *, proj_by_tracer=None,
                          row_fac_by_tracer=None):
    """Per-tracer moment tables ``(A_k, B_k)`` and constants ``(P_k, F_k)``.

    **Why the tables are PER TRACER and cannot be shared (PLAN §2.2 + §4.4's
    ``Z_k`` non-cancellation).**  Eq. (2)'s two reductions

        A(z; b) = sum_{p in F} e^{b f(p,z)},   B(z; b) = sum_{p in F} f_p e^{b f(p,z)}

    depend on the tracer through TWO things: the fitted footprint ``F_k`` the
    sums run over, and the per-pixel selection fraction ``f_p^(k)`` that weights
    ``B``.  Both are properties of catalog ``k``'s depth and mask, not of the
    field, so K tracers need K tables even though they share one ``xi`` and one
    ``b_GW`` grid.  Sharing one table across catalogs would normalize catalog
    2's budget against catalog 1's footprint — the exact error the K >= 2
    refusal in ``likelihood/factory.py`` exists to prevent.

    **This is where the closed form ``redshift/completion.py`` needs actually
    lives.**  ``completion.field_global_log_Z``'s own docstring records that at
    K = 1 the survey-global ``log Z`` cancels — it enters the ``N`` per-event PE
    terms and the ``-N log mu`` selection term the same number of times — while
    "at K >= 2 each catalog's ``log_Z_global`` sits inside its own mixture
    branch and does NOT cancel".  §2.2's decomposition is what makes that
    surviving object cheap: split the sum at ``f_p = 0``, and the occupied rows
    go through the existing row-wise path while the empty/off-footprint block is
    eq. (2) over the empty subset, closed-form in the scalar ``c = C(z; theta)``
    as ``(A_k - c B_k)/(P_k - c F_k)``.  In latent mode the off-footprint block
    is even simpler, because the footprint is fitted TO the counts: every empty
    pixel is outside ``F_k``, ``Q == 1`` there by the seam's convention, and the
    budget is the plain ``f_p`` formula (``completion.py``'s LATENT branch).  So
    the per-tracer ``Z_k`` needs the per-tracer ``(A_k, B_k, P_k, F_k)`` and
    nothing else — no per-member empty budget, no second reduction, and no
    derivation of our own.

    ``f_p_by_tracer`` is a length-K sequence of footprint completeness rows.
    ``proj_by_tracer`` is the matching sequence of basis row blocks (defaults to
    ``basis.proj_sph`` for every tracer, i.e. a shared footprint with
    per-tracer depth); ``row_fac_by_tracer`` is the matching sequence of stored
    f32 row factors, passed for the eq. (4) exactness reason
    :func:`sky_moments` documents at length.

    Returns ``(A, B, P, F)`` with ``A``/``B`` shaped ``(K, M_draw, n_b, N_z)``
    and ``P``/``F`` shaped ``(K,)``.
    """
    f_by_k = list(f_p_by_tracer)
    K = len(f_by_k)
    projs = ([basis.proj_sph] * K if proj_by_tracer is None
             else list(proj_by_tracer))
    rfs = ([None] * K if row_fac_by_tracer is None
           else list(row_fac_by_tracer))
    if len(projs) != K or len(rfs) != K:
        raise ValueError(
            f"sky_moments_by_tracer: {K} completeness blocks against "
            f"{len(projs)} row blocks and {len(rfs)} row-factor blocks.")
    As, Bs, Ps, Fs = [], [], [], []
    for f_p, proj, rf in zip(f_by_k, projs, rfs):
        # ``replace`` and not a re-construction: LatentBasis grows fields as the
        # ladder proceeds (PR-8 added the amplitude profile), and a positional
        # or exhaustive-keyword rebuild here would silently drop whichever field
        # arrives next while still type-checking.
        sub = dataclasses.replace(basis, proj_sph=proj)
        A, B = sky_moments(sub, xi_members, b_nodes, f_p, row_fac=rf)
        P, F = sky_constant_coeffs(f_p)
        As.append(A); Bs.append(B); Ps.append(P); Fs.append(F)
    return (jnp.stack(As), jnp.stack(Bs),
            np.asarray(Ps, dtype=float), np.asarray(Fs, dtype=float))


def rho_from_moments(A, B, c, b_index) -> jnp.ndarray:
    """Closed-form budget normalizer ``rho_m(z; c)`` at one ``b`` node:
    ``log[(A - c B) / (P_F - c F_F)]`` with ``(P_F, F_F)`` folded in by the
    caller via ``A, B`` normalization or passed explicitly at the seam
    (PR-5); here the bare ``log(A - c B)`` building block."""
    return jnp.log(A[..., b_index, :] - c * B[..., b_index, :])


def chebyshev_lobatto_nodes(n_b: int, b_max: float) -> np.ndarray:
    """The ``n_b`` Chebyshev–Lobatto nodes on ``[0, b_max]`` (the ``b_GW``
    interpolation grid of PLAN §2.2; endpoints included)."""
    k = np.arange(n_b)
    return 0.5 * b_max * (1.0 - np.cos(np.pi * k / (n_b - 1)))


def interp_moments_b(table, b_nodes, b) -> jnp.ndarray:
    """Barycentric Chebyshev–Lobatto interpolation of a moment table along
    its ``b``-node axis (axis -2, per :func:`sky_moments`'s layout), at one
    scalar ``b`` — the P9-pinned online path (1e-6).  Exact at the nodes by
    the barycentric formula's pole handling."""
    table = jnp.asarray(table)
    b_nodes = jnp.asarray(b_nodes, dtype=jnp.float64)
    n = b_nodes.shape[0]
    w = jnp.asarray((-1.0) ** np.arange(n))
    w = w.at[0].mul(0.5).at[-1].mul(0.5)
    d = b - b_nodes                                          # (n_b,)
    exact = jnp.any(d == 0.0)
    idx = jnp.argmin(jnp.abs(d))
    coef = w / jnp.where(d == 0.0, 1.0, d)
    coef = jnp.where(exact, jnp.zeros_like(coef).at[idx].set(1.0), coef)
    coef = coef / jnp.sum(coef)
    return jnp.tensordot(table, coef, axes=([-2], [0]))
