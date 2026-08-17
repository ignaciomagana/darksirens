"""Shell-total-conditioned angular count channel for the latent field (PR-3).

The partial likelihood of PLAN §1.1 eq. (1'), theta-SHAPED from the start:

    pi_pg(xi; theta) = f_p exp(eta_pg) / sum_p' f_p' exp(eta_p'g)
    eta_pg           = b_gal * (Phi_s[p] . Xi . phi~_z[g])
    phi~_z[g, :]     = sum_n W(theta)[g, n] phi_z_fine[n, :]
    log p_count(xi)  = sum_g sum_p N_pg log pi_pg(xi)

``W`` carries the within-shell weighting ``base(z; theta)`` (normalized per
shell), the photo-z forward convolution, and the shell indicator
(:func:`darksirens.redshift.latent_field.shell_response`); because ``W`` acts
on the RADIAL BASIS ROWS, ``eta`` is exactly linear in ``xi`` and eq. (3)'s
Hessian is exact.  The offline anchor is the special case
``theta = theta_ref``; rung 1 evaluates the linear response of the same
object (PLAN §1.7).

Conditioning on the shell totals ``T_g = sum_p N_pg`` deletes the monopole
(and with it ``n0`` and — via ``dV/dz = (c/H0)^3 shape(z)`` and the
h-firewall — all ``H0`` content) from the field posterior; what remains is
pure angular placement per shell.

Objective / gradient / Fisher (PLAN §3.4, all exact for eq. (1')):

    J(xi)  = 0.5 ||xi||^2 - log p_count(xi)
    grad J = xi - b sum_g Phi_g^T (N_g - T_g pi_g)
    H      = I + b^2 sum_g [ Phi_g^T diag(T_g pi_g) Phi_g - T_g u_g u_g^T ],
             u_g = Phi_g^T pi_g                                        (3)

The rank-1 subtraction is the multinomial's own Fisher term — dropping it
makes ``H`` too large and the Laplace draws UNDER-dispersed (R2-SEV2-10).
``H >= I`` holds for any link (it is a Fisher information), which is why
Fisher scoring is normative (PLAN §3.4).

Sensitivities: ``S = d xi_hat / d theta = -H^{-1} (d grad / d theta)`` at
``(xi_hat, theta_ref)`` — one triangular-solve pair per column against the
same ``H_chol``; K >= 2 tracers ADD COLUMNS (the stacked objective is a sum,
so its cross-derivatives stack), never a refactor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class TracerCounts:
    """One tracer's count-channel data (footprint rows only).

    ``counts`` are INTEGER shell counts per footprint pixel
    (``np.histogram(zgals, bins=z_count_edges)`` per pixel — the data stay
    counts; the photo-z kernel convolves the MODEL inside ``W``).
    ``completeness`` is the per-pixel selection fraction ``f_p`` on the same
    rows.  ``bias`` is ``b_gal``, FIXED at the anchor (its uncertainty is a
    rank-1 covariance inflation through ``sensitivity(wrt="b_gal")``).
    """
    pix: Any            # (n_fit,) int32 global pixel ids (rows of proj_sph)
    counts: Any         # (n_fit, G_s) f64 integer-valued
    completeness: Any   # (n_fit,) f64  f_p in [0, 1]
    bias: float = 1.0
    stratum: Any = None


@dataclass(frozen=True)
class CountOperator:
    """Everything ``shell_multinomial_logl`` needs at one ``theta``.

    ``proj_sph`` is the footprint block of the sphere factor basis
    (``LatentBasis.proj_sph`` rows aligned with ``TracerCounts.pix``);
    ``phi_shell = W(theta) @ phi_z_fine`` is the shell-collapsed radial
    factor.  The anchor operator freezes ``W`` at ``theta_ref``; a
    theta-live caller rebuilds ``phi_shell`` (and ``log_fp``) per theta and
    everything downstream is unchanged — theta is an argument, not a rebuild.
    """
    proj_sph: Any       # (n_fit, M_sph) f64
    phi_shell: Any      # (G_s, M_z) f64
    counts: Any         # (n_fit, G_s) f64
    log_fp: Any         # (n_fit,) f64  log f_p (-inf rows are excluded rows)
    bias: float

    @property
    def m_sph(self) -> int:
        return int(self.proj_sph.shape[1])

    @property
    def m_z(self) -> int:
        return int(self.phi_shell.shape[1])

    @property
    def rank(self) -> int:
        return self.m_sph * self.m_z

    @property
    def shell_totals(self) -> jnp.ndarray:
        return jnp.sum(self.counts, axis=0)                 # (G_s,)


def make_count_operator(proj_sph, phi_z_fine, W, tracer: TracerCounts
                        ) -> CountOperator:
    """Anchor operator: ``phi_shell = W @ phi_z_fine`` with frozen ``W``."""
    phi_shell = jnp.asarray(W) @ jnp.asarray(phi_z_fine)    # (G_s, M_z)
    fp = jnp.asarray(tracer.completeness, dtype=jnp.float64)
    return CountOperator(
        proj_sph=jnp.asarray(proj_sph, dtype=jnp.float64),
        phi_shell=phi_shell,
        counts=jnp.asarray(tracer.counts, dtype=jnp.float64),
        log_fp=jnp.log(jnp.maximum(fp, 1e-300)),
        bias=float(tracer.bias))


def _eta(op: CountOperator, xi) -> jnp.ndarray:
    """``eta_pg = b (Phi_s[p] . Xi . phi~_z[g])`` — (n_fit, G_s), exactly
    linear in ``xi`` (eq. 1')."""
    Xi = jnp.reshape(jnp.asarray(xi), (op.m_sph, op.m_z))
    return op.bias * (op.proj_sph @ Xi) @ op.phi_shell.T


def _log_pi(op: CountOperator, xi) -> jnp.ndarray:
    a = op.log_fp[:, None] + _eta(op, xi)                   # (n_fit, G_s)
    return a - jax.scipy.special.logsumexp(a, axis=0, keepdims=True)


def shell_multinomial_logl(xi, op: CountOperator) -> jnp.ndarray:
    """``log p_count = sum_g sum_p N_pg log pi_pg`` (partial likelihood,
    F3b; the multinomial coefficient is xi-free and omitted)."""
    return jnp.sum(op.counts * _log_pi(op, xi))


def objective(xi, op: CountOperator) -> jnp.ndarray:
    """``J(xi) = 0.5 ||xi||^2 - log p_count(xi)`` — convex in ``xi``."""
    xi = jnp.asarray(xi)
    return 0.5 * jnp.dot(xi, xi) - shell_multinomial_logl(xi, op)


def gradient(xi, op: CountOperator) -> jnp.ndarray:
    """``grad J = xi - b sum_g Phi_g^T (N_g - T_g pi_g)`` — separable form."""
    pi = jnp.exp(_log_pi(op, xi))                           # (n_fit, G_s)
    resid = op.counts - op.shell_totals[None, :] * pi       # (n_fit, G_s)
    G_sph = op.proj_sph.T @ resid                           # (M_sph, G_s)
    data = op.bias * (G_sph @ op.phi_shell)                 # (M_sph, M_z)
    return jnp.asarray(xi) - data.reshape(-1)


def hessian_separable(xi, op: CountOperator) -> jnp.ndarray:
    """Exact Fisher Hessian of eq. (1') with the rank-1 term (eq. 3).

    Two-stage contraction for the diagonal part (never materializes the
    Kronecker basis), plus ``G_s`` rank-1 Kronecker outer products.
    Returns the dense ``(M, M)`` SPD matrix (``H >= I``).
    """
    pi = jnp.exp(_log_pi(op, xi))
    lam = op.shell_totals[None, :] * pi                     # (n_fit, G_s) T_g pi
    A = op.proj_sph
    # stage 1: T[g, i, j] = sum_p lam[p, g] A[p, i] A[p, j]
    T1 = jnp.einsum("pg,pi,pj->gij", lam, A, A)             # (G_s, Ms, Ms)
    # stage 2: diag part = sum_g T1[g] (x) (phi~_g phi~_g^T)
    Hd = jnp.einsum("gij,ga,gb->iajb", T1, op.phi_shell, op.phi_shell)
    # rank-1 subtraction: u_g = (A^T pi_g) (x) phi~_g, weight T_g
    v = A.T @ pi                                            # (Ms, G_s)
    Hr = jnp.einsum("g,ig,jg,ga,gb->iajb",
                    op.shell_totals, v, v, op.phi_shell, op.phi_shell)
    M = op.rank
    Hdata = (Hd - Hr).reshape(M, M)
    return jnp.eye(M) + (op.bias ** 2) * Hdata


def count_map_solve(op: CountOperator, *, n_iter: int = 13,
                    xi0=None) -> dict:
    """Fixed-trip Fisher scoring on the convex ``J`` (no host control flow).

    ``H >= I`` makes the un-damped Fisher step globally well-posed for this
    objective in practice; the fixed trip count keeps the solve a
    deterministic function of the inputs (PLAN §10: no history-dependent
    stopping).  Returns ``{xi_hat, H_chol, n_iter, grad_inf, J}``; callers
    gate on ``grad_inf`` (pin P6: < 1e-8).
    """
    M = op.rank
    xi = jnp.zeros(M) if xi0 is None else jnp.asarray(xi0)

    def _step(xi, _):
        g = gradient(xi, op)
        H = hessian_separable(xi, op)
        L = jnp.linalg.cholesky(H)
        dx = jax.scipy.linalg.cho_solve((L, True), g)
        return xi - dx, None

    xi_hat, _ = lax.scan(_step, xi, None, length=n_iter)
    H = hessian_separable(xi_hat, op)
    L = jnp.linalg.cholesky(H)
    g = gradient(xi_hat, op)
    return dict(xi_hat=xi_hat, H_chol=L,
                grad_inf=jnp.max(jnp.abs(g)),
                J=objective(xi_hat, op), n_iter=n_iter)


def laplace_evidence(op: CountOperator, xi_hat, H_chol) -> jnp.ndarray:
    """The rung-1 galaxy-side evidence (PLAN eq. 5, §0.5 D2):
    ``l(xi_hat) - 0.5||xi_hat||^2 - 0.5 log det H`` — the CORRECT Laplace
    evidence, Occam term included (the envelope theorem removes
    ``d xi_hat/d theta`` from J, not from l)."""
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(H_chol)))
    return (shell_multinomial_logl(xi_hat, op)
            - 0.5 * jnp.dot(xi_hat, xi_hat) - 0.5 * logdet)


def sensitivity(xi_hat, H_chol, dgrad_dtheta) -> jnp.ndarray:
    """``S = d xi_hat/d theta = -H^{-1} (d grad/d theta)`` (IFT columns).

    ``dgrad_dtheta`` is ``(M, n_theta)`` — for a stacked K-tracer objective
    the gradients ADD, so catalog k's own theta contributes its own column
    built from its own operator; pass the horizontally stacked block and
    K >= 2 adds columns, not code.  ``wrt = b_gal`` is one more column:
    ``d grad/d b`` of :func:`gradient` at the anchor.
    """
    return -jax.scipy.linalg.cho_solve((H_chol, True),
                                       jnp.asarray(dgrad_dtheta))


def dgrad_db(xi, op: CountOperator) -> jnp.ndarray:
    """``d grad J / d b_gal`` at fixed ``xi`` — the ``b_gal`` column feeding
    the rank-1 covariance inflation (PLAN §3.4, R1-SEV2-5)."""
    return jax.jacfwd(
        lambda b: gradient(
            xi, CountOperator(op.proj_sph, op.phi_shell, op.counts,
                              op.log_fp, b)))(op.bias)


def laplace_draws(xi_hat, H_chol, n_draw: int, key, *, return_g: bool = False):
    """``xi_m = xi_hat + L_H^{-T} g_m`` with ANTITHETIC standard-normal pairs
    (``n_draw`` even; the partner of member ``k`` is ``k + n_draw//2``) —
    PLAN §6.5 item 3: free, cancels the odd part of the response exactly.

    ``return_g`` additionally returns the WHITENED draws ``g`` themselves.  The
    two are different objects and conflating them is not harmless: ``g`` is
    standard normal (unit sd, mean zero) while ``xi_m`` is centred on
    ``xi_hat``, whose own per-mode amplitude is 2.46 on the production anchor.
    Anything that reads ``g`` but is handed ``xi_m`` silently picks up
    ``a·xi_hat`` — a set of same-sign offsets with a spread several times too
    small, which is exactly how it presents.
    """
    if n_draw % 2:
        raise ValueError("laplace_draws: n_draw must be even (antithetic).")
    M = xi_hat.shape[0]
    g_half = jax.random.normal(key, (n_draw // 2, M))
    g = jnp.concatenate([g_half, -g_half], axis=0)
    steps = jax.scipy.linalg.solve_triangular(
        H_chol.T, g.T, lower=False).T                       # (n_draw, M)
    draws = jnp.asarray(xi_hat)[None, :] + steps
    return (draws, g) if return_g else draws


def counts_from_catalog(zgals, ngals, pix_ids, z_count_edges) -> np.ndarray:
    """Integer shell counts per footprint pixel — EXACTLY
    ``np.histogram(zgals[p, :n], bins=z_count_edges)`` per pixel (the PR-2
    round-trip gate's convention; weights never enter the counts)."""
    zg = np.asarray(zgals)
    ng = np.asarray(ngals)
    edges = np.asarray(z_count_edges, dtype=float)
    out = np.zeros((len(pix_ids), edges.size - 1))
    for k, p in enumerate(np.asarray(pix_ids)):
        n = int(ng[p])
        if n:
            out[k], _ = np.histogram(zg[p, :n], bins=edges)
    return out
