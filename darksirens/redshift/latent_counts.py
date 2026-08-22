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

K >= 2 (PR-7): :class:`MultiTracerCountOperator` stacks K tracers over ONE
shared ``xi`` — ``J = 0.5||xi||^2 - sum_k log p_count,k``, so the K data terms
and the K Fisher informations add while the ridge ``I`` is counted once.  That
single line is PLAN §4.4's shared-realization theorem: there is no catalog
index on ``xi`` for a ``realization_set_id`` to disagree about, so the check
is retired for want of a referent rather than satisfied.
:func:`decoupled_objective` is the model in which the property is FALSE (K
fields, K ridges — the independent-fields product prior that
``--allow_unverified_shared_lss_members`` actually marginalizes over), kept in
the module because Tier E (iii') is the comparison between the two.

``b_gal`` is fixed at the anchor in the solve, and its uncertainty is carried
by a RANK-1 INFLATION of the draw covariance (PLAN §3.4, R1-SEV2-5):

    Cov(xi) = H^{-1} + s_b^2 v v^T,   v = d xi_hat/d b = -H^{-1} (d grad/d b)

with ``s_b`` the PROFILE CURVATURE of the count channel in ``b_gal``
(:func:`b_gal_profile_sigma`), never a prior width — see that function's
docstring for why the distinction is load-bearing rather than cosmetic.
:func:`laplace_draws` samples that covariance WITHOUT any dense M x M
refactorization, by adding one scalar-normal step along ``v``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    ``label`` names the tracer at K >= 2 (PR-7).  It is not decoration: the
    per-tracer moment tables, the per-tracer ``b_k`` sensitivity columns and
    the artifact's tracer groups are all keyed by it, and OWNER DECISION 9's
    DISJOINT partition is a statement about which galaxies carry which label.
    ``pix`` is this tracer's own footprint ``F_k``, so two tracers may have
    DIFFERENT footprints and different ``completeness`` on the same sky.
    """
    pix: Any            # (n_fit,) int32 global pixel ids (rows of proj_sph)
    counts: Any         # (n_fit, G_s) f64 integer-valued
    completeness: Any   # (n_fit,) f64  f_p in [0, 1]
    bias: float = 1.0
    stratum: Any = None
    label: str = "tracer"


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


def _grad_data(xi, op: CountOperator) -> jnp.ndarray:
    """``b sum_g Phi_g^T (N_g - T_g pi_g)`` — the DATA term of ``grad J``,
    i.e. everything except the ridge ``xi``.

    Extracted verbatim from :func:`gradient` at PR-7 so the K-tracer stack can
    ADD data terms (they sum; the ridge does not) without a second copy of the
    contraction and without perturbing the K=1 arithmetic by a single
    operation: :func:`gradient` is now literally ``xi - _grad_data(xi, op)``,
    the same expression tree it always evaluated, which is what makes
    ``multi_gradient`` at K=1 bit-identical to ``gradient`` rather than merely
    equal to it.
    """
    pi = jnp.exp(_log_pi(op, xi))                           # (n_fit, G_s)
    resid = op.counts - op.shell_totals[None, :] * pi       # (n_fit, G_s)
    G_sph = op.proj_sph.T @ resid                           # (M_sph, G_s)
    data = op.bias * (G_sph @ op.phi_shell)                 # (M_sph, M_z)
    return data.reshape(-1)


def gradient(xi, op: CountOperator) -> jnp.ndarray:
    """``grad J = xi - b sum_g Phi_g^T (N_g - T_g pi_g)`` — separable form."""
    return jnp.asarray(xi) - _grad_data(xi, op)


def _hess_data(xi, op: CountOperator) -> jnp.ndarray:
    """``b^2 sum_g [Phi_g^T diag(T_g pi_g) Phi_g - T_g u_g u_g^T]`` — the DATA
    block of eq. (3), i.e. ``H - I``.

    Same extraction as :func:`_grad_data` and for the same reason: at K >= 2
    the Fisher informations of the K stacked multinomials ADD while the prior
    precision ``I`` is counted ONCE (there is one ``xi``, not K of them —
    PLAN §4.4's shared-realization theorem in its most concrete arithmetic
    form).  Adding ``I`` per tracer would be the decoupled model of §0.5
    finding 12, which is the thing Tier E (iii') is built to distinguish from
    this one.
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
    return (op.bias ** 2) * Hdata


def hessian_separable(xi, op: CountOperator) -> jnp.ndarray:
    """Exact Fisher Hessian of eq. (1') with the rank-1 term (eq. 3).

    Two-stage contraction for the diagonal part (never materializes the
    Kronecker basis), plus ``G_s`` rank-1 Kronecker outer products.
    Returns the dense ``(M, M)`` SPD matrix (``H >= I``).
    """
    M = op.rank
    return jnp.eye(M) + _hess_data(xi, op)


# ------------------------------------------------------- K >= 2: the stack
#
# PLAN §4.4, THE SHARED-REALIZATION THEOREM.  The whole of the K >= 2 design is
# in the arithmetic of the three functions below, so it is worth saying plainly
# what they do and do not assume.  There is ONE ``xi``.  The K tracers do not
# each get a field; they each get a LIKELIHOOD of the same field.  Therefore
#
#     J(xi) = 0.5 ||xi||^2 - sum_k log p_count,k(xi; b_k, f_p^(k), N^(k))
#
# and, differentiating, the K data terms ADD while the ridge is counted once:
#
#     grad J = xi - sum_k [d log p_k / d xi],    H = I + sum_k H_data,k .
#
# "Member m of every catalog is the same realization" is then not a stamp to be
# verified but a property of this expression: there is no index on ``xi`` for a
# catalog id to disagree about.  The table path has to CHECK that property with
# ``realization_set_id`` because it consumes K independently built Q tables;
# here there is nothing to check, which is what §4.4 means by "it ceases to have
# a referent" (and see :func:`decoupled_objective` for the model in which the
# property is FALSE, which is the object Tier E (iii') measures against).
#
# OWNER DECISION 9 — DISJOINT tracers.  The stacked objective above is a sum of
# K independent multinomial likelihoods, which is correct exactly when no galaxy
# is counted twice; an AGN sample that is a SUBSET of the galaxy sample would
# double-count its members and over-inform ``xi`` by the overlap fraction.  This
# module never has to detect that, because :func:`counts_from_catalog_by_tracer`
# builds the K count arrays from a per-galaxy LABEL and a label is single-valued
# — the partition is disjoint by construction, not by assertion.
# :func:`check_disjoint_tracers` is the guard for count arrays that arrive from
# somewhere else.

@dataclass(frozen=True)
class MultiTracerCountOperator:
    """K stacked :class:`CountOperator` s over ONE shared ``xi``.

    Each entry carries its own ``proj_sph`` rows (its own footprint ``F_k``),
    its own ``log_fp``, its own ``counts`` and its own ``bias`` ``b_k``; they
    share the radial factor ``phi_shell`` only because the shell edges are a
    property of the analysis, not of the tracer (a tracer with different shells
    is admissible — nothing below reads ``phi_shell`` across entries — but the
    rank must agree, which is checked here).

    K = 1 is admitted and is the identity wrapper: every function in this
    section then evaluates the single-tracer expression tree unchanged, which
    is the PR-7 gate (``tests/test_latent_multitracer.py``).
    """
    ops: tuple

    def __post_init__(self):
        if not self.ops:
            raise ValueError(
                "MultiTracerCountOperator: at least one tracer is required.")
        ranks = {op.rank for op in self.ops}
        if len(ranks) != 1:
            raise ValueError(
                f"MultiTracerCountOperator: the K tracers must share one "
                f"field, so they must share one mode space; got ranks "
                f"{sorted(ranks)}. This is the shared-realization property in "
                f"its bluntest form -- a per-tracer rank would be a per-tracer "
                f"field (PLAN §4.4).")

    @property
    def n_tracer(self) -> int:
        return len(self.ops)

    @property
    def rank(self) -> int:
        return int(self.ops[0].rank)

    @property
    def m_sph(self) -> int:
        return int(self.ops[0].m_sph)

    @property
    def m_z(self) -> int:
        return int(self.ops[0].m_z)


def make_multi_count_operator(proj_sph_by_tracer, phi_z_fine, W,
                              tracers) -> MultiTracerCountOperator:
    """Build the K-tracer stack from K :class:`TracerCounts`.

    ``proj_sph_by_tracer`` is either one basis row block shared by all tracers
    (the common case: one HEALPix map, K galaxy populations on it) or a
    length-K sequence of row blocks, one per tracer footprint.  ``W`` is
    likewise one response or a length-K sequence.
    """
    tracers = tuple(tracers)
    K = len(tracers)
    proj = (list(proj_sph_by_tracer) if isinstance(proj_sph_by_tracer, (list, tuple))
            else [proj_sph_by_tracer] * K)
    Ws = list(W) if isinstance(W, (list, tuple)) else [W] * K
    if len(proj) != K or len(Ws) != K:
        raise ValueError(
            f"make_multi_count_operator: got {K} tracers but {len(proj)} row "
            f"blocks and {len(Ws)} shell responses.")
    return MultiTracerCountOperator(tuple(
        make_count_operator(p, phi_z_fine, w, t)
        for p, w, t in zip(proj, Ws, tracers)))


def multi_shell_multinomial_logl(xi, mop: MultiTracerCountOperator):
    """``sum_k log p_count,k(xi)`` — the K partial likelihoods, added.

    Accumulated as a Python left fold starting from tracer 0 rather than a
    ``jnp.sum`` over a stacked array, so at K = 1 the result IS
    :func:`shell_multinomial_logl`'s value with no reduction wrapped around it.
    """
    total = shell_multinomial_logl(xi, mop.ops[0])
    for op in mop.ops[1:]:
        total = total + shell_multinomial_logl(xi, op)
    return total


def multi_objective(xi, mop: MultiTracerCountOperator):
    """``J(xi) = 0.5||xi||^2 - sum_k log p_count,k(xi)``.

    ONE ridge for K tracers.  See :func:`decoupled_objective` for the variant
    with K ridges and K fields, which is what the table path's
    ``--allow_unverified_shared_lss_members`` actually marginalizes over
    (``inference/loaders.py:352-395``).
    """
    xi = jnp.asarray(xi)
    return 0.5 * jnp.dot(xi, xi) - multi_shell_multinomial_logl(xi, mop)


def multi_gradient(xi, mop: MultiTracerCountOperator):
    """``grad J = xi - sum_k [d log p_k/d xi]`` — the data terms add."""
    data = _grad_data(xi, mop.ops[0])
    for op in mop.ops[1:]:
        data = data + _grad_data(xi, op)
    return jnp.asarray(xi) - data


def multi_hessian_separable(xi, mop: MultiTracerCountOperator):
    """``H = I + sum_k H_data,k`` — the K Fisher informations add, ``I`` once."""
    data = _hess_data(xi, mop.ops[0])
    for op in mop.ops[1:]:
        data = data + _hess_data(xi, op)
    return jnp.eye(mop.rank) + data


def as_multi(op) -> MultiTracerCountOperator:
    """Wrap a bare :class:`CountOperator` as a one-tracer stack (idempotent)."""
    if isinstance(op, MultiTracerCountOperator):
        return op
    return MultiTracerCountOperator((op,))


def _obj(xi, op):
    """Objective dispatch: single-tracer operators keep the ORIGINAL call."""
    return (multi_objective(xi, op) if isinstance(op, MultiTracerCountOperator)
            else objective(xi, op))


def _grad(xi, op):
    return (multi_gradient(xi, op) if isinstance(op, MultiTracerCountOperator)
            else gradient(xi, op))


def _hess(xi, op):
    return (multi_hessian_separable(xi, op)
            if isinstance(op, MultiTracerCountOperator)
            else hessian_separable(xi, op))


def _logl(xi, op):
    return (multi_shell_multinomial_logl(xi, op)
            if isinstance(op, MultiTracerCountOperator)
            else shell_multinomial_logl(xi, op))


def check_disjoint_tracers(tracers, parent_counts=None, *,
                           atol: float = 0.0) -> dict:
    """OWNER DECISION 9's guard for count arrays not built by
    :func:`counts_from_catalog_by_tracer`.

    Disjointness is a statement about GALAXIES, and count arrays have already
    thrown the galaxy identities away, so this can only ever be a NECESSARY
    condition — it is written to say so rather than to imply more.  What it
    checks, when a ``parent_counts`` array (the union catalog's own shell
    counts on the same rows) is supplied, is the shell-by-shell budget
    ``sum_k N^(k)_pg <= N^parent_pg``: an AGN sample that is a subset of the
    galaxy sample violates it wherever the subset is dense, which is the R14
    failure mode.  With no parent it checks only that the labels are distinct,
    and reports ``verified=False`` so a caller cannot mistake "nothing to check
    against" for "checked".

    Returns ``{verified, labels, max_excess, total_by_tracer}``.
    """
    tracers = tuple(tracers)
    labels = [str(t.label) for t in tracers]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"check_disjoint_tracers: tracer labels must be distinct, got "
            f"{labels}. Under OWNER DECISION 9 the K tracers PARTITION the "
            f"catalog, and a partition's blocks are named.")
    totals = [float(np.sum(np.asarray(t.counts))) for t in tracers]
    if parent_counts is None:
        return dict(verified=False, labels=labels, max_excess=None,
                    total_by_tracer=totals)
    par = np.asarray(parent_counts, dtype=float)
    stack = np.zeros_like(par)
    for t in tracers:
        c = np.asarray(t.counts, dtype=float)
        if c.shape != par.shape:
            raise ValueError(
                f"check_disjoint_tracers: tracer {t.label!r} has counts of "
                f"shape {c.shape} against a parent of shape {par.shape}; the "
                f"budget check needs the same (pixel, shell) rows.")
        stack = stack + c
    excess = float(np.max(stack - par))
    if excess > atol:
        raise ValueError(
            f"check_disjoint_tracers: the K tracers over-count the parent "
            f"catalog by up to {excess:.6g} galaxies in a single (pixel, "
            f"shell) cell. Under OWNER DECISION 9 the tracers are DISJOINT; "
            f"a stacked count-channel objective sums K multinomials and would "
            f"count the overlap twice, over-informing xi by the overlap "
            f"fraction (risk R14).")
    return dict(verified=True, labels=labels, max_excess=excess,
                total_by_tracer=totals)


def counts_from_catalog_by_tracer(zgals, ngals, pix_ids, z_count_edges,
                                  labels, n_tracer: int) -> list:
    """K integer shell-count arrays from a per-galaxy tracer LABEL.

    ``labels`` is shaped like ``zgals`` (``(n_pix, n_max)``) and holds, for
    each catalogued galaxy, the index of the tracer it belongs to; entries
    beyond ``ngals[p]`` are ignored.  Because a label is single-valued, the K
    arrays returned here are a DISJOINT PARTITION of the parent catalog by
    construction — OWNER DECISION 9 discharged structurally rather than by a
    check that can only ever be necessary (:func:`check_disjoint_tracers`).
    Summing the K arrays reproduces :func:`counts_from_catalog` exactly, which
    ``tests/test_latent_multitracer.py`` pins.
    """
    zg = np.asarray(zgals)
    ng = np.asarray(ngals)
    lab = np.asarray(labels)
    edges = np.asarray(z_count_edges, dtype=float)
    out = [np.zeros((len(pix_ids), edges.size - 1)) for _ in range(n_tracer)]
    for j, p in enumerate(np.asarray(pix_ids)):
        n = int(ng[p])
        if not n:
            continue
        zrow, lrow = zg[p, :n], lab[p, :n]
        for k in range(n_tracer):
            sel = lrow == k
            if np.any(sel):
                out[k][j], _ = np.histogram(zrow[sel], bins=edges)
    return out


#: Armijo sufficient-decrease constant.  1e-4 is the textbook value; the test
#: is so slack at this ``c1`` that it only ever refuses a step that makes ``J``
#: WORSE, which is precisely the failure mode being fixed.
_ARMIJO_C1 = 1e-4
#: Backtrack shrink factor and the hard cap on halvings per trip.  The cap is
#: what keeps the solve a bounded, deterministic function of its inputs; 30
#: halvings is ``alpha = 9.3e-10``, far past anything a convex ``J`` needs.
_ARMIJO_SHRINK = 0.5
_ARMIJO_MAX_BACKTRACK = 30
#: ABSOLUTE slack on the Armijo test, as a fraction of ``|J|``.  This is not a
#: tuning knob, it is a floating-point necessity, and the number is measured:
#: on the nside-16 closure world ``J ~ 8e6``, so once the iteration is at the
#: optimum the true decrease (~1e-10) sits BELOW the f64 resolution of ``J``
#: itself (``eps |J| = 1.8e-9``) and a strict test can find no acceptable
#: ``alpha``.  It then backtracks to the cap and freezes with ``grad_inf`` at
#: 1.4e-6 — a P6 failure for a rounding reason rather than a convergence one
#: (observed on 1 of 8 realizations by the PR-6a closure workstream, see
#: ``experiments/field_level_plan/pr6a/CLOSURE.md`` S-1).  1e-12 |J| is three
#: orders below the decrease any real Newton trip produces and three orders
#: above the noise floor.
_ARMIJO_SLACK_REL = 1e-12


def _damped_newton_step(xi, op: CountOperator, *, c1: float, shrink: float,
                        max_backtrack: int, slack_rel: float):
    """One Fisher/Newton trip with a bounded Armijo backtrack.

    Returns ``(xi_new, alpha, n_backtrack)``.  The FULL step is tried first
    and is formed as ``xi - dx`` — literally the expression the undamped
    solve used, with no ``alpha`` multiplying it — so that on any problem
    where the full step is accepted at every trip the iterate sequence is
    bit-identical to the pre-damping solve.  That is a hard requirement, not
    a nicety: the production anchor (``latent_anchor_v2a.h5``) converged
    undamped at ``grad_inf = 1.09e-10`` and must not move.  Writing the first
    trial as ``xi - 1.0 * dx`` would be exact in IEEE arithmetic but invites
    XLA to contract the multiply-add into an FMA and change the last bit, so
    the un-scaled expression is used deliberately.
    """
    g = _grad(xi, op)
    H = _hess(xi, op)
    L = jnp.linalg.cholesky(H)
    dx = jax.scipy.linalg.cho_solve((L, True), g)
    J0 = _obj(xi, op)
    # ``slope = g^T H^{-1} g >= 0`` because ``H >= I`` is SPD, so ``-dx`` is a
    # descent direction ALWAYS; what the line search supplies is the length.
    slope = jnp.dot(g, dx)
    slack = slack_rel * jnp.abs(J0)

    xi_full = xi - dx                     # the shipped expression, untouched
    J_full = _obj(xi_full, op)

    def _accepted(alpha, J_try):
        return J_try <= J0 - c1 * alpha * slope + slack

    def _cond(state):
        alpha, _, J_try, k = state
        return jnp.logical_and(jnp.logical_not(_accepted(alpha, J_try)),
                               k < max_backtrack)

    def _body(state):
        alpha, _, _, k = state
        a = alpha * shrink
        x_try = xi - a * dx
        return (a, x_try, _obj(x_try, op), k + 1)

    alpha, xi_new, _, n_bt = lax.while_loop(
        _cond, _body,
        (jnp.asarray(1.0), xi_full, J_full, jnp.asarray(0, dtype=jnp.int32)))
    return xi_new, alpha, n_bt


def count_map_solve(op: CountOperator, *, n_iter: int = 13, xi0=None,
                    c1: float = _ARMIJO_C1,
                    shrink: float = _ARMIJO_SHRINK,
                    max_backtrack: int = _ARMIJO_MAX_BACKTRACK,
                    slack_rel: float = _ARMIJO_SLACK_REL) -> dict:
    """Fixed-trip DAMPED Fisher scoring on the convex ``J`` (no host control
    flow).

    ``op`` is a :class:`CountOperator` or, from PR-7, a
    :class:`MultiTracerCountOperator`.  The dispatch is by type on the host and
    resolves before tracing, so a bare operator evaluates ``objective`` /
    ``gradient`` / ``hessian_separable`` -- the pre-PR-7 expressions,
    unaltered -- and the K = 1 solve is bit-identical trip for trip, line
    search included (``tests/test_latent_multitracer.py``).

    Each trip takes the exact Newton direction ``dx = H^{-1} grad`` and then
    backtracks on ``alpha`` until Armijo's sufficient-decrease test on ``J``
    passes.  The trip count is still fixed and the backtrack is capped, so
    the solve remains a deterministic, traceable function of its inputs
    (PLAN §10: no history-dependent stopping); the backtrack is a
    :func:`jax.lax.while_loop` inside the :func:`jax.lax.scan` body, not a
    Python loop, so nothing here is host control flow.

    **Why the damping exists.**  The previous version of this function took
    UNDAMPED steps and its docstring asserted that "``H >= I`` makes the
    un-damped Fisher step globally well-posed for this objective in
    practice".  That claim is false and was measured false.  For the
    multinomial logit the link is canonical, so the Fisher information EQUALS
    the observed Hessian (PLAN §3.4 — which is why Fisher scoring is
    normative here rather than merely convenient); this iteration is
    therefore EXACT Newton on a convex objective, and exact Newton on a
    convex objective is only LOCALLY convergent.  ``H >= I`` bounds the step,
    it does not make it a descent step of acceptable length.  Measured by the
    PR-6a closure workstream at nside 16 with 1.1e6 galaxies over 1854 x 12
    voxels, on ``xi_true`` drawn from the model's own prior, 8 realizations
    (``seed = 6100 + 977 k``):

        undamped: 3 of 8 converge (P6 pass rate 0.375), ``grad_inf`` up to
                  4.9e5 on the 5 that do not.  Realization ``k = 1``
                  (``seed = 7077``) leaves ``xi = 0`` with ``|dx| = 41.1``
                  against ``||xi_true|| = 16.9``, ``J`` RISES 8.275e6 ->
                  8.506e6 at trip 1, runs away to ``|dx| = 1.39e6`` and
                  settles into a period-2 limit cycle at
                  ``grad_inf = 4.41e5``.
        damped:   8 of 8 converge (P6 pass rate 1.000), ``grad_inf`` between
                  7.1e-12 and 4.3e-11 at the same ``n_iter = 13``.

    The cure is small because the disease is: on all 5 diverging
    realizations the line search halves the step exactly ONCE, on trip 0
    only, and every subsequent trip takes ``alpha = 1``
    (``alpha = [0.5, 1, 1, ...]``).  On the 3 that already converged it takes
    ``alpha = 1`` at every trip, and therefore reproduces the undamped
    iterate sequence bit for bit — see :func:`_damped_newton_step` for how
    that bit-identity is secured, and ``tests/test_latent_solve_damping.py``
    for the pin.  The failure was LOUD rather than silent
    (``cli/build_latent_field.py`` gates on ``grad_inf > 1e-8`` and the
    production anchor converged at 1.09e-10), so this is a robustness defect
    in the builder, not a correctness defect in any shipped artifact.

    Returns ``{xi_hat, H_chol, n_iter, grad_inf, J, alpha, n_backtrack}``;
    callers gate on ``grad_inf`` (pin P6: < 1e-8).  ``alpha`` and
    ``n_backtrack`` are the per-trip line-search trace — a solve that had to
    damp is a solve worth reporting, and the arrays are (n_iter,) so a caller
    can log ``float(sol["n_backtrack"].sum())`` without a second pass.
    """
    M = op.rank
    xi = jnp.zeros(M) if xi0 is None else jnp.asarray(xi0)

    def _step(xi, _):
        xi_new, alpha, n_bt = _damped_newton_step(
            xi, op, c1=c1, shrink=shrink, max_backtrack=max_backtrack,
            slack_rel=slack_rel)
        return xi_new, (alpha, n_bt)

    xi_hat, (alphas, n_bts) = lax.scan(_step, xi, None, length=n_iter)
    H = _hess(xi_hat, op)
    L = jnp.linalg.cholesky(H)
    g = _grad(xi_hat, op)
    return dict(xi_hat=xi_hat, H_chol=L,
                grad_inf=jnp.max(jnp.abs(g)),
                J=_obj(xi_hat, op), n_iter=n_iter,
                alpha=alphas, n_backtrack=n_bts)


def laplace_evidence(op: CountOperator, xi_hat, H_chol) -> jnp.ndarray:
    """The rung-1 galaxy-side evidence (PLAN eq. 5, §0.5 D2):
    ``l(xi_hat) - 0.5||xi_hat||^2 - 0.5 log det H`` — the CORRECT Laplace
    evidence, Occam term included (the envelope theorem removes
    ``d xi_hat/d theta`` from J, not from l)."""
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(H_chol)))
    return (_logl(xi_hat, op)
            - 0.5 * jnp.dot(xi_hat, xi_hat) - 0.5 * logdet)


def sensitivity(xi_hat, H_chol, dgrad_dtheta) -> jnp.ndarray:
    """``S = d xi_hat/d theta = -H^{-1} (d grad/d theta)`` (IFT columns).

    ``dgrad_dtheta`` is ``(M, n_theta)`` — for a stacked K-tracer objective
    the gradients ADD, so catalog k's own theta contributes its own column
    built from its own operator; pass the horizontally stacked block and
    K >= 2 adds columns, not code.  ``wrt = b_gal`` is one more column:
    ``d grad/d b`` of :func:`gradient` at the anchor.

    **PR-7 CHECKED that claim against this function rather than restating it.**
    It is linear in ``dgrad_dtheta``, never inspects the block's columns and
    never assumes one operator produced them, so the K-tracer stacking needed
    no change here at all — only a BUILDER for the extra columns, which is
    :func:`dgrad_db_by_tracer` (see its docstring for the finding in full).
    ``H_chol`` must be the STACKED Cholesky: that is where the tracers couple,
    and it is the only place they do.
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


def _with_bias(op: CountOperator, b) -> CountOperator:
    """``op`` with ``bias`` replaced — the one place a ``b`` is made traceable."""
    return CountOperator(op.proj_sph, op.phi_shell, op.counts, op.log_fp, b)


def with_biases(mop: MultiTracerCountOperator, biases
                ) -> MultiTracerCountOperator:
    """The K-tracer stack at a new bias vector ``(b_1, ..., b_K)``.

    Everything else — footprints, completeness, counts, the shell response —
    is carried through unchanged, so this is the ONE moving part of the bias
    profile of :func:`bias_profile`.
    """
    b = jnp.asarray(biases)
    if int(b.shape[0]) != mop.n_tracer:
        raise ValueError(
            f"with_biases: got {int(b.shape[0])} biases for "
            f"{mop.n_tracer} tracers.")
    return MultiTracerCountOperator(tuple(
        _with_bias(op, b[k]) for k, op in enumerate(mop.ops)))


def dgrad_db_by_tracer(xi, mop: MultiTracerCountOperator) -> jnp.ndarray:
    """``(M, K)`` — column ``k`` is ``d grad J / d b_k`` at fixed ``xi``.

    **This is the "K >= 2 adds columns" claim discharged, and it is worth
    being precise about what was and was not required to discharge it.**
    PLAN §3.4 states that ``S = d xi_hat/d theta`` is assembled from the
    STACKED K-tracer objective, so that catalog ``k``'s own selection theta and
    its own ``b_k`` each contribute their own column, and that K >= 2 then adds
    columns rather than forcing a refactor.  Checked against the shipped code:
    :func:`sensitivity` takes ``dgrad_dtheta`` as an ``(M, n_theta)`` BLOCK and
    does one ``cho_solve`` against one ``H_chol``.  It never inspects the
    columns, never assumes one operator produced them, and is linear in the
    block — so K columns from K operators are admissible in the shipped
    signature EXACTLY AS IT STANDS, and PR-7 adds no argument to it and changes
    no line of it.  What was missing at K = 1 was only the BUILDER of the extra
    columns, because ``dgrad_db`` closes over a single operator.  This function
    is that builder: the stacked gradient's ``b_k`` derivative is
    ``d/d b_k`` of tracer ``k``'s data term alone (the other tracers do not
    contain ``b_k``), so column ``k`` is exactly ``dgrad_db(xi, ops[k])`` with
    the ridge already cancelled by the difference — and the ``H_chol`` it is
    solved against is the STACKED one, which is where the coupling between
    tracers enters.
    """
    cols = [jnp.reshape(dgrad_db(xi, op), (-1,)) for op in mop.ops]
    return jnp.stack(cols, axis=1)


def log_bias_prior_potential(biases, log_b_prior) -> jnp.ndarray:
    """``sum_k 0.5 ((log b_k - mu_k)/s)^2`` — the OPTIONAL log-normal prior on
    the tracer biases, as a potential added to ``J``.

    **Why the profile needs one at all, and why it is disclosed rather than
    buried.**  The count channel of eq. (1') is invariant under
    ``xi -> xi/s``, ``b_k -> s b_k`` for a single common ``s``: the exponent
    ``b_k (Phi_s Xi phi_z)`` is unchanged.  The ridge ``0.5||xi||^2`` is the
    only term that sees ``s``, and it DECREASES monotonically as ``s`` grows,
    so the profile likelihood in the overall bias amplitude has no interior
    maximum — ``b_hat -> infinity`` with ``||xi_hat|| -> 0`` at fixed product.
    The RATIO ``b_2/b_1`` is exactly the invariant of that direction and is
    perfectly well determined; the amplitude is not.  A prior on ``log b_k``
    makes the amplitude proper so that a covariance can be quoted at all,
    and — this is the point — it leaves the ratio essentially untouched in the
    SHARED model while being the ONLY thing that determines it in the decoupled
    one, which is why Tier E reports the ratio width against a sweep in ``s``
    rather than at one value of it.

    ``log_b_prior`` is ``(mu, s)`` with ``mu`` a scalar or ``(K,)`` of
    ``log b_ref`` and ``s`` the prior sd in log-bias.  ``None`` is no prior and
    every function below is then exactly the flat-profile object.
    """
    mu, s = log_b_prior
    b = jnp.asarray(biases)
    mu = jnp.broadcast_to(jnp.asarray(mu, dtype=jnp.float64), b.shape)
    return jnp.sum(0.5 * ((jnp.log(b) - mu) / float(s)) ** 2)


def bias_profile_grad(xi_hat, mop: MultiTracerCountOperator, *,
                      log_b_prior=None) -> jnp.ndarray:
    """``dP/db_k`` of the profile ``P(b) = J(xi_hat(b); b)``, shape ``(K,)``.

    By the envelope theorem at ``xi_hat(b)`` the implicit term drops and the
    profile gradient is the PARTIAL derivative of ``J`` in ``b`` at fixed
    ``xi`` — one scalar jacobian per tracer, no re-solve.  ``J``'s ridge is
    ``b``-free, so this is ``-d log p_count,k/d b_k`` (plus the optional
    log-bias prior, which is a pure function of ``b`` and so needs no envelope
    argument at all).
    """
    def _P(b):
        val = multi_objective(xi_hat, with_biases(mop, b))
        if log_b_prior is not None:
            val = val + log_bias_prior_potential(b, log_b_prior)
        return val
    b0 = jnp.asarray([op.bias for op in mop.ops], dtype=jnp.float64)
    return jax.jacfwd(_P)(b0)


def bias_profile_hessian(xi_hat, mop: MultiTracerCountOperator, *,
                         H_chol=None, log_b_prior=None) -> jnp.ndarray:
    """``d^2 P/db_j db_k``, the ``(K, K)`` PROFILE curvature in the biases.

    The K-tracer generalization of :func:`b_gal_profile_sigma`'s scalar
    identity, and PLAN §3.4's "at K >= 2 the ratio ``b_2/b_1`` comes from its
    own 2x2 profile curvature" made computable:

        P(b)      = J(xi_hat(b); b)
        d^2P/db^2 = J_bb + (d grad/d b)^T V,   V = -H^{-1} (d grad/d b)  (K cols)

    ``J_bb`` is the partial ``(K, K)`` Hessian at FIXED ``xi`` — block-diagonal,
    because tracer ``j``'s likelihood does not contain ``b_k`` for ``k != j``.
    **Every off-diagonal element of the profile curvature therefore comes from
    the second term, i.e. from ``H^{-1}``, i.e. from the SHARED FIELD.**  That
    is the coupling Tier E (ii)/(iii') measure: decouple the fields and the
    ``H^{-1}`` that mixes the tracers becomes block diagonal, the off-diagonal
    vanishes, and the ratio's credible region changes in a predictable way.

    ``H_chol`` is the stacked Cholesky at ``xi_hat``; rebuilt here if omitted.
    """
    xi_hat = jnp.asarray(xi_hat)
    if H_chol is None:
        H_chol = jnp.linalg.cholesky(multi_hessian_separable(xi_hat, mop))
    b0 = jnp.asarray([op.bias for op in mop.ops], dtype=jnp.float64)

    def _P(b):
        return multi_objective(xi_hat, with_biases(mop, b))

    J_bb = jax.jacfwd(jax.jacfwd(_P))(b0)                   # (K, K)
    if log_b_prior is not None:
        J_bb = J_bb + jax.jacfwd(jax.jacfwd(
            lambda b: log_bias_prior_potential(b, log_b_prior)))(b0)
    dgb = dgrad_db_by_tracer(xi_hat, mop)                   # (M, K)
    V = sensitivity(xi_hat, H_chol, dgb)                    # (M, K)
    return J_bb + dgb.T @ V


def bias_profile_curvature_log(g_b, H_b, biases) -> jnp.ndarray:
    """The profile curvature re-expressed in ``u_k = log b_k``.

        dP/du_k        = b_k (dP/db_k)
        d2P/du_j du_k  = b_j b_k (d2P/db_j db_k) + delta_jk b_k (dP/db_k)

    **The log parametrization is not cosmetic and this is the one place to say
    why.**  In ``b`` coordinates the ``(K, K)`` curvature is nearly singular by
    construction: the common-rescaling direction ``xi -> xi/s, b -> s b``
    leaves every ``eta`` invariant (see :func:`log_bias_prior_potential`), so
    the amplitude direction is soft while the RATIO direction is stiff, and a
    Newton step in ``b`` is dominated by the soft direction — measured on the
    Tier-E world, ``cond(H_b) = 5.5e3`` with the step in the amplitude blowing
    past any admissible ``b`` while the ratio never moves.  In ``u`` the same
    curvature is positive definite and well conditioned, because the second
    term above adds exactly the gradient the reparametrization generates.  The
    log-normal bias prior is Gaussian in ``u``, so it transforms consistently
    and needs no separate handling: including it in ``g_b`` and ``H_b``
    reproduces its ``1/s^2`` diagonal here exactly.
    """
    b = jnp.asarray(biases, dtype=jnp.float64)
    g_b = jnp.asarray(g_b, dtype=jnp.float64)
    return (b[:, None] * jnp.asarray(H_b) * b[None, :]
            + jnp.diag(b * g_b))


def bias_profile(mop: MultiTracerCountOperator, *, n_outer: int = 8,
                 n_iter: int = 13, damp: float = 1.0, xi0=None,
                 log_b_prior=None, max_log_step: float = 0.5) -> dict:
    """Profile-maximize the stacked count channel over ``(b_1, ..., b_K)``.

    An outer Newton **in log bias** on ``P(b) = J(xi_hat(b); b)``, whose
    gradient and Hessian are the three functions above; each outer step costs
    ONE inner :func:`count_map_solve` (the anchor at the current ``b``) plus
    ``K`` triangular solves against that solve's own ``H_chol``.  The inner
    solve is warm-started from the previous ``xi_hat``, so the outer loop
    converges in a handful of trips.  Fixed trip count and no host-side
    stopping rule, for the reason PLAN §10 gives for the inner solve.

    The step is capped at ``max_log_step`` nats per component.  That is a
    trust region, not a tuning dial: it only ever shortens a step, it is
    inactive once the iteration is near the optimum (where the log-space
    curvature is well conditioned — see
    :func:`bias_profile_curvature_log`), and without it the first trip from a
    deliberately wrong starting ``b`` can leave the region where the inner
    solve's warm start is any use.

    Returns ``{b_hat, xi_hat, H_chol, cov_log_b, curvature_log_b, cov_b,
    grad_inf, profile_grad, profile_grad_log, P}``.  ``cov_log_b`` is the
    inverse of the ``(K, K)`` **log**-bias profile curvature at ``b_hat`` — the
    credible region, and via ``log r = u_2 - u_1``
    (:func:`bias_ratio_from_profile`) the object Tier E gates.  ``cov_b`` is
    the same object pushed to ``b`` coordinates for readers who want it there.
    """
    b = jnp.asarray([op.bias for op in mop.ops], dtype=jnp.float64)
    xi = xi0
    sol = None
    for _ in range(n_outer):
        cur = with_biases(mop, b)
        sol = count_map_solve(cur, n_iter=n_iter, xi0=xi)
        xi = sol["xi_hat"]
        g_b = bias_profile_grad(xi, cur, log_b_prior=log_b_prior)
        H_b = bias_profile_hessian(xi, cur, H_chol=sol["H_chol"],
                                   log_b_prior=log_b_prior)
        g_u = b * g_b
        H_u = bias_profile_curvature_log(g_b, H_b, b)
        step = jnp.linalg.solve(H_u, g_u)
        step = jnp.clip(damp * step, -max_log_step, max_log_step)
        b = b * jnp.exp(-step)
    cur = with_biases(mop, b)
    sol = count_map_solve(cur, n_iter=n_iter, xi0=xi)
    xi = sol["xi_hat"]
    g_b = bias_profile_grad(xi, cur, log_b_prior=log_b_prior)
    H_b = bias_profile_hessian(xi, cur, H_chol=sol["H_chol"],
                               log_b_prior=log_b_prior)
    H_u = bias_profile_curvature_log(g_b, H_b, b)
    cov_u = jnp.linalg.inv(H_u)
    inv_b = jnp.diag(1.0 / b)
    return dict(b_hat=b, xi_hat=xi, H_chol=sol["H_chol"],
                cov_log_b=cov_u, curvature_log_b=H_u,
                cov_b=jnp.diag(b) @ cov_u @ jnp.diag(b),
                curvature_b=H_b, grad_inf=sol["grad_inf"],
                profile_grad=g_b, profile_grad_log=b * g_b,
                P=multi_objective(xi, cur))


def bias_ratio_from_profile(b_hat, cov_log_b, *, num: int = 1, den: int = 0
                            ) -> dict:
    """``r = b_num/b_den`` and its width, from the LOG-bias profile covariance.

    ``log r = u_num - u_den`` is linear in the parameters the covariance is
    quoted in, so its variance is the exact contraction
    ``C_nn + C_dd - 2 C_nd`` with no delta-method step at all; only the push to
    ``sigma_r = r sigma_log r`` is first order, and both are returned so a
    reader can use whichever is appropriate.

    **The ``-2 C_nd`` is the entire content of Tier E (ii) and (iii').**  The
    off-diagonal exists only because the two tracers share one field: it is
    ``(d grad/d b_num)^T H^{-1} (d grad/d b_den)`` through the STACKED
    ``H_chol`` (:func:`bias_profile_hessian`).  Decouple the fields and the
    curvature is block diagonal, ``C_nd`` is exactly zero, and the ratio's
    variance becomes the SUM of two separately-unidentified amplitudes'
    variances instead of the difference of two strongly correlated ones.
    """
    b = np.asarray(b_hat, dtype=float)
    C = np.asarray(cov_log_b, dtype=float)
    r = float(b[num] / b[den])
    var_log = float(C[num, num] + C[den, den] - 2.0 * C[num, den])
    sig_log = float(np.sqrt(max(var_log, 0.0)))
    corr = float(C[num, den] / np.sqrt(C[num, num] * C[den, den]))
    return dict(ratio=r, sigma=r * sig_log, sigma_log=sig_log,
                log_ratio=float(np.log(r)),
                b_hat=b.tolist(),
                sigma_log_b=np.sqrt(np.diag(C)).tolist(),
                corr=corr)


def decoupled_objective(xi_stack, mop: MultiTracerCountOperator):
    """The DECOUPLED two-field model: K INDEPENDENT fields, K ridges.

        J_dec(xi_1..xi_K) = sum_k [ 0.5||xi_k||^2 - log p_count,k(xi_k; b_k) ]

    This is not a straw man and it is not a bug — it is what the table path
    actually marginalizes over when a K >= 2 shared-LSS run is forced through
    with ``--allow_unverified_shared_lss_members``
    (``inference/loaders.py:352-395``, in the code's own words: an
    INDEPENDENT-FIELDS PRODUCT PRIOR rather than the shared-field prior the
    estimator assumes).  PLAN §0.5 finding 12 makes running BOTH on the same
    mock the substantive replacement for v3's routing-tautology gate: v3 asked
    that a deleted check not fire, which is satisfied by deleting the check;
    v4 asks that the COUPLING the flag throws away be demonstrated, which is
    satisfied only by an estimator that has it and one that does not.

    ``xi_stack`` is ``(K, M)``.  Note what separates instantly here and does
    not in :func:`multi_objective`: ``J_dec`` is a SUM OF K INDEPENDENT
    PROBLEMS, so its Hessian is block diagonal, ``bias_profile_hessian``'s
    off-diagonal is exactly zero, and ``b_k`` is left degenerate with tracer
    ``k``'s own field amplitude — broken only by that tracer's ridge.  The
    shared model breaks the same degeneracy with the OTHER tracer's counts.
    That is the whole physical content of differentiator 2, in one line of
    arithmetic.
    """
    xs = jnp.asarray(xi_stack)
    total = objective(xs[0], mop.ops[0])
    for k, op in enumerate(mop.ops[1:], start=1):
        total = total + objective(xs[k], op)
    return total


def decoupled_bias_profile(mop: MultiTracerCountOperator, **kw) -> dict:
    """The decoupled twin of :func:`bias_profile` — K separate single-tracer
    profiles, assembled into one ``(K,)`` estimate and one BLOCK-DIAGONAL
    ``(K, K)`` covariance.

    Written as K calls to :func:`bias_profile` on one-tracer stacks rather than
    as a second solver, because that is literally what the decoupled model is:
    the shared-field estimator with the sharing removed.  The two functions
    consume the same counts, the same footprints, the same completeness, the
    same shell response and the same ``b`` starting point; the ONLY difference
    is whether the K tracers see one ``xi`` or K.
    """
    outs = [bias_profile(MultiTracerCountOperator((op,)), **kw)
            for op in mop.ops]
    b_hat = jnp.stack([o["b_hat"][0] for o in outs])
    cov = np.zeros((mop.n_tracer, mop.n_tracer))
    for k, o in enumerate(outs):
        cov[k, k] = float(np.asarray(o["cov_log_b"])[0, 0])
    return dict(b_hat=b_hat, cov_log_b=jnp.asarray(cov), per_tracer=outs,
                grad_inf=max(float(o["grad_inf"]) for o in outs))


#: Systematics floor on ``s_b``, as a fraction of ``b_gal`` (PLAN §3.4 v4:
#: "with a stated systematics floor").  5% is STATED here rather than tuned:
#: the count channel's statistical curvature is the uncertainty of a
#: SINGLE-NUMBER linear-bias model — one scale-independent, redshift-independent,
#: deterministic ``b_gal`` shared by every shell and every pixel — evaluated on
#: a catalog with ~1e6 galaxies, so it shrinks like 1/sqrt(N_gal) toward zero
#: while the model error it is standing in for does not.  Scale dependence of
#: bias across the multipoles this basis resolves, its evolution across
#: ``z_depth``, and tracer stochasticity are all outside eq. (1') and each is a
#: several-percent-level effect on the amplitude at these scales.  5% is the
#: order at which linear bias is quotable at all; the floor keeps the draw
#: covariance from claiming a field precision the model cannot support.  It is
#: a FLOOR, not a dial: it only ever raises ``s_b``, it is stamped into the
#: artifact, and if the statistical curvature is the larger of the two it is
#: the one that is used (:func:`b_gal_profile_sigma` reports which bound won).
B_GAL_SYSTEMATIC_FLOOR_FRAC = 0.05

#: ``fold_in`` offset for the rank-1 scalar stream.  NOT ``split(key)``: the
#: ``g`` stream must keep keying on ``key`` itself so that turning the
#: inflation ON leaves ``L^{-T} g_m`` bit-identical and the inflation is
#: visibly ADDITIVE (pinned by ``test_latent_b_gal_dispersion``); splitting
#: would re-key ``g`` and make every member move for two reasons at once.
_B_GAL_EPS_FOLD = 1


def b_gal_profile_sigma(xi_hat, op: CountOperator, *, H_chol=None,
                        dgrad_b=None, v_b=None,
                        systematic_floor_frac: float
                        = B_GAL_SYSTEMATIC_FLOOR_FRAC) -> dict:
    """``s_b`` — the PROFILE CURVATURE of the count channel in ``b_gal``.

    PLAN §3.4 [v4, §0.5 finding 11] — v3's "``s_b`` is a 20% prior width" is
    WITHDRAWN, and the reason is not bookkeeping.  §4.3 pins ``amp = 1``, so
    ``b_gal`` is the sole clustering amplitude of eq. (1') and the counts
    MEASURE it; a plan cannot claim that and simultaneously hand ``s_b`` a free
    dial.  With a dial, Tier-B's "latent-on CI >= table CI" acceptance
    criterion (PLAN §6.2) could be made to pass or fail by choice, which is
    exactly the criticism v4 levels at rev 1.  So ``s_b`` is measured:

        s_b^2 = [ -d^2 log p_count / db^2 ]^{-1}   at the anchor,

    the curvature of the PROFILE, not of the conditional slice.  Profiling
    ``xi`` out of ``J(xi; b) = 0.5||xi||^2 - log p_count(xi; b)`` and using the
    envelope theorem at ``xi_hat(b)``:

        P(b)   = J(xi_hat(b); b)
        P''(b) = J_bb - J_bxi^T H^{-1} J_bxi = J_bb + (d grad/d b) . v,
        v      = d xi_hat/d b = -H^{-1} (d grad/d b)                (§3.4)

    so the profile costs ONE 1-D second derivative on top of the ``b_gal``
    column of ``S`` that :func:`sensitivity` has already built against this
    same ``H_chol`` — "a 1-D profile against the same ``H_chol``", verbatim.
    The subtracted term is ``J_bxi^T H^{-1} J_bxi >= 0``: the profile is
    strictly BROADER than the conditional slice at fixed field, because the
    field can partly absorb a change in ``b``.  Both are returned, so the gap
    between them is on the record rather than asserted.

    ``dgrad_b`` / ``v_b`` are accepted so the builder passes the columns it has
    ALREADY computed (``dgrad_db`` and ``sensitivity_S[:, 'b_gal']``) instead of
    recomputing the same object a second way; pass ``H_chol`` alone and both are
    built here.

    Returns ``{s_b, s_b_stat, s_b_floor, floor_active, curvature_profile,
    curvature_conditional, v_b}``, all host floats except ``v_b``.
    """
    xi_hat = jnp.asarray(xi_hat)
    if dgrad_b is None:
        dgrad_b = dgrad_db(xi_hat, op)
    dgrad_b = jnp.reshape(jnp.asarray(dgrad_b), (-1,))
    if v_b is None:
        if H_chol is None:
            raise ValueError(
                "b_gal_profile_sigma: give either v_b (the b_gal column of "
                "sensitivity_S) or H_chol to build it from; the two must be "
                "the SAME object -- PLAN §3.4 has one construction, not two.")
        v_b = sensitivity(xi_hat, H_chol, dgrad_b[:, None])[:, 0]
    v_b = jnp.reshape(jnp.asarray(v_b), (-1,))

    def _logl_at_b(b):
        return shell_multinomial_logl(
            xi_hat, CountOperator(op.proj_sph, op.phi_shell, op.counts,
                                  op.log_fp, b))

    # J_bb = -d^2 log p_count/db^2 at FIXED xi (the ridge 0.5||xi||^2 is b-free).
    curv_cond = float(-jax.jacfwd(jax.jacfwd(_logl_at_b))(op.bias))
    curv_prof = curv_cond + float(jnp.dot(dgrad_b, v_b))
    floor = float(systematic_floor_frac) * abs(float(op.bias))
    if not np.isfinite(curv_prof) or curv_prof <= 0.0:
        # Fail LOUD: a non-positive profile curvature means the anchor is not a
        # maximum in b (or the operator is degenerate), and silently falling
        # back to the floor would report a systematics-limited s_b for what is
        # actually a broken solve.
        raise ValueError(
            f"b_gal_profile_sigma: profile curvature {curv_prof:.6e} is not "
            f"positive (conditional {curv_cond:.6e}); the anchor is not a "
            "maximum of the count channel in b_gal, so s_b^2 = 1/curvature is "
            "undefined. Check the solve (grad_inf) before inflating draws.")
    s_stat = float(curv_prof ** -0.5)
    return dict(s_b=max(s_stat, floor), s_b_stat=s_stat, s_b_floor=floor,
                floor_active=bool(floor > s_stat),
                curvature_profile=curv_prof,
                curvature_conditional=curv_cond,
                v_b=v_b)


def laplace_draws_multitracer(xi_hat, H_chol, n_draw: int, key, *, cov_b, V_b,
                              return_g: bool = False,
                              return_eps: bool = False):
    """The K-tracer draw covariance ``H^{-1} + V C_b V^T`` (PR-7).

    The rank-K generalization of :func:`laplace_draws`'s rank-1 inflation, and
    the generalization is forced rather than chosen: at K >= 2 the biases are
    **correlated**, because :func:`bias_profile_hessian`'s off-diagonal is
    ``(d grad/d b_j)^T H^{-1} (d grad/d b_k)`` and ``H^{-1}`` is the SHARED
    field's covariance.  Inflating each ``b_k`` independently by its own
    marginal ``s_k`` would put the wrong correlation (namely zero) into the
    member ensemble — it would draw the members of the DECOUPLED model
    (:func:`decoupled_objective`) while the counts were fitted with the shared
    one, which is precisely the mismatch PLAN §0.5 finding 12 is about.

        xi_m = xi_hat + L_H^{-T} g_m + V L_C eps_m,   eps_m ~ N(0, I_K)

    with ``L_C = chol(C_b)``.  ``C_b`` is the ``(K, K)`` profile covariance of
    the biases (``bias_profile``'s ``cov_b``), ``V_b`` the ``(M, K)`` block
    ``d xi_hat/d b`` (:func:`sensitivity` on :func:`dgrad_db_by_tracer`).
    ``eps`` is antithetic in the same pairing as ``g``, for the P14 reason
    :func:`laplace_draws` documents.  At K = 1 this reproduces
    :func:`laplace_draws`'s inflation in exact arithmetic but is NOT promised
    bit-identical to it (the RNG is drawn at shape ``(n_draw//2, 1)`` and the
    inflation is a matmul rather than a scalar multiply), which is why the
    K = 1 builder path still calls :func:`laplace_draws`.
    """
    if n_draw % 2:
        raise ValueError(
            "laplace_draws_multitracer: n_draw must be even (antithetic).")
    M = int(jnp.asarray(xi_hat).shape[0])
    V = jnp.asarray(V_b, dtype=jnp.float64)
    C = jnp.asarray(cov_b, dtype=jnp.float64)
    if V.ndim != 2 or V.shape[0] != M or V.shape[1] != C.shape[0]:
        raise ValueError(
            f"laplace_draws_multitracer: V_b is {V.shape} and cov_b is "
            f"{C.shape}; V_b must be (M, K) = ({M}, {C.shape[0]}). The "
            f"inflation is V C V^T -- K scales need K directions.")
    L_C = jnp.linalg.cholesky(C)
    g_half = jax.random.normal(key, (n_draw // 2, M))
    g = jnp.concatenate([g_half, -g_half], axis=0)
    steps = jax.scipy.linalg.solve_triangular(H_chol.T, g.T, lower=False).T
    e_half = jax.random.normal(
        jax.random.fold_in(key, _B_GAL_EPS_FOLD), (n_draw // 2, C.shape[0]))
    eps = jnp.concatenate([e_half, -e_half], axis=0)         # (n_draw, K)
    draws = jnp.asarray(xi_hat)[None, :] + steps + eps @ L_C.T @ V.T
    out = (draws,)
    if return_g:
        out = out + (g,)
    if return_eps:
        out = out + (eps,)
    return out[0] if len(out) == 1 else out


def laplace_draws(xi_hat, H_chol, n_draw: int, key, *, return_g: bool = False,
                  s_b=None, v_b=None, return_eps: bool = False):
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

    **The ``b_gal`` rank-1 inflation (PLAN §3.4).**  With ``s_b`` and
    ``v_b = d xi_hat/d b`` supplied, the members are drawn from

        Cov(xi) = H^{-1} + s_b^2 v v^T

    as ``xi_m = xi_hat + L_H^{-T} g_m + s_b eps_m v``, with ``eps_m`` an
    INDEPENDENT scalar standard normal.  That is exact and needs no dense
    ``M x M`` factorization of the inflated covariance: the two sources are
    independent, so their covariances add, and the second contributes
    ``s_b^2 v v^T`` by construction.  ``eps`` is antithetic in the SAME pairing
    as ``g`` (``eps = [e, -e]``, partner of ``k`` is ``k + n_draw//2``), which
    is what keeps PR-5b's balanced member ordering ``[0, M/2, 1, M/2+1, ...]``
    balanced in BOTH sources — a naive unpaired prefix loses the odd-part
    cancellation and fails P14 at ``M_draw = 4`` and 8 (PR-5b measured this on
    the ``g`` source alone; a second unpaired source would reintroduce it).

    With ``s_b`` omitted (or zero) this is bit-for-bit today's function: the
    ``g`` stream keys on ``key`` itself and the rank-1 stream on
    ``fold_in(key, 1)``, so switching the feature on ADDS a term and moves
    nothing else.
    """
    if n_draw % 2:
        raise ValueError("laplace_draws: n_draw must be even (antithetic).")
    M = xi_hat.shape[0]
    g_half = jax.random.normal(key, (n_draw // 2, M))
    g = jnp.concatenate([g_half, -g_half], axis=0)
    steps = jax.scipy.linalg.solve_triangular(
        H_chol.T, g.T, lower=False).T                       # (n_draw, M)
    draws = jnp.asarray(xi_hat)[None, :] + steps

    inflate = s_b is not None and v_b is not None and float(s_b) != 0.0
    if s_b is not None and v_b is None:
        raise ValueError(
            "laplace_draws: s_b was given without v_b (= d xi_hat/d b, the "
            "b_gal column of sensitivity_S). The rank-1 inflation is "
            "s_b^2 v v^T; a scale with no direction is not a covariance.")
    if inflate:
        v = jnp.reshape(jnp.asarray(v_b), (-1,))
        if int(v.size) != int(M):
            raise ValueError(
                f"laplace_draws: v_b has {int(v.size)} entries but xi_hat has "
                f"{int(M)}; the inflation direction lives in the same mode "
                "space as the field.")
        e_half = jax.random.normal(
            jax.random.fold_in(key, _B_GAL_EPS_FOLD), (n_draw // 2,))
        eps = jnp.concatenate([e_half, -e_half], axis=0)    # (n_draw,)
        draws = draws + float(s_b) * eps[:, None] * v[None, :]
    else:
        eps = jnp.zeros((n_draw,))

    out = (draws,)
    if return_g:
        out = out + (g,)
    if return_eps:
        out = out + (eps,)
    return out[0] if len(out) == 1 else out


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
