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
    g = gradient(xi, op)
    H = hessian_separable(xi, op)
    L = jnp.linalg.cholesky(H)
    dx = jax.scipy.linalg.cho_solve((L, True), g)
    J0 = objective(xi, op)
    # ``slope = g^T H^{-1} g >= 0`` because ``H >= I`` is SPD, so ``-dx`` is a
    # descent direction ALWAYS; what the line search supplies is the length.
    slope = jnp.dot(g, dx)
    slack = slack_rel * jnp.abs(J0)

    xi_full = xi - dx                     # the shipped expression, untouched
    J_full = objective(xi_full, op)

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
        return (a, x_try, objective(x_try, op), k + 1)

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
    H = hessian_separable(xi_hat, op)
    L = jnp.linalg.cholesky(H)
    g = gradient(xi_hat, op)
    return dict(xi_hat=xi_hat, H_chol=L,
                grad_inf=jnp.max(jnp.abs(g)),
                J=objective(xi_hat, op), n_iter=n_iter,
                alpha=alphas, n_backtrack=n_bts)


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
