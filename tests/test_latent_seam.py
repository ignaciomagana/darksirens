"""PR-5 pins for the latent seam (PLAN §6.3 P13b, P16, P18; §4.2 eq. 4).

The seam replaces a resident ``(M, N_rows, N_grid)`` log-Q table with a
generated ``logQ = 1[p in F](b_GW row_fac[p].phi_z[z] - rho[z])``.  These pins
fix the four properties that make that substitution legitimate:

P13b   off-footprint rows return **bit-zero** ``logQ`` -- the whole bracket,
       including ``-rho``, not just the field term.  ~38% of every production
       gather lands there (49,143 union rows vs 30,470 footprint rows), and
       PLAN eq. (4) conserves the off-footprint budget block only because
       ``Q == 1`` there exactly.
seam   the latent evaluator is **bit-identical** to the table evaluator fed the
       table the seam would generate, on a fixture inside both clip bounds.
       This is the pin that says the seam changed the SOURCE of ``Q`` and
       nothing else: same depth relaxation, same two-node interpolation, same
       normalizer selection.  The one deliberate difference is the clip BOUND
       (``_LATENT_LOGQ_CLIP`` vs ``_LOGQ_CLIP``), which is why that pin's
       fixture is the mild-amplitude one and the production-amplitude pins
       below are separate.
eq.(4) the generated ``Q`` conserves the CONSUMED missing budget
       ``sum_p (1 - f_p C(z)) Q_p(z)`` at every ``z``, member and ``b_GW`` --
       the identity ``rho`` is defined to enforce -- BOTH as an expression and
       through the clip the consumers apply to it at production amplitude.
P16    the complete-catalog limit: at ``C == 1`` (``f_p == 1``) the missing
       branch vanishes and the prior collapses to the observed host spikes
       **for every member**, so latent-on equals latent-off bit-identically.
       A physics identity, not a routing property -- strictly stronger than
       P12, which only tests the flag in its off position.
P18    at ``dtheta = 0`` the rung-1 corrections are exact identities, so rung 0
       is a branch of the rung-1 code path rather than a fork.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.likelihood.latent_q import (
    LatentQPlan,
    footprint_row_map,
    interp_b,
    latent_logq_at,
    latent_logq_rows,
    moments_at,
    on_footprint_mask,
    rho_from_moments,
    theta_shift,
)
from darksirens.redshift.completion import (
    _LATENT_LOGQ_CLIP,
    _LOGQ_CLIP,
    _member_q_eff_from_logq,
)
from darksirens.redshift.grid import zgrid
from darksirens.redshift.latent_field import build_latent_basis, sky_moments
from darksirens.redshift.prior import (
    eval_dark_member_completion,
    eval_dark_member_completion_latent,
)

N_GRID = int(zgrid.size)
Z_DEPTH = 0.30
M_SPH, M_Z = 24, 6
N_FIT, N_ROWS, M_DRAW = 40, 64, 3
#: The builder's shipped ``b_GW`` grid (``--n-b-nodes 33``, ``--b-max 4``).
#: The node count is not cosmetic: see
#: ``test_budget_identity_off_node_is_bought_by_the_node_count``.
N_B, B_MAX = 33, 4.0
#: The production anchor's moment scale, measured on ``latent_anchor_v2a.h5``:
#: 30,470 fitted-footprint pixels carrying a field COHERENT across that
#: footprint, at a level that puts ``A(z; b) = sum_{p in F} e^{b f}`` at 2.97e4
#: for ``b = 0`` and 2.37e19 for ``b = 4``.  ``N_FOOT_PROD_LEVEL`` is set from
#: that pair -- ``4 * level ~ ln(2.37e19 / 2.97e4)`` less the spread's Jensen
#: term -- and reproduces it to 3.047e4 / 2.366e19.  The SPREAD is small on
#: purpose: it is the coherence that keeps ``log A`` free of a freezing
#: crossover inside ``[0, B_MAX]``.
N_FOOT_PROD, N_FOOT_PROD_LEVEL, N_FOOT_PROD_SPREAD = 30470, 7.82, 0.6
#: The real anchor's per-mode field amplitude ``||xi_hat||/sqrt(M) = 2.46``
#: (experiments/field_level_plan/pr3/REPORT.md).  The 0.4 default above is a
#: sixth of it, and at 0.4 the consumer clip below never engages -- which is
#: exactly why the unclipped eq. (4) pins could not see the clip defect.
PROD_AMP = 2.46
#: The top of the shipped ``b_miss`` prior, ``U(0, 3)``.
PROD_B = 3.0
#: The shipped anchor's footprint size and depth-map maximum
#: (experiments/field_level_plan/pr5/latent_anchor_v2a.h5, ``completeness``:
#: 30,470 pixels, f_p in [0.001, 0.9558]).
DESI_P_F, DESI_F_P_MAX = 30470.0, 0.9558


# --------------------------------------------------------------------- fixture

def _tiny_plan(seed=7, f_p=None, n_b=N_B, b_max=B_MAX, amp=0.4):
    """A small but STRUCTURALLY complete plan: real basis, real moments.

    The footprint is the first ``N_FIT`` of ``N_ROWS`` catalog rows, so the
    remaining rows exercise the off-footprint branch the production run spends
    ~38% of its gathers in.

    ``amp`` is the per-mode field amplitude ``||xi||/sqrt(M)``.  The default
    0.4 is a mild field; ``PROD_AMP`` is the real anchor's.
    """
    rng = np.random.default_rng(seed)
    zg = np.asarray(zgrid)
    below = zg <= Z_DEPTH
    n_sub = int(below.sum())
    z_sub = zg[:n_sub]

    vec = rng.normal(size=(N_FIT, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_DEPTH, ls_sph=0.6, ls_z=0.12)

    if f_p is None:
        f_p = rng.uniform(0.5, 1.0, size=N_FIT)
    f_p = np.asarray(f_p, dtype=float)

    xi = rng.normal(size=(M_DRAW, M_SPH * M_Z)) * float(amp)
    row_fac_fit = np.stack([
        np.asarray(basis.phi_sph @ x.reshape(M_SPH, M_Z)) for x in xi
    ]).astype(np.float32)

    k = np.arange(n_b)
    b_nodes = 0.5 * b_max * (1.0 - np.cos(np.pi * k / (n_b - 1)))
    # Exactly as the builder does it: the moments must see the SAME (f32) row
    # factors the seam consumes, or eq. (4) closes only to ~b|f|eps_f32.
    A_sub, B_sub = sky_moments(basis, xi, b_nodes, f_p, row_fac=row_fac_fit)
    A = np.zeros((M_DRAW, n_b, N_GRID))
    B = np.zeros_like(A)
    A[:, :, :n_sub] = np.asarray(A_sub)
    B[:, :, :n_sub] = np.asarray(B_sub)

    phi_z = np.zeros((N_GRID, M_Z))
    phi_z[:n_sub] = np.asarray(basis.phi_z_out)

    pad = np.zeros((M_DRAW, 1, M_Z), dtype=np.float32)
    row_fac = np.concatenate([row_fac_fit, pad], axis=1)

    plan = LatentQPlan(
        phi_z=jnp.asarray(phi_z), below_depth=jnp.asarray(below),
        row_fac=jnp.asarray(row_fac), A=jnp.asarray(A), B=jnp.asarray(B),
        b_nodes=jnp.asarray(b_nodes), P_F=float(N_FIT), F_F=float(f_p.sum()),
        m_sph=M_SPH, m_z=M_Z)
    # Catalog rows 0..N_FIT-1 are the footprint; the rest are outside it.
    row_map = footprint_row_map(np.arange(N_ROWS), np.arange(N_FIT), N_FIT)
    return plan, row_map, f_p


def _c_curve(scale=1.0):
    """A smooth in-range completeness curve, zero above the depth."""
    zg = np.asarray(zgrid)
    c = scale * 0.6 * np.exp(-((zg / 0.22) ** 2))
    return jnp.asarray(np.where(zg <= Z_DEPTH, c, 0.0))


def _rho(plan, m, c, b):
    return rho_from_moments(plan.A[m], plan.B[m], c, b, plan.b_nodes,
                            plan.P_F, plan.F_F, plan.below_depth)


# ------------------------------------------------------------------- P13b

def test_p13b_off_footprint_logq_is_bit_zero():
    """Every off-footprint row returns EXACTLY 0.0 log-Q, at every z."""
    plan, row_map, _ = _tiny_plan()
    c, b = _c_curve(), 1.3
    on_fp = on_footprint_mask(row_map, N_FIT)
    for m in range(M_DRAW):
        rho = _rho(plan, m, c, b)
        rows = plan.row_fac[m][jnp.asarray(row_map)]
        lq = np.asarray(latent_logq_rows(plan, rows, rho, b, on_fp))
        off = lq[N_FIT:]
        assert off.shape == (N_ROWS - N_FIT, N_GRID)
        # Bit-zero, not merely small: the identity needs Q == 1 exactly.
        assert np.all(off == 0.0)
        # ... and the footprint block is NOT trivially zero (else the pin is
        # vacuous).
        assert np.any(np.abs(lq[:N_FIT]) > 1e-6)


def test_above_depth_logq_is_bit_zero():
    """Above ``z_depth`` the seam is bit-zero for every row -- the relaxation
    expressed in the basis (zero ``phi_z`` rows, zero ``rho``) rather than as a
    downstream mask, so no NaN/overflow can be produced there and then masked."""
    plan, row_map, _ = _tiny_plan()
    on_fp = on_footprint_mask(row_map, N_FIT)
    rho = _rho(plan, 0, _c_curve(), 1.7)
    rows = plan.row_fac[0][jnp.asarray(row_map)]
    lq = np.asarray(latent_logq_rows(plan, rows, rho, 1.7, on_fp))
    above = ~np.asarray(plan.below_depth)
    assert np.all(lq[:, above] == 0.0)
    assert np.isfinite(lq).all()


# ------------------------------------------------------- the eq. (4) identity

def _budget_residual(plan, row_map, f_p, b):
    """max_z |sum_p (1 - f_p C) Q_p / sum_p (1 - f_p C) - 1| over all members."""
    c = _c_curve()
    on_fp = on_footprint_mask(row_map, N_FIT)
    w_fit = 1.0 - np.asarray(f_p)[:, None] * np.asarray(c)[None, :]
    worst = 0.0
    for m in range(plan.n_draw):
        rho = _rho(plan, m, c, b)
        rows = plan.row_fac[m][jnp.asarray(row_map)]
        Q = np.exp(np.asarray(latent_logq_rows(plan, rows, rho, b, on_fp)))
        got = (w_fit * Q[:N_FIT]).sum(axis=0)
        worst = max(worst, float(np.abs(got / w_fit.sum(axis=0) - 1.0).max()))
    return worst


@pytest.mark.parametrize("b", [0.0, 0.37, 1.11, 2.5, 3.77, 4.0])
def test_budget_identity_eq4(b):
    """``sum_p (1 - f_p C) Q_p == sum_p (1 - f_p C)`` over the FULL sky.

    Split as PLAN §4.2 does: the footprint block carries the generated ``Q``,
    the off-footprint block carries ``Q == 1`` and is conserved trivially.  The
    identity must hold at every z, every member and every ``b_GW`` -- it is
    what ``rho`` is defined to enforce, and the +55% Jensen inflation the table
    path measured is removed by construction, not by a fitted correction.

    Machine-exact, NOT a fitted tolerance.  Two things buy that, and this pin
    catches the loss of either: the builder computes the moments from the same
    f32 row factors the seam consumes (else a ``b|f|eps_f32`` residual,
    measured 2.7e-7 at the production corner), and the shipped 33-node
    Chebyshev ``b`` grid resolves the moments spectrally (see the companion
    test).  Both node and off-node ``b`` values are exercised.
    """
    plan, row_map, f_p = _tiny_plan()
    assert _budget_residual(plan, row_map, f_p, b) < 1e-13


def test_budget_identity_off_node_is_bought_by_the_node_count():
    """Between ``b`` nodes the identity is only as exact as the interpolation.

    At the shipped 33 nodes it closes at machine precision; at 9 nodes over the
    same range it degrades by ~8 orders of magnitude.  Recorded because the
    node count is the ONLY thing standing between eq. (4) and an approximate
    budget at a sampled (hence generically off-node) ``b_GW`` -- the moment
    table's ``n_b`` is a correctness parameter, not a resolution nicety.
    """
    b_off = 0.37
    fine, _, f_fine = _tiny_plan(n_b=33, b_max=4.0)
    coarse, row_map, f_coarse = _tiny_plan(n_b=9, b_max=4.0)
    r_fine = _budget_residual(fine, row_map, f_fine, b_off)
    r_coarse = _budget_residual(coarse, row_map, f_coarse, b_off)
    print(f"\n[eq.4 off-node b={b_off}] n_b=33: {r_fine:.2e}   n_b=9: {r_coarse:.2e}")
    assert r_fine < 1e-13
    assert r_coarse > 100 * r_fine


# ------------------------------- eq. (4) THROUGH the clipped consumer path

def _prod_plan():
    """The fixture at the real anchor's amplitude, on a ``b`` grid whose TOP
    node is ``PROD_B``.

    ``b_max = PROD_B`` puts the sampled ``b`` exactly on a node, so
    :func:`interp_b` takes its exact branch and the residual measured below is
    the CLIP's alone -- the off-node interpolation error is a separate axis,
    already pinned by ``test_budget_identity_off_node_is_bought_by_the_node_count``.
    """
    return _tiny_plan(amp=PROD_AMP, b_max=PROD_B)


def _consumed_budget_ratio(plan, row_map, f_p, b, clip):
    """``sum_p (1 - f_p C) Q_eff_p / sum_p (1 - f_p C)`` per z, worst member.

    ``Q_eff`` comes from :func:`_member_q_eff_from_logq`, the ONE function every
    latent consumer routes through (``latent_member_N_miss_integrals``,
    ``latent_posterior_mean_q``, ``field_global_log_Z_members`` and the
    numerator's ``eval_dark_member_completion_latent``), so this measures the
    budget the run actually consumes rather than the expression it would consume
    with no clip at all.
    """
    c = _c_curve()
    on_fp = on_footprint_mask(row_map, N_FIT)
    below = np.asarray(plan.below_depth)
    w_fit = 1.0 - np.asarray(f_p)[:, None] * np.asarray(c)[None, :]
    den = w_fit.sum(axis=0)
    worst = 0.0
    for m in range(plan.n_draw):
        rho = _rho(plan, m, c, b)
        rows = plan.row_fac[m][jnp.asarray(row_map)]
        lq = latent_logq_rows(plan, rows, rho, b, on_fp)
        q = np.asarray(_member_q_eff_from_logq(
            lq, plan.below_depth, True, clip))
        got = (w_fit * q[:N_FIT]).sum(axis=0)
        worst = max(worst, float(np.abs(got[below] / den[below] - 1.0).max()))
    return worst


def test_eq4_survives_the_consumer_clip_at_production_amplitude():
    """eq. (4) must hold THROUGH the clip the consumers actually apply.

    The other eq. (4) pins evaluate ``exp(logQ)`` directly on a 0.4-amplitude
    fixture, where ``|logQ| < 7`` everywhere -- so they are blind to the bound
    ``_member_q_eff_from_logq`` imposes.  At the real anchor's amplitude
    (``PROD_AMP``) and the top of the ``b_miss`` prior (``PROD_B``) most rows
    are outside ``+/-_LOGQ_CLIP``, and clipping there does not tame a tail: it
    hands back missing hosts that ``rho`` had already spent, so the consumed
    budget stops being the budget the normalizer conserves.

    The second assertion is what keeps this pin from going quietly vacuous: the
    TABLE bound must still fail the same tolerance on the same fixture.  If a
    future fixture change makes it pass, this test has stopped measuring the
    thing it was written for.
    """
    plan, row_map, f_p = _prod_plan()
    lat = _consumed_budget_ratio(plan, row_map, f_p, PROD_B, _LATENT_LOGQ_CLIP)
    tab = _consumed_budget_ratio(plan, row_map, f_p, PROD_B, _LOGQ_CLIP)
    print(f"\n[eq.4 through the consumer, amp={PROD_AMP} b={PROD_B}] "
          f"latent bound {_LATENT_LOGQ_CLIP}: {lat:.2e}   "
          f"table bound {_LOGQ_CLIP}: {tab:.2e}")
    assert lat < 1e-6
    assert tab > 1e-6


def test_latent_bound_clears_the_identity_bound_at_the_production_footprint():
    """The latent bound is numerical safety; the upper rail must not engage.

    eq. (4) bounds the POSITIVE tail on its own.  Every ``w_p = 1 - f_p C`` is
    positive and every ``Q_p`` non-negative, so a single row's term cannot
    exceed the whole sum:

        w_p Q_p <= sum_q w_q Q_q == sum_q w_q   =>   logQ_p <= log(sum_q w_q / w_p),

    which is what the first assertion measures on the fixture.  Evaluated at the
    SHIPPED footprint (pr5/latent_anchor_v2a.h5: 30,470 pixels, ``f_p`` in
    [0.001, 0.9558], so ``w >= 1 - 0.9558`` at ``C == 1``) that bound is ~13.4:
    ABOVE ``_LOGQ_CLIP``, which is the structural statement of the defect --
    the table rail sits inside the range the identity legitimately produces, so
    it deletes budget instead of taming a tail.  ``_LATENT_LOGQ_CLIP`` must sit
    above it, leaving only the inert lower rail reachable.
    """
    plan, row_map, f_p = _prod_plan()
    c = _c_curve()
    on_fp = on_footprint_mask(row_map, N_FIT)
    below = np.asarray(plan.below_depth)
    w_fit = 1.0 - np.asarray(f_p)[:, None] * np.asarray(c)[None, :]
    hi = float(np.log(w_fit.sum(axis=0)[below] / w_fit[:, below].min()).max())
    worst_pos, worst_abs = -np.inf, 0.0
    for m in range(plan.n_draw):
        rows = plan.row_fac[m][jnp.asarray(row_map)]
        lq = np.asarray(latent_logq_rows(
            plan, rows, _rho(plan, m, c, PROD_B), PROD_B, on_fp))[:N_FIT, below]
        worst_pos = max(worst_pos, float(lq.max()))
        worst_abs = max(worst_abs, float(np.abs(lq).max()))
    assert worst_pos <= hi + 1e-9
    # ... and the fixture is genuinely outside the TABLE bound.
    assert worst_abs > _LOGQ_CLIP

    desi = float(np.log(DESI_P_F / (1.0 - DESI_F_P_MAX)))
    print(f"\n[bounds] fixture max logQ {worst_pos:.2f} <= identity bound "
          f"{hi:.2f}; DESI-footprint identity bound {desi:.2f}; "
          f"table {_LOGQ_CLIP}, latent {_LATENT_LOGQ_CLIP}")
    assert desi > _LOGQ_CLIP
    assert _LATENT_LOGQ_CLIP > desi


def test_numerator_hot_path_returns_the_unclipped_Q():
    """The per-event numerator carries the same ``Q``, not a railed one.

    ``eval_dark_member_completion_latent`` is the two-node gather every PE and
    injection sample goes through.  Driven with ``A_obs = -inf``,
    ``base_miss == 1`` and ``log_Z == 0`` it returns exactly ``log Q_eff``, so
    the clip it applies is measurable rather than inferred: at production
    amplitude the railed evaluator overstates ``Q`` on the deeply underdense
    rows by tens of nats, which is a per-event distortion of the missing-host
    density even where the aggregate budget nearly closes.
    """
    plan, row_map, f_p = _prod_plan()
    c = _c_curve()
    below = np.asarray(plan.below_depth)
    nodes = np.arange(N_GRID - 1)[below[:-1]]
    rows_idx, node_idx = np.meshgrid(np.arange(N_FIT), nodes, indexing="ij")
    pix = jnp.asarray(rows_idx.reshape(-1), dtype=jnp.int32)
    idx = jnp.asarray(node_idx.reshape(-1), dtype=jnp.int32)
    t = jnp.zeros(pix.shape, dtype=float)              # land ON the low node
    A_obs = jnp.full(pix.shape, -jnp.inf)
    base_miss = jnp.ones((N_ROWS, N_GRID))
    logZ = jnp.zeros(N_ROWS)
    on_fp = on_footprint_mask(row_map, N_FIT)
    pix_fit = jnp.asarray(row_map)[pix]
    on_fp_s = on_fp[pix]

    m = 0
    rho = _rho(plan, m, c, PROD_B)
    got = np.asarray(eval_dark_member_completion_latent(
        A_obs, idx, t, pix, pix_fit, on_fp_s, base_miss, plan.row_fac[m],
        plan.phi_z, rho, PROD_B, logZ, Z_DEPTH, False))
    want = np.asarray(latent_logq_rows(
        plan, plan.row_fac[m][jnp.asarray(row_map)], rho, PROD_B, on_fp)
    )[rows_idx.reshape(-1), node_idx.reshape(-1)]
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    # The old bound would have railed a majority of these samples.
    railed = float((np.abs(want) > _LOGQ_CLIP).mean())
    print(f"\n[numerator] {railed:.1%} of (row, node) samples exceed "
          f"+/-{_LOGQ_CLIP}; worst overstatement under it "
          f"{float(np.max(np.clip(want, -_LOGQ_CLIP, _LOGQ_CLIP) - want)):.1f} nats")
    assert railed > 0.25


def test_rho_vanishes_on_a_zero_field():
    """A zero field gives ``rho == 0`` and ``Q == 1`` identically -- the
    property P16 turns into a physics gate (and the sanity check that ``rho``
    is a normalizer, not an offset)."""
    plan, row_map, _ = _tiny_plan()
    zero = LatentQPlan(**{**plan.__dict__,
                          "row_fac": jnp.zeros_like(plan.row_fac)})
    # A(z; b) = |F| and B(z; b) = sum f_p when f == 0, for every b.
    A = jnp.broadcast_to(jnp.where(plan.below_depth, plan.P_F, 0.0),
                         plan.A.shape)
    B = jnp.broadcast_to(jnp.where(plan.below_depth, plan.F_F, 0.0),
                         plan.B.shape)
    rho = rho_from_moments(A[0], B[0], _c_curve(), 0.9, plan.b_nodes,
                           plan.P_F, plan.F_F, plan.below_depth)
    np.testing.assert_allclose(np.asarray(rho), 0.0, atol=1e-13)
    on_fp = on_footprint_mask(row_map, N_FIT)
    rows = zero.row_fac[0][jnp.asarray(row_map)]
    lq = np.asarray(latent_logq_rows(zero, rows, rho, 0.9, on_fp))
    np.testing.assert_allclose(lq, 0.0, atol=1e-13)


def test_interp_b_is_exact_at_nodes():
    """P9's online half: barycentric interpolation reproduces the moment table
    exactly at its own nodes (the pole branch), so a run at a node ``b_GW``
    consumes the built moments rather than an interpolant of them."""
    plan, _, _ = _tiny_plan()
    for i, b in enumerate(np.asarray(plan.b_nodes)):
        got = np.asarray(interp_b(plan.A[0], plan.b_nodes, float(b)))
        np.testing.assert_allclose(got, np.asarray(plan.A[0][i]), rtol=0, atol=0)


# ------------------------------------------------- the seam vs the table path

def _table_from_plan(plan, row_map, c, b, *, kernel):
    """The ``(M, N_rows, N_grid)`` log-Q table the seam would generate.

    Built ONLY so the two evaluators can be compared; the production latent
    path never forms this object (1.06 GB at ``M_draw = 8`` and DESI scale).

    ``kernel="gather"`` reproduces the HOT path's reduction exactly -- an
    elementwise product summed over ``M_z`` -- by broadcasting rows against
    nodes.  ``kernel="rows"`` uses :func:`latent_logq_rows`, whose ``(N_rows,
    M_z) @ (M_z, N_grid)`` matmul is the only affordable form at production
    scale but is a DIFFERENT reduction, so it agrees with the gather to
    floating-point re-association rather than bitwise.
    """
    on_fp = on_footprint_mask(row_map, N_FIT)
    out = []
    for m in range(plan.n_draw):
        rows = plan.row_fac[m][jnp.asarray(row_map)]
        rho = _rho(plan, m, c, b)
        if kernel == "rows":
            out.append(latent_logq_rows(plan, rows, rho, b, on_fp))
        else:
            out.append(latent_logq_at(
                rows[:, None, :], plan.phi_z[None, :, :], rho[None, :], b,
                on_fp[:, None]))
    return jnp.stack(out)


def test_seam_row_block_matches_the_gather_kernel():
    """The state-prep row block and the hot-path gather compute the same field.

    They cannot be bitwise equal -- one is a matmul over ``M_z``, the other an
    elementwise sum -- and the matmul is not negotiable at production scale
    (a broadcast ``(N_rows, N_grid, M_z)`` intermediate would be 3.4 TB at
    ``M_z = 12``).  Pin the agreement at floating point instead, so a genuine
    divergence between the normalizer's ``Q`` and the numerator's ``Q`` is
    still caught.
    """
    plan, row_map, _ = _tiny_plan()
    c, b = _c_curve(), 1.15
    a = np.asarray(_table_from_plan(plan, row_map, c, b, kernel="gather"))
    r = np.asarray(_table_from_plan(plan, row_map, c, b, kernel="rows"))
    np.testing.assert_allclose(r, a, rtol=0.0, atol=1e-14)


def test_seam_is_bit_identical_to_the_table_evaluator():
    """The ONE substitution, verified: feed the table evaluator the table the
    seam generates and the two return bit-identical values, member by member.

    This is what licenses reusing every downstream pin (the goldens, the
    marginalization tests) rather than re-deriving them for latent mode: the
    latent evaluator differs from the table evaluator in the SOURCE of
    ``logQ`` and in nothing else -- same depth relaxation, same two-node
    interpolation, same ``logaddexp``, same normalizer selection.

    The two evaluators DO carry different clip bounds, so this pin is stated on
    the mild-amplitude fixture, where ``|logQ| < _LOGQ_CLIP`` everywhere and
    neither bound engages; the assertion below records that.  What the seam is
    licensed to reuse is the evaluator, not the table path's bound.
    """
    plan, row_map, _ = _tiny_plan()
    c, b = _c_curve(), 1.15
    rng = np.random.default_rng(3)
    nsamp = 128
    z = jnp.asarray(rng.uniform(0.01, 0.55, size=nsamp))
    pix = jnp.asarray(rng.integers(0, N_ROWS, size=nsamp), dtype=jnp.int32)
    idx = jnp.clip(jnp.searchsorted(zgrid, z, side="right") - 1, 0, N_GRID - 2)
    t = jnp.clip((z - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx]), 0.0, 1.0)
    A_obs = jnp.asarray(rng.normal(size=nsamp))
    base_miss = jnp.asarray(np.abs(rng.normal(size=(N_ROWS, N_GRID))) + 1e-3)
    logZ = jnp.asarray(rng.normal(size=(plan.n_draw, N_ROWS)) + 5.0)

    table = _table_from_plan(plan, row_map, c, b, kernel="gather")
    assert float(np.max(np.abs(np.asarray(table)))) < _LOGQ_CLIP
    # The evaluator takes the footprint row index and mask PER SAMPLE: they are
    # member-independent, so production hoists them into the same precompute
    # that produces A_obs/idx/t rather than regathering them M times.
    on_fp = on_footprint_mask(row_map, N_FIT)
    pix_fit = jnp.asarray(row_map)[pix]
    on_fp_s = on_fp[pix]

    for m in range(plan.n_draw):
        want = eval_dark_member_completion(
            A_obs, idx, t, pix, base_miss, table[m], True, logZ[m],
            Z_DEPTH, False)
        got = eval_dark_member_completion_latent(
            A_obs, idx, t, pix, pix_fit, on_fp_s, base_miss,
            plan.row_fac[m], plan.phi_z, _rho(plan, m, c, b), b,
            logZ[m], Z_DEPTH, False)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# ---------------------------------------------------------------------- P16

def test_p16_complete_catalog_limit_is_member_exact():
    """PLAN §1.6 Limit I: at ``C == 1``, ``f_p == 1`` the missing branch
    vanishes and the prior is the observed host spikes -- **for every member
    and every field**, so ``logsumexp_m ll_m - log M == ll`` exactly.

    Here in the evaluator's own terms: with ``base_miss == 0`` below the depth
    the completion is independent of the member index, bit-identically.  The
    field is NOT small and ``b_GW`` is NOT zero -- the identity is structural.
    """
    plan, row_map, _ = _tiny_plan(f_p=np.ones(N_FIT))
    c, b = jnp.asarray(np.where(np.asarray(zgrid) <= Z_DEPTH, 1.0, 0.0)), 1.9
    rng = np.random.default_rng(11)
    nsamp = 96
    z = jnp.asarray(rng.uniform(0.01, Z_DEPTH - 1e-3, size=nsamp))
    pix = jnp.asarray(rng.integers(0, N_ROWS, size=nsamp), dtype=jnp.int32)
    idx = jnp.clip(jnp.searchsorted(zgrid, z, side="right") - 1, 0, N_GRID - 2)
    t = jnp.clip((z - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx]), 0.0, 1.0)
    A_obs = jnp.asarray(rng.normal(size=nsamp))
    # C == 1 below the depth => base_miss == (1 - C) dN_exp == 0 there.
    zg = np.asarray(zgrid)
    base = np.broadcast_to(np.where(zg <= Z_DEPTH, 0.0, 1.0), (N_ROWS, N_GRID))
    base_miss = jnp.asarray(np.array(base))
    logZ = jnp.asarray(np.full(N_ROWS, 4.0))
    on_fp = on_footprint_mask(row_map, N_FIT)
    pix_fit = jnp.asarray(row_map)[pix]
    on_fp_s = on_fp[pix]

    vals = [
        np.asarray(eval_dark_member_completion_latent(
            A_obs, idx, t, pix, pix_fit, on_fp_s, base_miss,
            plan.row_fac[m], plan.phi_z, _rho(plan, m, c, b), b,
            logZ, Z_DEPTH, False))
        for m in range(plan.n_draw)
    ]
    for v in vals[1:]:
        np.testing.assert_array_equal(v, vals[0])
    # The member average is then the single-member value, exactly.
    stack = np.stack(vals)
    lse = np.log(np.mean(np.exp(stack - stack.max(axis=0)), axis=0)) + stack.max(axis=0)
    np.testing.assert_allclose(lse, vals[0], rtol=0, atol=1e-13)


# ---------------------------------------------------------------------- P18

def test_p18_zero_dtheta_is_an_exact_identity():
    """Rung 0 is the ``dtheta = 0`` branch of the rung-1 code path, so the
    control arm is a flag and not a fork: the shift is exact zeros and the
    corrected moments are the uncorrected ones, bit-identically."""
    plan, _, _ = _tiny_plan()
    n_th = 4
    S = jnp.asarray(np.random.default_rng(5).normal(size=(M_SPH * M_Z, n_th)))
    dA = jnp.asarray(np.random.default_rng(6).normal(size=plan.A.shape + (n_th,)))
    shift = theta_shift(S, jnp.zeros(n_th), M_SPH, M_Z)
    assert shift.shape == (M_SPH, M_Z)
    assert np.all(np.asarray(shift) == 0.0)
    A2, B2 = moments_at(plan.A, plan.B, dA, dA, jnp.zeros(n_th))
    np.testing.assert_array_equal(np.asarray(A2), np.asarray(plan.A))
    np.testing.assert_array_equal(np.asarray(B2), np.asarray(plan.B))


def test_theta_shift_is_member_independent_and_o_m():
    """The rung-1 correction is ``(M_sph, M_z)`` -- it carries NO member axis
    and NO row axis.  A ``row_fac`` that acquired a theta index would be the
    per-proposal re-solve PLAN §10 refuses (3.0-24 GB at 256 concurrency); this
    pin is the shape-level statement of that refusal."""
    n_th = 3
    S = jnp.asarray(np.random.default_rng(9).normal(size=(M_SPH * M_Z, n_th)))
    shift = theta_shift(S, jnp.asarray([0.1, -0.2, 0.05]), M_SPH, M_Z)
    assert shift.shape == (M_SPH, M_Z)
    assert shift.size == M_SPH * M_Z


# ------------------------------------------------------------------ row map

def test_footprint_row_map_handles_unsorted_and_missing_pixels():
    """The map is by PIXEL ID, not by position: a catalog whose rows are an
    arbitrary superset of the footprint in arbitrary order must still land each
    row on its own footprint entry, and everything else on the pad row."""
    fit = np.array([10, 3, 77, 5])
    rows = np.array([77, 0, 3, 999, 10, 5, 4])
    m = footprint_row_map(rows, fit, fit.size)
    assert m.tolist() == [2, 4, 1, 4, 0, 3, 4]
    mask = np.asarray(on_footprint_mask(m, fit.size))
    assert mask.tolist() == [True, False, True, False, True, True, False]


# ------------------------------------------ moment interpolation dynamic range

def test_moment_interpolation_survives_the_production_dynamic_range():
    """``rho`` stays finite and ``A - cB`` stays POSITIVE at small ``b_GW``.

    ``A(z; b) = sum_{p in F} e^{b f}`` spans **14.9 orders of magnitude** across
    the shipped b-node range on the real anchor, and a polynomial interpolant of
    a function with that dynamic range oscillates at the LOW end.  Measured on
    ``latent_anchor_v2a.h5`` before the fix: linear-space barycentric
    interpolation returned ``A = -9.69e7`` at ``b_GW = 0.37`` where the true
    value is ``+3.07e4`` (bracketed by +2.97e4 at b = 0.1 and +3.20e4 at 0.5),
    so ``A - cB`` went negative, the ``1e-300`` floor caught it, and ``rho`` was
    silently garbage.  ``b_GW`` is SAMPLED, so the sampler visits that region.

    The fixture is the production moment scale, not a scaled-down cartoon: the
    footprint is the anchor's own 30,470 pixels and the field sits at the
    anchor's own level, which puts ``A(b = 0) = 3.047e4`` and
    ``A(b = 4) = 2.366e19`` against the artifact's measured 2.97e4 and 2.37e19
    -- 14.89 orders against 14.9.  Four orders less than that and linear
    interpolation survives (the earlier fixture spanned 11.8 and could not go
    red), so the range is a fixture INVARIANT, asserted below.

    Two properties of the fixture carry the discrimination and neither is
    cosmetic:

    * ``f_p`` VARIES pixel to pixel, so ``B`` is not a multiple of ``A``.  With
      a constant ``f_p`` the moments are exactly proportional, ``A - cB``
      collapses to ``(P_F - c F_F)/P_F * A``, and "num > 0" degenerates into
      "A_interp > 0" -- a strictly weaker statement than the one this pin
      exists to make.
    * the field is COHERENT across the footprint (one level, a small
      pixel-to-pixel spread) rather than a wide zero-mean scatter.  That is what
      the anchor looks like, and it is why ``log A`` has no freezing crossover
      inside ``[0, b_max]`` and the same 33 nodes reproduce it to ~1e-14 while
      they cannot reproduce ``A`` itself at all.

    The pin is that ``num > 0`` STRUCTURALLY (``B <= A`` and ``c <= 1``) and
    that ``rho`` matches the direct footprint reduction -- never that the
    ``1e-300`` floor rescued it.  The floor cannot be detected by ``num > 0``
    alone (a floored ``rho`` still exponentiates to a positive 1e-300), so the
    floor is caught by a magnitude threshold and by the reduction itself.
    """
    n_b, n_z = N_B, 64
    b_nodes = 0.5 * B_MAX * (1.0 - np.cos(np.pi * np.arange(n_b) / (n_b - 1)))
    rng = np.random.default_rng(11)
    z = np.linspace(0.0, 1.0, n_z)
    # f(p, z) = (level + spread) * mild z ramp, and f_p drawn per pixel.
    f = ((N_FOOT_PROD_LEVEL + N_FOOT_PROD_SPREAD * rng.normal(size=N_FOOT_PROD))
         [:, None] * (0.85 + 0.15 * z)[None, :])            # (n_pix, n_z)
    f_p = rng.uniform(0.3, 1.0, size=N_FOOT_PROD)
    P_F, F_F = float(N_FOOT_PROD), float(f_p.sum())
    # eq. (2) itself, one b node at a time (the cube is never materialized).
    A = np.empty((n_b, n_z))
    B = np.empty((n_b, n_z))
    for i, b_i in enumerate(b_nodes):
        e = np.exp(b_i * f)
        A[i] = e.sum(0)
        B[i] = f_p @ e
    assert np.log10(A.max() / A.min()) > 14.0, "fixture lost its dynamic range"
    ratio = B / A
    assert ratio.max() / ratio.min() > 1.05, "B is proportional to A"

    # C(z) varies with z, as the sky-aggregate completeness curve does.
    c = 0.4 + 0.5 * z
    den = P_F - c * F_F
    w = 1.0 - f_p[:, None] * c[None, :]                     # (n_pix, n_z)
    for b in (0.0, 0.1, 0.37, 0.5, 1.11, 2.5, 3.77, 4.0):
        rho = np.asarray(rho_from_moments(jnp.asarray(A), jnp.asarray(B),
                                          jnp.asarray(c), b,
                                          jnp.asarray(b_nodes), P_F, F_F, None))
        assert np.all(np.isfinite(rho)), f"rho non-finite at b={b}"
        num = np.exp(rho) * den
        assert np.all(num > 0.0), f"A - cB went non-positive at b={b}"
        # The floor is 1e-300; the true minimum of A - cB over this fixture is
        # 1.26e4 (the production anchor's own minimum is 1.70e4), so anything
        # below 1e3 means the floor, not the arithmetic, produced the sign.
        assert num.min() > 1e3, f"the 1e-300 floor is load-bearing at b={b}"
        # ... and against eq. (2) evaluated directly: worst |drho| = 6.8e-14.
        ref = np.log((w * np.exp(b * f)).sum(0) / w.sum(0))
        np.testing.assert_allclose(rho, ref, rtol=0, atol=1e-9)
