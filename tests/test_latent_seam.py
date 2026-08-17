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
       table the seam would generate.  This is the pin that says the seam
       changed the SOURCE of ``Q`` and nothing else: same clip, same depth
       relaxation, same two-node interpolation, same normalizer selection.
eq.(4) the generated ``Q`` conserves the CONSUMED missing budget
       ``sum_p (1 - f_p C(z)) Q_p(z)`` at every ``z``, member and ``b_GW`` --
       the identity ``rho`` is defined to enforce.
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


# --------------------------------------------------------------------- fixture

def _tiny_plan(seed=7, f_p=None, n_b=N_B, b_max=B_MAX):
    """A small but STRUCTURALLY complete plan: real basis, real moments.

    The footprint is the first ``N_FIT`` of ``N_ROWS`` catalog rows, so the
    remaining rows exercise the off-footprint branch the production run spends
    ~38% of its gathers in.
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

    xi = rng.normal(size=(M_DRAW, M_SPH * M_Z)) * 0.4
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
    ``logQ`` and in nothing else -- same clip, same depth relaxation, same
    two-node interpolation, same ``logaddexp``, same normalizer selection.
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
