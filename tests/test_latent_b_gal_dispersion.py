"""PLAN §3.4's ``b_gal`` rank-1 draw-covariance inflation (closure finding S-2).

Through PR-6a the member ensemble was ``N(xi_hat, H^{-1})`` at FIXED
``b_gal``: ``laplace_draws`` took ``H_chol`` and nothing else, and the
``b_gal`` column of ``sensitivity_S`` was built, stored and never consumed.
PLAN §3.4 asks for

    Cov(xi) = H^{-1} + s_b^2 v v^T,   v = d xi_hat/d b = -H^{-1}(d grad/d b)

with ``s_b`` the PROFILE curvature ``[-d^2 log p_count/db^2]^{-1/2}`` at the
anchor -- [v4, §0.5 finding 11] withdrew v3's "20% prior width", because §4.3
pins ``amp = 1`` and the counts therefore MEASURE ``b_gal``; a free dial would
let PLAN §6.2's Tier-B criterion "latent-on CI >= table CI" be chosen instead
of measured.  These pins are what make that criterion non-vacuous.

What is pinned here:

D1  ``s_b`` is the profile, not the conditional slice: the returned profile
    curvature equals a direct finite-difference of the RE-SOLVED (profiled)
    objective ``J(xi_hat(b); b)`` in ``b``, and is strictly smaller than the
    conditional curvature at fixed ``xi``.
D2  The systematics floor only ever RAISES ``s_b`` and reports which bound won.
D3  Feature OFF is BIT-IDENTICAL to the pre-S-2 function -- same key, same
    members, to the last bit -- so every shipped anchor and every existing pin
    is reproducible with the inflation off.
D4  Feature ON is ADDITIVE: ``xi_m`` moves by exactly ``s_b eps_m v`` and by
    nothing else.
D5  The drawn covariance is ``H^{-1} + s_b^2 v v^T`` to Monte-Carlo error at
    large ``M``, checked IN THE RANK-1 DIRECTION: the variance along ``v``
    inflates by the predicted factor while the variance along a
    ``H^{-1}``-orthogonal companion direction does not move at all (that one is
    exact, not statistical -- the rank-1 term contributes zero to it by
    construction, and the SAME ``g`` stream is used with the feature on and
    off, so the comparison is a bit-level identity).
D6  Antithetic structure holds in BOTH sources, and the balanced member
    ordering PR-5b ships (``[0, M/2, 1, M/2+1, ...]``) has every even prefix
    balanced in ``eps`` as well as in ``g``.  A naive unpaired prefix fails
    P14 at ``M_draw = 4`` and 8; a second unpaired source would put that back.
D7  The scale needs a direction: ``s_b`` without ``v_b`` is refused.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift.latent_counts import (
    B_GAL_SYSTEMATIC_FLOOR_FRAC,
    CountOperator,
    TracerCounts,
    b_gal_profile_sigma,
    count_map_solve,
    dgrad_db,
    laplace_draws,
    make_count_operator,
    objective,
    sensitivity,
)
from darksirens.redshift.latent_field import build_latent_basis, shell_response

M_SPH, M_Z, G_S, N_FIT = 12, 3, 5, 40
Z_HI = 0.3


def _fixture(seed=0, bias=1.3):
    """The ``test_latent_counts`` fixture, verbatim, so the two files pin the
    same operator and a change in one is visible in the other."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_fine = np.linspace(1e-3, Z_HI, 120)
    basis = build_latent_basis(
        v, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.6, ls_z=0.15, zeta_fine=np.log1p(z_fine))
    edges = np.linspace(0.02, Z_HI, G_S + 1)
    W = shell_response(edges, z_fine,
                       lambda z: 0.02 * np.ones_like(z),
                       lambda z: z ** 2 + 1e-6)
    counts = rng.poisson(30.0, size=(N_FIT, G_S)).astype(float)
    f_p = rng.uniform(0.5, 1.0, size=N_FIT)
    tracer = TracerCounts(pix=np.arange(N_FIT), counts=counts,
                          completeness=f_p, bias=bias)
    return make_count_operator(basis.phi_sph, basis.phi_z_fine, W, tracer)


def _anchor(seed=0, bias=1.3):
    op = _fixture(seed, bias)
    sol = count_map_solve(op)
    assert float(sol["grad_inf"]) < 1e-8
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    gb = dgrad_db(xi_hat, op)
    v = np.asarray(sensitivity(xi_hat, L, gb[:, None]))[:, 0]
    return op, np.asarray(xi_hat), np.asarray(L), np.asarray(gb), v


def _legacy_draws(xi_hat, H_chol, n_draw, key):
    """``laplace_draws`` EXACTLY as it stood before S-2 (PR-6a, f45be7c).

    Copied rather than imported: the point of D3 is that the shipped anchor is
    reproducible bit-for-bit, and a pin that calls the new code with a flag
    cannot distinguish "unchanged" from "changed in both branches"."""
    if n_draw % 2:
        raise ValueError("laplace_draws: n_draw must be even (antithetic).")
    M = xi_hat.shape[0]
    g_half = jax.random.normal(key, (n_draw // 2, M))
    g = jnp.concatenate([g_half, -g_half], axis=0)
    steps = jax.scipy.linalg.solve_triangular(H_chol.T, g.T, lower=False).T
    return jnp.asarray(xi_hat)[None, :] + steps, g


# --------------------------------------------------------------- D1, D2: s_b

def test_d1_s_b_is_the_profile_curvature_not_the_conditional_slice():
    op, xi_hat, L, gb, v = _anchor(seed=1)
    prof = b_gal_profile_sigma(xi_hat, op, dgrad_b=gb, v_b=v,
                               systematic_floor_frac=0.0)

    # The profile itself: RE-SOLVE at b +/- db and difference J(xi_hat(b); b).
    # This is the definition, computed the expensive way; the shipped function
    # gets the same number from one 1-D second derivative plus the b_gal
    # column that sensitivity() has already built (PLAN §3.4).
    db = 2e-3

    def _profiled(b):
        op_b = CountOperator(op.proj_sph, op.phi_shell, op.counts, op.log_fp, b)
        xb = count_map_solve(op_b, xi0=xi_hat)["xi_hat"]
        return float(objective(xb, op_b))

    p0, pp, pm = _profiled(op.bias), _profiled(op.bias + db), \
        _profiled(op.bias - db)
    fd = (pp - 2 * p0 + pm) / db ** 2
    assert np.isclose(prof["curvature_profile"], fd, rtol=5e-4), (
        f"profile curvature {prof['curvature_profile']:.8e} vs re-solved "
        f"finite difference {fd:.8e}")

    # The profile is BROADER than the conditional slice by the field's ability
    # to absorb a change in b: P'' = J_bb - J_bxi^T H^{-1} J_bxi, and the
    # subtracted term is a positive quadratic form.
    assert prof["curvature_profile"] < prof["curvature_conditional"]
    assert np.isclose(
        prof["curvature_conditional"] - prof["curvature_profile"],
        float(gb @ np.linalg.solve(L @ L.T, gb)), rtol=1e-8)
    assert np.isclose(prof["s_b"], prof["curvature_profile"] ** -0.5,
                      rtol=1e-12)

    # It is a MEASUREMENT, and this is the sharp discriminator against v3's
    # withdrawn 20% dial: give the same field more galaxies and s_b SHRINKS.
    # A prior width would not move.  (It is not exactly 1/sqrt(N_gal) -- the
    # anchor field sharpens too, and the profile's subtracted term moves with
    # it: measured 0.541945 -> 0.186852 -> 0.115325 at 1x, 4x, 16x the counts
    # on this fixture, i.e. faster than 1/sqrt(N) at first and slower later.)
    op16 = CountOperator(op.proj_sph, op.phi_shell, 16.0 * op.counts,
                         op.log_fp, op.bias)
    s16 = count_map_solve(op16)
    gb16 = dgrad_db(s16["xi_hat"], op16)
    v16 = np.asarray(sensitivity(s16["xi_hat"], s16["H_chol"],
                                 gb16[:, None]))[:, 0]
    p16 = b_gal_profile_sigma(s16["xi_hat"], op16, dgrad_b=gb16, v_b=v16,
                              systematic_floor_frac=0.0)
    assert p16["s_b_stat"] < 0.5 * prof["s_b_stat"]


def test_d2_systematics_floor_only_raises_and_reports_which_bound_won():
    op, xi_hat, L, gb, v = _anchor(seed=2)
    bare = b_gal_profile_sigma(xi_hat, op, dgrad_b=gb, v_b=v,
                               systematic_floor_frac=0.0)
    hi = b_gal_profile_sigma(xi_hat, op, dgrad_b=gb, v_b=v,
                             systematic_floor_frac=0.5)
    assert bare["floor_active"] is False
    assert hi["floor_active"] is True
    assert hi["s_b"] == pytest.approx(0.5 * op.bias)
    assert hi["s_b"] > bare["s_b"]
    assert hi["s_b_stat"] == pytest.approx(bare["s_b_stat"], rel=0, abs=0)
    # the default is a floor, i.e. it never LOWERS the measured curvature
    dflt = b_gal_profile_sigma(xi_hat, op, dgrad_b=gb, v_b=v)
    assert dflt["s_b"] >= dflt["s_b_stat"]
    assert dflt["s_b_floor"] == pytest.approx(
        B_GAL_SYSTEMATIC_FLOOR_FRAC * op.bias)
    # H_chol alone rebuilds the same v: one construction, two entry points.
    from_chol = b_gal_profile_sigma(xi_hat, op, H_chol=L)
    assert from_chol["s_b"] == pytest.approx(dflt["s_b"], rel=1e-12)
    np.testing.assert_allclose(np.asarray(from_chol["v_b"]), v, rtol=0,
                               atol=1e-12)


# ------------------------------------------------- D3, D4: off is off, on adds

def test_d3_feature_off_is_bit_identical_to_the_pre_s2_draws():
    op, xi_hat, L, gb, v = _anchor(seed=3)
    key = jax.random.PRNGKey(11)
    old, old_g = _legacy_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key)
    new, new_g, eps = laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8,
                                    key, return_g=True, return_eps=True)
    assert np.array_equal(np.asarray(new), np.asarray(old))
    assert np.array_equal(np.asarray(new_g), np.asarray(old_g))
    assert np.array_equal(np.asarray(eps), np.zeros(8))
    # the legacy call signature, untouched
    bare = laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key)
    assert np.array_equal(np.asarray(bare), np.asarray(old))
    # s_b = 0 is the same statement as s_b = None
    zero = laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key,
                         s_b=0.0, v_b=v)
    assert np.array_equal(np.asarray(zero), np.asarray(old))


def test_d4_feature_on_moves_the_members_by_exactly_the_rank1_term():
    op, xi_hat, L, gb, v = _anchor(seed=3)
    key = jax.random.PRNGKey(11)
    s_b = float(b_gal_profile_sigma(xi_hat, op, dgrad_b=gb, v_b=v)["s_b"])
    off = np.asarray(laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key))
    on, g_on, eps = laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key,
                                  return_g=True, return_eps=True,
                                  s_b=s_b, v_b=v)
    # the g stream is UNCHANGED by switching the feature on: the rank-1 stream
    # is fold_in(key, 1), not split(key), so exactly one thing moves.
    _, g_off = laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 8, key,
                             return_g=True)
    assert np.array_equal(np.asarray(g_on), np.asarray(g_off))
    delta = np.asarray(on) - off
    np.testing.assert_allclose(
        delta, s_b * np.asarray(eps)[:, None] * v[None, :], rtol=0, atol=1e-14)
    # and it is not a null edit: the members actually moved
    assert np.abs(delta).max() > 0.0


# ------------------------------------------------------ D5: the covariance

def test_d5_drawn_covariance_is_h_inv_plus_rank1_in_the_v_direction():
    op, xi_hat, L, gb, v = _anchor(seed=4)
    # A large s_b, so the rank-1 term is resolvable against MC error at an
    # affordable M; the SHAPE of the statement does not depend on the value.
    s_b = 0.35
    n = 40000
    draws = np.asarray(laplace_draws(
        jnp.asarray(xi_hat), jnp.asarray(L), n, jax.random.PRNGKey(5),
        s_b=s_b, v_b=v))
    r = draws - xi_hat[None, :]
    H = L @ L.T
    Hinv = np.linalg.inv(H)

    # (a) the full covariance, in a handful of coordinates
    cov = r.T @ r / n
    target = Hinv + s_b ** 2 * np.outer(v, v)
    idx = np.arange(0, xi_hat.size, max(1, xi_hat.size // 8))
    sub_c, sub_t = cov[np.ix_(idx, idx)], target[np.ix_(idx, idx)]
    # MC error on a covariance entry is ~ sqrt((C_ii C_jj + C_ij^2)/n)
    err = np.sqrt((np.outer(np.diag(sub_t), np.diag(sub_t)) + sub_t ** 2) / n)
    z = np.abs(sub_c - sub_t) / err
    assert z.max() < 5.0, f"covariance off by up to {z.max():.2f} MC sigma"

    # (b) THE RANK-1 DIRECTION, specifically.  Project on v: the variance must
    # be v^T H^{-1} v + s_b^2 (v.v)^2, i.e. inflated by a factor the anchor
    # itself predicts.
    pv = r @ v
    var_v = float(pv @ pv / n)
    base_v = float(v @ Hinv @ v)
    pred_v = base_v + s_b ** 2 * float(v @ v) ** 2
    assert pred_v / base_v > 1.5, "fixture does not exercise the inflation"
    assert abs(var_v - pred_v) / (pred_v * np.sqrt(2 / n)) < 5.0, (
        f"variance along v = {var_v:.6e}, predicted {pred_v:.6e} "
        f"(uninflated {base_v:.6e})")

    # (c) a companion direction with u^T H^{-1} v = 0 -- H^{-1}-orthogonal to
    # the inflation, so its variance must not move AT ALL.  This one is exact
    # rather than statistical: the same g stream is used with the feature on
    # and off, so the two projections differ only by s_b eps (u.v), and u.v is
    # zero here by construction.
    w = Hinv @ v
    e = np.zeros(xi_hat.size)
    e[int(np.argmin(np.abs(w)))] = 1.0
    u = e - (float(e @ Hinv @ v) / base_v) * v
    u = u - (float(u @ v) / float(v @ v)) * v      # also v-orthogonal: u.v = 0
    assert abs(float(u @ v)) < 1e-10
    off = np.asarray(laplace_draws(
        jnp.asarray(xi_hat), jnp.asarray(L), n, jax.random.PRNGKey(5)))
    pu_on = r @ u
    pu_off = (off - xi_hat[None, :]) @ u
    np.testing.assert_allclose(pu_on, pu_off, rtol=0, atol=1e-12)
    var_u = float(pu_on @ pu_on / n)
    pred_u = float(u @ Hinv @ u)
    assert abs(var_u - pred_u) / (pred_u * np.sqrt(2 / n)) < 5.0


# --------------------------------------------------- D6: antithetic in both

def test_d6_antithetic_in_both_sources_and_in_every_balanced_prefix():
    op, xi_hat, L, gb, v = _anchor(seed=6)
    s_b = 0.2
    m = 16
    draws, g, eps = laplace_draws(
        jnp.asarray(xi_hat), jnp.asarray(L), m, jax.random.PRNGKey(7),
        return_g=True, return_eps=True, s_b=s_b, v_b=v)
    draws, g, eps = np.asarray(draws), np.asarray(g), np.asarray(eps)
    half = m // 2
    # partner of k is k + M/2, in BOTH sources
    np.testing.assert_allclose(g[:half], -g[half:], rtol=0, atol=0)
    np.testing.assert_allclose(eps[:half], -eps[half:], rtol=0, atol=0)
    assert np.abs(eps).min() > 0.0
    # the full ensemble is centred on xi_hat EXACTLY
    np.testing.assert_allclose(draws.mean(0), xi_hat, rtol=0, atol=1e-12)
    # PR-5b's balanced ordering: every even prefix is centred too, in both
    # sources.  (The naive prefix draws[:k] is NOT -- that is the measured
    # value of the pairing, and it is why eps must share the pairing.)
    order = np.ravel(np.column_stack([np.arange(half),
                                      np.arange(half) + half]))
    for k in (2, 4, 8, m):
        pre = order[:k]
        np.testing.assert_allclose(draws[pre].mean(0), xi_hat,
                                   rtol=0, atol=1e-12)
        assert abs(float(eps[pre].sum())) < 1e-15
    # ... and the naive prefix is not balanced, so the pin above is a real one
    assert abs(float(eps[:4].sum())) > 1e-6


def test_d7_scale_without_direction_is_refused():
    op, xi_hat, L, gb, v = _anchor(seed=8)
    with pytest.raises(ValueError, match="without v_b"):
        laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 4,
                      jax.random.PRNGKey(0), s_b=0.1)
    with pytest.raises(ValueError, match="same mode space"):
        laplace_draws(jnp.asarray(xi_hat), jnp.asarray(L), 4,
                      jax.random.PRNGKey(0), s_b=0.1, v_b=v[:-1])
    with pytest.raises(ValueError, match="either v_b"):
        b_gal_profile_sigma(xi_hat, op)
