"""
test_population_sampling.py
---------------------------
Grid inverse-CDF samplers for the flow-surrogate path
(darksirens/gw/populations/sampling.py).

The load-bearing property is exactness of the proposal density: every
sampler returns log_s such that E_s[t/s] equals the integral of the target
t, at any grid resolution. That identity is tested directly, along with
empirical marginals, the analytic truncated normal, and common-random-number
continuity in the hyperparameters.
"""

import sys
from pathlib import Path

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.gw.populations.sampling import (
    cell_centers,
    make_mass_q_edges,
    resolve_mass_grid_bounds,
    sample_histogram_1d,
    sample_mass_q,
    sample_z_from_grid,
    truncnorm_sample,
)

RNG = np.random.default_rng(20260712)
J = 200_000
U = jnp.asarray(RNG.uniform(size=(J, 4)))


def _log_t_cells(model, theta, m1_edges, q_edges):
    tm = model.mixture_theta(theta)
    m1c = cell_centers(m1_edges)
    qc = cell_centers(q_edges)
    M1, Q = jnp.meshgrid(m1c, qc, indexing="ij")
    p = model.mixture.mass_q_density(M1.ravel(), Q.ravel(), tm).reshape(M1.shape)
    return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)


@pytest.fixture(scope="module")
def plp():
    model = get_model("powerlaw+peak")
    theta = jnp.asarray(np.asarray(get_fixed_population_params("powerlaw+peak")))
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges, q_edges = make_mass_q_edges(m1_lo, m1_hi, n_m1=256, n_q=128)
    log_t = _log_t_cells(model, theta, m1_edges, q_edges)
    return model, theta, m1_edges, q_edges, log_t


def test_histogram_1d_exactness_and_support():
    # Target: unnormalised linear ramp on [0, 2]; exact integral = 2.
    edges = jnp.linspace(0.0, 2.0, 41)
    dens = cell_centers(edges)  # t(x) = x at cell centers
    out = sample_histogram_1d(U[:, 0], edges, jnp.log(dens))
    assert out.x.shape == (J,)
    assert float(out.x.min()) >= 0.0 and float(out.x.max()) <= 2.0
    # E_s[t/s] with t the piecewise-constant cell density itself == norm.
    t_at = jnp.log(dens)[out.cell]
    est = jnp.exp(jax.scipy.special.logsumexp(t_at - out.log_s) - jnp.log(J))
    np.testing.assert_allclose(float(est), float(jnp.exp(out.log_norm)), rtol=1e-12)
    # Empirical CDF vs analytic CDF of the ramp: x^2/4.
    xs = np.sort(np.asarray(out.x))
    ref = xs**2 / 4.0
    emp = np.arange(1, J + 1) / J
    assert np.abs(emp - ref).max() < 5e-3


def test_mass_q_exact_reweight_recovers_target_integral(plp):
    model, theta, m1_edges, q_edges, log_t = plp
    out = sample_mass_q(U[:, 0], U[:, 1], m1_edges, q_edges, log_t)

    # Evaluate the TRUE target at the sampled points and reweight by the
    # exact proposal density: the mean of t/s estimates ∬ t dm1 dq.
    tm = model.mixture_theta(theta)
    p = model.mixture.mass_q_density(out.m1, out.q, tm)
    log_p = jnp.where(p > 0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
    est = float(jnp.exp(
        jax.scipy.special.logsumexp(log_p - out.log_s) - jnp.log(J)
    ))

    # Reference: dense trapezoid integral of the true target.
    m1g = jnp.geomspace(m1_edges[0], m1_edges[-1], 1024)
    qg = jnp.linspace(q_edges[0], q_edges[-1], 512)
    M1, Q = jnp.meshgrid(m1g, qg, indexing="ij")
    P = model.mixture.mass_q_density(M1.ravel(), Q.ravel(), tm).reshape(M1.shape)
    ref = float(jnp.trapezoid(jnp.trapezoid(P, qg, axis=1), m1g))

    assert est == pytest.approx(ref, rel=2e-2)


def test_mass_q_marginals_match_brute_force(plp):
    model, theta, m1_edges, q_edges, log_t = plp
    out = sample_mass_q(U[:, 0], U[:, 1], m1_edges, q_edges, log_t)
    m1 = np.asarray(out.m1)
    assert (m1 >= float(m1_edges[0])).all() and (m1 <= float(m1_edges[-1])).all()

    # Binned m1 marginal of the draws vs the (normalised) grid marginal.
    dm = np.diff(np.asarray(m1_edges))
    dq = np.diff(np.asarray(q_edges))
    mass = np.exp(np.asarray(log_t)) * dm[:, None] * dq[None, :]
    marg = mass.sum(axis=1)
    marg = marg / marg.sum()
    counts, _ = np.histogram(m1, bins=np.asarray(m1_edges))
    emp = counts / counts.sum()
    assert np.abs(emp - marg).sum() < 0.02  # total-variation distance

    q = np.asarray(out.q)
    margq = mass.sum(axis=0)
    margq = margq / margq.sum()
    countsq, _ = np.histogram(q, bins=np.asarray(q_edges))
    empq = countsq / countsq.sum()
    assert np.abs(empq - margq).sum() < 0.02


def test_truncnorm_against_scipy():
    from scipy import stats

    mu, sigma, lo, hi = 0.06, 0.12, -1.0, 1.0
    u = U[: 50_000, 2]
    out = truncnorm_sample(u, mu, sigma, lo, hi)
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    ref_x = stats.truncnorm.ppf(np.asarray(u), a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(np.asarray(out.x), ref_x, atol=1e-9)
    ref_logpdf = stats.truncnorm.logpdf(ref_x, a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(np.asarray(out.log_s), ref_logpdf, atol=1e-9)


def test_z_sampler_matches_analytic_cdf():
    # Toy target resembling a volumetric prior: t(z) ∝ z^2 on [0, 3].
    zgrid = jnp.linspace(0.0, 3.0, 400)
    log_t = jnp.where(zgrid > 0, 2.0 * jnp.log(jnp.maximum(zgrid, 1e-300)), -jnp.inf)
    out = sample_z_from_grid(U[:, 3], zgrid, log_t)
    zs = np.sort(np.asarray(out.x))
    ref = (zs / 3.0) ** 3
    emp = np.arange(1, J + 1) / J
    assert np.abs(emp - ref).max() < 5e-3


def test_common_random_numbers_continuity(plp):
    model, theta, m1_edges, q_edges, _ = plp
    theta2 = jnp.asarray(np.asarray(theta).copy())
    # Perturb alpha (index 1: [v1, alpha, m_min, m_max, dm_min, dm_max, ...]).
    theta2 = theta2.at[1].add(1e-4)

    u1, u2 = U[:4096, 0], U[:4096, 1]
    a = sample_mass_q(u1, u2, m1_edges, q_edges,
                      _log_t_cells(model, theta, m1_edges, q_edges))
    b = sample_mass_q(u1, u2, m1_edges, q_edges,
                      _log_t_cells(model, theta2, m1_edges, q_edges))
    # Same base uniforms + tiny hyperparameter step -> tiny coordinate step.
    assert float(jnp.median(jnp.abs(a.m1 - b.m1))) < 0.05
    assert float(jnp.median(jnp.abs(a.q - b.q))) < 0.01


def test_samplers_are_jittable(plp):
    model, theta, m1_edges, q_edges, log_t = plp

    @jax.jit
    def draw(u1, u2, log_t):
        out = sample_mass_q(u1, u2, m1_edges, q_edges, log_t)
        return out.m1, out.q, out.log_s

    m1, q, log_s = draw(U[:1024, 0], U[:1024, 1], log_t)
    assert np.isfinite(np.asarray(log_s)).all()


def test_truncated_histogram_exactness():
    from darksirens.gw.populations.sampling import sample_histogram_trunc

    # Ramp target on [0, 2] truncated to [0.5, 1.2]: E_s[t/s] over the window
    # must equal the window integral of t exactly (t = cell density itself).
    edges = jnp.linspace(0.0, 2.0, 81)
    dens = cell_centers(edges)
    lo, hi = 0.5, 1.2
    out = sample_histogram_trunc(U[:100_000, 0], edges, jnp.log(dens), lo, hi)
    assert float(out.x.min()) >= lo and float(out.x.max()) <= hi
    t_at = jnp.log(dens)[out.cell]
    est = float(jnp.exp(
        jax.scipy.special.logsumexp(t_at - out.log_s) - jnp.log(out.x.shape[0])
    ))
    # Exact window mass of the piecewise-constant density.
    widths = jnp.diff(edges)
    mass = float(jnp.sum(
        dens * jnp.clip(jnp.minimum(edges[1:], hi) - jnp.maximum(edges[:-1], lo), 0.0, None)
    ))
    assert est == pytest.approx(mass, rel=1e-10)
    # Per-draw (array) windows must work elementwise too.
    los = jnp.full(1000, 0.3).at[::2].set(0.8)
    his = jnp.full(1000, 1.5)
    out2 = sample_histogram_trunc(U[:1000, 1], edges, jnp.log(dens), los, his)
    x2 = np.asarray(out2.x)
    assert (x2[::2] >= 0.8 - 1e-12).all() and (x2 >= 0.3 - 1e-12).all() and (x2 <= 1.5 + 1e-12).all()


def test_chirp_band_two_stage_sampler_recovers_target_integral(plp):
    from darksirens.gw.populations.sampling import (
        _floored,
        sample_m1_given_q_trunc,
        sample_q_marginal_trunc,
    )

    model, theta, m1_edges, q_edges, log_t = plp
    log_t_fl = _floored(log_t)
    n = 100_000
    q_win = (0.4, 0.95)
    qs = sample_q_marginal_trunc(U[:n, 1], m1_edges, q_edges, log_t_fl, q_win)
    # Per-draw m1 windows: a (fat) chirp band Mc in [8, 12] at z=0.
    g = (1.0 + qs.x) ** 0.2 / qs.x**0.6
    m1_lo, m1_hi = 8.0 * g, 12.0 * g
    m1, log_s_m1 = sample_m1_given_q_trunc(
        U[:n, 0], m1_edges, log_t_fl, qs.cell, m1_lo, m1_hi
    )
    assert bool(jnp.all(m1 >= m1_lo - 1e-9)) and bool(jnp.all(m1 <= m1_hi + 1e-9))

    # Exact-reweight identity: E_s[t/s] = ∫∫_region t dm1 dq for the floored
    # piecewise-constant target restricted to the sampled region.
    tm = model.mixture_theta(theta)
    p = model.mixture.mass_q_density(m1, qs.x, tm)
    log_p = jnp.where(p > 0, jnp.log(jnp.maximum(p, 1e-300)), -jnp.inf)
    est = float(jnp.exp(
        jax.scipy.special.logsumexp(log_p - (qs.log_s + log_s_m1)) - jnp.log(n)
    ))

    # Brute-force reference on a fine grid over the same region.
    m1g = jnp.geomspace(float(m1_edges[0]), float(m1_edges[-1]), 2048)
    qg = jnp.linspace(q_win[0], q_win[1], 512)
    M1, Q = jnp.meshgrid(m1g, qg, indexing="ij")
    P = model.mixture.mass_q_density(M1.reshape(-1), Q.reshape(-1), tm).reshape(M1.shape)
    G = (1.0 + Q) ** 0.2 / Q**0.6
    band = (M1 >= 8.0 * G) & (M1 <= 12.0 * G)
    ref = float(jnp.trapezoid(jnp.trapezoid(jnp.where(band, P, 0.0), qg, axis=1), m1g))
    assert est == pytest.approx(ref, rel=5e-2)


# ── the m1|q draw reads the CDF tables by (draw, node) index ────────────────
#
# The draw used to slice a whole CDF column per draw and hand it to
# jnp.searchsorted, which XLA cannot fuse: 17.6 GB of scratch at the
# production flow shape (nEvents=259, J=16384, K=512).  It now probes the 2-D
# tables directly.  The tests below pin BOTH halves of that claim: bitwise
# equality with the column-slice reference, and scratch that no longer grows
# with the grid resolution.


def _m1_given_q_column_slice_reference(u, m1_edges, log_t_cells, q_cells, lo, hi, tab):
    """The pre-restructure draw: slice the column, then ``jnp.searchsorted``."""
    from darksirens.gw.populations.sampling import _sample_histogram_trunc_tab

    def _one(u_j, c_j, lo_j, hi_j):
        out = _sample_histogram_trunc_tab(
            u_j[None], m1_edges, log_t_cells[:, c_j],
            tab.cdf_nodes[c_j], tab.log_norm[c_j], lo_j, hi_j,
        )
        return out.x[0], out.log_s[0]

    return jax.vmap(_one)(u, q_cells, lo, hi)


def test_column_searchsorted_matches_jnp_searchsorted():
    """Including duplicate nodes and a table that is not quite monotone.

    A parallel ``jnp.cumsum`` can leave the CDF nodes non-monotone by an ulp
    near saturation, so the column search must agree with ``jnp.searchsorted``
    on unsorted input too — not just where the precondition holds.
    """
    from darksirens.gw.populations.sampling import _searchsorted_right_col

    rng = np.random.default_rng(7)
    for K, C in ((512, 16), (65, 4), (2, 3), (1, 1)):
        nodes = jnp.asarray(np.sort(rng.uniform(size=(C, K + 1)), axis=1))
        nodes = nodes.at[:, K // 2 : K // 2 + 3].set(nodes[:, K // 2 : K // 2 + 1])
        nodes = nodes.at[0, -1].set(nodes[0, -1] * (1.0 - 2.0**-52))
        col = jnp.asarray(rng.integers(0, C, 4096))
        v = jnp.asarray(rng.uniform(size=4096))
        # hit exact node values, not just generic points
        v = v.at[::4].set(nodes[col[::4], (col[::4] * 7) % (K + 1)])
        got = _searchsorted_right_col(nodes, col, v)
        want = jax.vmap(lambda c, x: jnp.searchsorted(nodes[c], x, side="right"))(col, v)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_m1_given_q_draw_is_bit_identical_to_column_slice_reference():
    from darksirens.gw.populations.sampling import (
        build_column_cdf_tables,
        sample_m1_given_q_trunc,
    )

    rng = np.random.default_rng(11)
    for K, C, n in ((512, 256, 4096), (64, 16, 257), (7, 3, 33), (1, 2, 17)):
        m1_edges = jnp.asarray(np.sort(rng.uniform(3.0, 100.0, K + 1)))
        dense = jnp.asarray(rng.normal(size=(K, C)) - 5.0)
        # 90% floored: duplicate CDF nodes, and a saturated (non-monotone) tail
        floored = jnp.where(jnp.asarray(rng.uniform(size=(K, C))) < 0.9, -700.0, dense)
        for tag, log_t in (("dense", dense), ("floored", floored)):
            tab = build_column_cdf_tables(m1_edges, log_t)
            u = jnp.asarray(rng.uniform(size=n)).at[0].set(0.0).at[-1].set(1.0)
            cells = jnp.asarray(rng.integers(0, C, n))
            lo = jnp.asarray(rng.uniform(0.0, 110.0, n))       # windows off-grid
            hi = lo + jnp.asarray(rng.uniform(-5.0, 60.0, n))  # and some hi < lo
            x, log_s = sample_m1_given_q_trunc(
                u, m1_edges, log_t, cells, lo, hi, tables=tab
            )
            x_ref, log_s_ref = _m1_given_q_column_slice_reference(
                u, m1_edges, log_t, cells, lo, hi, tab
            )
            assert bool(jnp.all(x == x_ref)), f"{tag} K={K}: m1 draws moved"
            assert bool(jnp.all(log_s == log_s_ref)), f"{tag} K={K}: log_s moved"

    # ... and under the vmap-over-events the flow likelihood wraps it in.
    K, C, nE, n = 64, 16, 5, 128
    m1_edges = jnp.asarray(np.sort(rng.uniform(3.0, 100.0, K + 1)))
    log_t = jnp.asarray(rng.normal(size=(K, C)) - 5.0)
    tab = build_column_cdf_tables(m1_edges, log_t)
    u = jnp.asarray(rng.uniform(size=(nE, n)))
    cells = jnp.asarray(rng.integers(0, C, (nE, n)))
    lo = jnp.asarray(rng.uniform(3.0, 90.0, (nE, n)))
    hi = lo + jnp.asarray(rng.uniform(0.0, 40.0, (nE, n)))

    def _per_event(f):
        return jax.vmap(lambda a, b, c, d: f(a, m1_edges, log_t, b, c, d, tab))

    x, log_s = _per_event(
        lambda u_, e, l, c, a, b, t: sample_m1_given_q_trunc(u_, e, l, c, a, b, tables=t)
    )(u, cells, lo, hi)
    x_ref, log_s_ref = _per_event(_m1_given_q_column_slice_reference)(u, cells, lo, hi)
    assert bool(jnp.all(x == x_ref)) and bool(jnp.all(log_s == log_s_ref))


def test_m1_given_q_draw_scratch_is_independent_of_grid_resolution():
    """Regression guard: the (draws, K+1) CDF tensor must not come back."""
    from darksirens.gw.populations.sampling import (
        build_column_cdf_tables,
        sample_m1_given_q_trunc,
    )

    def temp_bytes(K, C=8, n=1024):
        rng = np.random.default_rng(3)
        m1_edges = jnp.asarray(np.linspace(3.0, 100.0, K + 1))
        log_t = jnp.asarray(rng.normal(size=(K, C)))
        tab = build_column_cdf_tables(m1_edges, log_t)
        u = jnp.asarray(rng.uniform(size=n))
        cells = jnp.asarray(rng.integers(0, C, n))
        lo = jnp.asarray(rng.uniform(3.0, 40.0, n))
        f = jax.jit(
            lambda u_, c, a, b: sample_m1_given_q_trunc(
                u_, m1_edges, log_t, c, a, b, tables=tab
            )
        )
        mem = f.lower(u, cells, lo, lo + 10.0).compile().memory_analysis()
        if mem is None:
            pytest.skip("backend reports no memory analysis")
        return mem.temp_size_in_bytes

    small, big = temp_bytes(128), temp_bytes(1024)
    # The column slice cost n*(K+1)*8 B, i.e. an 8x jump across this pair.
    assert big < 1.5 * small, f"scratch grows with K: {small} -> {big} B"
