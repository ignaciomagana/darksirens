"""
test_analyze_ppd_chunking.py
----------------------------
Tests for the memory-safe posterior-predictive engine in
``darksirens.cli.analyze``:

* ``plan_ppd_sizing`` keeps the working-set estimate under the budget and
  honours user overrides;
* ``probe_device_memory_bytes`` returns a positive budget and a source tag;
* grid-chunked density evaluation is numerically identical to the unchunked
  (legacy) path, and the marginals stay normalised.
"""

import sys
import types

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

# darksirens.gw.populations imports tinygp lazily; stub it so importing the
# analyze module never requires the real package (mirrors test_analyze_bayes_factors).
if "tinygp" not in sys.modules:
    _stub = types.ModuleType("tinygp")

    class _GP:
        def __init__(self, *a, **k):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _Kernels:
        class Matern52:
            def __init__(self, *a, **k):
                pass

            def __rmul__(self, other):
                return self

    _stub.GaussianProcess = _GP
    _stub.kernels = _Kernels()
    sys.modules["tinygp"] = _stub

jnp = pytest.importorskip("jax.numpy")

from darksirens.cli.analyze import (
    gp_node_count,
    plan_ppd_sizing,
    probe_device_memory_bytes,
    posterior_predictive,
)
from darksirens.gw.populations import pop_model_parser


# --------------------------------------------------------------------------
# Sizing planner (pure arithmetic, no device needed)
# --------------------------------------------------------------------------

def test_plan_ppd_sizing_bounds_working_set_for_gp():
    n_outer, nz, nchi = 6144, 32, 24   # 128*48 outer rows; default (z,chi) plane
    n_nodes = 6400                     # gp4d
    budget = 8e9
    safe_frac, n_live, dtype = 0.4, 16, 8
    batch, slab = plan_ppd_sizing(
        nsamples=200, n_outer=n_outer, nz=nz, nchi=nchi, n_nodes=n_nodes,
        max_mem_bytes=budget, safe_frac=safe_frac, n_live=n_live, dtype_bytes=dtype,
    )
    assert batch >= 1 and slab is not None and slab <= n_outer
    # per-slab working set within budget
    assert batch * slab * nz * nchi * dtype * (n_nodes + n_live) <= safe_frac * budget


def test_plan_ppd_sizing_no_chunk_for_parametric_small_grid():
    batch, slab = plan_ppd_sizing(   # grid_points = 2500*4*2 = 20000 <= full_grid_max
        nsamples=500, n_outer=2500, nz=4, nchi=2, n_nodes=0,
        max_mem_bytes=16e9, safe_frac=0.4,
    )
    assert slab is None           # whole grid fits → unchunked (full) path
    assert batch >= 1


def test_plan_ppd_sizing_chunks_small_grid_when_gp_nodes_blow_the_budget():
    # grid_points = 1024*8*8 = 65536 <= full_grid_max, but with 6400 inducing
    # nodes the full plane needs ~215 GB at batch=64: the small-grid shortcut
    # must not bypass the memory model.
    n_outer, nz, nchi, n_nodes = 1024, 8, 8, 6400
    budget, safe_frac, n_live, dtype = 8e9, 0.4, 16, 8
    batch, slab = plan_ppd_sizing(
        nsamples=200, n_outer=n_outer, nz=nz, nchi=nchi, n_nodes=n_nodes,
        max_mem_bytes=budget, safe_frac=safe_frac, n_live=n_live, dtype_bytes=dtype,
    )
    assert slab is not None and 1 <= slab <= n_outer
    assert batch * slab * nz * nchi * dtype * (n_nodes + n_live) <= safe_frac * budget


def test_plan_ppd_sizing_honors_overrides():
    batch, slab = plan_ppd_sizing(
        nsamples=100, n_outer=1000, nz=32, nchi=24, n_nodes=10,
        max_mem_bytes=8e9, batch_size=3, grid_chunk=11,
    )
    assert batch == 3 and slab == 11


def test_gp_node_count_covers_the_additive_family():
    # The additive models keep their nodes per ANOVA term (no top-level ``M``),
    # so a bare getattr(model, "M", 0) probe would size their slab as parametric.
    assert gp_node_count(pop_model_parser("powerlaw")) == 0
    assert gp_node_count(pop_model_parser("gp4d")) == 6400
    assert gp_node_count(pop_model_parser("gp4d_additive")) == 268
    assert gp_node_count(pop_model_parser("gp_separable")) == 28


def test_probe_device_memory_positive():
    budget, source = probe_device_memory_bytes()
    assert isinstance(budget, int) and budget > 0
    assert isinstance(source, str) and source


# --------------------------------------------------------------------------
# Numerical equivalence of chunked vs unchunked evaluation
# --------------------------------------------------------------------------

def _powerlaw_setup():
    """A parametric model whose pop-extractor ignores theta (fix_population),
    so the test needs no valid sampled-coordinate construction."""
    settings = {
        "pop_model": "powerlaw",
        "fix_population": True,
        "shared_beta": True,
        "shared_spin": True,
        "shared_gamma": True,
    }
    pop_model = pop_model_parser("powerlaw")
    mgrid = np.linspace(6.0, 50.0, 10)
    qgrid = np.linspace(0.05, 1.0, 6)
    zgrid = np.linspace(0.0, 1.5, 5)
    chigrid = np.linspace(-1.0, 1.0, 4)
    samples = np.zeros((3, 1))      # theta is ignored by the fixed extractor
    return pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid


def test_chunked_matches_unchunked():
    pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid = _powerlaw_setup()
    common = dict(batch_size=2, max_mem_gb=8.0)
    ref = posterior_predictive(pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
                               grid_chunk=None, **common)
    # n_outer = nm*nq = 60; slab 7 does not divide it → exercises pad+trim
    new = posterior_predictive(pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
                               grid_chunk=7, **common)
    for a, b in zip(ref, new):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-12)


def test_marginals_are_normalised():
    pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid = _powerlaw_setup()
    p_m1, p_m2, p_q, p_z, p_chi, p_m1m2 = posterior_predictive(
        pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
        batch_size=2, grid_chunk=7, max_mem_gb=8.0,
    )
    for grid, marg in [(mgrid, p_m1), (mgrid, p_m2), (qgrid, p_q),
                       (zgrid, p_z), (chigrid, p_chi)]:
        integral = _trapezoid(np.asarray(marg[0]), grid)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# A degenerate (zero-density) posterior sample must not NaN the whole figure
# --------------------------------------------------------------------------

def test_zero_density_sample_yields_zero_curves_not_nan():
    """A constraint-violating draw (e.g. parametric.py's ``valid`` mask forces
    p = 0 everywhere) integrates to exactly 0 on the grid.  Unguarded ``/=``
    normalisations turn that into 0/0 = NaN, and jnp.median/jnp.percentile
    propagate NaN — one bad sample out of thousands would blank the credible band
    of every model in the figure, silently."""
    from darksirens.cli.analyze import make_single_theta_predictive, summarize_ppd

    settings = {"pop_model": "powerlaw", "fix_population": True,
                "shared_beta": True, "shared_spin": True, "shared_gamma": True}
    mgrid = np.linspace(6.0, 50.0, 8)
    qgrid = np.linspace(0.05, 1.0, 5)
    zgrid = np.linspace(0.0, 1.5, 4)
    chigrid = np.linspace(-1.0, 1.0, 3)

    def dead_model(m1, q, z, chi, theta):
        return jnp.full(jnp.shape(m1), -jnp.inf)      # p = 0 everywhere

    for grid_chunk in (None, 3):
        fn = make_single_theta_predictive(dead_model, settings, mgrid, qgrid, zgrid,
                                          chigrid, grid_chunk=grid_chunk)
        for arr in fn(jnp.zeros(1)):
            arr = np.asarray(arr)
            assert np.all(np.isfinite(arr)), grid_chunk
            assert np.all(arr == 0.0), grid_chunk

    # ... and the summary of a stack containing one dead sample is unaffected.
    good = np.tile(np.linspace(0.1, 1.0, 8), (4, 1))
    stack = np.vstack([good, np.zeros((1, 8))])
    med, lo, hi = summarize_ppd(stack)
    assert np.all(np.isfinite(med)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


# --------------------------------------------------------------------------
# The slab evaluator must hand the population model BROADCAST AXES, not a
# ravelled (S*nz*nchi,) mesh
# --------------------------------------------------------------------------

def test_slab_passes_broadcast_axes_not_a_ravelled_mesh():
    """Pin the coordinate shapes the PPD evaluator hands the population model.

    The pairing normaliser inside the model runs an N_Q-node q-quadrature per
    query ROW and depends on m1 alone, so a flattened ``(S*nz*nchi,)`` mesh
    repeats it ``nz*nchi`` times and materialises an ``(S*nz*nchi, N_Q)``
    scratch buffer — 15.17 GB of compiled scratch at the shipped defaults
    (--nm 128 --nq 48 --nz 32 --nchi 24), which is the reason ``--grid_chunk``
    exists at all.  Broadcast axes keep that at ``(S, 1, 1, N_Q)``: 39.7 MB,
    and 21.2 -> 0.58 ms per sample on GPU (1507 -> 16.7 ms on CPU).
    """
    from darksirens.cli.analyze import make_single_theta_predictive

    settings = {"pop_model": "powerlaw", "fix_population": True,
                "shared_beta": True, "shared_spin": True, "shared_gamma": True}
    mgrid = np.linspace(6.0, 50.0, 10)
    qgrid = np.linspace(0.05, 1.0, 6)
    zgrid = np.linspace(0.0, 1.5, 5)
    chigrid = np.linspace(-1.0, 1.0, 4)
    nz, nchi = zgrid.size, chigrid.size
    inner = pop_model_parser("powerlaw")

    for grid_chunk, n_outer in ((None, mgrid.size * qgrid.size), (7, 7)):
        seen = []

        def recording(m1, q, z, chi, theta):
            seen.append(tuple(jnp.shape(a) for a in (m1, q, z, chi)))
            return inner(m1, q, z, chi, theta)

        fn = make_single_theta_predictive(recording, settings, mgrid, qgrid,
                                          zgrid, chigrid, grid_chunk=grid_chunk)
        out = fn(jnp.zeros(1))
        assert seen, grid_chunk
        assert seen[0] == ((n_outer, 1, 1), (n_outer, 1, 1),
                           (1, nz, 1), (1, 1, nchi)), (grid_chunk, seen[0])
        for arr in out:
            assert np.all(np.isfinite(np.asarray(arr))), grid_chunk


def test_slab_accepts_a_model_with_degenerate_axes():
    """A density independent of z (or chi) legitimately returns size-1 axes
    under broadcast inputs; the slab evaluator must broadcast, not reshape."""
    from darksirens.cli.analyze import make_single_theta_predictive

    settings = {"pop_model": "powerlaw", "fix_population": True,
                "shared_beta": True, "shared_spin": True, "shared_gamma": True}
    mgrid = np.linspace(6.0, 50.0, 8)
    qgrid = np.linspace(0.05, 1.0, 5)
    zgrid = np.linspace(0.0, 1.5, 4)
    chigrid = np.linspace(-1.0, 1.0, 3)

    def flat_in_z_and_chi(m1, q, z, chi, theta):
        # shape (S, 1, 1): no z or chi dependence at all.
        return jnp.zeros(jnp.shape(m1))

    for grid_chunk in (None, 3):
        fn = make_single_theta_predictive(flat_in_z_and_chi, settings, mgrid,
                                          qgrid, zgrid, chigrid,
                                          grid_chunk=grid_chunk)
        for arr in fn(jnp.zeros(1)):
            arr = np.asarray(arr)
            assert np.all(np.isfinite(arr)), grid_chunk
