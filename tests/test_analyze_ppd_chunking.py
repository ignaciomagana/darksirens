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


def test_plan_ppd_sizing_honors_overrides():
    batch, slab = plan_ppd_sizing(
        nsamples=100, n_outer=1000, nz=32, nchi=24, n_nodes=10,
        max_mem_bytes=8e9, batch_size=3, grid_chunk=11,
    )
    assert batch == 3 and slab == 11


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
        integral = np.trapezoid(np.asarray(marg[0]), grid)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-6, atol=1e-9)
