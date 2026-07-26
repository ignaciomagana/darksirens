"""Regression tests for two GP-population fixes.

1. ``gp3d_m1_q_chi`` (axes m1,q,chi, no z) normalised over the FULL
   500x200x200 = 2e7-point tensor grid x ~800 nodes ~= 128 GB, OOM/hanging on
   the first likelihood call.  The normalisation now coarsens for any
   >=3-prob-axis model (as it already did for z-conditioned multi-axis models),
   so the model is runnable; the affordable 2-prob-axis ``gp2d_q_chi`` keeps its
   full grid unchanged.

2. ``JointGPPopulation`` / ``AdditiveGPPopulation`` assumed 1-D query inputs and
   crashed on the weak-lensing path's (Nsamp, Nnodes) mu-marginalisation mesh
   (``_broadcast_logp_inputs`` was defined but never called).  ``log_p_pop`` now
   flattens broadcastable inputs and reshapes the result back.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens.gw.populations.gp as gp
from darksirens.gw.populations.gp import GP_MODEL_NAMES, build_gp_model


@pytest.fixture(autouse=True)
def _evict_tinygp_stub():
    """Combined-run guard: several test files install a stub ``tinygp`` (a
    bare module with no ``__file__``) into ``sys.modules`` at import time so
    their own imports stay light, and in a batched run that stub leaks into
    THIS file's lazy GP imports — every kernel construction then dies with
    ``ImportError: cannot import name 'transforms' from 'tinygp' (unknown
    location)`` even though the file is green standalone.  Evict any stub
    (and its submodule entries) before each test so the real package imports;
    files that want the stub re-install it at their own import time, and
    already-bound references are unaffected."""
    stub = sys.modules.get("tinygp")
    if stub is not None and getattr(stub, "__file__", None) is None:
        for key in [k for k in list(sys.modules)
                    if k == "tinygp" or k.startswith("tinygp.")]:
            del sys.modules[key]
    yield


def test_gp3d_m1_q_chi_evaluates_finite():
    # Was unrunnable (128 GB normalisation grid); must now evaluate to finite
    # log-densities.  A regression to the full grid would OOM/hang here.
    m = build_gp_model("gp3d_m1_q_chi")
    th = jnp.asarray(m.fiducial())
    m1 = jnp.array([15.0, 25.0, 35.0, 45.0])
    q = jnp.array([0.5, 0.6, 0.7, 0.8])
    z = jnp.array([0.1, 0.2, 0.3, 0.4])
    chi = jnp.array([0.0, 0.1, -0.1, 0.2])
    lp = m.log_p_pop(m1, q, z, chi, th)
    assert lp.shape == m1.shape
    assert bool(jnp.all(jnp.isfinite(lp))), f"non-finite log_p_pop: {lp}"


@pytest.mark.parametrize("name", ["gp2d_q_chi", "gp_separable", "gp3d_m1_q_chi"])
def test_gp_log_p_pop_handles_2d_inputs(name):
    # A (Nsamp, Nnodes) query (the WL mesh) must return the same shape and match
    # a column-wise 1-D evaluation (the known-good path) entry for entry.
    m = build_gp_model(name)
    th = jnp.asarray(m.fiducial())
    m1a = jnp.array([20.0, 30.0, 40.0])
    qa = jnp.array([0.6, 0.7, 0.8])
    za = jnp.array([0.1, 0.2, 0.3])
    ca = jnp.array([0.0, 0.1, -0.1])
    m1b = jnp.stack([m1a, m1a + 1.0], axis=1)          # (3, 2)
    qb = jnp.stack([qa, qa], axis=1)
    zb = jnp.stack([za, za], axis=1)
    cb = jnp.stack([ca, ca], axis=1)

    r2 = m.log_p_pop(m1b, qb, zb, cb, th)
    assert r2.shape == (3, 2), f"{name}: 2-D query gave shape {r2.shape}"

    ref = jnp.stack(
        [m.log_p_pop(m1b[:, c], qb[:, c], zb[:, c], cb[:, c], th)
         for c in range(m1b.shape[1])],
        axis=1,
    )
    assert bool(jnp.allclose(r2, ref, atol=1e-9, equal_nan=True)), (
        f"{name}: 2-D result does not match column-wise 1-D evaluation"
    )


# ===========================================================================
# 3. The z-conditional normaliser must not batch the z grid.
# ===========================================================================
# ``_znorm_interp`` evaluated the probability-axis normalisation with
# ``jax.vmap(eval_norm_fn)(zg)``.  Inside, the field evaluation forms the
# explicit cross-kernel k(coords, Z) of shape (G, M) before its matvec, so vmap
# materialises the whole (N_z, G, M) cube: for the registered gp4d model
# (G = 48*24*24 = 27648 coarse nodes, M = 10*8*10*8 = 6400 inducing points,
# N_z = 40 at the default zMax = 5) that is 40*27648*6400*8 B = 52.74 GiB of XLA
# scratch for a query of only EIGHT points -- before any event or injection data
# enters, so no --sel_batch_size / auto-blocking setting can reduce it.  Measured
# temp_size_in_bytes (cpu, x64):
#
#     model            master (vmap)   now (lax.map + checkpoint)
#     gp4d forward        52.74 GiB          2.64 GiB
#     gp4d gradient      583.48 GiB         13.91 GiB
#
# The checkpoint matters on its own: a scan keeps its per-iteration residuals for
# the transpose, so a plain lax.map made the gradient WORSE than vmap (756 GiB).

_GIB = 1024 ** 3
# Every registered GP model's forward pass must stay far below the smallest GPU
# in the fleet (40 GB).  Measured maxima: gp4d 2.64 GiB, gp3d_m1_q_chi 0.17 GiB,
# everything else <= 0.08 GiB.
_FWD_TEMP_CEILING_GIB = 4.0


def _probe_query(n=8):
    return (jnp.linspace(6.0, 60.0, n), jnp.linspace(0.2, 0.95, n),
            jnp.linspace(0.05, 1.2, n), jnp.linspace(-0.3, 0.3, n))


@pytest.mark.parametrize("name", list(GP_MODEL_NAMES))
def test_gp_forward_transient_memory_stays_bounded(name):
    """No registered GP model may need multi-GiB scratch to evaluate 8 points.

    Uses XLA's own accounting (``memory_analysis``) on the COMPILED executable,
    so it never allocates the buffer it is guarding against -- the pre-fix gp4d
    would have needed 52.74 GiB to run this test's query."""
    pytest.importorskip("tinygp")
    m = build_gp_model(name)
    th = jnp.asarray(m.fiducial())
    comp = jax.jit(m.log_p_pop).lower(*_probe_query(), th).compile()
    temp_gib = comp.memory_analysis().temp_size_in_bytes / _GIB
    assert temp_gib < _FWD_TEMP_CEILING_GIB, (
        f"{name}: {temp_gib:.3f} GiB of transient memory for an 8-point query "
        f"(ceiling {_FWD_TEMP_CEILING_GIB} GiB); a z-grid or normalisation-grid "
        "change has reintroduced a batched (N_z, G, M) tensor."
    )


def test_gp4d_reverse_mode_transient_memory_stays_bounded():
    """Reverse mode (``--sampler numpyro``) must fit on a fleet GPU too.

    Master needed 583.48 GiB here; a lax.map WITHOUT jax.checkpoint needs
    756.19 GiB (the scan stores every z node's kernel on the tape)."""
    pytest.importorskip("tinygp")
    m = build_gp_model("gp4d")
    th = jnp.asarray(m.fiducial())
    m1, q, z, chi = _probe_query()
    grad = jax.grad(lambda t: jnp.sum(m.log_p_pop(m1, q, z, chi, t)))
    comp = jax.jit(grad).lower(th).compile()
    temp_gib = comp.memory_analysis().temp_size_in_bytes / _GIB
    assert temp_gib < 32.0, f"gp4d gradient needs {temp_gib:.2f} GiB of scratch"


@pytest.fixture
def _small_norm_grids():
    """Shrink the normalisation grids so the (N_z, G, M) cube is cheap enough to
    evaluate BOTH the sequential and the batched z-loop for comparison."""
    saved = (gp._KD_N, gp._ZNORM_N, gp._M1NORM_N, gp._znorm_interp)
    gp._KD_N = {"m1": 8, "q": 6, "chi": 6}
    gp._ZNORM_N = 7
    gp._M1NORM_N = 9
    yield
    gp._KD_N, gp._ZNORM_N, gp._M1NORM_N, gp._znorm_interp = saved


@pytest.mark.parametrize("name", ["gp1d_z", "gp2d_m1_z", "gp2d_q_z", "gp2d_chi_z",
                                  "gp3d_m1_q_z", "gp3d_m1_chi_z", "gp3d_q_chi_z",
                                  "gp4d"])
def test_sequential_znorm_matches_batched_znorm(_small_norm_grids, name):
    """Equivalence of the sequential z-loop with the historical batched one.

    Values agree to floating point (bit-identical for most models; the largest
    difference observed across models and hyperparameter draws is 7.1e-15 nats,
    XLA reassociating the same reduction under scan instead of vmap) and so do
    the reverse-mode gradients."""
    pytest.importorskip("tinygp")

    def _vmap_znorm(eval_norm_fn, z_query):
        zg = jnp.linspace(0.0, gp._ZNORM_HI, gp._ZNORM_N)
        norms = jax.vmap(eval_norm_fn)(zg)
        return jnp.exp(jnp.interp(z_query, zg,
                                  jnp.log(jnp.where(norms > 0, norms, gp._LOGSAFE))))

    m = build_gp_model(name)
    th = np.array(m.fiducial(), dtype=float, copy=True)
    rng = np.random.default_rng(7)
    for i, spec in enumerate(m.param_specs):
        if getattr(spec, "prior_kind", "uniform") == "normal":
            th[i] = rng.normal(0.0, 0.7)      # a non-trivial GP field draw
    th = jnp.asarray(th)

    rq = np.random.default_rng(3)
    n = 12
    m1 = jnp.asarray(rq.uniform(5.0, 90.0, n))
    q = jnp.asarray(rq.uniform(0.1, 1.0, n))
    z = jnp.asarray(rq.uniform(0.01, 4.5, n))
    chi = jnp.asarray(rq.uniform(-0.8, 0.8, n))

    def total(t):
        lp = m.log_p_pop(m1, q, z, chi, t)
        return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))

    out = {}
    for kind, fn in (("batched", _vmap_znorm), ("sequential", gp._znorm_interp)):
        gp._znorm_interp = fn
        out[kind] = (np.asarray(m.log_p_pop(m1, q, z, chi, th)),
                     np.asarray(jax.grad(total)(th)))
    (f_b, g_b), (f_s, g_s) = out["batched"], out["sequential"]
    fin = np.isfinite(f_b)
    assert np.array_equal(np.isfinite(f_b), np.isfinite(f_s)), name
    assert np.abs(f_s[fin] - f_b[fin]).max() < 1e-13, name
    scale = max(float(np.abs(g_b).max()), 1e-30)
    assert np.abs(g_s - g_b).max() / scale < 1e-11, name
