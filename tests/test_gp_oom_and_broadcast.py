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
import jax
import jax.numpy as jnp
import pytest

from darksirens.gw.populations.gp import build_gp_model


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
