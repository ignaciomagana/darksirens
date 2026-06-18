import jax.numpy as jnp

from darksirens.gw.populations.gp import _broadcast_logp_inputs


def test_broadcast_logp_inputs_accepts_sparse_predictive_mesh():
    m1 = jnp.ones((300, 1, 1, 1))
    q = jnp.ones((1, 100, 1, 1))
    z = jnp.ones((1, 1, 10, 1))
    chi = jnp.ones((1, 1, 1, 5))

    m1_f, q_f, z_f, chi_f, out_shape = _broadcast_logp_inputs(m1, q, z, chi)

    assert out_shape == (300, 100, 10, 5)
    assert m1_f.shape == q_f.shape == z_f.shape == chi_f.shape == (300 * 100 * 10 * 5,)


def test_broadcast_logp_inputs_preserves_scalar_output_shape():
    m1_f, q_f, z_f, chi_f, out_shape = _broadcast_logp_inputs(30.0, 0.8, 0.1, 0.0)

    assert out_shape == ()
    assert m1_f.shape == q_f.shape == z_f.shape == chi_f.shape == (1,)
