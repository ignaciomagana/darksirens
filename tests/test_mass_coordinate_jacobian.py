import jax.numpy as jnp
import pytest

from darksirens.inference import utils as inference_utils
from darksirens.utils.containers import CosmoParams


def test_log_sample_weight_includes_m1_q_coordinate_jacobian(monkeypatch):
    monkeypatch.setattr(
        inference_utils, "z_of_dL", lambda dL, H0, Om0: dL * 0.0 + 0.25
    )
    monkeypatch.setattr(
        inference_utils, "ddL_of_z", lambda z, dL, H0, Om0: dL * 0.0 + 4.0
    )

    def log_p_pop(m1src, q, z, chieff, pop_params):
        return jnp.zeros_like(m1src)

    def log_prior_z(z, pix, catalog):
        return jnp.zeros_like(z)

    weight = inference_utils.log_sample_weight(
        m1det=jnp.array(50.0),
        q=jnp.array(0.5),
        dL=jnp.array(100.0),
        chieff=jnp.array(0.0),
        pix=jnp.array(0),
        prior_wt=jnp.array(2.0),
        cosmo=CosmoParams(H0=jnp.array(70.0), Om0=jnp.array(0.3)),
        survey=None,
        pop_params=jnp.array([]),
        catalog=None,
        log_p_pop_fn=log_p_pop,
        log_prior_z_fn=log_prior_z,
    )

    z = 0.25
    m1src = 50.0 / (1.0 + z)
    expected = (
        -jnp.log(4.0)
        - 2.0 * jnp.log1p(z)
        - jnp.log(m1src)
        - jnp.log(2.0)
    )
    assert weight == pytest.approx(float(expected))
