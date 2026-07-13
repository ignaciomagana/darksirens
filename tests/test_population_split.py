"""
test_population_split.py
------------------------
The additive split of the population density used by the flow-surrogate
likelihood path:

    log_p_pop(m1, q, z, chieff, theta)
        == log_p_massspin(m1, q, chieff, theta) + log_rate_z(z, theta)

for shared redshift evolution (powerlaw with shared gamma, and @md), plus the
shared-spin factorisation

    mixture(m1, q, chieff, tm) == mass_q_density(m1, q, tm) * spin(chieff)

that the (m1, q) / chieff samplers rely on, and the guards for the
per-component cases where neither identity exists.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.gw.populations.registry import (
    get_fixed_population_params,
    get_model,
)

RNG = np.random.default_rng(20260712)


def _random_points(n=256):
    m1 = RNG.uniform(3.0, 120.0, n)
    q = RNG.uniform(0.05, 1.0, n)
    z = RNG.uniform(0.0, 3.0, n)
    chieff = RNG.uniform(-0.99, 0.99, n)
    return (jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z), jnp.asarray(chieff))


def _theta_variants(name):
    """Fiducial theta plus a couple of jittered variants inside the priors."""
    fid = np.asarray(get_fixed_population_params(name), dtype=np.float64)
    model = get_model(name)
    lows, highs, _ = model.prior_bounds()
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    variants = [fid]
    for frac in (0.35, 0.65):
        variants.append(lows + frac * (highs - lows))
    return model, [jnp.asarray(t) for t in variants]


@pytest.mark.parametrize("name", ["powerlaw+peak", "powerlaw+peak@md"])
def test_additive_split_matches_log_p_pop(name):
    model, thetas = _theta_variants(name)
    assert model.has_additive_rate_split
    m1, q, z, chieff = _random_points()
    for theta in thetas:
        full = model.log_p_pop(m1, q, z, chieff, theta)
        split = model.log_p_massspin(m1, q, chieff, theta) + model.log_rate_z(z, theta)
        # -inf sentinels must agree exactly; finite values to float tolerance.
        finite = np.isfinite(np.asarray(full))
        assert np.array_equal(finite, np.isfinite(np.asarray(split)))
        np.testing.assert_allclose(
            np.asarray(full)[finite], np.asarray(split)[finite], rtol=0, atol=1e-12
        )


def test_mass_q_density_times_spin_matches_mixture():
    model, thetas = _theta_variants("powerlaw+peak")
    mix = model.mixture
    assert mix.shared_spin
    m1, q, _, chieff = _random_points()
    for theta in thetas:
        tm = model.mixture_theta(theta)
        ts = mix.spin_theta(tm)
        spin = mix.spin_components[0]
        spin_density = spin(chieff, ts, norm=spin._norm(ts))
        joint = mix(m1, q, chieff, tm)
        product = mix.mass_q_density(m1, q, tm) * spin_density
        np.testing.assert_allclose(
            np.asarray(joint), np.asarray(product), rtol=1e-12, atol=0
        )


def test_per_component_gamma_guards():
    model = get_model("powerlaw+peak+powerlaw", shared_gamma=False)
    assert not model.has_additive_rate_split
    n = model.mixture.n_params + model.mixture.k
    theta = jnp.asarray(np.linspace(0.2, 0.8, n))
    z = jnp.asarray([0.1, 0.5])
    with pytest.raises(NotImplementedError):
        model.log_rate_z(z, theta)
    with pytest.raises(NotImplementedError):
        model.log_p_massspin(jnp.ones(2) * 30, jnp.ones(2) * 0.8, jnp.zeros(2), theta)


def test_per_component_spin_guards():
    # Two mass components with per-component spin -> no shared-spin factorisation.
    model = get_model("powerlaw+peak", shared_spin=False)
    mix = model.mixture
    assert not mix.shared_spin
    tm = jnp.asarray(
        np.linspace(0.2, 0.8, mix.n_params)
    )
    with pytest.raises(NotImplementedError):
        mix.mass_q_density(jnp.ones(2) * 30, jnp.ones(2) * 0.8, tm)
    with pytest.raises(NotImplementedError):
        mix.spin_theta(tm)


def test_log_p_pop_unchanged_for_per_component_gamma():
    # The non-split branch must keep working through the refactor.
    model = get_model("powerlaw+peak+powerlaw", shared_gamma=False)
    n = model.mixture.n_params + model.mixture.k
    theta = jnp.asarray(np.linspace(0.2, 0.8, n))
    m1, q, z, chieff = _random_points(64)
    out = model.log_p_pop(m1, q, z, chieff, theta)
    assert out.shape == (64,)
    assert bool(np.all(np.isfinite(np.asarray(out)) | (np.asarray(out) == -np.inf)))
