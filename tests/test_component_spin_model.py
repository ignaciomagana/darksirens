"""ComponentSpinModel (DS-08): 4-D component-spin population, analytic norm."""
import sys
import types

_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from darksirens.gw.populations.component_spin import (
    COMPONENT_SPIN_COLUMNS,
    ComponentSpinModel,
    default_component_spin,
)

THETAS = [
    (1.6, 4.5, 0.75, 0.9),   # GWTC-3 Default-like
    (2.0, 2.0, 0.0, 1.0),    # fully isotropic tilts
    (1.0, 1.0, 1.0, 0.3),    # flat magnitudes, narrow aligned tilts
    (5.0, 1.5, 0.5, 4.0),    # wide tilt Gaussian
]


@pytest.mark.parametrize("theta", THETAS)
def test_each_factor_normalises_to_one(theta):
    """Deterministic 1-D quadrature: both analytic factors integrate to 1."""
    model = default_component_spin()
    alpha, beta, zeta, sigma_t = theta

    a = jnp.linspace(1e-9, 1.0 - 1e-9, 200_001)
    p_a = jnp.exp(model._log_p_magnitude(a, alpha, beta))
    norm_a = float(jnp.trapezoid(p_a, a))
    # alpha=1 or beta=1 puts finite density at the endpoint; the open-interval
    # grid still integrates to 1 to quadrature accuracy.
    assert abs(norm_a - 1.0) < 1e-3

    c = jnp.linspace(-1.0, 1.0, 200_001)
    p_c = jnp.exp(model._log_p_tilt(c, zeta, sigma_t))
    norm_c = float(jnp.trapezoid(p_c, c))
    assert abs(norm_c - 1.0) < 1e-6


def test_joint_normalises_monte_carlo():
    """MC cross-check of the joint 4-D density (volume of the box = 4)."""
    model = default_component_spin()
    theta = jnp.asarray(THETAS[0])
    rng = np.random.default_rng(7)
    n = 2_000_000
    spin = np.column_stack([
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n),
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(-1.0, 1.0, n),
    ])
    p = np.exp(np.asarray(model.log_prob(jnp.asarray(spin), theta)))
    integral = 4.0 * p.mean()
    err = 4.0 * p.std() / np.sqrt(n)
    assert abs(integral - 1.0) < max(3.0 * err, 1e-3), (integral, err)


def test_out_of_support_is_neg_inf():
    model = default_component_spin()
    theta = jnp.asarray(THETAS[0])
    bad = jnp.asarray([
        [1.2, 0.5, 0.0, 0.0],    # a1 > 1
        [0.5, -0.1, 0.0, 0.0],   # a2 < 0
        [0.5, 0.5, 1.5, 0.0],    # cost1 > 1
        [0.5, 0.5, 0.0, -1.5],   # cost2 < -1
    ])
    assert np.all(np.isneginf(np.asarray(model.log_prob(bad, theta))))
    good = jnp.asarray([[0.3, 0.2, 0.5, -0.4]])
    assert np.isfinite(np.asarray(model.log_prob(good, theta))).all()


def test_requires_spin_block_and_has_no_chieff_density():
    model = default_component_spin()
    theta = jnp.zeros(4)
    with pytest.raises(TypeError, match="component spin basis"):
        model(jnp.zeros(3), theta)
    with pytest.raises(TypeError, match="chi_eff"):
        model._eval_unnorm(jnp.zeros(3), theta)
    assert model.consumes_spin_block
    assert model.spin_columns == COMPONENT_SPIN_COLUMNS == ("a1", "a2", "cost1", "cost2")


def test_registered_preset_labels_and_dispatch():
    from darksirens.gw.populations.registry import get_model

    model = get_model("gwtc3_plpeak_component_spin")
    names = [s.name or s.label for s in model.param_specs]
    for expected in ("alpha_chi", "beta_chi", "zeta_spin", "sigma_t"):
        assert expected in names
    assert len(model.param_specs) == 13

    # The chieff twin keeps its labels untouched.
    chieff_model = get_model("gwtc3_fiducial_plpeak")
    chieff_names = [s.name or s.label for s in chieff_model.param_specs]
    assert "mu_chi" in chieff_names and "alpha_chi" not in chieff_names


def test_log_p_pop_separates_into_mass_and_spin_terms():
    """log_p_pop(component preset) - log_prob(spin) equals the chieff twin's
    mass/pairing/rate part -- the spin factor is exactly the analytic 4-D
    density, with no hidden renormalisation."""
    from darksirens.gw.populations.registry import get_model

    comp = get_model("gwtc3_plpeak_component_spin")
    chieff_twin = get_model("gwtc3_fiducial_plpeak")

    mass_beta = (3.5, 5.0, 65.0, 0.038, 34.0, 5.5, 4.9, 1.1)
    theta_comp = jnp.asarray(mass_beta + (1.6, 4.5, 0.75, 0.9) + (2.9,))
    theta_chieff = jnp.asarray(mass_beta + (0.0, 0.1) + (2.9,))

    rng = np.random.default_rng(11)
    n = 64
    m1 = jnp.asarray(rng.uniform(8.0, 60.0, n))
    q = jnp.asarray(rng.uniform(0.3, 1.0, n))
    z = jnp.asarray(rng.uniform(0.05, 1.0, n))
    chieff = jnp.asarray(rng.uniform(-0.5, 0.5, n))
    spin = jnp.asarray(np.column_stack([
        rng.uniform(0.01, 0.99, n), rng.uniform(0.01, 0.99, n),
        rng.uniform(-0.99, 0.99, n), rng.uniform(-0.99, 0.99, n),
    ]))

    lp_comp = comp.log_p_pop(m1, q, z, chieff, theta_comp, spin=spin)
    lp_spin = comp.spin_component.log_prob(spin, jnp.asarray((1.6, 4.5, 0.75, 0.9)))

    lp_chieff = chieff_twin.log_p_pop(m1, q, z, chieff, theta_chieff)
    ts = jnp.asarray((0.0, 0.1))
    spin_norm = chieff_twin.spin_component._norm(ts)
    lp_chieff_spin = jnp.log(chieff_twin.spin_component(chieff, ts, norm=spin_norm))

    np.testing.assert_allclose(
        np.asarray(lp_comp - lp_spin),
        np.asarray(lp_chieff - lp_chieff_spin),
        rtol=1e-10, atol=1e-10,
    )

    # And a chieff-basis pairing (no spin block) is refused, not approximated.
    with pytest.raises(TypeError, match="component spin basis"):
        comp.log_p_pop(m1, q, z, chieff, theta_comp)
