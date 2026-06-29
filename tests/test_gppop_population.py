"""Tests for the binned-GP (gppop) population models.

``gppop`` is the mass-only binned Gaussian-process rate model (Ray et al. 2023,
arXiv:2304.08046) ported into darksirens' population framework; ``gppop_mz``
adds redshift bins.  These check registry wiring, the parameter contract, and
the binned-density behaviour (bin lookup, the (m1,q) Jacobian, normalization).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from darksirens.gw.populations.registry import (
    available_models,
    get_model,
    get_fixed_population_params,
    pop_model_prior_parser,
)
from darksirens.gw.populations.utils import get_mass_grid, get_q_grid


# --------------------------------------------------------------------------
# Registry wiring + parameter contract (no tinygp needed)
# --------------------------------------------------------------------------

def test_gppop_models_registered():
    models = available_models()
    assert "gppop" in models
    assert "gppop_mz" in models


def test_gppop_param_contract():
    model = get_model("gppop")
    names = [s.name for s in model.param_specs]
    for required in ("log_amp", "log_ls_m", "mu_chi", "sigma_chi", "gamma"):
        assert required in names, required
    assert any(n.startswith("xi_") for n in names)
    assert "log_ls_z" not in names  # mass-only

    lows, highs, labels, kinds, _latex = pop_model_prior_parser("gppop")
    assert len(lows) == len(highs) == len(labels) == len(kinds) == len(names)
    # whitened latents declare a standard-normal prior to the sampler.
    for spec, kind in zip(model.param_specs, kinds):
        if spec.name.startswith("xi_"):
            assert kind == ("normal", 0.0, 1.0)
    fid = get_fixed_population_params("gppop")
    assert len(fid) == len(labels)
    # xi fiducials are exactly zero (flat field).
    for spec, value in zip(model.param_specs, fid):
        if spec.name.startswith("xi_"):
            assert float(value) == 0.0


def test_gppop_mz_param_contract():
    model = get_model("gppop_mz")
    names = [s.name for s in model.param_specs]
    assert "log_ls_z" in names      # redshift length scale present
    assert "gamma" not in names     # z-bins carry the free-form rate instead
    fid = get_fixed_population_params("gppop_mz")
    _l, _h, labels, _k, _x = pop_model_prior_parser("gppop_mz")
    assert len(fid) == len(labels) == len(names)
    # gppop_mz has more bins (mass x z) than mass-only gppop.
    assert get_model("gppop_mz").M >= get_model("gppop").M


# --------------------------------------------------------------------------
# Binned density behaviour (needs tinygp for the kernel/Cholesky)
# --------------------------------------------------------------------------

def _require_tinygp_transforms():
    tinygp = pytest.importorskip("tinygp")
    if not hasattr(tinygp, "transforms"):
        pytest.skip("tinygp.transforms is required")


def test_gppop_density_finite_inside_inf_outside():
    _require_tinygp_transforms()
    model = get_model("gppop")
    theta = jnp.asarray(get_fixed_population_params("gppop"))

    lp_in = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                            jnp.array([0.3]), jnp.array([0.0]), theta)
    assert np.isfinite(np.asarray(lp_in)).all()

    lp_above = model.log_p_pop(jnp.array([500.0]), jnp.array([0.5]),
                               jnp.array([0.3]), jnp.array([0.0]), theta)
    assert np.isneginf(np.asarray(lp_above)).all()

    lp_below = model.log_p_pop(jnp.array([2.0]), jnp.array([0.9]),
                               jnp.array([0.3]), jnp.array([0.0]), theta)
    assert np.isneginf(np.asarray(lp_below)).all()


def test_gppop_mass_shape_is_normalised():
    _require_tinygp_transforms()
    model = get_model("gppop")
    theta = jnp.asarray(get_fixed_population_params("gppop"))
    amp = jnp.exp(theta[0])
    ls = jnp.exp(theta[1])
    xi = theta[2:2 + model.M]
    logn = model._bin_log_rates(amp, [ls, ls], xi)

    mg, qg = get_mass_grid(), get_q_grid()
    M1, Q = jnp.meshgrid(mg, qg, indexing="ij")
    pun = model._binned_density(M1.ravel(), Q.ravel(),
                                jnp.zeros(M1.size), logn).reshape(M1.shape)
    p = pun / model._mass_norm(logn)
    integral = jnp.trapezoid(jnp.trapezoid(p, qg, axis=1), mg)
    np.testing.assert_allclose(float(integral), 1.0, rtol=1e-5)


def test_gppop_rate_constant_within_bin():
    _require_tinygp_transforms()
    model = get_model("gppop")
    theta = jnp.asarray(get_fixed_population_params("gppop"))
    amp = jnp.exp(theta[0])
    ls = jnp.exp(theta[1])
    xi = theta[2:2 + model.M]
    logn = model._bin_log_rates(amp, [ls, ls], xi)

    # Two (m1, q) points sharing the same (m1, m2) bin (default edges
    # ...,25,40,65,...): m1 in [40,65), m2 in [25,40).  The recovered per-bin
    # rate n = density * (q m1) must be identical.
    p1 = model._binned_density(jnp.array([45.0]), jnp.array([30.0 / 45.0]),
                               jnp.zeros(1), logn)
    p2 = model._binned_density(jnp.array([60.0]), jnp.array([35.0 / 60.0]),
                               jnp.zeros(1), logn)
    n1 = float(p1[0] * (30.0 / 45.0) * 45.0)
    n2 = float(p2[0] * (35.0 / 60.0) * 60.0)
    np.testing.assert_allclose(n1, n2, rtol=1e-10)


def test_gppop_mz_redshift_binning():
    _require_tinygp_transforms()
    model = get_model("gppop_mz")
    theta = jnp.asarray(get_fixed_population_params("gppop_mz"))

    lp_in = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                            jnp.array([0.5]), jnp.array([0.0]), theta)
    assert np.isfinite(np.asarray(lp_in)).all()

    # z beyond the default z-edges (..., 1.2) -> out of bin -> zero density.
    lp_out = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                             jnp.array([5.0]), jnp.array([0.0]), theta)
    assert np.isneginf(np.asarray(lp_out)).all()
