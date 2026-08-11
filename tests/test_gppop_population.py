"""Tests for the binned-GP (gppop) population models.

``gppop`` is the mass-only binned Gaussian-process rate model (Ray et al. 2023,
arXiv:2304.08046) ported into darksirens' population framework; ``gppop_mz``
adds redshift bins.  These check registry wiring, the parameter contract, and
the binned-density behaviour (bin lookup, the (m1,q) Jacobian, normalization).
"""

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
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

    # z beyond the default z-edges (..., 1.2) -> the TOP bin is open-ended, so the
    # rate is the last bin's, carried down only by the 1/(1+z) time dilation.  A
    # hard zero there asserted no mergers above z = 1.2 while the pipeline runs to
    # zMax = 5, making the likelihood -inf for every proposal.
    lp_top = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                             jnp.array([1.0]), jnp.array([0.0]), theta)
    lp_out = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                             jnp.array([5.0]), jnp.array([0.0]), theta)
    assert np.isfinite(np.asarray(lp_out)).all()
    np.testing.assert_allclose(float(lp_out[0]) - float(lp_top[0]),
                               -np.log(6.0 / 2.0), atol=1e-10)

    # Below the first edge (z < 0) is still outside the model's support.
    lp_neg = model.log_p_pop(jnp.array([40.0]), jnp.array([0.5]),
                             jnp.array([-0.1]), jnp.array([0.0]), theta)
    assert np.isneginf(np.asarray(lp_neg)).all()


def test_gppop_mz_carries_source_time_dilation():
    """gppop_mz z-bins are the source-frame R(z); log_p_pop must apply the
    1/(1+z) observer/source time dilation itself (the likelihood's log1p is the
    detector-mass Jacobian, not dilation).  With a FLAT log-rate field (xi = 0,
    all bins equal) the only z-dependence is that dilation, so densities at
    z inside a common (m1, q) bin must scale as 1/(1+z): the pairwise log
    differences equal -log((1+z_i)/(1+z_j))."""
    _require_tinygp_transforms()
    model = get_model("gppop_mz")
    # Fiducial has xi = 0 -> logn = 0 in every bin -> flat source-frame rate.
    theta = jnp.asarray(get_fixed_population_params("gppop_mz"))

    m1, q, chi = jnp.array([40.0]), jnp.array([0.5]), jnp.array([0.0])
    # z values sit in distinct z-bins (edges ...,0.3,0.7,1.2) but the mass/spin
    # factors are identical, isolating the pure time-dilation z-dependence.
    zs = [0.1, 0.5, 0.9]
    lps = [float(model.log_p_pop(m1, q, jnp.array([z]), chi, theta)[0]) for z in zs]
    assert np.all(np.isfinite(lps))

    for a in range(len(zs)):
        for b in range(len(zs)):
            expected = -np.log((1.0 + zs[a]) / (1.0 + zs[b]))
            np.testing.assert_allclose(lps[a] - lps[b], expected, atol=1e-10)

    # Relative densities vs z = 0.1: {1, 1.1/1.5, 1.1/1.9}.
    rel = [np.exp(lp - lps[0]) for lp in lps]
    np.testing.assert_allclose(rel, [1.0, 1.1 / 1.5, 1.1 / 1.9], atol=1e-10)


def test_gppop_mass_only_z_scaling_unchanged():
    """The mass-only ``gppop`` branch (no z-bins) keeps the parametric rate
    evolution (gamma - 1) * log1p(z) -- the fix touches only the z-bearing
    branch.  With a fixed mass/spin point the z-dependence is exactly that."""
    _require_tinygp_transforms()
    model = get_model("gppop")
    theta = jnp.asarray(get_fixed_population_params("gppop"))
    names = [s.name for s in model.param_specs]
    gamma = float(theta[names.index("gamma")])

    m1, q, chi = jnp.array([40.0]), jnp.array([0.5]), jnp.array([0.0])
    zs = [0.1, 0.5, 0.9]
    lps = [float(model.log_p_pop(m1, q, jnp.array([z]), chi, theta)[0]) for z in zs]
    assert np.all(np.isfinite(lps))

    for a in range(len(zs)):
        for b in range(len(zs)):
            expected = (gamma - 1.0) * (np.log1p(zs[a]) - np.log1p(zs[b]))
            np.testing.assert_allclose(lps[a] - lps[b], expected, atol=1e-10)
