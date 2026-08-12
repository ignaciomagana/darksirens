import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import sys
import types

# Parametric tests do not evaluate GP models, but importing the population
# registry imports optional GP classes. Keep these tests independent of tinygp.
if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub


from darksirens.gw.populations.base import ParamSpec
from darksirens.gw.populations.parametric import (
    BrokenPowerLaw,
    PowerLaw,
    PowerLawPairing,
    TruncatedGaussianSpin,
)
from darksirens.gw.populations.registry import get_fixed_population_params, pop_model_parser
from darksirens.gw.populations.utils import (
    configure_normalization_grids,
    get_chi_grid,
    get_mass_grid,
    get_q_grid,
    normalization_grid_settings,
)


def teardown_function():
    configure_normalization_grids(n_mass=500, n_q=200, n_chi=200)


def _relative_difference(coarse, reference):
    coarse = float(coarse)
    reference = float(reference)
    return abs(coarse - reference) / max(abs(reference), 1.0e-300)


def test_normalization_grid_settings_are_configurable_and_cached():
    configure_normalization_grids(n_mass=321, n_q=123, n_chi=77)

    settings = normalization_grid_settings()
    assert settings.n_mass == 321
    assert settings.n_q == 123
    assert settings.n_chi == 77
    assert get_mass_grid().shape == (321,)
    assert get_q_grid().shape == (123,)
    assert get_chi_grid().shape == (77,)


def test_parametric_mass_norms_converge_near_minimum_smoothing_widths():
    pl = PowerLaw(
        ParamSpec("alpha", -4.0, 6.0),
        ParamSpec("mmin", 2.0, 10.0),
        ParamSpec("mmax", 50.0, 100.0),
        ParamSpec("dmmin", 0.01, 10.0),
        ParamSpec("dmmax", 0.01, 20.0),
    )
    bpl = BrokenPowerLaw(
        ParamSpec("alpha1", 0.0, 6.0),
        ParamSpec("alpha2", 0.0, 6.0),
        ParamSpec("mb", 20.0, 50.0),
        ParamSpec("mmin", 2.0, 10.0),
        ParamSpec("mmax", 50.0, 100.0),
        ParamSpec("dmmin", 0.01, 10.0),
        ParamSpec("dmmax", 0.01, 20.0),
    )
    cases = [
        (pl, jnp.array([2.3, 2.0, 50.0, 0.01, 0.01])),
        (bpl, jnp.array([1.6, 3.8, 35.0, 2.0, 50.0, 0.01, 0.01])),
    ]

    configure_normalization_grids(n_mass=500)
    coarse = [component._norm(theta) for component, theta in cases]

    configure_normalization_grids(n_mass=20000)
    reference = [component._norm(theta) for component, theta in cases]

    for coarse_norm, reference_norm in zip(coarse, reference):
        assert _relative_difference(coarse_norm, reference_norm) < 3.0e-3


def _dense_mass_grid():
    return jnp.linspace(1.0, 200.0, 200_001)      # h = 1e-3 Msun


def test_mass_norm_is_smooth_and_differentiable_in_the_sampled_edge():
    """The normaliser of a component whose truncation edge is SAMPLED must not be
    a fixed-grid staircase.

    With ``dm_min`` below the mass-grid spacing (h = 0.4 Msun at n_mass = 500,
    against a DM_MIN prior floor of 0.01) the old base-class trapezoid saw a hard
    edge at a sampled location: ``_norm`` was exactly flat over each 0.4-Msun cell
    and dropped 10.2% at every node crossing, producing 0.112-nat discontinuities
    in ``log_p_pop`` and ``d log p/d m_min`` = 0 inside a cell (1.37 at a node)
    against a dense-quadrature reference of 0.27.
    """
    pl = PowerLaw(
        ParamSpec("alpha", -4.0, 6.0),
        ParamSpec("mmin", 2.0, 10.0),
        ParamSpec("mmax", 50.0, 100.0),
        ParamSpec("dmmin", 0.01, 10.0),
        ParamSpec("dmmax", 0.01, 20.0),
    )

    def theta(mmin):
        return jnp.array([2.3, mmin, 80.0, 0.01, 10.0])

    def norm(mmin):
        return pl._norm(theta(mmin))

    def dense_norm(mmin):
        g = _dense_mass_grid()
        return float(jnp.trapezoid(pl._eval_unnorm(g, theta(mmin)), g))

    # (a) exact against a dense quadrature of the same integrand.
    for mmin in (5.0, 5.5, 6.0):
        np.testing.assert_allclose(float(norm(mmin)), dense_norm(mmin), rtol=1e-6)

    # (b) no staircase: strictly monotone with steps that never jump by a factor
    # of two (the old normaliser was flat over a cell, then dropped 10%).
    xs = np.arange(4.8, 6.4, 0.05)
    steps = np.abs(np.diff(np.array([float(norm(x)) for x in xs])))
    assert np.all(np.diff(np.array([float(norm(x)) for x in xs])) < 0.0)
    assert steps.max() < 2.0 * steps.min(), (steps.min(), steps.max())

    # (c) the d/dm_min channel is alive and correct.
    grad = jax.grad(lambda x: jnp.log(pl(jnp.array(8.0), theta(x))))
    for mmin in (5.0, 5.5, 6.0):
        h = 1.0e-3
        ref = (np.log(dense_norm(mmin - h)) - np.log(dense_norm(mmin + h))) / (2 * h)
        np.testing.assert_allclose(float(grad(mmin)), ref, rtol=1e-3)


def test_pairing_norm_is_smooth_and_differentiable_in_the_support_edge():
    """The pairing normaliser must follow the sampled support edge q_cut = m_min/m1.

    On the fixed q grid ``N(m1) = int p(q|m1) dq`` only changed when ``q_cut``
    crossed a node: ``p(q|m1)`` was bit-identical over m1 in [60, 62] (the true
    normaliser grows 3% there) and ``d p/d m_min`` was -0.0 almost everywhere with
    +1.37 spikes wherever a node landed inside the taper window, against a true
    value of 0.005.
    """
    pairing = PowerLawPairing(ParamSpec("beta", -2.0, 7.0))
    th = jnp.array([1.0])
    dm = 0.01

    def dens(m1, q, mmin):
        return pairing(jnp.asarray(m1), jnp.asarray(q), mmin, dm, th)

    def dense_dens(m1, q, mmin):
        qg = jnp.linspace(0.0, 1.0, 2_000_001)
        I = float(jnp.trapezoid(pairing._eval_unnorm(jnp.asarray(m1), qg, mmin, dm, th), qg))
        return float(pairing._eval_unnorm(jnp.asarray(m1), jnp.asarray(q), mmin, dm, th)) / I

    # (a) m1 dependence is resolved, not piecewise constant.
    m1s = np.linspace(60.0, 62.0, 9)
    vals = np.array([float(dens(m, 0.9, 5.0)) for m in m1s])
    assert np.all(np.diff(vals) != 0.0)
    np.testing.assert_allclose(vals[0], dense_dens(60.0, 0.9, 5.0), rtol=1e-3)
    np.testing.assert_allclose(vals[-1], dense_dens(62.0, 0.9, 5.0), rtol=1e-3)

    # (b) the d/dm_min channel matches a finite difference of the same quadrature.
    grad = jax.grad(lambda mm: dens(60.0, 0.9, mm))
    for mmin in (5.0, 5.5, 6.0):
        h = 0.25
        fd = (float(dens(60.0, 0.9, mmin + h)) - float(dens(60.0, 0.9, mmin - h))) / (2 * h)
        np.testing.assert_allclose(float(grad(mmin)), fd, rtol=5e-2)


def test_pairing_and_spin_norms_converge_near_narrow_features():
    pairing = PowerLawPairing(ParamSpec("beta", -2.0, 7.0))
    spin = TruncatedGaussianSpin(
        ParamSpec("mu_chi", -1.0, 1.0),
        ParamSpec("sigma_chi", 0.01, 1.0),
    )
    m1 = jnp.array([3.0, 10.0, 80.0])
    q = jnp.array([0.8, 0.5, 0.3])
    pairing_theta = jnp.array([2.0])
    spin_theta = jnp.array([0.0, 0.01])

    configure_normalization_grids(n_q=200, n_chi=200)
    coarse_pair = pairing(m1, q, 2.0, 0.01, pairing_theta)
    coarse_spin_norm = spin._norm(spin_theta)

    configure_normalization_grids(n_q=10000, n_chi=10000)
    ref_pair = pairing(m1, q, 2.0, 0.01, pairing_theta)
    ref_spin_norm = spin._norm(spin_theta)

    np.testing.assert_allclose(coarse_pair, ref_pair, rtol=1.5e-2, atol=0.0)
    assert _relative_difference(coarse_spin_norm, ref_spin_norm) < 1.0e-3


def test_representative_population_model_is_stable_at_production_grid_sizes():
    theta = get_fixed_population_params("powerlaw+peak")
    log_p_pop = pop_model_parser("powerlaw+peak")
    m1 = jnp.array([8.0, 20.0, 45.0, 70.0])
    q = jnp.array([0.4, 0.7, 0.9, 0.5])
    z = jnp.array([0.05, 0.2, 0.6, 1.0])
    chi = jnp.array([-0.1, 0.0, 0.2, 0.5])

    configure_normalization_grids(n_mass=500, n_q=200, n_chi=200)
    coarse = log_p_pop(m1, q, z, chi, theta)

    configure_normalization_grids(n_mass=3000, n_q=2000, n_chi=2000)
    reference = log_p_pop(m1, q, z, chi, theta)

    np.testing.assert_allclose(coarse, reference, rtol=2.0e-2, atol=2.0e-2)
