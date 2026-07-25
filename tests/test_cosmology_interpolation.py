import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import jax.numpy as jnp
import numpy as np

from darksirens.utils.cosmology import dL_of_z, dV_of_z, r_of_z, z_of_dL


def test_cpl_fiducial_reproduces_flat_lambdacdm_distances_and_volume():
    H0 = 67.74
    Om0 = 0.3075
    z = jnp.array([0.0, 0.03, 0.1, 0.5, 1.0, 2.0])
    lcdm = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0)

    np.testing.assert_allclose(
        np.asarray(r_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.comoving_distance(np.asarray(z)).to_value(u.Mpc),
        rtol=5e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(dL_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.luminosity_distance(np.asarray(z)).to_value(u.Mpc),
        rtol=5e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(dV_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.differential_comoving_volume(np.asarray(z)).to_value(u.Mpc**3 / u.sr),
        rtol=5e-5,
        atol=1e-8,
    )


def test_cpl_fiducial_inverse_distance_matches_lambdacdm_redshift():
    H0 = 67.74
    Om0 = 0.3075
    z = jnp.array([0.02, 0.1, 0.5, 1.0, 2.0])
    dL = dL_of_z(z, H0, Om0, w0=-1.0, wa=0.0)

    actual = z_of_dL(dL, H0, Om0, w0=-1.0, wa=0.0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(z), rtol=2e-4, atol=2e-6)


# ============================================================================
# Separable interpolation path (interpnd_scalar_head)
# ============================================================================
#
# `r_of_z` contracts the three scalar cosmology axes into a 1-D r(z) curve
# before the redshift lookup, instead of gathering 16 corners out of the full
# (Om0, w0, wa, z) table. Multilinear interpolation is a tensor product, so this
# is the same number up to floating-point reassociation — these tests pin that
# equivalence, including the NaN/out-of-range masking and the gradients (the
# masking exists to stop a NaN poisoning the backward pass, so it has to be
# checked on the backward pass too, not just the forward one).

import pytest

from darksirens.utils import cosmology as _cosmo
from darksirens.utils.interp2d import interpnd, interpnd_scalar_head

_GRIDS = (_cosmo.Om0grid, _cosmo.w0grid, _cosmo.wagrid, _cosmo.zgrid)


def _r_of_z_interpnd(z, H0, Om0, w0, wa):
    """The pre-optimisation general path, kept here as the reference."""
    return interpnd(
        (Om0, w0, wa, z), _GRIDS, _cosmo.rs, fill_value=jnp.nan
    ) * (_cosmo.H0Planck / H0)


def test_separable_matches_general_path_over_random_cosmologies():
    rng = np.random.default_rng(20260725)
    worst = 0.0
    for _ in range(40):
        Om0 = float(rng.uniform(0.16, 0.45))
        w0 = float(rng.uniform(-2.2, 0.2))
        wa = float(rng.uniform(-2.4, 2.4))
        H0 = float(rng.uniform(50.0, 110.0))
        z = jnp.asarray(rng.uniform(1e-3, 4.9, 4000))
        want = np.asarray(_r_of_z_interpnd(z, H0, Om0, w0, wa))
        got = np.asarray(r_of_z(z, H0, Om0, w0, wa))
        worst = max(worst, float(np.max(np.abs(got / want - 1.0))))
    assert worst < 1e-13, f"separable path drifted from interpnd: {worst:.2e}"


@pytest.mark.parametrize("Om0,w0,wa,label", [
    (0.05, -1.0, 0.0, "Om0 below grid"),
    (0.90, -1.0, 0.0, "Om0 above grid"),
    (0.31, -9.0, 0.0, "w0 below grid"),
    (0.31, 5.0, 0.0, "w0 above grid"),
    (0.31, -1.0, -9.0, "wa below grid"),
    (0.31, -1.0, 9.0, "wa above grid"),
    (0.31, -1.0, 0.0, "all in range"),
])
def test_separable_nan_masking_matches_per_axis(Om0, w0, wa, label):
    """Out-of-range on ANY axis -> NaN, identically on both paths."""
    z = jnp.array([0.5, -1.0, 99.0, jnp.nan])
    want = np.asarray(_r_of_z_interpnd(z, 70.0, Om0, w0, wa))
    got = np.asarray(r_of_z(z, 70.0, Om0, w0, wa))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=label)
    finite = ~np.isnan(want)
    if finite.any():
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-13)


def test_separable_gradients_match_general_path():
    """Gradients w.r.t. all five arguments, not just the forward value."""
    z = jnp.asarray(np.linspace(0.05, 4.0, 128))
    f = lambda H0, Om0, w0, wa: jnp.sum(r_of_z(z, H0, Om0, w0, wa))
    g = lambda H0, Om0, w0, wa: jnp.sum(_r_of_z_interpnd(z, H0, Om0, w0, wa))
    got = jax.grad(f, argnums=(0, 1, 2, 3))(70.0, 0.31, -0.95, 0.1)
    want = jax.grad(g, argnums=(0, 1, 2, 3))(70.0, 0.31, -0.95, 0.1)
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-11)


def test_separable_gradient_is_finite_for_out_of_range_queries():
    """The NaN-sanitisation contract: a masked query must not poison the grad."""
    z = jnp.array([0.5, 99.0, jnp.nan, 1.0])
    f = lambda Om0: jnp.sum(jnp.nan_to_num(r_of_z(z, 70.0, Om0, -1.0, 0.0)))
    assert np.isfinite(float(jax.grad(f)(0.31)))


def test_vmap_over_cosmology_samples_keeps_the_fast_path():
    """The inference pattern: one cosmology per posterior sample."""
    z = jnp.asarray(np.linspace(0.01, 4.5, 500))
    rng = np.random.default_rng(7)
    n = 64
    H0 = jnp.asarray(rng.uniform(60.0, 90.0, n))
    Om0 = jnp.asarray(rng.uniform(0.22, 0.40, n))
    w0 = jnp.asarray(rng.uniform(-1.8, -0.4, n))
    wa = jnp.asarray(rng.uniform(-1.5, 1.5, n))
    got = jax.vmap(r_of_z, in_axes=(None, 0, 0, 0, 0))(z, H0, Om0, w0, wa)
    want = jax.vmap(_r_of_z_interpnd, in_axes=(None, 0, 0, 0, 0))(z, H0, Om0, w0, wa)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13)


def test_array_valued_cosmology_falls_back_to_interpnd():
    """A caller that maps over Om0 directly still gets correct answers."""
    Om0 = jnp.asarray([0.28, 0.33, 0.36])
    z = jnp.asarray([1.0, 2.0, 3.0])
    got = np.asarray(r_of_z(z, 70.0, Om0, -1.0, 0.0))
    want = np.asarray(_r_of_z_interpnd(z, 70.0, Om0, -1.0, 0.0))
    np.testing.assert_allclose(got, want, rtol=1e-14)
    assert np.all(np.isfinite(got))


def test_interpnd_scalar_head_rejects_non_scalar_leading_axes():
    with pytest.raises(ValueError, match="scalar leading coordinates"):
        interpnd_scalar_head(
            (jnp.asarray([0.3, 0.31]), -1.0, 0.0, jnp.asarray([1.0, 2.0])),
            _GRIDS, _cosmo.rs,
        )


def test_interpnd_scalar_head_generalises_beyond_the_cosmology_table():
    """Same contract on a small hand-built table, any number of leading axes."""
    ax0 = jnp.asarray([0.0, 1.0, 2.0])
    ax1 = jnp.asarray([0.0, 0.5, 1.0, 1.5])
    tail = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.normal(size=(ax0.size, ax1.size, tail.size)))
    q = jnp.asarray([0.25, 1.75, 3.5])
    got = interpnd_scalar_head((0.7, 1.1, q), (ax0, ax1, tail), values, fill_value=jnp.nan)
    want = interpnd((0.7, 1.1, q), (ax0, ax1, tail), values, fill_value=jnp.nan)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13)
