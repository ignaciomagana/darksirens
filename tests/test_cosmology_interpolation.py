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
