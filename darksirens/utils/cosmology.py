"""JAX-compatible flat-ΛCDM distance and volume utilities.

The likelihood evaluates cosmological factors for every posterior sample and
selection injection, so these helpers avoid calling Astropy inside JIT-compiled
code.  At import time the module builds a two-dimensional interpolation table of
comoving distance as a function of redshift and ``Om0`` at the Planck-2015
``H0``.  Runtime functions rescale that table by ``H0Planck / H0`` and expose
JIT-able distance, inverse-distance, and differential-volume operations.

Distances are in Mpc, ``H0`` is in km/s/Mpc, and redshift is dimensionless.  The
interpolation grid covers the prior range used by :mod:`darksirens.inference`
with a guard band so that sampler proposals at the edge of the allowed prior do
not silently extrapolate.
"""

import astropy.constants as constants
import astropy.units as u
import jax
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
from jax import jit
from jax import numpy as jnp

from darksirens.utils.interp2d import interp2d

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

zMax = 5
"""Maximum redshift covered by the precomputed interpolation grid."""

H0Planck = Planck15.H0.value
"""Planck-2015 Hubble constant used to build the reference distance grid."""

Om0Planck = float(Planck15.Om0)
"""Planck-2015 matter density used as the center of the interpolation grid."""

speed_of_light = constants.c.to("km/s").value
"""Speed of light in km/s, matching the units of ``H0``."""

zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax + 1), 1000))

rs = []
# The prior in inference/prior.py covers [Om0Planck-0.1, Om0Planck+0.1].
# The interpolation grid must extend strictly beyond those bounds so that
# sampler proposals at or near the prior boundary never extrapolate outside
# the grid (which would return silently wrong distances).  A pad of 0.05
# on each side is sufficient for any reasonable sampler step size.
_OM0_PRIOR_HALF_WIDTH = 0.1
_OM0_GRID_PAD = 0.05
Om0grid = jnp.linspace(
    Om0Planck - _OM0_PRIOR_HALF_WIDTH - _OM0_GRID_PAD,
    Om0Planck + _OM0_PRIOR_HALF_WIDTH + _OM0_GRID_PAD,
    200,
)
for Om0 in Om0grid:
    cosmo = FlatLambdaCDM(H0=H0Planck, Om0=Om0)
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

zgrid = jnp.array(zgrid)
"""Redshift coordinates of the distance interpolation grid."""

rs = jnp.asarray(rs)
rs = rs.reshape(len(Om0grid), len(zgrid))
"""Comoving-distance table indexed by ``(Om0grid, zgrid)`` in Mpc."""


@jit
def E(z, Om0=Om0Planck):
    """Return the dimensionless expansion rate ``H(z) / H0``.

    The expression assumes a spatially flat ΛCDM cosmology with matter density
    ``Om0`` and dark-energy density ``1 - Om0``.  Inputs may be scalars or JAX
    arrays and are broadcast by JAX.
    """
    return jnp.sqrt(Om0 * (1 + z) ** 3 + (1.0 - Om0))


@jit
def r_of_z(z, H0, Om0=Om0Planck):
    """Return line-of-sight comoving distance at redshift ``z``.

    The distance is interpolated from the precomputed ``(Om0, z)`` grid and
    rescaled from the Planck reference Hubble constant to the requested ``H0``.
    Values outside the tabulated grid should be checked with
    :func:`dL_in_z_grid` before use in likelihood calculations.
    """
    return interp2d(Om0, z, Om0grid, zgrid, rs) * (H0Planck / H0)


@jit
def dL_of_z(z, H0, Om0=Om0Planck):
    """Return luminosity distance for redshift ``z`` in Mpc."""
    return (1 + z) * r_of_z(z, H0, Om0)


@jit
def dL_grid_bounds(H0, Om0=Om0Planck):
    """Return the luminosity-distance support covered by ``zgrid``.

    The pair ``(dL_min, dL_max)`` is useful for masking posterior samples before
    inverse interpolation.  The lower bound is slightly above zero because the
    redshift grid starts at ``z = 0`` represented through the log-spaced grid.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0)
    return dL_grid[0], dL_grid[-1]


@jit
def dL_in_z_grid(dL, H0, Om0=Om0Planck):
    """Return a boolean mask for distances supported by the redshift grid."""
    dL_min, dL_max = dL_grid_bounds(H0, Om0)
    return (dL >= dL_min) & (dL <= dL_max)


@jit
def z_of_dL(dL, H0, Om0=Om0Planck):
    """Invert luminosity distance to redshift by one-dimensional interpolation.

    Unsupported distances are returned as ``NaN`` rather than extrapolated.  The
    likelihood converts those values to zero or ``-inf`` contributions depending
    on the surrounding calculation, which prevents out-of-grid samples from
    looking artificially valid.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0)
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    z = jnp.interp(dL, dL_grid, zgrid)
    return jnp.where(in_grid, z, jnp.nan)


@jit
def dV_of_z(z, H0, Om0=Om0Planck):
    """Return the differential comoving-volume factor per steradian.

    The returned value is ``c * r(z)^2 / (H0 * E(z))`` in Mpc^3.  Angular pixel
    areas are applied by the electromagnetic prior code when a prior density per
    HEALPix pixel is required.
    """
    return speed_of_light * r_of_z(z, H0, Om0) ** 2 / (H0 * E(z, Om0))


@jit
def ddL_of_z(z, dL, H0, Om0=Om0Planck):
    """Return ``d dL / dz`` for the flat-ΛCDM luminosity-distance relation.

    The derivative is used when transforming redshift densities into
    luminosity-distance densities.  ``dL`` is accepted as an argument so callers
    that already computed the luminosity distance can avoid recomputing it.
    """
    return dL / (1 + z) + speed_of_light * (1 + z) / (H0 * E(z, Om0))
