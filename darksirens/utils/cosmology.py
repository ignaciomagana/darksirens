"""JAX-compatible flat-CPL distance and volume utilities.

The likelihood evaluates cosmological factors for every posterior sample and
selection injection, so these helpers avoid calling Astropy inside JIT-compiled
code.  At import time the module builds a four-dimensional interpolation table
of comoving distance as a function of redshift, ``Om0``, ``w0``, and ``wa`` at
the Planck-2015 ``H0``.  Runtime functions rescale that table by
``H0Planck / H0`` and expose JIT-able distance, inverse-distance, and
differential-volume operations.

Distances are in Mpc, ``H0`` is in km/s/Mpc, and redshift is dimensionless.  The
interpolation grid covers the prior range used by :mod:`darksirens.inference`
with guard bands so that sampler proposals at the edge of the allowed prior do
not silently extrapolate.
"""

import os

import astropy.constants as constants
import jax
import numpy as np
from astropy.cosmology import Flatw0waCDM, Planck15
from jax import jit
from jax import numpy as jnp

from darksirens.utils.interp2d import interpnd

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

zMax = float(os.environ.get("DARKSIRENS_ZMAX", 5))
"""Maximum redshift covered by the precomputed interpolation grid.

Default 5 (unchanged numerics). Override with the DARKSIRENS_ZMAX environment
variable (read once at import) for high-redshift studies — e.g. Madau-Dickinson
rate inference with strongly lensed sources whose true z exceeds 5. The z-grid
node count scales with the log range so low-z interpolation accuracy is
preserved; darksirens.redshift.grid reads the same variable, keeping the two
grids consistent."""

H0Planck = Planck15.H0.value
"""Planck-2015 Hubble constant used to build the reference distance grid."""

Om0Planck = float(Planck15.Om0)
"""Planck-2015 matter density used as the center of the interpolation grid."""

w0Fiducial = -1.0
"""Fiducial CPL present-day dark-energy equation-of-state parameter."""

waFiducial = 0.0
"""Fiducial CPL dark-energy evolution parameter."""

speed_of_light = constants.c.to("km/s").value
"""Speed of light in km/s, matching the units of ``H0``."""

# The default prior in inference/prior.py covers [Om0Planck-0.1,
# Om0Planck+0.1].  The interpolation grid must extend strictly beyond those
# bounds so that sampler proposals at or near the prior boundary never
# extrapolate outside the grid.  The CPL ranges below use the same convention:
# an intended prior range plus a guard pad on both sides.
_OM0_PRIOR_HALF_WIDTH = 0.1
_OM0_GRID_PAD = 0.05
_W0_PRIOR_HALF_WIDTH = 1.0
_W0_GRID_PAD = 0.25
_WA_PRIOR_HALF_WIDTH = 2.0
_WA_GRID_PAD = 0.5

Om0PriorLower = Om0Planck - _OM0_PRIOR_HALF_WIDTH
Om0PriorUpper = Om0Planck + _OM0_PRIOR_HALF_WIDTH
w0PriorLower = w0Fiducial - _W0_PRIOR_HALF_WIDTH
w0PriorUpper = w0Fiducial + _W0_PRIOR_HALF_WIDTH
waPriorLower = waFiducial - _WA_PRIOR_HALF_WIDTH
waPriorUpper = waFiducial + _WA_PRIOR_HALF_WIDTH

# Node count scales with the log-z range (500 at the default zMax=5) so a
# raised DARKSIRENS_ZMAX keeps the same low-z node density.
_ZGRID_NODES = max(500, int(round(500 * np.log(zMax + 1.0) / np.log(6.0))))
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax + 1), _ZGRID_NODES))

Om0grid = jnp.linspace(
    Om0PriorLower - _OM0_GRID_PAD,
    Om0PriorUpper + _OM0_GRID_PAD,
    31,
)
"""Matter-density coordinates of the distance interpolation grid."""

w0grid = jnp.linspace(
    w0PriorLower - _W0_GRID_PAD,
    w0PriorUpper + _W0_GRID_PAD,
    21,
)
"""CPL ``w0`` coordinates of the distance interpolation grid."""

wagrid = jnp.linspace(
    waPriorLower - _WA_GRID_PAD,
    waPriorUpper + _WA_GRID_PAD,
    21,
)
"""CPL ``wa`` coordinates of the distance interpolation grid."""

# Keep an Astropy cosmology object attached to the grid construction so that the
# tabulated model is explicitly the same flat CPL family Astropy calls
# ``Flatw0waCDM``.  The grid values themselves are produced by the equivalent
# analytic CPL expansion law below; this avoids thousands of slow Astropy
# distance integrations at import time while preserving JAX-free grid setup.
_reference_cosmology = Flatw0waCDM(
    H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial
)


def _cpl_E_numpy(z, Om0, w0, wa):
    """NumPy version of the flat CPL expansion law used during grid setup."""
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * np.exp(-3.0 * wa * z / one_plus_z)
    )
    return np.sqrt(Om0 * one_plus_z**3 + dark_energy)


def _cumulative_trapezoid(y, x):
    """Return cumulative trapezoid integrals with zero initial value."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _build_cpl_distance_grid():
    """Build ``r(Om0, w0, wa, z)`` in Mpc for the flat CPL grid."""
    # The Astropy object above documents the model family mirrored by this table.
    r_grid = np.empty(
        (len(Om0grid), len(w0grid), len(wagrid), len(zgrid)), dtype=np.float64
    )
    z = np.asarray(zgrid, dtype=np.float64)
    for i, Om0 in enumerate(np.asarray(Om0grid)):
        for j, w0 in enumerate(np.asarray(w0grid)):
            for k, wa in enumerate(np.asarray(wagrid)):
                inv_E = 1.0 / _cpl_E_numpy(z, float(Om0), float(w0), float(wa))
                r_grid[i, j, k, :] = (
                    speed_of_light / H0Planck
                ) * _cumulative_trapezoid(inv_E, z)
    return r_grid


zgrid = jnp.array(zgrid)
"""Redshift coordinates of the distance interpolation grid."""

rs = jnp.asarray(_build_cpl_distance_grid())
"""Comoving-distance table indexed by ``(Om0grid, w0grid, wagrid, zgrid)`` in Mpc."""


@jit
def E(z, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return the dimensionless CPL expansion rate ``H(z) / H0``.

    The expression assumes a spatially flat CPL cosmology with matter density
    ``Om0`` and dark-energy density ``1 - Om0``.  Inputs may be scalars or JAX
    arrays and are broadcast by JAX.
    """
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * jnp.exp(-3.0 * wa * z / one_plus_z)
    )
    return jnp.sqrt(Om0 * one_plus_z**3 + dark_energy)


@jit
def r_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return line-of-sight comoving distance at redshift ``z``.

    The distance is interpolated from the precomputed ``(Om0, w0, wa, z)`` grid
    and rescaled from the Planck reference Hubble constant to the requested
    ``H0``.  Values outside the tabulated grid are returned as ``NaN`` rather
    than extrapolated.
    """
    return interpnd(
        (Om0, w0, wa, z),
        (Om0grid, w0grid, wagrid, zgrid),
        rs,
        fill_value=jnp.nan,
    ) * (H0Planck / H0)


@jit
def dL_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return luminosity distance for redshift ``z`` in Mpc."""
    return (1 + z) * r_of_z(z, H0, Om0, w0, wa)


@jit
def dL_grid_bounds(H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return the luminosity-distance support covered by ``zgrid``.

    The pair ``(dL_min, dL_max)`` is useful for masking posterior samples before
    inverse interpolation.  The lower bound is zero because the redshift grid
    starts at ``z = 0``.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    return dL_grid[0], dL_grid[-1]


@jit
def dL_in_z_grid(dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return a boolean mask for distances supported by the redshift grid."""
    dL_min, dL_max = dL_grid_bounds(H0, Om0, w0, wa)
    return (dL >= dL_min) & (dL <= dL_max)


@jit
def z_of_dL(dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Invert luminosity distance to redshift by one-dimensional interpolation.

    Unsupported distances are returned as ``NaN`` rather than extrapolated.  The
    likelihood converts those values to zero or ``-inf`` contributions depending
    on the surrounding calculation, which prevents out-of-grid samples from
    looking artificially valid.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    z = jnp.interp(dL, dL_grid, zgrid)
    return jnp.where(in_grid, z, jnp.nan)


@jit
def dV_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return the differential comoving-volume factor per steradian.

    The returned value is ``c * r(z)^2 / (H0 * E(z))`` in Mpc^3.  Angular pixel
    areas are applied by the electromagnetic prior code when a prior density per
    HEALPix pixel is required.
    """
    return speed_of_light * r_of_z(z, H0, Om0, w0, wa) ** 2 / (H0 * E(z, Om0, w0, wa))


@jit
def ddL_of_z(z, dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return ``d dL / dz`` for the flat-CPL luminosity-distance relation.

    The derivative is used when transforming redshift densities into
    luminosity-distance densities.  ``dL`` is accepted as an argument so callers
    that already computed the luminosity distance can avoid recomputing it.
    """
    return dL / (1 + z) + speed_of_light * (1 + z) / (H0 * E(z, Om0, w0, wa))
