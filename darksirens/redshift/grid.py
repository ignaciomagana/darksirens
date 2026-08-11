"""Shared redshift grid for redshift priors and completion models."""

import math
import os

import jax
import jax.numpy as jnp

# Enable x64 BEFORE building the module-level ``zgrid`` below.  ``darksirens.
# redshift`` imports this module FIRST (before ``cosmology``, which self-enables
# x64), and the package root ``__init__`` is deliberately side-effect-free, so a
# cold ``import darksirens.redshift`` outside the CLI would otherwise freeze
# ``zgrid`` to float32 for the process lifetime -- silently degrading every
# downstream redshift/volume/completion computation (the import-order trap
# documented in tests/conftest.py).  Mirrors ``cosmology.py``.
jax.config.update("jax_enable_x64", True)

# Log-spaced from z~0 to zMax, giving 1000 points at the default zMax=5.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
# DARKSIRENS_ZMAX overrides the cap (read once at import; the cosmology
# module reads the same variable so the two grids stay consistent); the node
# count scales with the log range to preserve low-z density.
zMax: float = float(os.environ.get("DARKSIRENS_ZMAX", 5.0))
_ZGRID_NODES = max(1000, int(round(1000 * math.log(zMax + 1.0) / math.log(6.0))))
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), _ZGRID_NODES))
# The cosmology module tabulates ITS grid with numpy's expm1/log while the
# grid above uses jnp's (XLA libm): at some DARKSIRENS_ZMAX values the two
# endpoints differ by one ulp (e.g. 0.52, 0.65) and this grid's last node
# lands ABOVE the distance table, which the out-of-range mask turns into a
# NaN r(zMax) that poisons every consumer of chi(zgrid). Pin the endpoint to
# the numpy value whenever it is lower (numpy specifically -- math's libm
# differs from numpy's at exactly the pathological values); bit-inert at the
# default zMax=5 and at every zMax where the two agree.
import numpy as _np

zgrid = zgrid.at[-1].set(
    jnp.minimum(zgrid[-1], float(_np.expm1(_np.log(zMax + 1.0)))))


def log_interp_zgrid(z, log_grid):
    """Interpolate a LOG-valued grid on ``zgrid``, with the z -> 0 power law.

    ``zgrid[0]`` is exactly 0, where ``dV_c/dz`` vanishes, so node 0 of any log
    grid built from the volume element is a FLOOR SENTINEL (``log(tiny)`` =
    -708, ``log(1e-300)`` = -690.8) rather than a value.  Linear interpolation of
    the LOG array across the first cell then ramps ~300 decades from that
    sentinel up to the physical value at ``zgrid[1]``: at z = 9e-4 it
    underestimates the density by e^-339, i.e. deletes it (a nearby galaxy's host
    weight, or a PE sample's volume prior).  Since ``dV_c/dz ∝ z^2`` as z -> 0,
    take the power law off node 1 below the first node -- exact to O(z) there,
    and the ``maximum(z, tiny)`` keeps the value and its gradient finite at
    z = 0, which is what the sentinel was introduced for.
    """
    z1 = zgrid[1]
    below = log_grid[1] + 2.0 * jnp.log(
        jnp.maximum(z, jnp.finfo(zgrid.dtype).tiny) / z1)
    return jnp.where(z < z1, below, jnp.interp(z, zgrid, log_grid))
