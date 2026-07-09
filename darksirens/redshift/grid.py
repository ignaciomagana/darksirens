"""Shared redshift grid for redshift priors and completion models."""

import math
import os

import jax.numpy as jnp

# Log-spaced from z~0 to zMax, giving 1000 points at the default zMax=5.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
# DARKSIRENS_ZMAX overrides the cap (read once at import; the cosmology
# module reads the same variable so the two grids stay consistent); the node
# count scales with the log range to preserve low-z density.
zMax: float = float(os.environ.get("DARKSIRENS_ZMAX", 5.0))
_ZGRID_NODES = max(1000, int(round(1000 * math.log(zMax + 1.0) / math.log(6.0))))
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), _ZGRID_NODES))
