"""Shared redshift grid for redshift priors and completion models."""

import jax.numpy as jnp

# Log-spaced from z~0 to zMax, giving 1000 points.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
zMax: float = 5.0
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), 1000))
