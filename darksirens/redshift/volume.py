"""
volume.py
---------
Comoving volume prior: the agnostic redshift prior used in GW-only
(spectral sirens) inference.

    p(z) ∝ dV_c/dz

No galaxy number-density evolution enters here — `delta` describes
n(z) = n0 (1+z)^delta and is only relevant where galaxies are counted
(catalog.py, completion.py).  Merger rate evolution is accounted for
elsewhere in the pipeline and must not be included here.

Performance note
----------------
``_precompute_volume_grid`` separates the O(N_grid) normalisation
computation (which depends only on cosmo) from the O(1) interpolation
(which depends on z).  Inside a vmap over PE samples, JAX's CSE will
already hoist the normalisation out of the per-sample loop — but
making the split explicit:

  1. Clarifies intent in the source.
  2. Lets ``_log_prior_complete_catalog`` reuse the grid for the
     volume fallback without paying for a second normalisation.
  3. Makes the scalar ``log_volume_prior`` and the batched
     ``log_volume_prior_vmap`` share the same code path.
"""

import jax.numpy as jnp
from jax import jit, vmap

from darksirens.utils.cosmology import dV_of_z
from darksirens.core.types import CosmoParams, SurveyParams

from darksirens.em.utils import zgrid


def _precompute_volume_grid(cosmo: CosmoParams) -> jnp.ndarray:
    """
    Normalised comoving volume element on ``zgrid``.

    Depends only on ``cosmo`` — not on ``z`` or any survey parameters.
    Inside a JIT context, JAX's CSE hoists this computation once per
    cosmo proposal rather than repeating it per-sample.

    Returns
    -------
    pvol_norm : (N_grid,) — normalised so that trapezoid(pvol_norm, zgrid) = 1.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    pvol = dV_of_z(zgrid, H0, Om0, w0, wa)
    return pvol / jnp.trapezoid(pvol, zgrid)


@jit
def log_volume_prior(z: float, cosmo: CosmoParams, survey: SurveyParams) -> float:
    """
    Log of the normalised comoving-volume redshift prior evaluated at z.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate the prior.
    cosmo : CosmoParams
        Cosmological parameters (H0, Om0, w0, wa).
    survey : SurveyParams
        Accepted for API uniformity; not used here.

    Returns
    -------
    float : log p(z), normalised over zgrid.
    """
    pvol_norm = _precompute_volume_grid(cosmo)
    return jnp.interp(z, zgrid, jnp.log(pvol_norm))


#: Vectorised over z-samples only; cosmo and survey are broadcast constants.
#: Use this instead of manually vmapping to ensure the normalisation grid
#: is computed once (via CSE / explicit hoist) regardless of call site.
log_volume_prior_vmap = jit(
    vmap(log_volume_prior, in_axes=(0, None, None), out_axes=0)
)