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
from jax import vmap

from darksirens.utils.cosmology import dV_of_z, threads_distance_table
from darksirens.core.types import CosmoParams, SurveyParams

from darksirens.redshift.grid import log_interp_zgrid, zgrid


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


@threads_distance_table()
def log_volume_prior(z: float, cosmo: CosmoParams, survey: SurveyParams,
                     distance_table=None) -> float:
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
    distance_table : optional
        Comoving-distance interpolation table, threaded as a jit ARGUMENT (see
        ``utils.cosmology``).  ``None`` resolves to the caller's active table.

    Returns
    -------
    float : log p(z), normalised over zgrid.
    """
    pvol_norm = _precompute_volume_grid(cosmo)
    # NaN-guarded log: dV(z=0) == 0 exactly, and the naked log makes
    # d log_pvol[0]/dH0 = (1/0) * 0 = NaN in reverse mode for any query in the
    # first z-cell — the same defect fixed in the state path
    # (redshift/prior.py); this one-shot registry path is used by the
    # WL/cluster likelihood and checks.py. log(tiny) is numerically identical
    # downstream (exp underflows to exactly zero weight).
    #
    # The sentinel must not be LINEARLY interpolated in log space, though: node 0
    # sits at z = 0 and node 1 at z ~ 1.8e-3, so a straight line between -708 and
    # the physical ~+12 is a 300-decade ramp across the first cell (p(9e-4)
    # returned exp(-361) instead of the true density).  ``log_interp_zgrid``
    # follows dV_c/dz ∝ z^2 there instead.
    return log_interp_zgrid(
        z, jnp.log(jnp.maximum(pvol_norm, jnp.finfo(pvol_norm.dtype).tiny))
    )


@threads_distance_table()
def log_volume_prior_vmap(z, cosmo: CosmoParams, survey: SurveyParams,
                          distance_table=None):
    """Vectorised over z-samples only; cosmo and survey are broadcast constants.

    Use this instead of manually vmapping to ensure the normalisation grid is
    computed once (via CSE / explicit hoist) regardless of call site.  The
    distance table is a jit argument of this boundary and reaches
    ``log_volume_prior`` as an unmapped operand of the inner jit.
    """
    return vmap(log_volume_prior, in_axes=(0, None, None), out_axes=0)(z, cosmo, survey)