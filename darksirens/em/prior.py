"""
prior.py
--------
Redshift prior assembly for dark-siren and spectral-siren cosmological
inference with gravitational waves.

Physical picture
~~~~~~~~~~~~~~~~
We want p(z | pix, Θ) — the probability that a GW source at sky
position ``pix`` has redshift z, given cosmological parameters Θ.

Four regimes are supported:

1. ``"spectral_sirens"``
   GW data only.  Prior is the comoving volume element dV_c/dz.

2. ``"bright_sirens"``
   Counterpart-informed inference using a synthetic one-object catalog.
   By default, the counterpart redshift prior is finite only for samples in
   the global HEALPix counterpart pixel and is ``-inf`` elsewhere.  If the
   catalog explicitly requests sky marginalization, the same redshift prior is
   applied independent of sample sky pixel.

3. ``"dark_sirens_complete"``
   EM catalog assumed 100 % complete.

        p(z | pix) = p_cat(z | pix)

   Empty pixels are handled by an explicit policy.  The formal complete-catalog
   default is ``zero``: a pixel with no real catalog galaxies has zero host
   probability and returns ``-inf``.  The optional ``volume`` policy preserves
   the historical volume-prior fallback as a robustness approximation for
   sparse pixelations.

4. ``"dark_sirens"``  (default)
   Incomplete catalog; prior is a mixture weighted by C_eff(z):

        p(z | pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                   + (1 - C_eff(z|pix)) * p_miss(z|pix)

   Empty-pixel handling: C_eff → 0 automatically when the pixel has
   no observed galaxies, routing the prior entirely to p_miss.  No
   explicit fallback needed here.

Performance: single-vmap fusion in ``_log_prior_dark_sirens``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The previous implementation called ``catalog_completion_vmap`` and
``log_catalog_prior_vmap`` as two separate vmaps over the same (z, pix)
pairs.  The new implementation fuses them into a single inner function
that is vmapped once:

  1. ``_precompute_grids`` runs once (pixel-independent grids).
  2. A single vmap over (z, pix) evaluates completion + catalog prior
     in one sweep, halving the vmap invocation overhead.
  3. The per-pixel KDE lookup (cached) happens once per (z, pix) pair
     instead of once per vmap call.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .volume import log_volume_prior, log_volume_prior_vmap, _precompute_volume_grid
from .catalog import log_catalog_prior
from .completion import _precompute_grids, _catalog_completion_inner

from .utils import zgrid


COMPLETE_EMPTY_PIXEL_POLICY_ZERO = 0
COMPLETE_EMPTY_PIXEL_POLICY_VOLUME = 1


# ------------------------------------------------------------
# Individual prior implementations
# ------------------------------------------------------------

@jit
def _log_prior_spectral_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    GW-only prior: normalised comoving volume element.

    ``pix`` and ``em_catalog`` are accepted for API uniformity; ignored.
    """
    return log_volume_prior_vmap(z, cosmo, survey)


@jit
def _log_prior_complete_catalog(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior under the complete-catalog assumption.

        p(z | pix) = p_cat(z | pix)

    Empty-pixel policy
    ------------------
    The strict complete-catalog interpretation is controlled by
    ``survey.complete_empty_pixel_policy == 0`` (``zero``): if the requested
    catalog row has no real galaxies, the prior returns ``-inf`` because no
    host exists in that pixel.

    ``survey.complete_empty_pixel_policy == 1`` (``volume``) keeps the previous
    volume-prior fallback only for genuinely empty pixels.  This is a
    robustness approximation for sparse pixelations, not the formal
    complete-catalog model.

    Empty pixels are identified from ``em_catalog.ngals`` (or, for legacy
    catalogs lacking it, a positive-weight real-galaxy mask), not from
    ``isfinite(log_p_cat)``.  Therefore numerical underflow in a non-empty
    pixel does not silently switch to the volume prior.
    """
    from .catalog import log_catalog_prior_vmap  # local import avoids circular
    log_p_cat = log_catalog_prior_vmap(z, pix, cosmo, survey, em_catalog)

    if em_catalog.ngals is not None:
        row_has_galaxies = jnp.take(em_catalog.ngals, pix) > 0
    else:
        row_has_galaxies = jnp.any(jnp.take(em_catalog.wgals, pix, axis=0) > 0.0, axis=-1)

    # Precompute volume grid once (shared across all samples via CSE).
    pvol_norm  = _precompute_volume_grid(cosmo)
    log_p_vol  = vmap(lambda z_i: jnp.interp(z_i, zgrid, jnp.log(pvol_norm)))(z)

    empty_value = jnp.where(
        survey.complete_empty_pixel_policy == COMPLETE_EMPTY_PIXEL_POLICY_VOLUME,
        log_p_vol,
        -jnp.inf,
    )
    return jnp.where(row_has_galaxies, log_p_cat, empty_value)


@jit
def _log_prior_bright_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Bright-siren counterpart redshift likelihood with an optional sky gate.

    Multi-event bright-siren analyses store one counterpart redshift and one
    counterpart sky pixel per GW event.  The likelihood sets
    ``em_catalog.active_counterpart_index`` before evaluating each event, so the
    same vectorised prior can use the event-specific counterpart while retaining
    the compact catalog pixel mapping used by dark-siren analyses.

    For backward compatibility, if the per-event counterpart arrays are absent
    this falls back to the historical one-object synthetic catalog prior.
    """
    from jax.scipy.stats import norm
    from .catalog import log_catalog_prior_vmap  # local import avoids circular

    if em_catalog.counterpart_zs is not None:
        idx = jnp.asarray(em_catalog.active_counterpart_index, dtype=jnp.int32)
        counterpart_z = jnp.take(em_catalog.counterpart_zs, idx)
        counterpart_dz = jnp.take(em_catalog.counterpart_dzs, idx)
        counterpart_pixel = jnp.take(em_catalog.counterpart_pixels, idx)

        if em_catalog.unique_pixels is None:
            global_pix = pix
        else:
            global_pix = jnp.take(em_catalog.unique_pixels, pix)

        sky_marginalized = jnp.asarray(em_catalog.bright_siren_sky_marginalized)
        in_counterpart_pixel = global_pix == counterpart_pixel
        log_p_cp = norm.logpdf(z, counterpart_z, counterpart_dz)
        return jnp.where(sky_marginalized | in_counterpart_pixel, log_p_cp, -jnp.inf)

    counterpart_pixel = em_catalog.counterpart_pixel
    if counterpart_pixel is None:
        if em_catalog.ngals is None:
            counterpart_pixel = 0
        elif em_catalog.unique_pixels is None:
            counterpart_pixel = jnp.argmax(em_catalog.ngals > 0)
        else:
            counterpart_pixel = em_catalog.unique_pixels[jnp.argmax(em_catalog.ngals > 0)]

    if em_catalog.unique_pixels is None:
        global_pix = pix
        counterpart_row = counterpart_pixel
    else:
        global_pix = jnp.take(em_catalog.unique_pixels, pix)
        counterpart_row = jnp.argmax(em_catalog.unique_pixels == counterpart_pixel)

    sky_marginalized = jnp.asarray(em_catalog.bright_siren_sky_marginalized)
    prior_pix = jnp.where(sky_marginalized, counterpart_row, pix)
    log_p_cp = log_catalog_prior_vmap(z, prior_pix, cosmo, survey, em_catalog)

    in_counterpart_pixel = global_pix == counterpart_pixel
    return jnp.where(sky_marginalized | in_counterpart_pixel, log_p_cp, -jnp.inf)


@jit
def _log_prior_dark_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior with catalog completion (the general case).

    Mixes catalog and missing-galaxy densities using C_eff(z):

        p(z|pix) ∝ C_eff(z|pix) * p_cat(z|pix)
                 + (1 - C_eff(z|pix)) * p_miss(z|pix)

    Single-vmap implementation
    --------------------------
    ``_precompute_grids`` (pixel-independent) runs once.  A single inner
    function computes both the completion term and the catalog prior for
    each (z_i, pix_i) pair.  This function is vmapped once, replacing
    the previous two-vmap implementation (one for completion, one for
    catalog prior).

    The per-pixel KDE lookup uses ``em_catalog.dN_obs_kde`` (precomputed
    at startup by ``build_pixel_kde_cache``) — an O(1) index rather than
    an O(N_grid × N_max_gals) recomputation.
    """
    # Pixel-independent grids — hoisted out of the vmap by JAX CSE.
    grids = _precompute_grids(cosmo, survey, em_catalog)

    def _inner(z_i, pix_i):
        # --- Completion ---
        _, p_miss, C_z = _catalog_completion_inner(z_i, pix_i, grids, survey, em_catalog)

        # --- Catalog prior ---
        log_p_cat = log_catalog_prior(z_i, pix_i, cosmo, survey, em_catalog)

        # --- Numerically safe log-space mixture ---
        log_C      = jnp.where(C_z   >  0.0, jnp.log(C_z),       -jnp.inf)
        log_1mC    = jnp.where(C_z   <  1.0, jnp.log1p(-C_z),    -jnp.inf)
        log_p_miss = jnp.where(p_miss > 0.0, jnp.log(p_miss),    -jnp.inf)
        log_p_cat  = jnp.nan_to_num(log_p_cat, neginf=-jnp.inf)
        log_p_miss = jnp.nan_to_num(log_p_miss, neginf=-jnp.inf)

        return logsumexp(jnp.stack([
            log_C   + log_p_cat,
            log_1mC + log_p_miss,
        ]))

    return vmap(_inner)(z, pix)


# ------------------------------------------------------------
# Registry and factory
# ------------------------------------------------------------

#: Maps model name → compiled prior function.
#: Signature: f(z, pix, cosmo, survey, em_catalog) → log_prior (array).
PRIOR_REGISTRY: dict = {
    "spectral_sirens":      _log_prior_spectral_sirens,
    "bright_sirens":        _log_prior_bright_sirens,
    "dark_sirens_complete": _log_prior_complete_catalog,
    "dark_sirens":          _log_prior_dark_sirens,
}


def get_redshift_prior(model: str):
    """
    Return the compiled log-prior function for the requested model.

    Parameters
    ----------
    model : str
        One of ``"spectral_sirens"``, ``"bright_sirens"``,
        ``"dark_sirens_complete"``, or ``"dark_sirens"``.

    Returns
    -------
    callable with signature::

        log_prior(z, pix, cosmo, survey, em_catalog) -> jnp.ndarray

    Raises
    ------
    ValueError if model is not in ``PRIOR_REGISTRY``.
    """
    if model not in PRIOR_REGISTRY:
        available = ", ".join(f'"{k}"' for k in PRIOR_REGISTRY)
        raise ValueError(
            f"Unknown redshift prior model '{model}'. "
            f"Available: {available}."
        )
    return PRIOR_REGISTRY[model]