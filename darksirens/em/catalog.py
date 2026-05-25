"""
catalog.py
----------
EM catalog redshift prior: p_cat(z | pix).

Each galaxy in the pixel is treated as a Gaussian in redshift space
(centre = photo-z, width = photo-z uncertainty).  The mixture is
weighted by:

    w_i ∝ w_gal,i * dV_c(z_i) * (1+z_i)^delta

where the volume weight reflects the galaxy number density model

    n(z) = n0 (1+z)^delta

Note: the (1+z)^delta factor here is purely the number-density
evolution of the galaxy population.  Merger rate evolution is handled
elsewhere in the pipeline and must not appear in this weight.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

@jit
def log_catalog_prior(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> float:
    r"""
    Log of the EM-catalog redshift prior at redshift z for pixel pix.

    Computes $\ln p_{\text{gal}}(z_k | \Omega_p)$ for a single discrete Gravitational 
    Wave (GW) posterior sample $z_k$ using an exact KDE overlap integration method.

    Notes
    -----
    When cross-correlating a broad continuous GW posterior with a highly precise 
    discrete galaxy catalog (e.g., DESI), evaluating discrete GW samples at the 
    exact catalog redshifts causes numerical failure because the instrumental 
    errors are nearly Dirac delta functions. 

    To fix this, we evaluate the exact overlap integral between the GW posterior 
    KDE and the discrete catalog. Due to Gaussian overlap symmetry, applying a KDE 
    bandwidth to the discrete GW samples is mathematically identical to applying 
    the KDE bandwidth to the catalog galaxies and evaluating at the discrete GW points:

    $$ \int p_{\text{GW}}(z) p_{\text{gal}}(z) \, dz = \frac{1}{K} \sum_k \left[ \sum_{i \in \Omega_p} \tilde{W}_i \, \mathcal{N}(z_k; z_i, \sigma_{\text{eff}, i}) \right] $$

    Where the effective variance is the sum in quadrature of the instrumental 
    error and the LSS smoothing kernel:
    $$ \sigma_{\text{eff}, i} = \sqrt{\sigma_{\text{cat}, i}^2 + \sigma_{\text{kde}}^2} $$

    The individual galaxy weights $W_i$ inside the pixel are calculated by scaling 
    the base completeness/luminosity weights $w_i$ by the cosmological volume element:
    $$ W_i = w_i \cdot \frac{dV_c}{dz}(z_i | H_0, \Omega_m) $$
    
    These weights are then locally normalized to sum to 1 inside the pixel:
    $$ \tilde{W}_i = \frac{W_i}{\sum_{j} W_j} $$

    Finally, the logsumexp trick is used to stably compute the log probability:
    $$ \ln p_{\text{gal}}(z_k | \Omega_p) = \text{logsumexp} \left( \ln \tilde{W}_i + \ln \mathcal{N}(z_k; z_i, \sigma_{\text{eff}, i}) \right) $$

    Parameters
    ----------
    z : float
        Redshift $z_k$ at which to evaluate the prior (a single discrete GW sample).
    pix : int
        HEALPix pixel index $\Omega_p$.
    cosmo : CosmoParams
        Cosmological parameters ($H_0, \Omega_m$) for volume weighting.
    survey : SurveyParams
        Survey parameters.
    em_catalog : EMCatalog
        EM galaxy catalog arrays containing redshifts, errors, and base weights.

    Returns
    -------
    float
        The log probability $\ln p_{\text{gal}}(z_k | \Omega_p)$.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    sigma_kde = survey.sigma_kde

    # ``pix`` is the catalog-row index for compact/sliced catalogs (the
    # caller passes the per-sample ``sample_to_unique_idx`` map) and the raw
    # HEALPix pixel for legacy full catalogs.
    zs = zgals[pix]         # z_i: (N_max_gals,)
    sig = dzgals[pix]       # \sigma_{cat, i}: (N_max_gals,) Raw instrumental errors
    w = wgals[pix]          # w_i: (N_max_gals,) Base weights

    # 1. Calculate the log of the volume weights based on the current cosmology
    log_vol_weights = jnp.log(dV_of_z(zs, H0, Om0))
    
    # 2. Add it to the base log-weights
    safe_w = jnp.where(w > 0, w, 1.0)
    log_base_w = jnp.where(w > 0, jnp.log(safe_w), -jnp.inf)
    
    log_total_w = log_base_w + log_vol_weights
    
    # 3. Normalize the new log-weights locally inside the pixel
    log_w_norm = log_total_w - logsumexp(log_total_w)
    
    # ---------------------------------------------------------
    # 4. THE KDE OVERLAP INTEGRAL
    # ---------------------------------------------------------
    # We apply a KDE bandwidth to the galaxies to represent the 
    # continuous structure of the dark matter halos/LSS. 
    # dz = 0.0025 roughly corresponds to ~10 Mpc clustering scale at low redshift.
    
    # Sum the variances in quadrature (Analytical integral of two Gaussians)
    sig_eff = jnp.sqrt(sig**2 + sigma_kde**2)
    
    # Evaluate the discrete GW sample z_k against the smoothed catalog in log-space
    return logsumexp(log_w_norm + norm.logpdf(z, zs, sig_eff))


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.
# Outer function takes z_array (N_samples,) and pix_array (N_samples,)
log_catalog_prior_vmap = jit(
    vmap(log_catalog_prior, in_axes=(0, 0, None, None, None), out_axes=0)
)
