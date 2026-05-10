"""
likelihood.py
-------------
Hierarchical dark-siren log-likelihood.

Sentinel convention
-------------------
All log-probability floors are -jnp.inf, not finite magic numbers.
  - log_p_pop: p=0 → -inf  (changed in base.py)
  - log_prior_z: no finite guard; -inf propagates correctly through
    logsumexp and is caught by the final jnp.isfinite(ll) check.
  - Neff: guarded against NaN when log_mu=-inf (all weights -inf).

RAM note
--------
optimization_barrier MUST be applied before arrays enter any JIT closure
(i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from jax.scipy.special import logsumexp
import numpy as np

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
from darksirens.em import get_redshift_prior
from darksirens.em.completion import build_pixel_kde_cache
from darksirens.utils.utils import logdiffexp
from darksirens.inference.utils import log_sample_weight
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

from astropy.cosmology import Planck15

H0_FID          = float(Planck15.H0.value)
OM0_FID         = float(Planck15.Om0)
SURVEY_PARAMS_FID = jnp.array([-2.0, 1.0, 0.5, 0.0, 1.0, 0.5])


# ============================================================
# Core jitted likelihood
# ============================================================

@partial(
    jax.jit,
    static_argnames=["nEvents", "nsamp", "pop_model", "universe_model",
                     "sel_batch_size"],
)
def darksiren_log_likelihood(
    cosmo:          CosmoParams,
    survey:         SurveyParams,
    pop_params:     jnp.ndarray,
    gw_pe:          GWEvent,
    em_catalog_pe:  EMCatalog,
    gw_sel:         GWEvent,
    em_catalog_sel: EMCatalog,
    nEvents:        int,
    nsamp:          int,
    Ndraw:          float,
    pop_model:      str,
    universe_model: str,
    sel_batch_size: int | None = None,
) -> jnp.ndarray:
    """
    Hierarchical dark-siren log-likelihood.

    Returns log p({d_i} | cosmo, survey, pop_params).
    """
    log_p_pop        = pop_model_parser(pop_model=pop_model)
    raw_logPriorUniv = get_redshift_prior(universe_model)
    H0, Om0          = cosmo.H0, cosmo.Om0

    # No finite guard on the redshift prior.  -inf propagates correctly
    # through logsumexp and is caught by the final isfinite check.
    def log_prior_z(z, pix, catalog):
        return raw_logPriorUniv(z, pix, cosmo, survey, catalog)

    def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        return log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog, log_p_pop, log_prior_z,
        )

    # ------------------------------------------------------------------
    # Selection term
    # ------------------------------------------------------------------
    def _sel_batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        ldw = log_weight(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        ldw = jnp.where(jnp.isfinite(ldw), ldw, -jnp.inf)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        lse, lse2 = _sel_batch_lse(
            gw_sel.dL, gw_sel.m1det, gw_sel.q,
            gw_sel.chieff, gw_sel.pixels, gw_sel.prior_wt,
        )
        log_mu = lse  - jnp.log(Ndraw)
        log_s2 = lse2 - 2.0 * jnp.log(Ndraw)
    else:
        N_sel     = gw_sel.dL.shape[0]
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            lse_b, lse2_b = _sel_batch_lse(
                sl(gw_sel.dL), sl(gw_sel.m1det), sl(gw_sel.q),
                sl(gw_sel.chieff), sl(gw_sel.pixels), sl(gw_sel.prior_wt),
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(_scan_fn, None, jnp.arange(N_batches))
        log_mu = logsumexp(lse_all)  - jnp.log(Ndraw)
        log_s2 = logsumexp(lse2_all) - 2.0 * jnp.log(Ndraw)

    # Effective sample size (Farr 2019).
    # Guard: when all selection weights are -inf, log_mu = log_s2 = -inf.
    # The subtraction 2*(-inf) - (-inf) = nan without the guard.
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    Neff = jnp.where(
        jnp.isfinite(log_mu),
        jnp.exp(2.0 * log_mu - log_sigma2),
        0.0,                              # → too_sparse=True → ll=-inf below
    )

    ll  = jnp.where(Neff <= 5 * nEvents, -jnp.inf, 0.0)
    ll += -nEvents * log_mu + nEvents * (3 + nEvents) / (2.0 * Neff)

    # ------------------------------------------------------------------
    # PE term: scan over events
    # ------------------------------------------------------------------
    def _pe_event_fn(_, event_idx):
        s  = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
        ldw = log_weight(
            sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL),
            sl(gw_pe.chieff), sl(gw_pe.pixels), sl(gw_pe.prior_wt),
            em_catalog_pe,
        )
        ldw = jnp.where(jnp.isfinite(ldw), ldw, -jnp.inf)
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


# ============================================================
# Likelihood closure factory
# ============================================================

def _barrier(arr: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(jnp.asarray(arr))


def make_likelihood(opts, data: dict, pop_params_fid,
                    fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    Applies optimization_barrier to all large catalog and GW data arrays
    before they are captured in the JIT closure.
    """
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    nEvents        = data["nEvents"]
    nsamp          = data["nsamp"]
    Ndraw          = data["Ndraw"]
    apix           = data["apix"]
    pop_model      = opts.pop_model
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)

    def _to_jax(key):
        val = data.get(key)
        return jnp.asarray(val) if val is not None else jnp.array([0.0])

    def _catalog_array(full_key, subset_key):
        return _to_jax(full_key) if data.get(full_key) is not None else _to_jax(subset_key)

    dN_obs_kde = data.get("dN_obs_kde")
    pixel_to_cache_idx = data.get("pixel_to_cache_idx")
    if (
        universe_model == "dark_sirens"
        and data.get("zgals") is not None
        and data.get("pixels_pe") is not None
        and data.get("pixels_sel") is not None
        and (dN_obs_kde is None or pixel_to_cache_idx is None)
    ):
        pixels_pe_np = np.asarray(data["pixels_pe"], dtype=np.int32).ravel()
        pixels_sel_np = np.asarray(data["pixels_sel"], dtype=np.int32).ravel()
        unique_pixels = np.unique(np.concatenate([pixels_pe_np, pixels_sel_np]))
        n_pix_catalog = int(data.get("n_pix_catalog", np.asarray(data["zgals"]).shape[0]))
        dN_obs_kde, pixel_to_cache_idx = build_pixel_kde_cache(
            unique_pixels, data["zgals"], n_pix_catalog
        )
        data["dN_obs_kde"] = dN_obs_kde
        data["pixel_to_cache_idx"] = pixel_to_cache_idx

    # Catalog arrays — barrier-wrapped before closure capture.  Redshift
    # prior functions index these arrays by absolute HEALPix pixel, so use
    # full survey arrays when they are available rather than PE/selection
    # subsets.
    zgals_catalog  = _barrier(_catalog_array("zgals", "zgals_pe"))
    dzgals_catalog = _barrier(_catalog_array("dzgals", "dzgals_pe"))
    wgals_catalog  = _barrier(_catalog_array("wgals", "wgals_pe"))
    ngals_catalog  = _barrier(_catalog_array("ngals_catalog", "ngals_pe"))
    dN_obs_kde = _barrier(dN_obs_kde) if dN_obs_kde is not None else None
    pixel_to_cache_idx = (
        _barrier(jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32))
        if pixel_to_cache_idx is not None
        else None
    )
    delta_g_pix_z = _barrier(_to_jax("delta_g_pix_z"))
    sigma_kernel  = data["sigma_kernel"]

    # GW data arrays — barrier-wrapped.
    m1det_pe   = _barrier(_to_jax("m1det"))
    m2det_pe   = _barrier(_to_jax("m2det"))
    dL_pe      = _barrier(_to_jax("dL"))
    chieff_pe  = _barrier(_to_jax("chieff"))
    p_pe       = _barrier(_to_jax("p_pe"))
    pixels_pe  = _barrier(jnp.asarray(data["pixels_pe"], dtype=jnp.int32))
    q_pe       = _barrier(m2det_pe / m1det_pe)

    m1det_sel  = _barrier(_to_jax("m1detsels"))
    m2det_sel  = _barrier(_to_jax("m2detsels"))
    dL_sel     = _barrier(_to_jax("dLsels"))
    chieff_sel = _barrier(_to_jax("chieffsels"))
    p_draw     = _barrier(_to_jax("p_draw"))
    pixels_sel = _barrier(jnp.asarray(data["pixels_sel"], dtype=jnp.int32))
    q_sel      = _barrier(m2det_sel / m1det_sel)

    # Parameter space.
    _, _, pop_labels, _ = pop_model_prior_parser(pop_model)
    cosmo_labels  = ["H0", "Om0"]
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha_miss"]
    pop_params_fid_list = [float(v) for v in pop_params_fid]

    sampled_labels = []
    if not opts.fix_cosmology:  sampled_labels += cosmo_labels
    if not opts.fix_population: sampled_labels += pop_labels
    if not opts.fix_survey:     sampled_labels += survey_labels

    fixed_in_coord = {
        k: float(v) for k, v in fixed_parameter_values.items()
        if k in set(sampled_labels)
    }

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        coord = jnp.asarray(coord)

        values = {}
        offset = 0
        for label in sampled_labels:
            if label in fixed_in_coord:
                values[label] = fixed_in_coord[label]
                continue
            if offset >= coord.shape[0]:
                raise ValueError(
                    f"Too few coordinates: needed '{label}' at index {offset}, "
                    f"got {coord.shape[0]}."
                )
            values[label] = coord[offset]
            offset += 1

        if offset != coord.shape[0]:
            raise ValueError(
                f"Coordinate mismatch: consumed {offset}, got {coord.shape[0]}."
            )

        def _get(label, default):
            if label in values:                  return values[label]
            if label in fixed_parameter_values:  return fixed_parameter_values[label]
            return default

        if opts.fix_cosmology:
            H0, Om0 = _get("H0", H0_FID), _get("Om0", OM0_FID)
        else:
            H0, Om0 = values["H0"], values["Om0"]

        if opts.fix_population:
            pop_params = jnp.array([
                _get(label, pop_params_fid_list[i])
                for i, label in enumerate(pop_labels)
            ])
        else:
            pop_params = jnp.array([values[label] for label in pop_labels])

        if opts.fix_survey:
            sp = jnp.array([
                _get(label, float(SURVEY_PARAMS_FID[i]))
                for i, label in enumerate(survey_labels)
            ])
        else:
            sp = jnp.array([values[label] for label in survey_labels])

        cosmo  = CosmoParams(H0=H0, Om0=Om0)
        survey = SurveyParams(
            n0=10.0 ** sp[0], z50=sp[1], w=sp[2],
            delta=sp[3], b_miss=sp[4], alpha_miss=sp[5],
        )

        em_catalog_pe = EMCatalog(
            apix=apix, zgals=zgals_catalog, dzgals=dzgals_catalog,
            wgals=wgals_catalog, ngals=ngals_catalog,
            delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel,
            dN_obs_kde=dN_obs_kde, pixel_to_cache_idx=pixel_to_cache_idx,
        )
        em_catalog_sel = EMCatalog(
            apix=apix, zgals=zgals_catalog, dzgals=dzgals_catalog,
            wgals=wgals_catalog, ngals=ngals_catalog,
            delta_g_pix_z=delta_g_pix_z, sigma_kernel=sigma_kernel,
            dN_obs_kde=dN_obs_kde, pixel_to_cache_idx=pixel_to_cache_idx,
        )

        gw_pe = GWEvent(
            m1det=m1det_pe, m2det=m2det_pe, dL=dL_pe,
            chieff=chieff_pe, prior_wt=p_pe, pixels=pixels_pe, q=q_pe,
        )
        gw_sel = GWEvent(
            m1det=m1det_sel, m2det=m2det_sel, dL=dL_sel,
            chieff=chieff_sel, prior_wt=p_draw, pixels=pixels_sel, q=q_sel,
        )

        return darksiren_log_likelihood(
            cosmo, survey, pop_params,
            gw_pe,  em_catalog_pe,
            gw_sel, em_catalog_sel,
            nEvents, nsamp, Ndraw,
            pop_model, universe_model,
            sel_batch_size=sel_batch_size,
        )

    return likelihood