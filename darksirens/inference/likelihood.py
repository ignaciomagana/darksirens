"""
likelihood.py
-------------
Hierarchical dark-siren likelihood factory.

Sentinel convention
-------------------
All log-probability floors are -jnp.inf, not finite magic numbers.

RAM note
--------
optimization_barrier MUST be applied before arrays enter any JIT closure
(i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.
"""

from __future__ import annotations

import jax.numpy as jnp

from darksirens.em.completion import build_pixel_kde_cache
from darksirens.inference.catalog_views import barrier, prepare_catalog_views
from darksirens.inference.events import pad_gw_event_to_multiple
from darksirens.inference.likelihood_core import darksiren_log_likelihood
from darksirens.inference.parameters import (
    H0_FID,
    OM0_FID,
    SURVEY_PARAMS_FID,
    build_parameter_decoder,
    complete_empty_pixel_policy_code,
)
from darksirens.utils.containers import EMCatalog, GWEvent

# Backward-compatible aliases for callers/tests that imported private helpers.
_barrier = barrier
_complete_empty_pixel_policy_code = complete_empty_pixel_policy_code


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


def make_likelihood(opts, data: dict, pop_params_fid, fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    This wrapper prepares static catalog/GW views, decodes sampler coordinates,
    and delegates the pure JIT likelihood evaluation to
    :func:`darksirens.inference.likelihood_core.darksiren_log_likelihood`.
    """
    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    apix = data["apix"]
    pop_model = opts.pop_model
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)
    counterpart_pixel = data.get("counterpart_pixel")
    counterpart_pixels = (
        barrier(jnp.asarray(data["counterpart_pixels"], dtype=jnp.int32))
        if data.get("counterpart_pixels") is not None else None
    )
    counterpart_zs = (
        barrier(jnp.asarray(data["counterpart_zs"], dtype=float))
        if data.get("counterpart_zs") is not None else None
    )
    counterpart_dzs = (
        barrier(jnp.asarray(data["counterpart_dzs"], dtype=float))
        if data.get("counterpart_dzs") is not None else None
    )
    bright_siren_sky_marginalized = bool(
        data.get(
            "bright_siren_sky_marginalized",
            getattr(opts, "bright_siren_sky_marginalized", False),
        )
    )

    catalogs = prepare_catalog_views(
        opts,
        data,
        universe_model,
        counterpart_pixel,
        cache_builder=build_pixel_kde_cache,
    )

    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    pixels_pe = catalogs.sample_to_unique_pe
    q_pe = barrier(m2det_pe / m1det_pe)

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    pixels_sel = catalogs.sample_to_unique_sel
    q_sel = barrier(m2det_sel / m1det_sel)

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
    )

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        cosmo, survey, pop_params = parameter_decoder.decode(coord)

        em_catalog_pe = EMCatalog(
            apix=apix,
            zgals=catalogs.zgals_pe_catalog,
            dzgals=catalogs.dzgals_pe_catalog,
            wgals=catalogs.wgals_pe_catalog,
            ngals=catalogs.ngals_pe_catalog,
            delta_g_pix_z=catalogs.delta_g_pix_z,
            sigma_kernel=catalogs.sigma_kernel,
            dN_obs_kde=catalogs.dN_obs_kde_pe,
            pixel_to_cache_idx=catalogs.pixel_to_cache_idx_pe,
            unique_pixels=catalogs.unique_pixels_pe,
            sample_to_unique_idx=catalogs.sample_to_unique_pe,
            counterpart_pixel=counterpart_pixel,
            counterpart_pixels=counterpart_pixels,
            counterpart_zs=counterpart_zs,
            counterpart_dzs=counterpart_dzs,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        )
        em_catalog_sel = EMCatalog(
            apix=apix,
            zgals=catalogs.zgals_sel_catalog,
            dzgals=catalogs.dzgals_sel_catalog,
            wgals=catalogs.wgals_sel_catalog,
            ngals=catalogs.ngals_sel_catalog,
            delta_g_pix_z=catalogs.delta_g_pix_z,
            sigma_kernel=catalogs.sigma_kernel,
            dN_obs_kde=catalogs.dN_obs_kde_sel,
            pixel_to_cache_idx=catalogs.pixel_to_cache_idx_sel,
            unique_pixels=catalogs.unique_pixels_sel,
            sample_to_unique_idx=catalogs.sample_to_unique_sel,
            counterpart_pixel=counterpart_pixel,
            counterpart_pixels=counterpart_pixels,
            counterpart_zs=counterpart_zs,
            counterpart_dzs=counterpart_dzs,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        )

        gw_pe = GWEvent(
            m1det=m1det_pe,
            m2det=m2det_pe,
            dL=dL_pe,
            chieff=chieff_pe,
            prior_wt=p_pe,
            pixels=pixels_pe,
            q=q_pe,
            valid=jnp.ones_like(dL_pe, dtype=bool),
        )
        gw_sel = GWEvent(
            m1det=m1det_sel,
            m2det=m2det_sel,
            dL=dL_sel,
            chieff=chieff_sel,
            prior_wt=p_draw,
            pixels=pixels_sel,
            q=q_sel,
            valid=jnp.ones_like(dL_sel, dtype=bool),
        )
        if sel_batch_size is not None:
            gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

        return darksiren_log_likelihood(
            cosmo,
            survey,
            pop_params,
            gw_pe,
            em_catalog_pe,
            gw_sel,
            em_catalog_sel,
            nEvents,
            nsamp,
            Ndraw,
            pop_model,
            universe_model,
            sel_batch_size=sel_batch_size,
        )

    return likelihood
