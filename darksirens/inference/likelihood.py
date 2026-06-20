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
from darksirens.inference.likelihood_core import (
    darksiren_log_likelihood,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
)
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
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())

    # Weak-lensing magnification backend (resolved up front, before the heavy
    # catalog-view prep, so a missing WL config fails fast).  All values are
    # inert when wl_backend == WL_BACKEND_DISABLED, preserving behaviour for
    # every non-WL universe model.
    wl_backend = WL_BACKEND_DISABLED
    wl_a = jnp.asarray(0.0)
    wl_b = jnp.asarray(0.0)
    wl_z_grid = jnp.asarray([0.0, 1.0])
    wl_log_mu_grid = jnp.asarray([0.0, 1.0])
    wl_log_p_table = jnp.asarray([[0.0, 0.0], [0.0, 0.0]])
    wl_params = data.get("wl_params")
    if universe_model == "spectral_sirens_wl":
        if wl_params is None:
            raise ValueError(
                "universe_model='spectral_sirens_wl' requires data['wl_params'] "
                "to be present."
            )
        backend = int(wl_params.backend)
        if backend == WL_BACKEND_LOGNORMAL:
            wl_backend = WL_BACKEND_LOGNORMAL
            wl_a = jnp.asarray(wl_params.a)
            wl_b = jnp.asarray(wl_params.b)
        elif backend == WL_BACKEND_TABULATED:
            wl_backend = WL_BACKEND_TABULATED
            wl_z_grid = jnp.asarray(wl_params.z_grid)
            wl_log_mu_grid = jnp.asarray(wl_params.log_mu_grid)
            wl_log_p_table = jnp.asarray(wl_params.log_p_table)
        else:
            raise ValueError(
                "Unsupported weak-lensing backend in data['wl_params']: "
                f"{backend}. Expected {WL_BACKEND_LOGNORMAL} (LOGNORMAL) or "
                f"{WL_BACKEND_TABULATED} (TABULATED)."
            )

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

    # Slice the (global, host-side) Q_LSS table to each view's union pixels, so
    # only the compact (n_union, n_grid) block becomes a device/jit operand
    # rather than the full (n_pix, n_grid) table.  Returns (compact_logq, indexing).
    def _compact_lss_q(unique_pixels):
        full = catalogs.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            # already compact, or a legacy full catalog (rows are global pixels)
            return barrier(full_j), (1 if idx == 1 else idx)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    lss_q_pe, lss_idx_pe = _compact_lss_q(catalogs.unique_pixels_pe)
    lss_q_sel, lss_idx_sel = _compact_lss_q(catalogs.unique_pixels_sel)

    # Per-galaxy marks: gathered to the compact catalog rows using the SAME
    # unique-pixel map that compacts zgals, so they align row-for-row.  None
    # (mark absent) flows through to the legacy galaxy-count host model.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS

    def _compact_marks(unique_pixels):
        out = {}
        for field in _MARK_FIELDS.values():
            full = data.get(field)
            if full is None:
                out[field] = None
            else:
                full = jnp.asarray(full)
                arr = full if unique_pixels is None else full[jnp.asarray(unique_pixels)]
                out[field] = barrier(arr)
        return out

    marks_pe = _compact_marks(catalogs.unique_pixels_pe)
    marks_sel = _compact_marks(catalogs.unique_pixels_sel)

    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    pixels_pe = catalogs.sample_to_unique_pe
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    pixels_sel = catalogs.sample_to_unique_sel
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=data.get("wl_params"),
    )

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        cosmo, survey, pop_params, sky_params, mark_params = parameter_decoder.decode(coord)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}. "
                "Verify parameter-space construction for this population model."
            )

        em_catalog_pe = EMCatalog(
            apix=apix,
            zgals=catalogs.zgals_pe_catalog,
            dzgals=catalogs.dzgals_pe_catalog,
            wgals=catalogs.wgals_pe_catalog,
            ngals=catalogs.ngals_pe_catalog,
            delta_g_pix_z=catalogs.delta_g_pix_z,
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
            lss_completion_logq=lss_q_pe,
            lss_completion_indexing=lss_idx_pe,
            mark_logmstar=marks_pe["mark_logmstar"],
            mark_logssfr=marks_pe["mark_logssfr"],
            mark_metallicity=marks_pe["mark_metallicity"],
            mark_color=marks_pe["mark_color"],
        )
        em_catalog_sel = EMCatalog(
            apix=apix,
            zgals=catalogs.zgals_sel_catalog,
            dzgals=catalogs.dzgals_sel_catalog,
            wgals=catalogs.wgals_sel_catalog,
            ngals=catalogs.ngals_sel_catalog,
            delta_g_pix_z=catalogs.delta_g_pix_z,
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
            lss_completion_logq=lss_q_sel,
            lss_completion_indexing=lss_idx_sel,
            mark_logmstar=marks_sel["mark_logmstar"],
            mark_logssfr=marks_sel["mark_logssfr"],
            mark_metallicity=marks_sel["mark_metallicity"],
            mark_color=marks_sel["mark_color"],
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
            nx=nx_pe,
            ny=ny_pe,
            nz=nz_pe,
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
            nx=nx_sel,
            ny=ny_sel,
            nz=nz_sel,
        )
        if sel_batch_size is not None:
            gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

        if shared_beta and shared_spin and shared_gamma:
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
                sky_model=sky_model,
                sky_params=sky_params,
                mark_model=mark_model,
                mark_params=mark_params,
                mark_names=mark_names,
                wl_backend=wl_backend,
                wl_a=wl_a,
                wl_b=wl_b,
                wl_z_grid=wl_z_grid,
                wl_log_mu_grid=wl_log_mu_grid,
                wl_log_p_table=wl_log_p_table,
            )
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
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            sky_model=sky_model,
            sky_params=sky_params,
            mark_model=mark_model,
            mark_params=mark_params,
            mark_names=mark_names,
            wl_backend=wl_backend,
            wl_a=wl_a,
            wl_b=wl_b,
            wl_z_grid=wl_z_grid,
            wl_log_mu_grid=wl_log_mu_grid,
            wl_log_p_table=wl_log_p_table,
        )

    return likelihood
