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

from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.likelihood.catalog_views import barrier, prepare_catalog_views
from darksirens.likelihood.events import pad_gw_event_to_multiple
from darksirens.likelihood.core import (
    darksiren_log_likelihood,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
)
from darksirens.core.constants import H0_FID, OM0_FID, SURVEY_PARAMS_FID
from darksirens.inference.parameters import (
    build_parameter_decoder,
    complete_empty_pixel_policy_code,
)
from darksirens.core.types import EMCatalog, GWEvent

# Backward-compatible aliases for callers/tests that imported private helpers.
_barrier = barrier
_complete_empty_pixel_policy_code = complete_empty_pixel_policy_code


def _resolve_redshift_prior_materialization(opts) -> bool:
    """Resolve whether to keep the likelihood-internal redshift-prior barrier."""
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError(
            "redshift_prior_barrier must be one of {'auto', 'on', 'off'}, "
            f"got {mode!r}."
        )
    tinyns_cfg = getattr(opts, "tinyns_resolved_config", None) or {}
    is_tinyns_jax_rwalk = (
        getattr(opts, "sampler", None) == "tinyns"
        and tinyns_cfg.get("kernel") == "jax"
        and tinyns_cfg.get("sample") == "rwalk"
    )
    # NumPyro NUTS differentiates the likelihood w.r.t. theta; the redshift
    # prior STATE depends on theta and ``lax.optimization_barrier`` has no
    # differentiation rule, so the barrier must come off for gradient-based
    # sampling (otherwise dark/bright-siren NUTS dies with
    # NotImplementedError at the first gradient).
    is_numpyro = getattr(opts, "sampler", None) == "numpyro"
    return not (is_tinyns_jax_rwalk or is_numpyro)


def _redshift_prior_materialization_reason(opts, materialize: bool) -> str:
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode in {"on", "off"}:
        return f"forced {mode}"
    if materialize:
        return "auto -> on"
    if getattr(opts, "sampler", None) == "numpyro":
        return "auto -> off for NumPyro NUTS (optimization_barrier is not differentiable)"
    return "auto -> off for TinyNS JAX rwalk"


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


def _make_mixture_likelihood(
    opts,
    data: dict,
    pop_params_fid,
    fixed_parameter_values: dict | None,
    n_catalogs: int,
):
    """Build the K-catalog mixture likelihood callable (dark_sirens, K >= 2).

    Each catalog in ``data["catalogs"]`` carries its OWN nside/apix, compact
    PE/selection views, KDE cache, and Q_LSS table; the GW posterior and
    selection *physics* arrays (masses, distances, spins, sky vectors) are shared
    across catalogs.  The per-catalog compact pixel maps are stacked into ``(N,
    K)`` matrices and the mixture weights come from ``decode_mixture``.
    """
    bundles = data.get("catalogs")
    if bundles is None or len(bundles) != n_catalogs:
        raise ValueError(
            f"n_catalogs={n_catalogs} requires data['catalogs'] with "
            f"{n_catalogs} bundles; got "
            f"{None if bundles is None else len(bundles)}."
        )

    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    universe_model = opts.universe_model
    sel_batch_size = getattr(opts, "sel_batch_size", None)
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

    def _bundle_field_inputs(bundle):
        """Per-bundle FIELD-convention normalization inputs (survey-global),
        precomputed by the loader (loaders.py) or supplied directly in tests.
        Shared by the bundle's PE and selection EMCatalogs so their global Z is
        the SAME value for the same theta (constants cancel structurally)."""
        fobs = bundle.get("field_dN_obs_s")
        if fobs is None:
            return None, None, None
        return (
            barrier(jnp.asarray(fobs)),
            jnp.asarray(bundle["field_n_empty"], dtype=jnp.float64),
            jnp.asarray(bundle["field_N_obs_total"], dtype=jnp.float64),
        )

    # Shared (catalog-independent) GW / selection physics arrays.
    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    def _compact_lss_q_for(views, unique_pixels):
        # Per-catalog analogue of make_likelihood._compact_lss_q: slice each
        # catalog's own Q_LSS table to its union pixels so only the compact block
        # reaches the device.
        full = views.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(views.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            return barrier(full_j), (1 if idx == 1 else idx)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    em_catalogs_pe = []
    em_catalogs_sel = []
    pe_pixel_cols = []
    sel_pixel_cols = []
    for bundle in bundles:
        views = prepare_catalog_views(
            opts,
            bundle,
            universe_model,
            counterpart_pixel=None,
            cache_builder=build_pixel_kde_cache,
        )
        # Each catalog uses ITS OWN pixel area (its own nside): sharing a single
        # apix across catalogs of different resolutions would silently bias the
        # per-pixel galaxy densities that enter the completion.
        apix_k = bundle["apix"]
        lss_q_pe_k, lss_idx_pe_k = _compact_lss_q_for(views, views.unique_pixels_pe)
        lss_q_sel_k, lss_idx_sel_k = _compact_lss_q_for(views, views.unique_pixels_sel)
        field_obs_k, field_ne_k, field_Nobs_k = _bundle_field_inputs(bundle)

        em_catalogs_pe.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_pe_catalog,
            dzgals=views.dzgals_pe_catalog,
            wgals=views.wgals_pe_catalog,
            ngals=views.ngals_pe_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_pe,
            pixel_to_cache_idx=views.pixel_to_cache_idx_pe,
            unique_pixels=views.unique_pixels_pe,
            sample_to_unique_idx=views.sample_to_unique_pe,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_pe_k,
            lss_completion_indexing=lss_idx_pe_k,
            field_dN_obs_s=field_obs_k,
            field_n_empty=field_ne_k,
            field_N_obs_total=field_Nobs_k,
        ))
        em_catalogs_sel.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_sel_catalog,
            dzgals=views.dzgals_sel_catalog,
            wgals=views.wgals_sel_catalog,
            ngals=views.ngals_sel_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_sel,
            pixel_to_cache_idx=views.pixel_to_cache_idx_sel,
            unique_pixels=views.unique_pixels_sel,
            sample_to_unique_idx=views.sample_to_unique_sel,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_sel_k,
            lss_completion_indexing=lss_idx_sel_k,
            field_dN_obs_s=field_obs_k,
            field_n_empty=field_ne_k,
            field_N_obs_total=field_Nobs_k,
        ))
        pe_pixel_cols.append(jnp.asarray(views.sample_to_unique_pe, dtype=jnp.int32))
        sel_pixel_cols.append(jnp.asarray(views.sample_to_unique_sel, dtype=jnp.int32))

    # Stack the per-catalog compact pixel maps into (N, K) matrices.
    pixels_pe = barrier(jnp.stack(pe_pixel_cols, axis=1))
    pixels_sel = barrier(jnp.stack(sel_pixel_cols, axis=1))

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=None,
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

    em_catalog_pe_0 = em_catalogs_pe[0]
    em_catalog_sel_0 = em_catalogs_sel[0]
    mixture_em_catalogs_pe = tuple(em_catalogs_pe[1:])
    mixture_em_catalogs_sel = tuple(em_catalogs_sel[1:])

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        (
            cosmo,
            surveys,
            pop_params,
            sky_params,
            mark_params,
            log_w,
        ) = parameter_decoder.decode_mixture(coord)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}."
            )
        return darksiren_log_likelihood(
            cosmo,
            surveys[0],
            pop_params,
            gw_pe,
            em_catalog_pe_0,
            gw_sel,
            em_catalog_sel_0,
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
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            # Forward lss_marginalize so core's K>=2 NotImplementedError guard
            # is reachable for direct make_likelihood callers too (the CLI has
            # its own _fatal guard).
            lss_marginalize=bool(getattr(opts, "lss_marginalize", False)),
            n_catalogs=n_catalogs,
            mixture_surveys=tuple(surveys[1:]),
            mixture_em_catalogs_pe=mixture_em_catalogs_pe,
            mixture_em_catalogs_sel=mixture_em_catalogs_sel,
            mixture_log_weights=log_w,
            catalog_sky_weighting=catalog_sky_weighting,
        )

    return likelihood


def make_likelihood(opts, data: dict, pop_params_fid, fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    This wrapper prepares static catalog/GW views, decodes sampler coordinates,
    and delegates the pure JIT likelihood evaluation to
    :func:`darksirens.likelihood.core.darksiren_log_likelihood`.
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
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

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
    # Selection-side WL marginalization (opt-in; lognormal backend only —
    # the core silently keeps the legacy selection path otherwise, mirroring
    # the cluster wrapper's semantics).
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if getattr(opts, "wl_selection", "standard") == "wl_lognormal"
        else WL_SELECTION_STANDARD
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

    # Multitracer: for K >= 2 the catalog-completed redshift prior becomes a
    # per-catalog mixture; delegate to the dedicated builder and leave the
    # single-catalog path below bit-identical.
    n_catalogs = int(getattr(opts, "n_catalogs", 1))
    if n_catalogs >= 2:
        return _make_mixture_likelihood(
            opts, data, pop_params_fid, fixed_parameter_values, n_catalogs
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

    # Slice the (optional) Q_LSS ENSEMBLE (M, n_pix, n_grid) to each view's union
    # pixels the same way, for the fully-Bayesian marginalisation (--lss_marginalize).
    def _compact_lss_members(unique_pixels):
        full = catalogs.lss_completion_logq_members
        if full is None:
            return None
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            return barrier(full_j)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[1]:
            raise ValueError(
                f"LSS completion ensemble has {full_j.shape[1]} pixels but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[:, up])

    lss_qm_pe = _compact_lss_members(catalogs.unique_pixels_pe)
    lss_qm_sel = _compact_lss_members(catalogs.unique_pixels_sel)
    lss_marginalize = bool(getattr(opts, "lss_marginalize", False))

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
            lss_completion_logq_members=lss_qm_pe,
            lss_completion_indexing=lss_idx_pe,
            mark_logmstar=marks_pe["mark_logmstar"],
            mark_logssfr=marks_pe["mark_logssfr"],
            mark_metallicity=marks_pe["mark_metallicity"],
            mark_color=marks_pe["mark_color"],
            field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
            field_n_empty=getattr(catalogs, "field_n_empty", None),
            field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
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
            lss_completion_logq_members=lss_qm_sel,
            lss_completion_indexing=lss_idx_sel,
            mark_logmstar=marks_sel["mark_logmstar"],
            mark_logssfr=marks_sel["mark_logssfr"],
            mark_metallicity=marks_sel["mark_metallicity"],
            mark_color=marks_sel["mark_color"],
            field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
            field_n_empty=getattr(catalogs, "field_n_empty", None),
            field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
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
                wl_selection=wl_selection,
                lss_marginalize=lss_marginalize,
                materialize_redshift_prior_state=materialize_redshift_prior_state,
                selection_neff_soft_guard=selection_neff_soft_guard,
                catalog_sky_weighting=catalog_sky_weighting,
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
            wl_selection=wl_selection,
            lss_marginalize=lss_marginalize,
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            catalog_sky_weighting=catalog_sky_weighting,
        )

    return likelihood
