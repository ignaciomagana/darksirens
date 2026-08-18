"""Catalog compaction and cache setup helpers for dark-siren inference."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import jax.numpy as jnp
from jax import lax
import numpy as np

from darksirens.redshift.completion import (
    _LOGQ_CLIP,
    build_field_delta_g_inputs,
    build_field_depth_inputs,
    build_field_lss_q_fp_empty_sum,
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
    build_field_mark_inputs,
    build_field_normalization_inputs,
    build_pixel_kde_cache,
)

DARK_SIREN_CACHE_MODELS = {"dark_sirens"}


def barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """Apply the pre-JIT optimization barrier used by likelihood closures."""
    return lax.optimization_barrier(jnp.asarray(arr))


def field_depth_inputs_required(opts, z_depth_attr) -> bool:
    """Whether this catalog needs the flat FULL-SKY field DEPTH inputs.

    True iff FIELD sky weighting is active AND a survey depth will be in force for
    this catalog, in which case the survey-global normalizer's observed term must
    be the depth-scaled count ``Sum_pix N_obs,pix * m_pix(theta)`` and needs the
    per-galaxy kernel geometry (:func:`build_field_depth_inputs`).  The depth is in
    force iff the ``--survey_z_depth`` CLI override or the per-catalog file
    attribute is set -- the two inputs of ``cli.inference.resolve_survey_z_depth``,
    whose result is non-None iff one of them is.  Checked here (rather than
    building unconditionally) because the inputs are ``(N_gal_total,)`` f64 flat
    arrays that only the depth convention reads.
    """
    if getattr(opts, "catalog_sky_weighting", "conditional") != "field":
        return False
    return (
        getattr(opts, "survey_z_depth", None) is not None
        or z_depth_attr is not None
    )


def _global_cache_lookup(unique_pixels, n_rows: int, identity_lookup):
    """Return a cache lookup indexed by global HEALPix pixel.

    ``identity_lookup`` is the row-indexed lookup returned by the cache
    builder for ``unique_pixels=arange(n_rows)``.  When the compact view
    carries global pixel ids, the completion code indexes the lookup by
    global pixel, so it must be re-keyed; otherwise rows are the keys and
    the identity lookup is already correct.
    """
    if unique_pixels is None:
        return identity_lookup
    globals_np = np.asarray(unique_pixels, dtype=np.int64).reshape(-1)
    lookup = np.zeros(int(globals_np.max()) + 1, dtype=np.int32)
    lookup[globals_np] = np.arange(n_rows, dtype=np.int32)
    return lookup


@dataclass(frozen=True)
class CompactCatalogView:
    """Caller-independent compact catalog arrays for one sample view."""

    unique_pixels: np.ndarray | None
    sample_to_unique: np.ndarray | None
    zgals: np.ndarray
    dzgals: np.ndarray | None
    wgals: np.ndarray | None
    ngals: np.ndarray | None


@dataclass(frozen=True)
class CatalogViews:
    """Barrier-wrapped PE/selection catalog views captured by the closure."""

    zgals_pe_catalog: jnp.ndarray
    dzgals_pe_catalog: jnp.ndarray
    wgals_pe_catalog: jnp.ndarray
    ngals_pe_catalog: jnp.ndarray | None
    zgals_sel_catalog: jnp.ndarray
    dzgals_sel_catalog: jnp.ndarray
    wgals_sel_catalog: jnp.ndarray
    ngals_sel_catalog: jnp.ndarray | None
    unique_pixels_pe: jnp.ndarray | None
    unique_pixels_sel: jnp.ndarray | None
    sample_to_unique_pe: jnp.ndarray
    sample_to_unique_sel: jnp.ndarray
    delta_g_pix_z: jnp.ndarray
    dN_obs_kde_pe: jnp.ndarray | None
    dN_obs_kde_sel: jnp.ndarray | None
    pixel_to_cache_idx_pe: jnp.ndarray | None
    pixel_to_cache_idx_sel: jnp.ndarray | None
    # Optional deterministic LSS-conditioned lognormal completion table Q_LSS,
    # shared across PE/selection (the per-row resolution to compact rows happens
    # in completion.py via each view's unique_pixels).  ``None`` = legacy path.
    lss_completion_logq: jnp.ndarray | None = None
    lss_completion_logq_members: jnp.ndarray | None = None  # (M, n_pix|n_rows, n_grid)
    lss_completion_indexing: int = 0  # int enum: 0=auto, 1=compact, 2=global
    # FIELD-convention sky-weighting normalization inputs (built from the FULL-sky
    # catalog when ``opts.catalog_sky_weighting == "field"``; ``None`` otherwise).
    field_dN_obs_s: jnp.ndarray | None = None  # (n_occupied, n_grid) f32
    field_n_empty: jnp.ndarray | None = None   # scalar
    field_N_obs_total: jnp.ndarray | None = None  # scalar
    # FIELD-normalizer budget modulations (survey-global rows aligned with
    # field_dN_obs_s via field_occupied_pixels): Q_LSS rows / empty-pixel Q sum
    # and per-pixel delta_g rows, so the global Z carries the SAME
    # missing-galaxy budget as the numerator.
    field_occupied_pixels: jnp.ndarray | None = None  # (n_occupied,) int32
    field_lss_q: jnp.ndarray | None = None            # (n_occupied, n_grid) f32
    field_lss_q_empty_sum: jnp.ndarray | None = None  # (n_grid,) f64
    field_delta_g: jnp.ndarray | None = None          # (n_occupied, n_grid) f32
    field_lss_q_members: jnp.ndarray | None = None            # (M, n_occupied, n_grid) f32
    field_lss_q_empty_sum_members: jnp.ndarray | None = None  # (M, n_grid) f64
    # Marked-host field normalizer: flat FULL-SKY per-galaxy inputs.
    field_mark_z: jnp.ndarray | None = None           # (N_gal_total,) f32
    field_mark_w: jnp.ndarray | None = None           # (N_gal_total,) f32
    field_mark_values: jnp.ndarray | None = None      # (N_gal_total, n_marks) f32
    # DEPTH-truncated field normalizer: flat FULL-SKY per-galaxy kernel geometry,
    # built ONLY when a survey depth is active (build_field_depth_inputs).
    field_depth_z: jnp.ndarray | None = None          # (N_gal_total,) f64
    field_depth_dz: jnp.ndarray | None = None         # (N_gal_total,) f64
    field_depth_c: jnp.ndarray | None = None          # (N_gal_total,) f64
    # Stratified selection (SurveyParams.selection_strata): full-sky pixel ->
    # stratum lookup + per-stratum empty-pixel budgets (see EMCatalog docs).
    pixel_stratum_map: jnp.ndarray | None = None        # (n_pix_total,) int32
    empty_stratum_counts: jnp.ndarray | None = None     # (S,) f64
    field_lss_q_empty_sum_strata: jnp.ndarray | None = None  # (S, n_grid) f64
    field_lss_q_empty_sum_strata_members: jnp.ndarray | None = None  # (M, S, n_grid) f64
    # Per-pixel selection fraction f_p (field-level PR-2): per-view row
    # gathers of the full-sky map plus the footprint splits the field
    # normalizer consumes.  None when --per_pixel_completeness is off.
    f_p_rows_pe: jnp.ndarray | None = None       # (n_pe_rows,) f32
    f_p_rows_sel: jnp.ndarray | None = None      # (n_sel_rows,) f32
    field_f_p_occ: jnp.ndarray | None = None     # (n_occupied,) f32
    field_f_p_empty_sum: jnp.ndarray | None = None  # scalar f64
    # f_p x Q pairing (S-3): the f_p-weighted empty-pixel Q budget
    # ``Sum_{p empty} f_p Q_p(z)``.  Built only when a GLOBAL Q table and an
    # f_p map are both present; without it the field normalizer would model
    # unobserved sky as Cbar-complete (completion.build_field_lss_q_fp_empty_sum).
    field_lss_q_fp_empty_sum: jnp.ndarray | None = None  # (n_grid,) f64


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


def unique_inference_pixels(pixels_pe, pixels_sel, required_pixels=None) -> np.ndarray:
    """Return the sorted union of unique PE and selection HEALPix pixels."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    parts = [unique_pe, unique_sel]
    if required_pixels is not None:
        parts.append(np.asarray(required_pixels, dtype=np.int32).reshape(-1))
    return np.unique(np.concatenate(parts)).astype(np.int32, copy=False)


def _full_catalog_arrays(data: dict):
    full_z = (
        data.get("zgals_catalog")
        if data.get("zgals_catalog") is not None
        else data.get("zgals")
    )
    full_dz = (
        data.get("dzgals_catalog")
        if data.get("dzgals_catalog") is not None
        else data.get("dzgals")
    )
    full_w = (
        data.get("wgals_catalog")
        if data.get("wgals_catalog") is not None
        else data.get("wgals")
    )
    full_n = data.get("ngals_catalog")
    return full_z, full_dz, full_w, full_n


def _compact_view_from_data(data: dict, prefix: str) -> CompactCatalogView | None:
    """Return a caller-provided compact view, if present."""
    zgals = data.get(f"zgals_{prefix}")
    if zgals is None:
        return None

    return CompactCatalogView(
        unique_pixels=data.get(f"unique_pixels_{prefix}"),
        sample_to_unique=data.get(f"sample_to_unique_{prefix}"),
        zgals=zgals,
        dzgals=data.get(f"dzgals_{prefix}"),
        wgals=data.get(f"wgals_{prefix}"),
        ngals=data.get("ngals_pe" if prefix == "pe" else "ngals_sel"),
    )


def _ensure_compact(
    data: dict, prefix: str, pixels_key: str
) -> CompactCatalogView | None:
    """Return a compact view without mutating the caller-owned data dictionary."""
    caller_view = _compact_view_from_data(data, prefix)
    if caller_view is not None:
        return caller_view

    full_z, full_dz, full_w, full_n = _full_catalog_arrays(data)
    pixels = data.get(pixels_key)
    if any(value is None for value in (full_z, full_dz, full_w, full_n, pixels)):
        return None

    unique_pixels, sample_to_unique_idx = np.unique(
        np.asarray(pixels, dtype=np.int32), return_inverse=True
    )
    unique_pixels = unique_pixels.astype(np.int32, copy=False)
    return CompactCatalogView(
        unique_pixels=unique_pixels,
        sample_to_unique=sample_to_unique_idx.astype(np.int32, copy=False),
        zgals=full_z[unique_pixels],
        dzgals=full_dz[unique_pixels],
        wgals=full_w[unique_pixels],
        ngals=full_n[unique_pixels],
    )


def prepare_catalog_views(
    opts,
    data: dict,
    universe_model: str,
    counterpart_pixel: int | None,
    cache_builder=build_pixel_kde_cache,
) -> CatalogViews:
    """Compact catalogs, build sample-to-unique maps, and prebuild KDE caches."""
    # The LINEAR Q tables are not plumbed to the NUMERATOR: ``CatalogViews``
    # carries only the log forms and both factories compact only those, so a
    # linear-only data dict would give a Q-modulated field normalizer (the
    # budget rows below accept either form) with an UNMODULATED numerator --
    # precisely the numerator/denominator budget mismatch field_global_log_Z's
    # mutual-exclusion guard exists to prevent, and invisible to it.  Reject at
    # this single seam rather than letting the two disagree.
    for _linear_key in ("lss_completion_q", "lss_completion_q_members"):
        if data.get(_linear_key) is not None:
            raise ValueError(
                f"data['{_linear_key}'] is not supported by the likelihood "
                "factory: supply the LOG table instead "
                "('lss_completion_logq' / 'lss_completion_logq_members', which "
                "is what every builder emits). The linear table reaches only "
                "the field normalizer's missing-galaxy budget, never the "
                "per-pixel numerator, so the two seams would carry different "
                "budgets."
            )
    pe_view = _ensure_compact(data, "pe", "pixels_pe")
    sel_view = _ensure_compact(data, "sel", "pixels_sel")

    # Caller-provided union views: a multitracer bundle (loaders.py) compacts PE
    # and selection ONCE over their pixel union, so both views reference the SAME
    # galaxy table and unique-pixel array (only the per-view sample->row maps
    # differ).  Detect it by identity so the barriered device arrays are aliased
    # and a SINGLE KDE cache is built per bundle -- the compact-view analogue of
    # the flat single-catalog ``union_unique_pixels`` path below.
    caller_union_views = (
        pe_view is not None
        and sel_view is not None
        and pe_view.zgals is sel_view.zgals
        and pe_view.unique_pixels is sel_view.unique_pixels
    )

    full_z, full_dz, full_w, full_n = _full_catalog_arrays(data)

    def _full_or_default(full_value):
        return jnp.asarray(full_value) if full_value is not None else jnp.array([0.0])

    zgals_pe_catalog = barrier(
        jnp.asarray(pe_view.zgals) if pe_view is not None else _full_or_default(full_z)
    )
    dzgals_pe_catalog = barrier(
        _full_or_default(pe_view.dzgals if pe_view is not None else full_dz)
    )
    wgals_pe_catalog = barrier(
        _full_or_default(pe_view.wgals if pe_view is not None else full_w)
    )
    ngals_pe_raw = pe_view.ngals if pe_view is not None else full_n
    ngals_pe_catalog = (
        barrier(jnp.asarray(ngals_pe_raw, dtype=jnp.int32))
        if ngals_pe_raw is not None
        else None
    )

    if caller_union_views:
        # PE and selection share one galaxy table: alias the already-barriered
        # PE device arrays so the two EMCatalogs carry IS-identical leaves (one
        # device buffer per array, not two copies of the same values).
        zgals_sel_catalog = zgals_pe_catalog
        dzgals_sel_catalog = dzgals_pe_catalog
        wgals_sel_catalog = wgals_pe_catalog
        ngals_sel_catalog = ngals_pe_catalog
    else:
        zgals_sel_catalog = barrier(
            jnp.asarray(sel_view.zgals)
            if sel_view is not None
            else _full_or_default(full_z)
        )
        dzgals_sel_catalog = barrier(
            _full_or_default(sel_view.dzgals if sel_view is not None else full_dz)
        )
        wgals_sel_catalog = barrier(
            _full_or_default(sel_view.wgals if sel_view is not None else full_w)
        )
        ngals_sel_raw = sel_view.ngals if sel_view is not None else full_n
        ngals_sel_catalog = (
            barrier(jnp.asarray(ngals_sel_raw, dtype=jnp.int32))
            if ngals_sel_raw is not None
            else None
        )

    unique_pixels_pe_raw = pe_view.unique_pixels if pe_view is not None else None
    unique_pixels_sel_raw = sel_view.unique_pixels if sel_view is not None else None
    unique_pixels_pe = (
        barrier(jnp.asarray(unique_pixels_pe_raw, dtype=jnp.int32))
        if unique_pixels_pe_raw is not None
        else None
    )
    unique_pixels_sel = (
        unique_pixels_pe
        if caller_union_views
        else (
            barrier(jnp.asarray(unique_pixels_sel_raw, dtype=jnp.int32))
            if unique_pixels_sel_raw is not None
            else None
        )
    )
    sample_to_unique_pe_raw = (
        pe_view.sample_to_unique if pe_view is not None else data.get("pixels_pe")
    )
    sample_to_unique_sel_raw = (
        sel_view.sample_to_unique if sel_view is not None else data.get("pixels_sel")
    )
    # The PE side is absent entirely on the flow-surrogate path (no stored PE
    # samples): keep it None rather than crashing on jnp.asarray(None).
    sample_to_unique_pe = (
        barrier(jnp.asarray(sample_to_unique_pe_raw, dtype=jnp.int32))
        if sample_to_unique_pe_raw is not None
        else None
    )
    sample_to_unique_sel = barrier(
        jnp.asarray(sample_to_unique_sel_raw, dtype=jnp.int32)
    )

    union_unique_pixels = None
    if all(
        value is not None
        for value in (
            full_z,
            full_dz,
            full_w,
            full_n,
            data.get("pixels_pe"),
            data.get("pixels_sel"),
        )
    ):
        counterpart_pixels = data.get("counterpart_pixels")
        required_pixels = (
            counterpart_pixels
            if universe_model == "bright_sirens" and counterpart_pixels is not None
            else (
                [counterpart_pixel]
                if universe_model == "bright_sirens" and counterpart_pixel is not None
                else None
            )
        )
        union_unique_pixels = unique_inference_pixels(
            data["pixels_pe"], data["pixels_sel"], required_pixels=required_pixels
        )
        pe_global_pixels = np.asarray(data["pixels_pe"], dtype=np.int32)
        sel_global_pixels = np.asarray(data["pixels_sel"], dtype=np.int32)
        sample_to_union_pe_raw = np.searchsorted(
            union_unique_pixels, pe_global_pixels
        ).astype(np.int32, copy=False)
        sample_to_union_sel_raw = np.searchsorted(
            union_unique_pixels, sel_global_pixels
        ).astype(np.int32, copy=False)

        zgals_union_catalog = barrier(jnp.asarray(full_z[union_unique_pixels]))
        dzgals_union_catalog = barrier(jnp.asarray(full_dz[union_unique_pixels]))
        wgals_union_catalog = barrier(jnp.asarray(full_w[union_unique_pixels]))
        ngals_union_catalog = barrier(
            jnp.asarray(full_n[union_unique_pixels], dtype=jnp.int32)
        )
        unique_pixels_union = barrier(jnp.asarray(union_unique_pixels, dtype=jnp.int32))
        sample_to_unique_pe = barrier(
            jnp.asarray(sample_to_union_pe_raw, dtype=jnp.int32)
        )
        sample_to_unique_sel = barrier(
            jnp.asarray(sample_to_union_sel_raw, dtype=jnp.int32)
        )

        zgals_pe_catalog = zgals_sel_catalog = zgals_union_catalog
        dzgals_pe_catalog = dzgals_sel_catalog = dzgals_union_catalog
        wgals_pe_catalog = wgals_sel_catalog = wgals_union_catalog
        ngals_pe_catalog = ngals_sel_catalog = ngals_union_catalog
        unique_pixels_pe = unique_pixels_sel = unique_pixels_union

    delta_g_pix_z = barrier(_to_jax(data, "delta_g_pix_z"))

    # FIELD-convention sky weighting: build the survey-GLOBAL normalization inputs
    # from the FULL-sky catalog (occupied-pixel smoothed densities, empty-pixel
    # count, total observed count) BEFORE compaction/drop_full_catalog.  Only for
    # ``--catalog_sky_weighting field``; the conditional path leaves these None.
    field_dN_obs_s = field_n_empty = field_N_obs_total = None
    field_occupied_pixels = field_lss_q = field_lss_q_empty_sum = None
    field_delta_g = None
    field_lss_q_members = field_lss_q_empty_sum_members = None
    field_mark_z = field_mark_w = field_mark_values = None
    field_depth_z = field_depth_dz = field_depth_c = None
    pixel_stratum_map = empty_stratum_counts = None
    field_lss_q_empty_sum_strata = field_lss_q_empty_sum_strata_members = None
    if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
        occupied_np = None
        if data.get("field_dN_obs_s") is not None:
            # Caller-provided (already survey-global) field inputs -- symmetric with
            # the caller-provided compact-view path; used by tests and any caller
            # that precomputes them.
            field_dN_obs_s = barrier(jnp.asarray(data["field_dN_obs_s"]))
            field_n_empty = jnp.asarray(data["field_n_empty"], dtype=jnp.float64)
            field_N_obs_total = jnp.asarray(data["field_N_obs_total"], dtype=jnp.float64)
            if data.get("field_occupied_pixels") is not None:
                occupied_np = np.asarray(
                    data["field_occupied_pixels"], dtype=np.int32
                )
            # Depth inputs travel with them (the multitracer bundles are already
            # compacted here, so they cannot be rebuilt from full-sky rows).
            if data.get("field_depth_z") is not None:
                field_depth_z = barrier(
                    jnp.asarray(data["field_depth_z"], dtype=jnp.float64)
                )
                field_depth_dz = barrier(
                    jnp.asarray(data["field_depth_dz"], dtype=jnp.float64)
                )
                field_depth_c = barrier(
                    jnp.asarray(data["field_depth_c"], dtype=jnp.float64)
                )
        elif all(v is not None for v in (full_z, full_w, full_n)):
            field = build_field_normalization_inputs(full_z, full_w, full_n)
            field_dN_obs_s = barrier(field.dN_obs_s)
            field_n_empty = jnp.asarray(field.n_empty, dtype=jnp.float64)
            field_N_obs_total = jnp.asarray(field.N_obs_total, dtype=jnp.float64)
            occupied_np = np.asarray(field.occupied_pixels, dtype=np.int32)
            # DEPTH-consistent global observed term: with a survey depth the raw
            # field_N_obs_total counts every above-depth catalogued galaxy twice
            # (observed AND missing), so the normalizer needs the per-galaxy
            # below-depth kernel geometry.  Built from the FULL-sky rows, before
            # compaction / --drop_full_catalog, exactly like the mark inputs.
            if field_depth_inputs_required(opts, data.get("z_depth")) \
                    and full_dz is not None:
                depth_inputs = build_field_depth_inputs(
                    full_z, full_dz, full_w, full_n
                )
                field_depth_z = barrier(depth_inputs.z)
                field_depth_dz = barrier(depth_inputs.dz)
                field_depth_c = barrier(depth_inputs.c)
        # else: leave None -> the likelihood raises a clear scope error (field
        # mode needs the full-sky catalog rows to count empty pixels).

        # Budget-modulation rows: mirror the numerator's Q table / per-pixel
        # delta_g into the survey-global normalizer (field_global_log_Z), so
        # both carry the SAME missing-galaxy budget.  Requires the GLOBAL Q
        # table (compact per-view tables cannot supply the empty-pixel budget).
        if occupied_np is not None:
            n_pix_total = int(
                data.get(
                    "n_pix_catalog",
                    np.asarray(full_z).shape[0] if full_z is not None else 0,
                )
            )
            # Caller-supplied budget rows are used AS-IS: the multitracer
            # loader already built them from this catalog's FULL-sky rows
            # (loaders.load_multitracer_catalog_bundles), so rebuilding them
            # here repeats a float64 ``np.exp`` over the whole
            # (M, n_pix, N_grid) ensemble per bundle -- and the memory model
            # (block_sizing.estimate_pending_static_bytes) counts the members
            # build only for non-bundle sources.
            have_q_rows = (
                data.get("field_lss_q") is not None
                and data.get("field_lss_q_empty_sum") is not None
            )
            have_qm_rows = (
                data.get("field_lss_q_members") is not None
                and data.get("field_lss_q_empty_sum_members") is not None
            )
            # Only the LOG form reaches here (the linear keys are rejected at the
            # top of this function, since they never reach the numerator).
            logq_full = data.get("lss_completion_logq")
            if logq_full is not None:
                logq_np = np.asarray(logq_full)
                if have_q_rows:
                    field_lss_q = barrier(jnp.asarray(data["field_lss_q"]))
                    field_lss_q_empty_sum = barrier(
                        jnp.asarray(data["field_lss_q_empty_sum"])
                    )
                else:
                    q_occ, q_empty_sum = build_field_lss_q_inputs(
                        logq_np, occupied_np, n_pix_total
                    )
                    field_lss_q = barrier(q_occ)
                    field_lss_q_empty_sum = barrier(q_empty_sum)
            logq_members_full = data.get("lss_completion_logq_members")
            if logq_members_full is not None:
                if have_qm_rows:
                    field_lss_q_members = barrier(
                        jnp.asarray(data["field_lss_q_members"])
                    )
                    field_lss_q_empty_sum_members = barrier(
                        jnp.asarray(data["field_lss_q_empty_sum_members"])
                    )
                else:
                    qm_occ, qm_empty = build_field_lss_q_member_inputs(
                        np.asarray(logq_members_full), occupied_np, n_pix_total
                    )
                    field_lss_q_members = barrier(qm_occ)
                    field_lss_q_empty_sum_members = barrier(qm_empty)
            dg_full = data.get("delta_g_pix_z")
            if dg_full is not None and np.asarray(dg_full).ndim == 2 \
                    and np.asarray(dg_full).shape[0] > 1:
                field_delta_g = barrier(
                    build_field_delta_g_inputs(dg_full, occupied_np)
                )
            field_occupied_pixels = barrier(
                jnp.asarray(occupied_np, dtype=jnp.int32)
            )

            # Stratified selection: full-sky stratum map (attached to the data
            # bundle by the CLI from --stratum_map) + the per-stratum
            # decomposition of the empty-pixel budget the stratified global
            # normalizer needs (completion._field_missing_curve).
            strat_full = data.get("pixel_stratum_map")
            if strat_full is not None:
                strat_np = np.asarray(strat_full, dtype=np.int32)
                if strat_np.shape[0] != n_pix_total:
                    raise ValueError(
                        f"pixel_stratum_map covers {strat_np.shape[0]} pixels "
                        f"but the catalog sky has {n_pix_total}; the map must "
                        "be full-sky at the survey nside."
                    )
                if strat_np.min() < 0:
                    raise ValueError(
                        "pixel_stratum_map contains negative labels; the map "
                        "builder must assign every pixel a stratum in [0, S) "
                        "(off-footprint pixels get the documented policy "
                        "label, e.g. the majority stratum)."
                    )
                n_strata = int(strat_np.max()) + 1
                empty_mask = np.ones(n_pix_total, dtype=bool)
                empty_mask[occupied_np] = False
                empty_stratum_counts = barrier(jnp.asarray(
                    np.bincount(strat_np[empty_mask], minlength=n_strata)
                    .astype(np.float64)))
                pixel_stratum_map = barrier(jnp.asarray(strat_np))
                if field_lss_q_empty_sum is not None:
                    # Per-stratum twin of the empty-pixel Q budget, from the
                    # same global (log)Q table.
                    # SAME clip as the numerator (_to_q) and the whole-sky Q
                    # rows: the per-stratum empty budget must carry the same
                    # missing-galaxy total, not the raw table's tail.
                    q_lin = np.exp(np.clip(logq_np, -_LOGQ_CLIP, _LOGQ_CLIP))
                    sums = np.zeros((n_strata, q_lin.shape[1]))
                    for s in range(n_strata):
                        rows = empty_mask & (strat_np == s)
                        sums[s] = q_lin[rows].sum(axis=0)
                    field_lss_q_empty_sum_strata = barrier(jnp.asarray(sums))
                if field_lss_q_empty_sum_members is not None:
                    # PER-MEMBER twin: the stratified branch of
                    # completion._field_missing_curve reads ONLY the strata
                    # budget, so without this every ensemble member's global
                    # normalizer would carry the DETERMINISTIC empty-pixel
                    # budget while its numerator carries member m's Q --
                    # marginalizing over inconsistent estimands.  Streamed
                    # member by member over the EMPTY rows only, so the full
                    # (M, n_pix, n_grid) linear ensemble is never materialized.
                    logqm_np = np.asarray(logq_members_full)
                    strat_empty = strat_np[empty_mask]
                    rows_by_s = [strat_empty == s for s in range(n_strata)]
                    sums_m = np.zeros(
                        (logqm_np.shape[0], n_strata, logqm_np.shape[2])
                    )
                    for m in range(logqm_np.shape[0]):
                        q_lin_m = np.exp(
                            np.clip(logqm_np[m][empty_mask], -700.0, 700.0)
                        )
                        for s in range(n_strata):
                            sums_m[m, s] = q_lin_m[rows_by_s[s]].sum(axis=0)
                    field_lss_q_empty_sum_strata_members = barrier(
                        jnp.asarray(sums_m)
                    )

        # Marked-host field normalizer: flatten the FULL-SKY z-centred marks so
        # mu_miss and the observed marked mass are view-independent (PE and
        # selection share ONE global Z).  Needs the full arrays (pre-compaction).
        mark_model = getattr(opts, "mark_model", "none")
        mark_names = tuple(getattr(opts, "mark_names", ()) or ())
        if (mark_model not in (None, "none") and mark_names
                and full_z is not None):
            from darksirens.marks import MARK_FIELDS

            mark_tables = {
                name: data.get(MARK_FIELDS[name]) for name in mark_names
            }
            if all(v is not None for v in mark_tables.values()):
                fz, fw, fvals = build_field_mark_inputs(
                    full_z, full_w, full_n, mark_tables, mark_names
                )
                field_mark_z = barrier(fz)
                field_mark_w = barrier(fw)
                field_mark_values = barrier(fvals)

    # Carry the (global) Q table HOST-side, unbarriered: likelihood.py slices it
    # to the per-view union pixels so only the compact block reaches the device.
    lss_completion_logq = data.get("lss_completion_logq")
    lss_completion_logq_members = data.get("lss_completion_logq_members")
    lss_completion_indexing = int(data.get("lss_completion_indexing", 0))

    dN_obs_kde_pe = dN_obs_kde_sel = None
    # The KDE cache is always built row-for-row with the catalog's unique_pixels,
    # so the completion code indexes ``dN_obs_kde`` DIRECTLY by row (see
    # completion._row_C): the dense ``pixel_to_cache_idx`` lookup (12*nside^2 on
    # the flat union path) is a pure identity round-trip and is no longer built.
    # The EMCatalog field is retained (None) for pytree/treedef stability.
    pixel_to_cache_idx_pe = pixel_to_cache_idx_sel = None
    cache_required = universe_model in DARK_SIREN_CACHE_MODELS

    if cache_required:
        if union_unique_pixels is not None:
            dN_obs_kde_pe, _ = cache_builder(
                unique_pixels=union_unique_pixels,
                zgals=full_z,
                n_pix_catalog=int(
                    data.get("n_pix_catalog", np.asarray(full_z).shape[0])
                ),
                wgals=full_w,
                ngals=full_n,
            )
            dN_obs_kde_sel = dN_obs_kde_pe
        elif caller_union_views:
            # Bundle union (loaders.py): PE and selection share one compact
            # galaxy table, so build ONE cache over the union rows and alias it to
            # both views -- the compact-view analogue of the flat
            # ``union_unique_pixels`` branch above (one cache per bundle).
            n_union_rows = int(np.asarray(pe_view.zgals).shape[0])
            dN_obs_kde_pe, _ = cache_builder(
                unique_pixels=np.arange(n_union_rows, dtype=np.int32),
                zgals=pe_view.zgals,
                n_pix_catalog=n_union_rows,
                wgals=pe_view.wgals,
                ngals=pe_view.ngals,
            )
            dN_obs_kde_sel = dN_obs_kde_pe
        else:
            pe_mask = (
                None
                if pe_view is None
                else (pe_view.wgals if pe_view.wgals is not None else pe_view.ngals)
            )
            sel_mask = (
                None
                if sel_view is None
                else (sel_view.wgals if sel_view.wgals is not None else sel_view.ngals)
            )
            missing_cache_inputs = [
                name
                for name, value in (
                    (
                        "PE compact galaxy redshifts",
                        None if pe_view is None else pe_view.zgals,
                    ),
                    (
                        "selection compact galaxy redshifts",
                        None if sel_view is None else sel_view.zgals,
                    ),
                    (
                        "PE sample-to-unique map",
                        None if pe_view is None else pe_view.sample_to_unique,
                    ),
                    (
                        "selection sample-to-unique map",
                        None if sel_view is None else sel_view.sample_to_unique,
                    ),
                    ("PE galaxy mask (wgals or ngals)", pe_mask),
                    ("selection galaxy mask (wgals or ngals)", sel_mask),
                )
                if value is None
            ]
            if missing_cache_inputs:
                message = (
                    "Dark-siren inference requires the per-pixel KDE cache; "
                    f"cannot build it because these inputs are missing: {', '.join(missing_cache_inputs)}."
                )
                if getattr(opts, "allow_uncached_dark_sirens", False):
                    warnings.warn(
                        message
                        + " Falling back to uncached completion for tests/backward compatibility.",
                        RuntimeWarning,
                    )
                else:
                    raise RuntimeError(message)
            else:
                n_pe_rows = int(np.asarray(pe_view.zgals).shape[0])
                n_sel_rows = int(np.asarray(sel_view.zgals).shape[0])
                # Row-for-row caches (row k == compact row k); the completion code
                # indexes them by row, so no global re-key lookup is built.
                dN_obs_kde_pe, _ = cache_builder(
                    unique_pixels=np.arange(n_pe_rows, dtype=np.int32),
                    zgals=pe_view.zgals,
                    n_pix_catalog=n_pe_rows,
                    wgals=pe_view.wgals,
                    ngals=pe_view.ngals,
                )
                dN_obs_kde_sel, _ = cache_builder(
                    unique_pixels=np.arange(n_sel_rows, dtype=np.int32),
                    zgals=sel_view.zgals,
                    n_pix_catalog=n_sel_rows,
                    wgals=sel_view.wgals,
                    ngals=sel_view.ngals,
                )

    # When PE and selection share ONE cache (flat union or bundle union), barrier
    # it ONCE and alias, so the two EMCatalogs carry IS-identical cache leaves
    # (not two separately-barriered copies of the same values) -- required for the
    # redshift-prior-state sharing in likelihood/core.py and one fewer device
    # buffer.  ``is`` identity of the pre-barrier objects is the shared signal.
    shared_cache = (
        dN_obs_kde_pe is not None and dN_obs_kde_pe is dN_obs_kde_sel
    )
    dN_obs_kde_pe = barrier(dN_obs_kde_pe) if dN_obs_kde_pe is not None else None
    dN_obs_kde_sel = (
        dN_obs_kde_pe
        if shared_cache
        else (barrier(dN_obs_kde_sel) if dN_obs_kde_sel is not None else None)
    )
    # pixel_to_cache_idx is no longer built (see completion._row_C): the fields
    # stay None so the EMCatalog treedef is unchanged.

    # Per-pixel selection fraction f_p (field-level PR-2): row-align the
    # full-sky map with each view (compact views gather by unique_pixels;
    # legacy full catalogs use the map itself), and split the footprint sums
    # for the field normalizer.  ``None`` everywhere when the flag is off.
    f_p_rows_pe = f_p_rows_sel = None
    field_f_p_occ = field_f_p_empty_sum = None
    field_lss_q_fp_empty_sum = None
    f_p_full = data.get("f_p_map")
    if f_p_full is not None:
        fp_np = np.asarray(f_p_full, dtype=np.float32)

        def _fp_rows(up_raw):
            if up_raw is None:
                return barrier(jnp.asarray(fp_np))
            return barrier(jnp.asarray(
                fp_np[np.asarray(up_raw, dtype=np.int64)]))

        f_p_rows_pe = _fp_rows(unique_pixels_pe_raw)
        f_p_rows_sel = (
            f_p_rows_pe
            if unique_pixels_sel_raw is unique_pixels_pe_raw
            else _fp_rows(unique_pixels_sel_raw))
        if field_occupied_pixels is not None:
            occ_idx = np.asarray(field_occupied_pixels, dtype=np.int64)
            field_f_p_occ = barrier(jnp.asarray(fp_np[occ_idx]))
            field_f_p_empty_sum = jnp.asarray(
                float(fp_np.sum()) - float(fp_np[occ_idx].sum()),
                dtype=jnp.float64)
            # f_p x Q (S-3): the empty-pixel budget the Q path needs before it
            # may carry f_p at all.  A Q ENSEMBLE is refused rather than paired
            # with the deterministic budget -- see _field_missing_curve.
            logq_for_fp = data.get("lss_completion_logq")
            if logq_for_fp is not None:
                field_lss_q_fp_empty_sum = barrier(
                    build_field_lss_q_fp_empty_sum(
                        np.asarray(logq_for_fp), occ_idx, n_pix_total, fp_np))

    return CatalogViews(
        f_p_rows_pe=f_p_rows_pe,
        f_p_rows_sel=f_p_rows_sel,
        field_f_p_occ=field_f_p_occ,
        field_f_p_empty_sum=field_f_p_empty_sum,
        field_lss_q_fp_empty_sum=field_lss_q_fp_empty_sum,
        zgals_pe_catalog=zgals_pe_catalog,
        dzgals_pe_catalog=dzgals_pe_catalog,
        wgals_pe_catalog=wgals_pe_catalog,
        ngals_pe_catalog=ngals_pe_catalog,
        zgals_sel_catalog=zgals_sel_catalog,
        dzgals_sel_catalog=dzgals_sel_catalog,
        wgals_sel_catalog=wgals_sel_catalog,
        ngals_sel_catalog=ngals_sel_catalog,
        unique_pixels_pe=unique_pixels_pe,
        unique_pixels_sel=unique_pixels_sel,
        sample_to_unique_pe=sample_to_unique_pe,
        sample_to_unique_sel=sample_to_unique_sel,
        delta_g_pix_z=delta_g_pix_z,
        dN_obs_kde_pe=dN_obs_kde_pe,
        dN_obs_kde_sel=dN_obs_kde_sel,
        pixel_to_cache_idx_pe=pixel_to_cache_idx_pe,
        pixel_to_cache_idx_sel=pixel_to_cache_idx_sel,
        lss_completion_logq=lss_completion_logq,
        lss_completion_logq_members=lss_completion_logq_members,
        lss_completion_indexing=lss_completion_indexing,
        field_dN_obs_s=field_dN_obs_s,
        field_n_empty=field_n_empty,
        field_N_obs_total=field_N_obs_total,
        field_occupied_pixels=field_occupied_pixels,
        field_lss_q=field_lss_q,
        field_lss_q_empty_sum=field_lss_q_empty_sum,
        field_delta_g=field_delta_g,
        field_lss_q_members=field_lss_q_members,
        field_lss_q_empty_sum_members=field_lss_q_empty_sum_members,
        field_mark_z=field_mark_z,
        field_mark_w=field_mark_w,
        field_mark_values=field_mark_values,
        field_depth_z=field_depth_z,
        field_depth_dz=field_depth_dz,
        field_depth_c=field_depth_c,
        pixel_stratum_map=pixel_stratum_map,
        empty_stratum_counts=empty_stratum_counts,
        field_lss_q_empty_sum_strata=field_lss_q_empty_sum_strata,
        field_lss_q_empty_sum_strata_members=field_lss_q_empty_sum_strata_members,
    )
