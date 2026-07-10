"""Catalog compaction and cache setup helpers for dark-siren inference."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import jax.numpy as jnp
from jax import lax
import numpy as np

from darksirens.redshift.completion import (
    build_field_normalization_inputs,
    build_pixel_kde_cache,
)

DARK_SIREN_CACHE_MODELS = {"dark_sirens"}


def barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """Apply the pre-JIT optimization barrier used by likelihood closures."""
    return lax.optimization_barrier(jnp.asarray(arr))


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
    pe_view = _ensure_compact(data, "pe", "pixels_pe")
    sel_view = _ensure_compact(data, "sel", "pixels_sel")

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
        barrier(jnp.asarray(unique_pixels_sel_raw, dtype=jnp.int32))
        if unique_pixels_sel_raw is not None
        else None
    )
    sample_to_unique_pe_raw = (
        pe_view.sample_to_unique if pe_view is not None else data.get("pixels_pe")
    )
    sample_to_unique_sel_raw = (
        sel_view.sample_to_unique if sel_view is not None else data.get("pixels_sel")
    )
    sample_to_unique_pe = barrier(jnp.asarray(sample_to_unique_pe_raw, dtype=jnp.int32))
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
    if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
        if data.get("field_dN_obs_s") is not None:
            # Caller-provided (already survey-global) field inputs -- symmetric with
            # the caller-provided compact-view path; used by tests and any caller
            # that precomputes them.
            field_dN_obs_s = barrier(jnp.asarray(data["field_dN_obs_s"]))
            field_n_empty = jnp.asarray(data["field_n_empty"], dtype=jnp.float64)
            field_N_obs_total = jnp.asarray(data["field_N_obs_total"], dtype=jnp.float64)
        elif all(v is not None for v in (full_z, full_w, full_n)):
            fobs, n_empty, N_obs_total = build_field_normalization_inputs(
                full_z, full_w, full_n
            )
            field_dN_obs_s = barrier(fobs)
            field_n_empty = jnp.asarray(n_empty, dtype=jnp.float64)
            field_N_obs_total = jnp.asarray(N_obs_total, dtype=jnp.float64)
        # else: leave None -> the likelihood raises a clear scope error (field
        # mode needs the full-sky catalog rows to count empty pixels).

    # Carry the (global) Q table HOST-side, unbarriered: likelihood.py slices it
    # to the per-view union pixels so only the compact block reaches the device.
    lss_completion_logq = data.get("lss_completion_logq")
    lss_completion_logq_members = data.get("lss_completion_logq_members")
    lss_completion_indexing = int(data.get("lss_completion_indexing", 0))

    dN_obs_kde_pe = dN_obs_kde_sel = None
    pixel_to_cache_idx_pe = pixel_to_cache_idx_sel = None
    cache_required = universe_model in DARK_SIREN_CACHE_MODELS

    if cache_required:
        if union_unique_pixels is not None:
            dN_obs_kde_pe, pixel_to_cache_idx_pe = cache_builder(
                unique_pixels=union_unique_pixels,
                zgals=full_z,
                n_pix_catalog=int(
                    data.get("n_pix_catalog", np.asarray(full_z).shape[0])
                ),
                wgals=full_w,
                ngals=full_n,
            )
            dN_obs_kde_sel = dN_obs_kde_pe
            pixel_to_cache_idx_sel = pixel_to_cache_idx_pe
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
                dN_obs_kde_pe, pixel_to_cache_idx_pe = cache_builder(
                    unique_pixels=np.arange(n_pe_rows, dtype=np.int32),
                    zgals=pe_view.zgals,
                    n_pix_catalog=n_pe_rows,
                    wgals=pe_view.wgals,
                    ngals=pe_view.ngals,
                )
                dN_obs_kde_sel, pixel_to_cache_idx_sel = cache_builder(
                    unique_pixels=np.arange(n_sel_rows, dtype=np.int32),
                    zgals=sel_view.zgals,
                    n_pix_catalog=n_sel_rows,
                    wgals=sel_view.wgals,
                    ngals=sel_view.ngals,
                )
                # The KDE rows above are per *compact row*, but the completion
                # code looks the cache up by GLOBAL HEALPix pixel whenever the
                # catalog carries ``unique_pixels`` (it translates row ->
                # global before indexing).  A row-sized lookup indexed by a
                # global pixel id silently clamps out of bounds under JIT and
                # returns the wrong pixel's KDE — so rebuild the lookup in the
                # global key space here.  Without ``unique_pixels`` the rows
                # themselves are the keys and the identity lookup is correct.
                pixel_to_cache_idx_pe = _global_cache_lookup(
                    pe_view.unique_pixels, n_pe_rows, pixel_to_cache_idx_pe
                )
                pixel_to_cache_idx_sel = _global_cache_lookup(
                    sel_view.unique_pixels, n_sel_rows, pixel_to_cache_idx_sel
                )

    dN_obs_kde_pe = barrier(dN_obs_kde_pe) if dN_obs_kde_pe is not None else None
    dN_obs_kde_sel = barrier(dN_obs_kde_sel) if dN_obs_kde_sel is not None else None
    pixel_to_cache_idx_pe = (
        barrier(jnp.asarray(pixel_to_cache_idx_pe, dtype=jnp.int32))
        if pixel_to_cache_idx_pe is not None
        else None
    )
    pixel_to_cache_idx_sel = (
        barrier(jnp.asarray(pixel_to_cache_idx_sel, dtype=jnp.int32))
        if pixel_to_cache_idx_sel is not None
        else None
    )

    return CatalogViews(
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
    )
