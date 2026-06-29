"""Compact catalog views used by inference data loading."""

import numpy as np


def _compact_catalog_for_pixels(pixels, zgals, dzgals, wgals, ngals, required_pixels=None):
    """Return compact catalog rows and sample→row lookup for pixels.

    ``required_pixels`` are included in the compact catalog even if no sample
    falls in them.  The sample-to-row lookup still covers only ``pixels``.
    """
    pixels = np.asarray(pixels, dtype=np.int32)
    if required_pixels is None:
        unique_pixels, sample_to_unique_idx = np.unique(pixels, return_inverse=True)
    else:
        required_pixels = np.asarray(required_pixels, dtype=np.int32).reshape(-1)
        unique_pixels = np.unique(np.concatenate([pixels, required_pixels]))
        sample_to_unique_idx = np.searchsorted(unique_pixels, pixels)
    unique_pixels = unique_pixels.astype(np.int32, copy=False)
    sample_to_unique_idx = sample_to_unique_idx.astype(np.int32, copy=False)
    return (
        unique_pixels,
        sample_to_unique_idx,
        zgals[unique_pixels],
        dzgals[unique_pixels],
        wgals[unique_pixels],
        ngals[unique_pixels],
    )


def validate_loaded_survey_shapes(data):
    """Validate compact per-pixel galaxy counts returned by ``load_all_data``.

    ``ngals_pe`` and ``ngals_sel`` are compact catalog-row arrays, so they
    should match the corresponding compact pixel arrays whenever a survey (or
    bright-siren counterpart catalog) is available.
    """
    if data.get("zgals_catalog") is None:
        return

    compact_shape_checks = (
        ("ngals_pe", "unique_pixels_pe", "PE"),
        ("ngals_sel", "unique_pixels_sel", "selection"),
    )
    for ngals_key, pixels_key, label in compact_shape_checks:
        ngals_value = data.get(ngals_key)
        pixels_value = data.get(pixels_key)
        if ngals_value is None or pixels_value is None:
            continue
        ngals_n = int(np.asarray(ngals_value).shape[0])
        pixels_n = int(np.asarray(pixels_value).shape[0])
        if ngals_n != pixels_n:
            raise ValueError(
                f"Survey {label} count shape mismatch: {ngals_key}.shape[0] "
                f"({ngals_n}) must equal {pixels_key}.shape[0] ({pixels_n})."
            )

    sample_shape_checks = (
        ("sample_to_unique_pe", "pixels_pe", "PE"),
        ("sample_to_unique_sel", "pixels_sel", "selection"),
    )
    for sample_key, pixels_key, label in sample_shape_checks:
        sample_value = data.get(sample_key)
        pixels_value = data.get(pixels_key)
        if sample_value is None or pixels_value is None:
            continue
        sample_n = int(np.asarray(sample_value).shape[0])
        pixels_n = int(np.asarray(pixels_value).shape[0])
        if sample_n != pixels_n:
            raise ValueError(
                f"Survey {label} sample map shape mismatch: "
                f"{sample_key}.shape[0] ({sample_n}) must equal "
                f"{pixels_key}.shape[0] ({pixels_n})."
            )


def _catalog_memory_diagnostics(zgals, dzgals, wgals, pixels_pe, pixels_sel, ngals_pe, ngals_sel):
    """Summarise memory saved by compact unique-pixel catalog views."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    row_bytes = sum(arr.dtype.itemsize * arr.shape[1] for arr in (zgals, dzgals, wgals))
    duplicated_pe = max(0, np.asarray(pixels_pe).size - unique_pe.size) * row_bytes
    duplicated_sel = max(0, np.asarray(pixels_sel).size - unique_sel.size) * row_bytes
    max_gals = 0
    if ngals_pe is not None and ngals_pe.size:
        max_gals = max(max_gals, int(np.max(ngals_pe)))
    if ngals_sel is not None and ngals_sel.size:
        max_gals = max(max_gals, int(np.max(ngals_sel)))
    return {
        "unique_pe_pixels": int(unique_pe.size),
        "unique_sel_pixels": int(unique_sel.size),
        "duplicated_catalog_bytes_avoided": int(duplicated_pe + duplicated_sel),
        "max_galaxies_per_unique_pixel": max_gals,
    }
