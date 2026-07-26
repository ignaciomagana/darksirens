"""Survey catalog I/O helpers.

This module loads pixelated survey HDF5 files and optional per-galaxy mark
datasets. Redshift grids live in :mod:`darksirens.redshift.grid`.
"""

import jax.numpy as jnp
import numpy as np
import h5py


def _row_z_sort_order(zgals, ngals):
    """Per-row permutation sorting the real-galaxy prefix by ascending z.

    Padded slots (index >= ngal) keep their positions after the real prefix
    (stable argsort on a key of +inf), so ``arange < ngal`` remains the real
    mask and padding values are untouched.  Deterministic: two callers reading
    the same file datasets derive bit-identical permutations, which is how
    :func:`load_survey_marks` stays co-indexed with :func:`load_survey`
    without any cross-call plumbing.
    """
    z = np.asarray(zgals)
    ng = np.asarray(ngals)
    real = np.arange(z.shape[1])[None, :] < ng[:, None]
    key = np.where(real, z, np.inf)
    return np.argsort(key, axis=1, kind="stable")


def sort_survey_rows_by_z(zgals, dzgals, wgals, ngals, extras=()):
    """Sort every padded catalog row by galaxy redshift, coherently.

    Applies ONE per-row permutation (from :func:`_row_z_sort_order`) to every
    co-indexed per-galaxy array: ``zgals`` / ``dzgals`` / ``wgals`` and any
    ``extras`` (e.g. mark tables).  ``ngals`` is per-row and returned as-is.

    This establishes the row z-sort invariant required by the windowed
    catalog-KDE evaluator (``darksirens.redshift.catalog``): within each row
    the first ``ngal`` entries are non-decreasing in z, real galaxies form a
    contiguous prefix, and padding follows.  The invariant is hard-asserted
    here, at construction.

    Returns ``(zgals, dzgals, wgals, ngals, extras)`` with extras a tuple
    (``None`` entries pass through).
    """
    order = _row_z_sort_order(zgals, ngals)

    def _take(a):
        return np.take_along_axis(np.asarray(a), order, axis=1)

    z_s = _take(zgals)
    ng = np.asarray(ngals)
    # Sort invariant, asserted where it is constructed: real prefix ascending.
    cols = np.arange(1, z_s.shape[1])[None, :]
    ok = (np.diff(z_s, axis=1) >= 0) | (cols >= ng[:, None])
    if not bool(np.all(ok)):
        raise AssertionError(
            "row z-sort invariant violated after sorting (NaN redshifts or "
            "real galaxies outside the ngal prefix?)"
        )
    extras_s = tuple(None if e is None else _take(e) for e in extras)
    return z_s, _take(dzgals), _take(wgals), ng, extras_s


def load_survey(survey_path, to_device=True, sort_rows_by_z=True):
    """Load the pixelated survey. ``to_device=False`` keeps the dense full-sky
    arrays on the host so callers can compact before transferring to device.

    ``z_depth`` is the optional per-survey redshift depth written by
    ``darksirens_pixelate --z_depth`` (``f.attrs['z_depth']``): the redshift
    beyond which the survey catalogs no galaxies, used as the completeness prior
    (completeness is zero beyond it; hosts there are missing, not nonexistent --
    :mod:`darksirens.redshift.completion`).  ``None`` when the attribute is
    absent (completeness estimated over the full grid; legacy path).

    ``sort_rows_by_z`` (default True) applies :func:`sort_survey_rows_by_z` on
    the host before any device transfer, establishing the per-row z-sort
    invariant the windowed catalog-KDE hot path requires.  Per-galaxy mark
    datasets loaded via :func:`load_survey_marks` derive the SAME permutation
    from the file, so they stay co-indexed.  Pass False to keep the raw file
    order (disables KDE windowing downstream; row-reduction results are
    identical either way up to floating-point summation order).
    """
    asarray = jnp.asarray if to_device else np.asarray
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = np.asarray(f['zgals'])
        ngals = np.asarray(f['ngals'])
        dzgals = np.asarray(f['dzgals'])
        wgals = np.asarray(f['wgals'])
        z_depth = float(f.attrs['z_depth']) if 'z_depth' in f.attrs else None
    if sort_rows_by_z:
        zgals, dzgals, wgals, ngals, _ = sort_survey_rows_by_z(
            zgals, dzgals, wgals, ngals
        )
    return (
        nside, asarray(ngals), asarray(zgals), asarray(dzgals), asarray(wgals),
        z_depth,
    )


#: Per-galaxy "mark" datasets optionally written by ``darksirens_pixelate``
#: (padded ``(npix, maxgals)`` arrays), keyed by the EMCatalog field name.
MARK_DATASETS = ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color")


def load_survey_marks(survey_path, datasets=None, sort_rows_by_z=True):
    """Load per-galaxy mark datasets present in the pixelated survey file.

    Returns ``{dataset_name: (npix, maxgals) ndarray}`` for whichever of
    :data:`MARK_DATASETS` exist (empty dict if none).  These are the *raw* marks;
    z-centering happens at load (``inference/data.py``).

    ``datasets``, if given, restricts the read to that subset of
    :data:`MARK_DATASETS` (e.g. a K>=2 mixture catalog's requested marks only)
    so unrequested mark tables are never pulled off disk.  ``None`` (default)
    loads every mark dataset present, for callers that need to auto-detect
    which marks a catalog provides.

    ``sort_rows_by_z`` (default True, matching :func:`load_survey`) permutes
    each mark row through the SAME per-row z-sort permutation ``load_survey``
    applies, re-derived deterministically from this file's ``zgals``/``ngals``
    (stable argsort of identical inputs), so marks stay co-indexed with the
    sorted galaxy arrays.  Pass False when pairing with
    ``load_survey(..., sort_rows_by_z=False)``.
    """
    wanted = MARK_DATASETS if datasets is None else tuple(datasets)
    out = {}
    with h5py.File(survey_path, 'r') as f:
        for ds in wanted:
            if ds in f:
                out[ds] = np.asarray(f[ds])
        if sort_rows_by_z and out:
            order = _row_z_sort_order(
                np.asarray(f['zgals']), np.asarray(f['ngals'])
            )
            out = {
                ds: np.take_along_axis(arr, order, axis=1)
                for ds, arr in out.items()
            }
    return out
