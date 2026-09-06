"""Survey catalog I/O helpers.

This module loads pixelated survey HDF5 files and optional per-galaxy mark
datasets. Redshift grids live in :mod:`darksirens.redshift.grid`.
"""

import jax.numpy as jnp
import numpy as np
import h5py


#: Raised by :func:`sort_survey_rows_by_z` (both implementations) when the
#: real-galaxy prefix is not ascending after sorting; ``cli/pixelate.py``
#: refers to this message.
ROW_Z_SORT_INVARIANT_ERROR = (
    "row z-sort invariant violated after sorting (NaN redshifts or "
    "real galaxies outside the ngal prefix?)"
)


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


def _row_z_sort_order_device(zgals, ngals):
    """Device twin of :func:`_row_z_sort_order` (same permutation, exactly).

    ``jnp.argsort(..., stable=True)`` is the same stable sort on the same
    ``+inf``-padded key, so ties -- including every padding slot -- break by
    column index in both implementations and the two permutations are equal
    element for element (verified on the full production catalog).  The order
    is returned as ``int32``: the padded width is a few thousand columns, and
    halving the index table halves the transient device footprint of the
    load.
    """
    z = jnp.asarray(zgals)
    ng = jnp.asarray(ngals)
    real = jnp.arange(z.shape[1])[None, :] < ng[:, None]
    key = jnp.where(real, z, jnp.inf)
    return jnp.argsort(key, axis=1, stable=True).astype(jnp.int32)


def sort_survey_rows_by_z(zgals, dzgals, wgals, ngals, extras=(), to_device=False):
    """Sort every padded catalog row by galaxy redshift, coherently.

    Applies ONE per-row permutation (from :func:`_row_z_sort_order`) to every
    co-indexed per-galaxy array: ``zgals`` / ``dzgals`` / ``wgals`` and any
    ``extras`` (e.g. mark tables).  ``ngals`` is per-row and returned as-is.

    This establishes the row z-sort invariant required by the windowed
    catalog-KDE evaluator (``darksirens.redshift.catalog``): within each row
    the first ``ngal`` entries are non-decreasing in z, real galaxies form a
    contiguous prefix, and padding follows.  The invariant is hard-asserted
    here, at construction.

    ``to_device=True`` runs the identical operation on the accelerator
    (:func:`_sort_survey_rows_by_z_device`) and returns device arrays: the
    caller is about to upload these arrays anyway, so the sort rides along
    instead of costing seconds of single-threaded host work first.  The
    default host path stays for callers that must compact on the host before
    any transfer (``load_survey(..., to_device=False)``).

    Returns ``(zgals, dzgals, wgals, ngals, extras)`` with extras a tuple
    (``None`` entries pass through).
    """
    if to_device:
        return _sort_survey_rows_by_z_device(zgals, dzgals, wgals, ngals, extras)
    order = _row_z_sort_order(zgals, ngals)

    def _take(a):
        return np.take_along_axis(np.asarray(a), order, axis=1)

    z_s = _take(zgals)
    ng = np.asarray(ngals)
    # Sort invariant, asserted where it is constructed: real prefix ascending.
    cols = np.arange(1, z_s.shape[1])[None, :]
    ok = (np.diff(z_s, axis=1) >= 0) | (cols >= ng[:, None])
    if not bool(np.all(ok)):
        raise AssertionError(ROW_Z_SORT_INVARIANT_ERROR)
    extras_s = tuple(None if e is None else _take(e) for e in extras)
    return z_s, _take(dzgals), _take(wgals), ng, extras_s


def _sort_survey_rows_by_z_device(zgals, dzgals, wgals, ngals, extras=()):
    """Device implementation of :func:`sort_survey_rows_by_z`.

    Same permutation, same outputs, bit for bit -- only the machine differs:
    the (npix, maxgals) key, the stable argsort and the gathers run on the
    accelerator instead of one host core, and the invariant is a device
    reduction with a single scalar transfer instead of a full-width
    ``np.diff`` on the host.

    Each source array is uploaded, gathered and then dropped one at a time so
    the transient device footprint stays near two copies of one array plus the
    int32 order, rather than holding every raw and gathered array at once.
    """
    ng = jnp.asarray(ngals)
    z_d = jnp.asarray(zgals)
    order = _row_z_sort_order_device(z_d, ng)

    def _take(a):
        a_d = jnp.asarray(a)
        out = jnp.take_along_axis(a_d, order, axis=1)
        out.block_until_ready()  # raw upload is dead once the gather is done
        del a_d
        return out

    z_s = jnp.take_along_axis(z_d, order, axis=1)
    z_s.block_until_ready()
    del z_d
    # Sort invariant, asserted where it is constructed: real prefix ascending.
    cols = jnp.arange(1, z_s.shape[1])[None, :]
    ok = jnp.all((jnp.diff(z_s, axis=1) >= 0) | (cols >= ng[:, None]))
    if not bool(ok):
        raise AssertionError(ROW_Z_SORT_INVARIANT_ERROR)
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

    ``sort_rows_by_z`` (default True) applies :func:`sort_survey_rows_by_z`,
    establishing the per-row z-sort invariant the windowed catalog-KDE hot
    path requires.  It runs on the device when ``to_device=True`` (the arrays
    are bound for the device anyway) and on the host otherwise, so callers
    that compact before transferring still never touch the accelerator.  Per-galaxy mark
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
            zgals, dzgals, wgals, ngals, to_device=to_device
        )
    return (
        nside, asarray(ngals), asarray(zgals), asarray(dzgals), asarray(wgals),
        z_depth,
    )


#: Per-galaxy "mark" datasets optionally written by ``darksirens_pixelate``
#: (padded ``(npix, maxgals)`` arrays), keyed by the EMCatalog field name.
MARK_DATASETS = ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color")

#: Per-galaxy PROPERTY datasets (padded like marks) that are NOT marks: they
#: never enter the sampled likelihood and never enroll in the marked-host
#: model -- they exist for OFFLINE consumers only (the selection-function fit
#: and the Q-table builder).  Kept in a separate registry precisely so a
#: catalog carrying magnitudes cannot silently become a "marked" catalog.
#: Padding value is 0.0 (an absurd apparent magnitude): every reader MUST mask
#: real slots via ``ngals`` (``arange < ngal``), never by value.
#: ``gal_stratum`` carries an integer stratum label per galaxy (imaging side
#: or m_th stratum) as a padded float table; -pad 0.0 is NOT a label, mask by
#: ``ngals`` like every galprop.
GALPROP_DATASETS = ("gal_app_mag", "gal_stratum")


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


def load_survey_galprops(survey_path, datasets=None, sort_rows_by_z=True):
    """Load per-galaxy property datasets (:data:`GALPROP_DATASETS`).

    Same padded layout and row z-sort permutation as
    :func:`load_survey_marks` (the permutation is re-derived from this file's
    ``zgals``/``ngals``, so properties stay co-indexed with the sorted galaxy
    arrays) -- but from the separate non-mark registry, so requesting
    galaxy properties can never enroll a catalog into the marked-host model.
    Real slots must be masked via ``ngals``; the 0.0 padding is not a value.
    """
    wanted = GALPROP_DATASETS if datasets is None else tuple(datasets)
    return load_survey_marks(
        survey_path, datasets=wanted, sort_rows_by_z=sort_rows_by_z)
