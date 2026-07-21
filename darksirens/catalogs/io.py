"""Survey catalog I/O helpers.

This module loads pixelated survey HDF5 files and optional per-galaxy mark
datasets. Redshift grids live in :mod:`darksirens.redshift.grid`.
"""

import jax.numpy as jnp
import numpy as np
import h5py


def load_survey(survey_path, to_device=True):
    """Load the pixelated survey. ``to_device=False`` keeps the dense full-sky
    arrays on the host so callers can compact before transferring to device.

    ``z_depth`` is the optional per-survey redshift depth written by
    ``darksirens_pixelate --z_depth`` (``f.attrs['z_depth']``): the redshift
    beyond which the survey does not attempt to detect galaxies, used to
    bound the completion missing-galaxy budget
    (:mod:`darksirens.redshift.completion`).  ``None`` when the attribute is
    absent (legacy full-grid budget).
    """
    asarray = jnp.asarray if to_device else np.asarray
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = asarray(f['zgals'])
        ngals = asarray(f['ngals'])
        dzgals = asarray(f['dzgals'])
        wgals = asarray(f['wgals'])
        z_depth = float(f.attrs['z_depth']) if 'z_depth' in f.attrs else None
    return nside, ngals, zgals, dzgals, wgals, z_depth


#: Per-galaxy "mark" datasets optionally written by ``darksirens_pixelate``
#: (padded ``(npix, maxgals)`` arrays), keyed by the EMCatalog field name.
MARK_DATASETS = ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color")


def load_survey_marks(survey_path, datasets=None):
    """Load per-galaxy mark datasets present in the pixelated survey file.

    Returns ``{dataset_name: (npix, maxgals) ndarray}`` for whichever of
    :data:`MARK_DATASETS` exist (empty dict if none).  These are the *raw* marks;
    z-centering happens at load (``inference/data.py``).

    ``datasets``, if given, restricts the read to that subset of
    :data:`MARK_DATASETS` (e.g. a K>=2 mixture catalog's requested marks only)
    so unrequested mark tables are never pulled off disk.  ``None`` (default)
    loads every mark dataset present, for callers that need to auto-detect
    which marks a catalog provides.
    """
    wanted = MARK_DATASETS if datasets is None else tuple(datasets)
    out = {}
    with h5py.File(survey_path, 'r') as f:
        for ds in wanted:
            if ds in f:
                out[ds] = np.asarray(f[ds])
    return out
