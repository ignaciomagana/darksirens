"""
utils.py
--------
Module-level utilities and redshift grid shared across all submodules.

The grid is defined once here so that JAX can trace through it at
compile time (via `jit`) without recompilation when it is imported by
multiple submodules.  Using a log-spaced grid gives finer resolution
at low redshift where the catalog is densest, and coarser resolution
at high redshift where the prior is smooth.
"""

import jax.numpy as jnp
import numpy as np
import h5py

# Log-spaced from z~0 to zMax, giving 1000 points.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
zMax: float = 5.0
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), 1000))


def load_survey(survey_path, to_device=True):
    """Load the pixelated survey. ``to_device=False`` keeps the dense full-sky
    arrays on the host so callers can compact before transferring to device."""
    asarray = jnp.asarray if to_device else np.asarray
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = asarray(f['zgals'])
        ngals = asarray(f['ngals'])
        dzgals = asarray(f['dzgals'])
        wgals = asarray(f['wgals'])
    return nside, ngals, zgals, dzgals, wgals


#: Per-galaxy "mark" datasets optionally written by ``darksirens_pixelate``
#: (padded ``(npix, maxgals)`` arrays), keyed by the EMCatalog field name.
MARK_DATASETS = ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color")


def load_survey_marks(survey_path):
    """Load any per-galaxy mark datasets present in the pixelated survey file.

    Returns ``{dataset_name: (npix, maxgals) ndarray}`` for whichever of
    :data:`MARK_DATASETS` exist (empty dict if none).  These are the *raw* marks;
    z-centering happens at load (``inference/data.py``).
    """
    out = {}
    with h5py.File(survey_path, 'r') as f:
        for ds in MARK_DATASETS:
            if ds in f:
                out[ds] = np.asarray(f[ds])
    return out
