"""Galaxy mark loading and centering helpers."""

import jax.numpy as jnp
import numpy as np

from darksirens.redshift import zgrid
from darksirens.catalogs.io import load_survey_marks

#: z-bins for centering per-galaxy marks (subtract the running mean E[m|z]).
_MARK_CENTER_NBINS = 40


def _center_marks(raw_marks: dict, zgals, ngals) -> dict:
    """Return z-centred marks ``m - E[m|z]`` (the global running mean over real
    galaxies), so the sampled ``eta`` measure host preference at fixed redshift.
    Padded slots are set to 0 (masked downstream).

    Non-finite marks (contract)
    ---------------------------
    ``darksirens_pixelate`` REJECTS non-finite mark values for real galaxies, so
    a supported catalog never reaches here with one.  The finite mask below is a
    backstop for hand-built or third-party catalogs: the per-bin mean is taken
    over FINITE real marks only, so a single NaN/inf cannot turn a whole
    redshift bin's mean -- and therefore every centred mark in that bin, i.e.
    every galaxy at that redshift across the whole sky -- into NaN.  The
    offending slot itself stays non-finite (NaN in, NaN out): it is a data
    error, and silently replacing it with the bin mean would be exactly the kind
    of quiet repair this guard exists to prevent.
    """
    zgals = np.asarray(zgals, dtype=float)
    maxg = zgals.shape[1]
    real = np.arange(maxg)[None, :] < np.asarray(ngals)[:, None]
    z_hi = float(np.asarray(zgrid)[-1])
    edges = np.linspace(0.0, z_hi, _MARK_CENTER_NBINS + 1)
    binc = np.clip(np.searchsorted(edges, zgals, side="right") - 1, 0, _MARK_CENTER_NBINS - 1)
    out = {}
    for name, M in raw_marks.items():
        M = np.asarray(M, dtype=float)
        good = real & np.isfinite(M)
        sums = np.zeros(_MARK_CENTER_NBINS)
        cnts = np.zeros(_MARK_CENTER_NBINS)
        np.add.at(sums, binc[good], M[good])
        np.add.at(cnts, binc[good], 1.0)
        binmean = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1.0), 0.0)
        Mc = np.where(real, M - binmean[binc], 0.0)
        out[name] = Mc
    return out


def load_and_center_survey_marks(survey_path, zgals, ngals, datasets=None) -> dict:
    """Load survey marks and return z-centred JAX arrays keyed by mark name.

    ``datasets``, if given, restricts the read/centering to that subset of
    dataset names (see :func:`darksirens.catalogs.io.load_survey_marks`) --
    e.g. a K>=2 mixture catalog's requested marks only, so unrequested
    per-galaxy mark tables (up to ~0.83 GB each at DESI-wide shape) are never
    loaded or centred.
    """
    raw_marks = load_survey_marks(survey_path, datasets=datasets)
    if not raw_marks:
        return {}
    centered = _center_marks(raw_marks, zgals, ngals)
    out = {name: jnp.asarray(arr) for name, arr in centered.items()}
    print(f"    - Loaded galaxy marks {sorted(raw_marks)} (z-centred)")
    return out
