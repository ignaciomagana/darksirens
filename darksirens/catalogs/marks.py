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
    Padded slots are set to 0 (masked downstream)."""
    zgals = np.asarray(zgals, dtype=float)
    maxg = zgals.shape[1]
    real = np.arange(maxg)[None, :] < np.asarray(ngals)[:, None]
    z_hi = float(np.asarray(zgrid)[-1])
    edges = np.linspace(0.0, z_hi, _MARK_CENTER_NBINS + 1)
    binc = np.clip(np.searchsorted(edges, zgals, side="right") - 1, 0, _MARK_CENTER_NBINS - 1)
    out = {}
    for name, M in raw_marks.items():
        M = np.asarray(M, dtype=float)
        sums = np.zeros(_MARK_CENTER_NBINS)
        cnts = np.zeros(_MARK_CENTER_NBINS)
        np.add.at(sums, binc[real], M[real])
        np.add.at(cnts, binc[real], 1.0)
        binmean = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1.0), 0.0)
        Mc = np.where(real, M - binmean[binc], 0.0)
        out[name] = Mc
    return out


def load_and_center_survey_marks(survey_path, zgals, ngals) -> dict:
    """Load survey marks and return z-centred JAX arrays keyed by mark name."""
    raw_marks = load_survey_marks(survey_path)
    if not raw_marks:
        return {}
    centered = _center_marks(raw_marks, zgals, ngals)
    out = {name: jnp.asarray(arr) for name, arr in centered.items()}
    print(f"    - Loaded galaxy marks {sorted(raw_marks)} (z-centred)")
    return out
