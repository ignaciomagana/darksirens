"""Per-pixel selection fraction ``f_p`` from a magnitude-threshold depth map.

PR-2 of the field-level ladder (OWNER DECISION 4a): the per-pixel completeness
is ``C_p(z) = f_p * C(z; theta_sel)`` with ``f_p = 1 - masked_frac`` — the
fraction of the pixel's area not lost to survey mask bits — degraded from the
depth map's native nside to the catalog nside by equal-area (plain child)
averaging.  ``f_p`` multiplies the SURVEY-CURVE completeness on both sides of
the budget (the per-row missing density and the field normalizer), so the
missing-budget identity keeps holding row by row.

The map artifact is ``build_mth_map.py``'s output
(``mth_map_nside<N>.h5``: ``masked_frac``, ``counts``, ... RING ordering).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionFractionMap:
    """``f_p`` on the catalog grid plus the numbers the PR-2 gates quote."""
    f_p: np.ndarray            # (n_pix_out,) float64 in [0, 1], RING
    nside: int
    area_deg2: float           # sum_p f_p * Omega_pix over f_p > 0
    n_covered: int             # pixels with any source coverage (counts > 0)
    n_zero: int                # pixels with f_p == 0

    def coverage_report(self, ngals: np.ndarray) -> dict:
        """Occupied / in-footprint-partial / off-footprint pixel classes.

        ``ngals`` is the catalog's per-pixel galaxy count on the same grid.
        A zero-count pixel INSIDE coverage (f_p > 0) is a measurement (an
        empty but observed pixel); outside coverage it is no data.
        """
        occ = np.asarray(ngals) > 0
        cov = self.f_p > 0.0
        return dict(
            n_pix=int(self.f_p.size),
            n_occupied=int(occ.sum()),
            n_covered=int(cov.sum()),
            n_occupied_partial=int((occ & (self.f_p < 1.0) & cov).sum()),
            n_empty_covered=int((~occ & cov).sum()),
            n_off_footprint=int((~occ & ~cov).sum()),
            n_occupied_uncovered=int((occ & ~cov).sum()),
            f_p_occupied_mean=float(self.f_p[occ].mean()) if occ.any() else 0.0,
            f_p_occupied_min=float(self.f_p[occ].min()) if occ.any() else 0.0,
            area_deg2=self.area_deg2,
        )


def _degrade_ring(vals: np.ndarray, weights: np.ndarray, nside_in: int,
                  nside_out: int) -> np.ndarray:
    """Weighted equal-area degrade of a RING map (weights=1 -> plain mean).

    HEALPix children of a NESTED pixel are contiguous, so degrade in NEST:
    RING -> NEST reorder, reshape (n_out, ratio), weighted mean, NEST -> RING.
    Pixels with zero total weight degrade to 0.
    """
    import healpy as hp

    if nside_out > nside_in:
        raise ValueError(f"cannot degrade nside {nside_in} -> {nside_out}")
    ratio = (nside_in // nside_out) ** 2
    ring2nest = hp.ring2nest(nside_in, np.arange(12 * nside_in ** 2))
    v_nest = np.zeros_like(vals, dtype=float)
    w_nest = np.zeros_like(weights, dtype=float)
    v_nest[ring2nest] = vals
    w_nest[ring2nest] = weights
    v = (v_nest * w_nest).reshape(-1, ratio).sum(1)
    w = w_nest.reshape(-1, ratio).sum(1)
    out_nest = np.where(w > 0.0, v / np.where(w > 0.0, w, 1.0), 0.0)
    nest2ring_ids = hp.nest2ring(nside_out, np.arange(12 * nside_out ** 2))
    out = np.zeros_like(out_nest)
    out[nest2ring_ids] = out_nest
    return out


def load_selection_fraction(mth_map_path, nside_out: int) -> SelectionFractionMap:
    """``f_p = 1 - masked_frac`` degraded to ``nside_out`` by area weighting.

    Uncovered native pixels (``counts == 0``) carry no masked-fraction
    measurement; they enter the degrade with zero weight, and an output pixel
    with NO covered children gets ``f_p = 0`` (off-footprint: the survey saw
    nothing there, so its selection fraction for the catalog is zero — those
    pixels' missing budget is the full ``dN_exp``, exactly the current
    empty-pixel behaviour under ``C -> 0``).
    """
    import h5py

    with h5py.File(mth_map_path, "r") as f:
        if str(f.attrs.get("ordering", "RING")).upper() != "RING":
            raise ValueError(f"{mth_map_path}: expected RING ordering")
        nside_in = int(f.attrs["nside"])
        masked_frac = np.asarray(f["masked_frac"][...], dtype=float)
        counts = np.asarray(f["counts"][...], dtype=float)

    covered = counts > 0
    # masked_frac is NaN on uncovered pixels by construction (0/0); they
    # enter the degrade as f = 0 (no coverage -> no selection fraction).
    f_native = np.where(covered,
                        np.clip(1.0 - np.nan_to_num(masked_frac, nan=1.0),
                                0.0, 1.0),
                        0.0)
    # weight = coverage indicator: an output pixel's f_p is the mean over its
    # COVERED children (area weighting; children are equal-area), times the
    # covered-child fraction — i.e. uncovered area contributes f = 0.
    f_num = _degrade_ring(f_native, np.ones_like(f_native),
                          nside_in, nside_out)
    f_p = np.clip(f_num, 0.0, 1.0)

    import healpy as hp
    omega_deg2 = hp.nside2pixarea(nside_out, degrees=True)
    return SelectionFractionMap(
        f_p=f_p, nside=nside_out,
        area_deg2=float(f_p.sum() * omega_deg2),
        n_covered=int((f_p > 0).sum()),
        n_zero=int((f_p == 0).sum()))
