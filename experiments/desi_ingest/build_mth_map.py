"""Per-pixel magnitude-threshold (m_th) map from the Legacy Survey backbone.

For every nside-128 RING pixel, accumulates streaming histograms of
  * m_5sigma = 22.5 - 2.5 log10(5 / sqrt(flux_ivar_r))  (point-source depth
    proxy per object), and
  * dered_mag_r (for the faint-tail q99 turnover check),
over ls_dr9north_m21_v2.h5 (the _v2 file -- the original has int-truncated
photo-z/flux columns) and ls_dr10south_m21.h5.

The catalog retention cut is r <= 21, and LS r-band 5-sigma depth is
typically ~23, so detection at 21 is ~complete on most of the footprint; the
honest per-pixel effective limit is m_th_eff = min(21, median m_5sigma).
Strata = K=4 quantile bins of m_th_eff over occupied pixels (stratum -1 off
footprint), the input the Stage-B multi-stratum machinery consumes later.

Output: data/mth_map_nside128.h5 (mth_eff_map, median_m5_map, q99_mag_map,
stratum_map, counts, masked_frac + bin edges) with provenance JSON.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

import common as C

NSIDE = 128
NPIX = 12 * NSIDE * NSIDE
CHUNK = 20_000_000
M5_EDGES = np.linspace(19.0, 24.0, 101)     # 0.05-mag bins
MAG_EDGES = np.linspace(18.0, 21.0, 61)
K_STRATA = 4
M_RETENTION = 21.0


def _accumulate(path, h_m5, h_mag, counts, masked):
    with h5py.File(path, "r") as f:
        n = f["HPX128_RING"].shape[0]
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            pix = f["HPX128_RING"][lo:hi].astype(np.int64)
            ivar = f["flux_ivar_r"][lo:hi]
            mag = f["dered_mag_r"][lo:hi]
            mb = f["maskbits"][lo:hi]

            ok = (pix >= 0) & (pix < NPIX)
            np.add.at(counts, pix[ok], 1)
            np.add.at(masked, pix[ok & (mb != 0)], 1)

            good = ok & (ivar > 0)
            m5 = 22.5 - 2.5 * np.log10(5.0 / np.sqrt(ivar[good]))
            b5 = np.clip(np.digitize(m5, M5_EDGES) - 1, 0, len(M5_EDGES) - 2)
            np.add.at(h_m5, (pix[good], b5), 1)

            gm = ok & np.isfinite(mag)
            bm = np.clip(np.digitize(mag[gm], MAG_EDGES) - 1, 0,
                         len(MAG_EDGES) - 2)
            np.add.at(h_mag, (pix[gm], bm), 1)
            print(f"  {path.name}: {hi:,}/{n:,}", flush=True)


def _hist_quantile(hist, edges, q):
    """Per-pixel quantile from binned counts (bin-center convention)."""
    cum = np.cumsum(hist, axis=1)
    tot = cum[:, -1]
    out = np.full(hist.shape[0], np.nan)
    occ = tot > 0
    target = q * tot[occ]
    idx = np.array([np.searchsorted(c, t) for c, t in
                    zip(cum[occ], target)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    out[occ] = centers[np.clip(idx, 0, len(centers) - 1)]
    return out


def main() -> None:
    h_m5 = np.zeros((NPIX, len(M5_EDGES) - 1), dtype=np.uint32)
    h_mag = np.zeros((NPIX, len(MAG_EDGES) - 1), dtype=np.uint32)
    counts = np.zeros(NPIX, dtype=np.uint64)
    masked = np.zeros(NPIX, dtype=np.uint64)

    for p in (C.LS_NORTH, C.LS_SOUTH):
        print(f"[mth] streaming {p}")
        _accumulate(p, h_m5, h_mag, counts, masked)

    med_m5 = _hist_quantile(h_m5, M5_EDGES, 0.5)
    q99_mag = _hist_quantile(h_mag, MAG_EDGES, 0.99)
    mth_eff = np.minimum(M_RETENTION, med_m5)

    occ = counts > 0
    finite = occ & np.isfinite(mth_eff)
    qs = np.quantile(mth_eff[finite], np.linspace(0, 1, K_STRATA + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    stratum = np.full(NPIX, -1, dtype=np.int8)
    stratum[finite] = np.clip(np.digitize(mth_eff[finite], qs) - 1, 0,
                              K_STRATA - 1)

    out = C.DATA_DIR / "mth_map_nside128.h5"
    with h5py.File(out, "w") as f:
        f.attrs["nside"] = NSIDE
        f.attrs["ordering"] = "RING"
        f.attrs["m_retention"] = M_RETENTION
        f.attrs["k_strata"] = K_STRATA
        f.create_dataset("mth_eff_map", data=mth_eff, compression="gzip")
        f.create_dataset("median_m5_map", data=med_m5, compression="gzip")
        f.create_dataset("q99_mag_map", data=q99_mag, compression="gzip")
        f.create_dataset("stratum_map", data=stratum, compression="gzip")
        f.create_dataset("stratum_edges", data=np.quantile(
            mth_eff[finite], np.linspace(0, 1, K_STRATA + 1)))
        f.create_dataset("counts", data=counts, compression="gzip")
        with np.errstate(invalid="ignore"):
            f.create_dataset("masked_frac",
                             data=np.where(occ, masked / np.maximum(counts, 1),
                                           np.nan), compression="gzip")

    prov = {
        "sources": [str(C.LS_NORTH), str(C.LS_SOUTH)],
        "n_occupied_pixels": int(occ.sum()),
        "mth_eff_quantiles_10_50_90": [float(v) for v in np.quantile(
            mth_eff[finite], [0.1, 0.5, 0.9])],
        "frac_pixels_shallower_than_retention":
            float((mth_eff[finite] < M_RETENTION - 1e-9).mean()),
        "stratum_edges": [float(v) for v in np.quantile(
            mth_eff[finite], np.linspace(0, 1, K_STRATA + 1))],
        "note": "m5sigma is a point-source depth proxy; detection at the "
                "r<=21 retention cut is ~complete wherever median m5 > 21, "
                "so mth_eff = min(21, median m5). q99_mag flags pixels whose "
                "faint tail stops short of the cut.",
    }
    C.write_provenance(out, prov)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
