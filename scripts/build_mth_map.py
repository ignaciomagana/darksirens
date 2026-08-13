#!/usr/bin/env python
"""Per-pixel magnitude-threshold (m_th) depth map from survey backbone files.

Promoted from ``experiments/desi_ingest/build_mth_map.py`` (field-level PR-2)
with the paths parameterized; the algorithm is unchanged.  For every native
RING pixel, accumulates streaming histograms of

* ``m_5sigma = 22.5 - 2.5 log10(5 / sqrt(flux_ivar_r))`` (point-source depth
  proxy per object), and
* ``dered_mag_r`` (for the faint-tail q99 turnover check),

over the given source HDF5 files.  The catalog retention cut is ``m_ret``
(r <= 21 for the DESI union), and detection at the cut is ~complete wherever
the 5-sigma depth exceeds it, so the honest per-pixel effective limit is
``m_th_eff = min(m_ret, median m_5sigma)``.  Strata = K quantile bins of
``m_th_eff`` over occupied pixels (stratum -1 off footprint).

Output HDF5: ``mth_eff_map, median_m5_map, q99_mag_map, stratum_map,
stratum_edges, counts, masked_frac`` (+ provenance JSON).  ``masked_frac``
and ``counts`` are what ``darksirens.catalogs.depth_map.load_selection_fraction``
consumes for the ``--per_pixel_completeness`` selection fraction
``f_p = 1 - masked_frac``.

Usage:
    python scripts/build_mth_map.py --sources north.h5 south.h5 \
        --out mth_map_nside128.h5 [--nside 128] [--m-retention 21.0]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

CHUNK = 20_000_000
M5_EDGES = np.linspace(19.0, 24.0, 101)     # 0.05-mag bins
MAG_EDGES = np.linspace(18.0, 21.0, 61)


def _accumulate(path: Path, npix: int, h_m5, h_mag, counts, masked) -> None:
    with h5py.File(path, "r") as f:
        pix_key = next(k for k in f.keys() if k.upper().startswith("HPX"))
        n = f[pix_key].shape[0]
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            pix = f[pix_key][lo:hi].astype(np.int64)
            ivar = f["flux_ivar_r"][lo:hi]
            mag = f["dered_mag_r"][lo:hi]
            mb = f["maskbits"][lo:hi]

            ok = (pix >= 0) & (pix < npix)
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="source HDF5 files (HPX<nside>_RING pixel column, "
                         "flux_ivar_r, dered_mag_r, maskbits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nside", type=int, default=128)
    ap.add_argument("--m-retention", type=float, default=21.0)
    ap.add_argument("--k-strata", type=int, default=4)
    args = ap.parse_args(argv)

    npix = 12 * args.nside ** 2
    h_m5 = np.zeros((npix, len(M5_EDGES) - 1), dtype=np.uint32)
    h_mag = np.zeros((npix, len(MAG_EDGES) - 1), dtype=np.uint32)
    counts = np.zeros(npix, dtype=np.uint64)
    masked = np.zeros(npix, dtype=np.uint64)

    for p in args.sources:
        print(f"[mth] streaming {p}")
        _accumulate(Path(p), npix, h_m5, h_mag, counts, masked)

    med_m5 = _hist_quantile(h_m5, M5_EDGES, 0.5)
    q99_mag = _hist_quantile(h_mag, MAG_EDGES, 0.99)
    mth_eff = np.minimum(args.m_retention, med_m5)

    occ = counts > 0
    finite = occ & np.isfinite(mth_eff)
    qs = np.quantile(mth_eff[finite], np.linspace(0, 1, args.k_strata + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    stratum = np.full(npix, -1, dtype=np.int8)
    stratum[finite] = np.clip(np.digitize(mth_eff[finite], qs) - 1, 0,
                              args.k_strata - 1)

    out = Path(args.out)
    with h5py.File(out, "w") as f:
        f.attrs["nside"] = args.nside
        f.attrs["ordering"] = "RING"
        f.attrs["m_retention"] = args.m_retention
        f.attrs["k_strata"] = args.k_strata
        f.create_dataset("mth_eff_map", data=mth_eff, compression="gzip")
        f.create_dataset("median_m5_map", data=med_m5, compression="gzip")
        f.create_dataset("q99_mag_map", data=q99_mag, compression="gzip")
        f.create_dataset("stratum_map", data=stratum, compression="gzip")
        f.create_dataset("stratum_edges", data=np.quantile(
            mth_eff[finite], np.linspace(0, 1, args.k_strata + 1)))
        f.create_dataset("counts", data=counts, compression="gzip")
        with np.errstate(invalid="ignore"):
            f.create_dataset("masked_frac",
                             data=np.where(occ, masked / np.maximum(counts, 1),
                                           np.nan), compression="gzip")

    prov = {
        "sources": [str(p) for p in args.sources],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_occupied_pixels": int(occ.sum()),
        "mth_eff_quantiles_10_50_90": [float(v) for v in np.quantile(
            mth_eff[finite], [0.1, 0.5, 0.9])],
        "frac_pixels_shallower_than_retention":
            float((mth_eff[finite] < args.m_retention - 1e-9).mean()),
        "stratum_edges": [float(v) for v in np.quantile(
            mth_eff[finite], np.linspace(0, 1, args.k_strata + 1))],
        "note": "m5sigma is a point-source depth proxy; detection at the "
                "retention cut is ~complete wherever median m5 exceeds it, "
                "so mth_eff = min(m_ret, median m5). q99_mag flags pixels "
                "whose faint tail stops short of the cut.",
    }
    with open(out.with_suffix(out.suffix + ".provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
