"""Full-sky north/south stratum map at the survey nside (RING).

Per-pixel majority vote of the galaxies' STRATUM labels (0=LS-DR10 south,
1=LS-DR9 north); empty and off-footprint pixels get the label of their
NEAREST occupied neighbour along the healpix ring ordering fallback -- in
practice the map is used only through (a) occupied-pixel gathers and (b)
per-stratum empty-pixel counts, where the off-footprint policy label just
decides which stratum's (1 - C_sel) budget those pixels carry.  Policy:
majority stratum of the whole catalog (south), documented here and in the
provenance JSON.
"""

from __future__ import annotations

import h5py
import healpy as hp
import numpy as np

import common as C

NSIDE = C.NSIDE_PRIMARY


def main() -> None:
    with h5py.File(C.DATA_DIR / "desi_union_raw.h5", "r") as f:
        ra = np.radians(f["TARGET_RA"][...])
        dec = np.radians(f["TARGET_DEC"][...])
        stratum = f["STRATUM"][...].astype(np.int64)
    pix = hp.ang2pix(NSIDE, np.pi / 2.0 - dec, ra)
    npix = 12 * NSIDE * NSIDE
    n_north = np.bincount(pix[stratum == 1], minlength=npix)
    n_total = np.bincount(pix, minlength=npix)
    majority_global = int(np.round(stratum.mean()))     # 0 = south
    smap = np.full(npix, majority_global, dtype=np.int32)
    occ = n_total > 0
    smap[occ] = (n_north[occ] * 2 > n_total[occ]).astype(np.int32)

    mixed = occ & (n_north > 0) & (n_north < n_total)
    out = C.DATA_DIR / f"stratum_map_ns_nside{NSIDE}.h5"
    with h5py.File(out, "w") as f:
        f.attrs["nside"] = NSIDE
        f.attrs["ordering"] = "RING"
        f.attrs["labels"] = "0=south(DR10) 1=north(DR9)"
        f.attrs["off_footprint_policy"] = f"majority stratum ({majority_global})"
        f.create_dataset("stratum_map", data=smap, compression="gzip")
    C.write_provenance(out, {
        "n_occupied": int(occ.sum()),
        "n_north_pixels": int((smap[occ] == 1).sum()),
        "n_south_pixels": int((smap[occ] == 0).sum()),
        "n_boundary_mixed_pixels": int(mixed.sum()),
        "off_footprint_label": majority_global,
    })
    print(f"wrote {out}: {occ.sum():,} occupied "
          f"({(smap[occ] == 1).sum():,} north / {(smap[occ] == 0).sum():,} "
          f"south), {mixed.sum():,} boundary pixels majority-assigned")


if __name__ == "__main__":
    main()
