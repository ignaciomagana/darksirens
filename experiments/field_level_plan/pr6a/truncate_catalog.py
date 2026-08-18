"""Rule 7, tested: truncate the mock catalog at ``z_depth`` as production is.

Checklist rule 7 says a depth knob must truncate the rows the likelihood
actually reads, and must be verified on the built artifact rather than the
builder's intent.  Checked on both catalogs:

    PRODUCTION (DESI, nside 64):  22,787,566 galaxies, max z = 0.3000,
                                  **0** above z_depth = 0.30
    CLOSURE MOCK (nside 16):         192,757 galaxies, max z = 0.3847,
                                  9,384 above (4.868% of count AND weight),
                                  in 1,594 of 1,854 occupied rows

Production truncates exactly at the depth.  The mock does not -- it carries
``z_depth`` as an HDF5 attribute while the rows run 28% past it.  So the mock
is not faithful to the line it validates, which is a defect on its own terms
regardless of what it costs.

Whether it costs anything is what this measures.  ``c_mode="selection"`` makes
``C(z)`` a parametric function of the selection fit, NOT of the catalog counts,
so dropping the above-depth rows does not move the completeness model: the
intervention is clean, and the only thing that changes is which galaxies the
observed branch can place a host on.

Writes a truncated copy of a world's catalog, leaving every other product
untouched, so Tier C can be re-run on the same seeds with one variable changed.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def truncate(src, dst, z_depth):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # In-place truncation is the common case (the tier rewrites each
    # realization's own catalog), so copying onto itself must be a no-op
    # rather than an error.
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        z = np.asarray(f["zgals"][...])
        ng = np.asarray(f["ngals"][...])
        dz = np.asarray(f["dzgals"][...]) if "dzgals" in f else None
        w = np.asarray(f["wgals"][...]) if "wgals" in f else None
        mag = np.asarray(f["gal_app_mag"][...]) if "gal_app_mag" in f else None

        dropped = 0
        for i in range(z.shape[0]):
            n = int(ng[i])
            if not n:
                continue
            zi = z[i, :n]
            keep = np.where(zi <= z_depth)[0]
            k = keep.size
            dropped += n - k
            # COMPACT the kept entries to the front, exactly the layout the
            # loader assumes (it reads [:ngals] per row); leaving holes would
            # silently feed the prior whatever padding value sits there.
            for arr in (z, dz, w, mag):
                if arr is None:
                    continue
                arr[i, :k] = arr[i, keep]
                arr[i, k:n] = arr[i, n - 1] if k else 0.0
            ng[i] = k
        f["zgals"][...] = z
        f["ngals"][...] = ng
        if dz is not None:
            f["dzgals"][...] = dz
        if w is not None:
            f["wgals"][...] = w
        if mag is not None:
            f["gal_app_mag"][...] = mag
        f.attrs["truncated_at_z_depth"] = float(z_depth)

    with h5py.File(dst) as f:
        z = np.asarray(f["zgals"][...]); ng = np.asarray(f["ngals"][...])
        mx = max((float(z[i, :int(ng[i])].max()) for i in range(z.shape[0])
                  if int(ng[i])), default=0.0)
        tot = int(ng.sum())
    print(f"[truncate] dropped {dropped:,} galaxies above z={z_depth}")
    print(f"[truncate] now {tot:,} galaxies, max z = {mx:.4f}")
    return dropped, tot, mx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--z-depth", type=float, default=0.30)
    a = ap.parse_args(argv)
    truncate(a.src, a.dst, a.z_depth)


if __name__ == "__main__":
    main()
