"""Checklist gate 8(a): machinery closure with delta-function PE at truth.

Every Tier C test so far has varied something DOWNSTREAM of the likelihood's
core -- the field, the b_gal draw covariance, the injection count, the catalog
depth -- and five candidate causes have been eliminated with numbers while the
~2.5x width deficit has not moved.  This gate goes upstream instead.

Replace each event's parameter-estimation samples with a delta function at that
event's TRUE parameters.  The PE then contributes no uncertainty at all, so:

* **correct coverage under delta-PE** -> the likelihood machinery (catalog term,
  completion, selection term) is calibrated, and the dispersion enters through
  the synthetic PE or the event construction;
* **wrong coverage under delta-PE** -> the width deficit lives in the
  estimator's own terms and is independent of the mock's PE.

It is the checklist's FIRST gate for exactly this reason: everything else is
downstream of it, so running it fifth rather than first is the wrong order and
this closes that gap.

Two construction choices, both stated because they are the places a delta-PE
file can be quietly wrong:

1. **Frames.**  The stored PE carries DETECTOR-frame masses; ``truth`` carries
   source-frame ``m1``/``m2`` and the true ``z``.  The delta file therefore
   writes ``m1det = m1 (1+z)``, not ``m1``.  Writing the source-frame value into
   a detector-frame column would shift every event's population weight.
2. **The PE prior.**  The likelihood importance-reweights by dividing out the
   PE prior ``p_pe``.  A delta posterior has no prior to divide out, so ``p_pe``
   is set to a constant.  Leaving the original per-sample ``p_pe`` would
   reweight identical samples by unequal weights -- an ESS of 1 dressed up as
   512.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def make_delta(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        nobs = int(f.attrs["nobs"])
        nsamp = int(f.attrs["nsamp"])
        t = f["truth"]
        z = np.asarray(t["z"][...])
        dl = np.asarray(t["dl"][...])
        m1 = np.asarray(t["m1"][...])
        m2 = np.asarray(t["m2"][...])
        chi = np.asarray(t["chi"][...])
        ra = np.asarray(t["ra"][...])
        dec = np.asarray(t["dec"][...])
        assert z.size == nobs, (z.size, nobs)

        rep = lambda a: np.repeat(a, nsamp)          # noqa: E731
        f["dL"][...] = rep(dl)
        f["m1det"][...] = rep(m1 * (1.0 + z))        # frames: see docstring
        f["m2det"][...] = rep(m2 * (1.0 + z))
        f["m1src"][...] = rep(m1)
        f["m2src"][...] = rep(m2)
        f["chieff"][...] = rep(chi)
        f["ra"][...] = rep(ra)
        f["dec"][...] = rep(dec)
        f["p_pe"][...] = np.ones(nobs * nsamp)       # see docstring
        f.attrs["delta_pe_at_truth"] = True
    with h5py.File(dst) as f:
        d = np.asarray(f["dL"][...]).reshape(int(f.attrs["nobs"]), -1)
        spread = float(np.max(d.max(axis=1) - d.min(axis=1)))
    print(f"[delta-PE] {dst.name}: max within-event dL spread = {spread:.3e} "
          f"(0 required)")
    return spread


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    make_delta(a.src, a.dst)


if __name__ == "__main__":
    main()
