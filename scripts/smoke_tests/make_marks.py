#!/usr/bin/env python3
"""
make_marks.py — add synthetic per-galaxy marks to a pixelated survey catalog.

The mock generator (``scripts/mock_dark_sirens/generate_mock_data.py``) writes a
pixelated catalog with ``zgals/dzgals/wgals/ngals`` but no per-galaxy marks, so
the marked-host model (``--mark_model loglinear``) has nothing to read. This
helper copies a pixelated catalog and adds ``mark_logmstar`` / ``mark_logssfr``
columns (the dataset names ``darksirens_pixelate`` would emit), filled with
mildly z-correlated synthetic values for the real galaxies and 0 for padding.

It only needs to make ``--mark_model loglinear`` *run* end-to-end for the smoke
suite; it does not inject a recoverable host preference (the marks are
z-centred at load, so a smoke run simply exercises the code path).

Usage
-----
    python scripts/smoke_tests/make_marks.py --catalog cat_pixelated.h5 \
        [--out cat_marked.h5] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

# Dataset names must match darksirens_pixelate's mark columns so load_survey_marks
# picks them up (see darksirens/tool/darksirens_pixelate.py).
_MARK_DATASETS = ("mark_logmstar", "mark_logssfr")


def add_marks(catalog_path: str, out_path: str, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    with h5py.File(catalog_path, "r") as f:
        zgals = np.asarray(f["zgals"], dtype=float)
        ngals = np.asarray(f["ngals"]).astype(int)
        datasets = {k: np.asarray(f[k]) for k in f.keys()}
        attrs = dict(f.attrs)

    npix, maxg = zgals.shape
    real = np.arange(maxg)[None, :] < ngals[:, None]          # (npix, maxg) mask

    # Mildly z-correlated marks for real galaxies; 0 for padded slots.
    z = np.where(real, zgals, 0.0)
    logmstar = np.where(real, 10.5 + 0.6 * z + 0.3 * rng.standard_normal((npix, maxg)), 0.0)
    logssfr = np.where(real, -9.5 - 0.4 * z + 0.3 * rng.standard_normal((npix, maxg)), 0.0)
    marks = {"mark_logmstar": logmstar, "mark_logssfr": logssfr}

    with h5py.File(out_path, "w") as f:
        for k, v in datasets.items():
            f.create_dataset(k, data=v)
        for k, v in marks.items():
            if k not in datasets:
                f.create_dataset(k, data=v.astype(float))
        for k, v in attrs.items():
            f.attrs[k] = v
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Add synthetic marks to a pixelated catalog.")
    p.add_argument("--catalog", required=True, help="Pixelated survey catalog HDF5 (input).")
    p.add_argument("--out", default=None, help="Output HDF5 (default: <catalog>_marked.h5).")
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args(argv)
    out = opts.out
    if out is None:
        root, ext = os.path.splitext(opts.catalog)
        out = f"{root}_marked{ext or '.h5'}"
    add_marks(opts.catalog, out, seed=opts.seed)
    print(f"[make_marks] wrote {out} with marks {list(_MARK_DATASETS)}")


if __name__ == "__main__":
    main()
