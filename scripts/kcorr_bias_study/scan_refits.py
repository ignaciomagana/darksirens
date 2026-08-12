#!/usr/bin/env python
"""One-line-per-catalog-size summary of the refit-stage JSONs."""
from __future__ import annotations

import glob
import json
import sys

rows = []
for f in sorted(glob.glob(sys.argv[1])):
    r = json.load(open(f))["refit"]
    fid = r["fiducial"]
    by = {x["name"]: x for x in r["rows"]}
    rows.append((r["n_gal"], fid["laplace_sd_M0hat"], by))
rows.sort()
print(f"{'n_gal':>9} {'sd(M0hat)':>10} "
      + " ".join(f"{k:>22}" for k in
                 ("Om0_hi", "w0_hi", "Om0lo_w0lo_walo")))
for n, sd, by in rows:
    cells = []
    for k in ("Om0_hi", "w0_hi", "Om0lo_w0lo_walo"):
        x = by[k]
        cells.append(f"{x['delta_M0hat']:+.4f} ({x['delta_over_laplace_sd']:+.1f}sd)")
    print(f"{n:>9} {sd:>10.5f} " + " ".join(f"{c:>22}" for c in cells))
