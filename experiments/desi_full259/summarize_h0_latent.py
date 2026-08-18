"""Read h0_latent_scans.json and report the arm-by-arm decomposition.

The comparison that matters is NOT latent-vs-nofp.  PR-6a measured that 97.2%
of latent mode's runtime overhead and 91% of its Tier-B shift is the ``f_p``
channel, so nofp -> latent credits the field with PR-2's effect.  The
decomposition below separates them:

    nofp -> fp        the f_p channel      (PR-2)
    fp   -> latent    the field itself     (PR-5/PR-6a, the deliverable)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(path):
    res = json.load(open(path))["results"]
    order = [a for a in ("nofp", "fp", "latent") if a in res]

    print(f"{'arm':8} {'H0 median':>10} {'90% CI':>18} {'width':>7} "
          f"{'finite':>8} {'ms/eval':>8}")
    print("-" * 66)
    for a in order:
        r = res[a]
        if r.get("all_nonfinite"):
            print(f"{a:8} {'ALL -inf':>10}")
            continue
        ci = r["ci90"]
        print(f"{a:8} {r['median']:10.2f} "
              f"[{ci[0]:7.2f},{ci[1]:7.2f}] {ci[1]-ci[0]:7.2f} "
              f"{r['n_finite']:4d}/{len(r['h0']):3d} {r['ms_per_eval']:8.0f}")

    print()
    def shift(a, b, label):
        if a not in res or b not in res:
            return
        ra, rb = res[a], res[b]
        if ra.get("all_nonfinite") or rb.get("all_nonfinite"):
            return
        dm = rb["median"] - ra["median"]
        wa = ra["ci90"][1] - ra["ci90"][0]
        wb = rb["ci90"][1] - rb["ci90"][0]
        # express the shift in units of the WIDER arm's 90% half-width, so a
        # "sigma" here is conservative rather than flattering
        half = max(wa, wb) / 2.0 / 1.645
        print(f"{label:26} dH0 = {dm:+8.2f}  ({dm/half:+6.2f} sigma)   "
              f"width {wa:6.2f} -> {wb:6.2f}  ({wb/wa:5.3f}x)")

    shift("nofp", "fp", "f_p channel (PR-2)")
    shift("fp", "latent", "the FIELD (PR-5/6a)")
    shift("nofp", "latent", "both together")

    if "fp" in res and "latent" in res and not res["fp"].get("all_nonfinite"):
        d = abs(res["latent"]["median"] - res["fp"]["median"])
        t = abs(res["fp"]["median"] - res["nofp"]["median"]) if "nofp" in res else None
        if t:
            print(f"\nfield / f_p effect ratio: {d/t:.4f}  "
                  f"({100*d/t:.2f}% of the shift is the field)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "data/h0_latent/h0_latent_scans.json")
