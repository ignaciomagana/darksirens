"""Could ANY ``s_b`` have closed Tier C?  Measure ``dCI/ds_b``, don't argue it.

S-2 implements PLAN §3.4's ``Cov(xi) = H^-1 + s_b^2 v v^T`` with ``s_b`` MEASURED
(the profile curvature, floored at 5% of ``b_gal``), so ``s_b`` is not a dial and
this script is not proposing to turn one.  It is answering a different and
falsifiable question: the closure's diagnosis was that the ensemble was
under-dispersed and that ``b_gal`` was the missing dispersion.  If that
diagnosis were right, then SOME value of ``s_b`` -- defensible or not -- would
reproduce the ~2.6x interval the coverage needs.  If no value does, the
diagnosis is refuted independently of what ``s_b`` actually is, and the
refutation does not rest on the floor, on the curvature, or on any choice made
in S-2.

So: rebuild the Tier-B anchor at ``s_b`` spanning several decades, run the same
``latent`` arm on the same mock and the same data object, and report the 90%
``H0`` width as a function of ``s_b``.  The forced values are labelled as forced
in the output; only the first row is the shipped ``s_b``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import world16 as W16
import arms as A
import build_anchor16
import tier_b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="data/rb")
    p.add_argument("--h0-step", type=float, default=1.0)
    p.add_argument("--s-b", nargs="*", type=float,
                   default=[0.0, 0.05, 0.2, 1.0, 5.0, 25.0])
    p.add_argument("--out", default="s_b_sensitivity.json")
    a = p.parse_args(argv)

    d = Path(a.dir)
    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    world = W16.build_world()
    cache = {}
    rows = []
    t0 = time.time()
    for s_b in a.s_b:
        # ``s_b_floor_frac`` is what forces the value: the floor only ever
        # RAISES s_b, so setting the floor to the target and keeping b_gal = 1
        # pins s_b = target for every target above the profile curvature
        # (4.432363e-02 on this anchor).  s_b = 0 is the feature switched off.
        info = build_anchor16.build(
            survey=d / "catalog_pixelated_nside_16.h5",
            mth_map=d / "mth_map_nside16.h5",
            out=d / "latent_anchor_bgal.h5", world=world, verbose=False,
            b_gal_dispersion=(s_b > 0.0),
            s_b_floor_frac=(s_b if s_b > 0.0 else None))
        got = float(info["s_b"])
        res = tier_b.run(d, grid=grid, arm_names=("latent_bgal",),
                         quiet=True)["latent_bgal"]
        row = dict(s_b_requested=float(s_b), s_b_used=got,
                   forced=bool(s_b > 0.0
                               and abs(got - s_b) < 1e-12
                               and s_b != 0.05),
                   floor_active=bool(info["s_b_floor_active"]),
                   s_b_profile=float(info["s_b_profile"]),
                   member_sd=float(info["member_sd"]),
                   median=res["median"], sigma=res["sigma"],
                   width90=res["width90"],
                   cdf_at_truth=res["cdf_at_truth"],
                   in_ci90=bool(res["ci90"][0] <= W16.H0_TRUE
                                <= res["ci90"][1]))
        rows.append(row)
        print(f"[s_b] {got:9.5f}  member_sd {row['member_sd']:8.5f}  "
              f"H0 {row['median']:7.3f}  sigma {row['sigma']:6.3f}  "
              f"w90 {row['width90']:7.3f}  cdf {row['cdf_at_truth']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(W16.PR6A_DIR / a.out, "w") as f:
            json.dump({"rows": rows, "dir": str(d)}, f, indent=1)

    # Leave the directory as Tier B found it: the forced anchors are scratch,
    # and ``data/rb/latent_anchor_bgal.h5`` must go back to the SHIPPED s_b so
    # ``tier_b.py --dir data/rb`` reproduces ``tier_b_rb_v2.json``.
    build_anchor16.build(
        survey=d / "catalog_pixelated_nside_16.h5",
        mth_map=d / "mth_map_nside16.h5",
        out=d / "latent_anchor_bgal.h5", world=world, verbose=False,
        b_gal_dispersion=True)

    base = rows[0]
    for r in rows:
        r["width90_ratio_vs_s_b_0"] = float(r["width90"] / base["width90"])
        r["member_sd_ratio_vs_s_b_0"] = float(r["member_sd"]
                                              / base["member_sd"])
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump({"rows": rows, "dir": str(d)}, f, indent=1)
    print(json.dumps(rows, indent=1))
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
