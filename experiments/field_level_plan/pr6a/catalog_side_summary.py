"""Read the catalog-side interventions against the SAME-SEED baseline.

``tier_c_catalog_side.py`` runs Tier C's own loop with one catalog-side term
perturbed at a time, from ``--seed0 7001`` -- the same seed block as
``tier_c_n100.json``.  So the baseline is not another run: it is the first
``n`` realizations of the n=100 pass, matched seed for seed, which removes the
realization noise from the comparison entirely.

What each column means is in ``tier_c_catalog_side.py``'s docstring.  What to
read here: the overconfidence ratio.  If an intervention moves it toward 1, the
term it perturbs is carrying the width deficit; if it moves the quoted ``sigma``
without moving the ratio, it is a knob on precision and not on calibration.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def verdict_row(v, arm):
    a = v["verdict"][arm]
    return dict(
        n=a["n_real"], overconfidence=a["overconfidence"],
        spread=a["spread_of_medians"], sigma=a["mean_quoted_sigma"],
        frac_in_90=a["frac_in_90"], bias=a["median_bias_sigma"],
        median_width90=a.get("median_width90"))


def recompute(rows, arm, H0_true=67.74):
    med = np.array([r[arm]["median"] for r in rows])
    sig = np.array([r[arm]["sigma"] for r in rows])
    u = np.array([r[arm]["cdf_at_truth"] for r in rows])
    return dict(n=len(rows), overconfidence=float(med.std(ddof=1) / sig.mean()),
                spread=float(med.std(ddof=1)), sigma=float(sig.mean()),
                frac_in_90=float(((u >= 0.05) & (u <= 0.95)).mean()),
                bias=float(np.median((med - H0_true) / sig)),
                median_width90=float(np.median(
                    [r[arm]["width90"] for r in rows])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="tier_c_n100.json")
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=path.json pairs")
    ap.add_argument("--out", default="catalog_side_summary.json")
    a = ap.parse_args(argv)

    base_all = json.load(open(a.baseline))
    out, table = {}, []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        d = json.load(open(path))
        seeds = [r["seed"] for r in d["rows"]]
        base_rows = [r for r in base_all["rows"] if r["seed"] in set(seeds)]
        matched = len(base_rows) == len(seeds)
        b = recompute(base_rows, a.arm) if base_rows else None
        g = recompute(d["rows"], a.arm)
        out[label] = dict(path=path, seeds_matched=matched,
                          n_seeds=len(seeds), baseline=b, run=g,
                          delta_overconfidence=(
                              g["overconfidence"] - b["overconfidence"]
                              if b else None))
        table.append((label, b, g, matched))

    print(f"{'run':>14} {'n':>4} {'oc base':>9} {'oc run':>9} {'d(oc)':>8} "
          f"{'sig base':>9} {'sig run':>9} {'spread run':>11} {'f90':>6}")
    for label, b, g, matched in table:
        if b is None:
            print(f"{label:>14} {g['n']:>4} {'--':>9} {g['overconfidence']:>9.3f}"
                  f"  (no matching baseline seeds)")
            continue
        print(f"{label:>14} {g['n']:>4} {b['overconfidence']:>9.3f} "
              f"{g['overconfidence']:>9.3f} "
              f"{g['overconfidence'] - b['overconfidence']:>+8.3f} "
              f"{b['sigma']:>9.3f} {g['sigma']:>9.3f} {g['spread']:>11.3f} "
              f"{g['frac_in_90']:>6.2f}"
              + ("" if matched else "   [SEEDS NOT FULLY MATCHED]"))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
