"""Recompute the law-of-total-variance decomposition from ``variance_split``'s
raw cells.

Kept separate from ``variance_split.py`` so the campaign (~1 s of arithmetic on
top of ~25 minutes of GPU) never has to be re-run to change how it is read, and
so the decomposition is auditable against the cell table it came from.

    Var(H0) = E_cat[ Var_evt(H0 | cat) ]  +  Var_cat( E_evt[H0 | cat] )

The second term is debiased by ``within/n_event``: each row mean is itself an
average of ``n_event`` noisy cells.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="variance_split.json")
    p.add_argument("--h0-true", type=float, default=67.74)
    a = p.parse_args(argv)
    d = json.load(open(a.inp))
    cells = d["cells"]
    ns = max(c["i"] for c in cells) + 1
    ne = max(c["j"] for c in cells) + 1
    print(f"{len(cells)} cells, {ns} catalogs x {ne} event sets "
          f"({'complete' if len(cells) == ns * ne else 'PARTIAL'} grid)\n")
    out = {}
    for arm in d["arms"]:
        M = np.full((ns, ne), np.nan)
        S = np.full((ns, ne), np.nan)
        for c in cells:
            M[c["i"], c["j"]] = c[arm]["median"]
            S[c["i"], c["j"]] = c[arm]["sigma"]
        ok = np.isfinite(M).all(axis=1)
        M, S = M[ok], S[ok]
        n_row = M.shape[0]
        v_within = float(np.mean(np.var(M, axis=1, ddof=1)))
        v_total = float(np.var(M, ddof=1))
        row_mean = M.mean(axis=1)
        v_between_raw = float(np.var(row_mean, ddof=1))
        v_cat = v_between_raw - v_within / ne
        sg = float(np.mean(S))
        r = dict(
            n_catalogs=n_row, n_event_sets=ne, n_cells=int(M.size),
            mean_quoted_sigma=sg,
            total_spread=float(v_total ** 0.5),
            overconfidence_total=float(v_total ** 0.5 / sg),
            event_spread_within_catalog=float(v_within ** 0.5),
            overconfidence_events_only=float(v_within ** 0.5 / sg),
            catalog_spread_raw=float(v_between_raw ** 0.5),
            catalog_spread_debiased=(float(v_cat ** 0.5) if v_cat > 0
                                     else -float((-v_cat) ** 0.5)),
            frac_variance_from_events=float(v_within / v_total),
            grand_mean=float(M.mean()), H0_true=float(a.h0_true),
            per_catalog_mean=row_mean.tolist(),
            per_catalog_event_sd=np.std(M, axis=1, ddof=1).tolist(),
            medians=M.tolist())
        out[arm] = r
        print(f"--- {arm} ---")
        print(f"  mean quoted sigma                 {sg:8.4f} km/s")
        print(f"  TOTAL spread of medians           {r['total_spread']:8.4f}"
              f"   -> overconfidence {r['overconfidence_total']:6.3f}")
        print(f"  EVENT spread within one catalog   "
              f"{r['event_spread_within_catalog']:8.4f}"
              f"   -> overconfidence {r['overconfidence_events_only']:6.3f}")
        print(f"  CATALOG common mode (debiased)    "
              f"{r['catalog_spread_debiased']:8.4f}")
        print(f"  fraction of variance from events  "
              f"{r['frac_variance_from_events']:8.4f}")
        print(f"  per-catalog means   "
              f"{np.round(row_mean, 2).tolist()}")
        print(f"  per-catalog event sd "
              f"{np.round(np.std(M, axis=1, ddof=1), 2).tolist()}\n")
    with open(a.inp.replace(".json", "_decomposition.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {a.inp.replace('.json', '_decomposition.json')}")


if __name__ == "__main__":
    main()
