"""The f_p x Q closure validation: does the ADMITTED pairing recover truth?

Reruns Tier B's ``table`` and ``table_fp`` arms on the rb spec-z world with the
mask-free ``q_v4_maskfree.h5`` in place of the footprint-absorbing
``q_radial.h5``.  The old pairing measured H0 = 41.24 [36.1, 46.3] against a
truth of 67.74 (sz_tier_b_tablefp.json; DIAGNOSIS.md "A FINDING AGAINST S-3
ITSELF").  The loader now admits the pairing only on the artifact's EARNED
``f_p_aware`` stamp, so this run exercises the real gate, not an override.

Same grid as the record it answers: H0 in [20, 140] at 2 km/s (61 points).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

import world16 as W16  # noqa: F401  (side effect: pins DARKSIRENS_ZMAX=1.5)
import arms as A
import tier_b as TB

TRUTH = 67.74


def main():
    p = TB.paths_for("data/rb")
    p["q_table"] = Path("data/rb/q_v4_maskfree.h5")
    assert p["q_table"].exists(), "run build_q_v4_rb.sh first"
    grid = np.arange(20.0, 140.5, 2.0)
    cache = {}
    res = {}
    for arm in ("table", "table_fp"):
        logl, opts, data = A.build(p, arm, data_cache=cache)
        vals = A.scan_h0(logl, grid)
        s = A.summarize(grid, vals)
        res[arm] = s
        print(f"[{arm}] H0 = {s['median']:.2f} "
              f"90% [{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}] "
              f"sigma={s['sigma']:.2f} cdf(truth)={s['cdf_at_truth']:.4f}")

    old = json.load(open("sz_tier_b_tablefp.json"))
    res["_truth"] = TRUTH
    res["_h0"] = grid.tolist()
    res["_q_table"] = str(p["q_table"])
    res["_old_table_fp_median"] = old["table_fp"]["median"]
    res["_verdict"] = {
        "old_table_fp": old["table_fp"]["median"],
        "new_table_fp": res["table_fp"]["median"],
        "truth": TRUTH,
        "truth_in_new_90": bool(res["table_fp"]["ci90"][0] <= TRUTH
                                <= res["table_fp"]["ci90"][1]),
    }
    with open("sz_tier_b_tablefp_v4.json", "w") as fh:
        json.dump(res, fh, indent=1)
    v = res["_verdict"]
    print(f"[verdict] table_fp: {v['old_table_fp']:.2f} (mask double-counted) "
          f"-> {v['new_table_fp']:.2f} (mask-free Q, admitted by stamp); "
          f"truth {TRUTH} in 90% CI: {v['truth_in_new_90']}")


if __name__ == "__main__":
    main()
