#!/usr/bin/env bash
# The R14 overlap arm, one process per overlap fraction (same JIT reason as
# run_tier_e.sh).
set -eu
cd "$(dirname "$0")/../../.."
export JAX_PLATFORMS=cpu DARKSIRENS_ZMAX=1.5 PYTHONPATH=.
D=experiments/field_level_plan/pr7
for phi in 0.0 0.05 0.10 0.25 0.50; do
  python $D/tier_e_overlap.py --overlap $phi --n-real 8 \
      --out $D/tier_e_overlap_$phi.json 2>&1 | tee $D/tier_e_overlap_$phi.log
done
python - <<'PY'
import glob, json, numpy as np
rows=[]
for p in sorted(glob.glob("experiments/field_level_plan/pr7/tier_e_overlap_*.json")):
    rows += json.load(open(p))["rows"]
out=[]
for phi in sorted({r["overlap"] for r in rows}):
    sub=[r for r in rows if r["overlap"]==phi]
    pl=np.array([r["pull"] for r in sub])
    out.append(dict(overlap=phi, n=len(sub),
                    ratio_mean=float(np.mean([r["ratio"] for r in sub])),
                    ratio_sd=float(np.std([r["ratio"] for r in sub], ddof=1)),
                    sigma_mean=float(np.mean([r["sigma"] for r in sub])),
                    pull_mean=float(pl.mean()), pull_sd=float(pl.std(ddof=1)),
                    within_2sigma=int(np.sum(np.abs(pl)<2))))
print(json.dumps(out, indent=2))
json.dump(dict(summary=out, rows=rows),
          open("experiments/field_level_plan/pr7/tier_e_overlap.json","w"),
          indent=1)
PY
