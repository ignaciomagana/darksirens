#!/bin/bash
# The v2 MASK-FREE Q build (--depth-map), scripted so it survives a session.
#
# v1 (2026-08-20, commit 54de239 era) was invoked ad hoc, died with the session,
# and left its failure numbers only as prose in experiments/CHECKPOINT.md:
#   off-footprint logQ mean -0.073 sd 0.5694 ; corr(Q, f_p) +0.392 / +0.333 /
#   -0.995 at z slices 150 / 250 / 400.
# The two fixes measured here (commit 2e594fb): off-footprint pixels are
# EXCLUDED from the fit (zero-init keeps logQ = 0), and f_p_aware is EARNED by
# _verify_mask_free on the finished table rather than asserted by the flag.
#
# The open question this run answers: is exclusion sufficient, or does Q still
# correlate with f_p through the per-z budget renormalization alone?
#
# radial only: --mode gp3d and stratified selection REFUSE --depth-map (they
# never fold f_p into the fit, so the stamp would be unearned).  The radial
# builder forks workers, hence CPU JAX.  Empty covered pixels are deduped PER
# f_p VALUE (each f_p is its own N_obs = 0 fit problem).
set -euo pipefail
cd "$(dirname "$0")"

export DARKSIRENS_ZMAX=0.75
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

CATALOG=data/pixelated_n64/catalog_pixelated_nside_64.h5
FIT=data/selection_fit_union.json
DEPTH=data/mth_map_nside128.h5
OUT=data/fits/q_v2_depthmap.h5
LOG10N0=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['log10n0'])")
DELTA=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['delta'])")
mkdir -p data/fits logs

echo "log10n0=$LOG10N0 delta=$DELTA"

JAX_PLATFORMS=cpu python -m darksirens.cli.build_lognormal_completion \
  --catalog "$CATALOG" --out "$OUT" \
  --c-mode selection --selection-fit "$FIT" \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --depth-map "$DEPTH" \
  --mode radial --workers 16 --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_v2_depthmap.log

# Measure the artifact independently of the builder's own stamp, and leave a
# machine-readable record beside it (v1 left none, which is why its numbers
# are prose).  DO NOT trust the stamp without reading this report.
JAX_PLATFORMS=cpu python measure_maskfree_v2.py "$OUT" "$DEPTH" "$CATALOG" \
  --json data/fits/q_v2_depthmap_maskfree.json \
  2>&1 | tee -a logs/qbuild_v2_depthmap.log

echo "DONE"
