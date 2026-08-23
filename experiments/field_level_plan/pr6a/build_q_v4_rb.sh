#!/bin/bash
# Mask-free (v4-recipe) Q table for the rb spec-z closure world, to validate the
# f_p x Q pairing end to end against truth H0 = 67.74. The OLD data/rb/q_radial.h5
# was fit to observed counts with no depth map: it absorbed the footprint
# (on/off Q 1.624 vs 0.050, corr(Q, f_p) = +0.41) and paired with C_p = f_p C it
# put H0 at 41.24 [36.1, 46.3] (DIAGNOSIS.md "A FINDING AGAINST S-3 ITSELF").
# This build folds f_p in (--depth-map), pins Q = 1 above the mock's measured
# support (max observed z = 0.3847 -> --q-support-depth 0.39), and must EARN
# f_p_aware = True from the catalog-anchored verifier before validate_fp_q_v4.py
# is allowed to mean anything.
set -euo pipefail
cd "$(dirname "$0")"

export DARKSIRENS_ZMAX=1.5   # the pr6a world pin (world16.py)
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

LOG10N0=$(python -c "import json; print(json.load(open('data/rb/n0_calibration.json'))['log10n0'])")
DELTA=$(python -c "import json; print(json.load(open('data/rb/n0_calibration.json'))['delta'])")
mkdir -p logs

JAX_PLATFORMS=cpu python -m darksirens.cli.build_lognormal_completion \
  --catalog data/rb/catalog_pixelated_nside_16.h5 \
  --out data/rb/q_v4_maskfree.h5 \
  --c-mode selection --selection-fit data/rb/selection_fit.json \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --depth-map data/rb/mth_map_nside16.h5 \
  --q-support-depth 0.39 \
  --mode radial --workers 16 --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_v4_rb.log

echo "DONE"
