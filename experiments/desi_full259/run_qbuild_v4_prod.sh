#!/bin/bash
# Mask-free (v4) Q table on the PRODUCTION grid (DARKSIRENS_ZMAX=6.0, 1086
# nodes) -- the desi_ingest q_v4_depthmap.h5 is the same recipe on the 0.75
# ingest grid and the loader rightly refuses the grid mismatch. Same catalog
# (nside-64 union), same selection fit, this experiment's n0 calibration,
# --depth-map + --q-support-depth 0.30 (the catalog's measured truncation:
# max observed z = 0.3000 exactly). Replaces q_radial.h5 for the S-3 arms; the
# earned f_p_aware stamp is what admits the q_fp pairing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$HERE"
while [ ! -f "$REPO_ROOT/setup.py" ] && [ "$REPO_ROOT" != / ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"; done
. "$REPO_ROOT/experiments/py_env.sh"
cd "$HERE"

export DARKSIRENS_ZMAX=6.0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

CATALOG=../desi_ingest/data/pixelated_n64/catalog_pixelated_nside_64.h5
FIT=../desi_ingest/data/selection_fit_union.json
DEPTH=../desi_ingest/data/mth_map_nside128.h5
OUT=data/fits/q_v4_depthmap_prod.h5
LOG10N0=$("$PYTHON" -c "import json; print(json.load(open('data/n0_calibration.json'))['log10n0'])")
DELTA=$("$PYTHON" -c "import json; print(json.load(open('data/n0_calibration.json'))['delta'])")
mkdir -p data/fits logs

echo "log10n0=$LOG10N0 delta=$DELTA q_support_depth=0.30 ZMAX=6.0"

JAX_PLATFORMS=cpu "$PYTHON" -m darksirens.cli.build_lognormal_completion \
  --catalog "$CATALOG" --out "$OUT" \
  --c-mode selection --selection-fit "$FIT" \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --depth-map "$DEPTH" \
  --q-support-depth 0.30 \
  --mode radial --workers 16 --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_v4_prod.log

echo "DONE"
