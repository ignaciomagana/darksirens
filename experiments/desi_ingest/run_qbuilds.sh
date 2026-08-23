#!/bin/bash
# Stage A Q-table builds on the real DESI union catalog (c_mode=selection).
# Radial first (forks workers -> CPU JAX), then gp3d (GPU, shared H100).
# Requires data/selection_fit_union.json and data/n0_calibration.json.
set -euo pipefail
cd "$(dirname "$0")"

export DARKSIRENS_ZMAX=0.75
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

CATALOG=data/pixelated_n64/catalog_pixelated_nside_64.h5
FIT=data/selection_fit_union.json
LOG10N0=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['log10n0'])")
DELTA=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['delta'])")
mkdir -p data/fits

echo "log10n0=$LOG10N0 delta=$DELTA"

JAX_PLATFORMS=cpu python -m darksirens.cli.build_lognormal_completion \
  --catalog "$CATALOG" --out data/fits/q_radial.h5 \
  --c-mode selection --selection-fit "$FIT" \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --mode radial --workers 16 --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_radial.log

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m darksirens.cli.build_lognormal_completion \
  --catalog "$CATALOG" --out data/fits/q_gp3d.h5 \
  --c-mode selection --selection-fit "$FIT" \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --mode gp3d --gp3d-nz-nodes 12 --gp3d-z-node-hi 0.30 --gp3d-nsph-nodes 64 \
  --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_gp3d.log

echo "DONE"
