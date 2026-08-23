#!/bin/bash
#SBATCH -J qbuild-gp3d
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/qbuild_gp3d_recal_%j.out
#SBATCH -e logs/qbuild_gp3d_recal_%j.err
# gp3d Q rebuild at the RECALIBRATED n0 (task: Q rebuilds under evo-fix).
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_ingest
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
export DARKSIRENS_ZMAX=0.75 PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LOG10N0=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['log10n0'])")
DELTA=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['delta'])")
python -m darksirens.cli.build_lognormal_completion \
  --catalog data/pixelated_n64/catalog_pixelated_nside_64.h5 \
  --out data/fits/q_gp3d.h5 \
  --c-mode selection --selection-fit data/selection_fit_union.json \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --mode gp3d --gp3d-nz-nodes 27 --gp3d-z-node-hi 0.30 --gp3d-nsph-nodes 64 \
  --n-members 8 --seed 22
