#!/bin/bash
#SBATCH -J pr4-anchor
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr4/anchor_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr4/anchor_%j.err
# PR-4: build the latent anchor artifact TWICE at the same seed (sha gate).
set -euo pipefail
REPO=/hildafs/projects/phy230014p/magana/src/darksirens-dev
cd $REPO/experiments/field_level_plan/pr4
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=$REPO
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_ZMAX=6.0
INGEST=$REPO/experiments/desi_ingest/data
ARGS="--survey $INGEST/pixelated_n64/catalog_pixelated_nside_64.h5 \
  --selection-fit $INGEST/selection_fit_union.json \
  --n0-calibration $REPO/experiments/desi_full259/data/n0_calibration.json \
  --per-pixel-completeness $INGEST/mth_map_nside128.h5 \
  --om0 0.3089 --seed 22"
time python $REPO/darksirens/cli/build_latent_field.py $ARGS --out latent_anchor_a.h5
time python $REPO/darksirens/cli/build_latent_field.py $ARGS --out latent_anchor_b.h5
python - <<'EOF'
import h5py
a = h5py.File('latent_anchor_a.h5')['latent_field'].attrs['sha256']
b = h5py.File('latent_anchor_b.h5')['latent_field'].attrs['sha256']
print('sha a:', a); print('sha b:', b)
print('REPRODUCIBLE' if a == b else 'MISMATCH')
EOF
