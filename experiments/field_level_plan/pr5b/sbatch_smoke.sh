#!/bin/bash
#SBATCH -J pr5b-smoke
#SBATCH -p TWIG-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/smoke_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/smoke_%j.err
# PR-5b gate 0: the latent seam on the REAL production line, M_draw=8.
# TWIG-GPU rather than RITA-GPU: RITA is fully allocated (2/2 A100-80 in use)
# and the PR-5 anchor rebuild has been PD (Resources) there since 08:30.
set -euo pipefail
REPO=/hildafs/projects/phy230014p/magana/src/darksirens-dev
cd $REPO/experiments/desi_full259
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=$REPO:$REPO/experiments/field_level_plan/pr5b
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python $REPO/experiments/field_level_plan/pr5b/smoke_latent.py
