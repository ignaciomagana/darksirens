#!/bin/bash
#SBATCH -J pr3-gates
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr3/gates_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr3/gates_%j.err
# PR-0 item 3: guard-decomposed Q-on-at-anchor H0 oscillation (field_level_plan).
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_full259
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python ../field_level_plan/pr3/run_promotion_gates.py
