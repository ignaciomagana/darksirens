#!/bin/bash
#SBATCH -J s3-footprint
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/s3_footprint_%j.out
#SBATCH -e logs/s3_footprint_%j.err
# S-3 measured on the shipped Q-table line: selq_radial with and without the
# per-pixel selection fraction.  The masked arm is only runnable at all with
# the f_p-weighted empty-pixel Q budget.
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_full259
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python run_s3_footprint.py --h0-step 2.0 --arms q_nofp q_fp
