#!/bin/bash
#SBATCH -J h0-full259
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/h0_full259_%j.out
#SBATCH -e logs/h0_full259_%j.err
# Beta-consistent 259-event grid scans at the recalibrated budget.
# Q-free configs first; selq_radial LAST behind a wait for the recal
# radial table (rebuilding on CPU at submit time).
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_ingest
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
export DARKSIRENS_ZMAX=0.75 PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python run_h0_full259.py --configs complete per_pixel sel sel_strat
n=0
until [ -f data/fits/q_radial.h5 ]; do
  sleep 300; n=$((n+1))
  if [ $n -gt 72 ]; then echo "[h0-full259] q_radial.h5 never appeared (6h); skipping selq_radial"; exit 0; fi
done
python run_h0_full259.py --configs selq_radial
