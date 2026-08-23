#!/bin/bash
#SBATCH -J h0-full259-v2
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 96:00:00
#SBATCH -o logs/h0_scans_%j.out
#SBATCH -e logs/h0_scans_%j.err
# FULL-RANGE 259-event scans (owner-designated pair, ZMAX=6.0 line).
# Q-free configs first; selq_radial last, behind a wait for the 6.0-grid
# radial table (building on CPU at submit time). "complete" is omitted:
# all-sky events off the 62% footprint have zero support under
# dark_sirens_complete (structural, not a bug).
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_full259
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python run_h0_scans.py --configs per_pixel sel sel_strat
n=0
until [ -f data/fits/q_radial.h5 ]; do
  sleep 300; n=$((n+1))
  if [ $n -gt 96 ]; then echo "[h0-full259-v2] q_radial.h5 never appeared (8h); skipping selq_radial"; exit 0; fi
done
python run_h0_scans.py --configs selq_radial
