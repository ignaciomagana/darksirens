#!/bin/bash
#SBATCH -J ns-resume
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 168:00:00
#SBATCH -o logs/ns_resume_%j.out
#SBATCH -e logs/ns_resume_%j.err
# Resume the sampled-theta NS run from its checkpoint if the 4h-capped job
# (1118731) timed out; exit clean if it already finished.
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_ingest
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
if ls data/ns_sampled_theta/*/results.hdf5 >/dev/null 2>&1; then
  echo "[ns-resume] results.hdf5 already present; nothing to do."; exit 0
fi
exec bash run_ns_sampled_theta.sh data/ns_sampled_theta
