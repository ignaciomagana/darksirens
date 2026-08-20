#!/bin/bash
#SBATCH -J ns-sampled-theta
#SBATCH -p RITA-GPU
#SBATCH -A phy230014p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/ns_sampled_theta_%j.out
#SBATCH -e logs/ns_sampled_theta_%j.err
# Sampled-theta NS production run (see run_ns_sampled_theta.sh for the
# config rationale).  Checkpointed every 1800 s; requeue the identical
# command with the same OUT dir to resume (--resume auto).
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_ingest
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
exec bash run_ns_sampled_theta.sh "${1:-data/ns_sampled_theta}"
