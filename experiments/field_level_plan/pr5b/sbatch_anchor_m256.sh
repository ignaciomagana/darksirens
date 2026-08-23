#!/bin/bash
#SBATCH -J pr5b-anchor256
#SBATCH -p TWIG-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/anchor256_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/anchor256_%j.err
# PR-5b: the M_draw=256 anchor PLAN 6.5 item 1 asks for, plus the MAP twin
# (one "member" = xi_hat) that P17 arm (b) needs.  TWIG-GPU rather than
# RITA-GPU: RITA is fully allocated (2/2 A100-80) and the PR-5 v2a anchor
# rebuild has been PD (Resources) there all day.
set -euo pipefail
REPO=/hildafs/projects/phy230014p/magana/src/darksirens-dev
cd $REPO/experiments/desi_full259
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=$REPO:$REPO/experiments/field_level_plan/pr5b
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_ZMAX=6.0
time python $REPO/experiments/field_level_plan/pr5b/build_anchor_m256.py \
    --m-draw 256 --moment-chunk 16 --time-dadb
