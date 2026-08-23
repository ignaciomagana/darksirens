#!/bin/bash
#SBATCH -J pr5b-campaign
#SBATCH -p TWIG-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/campaign_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/campaign_%j.err
# PR-5b item 1: the ll_m matrix at M_draw = 256 over 33 H0 nodes in [20, 140]
# plus the anchor node 67.74, on the production 259-event line.
# Budget: PR-0's measured 3027 ms/eval x (34 nodes x [8 member-chunks + M=8
# arm + MAP arm]) ~= 20 min of evaluation; -t 08:00:00 is deliberate slack.
set -euo pipefail
REPO=/hildafs/projects/phy230014p/magana/src/darksirens-dev
cd $REPO/experiments/desi_full259
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=$REPO:$REPO/experiments/field_level_plan/pr5b
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_ZMAX=6.0
time python $REPO/experiments/field_level_plan/pr5b/run_member_spread.py \
    --chunk 32
