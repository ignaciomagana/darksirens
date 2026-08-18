#!/bin/bash
#SBATCH -J pr8-amp-scan
#SBATCH -p TWIG-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr8/scan_%j.out
#SBATCH -e /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr8/scan_%j.err
# PR-8's deliverable: H0 vs the ASSUMED amp(z > z_depth) on the pr6a nside-16
# realization (seed 7001, 60 events).  MOCK SCALE ONLY -- the 259-event
# production posterior is HELD by the owner and is not launched here.
# Budget: 8 arms x 121 H0 nodes; measured 8.4 s/eval on CPU, ~0.1 s/eval on the
# H100, so this is minutes of evaluation with hours of slack.
set -euo pipefail
REPO=/hildafs/projects/phy230014p/magana/src/darksirens-dev
cd $REPO/experiments/field_level_plan/pr8
export PATH=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=$REPO:$REPO/experiments/field_level_plan/pr6a:$REPO/experiments/field_level_plan/pr8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# world16.py pins DARKSIRENS_ZMAX = 1.5 on import and REFUSES a conflicting
# value; the anchors' z_sub is that grid, so it is set here explicitly rather
# than inherited.
export DARKSIRENS_ZMAX=1.5
time python scan_amp.py --out $REPO/experiments/field_level_plan/pr8/scan_amp.json
