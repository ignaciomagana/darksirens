#!/bin/bash
#SBATCH -J ns-joint-sel-259
#SBATCH -p RITA-GPU
#SBATCH -A phy220048p
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 96:00:00
#SBATCH -o logs/ns_joint_sel_%j.out
#SBATCH -e logs/ns_joint_sel_%j.err
# FULL 259-event joint population + H0 + theta run, selection channel.
# The production construction for the unwindowed set: the fixed-pop grid
# degenerates into a spectral-siren pin (see data/h0_scans/DIAGNOSTIC_ONLY.md),
# so the population backbone is SAMPLED jointly. Budget (n0, delta) fixed at
# the 6.0-grid calibration; theta anchored by the magnitude fit; Om0 = 0.3089
# (the pair's pe cosmology). Checkpointed; requeue same command to resume.
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/desi_full259
export PATH=/hildafs/home/magana/.conda/envs/jax/bin:$PATH
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export DARKSIRENS_ZMAX=6.0 XLA_PYTHON_CLIENT_PREALLOCATE=false
FIXED=$(python - <<'PYEOF'
import json
cal = json.load(open("data/n0_calibration.json"))
print(json.dumps({"Om0": 0.3089, "sigma_kde": 0.003,
                  "log10n0": cal["log10n0"], "delta": cal["delta"]}))
PYEOF
)
echo "[ns-joint] fixed: $FIXED"
python -m darksirens.cli.inference \
  --gw_path /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/GWTC5_S1_spin_fullnuts/gwsamples_bbh_whitelist_all_events_final.h5 \
  --gwselection_path /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/GWTC5_S1_spin_fullnuts/selection_o3o4ab_allsky.h5 \
  --survey_path ../desi_ingest/data/pixelated_n64/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens \
  --pop_model gwtc5_fiducial_bpl2peaks \
  --c_mode selection \
  --selection_fit ../desi_ingest/data/selection_fit_union.json \
  --catalog_sky_weighting field \
  --use_lss false \
  --fix_population false \
  --fix_cosmology false \
  --fix_de true \
  --fix_survey false \
  --fixed_parameter_values "$FIXED" \
  --prior_overrides '{"H0": [20.0, 140.0]}' \
  --sampler tinyns \
  --nlive "${NLIVE:-1000}" \
  --tinyns_preset adaptive_gpu \
  --selection_neff_guard soft \
  --sel_batch_size 131072 \
  --pe_event_block 32 \
  --max_samples 2000000 \
  --checkpoint_interval 1800 \
  --resume auto \
  --save_path data/ns_joint_sel
