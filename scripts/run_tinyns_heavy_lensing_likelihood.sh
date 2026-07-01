#!/usr/bin/env bash
set -euo pipefail

GW_PATH=${GW_PATH:?set GW_PATH}
GWSELECTION_PATH=${GWSELECTION_PATH:?set GWSELECTION_PATH}
LENSED_INJECTIONS_PATH=${LENSED_INJECTIONS_PATH:?set LENSED_INJECTIONS_PATH}
PAIR_PE_PATH=${PAIR_PE_PATH:?set PAIR_PE_PATH}
PARTITION_PATH=${PARTITION_PATH:?set PARTITION_PATH}
SAVE_PATH=${SAVE_PATH:-./runs/tinyns_heavy_lensing}
TINYNS_PRESET=${TINYNS_PRESET:-heavy_darksirens}
# For harder targets after validation, try: TINYNS_PRESET=heavy_darksirens_strong

darksirens_inference_lensing \
  --gw_path "$GW_PATH" \
  --gwselection_path "$GWSELECTION_PATH" \
  --lensed_injections_path "$LENSED_INJECTIONS_PATH" \
  --pair_pe_path "$PAIR_PE_PATH" \
  --partition_path "$PARTITION_PATH" \
  --cluster_mode "${CLUSTER_MODE:-j2}" \
  --wl_backend "${WL_BACKEND:-lognormal}" \
  --sampler tinyns \
  --tinyns_preset "$TINYNS_PRESET" \
  --nlive "${NLIVE:-2000}" \
  --dlogz "${DLOGZ:-0.11}" \
  --max_samples "${MAX_SAMPLES:-0}" \
  --pe_max_per_pair "${PE_MAX_PER_PAIR:-400}" \
  --show_progress \
  --save_path "$SAVE_PATH"
