#!/usr/bin/env bash
set -euo pipefail

# Template for a realistic heavy dark-sirens TinyNS run. Required input files:
GW_PATH=${GW_PATH:?set GW_PATH}
GWSELECTION_PATH=${GWSELECTION_PATH:?set GWSELECTION_PATH}
SURVEY_PATH=${SURVEY_PATH:?set SURVEY_PATH}
SAVE_PATH=${SAVE_PATH:-./runs/tinyns_heavy_darksirens}

# These science choices are environment-controlled so users can validate them
# for each target. Keep USE_LSS=false with DROP_FULL_CATALOG=true; full catalogs
# are required for LSS-conditioned modes.
FIXED_COSMOLOGY=${FIXED_COSMOLOGY:-true}
FIX_SURVEY=${FIX_SURVEY:-true}
USE_LSS=${USE_LSS:-false}
DROP_FULL_CATALOG=${DROP_FULL_CATALOG:-true}
TINYNS_PRESET=${TINYNS_PRESET:-heavy_darksirens}
# For harder targets after validation, try: TINYNS_PRESET=heavy_darksirens_strong

if [[ "$USE_LSS" == "true" && "$DROP_FULL_CATALOG" == "true" ]]; then
  echo "ERROR: DROP_FULL_CATALOG=true is incompatible with USE_LSS=true." >&2
  echo "Set DROP_FULL_CATALOG=false or USE_LSS=false." >&2
  exit 2
fi

darksirens_inference \
  --gw_path "$GW_PATH" \
  --gwselection_path "$GWSELECTION_PATH" \
  --survey_path "$SURVEY_PATH" \
  --sampler tinyns \
  --tinyns_preset "$TINYNS_PRESET" \
  --universe_model dark_sirens \
  --pop_model brokenpowerlaw+2peaks \
  --fix_cosmology "$FIXED_COSMOLOGY" \
  --fix_survey "$FIX_SURVEY" \
  --use_lss "$USE_LSS" \
  --nlive "${NLIVE:-2000}" \
  --dlogz "${DLOGZ:-0.11}" \
  --max_samples "${MAX_SAMPLES:-0}" \
  --sel_batch_size "${SEL_BATCH_SIZE:-4096}" \
  --drop_full_catalog "$DROP_FULL_CATALOG" \
  --show_progress true \
  --save_path "$SAVE_PATH"
