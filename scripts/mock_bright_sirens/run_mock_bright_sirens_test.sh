#!/usr/bin/env bash
# Generate a realistic multi-event bright-siren mock data set and validate that
# it can be ingested by the inference pipeline.  This intentionally lives
# outside scripts/mock_dark_sirens so the working dark-siren mock workflow is unchanged.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${OUTDIR:-${ROOT_DIR}/data/mock_bright_sirens_test}"
SEED="${SEED:-5678}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

N0="${N0:-1e-3}"
ZMAX="${ZMAX:-0.08}"
SURVEY_Z50="${SURVEY_Z50:-0.75}"
SURVEY_WIDTH="${SURVEY_WIDTH:-0.12}"
GALAXY_DENSITY_DELTA="${GALAXY_DENSITY_DELTA:-0.0}"
W0="${W0:--1.0}"
WA="${WA:-0.0}"

NOBS="${NOBS:-10}"
NSAMP="${NSAMP:-512}"
NDRAW="${NDRAW:-100000}"
SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-50000}"
SELECTION_PER_OBSERVATION_FACTOR="${SELECTION_PER_OBSERVATION_FACTOR:-}"
SELECTION_TARGET_DETECTIONS="${SELECTION_TARGET_DETECTIONS:-}"

COUNTERPART_NSIDE="${COUNTERPART_NSIDE:-1}"
COUNTERPART_DZ="${COUNTERPART_DZ:-1e-4}"
BRIGHT_SIREN_SKY_MARGINALIZED="${BRIGHT_SIREN_SKY_MARGINALIZED:-True}"

INFERENCE_NLIVE="${INFERENCE_NLIVE:-200}"
INFERENCE_DLOGZ="${INFERENCE_DLOGZ:-0.1}"
INFERENCE_MAX_SAMPLES="${INFERENCE_MAX_SAMPLES:-0}"
INFERENCE_SEL_BATCH_SIZE="${INFERENCE_SEL_BATCH_SIZE:-256}"

# Measurement family (all-observable): every width is set at the SNR threshold
# and scales as 1/rho_obs.  The old truth-scaled dL/m1det/m2det knobs are gone.
SNR_UNCERTAINTY="${SNR_UNCERTAINTY:-1.0}"
LNMC_UNCERTAINTY="${LNMC_UNCERTAINTY:-0.08}"
LNQ_UNCERTAINTY="${LNQ_UNCERTAINTY:-0.60}"
CHIEFF_UNCERTAINTY="${CHIEFF_UNCERTAINTY:-0.20}"

cd "${ROOT_DIR}"
mkdir -p "${OUTDIR}"

cat <<EOF
Starting bright-siren mock data validation.
  ROOT_DIR=${ROOT_DIR}
  OUTDIR=${OUTDIR}
  SEED=${SEED}
  N0=${N0}
  ZMAX=${ZMAX}
  NOBS=${NOBS}
  NSAMP=${NSAMP}
  NDRAW=${NDRAW}
  COUNTERPART_NSIDE=${COUNTERPART_NSIDE}
  BRIGHT_SIREN_SKY_MARGINALIZED=${BRIGHT_SIREN_SKY_MARGINALIZED}
  RUN_INFERENCE=${RUN_INFERENCE}
EOF

selection_target_args=""
if [ -n "${SELECTION_TARGET_DETECTIONS}" ]; then
  selection_target_args="--selection-target-detections ${SELECTION_TARGET_DETECTIONS}"
elif [ -n "${SELECTION_PER_OBSERVATION_FACTOR}" ]; then
  selection_target_args="--selection-per-observation-factor ${SELECTION_PER_OBSERVATION_FACTOR}"
fi

# shellcheck disable=SC2086
python scripts/mock_bright_sirens/generate_mock_bright_sirens.py \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --n0 "${N0}" \
  --zmax "${ZMAX}" \
  --survey-z50 "${SURVEY_Z50}" \
  --survey-width "${SURVEY_WIDTH}" \
  --galaxy-density-delta "${GALAXY_DENSITY_DELTA}" \
  --w0 "${W0}" \
  --wa "${WA}" \
  --nobs "${NOBS}" \
  --nsamp "${NSAMP}" \
  --ndraw "${NDRAW}" \
  --selection-batch-size "${SELECTION_BATCH_SIZE}" \
  ${selection_target_args} \
  --snr-uncertainty "${SNR_UNCERTAINTY}" \
  --lnmc-uncertainty "${LNMC_UNCERTAINTY}" \
  --lnq-uncertainty "${LNQ_UNCERTAINTY}" \
  --chieff-uncertainty "${CHIEFF_UNCERTAINTY}" \
  --counterpart-dz "${COUNTERPART_DZ}" \
  --verbose

export OUTDIR NOBS NSAMP COUNTERPART_NSIDE BRIGHT_SIREN_SKY_MARGINALIZED
COUNTERPART_ARGS="$(python - <<'PY'
import json, os
from pathlib import Path
items = json.loads((Path(os.environ["OUTDIR"]) / "bright_counterparts.json").read_text())["counterparts"]
print(" ".join(f"{c['ra_rad']} {c['dec_rad']} {c['z']}" for c in items))
PY
)"
export COUNTERPART_ARGS

echo "Starting bright-siren ingestion validation for generated products."
python - <<'PY'
from argparse import Namespace
from pathlib import Path
import json
import multiprocessing as mp
import os
import numpy as np

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
from darksirens.inference.data import load_all_data

out = Path(os.environ["OUTDIR"])
items = json.loads((out / "bright_counterparts.json").read_text())["counterparts"]
counterparts = tuple((c["ra_rad"], c["dec_rad"], c["z"]) for c in items)
opts = Namespace(
    universe_model="bright_sirens",
    survey_path=None,
    gw_path=str(out / "mock_bright_gw_events.h5"),
    gwselection_path=str(out / "mock_bright_gw_selection.h5"),
    use_LSS=False,
    counterpart=counterparts,
    counterpart_nside=int(os.environ["COUNTERPART_NSIDE"]),
    counterpart_dz=items[0]["counterpart_dz"],
    bright_siren_sky_marginalized=os.environ["BRIGHT_SIREN_SKY_MARGINALIZED"].lower() in {"true", "1", "yes"},
)
data = load_all_data(opts)
assert int(data["nEvents"]) == int(os.environ["NOBS"]), data["nEvents"]
assert int(data["nsamp"]) == int(os.environ["NSAMP"]), data["nsamp"]
assert len(data["counterpart_zs"]) == int(os.environ["NOBS"])
assert np.isfinite(np.asarray(data["p_draw"])).all(), "non-finite pdraw values"
assert len(data["p_draw"]) > 5 * int(os.environ["NOBS"]), "too few joint GW+EM detected selection samples"
print("Bright-siren ingestion validation passed.")
PY

if [ "${RUN_INFERENCE}" = "1" ]; then
  echo "Starting optional bright-siren darksirens_inference sampler run."
  # shellcheck disable=SC2086
  python -m darksirens.cli.inference \
    --gw_path "${OUTDIR}/mock_bright_gw_events.h5" \
    --gwselection_path "${OUTDIR}/mock_bright_gw_selection.h5" \
    --sampler dynesty \
    --pop_model powerlaw+peak \
    --universe_model bright_sirens \
    --counterpart ${COUNTERPART_ARGS} \
    --counterpart_dz "${COUNTERPART_DZ}" \
    --counterpart_nside "${COUNTERPART_NSIDE}" \
    --bright_siren_sky_marginalized "${BRIGHT_SIREN_SKY_MARGINALIZED}" \
    --fix_population False \
    --fix_cosmology False \
    --fix_survey True \
    --nlive "${INFERENCE_NLIVE}" \
    --dlogz "${INFERENCE_DLOGZ}" \
    --max_samples "${INFERENCE_MAX_SAMPLES}" \
    --sel_batch_size "${INFERENCE_SEL_BATCH_SIZE}" \
    --seed "${SEED}" \
    --show_progress True \
    --save_path "${OUTDIR}/inference_bright_sirens"
fi

cat <<EOF
Bright-siren mock validation complete.
Products are in: ${OUTDIR}
Set RUN_INFERENCE=1 to run the optional multi-event bright-siren sampler.
EOF
