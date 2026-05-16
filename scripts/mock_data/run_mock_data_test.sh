#!/usr/bin/env sh
# Generate a realistic low-redshift mock dark-sirens data set and verify that
# the inference data loaders can ingest it. Set RUN_INFERENCE=1 to launch a
# production-style dark-sirens sampler run using survey hyperparameters matched
# to the generated catalog while leaving cosmology free.
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${OUTDIR:-${ROOT_DIR}/data/mock_dark_sirens_test}"
SEED="${SEED:-1234}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

# Keep validation subprocesses deterministic on shared CPU machines and avoid
# fork-after-JAX deadlocks when libraries create worker processes after JAX has
# initialized its runtime. Users can still override these before invoking this
# script.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# Survey-generation defaults.  N0=1e-3 Mpc^-3 is intentionally inside the
# default inference prior log10n0 ∈ [-4, -1].  The low zmax keeps the fixture
# lightweight while still using a physically meaningful density normalization.
N0="${N0:-1e-3}"
ZMAX="${ZMAX:-0.3}"
SURVEY_Z50="${SURVEY_Z50:-0.75}"
SURVEY_WIDTH="${SURVEY_WIDTH:-0.12}"
GALAXY_DENSITY_DELTA="${GALAXY_DENSITY_DELTA:-0.0}"
LOG10N0="$(python -c 'import math, sys; print(math.log10(float(sys.argv[1])))' "${N0}")"

# Mock size/performance knobs.
NSIDE="${NSIDE:-128}"
NOBS="${NOBS:-100}"
NSAMP="${NSAMP:-128}"
NDRAW="${NDRAW:-5000000}"
SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-500000}"
SELECTION_PER_OBSERVATION_FACTOR="${SELECTION_PER_OBSERVATION_FACTOR:-}"
SELECTION_TARGET_DETECTIONS="${SELECTION_TARGET_DETECTIONS:-}"

# Optional sampler-run knobs.  The selection likelihood is batched by default
# so even large generated selection files do not have to be materialized as one
# XLA operation on GPU.  By default the sampler is not call-capped
# (INFERENCE_MAX_SAMPLES=0); set a positive value only for local debugging.
INFERENCE_NLIVE="${INFERENCE_NLIVE:-200}"
INFERENCE_DLOGZ="${INFERENCE_DLOGZ:-0.1}"
INFERENCE_MAX_SAMPLES="${INFERENCE_MAX_SAMPLES:-0}"
INFERENCE_SEL_BATCH_SIZE="${INFERENCE_SEL_BATCH_SIZE:-256}"

# Fractional/absolute widths used to generate mock GW PE samples.
DL_FRAC_UNCERTAINTY="${DL_FRAC_UNCERTAINTY:-0.03}"
M1DET_FRAC_UNCERTAINTY="${M1DET_FRAC_UNCERTAINTY:-0.03}"
M2DET_FRAC_UNCERTAINTY="${M2DET_FRAC_UNCERTAINTY:-0.15}"
CHIEFF_UNCERTAINTY="${CHIEFF_UNCERTAINTY:-0.08}"
SKY_UNCERTAINTY_DEG="${SKY_UNCERTAINTY_DEG:-0.5}"

FIXED_SURVEY_JSON="${FIXED_SURVEY_JSON:-{\"log10n0\": ${LOG10N0}, \"z50\": ${SURVEY_Z50}, \"w\": ${SURVEY_WIDTH}, \"delta\": ${GALAXY_DENSITY_DELTA}, \"b_miss\": 1.0, \"alpha_miss\": 0.5}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTDIR}"

cat <<EOF
Starting verbose mock data validation.
  ROOT_DIR=${ROOT_DIR}
  OUTDIR=${OUTDIR}
  SEED=${SEED}
  N0=${N0}
  ZMAX=${ZMAX}
  SURVEY_Z50=${SURVEY_Z50}
  SURVEY_WIDTH=${SURVEY_WIDTH}
  GALAXY_DENSITY_DELTA=${GALAXY_DENSITY_DELTA}
  NSIDE=${NSIDE}
  NOBS=${NOBS}
  NSAMP=${NSAMP}
  NDRAW=${NDRAW}
  SELECTION_BATCH_SIZE=${SELECTION_BATCH_SIZE}
  SELECTION_PER_OBSERVATION_FACTOR=${SELECTION_PER_OBSERVATION_FACTOR:-<disabled>}
  SELECTION_TARGET_DETECTIONS=${SELECTION_TARGET_DETECTIONS:-<disabled>}
  RUN_INFERENCE=${RUN_INFERENCE}
  INFERENCE_NLIVE=${INFERENCE_NLIVE}
  INFERENCE_DLOGZ=${INFERENCE_DLOGZ}
  INFERENCE_MAX_SAMPLES=${INFERENCE_MAX_SAMPLES}
  INFERENCE_SEL_BATCH_SIZE=${INFERENCE_SEL_BATCH_SIZE}
EOF

selection_target_args=""
if [ -n "${SELECTION_TARGET_DETECTIONS}" ]; then
  selection_target_args="--selection-target-detections ${SELECTION_TARGET_DETECTIONS}"
elif [ -n "${SELECTION_PER_OBSERVATION_FACTOR}" ]; then
  selection_target_args="--selection-per-observation-factor ${SELECTION_PER_OBSERVATION_FACTOR}"
fi

# shellcheck disable=SC2086 # intentional splitting of optional selection_target_args
python scripts/mock_data/generate_mock_data.py \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --n0 "${N0}" \
  --zmax "${ZMAX}" \
  --survey-z50 "${SURVEY_Z50}" \
  --survey-width "${SURVEY_WIDTH}" \
  --galaxy-density-delta "${GALAXY_DENSITY_DELTA}" \
  --nobs "${NOBS}" \
  --nsamp "${NSAMP}" \
  --ndraw "${NDRAW}" \
  --selection-batch-size "${SELECTION_BATCH_SIZE}" \
  ${selection_target_args} \
  --dL-fractional-uncertainty "${DL_FRAC_UNCERTAINTY}" \
  --m1det-fractional-uncertainty "${M1DET_FRAC_UNCERTAINTY}" \
  --m2det-fractional-uncertainty "${M2DET_FRAC_UNCERTAINTY}" \
  --chieff-uncertainty "${CHIEFF_UNCERTAINTY}" \
  --sky-uncertainty-deg "${SKY_UNCERTAINTY_DEG}" \
  --nside "${NSIDE}" \
  --verbose

export OUTDIR NSIDE NOBS NSAMP
echo "Starting verbose ingestion validation for generated products."
python - <<'PY'
from argparse import Namespace
from pathlib import Path
import multiprocessing as mp
import os
import numpy as np

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
from darksirens.inference.data import load_all_data

out = Path(os.environ["OUTDIR"])
nside = int(os.environ["NSIDE"])
nobs = int(os.environ["NOBS"])
nsamp = int(os.environ["NSAMP"])
opts = Namespace(
    universe_model="dark_sirens",
    survey_path=str(out / f"catalog_pixelated_nside_{nside}.h5"),
    gw_path=str(out / "mock_gw_events.h5"),
    gwselection_path=str(out / "mock_gw_selection.h5"),
    sigma_kernel=0.005,
    use_LSS=False,
    counterpart=None,
    counterpart_nside=1,
    counterpart_dz=1e-4,
)
data = load_all_data(opts)
assert int(data["nEvents"]) == nobs, data["nEvents"]
assert int(data["nsamp"]) == nsamp, data["nsamp"]
assert int(data["nside"]) == nside, data["nside"]
assert len(data["p_draw"]) > 5 * nobs, "too few detected selection samples"
assert np.isfinite(np.asarray(data["p_draw"])).all(), "non-finite p_draw values"
print("Ingestion validation passed.")
PY

if [ "${RUN_INFERENCE}" = "1" ]; then
  echo "Starting optional darksirens_inference sampler run."
  python -m darksirens.tool.darksirens_inference \
    --gw_path "${OUTDIR}/mock_gw_events.h5" \
    --gwselection_path "${OUTDIR}/mock_gw_selection.h5" \
    --survey_path "${OUTDIR}/catalog_pixelated_nside_${NSIDE}.h5" \
    --sampler dynesty \
    --pop_model powerlaw+peak_shared_beta_spin \
    --universe_model dark_sirens \
    --fix_population False \
    --fix_cosmology False \
    --fix_survey False \
    --nlive "${INFERENCE_NLIVE}" \
    --dlogz "${INFERENCE_DLOGZ}" \
    --seed "${SEED}" \
    --show_progress True \
    --save_path "${OUTDIR}/inference_realistic"
fi

cat <<EOF
Mock data validation complete.
Products are in: ${OUTDIR}
Generated survey density N0=${N0} Mpc^-3 and zmax=${ZMAX}; fixed inference survey JSON: ${FIXED_SURVEY_JSON}
Set RUN_INFERENCE=1 to run the optional darksirens_inference sampler with free H0 and Om0.
EOF
