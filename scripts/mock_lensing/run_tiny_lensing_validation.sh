#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

OUTDIR="${OUTDIR:-data/tiny_lens}"
RUNROOT="${RUNROOT:-runs/tiny_lens_validation}"
PYTHON="${PYTHON:-python}"
SAMPLER_ARGS="${SAMPLER_ARGS:---sampler dynesty --nlive 40 --dlogz 10 --max_samples 0 --pe_max_per_pair 64}"

mkdir -p "$OUTDIR" "$RUNROOT"

echo "[tiny-lens] generating mock in $OUTDIR"
"$PYTHON" scripts/mock_lensing/generate_mock_lensing.py \
  --outdir "$OUTDIR" \
  --conditioning poisson_counts \
  --n-universe "${N_UNIVERSE:-2000}" \
  --max-sing-keep "${MAX_SING_KEEP:-5}" \
  --max-pair-keep "${MAX_PAIR_KEEP:-2}" \
  --nsamp "${NSAMP:-64}" \
  --n-unlensed-inj "${N_UNLENSED_INJ:-1000}" \
  --n-lensed-inj "${N_LENSED_INJ:-1000}" \
  --seed "${SEED:-2026}"

COMMON=(
  # Unified observed-catalog layout (the legacy split mock_gw_pe/mock_pair_pe
  # layout has been dead since the 2026-07-01 unified migration).
  --gw_path "$OUTDIR/mock_observed_gw_pe.h5"
  --observed_catalog_path "$OUTDIR/observed_catalog.json"
  --gwselection_path "$OUTDIR/mock_gw_selection.h5"
  --wl_backend lognormal
  --pop_model powerlaw+peak
  --fix_cosmology true
  --fix_survey true
  --fix_population true
  --seed "${SEED:-2026}"
)

echo "[tiny-lens] running cluster_mode=off"
"$PYTHON" -m darksirens.cli.inference_lensing \
  "${COMMON[@]}" \
  --cluster_mode off \
  $SAMPLER_ARGS \
  --save_path "$RUNROOT/off"

echo "[tiny-lens] running cluster_mode=j2"
"$PYTHON" -m darksirens.cli.inference_lensing \
  "${COMMON[@]}" \
  --cluster_mode j2 \
  --lensed_injections_path "$OUTDIR/mock_lensed_injections.h5" \
  --pair_metadata_path "$OUTDIR/mock_pair_metadata.h5" \
  --partition_path "$OUTDIR/partition.json" \
  $SAMPLER_ARGS \
  --save_path "$RUNROOT/j2"

if [[ "${RUN_MARGINALIZE_EXACT:-0}" == "1" && -f "$OUTDIR/candidate_pairs.json" ]]; then
  echo "[tiny-lens] running cluster_mode=j2 with partition_mode=marginalize_exact"
  "$PYTHON" -m darksirens.cli.inference_lensing \
    "${COMMON[@]}" \
    --cluster_mode j2 \
    --lensed_injections_path "$OUTDIR/mock_lensed_injections.h5" \
    --pair_metadata_path "$OUTDIR/mock_pair_metadata.h5" \
    --candidate_pairs_path "$OUTDIR/candidate_pairs.json" \
    --partition_mode marginalize_exact \
    --max_exact_partitions "${MAX_EXACT_PARTITIONS:-10000}" \
    $SAMPLER_ARGS \
    --save_path "$RUNROOT/j2_marginalized"
fi

echo "[tiny-lens] mock files: $OUTDIR"
echo "[tiny-lens] inference outputs: $RUNROOT/off, $RUNROOT/j2"
if [[ "${RUN_MARGINALIZE_EXACT:-0}" == "1" && -f "$OUTDIR/candidate_pairs.json" ]]; then
  echo "[tiny-lens] marginalization output: $RUNROOT/j2_marginalized"
fi
