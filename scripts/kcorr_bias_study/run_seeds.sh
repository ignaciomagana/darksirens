#!/usr/bin/env bash
# Launch N independent mock realizations of the H0 propagation, in parallel.
# Usage: run_seeds.sh <outdir> <nseeds> [extra args to propagate_to_h0.py]
set -eu
OUT="$1"; shift
N="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for s in $(seq 1 "$N"); do
  python "$ROOT/scripts/kcorr_bias_study/propagate_to_h0.py" \
      --seed $((20260812 + s)) --out "$OUT/h0_s$s.json" "$@" \
      > "$OUT/h0_s$s.log" 2>&1 &
done
wait
echo "done: $N realizations in $OUT"
