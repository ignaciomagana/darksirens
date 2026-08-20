#!/usr/bin/env bash
# Clustered-mock completeness closure experiment: generate -> fit -> plot.
#
# Usage:  bash experiments/completeness_viz/run_all.sh
# Override any setting via the environment, e.g.:
#     OUT=output_ns32 NSIDE=32 WORKERS=16 bash experiments/completeness_viz/run_all.sh
#
# JAX_PLATFORMS=cpu: the radial Q build forks worker processes, which breaks
# an already-initialized CUDA context; these solves are CPU-bound anyway.
# Do NOT set DARKSIRENS_ZMAX — the package zgrid is read once at import and
# every step (build, fit, plot) must share the identical grid.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-output}"
SEED="${SEED:-42}"
NSIDE="${NSIDE:-16}"
ZMAX="${ZMAX:-0.5}"
NTARGET="${NTARGET:-300000}"
NMEMBERS="${NMEMBERS:-16}"
WORKERS="${WORKERS:-12}"
# Completeness base + gp3d inducing-grid overrides (empty = builder default,
# reproducing the legacy per-pixel reference run exactly).  Aggregate runs
# should use their own OUT: cached Q tables are c_mode-stamped and refuse
# reuse across bases.
C_MODE="${C_MODE:-per_pixel}"
GP3D_NZ="${GP3D_NZ:-}"
GP3D_NSPH="${GP3D_NSPH:-}"
ZNODE_HI="${ZNODE_HI:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

FIT_FLAGS=(--c-mode "$C_MODE")
[ -n "$GP3D_NZ" ] && FIT_FLAGS+=(--gp3d-nz-nodes "$GP3D_NZ")
[ -n "$GP3D_NSPH" ] && FIT_FLAGS+=(--gp3d-nsph-nodes "$GP3D_NSPH")
[ -n "$ZNODE_HI" ] && FIT_FLAGS+=(--gp3d-z-node-hi "$ZNODE_HI")

python generate_clustered_mock.py --outdir "$OUT" --seed "$SEED" \
    --nside "$NSIDE" --zmax "$ZMAX" --n-target "$NTARGET"
python fit_completeness.py --outdir "$OUT" \
    --catalog "$OUT/catalog_pixelated_nside_${NSIDE}.h5" \
    --truth "$OUT/truth.h5" --workers "$WORKERS" --n-members "$NMEMBERS" \
    "${FIT_FLAGS[@]}"
python plot_completeness.py --outdir "$OUT"

echo "Done. Figures in $OUT/plots/, closure metrics in $OUT/plots/closure_summary.json"
