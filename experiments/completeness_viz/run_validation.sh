#!/usr/bin/env bash
# Validation runs for feat/completeness-aggregate-budget (PLAN.md gates 2b-3).
# Two closure reruns against the frozen 2026-08-07 mock (output/):
#   output_pp_s0/  per-pixel mode + S0 fixes (quantifies S0-only deltas)
#   output_agg/    aggregate c_mode + resolved gp3d nodes (the S1 gates)
# Each run is two passes: CPU pass builds the fork-based radial fits
# (JAX_PLATFORMS=cpu, see run_all.sh), then a CUDA pass builds gp3d on the
# H100 while every cached fit reloads.
set -euo pipefail
cd "$(dirname "$0")"

CAT=output/catalog_pixelated_nside_16.h5
TRUTH=output/truth.h5
COMMON=(--catalog "$CAT" --truth "$TRUTH" --workers 12 --n-members 16)

run_mode () {
    local out="$1"; shift
    mkdir -p "$out"
    ln -sf "$(readlink -f "$CAT")" "$out/$(basename "$CAT")"
    ln -sf "$(readlink -f "$TRUTH")" "$out/truth.h5"
    echo "=== [$out] pass 1 (CPU: radial builds) $(date)"
    JAX_PLATFORMS=cpu python fit_completeness.py --outdir "$out" \
        "${COMMON[@]}" "$@" --skip q_gp3d priors 2>&1 | tee "$out/fit_pass1.log"
    echo "=== [$out] pass 2 (CUDA: gp3d + priors) $(date)"
    python fit_completeness.py --outdir "$out" \
        "${COMMON[@]}" "$@" 2>&1 | tee "$out/fit_pass2.log"
    echo "=== [$out] plots $(date)"
    JAX_PLATFORMS=cpu python plot_completeness.py --outdir "$out" \
        2>&1 | tee "$out/plot.log"
}

# Same resolved gp3d nodes in both runs so they differ by c_mode ALONE
# (the shipped 6-node default correctly trips the S0c resolution guard here).
NODES=(--gp3d-nz-nodes 12 --gp3d-z-node-hi 0.5 --gp3d-nsph-nodes 64)
run_mode output_pp_s0 "${NODES[@]}"
run_mode output_agg --c-mode aggregate "${NODES[@]}"

echo "=== validation runs complete $(date)"
