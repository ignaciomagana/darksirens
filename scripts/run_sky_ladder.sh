#!/usr/bin/env bash
#
# run_sky_ladder.sh — end-to-end sky-anisotropy / 3-D-clustering ladder.
#
# 1. Generate a mock with a KNOWN injected 3-D structure (a z-evolving dipole
#    + a localized (ra,dec,z0) blob); the selection injections stay isotropic.
# 2. Run inference once per sky model:
#       isotropic -> dipole -> sphere_gp -> sphere_gp_z -> overdensity_gp
# 3. Compare them with darksirens_analyze (relative evidences + pairwise Bayes
#    factors) and produce the recovered-structure plots (dipole posterior,
#    sphere-GP / 3-D z-shell maps).
#
# Usage:
#     bash scripts/run_sky_ladder.sh
# Override any setting via the environment, e.g.:
#     NOBS=400 NLIVE=2000 SKY_MODELS="isotropic dipole sphere_gp_z" \
#         OUTDIR=results/run1 bash scripts/run_sky_ladder.sh
#
# Notes
# -----
# * The Bayes factor needs a nested sampler (dynesty/jaxns) for the evidence.
#   The GP sky models are high-dimensional (sphere_gp ~50, sphere_gp_z /
#   overdensity_gp ~195 whitened latents), so nested sampling is SLOW there —
#   raise NLIVE and expect long runtimes, or trim SKY_MODELS for a quick check.
# * overdensity_gp (radial clustering) is partly degenerate with the population
#   redshift slope gamma; set FIX_POP=true for an identifiable measurement
#   (this drops 'isotropic', which would then have zero free parameters).
set -euo pipefail

OUTDIR="${OUTDIR:-results/sky_ladder}"
NOBS="${NOBS:-200}"
NDRAW="${NDRAW:-80000}"
SAMPLER="${SAMPLER:-dynesty}"
NLIVE="${NLIVE:-1000}"
SEED="${SEED:-22}"
SKY_MODELS="${SKY_MODELS:-isotropic dipole sphere_gp sphere_gp_z overdensity_gp}"
FIX_POP="${FIX_POP:-false}"

# Injected truth: a z-evolving dipole (amplitude ramps to z_pivot) + a 3-D blob.
DIP_AMP="${DIP_AMP:-0.5}"; DIP_RA="${DIP_RA:-120}"; DIP_ZPIV="${DIP_ZPIV:-1.0}"
BLOB_AMP="${BLOB_AMP:-0.8}"; BLOB_RA="${BLOB_RA:-250}"; BLOB_DEC="${BLOB_DEC:--30}"
BLOB_Z0="${BLOB_Z0:-0.5}"

DATA="$OUTDIR/data"
GW="$DATA/mock_gw_events.h5"
SEL="$DATA/mock_gw_selection.h5"
mkdir -p "$DATA"

extra_fix=""
if [ "$FIX_POP" = "true" ]; then
    extra_fix="--fix_population true"
    SKY_MODELS="$(echo "$SKY_MODELS" | tr ' ' '\n' | grep -v '^isotropic$' | tr '\n' ' ')"
    echo "[ladder] FIX_POP=true -> dropping 'isotropic' (0 free params); models: $SKY_MODELS"
fi

echo "==> [1/3] Generating mock with injected 3-D structure (nobs=$NOBS)"
python scripts/mock_dark_sirens/generate_mock_data.py \
    --outdir "$DATA" --nobs "$NOBS" --ndraw "$NDRAW" --seed "$SEED" --verbose \
    --sky-dipole-amp "$DIP_AMP" --sky-dipole-ra-deg "$DIP_RA" --sky-dipole-z-pivot "$DIP_ZPIV" \
    --sky-blob-amp "$BLOB_AMP" --sky-blob-ra-deg "$BLOB_RA" --sky-blob-dec-deg "$BLOB_DEC" \
    --sky-blob-z0 "$BLOB_Z0"

RUN_DIRS=()
for model in $SKY_MODELS; do
    echo "==> [2/3] Inference: sky_model=$model  (sampler=$SAMPLER, nlive=$NLIVE)"
    SAVE="$OUTDIR/runs/$model"
    mkdir -p "$SAVE"
    darksirens_inference \
        --gw_path "$GW" --gwselection_path "$SEL" \
        --sampler "$SAMPLER" --nlive "$NLIVE" --seed "$SEED" \
        --universe_model spectral_sirens --pop_model powerlaw+peak \
        --fixed_cosmology true $extra_fix \
        --sky_model "$model" \
        --save_path "$SAVE"
    # The tool writes results.hdf5 into a timestamped subdir of --save_path.
    run_dir=$(ls -dt "$SAVE"/*/ 2>/dev/null | head -1 || true)
    [ -z "$run_dir" ] && run_dir="$SAVE"
    RUN_DIRS+=("${run_dir%/}")
done

echo "==> [3/3] Model comparison + recovered-structure plots"
darksirens_analyze --run_dirs "${RUN_DIRS[@]}" --outdir "$OUTDIR/figs"

echo
echo "Done."
echo "  Relative evidences + Bayes factors: printed above; figures in $OUTDIR/figs"
echo "    (model_evidences.pdf, bayes_factors.pdf, sky_dipole_*.pdf, sky_gp_map_*.pdf)"
echo "  Injected truth: $GW  (attrs 'injected_sky' / 'injected_sky_dipole')"
echo "  Expectation: with a z-evolving structure, sphere_gp_z / overdensity_gp"
echo "  should out-score the flat sphere_gp, which averages the z-evolution away."
