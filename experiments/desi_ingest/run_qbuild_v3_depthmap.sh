#!/bin/bash
# The v3 MASK-FREE Q build: v2 plus --q-support-depth 0.30.
#
# WHY 0.30, MEASURED.  The real DESI catalog is HARD-TRUNCATED there: max
# observed z = 0.3000 exactly, 0 galaxies above it.  v2 nevertheless fit the
# whole DARKSIRENS_ZMAX=0.75 grid, so above the truncation every covered pixel
# handed the solver an N_obs = 0 row against a strictly positive model
# expectation f_p C(z) dN_exp -- and the MAP pushed each pixel's Q down in
# proportion to its own model rate, which under --depth-map IS f_p.  That
# manufactures a correlation out of empty sky, and the v2 record shows it lives
# ENTIRELY above 0.30 (data/fits/q_v2_depthmap_maskfree.json, nine-slice sweep;
# z read off the ZMAX=0.75 zgrid):
#
#   slice    0  z 0.000   corr +0.006      slice  499  z 0.322   corr -0.508
#   slice  124  z 0.072   corr +0.033      slice  624  z 0.418   corr -0.487
#   slice  249  z 0.150   corr +0.103      slice  749  z 0.521   corr -0.601
#   slice  374  z 0.233   corr +0.130      slice  874  z 0.632   corr -0.864  <-- worst
#                                          slice  999  z 0.750   corr +0.004
#
# Below 0.30 the worst |corr| is 0.130; the entire -0.5 .. -0.86 band is above
# it.  worst |corr| 0.864 > the 0.10 mask-free tolerance is what stamped v2
# f_p_aware=FALSE, which makes the loader refuse the f_p x Q pairing.
#
# WHAT THE CUT DOES.  --q-support-depth 0.30 truncates the FIT (the radial
# chi_u solve grid, its N_obs/C inputs, and the per-z budget renormalization)
# to zgrid nodes with z <= 0.30 -- 469 of 1000, top fitted node z = 0.29974 --
# and pins logQ to EXACTLY 0.0 (Q = 1) at the other 531 for every pixel.  Q
# owns PLACEMENT inside the catalog's support; the missing-host budget above it
# belongs to C(z)/n0.  Precedent: the latent seam pins Q = 1 outside its own
# support (darksirens/likelihood/latent_q.py).
#
# Everything else is v2 verbatim.  radial only: --mode gp3d and stratified
# selection REFUSE both --depth-map and --q-support-depth (neither folds f_p in
# nor cuts the radial grid, so the stamps would be unearned).  The radial
# builder forks workers, hence CPU JAX.  Empty covered pixels are deduped PER
# f_p VALUE (each f_p is its own N_obs = 0 fit problem).
set -euo pipefail
cd "$(dirname "$0")"

export DARKSIRENS_ZMAX=0.75
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

CATALOG=data/pixelated_n64/catalog_pixelated_nside_64.h5
FIT=data/selection_fit_union.json
DEPTH=data/mth_map_nside128.h5
OUT=data/fits/q_v3_depthmap.h5
Q_SUPPORT=0.30
LOG10N0=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['log10n0'])")
DELTA=$(python -c "import json; print(json.load(open('data/n0_calibration.json'))['delta'])")
mkdir -p data/fits logs

echo "log10n0=$LOG10N0 delta=$DELTA q_support_depth=$Q_SUPPORT"

JAX_PLATFORMS=cpu python -m darksirens.cli.build_lognormal_completion \
  --catalog "$CATALOG" --out "$OUT" \
  --c-mode selection --selection-fit "$FIT" \
  --log10n0 "$LOG10N0" --delta "$DELTA" \
  --depth-map "$DEPTH" \
  --q-support-depth "$Q_SUPPORT" \
  --mode radial --workers 16 --n-members 8 --seed 22 \
  2>&1 | tee logs/qbuild_v3_depthmap.log

# Measure the artifact independently of the builder's own stamp, exactly as v2
# was measured, so the two JSON records are directly comparable.  The nine-slice
# sweep above z = 0.30 must now read corr == 0 (those slices are bit-zero logQ
# for every pixel, so the sweep skips them for zero variance); the verdict rests
# on the slices BELOW the cut, where v2 already sat at |corr| <= 0.130.
# DO NOT trust the stamp without reading this report.
JAX_PLATFORMS=cpu python measure_maskfree_v2.py "$OUT" "$DEPTH" "$CATALOG" \
  --json data/fits/q_v3_depthmap_maskfree.json \
  2>&1 | tee -a logs/qbuild_v3_depthmap.log

echo "DONE"
