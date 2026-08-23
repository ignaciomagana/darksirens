#!/bin/bash
# The v4 MASK-FREE Q build: v3's inputs verbatim, with two DEFECTS of the v3
# build fixed.  Nothing on the command line changes -- both fixes live in
# darksirens/cli/build_lognormal_completion.py.
#
# ---------------------------------------------------------------------------
# DEFECT 1 (fixed in the builder): the periodic wrap at the truncation edge.
# ---------------------------------------------------------------------------
# The radial solve uses a CIRCULANT prior -- an FFT covariance
# c[d] = sigma^2 exp(-d^2 / 2 ell^2) in PERIODIC index distance
# (gaussian_correlation_spectrum) -- so the first and last nodes of the solve
# domain are nearest neighbours.  On the full DARKSIRENS_ZMAX = 0.75 grid that
# seam sat in empty high-z sky and cost nothing.  --q-support-depth 0.30 moved
# it onto the catalog's truncation edge, gluing z = 0 to z = 0.30.  Measured,
# in the two records (data/fits/q_v{2,3}_depthmap_maskfree.json):
#
#   corr(Q, f_p) at node 0:   v2 (uncut)  +0.006      wrap partner = empty z 0.75
#                             v3 (cut)    +0.191      wrap partner = node 468
#   corr(Q, f_p) at node 468 (v3, the truncation edge):  +0.189
#
# i.e. node 0 was reading the EDGE's field, not its own; the contamination
# decayed back to the interior baseline by node ~50 of 469 -- about 2.6 x the
# build's ell_grid = 19.0 nodes (lss_corr_length_mpc 50.0 / dchi_u 2.63 Mpc).
#
# THE FIX: solve on a LONGER domain than we output.  When the cut actually
# truncates, the solve grid is padded above the top fitted node with
# max(32, ceil(4 * ell_grid)) = 77 extra nodes here, which carry
#   * ZERO rate  (C and dN_exp both padded with 0), so _map_solve_row's
#     rate_base > 0 mask drops them from the Poisson term entirely -- data
#     gradient EXACTLY 0, not merely small.  They are unconstrained PRIOR
#     nodes, not the maximally informative "this volume is empty" bins that
#     --q-support-depth exists to remove;
#   * ZERO budget weight (w_budget is never padded), so the per-z mean-one
#     renormalization cannot see them;
#   * NO output (only the first 469 solved columns reach the zgrid; logQ above
#     the cut is still bit-zero for every pixel).
# Four correlation lengths leave the two ends of the FITTED domain correlated
# at exp(-(77+1)^2 / 2 / 19.0^2) ~ 3e-4 of the marginal variance, against the
# ~0.999 that distance 1 imposed in v3.
#
# The padding is applied ONLY when the cut truncates (n_fitted < n_grid), so a
# build with no --q-support-depth, or one cut at/above the grid top, is
# byte-identical to before.
#
# ---------------------------------------------------------------------------
# DEFECT 2 (fixed in the builder): a mask-free criterion anchored to ZERO.
# ---------------------------------------------------------------------------
# _verify_mask_free compared corr(Q, f_p) against 0 with tolerance 0.10.  That
# fails any FAITHFUL Q on this catalog, because the covered DESI sky itself
# carries a real density-depth correlation.  Measured on the pixelated catalog
# with f_p from data/mth_map_nside128.h5 degraded to nside 64, at 68-192
# galaxies per covered pixel (so not shrinkage noise):
#
#   corr(N/f_p, f_p),  z in [0.10, 0.15]:  +0.112
#                      z in [0.20, 0.25]:  +0.174
#                      z in [0.27, 0.30]:  +0.237
#
# and v3's Q interior profile tracks it (+0.035 ... +0.21, identical in v2 and
# v3, i.e. it is not an artifact of the cut).  The old test conflated "Q
# absorbed the mask SHAPE" -- the real hazard, which cost v1 its H0 (on/off
# mean contrast 1.62 vs 0.05, H0 = 41.24 against a truth of 67.74) -- with "Q
# records true covered-sky structure that happens to correlate with depth".
#
# THE FIX: per z slice, the criterion is now
#     |corr(Q, f_p) - corr(N/f_p, f_p)| <= 0.10
# with corr_data measured from THIS catalog in a band of +-n_fitted/18 zgrid
# nodes around the slice (+-26 of 469 here, dz ~ 0.03 at z ~ 0.25).  The
# OFF-FOOTPRINT check -- max |logQ| <= 1e-6 where f_p == 0 -- is UNCHANGED; it
# is the guard against the v1 failure and it does not move.
#
# Both correlation profiles and the worst delta are printed by the build and
# stamped into the artifact's diagnostics, and measure_maskfree_v2.py now calls
# the SAME function the stamp is earned with, so its recomputed verdict cannot
# drift from the stamp.
#
# ---------------------------------------------------------------------------
# Everything else is v3 verbatim.  radial only: --mode gp3d and stratified
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
OUT=data/fits/q_v4_depthmap.h5
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
  2>&1 | tee logs/qbuild_v4_depthmap.log

# Measure the artifact independently of the builder's own stamp, exactly as v2
# and v3 were measured, so the four JSON records stay directly comparable.  The
# sweep above z = 0.30 still reads nothing (those slices are bit-zero logQ for
# every pixel, so they are skipped for zero variance); the verdict rests on the
# slices BELOW the cut, and now on their DELTA against the catalog's own
# density-depth correlation rather than against zero.
# DO NOT trust the stamp without reading this report.
JAX_PLATFORMS=cpu python measure_maskfree_v2.py "$OUT" "$DEPTH" "$CATALOG" \
  --json data/fits/q_v4_depthmap_maskfree.json \
  2>&1 | tee -a logs/qbuild_v4_depthmap.log

echo "DONE"
