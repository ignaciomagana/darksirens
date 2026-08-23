#!/bin/bash
# Sampled-theta nested-sampling run: the PRODUCTION selection-channel posterior.
#
# Everything identical to the "sel" grid-scan config (run_h0_real.py: 44
# S>=0.495 events + MATCHED betaS, union catalog n64, no Q, homogeneous
# missing branch) except theta: the grid scans FIXED (M0hat, sigma_M) at
# theta_hat; here they are SAMPLED under the fit's truncated-normal prior
# (--selection_fit flips the prior kinds; m_lim stays a pinned datum).
#
# Two deliverables:
#   1. H0 marginalized over selection-model uncertainty (the honest budget)
#      vs the theta-fixed grid scan (sel: median 75.7 [57.2, 87.8]).
#   2. The theta posterior-vs-prior pulls as a misspecification gate: a
#      posterior pulled off the 22.8M-galaxy prior = selection model wrong.
#      Expectation from the +/-5sigma ablation (dlogL <= 5e-4): posterior
#      sits on the prior, H0 unchanged.
#
# Pins (must match the fit/build): DARKSIRENS_ZMAX=0.75; z_depth=0.30 rides
# the survey attr; m_lim=21.0 pinned by the resolver from the fit JSON;
# Om0/log10n0/delta/sigma_kde fixed at the grid-scan values
# (data/n0_calibration.json).  Sampled space: [H0, M0hat, sigma_M].
set -euo pipefail
cd "$(dirname "$0")"

OUT=${1:-data/ns_sampled_theta}
mkdir -p "$OUT"

# Fixed budget values from the CURRENT calibration (single source of truth;
# includes the 2026-08-09 (1+z)^delta normalization fix -- n0 is now
# calibrated against the FULL likelihood budget integrand).
FIXED_JSON=$(python - <<'EOF'
import json
cal = json.load(open("data/n0_calibration.json"))
print(json.dumps({"Om0": 0.3075, "sigma_kde": 0.003,
                  "log10n0": cal["log10n0"], "delta": cal["delta"]}))
EOF
)
echo "[run_ns] fixed_parameter_values: $FIXED_JSON"

# PYTHONPATH: the conda env's darksirens_inference entry point resolves to an
# OLDER installed copy (no --c_mode/--selection_fit); the dev repo must win.
DARKSIRENS_ZMAX=0.75 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-dev \
python -m darksirens.cli.inference \
  --gw_path /hildafs/projects/phy230014p/magana/desi_darksirens_selection/final/experiments/experiment_loa_rebuild/inputs/gwsamples_44.h5 \
  --gwselection_path /hildafs/projects/phy230014p/magana/desi_darksirens_selection/final/experiments/experiment_loa_rebuild/inputs/selection_betaS_v2_loaFaint_marg_s0495_noom.h5 \
  --survey_path data/pixelated_n64/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens \
  --pop_model gwtc5_fiducial_bpl2peaks \
  --c_mode selection \
  --selection_fit data/selection_fit_union.json \
  --catalog_sky_weighting field \
  --use_lss false \
  --fix_population true \
  --fix_cosmology false \
  --fix_de true \
  --fix_survey false \
  --fixed_parameter_values "$FIXED_JSON" \
  --prior_overrides '{"H0": [20.0, 140.0]}' \
  --sampler tinyns \
  --nlive "${NLIVE:-1000}" \
  --tinyns_preset "${TINYNS_PRESET:-adaptive_gpu}" \
  --max_samples "${MAX_SAMPLES:-1000000}" \
  --checkpoint_interval 1800 \
  --resume auto \
  --save_path "$OUT"
