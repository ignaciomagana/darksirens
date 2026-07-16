#!/usr/bin/env bash
# =============================================================================
# run_smoke_tests.sh  —  one-command smoke tests across the darksirens use cases
# =============================================================================
# Generates tiny mock data ONCE, then runs every use case (universe models, pop
# models, sky models, marks, LSS completion, weak + strong lensing, samplers) at
# a fast "smoke" profile and prints a PASS / FAIL / SKIP summary. A single
# failure never aborts the run — the point is to see what works/breaks.
#
#   bash scripts/smoke_tests/run_smoke_tests.sh            # run everything (smoke)
#   bash scripts/smoke_tests/run_smoke_tests.sh --list     # list cases, run nothing
#   bash scripts/smoke_tests/run_smoke_tests.sh --cases U-spec,S-dip   # a subset
#   bash scripts/smoke_tests/run_smoke_tests.sh --full     # realistic settings
#   bash scripts/smoke_tests/run_smoke_tests.sh --slow     # also the ~10-20min GP-sky cases
#   bash scripts/smoke_tests/run_smoke_tests.sh --pytest   # also run the pytest layer
#   bash scripts/smoke_tests/run_smoke_tests.sh --keep     # keep the _out/ workdir
#
# Env overrides: CONDA=<path to conda.exe>  ENV=<conda env>  NOBS NSIDE NLIVE
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# ---- configuration (overridable via env) ------------------------------------
CONDA="${CONDA:-/c/Users/Alien/anaconda3/Scripts/conda.exe}"
ENV="${ENV:-darksirens-dev}"
# Repo-root-RELATIVE output paths: the run cwd is the repo root, so both the
# tools (h5py reads) and the local sim pipeline (Python os.makedirs writes)
# resolve them identically. Absolute Git-Bash paths like /c/Users/... are
# mis-resolved by Windows Python on write, so do not use them here.
OUT="scripts/smoke_tests/_out"
LOGS="$OUT/logs"; RUNS="$OUT/runs"; FIGS="$OUT/figs"
DATA="$OUT/data"; BDATA="$OUT/data_bright"; LENSD="$OUT/lensing"

ONLY=""; DO_LIST=0; FULL=0; KEEP=0; DO_PYTEST=0; DO_SLOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) DO_LIST=1 ;;
    --full) FULL=1 ;;
    --slow) DO_SLOW=1 ;;
    --keep) KEEP=1 ;;
    --pytest) DO_PYTEST=1 ;;
    --cases) ONLY="$2"; shift ;;
    --cases=*) ONLY="${1#*=}" ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
# The 192-latent GP sky models (sphere_gp_z / overdensity_gp) take ~10-20 min each
# under NUTS (gradient compile dominates); gate them behind --slow (or --full).
RUN_SLOW=0; { [ "$DO_SLOW" = 1 ] || [ "$FULL" = 1 ]; } && RUN_SLOW=1

# Smoke vs full profiles.
if [ "$FULL" = 1 ]; then
  NOBS="${NOBS:-100}"; NSAMP=512; NDRAW=80000; NSIDE="${NSIDE:-32}"; NLIVE="${NLIVE:-1000}"
  DYN="--sampler dynesty --nlive $NLIVE"
  DYN_HI="--sampler dynesty --nlive $NLIVE"
  NUTS="--sampler numpyro --nuts_warmup 300 --nuts_samples 300 --nuts_chains 1"
  NORM_ENV=""   # full: use the tool's default normalisation grids
  SIM_FLAGS="--n-universe 120000 --seed 7 --nsamp 500 --n-sing-keep 80 --n-pair-keep 20 --n-unlensed-inj 120000 --n-lensed-inj 120000"
else
  # Smoke = "does it run", not accuracy: tiny mock, early-stop sampling, coarse
  # normalisation grids (via env vars the tool reads).
  NOBS="${NOBS:-6}"; NSAMP=64; NDRAW=4000; NSIDE="${NSIDE:-8}"; NLIVE="${NLIVE:-40}"
  DYN="--sampler dynesty --nlive $NLIVE --dlogz 10"
  DYN_HI="--sampler dynesty --nlive 80 --dlogz 10"   # gradient-free, for higher-dim models
  # shallow NUTS tree (default depth 10 ⇒ up to 1024 leapfrogs/step ⇒ minutes for 192-dim GPs)
  NUTS="--sampler numpyro --nuts_warmup 8 --nuts_samples 8 --nuts_chains 1 --nuts_max_tree_depth 3"
  NORM_ENV="export DARKSIRENS_GW_N_MASS=16 DARKSIRENS_GW_N_Q=8 DARKSIRENS_GW_N_CHI=8;"
  SIM_FLAGS="--n-universe 8000 --seed 7 --nsamp 200 --n-sing-keep 15 --n-pair-keep 5 --n-unlensed-inj 20000 --n-lensed-inj 20000"
fi

# ---- helpers ----------------------------------------------------------------
dsrun_eval() { "$CONDA" run --no-capture-output -n "$ENV" bash -c "${NORM_ENV:-} $1"; }
hr() { printf '%s\n' "------------------------------------------------------------"; }

# Tools (run as modules → PATH-independent; two console scripts aren't installed).
INFER="python -m darksirens.cli.inference"
LENS="python -m darksirens.cli.inference_lensing"
BUILDQ="python -m darksirens.cli.build_lognormal_completion"
ANALYZE="python -m darksirens.cli.analyze"
MOCK="python scripts/mock_dark_sirens/generate_mock_data.py"
BMOCK="python scripts/mock_bright_sirens/generate_mock_bright_sirens.py"
LMOCK="python scripts/mock_lensing/generate_mock_lensing.py"

# ---- data paths -------------------------------------------------------------
GWE="$DATA/mock_gw_events.h5";        SEL="$DATA/mock_gw_selection.h5"
CAT="$DATA/catalog_pixelated_nside_${NSIDE}.h5"
CATM="$DATA/catalog_pixelated_nside_${NSIDE}_marked.h5"
QRAD="$DATA/q_radial.h5";             QGP3D="$DATA/q_gp3d.h5"
BGWE="$BDATA/mock_bright_gw_events.h5"; BSEL="$BDATA/mock_bright_gw_selection.h5"
BCP="$BDATA/bright_counterparts.json"
CPARGS=""

# ---- case registry ----------------------------------------------------------
IDS=(); DESCS=(); PRECONDS=(); CMDS=()
add(){ IDS+=("$1"); DESCS+=("$2"); PRECONDS+=("$3"); CMDS+=("$4"); }

# Universe models
add U-spec     "spectral_sirens (GW only)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-spec"
add U-wl       "spectral_sirens_wl (weak-lensing magnification)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens_wl --lensing_wl_model lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-wl"
add U-dark     "dark_sirens (incomplete catalog)" HAVE_DARK \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT --universe_model dark_sirens --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-dark"
add U-complete "dark_sirens_complete (complete catalog)" HAVE_DARK \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT --universe_model dark_sirens_complete --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-complete"
add U-bright   "bright_sirens (EM counterparts)" HAVE_BRIGHT \
  "$INFER --gw_path $BGWE --gwselection_path $BSEL --universe_model bright_sirens --counterpart $CPARGS --counterpart_dz 1e-4 --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-bright"

# Population models (spectral, fixed cosmology)
add P-bpl2pk "pop: brokenpowerlaw+2peaks" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model brokenpowerlaw+2peaks --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/P-bpl2pk"
add P-gp1d   "pop: gp1d_m1 (GP, needs tinygp)" HAVE_TINYGP \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model gp1d_m1 --fixed_cosmology true --fix_survey true $NUTS --seed 1 --save_path $RUNS/P-gp1d"
add P-gppop  "pop: gppop (binned-GP, needs tinygp; dynesty=gradient-free)" HAVE_TINYGP \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model gppop --fixed_cosmology true --fix_survey true $DYN_HI --seed 1 --save_path $RUNS/P-gppop"

# Sky models (spectral, fixed cosmology + population)
add S-iso   "sky: isotropic" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model isotropic --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $DYN --seed 1 --save_path $RUNS/S-iso"
add S-dip   "sky: dipole" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model dipole --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $DYN --seed 1 --save_path $RUNS/S-dip"
add S-mult  "sky: multipole (l<=2)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model multipole --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $DYN --seed 1 --save_path $RUNS/S-mult"
add S-mult3 "sky: multipole_l3 (l<=3)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model multipole_l3 --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $DYN --seed 1 --save_path $RUNS/S-mult3"
add S-sgp   "sky: sphere_gp (GP on S^2)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model sphere_gp --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $NUTS --seed 1 --save_path $RUNS/S-sgp"
add S-sgpz  "sky: sphere_gp_z ((sphere x z) GP) [slow]" RUN_SLOW \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model sphere_gp_z --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $NUTS --seed 1 --save_path $RUNS/S-sgpz"
add S-od    "sky: overdensity_gp (3-D clustering) [slow]" RUN_SLOW \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --sky_model overdensity_gp --pop_model powerlaw+peak --fixed_cosmology true --fix_population true --fix_survey true $NUTS --seed 1 --save_path $RUNS/S-od"

# Marked-host model (dark sirens)
add M-loglin "marks: loglinear (logmstar,logssfr)" HAVE_MARKS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CATM --universe_model dark_sirens --mark_model loglinear --marks logmstar,logssfr --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/M-loglin"

# LSS completion (dark sirens)
add L-legacy "LSS: legacy delta_g (--use_LSS)" HAVE_DARK \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT --universe_model dark_sirens --use_LSS true --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/L-legacy"
add L-radial "LSS: radial lognormal Q_LSS" HAVE_QRAD \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT --universe_model dark_sirens --lss_completion $QRAD --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/L-radial"
add L-gp3d   "LSS: 3-D angular-coupling Q_LSS" HAVE_QGP3D \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT --universe_model dark_sirens --lss_completion $QGP3D --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/L-gp3d"

# Multitracer K-catalog mixture (duplicated catalog exercises the full
# CLI -> bundle loader -> mixture pipeline end-to-end; sky weighting
# auto-resolves to field and fcat_2 is sampled alongside the population).
add K2-field  "multitracer: K=2 duplicated catalog (auto field weighting)" HAVE_DARK \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT $CAT --universe_model dark_sirens --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/K2-field"
add K2-marks  "multitracer: K=2 + per-catalog marked-host etas" HAVE_MARKS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CATM $CATM --universe_model dark_sirens --mark_model loglinear --marks logmstar,logssfr --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/K2-marks"
add K2-qlss   "multitracer: K=2 + per-catalog Q_LSS (field-modulated budget)" HAVE_QRAD \
  "$INFER --gw_path $GWE --gwselection_path $SEL --survey_path $CAT $CAT --universe_model dark_sirens --lss_completion $QRAD $QRAD --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/K2-qlss"

# Lensing: strong-lensing clusters
add X-cloff "SL clusters: singleton-only (cluster_mode off)" HAVE_BASE \
  "$LENS --gw_path $GWE --gwselection_path $SEL --cluster_mode off --wl_backend lognormal --pop_model powerlaw+peak --fix_cosmology true --fix_survey true --fix_population true $DYN --seed 1 --save_path $RUNS/X-cloff"
add X-cl2   "SL clusters: J=2 pairs (cluster_mode j2)" HAVE_LENS \
  "$LENS --gw_path $LENSD/mock_gw_pe.h5 --gwselection_path $LENSD/mock_gw_selection.h5 --lensed_injections_path $LENSD/mock_lensed_injections.h5 --pair_pe_path $LENSD/mock_pair_pe.h5 --partition_path $LENSD/partition.json --cluster_mode j2 --wl_backend lognormal --pop_model powerlaw+peak --fix_cosmology true --fix_survey true --fix_population false $DYN --seed 1 --save_path $RUNS/X-cl2"

# Samplers (tiny spectral case, fixed cosmology, free population)
add R-dynesty "sampler: dynesty" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 3 --save_path $RUNS/R-dynesty"
add R-numpyro "sampler: numpyro (NUTS)" ALWAYS \
  "$INFER --gw_path $GWE --gwselection_path $SEL --universe_model spectral_sirens --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $NUTS --seed 3 --save_path $RUNS/R-numpyro"

# Analyze (posterior-predictive + sky-ladder Bayes factors)
add A-analyze "analyze: spectra + Bayes-factor matrix (sky ladder)" HAVE_BASE \
  "$ANALYZE --run_dirs \$(find $RUNS/S-iso $RUNS/S-dip $RUNS/S-mult -name results.hdf5 -printf '%h ' 2>/dev/null) --outdir $FIGS"

# ---- --list -----------------------------------------------------------------
if [ "$DO_LIST" = 1 ]; then
  printf '%-11s  %s\n' "CASE" "DESCRIPTION"; hr
  for i in "${!IDS[@]}"; do printf '%-11s  %s\n' "${IDS[$i]}" "${DESCS[$i]}"; done
  exit 0
fi

# ---- prep -------------------------------------------------------------------
rm -rf "$OUT"; mkdir -p "$LOGS" "$RUNS" "$FIGS" "$DATA" "$BDATA"
echo "darksirens smoke tests  |  env=$ENV  profile=$([ $FULL = 1 ] && echo full || echo smoke)  out=$OUT"
echo "tinygp / numpyro must be importable for GP / numpyro cases."; hr

HAVE_BASE=0 HAVE_DARK=0 HAVE_BRIGHT=0 HAVE_QRAD=0 HAVE_QGP3D=0 HAVE_MARKS=0 HAVE_LENS=0 HAVE_TINYGP=0
prep() {  # name, cond-to-set, command
  printf '[prep] %-26s ' "$1"; local t0=$SECONDS
  if dsrun_eval "$3" >"$LOGS/prep_$1.log" 2>&1; then echo "ok ($((SECONDS-t0))s)"; eval "$2=1"; else echo "FAILED (see $LOGS/prep_$1.log)"; fi
}

dsrun_eval "python -c 'import tinygp'" >/dev/null 2>&1 && HAVE_TINYGP=1 || echo "[prep] tinygp not importable — GP cases will SKIP (pip install tinygp)"
# --n-galaxies caps the catalog size (a small mock catalog keeps the dark-siren
# per-proposal precompute/JIT-compile fast); --n0 would scale with volume to ~1e5.
if [ "$FULL" = 1 ]; then NGAL="${NGAL:-40000}"; else NGAL="${NGAL:-3000}"; fi
prep mock_dark HAVE_BASE "$MOCK --outdir $DATA --seed 1 --n-galaxies $NGAL --nobs $NOBS --nsamp $NSAMP --ndraw $NDRAW --nside $NSIDE --zmax 0.1"
[ -f "$CAT" ] && HAVE_DARK=1
[ "$HAVE_DARK" = 1 ] && prep marks HAVE_MARKS "python scripts/smoke_tests/make_marks.py --catalog $CAT --out $CATM"
[ "$HAVE_DARK" = 1 ] && prep q_radial HAVE_QRAD "$BUILDQ --catalog $CAT --out $QRAD --mode radial --n-members 0"
[ "$HAVE_DARK" = 1 ] && prep q_gp3d  HAVE_QGP3D "$BUILDQ --catalog $CAT --out $QGP3D --mode gp3d --n-members 0 --gp3d-pix-chunk 256"
# the bright generator takes --n0 (no --n-galaxies); its catalog size doesn't slow
# bright inference (which is counterpart-based, not catalog-based).
prep mock_bright HAVE_BRIGHT "$BMOCK --outdir $BDATA --seed 2 --n0 1e-3 --nobs 3 --nsamp $NSAMP --ndraw $NDRAW --zmax 0.1"
if [ "$HAVE_BRIGHT" = 1 ] && [ -f "$BCP" ]; then
  CPARGS="$("$CONDA" run --no-capture-output -n "$ENV" python - "$BCP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
it = d["counterparts"] if isinstance(d, dict) else d
print(" ".join(f"{c['ra_rad']} {c['dec_rad']} {c['z']}" for c in it))
PY
)"
  CMDS[4]="$INFER --gw_path $BGWE --gwselection_path $BSEL --universe_model bright_sirens --counterpart $CPARGS --counterpart_dz 1e-4 --pop_model powerlaw+peak --fixed_cosmology true --fix_survey true $DYN --seed 1 --save_path $RUNS/U-bright"
fi
# Strong-lensing mock: standalone in-repo generator (writes the current gwcat
# schema, so the files load through darksirens_inference_lensing's loaders).
prep lensing_sim HAVE_LENS "$LMOCK --outdir $LENSD $SIM_FLAGS"
hr

# ---- run cases --------------------------------------------------------------
declare -a STATUS TIMES
n_pass=0; n_fail=0; n_skip=0
for i in "${!IDS[@]}"; do
  id="${IDS[$i]}"; desc="${DESCS[$i]}"; pre="${PRECONDS[$i]}"; cmd="${CMDS[$i]}"
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$id,"* ]]; then STATUS[$i]="-"; TIMES[$i]="-"; continue; fi
  if [ "$pre" != "ALWAYS" ] && [ "$(eval echo \$$pre)" != "1" ]; then
    STATUS[$i]="SKIP"; TIMES[$i]="-"; printf 'SKIP  %-11s  %s (needs %s)\n' "$id" "$desc" "$pre"; n_skip=$((n_skip+1)); continue
  fi
  printf 'RUN   %-11s  %s\n' "$id" "$desc"; t0=$SECONDS
  if dsrun_eval "$cmd" >"$LOGS/$id.log" 2>&1; then
    STATUS[$i]="PASS"; n_pass=$((n_pass+1)); printf '  -> PASS (%ss)\n' "$((SECONDS-t0))"
  else
    STATUS[$i]="FAIL"; n_fail=$((n_fail+1)); printf '  -> FAIL (%ss)  log: %s\n' "$((SECONDS-t0))" "$LOGS/$id.log"
    tail -n 3 "$LOGS/$id.log" 2>/dev/null | sed 's/^/       | /'
  fi
  TIMES[$i]="$((SECONDS-t0))"
done

# ---- optional pytest layer --------------------------------------------------
if [ "$DO_PYTEST" = 1 ]; then
  hr; echo "[pytest] Tier-0 unit/integration suite"
  dsrun_eval "python -m pytest tests/ --ignore=tests/test_fixed_parameter_coordinates.py -q" 2>&1 | tail -8
fi

# ---- summary ----------------------------------------------------------------
hr; printf '%-11s  %-5s  %-6s  %s\n' "CASE" "STAT" "TIME" "DESCRIPTION"; hr
for i in "${!IDS[@]}"; do
  st="${STATUS[$i]:--}"; [ "$st" = "-" ] && continue
  printf '%-11s  %-5s  %-6s  %s\n' "${IDS[$i]}" "$st" "${TIMES[$i]:--}s" "${DESCS[$i]}"
done
hr
echo "PASS=$n_pass  FAIL=$n_fail  SKIP=$n_skip   (logs in $LOGS)"
[ "$KEEP" = 1 ] || echo "(_out kept; pass --keep to retain across runs, or delete $OUT)"
[ "$n_fail" -eq 0 ]
