#!/bin/bash
# Paired comparison runs: stored-PE baseline (A) vs flow surrogates (B).
# powerlaw+peak, fixed cosmology + survey, sampling only {alpha_PL, mu_G, gamma};
# all other population parameters pinned to the registry fiducials.
set -u
cd /hildafs/projects/phy230014p/magana/src/darksirens

FIX='{"$v_1$": 0.1, "$m_{\\min,\\rm PL}$": 5.0, "$m_{\\max,\\rm PL}$": 80.0, "$\\delta m_{\\min,\\rm PL}$": 3.0, "$\\delta m_{\\max,\\rm PL}$": 10.0, "$\\sigma_{\\rm G}$": 5.0, "$\\beta$": 1.0, "$\\mu_\\chi$": 0.0, "$\\sigma_\\chi$": 0.1}'

COMMON=(--gwselection_path selection_o3o4ab_allsky.h5
        --universe_model spectral_sirens
        --pop_model powerlaw+peak
        --fixed_cosmology true --fix_survey true
        --sampler tinyns --nlive 300 --seed 7
        --save_path flows/runs)

case "${1:?usage: run_pair.sh A|B}" in
  A)
    exec python -m darksirens.cli.inference \
      --gw_path flows/pe_store_flowsubset.h5 \
      "${COMMON[@]}" --fixed_parameter_values "$FIX"
    ;;
  B)
    exec python -m darksirens.cli.inference \
      --gw_flows_path flows/m1m2dLchi_eff \
      --flows_nsamp 16384 --flows_seed 42 \
      "${COMMON[@]}" --fixed_parameter_values "$FIX"
    ;;
esac
