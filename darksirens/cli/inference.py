#!/usr/bin/env python3
"""
darksirens_inference.py
-----------------------
Entry point for the dark-siren / spectral-siren hierarchical inference pipeline.

Usage examples
--------------
# Spectral sirens, dynesty, free cosmology + population:
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --sampler           dynesty \
    --pop_model         powerlaw+peak \
    --universe_model    spectral_sirens \
    --nlive             2000

# Dark sirens with galaxy catalog, fixed cosmology, tinyns (rwalk + jax kernel):
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --survey_path       catalog_nside64.h5 \
    --sampler           tinyns \
    --nlive             1000 \
    --tinyns_sample     rwalk \
    --tinyns_kernel     jax \
    --pop_model         brokenpowerlaw+2peaks \
    --universe_model    dark_sirens \
    --fixed_cosmology   true \

# Fix individual parameters via JSON:
    --fixed_parameter_values '{"$v_1$": 0.1}'
    --prior_overrides        '{"H0": [60.0, 80.0]}'
"""

import os

# ── JAX memory configuration (before any JAX import) ──────────────────────────
from darksirens.core.jax_config import configure_jax_runtime

configure_jax_runtime()

import sys
import json
import datetime
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import h5py

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from darksirens.gw.populations import get_fixed_population_params, pop_model_prior_parser
from darksirens.gw.populations.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.likelihood.factory import (
    _redshift_prior_materialization_reason,
    _resolve_redshift_prior_materialization,
    make_likelihood,
)
from darksirens.redshift.completion import build_pixel_kde_cache, completion_clip_diagnostics
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.inference.sampling import run_sampler
from darksirens.inference.tinyns_config import add_tinyns_arguments, build_tinyns_config
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID

warnings.simplefilter("ignore", FutureWarning)


# ── Imports from refactored CLI/IO modules ─────────────────────────────────────

from darksirens.cli.common import (
    _banner,
    _section,
    _row,
    _end,
    _ok,
    _warn,
    _err,
    _fatal,
    _fixed_dark_energy_metadata,
    _format_fixed_dark_energy_summary,
    _format_option_value,
    _print_all_cli_options,
    str_to_bool,
    parse_json_arg,
    parse_counterpart_arg,
)
from darksirens.io.results import save_results_hdf5
from darksirens.io.settings import save_settings_json


# ── Parameter table ────────────────────────────────────────────────────────────

def _print_parameter_table(
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    fixed_parameter_statuses: dict,
    fix_cosmology:          bool,
    fix_de:                 bool,
    fix_population:         bool,
    fix_survey:             bool,
    pop_params_fid:         np.ndarray,
    pop_labels_all:         list,
):
    """
    Print sampled parameters with bounds, individually-fixed params with their
    values, and block-fixed parameters with their fiducial values.
    """
    COSMO_FID  = {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID}
    DE_FID = {"w0": W0_FID, "wa": WA_FID}
    SURVEY_FID = {"log10n0": -2.0, "z50": 1.0, "w": 0.5,
                  "delta": 0.0, "b_miss": 1.0, "alpha_miss": 1.0, "sigma_kde": 0.0}
    pop_fid_map = {lbl: float(pop_params_fid[i])
                   for i, lbl in enumerate(pop_labels_all)}

    block_fixed_sections: list[tuple[str, dict[str, float]]] = []
    if fix_cosmology:
        block_fixed_sections.append(("cosmology", COSMO_FID))
    elif fix_de:
        block_fixed_sections.append(("dark energy", DE_FID))
    if fix_population:
        block_fixed_sections.append(("population", pop_fid_map))
    if fix_survey:
        block_fixed_sections.append(("survey", SURVEY_FID))

    _section("Parameter Space")

    _row("Sampled parameters", "")
    _row("Parameter", f"{'Lower':>12}  {'Upper':>12}  Status", width=24)
    _row("─" * 24,    f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)
    for label, lo, hi in zip(labels, lower_bound, upper_bound):
        note = "← overridden" if label in prior_overrides else ""
        print(f"  │    {label:<24} {lo:>12.4g}  {hi:>12.4g}  {note}")

    _row("─" * 24, f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)
    _row("Individually fixed parameters", "")
    if fixed_parameter_values:
        for label, value in fixed_parameter_values.items():
            note = fixed_parameter_statuses.get(label, f"fixed = {value:.6g}")
            if label in prior_overrides:
                lo, hi = prior_overrides[label]
                print(
                    f"  │    {label:<24} {float(lo):>12.4g}  "
                    f"{float(hi):>12.4g}  {note}"
                )
            else:
                print(f"  │    {label:<24} {'—':>12}  {'—':>12}  {note}")
    else:
        print(f"  │    {'(none)':<24}")

    _row("─" * 24, f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)
    _row("Block-fixed parameters", "")
    if block_fixed_sections:
        for block_name, fiducials in block_fixed_sections:
            print(f"  │    [{block_name}]")
            for label, value in fiducials.items():
                print(
                    f"  │    {label:<24} {'—':>12}  {'—':>12}  "
                    f"fixed = {value:.6g}  (block)"
                )
    else:
        print(f"  │    {'(none)':<24}")

    _row("─" * 24, f"{'─' * 12}  {'─' * 12}  {'─' * 20}", width=24)

    n_free = sum(1 for lo, hi in zip(lower_bound, upper_bound) if lo != hi)
    n_fix_ind   = len(fixed_parameter_values)
    n_fix_block = ((len(COSMO_FID) if fix_cosmology else len(DE_FID) if fix_de else 0)
                   + (len(pop_labels_all) if fix_population else 0)
                   + (6 if fix_survey else 0))
    _row("Free (sampled)",      n_free)
    if n_fix_ind:   _row("Fixed individually", n_fix_ind)
    if n_fix_block: _row("Fixed (block)",      n_fix_block)
    _row("Total in coord vec",  len(labels))
    _end()

def _completion_validation_survey_values(
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> dict[str, float]:
    """Choose representative survey values for dry-run clipping diagnostics."""
    fid = {
        "log10n0": -2.0,
        "z50": 1.0,
        "w": 0.5,
        "delta": 0.0,
        "b_miss": 1.0,
        "alpha_miss": 1.0,
        "sigma_kde": 0.0,
    }
    values = dict(fid)
    for label in values:
        if label in prior_overrides:
            lo, hi = prior_overrides[label]
            values[label] = 0.5 * (float(lo) + float(hi))
        if label in fixed_parameter_values:
            values[label] = float(fixed_parameter_values[label])
    return values


def run_completion_validation(
    opts,
    data: dict,
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> str:
    """Save a dry-run completion clipping diagnostic and return its path."""
    required = ["zgals_catalog", "dzgals_catalog", "wgals_catalog", "ngals_catalog"]
    if any(data.get(key) is None for key in required):
        # Backward-compatible tests/callers may use unsuffixed catalog keys.
        fallback = {
            "zgals_catalog": "zgals",
            "dzgals_catalog": "dzgals",
            "wgals_catalog": "wgals",
        }
        for dst, src in fallback.items():
            if data.get(dst) is None and data.get(src) is not None:
                data[dst] = data[src]
    full_z = data.get("zgals_catalog")
    full_dz = data.get("dzgals_catalog")
    full_w = data.get("wgals_catalog")
    full_n = data.get("ngals_catalog")
    if any(value is None for value in (full_z, full_dz, full_w, full_n)):
        _fatal(
            "--validate_completion requires a loaded galaxy catalog with "
            "zgals/dzgals/wgals/ngals arrays."
        )

    pixels_pe = np.asarray(data["pixels_pe"], dtype=np.int32)
    pixels_sel = np.asarray(data["pixels_sel"], dtype=np.int32)
    unique_pixels = np.unique(np.concatenate([pixels_pe, pixels_sel])).astype(
        np.int32, copy=False
    )
    max_pixels = max(1, int(opts.completion_validation_pixels))
    unique_pixels = unique_pixels[:max_pixels]

    dN_obs_kde, pixel_to_cache_idx = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=full_z,
        n_pix_catalog=int(data.get("n_pix_catalog", np.asarray(full_z).shape[0])),
        wgals=full_w,
        ngals=full_n,
    )

    survey_values = _completion_validation_survey_values(
        prior_overrides, fixed_parameter_values
    )
    # Completion validation is a dry run, so unsampled cosmological values must
    # be represented by the fixed/fiducial values used by the likelihood decoder.
    cosmo = CosmoParams(
        H0=float(fixed_parameter_values.get("H0", H0_FID)),
        Om0=float(fixed_parameter_values.get("Om0", OM0_FID)),
        w0=float(fixed_parameter_values.get("w0", -1.0)),
        wa=float(fixed_parameter_values.get("wa", 0.0)),
    )
    survey = SurveyParams(
        n0=10.0 ** survey_values["log10n0"],
        z50=survey_values["z50"],
        w=survey_values["w"],
        delta=survey_values["delta"],
        b_miss=survey_values["b_miss"],
        alpha_miss=survey_values["alpha_miss"],
        sigma_kde=survey_values["sigma_kde"],
    )
    em_catalog = EMCatalog(
        apix=data["apix"],
        zgals=jnp.asarray(full_z[unique_pixels]),
        dzgals=jnp.asarray(full_dz[unique_pixels]),
        wgals=jnp.asarray(full_w[unique_pixels]),
        ngals=jnp.asarray(full_n[unique_pixels], dtype=jnp.int32),
        delta_g_pix_z=jnp.asarray(
            data.get("delta_g_pix_z", jnp.zeros((1, dN_obs_kde.shape[1])))
        ),
        dN_obs_kde=dN_obs_kde,
        pixel_to_cache_idx=pixel_to_cache_idx,
        unique_pixels=jnp.asarray(unique_pixels, dtype=jnp.int32),
    )
    diagnostics = completion_clip_diagnostics(
        cosmo=cosmo,
        survey=survey,
        em_catalog=em_catalog,
        max_pixels=max_pixels,
    )
    diagnostics["survey_values"] = survey_values
    diagnostics["cosmology_values"] = {
        "H0": float(cosmo.H0),
        "Om0": float(cosmo.Om0),
        "w0": float(cosmo.w0),
        "wa": float(cosmo.wa),
    }
    diagnostics["dark_energy_fixed"] = _fixed_dark_energy_metadata(
        opts, fixed_parameter_values
    )
    diagnostics["prior_overrides"] = prior_overrides or None
    diagnostics["fixed_parameter_values"] = fixed_parameter_values or None

    os.makedirs(opts.save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = os.path.join(opts.save_path, f"completion_validation__{timestamp}.json")
    with open(path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = datetime.datetime.now()
    print()
    _banner(f"DARK SIRENS  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    # ── Argument parsing ───────────────────────────────────────────

    optp = ArgumentParser(description=__doc__,
                          formatter_class=RawDescriptionHelpFormatter)

    g = optp.add_argument_group("Data")
    g.add_argument("--gw_path",          required=True)
    g.add_argument("--gwselection_path", required=True)
    g.add_argument("--survey_path",      default=None, nargs="+", metavar="PATH",
                   help=("Galaxy survey catalog(s). One path = current "
                         "single-catalog behaviour. Multiple paths define a "
                         "K-catalog dark-siren mixture: the redshift prior "
                         "becomes sum_k w_k p_k(z) with per-catalog survey "
                         "blocks (_c{k} labels) and sampled stick weights "
                         "fcat_2..fcat_K (dark_sirens only)."))
    g.add_argument("--save_path",        default="./")

    g = optp.add_argument_group("Physical model")
    g.add_argument("--universe_model", default="spectral_sirens",
                   choices=["spectral_sirens", "spectral_sirens_wl", "dark_sirens",
                            "dark_sirens_complete", "bright_sirens"])
    g.add_argument(
        "--sky_model", default="isotropic",
        choices=["isotropic", "dipole", "sphere_gp", "sphere_gp_z", "overdensity_gp",
                 "multipole", "multipole_l3"],
        help=(
            "Sky distribution of the source rate. 'isotropic' (default) is the "
            "null; 'dipole' (Isi, Farr & Varma 2023) and 'sphere_gp' "
            "(log-Gaussian random field, Essick et al. 2023) are angular g(n). "
            "'sphere_gp_z' is a (sphere x z) GP normalised per z-shell "
            "(directional anisotropy evolving with distance); 'overdensity_gp' "
            "is the same field normalised over the comoving volume (full 3-D "
            "clustering, use with gamma fixed). 'multipole'/'multipole_l3' are "
            "low-order spherical-harmonic expansions g=1+sum a_lm Y_lm (l<=2/3) "
            "-- the perturbative, sharply-constrained choice for a small "
            "deviation, giving the angular power spectrum C_l. All compared to "
            "isotropy by evidence; forced to 'isotropic' for bright_sirens."
        ),
    )
    g.add_argument(
        "--mark_model", default="none", choices=["none", "loglinear"],
        help=(
            "Marked-host model (dark_sirens): reweight catalog galaxies by a "
            "BBH-host efficiency h(m|eta) over per-galaxy marks. 'none' (default) "
            "is the legacy galaxy-count host model. 'loglinear' fits "
            "h=exp(sum_k eta_k m_tilde_k) over the (z-centred) marks selected by "
            "--marks. Measures whether GW hosts prefer high-M*/sSFR/low-Z "
            "galaxies at fixed redshift."
        ),
    )
    g.add_argument(
        "--marks", default=None, metavar="LIST",
        help=(
            "Comma-separated marks for --mark_model loglinear "
            "(subset of: logmstar,logssfr,metallicity,color). Default: all marks "
            "present in the catalog."
        ),
    )
    g.add_argument(
        "--pop_model",
        default="powerlaw+peak",
        help=(
            "Population model composition. Grammar mixtures combine mass "
            "tokens (powerlaw, brokenpowerlaw, peak) with '+' and optional "
            "count prefixes (for example, brokenpowerlaw+2peaks). Sharing is "
            "controlled separately by --shared_beta, --shared_spin, and "
            "--shared_gamma."
        ),
    )
    g.add_argument(
        "--shared_beta",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared beta/pairing distribution; false gives one beta per mass component.",
    )
    g.add_argument(
        "--shared_spin",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared spin distribution; false gives one spin distribution per mass component.",
    )
    g.add_argument(
        "--shared_gamma",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared redshift-evolution gamma; false gives one gamma per mass component.",
    )
    g.add_argument("--fix_population",  type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--fixed_cosmology", "--fix_cosmology", dest="fix_cosmology",
                   type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fix the full cosmology block (H0, Om0, w0, wa). "
                         "--fix_cosmology is a backward-compatible alias."))
    g.add_argument("--fixed_de",        dest="fix_de", type=str_to_bool,
                   default=False, metavar="BOOL",
                   help=("Fix only dark-energy cosmology parameters (w0, wa); "
                         "ignored when --fixed_cosmology is true."))
    g.add_argument("--fix_survey",      type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument(
        "--redshift_prior_barrier",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Controls the internal redshift-prior state optimization barrier. 'auto' keeps "
            "the barrier for ordinary JIT paths but disables the likelihood-internal barrier "
            "for TinyNS JAX rwalk because JAX cannot batch optimization_barrier under vmap. "
            "Use 'on' only for non-vmapped samplers; use 'off' to force vmappable behavior."
        ),
    )
    g.add_argument(
        "--selection_neff_guard",
        choices=["auto", "hard", "soft"],
        default="auto",
        help=(
            "Sparse-selection (Neff <= 5 N_obs) validity guard. 'hard' returns -inf "
            "(nested samplers; the historical behavior). 'soft' replaces the wall with a "
            "steep smooth penalty so gradient-based samplers are not divergence-flagged "
            "on every trajectory that brushes it. 'auto' picks soft for --sampler numpyro "
            "and hard otherwise. Check the posterior clears the Neff boundary post hoc "
            "whenever the soft guard ran."
        ),
    )
    g.add_argument("--prior_overrides", default=None, metavar="JSON")
    g.add_argument("--fixed_parameter_values", default=None, metavar="JSON")
    g.add_argument("--counterpart", nargs="+", metavar="RA_DEC_Z",
                   help=("Bright-siren counterpart RA DEC Z triplet(s), ordered by event; "
                         "angles are radians."))
    g.add_argument("--counterpart_dz", type=float, default=1.0e-4,
                   help="Gaussian redshift uncertainty for --counterpart.")
    g.add_argument("--counterpart_nside", type=int, default=1,
                   help="HEALPix NSIDE for the synthetic bright-siren counterpart catalog.")
    g.add_argument("--bright_siren_sky_marginalized", type=str_to_bool, default=False, metavar="BOOL",
                   help=("For bright_sirens, ignore the counterpart sky-pixel gate and "
                         "apply only the counterpart redshift prior."))
    g.add_argument("--complete_empty_pixel_policy", default="zero",
                   choices=["zero", "volume"],
                   help=("Policy for genuinely empty pixels in complete-catalog models: "
                         "'zero' is the formal complete-catalog default; "
                         "'volume' preserves the historical volume-prior robustness approximation."))

    g = optp.add_argument_group("Catalog")
    g.add_argument("--use_LSS",      type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--validate_completion", type=str_to_bool, default=False, metavar="BOOL",
                   help=("Run a dry-run completion clipping diagnostic, save JSON under "
                         "--save_path, and exit before building the likelihood."))
    g.add_argument("--completion_validation_pixels", type=int, default=64, metavar="N",
                   help="Maximum number of unique catalog pixels to inspect in --validate_completion.")
    g.add_argument("--lss_completion", default=None, nargs="+", metavar="PATH",
                   help=("Path to a precomputed LSS-conditioned lognormal completion file "
                         "(/lss_completion/logq_map), built offline by "
                         "darksirens-build-lognormal-completion. Replaces the legacy "
                         "max(1+b*delta_g,0) missing-galaxy factor for dark_sirens. If unset, "
                         "an in-catalog /lss_completion group (if present) is used automatically. "
                         "For a multi-catalog mixture pass 0, 1, or n_catalogs paths, "
                         "positionally aligned with --survey_path (use \"\" as a placeholder "
                         "for a catalog with no external completion)."))
    g.add_argument("--lss_marginalize", type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fully-Bayesian marginalisation of the GW likelihood over the "
                         "Q_LSS ENSEMBLE: logL = logsumexp_m logL(Q_m) - log M, instead of "
                         "the deterministic posterior-mean Q. Requires --lss_completion to "
                         "point at a file built with members "
                         "(darksirens-build-lognormal-completion --n-members M>0); "
                         "dark_sirens only. Off (default) = deterministic Q, unchanged."))

    g = optp.add_argument_group("Sampler")
    g.add_argument("--sampler",      required=True, choices=["tinyns", "dynesty", "numpyro"])
    g.add_argument("--nlive",        type=int,   default=1000)
    g.add_argument("--dlogz",        type=float, default=0.1)
    g.add_argument("--max_samples",  type=int,   default=1_000_000,
                   help="Max call/iteration budget for nested samplers "
                        "(dynesty call cap, tinyns iteration cap); 0 = unlimited.")
    add_tinyns_arguments(g, bool_type=str_to_bool)
    g.add_argument("--nuts_warmup",  type=int,   default=500)
    g.add_argument("--nuts_samples", type=int,   default=1000)
    g.add_argument("--nuts_chains",  type=int,   default=1)
    g.add_argument("--nuts_target_accept", type=float, default=0.8)
    g.add_argument("--nuts_max_tree_depth", type=int, default=10)
    g.add_argument("--nuts_chain_method", default="sequential", choices=["sequential", "parallel", "vectorized"])
    g.add_argument("--nuts_init_tries", type=int, default=32)
    g.add_argument("--nuts_init_seed_offset", type=int, default=100_000)
    g.add_argument("--seed",         type=int,   default=22)
    g.add_argument("--show_progress",type=str_to_bool, default=True, metavar="BOOL")
    g.add_argument("--dynesty_diagnostics", type=str_to_bool, default=False, metavar="BOOL",
                   help="Write dynesty runplot/traceplot PDFs every 10 minutes to "
                        "<save_path>/dynesty_diagnostics/. Only used with --sampler dynesty.")

    g = optp.add_argument_group("Performance")
    g.add_argument("--sel_batch_size", type=int, default=None, metavar="N")
    g.add_argument("--drop_full_catalog", type=str_to_bool, default=False, metavar="BOOL",
                   help="Discard the dense full-sky (npix, n_max_gals) galaxy arrays after "
                        "compacting to inference pixels, keeping only the compact PE/selection "
                        "views on device. Cuts startup GPU memory drastically for large nside. "
                        "Incompatible with --use_LSS and bright-siren models, which need the "
                        "full-sky rows.")
    g.add_argument("--norm_nmass", type=int, default=None, metavar="N",
                   help="Mass-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_MASS).")
    g.add_argument("--norm_nq", type=int, default=None, metavar="N",
                   help="Mass-ratio-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_Q).")
    g.add_argument("--norm_nchi", type=int, default=None, metavar="N",
                   help="Spin-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_CHI).")
    g.add_argument("--kernel_gl_nodes", type=int, default=None, metavar="N",
                   help="Gauss-Legendre nodes for the per-galaxy kernel normalisation Z_i "
                        "(default 24). Spectroscopic catalogs (sigma_eff <~ 5e-3) are exact "
                        "to likelihood precision at 4-8 nodes and the quadrature dominates "
                        "wide-sky dark-siren runs; do NOT reduce for broad photo-z kernels.")

    g = optp.add_argument_group("Lensing")
    g.add_argument("--lensing_wl_model", choices=["lognormal", "tabulated"], default="lognormal",
                   help="Weak-lensing magnification PDF model "
                        "(used only with --universe_model spectral_sirens_wl).")
    g.add_argument("--lensing_wl_a", type=float, default=4e-3,
                   help="Lognormal WL variance amplitude: s^2(z) = a*z^b. "
                        "Default 4e-3 ~ Takahashi+11 fit at z<2.")
    g.add_argument("--lensing_wl_b", type=float, default=1.5,
                   help="Lognormal WL variance slope.")
    g.add_argument("--lensing_wl_table_path", type=str, default=None,
                   help="Path to HDF5 table of log p_WL(mu|z) "
                        "(used with --lensing_wl_model tabulated).")

    opts = optp.parse_args()

    # ── Multitracer survey-path normalization ──────────────────────
    # --survey_path / --lss_completion accept multiple values (nargs="+"); a
    # single path reproduces the legacy single-catalog behaviour exactly.  We
    # keep the aligned lists (opts.survey_paths / opts.lss_completions) for the
    # mixture loader and re-scalarize opts.survey_path / opts.lss_completion so
    # every single-catalog reader (loaders, catalogs/lss, io/results) is
    # untouched.
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    survey_paths = _as_list(opts.survey_path)
    opts.survey_paths = survey_paths
    opts.n_catalogs = max(1, len(survey_paths))

    lss_values = _as_list(opts.lss_completion)
    if opts.n_catalogs == 1:
        valid_counts = (0, 1)
    else:
        # With K catalogs the completion tables MUST be spelled out per catalog:
        # broadcasting one table across catalogs would silently apply e.g. a
        # galaxy Q table to the AGN catalog whenever the nsides happen to match.
        valid_counts = (0, opts.n_catalogs)
    if len(lss_values) not in valid_counts:
        _fatal(
            "--lss_completion must have "
            + ("0 or 1 entries" if opts.n_catalogs == 1
               else f"0 or exactly n_catalogs={opts.n_catalogs} entries")
            + f" (got {len(lss_values)}); use \"\" as a placeholder for a "
            "catalog with no external completion."
        )
    opts.lss_completions = [v if v not in (None, "") else None for v in lss_values]
    opts.survey_path = survey_paths[0] if survey_paths else None
    opts.lss_completion = opts.lss_completions[0] if opts.lss_completions else None

    if opts.n_catalogs >= 2:
        if opts.universe_model != "dark_sirens":
            _fatal("Multiple --survey_path catalogs (a K>=2 mixture) require "
                   "--universe_model dark_sirens.")
        if getattr(opts, "use_LSS", False):
            _fatal("--use_LSS is not supported with a multi-catalog mixture.")
        if getattr(opts, "lss_marginalize", False):
            _fatal("--lss_marginalize is not supported with a multi-catalog mixture.")
        if getattr(opts, "mark_model", "none") not in (None, "none"):
            _fatal("--mark_model is not supported with a multi-catalog mixture.")
        if opts.counterpart is not None:
            _fatal("--counterpart is not supported with a multi-catalog mixture.")
        if getattr(opts, "validate_completion", False):
            _fatal("--validate_completion is not supported with a multi-catalog mixture.")

    # Persist the canonical names in settings while keeping opts.fix_cosmology
    # for backward-compatible internal callers and saved metadata.
    opts.fixed_cosmology = bool(opts.fix_cosmology)
    opts.fixed_de = bool(opts.fix_de)
    opts._fixed_de_superseded = bool(opts.fix_cosmology and opts.fix_de)
    if opts._fixed_de_superseded:
        opts.fix_de = False
        opts.fixed_de = False

    try:
        configure_normalization_grids(
            n_mass=opts.norm_nmass,
            n_q=opts.norm_nq,
            n_chi=opts.norm_nchi,
        )
    except ValueError as e:
        _fatal(str(e))

    if opts.kernel_gl_nodes is not None:
        from darksirens.redshift.catalog import configure_kernel_quadrature
        if opts.kernel_gl_nodes < 2:
            _fatal("--kernel_gl_nodes must be >= 2")
        configure_kernel_quadrature(opts.kernel_gl_nodes)

    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    opts.prior_overrides        = prior_overrides if prior_overrides else None
    opts.fixed_parameter_values = (
        fixed_parameter_values if fixed_parameter_values else None
    )
    opts.counterpart            = parse_counterpart_arg(opts.counterpart)

    if opts.sampler == "tinyns":
        try:
            build_tinyns_config(opts)
        except ValueError as e:
            _fatal(str(e))

    opts.materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    opts.redshift_prior_barrier_resolved = _redshift_prior_materialization_reason(
        opts, opts.materialize_redshift_prior_state
    )
    if (
        opts.redshift_prior_barrier == "auto"
        and not opts.materialize_redshift_prior_state
    ):
        print(
            "  [i] Disabling likelihood-internal redshift-prior optimization_barrier "
            f"({opts.redshift_prior_barrier_resolved})."
        )

    guard_mode = getattr(opts, "selection_neff_guard", "auto")
    opts.selection_neff_soft_guard = (
        guard_mode == "soft"
        or (guard_mode == "auto" and opts.sampler == "numpyro")
    )
    if opts.selection_neff_soft_guard:
        print(
            "  [i] Sparse-selection Neff guard: SOFT (smooth wall for gradient-based "
            f"sampling; mode={guard_mode}). Verify the posterior clears the "
            "Neff <= 5 N_obs boundary post hoc."
        )

    if opts.universe_model == "bright_sirens":
        # Bright sirens use a synthetic one-object catalog fixed by the
        # counterpart rather than survey-completion hyperparameters.
        opts.fix_survey = True
        # The sky direction is pinned to the counterpart, so an anisotropic
        # source-rate model is not identifiable; force the isotropic null.
        if getattr(opts, "sky_model", "isotropic") != "isotropic":
            print(
                f"  [!] --sky_model '{opts.sky_model}' is not supported with "
                "universe_model 'bright_sirens'; forcing 'isotropic'."
            )
            opts.sky_model = "isotropic"

    _print_all_cli_options(
        optp,
        opts,
        normalization_grid=normalization_grid_settings().to_dict(),
    )

    # ── Validation ─────────────────────────────────────────────────

    _section("Validating configuration")
    GALAXY_AWARE = {"dark_sirens", "dark_sirens_complete"}

    if opts.universe_model == "bright_sirens" and opts.counterpart is None:
        _fatal("'bright_sirens' requires --counterpart RA DEC Z triplet(s) (angles in radians).")
    if opts.universe_model != "bright_sirens" and opts.counterpart is not None:
        _warn("--counterpart is ignored unless --universe_model bright_sirens.")
    if opts.counterpart_dz <= 0.0:
        _fatal("--counterpart_dz must be positive.")
    if opts.counterpart_nside < 1 or not hp.isnsideok(opts.counterpart_nside):
        _fatal("--counterpart_nside must be a valid positive HEALPix NSIDE.")

    if opts.universe_model in GALAXY_AWARE and not opts.survey_path:
        _fatal(f"'{opts.universe_model}' requires --survey_path.")
    if opts.universe_model not in GALAXY_AWARE and opts.survey_path:
        _warn(f"--survey_path provided but '{opts.universe_model}' does not use it.")
    if getattr(opts, "_fixed_de_superseded", False):
        _warn(
            "--fixed_cosmology supersedes --fixed_de; "
            "dark energy is included in the fixed cosmology block."
        )
    if opts.fix_population and opts.fix_cosmology and opts.fix_survey:
        _warn("All blocks fixed — nothing will be inferred.")

    _ok("Configuration is valid.")
    _end()

    # ── Run configuration printout ─────────────────────────────────

    _section("Run Configuration")
    _row("Universe model",   opts.universe_model)
    if opts.counterpart is not None:
        first_cp = opts.counterpart[0]
        _row("Counterparts", len(opts.counterpart))
        _row("First counterpart", f"ra={first_cp[0]:.8g}, dec={first_cp[1]:.8g}, z={first_cp[2]:.8g}")
        _row("Counterpart dz", opts.counterpart_dz)
        _row("Counterpart nside", opts.counterpart_nside)
        _row("Bright-siren sky marginalized", opts.bright_siren_sky_marginalized)
    _row("Population model", opts.pop_model)
    _row("Shared beta", "yes" if opts.shared_beta else "no")
    _row("Shared spin", "yes" if opts.shared_spin else "no")
    _row("Shared gamma", "yes" if opts.shared_gamma else "no")
    if opts.universe_model in {"dark_sirens_complete", "bright_sirens"}:
        _row("Empty-pixel policy", opts.complete_empty_pixel_policy)
    print("  │")
    _row("Fix cosmology",    "yes" if opts.fix_cosmology  else "no")
    _row(
        "Fix dark energy",
        _format_fixed_dark_energy_summary(opts, fixed_parameter_values),
    )
    _row("Fix population",   "yes" if opts.fix_population else "no")
    _row("Fix survey",       "yes" if opts.fix_survey     else "no")
    _row("Prior overrides",  json.dumps(prior_overrides) if prior_overrides else "none")
    _row("Validate completion", "yes" if opts.validate_completion else "no")
    _row("Redshift prior barrier", opts.redshift_prior_barrier_resolved)
    if fixed_parameter_values:
        for lbl, val in fixed_parameter_values.items():
            _row(f"  fixed: {lbl}", val)
    else:
        _row("Fixed param values", "none")
    print("  │")
    _row("Sampler", opts.sampler)
    if opts.sampler in ("tinyns", "dynesty"):
        _row("  live points", opts.nlive)
        _row("  ΔlogZ stop",  opts.dlogz)
    if opts.sampler == "tinyns":
        cfg = opts.tinyns_resolved_config
        _row("  preset", cfg["preset"])
        _row("  sample", cfg["sample"])
        _row("  kernel", cfg["kernel"])
        _row("  bound", cfg["bound"])
        _row("  rwalk seed", cfg["rwalk_seed"])
        _row("  rwalk proposal", cfg["rwalk_proposal"])
        _row("  walks", cfg["walks"])
        _row("  step scale", cfg["step_scale"])
        _row("  min accepts", cfg["min_accepts"])
        _row("  repl. chains", cfg["replacement_chains"])
        _row("  chain sched.", cfg["replacement_chain_schedule"] or "none")
        _row("  max attempts", cfg["max_attempts"])
        _row("  jax block size", cfg["jax_block_size"])
        _row("  jax vectorized", cfg.get("jax_vectorized"))
        _row("  progress interval", cfg.get("progress_interval"))
        if cfg.get("checkpoint_path") or cfg.get("resume_from") or cfg.get("checkpoint_path_out"):
            _row("  checkpoint", cfg.get("checkpoint_path") or "none")
            _row("  resume from", cfg.get("resume_from") or "none")
            _row("  checkpoint out", cfg.get("checkpoint_path_out") or "none")
    if opts.sampler == "numpyro":
        _row("  warmup", opts.nuts_warmup)
        _row("  samples", opts.nuts_samples)
        _row("  chains", opts.nuts_chains)
        _row("  target accept", opts.nuts_target_accept)
        _row("  init tries", opts.nuts_init_tries)
        _row("  init seed offset", opts.nuts_init_seed_offset)
    _row("  seed", opts.seed)
    print("  │")
    norm_grid = normalization_grid_settings()
    _row("Norm grids", (
        f"mass={norm_grid.n_mass}, q={norm_grid.n_q}, chi={norm_grid.n_chi}"
    ))
    _row("JAX backend", jax.default_backend())
    _row("JAX devices",  ", ".join(str(d) for d in jax.devices()))
    print("  │")
    _row("GW events path",  opts.gw_path)
    _row("Selection path",  opts.gwselection_path)
    if opts.survey_path:
        if opts.n_catalogs == 1:
            _row("Survey path",  opts.survey_path)
        else:
            _row("Catalog mixture", f"{opts.n_catalogs} catalogs")
            for _i, _p in enumerate(opts.survey_paths):
                _row(f"  catalog {_i + 1}", _p)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
    _row("Output root",     opts.save_path)
    if opts.sel_batch_size:
        _row("Sel. batch",   f"{opts.sel_batch_size:,} samples/batch")
    if opts.drop_full_catalog:
        _row("Drop full catalog", "yes (compact views only)")
    _end()

    # ── Load data ──────────────────────────────────────────────────

    _section("Loading data")
    print("  │")
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)

    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    nside   = data.get("nside", "N/A")

    _ok(f"GW posterior samples:   {nEvents} events × {nsamp} samples/event = {nEvents*nsamp:,} total")
    _ok(f"Selection injections:   {int(Ndraw):,} total generated")

    if opts.survey_path:
        ngals_pe  = data.get("ngals_pe",  None)
        ngals_sel = data.get("ngals_sel", None)
        _ok(f"HEALPix nside:          {nside}")
        if ngals_pe  is not None:
            _ok(f"Catalog galaxies (PE pixels):  {int(np.asarray(ngals_pe).sum()):,}")
        if ngals_sel is not None:
            _ok(f"Catalog galaxies (sel pixels): {int(np.asarray(ngals_sel).sum()):,}")
        catalog_memory = data.get("catalog_memory")
        if catalog_memory is not None:
            _ok(
                "Unique catalog pixels:   "
                f"PE {catalog_memory['unique_pe_pixels']:,}, "
                f"selection {catalog_memory['unique_sel_pixels']:,}"
            )
            _ok(
                "Duplicated catalog bytes avoided: "
                f"{catalog_memory['duplicated_catalog_bytes_avoided'] / 1e9:.3f} GB"
            )
            _ok(
                "Max galaxies/unique pixel: "
                f"{catalog_memory['max_galaxies_per_unique_pixel']:,}"
            )

    dg = data.get("delta_g_pix_z")
    if dg is not None:
        gb = np.asarray(dg).nbytes / 1e9
        _ok(f"δ_g field shape:        {np.asarray(dg).shape}  ({gb:.3f} GB)")
    _end()

    if opts.validate_completion:
        _section("Completion validation dry run")
        validation_path = run_completion_validation(
            opts, data, prior_overrides, fixed_parameter_values
        )
        _ok(f"completion_validation JSON → {validation_path}")
        _row("Action", "exiting before likelihood/sampling")
        _end()
        return

    # ── Parameter space ────────────────────────────────────────────

    # Resolve the galaxy marks for the marked-host model (dark_sirens): those
    # present in the catalog, optionally narrowed by --marks.  Stored on opts so
    # the decoder/likelihood see the same ordered list as the parameter space.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS
    _present_marks = tuple(n for n in _MARK_FIELDS if data.get(_MARK_FIELDS[n]) is not None)
    if opts.marks:
        _req = tuple(s.strip() for s in opts.marks.split(",") if s.strip())
        _missing = [m for m in _req if m not in _present_marks]
        if _missing:
            _fatal(f"--marks requested {_missing} but the catalog provides "
                   f"{list(_present_marks)}.")
        opts.mark_names = _req
    else:
        opts.mark_names = _present_marks
    if opts.mark_model != "none" and not opts.mark_names:
        _fatal("--mark_model loglinear requires per-galaxy marks, but the catalog "
               "provides none (expected datasets like LOGMSTAR/LOGSSFR).")

    _section("Building parameter space")
    res = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        fix_de                 = opts.fix_de,
        prior_overrides        = prior_overrides,
        fixed_parameter_values = fixed_parameter_values,
        universe_model         = opts.universe_model,
        shared_beta            = opts.shared_beta,
        shared_spin            = opts.shared_spin,
        shared_gamma           = opts.shared_gamma,
        sky_model              = opts.sky_model,
        mark_model             = opts.mark_model,
        mark_names             = opts.mark_names,
        n_catalogs             = opts.n_catalogs,
    )
    labels, lower_bound, upper_bound = res[0], res[1], res[2]
    n_pop_eff, n_cosmo_eff, n_survey_eff, model_name = res[3], res[7], res[8], res[9]
    fixed_parameter_statuses = res[10]
    prior_kinds = res[11]

    _, _, pop_labels_all, _, _ = pop_model_prior_parser(
        opts.pop_model,
        shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    pop_params_fid  = get_fixed_population_params(
        opts.pop_model,
        shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    prior_transform = make_prior_transform(lower_bound, upper_bound, prior_kinds)

    _ok(f"Parameter space built:  {len(labels)} free dimensions")
    _end()

    _print_parameter_table(
        labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, fixed_parameter_statuses,
        opts.fix_cosmology, opts.fix_de, opts.fix_population, opts.fix_survey,
        pop_params_fid, pop_labels_all,
    )

    # ── Build likelihood ───────────────────────────────────────────

    _section("Building likelihood")
    print("  │  Applying optimization barriers...")
    print("  │  JIT compilation deferred to first call.")
    print("  │")
    likelihood = make_likelihood(
        opts                   = opts,
        data                   = data,
        pop_params_fid         = pop_params_fid,
        fixed_parameter_values = fixed_parameter_values,
    )
    _ok("Likelihood closure ready.")
    _end()

    # ── Sampling ───────────────────────────────────────────────────

    _section(f"Sampling  [{opts.sampler.upper()}]")
    tinyns_cfg = getattr(opts, "tinyns_resolved_config", {})
    sampler_info = {
        "tinyns": (
            f"nlive={opts.nlive}  dlogz={opts.dlogz}  "
            f"sample={tinyns_cfg.get('sample')}  kernel={tinyns_cfg.get('kernel')}  "
            f"seed={opts.seed}"
        ),
        "dynesty": f"nlive={opts.nlive}  dlogz={opts.dlogz}  seed={opts.seed}",
        "numpyro": f"warmup={opts.nuts_warmup}  samples={opts.nuts_samples}  chains={opts.nuts_chains}  seed={opts.seed}",
    }
    _row("Configuration", sampler_info[opts.sampler])
    _row("ndim", len(labels))
    print("  │")

    t_sample_start = datetime.datetime.now()
    results = run_sampler(
        method=opts.sampler, likelihood=likelihood,
        prior_transform=prior_transform, labels=labels,
        lower_bound=lower_bound, upper_bound=upper_bound, opts=opts,
        prior_kinds=prior_kinds,
    )
    t_sample_end  = datetime.datetime.now()
    wall_sampling = t_sample_end - t_sample_start

    if results is None or results.get("samples") is None:
        _fatal("Sampler returned no results.")

    n_samples = np.asarray(results["samples"]).shape[0]
    print("  │")
    _ok(f"Sampling complete.  Wall time: {wall_sampling}")
    _ok(f"Posterior samples:  {n_samples:,}")

    logZ    = results.get("logZ")
    logZerr = results.get("logZerr")
    if logZ is not None:
        zerr = float(logZerr) if logZerr is not None else float("nan")
        _ok(f"log Z = {float(logZ):.3f} ± {zerr:.3f}")
    _end()

    # ── Save outputs ───────────────────────────────────────────────

    t_end     = datetime.datetime.now()
    timestamp = t_end.strftime("%Y-%m-%dT%H-%M-%S")
    run_name  = f"{opts.pop_model}__{opts.universe_model}__{opts.sampler}__{timestamp}"
    run_dir   = os.path.join(opts.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    meta = {
        "n_events":         nEvents,
        "n_samp_per_event": nsamp,
        "n_draw":           int(Ndraw),
        "n_pop_eff":        n_pop_eff,
        "n_cosmo_eff":      n_cosmo_eff,
        "n_survey_eff":     n_survey_eff,
        "model_name":       model_name,
        "total_runtime":    str(t_end - t_start),
        "sampling_runtime": str(wall_sampling),
        "timestamp":        timestamp,
    }

    _section("Saving outputs")
    _row("Run directory", run_dir)
    print("  │")

    hdf5_path = save_results_hdf5(
        results, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, opts, meta,
        prior_kinds=prior_kinds,
    )
    _ok(f"results.hdf5   →  {hdf5_path}")

    json_path = save_settings_json(
        opts, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, meta,
    )
    _ok(f"settings.json  →  {json_path}")

    if not labels:
        # 0 free parameters (all blocks fixed) — there is nothing to plot.
        _ok("0 free parameters (all fixed) - corner plot skipped.")
    else:
        print("  │  Generating corner plot...")
        try:
            import matplotlib.pyplot as _plt
            from darksirens.utils.plotting import make_production_corner, make_latent_summary

            # Headline corner: cosmology + physical hyperparameters + survey.
            # For high-dimensional models (GP / gppop) the dozens of xi_* latents
            # are excluded for legibility and summarized separately below.
            fig = make_production_corner(results["samples"], labels)
            corner_path = os.path.join(run_dir, "corner.pdf")
            fig.savefig(corner_path, bbox_inches="tight", dpi=200)
            _plt.close(fig)
            _ok(f"corner.pdf     →  {corner_path}")

            fig_lat = make_latent_summary(results["samples"], labels)
            if fig_lat is not None:
                latents_path = os.path.join(run_dir, "latents.pdf")
                fig_lat.savefig(latents_path, bbox_inches="tight", dpi=200)
                _plt.close(fig_lat)
                _ok(f"latents.pdf    →  {latents_path}")
        except ModuleNotFoundError as e:
            _warn(f"Corner plot skipped; optional plotting dependency is missing: {e.name}")
        except Exception as e:
            _warn(f"Corner plot failed: {e}")

    _end()

    print()
    _banner(f"DONE  │  total wall time {t_end - t_start}")
    print()


if __name__ == "__main__":
    main()
