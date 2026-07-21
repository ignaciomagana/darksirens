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
    --fix_cosmology     true \

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
import math
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import h5py

from argparse import ArgumentParser, ArgumentTypeError, RawDescriptionHelpFormatter

from darksirens.gw.populations import get_fixed_population_params, pop_model_prior_parser
from darksirens.gw.populations.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.inference.validation import validate_multitracer_run
from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.likelihood.factory import make_likelihood
from darksirens.likelihood.block_sizing import (
    block_size_arg,
    resolve_block_sizes,
    format_block_size_request,
)
from darksirens.redshift.completion import build_pixel_kde_cache, completion_clip_diagnostics
from darksirens.redshift.grid import zMax as _REDSHIFT_GRID_ZMAX
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
    DeprecatedSpellingAction,
    resolve_redshift_prior_barrier,
    resolve_selection_neff_guard,
    run_cli,
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
    rule = f"  │    {'─' * 24} {'─' * 12}  {'─' * 12}  {'─' * 20}"

    _row("Sampled parameters", "")
    print(f"  │    {'Parameter':<24} {'Lower':>12}  {'Upper':>12}  Status")
    print(rule)
    for label, lo, hi in zip(labels, lower_bound, upper_bound):
        note = "← overridden" if label in prior_overrides else ""
        print(f"  │    {label:<24} {lo:>12.4g}  {hi:>12.4g}  {note}")

    print(rule)
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

    print(rule)
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

    print(rule)

    n_free = sum(1 for lo, hi in zip(lower_bound, upper_bound) if lo != hi)
    n_fix_ind   = len(fixed_parameter_values)
    n_fix_block = ((len(COSMO_FID) if fix_cosmology else len(DE_FID) if fix_de else 0)
                   + (len(pop_labels_all) if fix_population else 0)
                   + (len(SURVEY_FID) if fix_survey else 0))
    _row("Free (sampled)",      n_free)
    if n_fix_ind:   _row("Fixed individually", n_fix_ind)
    if n_fix_block: _row("Fixed (block)",      n_fix_block)
    _row("Total in coord vec",  len(labels))
    _end()


def resolve_survey_z_depth(cli_override, file_attr, zmax=None):
    """Resolve one catalog's ``z_depth``: CLI override > per-catalog file attr > None.

    Precedence: ``cli_override`` (the ``--survey_z_depth`` value, applied to
    every catalog when set) wins over ``file_attr`` (the per-catalog
    ``f.attrs['z_depth']`` written by ``darksirens_pixelate --z_depth``);
    ``None`` means the legacy full-grid missing-galaxy budget
    (:mod:`darksirens.redshift.completion`).  A resolved value above the
    redshift grid ``zMax`` (:data:`darksirens.redshift.grid.zMax`) is
    equivalent to no bound at all, so it is clamped to ``zMax`` with a
    warning rather than silently accepted or rejected.
    """
    if zmax is None:
        zmax = _REDSHIFT_GRID_ZMAX
    z = cli_override if cli_override is not None else file_attr
    if z is None:
        return None
    z = float(z)
    if not math.isfinite(z) or z <= 0.0:
        # NaN slips through every comparison guard (NaN > zmax is False) and
        # would silently zero the whole missing-galaxy budget downstream
        # (depth_mask = zgrid <= NaN is all-False); 0/negative likewise.
        raise ValueError(
            f"survey z_depth must be a finite positive redshift; got {z!r}."
        )
    if z > zmax:
        msg = (
            f"z_depth={z:g} exceeds the redshift grid zMax={zmax:g}; clamping "
            "to zMax (equivalent to the legacy full-grid missing-galaxy budget)."
        )
        warnings.warn(msg)
        _warn(msg)
        z = float(zmax)
    return z


def _report_survey_z_depth(label: str, resolved) -> None:
    """Print one loud config line for a catalog's resolved ``z_depth``."""
    _ok(
        f"{label}: "
        + (
            f"{resolved:g} (bounds missing-galaxy budget to z <= {resolved:g})"
            if resolved is not None
            else "None -> full-grid budget (legacy)"
        )
    )


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
    suffix: str = "",
) -> str:
    """Save a dry-run completion clipping diagnostic and return its path.

    ``suffix`` distinguishes per-catalog outputs for a K>=2 mixture run
    (e.g. ``"_c2"`` -> ``completion_validation_c2__<ts>.json``).
    """
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

    # The completion code indexes the KDE cache directly by row (cache row k ==
    # unique_pixels[k]); the dense pixel_to_cache_idx lookup is no longer used.
    dN_obs_kde, _ = build_pixel_kde_cache(
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
    # NOTE: this dry-run deliberately leaves z_depth at its default (None):
    # completion_clip_diagnostics reports clip fractions of the completeness
    # ratio only, which are z_depth-independent, so threading the resolved
    # depth here would change nothing. Do not "fix" this without a reason.
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
        pixel_to_cache_idx=None,
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
    path = os.path.join(
        opts.save_path, f"completion_validation{suffix}__{timestamp}.json"
    )
    with open(path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    return path


def run_completion_validation_multitracer(
    opts,
    data: dict,
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> list:
    """Per-catalog completion validation for a K>=2 mixture run.

    Each catalog gets the SAME dry-run diagnostic as a single-catalog run:
    the full survey rows are re-read host-side (bundles keep only compact
    views) and paired with that catalog's own per-sample pixel indices,
    stashed on the bundle by ``load_multitracer_catalog_bundles`` when
    ``--validate_completion`` is set.  Returns one JSON path per catalog,
    suffixed ``_c{k}``.
    """
    from darksirens.inference.loaders import load_survey

    bundles = data.get("catalogs") or []
    if len(bundles) != len(opts.survey_paths):
        _fatal(
            "Per-catalog completion validation needs one loaded bundle per "
            f"--survey_path (got {len(bundles)} bundles for "
            f"{len(opts.survey_paths)} paths)."
        )
    paths = []
    for k, (survey_path, bundle) in enumerate(
        zip(opts.survey_paths, bundles), start=1
    ):
        if bundle.get("pixels_pe_full") is None or bundle.get("pixels_sel_full") is None:
            _fatal(
                "Per-catalog completion validation requires the per-sample "
                f"pixel indices for catalog {k}; the bundle loader stashes "
                "them only when --validate_completion is set."
            )
        _, ngals, zgals, dzgals, wgals, _ = load_survey(survey_path, to_device=False)
        data_k = dict(
            zgals_catalog=zgals,
            dzgals_catalog=dzgals,
            wgals_catalog=wgals,
            ngals_catalog=ngals,
            pixels_pe=bundle["pixels_pe_full"],
            pixels_sel=bundle["pixels_sel_full"],
            apix=bundle["apix"],
            n_pix_catalog=bundle["n_pix_catalog"],
        )
        paths.append(
            run_completion_validation(
                opts, data_k, prior_overrides, fixed_parameter_values,
                suffix=f"_c{k}",
            )
        )
    return paths


def format_selection_guard_summary(
    guard_mode, sampler, soft_guard, max_likelihood_variance
):
    """One-line, always-printed summary of the resolved sparse-selection guard.

    Pure (string in, string out) so it is unit-testable without a full CLI
    run.  ``guard_mode`` is the raw ``--selection_neff_guard`` value
    ("auto"/"hard"/"soft"); ``soft_guard`` is the resolved boolean (which the
    CLI derives from ``guard_mode`` and ``sampler``).  The returned line is
    printed with the surrounding block's ``  [i] `` prefix.
    """
    resolved = "SOFT" if soft_guard else "HARD"
    why = f"auto: {sampler}" if guard_mode == "auto" else "explicit"
    criterion = (
        "sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel > "
        f"{max_likelihood_variance} (--max_likelihood_variance)"
    )
    if soft_guard:
        # SOFT never returns -inf; it replaces the wall with a smooth penalty
        # and the posterior must be checked to clear the boundary post hoc.
        action = f"smooth penalty wall as {criterion}; verify the posterior clears it post hoc"
    else:
        action = f"-inf when {criterion}"
    return (
        f"selection guard: {resolved} ({why}) — {action}; "
        "Vitale floor Neff > 5*N_obs always applies"
    )


def _universe_model_arg(value):
    """argparse ``type`` for --universe_model.

    The weak-lensing (``spectral_sirens_wl``) universe model moved to the
    ``darksirens_inference_lensing`` CLI. argparse runs ``type`` before
    ``choices``, so intercept the retired value here and raise a migration
    hint instead of the generic "invalid choice" message.
    """
    if value == "spectral_sirens_wl":
        raise ArgumentTypeError(
            "universe_model 'spectral_sirens_wl' moved to the lensing CLI; run "
            "`darksirens_inference_lensing --cluster_mode off "
            "--wl_backend lognormal|tabulated ...` instead."
        )
    return value


# ── Main ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ParameterSpace:
    """What the likelihood/sampling/save phases need from the built space."""
    labels: object
    lower_bound: object
    upper_bound: object
    prior_kinds: object
    prior_transform: object
    pop_params_fid: object
    n_pop_eff: object
    n_cosmo_eff: object
    n_survey_eff: object
    model_name: object


def build_parser():
    """Construct the darksirens_inference ArgumentParser.

    Extracted from main() so tooling can build the parser without running a
    full inference; PR 3 depends on this contract (returns the parser).
    """
    optp = ArgumentParser(description=__doc__,
                          formatter_class=RawDescriptionHelpFormatter)

    g = optp.add_argument_group("Data")
    g.add_argument("--gw_path",          default=None,
                   help=("gwcat PE posterior-samples file (format_version "
                         "gwcat-1.0 or gwcat-pe-2.0; a 2.0 file must use "
                         "spin_basis=chieff). Exactly one of "
                         "--gw_path / --gw_flows_path is required."))
    g.add_argument("--gw_flows_path",    default=None,
                   help=("Directory of per-event normalizing-flow checkpoints "
                         "(<EVENT>/<EVENT>_flow.npz) replacing stored PE "
                         "samples. Mutually exclusive with --gw_path; "
                         "spectral_sirens only."))
    g.add_argument("--gwselection_path", default=None,
                   help=("gwcat selection/injection file (format_version "
                         "gwcat-selection-1.0 or gwcat-selection-2.0; a 2.0 "
                         "file must use spin_basis=chieff). Exactly one of "
                         "--gwselection_path / --pdet_flow_path is required."))
    g.add_argument("--survey_path",      default=None, nargs="+", metavar="PATH",
                   help=("Galaxy survey catalog(s). One path = current "
                         "single-catalog behaviour. Multiple paths define a "
                         "K-catalog dark-siren mixture: the redshift prior "
                         "becomes sum_k w_k p_k(z) with per-catalog survey "
                         "blocks (_c{k} labels) and sampled stick weights "
                         "fcat_2..fcat_K (dark_sirens only)."))
    g.add_argument("--save_path",        default="./")

    g = optp.add_argument_group("Flow surrogates (with --gw_flows_path)")
    g.add_argument("--flows_nsamp", type=int, default=16384,
                   help=("Population draws J per event per likelihood call "
                         "(common random numbers, event-windowed proposal). "
                         "Per-event ESS is typically 1-5%% of J on real "
                         "events; the total-lnL variance guard "
                         "(--max_likelihood_variance) rejects undersampled "
                         "settings."))
    g.add_argument("--flows_seed", type=int, default=42,
                   help="Seed of the fixed base-uniform array (CRN).")
    g.add_argument("--flows_pattern", default="*/*_flow.npz",
                   help="Checkpoint glob relative to --gw_flows_path.")
    g.add_argument("--flows_on_mismatch", default="error",
                   choices=["error", "skip", "load"],
                   help=("Structural-check policy for checkpoints that do not "
                         "match the installed flowjax (version drift)."))
    g.add_argument("--flows_chieff_amax", type=float, default=0.99,
                   help="amax of the 1-D isotropic chi_eff PE prior.")
    g.add_argument("--flows_pe_cosmology", default="67.74,0.3089",
                   help=("'H0,Om0' of the PE prior cosmology (UniformSourceFrame "
                         "distance prior), matching the PE release."))
    g.add_argument("--flows_grid_nm", type=int, default=512,
                   help="m1 cells of the (m1, q) population sampling grid.")
    g.add_argument("--flows_grid_nq", type=int, default=256,
                   help="q cells of the (m1, q) population sampling grid.")
    g.add_argument("--flows_support_margin", type=float, default=0.25,
                   help=("Per-event proposal window: expand each flow's "
                         "sampled parameter range by this fraction per side "
                         "(exactly corrected in the proposal density)."))
    g.add_argument("--flows_support_nsamples", type=int, default=4096,
                   help="Flow draws used to measure each event's support box.")

    g = optp.add_argument_group(
        "Selection-function emulator (alternative to --gwselection_path)")
    g.add_argument("--pdet_flow_path", default=None,
                   help=("NF P_det emulator checkpoint (npz: found-injection "
                         "flow + xi). Drives the selection integral through "
                         "pseudo-injections drawn once at load time. Exactly "
                         "one of --gwselection_path / --pdet_flow_path is "
                         "required."))
    g.add_argument("--pdet_nsamp", type=int, default=1_000_000,
                   help=("Pseudo-injections drawn from the emulator flow at "
                         "load (Ndraw = nsamp / xi)."))
    g.add_argument("--pdet_seed", type=int, default=42,
                   help="Seed of the emulator pseudo-injection draw.")
    g.add_argument("--pdet_cosmology", default="67.9,0.3065",
                   help=("'H0,Om0' of the injection-campaign cosmology used "
                         "for z->dL and the Jacobian; must match the campaign "
                         "the emulator was trained on."))
    g.add_argument("--pdet_chieff_amax", type=float, default=0.99,
                   help=("Reference amax of the chi_eff spin-prior swap; keep "
                         "equal to --flows_chieff_amax and the injection-file "
                         "convention."))

    g = optp.add_argument_group("Physical model")
    g.add_argument("--universe_model", default="spectral_sirens",
                   type=_universe_model_arg,
                   choices=["spectral_sirens", "dark_sirens",
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
    g.add_argument("--fix_cosmology", "--fixed_cosmology", dest="fix_cosmology",
                   action=DeprecatedSpellingAction, deprecated=["--fixed_cosmology"],
                   type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fix the full cosmology block (H0, Om0, w0, wa). "
                         "--fixed_cosmology is a deprecated alias."))
    g.add_argument("--fix_de",          "--fixed_de", dest="fix_de",
                   action=DeprecatedSpellingAction, deprecated=["--fixed_de"],
                   type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fix only dark-energy cosmology parameters (w0, wa); "
                         "ignored when --fix_cosmology is true. "
                         "--fixed_de is a deprecated alias."))
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
    g.add_argument(
        "--max_likelihood_variance", type=float, default=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        help=("Cap on the Monte-Carlo variance of the TOTAL log-likelihood "
              "estimator, sigma^2_lnL = sum_i sigma^2_i + N_obs^2/Neff_sel "
              "(Essick & Farr 2022; Talbot & Golomb 2023, arXiv:2304.06138; the "
              "GWTC-4.0/5.0 population-analysis criterion, default 1.0). Proposals "
              "exceeding it are guarded (hard -inf or the soft wall per "
              "--selection_neff_guard). The Vitale 5 N_obs mean floor always applies."))
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
    g.add_argument("--use_lss", "--use_LSS", dest="use_LSS",
                   action=DeprecatedSpellingAction, deprecated=["--use_LSS"],
                   type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--catalog_sky_weighting", default=None,
                   choices=["conditional", "field"],
                   help=("Catalog redshift-prior normalization convention (dark_sirens). "
                         "'conditional' normalizes each pixel by its own "
                         "Z[pix]=N_obs+N_miss -- bit-identical to the pre-existing behaviour "
                         "and the per-pixel z-shape estimand. 'field' normalizes by the "
                         "survey-GLOBAL Z(theta)=Sum_all-pixels[N_obs+N_miss], so a K>=2 "
                         "mixture weight fcat_k measures the host FRACTION (number-density / "
                         "sky-clustering contrast), empty pixels carry weight proportional to "
                         "their n0-implied missing count, and log10n0 becomes informative. "
                         "Unset auto-resolves to the coherent convention for the run's K: "
                         "conditional for a single catalog, field for a K>=2 mixture; for "
                         "dark_sirens the degenerate explicit combinations (field at K=1, "
                         "conditional at K>=2) are fatal. "
                         "Field requires the full-sky catalog (incompatible with "
                         "--drop_full_catalog); composes with --lss_completion, --use_lss "
                         "and --mark_model (the global normalizer carries the same "
                         "modulated missing budget as the numerator, per member under "
                         "--lss_marginalize); incompatible with universe models other than "
                         "dark_sirens/dark_sirens_complete."))
    g.add_argument("--survey_z_depth", type=float, default=None, metavar="Z",
                   help=("Redshift depth bounding the completion missing-galaxy budget to "
                         "z <= z_depth instead of the full [0, DARKSIRENS_ZMAX] grid, avoiding "
                         "the dilution of the catalog term by galaxies the survey was never "
                         "designed to detect. Applies to ALL catalogs and overrides any "
                         "per-catalog f.attrs['z_depth'] written by darksirens_pixelate "
                         "--z_depth. Default None: per-catalog file attr if present, else the "
                         "legacy full-grid budget (bit-identical to pre-existing behaviour)."))
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
                         "dark_sirens only. Off (default) = deterministic Q, unchanged. "
                         "For a K>=2 mixture the marginalisation uses ONE SHARED member "
                         "index (member m of every catalog must sample the SAME LSS "
                         "realization): build the per-catalog ensembles jointly with "
                         "darksirens_build_joint_lognormal_completion (equal --n-members), "
                         "or the run aborts. "
                         "The matched-realizations requirement is verified from each "
                         "file's /lss_completion realization_set_id (they must be equal "
                         "and non-null); darksirens_build_joint_lognormal_completion "
                         "stamps one shared id across the K files, or pass "
                         "--allow_unverified_shared_lss_members to accept the "
                         "independent-fields approximation."))
    g.add_argument("--allow_unverified_shared_lss_members",
                   type=str_to_bool, nargs="?", const=True, default=False, metavar="BOOL",
                   help=("Bypass the K>=2 --lss_marginalize provenance check that the "
                         "per-catalog Q_LSS ensembles share a matched realization set "
                         "(equal, non-null realization_set_id). With this flag, catalogs "
                         "whose ensembles were built independently are marginalized "
                         "anyway, treating member m of each catalog as an INDEPENDENT LSS "
                         "draw -- an independent-fields product prior, NOT the matched "
                         "shared-field prior the estimator assumes. Emits a loud warning "
                         "and proceeds; use only if you understand the approximation."))

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
    g.add_argument("--sampler_preflight", choices=["on", "off"], default="on",
                   help="Probe 32 prior draws before nested sampling (dynesty/tinyns) and "
                        "fail fast if the likelihood is -inf on all of them (usually the "
                        "selection variance guard); 'off' skips the probe.")

    g = optp.add_argument_group("Performance")
    g.add_argument("--sel_batch_size", type=block_size_arg, default="auto",
                   metavar="N|auto|off",
                   help="Injections per selection-integral chunk. 'auto' (default) "
                        "sizes the block from probed free device memory after the "
                        "data load; 'off'/'none'/'0' forces a single pass (the "
                        "historical default); a positive integer pins the block.")
    g.add_argument("--pe_event_block", type=block_size_arg, default="auto",
                   metavar="N|auto|off",
                   help="Events per PE-reduction chunk. 'auto' (default) sizes from "
                        "free memory; 'off'/'none'/'0' keeps all events in one "
                        "vectorized block (fastest); a positive integer (e.g. 32) "
                        "bounds memory for dense dark-siren catalogs.")
    g.add_argument("--drop_full_catalog", type=str_to_bool, default=False, metavar="BOOL",
                   help="Discard the dense full-sky (npix, n_max_gals) galaxy arrays after "
                        "compacting to inference pixels, keeping only the compact PE/selection "
                        "views on device. Cuts startup GPU memory drastically for large nside. "
                        "Incompatible with --use_lss and bright-siren models, which need the "
                        "full-sky rows.")
    g.add_argument("--norm_nmass", type=int, default=None, metavar="N",
                   help="Mass-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_MASS).")
    g.add_argument("--norm_nq", type=int, default=None, metavar="N",
                   help="Mass-ratio-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_Q).")
    g.add_argument("--norm_nchi", type=int, default=None, metavar="N",
                   help="Spin-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_CHI).")
    g.add_argument("--pairing_norm_grid", type=int, default=None, metavar="N",
                   help="opt-in: interpolate the pairing model's per-sample "
                        "q-normalization from an N-node m1 grid instead of exact "
                        "per-sample integration (~1.3-1.6x faster likelihood "
                        "calls measured). Accuracy at N=2048: ~1e-8 relative "
                        "TYPICAL, up to ~2e-4 WORST-CASE for samples right at "
                        "the m_min support edge (the normaliser has a log-kink "
                        "there); halves ~4x per grid doubling. NOT recommended "
                        "when the run sits near the selection variance guard "
                        "boundary (a ~1e-4 logL perturbation can flip the hard "
                        "-inf wall). Default: exact "
                        "(env: DARKSIRENS_GW_PAIRING_M1_GRID).")
    g.add_argument("--kernel_gl_nodes", type=int, default=None, metavar="N",
                   help="Gauss-Legendre nodes for the per-galaxy kernel normalisation Z_i "
                        "(default 24). Spectroscopic catalogs (sigma_eff <~ 5e-3) are exact "
                        "to likelihood precision at 4-8 nodes and the quadrature dominates "
                        "wide-sky dark-siren runs; do NOT reduce for broad photo-z kernels.")
    return optp


def _normalize_multitracer_paths(opts):
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


def _resolve_catalog_sky_weighting(opts):
    # ── Catalog sky-weighting resolution (estimand coherence) ──────
    # The two dark-siren normalization conventions are DIFFERENT estimands and
    # each is degenerate in the wrong K regime: at K=1 the field
    # (survey-global) normalizer log_Z_global is one Lambda-dependent constant
    # that cancels exactly between the PE and selection terms, so the
    # number-density channel it exists for vanishes (NOTE (K=1) in
    # darksirens/redshift/prior.py); at K>=2 the conditional (per-pixel)
    # normalizer strips the number-density channel from the mixture weight and
    # fcat rails on clustered catalogs (tests/test_multitracer_field_recovery.py).
    # Unset resolves to the coherent convention for the run's K; the degenerate
    # explicit combinations are fatal for dark_sirens.  dark_sirens_complete
    # keeps its own pre-existing rules (K>=2 requires field, checked below;
    # K=1 allows both -- its complete-catalog field density has no n0 budget).
    opts.catalog_sky_weighting_source = (
        "explicit" if opts.catalog_sky_weighting is not None else "auto"
    )
    if opts.catalog_sky_weighting is None:
        opts.catalog_sky_weighting = (
            "conditional" if opts.n_catalogs == 1 else "field"
        )
    if (opts.universe_model == "dark_sirens"
            and opts.catalog_sky_weighting_source == "explicit"):
        if opts.n_catalogs == 1 and opts.catalog_sky_weighting == "field":
            _fatal(
                "--catalog_sky_weighting field is degenerate with a single "
                "catalog: the survey-global normalizer is one Lambda-dependent "
                "constant that cancels exactly between the PE and selection "
                "terms, so the number-density (log10n0) channel field mode "
                "exists for vanishes. Omit the flag (K=1 resolves to "
                "conditional), or add a second --survey_path catalog for a "
                "host-fraction (field) analysis."
            )
        if opts.n_catalogs >= 2 and opts.catalog_sky_weighting == "conditional":
            _fatal(
                "--catalog_sky_weighting conditional is not a host-fraction "
                "estimand for a K>=2 mixture: each catalog's prior is "
                "normalized per pixel, so fcat_k carries no number-density / "
                "sky-clustering information and rails to 1 on clustered "
                "sparse-contrast catalogs (measured: "
                "tests/test_multitracer_field_recovery.py). Omit the flag "
                "(K>=2 resolves to field), which makes fcat_k the catalog-k "
                "GW host fraction."
            )

    if opts.n_catalogs >= 2:
        _sky_w = getattr(opts, "catalog_sky_weighting", "conditional")
        if opts.universe_model == "dark_sirens_complete":
            # Complete-model mixture is coherent only under the field
            # (survey-global) normalizer; the conditional (per-pixel) normalizer
            # would mix per-pixel-normalized complete priors (incoherent estimand).
            if _sky_w != "field":
                _fatal("A K>=2 mixture with --universe_model dark_sirens_complete "
                       "requires --catalog_sky_weighting field; the conditional "
                       "(per-pixel) normalizer makes the complete-model mixture an "
                       "incoherent estimand (mixing per-pixel-normalized complete "
                       "priors).")
        elif opts.universe_model != "dark_sirens":
            _fatal("Multiple --survey_path catalogs (a K>=2 mixture) require "
                   "--universe_model dark_sirens, or dark_sirens_complete with "
                   "--catalog_sky_weighting field. spectral_sirens / "
                   "bright_sirens are inherently "
                   "single-catalog (catalog-free or counterpart-pinned) and "
                   "have no K>=2 mixture.")
        # Marked-host models compose with the mixture via PER-CATALOG eta
        # blocks (eta_<mark> for catalog 1, eta_<mark>_c{k} for catalogs
        # 2..K).  Resolve each catalog's marks from its FILE's datasets here
        # (pre-load), so the bundle loader and the parameter space see the
        # same ordered per-catalog lists.  mark_model == none still needs the
        # empty defaults (the K=1 data-based resolution below is skipped for
        # a mixture).
        opts.mark_names = ()
        opts.mark_names_by_catalog = ((),) * opts.n_catalogs
        if getattr(opts, "mark_model", "none") not in (None, "none"):
            import h5py as _h5py
            from darksirens.marks import MARK_FIELDS as _MF_K2

            _req = tuple(
                s.strip() for s in (opts.marks or "").split(",") if s.strip()
            ) or None
            _by_catalog = []
            for _p in opts.survey_paths:
                try:
                    with _h5py.File(_p, "r") as _f:
                        _avail = tuple(
                            n for n, ds in _MF_K2.items() if ds in _f
                        )
                except OSError as _exc:
                    _fatal(f"Cannot open --survey_path '{_p}' to resolve "
                           f"galaxy marks: {_exc}")
                _by_catalog.append(
                    tuple(n for n in (_req or _avail) if n in _avail)
                )
            if _req:
                _missing_everywhere = [
                    m for m in _req
                    if all(m not in names for names in _by_catalog)
                ]
                if _missing_everywhere:
                    _fatal(f"--marks requested {_missing_everywhere} but no "
                           "catalog provides them.")
            if all(not names for names in _by_catalog):
                _fatal("--mark_model loglinear requires per-galaxy marks, but "
                       "none of the catalogs provide any (expected datasets "
                       "like MARK_LOGMSTAR/MARK_LOGSSFR).")
            opts.mark_names_by_catalog = tuple(_by_catalog)
            opts.mark_names = _by_catalog[0]
            for _i, _names in enumerate(_by_catalog, start=1):
                if _names:
                    print(f"  [multitracer] catalog {_i} marks: "
                          f"{', '.join(_names)}")
                else:
                    print(f"  [multitracer] catalog {_i} has NO selected "
                          "marks -> plain galaxy-count host model (h = 1) "
                          "inside the mixture.")
        if opts.counterpart is not None:
            _fatal("--counterpart is not supported with a multi-catalog mixture.")


def _validate_multitracer_config(opts):
    # FIELD-convention sky weighting scope (host-fraction estimand).  The
    # missing-galaxy budget modulations (deterministic Q_LSS, use_LSS delta_g)
    # and the marked-host model ARE supported: the survey-global normalizer
    # carries the SAME per-pixel budget as the numerator (field_global_log_Z /
    # field_global_log_Z_marked; per-member normalizers under
    # --lss_marginalize).
    if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
        if opts.universe_model not in ("dark_sirens", "dark_sirens_complete"):
            _fatal("--catalog_sky_weighting field supports --universe_model "
                   "dark_sirens or dark_sirens_complete only (got "
                   f"'{opts.universe_model}').")
        if getattr(opts, "drop_full_catalog", False):
            _fatal("--catalog_sky_weighting field needs the full-sky catalog rows to "
                   "count empty pixels; it is incompatible with --drop_full_catalog.")

    # Marked-host model scope: the eta reweighting enters the dark_sirens
    # catalog prior only (prepare_redshift_prior_state ignores marks for every
    # other model), so any other universe_model would silently sample phantom
    # flat eta dimensions with zero likelihood effect.
    if (getattr(opts, "mark_model", "none") not in (None, "none")
            and opts.universe_model != "dark_sirens"):
        _fatal(f"--mark_model {opts.mark_model} requires --universe_model "
               f"dark_sirens (got '{opts.universe_model}'): marks reweight the "
               "dark-siren catalog prior and are inert for every other model, "
               "leaving phantom flat eta dimensions in the sampled space.")


def _canonicalize_fixed_flags(opts):
    # Persist the canonical names in settings while keeping opts.fix_cosmology
    # for backward-compatible internal callers and saved metadata.
    opts.fixed_cosmology = bool(opts.fix_cosmology)
    opts.fixed_de = bool(opts.fix_de)
    opts._fixed_de_superseded = bool(opts.fix_cosmology and opts.fix_de)
    if opts._fixed_de_superseded:
        opts.fix_de = False
        opts.fixed_de = False


def _configure_performance_grids(opts):
    try:
        configure_normalization_grids(
            n_mass=opts.norm_nmass,
            n_q=opts.norm_nq,
            n_chi=opts.norm_nchi,
            pairing_m1_grid=opts.pairing_norm_grid,
        )
    except ValueError as e:
        _fatal(str(e))

    if opts.kernel_gl_nodes is not None:
        from darksirens.redshift.catalog import configure_kernel_quadrature
        if opts.kernel_gl_nodes < 2:
            _fatal("--kernel_gl_nodes must be >= 2")
        configure_kernel_quadrature(opts.kernel_gl_nodes)


def _parse_structured_options(opts):
    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    opts.prior_overrides        = prior_overrides if prior_overrides else None
    opts.fixed_parameter_values = (
        fixed_parameter_values if fixed_parameter_values else None
    )
    opts.counterpart            = parse_counterpart_arg(opts.counterpart)
    return prior_overrides, fixed_parameter_values


def _resolve_sampler_config(opts):
    if opts.sampler == "tinyns":
        try:
            build_tinyns_config(opts)
        except ValueError as e:
            _fatal(str(e))

    resolve_redshift_prior_barrier(opts)

    guard_mode, max_likelihood_variance = resolve_selection_neff_guard(opts)
    # Always print a compact, resolved guard summary (mode + why, cap, and the
    # criterion) so the log records why the likelihood may return -inf; the
    # cap and the soft-mode post-hoc caveat are folded into the one line.
    print(
        "  [i] "
        + format_selection_guard_summary(
            guard_mode,
            opts.sampler,
            opts.selection_neff_soft_guard,
            max_likelihood_variance,
        )
    )


def _apply_bright_siren_overrides(opts):
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


def _validate_run_config(opts):
    # ── Validation ─────────────────────────────────────────────────

    _section("Validating configuration")
    GALAXY_AWARE = {"dark_sirens", "dark_sirens_complete"}

    if bool(opts.gw_path) == bool(opts.gw_flows_path):
        _fatal(
            "Exactly one of --gw_path (stored PE samples) or --gw_flows_path "
            "(per-event flow surrogates) is required."
        )
    if bool(opts.gwselection_path) == bool(opts.pdet_flow_path):
        _fatal(
            "Exactly one of --gwselection_path (injection file) or "
            "--pdet_flow_path (P_det emulator) is required."
        )
    if opts.pdet_flow_path:
        # Orthogonal to the event source: valid with --gw_path or
        # --gw_flows_path alike.
        if not os.path.isfile(opts.pdet_flow_path):
            _fatal(f"--pdet_flow_path is not a file: {opts.pdet_flow_path}")
        if opts.pdet_nsamp <= 0:
            _fatal("--pdet_nsamp must be positive.")
        try:
            _pdet_h0, _pdet_om0 = (float(v) for v in
                                   str(opts.pdet_cosmology).split(","))
        except ValueError:
            _fatal("--pdet_cosmology must be 'H0,Om0' (e.g. '67.9,0.3065'); "
                   f"got {opts.pdet_cosmology!r}.")
        else:
            if _pdet_h0 <= 0.0 or not (0.0 < _pdet_om0 < 1.0):
                _fatal("--pdet_cosmology needs H0 > 0 and 0 < Om0 < 1; "
                       f"got {opts.pdet_cosmology!r}.")
        if not (0.0 < opts.pdet_chieff_amax <= 1.0):
            _fatal("--pdet_chieff_amax must be in (0, 1].")
    if opts.gw_flows_path:
        if not os.path.isdir(opts.gw_flows_path):
            _fatal(f"--gw_flows_path is not a directory: {opts.gw_flows_path}")
        if opts.universe_model != "spectral_sirens":
            _fatal(
                "--gw_flows_path currently supports --universe_model "
                "spectral_sirens only. Dark sirens need 6-D flows with "
                "(ra, dec) plus the per-pixel catalog redshift-prior sampler "
                "(scaffolded: columns registry in darksirens/gw/flows.py, "
                "z-target hook in likelihood/flow_events.py). "
                f"Got '{opts.universe_model}'."
            )
        if not opts.shared_gamma:
            _fatal("--gw_flows_path requires shared redshift evolution "
                   "(--shared_gamma true).")
        if not opts.shared_spin:
            _fatal("--gw_flows_path requires a shared spin component "
                   "(--shared_spin true).")
        if opts.sky_model != "isotropic":
            _fatal("--gw_flows_path supports --sky_model isotropic only.")
        if getattr(opts, "mark_model", "none") not in (None, "none"):
            _fatal("--gw_flows_path does not support a marked-host model.")
        if opts.survey_path:
            _fatal("--gw_flows_path (spectral sirens) does not use --survey_path.")
        if opts.flows_nsamp <= 0:
            _fatal("--flows_nsamp must be positive.")
        if opts.sampler == "numpyro":
            _warn(
                "flows + NumPyro NUTS: the grid inverse-CDF samplers give "
                "piecewise gradients; nested sampling is the validated path."
            )

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
            "--fix_cosmology supersedes --fix_de; "
            "dark energy is included in the fixed cosmology block."
        )
    if opts.fix_population and opts.fix_cosmology and opts.fix_survey:
        _warn("All blocks fixed — nothing will be inferred.")

    _ok("Configuration is valid.")
    _end()


def _print_run_configuration(opts, prior_overrides, fixed_parameter_values):
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
    if opts.survey_path or opts.survey_paths:
        _row(
            "Survey z_depth override",
            opts.survey_z_depth if opts.survey_z_depth is not None
            else "none (per-catalog file attr, else legacy full-grid)",
        )
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
    _pairing_grid = (
        "exact" if norm_grid.pairing_m1_grid is None
        else f"grid({norm_grid.pairing_m1_grid})"
    )
    _row("Norm grids", (
        f"mass={norm_grid.n_mass}, q={norm_grid.n_q}, chi={norm_grid.n_chi}, "
        f"pairing={_pairing_grid}"
    ))
    _row("JAX backend", jax.default_backend())
    _row("JAX devices",  ", ".join(str(d) for d in jax.devices()))
    print("  │")
    if opts.gw_flows_path:
        _row("GW flows path", opts.gw_flows_path)
        _row("Flow draws J",  f"{opts.flows_nsamp:,} (seed {opts.flows_seed})")
        _row("Flow PE cosmo", opts.flows_pe_cosmology)
    else:
        _row("GW events path",  opts.gw_path)
    if opts.pdet_flow_path:
        _row("Pdet emulator",   opts.pdet_flow_path)
        _row("Pdet draws M",    f"{opts.pdet_nsamp:,} (seed {opts.pdet_seed})")
        _row("Pdet inj. cosmo", opts.pdet_cosmology)
        _row("Pdet chieff amax", opts.pdet_chieff_amax)
    else:
        _row("Selection path",  opts.gwselection_path)
    if opts.survey_path:
        if opts.n_catalogs == 1:
            _row("Survey path",  opts.survey_path)
        else:
            _row("Catalog mixture", f"{opts.n_catalogs} catalogs")
            for _i, _p in enumerate(opts.survey_paths):
                _row(f"  catalog {_i + 1}", _p)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
        _row("Catalog sky weighting",
             f"{getattr(opts, 'catalog_sky_weighting', 'conditional')} "
             f"({getattr(opts, 'catalog_sky_weighting_source', 'explicit')})")
    _row("Output root",     opts.save_path)
    _row("Sel. batch",     format_block_size_request(opts.sel_batch_size))
    _row("PE event block", format_block_size_request(getattr(opts, "pe_event_block", None)))
    if opts.drop_full_catalog:
        _row("Drop full catalog", "yes (compact views only)")
    _end()


def _resolve_and_report_block_sizes(opts, data):
    """Resolve the 'auto' block-size knobs from a probed memory budget (post-load).

    Mutates ``opts.sel_batch_size`` / ``opts.pe_event_block`` to concrete
    ``int``/``None`` values and records the provenance on
    ``opts.block_size_resolution`` (persisted via settings.json), so the factory,
    flow builder and settings dump downstream all see plain ints/None.
    """
    _msel = data.get("m1detsels")
    n_sel = int(np.asarray(_msel).shape[0]) if _msel is not None else 0
    plan = resolve_block_sizes(
        n_events=int(data.get("nEvents", 0)),
        n_samp=int(data.get("nsamp", 0)),
        n_sel=n_sel,
        sel_requested=getattr(opts, "sel_batch_size", None),
        pe_requested=getattr(opts, "pe_event_block", None),
        has_catalog=bool(getattr(opts, "survey_path", None)),
        flow_path=bool(getattr(opts, "gw_flows_path", None)),
        backend=jax.default_backend(),
    )
    opts.sel_batch_size = plan.sel_batch_size
    opts.pe_event_block = plan.pe_event_block
    opts.block_size_resolution = plan.source
    _sel = plan.sel_batch_size if plan.sel_batch_size is not None else "single pass"
    _pe = plan.pe_event_block if plan.pe_event_block is not None else "single pass"
    _ok(f"Block sizes [{plan.source}]: sel_batch_size={_sel}, pe_event_block={_pe}")


def _load_and_report_data(opts):
    # ── Load data ──────────────────────────────────────────────────

    _section("Loading data")
    print("  │")
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)

    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    nside   = data.get("nside", "N/A")

    if data.get("flow_event_names"):
        # Persist the ensemble's event identity in settings.json (which
        # serialises opts attributes) alongside the results.hdf5 meta.
        opts.flow_event_names = list(data["flow_event_names"])
        _ok(
            f"Flow surrogates:        {nEvents} events × J={nsamp:,} population "
            "draws/likelihood call"
        )
    else:
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

    # ── Survey z_depth resolution (bounds the completion missing-galaxy
    #    budget) ─────────────────────────────────────────────────────────
    # Precedence: --survey_z_depth (CLI, applies to every catalog) >
    # per-catalog f.attrs['z_depth'] (written by darksirens_pixelate
    # --z_depth) > None (legacy full-grid budget, bit-identical). Resolved
    # host-side here (files are already loaded) and threaded to SurveyParams
    # via ParameterDecoder.z_depths (inference/parameters.py).
    if opts.survey_path:
        if opts.n_catalogs == 1:
            resolved = resolve_survey_z_depth(opts.survey_z_depth, data.get("z_depth"))
            opts.resolved_survey_z_depths = [resolved]
            _report_survey_z_depth("Survey z_depth", resolved)
        else:
            bundles = data.get("catalogs") or []
            resolved_list = [
                resolve_survey_z_depth(opts.survey_z_depth, b.get("z_depth"))
                for b in bundles
            ]
            opts.resolved_survey_z_depths = resolved_list
            for _i, _resolved in enumerate(resolved_list):
                _report_survey_z_depth(f"Catalog {_i + 1} z_depth", _resolved)
    else:
        opts.resolved_survey_z_depths = []

    # Canonical post-load validation: assert the per-catalog config sequences
    # resolved above agree with n_catalogs before anything downstream consumes
    # them (single place; no behaviour change for a valid config).
    validate_multitracer_run(opts, data)

    _resolve_and_report_block_sizes(opts, data)

    _end()
    return data


def _maybe_run_completion_validation(opts, data, prior_overrides, fixed_parameter_values):
    if opts.validate_completion:
        _section("Completion validation dry run")
        if opts.n_catalogs >= 2:
            for validation_path in run_completion_validation_multitracer(
                opts, data, prior_overrides, fixed_parameter_values
            ):
                _ok(f"completion_validation JSON → {validation_path}")
        else:
            validation_path = run_completion_validation(
                opts, data, prior_overrides, fixed_parameter_values
            )
            _ok(f"completion_validation JSON → {validation_path}")
        _row("Action", "exiting before likelihood/sampling")
        _end()
        return True
    return False


def _resolve_single_catalog_marks(opts, data):
    # ── Parameter space ────────────────────────────────────────────

    # Resolve the galaxy marks for the marked-host model (dark_sirens): those
    # present in the catalog, optionally narrowed by --marks.  Stored on opts so
    # the decoder/likelihood see the same ordered list as the parameter space.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS
    if opts.n_catalogs == 1:
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
        opts.mark_names_by_catalog = (tuple(opts.mark_names),)
    # K >= 2: opts.mark_names / opts.mark_names_by_catalog were resolved
    # PRE-LOAD from each catalog file's datasets (see the multitracer guard
    # block); the flat data dict carries no top-level marks for a mixture.


def _build_and_report_parameter_space(opts, data, prior_overrides, fixed_parameter_values):
    _section("Building parameter space")
    # A Q_LSS completion table (explicit --lss_completion path or an in-catalog
    # /lss_completion group) REPLACES the b_miss local-overdensity factor, so
    # b_miss must be dropped from the sampled dark_sirens block or it is a
    # phantom flat nuisance dimension. Mirror the likelihood's Q-active test:
    # an explicit path, a table on this catalog's loaded bundle (K>=2), or (K=1
    # flat path, no bundle) the in-catalog table already merged into flat data.
    # PER-CATALOG Q activity: the likelihood chooses Q-vs-b_miss per catalog
    # (completion.field_lss_q is not None), so b_miss must be dropped ONLY for
    # the catalogs that actually carry a Q table.  A global OR would drop
    # b_miss_c{k} for a Q-free catalog too (a silently frozen nuisance).
    _lss_bundles = data.get("catalogs") or []
    _lss_explicit = list(getattr(opts, "lss_completions", None) or [])
    _LSS_Q_KEYS = ("lss_completion_logq", "lss_completion_q",
                   "lss_completion_logq_members", "lss_completion_q_members")

    def _catalog_lss_active(i):
        # Explicit --lss_completion path for catalog i (positionally aligned).
        if i < len(_lss_explicit) and _lss_explicit[i] is not None:
            return True
        # K>=2: this catalog's own bundle carrying any Q table/ensemble.
        if i < len(_lss_bundles):
            b = _lss_bundles[i]
            return any(b.get(k) is not None for k in _LSS_Q_KEYS)
        # K=1 flat path: no per-catalog bundle, so an in-catalog /lss_completion
        # group (or explicit table) landed in the flat data dict via
        # attach_lss_inputs -> maybe_load_lss_completion; mirror the same key
        # probe there so an in-catalog K=1 Q table also drops b_miss.
        if not _lss_bundles:
            return any(data.get(k) is not None for k in _LSS_Q_KEYS)
        return False

    lss_completion_active_by_catalog = tuple(
        bool(_catalog_lss_active(i)) for i in range(opts.n_catalogs)
    )
    lss_completion_active = any(lss_completion_active_by_catalog)
    # Persist BOTH on opts (-> settings.json) so post-processing rebuilds the
    # SAME parameter space (b_miss dropped per catalog when Q is active) as this
    # run sampled: the per-catalog tuple is authoritative, the scalar is the
    # legacy back-compat fallback.
    opts.lss_completion_active_by_catalog = lss_completion_active_by_catalog
    opts.lss_completion_active = bool(lss_completion_active)
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
        lss_completion_active  = lss_completion_active_by_catalog,
        mark_names_by_catalog  = getattr(opts, "mark_names_by_catalog", None),
    )
    labels, lower_bound, upper_bound = res[0], res[1], res[2]
    n_pop_eff, n_cosmo_eff, n_survey_eff, model_name = res[3], res[7], res[8], res[9]
    fixed_parameter_statuses = res[10]
    prior_kinds = res[11]
    # Record the exact sampler labels so build_parameter_decoder (inside
    # make_likelihood) can fail fast if it re-derives a different set -- the
    # P0.1 fail-fast net against any future CLI/decoder flag drift.
    opts.expected_sampled_labels = tuple(labels)

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
    return _ParameterSpace(
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prior_kinds=prior_kinds,
        prior_transform=prior_transform,
        pop_params_fid=pop_params_fid,
        n_pop_eff=n_pop_eff,
        n_cosmo_eff=n_cosmo_eff,
        n_survey_eff=n_survey_eff,
        model_name=model_name,
    )


def _build_likelihood(opts, data, pspace, fixed_parameter_values):
    pop_params_fid = pspace.pop_params_fid
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
    return likelihood


def _run_sampling(opts, likelihood, pspace):
    labels = pspace.labels
    lower_bound = pspace.lower_bound
    upper_bound = pspace.upper_bound
    prior_kinds = pspace.prior_kinds
    prior_transform = pspace.prior_transform
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
    return results, wall_sampling


def _make_run_dir(opts, timestamp):
    """Create opts.save_path/<run_name>, disambiguating name collisions.

    run_name embeds pop_model/universe_model/sampler/seed/timestamp
    (second resolution); two same-config jobs (same seed too) launched
    within the same second would otherwise collide under exist_ok=True
    and later overwrite each other's results.hdf5. Retry with a
    monotonically increasing numeric suffix instead.
    """
    run_name = (
        f"{opts.pop_model}__{opts.universe_model}__{opts.sampler}"
        f"__seed{opts.seed}__{timestamp}"
    )
    run_dir = os.path.join(opts.save_path, run_name)
    suffix = 0
    while True:
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = os.path.join(opts.save_path, f"{run_name}-{suffix:02d}")


def _save_outputs(opts, data, results, pspace, fixed_parameter_values, prior_overrides, wall_sampling, t_start):
    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    labels = pspace.labels
    lower_bound = pspace.lower_bound
    upper_bound = pspace.upper_bound
    prior_kinds = pspace.prior_kinds
    n_pop_eff = pspace.n_pop_eff
    n_cosmo_eff = pspace.n_cosmo_eff
    n_survey_eff = pspace.n_survey_eff
    model_name = pspace.model_name
    # ── Save outputs ───────────────────────────────────────────────

    t_end     = datetime.datetime.now()
    timestamp = t_end.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir   = _make_run_dir(opts, timestamp)

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
    if data.get("flow_event_names"):
        # Flow runs have no event names in a PE file: the ensemble's sorted
        # checkpoint order IS the event identity; persist it for diagnostics.
        # (JSON-encoded, and keyed *_json so it does not shadow the plain list
        # that settings.json serialises from opts.flow_event_names.)
        meta["flow_event_names_json"] = json.dumps(list(data["flow_event_names"]))

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


def main(argv=None):
    t_start = datetime.datetime.now()
    print()
    _banner(f"DARK SIRENS  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    optp = build_parser()
    opts = optp.parse_args(argv)

    _normalize_multitracer_paths(opts)
    _resolve_catalog_sky_weighting(opts)
    _validate_multitracer_config(opts)
    _canonicalize_fixed_flags(opts)
    _configure_performance_grids(opts)
    prior_overrides, fixed_parameter_values = _parse_structured_options(opts)
    _resolve_sampler_config(opts)
    _apply_bright_siren_overrides(opts)
    _print_all_cli_options(
        optp,
        opts,
        normalization_grid=normalization_grid_settings().to_dict(),
    )
    _validate_run_config(opts)
    _print_run_configuration(opts, prior_overrides, fixed_parameter_values)
    data = _load_and_report_data(opts)
    if _maybe_run_completion_validation(
        opts, data, prior_overrides, fixed_parameter_values
    ):
        return
    _resolve_single_catalog_marks(opts, data)
    pspace = _build_and_report_parameter_space(
        opts, data, prior_overrides, fixed_parameter_values
    )
    likelihood = _build_likelihood(opts, data, pspace, fixed_parameter_values)
    results, wall_sampling = _run_sampling(opts, likelihood, pspace)
    _save_outputs(
        opts, data, results, pspace,
        fixed_parameter_values, prior_overrides, wall_sampling, t_start,
    )


if __name__ == "__main__":
    run_cli(main)
