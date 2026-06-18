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

# Dark sirens with galaxy catalog, fixed cosmology, emcee:
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --survey_path       catalog_nside64.h5 \
    --sampler           emcee \
    --pop_model         brokenpowerlaw+2peaks \
    --universe_model    dark_sirens \
    --fixed_cosmology   true \

# Fix individual parameters via JSON:
    --fixed_parameter_values '{"$v_1$": 0.1}'
    --prior_overrides        '{"H0": [60.0, 80.0]}'
"""

import os

# ── JAX memory configuration (before any JAX import) ──────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",  "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR",    "platform")

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
from darksirens.inference.likelihood import make_likelihood
from darksirens.em.completion import build_pixel_kde_cache, completion_clip_diagnostics
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.inference.sampling import run_sampler
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.parameters import H0_FID, OM0_FID, W0_FID, WA_FID

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
warnings.simplefilter("ignore", FutureWarning)


# ── Formatting helpers ─────────────────────────────────────────────────────────

W = 72

def _banner(text: str):
    pad   = max(0, W - 4 - len(text))
    left  = pad // 2
    right = pad - left
    print(f"{'─' * W}")
    print(f"  {'·' * left} {text} {'·' * right}  ")
    print(f"{'─' * W}")

def _section(title: str):
    print()
    print(f"  ┌─ {title} {'─' * max(0, W - 6 - len(title))}┐")

def _row(label: str, value, width: int = 26):
    print(f"  │  {label:<{width}} {value}")

def _end():
    print(f"  └{'─' * (W - 3)}┘")

def _ok(msg: str):   print(f"  ✓  {msg}")
def _warn(msg: str): print(f"  ⚠  {msg}")
def _err(msg: str):  print(f"  ✗  {msg}")

def _fatal(msg: str):
    print()
    _err(f"FATAL: {msg}")
    print()
    sys.exit(1)


def _fixed_dark_energy_metadata(opts, fixed_parameter_values: dict | None) -> dict:
    """Return fixed-state metadata for CPL dark-energy parameters."""
    fixed_parameter_values = fixed_parameter_values or {}
    block_fixed = bool(
        getattr(opts, "fix_cosmology", False) or getattr(opts, "fix_de", False)
    )
    w0_fixed = block_fixed or "w0" in fixed_parameter_values
    wa_fixed = block_fixed or "wa" in fixed_parameter_values

    return {
        "fixed_dark_energy": bool(w0_fixed and wa_fixed),
        "w0_fixed": bool(w0_fixed),
        "wa_fixed": bool(wa_fixed),
        "w0_value": (
            float(fixed_parameter_values["w0"])
            if "w0" in fixed_parameter_values
            else float(W0_FID)
            if w0_fixed
            else None
        ),
        "wa_value": (
            float(fixed_parameter_values["wa"])
            if "wa" in fixed_parameter_values
            else float(WA_FID)
            if wa_fixed
            else None
        ),
    }


def _format_fixed_dark_energy_summary(opts, fixed_parameter_values: dict | None) -> str:
    """Format fixed CPL dark-energy state for human-readable summaries."""
    meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
    if not (meta["w0_fixed"] or meta["wa_fixed"]):
        return "no"
    pieces = ["yes" if meta["fixed_dark_energy"] else "partial"]
    fixed_values = []
    if meta["w0_fixed"]:
        fixed_values.append(f"w0={meta['w0_value']:.6g}")
    if meta["wa_fixed"]:
        fixed_values.append(f"wa={meta['wa_value']:.6g}")
    if fixed_values:
        pieces.append(f"({', '.join(fixed_values)})")
    return " ".join(pieces)

def _format_option_value(value):
    """Format parsed CLI option values for human-readable config tables."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _print_all_cli_options(optp: ArgumentParser, opts, *, normalization_grid: dict):
    """Print every parsed CLI option in argparse group order."""
    _section("All CLI Options")
    first_group = True
    seen: set[str] = set()

    for group in optp._action_groups:
        group_rows = []
        for action in group._group_actions:
            if action.dest == "help" or not hasattr(opts, action.dest):
                continue
            group_rows.append(action.dest)

        if not group_rows:
            continue

        if not first_group:
            print("  │")
        first_group = False

        _row(f"[{group.title}]", "")
        for dest in group_rows:
            seen.add(dest)
            _row(f"  {dest}", _format_option_value(getattr(opts, dest)))

    ungrouped = sorted(key for key in set(vars(opts)) - seen if not key.startswith("_"))
    if ungrouped:
        if not first_group:
            print("  │")
        _row("[Other]", "")
        for dest in ungrouped:
            _row(f"  {dest}", _format_option_value(getattr(opts, dest)))

    print("  │")
    _row("[Derived]", "")
    _row("  normalization_grid", _format_option_value(normalization_grid))
    _end()


# ── CLI helpers ────────────────────────────────────────────────────────────────

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "t", "1", "yes", "y"}:
        return True
    if str(value).lower() in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean.")


def parse_json_arg(value: str | None, argname: str) -> dict:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object (dict).")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        _fatal(f"--{argname} must be a valid JSON object. Error: {e}\n"
               f"  Example: --{argname} '{{\"H0\": [60, 80]}}'")


def parse_counterpart_arg(value: list[str] | None) -> tuple[tuple[float, float, float], ...] | None:
    """Parse one or more ``--counterpart RA DEC Z`` triplets into floats.

    Angles are expected in radians, matching the GW sample convention used by
    ``load_gw_samples`` and HEALPix indexing throughout the pipeline.  Multiple
    triplets are ordered by GW event, enabling multi-bright-siren analyses.
    """
    if value is None:
        return None
    if len(value) % 3 != 0:
        _fatal("--counterpart requires RA DEC Z triplets (angles in radians).")
    try:
        vals = [float(x) for x in value]
    except ValueError as e:
        _fatal(f"--counterpart values must be numeric RA DEC Z triplets. Error: {e}")
    out = []
    for i in range(0, len(vals), 3):
        ra, dec, z = vals[i : i + 3]
        if not (0.0 <= ra < 2.0 * np.pi):
            _fatal("--counterpart RA must be in radians with 0 <= RA < 2π.")
        if not (-0.5 * np.pi <= dec <= 0.5 * np.pi):
            _fatal("--counterpart Dec must be in radians with -π/2 <= Dec <= π/2.")
        if z <= 0.0:
            _fatal("--counterpart redshift Z must be positive.")
        out.append((ra, dec, z))
    return tuple(out)


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

# ── Data saving ────────────────────────────────────────────────────────────────

def save_results_hdf5(
    results:                dict,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    opts,
    meta:                   dict,
    prior_kinds=None,
) -> str:
    """
    Save posterior samples and all metadata to a single HDF5 file.

    Structure
    ---------
    results.hdf5
    ├── attrs           — run metadata (model, sampler, evidence, runtime, …)
    ├── samples         — (N_samples, N_dim) posterior samples
    ├── labels          — (N_dim,) parameter label strings (UTF-8)
    ├── lower_bound     — (N_dim,) prior lower bounds
    ├── upper_bound     — (N_dim,) prior upper bounds
    ├── fixed_labels    — (N_fixed,) individually-fixed param labels
    ├── fixed_values    — (N_fixed,) their values
    ├── log_weights     — (N_samples,) log importance weights  [if available]
    └── log_likelihood  — (N_samples,) per-sample log-likelihoods [if available]
    """
    path = os.path.join(run_dir, "results.hdf5")
    kw   = dict(compression="gzip", shuffle=True)
    dt   = h5py.string_dtype(encoding="utf-8")

    samples = np.asarray(results["samples"])
    N, ndim = samples.shape

    with h5py.File(path, "w") as f:

        # Samples and bounds
        f.create_dataset("samples",     data=samples,                          **kw)
        f.create_dataset("lower_bound", data=np.array(lower_bound, dtype=float), **kw)
        f.create_dataset("upper_bound", data=np.array(upper_bound, dtype=float), **kw)
        f.create_dataset("labels",      data=np.array(labels, dtype=object), dtype=dt)

        # Optional per-sample arrays
        if results.get("log_weights") is not None:
            f.create_dataset("log_weights",    data=np.asarray(results["log_weights"]), **kw)
        if results.get("log_likelihood") is not None:
            f.create_dataset("log_likelihood", data=np.asarray(results["log_likelihood"]), **kw)

        # Individually-fixed parameters — store so post-processing can reconstruct
        # the full parameter vector without reading the settings JSON separately.
        if fixed_parameter_values:
            fix_labels = list(fixed_parameter_values.keys())
            fix_vals   = [float(v) for v in fixed_parameter_values.values()]
            f.create_dataset("fixed_labels", data=np.array(fix_labels, dtype=object), dtype=dt)
            f.create_dataset("fixed_values", data=np.array(fix_vals, dtype=float), **kw)

        # Run metadata
        f.attrs["pop_model"]       = opts.pop_model
        f.attrs["shared_beta"]     = bool(getattr(opts, "shared_beta", True))
        f.attrs["shared_spin"]     = bool(getattr(opts, "shared_spin", True))
        f.attrs["shared_gamma"]    = bool(getattr(opts, "shared_gamma", True))
        # Per-parameter prior family (Option A); lets post-processing reconstruct
        # the exact prior. Default uniform when unset.
        if prior_kinds is not None:
            f.attrs["prior_kinds"] = json.dumps(
                {lbl: [k[0], k[1], k[2]] for lbl, k in zip(labels, prior_kinds)},
                default=str,
            )
        f.attrs["universe_model"]  = opts.universe_model
        f.attrs["sky_model"]       = getattr(opts, "sky_model", "isotropic")
        f.attrs["complete_empty_pixel_policy"] = opts.complete_empty_pixel_policy
        f.attrs["sampler"]         = opts.sampler
        f.attrs["fix_cosmology"]   = bool(opts.fix_cosmology)
        f.attrs["fixed_cosmology"] = bool(opts.fix_cosmology)
        f.attrs["fix_de"]          = bool(getattr(opts, "fix_de", False))
        f.attrs["fixed_de"]        = bool(getattr(opts, "fix_de", False))
        de_meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
        f.attrs["fixed_dark_energy"] = bool(de_meta["fixed_dark_energy"])
        f.attrs["w0_fixed"]        = bool(de_meta["w0_fixed"])
        f.attrs["wa_fixed"]        = bool(de_meta["wa_fixed"])
        if de_meta["w0_value"] is not None:
            f.attrs["fixed_w0"]    = float(de_meta["w0_value"])
        if de_meta["wa_value"] is not None:
            f.attrs["fixed_wa"]    = float(de_meta["wa_value"])
        f.attrs["fix_population"]  = bool(opts.fix_population)
        f.attrs["fix_survey"]      = bool(opts.fix_survey)
        f.attrs["gw_path"]         = opts.gw_path
        f.attrs["gwselection_path"] = opts.gwselection_path
        f.attrs["survey_path"]     = opts.survey_path or ""
        if getattr(opts, "counterpart", None) is not None:
            counterpart_arr = np.asarray(opts.counterpart, dtype=float)
            f.create_dataset("counterparts", data=counterpart_arr, **kw)
            f.attrs["counterpart_ra"] = float(counterpart_arr[0, 0])
            f.attrs["counterpart_dec"] = float(counterpart_arr[0, 1])
            f.attrs["counterpart_z"] = float(counterpart_arr[0, 2])
            f.attrs["n_counterparts"] = int(counterpart_arr.shape[0])
            f.attrs["counterpart_dz"] = float(opts.counterpart_dz)
            f.attrs["counterpart_nside"] = int(opts.counterpart_nside)
            f.attrs["bright_siren_sky_marginalized"] = bool(opts.bright_siren_sky_marginalized)
        f.attrs["nlive"]           = int(opts.nlive)
        f.attrs["dlogz"]           = float(opts.dlogz)
        f.attrs["nwalkers"]        = int(opts.nwalkers)
        f.attrs["nsteps"]          = int(opts.nsteps)
        f.attrs["nuts_warmup"]     = int(getattr(opts, "nuts_warmup", 0))
        f.attrs["nuts_samples"]    = int(getattr(opts, "nuts_samples", 0))
        f.attrs["nuts_chains"]     = int(getattr(opts, "nuts_chains", 0))
        f.attrs["nuts_target_accept"] = float(getattr(opts, "nuts_target_accept", 0.0))
        f.attrs["nuts_max_tree_depth"] = int(getattr(opts, "nuts_max_tree_depth", 0))
        f.attrs["nuts_init_tries"] = int(getattr(opts, "nuts_init_tries", 0))
        f.attrs["nuts_init_seed_offset"] = int(getattr(opts, "nuts_init_seed_offset", 0))
        f.attrs["seed"]            = int(opts.seed)
        f.attrs["n_samples"]       = N
        f.attrs["n_dim"]           = ndim
        f.attrs["n_events"]        = int(meta["n_events"])
        f.attrs["n_samp_per_event"] = int(meta["n_samp_per_event"])
        f.attrs["n_draw"]          = int(meta["n_draw"])
        f.attrs["total_runtime"]   = meta["total_runtime"]
        f.attrs["sampling_runtime"] = meta["sampling_runtime"]
        f.attrs["timestamp"]       = meta["timestamp"]

        logZ    = results.get("logZ")
        logZerr = results.get("logZerr")
        if logZ is not None:
            f.attrs["logZ"]    = float(logZ)
            f.attrs["logZerr"] = float(logZerr) if logZerr is not None else float("nan")

        if prior_overrides:
            f.attrs["prior_overrides"] = json.dumps(prior_overrides)

        f.attrs["environment"] = json.dumps({
            "jax_version":    jax.__version__,
            "numpy_version":  np.__version__,
            "healpy_version": hp.__version__,
            "jax_backend":    jax.default_backend(),
            "jax_devices":    [str(d) for d in jax.devices()],
            "python_version": sys.version,
        })

    return path


def save_settings_json(
    opts,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    meta:                   dict,
) -> str:
    """Human-readable settings.json for easy inspection and re-runs."""
    d: dict = {}

    for key, val in vars(opts).items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(val)
            d[key] = val
        except (TypeError, ValueError):
            d[key] = str(val)

    # Emit None explicitly so it's obvious when not set — not an empty dict
    d["fixed_parameter_values"] = fixed_parameter_values if fixed_parameter_values else None
    d["prior_overrides"]        = prior_overrides        if prior_overrides        else None
    d["fixed_cosmology"]        = bool(getattr(opts, "fix_cosmology", False))
    d["fixed_de"]               = bool(getattr(opts, "fix_de", False))
    de_meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
    d["fixed_dark_energy"]      = de_meta["fixed_dark_energy"]
    d["dark_energy_fixed_values"] = {
        label: value
        for label, value in (
            ("w0", de_meta["w0_value"]),
            ("wa", de_meta["wa_value"]),
        )
        if value is not None
    } or None
    d["w0_fixed"]               = de_meta["w0_fixed"]
    d["wa_fixed"]               = de_meta["wa_fixed"]

    d["labels"]      = list(labels)
    d["lower_bound"] = list(map(float, lower_bound))
    d["upper_bound"] = list(map(float, upper_bound))
    d.update(meta)
    d["normalization_grid"] = normalization_grid_settings().to_dict()

    d["environment"] = {
        "jax_version":    jax.__version__,
        "numpy_version":  np.__version__,
        "healpy_version": hp.__version__,
        "jax_backend":    jax.default_backend(),
        "jax_devices":    [str(dv) for dv in jax.devices()],
        "python_version": sys.version,
    }

    path = os.path.join(run_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    return path



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
    g.add_argument("--survey_path",      default=None)
    g.add_argument("--save_path",        default="./")

    g = optp.add_argument_group("Physical model")
    g.add_argument("--universe_model", default="spectral_sirens",
                   choices=["spectral_sirens", "dark_sirens", "dark_sirens_complete", "bright_sirens"])
    g.add_argument(
        "--sky_model", default="isotropic",
        choices=["isotropic", "dipole", "sphere_gp", "sphere_gp_z", "overdensity_gp"],
        help=(
            "Sky distribution of the source rate. 'isotropic' (default) is the "
            "null; 'dipole' (Isi, Farr & Varma 2023) and 'sphere_gp' "
            "(log-Gaussian random field, Essick et al. 2023) are angular g(n). "
            "'sphere_gp_z' is a (sphere x z) GP normalised per z-shell "
            "(directional anisotropy evolving with distance); 'overdensity_gp' "
            "is the same field normalised over the comoving volume (full 3-D "
            "clustering, use with gamma fixed). All compared to isotropy by "
            "evidence; forced to 'isotropic' for bright_sirens."
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

    g = optp.add_argument_group("Sampler")
    g.add_argument("--sampler",      required=True, choices=["jaxns", "dynesty", "emcee", "numpyro"])
    g.add_argument("--nlive",        type=int,   default=1000)
    g.add_argument("--dlogz",        type=float, default=0.1)
    g.add_argument("--max_samples",  type=int,   default=1_000_000)
    g.add_argument("--nwalkers",     type=int,   default=32)
    g.add_argument("--nsteps",       type=int,   default=1000)
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
    g.add_argument("--norm_nmass", type=int, default=None, metavar="N",
                   help="Mass-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_MASS).")
    g.add_argument("--norm_nq", type=int, default=None, metavar="N",
                   help="Mass-ratio-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_Q).")
    g.add_argument("--norm_nchi", type=int, default=None, metavar="N",
                   help="Spin-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_CHI).")

    opts = optp.parse_args()
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

    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    opts.prior_overrides        = prior_overrides if prior_overrides else None
    opts.fixed_parameter_values = (
        fixed_parameter_values if fixed_parameter_values else None
    )
    opts.counterpart            = parse_counterpart_arg(opts.counterpart)

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
    if fixed_parameter_values:
        for lbl, val in fixed_parameter_values.items():
            _row(f"  fixed: {lbl}", val)
    else:
        _row("Fixed param values", "none")
    print("  │")
    _row("Sampler", opts.sampler)
    if opts.sampler in ("jaxns", "dynesty"):
        _row("  live points", opts.nlive)
    if opts.sampler == "dynesty":
        _row("  ΔlogZ stop",  opts.dlogz)
    if opts.sampler == "jaxns":
        _row("  max samples", f"{opts.max_samples:,}")
    if opts.sampler == "emcee":
        _row("  walkers", opts.nwalkers)
        _row("  steps",   opts.nsteps)
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
        _row("Survey path",  opts.survey_path)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
    _row("Output root",     opts.save_path)
    if opts.sel_batch_size:
        _row("Sel. batch",   f"{opts.sel_batch_size:,} samples/batch")
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
    sampler_info = {
        "jaxns":   f"nlive={opts.nlive}  max_samples={opts.max_samples:,}  seed={opts.seed}",
        "dynesty": f"nlive={opts.nlive}  dlogz={opts.dlogz}  seed={opts.seed}",
        "emcee":   f"nwalkers={opts.nwalkers}  nsteps={opts.nsteps}  seed={opts.seed}",
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
