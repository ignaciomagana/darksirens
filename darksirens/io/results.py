import json
import os
import sys

import h5py
import healpy as hp
import jax
import numpy as np

from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.cli.common import _fixed_dark_energy_metadata
from darksirens.inference.sampling import _json_safe_tinyns_value

# ── Data saving ────────────────────────────────────────────────────────────────

_TINYNS_SCALAR_ATTRS = {
    "niter": "tinyns_niter",
    "ncall": "tinyns_ncall",
    "seconds": "tinyns_seconds",
    "niter_per_sec": "tinyns_niter_per_sec",
    "ncall_per_sec": "tinyns_ncall_per_sec",
    "calls_per_iter": "tinyns_calls_per_iter",
    "final_delta_logz": "tinyns_final_delta_logz",
    "live_weight_fraction": "tinyns_live_weight_fraction",
    "replacement_mean_batches": "tinyns_replacement_mean_batches",
    "replacement_max_batches": "tinyns_replacement_max_batches",
    "replacement_failures": "tinyns_replacement_failures",
    "replacement_rescue_used": "tinyns_replacement_rescue_used",
    "insertion_rank_mean_z": "tinyns_insertion_rank_mean_z",
    "insertion_rank_std_ratio": "tinyns_insertion_rank_std_ratio",
}


def _json_attr(value):
    return json.dumps(_json_safe_tinyns_value(value), default=str)


def write_tinyns_metadata(attrs, results, opts) -> None:
    if getattr(opts, "tinyns_resolved_config", None) is not None:
        attrs["tinyns_resolved_config"] = _json_attr(opts.tinyns_resolved_config)
    if results.get("tinyns_runtime_diagnostics") is not None:
        attrs["tinyns_runtime_diagnostics"] = _json_attr(results["tinyns_runtime_diagnostics"])
    if results.get("tinyns_summary") is not None:
        attrs["tinyns_summary"] = str(results["tinyns_summary"])
    if results.get("tinyns_diagnostics") is not None:
        attrs["tinyns_diagnostics"] = _json_attr(results["tinyns_diagnostics"])
    diag = results.get("tinyns_runtime_diagnostics") or {}
    for key, attr in _TINYNS_SCALAR_ATTRS.items():
        value = diag.get(key)
        if value is None:
            continue
        try:
            safe = _json_safe_tinyns_value(value)
            if isinstance(safe, (str, list, dict)):
                continue
            attrs[attr] = safe
        except (TypeError, ValueError):
            continue


def save_tinyns_diagnostics_json(results: dict, run_dir: str, opts) -> str | None:
    if getattr(opts, "sampler", None) != "tinyns" and not results.get("tinyns_runtime_diagnostics"):
        return None
    payload = {
        "tinyns_runtime_diagnostics": results.get("tinyns_runtime_diagnostics") or {},
        "tinyns_resolved_config": getattr(opts, "tinyns_resolved_config", None),
    }
    if results.get("tinyns_diagnostics") is not None:
        payload["tinyns_diagnostics"] = results["tinyns_diagnostics"]
    if results.get("tinyns_summary") is not None:
        payload["tinyns_summary"] = results["tinyns_summary"]
    path = os.path.join(run_dir, "tinyns_diagnostics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe_tinyns_value(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


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
    path     = os.path.join(run_dir, "results.hdf5")
    tmp_path = path + ".tmp"
    kw       = dict(compression="gzip", shuffle=True)
    dt       = h5py.string_dtype(encoding="utf-8")

    samples = np.asarray(results["samples"])
    N, ndim = samples.shape

    with h5py.File(tmp_path, "w") as f:

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
        f.attrs["mark_model"]      = getattr(opts, "mark_model", "none")
        f.attrs["mark_names"]      = ",".join(getattr(opts, "mark_names", ()) or ())
        f.attrs["complete_empty_pixel_policy"] = opts.complete_empty_pixel_policy
        f.attrs["catalog_sky_weighting"] = getattr(
            opts, "catalog_sky_weighting", "conditional"
        )
        # Survey z_depth (bounds the completion missing-galaxy budget): the CLI
        # override (None = unset) plus the resolved per-catalog values actually
        # used to build SurveyParams, JSON-encoded so a None entry (legacy
        # full-grid budget) round-trips distinctly from an unset attribute.
        f.attrs["survey_z_depth"] = (
            float(opts.survey_z_depth) if getattr(opts, "survey_z_depth", None) is not None
            else float("nan")
        )
        f.attrs["resolved_survey_z_depths"] = json.dumps(
            list(getattr(opts, "resolved_survey_z_depths", None) or [])
        )
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
        # Exactly one of gw_path (stored PE) / gw_flows_path (flow surrogates)
        # is set; record both so analyzers can tell the input mode apart.
        f.attrs["gw_path"]         = opts.gw_path or ""
        f.attrs["gw_flows_path"]   = getattr(opts, "gw_flows_path", None) or ""
        f.attrs["gwselection_path"] = opts.gwselection_path or ""
        f.attrs["pdet_flow_path"]  = getattr(opts, "pdet_flow_path", None) or ""
        f.attrs["survey_path"]     = opts.survey_path or ""
        # Multitracer: number of catalogs and the full aligned path list; the
        # scalar survey_path attr above is retained for single-catalog analyzers.
        f.attrs["n_catalogs"]      = int(getattr(opts, "n_catalogs", 1))
        _survey_paths = getattr(opts, "survey_paths", None)
        if not _survey_paths:
            _survey_paths = [opts.survey_path] if opts.survey_path else []
        # JSON-encoded so paths containing commas round-trip safely.
        f.attrs["survey_paths"]    = json.dumps(list(_survey_paths))
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
        f.attrs["tinyns_sample"]   = str(getattr(opts, "tinyns_sample", ""))
        f.attrs["tinyns_kernel"]   = str(getattr(opts, "tinyns_kernel", ""))
        f.attrs["tinyns_walks"]    = int(getattr(opts, "tinyns_walks", None) or 0)
        f.attrs["tinyns_replacement_chains"] = int(getattr(opts, "tinyns_replacement_chains", None) or 0)
        # "" = unset (fixed replacement_chains path); else the escalation schedule.
        f.attrs["tinyns_replacement_chain_schedule"] = str(
            getattr(opts, "tinyns_replacement_chain_schedule", None) or "")
        # 0 = unset (sampler auto-picks max(10000, walks*replacement_chains)).
        f.attrs["tinyns_max_attempts"] = int(getattr(opts, "tinyns_max_attempts", None) or 0)
        f.attrs["tinyns_preset"]   = str(getattr(opts, "tinyns_preset", ""))
        f.attrs["tinyns_bound"]    = str(getattr(opts, "tinyns_bound", ""))
        f.attrs["tinyns_step_scale"]  = float(getattr(opts, "tinyns_step_scale", None) or 0.0)
        write_tinyns_metadata(f.attrs, results, opts)
        if results.get("numpyro_diagnostics") is not None:
            try:
                f.attrs["numpyro_diagnostics"] = json.dumps(results["numpyro_diagnostics"], default=str)
            except (TypeError, ValueError):
                pass
        f.attrs["selection_neff_soft_guard"] = bool(
            getattr(opts, "selection_neff_soft_guard", False))
        f.attrs["max_likelihood_variance"] = float(
            getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
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
        # Flow-surrogate runs: JSON list of per-event flow names in row order.
        if meta.get("flow_event_names_json"):
            f.attrs["flow_event_names"] = meta["flow_event_names_json"]
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

    # Rename only after the file closes cleanly, so a failure mid-write (e.g.
    # an attr with no native HDF5 equivalent) leaves the old results.hdf5, if
    # any, untouched instead of a truncated one. os.replace is atomic on the
    # same filesystem, which run_dir always is (tmp_path is a sibling).
    os.replace(tmp_path, path)

    save_tinyns_diagnostics_json(results, run_dir, opts)
    return path


