import json
import os
import sys

import h5py
import healpy as hp
import jax
import numpy as np

from darksirens.cli.common import _fixed_dark_energy_metadata

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
        f.attrs["mark_model"]      = getattr(opts, "mark_model", "none")
        f.attrs["mark_names"]      = ",".join(getattr(opts, "mark_names", ()) or ())
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
        f.attrs["tinyns_sample"]   = str(getattr(opts, "tinyns_sample", ""))
        f.attrs["tinyns_kernel"]   = str(getattr(opts, "tinyns_kernel", ""))
        f.attrs["tinyns_walks"]    = int(getattr(opts, "tinyns_walks", 0))
        f.attrs["tinyns_replacement_chains"] = int(getattr(opts, "tinyns_replacement_chains", 0))
        # "" = unset (fixed replacement_chains path); else the escalation schedule.
        f.attrs["tinyns_replacement_chain_schedule"] = str(
            getattr(opts, "tinyns_replacement_chain_schedule", None) or "")
        # 0 = unset (sampler auto-picks max(10000, walks*replacement_chains)).
        f.attrs["tinyns_max_attempts"] = int(getattr(opts, "tinyns_max_attempts", None) or 0)
        f.attrs["tinyns_slices"]   = int(getattr(opts, "tinyns_slices", 0))
        f.attrs["tinyns_slice_steps"] = int(getattr(opts, "tinyns_slice_steps", 0))
        f.attrs["tinyns_step_scale"]  = float(getattr(opts, "tinyns_step_scale", 0.0))
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


