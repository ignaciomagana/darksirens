"""Preflight validation for spectral-siren strong-lensing inference inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from darksirens.lensing.partitions import validate_candidate_pairs, enumerate_compatible_partitions
from darksirens.likelihood.pair_kde import validate_pair_prior_wt


def _get(opts: Any, name: str, default=None):
    return getattr(opts, name, default)


def _read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _exists(path, errors, label):
    if not path:
        errors.append(f"missing {label}")
        return False
    if not Path(path).exists():
        errors.append(f"{label} does not exist: {path}")
        return False
    return True


def _infer_gw(path, errors, summary):
    if not _exists(path, errors, "gw_path"):
        return None, None
    try:
        with h5py.File(path, "r") as f:
            n_events = int(f.attrs.get("nobs", f.attrs.get("n_events", -1)))
            nsamp = int(f.attrs.get("nsamp", -1))
            if n_events < 0 and "m1det" in f and nsamp > 0:
                n_events = int(len(f["m1det"]) // nsamp)
            if nsamp < 0 and n_events > 0 and "m1det" in f:
                nsamp = int(len(f["m1det"]) // n_events)
            if n_events >= 0:
                summary["n_events"] = n_events
            if nsamp >= 0:
                summary["nsamp"] = nsamp
            return (n_events if n_events >= 0 else None), (nsamp if nsamp >= 0 else None)
    except Exception as exc:
        errors.append(f"gw_path not readable: {path}: {exc}")
        return None, None


def _check_partition(path, n_events, errors, summary):
    if not _exists(path, errors, "partition_path"):
        return []
    try:
        data = _read_json(path)
    except Exception as exc:
        errors.append(f"partition_path not readable: {path}: {exc}")
        return []
    singletons = np.asarray(data.get("singleton_indices", []), dtype=int)
    pairs = np.asarray(data.get("pair_indices", []), dtype=int)
    if pairs.size == 0:
        pairs = pairs.reshape((0, 2))
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        errors.append(f"partition pair_indices shape must be (n_pairs, 2), got {pairs.shape}")
    if int(data.get("n_singletons", len(singletons))) != len(singletons):
        errors.append("partition n_singletons does not match singleton_indices")
    if int(data.get("n_pairs", pairs.shape[0] if pairs.ndim == 2 else -1)) != (pairs.shape[0] if pairs.ndim == 2 else -1):
        errors.append("partition n_pairs does not match pair_indices")
    used = list(map(int, singletons)) + [int(x) for x in pairs.reshape(-1)]
    if n_events is not None and any(i < 0 or i >= n_events for i in used):
        errors.append(f"partition index out of range for n_events={n_events}")
    if len(set(used)) != len(used):
        errors.append("fixed partition uses at least one event more than once")
    summary["n_pairs_partition"] = int(pairs.shape[0]) if pairs.ndim == 2 else None
    return [tuple(map(int, p)) for p in pairs.reshape((-1, 2))] if pairs.ndim == 2 else []


def _check_pair_pe(path, n_events, partition_pairs, opts, errors, summary, *, unified_observed_mode=False):
    if not path:
        if unified_observed_mode and _get(opts, "pair_marks", "none") == "none":
            summary["pair_pe_metadata_optional"] = True
            return
        errors.append("missing pair_pe_path")
        return
    if not Path(path).exists():
        errors.append(f"pair_pe_path does not exist: {path}")
        return
    try:
        with h5py.File(path, "r") as f:
            if "npairs" not in f.attrs:
                errors.append("pair_pe_path missing npairs attribute")
                npairs = 0
            else:
                npairs = int(f.attrs["npairs"])
            summary["n_pairs_pair_pe"] = npairs
            for k in range(npairs):
                pname = f"pair_{k}"
                if pname not in f:
                    errors.append(f"pair_pe_path missing group {pname}")
                    continue
                g = f[pname]
                if ("event_index_image0" in g.attrs) != ("event_index_image1" in g.attrs):
                    errors.append(f"{pname} must define both event_index_image0 and event_index_image1")
                elif "event_index_image0" in g.attrs:
                    pair = (int(g.attrs["event_index_image0"]), int(g.attrs["event_index_image1"]))
                    if pair[0] == pair[1] or pair[0] < 0 or pair[1] < 0 or (n_events is not None and (pair[0] >= n_events or pair[1] >= n_events)):
                        errors.append(f"{pname} event-index metadata out of range: {pair}")
                    if partition_pairs and k < len(partition_pairs) and pair != partition_pairs[k]:
                        errors.append(f"{pname} event-index metadata {pair} does not match partition pair {partition_pairs[k]}")
                has_dt = "delta_t_obs" in g.attrs or "delta_t_obs" in g
                has_sig = "sigma_delta_t" in g.attrs or "sigma_delta_t" in g or _get(opts, "pair_time_sigma_sec") is not None
                fixed_time_marks = (
                    _get(opts, "pair_marks", "none") == "time"
                    and _get(opts, "partition_mode", "fixed") == "fixed"
                )
                if fixed_time_marks:
                    if not has_dt:
                        errors.append(f"pair_marks=time requires delta_t_obs metadata for {pname}")
                    if not has_sig:
                        errors.append(f"pair_marks=time requires sigma_delta_t in {pname} or --pair_time_sigma_sec")
                if has_sig:
                    try:
                        sig = float(g.attrs["sigma_delta_t"] if "sigma_delta_t" in g.attrs else (np.asarray(g["sigma_delta_t"])[()] if "sigma_delta_t" in g else _get(opts, "pair_time_sigma_sec")))
                        if not np.isfinite(sig) or sig <= 0:
                            errors.append(f"{pname} sigma_delta_t must be positive")
                    except Exception as exc:
                        errors.append(f"{pname} sigma_delta_t not readable: {exc}")
                if unified_observed_mode:
                    # Unified mode treats pair_pe_path as optional metadata only;
                    # image PE groups may be absent and are never inference inputs.
                    continue
                for img in ("image0", "image1"):
                    if img not in g:
                        errors.append(f"{pname} missing {img} group")
                        continue
                    gi = g[img]
                    for dset in ("m1det", "q", "dL_app", "chieff", "prior_wt"):
                        if dset not in gi:
                            errors.append(f"{pname}/{img} missing dataset {dset}")
                    if "prior_wt" in gi:
                        try:
                            validate_pair_prior_wt(np.asarray(gi["prior_wt"]), context=f"{pname}/{img}/prior_wt")
                        except Exception as exc:
                            errors.append(str(exc))
    except Exception as exc:
        errors.append(f"pair_pe_path not readable: {path}: {exc}")


def _check_candidates(path, n_events, opts, errors, summary):
    if not _exists(path, errors, "candidate_pairs_path"):
        return
    try:
        data = _read_json(path)
        cand_n, pairs = validate_candidate_pairs(data)
        summary["n_candidate_pairs"] = len(pairs)
        if n_events is not None and cand_n != n_events:
            errors.append(f"candidate_pairs n_events={cand_n} does not match gw n_events={n_events}")
        if _get(opts, "pair_marks", "none") == "time":
            for pair in pairs:
                if pair.delta_t_obs is None or pair.sigma_delta_t is None:
                    errors.append(
                        "candidate pair "
                        f"({pair.i},{pair.j}) missing marks.delta_t_obs/sigma_delta_t "
                        "required by pair_marks=time"
                    )
        states = enumerate_compatible_partitions(cand_n, pairs, max_partitions=int(_get(opts, "max_exact_partitions", 10000)))
        summary["n_exact_partitions"] = len(states)
    except Exception as exc:
        errors.append(f"candidate_pairs invalid: {exc}")


def _check_lensed(path, errors, summary):
    if not _exists(path, errors, "lensed_injections_path"):
        return
    try:
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("Ndraw_sources", f.attrs.get("n_draw_sources", -1)))
            if n <= 0:
                errors.append("lensed_injections n_draw_sources/Ndraw_sources must be positive")
            ptag_present = False
            for name, is_log in (("log_p_tag_per_source", True), ("p_tag_per_source", False), ("log_p_tag", True), ("p_tag", False)):
                if name in f:
                    ptag_present = True
                    arr = np.asarray(f[name], dtype=float)
                    if is_log and (not np.all(np.isfinite(arr) | np.isneginf(arr)) or np.any(arr > 0)):
                        errors.append(f"{name} must be finite/-inf and <= 0 in log-space")
                    if not is_log and (not np.all(np.isfinite(arr)) or np.any((arr < 0) | (arr > 1))):
                        errors.append(f"{name} must be finite and in [0, 1]")
            summary["p_tag_present"] = ptag_present
            if not ptag_present:
                summary["p_tag_default"] = 1
    except Exception as exc:
        errors.append(f"lensed_injections_path not readable: {path}: {exc}")



def run_lensing_preflight(opts) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    summary = {"cluster_mode": _get(opts, "cluster_mode"), "partition_mode": _get(opts, "partition_mode", "fixed"), "pair_marks": _get(opts, "pair_marks", "none"), "p_tag_present": False}
    n_events, _ = _infer_gw(_get(opts, "gw_path"), errors, summary)
    _exists(_get(opts, "gwselection_path"), errors, "gwselection_path")
    partition_pairs = []
    unified_observed_mode = False
    if _get(opts, "cluster_mode") == "j2":
        if _get(opts, "partition_mode", "fixed") == "fixed":
            partition_pairs = _check_partition(_get(opts, "partition_path"), n_events, errors, summary)
            if n_events is not None and partition_pairs:
                used = [i for pair in partition_pairs for i in pair]
                singletons = []
                try:
                    singletons = list(map(int, _read_json(_get(opts, "partition_path")).get("singleton_indices", [])))
                except Exception:
                    pass
                unified_observed_mode = (max(singletons + used, default=-1) + 1 == n_events)
        elif _get(opts, "partition_mode") == "marginalize_exact":
            _check_candidates(_get(opts, "candidate_pairs_path"), n_events, opts, errors, summary)
            try:
                cand = _read_json(_get(opts, "candidate_pairs_path"))
                unified_observed_mode = n_events is not None and int(cand.get("n_events", -1)) == int(n_events)
            except Exception:
                unified_observed_mode = False
        summary["unified_observed_mode"] = bool(unified_observed_mode)
        _check_lensed(_get(opts, "lensed_injections_path"), errors, summary)
        _check_pair_pe(_get(opts, "pair_pe_path"), n_events, partition_pairs, opts, errors, summary, unified_observed_mode=unified_observed_mode)
    if _get(opts, "wl_selection") == "wl_lognormal" and _get(opts, "wl_backend") != "lognormal":
        warnings.append("wl_selection=wl_lognormal is only meaningful with wl_backend=lognormal")
    if _get(opts, "fix_lens_rate", True):
        if not np.isfinite(float(_get(opts, "sl_tau_A", 0))) or float(_get(opts, "sl_tau_A", 0)) <= 0:
            errors.append("sl_tau_A must be positive when lens rate is fixed")
        if not np.isfinite(float(_get(opts, "sl_tau_n", np.nan))):
            errors.append("sl_tau_n must be finite")
    elif _get(opts, "lens_prior_overrides"):
        try:
            over = json.loads(_get(opts, "lens_prior_overrides")) if isinstance(_get(opts, "lens_prior_overrides"), str) else _get(opts, "lens_prior_overrides")
            for key, bounds in over.items():
                lo, hi = map(float, bounds)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    errors.append(f"invalid lens_prior_overrides bounds for {key}")
        except Exception as exc:
            errors.append(f"lens_prior_overrides must be valid JSON bounds: {exc}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}
