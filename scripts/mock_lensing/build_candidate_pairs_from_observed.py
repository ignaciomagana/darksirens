#!/usr/bin/env python
"""Build simulated lensing candidate-pair graphs from observed-event data.

The builder scores unordered event pairs using only observed catalog metadata and
posterior samples.  Truth metadata, when present, is used only to attach optional
validation labels and never to decide which edges are kept.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from darksirens.lensing.observed_catalog import load_observed_catalog, validate_observed_catalog

FORMAT_VERSION = "candidate-pairs-1.0"


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _dataset(f: h5py.File, *names: str) -> np.ndarray:
    for name in names:
        if name in f:
            return np.asarray(f[name], dtype=float)
    raise KeyError(f"none of datasets {names!r} found in {f.filename}")


def _read_event_samples(path: str | Path, n_events: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        nsamp = int(f.attrs.get("nsamp", -1))
        if nsamp <= 0:
            m1 = _dataset(f, "m1det")
            if len(m1) % n_events != 0:
                raise ValueError("cannot infer nsamp from m1det length and n_events")
            nsamp = len(m1) // n_events
        arrays = {
            "m1det": _dataset(f, "m1det").reshape(n_events, nsamp),
            "dL": _dataset(f, "dL", "dL_app").reshape(n_events, nsamp),
        }
        if "m2det" in f:
            arrays["q"] = (_dataset(f, "m2det") / np.maximum(_dataset(f, "m1det"), 1e-12)).reshape(n_events, nsamp)
        elif "q" in f:
            arrays["q"] = _dataset(f, "q").reshape(n_events, nsamp)
        else:
            arrays["q"] = np.ones((n_events, nsamp))
        if "chieff" in f:
            arrays["chieff"] = _dataset(f, "chieff").reshape(n_events, nsamp)
        return arrays


def _event_summary(samples: dict[str, np.ndarray], idx: int) -> dict[str, tuple[float, float]]:
    out = {}
    for key, arr in samples.items():
        vals = np.asarray(arr[idx], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = (math.nan, math.inf)
            continue
        med = float(np.median(vals))
        sig = float(0.5 * (np.percentile(vals, 84) - np.percentile(vals, 16)))
        out[key] = (med, max(sig, 1e-6 * max(abs(med), 1.0)))
    return out


def _gaussian_log_score(a: tuple[float, float], b: tuple[float, float], *, log_space: bool = False) -> float:
    ma, sa = a; mb, sb = b
    if not all(np.isfinite([ma, sa, mb, sb])) or sa <= 0 or sb <= 0 or ma <= 0 and log_space or mb <= 0 and log_space:
        return -50.0
    if log_space:
        ma, mb = math.log(ma), math.log(mb)
        sa = sa / max(a[0], 1e-12); sb = sb / max(b[0], 1e-12)
    sig = math.sqrt(sa * sa + sb * sb)
    z = (ma - mb) / max(sig, 1e-12)
    return float(-0.5 * z * z)


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _log_sky_overlap(
    ei: dict[str, Any], ej: dict[str, Any], *, sigma_floor_rad: float = 0.0
) -> float:
    try:
        rai = float(ei["ra_mean"])
        deci = float(ei["dec_mean"])
        si = float(ei["sky_sigma_rad"])
        raj = float(ej["ra_mean"])
        decj = float(ej["dec_mean"])
        sj = float(ej["sky_sigma_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "sky marks require ra_mean, dec_mean, and sky_sigma_rad for every event"
        ) from exc
    si = max(si, float(sigma_floor_rad), 1e-12)
    sj = max(sj, float(sigma_floor_rad), 1e-12)
    dec_bar = 0.5 * (deci + decj)
    dra = _angle_delta(rai, raj) * math.cos(dec_bar)
    ddec = deci - decj
    d2 = dra * dra + ddec * ddec
    var = si * si + sj * sj
    out = -math.log(2.0 * math.pi * var) - 0.5 * d2 / var
    return float(out) if math.isfinite(out) else -1.0e300

def _truth_label(ei: dict[str, Any], ej: dict[str, Any]) -> str:
    same_source = ei.get("truth_source_id") is not None and ei.get("truth_source_id") == ej.get("truth_source_id")
    both_lensed = bool(ei.get("truth_is_lensed_image")) and bool(ej.get("truth_is_lensed_image"))
    different_images = ei.get("truth_image_index") != ej.get("truth_image_index")
    return "true" if same_source and both_lensed and different_images else "wrong"


def _looks_like_placeholder_time_marks(deltas: list[float], sigmas: list[float]) -> bool:
    """Return True when all requested time marks match the legacy 1s placeholder."""
    if not deltas or len(deltas) != len(sigmas):
        return False
    return all(
        math.isclose(float(dt), 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(sig), 1.0, rel_tol=0.0, abs_tol=1e-12)
        for dt, sig in zip(deltas, sigmas)
    )


def build_candidate_pairs(*, gw_path: str | Path, observed_catalog_path: str | Path, truth_path: str | Path | None = None,
                          max_edges_per_event: int = 4, max_total_edges: int = 1000,
                          time_window_sec: float = math.inf, mass_distance_top_k: int = 0,
                          include_time_marks: bool = False, include_truth_labels: bool = False,
                          include_sky_marks: bool = True, sky_overlap_weight: float = 0.0,
                          sky_sigma_floor_rad: float = 1e-3, seed: int = 0) -> dict[str, Any]:
    del truth_path  # Labels are read from observed-catalog truth fields when present.
    rng = np.random.default_rng(seed)
    catalog = load_observed_catalog(observed_catalog_path)
    validate_observed_catalog(catalog, path=observed_catalog_path)
    events = catalog["events"]
    n_events = int(catalog["n_events"])
    samples = _read_event_samples(gw_path, n_events)
    summaries = [_event_summary(samples, i) for i in range(n_events)]

    candidates: list[dict[str, Any]] = []
    for i in range(n_events):
        for j in range(i + 1, n_events):
            ti = events[i].get("gps_time")
            tj = events[j].get("gps_time")
            delta_t = abs(float(ti) - float(tj)) if ti is not None and tj is not None else None
            if delta_t is not None and np.isfinite(time_window_sec) and delta_t > time_window_sec:
                continue
            mass_score = _gaussian_log_score(summaries[i]["m1det"], summaries[j]["m1det"], log_space=True)
            q_score = _gaussian_log_score(summaries[i]["q"], summaries[j]["q"])
            # Lensed images of the same source can differ in apparent distance; penalize only extreme mismatch.
            dli, _ = summaries[i]["dL"]; dlj, _ = summaries[j]["dL"]
            distance_score = -0.5 * max(0.0, abs(math.log(max(dli, 1e-9) / max(dlj, 1e-9))) - math.log(10.0)) ** 2
            chi_score = _gaussian_log_score(summaries[i]["chieff"], summaries[j]["chieff"]) if "chieff" in samples else 0.0
            time_score = 0.0 if delta_t is None else -math.log1p(delta_t / max(time_window_sec if np.isfinite(time_window_sec) else 86400.0, 1.0))
            log_mass_distance_score = float(mass_score + 0.25 * q_score + distance_score + 0.25 * chi_score)
            log_sky_overlap = None
            if include_sky_marks:
                log_sky_overlap = _log_sky_overlap(
                    events[i], events[j], sigma_floor_rad=sky_sigma_floor_rad
                )
            score = float(-3.0 + log_mass_distance_score + 0.25 * time_score
                          + float(sky_overlap_weight) * (log_sky_overlap or 0.0)
                          + rng.uniform(0.0, 1e-12))
            edge = {"i": i, "j": j, "log_prior_odds": score}
            if include_truth_labels:
                edge["label"] = _truth_label(events[i], events[j])
            marks = {"log_mass_distance_score": log_mass_distance_score}
            if delta_t is not None and include_time_marks:
                marks["delta_t_obs"] = float(delta_t)
                marks["sigma_delta_t"] = float(max(1.0, min(abs(delta_t) * 0.1, 86400.0)))
            if include_sky_marks:
                marks["log_sky_overlap"] = float(log_sky_overlap)
            edge["marks"] = marks
            candidates.append(edge)

    if mass_distance_top_k and mass_distance_top_k > 0:
        candidates.sort(key=lambda e: e["marks"].get("log_mass_distance_score", -np.inf), reverse=True)
        candidates = candidates[:mass_distance_top_k]
    candidates.sort(key=lambda e: (e["log_prior_odds"], -e["i"], -e["j"]), reverse=True)

    kept: list[dict[str, Any]] = []
    degree = [0] * n_events
    for edge in candidates:
        if len(kept) >= max_total_edges:
            break
        i = int(edge["i"]); j = int(edge["j"])
        if degree[i] >= max_edges_per_event or degree[j] >= max_edges_per_event:
            continue
        degree[i] += 1; degree[j] += 1
        kept.append(edge)

    if include_time_marks:
        deltas = [
            float(edge["marks"]["delta_t_obs"])
            for edge in kept
            if "delta_t_obs" in edge.get("marks", {})
        ]
        sigmas = [
            float(edge["marks"]["sigma_delta_t"])
            for edge in kept
            if "sigma_delta_t" in edge.get("marks", {})
        ]
        if _looks_like_placeholder_time_marks(deltas, sigmas):
            for edge in kept:
                edge.get("marks", {}).pop("delta_t_obs", None)
                edge.get("marks", {}).pop("sigma_delta_t", None)

    return {
        "format_version": FORMAT_VERSION,
        "n_events": n_events,
        "pairs": kept,
        "candidate_pairs": kept,  # Backward-compatible alias for existing inference releases.
        "builder": {
            "name": "build_candidate_pairs_from_observed",
            "gw_path": str(gw_path),
            "observed_catalog_path": str(observed_catalog_path),
            "max_edges_per_event": int(max_edges_per_event),
            "max_total_edges": int(max_total_edges),
            "time_window_sec": None if not np.isfinite(time_window_sec) else float(time_window_sec),
            "mass_distance_top_k": int(mass_distance_top_k),
            "include_time_marks": bool(include_time_marks),
            "include_truth_labels": bool(include_truth_labels),
            "include_sky_marks": bool(include_sky_marks),
            "sky_overlap_weight": float(sky_overlap_weight),
            "sky_sigma_floor_rad": float(sky_sigma_floor_rad),
            "seed": int(seed),
        },
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gw_path", required=True)
    p.add_argument("--observed_catalog_path", required=True)
    p.add_argument("--truth_path", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max_edges_per_event", type=int, default=4)
    p.add_argument("--max_total_edges", type=int, default=1000)
    p.add_argument("--time_window_sec", type=float, default=math.inf)
    p.add_argument("--mass_distance_top_k", type=int, default=0)
    p.add_argument("--include_time_marks", type=_str_to_bool, default=False)
    p.add_argument("--include_truth_labels", type=_str_to_bool, default=False)
    p.add_argument("--include_sky_marks", type=_str_to_bool, default=True)
    p.add_argument("--sky_overlap_weight", type=float, default=0.0)
    p.add_argument("--sky_sigma_floor_rad", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    data = build_candidate_pairs(
        gw_path=args.gw_path,
        observed_catalog_path=args.observed_catalog_path,
        truth_path=args.truth_path,
        max_edges_per_event=args.max_edges_per_event,
        max_total_edges=args.max_total_edges,
        time_window_sec=args.time_window_sec,
        mass_distance_top_k=args.mass_distance_top_k,
        include_time_marks=args.include_time_marks,
        include_truth_labels=args.include_truth_labels,
        include_sky_marks=args.include_sky_marks,
        sky_overlap_weight=args.sky_overlap_weight,
        sky_sigma_floor_rad=args.sky_sigma_floor_rad,
        seed=args.seed,
    )
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
