"""Configuration helpers for simulated end-to-end lensing studies."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

SCHEMA: dict[str, dict[str, tuple[type | tuple[type, ...], Any, Any]]] = {
    "mock": {
        "n_universe": (int, 1, None), "n_singletons": (int, 0, None), "n_lensed_pairs": (int, 0, None),
        "nsamp": (int, 1, None), "n_unlensed_inj": (int, 1, None), "n_lensed_inj": (int, 1, None),
        "conditioning": (str, {"fixed_counts", "poisson_counts"}, None),
    },
    "candidate_graph": {
        "max_edges_per_event": (int, 0, None), "max_total_edges": (int, 0, None),
        "include_time_marks": (bool, None, None), "include_sky_marks": (bool, None, None),
        "include_mass_distance_score": (bool, None, None), "edge_mark_prior_keys": (list, None, None),
    },
    "selection": {
        "pair_tag_model": (str, None, None), "pair_tag_constant": ((int, float), 0.0, None),
        "pair_tag_perturb_logit": ((int, float), None, None),
    },
    "inference": {
        "partition_mode": (str, None, None), "sampler": (str, {"dynesty", "tinyns"}, None),
        "nlive": (int, 1, None), "dlogz": ((int, float), 0.0, None), "pair_batch_size": (int, 1, None),
        "y_nodes_pair": (int, 1, None), "diagnostics_only": (bool, None, None),
    },
    "study": {"cases": (list, None, None), "seed": (int, None, None), "profile": (str, None, None)},
}

DEFAULTS: dict[str, Any] = {
    "mock": {"n_universe": 4000, "n_singletons": 2, "n_lensed_pairs": 2, "nsamp": 48, "n_unlensed_inj": 1000, "n_lensed_inj": 1000, "conditioning": "fixed_counts"},
    "candidate_graph": {"max_edges_per_event": 2, "max_total_edges": 8, "include_time_marks": True, "include_sky_marks": True, "include_mass_distance_score": True, "edge_mark_prior_keys": ["log_sky_overlap"]},
    "selection": {"pair_tag_model": "snr_time_sky", "pair_tag_constant": 1.0, "pair_tag_perturb_logit": 0.0},
    "inference": {"partition_mode": "marginalize_exact", "sampler": "dynesty", "nlive": 32, "dlogz": 10.0, "pair_batch_size": 256, "y_nodes_pair": 64, "diagnostics_only": False},
    "study": {"cases": None, "seed": 2026, "profile": "tiny"},
}

PROFILE_DEFAULTS = {
    "tiny": DEFAULTS,
    "small": {**DEFAULTS, "mock": {**DEFAULTS["mock"], "n_universe": 12000, "n_singletons": 8, "n_lensed_pairs": 4, "nsamp": 96, "n_unlensed_inj": 4000, "n_lensed_inj": 5000}, "candidate_graph": {**DEFAULTS["candidate_graph"], "max_total_edges": 24}, "inference": {**DEFAULTS["inference"], "nlive": 80}},
    "paper": {**DEFAULTS, "mock": {**DEFAULTS["mock"], "n_universe": 120000, "n_singletons": 200, "n_lensed_pairs": 40, "nsamp": 1000, "n_unlensed_inj": 200000, "n_lensed_inj": 300000}, "candidate_graph": {**DEFAULTS["candidate_graph"], "max_total_edges": 400}, "inference": {**DEFAULTS["inference"], "nlive": 1000}},
}

def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out

def read_config(path: str | Path) -> dict[str, Any]:
    p = Path(path); text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ValueError("YAML config requested but PyYAML is not installed; use JSON or install PyYAML")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ValueError("simulation config must be a mapping/object")
    return data

def parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        low = value.lower()
        if low == "true": return True
        if low == "false": return False
        if low == "null": return None
        return value

def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be dotted.key=value, got {item!r}")
        key, raw = item.split("=", 1); parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise ValueError(f"override parent {part!r} is not a mapping")
        cur[parts[-1]] = parse_scalar(raw)
    return out

def validate_config(config: dict[str, Any], *, allow_unknown: bool = False) -> None:
    errors: list[str] = []
    for section, values in config.items():
        if section not in SCHEMA:
            if not allow_unknown: errors.append(f"unknown section {section!r}")
            continue
        if not isinstance(values, dict):
            errors.append(f"section {section!r} must be a mapping"); continue
        for key, value in values.items():
            if key not in SCHEMA[section]:
                if not allow_unknown: errors.append(f"unknown key {section}.{key}")
                continue
            typ, min_or_allowed, _ = SCHEMA[section][key]
            if value is None:
                continue
            if typ is bool:
                ok = isinstance(value, bool)
            elif typ is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            else:
                ok = isinstance(value, typ)
            if not ok:
                errors.append(f"{section}.{key} has invalid type {type(value).__name__}"); continue
            if isinstance(min_or_allowed, set) and value not in min_or_allowed:
                errors.append(f"{section}.{key} must be one of {sorted(min_or_allowed)}")
            elif isinstance(min_or_allowed, (int, float)) and float(value) < float(min_or_allowed):
                errors.append(f"{section}.{key} must be >= {min_or_allowed}")
    if errors:
        raise ValueError("invalid simulation config: " + "; ".join(errors))

def resolve_config(path: str | Path | None, overrides: list[str] | None = None, *, profile: str | None = None, allow_unknown: bool = False) -> dict[str, Any]:
    raw = read_config(path) if path else {}
    selected = raw.get("study", {}).get("profile", profile or DEFAULTS["study"]["profile"])
    base = PROFILE_DEFAULTS.get(selected, DEFAULTS)
    cfg = _merge(base, raw)
    cfg = apply_overrides(cfg, overrides or [])
    validate_config(cfg, allow_unknown=allow_unknown)
    return cfg

def write_config(path: str | Path, config: dict[str, Any]) -> None:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            p.write_text(yaml.safe_dump(config, sort_keys=False))
            return
        p = p.with_suffix(".json")
    p.write_text(json.dumps(config, indent=2, allow_nan=True) + "\n")
