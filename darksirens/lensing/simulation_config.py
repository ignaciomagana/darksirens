"""Configuration helpers for simulated end-to-end lensing studies."""
from __future__ import annotations

import copy
import json
import math
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
        "pop_model": (str, {"powerlaw+peak", "powerlaw+peak@md"}, None),
        "include_lensed_singletons": (bool, None, None),
        "tau_A": ((int, float), 0.0, None), "tau_n": ((int, float), 0.0, None),
        "observation_times": (str, {"placeholder", "uniform"}, None),
        "t_obs_days": ((int, float), 0.0, None),
        "injection_proposal": (str, {"broad", "matched"}, None),
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
        "partition_mode": (str, None, None), "partition_component_mode": (str, {"global", "componentwise"}, None), "sampler": (str, {"dynesty", "tinyns"}, None),
        "nlive": (int, 1, None), "dlogz": ((int, float), 0.0, None), "max_exact_partitions": (int, 1, None), "max_component_events": (int, 1, None), "max_component_edges": (int, 1, None), "max_component_partitions": (int, 1, None), "max_total_partitions": (int, 1, None), "pair_batch_size": (int, 1, None),
        "y_nodes_pair": (int, 1, None), "diagnostics_only": (bool, None, None),
        "lens_prior_overrides": (dict, None, None), "fixed_parameter_values": (dict, None, None),
        "fix_lens_rate": (bool, None, None),
        "fix_population": (bool, None, None),
        "singleton_lensing": (str, {"off", "sl_mixture"}, None),
        "prior_overrides": (dict, None, None),
        "run_singles_only_ablation": (bool, None, None),
        "pe_max_per_pair": (int, 1, None),
        "sel_batch_size": (int, 1, None),
        "selection_neff_guard": (str, {"auto", "hard", "soft"}, None),
        "max_samples": (int, 0, None),
        "off_retry_on_nonfinite": (bool, None, None), "off_retry_n_unlensed_inj": (int, 1, None),
        "off_retry_nlive": (int, 1, None),
    },
    "study": {"cases": (list, None, None), "seed": (int, None, None), "profile": (str, None, None)},
}

DEFAULTS: dict[str, Any] = {
    "mock": {"n_universe": 4000, "n_singletons": 2, "n_lensed_pairs": 2, "nsamp": 48, "n_unlensed_inj": 1000, "n_lensed_inj": 1000, "conditioning": "fixed_counts", "pop_model": "powerlaw+peak", "include_lensed_singletons": False, "tau_A": None, "tau_n": None, "observation_times": "placeholder", "t_obs_days": 365.25, "injection_proposal": "broad"},
    "candidate_graph": {"max_edges_per_event": 2, "max_total_edges": 8, "include_time_marks": True, "include_sky_marks": True, "include_mass_distance_score": True, "edge_mark_prior_keys": ["log_sky_overlap"]},
    "selection": {"pair_tag_model": "snr_time_sky", "pair_tag_constant": 1.0, "pair_tag_perturb_logit": 0.0},
    "inference": {"partition_mode": "marginalize_exact", "partition_component_mode": "componentwise", "max_exact_partitions": 10000, "max_component_events": None, "max_component_edges": None, "max_component_partitions": 10000, "max_total_partitions": 10000, "sampler": "dynesty", "nlive": 32, "dlogz": 10.0, "pair_batch_size": 256, "y_nodes_pair": 64, "diagnostics_only": False, "fix_lens_rate": False, "fix_population": True, "singleton_lensing": "off", "prior_overrides": {}, "run_singles_only_ablation": False, "pe_max_per_pair": None, "sel_batch_size": None, "selection_neff_guard": None, "max_samples": 0, "fixed_parameter_values": {"tau_n": 3.0}, "lens_prior_overrides": {"log10_tau_A": [-5.0, -2.5]}, "off_retry_on_nonfinite": False, "off_retry_n_unlensed_inj": None, "off_retry_nlive": None},
    "study": {"cases": None, "seed": 2026, "profile": "tiny"},
}

PROFILE_DEFAULTS = {
    "tiny": DEFAULTS,
    "small": {**DEFAULTS, "mock": {**DEFAULTS["mock"], "n_universe": 12000, "n_singletons": 8, "n_lensed_pairs": 4, "nsamp": 96, "n_unlensed_inj": 4000, "n_lensed_inj": 5000}, "candidate_graph": {**DEFAULTS["candidate_graph"], "max_total_edges": 24}, "inference": {**DEFAULTS["inference"], "nlive": 80}},
    "paper": {**DEFAULTS, "mock": {**DEFAULTS["mock"], "n_universe": 120000, "n_singletons": 200, "n_lensed_pairs": 40, "nsamp": 1000, "n_unlensed_inj": 1000000, "n_lensed_inj": 300000, "injection_proposal": "matched"}, "candidate_graph": {**DEFAULTS["candidate_graph"], "max_total_edges": 400}, "inference": {**DEFAULTS["inference"], "nlive": 1000}},
}

# Range-check ``study.profile`` against the presets that actually exist.  The
# patch is late because PROFILE_DEFAULTS is defined below SCHEMA.  Without it
# an unknown name was accepted, stamped into the run metadata, and silently
# resolved to the TINY defaults -- a typo'd 'papre' ran a 4000-source toy
# under a "paper" label.
SCHEMA["study"]["profile"] = (str, set(PROFILE_DEFAULTS), None)

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
            numeric = typ in (int, float) or (
                isinstance(typ, tuple) and (int in typ or float in typ)
            )
            if typ is bool:
                ok = isinstance(value, bool)
            elif numeric:
                # ``True`` IS an int in Python, so bool has to be excluded
                # explicitly for every numeric key -- otherwise `tau_A: true`
                # validates and silently configures the sim with tau_A = 1.0.
                ok = isinstance(value, typ) and not isinstance(value, bool)
            else:
                ok = isinstance(value, typ)
            if not ok:
                errors.append(f"{section}.{key} has invalid type {type(value).__name__}"); continue
            if numeric and isinstance(value, float) and not math.isfinite(value):
                # json.loads accepts NaN/Infinity, and a NaN passes every
                # ``value < minimum`` comparison below (all comparisons with
                # NaN are False), so it would reach the simulation intact.
                errors.append(f"{section}.{key} must be a finite number, got {value!r}"); continue
            if isinstance(min_or_allowed, set) and value not in min_or_allowed:
                errors.append(f"{section}.{key} must be one of {sorted(min_or_allowed)}")
            elif isinstance(min_or_allowed, (int, float)) and float(value) < float(min_or_allowed):
                errors.append(f"{section}.{key} must be >= {min_or_allowed}")
    if errors:
        raise ValueError("invalid simulation config: " + "; ".join(errors))

def resolve_config(path: str | Path | None, overrides: list[str] | None = None, *, profile: str | None = None, allow_unknown: bool = False) -> dict[str, Any]:
    raw = read_config(path) if path else {}
    raw_study = raw.get("study") or {}
    selected = raw_study.get("profile") if isinstance(raw_study, dict) else None
    if selected is None:
        selected = profile or DEFAULTS["study"]["profile"]
    if not isinstance(selected, str) or selected not in PROFILE_DEFAULTS:
        # Fail closed: the old ``PROFILE_DEFAULTS.get(selected, DEFAULTS)``
        # answered a typo with the tiny preset while writing the typo into
        # resolved_config/run_manifest, so the run LOOKED like the profile it
        # was asked for and was a hundredth of its size.
        raise ValueError(
            f"unknown study.profile {selected!r}; valid choices are "
            f"{sorted(PROFILE_DEFAULTS)}"
        )
    base = PROFILE_DEFAULTS[selected]
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
