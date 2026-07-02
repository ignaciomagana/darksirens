"""Formal file-contract validators for unified lensing analyses.

The simulated lensing study and future real-data adapters communicate with the
inference CLI through these files.  Validators return structured reports instead
of raising so they can be used by preflight and standalone scripts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from darksirens.lensing.observed_catalog import (
    EVENT_INDEXING,
    PE_FORMAT_VERSION,
    load_observed_catalog,
    validate_observed_catalog as _validate_observed_catalog_obj,
)
from darksirens.lensing.partitions import validate_candidate_pairs as _validate_candidate_pairs_obj

CANDIDATE_PAIRS_FORMAT_VERSION = "candidate-pairs-1.0"
SELECTION_INPUTS_FORMAT_VERSION = "lensing-selection-inputs-1.0"
RUN_CONFIG_FORMAT_VERSIONS = {"lensing-run-config-1.0"}


def _report(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}
    try:
        result = fn()
        warnings.extend(result.pop("warnings", []))
        summary.update(result)
    except Exception as exc:  # validators are report-oriented by contract
        errors.append(str(exc))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


def _finite_dataset(f: h5py.File, name: str, *, size: int | None = None) -> int:
    if name not in f:
        raise ValueError(f"missing required dataset {name!r}")
    arr = np.asarray(f[name])
    if arr.ndim != 1:
        raise ValueError(f"dataset {name!r} must be one-dimensional")
    if size is not None and arr.size != size:
        raise ValueError(f"dataset {name!r} length {arr.size} != expected {size}")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"dataset {name!r} contains non-finite values")
    return int(arr.size)


def validate_observed_gw_pe(path: str | Path) -> dict[str, Any]:
    """Validate ``observed_gw_pe.h5`` / ``mock_observed_gw_pe.h5``."""

    def _impl() -> dict[str, Any]:
        with h5py.File(path, "r") as f:
            fmt = _decode(f.attrs.get("format_version"))
            if fmt != PE_FORMAT_VERSION:
                raise ValueError(f"observed GW PE format_version must be {PE_FORMAT_VERSION!r}, got {fmt!r}")
            indexing = _decode(f.attrs.get("event_indexing"))
            if indexing != EVENT_INDEXING:
                raise ValueError(f"observed GW PE event_indexing must be {EVENT_INDEXING!r}, got {indexing!r}")
            n_events = int(f.attrs.get("n_events", f.attrs.get("nobs", -1)))
            nsamp = int(f.attrs.get("nsamp", -1))
            if n_events < 0 or nsamp <= 0:
                raise ValueError("observed GW PE requires non-negative n_events and positive nsamp attrs")
            expected = n_events * nsamp
            for name in ("m1det", "m2det", "dL", "chieff", "p_pe"):
                _finite_dataset(f, name, size=expected)
            warnings = []
            if "nobs" in f.attrs and int(f.attrs["nobs"]) != n_events:
                raise ValueError("nobs attr must match n_events")
            if "q" not in f and "m2det" not in f:
                raise ValueError("observed GW PE requires m2det or q samples")
            if "observed_catalog_sha256" not in f.attrs:
                warnings.append("observed_catalog_sha256 attr is recommended for provenance")
            return {"format_version": fmt, "event_indexing": indexing, "n_events": n_events, "nsamp": nsamp, "warnings": warnings}

    return _report(_impl)


def validate_observed_catalog(path: str | Path) -> dict[str, Any]:
    def _impl() -> dict[str, Any]:
        data = load_observed_catalog(path)
        meta = _validate_observed_catalog_obj(data, path=path)
        truth_keys = [k for k in data if k.startswith("truth") or k == "event_records"]
        return {**meta, "truth_fields_present": bool(truth_keys), "truth_fields": truth_keys}
    return _report(_impl)


def validate_candidate_pairs(path: str | Path) -> dict[str, Any]:
    def _impl() -> dict[str, Any]:
        data = _read_json(path)
        n_events, pairs = _validate_candidate_pairs_obj(data)
        warnings = []
        if data.get("format_version") is None:
            warnings.append("candidate_pairs format_version is missing; accepted as legacy candidate-pairs-1.0")
        if "pairs" not in data and "candidate_pairs" in data:
            warnings.append("candidate_pairs uses legacy 'candidate_pairs' edge-list alias; prefer 'pairs'")
        label_count = sum(1 for p in pairs if p.label is not None)
        mark_keys = sorted({key for p in pairs for key in p.marks.keys})
        return {"format_version": data.get("format_version", CANDIDATE_PAIRS_FORMAT_VERSION), "n_events": n_events, "n_candidate_pairs": len(pairs), "labels_present": label_count > 0, "labels_ignored_by_inference": True, "mark_keys": mark_keys, "warnings": warnings}
    return _report(_impl)


def validate_selection_inputs(path: str | Path) -> dict[str, Any]:
    """Validate consolidated or legacy selection HDF5 inputs.

    Consolidated files use root ``format_version='lensing-selection-inputs-1.0'``.
    Legacy simulation component files (``gwcat-selection-1.0`` and lensed
    injection files) are accepted so existing split-pair simulations remain
    preflightable while the new contract is adopted.
    """

    def _impl() -> dict[str, Any]:
        with h5py.File(path, "r") as f:
            fmt = _decode(f.attrs.get("format_version"))
            warnings = []
            if fmt == SELECTION_INPUTS_FORMAT_VERSION:
                groups = list(f.keys())
                if "unlensed" not in f and "lensed" not in f:
                    raise ValueError("selection_inputs.h5 requires at least an 'unlensed' or 'lensed' group")
                return {"format_version": fmt, "groups": groups, "warnings": warnings}
            if fmt == "gwcat-selection-1.0":
                for name in ("m1det", "m2det", "dL", "chieff", "pdraw"):
                    _finite_dataset(f, name)
                warnings.append("legacy unlensed selection file accepted; prefer consolidated selection_inputs.h5")
                return {"format_version": fmt, "selection_kind": "unlensed", "n_injections": int(len(f["dL"])), "ndraw": int(f.attrs.get("ndraw", len(f["dL"]))), "warnings": warnings}
            # Lensed injection writer has varied format strings; accept both legacy
            # names and the canonical mock writer output used by simulated studies.
            canonical_lensed = {
                "source_id",
                "image_id",
                "m1_src",
                "q_src",
                "z_src",
                "chieff",
                "y_source",
                "mu",
                "detected",
                "p_prop_src",
                "p_prop_y",
            }
            datasets = set(f.keys())
            if canonical_lensed.issubset(datasets):
                n = _finite_dataset(f, "source_id")
                for name in sorted(canonical_lensed - {"source_id"}):
                    _finite_dataset(f, name, size=n)
                warnings.append("canonical lensed injection file accepted; prefer consolidated selection_inputs.h5")
                return {"format_version": fmt, "selection_kind": "lensed", "datasets": list(f.keys()), "canonical_lensed_injections": True, "n_injections": n, "p_tag_present": "p_tag_per_source" in f, "warnings": warnings}
            if "p_tag_per_source" in f or "m1src" in f or "m1det_image0" in f:
                warnings.append("legacy lensed injection file accepted; prefer consolidated selection_inputs.h5")
                return {"format_version": fmt, "selection_kind": "lensed", "datasets": list(f.keys()), "p_tag_present": "p_tag_per_source" in f, "warnings": warnings}
            raise ValueError(f"unrecognized selection input format_version {fmt!r}")
    return _report(_impl)


def validate_run_config(path: str | Path) -> dict[str, Any]:
    def _impl() -> dict[str, Any]:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            data = _read_json(path)
        else:
            try:
                import yaml
            except Exception as exc:
                raise ValueError("YAML run_config validation requires PyYAML") from exc
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("run_config must be a mapping/object")
        warnings = []
        fmt = data.get("format_version")
        if fmt is None:
            warnings.append("run_config format_version missing; accepted for existing resolved simulation configs")
        elif fmt not in RUN_CONFIG_FORMAT_VERSIONS:
            raise ValueError(f"run_config format_version must be one of {sorted(RUN_CONFIG_FORMAT_VERSIONS)}")
        return {"format_version": fmt, "top_level_keys": sorted(map(str, data.keys())), "warnings": warnings}
    return _report(_impl)
