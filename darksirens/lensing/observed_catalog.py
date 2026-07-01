"""Schema helpers for unified observed spectral-siren lensing catalogs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FORMAT_VERSION = "observed-lensing-catalog-1.0"
PE_FORMAT_VERSION = "observed-lensing-pe-1.0"
EVENT_INDEXING = "global"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_observed_catalog(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_observed_catalog(data: dict[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    """Validate and return compact metadata for an observed catalog JSON object."""
    label = f"observed_catalog {path}" if path is not None else "observed_catalog"
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{label} format_version must be {FORMAT_VERSION!r}")
    if data.get("event_indexing") != EVENT_INDEXING:
        raise ValueError(f"{label} event_indexing must be {EVENT_INDEXING!r}")
    if "n_events" not in data:
        raise ValueError(f"{label} missing n_events")
    if "events" not in data:
        raise ValueError(f"{label} missing events")
    try:
        n_events = int(data["n_events"])
    except Exception as exc:
        raise ValueError(f"{label} n_events must be an integer") from exc
    events = data["events"]
    if not isinstance(events, list):
        raise ValueError(f"{label} events must be a list")
    if n_events != len(events):
        raise ValueError(f"{label} n_events={n_events} does not equal len(events)={len(events)}")
    seen_ids: set[str] = set()
    for expected, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"{label} events[{expected}] must be an object")
        if "event_index" not in event:
            raise ValueError(f"{label} events[{expected}] missing event_index")
        if int(event["event_index"]) != expected:
            raise ValueError(f"{label} event_index must be contiguous 0..n_events-1")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{label} events[{expected}] missing non-empty event_id")
        if event_id in seen_ids:
            raise ValueError(f"{label} event_id values must be unique")
        seen_ids.add(event_id)
        if "gps_time" in event and event["gps_time"] is not None:
            try:
                gps_time = float(event["gps_time"])
            except Exception as exc:
                raise ValueError(f"{label} events[{expected}] gps_time must be finite if present") from exc
            if not np.isfinite(gps_time):
                raise ValueError(f"{label} events[{expected}] gps_time must be finite if present")
    return {
        "format_version": FORMAT_VERSION,
        "event_indexing": EVENT_INDEXING,
        "n_events": n_events,
        "path": str(path) if path is not None else None,
    }


def validate_observed_catalog_file(path: str | Path) -> dict[str, Any]:
    meta = validate_observed_catalog(load_observed_catalog(path), path=path)
    meta["sha256"] = sha256_file(path)
    return meta


def observed_catalog_metadata_from_hdf5(path: str | Path) -> dict[str, Any] | None:
    """Return observed-catalog PE attrs if an HDF5 GW PE file declares them."""
    with h5py.File(path, "r") as f:
        fmt = f.attrs.get("format_version", None)
        indexing = f.attrs.get("event_indexing", None)
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        if isinstance(indexing, bytes):
            indexing = indexing.decode()
        if fmt != PE_FORMAT_VERSION or indexing != EVENT_INDEXING:
            return None
        n_events = int(f.attrs.get("n_events", f.attrs.get("nobs", -1)))
        meta = {
            "pe_format_version": PE_FORMAT_VERSION,
            "event_indexing": EVENT_INDEXING,
            "n_events": n_events,
            "source": f.attrs.get("source", None),
        }
        if isinstance(meta["source"], bytes):
            meta["source"] = meta["source"].decode()
        for key in ("observed_catalog_path", "observed_catalog_sha256"):
            if key in f.attrs:
                value = f.attrs[key]
                meta[key] = value.decode() if isinstance(value, bytes) else str(value)
        return meta


def write_observed_pe_attrs(path: str | Path, *, n_events: int, catalog_path: str | Path | None = None, source: str = "mock_lensing") -> None:
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = PE_FORMAT_VERSION
        f.attrs["event_indexing"] = EVENT_INDEXING
        f.attrs["n_events"] = int(n_events)
        f.attrs["source"] = source
        if catalog_path is not None:
            f.attrs["observed_catalog_path"] = str(catalog_path)
            f.attrs["observed_catalog_sha256"] = sha256_file(catalog_path)
