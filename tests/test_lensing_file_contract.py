import json
from pathlib import Path

import h5py
import numpy as np

from darksirens.lensing import file_contract


def _write_pe(path: Path, n_events=2, nsamp=3):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "observed-lensing-pe-1.0"
        f.attrs["event_indexing"] = "global"
        f.attrs["n_events"] = n_events
        f.attrs["nobs"] = n_events
        f.attrs["nsamp"] = nsamp
        for name in ("m1det", "m2det", "dL", "chieff", "p_pe"):
            f.create_dataset(name, data=np.ones(n_events * nsamp))


def _write_catalog(path: Path, include_truth=True):
    events = [{"event_index": 0, "event_id": "e0", "gps_time": 1.0}, {"event_index": 1, "event_id": "e1", "gps_time": 2.0}]
    if include_truth:
        events[0]["truth_source_id"] = 7
    path.write_text(json.dumps({"format_version": "observed-lensing-catalog-1.0", "event_indexing": "global", "n_events": 2, "events": events}))


def test_valid_contract_files_pass_without_truth_or_labels(tmp_path):
    pe = tmp_path / "observed_gw_pe.h5"; _write_pe(pe)
    cat = tmp_path / "observed_catalog.json"; _write_catalog(cat, include_truth=False)
    cand = tmp_path / "candidate_pairs.json"
    cand.write_text(json.dumps({"format_version": "candidate-pairs-1.0", "n_events": 2, "pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0}]}))
    sel = tmp_path / "selection_inputs.h5"
    with h5py.File(sel, "w") as f:
        f.attrs["format_version"] = "lensing-selection-inputs-1.0"
        f.create_group("unlensed")
    cfg = tmp_path / "run_config.json"; cfg.write_text(json.dumps({"format_version": "lensing-run-config-1.0"}))

    assert file_contract.validate_observed_gw_pe(pe)["ok"]
    assert file_contract.validate_observed_catalog(cat)["ok"]
    report = file_contract.validate_candidate_pairs(cand)
    assert report["ok"]
    assert report["summary"]["labels_ignored_by_inference"] is True
    assert file_contract.validate_selection_inputs(sel)["ok"]
    assert file_contract.validate_run_config(cfg)["ok"]


def test_broken_contract_files_fail_usefully(tmp_path):
    pe = tmp_path / "bad_pe.h5"
    with h5py.File(pe, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
    report = file_contract.validate_observed_gw_pe(pe)
    assert not report["ok"]
    assert "format_version" in report["errors"][0]

    cand = tmp_path / "bad_candidate_pairs.json"
    cand.write_text(json.dumps({"format_version": "candidate-pairs-1.0", "n_events": 1, "pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0}]}))
    report = file_contract.validate_candidate_pairs(cand)
    assert not report["ok"]
    assert "out of range" in report["errors"][0]


def test_legacy_candidate_pairs_alias_is_accepted_with_warning(tmp_path):
    cand = tmp_path / "candidate_pairs.json"
    cand.write_text(json.dumps({"n_events": 2, "candidate_pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0, "label": "true"}]}))
    report = file_contract.validate_candidate_pairs(cand)
    assert report["ok"]
    assert report["summary"]["labels_present"] is True
    assert any("legacy" in w for w in report["warnings"])
