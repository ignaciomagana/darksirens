import json
from pathlib import Path

import h5py
import numpy as np

from darksirens.lensing import file_contract


def _write_pe(path: Path, n_events=2, nsamp=3):
    # Loader-complete on purpose: the preflight now enforces the same
    # requirement table as load_gw_samples (review DS-02), so a fixture
    # carrying only the old five-dataset subset would (correctly) fail it.
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "observed-lensing-pe-1.0"
        f.attrs["event_indexing"] = "global"
        f.attrs["n_events"] = n_events
        f.attrs["nobs"] = n_events
        f.attrs["nsamp"] = nsamp
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        n = n_events * nsamp
        for name in ("m1det", "m2det", "dL", "p_pe", "m1src", "m2src"):
            f.create_dataset(name, data=np.ones(n))
        f.create_dataset("chieff", data=np.zeros(n))
        f.create_dataset("ra", data=np.linspace(0.1, 1.0, n))
        f.create_dataset("dec", data=np.linspace(-0.5, 0.5, n))


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
        # A present group must carry its datasets: an EMPTY group used to be
        # "contract valid" and then fail at load time (review P2-06).
        f.create_group("unlensed").create_dataset("dL", data=np.ones(4))
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


def test_canonical_lensed_injection_selection_file_is_accepted(tmp_path):
    path = tmp_path / "mock_lensed_injections.h5"
    with h5py.File(path, "w") as f:
        for name in (
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
        ):
            f.create_dataset(name, data=np.ones(3))
    report = file_contract.validate_selection_inputs(path)
    assert report["ok"]
    assert report["summary"]["selection_kind"] == "lensed"
    assert report["summary"]["canonical_lensed_injections"] is True


def test_preflight_and_loader_agree_on_requirements(tmp_path):
    """Property test over the shared table: dropping ANY required dataset or
    attr from a selection file must fail the preflight (review DS-02 -- the
    old preflight passed files that then failed at load on m1src/m2src)."""
    from darksirens.gw import store_contract

    contract = store_contract.contract_for("gwcat-selection-1.0")

    def _write_full(path):
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["ndraw"] = 100
            f.attrs["chi_eff_swap_applied"] = True
            n = 4
            for name in contract.datasets:
                if name == "chieff":
                    f.create_dataset(name, data=np.zeros(n))
                elif name in ("ra", "dec"):
                    f.create_dataset(name, data=np.linspace(0.1, 0.9, n))
                else:
                    f.create_dataset(name, data=np.ones(n))

    full = tmp_path / "sel_full.h5"
    _write_full(full)
    assert file_contract.validate_selection_inputs(full)["ok"]

    for missing in contract.datasets:
        path = tmp_path / f"sel_missing_{missing}.h5"
        _write_full(path)
        with h5py.File(path, "a") as f:
            del f[missing]
        report = file_contract.validate_selection_inputs(path)
        assert not report["ok"], f"preflight passed without dataset {missing!r}"
        assert any(missing in err for err in report["errors"])

    for missing in contract.attrs:
        path = tmp_path / f"sel_missing_attr_{missing}.h5"
        _write_full(path)
        with h5py.File(path, "a") as f:
            del f.attrs[missing]
        report = file_contract.validate_selection_inputs(path)
        assert not report["ok"], f"preflight passed without attr {missing!r}"
        assert any(missing in err for err in report["errors"])


def test_zero_pdraw_rejected(tmp_path):
    """A detected injection with zero draw density is a mis-specified prior:
    the likelihood would silently exclude it with the same mask that drops
    padding rows while Ndraw keeps counting it, biasing mu low."""
    from darksirens.gw import store_contract

    contract = store_contract.contract_for("gwcat-selection-1.0")
    path = tmp_path / "sel_zero_pdraw.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["ndraw"] = 100
        f.attrs["chi_eff_swap_applied"] = True
        n = 4
        for name in contract.datasets:
            if name == "chieff":
                f.create_dataset(name, data=np.zeros(n))
            elif name in ("ra", "dec"):
                f.create_dataset(name, data=np.linspace(0.1, 0.9, n))
            elif name == "pdraw":
                data = np.ones(n)
                data[1] = 0.0
                f.create_dataset(name, data=data)
            else:
                f.create_dataset(name, data=np.ones(n))
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("pdraw" in err and "non-positive" in err for err in report["errors"])


def test_zero_p_pe_accepted_negative_rejected(tmp_path):
    """Zero p_pe is legal (the likelihood masks prior_wt > 0, and the shipped
    whitelist product carries exactly one zero); NEGATIVE density is not."""
    pe_ok = tmp_path / "pe_zero_ppe.h5"
    _write_pe(pe_ok)
    with h5py.File(pe_ok, "a") as f:
        data = np.asarray(f["p_pe"])
        data[0] = 0.0
        del f["p_pe"]
        f.create_dataset("p_pe", data=data)
    assert file_contract.validate_observed_gw_pe(pe_ok)["ok"]

    pe_bad = tmp_path / "pe_negative_ppe.h5"
    _write_pe(pe_bad)
    with h5py.File(pe_bad, "a") as f:
        data = np.asarray(f["p_pe"])
        data[0] = -1.0
        del f["p_pe"]
        f.create_dataset("p_pe", data=data)
    report = file_contract.validate_observed_gw_pe(pe_bad)
    assert not report["ok"]
    assert any("p_pe" in err and "negative" in err for err in report["errors"])
