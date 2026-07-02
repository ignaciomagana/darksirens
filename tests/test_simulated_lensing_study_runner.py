import json
import subprocess
import sys
from pathlib import Path

from scripts.mock_lensing.run_simulated_lensing_study import classify_run_outputs, combined_inference_status, evidence_delta, extract_logz, extract_midpoint_loglike, latest_attempt, preflight_status, recovery_metrics, posterior_probability_items, validate_known_inference_flags, write_preflight_summary, _run_logged


def test_simulated_lensing_study_dry_run_writes_plan(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "study"
    cmd = [sys.executable, "scripts/mock_lensing/run_simulated_lensing_study.py", "--workdir", str(workdir), "--profile", "tiny", "--dry_run", "true"]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    plan = json.loads((workdir / "validation_plan.json").read_text())
    assert plan["profile"] == "tiny"
    assert "A_no_true_pairs_sparse_wrong_graph" in plan["cases"]
    assert "H_ambiguous_components" in plan["cases"]
    assert (workdir / "run_manifest.json").exists()
    commands = json.dumps(plan)
    assert "--edge_mark_prior_keys" in commands
    assert "--edge_prior_marks" not in commands
    first = plan["cases"]["A_no_true_pairs_sparse_wrong_graph"]
    assert "off" in first
    assert "off_preflight" in first
    off_cmd = first["off"]
    assert "--cluster_mode" in off_cmd
    assert off_cmd[off_cmd.index("--cluster_mode") + 1] == "off"
    for j2_only_flag in ["--lensed_injections_path", "--pair_metadata_path", "--candidate_pairs_path", "--partition_mode", "--pair_marks", "--pair_tag_model"]:
        assert j2_only_flag not in off_cmd
    assert "--fix_lens_rate" in off_cmd
    assert off_cmd[off_cmd.index("--fix_lens_rate") + 1] == "true"
    f_case = plan["cases"]["F_true_pairs_no_time_marks"]["inference"]
    assert f_case[f_case.index("--pair_tag_model") + 1] == "snr_sky"


def test_run_logged_writes_stdout_stderr_for_nonzero_exit(tmp_path):
    log = _run_logged([sys.executable, "-c", "import sys; print(\"out msg\"); print(\"err msg\", file=sys.stderr); sys.exit(3)"], tmp_path / "cmd")
    assert log["return_code"] == 3
    assert Path(log["stdout_path"]).read_text().strip() == "out msg"
    assert Path(log["stderr_path"]).read_text().strip() == "err msg"


def test_latest_attempt_finds_failure_json(tmp_path):
    run = tmp_path / "run_root" / "pop__j2__dynesty__2026-01-01T00-00-00"
    run.mkdir(parents=True)
    (run / "failure.json").write_text(json.dumps({"stage": "sampler", "error_message": "boom"}))
    assert latest_attempt(tmp_path / "run_root") == run


def test_failure_payload_can_be_included_in_summary_record(tmp_path):
    run = tmp_path / "runs" / "case" / "attempt"
    run.mkdir(parents=True)
    failure = {"stage": "midpoint_loglike", "error_message": "bad shape"}
    (run / "failure.json").write_text(json.dumps(failure))
    summary = {"cases": {"case": {"status": "failed_j2_inference", "j2": {"run_dir": str(latest_attempt(tmp_path / "runs" / "case")), "failure": json.loads((run / "failure.json").read_text())}}}}
    assert summary["cases"]["case"]["j2"]["failure"] == failure


def test_truth_recovery_metrics_with_fake_posterior_probabilities():
    metrics = recovery_metrics({(0, 1), (2, 3)}, [((0, 1), 0.8), ((2, 3), 0.6), ((1, 2), 0.2)], {"expected_n_pairs": 1.6, "map_partition_pairs": [[0, 1], [2, 3]], "map_partition_n_pairs": 2})
    assert metrics["injected_n_pairs"] == 2
    assert metrics["map_n_pairs"] == 2
    assert metrics["true_edge_posterior_probability_mean"] == 0.7
    assert metrics["false_edge_posterior_probability_max"] == 0.2
    assert metrics["false_edge_posterior_probability_sum"] == 0.2
    assert metrics["map_partition_exact_truth_match"] is True


def test_posterior_probability_items_from_list(tmp_path):
    cand = {"pairs": [{"i": 4, "j": 1}, {"i": 2, "j": 3}]}
    path = tmp_path / "candidate_pairs.json"
    path.write_text(json.dumps(cand))
    assert posterior_probability_items({"posterior_pair_probabilities": [0.4, 0.5]}, path) == [((1, 4), 0.4), ((2, 3), 0.5)]


def test_command_validator_rejects_stale_edge_prior_marks():
    try:
        validate_known_inference_flags([sys.executable, "-m", "darksirens.cli.inference_lensing", "--edge_prior_marks", "log_sky_overlap"])
    except ValueError as exc:
        assert "--edge_prior_marks" in str(exc)
    else:
        raise AssertionError("validate_known_inference_flags should reject stale --edge_prior_marks")


def test_command_validator_accepts_edge_mark_prior_keys(tmp_path):
    cmd = [
        sys.executable, "-m", "darksirens.cli.inference_lensing",
        "--gw_path", "gw.h5", "--gwselection_path", "sel.h5", "--lensed_injections_path", "lens.h5",
        "--candidate_pairs_path", "pairs.json", "--partition_mode", "marginalize_exact", "--cluster_mode", "j2",
        "--edge_mark_prior_keys", "log_sky_overlap", "--save_path", str(tmp_path),
    ]
    validate_known_inference_flags(cmd)


def test_preflight_summary_writer_with_fake_records(tmp_path):
    summary = {"profile": "tiny", "cases": {"A": {"status": "passed_preflight", "j2": {"status": "passed_preflight"}, "off": {"status": "passed_preflight"}, "candidate_graph_summary": {"n_events": 2, "n_candidate_edges": 1, "n_components": 1, "component_n_partitions": [2]}, "warnings": ["w"], "errors": []}}}
    write_preflight_summary(tmp_path, summary)
    data = json.loads((tmp_path / "preflight_summary.json").read_text())
    assert data["cases"]["A"]["status"] == "passed_preflight"
    md = (tmp_path / "preflight_summary.md").read_text()
    assert "| case | j2 preflight | off preflight | n_events | n_edges | n_components | n_partitions | warnings | errors |" in md
    assert "| A | passed_preflight | passed_preflight | 2 | 1 | 1 | 2 | 1 | 0 |" in md


def test_preflight_only_dry_run_plan_has_preflight_no_generated_outputs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "study"
    cmd = [sys.executable, "scripts/mock_lensing/run_simulated_lensing_study.py", "--workdir", str(workdir), "--profile", "tiny", "--preflight_only", "true", "--dry_run", "true"]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    plan = json.loads((workdir / "validation_plan.json").read_text())
    commands = json.dumps(plan)
    assert "--preflight_only" in commands
    assert "preflight.json" in commands
    assert not (workdir / "cases" / "A_no_true_pairs_sparse_wrong_graph" / "mock_observed_gw_pe.h5").exists()



def test_preflight_status_accounts_for_off_controls():
    assert preflight_status(0, {"ok": True}, 0, {"ok": True}, True) == "passed_preflight"
    assert preflight_status(0, {"ok": True}, 1, {"ok": False}, True) == "failed_off_preflight"
    assert preflight_status(1, {"ok": False}, 0, {"ok": True}, True) == "failed_preflight"


def test_combined_inference_status_accounts_for_off_controls():
    assert combined_inference_status("passed", "passed", True) == "passed"
    assert combined_inference_status("passed", "failed_inference", True) == "failed_off_inference"
    assert combined_inference_status("failed_inference", "passed", True) == "failed_j2_inference"
    assert combined_inference_status("failed_inference", "failed_inference", True) == "failed_both"
    assert combined_inference_status("passed", "failed_inference", False) == "passed"



def test_classify_run_outputs_distinguishes_nonfinite_and_missing_logz():
    assert classify_run_outputs({"logZ": "-inf"}, {"logL_total": -1.0}, diagnostics_only=False)["evidence_status"] == "nonfinite_logZ"
    assert classify_run_outputs({}, {"logL_total": -1.0}, diagnostics_only=False)["evidence_status"] == "missing_logZ"


def test_classify_run_outputs_marks_diagnostics_only_evidence_not_meaningful():
    statuses = classify_run_outputs({"logZ": -12.0}, {"logL_total": -1.0}, diagnostics_only=True)
    assert statuses["evidence_status"] == "diagnostics_only_not_meaningful"
    assert statuses["midpoint_status"] == "finite_midpoint"


def test_classify_run_outputs_marks_nonfinite_midpoint():
    statuses = classify_run_outputs({"logZ": -12.0}, {"logL_total": "-inf", "Neff_singleton": 25.36}, diagnostics_only=True)
    assert extract_midpoint_loglike({"logL_total": "-inf"}) is None
    assert statuses["midpoint_status"] == "nonfinite_midpoint"


def test_delta_logz_requires_both_finite_logz():
    assert evidence_delta(extract_logz({"logZ": 1.0}), extract_logz({"logZ": 0.5})) == 0.5
    assert evidence_delta(extract_logz({"logZ": 1.0}), extract_logz({"logZ": "-inf"})) is None
    assert evidence_delta(extract_logz({"logZ": "nan"}), extract_logz({"logZ": 0.5})) is None

def test_extract_logz_from_fake_attrs():
    assert extract_logz({"logZ": "12.5"}) == 12.5
    assert extract_logz({"logz": -3}) == -3.0
    assert extract_logz({"logZ": "nan"}) is None
    assert extract_logz({}) is None


def test_delta_logz_arithmetic_for_fake_attrs():
    j2 = extract_logz({"logZ": 10.0})
    off = extract_logz({"logZ": 7.5})
    assert evidence_delta(j2, off) == 2.5
    assert evidence_delta(j2, extract_logz({})) is None
    assert evidence_delta(j2, off, diagnostics_only=True) is None


def test_diagnostics_only_plan_records_evidence_warning_and_off_max_samples_zero(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "study"
    cmd = [sys.executable, "scripts/mock_lensing/run_simulated_lensing_study.py", "--workdir", str(workdir), "--profile", "tiny", "--diagnostics_only", "true", "--dry_run", "true"]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    plan = json.loads((workdir / "validation_plan.json").read_text())
    case = plan["cases"]["A_no_true_pairs_sparse_wrong_graph"]
    assert case["inference"][case["inference"].index("--max_samples") + 1] == "0"
    assert case["off"][case["off"].index("--max_samples") + 1] == "0"
    b_cmd = plan["cases"]["B_true_pairs_clean_graph"]["inference"]
    assert b_cmd[b_cmd.index("--fixed_parameter_values") + 1] == '{"tau_n": 3.0}'
    assert b_cmd[b_cmd.index("--lens_prior_overrides") + 1] == '{"log10_tau_A": [-5.0, -2.5]}'
    assert plan["diagnostics_only"] is True

from scripts.mock_lensing.run_simulated_lensing_study import audit_candidate_graph, _candidate_audit_csv_row, _write_csv


def _write_audit_inputs(tmp_path, *, pairs, events):
    cand = {"format_version": "candidate-pairs-1.0", "n_events": len(events), "pairs": pairs}
    cat = {"n_events": len(events), "events": events}
    cp = tmp_path / "candidate_pairs.json"
    op = tmp_path / "observed_catalog.json"
    cp.write_text(json.dumps(cand))
    op.write_text(json.dumps(cat))
    return cp, op


def _event(source=None, image=None, lensed=False):
    return {"truth_source_id": source, "truth_image_index": image, "truth_is_lensed_image": lensed}


def test_candidate_graph_audit_detects_true_edge_survival(tmp_path):
    cp, op = _write_audit_inputs(
        tmp_path,
        pairs=[{"i": 0, "j": 1, "log_prior_odds": 2.0, "marks": {"log_mass_distance_score": 1.0}}],
        events=[_event("s", 0, True), _event("s", 1, True)],
    )
    audit = audit_candidate_graph(cp, op)
    assert audit["n_true_edges_in_catalog"] == 1
    assert audit["n_true_edges_in_candidate_graph"] == 1
    assert audit["true_edge_survival_fraction"] == 1.0
    assert audit["n_false_edges"] == 0


def test_candidate_graph_audit_reports_missing_true_edge(tmp_path):
    cp, op = _write_audit_inputs(
        tmp_path,
        pairs=[{"i": 0, "j": 2, "log_prior_odds": -1.0, "marks": {}}],
        events=[_event("s", 0, True), _event("s", 1, True), _event("x", 0, False)],
    )
    audit = audit_candidate_graph(cp, op)
    assert audit["n_true_edges_in_candidate_graph"] == 0
    assert audit["true_edge_survival_fraction"] == 0.0
    assert audit["missing_true_edges"] == [[0, 1]]
    assert audit["n_false_edges"] == 1


def test_candidate_graph_audit_works_with_no_true_pairs(tmp_path):
    cp, op = _write_audit_inputs(
        tmp_path,
        pairs=[{"i": 0, "j": 1, "log_prior_odds": -1.0, "marks": {"delta_t_obs": 2.0, "sigma_delta_t": 1.0}}],
        events=[_event("a", 0, False), _event("b", 0, False)],
    )
    audit = audit_candidate_graph(cp, op)
    assert audit["n_true_edges_in_catalog"] == 0
    assert audit["true_edge_survival_fraction"] is None
    assert audit["n_false_edges"] == 1
    assert "delta_t_obs" in audit["available_mark_keys"]


def test_candidate_graph_audit_does_not_require_candidate_labels(tmp_path):
    cp, op = _write_audit_inputs(
        tmp_path,
        pairs=[{"i": 0, "j": 1, "log_prior_odds": 0.5, "marks": {"log_sky_overlap": -3.0, "log_mass_distance_score": -0.2}}],
        events=[_event("s", 0, True), _event("s", 1, True)],
    )
    raw = json.loads(cp.read_text())
    assert "label" not in raw["pairs"][0]
    audit = audit_candidate_graph(cp, op)
    assert audit["n_true_edges_in_candidate_graph"] == 1
    assert audit["mark_summary_by_key"]["log_sky_overlap"]["median"] == -3.0


def test_candidate_graph_audit_csv_writer(tmp_path):
    row = _candidate_audit_csv_row("case", {"n_events": 2, "n_candidate_edges": 1, "n_true_edges_in_catalog": 1, "n_true_edges_in_candidate_graph": 1, "true_edge_survival_fraction": 1.0, "n_false_edges": 0, "n_components": 1, "component_sizes": [2], "component_edge_counts": [1], "component_partition_counts": [2], "available_mark_keys": ["log_mass_distance_score"]})
    out = tmp_path / "candidate_graph_audit.csv"
    fields = ["case","n_events","n_candidate_edges","n_true_edges","n_true_edges_kept","true_edge_survival_fraction","n_false_edges","n_components","max_component_events","max_component_edges","max_component_partitions","available_mark_keys"]
    _write_csv(out, [row], fields)
    text = out.read_text()
    assert "case,n_events,n_candidate_edges" in text
    assert "log_mass_distance_score" in text
