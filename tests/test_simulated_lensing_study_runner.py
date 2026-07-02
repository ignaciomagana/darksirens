import json
import subprocess
import sys
from pathlib import Path

from scripts.mock_lensing.run_simulated_lensing_study import evidence_delta, extract_logz, recovery_metrics, posterior_probability_items, validate_known_inference_flags, write_preflight_summary


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
    summary = {"profile": "tiny", "cases": {"A": {"status": "passed_preflight", "candidate_graph_summary": {"n_events": 2, "n_candidate_edges": 1, "n_components": 1, "component_n_partitions": [2]}, "warnings": ["w"], "errors": []}}}
    write_preflight_summary(tmp_path, summary)
    data = json.loads((tmp_path / "preflight_summary.json").read_text())
    assert data["cases"]["A"]["status"] == "passed_preflight"
    md = (tmp_path / "preflight_summary.md").read_text()
    assert "| case | status | n_events | n_edges | n_components | n_partitions | warnings | errors |" in md
    assert "| A | passed_preflight | 2 | 1 | 1 | 2 | 1 | 0 |" in md


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
    assert plan["diagnostics_only"] is True
