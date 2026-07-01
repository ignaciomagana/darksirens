import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fake_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    cases = ["D_true_pairs_bad_pair_tag", "E_true_pairs_no_sky_marks", "F_true_pairs_no_time_marks", "G_true_pairs_full_marks"]
    (study / "run_manifest.json").write_text(json.dumps({"cases": {case: {} for case in cases}}) + "\n")
    (study / "validation_summary.json").write_text(json.dumps({
        "diagnostics_only": False,
        "cases": {
            "G_true_pairs_full_marks": {"results_attrs": {"logZ": -10.0}, "lens_rate_posterior_summary": {"median": -3.5, "q05": -4.0, "q95": -3.0}},
            "D_true_pairs_bad_pair_tag": {"results_attrs": {"logZ": -11.5}},
        },
    }) + "\n")
    _write_csv(study / "posterior_pair_probabilities.csv", [
        {"case": "G_true_pairs_full_marks", "i": 0, "j": 1, "posterior_probability": 0.8, "is_true_edge": True},
        {"case": "G_true_pairs_full_marks", "i": 0, "j": 2, "posterior_probability": 0.1, "is_true_edge": False},
        {"case": "D_true_pairs_bad_pair_tag", "i": 0, "j": 1, "posterior_probability": 0.3, "is_true_edge": True},
    ], ["case", "i", "j", "posterior_probability", "is_true_edge"])
    _write_csv(study / "truth_recovery_summary.csv", [
        {"case": "D_true_pairs_bad_pair_tag", "injected_n_pairs": 1, "expected_n_pairs": 0.3, "map_n_pairs": 0, "true_edge_posterior_probability_mean": 0.3, "false_edge_posterior_probability_max": 0.0, "false_edge_posterior_probability_sum": 0.0, "map_partition_exact_truth_match": False},
        {"case": "E_true_pairs_no_sky_marks", "injected_n_pairs": 1, "expected_n_pairs": 0.4, "map_n_pairs": 0, "true_edge_posterior_probability_mean": 0.4, "false_edge_posterior_probability_max": 0.2, "false_edge_posterior_probability_sum": 0.2, "map_partition_exact_truth_match": False},
        {"case": "F_true_pairs_no_time_marks", "injected_n_pairs": 1, "expected_n_pairs": 0.5, "map_n_pairs": 1, "true_edge_posterior_probability_mean": 0.5, "false_edge_posterior_probability_max": 0.3, "false_edge_posterior_probability_sum": 0.3, "map_partition_exact_truth_match": False},
        {"case": "G_true_pairs_full_marks", "injected_n_pairs": 1, "expected_n_pairs": 0.9, "map_n_pairs": 1, "true_edge_posterior_probability_mean": 0.8, "false_edge_posterior_probability_max": 0.1, "false_edge_posterior_probability_sum": 0.1, "map_partition_exact_truth_match": True},
    ], ["case", "injected_n_pairs", "expected_n_pairs", "map_n_pairs", "true_edge_posterior_probability_mean", "false_edge_posterior_probability_max", "false_edge_posterior_probability_sum", "map_partition_exact_truth_match"])
    _write_csv(study / "bias_summary.csv", [{"case": "G_true_pairs_full_marks", "p_tag_model": "snr_time_sky", "pair_tag_perturb_logit": 0, "delta_expected_n_pairs_minus_injected": -0.1}], ["case", "p_tag_model", "pair_tag_perturb_logit", "delta_expected_n_pairs_minus_injected"])
    _write_csv(study / "partition_component_summary.csv", [{"case": case, "n_events": 4, "n_candidate_edges": 2, "expected_n_pairs": 1, "map_n_pairs": 1} for case in cases], ["case", "n_events", "n_candidate_edges", "expected_n_pairs", "map_n_pairs"])
    for case in cases:
        cdir = study / "cases" / case
        cdir.mkdir(parents=True)
        (cdir / "candidate_pairs.json").write_text(json.dumps({"pairs": [{"i": 0, "j": 1, "log_prior_odds": 2.0}, {"i": 0, "j": 2, "log_prior_odds": -1.0}]}) + "\n")
    return study


def test_simulated_lensing_study_plot_script_creates_figures(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    study = _fake_study(tmp_path)
    outdir = tmp_path / "plots"
    cmd = [sys.executable, "scripts/mock_lensing/plot_simulated_lensing_study.py", "--study_dir", str(study), "--outdir", str(outdir), "--format", "png"]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    manifest = json.loads((outdir / "plot_manifest.json").read_text())
    produced = {item["name"] for item in manifest["produced"]}
    assert "fig_pair_probabilities" in produced
    assert "fig_candidate_graph_summary" in produced
    assert "fig_false_positive_summary" in produced
    for item in manifest["produced"]:
        assert Path(item["path"]).exists()


def test_simulated_lensing_study_plot_script_missing_optional_inputs_do_not_crash(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    study = tmp_path / "study"
    study.mkdir()
    (study / "validation_summary.json").write_text(json.dumps({"diagnostics_only": True, "cases": {}}) + "\n")
    outdir = tmp_path / "plots"
    cmd = [sys.executable, "scripts/mock_lensing/plot_simulated_lensing_study.py", "--study_dir", str(study), "--outdir", str(outdir)]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    manifest = json.loads((outdir / "plot_manifest.json").read_text())
    assert manifest["produced"] == []
    assert manifest["skipped"]
    assert any("missing artifact" in warning for warning in manifest["warnings"])
