"""Tests for scripts/mock_lensing/summarize_simulated_lensing_runs.py."""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from scripts.mock_lensing.summarize_simulated_lensing_runs import main


def _write_study(workdir: Path, seed: int) -> None:
    workdir.mkdir(parents=True)
    summary = {
        "profile": "small",
        "diagnostics_only": False,
        "resolved_config": {
            "study": {"seed": seed, "profile": "small"},
            "inference": {"sampler": "dynesty", "nlive": 256, "dlogz": 0.25,
                          "fix_population": False, "singleton_lensing": "sl_mixture"},
            "mock": {"pop_model": "powerlaw+peak@md"},
        },
        "cases": {
            "B_true_pairs_clean_graph": {
                "status": "passed",
                "candidate_graph_summary": {"n_events": 24},
                "candidate_graph_audit": {"n_events": 24, "n_candidate_edges": 4,
                                          "n_components": 20},
                "j2": {"status": "passed", "logZ": -100.0, "logZerr": 0.2,
                       "run_dir": "x"},
                "off": {"status": "passed", "logZ": -104.0, "logZerr": 0.2,
                        "run_dir": "y"},
                "singles_only": {"status": "passed", "logZ": -102.0,
                                 "logZerr": 0.3, "run_dir": "z"},
                "delta_logZ_j2_minus_off": 4.0,
                "delta_logZerr": 0.28,
            }
        },
    }
    (workdir / "validation_summary.json").write_text(json.dumps(summary))
    with (workdir / "truth_recovery_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "injected_n_pairs", "expected_n_pairs",
                                          "map_n_pairs", "true_edge_posterior_probability_mean",
                                          "false_edge_posterior_probability_max",
                                          "false_edge_posterior_probability_sum",
                                          "map_partition_exact_truth_match"])
        w.writeheader()
        w.writerow({"case": "B_true_pairs_clean_graph", "injected_n_pairs": 4,
                    "expected_n_pairs": 3.5, "map_n_pairs": 4,
                    "true_edge_posterior_probability_mean": 0.8,
                    "false_edge_posterior_probability_max": 0.01,
                    "false_edge_posterior_probability_sum": 0.02,
                    "map_partition_exact_truth_match": True})
    with (workdir / "hyperparameter_recovery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "run", "label", "truth", "mean",
                                          "median", "q05", "q95", "truth_in_90ci"])
        w.writeheader()
        in_ci = seed % 2 == 0
        w.writerow({"case": "B_true_pairs_clean_graph", "run": "j2",
                    "label": r"$z_{\rm peak}$", "truth": 1.9, "mean": 2.0,
                    "median": 2.0, "q05": 1.5, "q95": 2.5,
                    "truth_in_90ci": in_ci})


def test_summarizer_combines_seeds_and_coverage(tmp_path):
    for seed in (3001, 3002):
        _write_study(tmp_path / f"study_{seed}", seed)
    outdir = tmp_path / "combined"
    rc = main(["--runs", str(tmp_path / "study_*"), "--outdir", str(outdir)])
    assert rc == 0

    run_rows = list(csv.DictReader((outdir / "combined_run_table.csv").open()))
    assert len(run_rows) == 6  # 2 seeds x (j2, off, singles_only)
    assert {r["run"] for r in run_rows} == {"j2", "off", "singles_only"}
    assert run_rows[0]["n_candidate_edges"] == "4"
    assert run_rows[0]["pop_model"] == "powerlaw+peak@md"

    evid = list(csv.DictReader((outdir / "combined_evidence.csv").open()))
    assert len(evid) == 2
    assert evid[0]["logZ_singles"] == "-102.0"

    cov = list(csv.DictReader((outdir / "hyperparameter_coverage.csv").open()))
    assert len(cov) == 1
    # seed 3001 (odd) out of CI, seed 3002 (even) in CI -> coverage 0.5
    assert cov[0]["run"] == "j2"
    assert float(cov[0]["coverage_90"]) == 0.5
    assert cov[0]["n"] == "2"

    meta = json.loads((outdir / "summary_metadata.json").read_text())
    assert meta["n_hyper_rows"] == 2


def test_summarizer_errors_on_no_matches(tmp_path):
    rc = main(["--runs", str(tmp_path / "nothing_*"), "--outdir", str(tmp_path / "out")])
    assert rc == 1
