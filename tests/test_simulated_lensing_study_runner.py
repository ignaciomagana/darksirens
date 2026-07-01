import json
import subprocess
import sys
from pathlib import Path

from scripts.mock_lensing.run_simulated_lensing_study import recovery_metrics, posterior_probability_items


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
