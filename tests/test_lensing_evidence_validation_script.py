import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.mock_lensing.run_lensing_evidence_validation import (
    _format_status_object,
    _posterior_pair_probability_check,
    _write_markdown,
)


def test_lensing_evidence_validation_help():
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "scripts/mock_lensing/run_lensing_evidence_validation.py", "--help"]
    completed = subprocess.run(cmd, cwd=repo, check=True, text=True, capture_output=True)
    assert "tiny_evidence" in completed.stdout
    assert "--dry_run" in completed.stdout
    assert "--diagnostics_only" in completed.stdout
    assert "--skip_preflight" in completed.stdout


def test_lensing_evidence_validation_dry_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "ds_lens_evidence"
    cmd = [
        sys.executable,
        "scripts/mock_lensing/run_lensing_evidence_validation.py",
        "--profile",
        "tiny_evidence",
        "--workdir",
        str(workdir),
        "--dry_run",
        "true",
    ]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    plan_path = workdir / "validation_plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["profile"] == "tiny_evidence"
    assert "j2_fixed_true" in plan["cases"]
    assert "off_true_catalog" in plan["cases"]


@pytest.mark.slow
def test_lensing_evidence_validation_diagnostics_only(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "ds_lens_evidence_diag"
    cmd = [
        sys.executable,
        "scripts/mock_lensing/run_lensing_evidence_validation.py",
        "--profile",
        "tiny_evidence",
        "--workdir",
        str(workdir),
        "--use_unified_observed_catalog",
        "true",
        "--diagnostics_only",
        "true",
    ]
    subprocess.run(cmd, cwd=repo, check=True, timeout=900)  # CPU-only boxes need headroom; still guards against hangs
    summary = json.loads((workdir / "validation_summary.json").read_text())
    assert summary["diagnostics_only"] is True
    assert all(run["preflight_json"] for run in summary["runs"].values())
    assert all(run["status"] == "passed" for run in summary["runs"].values())
    assert (workdir / "validation_summary.md").exists()


def test_posterior_pair_probability_sum_check():
    # The REAL diagnostics format: marginal_diagnostics emits a list of dicts
    # with per-edge p_pair fields (the check previously crashed on it with
    # TypeError — library review, lensing pkg finding 3).
    ok, msg = _posterior_pair_probability_check(
        {
            "posterior_pair_probabilities": [
                {"pair": [0, 1], "p_pair": 0.25},
                {"pair": [2, 3], "p_pair": 0.75},
            ],
            "expected_n_pairs": 1.0,
        }
    )
    assert ok
    assert msg is None

    # Compatibility forms still accepted:
    ok, msg = _posterior_pair_probability_check(
        {"posterior_pair_probabilities": {"0-1": 0.25, "2-3": 0.75}, "expected_n_pairs": 1.0}
    )
    assert ok
    assert msg is None

    ok, msg = _posterior_pair_probability_check(
        {"posterior_pair_probabilities": [0.25, 0.25], "expected_n_pairs": 1.0}
    )
    assert not ok
    assert "sum" in msg

    ok, msg = _posterior_pair_probability_check(
        {"posterior_pair_probabilities": [{"p_pair": 1.2}], "expected_n_pairs": 1.2}
    )
    assert not ok
    assert "[0, 1]" in msg


def test_status_object_formatting():
    row = _format_status_object(
        "j2_fixed_true",
        {
            "status": "passed",
            "logZ": -12.0,
            "logZerr": None,
            "diagnostics": {"logL_total": -10.0, "pair_logL_sum": 3.0, "n_pairs": 2},
            "runtime_s": 1.5,
            "run_dir": "/tmp/run",
        },
    )
    assert row == {
        "case": "j2_fixed_true",
        "status": "passed",
        "logZ": -12.0,
        "logZerr": None,
        "logL_total": -10.0,
        "pair_logL_sum": 3.0,
        "n_pairs": 2,
        "runtime_s": 1.5,
        "run_dir": "/tmp/run",
    }


def test_markdown_summary_generation(tmp_path):
    out = tmp_path / "validation_summary.md"
    _write_markdown(
        out,
        {
            "profile": "tiny_evidence",
            "checks": {"all_cases_passed": True},
            "runs": {
                "case_a": {
                    "status": "passed",
                    "logZ": -1.0,
                    "logZerr": 0.2,
                    "diagnostics": {"logL_total": -2.0, "pair_logL_sum": 0.5, "n_pairs": 1},
                    "runtime_s": 0.25,
                    "run_dir": "/tmp/case_a",
                }
            },
        },
    )
    text = out.read_text()
    assert "| case | status | logZ | logZerr | logL_total |" in text
    assert "| case_a | passed | -1.0 | 0.2 | -2.0 | 0.5 | 1 | 0.2 | /tmp/case_a |" in text
