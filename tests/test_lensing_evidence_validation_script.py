import json
import subprocess
import sys
from pathlib import Path


def test_lensing_evidence_validation_help():
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "scripts/mock_lensing/run_lensing_evidence_validation.py", "--help"]
    completed = subprocess.run(cmd, cwd=repo, check=True, text=True, capture_output=True)
    assert "tiny_evidence" in completed.stdout
    assert "--dry_run" in completed.stdout


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
