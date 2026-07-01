import json
import subprocess
import sys
from pathlib import Path

import pytest

from darksirens.lensing import simulation_config as sc


def test_json_config_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"study": {"profile": "tiny"}, "mock": {"n_universe": 1234}}))
    cfg = sc.resolve_config(path)
    assert cfg["mock"]["n_universe"] == 1234
    assert cfg["study"]["profile"] == "tiny"


def test_yaml_config_loads_if_available(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text("study:\n  profile: tiny\nmock:\n  n_singletons: 3\n")
    cfg = sc.resolve_config(path)
    assert cfg["mock"]["n_singletons"] == 3


def test_override_works(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    cfg = sc.resolve_config(path, ["mock.n_singletons=7", "inference.diagnostics_only=true"])
    assert cfg["mock"]["n_singletons"] == 7
    assert cfg["inference"]["diagnostics_only"] is True


def test_invalid_config_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"mock": {"n_universe": 0}, "extra": {}}))
    with pytest.raises(ValueError, match="n_universe.*>= 1"):
        sc.resolve_config(path)


def test_resolved_config_written_in_dry_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "study"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"study": {"profile": "tiny", "cases": ["B_true_pairs_clean_graph"]}}))
    cmd = [sys.executable, "scripts/mock_lensing/run_simulated_lensing_study.py", "--config", str(config), "--workdir", str(workdir), "--dry_run", "true", "--override", "mock.n_singletons=1"]
    subprocess.run(cmd, cwd=repo, check=True, timeout=60)
    resolved = workdir / "resolved_config.yaml"
    if not resolved.exists():
        resolved = workdir / "resolved_config.json"
    assert resolved.exists()
    manifest = json.loads((workdir / "run_manifest.json").read_text())
    assert manifest["resolved_config"]["mock"]["n_singletons"] == 1
    assert list(manifest["cases"]) == ["B_true_pairs_clean_graph"]
