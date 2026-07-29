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


def test_lensing_inference_config_overrides_parse_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    cfg = sc.resolve_config(path, [
        'inference.lens_prior_overrides={"log10_tau_A": [-4.0, -1.5]}',
        'inference.fixed_parameter_values={"tau_n": 3.5}',
        "inference.fix_lens_rate=false",
    ])
    assert cfg["inference"]["lens_prior_overrides"] == {"log10_tau_A": [-4.0, -1.5]}
    assert cfg["inference"]["fixed_parameter_values"] == {"tau_n": 3.5}
    assert cfg["inference"]["fix_lens_rate"] is False


def test_lensing_inference_config_validation_rejects_bad_types(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"inference": {"lens_prior_overrides": [], "fixed_parameter_values": [], "fix_lens_rate": "false"}}))
    with pytest.raises(ValueError, match="fix_lens_rate.*invalid type.*fixed_parameter_values.*invalid type.*lens_prior_overrides.*invalid type"):
        sc.resolve_config(path)


def test_invalid_config_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"mock": {"n_universe": 0}, "extra": {}}))
    with pytest.raises(ValueError, match="n_universe.*>= 1"):
        sc.resolve_config(path)


def test_unknown_profile_rejected_with_choices(tmp_path):
    """A typo'd profile used to resolve to the TINY preset while the typo was
    stamped into the run metadata: the run looked like 'paper' and was 1/30th
    of its size."""
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"study": {"profile": "papre"}}))
    with pytest.raises(ValueError, match=r"unknown study.profile 'papre'.*paper.*small.*tiny"):
        sc.resolve_config(path)
    # Same gate for the caller-supplied profile argument...
    with pytest.raises(ValueError, match="unknown study.profile 'papre'"):
        sc.resolve_config(None, profile="papre")
    # ...and for an override, which cannot retro-select the base preset.
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="study.profile must be one of"):
        sc.resolve_config(empty, ["study.profile=papre"])
    # An explicit null profile still means "use the default".
    nulled = tmp_path / "nulled.json"
    nulled.write_text(json.dumps({"study": {"profile": None}}))
    assert sc.resolve_config(nulled, profile="small")["mock"]["n_universe"] == 12000


@pytest.mark.parametrize("override, match", [
    ("mock.tau_A=true", "tau_A has invalid type bool"),
    ("mock.t_obs_days=false", "t_obs_days has invalid type bool"),
    ("mock.tau_A=NaN", "tau_A must be a finite number"),
    ("mock.tau_n=Infinity", "tau_n must be a finite number"),
    ("inference.dlogz=-Infinity", "dlogz must be a finite number"),
])
def test_numeric_overrides_reject_bool_and_nonfinite(tmp_path, override, match):
    """``True`` is an int in Python and NaN loses every ``<`` comparison, so
    both slipped through the type gate and the range gate untouched."""
    path = tmp_path / "config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match=match):
        sc.resolve_config(path, [override])


def test_valid_numeric_overrides_still_accepted(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    cfg = sc.resolve_config(path, ["mock.tau_A=0.0005", "mock.tau_n=3", "inference.dlogz=1.5"])
    assert cfg["mock"]["tau_A"] == 0.0005
    assert cfg["mock"]["tau_n"] == 3
    assert cfg["inference"]["dlogz"] == 1.5


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
