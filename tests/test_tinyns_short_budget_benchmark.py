import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_tinyns_darksirens_short_budget.py"
spec = importlib.util.spec_from_file_location("tinyns_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def _opts(**kwargs):
    base = dict(
        executable="darksirens_inference",
        gw_path="gw.h5",
        gwselection_path="sel.h5",
        survey_path=None,
        universe_model="spectral_sirens",
        pop_model="powerlaw+peak",
        nlive=400,
        dlogz=0.5,
        max_samples=2000,
        seed=21,
        extra_arg=[],
        extra_args_json=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_default_sweep_constructs_tinyns_only_commands(tmp_path):
    names = [entry["name"] for entry in bench.DEFAULT_SWEEP]
    assert names == ["recommended", "cheap", "fast16", "fast32", "fast16_B128"]
    cmd = bench.build_command(_opts(), bench.DEFAULT_SWEEP[2], tmp_path / "run")
    assert "--sampler" in cmd and cmd[cmd.index("--sampler") + 1] == "tinyns"
    assert "--tinyns_progress_interval" in cmd
    assert "--max_samples" in cmd
    assert "--tinyns_replacement_chains" in cmd
    assert "--tinyns_walks" in cmd
    assert not (set(cmd) & bench.NO_NORMALIZATION_GRID_FLAGS)


def test_survey_path_optional_for_spectral_and_included_when_provided(tmp_path):
    spectral = bench.build_command(_opts(universe_model="spectral_sirens", survey_path=None), bench.DEFAULT_SWEEP[0], tmp_path / "spectral")
    assert "--survey_path" not in spectral
    dark = bench.build_command(_opts(universe_model="dark_sirens", survey_path="cat.h5"), bench.DEFAULT_SWEEP[0], tmp_path / "dark")
    assert dark[dark.index("--survey_path") + 1] == "cat.h5"


def test_parse_diagnostics_prefers_sidecar(tmp_path):
    run = tmp_path / "run" / "actual"
    run.mkdir(parents=True)
    payload = {
        "tinyns_resolved_config": {"sample": "rwalk"},
        "tinyns_runtime_diagnostics": {
            "niter": 12,
            "ncall": 345,
            "niter_per_sec": 1.2,
            "replacement_mean_batches": 2.5,
            "replacement_failures": 0,
            "replacement_rescue_used": False,
        },
    }
    (run / "tinyns_diagnostics.json").write_text(json.dumps(payload))
    parsed = bench.parse_diagnostics(tmp_path / "run")
    assert parsed["niter"] == 12
    assert parsed["ncall"] == 345
    assert parsed["niter_per_sec"] == 1.2
    assert parsed["replacement_mean_batches"] == 2.5
    assert parsed["replacement_failures"] == 0
    assert parsed["replacement_rescue_used"] is False


def test_write_summary_creates_csv_and_json(tmp_path):
    rows = [{"config_name": "cheap", "status": "completed", "niter": 10}]
    bench.write_summary(rows, tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert "cheap" in (tmp_path / "summary.csv").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())[0]["config_name"] == "cheap"
