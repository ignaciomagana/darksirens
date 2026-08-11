"""The TinyNS tuning harness must not recommend a fast, biased configuration.

The sweep varies exactly the knobs that decide whether the live-point
replacement is a valid Markov move (walks, step_scale, min_accepts).
Under-mixing there makes the run FASTER while biasing logZ and the posterior, so
ranking on niter/sec behind only the replacement-failure flags selects for the
bias.  The insertion-rank statistic and cross-config logZ agreement are the
diagnostics that catch it, and they must gate the recommendation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "benchmark_tinyns_darksirens_short_budget.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("_tinyns_bench_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(name, rate, z=0.2, ratio=1.0, logZ=-100.0, logZerr=0.2):
    return {"config_name": name, "status": "completed", "niter_per_sec": rate,
            "replacement_failures": 0, "replacement_rescue_used": "False",
            "insertion_rank_mean_z": z, "insertion_rank_std_ratio": ratio,
            "logZ": logZ, "logZerr": logZerr}


def test_undermixed_config_fails_the_correctness_gate(bench):
    ok, reason = bench._insertion_rank_ok(_row("cheap", 50.0, z=-4.5))
    assert not ok and "insertion_rank_mean_z" in reason
    ok, reason = bench._insertion_rank_ok(_row("cheap", 50.0, ratio=0.5))
    assert not ok and "std_ratio" in reason
    assert bench._insertion_rank_ok(_row("ok", 10.0))[0]
    # Missing diagnostics are not silently failed.
    assert bench._insertion_rank_ok({"config_name": "x"})[0]


def test_fastest_but_undermixed_is_not_the_best_candidate(bench, capsys):
    rows = [_row("recommended", 10.0), _row("cheap", 99.0, z=-6.0)]
    bench.print_ranking(rows)
    out = capsys.readouterr().out
    assert "Best healthy candidate by niter/sec: recommended" in out
    assert "cheap FAILS the sampler-correctness gate" in out
    # ... and the ranking itself puts correctness ahead of speed.
    ranked = sorted(rows, key=lambda r: bench._score(r, bench.logz_outliers(rows)),
                    reverse=True)
    assert ranked[0]["config_name"] == "recommended"


def test_logz_disagreement_is_reported_and_excluded(bench, capsys):
    rows = [_row("recommended", 10.0, logZ=-100.0, logZerr=0.1),
            _row("fast16", 90.0, logZ=-104.0, logZerr=0.1)]
    outliers = bench.logz_outliers(rows)
    assert "fast16" in outliers
    bench.print_ranking(rows)
    out = capsys.readouterr().out
    assert "Best healthy candidate by niter/sec: recommended" in out
    assert "disagrees with recommended" in out


def test_all_gates_failing_refuses_to_recommend(bench, capsys):
    bench.print_ranking([_row("cheap", 99.0, z=-6.0), _row("fast16", 80.0, ratio=0.2)])
    out = capsys.readouterr().out
    assert "No config passed both" in out
    assert "Best healthy candidate" not in out
