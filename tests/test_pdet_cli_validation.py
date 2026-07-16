"""CLI surface for the P_det emulator selection path (--pdet_flow_path and
the validation guard block in darksirens/cli/inference.py).  Mirrors
test_flow_cli_validation.py's subprocess style: the guards fire before any
data file is opened, so dummy paths must fail from the guard, never from IO.
"""
import subprocess
import sys

import numpy as np
import pytest


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def dummy_npz(tmp_path_factory):
    """An existing file for --pdet_flow_path so only later guards fire."""
    path = tmp_path_factory.mktemp("pdet_cli") / "pdet.npz"
    np.savez(path, arr_0=np.zeros(1))
    return str(path)


BASE = [
    "--gw_path", "/nonexistent/gw.h5",
    "--sampler", "tinyns",
]


def test_help_mentions_pdet_flags():
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in ("--pdet_flow_path", "--pdet_nsamp", "--pdet_seed",
                 "--pdet_cosmology", "--pdet_chieff_amax"):
        assert flag in result.stdout


def test_neither_selection_input_exits_nonzero():
    result = _run(BASE)
    assert result.returncode != 0
    assert "Exactly one of --gwselection_path" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


def test_both_selection_inputs_exit_nonzero(dummy_npz):
    result = _run(BASE + [
        "--gwselection_path", "/nonexistent/sel.h5",
        "--pdet_flow_path", dummy_npz,
    ])
    assert result.returncode != 0
    assert "Exactly one of --gwselection_path" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


def test_pdet_missing_file_exits_nonzero():
    result = _run(BASE + ["--pdet_flow_path", "/nonexistent/pdet.npz"])
    assert result.returncode != 0
    assert "not a file" in result.stdout


def test_pdet_nonpositive_nsamp_exits_nonzero(dummy_npz):
    result = _run(BASE + [
        "--pdet_flow_path", dummy_npz,
        "--pdet_nsamp", "0",
    ])
    assert result.returncode != 0
    assert "--pdet_nsamp must be positive" in result.stdout


def test_pdet_bad_cosmology_exits_nonzero(dummy_npz):
    result = _run(BASE + [
        "--pdet_flow_path", dummy_npz,
        "--pdet_cosmology", "not-a-pair",
    ])
    assert result.returncode != 0
    assert "--pdet_cosmology must be 'H0,Om0'" in result.stdout


def test_pdet_bad_amax_exits_nonzero(dummy_npz):
    result = _run(BASE + [
        "--pdet_flow_path", dummy_npz,
        "--pdet_chieff_amax", "1.5",
    ])
    assert result.returncode != 0
    assert "--pdet_chieff_amax must be in (0, 1]" in result.stdout


def test_pdet_with_flow_events_passes_guards(dummy_npz, tmp_path):
    """The emulator must be orthogonal to the event source: with
    --gw_flows_path it must sail past both exactly-one guards (and die
    later at data loading, not in validation)."""
    result = _run([
        "--gw_flows_path", str(tmp_path),
        "--pdet_flow_path", dummy_npz,
        "--sampler", "tinyns",
        "--universe_model", "spectral_sirens",
    ])
    assert result.returncode != 0
    assert "Exactly one of --gw_path" not in result.stdout
    assert "Exactly one of --gwselection_path" not in result.stdout


def test_gwselection_only_passes_pdet_guards():
    """The legacy injection-file path must sail past the pdet guards."""
    result = _run(BASE + ["--gwselection_path", "/nonexistent/sel.h5"])
    assert result.returncode != 0
    assert "Exactly one of --gwselection_path" not in result.stdout
