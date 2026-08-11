"""CLI surface for the flow-surrogate path (--gw_flows_path and the
validation guard block in darksirens/cli/inference.py).  Mirrors
test_cli_multitracer.py's subprocess style: the guards fire before any data
file is opened, so dummy paths must fail from the guard, never from IO.
"""
import subprocess
import sys


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True,
        text=True,
    )


BASE = [
    "--gwselection_path", "/nonexistent/sel.h5",
    "--sampler", "tinyns",
]


def test_help_mentions_flows():
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--gw_flows_path" in result.stdout
    assert "--flows_nsamp" in result.stdout


def test_neither_gw_input_exits_nonzero():
    result = _run(BASE)
    assert result.returncode != 0
    assert "Exactly one of --gw_path" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


def test_both_gw_inputs_exit_nonzero(tmp_path):
    result = _run(BASE + [
        "--gw_path", "/nonexistent/gw.h5",
        "--gw_flows_path", str(tmp_path),
    ])
    assert result.returncode != 0
    assert "Exactly one of --gw_path" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


def test_flows_with_dark_sirens_exits_nonzero(tmp_path):
    result = _run(BASE + [
        "--gw_flows_path", str(tmp_path),
        "--universe_model", "dark_sirens",
        "--survey_path", "/nonexistent/cat.h5",
    ])
    assert result.returncode != 0
    assert "spectral_sirens only" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


def test_flows_nonpositive_nsamp_exits_nonzero(tmp_path):
    result = _run(BASE + [
        "--gw_flows_path", str(tmp_path),
        "--universe_model", "spectral_sirens",
        "--flows_nsamp", "0",
    ])
    assert result.returncode != 0
    assert "--flows_nsamp must be positive" in result.stdout


def test_flows_missing_directory_exits_nonzero():
    result = _run(BASE + [
        "--gw_flows_path", "/nonexistent/flows_dir",
        "--universe_model", "spectral_sirens",
    ])
    assert result.returncode != 0
    assert "not a directory" in result.stdout


def _pdet_flows_args(tmp_path, *extra):
    """--gw_flows_path + --pdet_flow_path (exactly one selection source)."""
    pdet = tmp_path / "pdet_flow.npz"
    pdet.write_bytes(b"not a real checkpoint")
    flows = tmp_path / "flows"
    flows.mkdir()
    return [
        "--gw_flows_path", str(flows),
        "--pdet_flow_path", str(pdet),
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ] + list(extra)


def test_flows_chieff_amax_out_of_range_exits_nonzero(tmp_path):
    result = _run(_pdet_flows_args(tmp_path, "--flows_chieff_amax", "1.5"))
    assert result.returncode != 0
    assert "--flows_chieff_amax must be in (0, 1]" in result.stdout


def test_mismatched_chieff_amax_between_flows_and_pdet_exits_nonzero(tmp_path):
    """The chi_eff PE prior is divided out of the event flows and baked into the
    pseudo-injection p_draw: two different amax leave a chi_eff/q-dependent
    residual in the hierarchical ratio, so the pair must be equal."""
    result = _run(_pdet_flows_args(tmp_path, "--pdet_chieff_amax", "1.0"))
    assert result.returncode != 0
    assert "--flows_chieff_amax 0.99 != --pdet_chieff_amax 1.0" in result.stdout
    assert "same reference convention" in result.stdout.lower()


def test_matching_chieff_amax_passes_the_cross_check(tmp_path):
    """Equal values (the shared default) sail past the guard and fail later."""
    result = _run(_pdet_flows_args(tmp_path, "--pdet_chieff_amax", "0.99"))
    assert result.returncode != 0
    assert "!= --pdet_chieff_amax" not in result.stdout


def test_gw_path_only_passes_flow_guards():
    """A stored-PE run must sail past the flow guards (and die later on IO)."""
    result = _run(BASE + [
        "--gw_path", "/nonexistent/gw.h5",
        "--universe_model", "spectral_sirens",
    ])
    assert result.returncode != 0
    assert "Exactly one of --gw_path" not in result.stdout
