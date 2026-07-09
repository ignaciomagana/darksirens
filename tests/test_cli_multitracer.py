"""CLI surface for the K-catalog multitracer mixture (--survey_path nargs='+'
and the post-parse guard block in darksirens/cli/inference.py).  Mirrors
test_cli_sigma_kernel_removed.py's subprocess style.

The guard block runs immediately after argparse, before any file is opened
(load_all_data is only reached much later, in the "Loading data" section), so
these tests use deliberately nonexistent dummy paths -- the process must exit
nonzero from the guard, never from a file-not-found error."""
import subprocess
import sys


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True,
        text=True,
    )


def test_cli_help_mentions_multi_catalog_survey_path_usage():
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # argparse nargs="+" usage marker for --survey_path.
    assert "--survey_path PATH [PATH ...]" in result.stdout
    # The help text documents the K-catalog mixture semantics.
    lowered = result.stdout.lower()
    assert "k-catalog" in lowered or "multiple paths" in lowered
    assert "fcat_2" in result.stdout


def test_two_survey_paths_with_non_dark_sirens_universe_model_exits_nonzero():
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "dark_sirens" in result.stdout
    # Must fail before any file access is attempted.
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_two_survey_paths_with_one_lss_completion_entry_exits_nonzero():
    """K=2 requires --lss_completion to have 0 or exactly n_catalogs entries;
    a single entry (ambiguous which catalog it belongs to) must be rejected
    explicitly rather than silently broadcast."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--lss_completion", "/nonexistent/lssA.h5",
        "--universe_model", "dark_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "lss_completion" in result.stdout
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_single_survey_path_is_unaffected_by_multi_catalog_guards():
    """A single --survey_path (the legacy single-catalog invocation) must NOT
    trip the K>=2 guards -- it should fail later, for the ordinary reason
    (missing GW/selection files), not on universe_model/lss_completion."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "Multiple --survey_path catalogs" not in result.stdout
