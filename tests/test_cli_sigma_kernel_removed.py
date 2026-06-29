import subprocess
import sys

from darksirens.inference.prior import build_parameter_space
from darksirens.cli.inference import _completion_validation_survey_values


def test_cli_help_does_not_list_sigma_kernel():
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sigma_kernel" not in result.stdout
    assert "--prior_overrides" in result.stdout
    assert "--fixed_parameter_values" in result.stdout


def test_cli_rejects_sigma_kernel_argument():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "darksirens.cli.inference",
            "--gw_path",
            "gw.hdf5",
            "--gwselection_path",
            "selection.hdf5",
            "--sampler",
            "tinyns",
            "--sigma_kernel",
            "0.01",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --sigma_kernel" in result.stderr


def test_sigma_kde_remains_available_for_prior_and_fixed_parameter_values():
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={"sigma_kde": [0.01, 0.03]},
        fixed_parameter_values={},
    )

    sigma_idx = labels.index("sigma_kde")
    assert lower[sigma_idx] == 0.01
    assert upper[sigma_idx] == 0.03

    fixed = _completion_validation_survey_values(
        prior_overrides={"sigma_kde": [0.01, 0.03]},
        fixed_parameter_values={"sigma_kde": 0.02},
    )
    assert fixed["sigma_kde"] == 0.02
