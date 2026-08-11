import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_lensing_validation_tiny_diagnostics(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/mock_lensing/run_lensing_validation.py",
        "--profile",
        "tiny",
        "--workdir",
        str(tmp_path / "ds_lensing_validation"),
    ]
    subprocess.run(cmd, cwd=repo, check=True, timeout=300)


def test_marginalized_result_metadata_is_unambiguous():
    from types import SimpleNamespace

    from darksirens.cli.inference_lensing import _write_result_partition_metadata

    attrs = {}
    opts = SimpleNamespace(
        partition_mode="marginalize_exact",
        cluster_mode="j2",
        wl_backend="lognormal",
        wl_selection="standard",
    )
    inp = {"nEvents": 4, "n_singletons": 4, "n_pairs": 0}
    diagnostics = {
        "n_partitions": 3,
        "expected_n_singletons": 2.5,
        "expected_n_pairs": 0.75,
        "map_partition_index": 2,
        "map_partition": {"n_singletons": 2, "n_pairs": 1},
        "logL_marginalized": -12.5,
        "log_z_partition_prior": 0.4,
    }

    _write_result_partition_metadata(attrs, opts=opts, inp=inp, diagnostics=diagnostics)

    assert attrs["partition_mode"] == "marginalize_exact"
    assert attrs["n_pairs"] == 0
    assert attrs["n_pairs_meaning"] == "reference_partition_n_pairs"
    assert attrs["reference_partition_n_pairs"] == 0
    # Midpoint-derived quantities are now explicitly labelled (library
    # review, lensing CLI finding 6): the attrs carry a prior_midpoint_
    # prefix + an eval-point marker so pre-posterior numbers cannot be
    # traced into a paper as run results.
    assert attrs["partition_diagnostics_eval_point"] == "prior_midpoint"
    assert attrs["prior_midpoint_expected_n_pairs"] == 0.75
    assert attrs["prior_midpoint_map_partition_n_pairs"] == 1
    assert "expected_n_pairs" not in attrs
    assert attrs["n_partitions"] == 3


def test_fallback_diagnostics_point_is_not_filed_as_the_prior_midpoint():
    """The reliability guard often fires at the midpoint on paper-scale joint
    runs, and the fallback evaluates at the registry fiducial (or a seeded prior
    draw) instead. Those numbers must not be archived as prior_midpoint_*
    (review F-006)."""
    from types import SimpleNamespace

    from darksirens.cli.inference_lensing import _write_result_partition_metadata

    attrs = {}
    opts = SimpleNamespace(
        partition_mode="marginalize_exact",
        cluster_mode="j2",
        wl_backend="lognormal",
        wl_selection="standard",
    )
    inp = {"nEvents": 4, "n_singletons": 4, "n_pairs": 0}
    diagnostics = {
        "n_partitions": 3,
        "expected_n_singletons": 2.5,
        "expected_n_pairs": 0.75,
        "map_partition_index": 2,
        "map_partition": {"n_singletons": 2, "n_pairs": 1},
        "logL_marginalized": -12.5,
        "log_z_partition_prior": 0.4,
    }

    _write_result_partition_metadata(
        attrs, opts=opts, inp=inp, diagnostics=diagnostics,
        eval_point_label="registry_fiducial", eval_point=[70.0, 0.3],
    )

    assert attrs["partition_diagnostics_eval_point"] == "registry_fiducial"
    assert attrs["diagnostics_point_expected_n_pairs"] == 0.75
    assert attrs["diagnostics_point_logL_marginalized"] == -12.5
    assert "prior_midpoint_expected_n_pairs" not in attrs
    # the numeric point is archived too, not only in midpoint_diagnostics.json
    assert json.loads(attrs["partition_diagnostics_eval_point_values"]) == [70.0, 0.3]


def test_fixed_result_metadata_preserves_legacy_n_pairs():
    from types import SimpleNamespace

    from darksirens.cli.inference_lensing import _write_result_partition_metadata

    attrs = {}
    opts = SimpleNamespace(
        partition_mode="fixed",
        cluster_mode="j2",
        wl_backend="lognormal",
        wl_selection="standard",
    )
    inp = {"nEvents": 4, "n_singletons": 2, "n_pairs": 1}

    _write_result_partition_metadata(attrs, opts=opts, inp=inp, diagnostics={})

    assert attrs["partition_mode"] == "fixed"
    assert attrs["n_pairs"] == 1
    assert attrs["reference_partition_n_pairs"] == 1
    assert "n_pairs_meaning" not in attrs
    assert "expected_n_pairs" not in attrs
