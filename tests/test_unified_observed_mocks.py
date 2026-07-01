import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import pytest


def _generate_unified_mock(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "mock"
    cmd = [
        sys.executable,
        "scripts/mock_lensing/generate_mock_lensing.py",
        "--outdir",
        str(out),
        "--conditioning",
        "fixed_counts",
        "--n-universe",
        "3000",
        "--n-sing-keep",
        "2",
        "--n-pair-keep",
        "2",
        "--n-unlensed-inj",
        "1000",
        "--n-lensed-inj",
        "1000",
        "--nsamp",
        "16",
        "--write-unified-observed-catalog",
        "true",
    ]
    subprocess.run(cmd, cwd=repo, check=True, timeout=180)
    return out


@pytest.mark.integration
@pytest.mark.slow
def test_generator_writes_unified_observed_catalog_metadata(tmp_path):
    mock = _generate_unified_mock(tmp_path)

    assert (mock / "mock_observed_gw_pe.h5").exists()
    assert (mock / "observed_catalog.json").exists()

    observed = json.loads((mock / "observed_catalog.json").read_text())
    candidate_pairs = json.loads((mock / "candidate_pairs.json").read_text())
    partition = json.loads((mock / "partition.json").read_text())

    assert observed["n_events"] == candidate_pairs["n_events"]
    assert observed["truth_partition"]["singleton_indices"] == partition["singleton_indices"]
    assert observed["truth_partition"]["pair_indices"] == partition["pair_indices"]

    with h5py.File(mock / "mock_observed_gw_pe.h5") as f:
        assert int(f.attrs["nobs"]) == observed["n_events"]
    with h5py.File(mock / "mock_pair_pe.h5") as f:
        for k, pair_indices in enumerate(partition["pair_indices"]):
            group = f[f"pair_{k}"]
            assert int(group.attrs["event_index_image0"]) == pair_indices[0]
            assert int(group.attrs["event_index_image1"]) == pair_indices[1]
            assert int(group.attrs["pair_index"]) == k


@pytest.mark.integration
@pytest.mark.slow
def test_load_inputs_accepts_unified_observed_catalog_fixed_and_marginalized(tmp_path):
    from darksirens.cli.inference_lensing import load_inputs

    mock = _generate_unified_mock(tmp_path)
    base = dict(
        seed=1,
        gw_path=str(mock / "mock_observed_gw_pe.h5"),
        gwselection_path=str(mock / "mock_gw_selection.h5"),
        cluster_mode="j2",
        lensed_injections_path=str(mock / "mock_lensed_injections.h5"),
        pair_pe_path=str(mock / "mock_pair_pe.h5"),
        pe_max_per_pair=8,
        pair_marks="none",
        pair_time_sigma_sec=None,
        max_exact_partitions=1000,
    )

    fixed = load_inputs(Namespace(
        **base,
        partition_mode="fixed",
        partition_path=str(mock / "partition.json"),
        candidate_pairs_path=None,
    ))
    assert fixed["nEvents"] == 6
    assert fixed["n_pairs"] == 2

    marginalized = load_inputs(Namespace(
        **base,
        partition_mode="marginalize_exact",
        partition_path=None,
        candidate_pairs_path=str(mock / "candidate_pairs.json"),
    ))
    assert marginalized["nEvents"] == 6
    assert marginalized["partition_states"] is not None
