"""scripts/subset_pe_store.py: format guard and per-event attr reindexing (DS-11)."""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "subset_pe_store.py"

NOBS, NSAMP = 3, 4
NAMES = ["GW_A", "GW_B", "GW_C"]


def _write_store(path, fmt="gwcat-pe-2.1"):
    n = NOBS * NSAMP
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = fmt
        f.attrs["nobs"] = NOBS
        f.attrs["nsamp"] = NSAMP
        f.attrs["event_names"] = np.array(NAMES, dtype=h5py.string_dtype())
        f.attrs["spin_amax_1_per_event"] = np.array([0.99, 0.05, 0.8])
        f.attrs["contract_hash"] = "deadbeefdeadbeef"
        for name in ("m1det", "m2det", "dL", "chieff", "p_pe", "ra", "dec",
                     "m1src", "m2src"):
            f.create_dataset(name, data=np.arange(n, dtype=float))


def _make_flows(tmp_path, names):
    flows = tmp_path / "flows"
    for name in names:
        d = flows / name
        d.mkdir(parents=True)
        np.savez(d / f"{name}_flow.npz", dummy=np.zeros(1))
    return flows


def _run(store, flows, output):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(store),
         "--flows", str(flows), "--output", str(output)],
        capture_output=True, text=True,
    )


def test_subsets_and_reindexes_per_event_attrs(tmp_path):
    store = tmp_path / "store.h5"
    _write_store(store)
    flows = _make_flows(tmp_path, ["GW_C", "GW_A"])  # sorted: GW_A, GW_C
    out = tmp_path / "subset.h5"
    res = _run(store, flows, out)
    assert res.returncode == 0, res.stderr

    with h5py.File(out) as f:
        assert int(f.attrs["nobs"]) == 2
        got_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f.attrs["event_names"]]
        assert got_names == ["GW_A", "GW_C"]
        # Per-event attr reindexed to the subset rows, in flow order.
        np.testing.assert_allclose(
            np.asarray(f.attrs["spin_amax_1_per_event"]), [0.99, 0.8])
        # Space-level attrs invariant under an event subset: copied verbatim.
        assert f.attrs["contract_hash"] == "deadbeefdeadbeef"
        assert f.attrs["subset_n_original_events"] == NOBS
        # Sample blocks follow the same rows.
        m1 = np.asarray(f["m1det"]).reshape(2, NSAMP)
        np.testing.assert_allclose(m1[0], np.arange(0, 4, dtype=float))
        np.testing.assert_allclose(m1[1], np.arange(8, 12, dtype=float))


def test_unknown_format_refused(tmp_path):
    store = tmp_path / "store_sel.h5"
    _write_store(store, fmt="gwcat-selection-2.0")
    flows = _make_flows(tmp_path, ["GW_A"])
    res = _run(store, flows, tmp_path / "subset.h5")
    assert res.returncode != 0
    assert "Refusing to subset" in res.stderr
