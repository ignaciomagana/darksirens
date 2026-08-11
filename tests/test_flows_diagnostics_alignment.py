"""scripts/flows_diagnostics.py must not compare a PERMUTED PE store.

The flow ensemble is ordered by lexicographic checkpoint name; the PE store
carries its own ``event_names`` order.  Every per-event product in the script
joins the two by POSITION, so a store with the right event COUNT but a different
order silently reports a permuted comparison -- which looks exactly like "the
flows are a poor fit", the failure the script exists to detect.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "flows_diagnostics.py"


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("_flows_diag_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _store(path, names):
    with h5py.File(path, "w") as f:
        f.attrs["event_names"] = np.asarray(names, dtype=object)


def test_matching_order_passes(diag, tmp_path):
    p = tmp_path / "store.h5"
    _store(p, ["GW150914", "GW170817"])
    diag._assert_store_matches_ensemble(p, SimpleNamespace(names=["GW150914", "GW170817"]))


def test_permuted_order_is_refused(diag, tmp_path):
    p = tmp_path / "store.h5"
    _store(p, ["GW170817", "GW150914"])
    with pytest.raises(SystemExit, match="BY POSITION"):
        diag._assert_store_matches_ensemble(
            p, SimpleNamespace(names=["GW150914", "GW170817"]))


def test_missing_event_names_is_refused(diag, tmp_path):
    p = tmp_path / "store.h5"
    with h5py.File(p, "w") as f:
        f.attrs["nobs"] = 2
    with pytest.raises(SystemExit, match="subset_pe_store"):
        diag._assert_store_matches_ensemble(
            p, SimpleNamespace(names=["GW150914", "GW170817"]))
