"""scripts/diagnose_selection_guard.py: the guard wrapper must reach EVERY stack.

The script measures the selection variance guard by wrapping
``selection_log_correction``.  Each likelihood stack binds that symbol in its own
module namespace (``core``, ``flow_events`` for ``--gw_flows_path``,
``cluster_selection`` for the lensing stack), so a single-module patch measured
one code path and reported "the guard was not evaluated" for the others -- the
opposite of the truth on exactly the runs the script exists to diagnose.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "diagnose_selection_guard.py"


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("_diag_guard_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wrapper_reaches_core_flows_and_cluster_bindings(diag):
    import darksirens.likelihood.selection as selection_mod

    original = selection_mod.selection_log_correction
    # core is imported eagerly by the CLI; the other two are imported lazily
    # inside it, so they must pick the wrapper up through the source module.
    importlib.import_module("darksirens.likelihood.core")

    def _wrapper(*a, **k):  # pragma: no cover - never called here
        return original(*a, **k)

    patched = diag._patch_selection_guard(_wrapper, original)
    try:
        assert selection_mod.selection_log_correction is _wrapper
        assert sys.modules["darksirens.likelihood.core"].selection_log_correction \
            is _wrapper
        for name in ("darksirens.likelihood.flow_events",
                     "darksirens.likelihood.cluster_selection"):
            mod = importlib.import_module(name)
            assert mod.selection_log_correction is _wrapper, name
    finally:
        diag._restore_selection_guard(patched, original)
        for name in ("darksirens.likelihood.flow_events",
                     "darksirens.likelihood.cluster_selection"):
            sys.modules[name].selection_log_correction = original

    assert selection_mod.selection_log_correction is original
    assert sys.modules["darksirens.likelihood.core"].selection_log_correction \
        is original
