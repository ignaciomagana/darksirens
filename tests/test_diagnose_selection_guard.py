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


def _fake_likelihood(live, records):
    """A likelihood stand-in that "evaluates the guard" once per call."""
    import numpy as np

    def _ll(theta):
        live.extend(records)
        return np.float64(-1.0)

    return _ll


def _rec(neff, n_obs=20.0, pe_var=0.0):
    return dict(Neff=neff, pe_variance_sum=pe_var, log_mu=-1.0,
                nEvents=n_obs, max_likelihood_variance=1.0)


def test_worst_record_decides_not_the_last(diag, capsys):
    """Under the member vmap / a K-catalog mixture the guard is evaluated many
    times per likelihood call; the run is decided by the WORST evaluation."""
    import numpy as np
    from types import SimpleNamespace

    records = [_rec(1e6), _rec(50.0), _rec(1e6)]   # the sparse one is NOT last
    live = []          # _diagnose clears it, then the likelihood call refills it
    captured = dict(
        likelihood=_fake_likelihood(live, records),
        prior_transform=lambda u: u,
        labels=["H0"], lower=np.array([20.0]), upper=np.array([120.0]),
        opts=SimpleNamespace(pop_model="powerlaw", max_likelihood_variance=1.0,
                             selection_neff_soft_guard=False),
    )
    ns = SimpleNamespace(diagnose_draws=1, diagnose_seed=0)
    assert diag._diagnose(captured, live, ns) == 0
    out = capsys.readouterr().out
    assert "guard evaluations     = 3" in out
    assert "Neff_sel              = 50" in out          # worst, not last
    assert "GUARDED (-inf)" in out


def test_soft_guard_run_is_not_told_to_enable_the_soft_guard(diag, capsys):
    import numpy as np
    from types import SimpleNamespace

    live = []
    captured = dict(
        likelihood=_fake_likelihood(live, [_rec(50.0)]),
        prior_transform=lambda u: u,
        labels=["H0"], lower=np.array([20.0]), upper=np.array([120.0]),
        opts=SimpleNamespace(pop_model="powerlaw", max_likelihood_variance=1.0,
                             selection_neff_soft_guard=True),
    )
    ns = SimpleNamespace(diagnose_draws=1, diagnose_seed=0)
    diag._diagnose(captured, live, ns)
    out = capsys.readouterr().out
    assert "PENALIZED (soft) region" in out
    assert "GUARDED (-inf)" not in out
    assert "--selection_neff_guard soft" not in out
