"""Fail-closed convergence gating for LSS completion artifacts (P1-01/P1-02).

Before these fixes the completion pipeline failed OPEN at every layer:

* ``poisson_lognormal_map`` never set L-BFGS-B's ``maxfun``, so scipy's
  15000-EVALUATION default silently capped every large-``maxiter`` solve with
  ``success=False`` near (but not at) its fixed point — the mechanism behind
  production tables reading ``n_converged = 0`` while raising ``--maxiter``
  changed nothing;
* the builder saved whatever the optimizer's last iterate was (measured: a
  4-galaxy pixel's 10000-iterate reads logQ 0.556 where the fixed point is
  0.924 — a 40% under-relaxation that downstream inference cannot detect);
* non-finite gp3d cells were silently replaced with Q = 1;
* ``save_lss_completion_hdf5`` truncated its destination in place, so a died
  build could leave a partial artifact that later inference trusted;
* the radial builder accepted ``--indexing compact`` while always writing a
  GLOBAL table, mis-stamping the file so the likelihood factory consumed
  wrong rows (or crashed inside JIT).
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("jax")
import h5py

from darksirens.redshift import zgrid
from darksirens.redshift.lognormal_completion import (
    gaussian_correlation_spectrum,
    load_lss_completion_hdf5,
    poisson_lognormal_map,
    save_lss_completion_hdf5,
)

NG = int(zgrid.size)


def _radial_problem(n_grid=64):
    N = np.zeros(n_grid)
    N[[10, 15, 20, 40]] = [2, 1, 1, 1]
    C = np.ones(n_grid) * 0.5
    dN = np.full(n_grid, 0.01)
    pk = gaussian_correlation_spectrum(n_grid, 4.0, 0.8)
    return N, C, dN, pk


# ---------------------------------------------------------------------------
# The maxfun mechanism and honest convergence accounting
# ---------------------------------------------------------------------------

def test_maxfun_scales_with_maxiter(monkeypatch):
    """scipy's maxfun default (15000 EVALUATIONS) must never silently bind
    before the requested iteration cap does."""
    import scipy.optimize as so

    captured = {}
    real_minimize = so.minimize

    def spy(*args, **kwargs):
        captured.update(kwargs.get("options") or {})
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(so, "minimize", spy)
    N, C, dN, pk = _radial_problem()
    poisson_lognormal_map(N, C, dN, pk, maxiter=30000)
    assert captured.get("maxiter") == 30000
    # maxls (default 20) bounds evaluations per iteration, so this maxfun can
    # never bind before maxiter itself does.
    assert captured.get("maxfun", 0) >= 21 * 30000


def test_unconverged_solve_is_reported_honestly():
    N, C, dN, pk = _radial_problem()
    out = poisson_lognormal_map(N, C, dN, pk, maxiter=1)
    d = out["diagnostics"]
    assert d["n_converged"] == 0
    assert d["converged"] is False
    assert "LIMIT" in (d["last_failure_message"] or "")


def test_converged_solve_is_reported_honestly():
    N, C, dN, pk = _radial_problem()
    out = poisson_lognormal_map(N, C, dN, pk)  # default cap: self-terminates
    d = out["diagnostics"]
    assert d["converged"] is True
    assert d["n_converged"] == d["n_rows"]
    assert d["last_failure_message"] is None


# ---------------------------------------------------------------------------
# save: finiteness + atomicity
# ---------------------------------------------------------------------------

def test_save_rejects_nonfinite_and_preserves_existing_file(tmp_path):
    path = str(tmp_path / "q.h5")
    good = np.zeros((2, NG))
    save_lss_completion_hdf5(path, logq_map=good, zgrid=np.asarray(zgrid),
                             indexing="global")
    original = open(path, "rb").read()

    bad = good.copy()
    bad[1, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_lss_completion_hdf5(path, logq_map=bad, zgrid=np.asarray(zgrid),
                                 indexing="global")
    assert open(path, "rb").read() == original, (
        "a rejected save must leave the existing artifact byte-identical"
    )
    assert not [n for n in os.listdir(tmp_path) if ".tmp-" in n], (
        "no temp file may survive a rejected save"
    )

    bad_members = np.zeros((2, 2, NG))
    bad_members[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        save_lss_completion_hdf5(path, logq_map=good, logq_members=bad_members,
                                 zgrid=np.asarray(zgrid), indexing="global")
    assert open(path, "rb").read() == original


# ---------------------------------------------------------------------------
# load: finiteness + convergence surfacing (legacy vs stamped files)
# ---------------------------------------------------------------------------

def _write_raw_completion(path, logq, diagnostics=None, indexing="global"):
    """Write a completion file directly (bypassing save's validation), the way
    a legacy builder or a corrupted transfer could have produced it."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("lss_completion")
        grp.create_dataset("logq_map", data=logq)
        grp.attrs["indexing"] = indexing
        if diagnostics is not None:
            grp.attrs["diagnostics"] = json.dumps(diagnostics)


def test_load_rejects_nonfinite_content(tmp_path):
    path = str(tmp_path / "q.h5")
    logq = np.zeros((2, 8))
    logq[0, 1] = np.nan
    _write_raw_completion(path, logq)
    with pytest.raises(ValueError, match="non-finite"):
        load_lss_completion_hdf5(path)


def test_load_warns_but_accepts_legacy_unconverged_diagnostics(tmp_path):
    """A LEGACY file's n_converged counter is unreliable (the maxfun cap), so
    a verified-good production table must keep loading — with a loud warning,
    not a rejection."""
    path = str(tmp_path / "q.h5")
    _write_raw_completion(
        path, np.zeros((2, 8)),
        diagnostics={"n_converged": 0, "n_occupied": 2},
    )
    with pytest.warns(RuntimeWarning, match="maxfun"):
        out = load_lss_completion_hdf5(path)
    assert out["logq_map"].shape == (2, 8)


def test_load_warns_on_stamped_allow_unconverged(tmp_path):
    path = str(tmp_path / "q.h5")
    _write_raw_completion(
        path, np.zeros((2, 8)),
        diagnostics={"converged": False, "allow_unconverged": True},
    )
    with pytest.warns(RuntimeWarning, match="allow-unconverged"):
        load_lss_completion_hdf5(path)


def test_load_is_quiet_for_converged_files(tmp_path):
    import warnings as _warnings

    path = str(tmp_path / "q.h5")
    _write_raw_completion(
        path, np.zeros((2, 8)),
        diagnostics={"converged": True, "n_converged": 2, "n_occupied": 2,
                     "allow_unconverged": False},
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        out = load_lss_completion_hdf5(path)
    assert out["diagnostics"]["converged"] is True


# ---------------------------------------------------------------------------
# Builder CLI gate (radial, tiny survey)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_survey(tmp_path):
    hp = pytest.importorskip("healpy")
    nside = 1
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int32)
    counts[2] = 3
    zgals = np.zeros((npix, 3))
    zgals[2, :3] = [0.1, 0.2, 0.3]
    path = tmp_path / "survey.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=np.full((npix, 3), 0.01))
        f.create_dataset("wgals", data=(zgals > 0).astype(float))
    return str(path), npix


def test_builder_cli_refuses_unconverged_and_writes_nothing(tiny_survey, tmp_path):
    from darksirens.cli.build_lognormal_completion import main as build_main

    path, _ = tiny_survey
    qfile = str(tmp_path / "q_gate.h5")
    with pytest.raises(SystemExit, match="unconverged"):
        build_main(["--catalog", path, "--out", qfile, "--n-members", "0",
                    "--maxiter", "1"])
    assert not os.path.exists(qfile), "the gate must not leave an artifact"


def test_builder_cli_allow_unconverged_stamps_the_override(tiny_survey, tmp_path):
    from darksirens.cli.build_lognormal_completion import main as build_main

    path, npix = tiny_survey
    qfile = str(tmp_path / "q_ablation.h5")
    build_main(["--catalog", path, "--out", qfile, "--n-members", "0",
                "--maxiter", "1", "--allow-unconverged"])
    with pytest.warns(RuntimeWarning, match="allow-unconverged"):
        out = load_lss_completion_hdf5(qfile)
    d = out["diagnostics"]
    assert d["allow_unconverged"] is True
    assert d["converged"] is False
    assert out["logq_map"].shape == (npix, NG)


def test_builder_cli_forces_global_indexing(tiny_survey, tmp_path):
    """P1-02: both build modes emit a full global table; a 'compact' stamp
    would make the factory consume wrong rows (or crash inside JIT)."""
    from darksirens.cli.build_lognormal_completion import main as build_main

    path, _ = tiny_survey
    qfile = str(tmp_path / "q_idx.h5")
    build_main(["--catalog", path, "--out", qfile, "--n-members", "0",
                "--maxiter", "1", "--allow-unconverged",
                "--indexing", "compact"])
    out = load_lss_completion_hdf5(qfile)
    assert out["indexing"] == "global"


# ---------------------------------------------------------------------------
# Factory: a compact-stamped table must be row-aligned before JIT
# ---------------------------------------------------------------------------

def test_factory_rejects_misaligned_compact_stamp():
    """P1-02 end-to-end: a 'compact'-stamped table whose rows are NOT aligned
    with the run's union pixel set must fail while building the likelihood --
    the verified failure mode was wrong rows consumed silently when shapes
    coincided, and a ConcretizationTypeError inside JIT when they did not."""
    from test_multitracer_likelihood import (
        APIX1, Z_A, Z_B, _base_opts, _bundle, _pop_bits, _shared_physics,
    )

    from darksirens.likelihood.factory import make_likelihood

    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    opts = _base_opts(
        n_catalogs=2,
        fix_survey=False,
        lss_completion_active_by_catalog=(True, False),
    )
    bundle_q = _bundle(APIX1, Z_A)
    bundle_q["lss_completion_logq"] = np.zeros((12, NG))  # 12 global rows...
    bundle_q["lss_completion_indexing"] = 1               # ...stamped compact
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [bundle_q, _bundle(APIX1, Z_B)]

    with pytest.raises(ValueError, match="compact"):
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
