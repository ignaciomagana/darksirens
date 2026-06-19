"""
End-to-end pipeline tests for LSS-conditioned lognormal completion:
the offline build CLI, loading the result into the inference data path, and the
diagnostic CLI.  These are heavier than ``test_lss_completion.py`` (they touch
the survey loader, ``load_all_data`` and matplotlib).
"""
from types import SimpleNamespace
import sys
import types

# darksirens.gw.utils imports tqdm at import time; the loader test stubs it.
sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
sys.modules["tqdm"].tqdm = lambda iterable=None, *a, **k: iterable
for _name in ("gwdistributions", "gwdistributions.distributions", "gwdistributions.distributions.spin"):
    sys.modules.setdefault(_name, types.ModuleType(_name))


class _SpinPriorStub:
    def _init_values(self, *a, **k):
        return None

    def _logprob(self, *a, **k):
        return 0.0


sys.modules["gwdistributions.distributions.spin"].IsotropicUniformMagnitudeChiEffGivenComponentMass = _SpinPriorStub

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")
import h5py
import healpy as hp

from darksirens.em import zgrid
from darksirens.em.lognormal_completion import load_lss_completion_hdf5
from darksirens.tool.darksirens_build_lognormal_completion import build_completion, main as build_main
from darksirens.tool.darksirens_diagnose_lognormal_completion import main as diagnose_main

NG = int(zgrid.size)


@pytest.fixture
def survey_path(tmp_path):
    nside = 1
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int32)
    counts[2], counts[5], counts[7] = 4, 2, 3
    max_gals = int(counts.max())
    zgals = np.zeros((npix, max_gals))
    dzgals = np.ones((npix, max_gals)) * 0.01
    wgals = np.zeros((npix, max_gals))
    rng = np.random.default_rng(0)
    for pix, n in enumerate(counts):
        if n:
            zgals[pix, :n] = rng.uniform(0.05, 0.6, size=n)
            wgals[pix, :n] = 1.0
    path = tmp_path / "survey.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return str(path), npix


def test_build_completion_shapes_finite(survey_path):
    path, npix = survey_path
    logq_map, logq_members, diag = build_completion(path, n_members=4, seed=1, maxiter=40)
    assert logq_map.shape == (npix, NG)
    assert np.all(np.isfinite(logq_map))
    assert logq_members.shape == (4, npix, NG)
    assert np.all(np.isfinite(logq_members))
    assert diag["n_grid"] == NG and diag["nside"] == 1


def test_build_cli_then_load_into_inference(survey_path, tmp_path, monkeypatch):
    path, npix = survey_path
    qfile = str(tmp_path / "lss_completion.h5")
    build_main(["--catalog", path, "--out", qfile, "--n-members", "0",
                "--maxiter", "30", "--indexing", "global"])
    loaded = load_lss_completion_hdf5(qfile)
    assert loaded["logq_map"].shape == (npix, NG)
    assert loaded["indexing"] == "global"

    # Load into the inference data path (mock the GW readers, dark_sirens).
    from darksirens.inference import data as data_module

    pe_pix = np.array([5, 2], dtype=np.int32)
    sel_pix = np.array([7, 2], dtype=np.int32)

    def _angles(pixels):
        theta, phi = hp.pix2ang(1, np.asarray(pixels, dtype=np.int64))
        return phi, np.pi / 2.0 - theta

    pe_ra, pe_dec = _angles(pe_pix)
    sel_ra, sel_dec = _angles(sel_pix)
    monkeypatch.setattr(data_module, "load_gw_samples", lambda _p: (
        np.array([36.0, 38.0]), np.array([28.8, 30.4]), np.array([460.0, 500.0]),
        np.array([0.0, 0.02]), pe_ra, pe_dec, np.ones(2), 1, 2))
    monkeypatch.setattr(data_module, "load_selection_samples", lambda _p: (
        np.array([34.0, 40.0]), np.array([27.2, 32.0]), np.array([430.0, 530.0]),
        np.zeros(2), sel_ra, sel_dec, np.ones(2), 2))

    opts = SimpleNamespace(
        universe_model="dark_sirens", survey_path=path,
        gw_path="x", gwselection_path="x", use_LSS=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=0.01,
        lss_completion=qfile,
    )
    data = data_module.load_all_data(opts)
    assert data["lss_completion_logq"] is not None
    assert np.asarray(data["lss_completion_logq"]).shape == (npix, NG)
    assert data["lss_completion_indexing"] == 2  # global


def test_diagnose_cli_smoke(survey_path, tmp_path):
    path, npix = survey_path
    qfile = str(tmp_path / "lss_completion.h5")
    build_main(["--catalog", path, "--out", qfile, "--n-members", "4",
                "--maxiter", "30", "--seed", "3", "--indexing", "global"])
    outdir = tmp_path / "figs"
    diagnose_main(["--catalog", path, "--lss-completion", qfile,
                   "--pixel", "2", "--outdir", str(outdir)])
    assert (outdir / "lss_completion_pixel2.pdf").exists()
