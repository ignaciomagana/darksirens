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

from darksirens.redshift import zgrid
from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
from darksirens.cli.build_lognormal_completion import build_completion, main as build_main
from darksirens.cli.diagnose_lognormal_completion import main as diagnose_main

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
    monkeypatch.setattr(data_module.loaders, "load_gw_samples", lambda _p: (
        np.array([36.0, 38.0]), np.array([28.8, 30.4]), np.array([460.0, 500.0]),
        np.array([0.0, 0.02]), pe_ra, pe_dec, np.ones(2), 1, 2))
    monkeypatch.setattr(data_module.loaders, "load_selection_samples", lambda _p: (
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


def test_center_marks_zero_global_mean():
    """_center_marks subtracts the running E[m|z], leaving zero global mean
    over real galaxies (pure within-z reweighting)."""
    from darksirens.catalogs.marks import _center_marks
    zg = np.array([[0.10, 0.50, 1.00], [0.20, 0.60, 1.10]])
    ng = np.array([3, 3], dtype=np.int32)
    M = 2.0 + 3.0 * zg  # linear-in-z mark
    out = _center_marks({"mark_logmstar": M}, zg, ng)["mark_logmstar"]
    assert out.shape == (2, 3)
    real = np.arange(3)[None, :] < ng[:, None]
    assert abs(float(out[real].mean())) < 1e-10  # per-bin centering => zero global mean


def test_pixelate_writes_marks_and_load_all_data_centers_them(tmp_path, monkeypatch):
    from darksirens.cli.pixelate import main as pixelate_main
    from darksirens.catalogs.io import load_survey_marks
    from darksirens.inference import data as data_module

    # Raw DESI-like catalog with a LOGMSTAR mark column.
    nside = 1
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    n = 60
    pix = rng.integers(0, npix, size=n)
    theta, phi = hp.pix2ang(nside, pix)
    raw = tmp_path / "desi_raw.h5"
    with h5py.File(raw, "w") as f:
        f.create_dataset("TARGET_RA", data=np.degrees(phi))
        f.create_dataset("TARGET_DEC", data=np.degrees(np.pi / 2 - theta))
        zs = rng.uniform(0.05, 0.8, size=n)
        f.create_dataset("Z", data=zs)
        f.create_dataset("ZERR", data=np.full(n, 0.001))
        f.create_dataset("WEIGHT", data=np.ones(n))
        f.create_dataset("LOGMSTAR", data=10.0 + 2.0 * zs)  # correlated with z

    pixelate_main(["--survey_path", str(raw), "--save_path", str(tmp_path), "--nside", str(nside)])
    cat = str(tmp_path / f"catalog_pixelated_nside_{nside}.h5")
    marks = load_survey_marks(cat)
    assert "mark_logmstar" in marks and marks["mark_logmstar"].shape[0] == npix

    # Load into the inference data path (dark_sirens) -> centered full-catalog marks.
    pe_pix = np.array([int(pix[0])], dtype=np.int32)
    sel_pix = np.array([int(pix[1])], dtype=np.int32)

    def _ang(p):
        th, ph = hp.pix2ang(nside, np.asarray(p, dtype=np.int64))
        return ph, np.pi / 2 - th

    pe_ra, pe_dec = _ang(pe_pix)
    sel_ra, sel_dec = _ang(sel_pix)
    monkeypatch.setattr(data_module.loaders, "load_gw_samples", lambda _p: (
        np.array([36.0]), np.array([28.8]), np.array([460.0]), np.array([0.0]),
        pe_ra, pe_dec, np.ones(1), 1, 1))
    monkeypatch.setattr(data_module.loaders, "load_selection_samples", lambda _p: (
        np.array([34.0, 40.0]), np.array([27.2, 32.0]), np.array([430.0, 530.0]),
        np.zeros(2), sel_ra, sel_dec, np.ones(2), 2))

    opts = SimpleNamespace(
        universe_model="dark_sirens", survey_path=cat,
        gw_path="x", gwselection_path="x", use_LSS=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=0.01, lss_completion=None,
    )
    data = data_module.load_all_data(opts)
    cm = data["mark_logmstar"]
    assert cm is not None and np.asarray(cm).shape[0] == npix
    # centered: zero global mean over real galaxies
    ng = np.asarray(data["ngals_catalog"])
    real = np.arange(np.asarray(cm).shape[1])[None, :] < ng[:, None]
    assert abs(float(np.asarray(cm)[real].mean())) < 1e-8


def test_diagnose_cli_smoke(survey_path, tmp_path):
    path, npix = survey_path
    qfile = str(tmp_path / "lss_completion.h5")
    build_main(["--catalog", path, "--out", qfile, "--n-members", "4",
                "--maxiter", "30", "--seed", "3", "--indexing", "global"])
    outdir = tmp_path / "figs"
    diagnose_main(["--catalog", path, "--lss-completion", qfile,
                   "--pixel", "2", "--outdir", str(outdir)])
    assert (outdir / "lss_completion_pixel2.pdf").exists()


def test_build_occupied_only_and_uniform_chi(survey_path):
    """Tier-2: build only occupied pixels (empties -> logQ=0) on a uniform-chi grid."""
    path, npix = survey_path
    logq_map, _, diag = build_completion(path, n_members=0, maxiter=40)
    assert logq_map.shape == (npix, NG)
    assert np.all(np.isfinite(logq_map))
    assert 0 < diag["n_occupied"] <= npix
    assert diag["dchi_uniform_mpc"] > 0.0          # uniform comoving-distance grid
    with h5py.File(path, "r") as f:
        ng = np.asarray(f["ngals"])
    empty = ng == 0
    assert empty.any()
    assert np.allclose(logq_map[empty], 0.0)       # empty pixels -> Q = 1
