from types import SimpleNamespace
import sys
import types

# ``darksirens.gw.utils`` imports tqdm at module import time, but this loader
# test monkeypatches the GW file readers and does not need tqdm itself.
tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", tqdm_stub)

gwdistributions_stub = types.ModuleType("gwdistributions")
distributions_stub = types.ModuleType("gwdistributions.distributions")
spin_stub = types.ModuleType("gwdistributions.distributions.spin")


class _SpinPriorStub:
    def _init_values(self, *args, **kwargs):
        return None

    def _logprob(self, *args, **kwargs):
        return 0.0


spin_stub.IsotropicUniformMagnitudeChiEffGivenComponentMass = _SpinPriorStub
sys.modules.setdefault("gwdistributions", gwdistributions_stub)
sys.modules.setdefault("gwdistributions.distributions", distributions_stub)
sys.modules.setdefault("gwdistributions.distributions.spin", spin_stub)

import h5py
import healpy as hp
import numpy as np
import pytest

from darksirens.inference import data as data_module


def _angles_for_pixels(nside, pixels):
    theta, phi = hp.pix2ang(nside, np.asarray(pixels, dtype=np.int64))
    ra = phi
    dec = np.pi / 2.0 - theta
    return ra, dec


@pytest.fixture
def survey_fixture(tmp_path):
    nside = 1
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int32)
    counts[2] = 3
    counts[5] = 1
    counts[7] = 2
    counts[10] = 0
    max_gals = int(counts.max())

    zgals = np.zeros((npix, max_gals), dtype=float)
    dzgals = np.ones((npix, max_gals), dtype=float) * 0.01
    wgals = np.zeros((npix, max_gals), dtype=float)
    for pix, n_gal in enumerate(counts):
        if n_gal:
            zgals[pix, :n_gal] = 0.01 * (pix + np.arange(n_gal) + 1)
            wgals[pix, :n_gal] = 1.0

    path = tmp_path / "survey.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)

    return path, counts


def test_load_all_data_returns_named_pe_counts_for_survey_fixture(
    survey_fixture, monkeypatch
):
    survey_path, counts = survey_fixture
    nside = 1
    pe_pixels = np.array([5, 2], dtype=np.int32)
    sel_pixels = np.array([7, 10], dtype=np.int32)
    pe_ra, pe_dec = _angles_for_pixels(nside, pe_pixels)
    sel_ra, sel_dec = _angles_for_pixels(nside, sel_pixels)

    def fake_load_gw_samples(_path):
        return (
            np.array([36.0, 38.0]),
            np.array([28.8, 30.4]),
            np.array([460.0, 500.0]),
            np.array([0.0, 0.02]),
            pe_ra,
            pe_dec,
            np.ones(2),
            1,
            2,
        )

    def fake_load_selection_samples(_path):
        return (
            np.array([34.0, 40.0]),
            np.array([27.2, 32.0]),
            np.array([430.0, 530.0]),
            np.zeros(2),
            sel_ra,
            sel_dec,
            np.ones(2),
            2,
        )

    monkeypatch.setattr(data_module.loaders, "load_gw_samples", fake_load_gw_samples)
    monkeypatch.setattr(
        data_module.loaders, "load_selection_samples", fake_load_selection_samples
    )

    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_path=str(survey_path),
        gw_path="unused-gw.hdf5",
        gwselection_path="unused-selection.hdf5",
        use_LSS=False,
        counterpart=None,
        counterpart_nside=1,
        counterpart_dz=0.01,
    )

    loaded = data_module.load_all_data(opts)

    assert "ngals" not in loaded
    np.testing.assert_array_equal(loaded["pixels_pe"], pe_pixels)
    np.testing.assert_array_equal(loaded["pixels_sel"], sel_pixels)
    np.testing.assert_array_equal(loaded["unique_pixels_pe"], np.array([2, 5]))
    np.testing.assert_array_equal(loaded["unique_pixels_sel"], np.array([7, 10]))
    np.testing.assert_array_equal(loaded["ngals_pe"], counts[[2, 5]])
    np.testing.assert_array_equal(loaded["ngals_sel"], counts[[7, 10]])
    assert loaded["ngals_pe"].shape[0] == loaded["unique_pixels_pe"].shape[0]
    assert loaded["ngals_sel"].shape[0] == loaded["unique_pixels_sel"].shape[0]
    assert loaded["sample_to_unique_pe"].shape[0] == loaded["pixels_pe"].shape[0]
    assert loaded["sample_to_unique_sel"].shape[0] == loaded["pixels_sel"].shape[0]


def test_load_all_data_stores_bright_siren_counterpart_pixel_and_keeps_it_compact(monkeypatch):
    nside = 2
    counterpart_pixel = 7
    non_counterpart_pixel = 8
    cp_ra, cp_dec = _angles_for_pixels(nside, np.array([counterpart_pixel], dtype=np.int32))
    pe_ra, pe_dec = _angles_for_pixels(
        nside, np.array([counterpart_pixel, non_counterpart_pixel], dtype=np.int32)
    )
    sel_ra, sel_dec = _angles_for_pixels(
        nside, np.array([non_counterpart_pixel], dtype=np.int32)
    )

    def fake_load_gw_samples(_path):
        return (
            np.array([36.0, 38.0]),
            np.array([28.8, 30.4]),
            np.array([460.0, 500.0]),
            np.array([0.0, 0.02]),
            pe_ra,
            pe_dec,
            np.ones(2),
            1,
            2,
        )

    def fake_load_selection_samples(_path):
        return (
            np.array([34.0]),
            np.array([27.2]),
            np.array([430.0]),
            np.zeros(1),
            sel_ra,
            sel_dec,
            np.ones(1),
            1,
        )

    monkeypatch.setattr(data_module.loaders, "load_gw_samples", fake_load_gw_samples)
    monkeypatch.setattr(
        data_module.loaders, "load_selection_samples", fake_load_selection_samples
    )

    opts = SimpleNamespace(
        universe_model="bright_sirens",
        survey_path=None,
        gw_path="unused-gw.hdf5",
        gwselection_path="unused-selection.hdf5",
        use_LSS=False,
        counterpart=(float(cp_ra[0]), float(cp_dec[0]), 0.2),
        counterpart_nside=nside,
        counterpart_dz=0.01,
        bright_siren_sky_marginalized=False,
    )

    loaded = data_module.load_all_data(opts)

    assert loaded["counterpart_pixel"] == counterpart_pixel
    assert loaded["bright_siren_sky_marginalized"] is False
    assert loaded["nside"] == nside
    np.testing.assert_array_equal(
        loaded["pixels_pe"], np.array([counterpart_pixel, non_counterpart_pixel])
    )
    assert counterpart_pixel in set(loaded["unique_pixels_pe"].tolist())
    assert counterpart_pixel in set(loaded["unique_pixels_sel"].tolist())


def test_load_all_data_accepts_multiple_bright_siren_counterparts(monkeypatch):
    nside = 2
    counterpart_pixels = np.array([7, 8], dtype=np.int32)
    cp_ra, cp_dec = _angles_for_pixels(nside, counterpart_pixels)

    def fake_load_gw_samples(_path):
        return (
            np.array([36.0, 38.0, 34.0, 35.0]),
            np.array([28.8, 30.4, 27.2, 28.0]),
            np.array([460.0, 500.0, 800.0, 820.0]),
            np.array([0.0, 0.02, -0.01, 0.01]),
            np.repeat(cp_ra, 2),
            np.repeat(cp_dec, 2),
            np.ones(4),
            2,
            2,
        )

    def fake_load_selection_samples(_path):
        return (
            np.array([34.0]),
            np.array([27.2]),
            np.array([430.0]),
            np.zeros(1),
            np.array([cp_ra[0]]),
            np.array([cp_dec[0]]),
            np.ones(1),
            1,
        )

    monkeypatch.setattr(data_module.loaders, "load_gw_samples", fake_load_gw_samples)
    monkeypatch.setattr(data_module.loaders, "load_selection_samples", fake_load_selection_samples)

    opts = SimpleNamespace(
        universe_model="bright_sirens",
        survey_path=None,
        gw_path="unused-gw.hdf5",
        gwselection_path="unused-selection.hdf5",
        use_LSS=False,
        counterpart=((float(cp_ra[0]), float(cp_dec[0]), 0.2), (float(cp_ra[1]), float(cp_dec[1]), 0.35)),
        counterpart_nside=nside,
        counterpart_dz=0.01,
        bright_siren_sky_marginalized=True,
    )

    loaded = data_module.load_all_data(opts)

    np.testing.assert_array_equal(loaded["counterpart_pixels"], counterpart_pixels)
    np.testing.assert_allclose(loaded["counterpart_zs"], np.array([0.2, 0.35]))
    assert loaded["bright_siren_sky_marginalized"] is True
    for pix in counterpart_pixels:
        assert pix in set(loaded["unique_pixels_pe"].tolist())
        assert pix in set(loaded["unique_pixels_sel"].tolist())
