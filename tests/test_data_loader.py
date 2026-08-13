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

    def fake_load_selection_samples(_path, **_kwargs):
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

    def fake_load_selection_samples(_path, **_kwargs):
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

    def fake_load_selection_samples(_path, **_kwargs):
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


# ---------------------------------------------------------------------------
# PERF-3: K>=2 must not build the unused top-level single-catalog object, and
# --drop_full_catalog must not be silently ignored for a multitracer mixture
# (each catalog is loaded exactly ONCE, per bundle).
# ---------------------------------------------------------------------------

def _write_survey_file(path, nside, counts):
    max_gals = int(counts.max())
    zgals = np.zeros((len(counts), max_gals), dtype=float)
    dzgals = np.ones((len(counts), max_gals), dtype=float) * 0.01
    wgals = np.zeros((len(counts), max_gals), dtype=float)
    for pix, n_gal in enumerate(counts):
        if n_gal:
            zgals[pix, :n_gal] = 0.01 * (pix + np.arange(n_gal) + 1)
            wgals[pix, :n_gal] = 1.0
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return path


def test_load_all_data_multitracer_drops_top_level_catalog_and_loads_each_path_once(
    tmp_path, monkeypatch
):
    nside = 1
    npix = hp.nside2npix(nside)
    counts1 = np.zeros(npix, dtype=np.int32)
    counts1[2] = 3
    counts1[5] = 1
    counts2 = np.zeros(npix, dtype=np.int32)
    counts2[2] = 2
    counts2[7] = 4

    path1 = _write_survey_file(tmp_path / "survey1.hdf5", nside, counts1)
    path2 = _write_survey_file(tmp_path / "survey2.hdf5", nside, counts2)

    pe_pixels = np.array([5, 2], dtype=np.int32)
    sel_pixels = np.array([7, 2], dtype=np.int32)
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

    def fake_load_selection_samples(_path, **_kwargs):
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

    load_calls = []
    real_load_survey = data_module.loaders.load_survey

    def spy_load_survey(survey_path, to_device=True):
        load_calls.append(survey_path)
        return real_load_survey(survey_path, to_device=to_device)

    monkeypatch.setattr(data_module.loaders, "load_survey", spy_load_survey)

    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_path=str(path1),  # cli/inference.py sets this to survey_paths[0]
        survey_paths=[str(path1), str(path2)],
        n_catalogs=2,
        gw_path="unused-gw.hdf5",
        gwselection_path="unused-selection.hdf5",
        use_LSS=False,
        counterpart=None,
        counterpart_nside=1,
        counterpart_dz=0.01,
        drop_full_catalog=True,
        mark_model="none",
    )

    loaded = data_module.load_all_data(opts)

    # (a) K>=2 never carries a full top-level catalog -- --drop_full_catalog is
    # honored (there is nothing left for it to silently ignore).
    for key in (
        "zgals", "dzgals", "wgals", "ngals_catalog",
        "zgals_catalog", "dzgals_catalog", "wgals_catalog",
        "zgals_pe", "dzgals_pe", "wgals_pe", "ngals_pe",
        "zgals_sel", "dzgals_sel", "wgals_sel", "ngals_sel",
        "catalog_memory",
    ):
        assert loaded.get(key) is None, f"{key} should be None for a K>=2 mixture"

    assert loaded.get("catalogs") is not None
    assert len(loaded["catalogs"]) == 2

    # (b) each catalog path is loaded exactly once -- no double-load of catalog 1
    # via a wasted top-level load_or_build_catalog_inputs call.
    assert load_calls.count(str(path1)) == 1
    assert load_calls.count(str(path2)) == 1
    assert len(load_calls) == 2


# ---------------------------------------------------------------------------
# PERF-5: attach_mark_inputs must skip mark I/O entirely when mark_model is
# "none" (the default) -- loading and z-centering every full-size mark table
# is pure waste when no mark model is selected.
# ---------------------------------------------------------------------------

def test_attach_mark_inputs_skips_mark_io_when_mark_model_is_none(monkeypatch):
    calls = []

    def spy_load_and_center(survey_path, zgals, ngals, datasets=None):
        calls.append((survey_path, datasets))
        return {"mark_logmstar": np.zeros((1, 1))}

    monkeypatch.setattr(
        data_module.loaders, "load_and_center_survey_marks", spy_load_and_center
    )

    opts = SimpleNamespace(
        survey_path="unused-survey.hdf5",
        universe_model="dark_sirens",
        mark_model="none",
    )
    data = {"zgals": np.zeros((1, 1)), "ngals_catalog": np.zeros(1)}

    out = data_module.loaders.attach_mark_inputs(opts, data)

    assert calls == []
    for ds in ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color"):
        assert out[ds] is None


def test_attach_mark_inputs_loads_marks_when_mark_model_is_set(monkeypatch):
    calls = []

    def spy_load_and_center(survey_path, zgals, ngals, datasets=None):
        calls.append((survey_path, datasets))
        return {"mark_logmstar": np.ones((1, 1))}

    monkeypatch.setattr(
        data_module.loaders, "load_and_center_survey_marks", spy_load_and_center
    )

    opts = SimpleNamespace(
        survey_path="unused-survey.hdf5",
        universe_model="dark_sirens",
        mark_model="loglinear",
    )
    data = {"zgals": np.zeros((1, 1)), "ngals_catalog": np.zeros(1)}

    out = data_module.loaders.attach_mark_inputs(opts, data)

    assert len(calls) == 1
    assert calls[0][0] == "unused-survey.hdf5"
    np.testing.assert_array_equal(out["mark_logmstar"], np.ones((1, 1)))


def test_multitracer_bundle_loads_only_requested_mark_datasets(tmp_path, monkeypatch):
    """The K>=2 marked path must not read mark datasets the mixture never
    selected for that catalog (a cheap dataset-level filter on top of the
    mark_model="none" fix above)."""
    import darksirens.catalogs.marks as marks_module

    nside = 1
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int32)
    counts[2] = 2
    max_gals = 2
    zgals = np.zeros((npix, max_gals))
    zgals[2, :2] = [0.1, 0.2]
    dzgals = np.ones((npix, max_gals)) * 0.01
    wgals = np.zeros((npix, max_gals))
    wgals[2, :2] = 1.0
    mark_logmstar = np.zeros((npix, max_gals))
    mark_logssfr = np.zeros((npix, max_gals))

    path = tmp_path / "survey_marked.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
        f.create_dataset("mark_logmstar", data=mark_logmstar)
        f.create_dataset("mark_logssfr", data=mark_logssfr)

    captured = {}
    real_load_survey_marks = marks_module.load_survey_marks

    def spy_load_survey_marks(survey_path, datasets=None):
        captured["datasets"] = datasets
        return real_load_survey_marks(survey_path, datasets=datasets)

    # load_and_center_survey_marks (catalogs/marks.py) calls load_survey_marks
    # via a name bound in its OWN module namespace; patch it there so the
    # local re-import inside load_multitracer_catalog_bundles picks it up.
    monkeypatch.setattr(marks_module, "load_survey_marks", spy_load_survey_marks)

    ra = np.zeros(2)
    dec = np.zeros(2)
    gw_inputs = dict(ra=ra, dec=dec, rasels=ra.copy(), decsels=dec.copy())

    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_paths=[str(path)],
        lss_completions=[],
        mark_model="loglinear",
        mark_names_by_catalog=(("logmstar",),),
        use_LSS=False,
        lss_marginalize=False,
        catalog_sky_weighting="conditional",
        validate_completion=False,
    )

    bundles = data_module.loaders.load_multitracer_catalog_bundles(opts, gw_inputs)

    assert captured["datasets"] == ("mark_logmstar",)
    assert bundles[0].get("mark_logmstar") is not None
    assert "mark_logssfr" not in bundles[0]


def _write_depth_survey(path, nside=1, z_depth_attr=None):
    """Tiny full-sky survey whose galaxies straddle a 0.3 depth."""
    npix = hp.nside2npix(nside)
    max_gals = 2
    counts = np.zeros(npix, dtype=np.int32)
    zgals = np.zeros((npix, max_gals))
    dzgals = np.full((npix, max_gals), 0.02)
    wgals = np.zeros((npix, max_gals))
    for pix in (2, 5, 7):
        counts[pix] = 2
        zgals[pix, :2] = [0.10, 0.70]     # one below, one beyond the depth
        wgals[pix, :2] = [1.0, 3.0]       # non-uniform: c_i = n_pix w_i / W_pix
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        if z_depth_attr is not None:
            f.attrs["z_depth"] = float(z_depth_attr)
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return str(path), counts, zgals, dzgals, wgals


def _depth_opts(paths, **overrides):
    base = dict(
        universe_model="dark_sirens",
        survey_paths=list(paths),
        lss_completions=[],
        mark_model="none",
        mark_names_by_catalog=None,
        use_LSS=False,
        lss_marginalize=False,
        catalog_sky_weighting="field",
        validate_completion=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("via", ["cli_override", "file_attr"])
def test_multitracer_bundles_carry_field_depth_inputs(tmp_path, via):
    """FIELD weighting + a survey depth: each bundle must carry the flat FULL-SKY
    depth inputs, from EITHER resolution source (--survey_z_depth or the
    per-catalog f.attrs['z_depth']).  Without them the survey-global normalizer
    would fall back to the raw N_obs_total and double count every above-depth
    catalogued galaxy (the numerator scales N_obs by the below-depth mass), which
    the likelihood now rejects loudly -- so the loader must build them where the
    full-sky rows still exist, before compaction drops them."""
    attr = 0.3 if via == "file_attr" else None
    path, counts, zgals, _dz, wgals = _write_depth_survey(
        tmp_path / "survey_depth.hdf5", z_depth_attr=attr
    )
    opts = _depth_opts([path], survey_z_depth=(0.3 if via == "cli_override" else None))
    ra = np.zeros(2)
    gw_inputs = dict(ra=ra, dec=ra.copy(), rasels=ra.copy(), decsels=ra.copy())

    bundle = data_module.loaders.load_multitracer_catalog_bundles(opts, gw_inputs)[0]
    n_gal = int(counts.sum())
    for key in ("field_depth_z", "field_depth_dz", "field_depth_c"):
        assert bundle.get(key) is not None, key
        assert np.asarray(bundle[key]).shape == (n_gal,), key
    # Row-major over occupied pixels, real slots only.
    np.testing.assert_allclose(
        np.asarray(bundle["field_depth_z"]), [0.10, 0.70] * 3, rtol=1e-12
    )
    # c_i = N_obs,pix * w_i / W_pix  ->  2 * [1, 3] / 4 = [0.5, 1.5]
    np.testing.assert_allclose(
        np.asarray(bundle["field_depth_c"]), [0.5, 1.5] * 3, rtol=1e-12
    )


def test_multitracer_bundles_omit_field_depth_inputs_without_a_depth(tmp_path):
    """No depth in force -> the (N_gal_total,) f64 flat arrays are NOT built: they
    are read only by the depth convention, and this is the memory contract that
    keeps them off the device for every other run."""
    path, *_ = _write_depth_survey(tmp_path / "survey_nodepth.hdf5")
    opts = _depth_opts([path], survey_z_depth=None)
    ra = np.zeros(2)
    gw_inputs = dict(ra=ra, dec=ra.copy(), rasels=ra.copy(), decsels=ra.copy())
    bundle = data_module.loaders.load_multitracer_catalog_bundles(opts, gw_inputs)[0]
    assert bundle.get("field_dN_obs_s") is not None      # field inputs ARE built
    for key in ("field_depth_z", "field_depth_dz", "field_depth_c"):
        assert key not in bundle, key
