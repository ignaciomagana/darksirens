"""Dead per-view compact catalogs on the flat single-catalog paths.

On the DEFAULT flat path (drop_full_catalog=False) the loader must NOT build
per-view compact galaxy tables: ``prepare_catalog_views``'s union branch
gathers the single PE-union-selection table from the retained full-sky rows at
factory time and rebinds both views to it, so per-view tables built at load
time sat in ``data`` as dead device memory for the whole run (~2-3 GB at the
DESI reference shape), with the factory barriering discarded copies of them on
top.  Regressions pinned here:

  * the loader keeps only the small host-side row bookkeeping (unique pixels,
    sample->row maps, per-row counts) and leaves every ``zgals_pe``-style
    galaxy table None;
  * ``prepare_catalog_views`` does not build (``_ensure_compact``) or barrier
    per-view compacts when the union branch will fire, and its union result is
    unchanged: one aliased table gathered from the full rows;
  * with --drop_full_catalog the loader compacts ONCE over the pixel union so
    both views share ONE table (mirroring the K>=2 bundle loader), which
    ``prepare_catalog_views`` detects by identity to alias the device arrays
    and build a single KDE cache.
"""
from types import SimpleNamespace
import sys
import types

# ``darksirens.gw.utils`` imports tqdm at module import time, but these loader
# tests monkeypatch the GW file readers and do not need tqdm itself.
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

# The population registry imports optional GP model classes; keep this file
# independent of the optional tinygp dependency (mirrors
# tests/test_dark_sirens_startup_likelihood.py).
if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub

import h5py
import healpy as hp
import numpy as np
import pytest

from darksirens.inference import data as data_module
import darksirens.likelihood.catalog_views as catalog_views_module
from darksirens.likelihood.catalog_views import prepare_catalog_views

NSIDE = 1
PE_PIXELS = np.array([5, 2], dtype=np.int32)
SEL_PIXELS = np.array([7, 10], dtype=np.int32)
UNION_PIXELS = np.array([2, 5, 7, 10], dtype=np.int32)
S2U_PE = np.array([1, 0], dtype=np.int32)     # searchsorted(union, [5, 2])
S2U_SEL = np.array([2, 3], dtype=np.int32)    # searchsorted(union, [7, 10])


def _angles_for_pixels(nside, pixels):
    theta, phi = hp.pix2ang(nside, np.asarray(pixels, dtype=np.int64))
    return phi, np.pi / 2.0 - theta


@pytest.fixture
def survey_fixture(tmp_path):
    npix = hp.nside2npix(NSIDE)
    counts = np.zeros(npix, dtype=np.int32)
    counts[2] = 3
    counts[5] = 1
    counts[7] = 2
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
        f.attrs["nside"] = NSIDE
        f.create_dataset("ngals", data=counts)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)

    return path, counts, zgals, dzgals, wgals


def _patch_gw_loaders(monkeypatch):
    pe_ra, pe_dec = _angles_for_pixels(NSIDE, PE_PIXELS)
    sel_ra, sel_dec = _angles_for_pixels(NSIDE, SEL_PIXELS)

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


def _opts(survey_path, **overrides):
    base = dict(
        universe_model="dark_sirens",
        survey_path=str(survey_path),
        gw_path="unused-gw.hdf5",
        gwselection_path="unused-selection.hdf5",
        use_LSS=False,
        counterpart=None,
        counterpart_nside=1,
        counterpart_dz=0.01,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _view_opts():
    return SimpleNamespace(
        catalog_sky_weighting="conditional",
        mark_model="none",
    )


def test_retained_catalog_builds_no_per_view_galaxy_tables(
    survey_fixture, monkeypatch
):
    """Default path: zero per-view galaxy tables in ``data``, bookkeeping kept."""
    survey_path, counts, *_ = survey_fixture
    _patch_gw_loaders(monkeypatch)

    loaded = data_module.load_all_data(_opts(survey_path))

    # The superseded per-view galaxy tables must not exist at all.
    for key in (
        "zgals_pe", "dzgals_pe", "wgals_pe",
        "zgals_sel", "dzgals_sel", "wgals_sel",
    ):
        assert loaded[key] is None, f"{key} should be None on the retained path"

    # The full-sky rows stay (the factory's union branch gathers from them)...
    assert loaded["zgals_catalog"] is not None
    # ...and the small host-side bookkeeping keeps its per-view semantics.
    np.testing.assert_array_equal(loaded["unique_pixels_pe"], np.array([2, 5]))
    np.testing.assert_array_equal(loaded["unique_pixels_sel"], np.array([7, 10]))
    np.testing.assert_array_equal(loaded["ngals_pe"], counts[[2, 5]])
    np.testing.assert_array_equal(loaded["ngals_sel"], counts[[7, 10]])
    assert loaded["sample_to_unique_pe"].shape[0] == loaded["pixels_pe"].shape[0]
    assert loaded["sample_to_unique_sel"].shape[0] == loaded["pixels_sel"].shape[0]
    assert loaded["catalog_memory"]["max_galaxies_per_unique_pixel"] == 3


def test_union_branch_result_is_unchanged_and_aliased(survey_fixture, monkeypatch):
    """The factory's union branch still yields ONE aliased full[union] table."""
    survey_path, counts, *_ = survey_fixture
    _patch_gw_loaders(monkeypatch)
    loaded = data_module.load_all_data(_opts(survey_path))

    views = prepare_catalog_views(_view_opts(), loaded, "dark_sirens", None)

    full_z = np.asarray(loaded["zgals_catalog"])
    full_dz = np.asarray(loaded["dzgals_catalog"])
    full_w = np.asarray(loaded["wgals_catalog"])
    np.testing.assert_array_equal(
        np.asarray(views.unique_pixels_pe), UNION_PIXELS
    )
    np.testing.assert_array_equal(
        np.asarray(views.zgals_pe_catalog), full_z[UNION_PIXELS]
    )
    np.testing.assert_array_equal(
        np.asarray(views.dzgals_pe_catalog), full_dz[UNION_PIXELS]
    )
    np.testing.assert_array_equal(
        np.asarray(views.wgals_pe_catalog), full_w[UNION_PIXELS]
    )
    np.testing.assert_array_equal(
        np.asarray(views.ngals_pe_catalog), counts[UNION_PIXELS]
    )
    np.testing.assert_array_equal(np.asarray(views.sample_to_unique_pe), S2U_PE)
    np.testing.assert_array_equal(np.asarray(views.sample_to_unique_sel), S2U_SEL)
    # One table, aliased between the views (identity, never a value copy).
    assert views.zgals_pe_catalog is views.zgals_sel_catalog
    assert views.dzgals_pe_catalog is views.dzgals_sel_catalog
    assert views.wgals_pe_catalog is views.wgals_sel_catalog
    assert views.ngals_pe_catalog is views.ngals_sel_catalog
    assert views.unique_pixels_pe is views.unique_pixels_sel
    assert views.dN_obs_kde_pe is views.dN_obs_kde_sel


def test_union_path_never_builds_per_view_compacts(survey_fixture, monkeypatch):
    """With full rows + both pixel maps present, ``_ensure_compact`` (and the
    per-view barriers behind it) must never run: everything it would produce is
    superseded by the union rebind, so each call is a discarded device copy."""
    survey_path, *_ = survey_fixture
    _patch_gw_loaders(monkeypatch)
    loaded = data_module.load_all_data(_opts(survey_path))

    calls = []
    real_ensure_compact = catalog_views_module._ensure_compact

    def spy_ensure_compact(data, prefix, pixels_key):
        calls.append(prefix)
        return real_ensure_compact(data, prefix, pixels_key)

    monkeypatch.setattr(catalog_views_module, "_ensure_compact", spy_ensure_compact)

    views = prepare_catalog_views(_view_opts(), loaded, "dark_sirens", None)

    assert calls == [], (
        "prepare_catalog_views built per-view compact catalogs on the union "
        f"path (calls for views: {calls}); they are superseded by the union "
        "table and every one is a dead device copy."
    )
    assert views.zgals_pe_catalog is views.zgals_sel_catalog


def test_drop_full_catalog_shares_one_union_table(survey_fixture, monkeypatch):
    """--drop_full_catalog: ONE union-compacted table shared by both views,
    detected by identity downstream (aliasing + a single KDE cache)."""
    survey_path, counts, zgals, dzgals, wgals = survey_fixture
    _patch_gw_loaders(monkeypatch)

    loaded = data_module.load_all_data(
        _opts(survey_path, drop_full_catalog=True)
    )

    # Full-sky rows are gone; the compact views are the only catalog state.
    assert loaded["zgals_catalog"] is None
    # PE and selection views are the SAME objects over the pixel union.
    assert loaded["zgals_pe"] is loaded["zgals_sel"]
    assert loaded["dzgals_pe"] is loaded["dzgals_sel"]
    assert loaded["wgals_pe"] is loaded["wgals_sel"]
    assert loaded["ngals_pe"] is loaded["ngals_sel"]
    assert loaded["unique_pixels_pe"] is loaded["unique_pixels_sel"]
    np.testing.assert_array_equal(loaded["unique_pixels_pe"], UNION_PIXELS)
    np.testing.assert_array_equal(loaded["zgals_pe"], zgals[UNION_PIXELS])
    np.testing.assert_array_equal(loaded["ngals_pe"], counts[UNION_PIXELS])
    np.testing.assert_array_equal(loaded["sample_to_unique_pe"], S2U_PE)
    np.testing.assert_array_equal(loaded["sample_to_unique_sel"], S2U_SEL)

    views = prepare_catalog_views(_view_opts(), loaded, "dark_sirens", None)
    np.testing.assert_array_equal(
        np.asarray(views.zgals_pe_catalog), zgals[UNION_PIXELS]
    )
    np.testing.assert_array_equal(
        np.asarray(views.dzgals_pe_catalog), dzgals[UNION_PIXELS]
    )
    np.testing.assert_array_equal(
        np.asarray(views.wgals_pe_catalog), wgals[UNION_PIXELS]
    )
    np.testing.assert_array_equal(np.asarray(views.sample_to_unique_pe), S2U_PE)
    np.testing.assert_array_equal(np.asarray(views.sample_to_unique_sel), S2U_SEL)
    assert views.zgals_pe_catalog is views.zgals_sel_catalog
    assert views.unique_pixels_pe is views.unique_pixels_sel
    assert views.dN_obs_kde_pe is views.dN_obs_kde_sel


def test_drop_full_catalog_matches_retained_likelihood(survey_fixture, monkeypatch):
    """--drop_full_catalog is a memory optimization, never a value change: the
    union-compacted drop path and the retained full-catalog (factory union)
    path evaluate the identical likelihood on the same inputs."""
    import jax.numpy as jnp

    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    survey_path, *_ = survey_fixture
    _patch_gw_loaders(monkeypatch)

    def _value(drop):
        opts = _opts(
            survey_path,
            drop_full_catalog=drop,
            pop_model="powerlaw+peak",
            sel_batch_size=None,
            fix_cosmology=True,
            fix_population=True,
            fix_survey=True,
        )
        data = data_module.load_all_data(opts)
        ll = make_likelihood(
            opts, data, get_fixed_population_params("powerlaw+peak")
        )
        return float(ll(jnp.array([])))

    v_retained = _value(False)
    v_dropped = _value(True)
    assert np.isfinite(v_retained)
    np.testing.assert_allclose(v_dropped, v_retained, rtol=1e-12)
