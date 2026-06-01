import copy
from types import SimpleNamespace
import sys
import types

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np

# The lightweight likelihood startup fixture uses a parametric population model,
# but importing the registry also imports optional GP model classes.  Keep this
# regression test independent of the optional tinygp dependency.
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


from darksirens.em import zgrid
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.likelihood import make_likelihood


def test_dark_sirens_likelihood_evaluates_once_before_sampling():
    """Small startup regression: one dark-sirens likelihood call must not fail."""
    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    n_events = 1
    nsamp = 2
    n_sel = 16

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)

    pixels_pe = jnp.array([7, 7], dtype=jnp.int32)
    pixels_sel = jnp.array([2, 7, *([2] * (n_sel - 2))], dtype=jnp.int32)

    data = {
        "nEvents": n_events,
        "nsamp": nsamp,
        "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside),
        "nside": nside,
        "n_pix_catalog": n_pix_catalog,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((n_pix_catalog, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0]),
        "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]),
        "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp),
        "pixels_pe": pixels_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": pixels_sel,
    }
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    likelihood = make_likelihood(
        opts,
        data,
        get_fixed_population_params(opts.pop_model),
    )

    value = likelihood(jnp.array([]))
    assert value.shape == ()
    assert not bool(jnp.isnan(value))
    # Reference value from the pre-refactor monolithic likelihood implementation.
    np.testing.assert_allclose(float(value), 0.016347464281930607, rtol=1e-12)


def test_make_likelihood_does_not_mutate_data_when_compacting_catalogs():
    """Regression: synthetic compact catalog views stay local to make_likelihood."""
    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    nsamp = 2
    n_sel = 4

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)

    data = {
        "nEvents": 1,
        "nsamp": nsamp,
        "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside),
        "nside": nside,
        "n_pix_catalog": n_pix_catalog,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((n_pix_catalog, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0]),
        "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]),
        "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp),
        "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.array([2, 7, 2, 2], dtype=jnp.int32),
    }
    data_before = copy.deepcopy(data)
    keys_before = set(data)
    ids_before = {key: id(value) for key, value in data.items()}

    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    make_likelihood(
        opts,
        data,
        get_fixed_population_params(opts.pop_model),
    )

    assert set(data) == keys_before
    assert {key: id(value) for key, value in data.items()} == ids_before
    for key, before_value in data_before.items():
        after_value = data[key]
        if hasattr(before_value, "shape"):
            np.testing.assert_array_equal(
                np.asarray(after_value), np.asarray(before_value)
            )
        else:
            assert after_value == before_value


def test_dark_sirens_cache_is_built_once_for_unique_pixels(monkeypatch):
    """Regression: likelihood evaluation uses the prebuilt unique-pixel cache."""
    import darksirens.em.completion as completion
    import darksirens.inference.likelihood as likelihood_module

    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    n_events = 1
    nsamp = 4
    n_sel = 12

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)

    pixels_pe = jnp.array([7, 7, 3, 3], dtype=jnp.int32)
    pixels_sel = jnp.array([2, 7, 2, 5, *([5] * (n_sel - 4))], dtype=jnp.int32)

    data = {
        "nEvents": n_events,
        "nsamp": nsamp,
        "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside),
        "nside": nside,
        "n_pix_catalog": n_pix_catalog,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((n_pix_catalog, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0, 35.0, 37.0]),
        "m2det": jnp.array([28.8, 30.4, 28.0, 29.6]),
        "dL": jnp.array([460.0, 500.0, 470.0, 490.0]),
        "chieff": jnp.array([0.0, 0.02, 0.01, -0.01]),
        "p_pe": jnp.ones(nsamp),
        "pixels_pe": pixels_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": pixels_sel,
    }
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    cache_calls = []

    def fake_build_pixel_kde_cache(
        unique_pixels, zgals, n_pix_catalog, wgals=None, ngals=None
    ):
        unique_pixels = np.asarray(unique_pixels, dtype=np.int32)
        cache_calls.append(unique_pixels.copy())
        pixel_to_cache_idx = np.zeros(n_pix_catalog, dtype=np.int32)
        for idx, pix in enumerate(unique_pixels):
            pixel_to_cache_idx[int(pix)] = idx
        return (
            jnp.zeros((unique_pixels.size, len(zgrid))),
            jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32),
        )

    monkeypatch.setattr(
        likelihood_module, "build_pixel_kde_cache", fake_build_pixel_kde_cache
    )

    likelihood = likelihood_module.make_likelihood(
        opts,
        data,
        get_fixed_population_params(opts.pop_model),
    )

    def fail_if_uncached_path_is_used(*args, **kwargs):
        raise AssertionError(
            "_kde_dndz_obs should not be called during likelihood evaluation"
        )

    monkeypatch.setattr(completion, "_kde_dndz_obs", fail_if_uncached_path_is_used)

    value = likelihood(jnp.array([]))
    assert value.shape == ()
    assert len(cache_calls) == 1
    np.testing.assert_array_equal(
        cache_calls[0], np.array([2, 3, 5, 7], dtype=np.int32)
    )
