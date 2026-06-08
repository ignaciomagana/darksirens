from types import SimpleNamespace
import sys
import types

import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

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
from darksirens.inference.prior import build_parameter_space
from darksirens.inference.pop_extractor import make_pop_extractor
import darksirens.inference.likelihood as likelihood_module


def _opts():
    return SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=False,
        fix_population=False,
        fix_survey=False,
    )


def _small_data():
    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    nsamp = 2
    n_sel = 4

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)
    pixels_pe = jnp.array([7, 7], dtype=jnp.int32)
    pixels_sel = jnp.array([2, 7, 2, 2], dtype=jnp.int32)

    return {
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
        "pixels_pe": pixels_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": pixels_sel,
    }


@pytest.mark.parametrize(
    "fixed_values",
    [
        {"H0": 67.74},
        {"Om0": 0.3075},
        {"w0": -0.9},
        {"wa": 0.1},
        {"$\\alpha$": 1.5},
        {"z50": 1.2},
    ],
)
def test_fixed_parameters_are_removed_from_sampler_coordinates_and_likelihood(monkeypatch, fixed_values):
    opts = _opts()
    labels, lower, upper, *_ = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        fixed_parameter_values=fixed_values,
    )
    assert len(labels) == len(lower) == len(upper)
    assert all(label not in labels for label in fixed_values)

    def fake_log_likelihood(
        cosmo,
        survey,
        pop_params,
        gw_pe,
        em_catalog_pe,
        gw_sel,
        em_catalog_sel,
        nEvents,
        nsamp,
        Ndraw,
        pop_model,
        universe_model,
        sel_batch_size=None,
    ):
        del gw_pe, em_catalog_pe, gw_sel, em_catalog_sel
        del nEvents, nsamp, Ndraw, pop_model, universe_model, sel_batch_size
        return jnp.asarray(
            cosmo.H0 + cosmo.Om0 + cosmo.w0 + cosmo.wa + survey.z50 + jnp.sum(pop_params)
        )

    monkeypatch.setattr(likelihood_module, "darksiren_log_likelihood", fake_log_likelihood)
    likelihood = likelihood_module.make_likelihood(
        opts,
        _small_data(),
        get_fixed_population_params(opts.pop_model),
        fixed_parameter_values=fixed_values,
    )

    theta = jnp.linspace(0.2, 0.8, len(labels))
    value = likelihood(theta)
    assert value.shape == ()
    assert bool(jnp.isfinite(value))


def test_pop_extractor_accepts_sampled_coordinate_length_with_fixed_population_parameter():
    fixed_values = {"$\\alpha$": 1.5}
    settings = {
        "pop_model": "powerlaw+peak",
        "fix_cosmology": False,
        "fix_population": False,
        "fix_survey": False,
        "fixed_parameter_values": fixed_values,
    }
    res = build_parameter_space(
        settings["pop_model"],
        settings["fix_population"],
        settings["fix_cosmology"],
        settings["fix_survey"],
        fixed_parameter_values=fixed_values,
    )
    labels = res[0]
    pop_labels = res[4]
    theta = jnp.arange(len(labels), dtype=jnp.float64)
    pop_theta = make_pop_extractor(settings)(theta)

    assert pop_theta.shape == (len(pop_labels),)
    assert float(pop_theta[pop_labels.index("$\\alpha$")]) == fixed_values["$\\alpha$"]


def _sampled_labels(*, fix_cosmology=False, fix_de=False, fixed_parameter_values=None):
    labels, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=fix_cosmology,
        fix_survey=True,
        fix_de=fix_de,
        fixed_parameter_values=fixed_parameter_values,
    )
    return labels


def test_default_sampled_cosmology_labels_include_cpl_dark_energy():
    labels = _sampled_labels()

    assert labels == ["H0", "Om0", "w0", "wa"]


def test_fixed_cosmology_removes_all_cosmology_labels():
    labels = _sampled_labels(fix_cosmology=True)

    for label in ("H0", "Om0", "w0", "wa"):
        assert label not in labels


def test_fixed_dark_energy_removes_only_cpl_dark_energy_labels():
    labels = _sampled_labels(fix_de=True)

    assert "H0" in labels
    assert "Om0" in labels
    assert "w0" not in labels
    assert "wa" not in labels


@pytest.mark.parametrize(
    "label,value",
    [
        ("H0", 67.74),
        ("Om0", 0.3075),
        ("w0", -1.0),
        ("wa", 0.0),
    ],
)
def test_individual_fixed_cosmology_values_remove_matching_label(label, value):
    labels = _sampled_labels(fixed_parameter_values={label: value})

    assert label not in labels
    assert all(other in labels for other in {"H0", "Om0", "w0", "wa"} - {label})
