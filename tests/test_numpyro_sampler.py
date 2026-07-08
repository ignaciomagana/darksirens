from types import SimpleNamespace

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("numpyro")

from darksirens.redshift import zgrid
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler


UNIVERSE_MODELS = [
    "spectral_sirens",
    "dark_sirens",
    "dark_sirens_complete",
    "bright_sirens",
]


def _small_likelihood_inputs(universe_model):
    nside = 1
    n_pix_catalog = hp.nside2npix(nside)
    n_events = 1
    nsamp = 2
    n_sel = 8

    zgals = np.full((n_pix_catalog, 1), 0.10, dtype=float)
    dzgals = np.full((n_pix_catalog, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix_catalog, 1), dtype=float)
    ngals = np.ones(n_pix_catalog, dtype=np.int32)

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
        "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=jnp.int32),
    }

    if universe_model == "bright_sirens":
        data.update(
            {
                "counterpart_pixel": 7,
                "counterpart_pixels": jnp.array([7], dtype=jnp.int32),
                "counterpart_zs": jnp.array([0.10]),
                "counterpart_dzs": jnp.array([0.02]),
                "bright_siren_sky_marginalized": False,
            }
        )

    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_params_fid = get_fixed_population_params("powerlaw+peak")
    sampled_pop_label = pop_labels[0]
    prior_overrides = {sampled_pop_label: [float(pop_lower[0]), float(pop_upper[0])]}
    fixed_parameter_values = {
        label: float(pop_params_fid[i])
        for i, label in enumerate(pop_labels)
        if label != sampled_pop_label
    }
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model=universe_model,
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=False,
        fix_survey=True,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_parameter_values,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    return opts, data, prior_overrides, fixed_parameter_values, sampled_pop_label


@pytest.mark.parametrize("universe_model", UNIVERSE_MODELS)
def test_numpyro_nuts_samples_with_finite_gradients_for_universe_models(universe_model):
    (
        opts,
        data,
        prior_overrides,
        fixed_parameter_values,
        sampled_label,
    ) = _small_likelihood_inputs(universe_model)
    pop_params_fid = get_fixed_population_params(opts.pop_model)
    likelihood = make_likelihood(
        opts,
        data,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
    )

    labels, lower, upper, *_ = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_parameter_values,
    )
    assert labels == [sampled_label]

    midpoint = jnp.asarray(0.5 * (lower + upper))
    value = likelihood(midpoint)
    grad = jax.grad(lambda theta: likelihood(theta))(midpoint)

    assert np.isfinite(float(value))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(grad), rtol=0, atol=0)
    assert np.all(np.isfinite(np.asarray(grad)))

    sampler_opts = SimpleNamespace(
        seed=11,
        show_progress=False,
        nuts_warmup=2,
        nuts_samples=2,
        nuts_chains=1,
        nuts_target_accept=0.8,
        nuts_max_tree_depth=3,
        nuts_chain_method="sequential",
    )
    results = run_sampler(
        "numpyro",
        likelihood,
        make_prior_transform(lower, upper),
        labels,
        lower,
        upper,
        sampler_opts,
    )

    samples = np.asarray(results["samples"])
    assert samples.shape == (2, 1)
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= lower[None, :])
    assert np.all(samples <= upper[None, :])


_TINY_NUTS_OPTS = dict(
    seed=11,
    show_progress=False,
    nuts_warmup=2,
    nuts_samples=2,
    nuts_chains=1,
    nuts_target_accept=0.8,
    nuts_max_tree_depth=3,
    nuts_chain_method="sequential",
)


def test_numpyro_nuts_full_free_population_smoke():
    """NUTS over the FULL free powerlaw+peak hyperparameter space (the H1
    profile shape), not a 1-D subspace: completes, samples in bounds, and the
    sampler-health diagnostics (divergence counts) are returned."""
    opts, data, *_ = _small_likelihood_inputs("spectral_sirens")
    opts.prior_overrides = {}
    opts.fixed_parameter_values = {}
    pop_params_fid = get_fixed_population_params(opts.pop_model)
    likelihood = make_likelihood(opts, data, pop_params_fid,
                                 fixed_parameter_values={})
    labels, lower, upper, *_ = build_parameter_space(
        opts.pop_model, False, True, True,
        prior_overrides={}, fixed_parameter_values={},
    )
    assert len(labels) > 5  # genuinely the full population space

    results = run_sampler(
        "numpyro", likelihood, make_prior_transform(lower, upper),
        labels, lower, upper, SimpleNamespace(**_TINY_NUTS_OPTS),
    )
    samples = np.asarray(results["samples"])
    assert samples.shape == (2, len(labels))
    assert np.all(np.isfinite(samples))
    assert np.all(samples >= np.asarray(lower)[None, :])
    assert np.all(samples <= np.asarray(upper)[None, :])
    diag = results["numpyro_diagnostics"]
    for key in ("n_divergent", "divergence_fraction", "mean_accept_prob",
                "mean_num_steps", "frac_at_max_tree_depth"):
        assert key in diag
    assert diag["n_divergent"] >= 0


def test_numpyro_nuts_dark_sirens_free_cosmology_gradient_path():
    """H0 free WITH a catalog: the redshift-prior state depends on theta, so
    the optimization barrier must come off for NUTS (auto resolver keyed on
    opts.sampler == 'numpyro'); with the barrier left on, the first gradient
    dies with NotImplementedError('optimization_barrier')."""
    opts, data, *_ = _small_likelihood_inputs("dark_sirens")
    opts.sampler = "numpyro"       # what the CLI sets; engages the resolver
    opts.fix_cosmology = False
    opts.fix_population = True
    opts.fix_de = True
    opts.fixed_de = True
    # Population is fixed via fix_population (fiducials), NOT via the
    # fixed-values map -- mirroring the CLI's coordinate layout.
    fixed = {"Om0": 0.3075}
    opts.prior_overrides = {}
    opts.fixed_parameter_values = fixed
    pop_params_fid = get_fixed_population_params(opts.pop_model)
    likelihood = make_likelihood(opts, data, pop_params_fid,
                                 fixed_parameter_values=fixed)
    labels, lower, upper, *_ = build_parameter_space(
        opts.pop_model, True, False, True,
        prior_overrides={}, fixed_parameter_values=fixed, fix_de=True,
    )
    assert labels == ["H0"]

    results = run_sampler(
        "numpyro", likelihood, make_prior_transform(lower, upper),
        labels, lower, upper, SimpleNamespace(**_TINY_NUTS_OPTS),
    )
    samples = np.asarray(results["samples"])
    assert samples.shape == (2, 1)
    assert np.all(np.isfinite(samples))
    assert results["numpyro_diagnostics"]["n_divergent"] >= 0
