"""Public import contract guardrails for the package refactor.

These tests intentionally cover the documented module paths after the compatibility
wrapper cleanup.
"""

import importlib


def test_cli_entry_imports():
    from darksirens.cli.inference import main as inference_main
    from darksirens.cli.analyze import main as analyze_main
    from darksirens.cli.pixelate import main as pixelate_main
    from darksirens.cli.skymaps_to_samples import main as skymaps_main
    from darksirens.cli.build_lognormal_completion import main as build_lognormal_main
    from darksirens.cli.diagnose_lognormal_completion import main as diagnose_lognormal_main
    from darksirens.cli.inference_lensing import main as inference_lensing_main

    assert callable(inference_main)
    assert callable(analyze_main)
    assert callable(pixelate_main)
    assert callable(skymaps_main)
    assert callable(build_lognormal_main)
    assert callable(diagnose_lognormal_main)
    assert callable(inference_lensing_main)


def test_core_inference_imports():
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.inference.parameters import H0_FID, OM0_FID
    from darksirens.inference.prior import build_parameter_space, make_prior_transform
    from darksirens.inference.sampling import run_sampler

    assert callable(load_all_data)
    assert callable(validate_loaded_survey_shapes)
    assert callable(make_likelihood)
    assert callable(build_parameter_space)
    assert callable(make_prior_transform)
    assert callable(run_sampler)
    assert isinstance(H0_FID, float)
    assert isinstance(OM0_FID, float)


def test_container_imports():
    from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent

    assert CosmoParams is not None
    assert SurveyParams is not None
    assert EMCatalog is not None
    assert GWEvent is not None


def test_population_imports():
    from darksirens.gw.populations import (
        get_model,
        pop_model_parser,
        pop_model_prior_parser,
        get_fixed_population_params,
    )

    assert callable(get_model)
    assert callable(pop_model_parser)
    assert callable(pop_model_prior_parser)
    assert callable(get_fixed_population_params)


def test_documented_stable_module_imports():
    documented_modules = [
        "darksirens.redshift",
        "darksirens.redshift.catalog",
        "darksirens.redshift.completion",
        "darksirens.redshift.lognormal_completion",
        "darksirens.redshift.prior",
        "darksirens.catalogs.io",
        "darksirens.redshift.volume",
        "darksirens.gw",
        "darksirens.gw.populations",
        "darksirens.gw.populations.base",
        "darksirens.gw.populations.components",
        "darksirens.gw.populations.grammar",
        "darksirens.gw.populations.parametric",
        "darksirens.gw.populations.gp",
        "darksirens.gw.populations.registry",
        "darksirens.gw.populations.utils",
        "darksirens.gw.selection",
        "darksirens.gw.utils",
        "darksirens.likelihood.catalog_views",
        "darksirens.likelihood.cluster_likelihood",
        "darksirens.likelihood.cluster_selection",
        "darksirens.likelihood.events",
        "darksirens.likelihood.core",
        "darksirens.likelihood.likelihood_with_clusters",
        "darksirens.likelihood.pair_kde",
        "darksirens.likelihood.selection",
        "darksirens.inference.utils",
        "darksirens.likelihood.wl_weight",
        "darksirens.lensing.grids",
        "darksirens.lensing.lensed_injections",
        "darksirens.lensing.slmarks",
        "darksirens.lensing.wlmagnification",
        "darksirens.marks",
        "darksirens.marks.models",
        "darksirens.marks.registry",
        "darksirens.sky",
        "darksirens.sky.analyze",
        "darksirens.sky.models",
        "darksirens.sky.registry",
        "darksirens.utils.cosmology",
        "darksirens.utils.utils",
    ]

    for module_name in documented_modules:
        assert importlib.import_module(module_name) is not None
