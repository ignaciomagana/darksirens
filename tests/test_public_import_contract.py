"""Public import contract guardrails for the package refactor.

These tests intentionally cover the currently documented/imported module paths so
future structure-only moves can keep compatibility wrappers in place until the
explicit wrapper-removal step.
"""

import importlib


def test_cli_entry_imports():
    from darksirens.tool.darksirens_inference import main as inference_main
    from darksirens.tool.darksirens_analyze import main as analyze_main
    from darksirens.tool.darksirens_pixelate import main as pixelate_main
    from darksirens.tool.darksirens_skymaps_to_samples import main as skymaps_main
    from darksirens.tool.darksirens_build_lognormal_completion import main as build_lognormal_main
    from darksirens.tool.darksirens_diagnose_lognormal_completion import main as diagnose_lognormal_main
    from darksirens.tool.darksirens_inference_lensing import main as inference_lensing_main

    assert callable(inference_main)
    assert callable(analyze_main)
    assert callable(pixelate_main)
    assert callable(skymaps_main)
    assert callable(build_lognormal_main)
    assert callable(diagnose_lognormal_main)
    assert callable(inference_lensing_main)


def test_core_inference_imports():
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.inference.likelihood import make_likelihood
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
    from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent

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
        "darksirens.em",
        "darksirens.em.catalog",
        "darksirens.em.completion",
        "darksirens.em.lognormal_completion",
        "darksirens.em.prior",
        "darksirens.em.utils",
        "darksirens.em.volume",
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
        "darksirens.inference.catalog_views",
        "darksirens.inference.cluster_likelihood",
        "darksirens.inference.cluster_selection",
        "darksirens.inference.events",
        "darksirens.inference.likelihood_core",
        "darksirens.inference.likelihood_with_clusters",
        "darksirens.inference.pair_kde",
        "darksirens.inference.selection",
        "darksirens.inference.utils",
        "darksirens.inference.wl_weight",
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
