"""Import guardrails for the staged package-layout refactor.

These tests document the intended migration path: new production code should use
``darksirens.core``, ``darksirens.cli``, ``darksirens.likelihood``, and
``darksirens.redshift`` while the legacy wrapper modules continue to import until
an explicit wrapper-removal step.
"""


def test_new_core_imports():
    from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
    from darksirens.core.constants import H0_FID, OM0_FID

    assert CosmoParams is not None
    assert SurveyParams is not None
    assert EMCatalog is not None
    assert GWEvent is not None
    assert isinstance(H0_FID, float)
    assert isinstance(OM0_FID, float)


def test_new_cli_imports():
    from darksirens.cli.inference import main

    assert callable(main)


def test_new_likelihood_imports():
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.likelihood.core import darksiren_log_likelihood

    assert callable(make_likelihood)
    assert callable(darksiren_log_likelihood)


def test_new_redshift_imports():
    from darksirens.redshift.completion import build_pixel_kde_cache

    assert callable(build_pixel_kde_cache)


def test_compatibility_imports_still_work():
    from darksirens.tool.darksirens_inference import main as old_main
    from darksirens.utils.containers import GWEvent
    from darksirens.em.completion import build_pixel_kde_cache
    from darksirens.inference.likelihood import make_likelihood

    assert callable(old_main)
    assert GWEvent is not None
    assert callable(build_pixel_kde_cache)
    assert callable(make_likelihood)
