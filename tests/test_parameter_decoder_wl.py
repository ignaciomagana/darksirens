from types import SimpleNamespace

from darksirens.inference import parameters as parameters_module


def _stub_build_parameter_space(*args, **kwargs):
    del args, kwargs
    return (
        (),  # sampled_labels
        (),
        (),
        0,
        (),  # pop_labels
        # survey_labels (master's 7-parameter block, incl. sigma_kde)
        ("log10n0", "z50", "w", "delta", "b_miss", "alpha_miss", "sigma_kde"),
        (),
        0,
        0,
        "model",
        {},   # fixed_parameter_statuses
        [],   # prior_kinds
        (),   # sky_labels
        (),   # mark_labels
    )


def test_decode_includes_wl_params_for_spectral_sirens_wl(monkeypatch):
    monkeypatch.setattr(parameters_module, "build_parameter_space", _stub_build_parameter_space)

    opts = SimpleNamespace(
        pop_model="BBH-powerlaw",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
        universe_model="spectral_sirens_wl",
    )
    wl_params = object()

    decoder = parameters_module.build_parameter_decoder(opts, pop_params_fid=(), wl_params=wl_params)
    _cosmo, survey, _pop_params, _sky, _mark = decoder.decode(())

    assert survey.wl_params is wl_params


def test_decode_has_none_wl_params_for_non_wl_universe_model(monkeypatch):
    monkeypatch.setattr(parameters_module, "build_parameter_space", _stub_build_parameter_space)

    opts = SimpleNamespace(
        pop_model="BBH-powerlaw",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
        universe_model="spectral_sirens",
    )

    decoder = parameters_module.build_parameter_decoder(
        opts,
        pop_params_fid=(),
        wl_params=object(),
    )
    _cosmo, survey, _pop_params, _sky, _mark = decoder.decode(())

    assert survey.wl_params is None


def test_wl_universe_model_samples_no_survey_block():
    """spectral_sirens_wl is catalog-free: with fix_survey=False it must NOT
    sample the 7-parameter survey block (library review, CLI finding 1: the
    missing _ACTIVE_SURVEY_PARAMS entry left 7 phantom flat dimensions in the
    space, inflating 12 -> 19 dims and corrupting Bayes factors against
    spectral_sirens)."""
    from darksirens.inference.prior import build_parameter_space

    labels_wl, *_ = build_parameter_space(
        "powerlaw+peak", False, True, False,
        prior_overrides={}, fixed_parameter_values={},
        universe_model="spectral_sirens_wl",
    )
    labels_spec, *_ = build_parameter_space(
        "powerlaw+peak", False, True, False,
        prior_overrides={}, fixed_parameter_values={},
        universe_model="spectral_sirens",
    )
    survey_like = {"log10n0", "z50", "w", "delta", "b_miss", "alpha_miss", "sigma_kde"}
    assert not (set(map(str, labels_wl)) & survey_like)
    assert list(map(str, labels_wl)) == list(map(str, labels_spec))
