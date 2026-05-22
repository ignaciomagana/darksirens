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
        ("log_n0", "z50", "w", "delta", "b_miss", "alpha_miss"),  # survey_labels
        (),
        0,
        0,
        "model",
        {},
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
    _cosmo, survey, _pop_params = decoder.decode(())

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
    _cosmo, survey, _pop_params = decoder.decode(())

    assert survey.wl_params is None
