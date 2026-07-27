from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from darksirens.inference import parameters as parameters_module


def _stub_build_parameter_space(*args, **kwargs):
    del args, kwargs
    return (
        (),  # sampled_labels
        (),
        (),
        0,
        (),  # pop_labels
        # survey_labels: the sampleable survey block (the survey registry's
        # labels; the decoder fills the rest of SurveyParams from fiducials)
        ("log10n0", "delta", "b_miss", "sigma_kde"),
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
    sample any survey parameter (library review, CLI finding 1: the missing
    universe-model entry left 7 phantom flat dimensions in the space, inflating
    12 -> 19 dims and corrupting Bayes factors against spectral_sirens)."""
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


# ---------------------------------------------------------------------------
# decode_mixture (K-catalog multitracer mixture): ParameterDecoder.n_catalogs
# and the module-level _sticks_to_log_weights are exercised via a decoder
# built directly (no build_parameter_space stubbing needed -- decode_mixture
# only reads self.sampled_labels / self.n_catalogs; the SurveyParams fields are
# addressed by name against the shared fiducial table).
# ---------------------------------------------------------------------------

def _decoder_k2(sampled_labels, fixed_parameter_values=None):
    return parameters_module.ParameterDecoder(
        sampled_labels=tuple(sampled_labels),
        fixed_parameter_values=fixed_parameter_values or {},
        pop_labels=(),
        pop_params_fid=(),
        complete_empty_pixel_policy=0,
        n_catalogs=2,
    )


def test_decode_mixture_returns_k_surveys_and_normalized_weights():
    decoder = _decoder_k2(sampled_labels=("log10n0", "log10n0_c2", "fcat_2"))
    coord = jnp.array([-2.0, -1.5, 0.3])

    cosmo, surveys, pop_params, sky_params, mark_params, log_w = decoder.decode_mixture(coord)

    assert len(surveys) == 2
    assert float(surveys[1].n0) == pytest.approx(10.0 ** -1.5)
    assert surveys[1].wl_params is None

    w = jnp.exp(log_w)
    assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-10)
    assert float(w[1]) == pytest.approx(0.3, abs=1e-10)


def test_decode_mixture_catalog1_matches_plain_decode():
    """Catalog 1 (cosmo/pop/sky/mark too) must be bit-identical between
    decode() and the first mixture component of decode_mixture()."""
    decoder = _decoder_k2(sampled_labels=("log10n0", "log10n0_c2", "fcat_2"))
    coord = jnp.array([-2.0, -1.5, 0.3])

    cosmo1, survey1, pop1, sky1, mark1 = decoder.decode(coord)
    cosmo2, surveys, pop2, sky2, mark2, _log_w = decoder.decode_mixture(coord)

    assert survey1.n0 == surveys[0].n0
    assert survey1.delta == surveys[0].delta
    assert cosmo1.H0 == cosmo2.H0
