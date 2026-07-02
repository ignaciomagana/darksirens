from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.cli import inference_lensing
from darksirens.core.types import CosmoParams, SurveyParams
from darksirens.utils.cosmology import H0Planck, Om0Planck


def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(
        n0=1e-3,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
    )


def test_lensing_cli_split_fixed_parameters_keeps_tau_n_out_of_base_space():
    """Lens-only fixed values must not reach the shared base parameter space."""
    with pytest.raises(KeyError, match="tau_n"):
        inference_lensing.build_parameter_space(
            "powerlaw+peak",
            True,
            True,
            True,
            fixed_parameter_values={"tau_n": 3.0},
        )

    base_fixed, lens_fixed = inference_lensing._split_lensing_fixed_parameters(
        {"tau_n": 3.0, "H0": 67.0}
    )
    assert base_fixed == {"H0": 67.0}
    assert lens_fixed == {"tau_n": 3.0}

    space = inference_lensing.build_parameter_space(
        "powerlaw+peak",
        True,
        True,
        True,
        fixed_parameter_values=base_fixed,
    )
    assert "tau_n" not in space[0]


def test_lensing_cli_fixed_tau_n_samples_only_log10_tau_A():
    opts = SimpleNamespace(fix_lens_rate=False)
    labels, lower, upper = inference_lensing._build_lens_parameter_space(
        opts, {"tau_n": 3.0}, {"log10_tau_A": [-5.0, -2.5]}
    )

    assert labels == ["log10_tau_A"]
    assert lower.tolist() == [-5.0]
    assert upper.tolist() == [-2.5]


def test_lensing_cli_decode_lens_params_uses_fixed_tau_n_and_sampled_tau_A():
    """A fixed local lens tau_n combines with sampled log10_tau_A."""
    opts = SimpleNamespace(fix_lens_rate=False, sl_tau_A=5e-4, sl_tau_n=1.0)
    sis = inference_lensing._decode_lens_params(
        jnp.asarray([-4.0]), ["log10_tau_A"], {"tau_n": 3.0}, opts
    )

    assert float(np.asarray(sis.A_tau)) == pytest.approx(1.0e-4)
    assert float(np.asarray(sis.n_tau)) == pytest.approx(3.0)


def test_lensing_cli_base_decoder_ignores_appended_lens_coordinate(monkeypatch):
    """The base decoder receives only base coordinates when lens labels are appended."""
    seen_decoder_shapes = []
    seen_sis = []

    class _Decoder:
        sampled_labels = ()

        def decode(self, coord):
            seen_decoder_shapes.append(tuple(np.asarray(coord).shape))
            return _cosmo(), _survey(), jnp.ones(1), None, None

    def _fake_cluster_likelihood(*args, **kwargs):
        del kwargs
        sis_params = args[16]
        seen_sis.append((float(sis_params.A_tau), float(sis_params.n_tau)))
        return sis_params.A_tau + sis_params.n_tau

    monkeypatch.setattr(
        inference_lensing,
        "darksiren_log_likelihood_with_clusters",
        _fake_cluster_likelihood,
    )
    inp = dict(
        gw_pe=None,
        gw_sel=None,
        nEvents=2,
        nsamp=1,
        Ndraw=1.0,
        singleton_indices=jnp.asarray([], dtype=jnp.int32),
        pair_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        n_singletons=0,
        n_pairs=1,
        pair_kdes=None,
        lensed=SimpleNamespace(m1_src=jnp.ones(2)),
    )
    opts = SimpleNamespace(
        fix_lens_rate=False,
        sl_tau_A=5e-4,
        sl_tau_n=1.0,
        cluster_mode="j2",
        wl_backend="disabled",
        wl_selection="standard",
        universe_model="spectral_sirens",
        pop_model="powerlaw+peak",
        sel_batch_size=None,
        lensing_wl_a=4e-3,
        lensing_wl_b=1.5,
        pair_marks="none",
    )

    loglike = inference_lensing.build_cluster_likelihood(
        opts, inp, _Decoder(), ["log10_tau_A"], {"tau_n": 3.0}
    )
    value = float(loglike(jnp.asarray([-4.0])))

    assert seen_decoder_shapes == [(0,)]
    assert seen_sis == pytest.approx([(1.0e-4, 3.0)])
    assert value == pytest.approx(3.0001)
