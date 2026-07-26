"""Four independent defects in the lensing CLI, each already fixed in its twin.

* prior_kinds was never passed to make_prior_transform / run_sampler, so every
  non-uniform prior family (the whitened GP latents, xi ~ N(0,1)) was silently
  sampled FLAT. The main CLI threads them.
* _make_run_dir omitted the seed and used exist_ok=True, so same-config jobs
  started in the same second clobbered each other. The main CLI embeds the seed
  and retries with a numeric suffix.
* _resolve_pair_marks read inp["pair_time_sigma"], which is empty under
  --partition_mode marginalize_exact (the widths live per-partition), so the
  auto rule fell through to the quadrature implementation it exists to avoid.
* The lensed-injection loaders read attrs["n_draw_sources"] unconditionally
  while the module docstring documents Ndraw_sources and preflight accepts
  either, so a file matching the documented schema passed preflight then
  raised KeyError.
"""
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# prior_kinds
# ---------------------------------------------------------------------------

def test_lensing_cli_threads_prior_kinds_to_transform_and_sampler():
    import darksirens.cli.inference_lensing as cli

    build_src = inspect.getsource(cli._build_space_and_closures)
    assert "make_prior_transform(lower, upper, prior_kinds)" in build_src, (
        "make_prior_transform called without prior_kinds -> every non-uniform "
        "prior silently becomes uniform"
    )
    run_src = inspect.getsource(cli._run_lensing_sampling)
    assert "prior_kinds=prior_kinds" in run_src, (
        "run_sampler called without prior_kinds -> the numpyro path builds its "
        "own prior and keeps sampling GP latents uniform"
    )
    # and the value must actually reach the sampler function
    assert "prior_kinds" in inspect.signature(cli._run_lensing_sampling).parameters


def test_gp_models_declare_a_non_uniform_prior():
    """Establishes the stake: there really are non-uniform priors to lose."""
    from darksirens.sky.registry import get_sky_model

    specs = get_sky_model("sphere_gp").param_specs
    kinds = {s.prior_kind for s in specs}
    assert "normal" in kinds, "no normal-prior latents -> this defect is moot"


def test_uniform_only_prior_kinds_reproduce_the_affine_map():
    """Threading prior_kinds must not perturb an all-uniform lensing run."""
    import jax.numpy as jnp

    from darksirens.inference.prior import make_prior_transform

    lo = np.array([0.0, -1.0, 10.0])
    hi = np.array([1.0, 2.0, 20.0])
    kinds = [("uniform", None, None)] * 3
    u = jnp.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        np.asarray(make_prior_transform(lo, hi)(u)),
        np.asarray(make_prior_transform(lo, hi, kinds)(u)),
        rtol=0, atol=0,
    )


# ---------------------------------------------------------------------------
# run directory collisions
# ---------------------------------------------------------------------------

def test_lensing_run_dir_embeds_seed_and_never_collides(tmp_path):
    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(save_path=str(tmp_path), pop_model="powerlaw+peak",
                           cluster_mode="off", sampler="tinyns", seed=7)
    first = cli._make_run_dir(opts)
    second = cli._make_run_dir(opts)          # same config, same second
    assert first != second, "two same-config runs shared a directory"
    assert "seed7" in os.path.basename(first)
    assert os.path.isdir(first) and os.path.isdir(second)

    other = SimpleNamespace(**{**vars(opts), "seed": 8})
    assert "seed8" in os.path.basename(cli._make_run_dir(other))


# ---------------------------------------------------------------------------
# pair_time_sigma under marginalize_exact
# ---------------------------------------------------------------------------

def _pair(sigma):
    return SimpleNamespace(i=0, j=1, delta_t_obs=1.0, sigma_delta_t=sigma)


def test_resolve_pair_marks_falls_back_to_candidate_pairs():
    """marginalize_exact keeps the widths on candidate_pairs, not on inp."""
    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl="auto",
                           sl_tau_A=1.0, sl_tau_n=1.0)
    from darksirens.lensing.slmarks import make_sis_lens_params

    T0 = float(make_sis_lens_params(A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n).T0)
    sharp = 0.5 * cli._TIME_DELTA_SHARPNESS * T0     # well inside the delta regime

    # No top-level pair_time_sigma (the marginalize_exact case), sharp marks.
    inp = {"candidate_pairs": [_pair(sharp), _pair(sharp)]}
    assert cli._resolve_pair_marks(opts, inp) == cli.PAIR_MARKS_TIME_DELTA

    # Broad marks still select the quadrature implementation.
    broad = 5.0 * cli._TIME_DELTA_SHARPNESS * T0
    inp_broad = {"candidate_pairs": [_pair(broad)]}
    assert cli._resolve_pair_marks(opts, inp_broad) == cli.PAIR_MARKS_TIME


def test_resolve_pair_marks_prefers_explicit_inp_sigmas():
    """The pre-existing path (sigmas on inp) is unchanged."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.lensing.slmarks import make_sis_lens_params

    opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl="auto",
                           sl_tau_A=1.0, sl_tau_n=1.0)
    T0 = float(make_sis_lens_params(A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n).T0)
    inp = {"pair_time_sigma": np.array([0.5 * cli._TIME_DELTA_SHARPNESS * T0])}
    assert cli._resolve_pair_marks(opts, inp) == cli.PAIR_MARKS_TIME_DELTA


def test_resolve_pair_marks_explicit_impl_still_wins():
    import darksirens.cli.inference_lensing as cli

    for impl, expected in (("quadrature", cli.PAIR_MARKS_TIME),
                           ("delta", cli.PAIR_MARKS_TIME_DELTA)):
        opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl=impl,
                               sl_tau_A=1.0, sl_tau_n=1.0)
        assert cli._resolve_pair_marks(opts, {}) == expected


# ---------------------------------------------------------------------------
# Ndraw_sources / n_draw_sources
# ---------------------------------------------------------------------------

def test_both_ndraw_source_spellings_are_accepted():
    from darksirens.lensing.lensed_injections import _read_n_draw_sources

    assert _read_n_draw_sources({"n_draw_sources": 11}) == 11.0
    assert _read_n_draw_sources({"Ndraw_sources": 13}) == 13.0
    # the documented spelling is what preflight validates under
    assert _read_n_draw_sources({"Ndraw_sources": 5, "n_draw_sources": 5}) == 5.0


def test_missing_ndraw_sources_names_both_spellings():
    from darksirens.lensing.lensed_injections import _read_n_draw_sources

    with pytest.raises(KeyError, match="Ndraw_sources"):
        _read_n_draw_sources({})


# ---------------------------------------------------------------------------
# universe_model / expected_sampled_labels (phantom survey dimensions)
# ---------------------------------------------------------------------------

def _lensing_opts(*extra):
    import darksirens.cli.inference_lensing as cli

    return cli.build_parser().parse_args(
        [
            "--gw_path", "gw.h5",
            "--gwselection_path", "sel.h5",
            "--sampler", "tinyns",
            "--cluster_mode", "off",
            "--pop_model", "powerlaw+peak",
            *extra,
        ]
    )


def _space_and_decoder(opts, tmp_path, monkeypatch):
    """Run the CLI's real space+decoder construction with the data-hungry
    closure builders stubbed out."""
    import darksirens.cli.inference_lensing as cli

    monkeypatch.setattr(cli, "build_cluster_likelihood", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_cluster_diagnostics", lambda *a, **k: None)
    out = cli._build_space_and_closures(
        opts, {}, str(tmp_path), {}, {}, {}
    )
    return out[0]


def test_lensing_space_matches_decoder_with_fix_survey_off(tmp_path, monkeypatch):
    """universe_model gates the sampled survey block (prior._ACTIVE_SURVEY_PARAMS).
    The lensing CLI omitted it, so --fix_survey false sampled seven flat survey
    nuisance dimensions (12 -> 19) that the decoder -- which does read
    opts.universe_model = spectral_sirens_wl -- dropped, and which
    _decode_base_parameters therefore never read."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.parameters import build_parameter_decoder

    opts = _lensing_opts("--fix_cosmology", "true", "--fix_survey", "false")
    cli._resolve_lensing_run_config(opts)
    assert opts.universe_model == "spectral_sirens_wl"

    labels = _space_and_decoder(opts, tmp_path, monkeypatch)
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(opts.pop_model), fixed_parameter_values={}
    )
    # --fix_lens_rate defaults true, so the space is base-only here
    assert list(labels) == list(decoder.sampled_labels)
    for phantom in ("log10n0", "z50", "w", "delta", "b_miss", "alpha_miss", "sigma_kde"):
        assert phantom not in labels


def test_lensing_cli_arms_the_expected_sampled_labels_net(tmp_path, monkeypatch):
    """The CLI must record the BASE sampler labels on opts so
    build_parameter_decoder's fail-fast net catches any future flag drift; the
    net was armed only in cli/inference.py, leaving every lensing run unguarded."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts("--fix_cosmology", "true", "--fix_survey", "false",
                         "--fix_lens_rate", "false")
    cli._resolve_lensing_run_config(opts)
    labels = _space_and_decoder(opts, tmp_path, monkeypatch)

    assert getattr(opts, "expected_sampled_labels", None) is not None
    # lens-only labels are appended after the base block and are not part of the
    # decoder's space, so the recorded net must exclude them
    assert "log10_tau_A" in labels
    assert "log10_tau_A" not in opts.expected_sampled_labels
    assert list(opts.expected_sampled_labels) == [
        lbl for lbl in labels if lbl not in cli.LENS_PARAMETER_PRIORS
    ]


def test_lensing_decoder_net_fires_on_injected_label_drift(tmp_path, monkeypatch):
    """The armed net must actually raise, not just be present."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.parameters import build_parameter_decoder

    opts = _lensing_opts("--fix_cosmology", "true", "--fix_survey", "false")
    cli._resolve_lensing_run_config(opts)
    _space_and_decoder(opts, tmp_path, monkeypatch)

    opts.expected_sampled_labels = tuple(opts.expected_sampled_labels) + ("bogus",)
    with pytest.raises(ValueError, match="diverge"):
        build_parameter_decoder(
            opts, get_fixed_population_params(opts.pop_model),
            fixed_parameter_values={},
        )


# ---------------------------------------------------------------------------
# archived lens optical depth / logZerr
# ---------------------------------------------------------------------------

def _run_save_phase(tmp_path, extra_args=(), lens_fixed=None, logZerr=0.21):
    """Drive _save_lensing_outputs on a minimal fixed-partition bundle and read
    back (results.hdf5 attrs, settings.json)."""
    import json

    import h5py

    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts(*extra_args)
    cli._resolve_lensing_run_config(opts)
    lens_fixed = dict(lens_fixed or {})
    lens_labels, lens_lower, lens_upper = cli._build_lens_parameter_space(
        opts, lens_fixed, {}
    )
    labels = ["H0"] + list(lens_labels)
    lower = np.concatenate([[50.0], lens_lower])
    upper = np.concatenate([[100.0], lens_upper])
    mid = 0.5 * (lower + upper)
    inp = {"nEvents": 4, "n_singletons": 4, "n_pairs": 0}
    results = {
        "samples": np.zeros((3, len(labels))),
        "logZ": -67.4497,
        "logZerr": logZerr,
    }
    settings = {}
    cli._save_lensing_outputs(
        opts, str(tmp_path), settings, inp, results, {}, labels, mid,
        {}, {}, lens_fixed, {},
    )
    with h5py.File(tmp_path / "results.hdf5", "r") as f:
        attrs = dict(f.attrs)
    with open(tmp_path / "settings.json") as f:
        saved_settings = json.load(f)
    return attrs, saved_settings, lens_lower, lens_upper


def test_sampled_lens_optical_depth_is_not_archived_under_a_bare_name(tmp_path):
    """A SAMPLED log10_tau_A has no run-level value: `mid` is the prior
    midpoint, so results.hdf5 attrs['lens_A_tau'] used to report 10**((lo+hi)/2)
    from the prior box as 'the SIS optical-depth normalisation of this run',
    silently tracking the prior bounds rather than the data."""
    import json

    attrs, settings, lo, hi = _run_save_phase(
        tmp_path, extra_args=("--fix_lens_rate", "false")
    )
    midpoint_A = 10.0 ** (0.5 * (lo[0] + hi[0]))

    for store in (attrs, settings):
        assert "lens_A_tau" not in store
        assert "lens_n_tau" not in store
        assert store["prior_midpoint_lens_A_tau"] == pytest.approx(midpoint_A)
        assert store["lens_parameters_eval_point"] == "prior_midpoint"
    assert json.loads(attrs["lens_labels"]) == ["log10_tau_A", "tau_n"]


def test_fixed_lens_rate_keeps_the_bare_lens_attribute_names(tmp_path):
    """--fix_lens_rate true: the archived value IS the run's fixed value, so the
    bare names stay (and no eval-point stamp is added)."""
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path)

    for store in (attrs, settings):
        assert store["lens_A_tau"] == pytest.approx(5e-4)   # --sl_tau_A default
        assert store["lens_n_tau"] == pytest.approx(3.0)    # --sl_tau_n default
        assert "prior_midpoint_lens_A_tau" not in store
        assert "lens_parameters_eval_point" not in store


def test_partially_fixed_lens_block_splits_bare_and_prefixed_names(tmp_path):
    """tau_n pinned via --fixed_parameter_values while log10_tau_A is sampled:
    only the sampled one is eval-point-prefixed."""
    attrs, settings, _lo, _hi = _run_save_phase(
        tmp_path,
        extra_args=("--fix_lens_rate", "false"),
        lens_fixed={"tau_n": 3.0},
    )
    for store in (attrs, settings):
        assert store["lens_n_tau"] == pytest.approx(3.0)
        assert "lens_A_tau" not in store
        assert "prior_midpoint_lens_A_tau" in store
        assert store["lens_parameters_eval_point"] == "prior_midpoint"


def test_diagnostics_lens_values_use_a_generic_eval_point_stamp():
    """build_cluster_diagnostics evaluates at whatever point cleared the
    reliability guard -- possibly a seeded prior draw -- so its sampled lens
    values must not be labelled prior_midpoint_*."""
    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(fix_lens_rate=False, sl_tau_A=5e-4, sl_tau_n=1.0)
    out = cli._lens_settings_dict(
        np.asarray([-4.0, 2.0]), ["log10_tau_A", "tau_n"], {}, opts,
        eval_point="diagnostics_point",
    )
    assert out["lens_parameters_eval_point"] == "diagnostics_point"
    assert out["diagnostics_point_lens_A_tau"] == pytest.approx(1e-4)
    assert out["diagnostics_point_lens_n_tau"] == pytest.approx(2.0)
    assert "lens_A_tau" not in out and "lens_n_tau" not in out
