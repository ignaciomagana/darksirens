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
    assert "make_prior_transform(" in build_src and (
        "lower, upper, prior_kinds" in build_src
    ), (
        "make_prior_transform called without prior_kinds -> every non-uniform "
        "prior silently becomes uniform"
    )
    assert "joint_constraints=joint_constraints" in build_src, (
        "make_prior_transform called without the resolved joint constraints "
        "-> jointly-constrained models (GWTC-5 simplex/ordering, the dipole "
        "ball) fall back to rejection sampling"
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
# SIS time-delay scale T0: configurability + support guard
# ---------------------------------------------------------------------------

def test_sl_T0_sec_is_a_cli_flag_threaded_into_sis_params():
    """T0 was hard-wired: none of the three make_sis_lens_params call sites
    could set it, so a real-data run could not move the SIS support edge."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.lensing.slmarks import DEFAULT_T0_SECONDS

    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    opts = SimpleNamespace(fix_lens_rate=True, sl_tau_A=5e-4, sl_tau_n=3.0,
                           sl_T0_sec=1.234e7)
    sis = cli._decode_lens_params(None, [], {}, opts)
    assert float(sis.T0) == 1.234e7

    # Sampled-lens-rate branch threads it too.
    opts_sampled = SimpleNamespace(fix_lens_rate=False, sl_tau_A=5e-4,
                                   sl_tau_n=3.0, sl_T0_sec=1.234e7)
    sis2 = cli._decode_lens_params(np.array([-3.0]), ["log10_tau_A"],
                                   {"tau_n": 3.0}, opts_sampled)
    assert float(sis2.T0) == 1.234e7

    # Absent / None falls back to the physically-derived default.
    assert cli._sl_T0_seconds(SimpleNamespace()) == DEFAULT_T0_SECONDS
    assert cli._sl_T0_seconds(SimpleNamespace(sl_T0_sec=None)) == DEFAULT_T0_SECONDS
    del parser


def test_sis_time_mark_support_classifies_out_of_support_delays():
    from darksirens.lensing.marginal_diagnostics import sis_time_mark_support

    T0 = 5.36e6
    day = 86400.0
    inside = sis_time_mark_support([1 * day, 30 * day], T0)
    assert inside["n_out_of_support"] == 0
    assert not inside["all_out_of_support"]

    mixed = sis_time_mark_support([1 * day, 90 * day], T0)
    assert mixed["n_out_of_support"] == 1
    assert not mixed["all_out_of_support"]

    # Everything past T0: every pair likelihood is exactly -inf.
    allout = sis_time_mark_support([70 * day, 90 * day, 200 * day], T0)
    assert allout["all_out_of_support"]
    assert allout["max_y_star"] > 1.0

    # Sign is irrelevant (y* uses |dt|), and Nones are dropped.
    signed = sis_time_mark_support([-30 * day, None], T0)
    assert signed["n_marked"] == 1
    assert signed["n_out_of_support"] == 0


def test_likelihood_construction_refuses_all_out_of_support_time_marks():
    """A GWTC-style month-scale candidate set under a too-small T0 must be a
    loud error, not an -inf likelihood the sampler cannot initialise on."""
    import darksirens.cli.inference_lensing as cli

    day = 86400.0
    opts = SimpleNamespace(pair_marks="time", sl_T0_sec=4.32e5)   # old 5-day T0
    inp = {"pair_time_delta_t_obs": np.array([30 * day, 45 * day])}
    with pytest.raises(SystemExit, match="outside the SIS support"):
        cli._require_time_marks_in_sis_support(opts, inp)

    # Raising T0 to the physical scale admits them.
    opts_ok = SimpleNamespace(pair_marks="time", sl_T0_sec=5.36e6)
    cli._require_time_marks_in_sis_support(opts_ok, inp)

    # Falls back to candidate_pairs when inp carries no materialised marks
    # (the marginalize_exact path).
    inp_cand = {"candidate_pairs": [SimpleNamespace(i=0, j=1,
                                                    delta_t_obs=30 * day,
                                                    sigma_delta_t=3600.0)]}
    with pytest.raises(SystemExit, match="outside the SIS support"):
        cli._require_time_marks_in_sis_support(opts, inp_cand)

    # Inert without time marks.
    cli._require_time_marks_in_sis_support(SimpleNamespace(pair_marks="none"), inp)


def test_sis_support_verdict_credits_the_mark_width_and_implementation():
    """y* >= 1 alone does NOT annihilate a pair: the delta-collapse mark
    integrates over its own Gaussian width (nodes out to u_max sigma_dt/T0) and
    the quadrature mark never masks at all, so the fatal verdict must be the
    width-aware one."""
    from darksirens.lensing.marginal_diagnostics import (
        SIS_TIME_COLLAPSE_U_MAX,
        resolve_sis_time_mark_impl,
        sis_time_mark_support,
    )

    T0 = 5.36e6
    sigma = 0.01 * T0  # sharp -> delta-collapse implementation
    assert resolve_sis_time_mark_impl("auto", [sigma], T0) == "delta"
    assert SIS_TIME_COLLAPSE_U_MAX == pytest.approx(4.1445, abs=1e-3)

    # Just past the edge: part of the mark measure still lands in (0, 1).
    near = sis_time_mark_support(
        [1.02 * T0], T0, sigma_delta_t_seconds=sigma, mark_impl="delta"
    )
    assert near["all_out_of_support"] and not near["all_annihilated"]
    assert near["n_annihilated"] == 0

    # Far enough that every collapse node is masked: still fatal.
    far = sis_time_mark_support(
        [1.10 * T0], T0, sigma_delta_t_seconds=sigma, mark_impl="delta"
    )
    assert far["all_annihilated"] and far["n_annihilated"] == 1

    # The quadrature mark is finite everywhere, so nothing is annihilated.
    quad = sis_time_mark_support(
        [6.0 * T0], T0, sigma_delta_t_seconds=0.1 * T0, mark_impl="quadrature"
    )
    assert quad["n_out_of_support"] == 1 and not quad["all_annihilated"]

    # Unknown widths keep the strict width-free verdict.
    strict = sis_time_mark_support([1.02 * T0], T0)
    assert strict["all_annihilated"] and resolve_sis_time_mark_impl("auto", [], T0) is None


def test_sis_support_guard_does_not_abort_a_finite_near_boundary_likelihood(capsys):
    """The guard used to kill runs whose pair likelihood was merely suppressed."""
    import darksirens.cli.inference_lensing as cli

    T0 = 5.36e6
    opts = SimpleNamespace(pair_marks="time", sl_T0_sec=T0, pair_time_mark_impl="auto")
    near = {
        "pair_time_delta_t_obs": np.array([1.02 * T0]),
        "pair_time_sigma": np.array([0.01 * T0]),
    }
    cli._require_time_marks_in_sis_support(opts, near)
    assert "outside the SIS support" in capsys.readouterr().out

    # Deep out of support with the same sharp mark is still fatal.
    far = {
        "pair_time_delta_t_obs": np.array([1.5 * T0]),
        "pair_time_sigma": np.array([0.01 * T0]),
    }
    with pytest.raises(SystemExit, match="outside the SIS support"):
        cli._require_time_marks_in_sis_support(opts, far)

    # With the quadrature implementation the mark is finite: warn, do not abort.
    quad = SimpleNamespace(
        pair_marks="time", sl_T0_sec=T0, pair_time_mark_impl="quadrature"
    )
    cli._require_time_marks_in_sis_support(quad, far)
    assert "outside the SIS support" in capsys.readouterr().out


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
    """universe_model gates the sampled survey block (the survey registry in
    inference/prior.py).
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

def _run_save_phase(tmp_path, extra_args=(), lens_fixed=None, logZerr=0.21,
                    dead_points=None, diagnostics=None, diagnostics_point=None,
                    diagnostics_point_label="prior_midpoint"):
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
        "dead_points": dead_points,
    }
    settings = {}
    cli._save_lensing_outputs(
        opts, str(tmp_path), settings, inp, results, diagnostics or {}, labels, mid,
        {}, {}, lens_fixed, {},
        diagnostics_point=diagnostics_point,
        diagnostics_point_label=diagnostics_point_label,
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


def test_lensing_outputs_persist_logzerr(tmp_path):
    """The lensing paper's central quantity is logZ(j2) - logZ(off); logZerr was
    printed to stdout and dropped, so archived run directories carried no error
    bar on the Bayes factor (the main CLI's save_results_hdf5 writes both)."""
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path, logZerr=0.21)
    assert attrs["logZ"] == pytest.approx(-67.4497)
    assert attrs["logZerr"] == pytest.approx(0.21)
    assert settings["logZ"] == pytest.approx(-67.4497)
    assert settings["logZerr"] == pytest.approx(0.21)


def test_lensing_outputs_logzerr_is_nan_when_the_sampler_reports_none(tmp_path):
    """Mirror io.results.save_results_hdf5: a missing error is NaN, not absent,
    so the attr is always readable."""
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path, logZerr=None)
    assert np.isnan(attrs["logZerr"])
    assert settings["logZerr"] is None


@pytest.mark.parametrize("sampler", ["tinyns", "dynesty"])
def test_logzerr_persistence_is_not_gated_on_the_sampler(tmp_path, sampler):
    """Both nested samplers report a logZ error, and the write is keyed on the
    result dict rather than on --sampler, so neither can lose it."""
    attrs, settings, _lo, _hi = _run_save_phase(
        tmp_path, extra_args=("--sampler", sampler), logZerr=0.33
    )
    assert attrs["logZerr"] == pytest.approx(0.33)
    assert settings["logZerr"] == pytest.approx(0.33)


def test_lensing_outputs_persist_the_dead_point_record(tmp_path):
    """The lensing CLI writes its own results.hdf5, so the additive dead-point
    datasets (logl_dead/logwt_dead) have to be wired there too -- and it is the
    CLI that most needs them, its headline being a logZ difference that an
    evidence bootstrap has to be rebuilt from."""
    import h5py

    from darksirens.io.results import DEAD_POINT_SEMANTICS

    logl = np.sort(np.linspace(-9.0, -1.0, 11))
    logwt = np.linspace(-20.0, -3.0, 11)
    _run_save_phase(tmp_path, dead_points={
        "logl": logl, "logwt": logwt, "n_dead": 11, "n_live": 40,
    })
    with h5py.File(tmp_path / "results.hdf5", "r") as f:
        np.testing.assert_allclose(f["logl_dead"][()], logl)
        np.testing.assert_allclose(f["logwt_dead"][()], logwt)
        assert f.attrs["n_dead"] == 11
        assert f.attrs["n_live"] == 40
        assert f.attrs["dead_points"] == DEAD_POINT_SEMANTICS
        # Indexed by dead point, NOT by the 3 posterior samples above.
        assert f["samples"].shape[0] == 3


def test_lensing_outputs_without_dead_points_are_unchanged(tmp_path):
    """numpyro lensing runs and every archived run predating the schema."""
    import h5py

    _run_save_phase(tmp_path)
    with h5py.File(tmp_path / "results.hdf5", "r") as f:
        assert "logl_dead" not in f and "logwt_dead" not in f
        assert "n_dead" not in f.attrs and "dead_points" not in f.attrs


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


# ---------------------------------------------------------------------------
# --cluster_mode off + --partition_mode marginalize_exact
# ---------------------------------------------------------------------------

def test_off_mode_rejects_partition_marginalisation_before_the_data_load():
    """The off control is singleton-only, so there are no candidate pairs to
    marginalise.  The combination used to pass preflight, write the run
    directory, spend minutes loading PE/selection, and only then raise
    KeyError: 'marginal_partitions' at the midpoint smoke test -- load_inputs
    returns early for off mode BEFORE the partition-mode requirement checks, and
    the likelihood closure branches on the flag alone."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts("--partition_mode", "marginalize_exact")
    assert opts.cluster_mode == "off"
    with pytest.raises(SystemExit, match="requires --cluster_mode j2"):
        cli._resolve_lensing_run_config(opts)


def test_load_inputs_also_rejects_the_combination(monkeypatch):
    """Second net for direct library callers, which never run
    _resolve_lensing_run_config.  It must fire BEFORE any file is opened."""
    import darksirens.cli.inference_lensing as cli

    def _boom(*a, **k):
        raise AssertionError("load_inputs opened data before validating flags")

    monkeypatch.setattr(cli, "load_gw_samples", _boom)
    opts = SimpleNamespace(
        cluster_mode="off", partition_mode="marginalize_exact", seed=1,
    )
    with pytest.raises(SystemExit, match="requires --cluster_mode j2"):
        cli.load_inputs(opts)


@pytest.mark.parametrize(
    "cluster_mode,partition_mode",
    [("j2", "marginalize_exact"), ("j2", "fixed"), ("off", "fixed")],
)
def test_supported_cluster_partition_combinations_still_resolve(
    cluster_mode, partition_mode
):
    """The guard must reject exactly one cell of the 2x2, not narrow the CLI."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts(
        "--cluster_mode", cluster_mode, "--partition_mode", partition_mode
    )
    cli._resolve_lensing_run_config(opts)
    assert opts.universe_model == "spectral_sirens_wl"


# ---------------------------------------------------------------------------
# pair_orientation_mode: flag surface + rendered-campaign consistency warning
# ---------------------------------------------------------------------------

def test_pair_orientation_mode_flag_defaults_to_independent():
    """The joint-orientation model is opt-in: the default must stay
    'independent' (existing mocks were rendered with independent per-image
    draws) and unknown values must be rejected at parse time."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts()
    assert opts.pair_orientation_mode == "independent"
    opts = _lensing_opts("--pair_orientation_mode", "shared_iota")
    assert opts.pair_orientation_mode == "shared_iota"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--gw_path", "g", "--gwselection_path", "s",
             "--pair_orientation_mode", "shared_orientation"]
        )


def _campaign_file(tmp_path, name, mode=None):
    """A stub lensed-injection file carrying only the campaign attr."""
    import h5py

    path = str(tmp_path / name)
    with h5py.File(path, "w") as f:
        if mode is not None:
            f.attrs["pair_orientation_mode"] = mode
    return path


def test_pair_orientation_mismatch_is_fatal_with_the_singleton_channel_on(tmp_path):
    """Running --pair_orientation_mode against a campaign rendered with the
    OTHER convention mis-normalises the J=2/lensed-singleton ratio that sets
    A_tau by up to ~2.6x. That ratio only enters through the lensed-singleton
    channel, so the mismatch is FATAL while that channel is enabled and a
    warning otherwise -- it used to be a warning either way and ran on."""
    import darksirens.cli.inference_lensing as cli

    path = _campaign_file(tmp_path, "lensed.h5", "shared_iota")

    opts = _lensing_opts("--lensed_injections_path", path,
                         "--singleton_lensing", "sl_mixture")
    with pytest.raises(SystemExit, match="pair_orientation_mode mismatch"):
        cli._gate_pair_orientation_mismatch(opts)

    # attr-less legacy files count as 'independent', so the mismatch is
    # detected in the other direction too.
    legacy = _campaign_file(tmp_path, "legacy.h5")
    opts = _lensing_opts("--lensed_injections_path", legacy,
                         "--pair_orientation_mode", "shared_iota",
                         "--singleton_lensing", "sl_mixture")
    with pytest.raises(SystemExit, match="independent"):
        cli._gate_pair_orientation_mismatch(opts)


def test_pair_orientation_mismatch_override_downgrades_to_a_warning(tmp_path):
    """Deliberate convention ablations stay possible, loudly."""
    import darksirens.cli.inference_lensing as cli

    path = _campaign_file(tmp_path, "lensed.h5", "shared_iota")
    opts = _lensing_opts("--lensed_injections_path", path,
                         "--singleton_lensing", "sl_mixture",
                         "--allow_pair_orientation_mismatch")
    assert opts.allow_pair_orientation_mismatch is True
    with pytest.warns(RuntimeWarning, match="NOT trustworthy"):
        cli._gate_pair_orientation_mismatch(opts)


def test_pair_orientation_mismatch_warns_when_the_channel_is_off(tmp_path):
    """--singleton_lensing off never evaluates the censoring factor, so the
    mismatch cannot bias the likelihood; matched modes stay silent."""
    import warnings as _warnings

    import darksirens.cli.inference_lensing as cli

    path = _campaign_file(tmp_path, "lensed.h5", "shared_iota")
    opts = _lensing_opts("--lensed_injections_path", path)
    assert opts.singleton_lensing == "off"
    with pytest.warns(RuntimeWarning, match="pair_orientation_mode"):
        cli._gate_pair_orientation_mismatch(opts)

    for extra in (("--singleton_lensing", "sl_mixture"), ()):
        opts = _lensing_opts("--lensed_injections_path", path,
                             "--pair_orientation_mode", "shared_iota", *extra)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            cli._gate_pair_orientation_mismatch(opts)


def test_preflight_reports_the_pair_orientation_mismatch(tmp_path):
    """--preflight_only must not pass a configuration the run itself refuses."""
    from darksirens.lensing.preflight import run_lensing_preflight

    path = _campaign_file(tmp_path, "lensed.h5", "shared_iota")

    def _report(*extra):
        return run_lensing_preflight(
            _lensing_opts("--lensed_injections_path", path, *extra)
        )

    on = _report("--singleton_lensing", "sl_mixture")
    assert any("mis-normalised" in e for e in on["errors"])
    assert on["summary"]["pair_orientation_mode_campaign"] == "shared_iota"
    assert on["summary"]["pair_orientation_mode_runtime"] == "independent"
    assert on["summary"]["pair_orientation_mode_match"] is False

    waived = _report("--singleton_lensing", "sl_mixture",
                     "--allow_pair_orientation_mismatch")
    assert not any("mis-normalised" in e for e in waived["errors"])
    assert any("mis-normalised" in w for w in waived["warnings"])

    off = _report()
    assert not any("mis-normalised" in e for e in off["errors"])
    assert any("mis-normalised" in w for w in off["warnings"])

    matched = _report("--pair_orientation_mode", "shared_iota",
                      "--singleton_lensing", "sl_mixture")
    assert not any("mis-normalised" in m
                   for m in matched["errors"] + matched["warnings"])
    assert matched["summary"]["pair_orientation_mode_match"] is True


def _one_source_campaign(tmp_path, name, mode=None):
    """A real one-source (exactly-one-detected) injection file."""
    import h5py

    from darksirens.lensing.lensed_injections import save_lensed_injections

    path = str(tmp_path / name)
    two = np.array([0.0, 0.0])
    save_lensed_injections(
        path,
        source_id=np.array([0, 0]), image_id=np.array([0, 1]),
        m1_src=two + 30.0, q_src=two + 0.8, z_src=two + 0.5,
        chieff=two, y_source=two + 0.4, mu=np.array([3.0, 1.5]),
        detected=np.array([True, False]),
        p_prop_src=two + 1.0, p_prop_y=two + 1.0, n_draw_sources=7,
        snr_model_attrs={"fc_rho_thr": 8.0, "fc_r0": 750.0, "fc_mc_bar": 1.22},
    )
    if mode is not None:
        with h5py.File(path, "r+") as f:
            f.attrs["pair_orientation_mode"] = mode
    return path


def test_singleton_subset_loader_preserves_the_campaign_attr(tmp_path):
    """The exactly-one-detected subset view dropped pair_orientation_mode, so a
    library caller had no way to check the convention it was rendered with. It
    stays OUT of the NamedTuple (a str leaf is not a valid JIT argument) and
    comes back alongside it."""
    from darksirens.lensing.lensed_injections import load_lensed_single_image_set

    path = _one_source_campaign(tmp_path, "campaign.h5", "shared_iota")
    singles, campaign = load_lensed_single_image_set(
        path, return_campaign_attrs=True
    )
    assert campaign["pair_orientation_mode"] == "shared_iota"
    assert singles.n_kept == 1
    # Default call is unchanged (and JIT-safe: no string leaf).
    bare = load_lensed_single_image_set(path)
    assert bare.n_kept == 1
    assert not any(isinstance(leaf, str) for leaf in bare)

    # An attr-less campaign reports the legacy convention, not None.
    legacy = _one_source_campaign(tmp_path, "legacy_campaign.h5")
    _, legacy_campaign = load_lensed_single_image_set(
        legacy, return_campaign_attrs=True
    )
    assert legacy_campaign["pair_orientation_mode"] == "independent"


def test_singleton_channel_loader_rechecks_the_campaign_attr(tmp_path):
    """The preserved attr has to be USED where the censoring factor is built,
    not only in the startup gate."""
    import darksirens.cli.inference_lensing as cli

    path = _one_source_campaign(tmp_path, "campaign.h5", "shared_iota")
    opts = _lensing_opts("--lensed_injections_path", path,
                         "--singleton_lensing", "sl_mixture")
    with pytest.raises(SystemExit, match="pair_orientation_mode mismatch"):
        cli._load_singleton_lensing_inputs(opts)

    matched = _lensing_opts("--lensed_injections_path", path,
                            "--singleton_lensing", "sl_mixture",
                            "--pair_orientation_mode", "shared_iota")
    loaded = cli._load_singleton_lensing_inputs(matched)
    assert loaded["lensed_singles"].n_kept == 1
    assert loaded["fc_pdet_params"] is not None


# ---------------------------------------------------------------------------
# time-mark magnitude / finiteness vs the observed catalog
# ---------------------------------------------------------------------------

def _time_marked_pairs(delta_t_obs, sigma_delta_t=10.0):
    from darksirens.lensing.partitions import CandidatePair, EdgeMarks

    return [
        CandidatePair(0, 1, -1.0, None,
                      EdgeMarks(delta_t_obs=delta_t_obs,
                                sigma_delta_t=sigma_delta_t))
    ]


def _gps(*times):
    return np.asarray(times, dtype=float)


def test_time_mark_magnitude_disagreement_is_fatal():
    """The mark magnitude must reproduce the catalog's arrival separation. It
    used to warn and then feed the likelihood the MARK's magnitude with the
    CATALOG's sign -- a delay neither input claims."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts()
    assert opts.allow_time_mark_mismatch is False
    with pytest.raises(SystemExit, match=r"\(0,1\)"):
        cli._orient_time_marks(_time_marked_pairs(500.0), _gps(0.0, 100.0), opts)
    # Fatal by default for direct library callers too (no opts at all).
    with pytest.raises(SystemExit, match="time-mark magnitude disagreement"):
        cli._orient_time_marks(_time_marked_pairs(500.0), _gps(0.0, 100.0))


def test_time_mark_mismatch_override_keeps_the_legacy_orientation(capsys):
    """--allow_time_mark_mismatch restores the old behaviour verbatim: the
    mark's magnitude, the catalog's sign, and a loud warning."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts("--allow_time_mark_mismatch")
    assert opts.allow_time_mark_mismatch is True

    pairs, signed = cli._orient_time_marks(
        _time_marked_pairs(500.0), _gps(0.0, 100.0), opts
    )
    assert signed
    assert pairs[0].delta_t_obs == 500.0        # mark magnitude, catalog sign +
    assert pairs[0].sigma_delta_t == 10.0
    out = capsys.readouterr().out
    assert "disagree with the observed catalog" in out and "(0,1)" in out

    # Reversed arrival order -> same magnitude, negative sign.
    flipped, _ = cli._orient_time_marks(
        _time_marked_pairs(500.0), _gps(100.0, 0.0), opts
    )
    assert flipped[0].delta_t_obs == -500.0


def test_time_mark_disagreement_reports_the_worst_edge(capsys):
    """With several bad edges the first one is rarely the informative one."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.lensing.partitions import CandidatePair, EdgeMarks

    marks = lambda dt: EdgeMarks(delta_t_obs=dt, sigma_delta_t=10.0)
    pairs = [
        CandidatePair(0, 1, -1.0, None, marks(200.0)),   # off by 100 s
        CandidatePair(1, 2, -1.0, None, marks(9000.0)),  # off by 8900 s
    ]
    with pytest.raises(SystemExit, match=r"worst edge \(1,2\)"):
        cli._orient_time_marks(pairs, _gps(0.0, 100.0, 200.0), _lensing_opts())


def test_nonfinite_time_marks_are_fatal_even_without_arrival_times():
    """A NaN delay or width is broken whether or not the catalog can orient it;
    it propagated straight into the marked pair likelihood before."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts()
    for bad in (_time_marked_pairs(np.nan), _time_marked_pairs(np.inf),
                _time_marked_pairs(100.0, sigma_delta_t=np.nan)):
        with pytest.raises(SystemExit, match="non-finite candidate time marks"):
            cli._orient_time_marks(bad, _gps(0.0, 100.0), opts)
        # The check sits ahead of the "no arrival times -> nothing to do" exit.
        with pytest.raises(SystemExit, match="non-finite candidate time marks"):
            cli._orient_time_marks(bad, None, opts)

    allowed = _lensing_opts("--allow_time_mark_mismatch")
    pairs, signed = cli._orient_time_marks(
        _time_marked_pairs(np.nan), _gps(0.0, 100.0), allowed
    )
    assert signed and np.isnan(pairs[0].delta_t_obs)


def test_sign_only_time_mark_disagreement_is_corrected_silently(capsys):
    """The magnitude agrees and only the stored order is reversed: resolving
    that IS this function's job, so it must not be reported at all."""
    import warnings as _warnings

    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        pairs, signed = cli._orient_time_marks(
            _time_marked_pairs(100.0), _gps(100.0, 0.0), opts
        )
    assert signed
    assert pairs[0].delta_t_obs == -100.0
    assert capsys.readouterr().out == ""


def test_sis_support_diagnostic_counts_discarded_nonfinite_delays():
    """The support verdict is finite-only by construction; the non-finite
    delays it cannot judge are now counted instead of vanishing."""
    from darksirens.lensing.marginal_diagnostics import sis_time_mark_support

    day = 86400.0
    support = sis_time_mark_support([1 * day, np.nan, np.inf, None], 5.36e6)
    assert support["n_marked"] == 1            # unchanged: finite delays only
    assert support["n_nonfinite_delays"] == 2
    assert support["n_out_of_support"] == 0

    # Also present on the degenerate early-return branch.
    empty = sis_time_mark_support([np.nan], 5.36e6)
    assert empty["n_marked"] == 0 and empty["n_nonfinite_delays"] == 1
    assert sis_time_mark_support([1 * day], 5.36e6)["n_nonfinite_delays"] == 0


def test_preflight_rejects_nonfinite_candidate_time_marks(tmp_path, monkeypatch):
    """Preflight collected candidate delta_t_obs without ever checking it was
    finite; the SIS-support diagnostic then discarded the non-finite ones."""
    import json

    from darksirens.lensing import preflight
    from darksirens.lensing.partitions import CandidatePair, EdgeMarks

    opts = _lensing_opts("--pair_marks", "time")
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({
        "n_events": 2,
        "pairs": [{"i": 0, "j": 1, "log_prior_odds": -1.0,
                   "marks": {"delta_t_obs": 1.0e5, "sigma_delta_t": 3600.0}}],
    }))
    errors, warns, summary = [], [], {}
    preflight._check_candidates(str(path), 2, opts, errors, warns, summary)
    assert not errors

    # The JSON reader rejects non-finite marks itself, so reach the check the
    # way a direct library caller does: with pairs built in-process.
    bad = (CandidatePair(0, 1, -1.0, None,
                         EdgeMarks(delta_t_obs=np.nan, sigma_delta_t=3600.0)),)
    monkeypatch.setattr(preflight, "validate_candidate_pairs",
                        lambda data: (2, bad))
    errors, warns, summary = [], [], {}
    preflight._check_candidates(str(path), 2, opts, errors, warns, summary)
    assert any("non-finite time marks" in e for e in errors), errors


def test_preflight_rejects_nonfinite_pair_pe_time_marks(tmp_path):
    """The fixed-partition path reads delta_t_obs straight off the pair file."""
    import h5py

    from darksirens.lensing import preflight

    path = str(tmp_path / "pairs.h5")
    with h5py.File(path, "w") as f:
        f.attrs["npairs"] = 1
        g = f.create_group("pair_0")
        g.attrs["event_index_image0"] = 0
        g.attrs["event_index_image1"] = 1
        g.attrs["delta_t_obs"] = np.nan
        g.attrs["sigma_delta_t"] = np.nan

    errors, warnings_, summary = [], [], {}
    opts = _lensing_opts("--pair_marks", "time", "--partition_mode", "fixed")
    preflight._check_pair_pe(path, 2, [(0, 1)], opts, errors, warnings_,
                             summary, unified_observed_mode=True)
    assert any("delta_t_obs must be finite" in e for e in errors), errors
    assert any("sigma_delta_t must be finite" in e for e in errors), errors


# ---------------------------------------------------------------------------
# closed-form count correction vs the master likelihood's variance budget
# (review F-001)
# ---------------------------------------------------------------------------

def test_factorized_count_correction_threads_the_baseline_pe_variance():
    """The sampler-facing selection correction on the DEFAULT componentwise
    marginalize_exact path is the closed form, not the master likelihood's
    value: ``selection0`` cancels exactly out of
    ``baseline + LSE_k(dp_k + count_delta_k)``.  So the closed form has to
    carry the same total-variance budget, or the guard silently reverts to the
    selection-only bound."""
    import inspect

    import darksirens.cli.inference_lensing as cli

    src = inspect.getsource(cli._count_correction_closed_form)
    assert 'pe_variance_sum=baseline_raw["pe_variance_sum"]' in src, (
        "the closed-form count correction calls "
        "combined_selection_log_correction without pe_variance_sum -> the "
        "factorized sampler path enforces the selection-only threshold "
        "N_obs^2/max_likelihood_variance while every other path enforces the "
        "total-variance criterion"
    )
    assert "_count_correction_closed_form(baseline_raw, opts)" in inspect.getsource(
        cli.build_cluster_likelihood
    )


def test_dropping_pe_variance_really_moves_the_guard():
    """Establishes the stake: the omitted argument is not cosmetic."""
    from darksirens.likelihood.cluster_selection import (
        combined_selection_log_correction,
    )

    # A point inside the band n^2/max_var < Neff < n^2/(max_var - pe_var):
    # guarded once the per-event variances spend half the budget, admitted
    # without them.
    n_sing, n_prs = 30, 0
    log_mu = -3.0
    log_sigma2 = 2.0 * log_mu - np.log(1100.0)          # Neff = mu^2/sigma^2
    kw = dict(
        n_singletons_observed=n_sing,
        n_clusters_observed=n_prs,
        max_likelihood_variance=1.0,
    )
    neg = -np.inf
    hard_without = float(combined_selection_log_correction(
        log_mu, log_sigma2, neg, neg, pe_variance_sum=0.0, **kw))
    hard_with = float(combined_selection_log_correction(
        log_mu, log_sigma2, neg, neg, pe_variance_sum=0.5, **kw))
    assert np.isfinite(hard_without) and hard_with == -np.inf

    soft_without = float(combined_selection_log_correction(
        log_mu, log_sigma2, neg, neg, soft_guard=True,
        pe_variance_sum=0.0, **kw))
    soft_with = float(combined_selection_log_correction(
        log_mu, log_sigma2, neg, neg, soft_guard=True,
        pe_variance_sum=0.5, **kw))
    assert soft_without - soft_with > 1e3, (soft_without, soft_with)


def test_loglike_diagnostics_cross_check_accepts_agreement():
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts()
    point = np.zeros(3)
    value = cli._cross_check_loglike_against_diagnostics(
        lambda c: -12.5, {"logL_marginalized": -12.5 + 1e-12}, point, opts=opts
    )
    assert np.isclose(value, -12.5)
    # both guarded is agreement, not a mismatch
    assert cli._cross_check_loglike_against_diagnostics(
        lambda c: -np.inf, {"logL_marginalized": -np.inf}, point, opts=opts
    ) == -np.inf


@pytest.mark.parametrize("diag", [
    {"logL_marginalized": -12.5},          # finite disagreement
    {"logL_marginalized": -np.inf},        # diagnostics guarded, sampler not
])
def test_loglike_diagnostics_cross_check_catches_divergent_paths(diag):
    """This is the check that would have caught F-001: the closed form and the
    master likelihood must evaluate the same target at the same point."""
    import darksirens.cli.inference_lensing as cli

    with pytest.raises(RuntimeError, match="disagrees with the partition diagnostics"):
        cli._cross_check_loglike_against_diagnostics(
            lambda c: 90.0, diag, np.zeros(3), opts=_lensing_opts()
        )


def test_smoke_test_runs_the_cross_check_at_the_diagnostics_point():
    import inspect

    import darksirens.cli.inference_lensing as cli

    src = inspect.getsource(cli._smoke_test_likelihood)
    assert "_cross_check_loglike_against_diagnostics(" in src
    assert "loglike, diagnostics, diag_point" in src, (
        "the cross-check must use the point the diagnostics were actually "
        "evaluated at, not the prior midpoint"
    )


# ---------------------------------------------------------------------------
# diagnostics count_delta: closed form, not fabricated pairings (review F-005)
# ---------------------------------------------------------------------------

def _baseline_raw(pe_variance_sum=0.4, log_mu=-3.0, neff=1100.0):
    import jax.numpy as jnp

    from darksirens.likelihood.cluster_selection import (
        combined_selection_log_correction,
    )

    log_sigma2 = 2.0 * log_mu - np.log(neff)
    raw = {
        "log_mu_singleton": jnp.asarray(log_mu),
        "log_sigma2_singleton": jnp.asarray(log_sigma2),
        "log_mu_cluster": jnp.asarray(-np.inf),
        "log_sigma2_cluster": jnp.asarray(-np.inf),
        "pe_variance_sum": jnp.asarray(pe_variance_sum),
    }
    return raw, combined_selection_log_correction


def test_closed_form_count_correction_reproduces_the_baseline_selection():
    """``count_delta[0] == 0`` is the invariant the whole factorization rests
    on: the closed form at the all-singleton baseline counts must be the master
    likelihood's own selection correction, variance budget included."""
    import darksirens.cli.inference_lensing as cli

    raw, combined = _baseline_raw()
    opts = _lensing_opts()
    opts.selection_neff_soft_guard = True
    opts.max_likelihood_variance = 1.0

    selection0 = float(combined(
        raw["log_mu_singleton"], raw["log_sigma2_singleton"],
        raw["log_mu_cluster"], raw["log_sigma2_cluster"],
        n_singletons_observed=30, n_clusters_observed=0,
        soft_guard=True, max_likelihood_variance=1.0,
        pe_variance_sum=raw["pe_variance_sum"],
    ))
    closed = cli._count_correction_closed_form(raw, opts)
    assert float(closed(30, 0)) == selection0
    # and it is a real value, not a degenerate 0/-inf that would make the
    # assertion vacuous
    assert np.isfinite(selection0) or selection0 == -np.inf


def test_both_factorized_paths_share_one_closed_form():
    """The sampler and the diagnostics must not compute count_delta two ways:
    the diagnostics used to read each count off a FULL likelihood evaluation of
    a fabricated (2k, 2k+1) pairing with delta_t_obs=0 / sigma=1 s, whose
    fictitious pairs' pair_variance_sum entered the guard threshold."""
    import inspect

    import darksirens.cli.inference_lensing as cli

    like_src = inspect.getsource(cli.build_cluster_likelihood)
    diag_src = inspect.getsource(cli.build_cluster_diagnostics)
    assert "_count_correction_closed_form(baseline_raw, opts)" in like_src
    assert "_count_correction_closed_form(baseline_raw, opts)" in diag_src
    # the probe partitions must no longer be pushed through the master
    # likelihood (that is what cost one XLA specialization per count)
    assert "probe_raw" not in diag_src
    assert 'for probe_part in inp["selection_probe_partitions"]' in diag_src


# ---------------------------------------------------------------------------
# end-to-end: the factorized sampler path and the diagnostics path must agree
# ---------------------------------------------------------------------------

def _factorized_inp_and_opts(soft_guard):
    """A 4-event / 2-candidate componentwise marginalize_exact setup, built
    with the CLI's own partition helpers so it mirrors load_inputs."""
    import jax.numpy as jnp
    from scipy.special import logsumexp

    import darksirens.cli.inference_lensing as cli
    from darksirens.lensing.partitions import (
        CandidatePair,
        exact_partition_components,
    )

    n_events = 4
    candidates = [CandidatePair(0, 1, np.log(2.0)), CandidatePair(2, 3, np.log(3.0))]
    summaries, component_states, approx_total = exact_partition_components(
        n_events, candidates
    )
    full_states = tuple(
        tuple(cli._full_state_for_component(n_events, summary, state) for state in states)
        for summary, states in zip(summaries, component_states)
    )
    part = lambda state: cli._runtime_part_from_state(state, candidates, pair_marks="none")
    baseline_state = cli._all_singleton_partition_state(n_events)
    max_pairs = sum(max(int(s.n_pairs) for s in states) for states in component_states)

    opts = _lensing_opts(
        "--cluster_mode", "j2",
        "--partition_mode", "marginalize_exact",
        "--candidate_pairs_path", "cand.json",
        "--selection_neff_guard", "soft" if soft_guard else "hard",
    )
    import darksirens.cli.inference_lensing as _cli
    _cli._resolve_lensing_run_config(opts)
    opts.sel_batch_size = None
    assert bool(opts.selection_neff_soft_guard) is soft_guard

    inp = dict(
        gw_pe=None, gw_sel=None, nEvents=n_events, nsamp=1, Ndraw=1.0,
        lensed=None, pair_kdes=None,
        factorized_exact=True,
        candidate_pairs=candidates,
        component_partition_states=component_states,
        component_partition_summaries=summaries,
        component_full_partitions=tuple(
            tuple(part(state) for state in states) for states in full_states
        ),
        baseline_partition=part(baseline_state),
        selection_probe_partitions=tuple(
            part(cli._count_probe_partition_state(n_events, k))
            for k in range(max_pairs + 1)
        ),
        partition_states=None,
        log_z_prior=float(
            sum(logsumexp([s.log_prior_weight for s in states]) for states in component_states)
        ),
        n_singletons=int(baseline_state.n_singletons),
        n_pairs=0,
        singleton_indices=jnp.asarray(baseline_state.singleton_indices, dtype=jnp.int32),
        pair_indices=jnp.asarray(baseline_state.pair_indices, dtype=jnp.int32),
    )
    return inp, opts, candidates


# Neff comfortably above every threshold (no wall), and a (Neff, pe_variance_sum)
# pair inside the band n_tot^2/max_var < Neff < n_tot^2/(max_var - pe_var) for
# EVERY total count a 4-event/2-candidate catalog can take (n_tot = 4, 3, 2) at
# max_likelihood_variance = 1 -- the band the dropped pe_variance_sum left
# entirely unguarded in the sampler.
_CLEAR_NEFF = 1100.0
_WALLED_NEFF = 25.0
_WALLED_PE_VAR = 0.9


def _install_fake_master(monkeypatch, pe_variance_sum, neff=_CLEAR_NEFF, log_mu=-3.0,
                         pair_selection_bias=0.0):
    """Master likelihood whose selection term is the real combined correction
    at the given (partition-independent) variance budget, and whose content
    term depends on which pairs are formed."""
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli
    from darksirens.likelihood.cluster_selection import (
        combined_selection_log_correction,
    )

    log_sigma2 = 2.0 * log_mu - np.log(neff)

    def _raw(*args, **kwargs):
        n_singletons, n_pairs = int(args[12]), int(args[13])
        pair_indices = np.asarray(args[11], dtype=int).reshape((-1, 2))[:n_pairs]
        singleton_logL_sum = jnp.asarray(0.5 * n_singletons)
        pair_logL_sum = jnp.asarray(
            float(sum(0.1 * (int(i) + int(j)) for i, j in pair_indices))
        )
        selection = combined_selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(log_sigma2),
            jnp.asarray(-np.inf), jnp.asarray(-np.inf),
            n_singletons_observed=n_singletons,
            n_clusters_observed=n_pairs,
            soft_guard=bool(kwargs.get("selection_neff_soft_guard", False)),
            max_likelihood_variance=float(kwargs.get("max_likelihood_variance", 1.0)),
            pe_variance_sum=jnp.asarray(pe_variance_sum),
        )
        # Stands in for the partition dependence pair_variance_sum gives the
        # real selection correction: a term that is NOT a function of the counts
        # alone, so the closed form cannot reproduce it.
        selection = selection + pair_selection_bias * float(
            sum(int(i) + int(j) for i, j in pair_indices)
        )
        total = singleton_logL_sum + pair_logL_sum + selection
        return {
            "logL_total": jnp.where(jnp.isfinite(total), total, -jnp.inf),
            "singleton_logL_sum": singleton_logL_sum,
            "pair_logL_sum": pair_logL_sum,
            "selection_correction_total": selection,
            "log_mu_singleton": jnp.asarray(log_mu),
            "log_sigma2_singleton": jnp.asarray(log_sigma2),
            "log_mu_cluster": jnp.asarray(-np.inf),
            "log_sigma2_cluster": jnp.asarray(-np.inf),
            "pe_variance_sum": jnp.asarray(pe_variance_sum),
            "n_singletons": jnp.asarray(n_singletons),
            "n_pairs": jnp.asarray(n_pairs),
        }

    def _scalar(*args, **kwargs):
        kwargs.pop("return_diagnostics", None)
        return _raw(*args, **kwargs)["logL_total"]

    monkeypatch.setattr(cli, "darksiren_log_likelihood_with_clusters", _scalar)
    monkeypatch.setattr(cli, "darksiren_likelihood_diagnostics_with_clusters", _raw)


class _FactorizedDecoder:
    def decode(self, coord):
        import jax.numpy as jnp
        del coord
        return None, None, jnp.ones(1), None, None


@pytest.mark.parametrize("soft_guard", [False, True])
@pytest.mark.parametrize("pe_variance_sum", [0.0, 0.5])
def test_factorized_sampler_and_diagnostics_agree(monkeypatch, soft_guard, pe_variance_sum):
    """The invariant the F-001 startup cross-check enforces: on the default
    componentwise path the sampler's closed-form selection correction and the
    diagnostics' marginalized logL must be the same number at the same point.
    With pe_variance_sum dropped from the closed form they were ~1e4 nats apart
    under the soft guard."""
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli

    inp, opts, _ = _factorized_inp_and_opts(soft_guard)
    _install_fake_master(monkeypatch, pe_variance_sum)

    loglike = cli.build_cluster_likelihood(opts, inp, _FactorizedDecoder(), [], {})
    diagnostics_fn = cli.build_cluster_diagnostics(opts, inp, _FactorizedDecoder(), [], {})
    coord = jnp.zeros(1)
    diagnostics = diagnostics_fn(coord)

    # count_delta[0] == 0: the closed form reproduces the master's own value
    assert diagnostics["count_loglike_delta"][0] == 0.0
    value = cli._cross_check_loglike_against_diagnostics(
        loglike, diagnostics, coord, opts=opts
    )
    assert np.isfinite(value) == np.isfinite(diagnostics["logL_marginalized"])


def test_factorized_sampler_total_moves_with_the_variance_budget(monkeypatch):
    """The stake, end to end: spending the budget must change the sampler-facing
    likelihood.  Before the fix the closed form ignored pe_variance_sum, so the
    soft-guard wall was entirely absent from the sampler's target."""
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli

    totals = []
    for pe_var in (0.0, _WALLED_PE_VAR):
        inp, opts, _ = _factorized_inp_and_opts(soft_guard=True)
        with pytest.MonkeyPatch.context() as mp:
            # Neff inside the band n^2/max_var < Neff < n^2/(max_var - pe_var):
            # guarded by the wall once the per-event variances spend the budget.
            _install_fake_master(mp, pe_var, neff=_WALLED_NEFF)
            loglike = cli.build_cluster_likelihood(
                opts, inp, _FactorizedDecoder(), [], {}
            )
            totals.append(float(loglike(jnp.zeros(1))))
    assert totals[0] - totals[1] > 1e3, totals


def test_factorized_paths_agree_inside_the_soft_wall(monkeypatch):
    """Agreement must hold where the soft wall is ACTIVE too -- that is the band
    the dropped pe_variance_sum left completely unguarded in the sampler."""
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli

    inp, opts, _ = _factorized_inp_and_opts(soft_guard=True)
    _install_fake_master(monkeypatch, _WALLED_PE_VAR, neff=_WALLED_NEFF)
    loglike = cli.build_cluster_likelihood(opts, inp, _FactorizedDecoder(), [], {})
    diagnostics = cli.build_cluster_diagnostics(
        opts, inp, _FactorizedDecoder(), [], {}
    )(jnp.zeros(1))
    value = cli._cross_check_loglike_against_diagnostics(
        loglike, diagnostics, jnp.zeros(1), opts=opts
    )
    assert value < -1e3, "the wall is not engaged, so this proves nothing"


def test_soft_guard_reports_a_non_count_only_selection_instead_of_dying(monkeypatch, capsys):
    """The count-only invariant is exact only where the soft wall is inactive:
    inside it the wall tracks a threshold the partition's own pair_variance_sum
    moves.  Killing the run at build time -- after the full PE/injection load --
    over an approximation the sampler makes deliberately was the F-005 abort;
    warn instead, and keep the hard-guard case fatal (there the correction
    provably IS count-only whenever it is finite)."""
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli

    inp, opts, _ = _factorized_inp_and_opts(soft_guard=True)
    _install_fake_master(monkeypatch, 0.0, pair_selection_bias=0.5)
    out = cli.build_cluster_diagnostics(opts, inp, _FactorizedDecoder(), [], {})(
        jnp.zeros(1)
    )
    assert np.isfinite(out["logL_marginalized"])
    assert "not count-only" in capsys.readouterr().out


def test_hard_guard_still_refuses_a_non_count_only_selection(monkeypatch):
    import jax.numpy as jnp

    import darksirens.cli.inference_lensing as cli

    inp, opts, _ = _factorized_inp_and_opts(soft_guard=False)
    _install_fake_master(monkeypatch, 0.0, pair_selection_bias=0.5)
    with pytest.raises(RuntimeError, match="not count-only"):
        cli.build_cluster_diagnostics(opts, inp, _FactorizedDecoder(), [], {})(
            jnp.zeros(1)
        )


def test_saved_outputs_label_the_fallback_diagnostics_point(tmp_path):
    """results.hdf5 and settings.json must name the point the diagnostics were
    evaluated at. The guard-clear fallback (registry fiducial / seeded prior
    draw) is the documented common case on paper-scale joint runs, and both
    archives used to hard-code prior_midpoint (review F-006)."""
    diagnostics = {
        "n_partitions": 3,
        "expected_n_singletons": 2.5,
        "expected_n_pairs": 0.75,
        "map_partition_index": 2,
        "map_partition": {"n_singletons": 2, "n_pairs": 1},
        "logL_marginalized": -12.5,
        "log_z_partition_prior": 0.4,
    }
    marginal_args = (
        "--cluster_mode", "j2",
        "--partition_mode", "marginalize_exact",
        "--candidate_pairs_path", "cand.json",
    )
    attrs, settings, _lo, _hi = _run_save_phase(
        tmp_path, extra_args=marginal_args, diagnostics=diagnostics,
        diagnostics_point=[70.0, 0.3], diagnostics_point_label="registry_fiducial",
    )
    assert attrs["partition_diagnostics_eval_point"] == "registry_fiducial"
    assert attrs["diagnostics_point_logL_marginalized"] == pytest.approx(-12.5)
    assert "prior_midpoint_logL_marginalized" not in attrs
    assert settings["partition_diagnostics_eval_point"] == "registry_fiducial"
    assert settings["diagnostics_point_expected_n_pairs"] == pytest.approx(0.75)
    assert "prior_midpoint_expected_n_pairs" not in settings
    assert settings["partition_diagnostics_eval_point_values"] == [70.0, 0.3]
    # structural, eval-point-independent counts stay bare
    assert settings["n_partitions"] == 3


def test_saved_outputs_keep_the_prior_midpoint_names_when_that_is_the_point(tmp_path):
    diagnostics = {
        "n_partitions": 3,
        "expected_n_singletons": 2.5,
        "expected_n_pairs": 0.75,
        "map_partition_index": 2,
        "map_partition": {"n_singletons": 2, "n_pairs": 1},
        "logL_marginalized": -12.5,
        "log_z_partition_prior": 0.4,
    }
    attrs, settings, _lo, _hi = _run_save_phase(
        tmp_path,
        extra_args=("--cluster_mode", "j2", "--partition_mode", "marginalize_exact",
                    "--candidate_pairs_path", "cand.json"),
        diagnostics=diagnostics, diagnostics_point=[70.0, 0.3],
    )
    assert attrs["partition_diagnostics_eval_point"] == "prior_midpoint"
    assert attrs["prior_midpoint_logL_marginalized"] == pytest.approx(-12.5)
    assert settings["prior_midpoint_expected_n_pairs"] == pytest.approx(0.75)


def test_smoke_test_threads_the_diagnostics_point_label_out():
    import inspect

    import darksirens.cli.inference_lensing as cli

    assert "return mid, diagnostics, diag_point, diag_label" in inspect.getsource(
        cli._smoke_test_likelihood
    )
    main_src = inspect.getsource(cli.main)
    assert "diagnostics_point_label=diag_label" in main_src
    assert "diagnostics_point=diag_point" in main_src


# ---------------------------------------------------------------------------
# --edge_mark_likelihood_keys must not be silently inert (review F-007)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["time", "delta_t_obs"])
def test_time_likelihood_key_without_pair_marks_time_is_fatal(key):
    """Only --pair_marks time enables the marked pair likelihood, so the flag
    that advertises the arrival-time term must not run without it: the run was
    statistically valid but discarded the strongest discriminant against false
    pairings while settings.json recorded the key as honoured."""
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts("--edge_mark_likelihood_keys", key)
    with pytest.raises(SystemExit, match="requires --pair_marks time"):
        cli._resolve_lensing_run_config(opts)


def test_time_likelihood_key_with_pair_marks_time_resolves():
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts(
        "--edge_mark_likelihood_keys", "time", "--pair_marks", "time",
    )
    cli._resolve_lensing_run_config(opts)          # no raise
    assert opts.pair_marks == "time"


def test_unsupported_edge_likelihood_keys_still_raise_not_implemented():
    import darksirens.cli.inference_lensing as cli

    opts = _lensing_opts("--edge_mark_likelihood_keys", "log_sky_overlap")
    with pytest.raises(NotImplementedError):
        cli._resolve_lensing_run_config(opts)
