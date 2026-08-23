"""Sampler checkpoint/resume: a killed run must be resumable, not a total loss.

Before these fixes a nested-sampling run that died after sampling started lost
everything:

* dynesty was called without ``checkpoint_file``/``checkpoint_every`` at all,
  so a SLURM TIMEOUT at hour 47 discarded 47 h of likelihood evaluations (the
  failure mode that halted the 13-job lensing campaign on 2026-07-21);
* tinyns' checkpoint flag raised ``TypeError: '<=' not supported between
  instances of 'NoneType' and 'int'`` before the first iteration whenever
  ``--tinyns_checkpoint_interval`` was omitted, because ``checkpoint_interval``
  defaulted to None and was forwarded unconditionally;
* the main CLI created its run directory only AFTER sampling returned, so
  there was nowhere for a checkpoint to live and no settings.json to
  reconstruct the configuration from.

The dynesty test here interrupts a real run mid-iteration and resumes it to
completion, and pins the reproducibility contract: the checkpoint carries the
``--seed``-derived Generator, so resume never draws fresh entropy.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dynesty")

import jax.numpy as jnp

from darksirens.inference.checkpointing import (
    CHECKPOINT_BASENAMES,
    DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    find_resume_target,
    parse_checkpoint_interval,
    plan_from_opts,
    resolve_checkpoint_plan,
)
from darksirens.inference.sampling import run_sampler
from darksirens.inference.tinyns_config import (
    build_tinyns_config,
    tinyns_run_kwargs,
)


LABELS = ["a", "b"]
LOWER = np.array([-5.0, -5.0])
UPPER = np.array([5.0, 5.0])


def _loglike(theta):
    return -0.5 * jnp.sum(jnp.asarray(theta) ** 2)


def _ptform(u):
    return 10.0 * (jnp.asarray(u) - 0.5)


class _SimulatedKill(RuntimeError):
    """Stands in for SIGTERM/OOM: dies mid-run with no chance to save."""


def _loglike_that_dies_after(n_calls):
    counter = {"n": 0}

    def f(theta):
        counter["n"] += 1
        if counter["n"] > n_calls:
            raise _SimulatedKill(f"killed after {n_calls} likelihood calls")
        return _loglike(theta)

    return f, counter


def _opts(save_path, run_dir, sampler, **overrides):
    # checkpoint_interval=0.001 s makes dynesty checkpoint essentially every
    # iteration, so the test does not have to wait 30 production minutes.
    settings = dict(
        sampler=sampler, seed=17, show_progress=False, nlive=40, dlogz=1.0,
        max_samples=0, dynesty_diagnostics=False, sampler_preflight="off",
        save_path=str(save_path), checkpoint_interval="0.001", resume="off",
    )
    settings.update(overrides)
    opts = SimpleNamespace(**settings)
    resolve_checkpoint_plan(opts, str(run_dir))
    return opts


# ---------------------------------------------------------------------------
# --checkpoint_interval / --resume resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("1800", 1800.0), (1800, 1800.0), ("600s", 600.0), ("0.5", 0.5),
    ("off", 0.0), ("OFF", 0.0), ("none", 0.0), ("0", 0.0), (None, 0.0),
])
def test_parse_checkpoint_interval(spec, expected):
    assert parse_checkpoint_interval(spec) == expected


@pytest.mark.parametrize("spec", ["fortnightly", "-1", "1e"])
def test_parse_checkpoint_interval_rejects_garbage(spec):
    with pytest.raises(ValueError):
        parse_checkpoint_interval(spec)


def test_checkpointing_is_on_by_default():
    """Production default: a timeout costs at most one interval, not the run."""
    assert DEFAULT_CHECKPOINT_INTERVAL_SECONDS > 0


def test_resume_auto_starts_fresh_when_no_checkpoint(tmp_path):
    """--resume auto must be safe to hard-code in a submit script."""
    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (None, None)


def test_resume_auto_picks_newest_checkpoint_and_its_run_dir(tmp_path):
    older = tmp_path / "run_old"
    newer = tmp_path / "run_new"
    for d in (older, newer):
        d.mkdir()
        (d / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    os.utime(older / CHECKPOINT_BASENAMES["dynesty"], (1_000_000, 1_000_000))

    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    ckpt, run_dir = find_resume_target(opts, "dynesty")
    assert run_dir == str(newer)
    assert ckpt == str(newer / CHECKPOINT_BASENAMES["dynesty"])


def _write_complete_results_hdf5(path):
    """A results.hdf5 the way a finished writer leaves it (completion marker)."""
    import h5py

    from darksirens.io.results import atomic_result_hdf5

    with atomic_result_hdf5(str(path)) as f:
        f.create_dataset("samples", data=np.zeros((2, 1)))
        f.attrs["labels"] = '["H0"]'
    with h5py.File(str(path), "r") as f:
        assert f.attrs["result_complete"]


def test_resume_auto_skips_finished_runs(tmp_path):
    """A run directory holding a COMPLETE results.hdf5 is a run that converged;
    its end-of-run checkpoint must not hijack a fresh submission."""
    done = tmp_path / "run_done"
    done.mkdir()
    (done / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    _write_complete_results_hdf5(done / "results.hdf5")
    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (None, None)
    # An explicit path still resumes it (re-saving a converged run is a valid
    # recovery from a crash inside the save step).
    opts.resume = str(done)
    ckpt, run_dir = find_resume_target(opts, "dynesty")
    assert run_dir == str(done)


def test_resume_auto_is_scoped_to_this_configuration(tmp_path):
    """One --save_path holds many runs; auto must not resume another model's
    or another seed's half-finished sampler."""
    mine = tmp_path / "pl+peak__dark_sirens__dynesty__seed4001__T1"
    theirs = tmp_path / "pl+peak__dark_sirens__dynesty__seed4002__T2"
    for d in (mine, theirs):
        d.mkdir()
        (d / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    # theirs is newer, so an unscoped auto would pick it.
    os.utime(mine / CHECKPOINT_BASENAMES["dynesty"], (1_000_000, 1_000_000))

    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    prefix = "pl+peak__dark_sirens__dynesty__seed4001__"
    _, run_dir = find_resume_target(opts, "dynesty", name_prefix=prefix)
    assert run_dir == str(mine)


def test_resume_ignores_the_other_samplers_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / CHECKPOINT_BASENAMES["tinyns"]).write_bytes(b"x")
    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    assert find_resume_target(opts, "dynesty") == (None, None)


def test_resume_explicit_missing_path_is_an_error(tmp_path):
    opts = SimpleNamespace(sampler="dynesty", resume=str(tmp_path / "nope"),
                           save_path=str(tmp_path))
    with pytest.raises(ValueError):
        find_resume_target(opts, "dynesty")


def test_resume_rejects_samplers_without_checkpointing(tmp_path):
    opts = SimpleNamespace(sampler="numpyro", resume="auto", save_path=str(tmp_path))
    with pytest.raises(ValueError):
        find_resume_target(opts, "numpyro")


def test_resolve_plan_places_checkpoint_in_run_dir_and_records_it(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    opts = SimpleNamespace(sampler="tinyns", checkpoint_interval="1800",
                           resume="off", save_path=str(tmp_path))
    plan = resolve_checkpoint_plan(opts, str(run_dir))
    assert plan.enabled and plan.interval_seconds == 1800.0
    assert plan.path == str(run_dir / CHECKPOINT_BASENAMES["tinyns"])
    # Mirrored on opts so settings.json records the resolved decision.
    assert opts.checkpoint_file_resolved == plan.path
    assert opts.checkpoint_interval_seconds == 1800.0
    assert opts.resume_from_resolved is None
    # And rebuildable inside run_sampler.
    assert plan_from_opts(opts, "tinyns") == plan


def test_checkpoint_interval_off_disables(tmp_path):
    opts = SimpleNamespace(sampler="dynesty", checkpoint_interval="off",
                           resume="off", save_path=str(tmp_path))
    plan = resolve_checkpoint_plan(opts, str(tmp_path))
    assert not plan.enabled and plan.path is None


def test_bare_namespace_callers_get_checkpointing_off():
    """Library/scripts callers of run_sampler must not start writing files."""
    plan = plan_from_opts(SimpleNamespace(), "dynesty")
    assert not plan.enabled and not plan.resuming


# ---------------------------------------------------------------------------
# dynesty: interrupt a real run and resume it to completion
# ---------------------------------------------------------------------------

def _run_dynesty(save_path, run_dir, likelihood, **overrides):
    opts = _opts(save_path, run_dir, "dynesty", **overrides)
    return run_sampler("dynesty", likelihood, _ptform, LABELS, LOWER, UPPER, opts)


def test_dynesty_interrupted_run_resumes_to_completion(tmp_path):
    run_dir = tmp_path / "run_interrupted"
    run_dir.mkdir()
    ckpt = run_dir / CHECKPOINT_BASENAMES["dynesty"]

    # (a) start sampling with checkpointing on, (b) get killed mid-run.
    # The uninterrupted run needs ~860 likelihood calls at these settings, so
    # 400 kills it roughly halfway through.
    dying, counter = _loglike_that_dies_after(400)
    with pytest.raises(_SimulatedKill):
        _run_dynesty(tmp_path, run_dir, dying)
    assert counter["n"] > 400
    assert ckpt.is_file(), "no checkpoint survived the simulated kill"
    # State only: a checkpoint that pickled the JAX likelihood closure would be
    # huge (or, in practice, would refuse to pickle at all).
    assert ckpt.stat().st_size < 1_000_000

    # (c) resume from the checkpoint and finish.
    resumed_calls = {"n": 0}

    def counted_resume(theta):
        resumed_calls["n"] += 1
        return _loglike(theta)

    resumed = _run_dynesty(tmp_path, run_dir, counted_resume, resume=str(run_dir))
    assert np.all(np.isfinite(resumed["samples"]))
    assert resumed["samples"].shape[1] == len(LABELS)
    assert np.isfinite(resumed["logZ"])

    # Seed contract: the checkpoint carries the --seed-derived Generator, so the
    # resumed run continues that exact stream instead of drawing fresh entropy.
    fresh_dir = tmp_path / "run_uninterrupted"
    fresh_dir.mkdir()
    reference_calls = {"n": 0}

    def counted_reference(theta):
        reference_calls["n"] += 1
        return _loglike(theta)

    reference = _run_dynesty(
        tmp_path, fresh_dir, counted_reference, checkpoint_interval="off"
    )
    assert resumed["logZ"] == pytest.approx(reference["logZ"], rel=1e-12)
    assert resumed["samples"].shape == reference["samples"].shape
    np.testing.assert_array_equal(resumed["samples"], reference["samples"])

    # The whole point: the resumed leg did NOT redo the work already in the
    # checkpoint, so it needed materially fewer likelihood evaluations than a
    # cold start to reach the same answer.
    assert resumed_calls["n"] < reference_calls["n"]


def test_resumed_dynesty_reports_the_actual_nlive(tmp_path):
    """P2-19: on resume the checkpoint's live-point count wins over --nlive
    (warned about), so the results must record the ACTUAL value the sampler
    ran with, with the request kept beside it for provenance."""
    run_dir = tmp_path / "run_nlive"
    run_dir.mkdir()

    dying, _ = _loglike_that_dies_after(400)
    with pytest.raises(_SimulatedKill):
        _run_dynesty(tmp_path, run_dir, dying)  # nlive=40 checkpoints

    # Requeue asks for a DIFFERENT nlive; the checkpoint's 40 wins.
    resumed = _run_dynesty(
        tmp_path, run_dir, _loglike, resume=str(run_dir), nlive=64
    )
    assert resumed["nlive_actual"] == 40
    # results.hdf5's actual-vs-requested attrs are pinned in
    # tests/test_results_saving.py::test_nlive_actual_wins_over_the_request.


def test_dynesty_resume_of_a_finished_run_is_idempotent(tmp_path):
    """Requeueing a job whose run already converged must not redo the run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = _run_dynesty(tmp_path, run_dir, _loglike)
    assert (run_dir / CHECKPOINT_BASENAMES["dynesty"]).is_file()

    second = _run_dynesty(tmp_path, run_dir, _loglike, resume=str(run_dir))
    assert second["logZ"] == pytest.approx(first["logZ"], rel=1e-12)


def test_dynesty_diagnostics_land_in_the_run_dir():
    """Two concurrent jobs sharing --save_path used to overwrite each other's
    runplot_0001.pdf: the plots went to the shared root while the plot counter
    restarts at 0 in every process."""
    from darksirens.inference.sampling import dynesty_diagnostics_dir

    per_run = dynesty_diagnostics_dir(
        SimpleNamespace(save_path="/out", run_dir="/out/run_a")
    )
    assert per_run == os.path.join("/out", "run_a", "dynesty_diagnostics")
    # Library callers without a run_dir keep the historical location.
    assert dynesty_diagnostics_dir(SimpleNamespace(save_path="/out")) == os.path.join(
        "/out", "dynesty_diagnostics"
    )


# ---------------------------------------------------------------------------
# tinyns: the TypeError, and an end-to-end resume
# ---------------------------------------------------------------------------

def _tinyns_opts(save_path, run_dir, **overrides):
    settings = dict(nlive=20, max_samples=25, checkpoint_interval="1800")
    settings.update(overrides)
    return _opts(save_path, run_dir, "tinyns", **settings)


def test_tinyns_run_kwargs_forward_a_positive_interval_when_flag_omitted():
    """The C-1 regression: an omitted --tinyns_checkpoint_interval used to
    forward None, defeating tinyns' own default and making its
    ``checkpoint_interval <= 0`` guard raise a bare TypeError."""
    cfg = build_tinyns_config(SimpleNamespace(
        tinyns_checkpoint_path="x.npz", nlive=10, dlogz=0.1, max_samples=10,
        seed=0, show_progress=False,
    ))
    kwargs = tinyns_run_kwargs(cfg)
    assert isinstance(kwargs["checkpoint_interval"], int)
    assert kwargs["checkpoint_interval"] > 0


def test_tinyns_run_kwargs_omit_a_none_interval():
    """Belt and braces: never forward None even if a config carries one."""
    cfg = build_tinyns_config(SimpleNamespace(
        nlive=10, dlogz=0.1, max_samples=10, seed=0, show_progress=False,
    ))
    cfg = type(cfg)(**{**cfg.to_json_dict(), "replacement_chain_schedule": None,
                       "explicit": (), "checkpoint_interval": None})
    assert "checkpoint_interval" not in tinyns_run_kwargs(cfg)


def test_tinyns_config_rejects_nonpositive_interval_with_a_checkpoint():
    with pytest.raises(ValueError):
        build_tinyns_config(SimpleNamespace(
            tinyns_checkpoint_path="x.npz", tinyns_checkpoint_interval=0,
            nlive=10, dlogz=0.1, max_samples=10, seed=0, show_progress=False,
        ))


@pytest.mark.parametrize("interval", [0, -5])
def test_tinyns_config_rejects_nonpositive_interval_without_a_config_path(
        interval):
    """The guard used to require a CONFIG-level checkpoint path, but
    run_sampler also enables tinyns checkpointing from the shared
    --checkpoint_interval plan (invisible to the validator): with only
    --tinyns_checkpoint_interval 0 the value reached tinyns and raised after the
    catalog load, KDE-cache build and likelihood trace."""
    with pytest.raises(ValueError, match="checkpoint_interval must be a positive"):
        build_tinyns_config(SimpleNamespace(
            tinyns_checkpoint_interval=interval,
            nlive=10, dlogz=0.1, max_samples=10, seed=0, show_progress=False,
        ))


def test_tinyns_checkpoints_without_an_explicit_interval_and_resumes(tmp_path):
    """End-to-end: this whole test used to die with the C-1 TypeError before
    the first nested-sampling iteration."""
    pytest.importorskip("tinyns")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = run_dir / CHECKPOINT_BASENAMES["tinyns"]

    opts = _tinyns_opts(tmp_path, run_dir)
    # Checkpointing is on via --checkpoint_interval alone; no explicit
    # --tinyns_checkpoint_interval anywhere in this test.
    assert not hasattr(opts, "tinyns_checkpoint_interval")
    first = run_sampler("tinyns", _loglike, _ptform, LABELS, LOWER, UPPER, opts)
    assert first["samples"].shape[1] == len(LABELS)
    assert ckpt.is_file(), "tinyns wrote no checkpoint"

    resume_opts = _tinyns_opts(tmp_path, run_dir, resume=str(run_dir),
                               max_samples=60)
    assert resume_opts.resume_from_resolved == str(ckpt)
    resumed = run_sampler(
        "tinyns", _loglike, _ptform, LABELS, LOWER, UPPER, resume_opts,
    )
    assert resumed["samples"].shape[1] == len(LABELS)
    assert np.isfinite(resumed["logZ"])
    # It carried on from the checkpoint's 25 iterations rather than restarting.
    niter = resumed.get("tinyns_runtime_diagnostics", {}).get("niter")
    assert niter is None or niter > 25


def test_tinyns_prints_the_resolved_iteration_cap_and_call_equivalent(
        tmp_path, capsys):
    """--max_samples is dynesty's CALL cap but tinyns' ITERATION cap, and one
    rwalk iteration costs walks x max_active_chains evaluations, so the same
    number buys a different budget per sampler.  The resolved cap and its call
    equivalent must be in the log (the old code left a dead `maxiter` local and
    a comment claiming the two budgets were comparable)."""
    pytest.importorskip("tinyns")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    opts = _tinyns_opts(tmp_path, run_dir, max_samples=20)
    run_sampler("tinyns", _loglike, _ptform, LABELS, LOWER, UPPER, opts)
    out = capsys.readouterr().out
    assert "tinyns iteration cap: maxiter=20 (--max_samples)" in out
    # recommended preset: walks=5, replacement_chains=1 -> 100 calls.
    assert "up to ~100 likelihood calls" in out


def test_tinyns_run_and_resample_keys_are_independent_streams(tmp_path):
    """P2-20: the run key and the posterior-resample key must be distinct
    split outputs of the seed key.  The old flow handed PRNGKey(seed) itself
    to sampler.run and later split that SAME key for resampling — split() is
    pure, so the resample stream overlapped whatever subkeys the sampler
    derived internally.  Determinism contract: same seed => identical
    samples; different seed => different resample."""
    pytest.importorskip("tinyns")
    import inspect

    from darksirens.inference import sampling as sampling_mod

    src = inspect.getsource(sampling_mod.run_sampler)
    assert "run_key, resample_key = jax.random.split" in src, (
        "run/resample keys must be split from the seed BEFORE the run"
    )
    assert "sampler.run(\n                run_key," in src

    d1 = tmp_path / "a"; d1.mkdir()
    d2 = tmp_path / "b"; d2.mkdir()
    r1 = run_sampler("tinyns", _loglike, _ptform, LABELS, LOWER, UPPER,
                     _tinyns_opts(tmp_path, d1))
    r2 = run_sampler("tinyns", _loglike, _ptform, LABELS, LOWER, UPPER,
                     _tinyns_opts(tmp_path, d2))
    np.testing.assert_array_equal(r1["samples"], r2["samples"])


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_both_clis_expose_the_same_checkpoint_flags():
    from darksirens.cli.inference import build_parser as main_parser
    from darksirens.cli.inference_lensing import build_parser as lensing_parser

    main = main_parser().parse_args(["--sampler", "dynesty"])
    lensing = lensing_parser().parse_args([
        "--sampler", "dynesty", "--gw_path", "gw.h5",
        "--gwselection_path", "inj.h5",
    ])
    for opts in (main, lensing):
        assert opts.checkpoint_interval == str(int(DEFAULT_CHECKPOINT_INTERVAL_SECONDS))
        assert opts.resume == "off"


def test_main_cli_writes_the_run_dir_and_settings_before_sampling(tmp_path):
    """C-3: a run killed during sampling used to leave --save_path empty."""
    from darksirens.cli.inference import (
        build_parser, _prepare_run_dir, _ParameterSpace, _write_failure,
    )

    opts = build_parser().parse_args(
        ["--sampler", "dynesty", "--save_path", str(tmp_path)]
    )
    pspace = _ParameterSpace(
        labels=["H0"], lower_bound=[50.0], upper_bound=[100.0],
        prior_kinds=[("uniform", None, None)], prior_transform=None,
        pop_params_fid={}, n_pop_eff=0, n_cosmo_eff=1, n_survey_eff=0,
        model_name="test",
    )
    data = {"nEvents": 3, "nsamp": 4, "Ndraw": 5}

    run_dir, timestamp, settings = _prepare_run_dir(opts, data, pspace, {}, {})

    # settings.json exists BEFORE any sampling happened.
    with open(os.path.join(run_dir, "settings.json")) as fh:
        saved = json.load(fh)
    assert saved["run_status"] == "sampling"
    assert saved["timestamp"] == timestamp
    assert saved["labels"] == ["H0"]
    assert saved["n_events"] == 3
    # The checkpoint target is inside the run directory and recorded on disk.
    assert saved["checkpoint_file_resolved"] == os.path.join(
        run_dir, CHECKPOINT_BASENAMES["dynesty"]
    )
    assert opts.run_dir == run_dir
    assert settings["run_dir"] == run_dir

    # And a death during sampling leaves a diagnosable record.
    try:
        raise RuntimeError("host-RAM OOM")
    except RuntimeError as exc:
        _write_failure(run_dir, "sampler", exc, labels=["H0"], settings=settings)
    with open(os.path.join(run_dir, "failure.json")) as fh:
        failure = json.load(fh)
    assert failure["stage"] == "sampler"
    assert failure["error_type"] == "RuntimeError"
    assert "host-RAM OOM" in failure["error_message"]


def test_main_cli_resume_reuses_the_original_run_dir(tmp_path):
    from darksirens.cli.inference import build_parser, _prepare_run_dir, _ParameterSpace

    pspace = _ParameterSpace(
        labels=["H0"], lower_bound=[50.0], upper_bound=[100.0],
        prior_kinds=[("uniform", None, None)], prior_transform=None,
        pop_params_fid={}, n_pop_eff=0, n_cosmo_eff=1, n_survey_eff=0,
        model_name="test",
    )
    data = {"nEvents": 3, "nsamp": 4, "Ndraw": 5}
    args = ["--sampler", "dynesty", "--save_path", str(tmp_path)]

    first, _, _ = _prepare_run_dir(
        build_parser().parse_args(args), data, pspace, {}, {}
    )
    # Pretend the first attempt checkpointed and was then killed.
    open(os.path.join(first, CHECKPOINT_BASENAMES["dynesty"]), "wb").close()

    opts = build_parser().parse_args(args + ["--resume", "auto"])
    second, _, _ = _prepare_run_dir(opts, data, pspace, {}, {})
    assert second == first, "a requeued run must not sprout a new run directory"
    assert opts.resume_from_resolved == os.path.join(
        first, CHECKPOINT_BASENAMES["dynesty"]
    )


def _test_pspace(upper=100.0):
    from darksirens.cli.inference import _ParameterSpace

    return _ParameterSpace(
        labels=["H0"], lower_bound=[50.0], upper_bound=[upper],
        prior_kinds=[("uniform", None, None)], prior_transform=None,
        pop_params_fid={}, n_pop_eff=0, n_cosmo_eff=1, n_survey_eff=0,
        model_name="test",
    )


def test_main_cli_resume_rejects_a_different_configuration(tmp_path):
    """P0-01: the run-name prefix only carries model/sampler/seed, so before
    the fingerprint gate a same-prefix run with different bounds, fixed
    values, or inputs resumed silently -- mixing nested-sampling state from
    two statistical targets into one output."""
    from darksirens.cli.inference import build_parser, _prepare_run_dir

    data = {"nEvents": 3, "nsamp": 4, "Ndraw": 5}
    args = ["--sampler", "dynesty", "--save_path", str(tmp_path)]

    first, _, _ = _prepare_run_dir(
        build_parser().parse_args(args), data, _test_pspace(100.0), {}, {}
    )
    assert os.path.isfile(os.path.join(first, "run_fingerprint.json"))
    # Pretend the first attempt checkpointed and was then killed.
    open(os.path.join(first, CHECKPOINT_BASENAMES["dynesty"]), "wb").close()

    # Same prefix, different prior bound: must refuse to resume.
    with pytest.raises(SystemExit):
        _prepare_run_dir(
            build_parser().parse_args(args + ["--resume", "auto"]),
            data, _test_pspace(120.0), {}, {},
        )

    # Same prefix, different fixed value: must refuse to resume.
    with pytest.raises(SystemExit):
        _prepare_run_dir(
            build_parser().parse_args(args + ["--resume", "auto"]),
            data, _test_pspace(100.0), {"Om0": 0.3}, {},
        )

    # --resume_force is the explicit, loud, expert override.
    opts = build_parser().parse_args(
        args + ["--resume", "auto", "--resume_force"]
    )
    with pytest.warns(RuntimeWarning, match="mismatch"):
        second, _, _ = _prepare_run_dir(opts, data, _test_pspace(120.0), {}, {})
    assert second == first


def test_main_cli_resume_preserves_the_original_settings_json(tmp_path):
    """The original settings.json is the record of the configuration that
    created the checkpoint; a resume attempt must not overwrite it."""
    from darksirens.cli.inference import build_parser, _prepare_run_dir

    data = {"nEvents": 3, "nsamp": 4, "Ndraw": 5}
    args = ["--sampler", "dynesty", "--save_path", str(tmp_path)]

    first, _, _ = _prepare_run_dir(
        build_parser().parse_args(args), data, _test_pspace(), {}, {}
    )
    open(os.path.join(first, CHECKPOINT_BASENAMES["dynesty"]), "wb").close()
    with open(os.path.join(first, "settings.json")) as fh:
        original = fh.read()

    second, _, _ = _prepare_run_dir(
        build_parser().parse_args(args + ["--resume", "auto"]),
        data, _test_pspace(), {}, {},
    )
    assert second == first
    with open(os.path.join(first, "settings.json")) as fh:
        assert fh.read() == original, "resume overwrote the original settings.json"
    resumes = [
        name for name in os.listdir(first)
        if name.startswith("settings.resume-") and name.endswith(".json")
    ]
    assert len(resumes) == 1, "each resume attempt must leave its own record"


# ---------------------------------------------------------------------------
# Full main() lifecycle: kill it during sampling, requeue with --resume auto
# ---------------------------------------------------------------------------

def _stub_pipeline(monkeypatch, likelihood):
    """Stub only the data / parameter-space seams of the main CLI.

    Everything this batch touched -- _prepare_run_dir, run_sampler's dynesty
    branch, _failure_guard, _save_outputs -- then runs for real, including the
    settings.json and results.hdf5 writes.  (The repo's mock-data generator
    needs a newer numpy than this environment has, so the tiny synthetic
    end-to-end run lives here instead of in scripts/smoke_tests.)
    """
    import darksirens.cli.inference as cli
    from darksirens.inference.prior import make_prior_transform

    data = {"nEvents": 5, "nsamp": 64, "Ndraw": 3000}
    pspace = cli._ParameterSpace(
        labels=list(LABELS), lower_bound=LOWER, upper_bound=UPPER,
        prior_kinds=[("uniform", None, None)] * len(LABELS),
        prior_transform=make_prior_transform(LOWER, UPPER),
        pop_params_fid={}, n_pop_eff=0, n_cosmo_eff=len(LABELS), n_survey_eff=0,
        model_name="synthetic",
    )
    monkeypatch.setattr(cli, "_load_and_report_data", lambda opts: data)
    monkeypatch.setattr(cli, "_maybe_run_completion_validation",
                        lambda *a, **k: False)
    monkeypatch.setattr(cli, "_resolve_single_catalog_marks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_build_and_report_parameter_space",
                        lambda *a, **k: pspace)
    monkeypatch.setattr(cli, "_build_likelihood", lambda *a, **k: likelihood)
    return cli


def _cli_args(tmp_path, extra=()):
    gw = tmp_path / "gw.h5"
    sel = tmp_path / "inj.h5"
    for path in (gw, sel):
        path.write_bytes(b"")
    return [
        "--sampler", "dynesty", "--gw_path", str(gw),
        "--gwselection_path", str(sel), "--universe_model", "spectral_sirens",
        "--save_path", str(tmp_path / "out"), "--nlive", "40", "--dlogz", "1.0",
        "--seed", "17", "--show_progress", "false",
        "--sampler_preflight", "off", "--checkpoint_interval", "0.001",
        *extra,
    ]


def test_main_cli_killed_during_sampling_resumes_with_auto(tmp_path, monkeypatch):
    """The production scenario end to end: a run dies mid-sampling, leaves a
    settings.json / failure.json / checkpoint behind, and the identical command
    plus --resume auto finishes it inside the SAME run directory."""
    out = tmp_path / "out"
    dying, _ = _loglike_that_dies_after(400)

    cli = _stub_pipeline(monkeypatch, dying)
    with pytest.raises(_SimulatedKill):
        cli.main(_cli_args(tmp_path))

    run_dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "the killed run left no run directory"
    run_dir = run_dirs[0]
    # A killed run is now diagnosable AND resumable.
    assert (run_dir / "settings.json").is_file()
    assert (run_dir / "failure.json").is_file()
    assert (run_dir / CHECKPOINT_BASENAMES["dynesty"]).is_file()
    with open(run_dir / "settings.json") as fh:
        assert json.load(fh)["run_status"] == "sampling"
    with open(run_dir / "failure.json") as fh:
        failure = json.load(fh)
    assert failure["stage"] == "sampler"
    assert failure["error_type"] == "_SimulatedKill"
    assert not (run_dir / "results.hdf5").exists()

    # Requeue the identical command plus --resume auto.
    cli = _stub_pipeline(monkeypatch, _loglike)
    cli.main(_cli_args(tmp_path, ["--resume", "auto"]))

    assert [d for d in out.iterdir() if d.is_dir()] == [run_dir], \
        "the resumed run opened a second run directory"
    assert (run_dir / "samples.npy").is_file()
    assert (run_dir / "results.hdf5").is_file()
    with open(run_dir / "settings.json") as fh:
        final = json.load(fh)
    assert final["run_status"] == "complete"
    assert final["sampling_runtime"] is not None
    assert final["resume_from_resolved"] == str(
        run_dir / CHECKPOINT_BASENAMES["dynesty"]
    )
    # The chain is on disk independently of the HDF5.
    chain = np.load(run_dir / "samples.npy")
    assert chain.ndim == 2 and chain.shape[1] == len(LABELS)
