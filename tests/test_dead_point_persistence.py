"""Nested-sampler DEAD POINTS survive into results.hdf5 (logl_dead/logwt_dead).

A nested sampler produces two different point sets, and only one of them used to
be archived.  ``results["samples"]`` is the EQUAL-WEIGHT resample: rows drawn
with replacement against exp(logwt), which throws the shrinkage record away.
The dead points -- every retired live point with its own logL and log importance
weight -- are what logZ, the logX ladder, the information H, an evidence
bootstrap or a dynesty runplot are computed from, and once the resample has been
taken they cannot be recovered.  For the lensing paper, whose headline is a logZ
difference, that record is the primary product.

The pre-existing ``log_weights``/``log_likelihood`` datasets are per POSTERIOR
SAMPLE (numpyro writes them), so the dead-point arrays are additive, under names
of their own, with an explicit length contract:

    logl_dead, logwt_dead : (n_dead,)   n_dead = niter + n_live

The trap this file pins down is that for dynesty n_dead happens to EQUAL
n_samples -- ``resample_equal`` returns as many rows as it was handed -- so a
reader that zips ``samples`` with ``logl_dead`` gets no error, just nonsense.
For tinyns the two differ outright (n_samples is the posterior ESS).
"""
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import jax.numpy as jnp

from darksirens.inference.checkpointing import resolve_checkpoint_plan
from darksirens.inference.sampling import _dead_point_block, run_sampler
from darksirens.io.results import (
    DEAD_POINT_SEMANTICS,
    save_results_hdf5,
    write_dead_point_datasets,
)

LABELS = ["a", "b"]
LOWER = np.array([-5.0, -5.0])
UPPER = np.array([5.0, 5.0])
NLIVE = 40


def _loglike(theta):
    return -0.5 * jnp.sum(jnp.asarray(theta) ** 2)


def _ptform(u):
    return 10.0 * (jnp.asarray(u) - 0.5)


def _sampler_opts(sampler, run_dir, **overrides):
    settings = dict(
        sampler=sampler, seed=17, show_progress=False, nlive=NLIVE, dlogz=1.0,
        max_samples=0, dynesty_diagnostics=False, sampler_preflight="off",
        save_path=str(run_dir), checkpoint_interval="off", resume="off",
    )
    settings.update(overrides)
    opts = SimpleNamespace(**settings)
    resolve_checkpoint_plan(opts, str(run_dir))
    return opts


# ---------------------------------------------------------------------------
# Both nested samplers hand back a dead-point block
# ---------------------------------------------------------------------------

def test_dynesty_returns_the_dead_point_record(tmp_path):
    pytest.importorskip("dynesty")
    results = run_sampler(
        "dynesty", _loglike, _ptform, LABELS, LOWER, UPPER,
        _sampler_opts("dynesty", tmp_path),
    )
    dead = results["dead_points"]

    assert dead["logl"].shape == dead["logwt"].shape == (dead["n_dead"],)
    assert dead["n_live"] == NLIVE
    assert dead["n_dead"] > NLIVE          # the run retired more than one shell
    assert np.all(np.isfinite(dead["logl"]))

    # THE trap: dynesty's equal-weight resample has exactly as many rows as
    # there are dead points, so the lengths agree while the point sets do not.
    assert dead["n_dead"] == results["samples"].shape[0]
    # Dead points come out in retirement (increasing-logL) order; a with-
    # replacement resample never does. This is the row-alignment contract.
    assert np.all(np.diff(dead["logl"]) >= 0.0)


def test_tinyns_returns_the_dead_point_record(tmp_path):
    pytest.importorskip("tinyns")
    results = run_sampler(
        "tinyns", _loglike, _ptform, LABELS, LOWER, UPPER,
        _sampler_opts("tinyns", tmp_path, max_samples=200,
                      tinyns_preset="recommended"),
    )
    dead = results["dead_points"]

    assert dead["logl"].shape == dead["logwt"].shape == (dead["n_dead"],)
    assert dead["n_live"] == NLIVE
    assert np.all(np.isfinite(dead["logl"]))

    # tinyns' record is the dead points plus the final live points, so the
    # length contract is exactly niter + n_live ...
    niter = results["tinyns_runtime_diagnostics"]["niter"]
    assert dead["n_dead"] == int(niter) + NLIVE
    # ... and it does NOT match the equal-weight resample, whose length is the
    # posterior ESS.
    assert dead["n_dead"] != results["samples"].shape[0]


def test_zero_free_parameter_shortcircuit_has_no_dead_points():
    """No sampler ran, so there is no shrinkage record to archive; the exact
    logZ = logL(fixed point) path must not invent one."""
    results = run_sampler(
        "tinyns", lambda theta: jnp.asarray(-1.25), _ptform, [],
        np.zeros(0), np.zeros(0), SimpleNamespace(),
    )
    assert results.get("dead_points") is None
    assert results["log_likelihood"].shape == (1,)   # per-sample, unchanged


# ---------------------------------------------------------------------------
# _dead_point_block: never kill a multi-day run at the save step
# ---------------------------------------------------------------------------

def test_dead_point_block_returns_none_without_arrays():
    assert _dead_point_block(None, None) is None
    assert _dead_point_block(np.zeros(3), None, n_live=5) is None


def test_dead_point_block_refuses_mismatched_arrays(capsys):
    assert _dead_point_block(np.zeros(3), np.zeros(4), n_live=5) is None
    assert "dead-point" in capsys.readouterr().out


def test_dead_point_block_flattens_and_records_lengths():
    block = _dead_point_block(np.arange(4.0), np.arange(4.0), n_live=2)
    assert block["n_dead"] == 4 and block["n_live"] == 2
    assert block["logl"].dtype == np.float64


def test_dead_point_block_without_n_live_omits_the_attr(tmp_path):
    """A dynamic dynesty Results carries samples_n, not nlive."""
    block = _dead_point_block(np.arange(3.0), np.arange(3.0))
    assert "n_live" not in block
    with h5py.File(tmp_path / "r.hdf5", "w") as f:
        assert write_dead_point_datasets(f, {"dead_points": block})
        assert "n_live" not in f.attrs
        assert f.attrs["n_dead"] == 3


# ---------------------------------------------------------------------------
# results.hdf5 round trip
# ---------------------------------------------------------------------------

def _opts(**overrides):
    base = dict(
        pop_model="powerlaw", universe_model="spectral_sirens",
        sky_model="isotropic", mark_model="none", mark_names=(),
        complete_empty_pixel_policy="zero", sampler="dynesty",
        fix_cosmology=False, fix_de=False, fix_population=False,
        fix_survey=False, gw_path="gw.h5", gwselection_path="sel.h5",
        pdet_flow_path=None, survey_path="",
        nlive=NLIVE, dlogz=0.1, tinyns_sample=None, tinyns_kernel=None,
        tinyns_walks=None, tinyns_replacement_chains=None,
        tinyns_replacement_chain_schedule=None, tinyns_max_attempts=None,
        tinyns_preset="recommended", tinyns_bound=None, tinyns_step_scale=None,
        tinyns_resolved_config=None,
        nuts_warmup=0, nuts_samples=0, nuts_chains=0, nuts_target_accept=0.0,
        nuts_max_tree_depth=0, nuts_init_tries=0, nuts_init_seed_offset=0,
        seed=123,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _meta():
    return {
        "n_events": 0, "n_samp_per_event": 0, "n_draw": 0,
        "total_runtime": "0:00:00", "sampling_runtime": "0:00:00",
        "timestamp": "2026-07-27T00-00-00",
    }


def _results_with_dead_points(n_samples=5, n_dead=13):
    rng = np.random.default_rng(0)
    return {
        "samples": rng.normal(size=(n_samples, 1)),
        "logZ": -1.0,
        "logZerr": 0.1,
        "dead_points": {
            "logl": np.sort(rng.normal(size=n_dead)),
            "logwt": rng.normal(size=n_dead),
            "n_dead": n_dead,
            "n_live": NLIVE,
        },
    }


def _save(tmp_path, results, opts=None):
    return save_results_hdf5(
        results, str(tmp_path), ["H0"], [10.0], [100.0], {}, {},
        opts or _opts(), _meta(),
    )


def test_save_results_hdf5_round_trips_dead_points(tmp_path):
    results = _results_with_dead_points()
    path = _save(tmp_path, results)
    with h5py.File(path, "r") as f:
        assert f["logl_dead"].shape == (13,)
        assert f["logwt_dead"].shape == (13,)
        np.testing.assert_allclose(f["logl_dead"][()],
                                   results["dead_points"]["logl"])
        np.testing.assert_allclose(f["logwt_dead"][()],
                                   results["dead_points"]["logwt"])
        assert f.attrs["n_dead"] == 13
        assert f.attrs["n_live"] == NLIVE
        # The semantics are stated in the file itself, not only in the source.
        assert f.attrs["dead_points"] == DEAD_POINT_SEMANTICS
        # ... and the length contract is genuinely independent of n_samples.
        assert f.attrs["n_samples"] == 5
        assert f["samples"].shape == (5, 1)


def test_dead_points_do_not_repurpose_the_per_sample_datasets(tmp_path):
    """log_weights/log_likelihood keep their (N_samples,) contract while the
    dead-point arrays sit alongside them at their own length."""
    results = _results_with_dead_points(n_samples=5, n_dead=13)
    results["log_weights"] = np.zeros(5)
    results["log_likelihood"] = np.ones(5)
    path = _save(tmp_path, results)
    with h5py.File(path, "r") as f:
        assert f["log_weights"].shape == (5,)
        assert f["log_likelihood"].shape == (5,)
        np.testing.assert_allclose(f["log_likelihood"][()], np.ones(5))
        assert f["logl_dead"].shape == (13,)


def test_no_dead_points_writes_nothing_new(tmp_path):
    """numpyro (and every legacy caller) must produce a byte-for-byte legacy
    layout: no new datasets, no new attrs."""
    path = _save(tmp_path, {"samples": np.zeros((2, 1)), "logZ": -1.0,
                            "logZerr": 0.1})
    with h5py.File(path, "r") as f:
        assert "logl_dead" not in f
        assert "logwt_dead" not in f
        for attr in ("n_dead", "n_live", "dead_points"):
            assert attr not in f.attrs


def test_dead_points_are_gzip_compressed_like_the_other_arrays(tmp_path):
    path = _save(tmp_path, _results_with_dead_points())
    with h5py.File(path, "r") as f:
        assert f["logl_dead"].compression == "gzip"


# ---------------------------------------------------------------------------
# Legacy readers are unaffected
# ---------------------------------------------------------------------------

def test_analyze_load_run_is_unaffected_by_dead_points(tmp_path):
    """cli/analyze's loader reads samples + evidence; the additive datasets
    must be invisible to it."""
    from darksirens.cli.analyze import load_run

    results = _results_with_dead_points()
    _save(tmp_path, results)
    settings, samples, logZ, logZerr = load_run(str(tmp_path))

    np.testing.assert_allclose(samples, results["samples"])
    assert logZ == -1.0
    assert logZerr == pytest.approx(0.1)
    assert settings["labels"] == ["H0"]
    # The attrs are merged into settings (every attr is), which must not
    # shadow anything the analyzer relies on.
    assert settings["n_dead"] == 13
    assert settings["n_samples"] == 5


def test_bayes_factor_reader_is_unaffected_by_dead_points(tmp_path):
    from darksirens.cli.analyze import load_run

    for tag, logZ in (("a", -10.0), ("b", -12.5)):
        run = tmp_path / tag
        run.mkdir()
        results = _results_with_dead_points()
        results["logZ"] = logZ
        _save(run, results)
    assert load_run(str(tmp_path / "a"))[2] - load_run(str(tmp_path / "b"))[2] \
        == pytest.approx(2.5)


def test_a_legacy_file_without_dead_points_still_loads(tmp_path):
    """Runs archived before this schema existed keep working unchanged."""
    from darksirens.cli.analyze import load_run

    _save(tmp_path, {"samples": np.zeros((2, 1)), "logZ": -3.0, "logZerr": 0.2})
    settings, samples, logZ, _ = load_run(str(tmp_path))
    assert samples.shape == (2, 1) and logZ == -3.0
    assert "n_dead" not in settings


# ---------------------------------------------------------------------------
# End to end: a real dynesty run through the real writer
# ---------------------------------------------------------------------------

def test_dynesty_run_persists_its_own_dead_points(tmp_path):
    pytest.importorskip("dynesty")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = run_sampler(
        "dynesty", _loglike, _ptform, LABELS, LOWER, UPPER,
        _sampler_opts("dynesty", run_dir),
    )
    path = save_results_hdf5(
        results, str(run_dir), LABELS, list(LOWER), list(UPPER), {}, {},
        _opts(), _meta(),
    )
    assert os.path.basename(path) == "results.hdf5"
    with h5py.File(path, "r") as f:
        n_dead = int(f.attrs["n_dead"])
        assert f["logl_dead"].shape == (n_dead,)
        assert f.attrs["n_live"] == NLIVE
        # logZ is recomputable from the persisted record alone, which is the
        # whole point of keeping it.
        logz = float(np.logaddexp.reduce(f["logwt_dead"][()]))
        assert logz == pytest.approx(float(f.attrs["logZ"]), abs=1e-6)
