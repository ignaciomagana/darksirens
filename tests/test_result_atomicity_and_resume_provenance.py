"""results.hdf5 durability and forced-resume provenance, for BOTH CLIs.

Two failure modes, both reproduced against the pre-fix code:

* LENS-01 — the lensing CLI opened the final results.hdf5 directly in "w" mode.
  A fault injected into ``write_dead_point_datasets`` left a truncated file
  holding ``["samples"]`` and no attrs.  That file is poison twice over:
  ``find_resume_target`` reads any results.hdf5 as "this run finished" and
  skips the checkpoint, and ``analyze.load_run`` prefers it over the
  samples.npy recovery chain.  The main CLI already wrote temp-then-replace;
  the writer is now shared, deletes the temporary on BaseException, and stamps
  a completion marker that resume and analyze both require.

* LENS-02 — the lensing fingerprint gate discarded the stored fingerprint
  returned by ``check_resume_fingerprint``, so a ``--resume_force`` across a
  MISMATCH (output that mixes two statistical targets by construction) left no
  digest, no ``resume_forced_mismatch`` flag, no
  ``run_fingerprint.forced-<ts>.json``, and no HDF5 provenance.  Both CLIs now
  run the same shared gate.
"""
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from darksirens.inference.checkpointing import (
    CHECKPOINT_BASENAMES,
    find_resume_target,
)
from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    FINGERPRINT_SCHEMA_VERSION,
    gate_and_stamp_resume_fingerprint,
    resume_provenance_attrs,
    save_run_fingerprint,
)
from darksirens.io.results import (
    RESULT_COMPLETE_ATTR,
    atomic_result_hdf5,
    result_is_complete,
)


# ---------------------------------------------------------------------------
# LENS-01: the shared atomic writer
# ---------------------------------------------------------------------------

def test_atomic_writer_publishes_a_marked_file(tmp_path):
    path = tmp_path / "results.hdf5"
    with atomic_result_hdf5(str(path)) as f:
        f.create_dataset("samples", data=np.zeros((4, 2)))
    assert path.exists()
    assert not (tmp_path / "results.hdf5.tmp").exists()
    with h5py.File(str(path), "r") as f:
        assert bool(f.attrs[RESULT_COMPLETE_ATTR])
    assert result_is_complete(str(path))


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_atomic_writer_leaves_nothing_behind_on_a_fault(tmp_path, exc_type):
    """A SLURM SIGTERM mid-write is a BaseException, not an Exception."""
    path = tmp_path / "results.hdf5"
    with pytest.raises(exc_type):
        with atomic_result_hdf5(str(path)) as f:
            f.create_dataset("samples", data=np.zeros((4, 2)))
            raise exc_type("injected fault mid-write")
    assert not path.exists(), "a partial results.hdf5 was published"
    assert not (tmp_path / "results.hdf5.tmp").exists(), "temporary left behind"


def test_atomic_writer_keeps_the_previous_complete_result(tmp_path):
    path = tmp_path / "results.hdf5"
    with atomic_result_hdf5(str(path)) as f:
        f.create_dataset("samples", data=np.full((3, 1), 7.0))
    with pytest.raises(RuntimeError):
        with atomic_result_hdf5(str(path)) as f:
            f.create_dataset("samples", data=np.zeros((3, 1)))
            raise RuntimeError("injected fault mid-write")
    with h5py.File(str(path), "r") as f:
        assert float(f["samples"][0, 0]) == 7.0


def _write_partial_result(path):
    """What the pre-fix lensing writer left: samples, then a fault."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("samples", data=np.zeros((4, 2)))


def test_a_truncated_result_is_not_complete(tmp_path):
    path = tmp_path / "results.hdf5"
    _write_partial_result(path)
    assert not result_is_complete(str(path))


def test_a_legacy_unmarked_result_is_still_accepted(tmp_path):
    """Archives written before the marker existed must keep loading."""
    path = tmp_path / "results.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("samples", data=np.zeros((4, 2)))
        f.create_dataset("labels", data=np.array(["H0", "Om0"], dtype="S"))
    assert result_is_complete(str(path))
    with h5py.File(str(path), "r") as f:
        assert RESULT_COMPLETE_ATTR not in f.attrs


def test_a_legacy_grouped_layout_result_is_still_accepted(tmp_path):
    """The grouped posterior/samples layout with only evidence aliases in attrs
    -- an archive shape analyze already supports."""
    path = tmp_path / "results.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_group("posterior").create_dataset("samples", data=np.zeros((4, 1)))
        f.attrs["log_evidence"] = 1.25
    assert result_is_complete(str(path))


def test_a_non_hdf5_or_missing_file_is_not_complete(tmp_path):
    (tmp_path / "junk.hdf5").write_bytes(b"not hdf5")
    assert not result_is_complete(str(tmp_path / "junk.hdf5"))
    assert not result_is_complete(str(tmp_path / "absent.hdf5"))


def test_auto_resume_recovers_a_run_whose_final_write_died(tmp_path):
    """The LENS-01 poisoning, end to end: a truncated results.hdf5 must NOT
    lock the run out of the checkpoint that would rebuild it."""
    run = tmp_path / "run_crashed"
    run.mkdir()
    (run / CHECKPOINT_BASENAMES["dynesty"]).write_bytes(b"x")
    _write_partial_result(run / "results.hdf5")
    opts = SimpleNamespace(sampler="dynesty", resume="auto", save_path=str(tmp_path))
    ckpt, run_dir = find_resume_target(opts, "dynesty")
    assert run_dir == str(run)
    assert ckpt == str(run / CHECKPOINT_BASENAMES["dynesty"])


def test_analyze_falls_back_to_the_recovery_chain_on_a_partial_result(tmp_path):
    from darksirens.cli.analyze import load_run

    run = tmp_path / "run"
    run.mkdir()
    _write_partial_result(run / "results.hdf5")
    np.save(str(run / "samples.npy"), np.full((5, 1), 3.0))
    (run / "settings.json").write_text(json.dumps({"labels": ["H0"]}))
    settings, samples, logZ, logZerr = load_run(str(run))
    assert samples.shape == (5, 1) and float(samples[0, 0]) == 3.0
    assert logZ is None and logZerr is None
    assert settings["labels"] == ["H0"]


def test_analyze_refuses_a_partial_result_with_no_recovery_chain(tmp_path):
    from darksirens.cli.analyze import load_run

    run = tmp_path / "run"
    run.mkdir()
    _write_partial_result(run / "results.hdf5")
    with pytest.raises(ValueError, match="PARTIAL result"):
        load_run(str(run))


def test_lensing_save_publishes_nothing_when_the_final_write_faults(tmp_path):
    """The reproduction from the review, against the real lensing save path."""
    import darksirens.cli.inference_lensing as cli

    from tests.test_lensing_cli_defects import _lensing_opts

    def boom(f, results):
        f.create_dataset("logl_dead", data=np.zeros(3))
        raise RuntimeError("injected fault mid-write")

    opts = _lensing_opts()
    cli._resolve_lensing_run_config(opts)
    lens_labels, lens_lower, lens_upper = cli._build_lens_parameter_space(
        opts, {}, {}
    )
    labels = ["H0"] + list(lens_labels)
    mid = 0.5 * (
        np.concatenate([[50.0], lens_lower]) + np.concatenate([[100.0], lens_upper])
    )
    results = {"samples": np.zeros((3, len(labels))), "logZ": -1.0, "logZerr": 0.1}
    inp = {"nEvents": 4, "n_singletons": 4, "n_pairs": 0}

    saved = cli.write_dead_point_datasets
    cli.write_dead_point_datasets = boom
    try:
        with pytest.raises(RuntimeError, match="injected fault"):
            cli._save_lensing_outputs(
                opts, str(tmp_path), {}, inp, results, {}, labels, mid,
                {}, {}, {}, {},
            )
    finally:
        cli.write_dead_point_datasets = saved

    assert not (tmp_path / "results.hdf5").exists(), (
        "a truncated results.hdf5 was published: auto-resume would read it as "
        "completion and analyze would read it as the posterior"
    )
    assert not (tmp_path / "results.hdf5.tmp").exists()
    # samples.npy is written first on purpose: it IS the recovery chain.
    assert (tmp_path / "samples.npy").exists()


def test_lensing_save_marks_a_finished_result(tmp_path):
    from tests.test_lensing_cli_defects import _run_save_phase

    attrs, _settings, _lo, _hi = _run_save_phase(tmp_path)
    assert bool(attrs[RESULT_COMPLETE_ATTR])
    assert result_is_complete(str(tmp_path / "results.hdf5"))


# ---------------------------------------------------------------------------
# LENS-02: the shared forced-resume gate
# ---------------------------------------------------------------------------

def _fingerprint(digest, **semantic):
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": digest,
        "semantic": dict(semantic),
        "advisory": {"code": {}},
    }


def _gate_opts(**kw):
    base = dict(resume_force=False, resume_from_resolved=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fresh_run_stamps_the_fingerprint_and_no_mismatch(tmp_path):
    opts = _gate_opts()
    stored = gate_and_stamp_resume_fingerprint(
        opts, str(tmp_path), None, _fingerprint("aaa", H0=[50, 100]), "TS"
    )
    assert stored is None
    assert opts.run_fingerprint_digest == "aaa"
    assert opts.resume_forced_mismatch is False
    assert (tmp_path / FINGERPRINT_BASENAME).exists()


def test_matched_resume_records_the_digest_without_a_forced_flag(tmp_path):
    fp = _fingerprint("aaa", H0=[50, 100])
    save_run_fingerprint(str(tmp_path), fp)
    opts = _gate_opts(resume_from_resolved=str(tmp_path / "checkpoint.dynesty.pkl"))
    stored = gate_and_stamp_resume_fingerprint(
        opts, str(tmp_path), str(tmp_path), fp, "TS"
    )
    assert stored["digest"] == "aaa"
    assert opts.run_fingerprint_digest == "aaa"
    assert opts.resume_forced_mismatch is False
    assert not list(tmp_path.glob("run_fingerprint.forced-*.json"))


def test_forced_mismatch_records_digest_flag_and_sibling_fingerprint(tmp_path):
    save_run_fingerprint(str(tmp_path), _fingerprint("aaa", H0=[50, 100]))
    new = _fingerprint("bbb", H0=[20, 200])
    opts = _gate_opts(
        resume_force=True,
        resume_from_resolved=str(tmp_path / "checkpoint.dynesty.pkl"),
    )
    with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint"):
        stored = gate_and_stamp_resume_fingerprint(
            opts, str(tmp_path), str(tmp_path), new, "2026-08-23T00-00-00"
        )
    assert stored["digest"] == "aaa"
    assert opts.run_fingerprint_digest == "bbb"
    assert opts.resume_forced_mismatch is True
    # The stored fingerprint stays: it is the record of what made the checkpoint.
    with open(tmp_path / FINGERPRINT_BASENAME) as fh:
        assert json.load(fh)["digest"] == "aaa"
    sibling = tmp_path / "run_fingerprint.forced-2026-08-23T00-00-00.json"
    assert sibling.exists()
    with open(sibling) as fh:
        assert json.load(fh)["digest"] == "bbb"
    prov = resume_provenance_attrs(opts)
    assert prov["resume_forced"] and prov["resume_forced_mismatch"]
    assert prov["run_fingerprint_digest"] == "bbb"
    assert prov["resumed"]


def test_unforced_mismatch_still_refuses(tmp_path):
    save_run_fingerprint(str(tmp_path), _fingerprint("aaa", H0=[50, 100]))
    opts = _gate_opts(resume_from_resolved="ckpt")
    seen = []
    gate_and_stamp_resume_fingerprint(
        opts, str(tmp_path), str(tmp_path), _fingerprint("bbb", H0=[20, 200]),
        "TS", on_error=seen.append,
    )
    assert seen and "does not match" in seen[0]


def test_lensing_forced_mismatch_leaves_provenance_in_hdf5(tmp_path):
    """LENS-02 end to end: drive the lensing gate across a forced mismatch and
    read the provenance back out of the lensing results.hdf5."""
    import darksirens.cli.inference_lensing as cli

    from tests.test_lensing_cli_defects import _lensing_opts, _run_save_phase

    opts = _lensing_opts("--resume", str(tmp_path), "--resume_force")
    cli._resolve_lensing_run_config(opts)
    lens_labels, lens_lower, lens_upper = cli._build_lens_parameter_space(
        opts, {}, {}
    )
    labels = ["H0"] + list(lens_labels)
    lower = np.concatenate([[50.0], lens_lower])
    upper = np.concatenate([[100.0], lens_upper])
    prior_kinds = ["uniform"] * len(labels)

    # A stored fingerprint from a DIFFERENT target.
    save_run_fingerprint(str(tmp_path), _fingerprint("stale-digest", H0=[1, 2]))
    opts.resume_from_resolved = str(tmp_path / "checkpoint.tinyns.npz")
    with pytest.warns(RuntimeWarning, match="DESPITE a fingerprint"):
        cli._gate_or_stamp_resume_fingerprint(
            opts, str(tmp_path), str(tmp_path), labels, lower, upper,
            prior_kinds, {},
        )
    assert opts.resume_forced_mismatch is True
    digest = opts.run_fingerprint_digest
    assert digest and digest != "stale-digest"
    assert list(tmp_path.glob("run_fingerprint.forced-*.json"))

    # Now the save phase must archive that verdict.
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path, opts=opts)
    assert attrs["run_fingerprint_digest"] == digest
    assert bool(attrs["resume_forced"])
    assert bool(attrs["resume_forced_mismatch"])
    assert bool(attrs["resumed"])
    assert settings["resume_forced_mismatch"] is True
    assert settings["run_fingerprint_digest"] == digest


def test_lensing_unforced_mismatch_exits(tmp_path):
    import darksirens.cli.inference_lensing as cli

    from tests.test_lensing_cli_defects import _lensing_opts

    opts = _lensing_opts("--resume", str(tmp_path))
    cli._resolve_lensing_run_config(opts)
    lens_labels, lens_lower, lens_upper = cli._build_lens_parameter_space(
        opts, {}, {}
    )
    labels = ["H0"] + list(lens_labels)
    lower = np.concatenate([[50.0], lens_lower])
    upper = np.concatenate([[100.0], lens_upper])
    save_run_fingerprint(str(tmp_path), _fingerprint("stale-digest", H0=[1, 2]))
    with pytest.raises(SystemExit, match="does not match"):
        cli._gate_or_stamp_resume_fingerprint(
            opts, str(tmp_path), str(tmp_path), labels, lower, upper,
            ["uniform"] * len(labels), {},
        )


def test_main_cli_result_carries_the_forced_mismatch_flag(tmp_path):
    """The main CLI's archive keeps its provenance block through the shared
    helper (io.results now builds it from resume_provenance_attrs)."""
    from darksirens.io.results import save_results_hdf5

    from tests.test_results_saving import _meta, _opts, _results

    opts = _opts(
        resume_force=True, resume_forced_mismatch=True,
        resume_from_resolved=str(tmp_path / "checkpoint.dynesty.pkl"),
        run_fingerprint_digest="bbb",
    )
    path = save_results_hdf5(
        _results(), str(tmp_path), ["H0"], [50.0], [100.0], {}, {}, opts, _meta(),
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["run_fingerprint_digest"] == "bbb"
        assert bool(f.attrs["resume_forced_mismatch"])
        assert bool(f.attrs[RESULT_COMPLETE_ATTR])
    assert not os.path.exists(path + ".tmp")
