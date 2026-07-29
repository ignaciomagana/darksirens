"""Run-fingerprint resume gate (review finding P0-01).

``--resume`` restores nested-sampling state and rebinds the CURRENT run's
likelihood/prior closures onto it; before the fingerprint gate, the only
compatibility check was the dimension, so a resume under a different
statistical model (different inputs, bounds, fixed values, model flags)
silently mixed samples and evidence from two targets.  These tests pin:

* the digest is STABLE across operational knobs -- a requeued SLURM job with
  an identical command line (production submit scripts hard-code
  ``--resume auto``) must always match its own checkpoint;
* the digest is SENSITIVE to every class of semantic change, including the
  CONTENT of input files (a regenerated file at the same path must mismatch);
* the gate fails closed (missing/corrupt/mismatched fingerprint) and
  ``--resume_force`` is the only way past it, loudly.
"""
import json
import os
from types import SimpleNamespace

import pytest

from darksirens.inference.run_fingerprint import (
    FINGERPRINT_BASENAME,
    ResumeFingerprintError,
    build_run_fingerprint,
    check_resume_fingerprint,
    save_run_fingerprint,
)


def _opts(**overrides):
    base = dict(
        sampler="dynesty", seed=17, nlive=100, dlogz=0.1,
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        gw_path=None, fix_population=False,
        save_path="/nonexistent/save", resume="off", resume_force=False,
        checkpoint_interval="1800", show_progress=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fingerprint(opts=None, **kw):
    defaults = dict(
        labels=["H0"], lower_bound=[20.0], upper_bound=[140.0],
        prior_kinds=[("uniform", None, None)],
        prior_overrides=None, fixed_parameter_values=None,
    )
    defaults.update(kw)
    return build_run_fingerprint(opts if opts is not None else _opts(), **defaults)


# ---------------------------------------------------------------------------
# Digest stability: requeues must match themselves
# ---------------------------------------------------------------------------

def test_operational_knobs_do_not_change_the_digest():
    """A resume attempt legitimately differs from the original submission in
    exactly these knobs; none of them may break a requeue chain."""
    reference = _fingerprint()
    operational = dict(
        resume="auto",
        resume_force=True,
        checkpoint_interval="off",
        checkpoint_interval_seconds=0.0,
        checkpoint_file_resolved="/some/run/checkpoint.dynesty.pkl",
        resume_from_resolved="/some/run/checkpoint.dynesty.pkl",
        run_dir="/some/run",
        save_path="/a/completely/different/save",
        show_progress=True,
        dynesty_diagnostics=True,
    )
    for key, value in operational.items():
        changed = _fingerprint(_opts(**{key: value}))
        assert changed["digest"] == reference["digest"], (
            f"operational knob {key!r} changed the fingerprint digest; this "
            "would break every SLURM requeue that toggles it"
        )


def test_underscore_attributes_are_ignored():
    reference = _fingerprint()
    changed = _fingerprint(_opts(_private_scratch="anything"))
    assert changed["digest"] == reference["digest"]


# ---------------------------------------------------------------------------
# Digest sensitivity: every semantic class must be caught
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutation", [
    dict(opts=_opts(pop_model="powerlaw")),
    dict(opts=_opts(seed=18)),
    dict(opts=_opts(nlive=101)),
    dict(opts=_opts(fix_population=True)),
    dict(opts=_opts(new_model_flag=True)),        # a flag the ref lacks
    dict(labels=["H0", "Om0"], lower_bound=[20.0, 0.1],
         upper_bound=[140.0, 0.5],
         prior_kinds=[("uniform", None, None)] * 2),
    dict(upper_bound=[141.0]),
    dict(prior_kinds=[("gaussian", 70.0, 5.0)]),
    dict(prior_overrides={"H0": [30.0, 120.0]}),
    dict(fixed_parameter_values={"Om0": 0.3075}),
])
def test_semantic_changes_change_the_digest(mutation):
    reference = _fingerprint()
    opts = mutation.pop("opts", None)
    changed = _fingerprint(opts, **mutation)
    assert changed["digest"] != reference["digest"]


def test_input_file_content_is_fingerprinted(tmp_path):
    """Path strings alone cannot detect a regenerated input file."""
    payload = tmp_path / "gwsamples.h5"
    payload.write_bytes(b"\x00" * 64)
    reference = _fingerprint(_opts(gw_path=str(payload)))
    assert f"{os.sep}gwsamples.h5" in json.dumps(reference["semantic"])

    same = _fingerprint(_opts(gw_path=str(payload)))
    assert same["digest"] == reference["digest"]

    # Same path, same size, one byte different: must mismatch.
    payload.write_bytes(b"\x00" * 63 + b"\x01")
    regenerated = _fingerprint(_opts(gw_path=str(payload)))
    assert regenerated["digest"] != reference["digest"]


def test_list_valued_path_options_are_fingerprinted(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    reference = _fingerprint(_opts(lss_completions=[str(a), str(b)]))
    b.write_bytes(b"B")
    changed = _fingerprint(_opts(lss_completions=[str(a), str(b)]))
    assert changed["digest"] != reference["digest"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_gate_passes_on_exact_match(tmp_path):
    fp = _fingerprint()
    save_run_fingerprint(str(tmp_path), fp)
    stored = check_resume_fingerprint(str(tmp_path), _fingerprint())
    assert stored["digest"] == fp["digest"]


def test_gate_fails_closed_on_mismatch_and_names_the_culprit(tmp_path):
    save_run_fingerprint(str(tmp_path), _fingerprint())
    current = _fingerprint(_opts(seed=18))
    with pytest.raises(ResumeFingerprintError) as excinfo:
        check_resume_fingerprint(str(tmp_path), current)
    message = str(excinfo.value)
    assert "seed" in message
    assert "--resume_force" in message


def test_gate_force_overrides_mismatch_with_a_loud_warning(tmp_path):
    save_run_fingerprint(str(tmp_path), _fingerprint())
    current = _fingerprint(_opts(seed=18))
    with pytest.warns(RuntimeWarning, match="mismatch"):
        stored = check_resume_fingerprint(str(tmp_path), current, force=True)
    assert stored is not None


def test_gate_fails_closed_on_missing_fingerprint(tmp_path):
    with pytest.raises(ResumeFingerprintError, match="run_fingerprint.json"):
        check_resume_fingerprint(str(tmp_path), _fingerprint())
    with pytest.warns(RuntimeWarning, match="WITHOUT a fingerprint"):
        stored = check_resume_fingerprint(str(tmp_path), _fingerprint(), force=True)
    assert stored is None


def test_gate_fails_closed_on_corrupt_fingerprint(tmp_path):
    (tmp_path / FINGERPRINT_BASENAME).write_text("{not json")
    with pytest.raises(ResumeFingerprintError, match="unreadable"):
        check_resume_fingerprint(str(tmp_path), _fingerprint())


def test_code_identity_drift_warns_but_does_not_block(tmp_path):
    """A requeue that straddles a `git pull` must proceed (advisory only)."""
    fp = _fingerprint()
    stored = json.loads(json.dumps(fp))
    stored["advisory"]["code"]["git_sha"] = "0" * 40
    save_run_fingerprint(str(tmp_path), stored)
    current = _fingerprint()
    if current["advisory"]["code"].get("git_sha") in (None, "unknown", "0" * 40):
        pytest.skip("no usable git identity in this environment")
    with pytest.warns(RuntimeWarning, match="code identity"):
        out = check_resume_fingerprint(str(tmp_path), current)
    assert out is not None


# ---------------------------------------------------------------------------
# Single-resolution contract for the checkpoint plan
# ---------------------------------------------------------------------------

def test_resolve_checkpoint_plan_honours_the_callers_resume_target(tmp_path):
    """An `auto` target must not be re-resolved between the run-directory
    decision and the plan: the caller's resolution is authoritative."""
    from darksirens.inference.checkpointing import (
        CHECKPOINT_BASENAMES, resolve_checkpoint_plan,
    )

    run_dir = tmp_path / "existing_run"
    run_dir.mkdir()
    ckpt = run_dir / CHECKPOINT_BASENAMES["dynesty"]
    ckpt.write_bytes(b"")

    # opts say `auto` (which WOULD find ckpt), but the caller already decided
    # not to resume: the explicit None wins.
    opts = SimpleNamespace(
        sampler="dynesty", resume="auto", save_path=str(tmp_path),
        checkpoint_interval="1800",
    )
    plan = resolve_checkpoint_plan(opts, str(run_dir), resume_from=None)
    assert plan.resume_from is None

    # And the caller's explicit target wins over a fresh glob.
    plan = resolve_checkpoint_plan(opts, str(run_dir), resume_from=str(ckpt))
    assert plan.resume_from == str(ckpt)
