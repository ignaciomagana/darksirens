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
        # memory-probed block sizes: resolved from live free VRAM before the
        # fingerprint is built, so they differ between two nodes / two requeues
        sel_batch_size=4096,
        block_size_static_state_bytes=1 << 33,
        # legacy tinyns checkpoint flags (they also name the evolving checkpoint)
        tinyns_checkpoint_path="/some/run/checkpoint.tinyns.npz",
        tinyns_checkpoint_path_out="/some/run/checkpoint.tinyns.npz",
        tinyns_resume_from="/some/run/checkpoint.tinyns.npz",
        tinyns_progress_interval=50,
    )
    for key, value in operational.items():
        changed = _fingerprint(_opts(**{key: value}))
        assert changed["digest"] == reference["digest"], (
            f"operational knob {key!r} changed the fingerprint digest; this "
            "would break every SLURM requeue that toggles it"
        )


def test_legacy_tinyns_checkpoint_is_never_hashed_as_an_input(tmp_path):
    """--tinyns_resume_from names the run's OWN checkpoint, whose bytes are
    rewritten every --tinyns_checkpoint_interval iterations: hashing it as an
    input file made the digest unstable between two consecutive resume attempts,
    so the gate was guaranteed to reject."""
    ckpt = tmp_path / "checkpoint.tinyns.npz"
    ckpt.write_bytes(b"\x00" * 64)
    reference = _fingerprint(_opts(tinyns_resume_from=str(ckpt)))
    ckpt.write_bytes(b"\x01" * 128)          # sampler wrote a new checkpoint
    assert _fingerprint(_opts(tinyns_resume_from=str(ckpt)))["digest"] == (
        reference["digest"]
    )


def test_memory_probed_block_sizes_do_not_change_the_digest():
    """The block-size resolvers overwrite ``opts.sel_batch_size`` (and stamp
    ``opts.block_size_static_state_bytes``) from PROBED FREE DEVICE MEMORY
    before the fingerprint is built, so a requeue onto a node with different
    free memory used to resolve a different integer and lock itself out of its
    own checkpoint (review F-002)."""
    submitted = dict(
        sel_batch_size=267520,
        block_size_static_state_bytes=1 << 30,
        block_size_resolution="blocked:auto",
    )
    requeued = dict(                    # same command line, emptier GPU
        sel_batch_size=356864,
        block_size_static_state_bytes=(1 << 30) + 4096,
        block_size_resolution="single-pass:auto",
    )
    reference = _fingerprint(_opts(**submitted))
    changed = _fingerprint(_opts(**requeued))
    assert changed["digest"] == reference["digest"], (
        "a memory-probed block size changed the fingerprint digest; --resume "
        "auto then refuses to continue its own checkpoint on a node with "
        "different free GPU memory"
    )
    # single pass (None) vs blocked must not split the digest either
    assert _fingerprint(_opts(**{**submitted, "sel_batch_size": None}))[
        "digest"
    ] == reference["digest"]


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
    dict(joint_constraints=[("ordered_le", (0, 1))]),
])
def test_semantic_changes_change_the_digest(mutation):
    reference = _fingerprint()
    opts = mutation.pop("opts", None)
    changed = _fingerprint(opts, **mutation)
    assert changed["digest"] != reference["digest"]


def test_joint_constraint_kind_is_fingerprinted(tmp_path):
    """The cube maps are a semantic input the labels/bounds cannot express.

    ``ordered_le`` and ``conditional_upper`` occupy the SAME ordered triangle
    with the same labels, the same bounds and the same per-parameter uniform
    families -- they differ only in the density they put on it (F-115: the
    conditional carries the 1/(x_j - lo) factor the sort does not).  Without
    this key the two configurations produced identical digests, so a
    checkpoint taken under one would resume under the other and mix two
    priors into one posterior and logZ.
    """
    labels = ["m2low", "m1low"]
    common = dict(
        labels=labels, lower_bound=[3.0, 3.0], upper_bound=[10.0, 10.0],
        prior_kinds=[("uniform", None, None)] * 2,
    )
    sorted_fp = _fingerprint(
        **common, joint_constraints=[("ordered_le", (0, 1))])
    conditional_fp = _fingerprint(
        **common, joint_constraints=[("conditional_upper", (0, 1))])

    assert sorted_fp["digest"] != conditional_fp["digest"]
    # The resume diff must NAME the map, not just report an opaque digest gap.
    save_run_fingerprint(str(tmp_path), sorted_fp)
    with pytest.raises(ResumeFingerprintError) as exc:
        check_resume_fingerprint(str(tmp_path), conditional_fp)
    assert "joint_constraints" in str(exc.value)
    assert "conditional_upper" in str(exc.value)


def test_schema_version_bump_refuses_pre_bump_fingerprints(tmp_path):
    """A fingerprint written before joint_constraints entered the semantic
    block cannot be compared against one written after: its digest was
    computed over a block that could not distinguish the GWTC-5 m_low prior
    from its replacement.  The gate must say so in the schema_version terms an
    operator can act on (re-run, or --resume_force after checking the model has
    no constraint groups), not silently accept."""
    current = _fingerprint()
    stale = json.loads(json.dumps(current))
    stale["schema_version"] = 1
    save_run_fingerprint(str(tmp_path), stale)

    with pytest.raises(ResumeFingerprintError) as exc:
        check_resume_fingerprint(str(tmp_path), current)
    assert "schema_version" in str(exc.value)

    with pytest.warns(RuntimeWarning, match="resume_force"):
        assert check_resume_fingerprint(
            str(tmp_path), current, force=True) is not None


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


def test_redshift_normalization_domain_is_fingerprinted(monkeypatch):
    """DARKSIRENS_ZMAX / *_ZNORM_HI change the integration domain of the
    redshift prior, the missing-galaxy budget and the selection integral -- i.e.
    both the numerator and beta -- so a requeue that resolves them differently
    must NOT be accepted against its own checkpoint."""
    from darksirens.redshift import grid as zgrid_module
    import darksirens.sky.models as sky_models

    reference = _fingerprint()
    block = reference["semantic"]["redshift_grid"]
    assert block["zMax"] == pytest.approx(float(zgrid_module.zMax))
    assert block["n_nodes"] == len(zgrid_module.zgrid)
    assert block["sky_znorm_hi"] == pytest.approx(float(sky_models._ZNORM_HI))

    monkeypatch.setattr(zgrid_module, "zMax", float(zgrid_module.zMax) + 1.0)
    assert _fingerprint()["digest"] != reference["digest"]
    monkeypatch.undo()

    monkeypatch.setattr(sky_models, "_ZNORM_HI", float(sky_models._ZNORM_HI) + 1.0)
    assert _fingerprint()["digest"] != reference["digest"]


def test_environment_block_records_the_darksirens_env(monkeypatch):
    """An archived logZ must be attributable to the zMax that produced it."""
    from darksirens.io.settings import environment_block

    monkeypatch.setenv("DARKSIRENS_ZMAX", "2.5")
    env = environment_block()["darksirens_env"]
    assert env["DARKSIRENS_ZMAX"] == "2.5"
    assert all(key.startswith("DARKSIRENS_") for key in env)


def test_flow_ensemble_directory_content_is_fingerprinted(tmp_path):
    """--gw_flows_path is a DIRECTORY, and for a flow-surrogate run those
    checkpoints ARE the PE likelihood: a retrained ensemble, or one event
    added/removed, must not pass the gate as an identical configuration."""
    flows = tmp_path / "flows"
    for name in ("EV1", "EV2"):
        (flows / name).mkdir(parents=True)
        (flows / name / f"{name}_flow.npz").write_bytes(b"\x00" * 32)

    opts = dict(gw_flows_path=str(flows), flows_pattern="*/*_flow.npz")
    reference = _fingerprint(_opts(**opts))
    slots = reference["semantic"]["data_files"]
    assert sum(k.startswith("gw_flows_path/") for k in slots) == 2
    assert _fingerprint(_opts(**opts))["digest"] == reference["digest"]

    # One checkpoint retrained in place (same path, same size).
    (flows / "EV2" / "EV2_flow.npz").write_bytes(b"\x00" * 31 + b"\x01")
    assert _fingerprint(_opts(**opts))["digest"] != reference["digest"]

    # An event added to the ensemble changes the event identity (sorted order).
    (flows / "EV2" / "EV2_flow.npz").write_bytes(b"\x00" * 32)
    (flows / "EV3").mkdir()
    (flows / "EV3" / "EV3_flow.npz").write_bytes(b"\x00" * 32)
    assert _fingerprint(_opts(**opts))["digest"] != reference["digest"]

def test_tinyns_resolved_config_does_not_smuggle_non_semantic_keys():
    """``build_tinyns_config`` mirrors the resolved config onto opts, and the
    fingerprint filters only TOP-LEVEL keys -- so the mirror used to re-admit
    ``show_progress`` and ``tinyns_checkpoint_interval`` (both declared
    non-semantic) through the nested dict, turning a requeue that changed the
    tinyns checkpoint cadence into a fatal resume mismatch."""
    from darksirens.inference.tinyns_config import (
        BASE_DEFAULTS, build_tinyns_config,
    )

    def _tinyns_opts(**over):
        kw = {f"tinyns_{name}": None for name in BASE_DEFAULTS}
        kw.update(over)
        opts = _opts(sampler="tinyns", tinyns_preset="recommended",
                     max_samples=1000, **kw)
        build_tinyns_config(opts)
        return opts

    reference = _fingerprint(_tinyns_opts())
    for key, value in (("tinyns_checkpoint_interval", 500),
                       ("show_progress", True)):
        changed = _fingerprint(_tinyns_opts(**{key: value}))
        assert changed["digest"] == reference["digest"], (
            f"{key!r} reached the digest through tinyns_resolved_config"
        )
    # Genuinely semantic tinyns knobs must still be caught.
    assert _fingerprint(_tinyns_opts(tinyns_walks=80))["digest"] != \
        reference["digest"]


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


def test_forced_mismatch_fingerprint_can_be_stamped_beside_the_stored_one(tmp_path):
    """A --resume_force run across a MISMATCH keeps the checkpoint's fingerprint
    (its true record) and records its own configuration beside it, so the
    directory never advertises only a configuration that did not produce its
    results.hdf5."""
    stored = _fingerprint()
    save_run_fingerprint(str(tmp_path), stored)
    current = _fingerprint(_opts(seed=18))
    path = save_run_fingerprint(
        str(tmp_path), current, basename="run_fingerprint.forced-TS.json"
    )
    assert os.path.basename(path) == "run_fingerprint.forced-TS.json"
    with open(os.path.join(tmp_path, FINGERPRINT_BASENAME)) as f:
        assert json.load(f)["digest"] == stored["digest"]
    with open(path) as f:
        assert json.load(f)["digest"] == current["digest"]
    assert not any(p.endswith(".tmp") for p in os.listdir(tmp_path))
