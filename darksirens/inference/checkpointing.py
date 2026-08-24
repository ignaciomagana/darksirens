"""Sampler checkpoint/resume policy shared by both inference CLIs.

Why this module exists
---------------------
Until this was added, a nested-sampling run that died after sampling started
was a TOTAL loss: dynesty was called without any of its checkpoint arguments,
tinyns' checkpoint flag raised a ``TypeError`` before the first iteration, and
the main CLI did not even create its run directory until sampling had already
returned.  The July 2026 lensing campaign was halted after 13 SLURM jobs were
killed by wall-clock timeouts and host-RAM OOM with nothing recoverable on
disk.  Every multi-day run must now be resumable.

The user-facing contract (identical in both CLIs)
-------------------------------------------------
``--checkpoint_interval SECONDS``
    Wall-clock cadence for sampler checkpoints, written INTO THE RUN
    DIRECTORY.  Default 1800 s (30 min), i.e. checkpointing is ON by default
    and a timeout costs at most half an hour.  ``off``/``none``/``0``
    disables it.

    * dynesty consumes this natively (``run_nested(checkpoint_every=...)``),
      so the cadence is exact.
    * tinyns counts ITERATIONS, not seconds (``checkpoint_interval`` in
      ``tinyns.NestedSampler.run``).  For tinyns this flag therefore only
      switches checkpointing on/off and places the file; the cadence is
      ``--tinyns_checkpoint_interval`` iterations (default 100).

``--resume auto|PATH|off``
    ``auto`` finds the most recently modified checkpoint for the selected
    sampler under ``--save_path`` and CONTINUES INSIDE THAT RUN DIRECTORY
    (so a requeued SLURM job keeps one run directory rather than sprouting a
    new one per attempt); with no checkpoint found it silently starts fresh,
    which is what makes ``--resume auto`` safe to hard-code in a submit
    script.  ``PATH`` may name either a run directory or a checkpoint file
    and must exist.  ``off`` (the default) always starts fresh.

    ``auto`` is deliberately narrow, because one ``--save_path`` normally
    holds many runs: it only considers run directories whose name carries
    THIS run's configuration prefix (model/sampler/seed — the CLI passes
    ``name_prefix``), and it skips any directory that already holds a
    COMPLETE ``results.hdf5``, i.e. a run that finished.  A results.hdf5 that
    merely exists is not enough: a final write killed half-way leaves a
    truncated file, and treating that as completion would disable the very
    recovery the checkpoint is for.  So a SLURM array that varies
    only ``--seed`` resumes each member into its own directory, and a fresh
    submission after a successful run starts fresh instead of silently
    "resuming" a converged sampler.

Seed / reproducibility contract
-------------------------------
Resuming never introduces fresh entropy.  dynesty's checkpoint pickles the
``numpy.random.Generator`` that ``--seed`` created, and the restored sampler
keeps drawing from that exact stream; tinyns checkpoints its ``PRNGKey``.
``tests/test_sampler_checkpoint_resume.py`` asserts that an
interrupted-then-resumed dynesty run reproduces the uninterrupted run
bit-for-bit, which is the same property ``tests/test_dynesty_seed.py`` pins
for a single-shot run.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

# Safe at module scope only because io.results defers its likelihood/sampler
# imports to call time: at module scope it must never reach back into
# inference.sampling (which imports this module), or the two deadlock on
# partial init.
from darksirens.io.results import result_is_complete


# One checkpoint file per sampler, so a run directory can hold a dynesty and a
# tinyns attempt without them fighting over the same name.
CHECKPOINT_BASENAMES = {
    "dynesty": "checkpoint.dynesty.pkl",
    "tinyns": "checkpoint.tinyns.npz",
}

# 30 min: short enough that a wall-clock kill costs at most one interval of
# likelihood evaluations, long enough that the checkpoint I/O (tens of kB for
# dynesty, a small .npz for tinyns) is free at production nlive.
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 1800.0

_OFF_WORDS = {"off", "none", "no", "false", "disabled", ""}


@dataclass(frozen=True)
class CheckpointPlan:
    """Resolved checkpoint/resume decision for one sampler run."""

    sampler: str
    enabled: bool
    interval_seconds: float
    path: str | None
    resume_from: str | None

    @property
    def resuming(self) -> bool:
        return self.resume_from is not None

    def summary(self) -> str:
        if self.sampler not in CHECKPOINT_BASENAMES:
            return "not supported for this sampler"
        if not self.enabled:
            parts = ["off"]
        elif self.sampler == "tinyns":
            # tinyns' cadence is iteration-based; do not pretend otherwise.
            parts = [f"{os.path.basename(self.path)} (every N iterations, see "
                     "--tinyns_checkpoint_interval)"]
        else:
            parts = [f"{os.path.basename(self.path)} every "
                     f"{self.interval_seconds:g} s"]
        if self.resuming:
            parts.append(f"resuming from {self.resume_from}")
        return "; ".join(parts)


def add_checkpoint_arguments(parser_or_group):
    """Register the shared ``--checkpoint_interval`` / ``--resume`` flags."""
    g = parser_or_group
    g.add_argument(
        "--checkpoint_interval",
        default=str(int(DEFAULT_CHECKPOINT_INTERVAL_SECONDS)),
        metavar="SECONDS|off",
        help=(
            "Seconds between sampler checkpoints written into the run "
            f"directory (default {int(DEFAULT_CHECKPOINT_INTERVAL_SECONDS)}; "
            "'off'/'0' disables). dynesty honours the wall-clock cadence "
            "exactly; tinyns checkpoints every "
            "--tinyns_checkpoint_interval ITERATIONS and this flag only "
            "enables/disables it. Resume the run with --resume auto."
        ),
    )
    g.add_argument(
        "--resume",
        default="off",
        metavar="auto|PATH|off",
        help=(
            "Restore a checkpointed run and continue it. 'auto' picks the "
            "most recently modified checkpoint under --save_path for this "
            "sampler AND this configuration (model/sampler/seed prefix), "
            "skipping run directories that already hold a results.hdf5, and "
            "continues inside that run directory; with nothing to resume it "
            "starts fresh silently, so it is safe to hard-code in a SLURM "
            "submit script. PATH may be a run directory or a checkpoint file "
            "and must exist. 'off' (default) always starts fresh. Resuming "
            "requires the run directory's run_fingerprint.json to match this "
            "run's full semantic configuration (inputs by content, priors, "
            "fixed values, model flags, sampler settings)."
        ),
    )
    g.add_argument(
        "--resume_force",
        action="store_true",
        default=False,
        help=(
            "UNSAFE: resume even when the checkpoint's run_fingerprint.json "
            "is missing or does not match this configuration. The resumed "
            "sampler state then mixes two statistical targets and the output "
            "is not a science result. Exists for deliberate expert use and "
            "for continuing checkpoints created before fingerprinting."
        ),
    )


def parse_checkpoint_interval(spec) -> float:
    """``spec`` -> seconds (0.0 meaning "checkpointing disabled")."""
    if spec is None:
        return 0.0
    if isinstance(spec, bool):
        raise ValueError("--checkpoint_interval must be seconds or 'off'.")
    if isinstance(spec, (int, float)):
        seconds = float(spec)
    else:
        text = str(spec).strip().lower()
        if text in _OFF_WORDS:
            return 0.0
        if text.endswith("s"):
            text = text[:-1].strip()
        try:
            seconds = float(text)
        except ValueError as exc:
            raise ValueError(
                f"--checkpoint_interval {spec!r} is not a number of seconds "
                "or 'off'."
            ) from exc
    if seconds < 0.0:
        raise ValueError("--checkpoint_interval must be >= 0 ('off' or 0 disables).")
    return seconds


def _resume_spec(opts) -> str:
    spec = getattr(opts, "resume", None)
    return "" if spec is None else str(spec).strip()


def find_resume_target(opts, sampler, name_prefix=None):
    """Resolve ``--resume`` to ``(checkpoint_path, run_dir)``.

    Returns ``(None, None)`` when not resuming (``off``, or ``auto`` with no
    resumable checkpoint on disk).  Raises ``ValueError`` for an explicit path
    that does not exist or holds no checkpoint for this sampler, and for
    ``--resume`` with a sampler that has no checkpoint support.

    ``name_prefix`` (the CLI's run-name prefix: model/sampler/seed) restricts
    ``auto`` to run directories belonging to THIS configuration, so a shared
    ``--save_path`` holding several models cannot hand a job someone else's
    half-finished sampler.
    """
    spec = _resume_spec(opts)
    if spec.lower() in _OFF_WORDS:
        return None, None

    basename = CHECKPOINT_BASENAMES.get(sampler)
    if basename is None:
        raise ValueError(
            f"--resume is not supported for --sampler {sampler} "
            f"(checkpointing exists for: {', '.join(sorted(CHECKPOINT_BASENAMES))})."
        )

    if spec.lower() == "auto":
        save_path = str(getattr(opts, "save_path", ".") or ".")
        candidates = []
        for path in glob.glob(os.path.join(save_path, "*", basename)):
            if not os.path.isfile(path):
                continue
            run_dir = os.path.dirname(path)
            if name_prefix and not os.path.basename(run_dir).startswith(name_prefix):
                continue
            # A run directory holding a COMPLETE results.hdf5 is a run that
            # FINISHED; its end-of-run checkpoint must not silently hijack a
            # fresh submission.  Bare existence is not completion: a final write
            # that died half-way used to poison this test and lock the run out
            # of the automatic recovery it needed most.
            if result_is_complete(os.path.join(run_dir, "results.hdf5")):
                continue
            candidates.append(path)
        if not candidates:
            return None, None
        newest = max(candidates, key=os.path.getmtime)
        return newest, os.path.dirname(newest) or "."

    if os.path.isdir(spec):
        path = os.path.join(spec, basename)
        if not os.path.isfile(path):
            raise ValueError(
                f"--resume {spec!r} is a directory with no {basename} in it."
            )
        return path, spec
    if os.path.isfile(spec):
        return spec, os.path.dirname(spec) or "."
    raise ValueError(f"--resume {spec!r} does not exist.")


_UNRESOLVED = object()


def resolve_checkpoint_plan(
    opts, run_dir, sampler=None, name_prefix=None, resume_from=_UNRESOLVED
) -> CheckpointPlan:
    """Resolve the plan and record it on ``opts`` (so it reaches settings.json).

    ``find_resume_target`` must already have been consulted to pick
    ``run_dir`` — the resumed run continues inside its ORIGINAL run directory,
    so the checkpoint we write is the checkpoint we restored.

    Callers that already resolved the resume target MUST pass it as
    ``resume_from`` (a path or None).  The legacy default re-runs
    ``find_resume_target``, and for ``--resume auto`` a second lookup can
    resolve to a DIFFERENT directory than the one the caller committed to (a
    newer checkpoint appearing between the two globs), silently splitting the
    plan across two runs.
    """
    sampler = sampler or getattr(opts, "sampler", "")
    seconds = parse_checkpoint_interval(getattr(opts, "checkpoint_interval", None))
    basename = CHECKPOINT_BASENAMES.get(sampler)
    if resume_from is _UNRESOLVED:
        resume_from, _ = find_resume_target(opts, sampler, name_prefix=name_prefix)

    enabled = bool(seconds > 0.0 and basename is not None and run_dir)
    path = os.path.join(run_dir, basename) if enabled else None

    plan = CheckpointPlan(
        sampler=sampler,
        enabled=enabled,
        interval_seconds=seconds,
        path=path,
        resume_from=resume_from,
    )
    # Plain JSON-able mirrors of the plan: settings.json serialises vars(opts),
    # so the resolved decision is recorded in the run directory rather than
    # only in the terminal log.
    opts.checkpoint_interval_seconds = seconds
    opts.checkpoint_file_resolved = path
    opts.resume_from_resolved = resume_from
    return plan


def plan_from_opts(opts, sampler) -> CheckpointPlan:
    """Rebuild the plan inside ``run_sampler`` from the mirrors above.

    Library callers that hand ``run_sampler`` a bare namespace (the sampler
    unit tests, ``scripts/``) never set these, and get checkpointing OFF —
    the CLI is what turns it on.
    """
    seconds = float(getattr(opts, "checkpoint_interval_seconds", 0.0) or 0.0)
    path = getattr(opts, "checkpoint_file_resolved", None)
    return CheckpointPlan(
        sampler=sampler,
        enabled=bool(seconds > 0.0 and path),
        interval_seconds=seconds,
        path=path,
        resume_from=getattr(opts, "resume_from_resolved", None),
    )


# ----------------------------------------------------------------------------
# dynesty: checkpoint the sampler STATE without pickling the JAX likelihood
# ----------------------------------------------------------------------------
# dynesty checkpoints by pickling the whole sampler, loglikelihood and prior
# transform included.  Ours are local closures over jitted JAX functions and
# gigabytes of device-resident PE/catalog data, so a naive
# ``run_nested(checkpoint_file=...)`` dies immediately with
# ``AttributeError: Can't pickle local object`` (verified against dynesty
# 2.1.4).  We therefore swap both callables for a module-level placeholder for
# the duration of the pickle and rebind the live ones on resume: the
# checkpoint stays a few tens of kB of pure sampler state, and it is portable
# across processes.


class _DetachedCallable:
    """Placeholder standing in for a live closure inside a checkpoint."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError(
            "This dynesty checkpoint stores sampler state only; rebind the "
            "likelihood and prior transform with "
            "darksirens.inference.checkpointing.rebind_dynesty_callables "
            "before running it."
        )


_DETACHED = _DetachedCallable()


def save_dynesty_checkpoint(sampler, fname):
    """Pickle ``sampler``'s state with its callables detached."""
    from dynesty.utils import save_sampler

    holder = sampler.loglikelihood
    held_loglike = holder.loglikelihood
    held_ptform = sampler.prior_transform
    # Our own instance-level ``save`` override must not be pickled either: it
    # is a bound method that unpickling would look up on the sampler.
    held_save = sampler.__dict__.pop("save", None)
    holder.loglikelihood = _DETACHED
    sampler.prior_transform = _DETACHED
    try:
        save_sampler(sampler, fname)
    finally:
        holder.loglikelihood = held_loglike
        sampler.prior_transform = held_ptform
        if held_save is not None:
            sampler.__dict__["save"] = held_save


def install_dynesty_checkpointing(sampler):
    """Route ``sampler.save`` (called by ``run_nested``) through the detached save."""
    import types

    sampler.save = types.MethodType(
        lambda self, fname: save_dynesty_checkpoint(self, fname), sampler
    )
    return sampler


def restore_dynesty_sampler(path, loglike, prior_transform):
    """Restore a checkpoint and rebind the live callables onto it."""
    from dynesty import NestedSampler

    sampler = NestedSampler.restore(path)
    rebind_dynesty_callables(sampler, loglike, prior_transform)
    return sampler


def rebind_dynesty_callables(sampler, loglike, prior_transform):
    sampler.loglikelihood.loglikelihood = loglike
    sampler.prior_transform = prior_transform
    return sampler
