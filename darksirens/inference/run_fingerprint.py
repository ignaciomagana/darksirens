"""Semantic run fingerprint: the resume-compatibility gate.

Why this module exists
----------------------
``--resume`` restores nested-sampling state (live points, dead points, their
stored log-likelihood values, and the accumulated evidence) and rebinds the
CURRENT run's likelihood and prior-transform closures onto it.  Neither
sampler can tell whether those closures describe the statistical target the
checkpoint was created under: dynesty's restore path checks only the
dimension, and the ``--resume auto`` directory filter carries only
model/sampler/seed.  A resume whose configuration differs in any other
semantic way -- different input files, different prior bounds, a different
fixed value, a different model flag -- therefore mixes samples, likelihood
values, and evidence increments from two different targets, and the completed
output looks normal while its posterior and logZ have no valid
interpretation.

The gate: every CLI run writes ``run_fingerprint.json`` into its run
directory at creation time; every resume rebuilds the fingerprint from its
own configuration and refuses to restore unless the two match exactly.
``--resume_force`` overrides the gate for deliberate expert use (and for
continuing pre-fingerprint checkpoints) and is stamped loudly.

What counts as semantic
-----------------------
Everything that changes the posterior or the evidence: the sampled labels,
their bounds and prior families, fixed/overridden parameter values, every
model flag, the sampler and its stopping/live-point settings, the seed, the
normalization grid, and the CONTENT of every input file -- a path string
alone cannot detect a regenerated file.  Operational knobs (where the run
directory lives, checkpoint cadence, progress printing, performance
chunking) are excluded, so a requeued SLURM job re-running its identical
command line always matches its own checkpoint.  That stability matters:
production submit scripts hard-code ``--resume auto`` and rely on requeues
resuming without human intervention.

File identity is a full SHA-256 up to 1 GiB; larger files get a sampled
digest (size + head/middle/tail 16 MiB windows), labeled as such, so a 26 GB
Q table does not add minutes to every submission.  Code identity (git sha,
package versions) is recorded but ADVISORY: refusing to resume across a
deployment would break every requeue that straddles a ``git pull``, so a
code mismatch warns instead of failing.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import tempfile
import warnings

FINGERPRINT_BASENAME = "run_fingerprint.json"
FINGERPRINT_SCHEMA_VERSION = 1

# Full-file SHA-256 up to this size; sampled digest above it.
_FULL_HASH_MAX_BYTES = 1 << 30
_SAMPLE_WINDOW_BYTES = 16 << 20

# Options that do NOT change the statistical target.  Everything else in
# vars(opts) is fingerprinted.  Two classes live here:
#   * resume/checkpoint machinery -- these legitimately differ between the
#     original submission and a resume attempt (resume_from_resolved is None
#     on the first run and a path afterwards);
#   * presentation and performance-only execution knobs -- chunk/block sizes
#     change summation batching, not the target density.
# When adding a CLI flag, ask "does this change the posterior or logZ?".
# If yes, leave it out of this set; the fingerprint picks it up automatically.
_NON_SEMANTIC_KEYS = frozenset({
    # resume / checkpoint machinery
    "resume",
    "resume_force",
    "checkpoint_interval",
    "checkpoint_interval_seconds",
    "checkpoint_file_resolved",
    "resume_from_resolved",
    "run_dir",
    "save_path",
    # legacy tinyns-specific checkpoint flags (sampling.py still honours them
    # over the shared plan).  They differ between a submission and its resume
    # exactly like resume_from_resolved -- and worse, once sampling has started
    # they name an EXISTING file, so the data_files scan below would hash the
    # checkpoint itself, whose bytes are rewritten every
    # --tinyns_checkpoint_interval iterations.
    "tinyns_checkpoint_path",
    "tinyns_checkpoint_path_out",
    "tinyns_resume_from",
    "tinyns_progress_interval",
    # presentation / diagnostics
    "show_progress",
    "dynesty_diagnostics",
    "preflight_only",
    "sampler_preflight",
    # start-time gates: they decide whether a run is ALLOWED to begin, not what
    # it targets.  --allow_skymap_population only suppresses the refusal to fit
    # a population from skymap-surrogate mass/spin draws; the target difference
    # it permits is carried by fix_population, which is semantic.  Keeping it
    # out preserves the digest of every run that predates the flag.
    "allow_skymap_population",
    # performance-only execution knobs
    "row_chunk",
    "gp3d_pix_chunk",
    "workers",
    "block_size_resolution",
    "pe_block",
    "pe_event_block",
    "selection_block",
    # ... including the two other values _resolve_and_report_block_sizes writes
    # back onto opts from a LIVE device-memory probe: a requeue landing on a GPU
    # with a different resident footprint resolves a different sel_batch_size (or
    # None), which would refuse the resume instead of continuing it.
    "sel_batch_size",
    "block_size_static_state_bytes",
    "max_component_partitions",
    "max_total_partitions",
    "tinyns_checkpoint_interval",
})


class ResumeFingerprintError(ValueError):
    """Raised when a resume target does not match the current configuration."""


def _jsonable(value):
    """Best-effort JSON-stable mirror of a settings value.

    Mirrors save_settings_json's convention: values that json can serialize
    pass through; everything else is stringified.  Tuples become lists so the
    canonical serialization does not depend on the Python container type.
    """
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _file_identity(path: str) -> dict:
    """Content identity for one input file.

    Full SHA-256 for ordinary files; for very large inputs, a sampled digest
    over (size, head, middle, tail windows).  A sampled digest still catches
    any size change and any byte change inside the windows, which covers the
    realistic failure modes (regenerated file, different table, truncation)
    without re-reading tens of GB on every submission.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= _FULL_HASH_MAX_BYTES:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
            return {"bytes": int(size), "sha256": h.hexdigest()}
        h.update(str(size).encode())
        for offset in (0, max(0, size // 2 - _SAMPLE_WINDOW_BYTES // 2),
                       max(0, size - _SAMPLE_WINDOW_BYTES)):
            f.seek(offset)
            h.update(f.read(_SAMPLE_WINDOW_BYTES))
    return {"bytes": int(size), "sampled_sha256": h.hexdigest()}


def _iter_file_candidates(value):
    """Yield the string values inside ``value`` that may name input files."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                yield v


def _add_flow_ensemble_identity(sem_opts, data_files):
    """Content identity for the flow-surrogate ensemble under ``gw_flows_path``.

    ``--gw_flows_path`` names a DIRECTORY, so the generic ``os.path.isfile``
    scan above records only its path string -- yet for a flow-surrogate run the
    per-event checkpoints ARE the PE likelihood (they replace the stored
    posterior samples), and the ensemble's sorted checkpoint order IS the event
    identity.  Retraining the ensemble in place, adding or removing an event,
    or resuming against a partially written directory would otherwise pass the
    resume gate silently.  Only this option is walked: a generic directory walk
    could recurse into an arbitrarily large data root.
    """
    flows_dir = sem_opts.get("gw_flows_path")
    if not isinstance(flows_dir, str) or not os.path.isdir(flows_dir):
        return
    pattern = sem_opts.get("flows_pattern") or "*/*_flow.npz"
    for path in sorted(glob.glob(os.path.join(flows_dir, str(pattern)))):
        if os.path.isfile(path):
            rel = os.path.relpath(path, flows_dir)
            data_files[f"gw_flows_path/{rel}"] = _file_identity(path)


def _redshift_grid_identity():
    """Resolved redshift-normalisation domain: the other env-driven target.

    ``normalization_grid_settings()`` below covers the mass/q/chi quadrature,
    but three environment knobs change the statistical target and were recorded
    nowhere: ``DARKSIRENS_ZMAX`` sets the shared ``zgrid`` -- the integration
    domain of the redshift prior, the missing-galaxy budget and the selection
    integral, i.e. both the likelihood numerator and beta -- while
    ``DARKSIRENS_GP_ZNORM_HI`` / ``DARKSIRENS_SKY_ZNORM_HI`` set the ceilings
    above which the GP-population and sky-model z-normalisers freeze.  A
    requeue whose submit environment resolves any of them differently would
    otherwise restore state whose stored logL came from a different target.
    Read through the modules (not os.environ) so the recorded values are the
    ones actually in force.
    """
    out = {}
    try:
        from darksirens.redshift import grid as _grid
        out["zMax"] = float(_grid.zMax)
        out["n_nodes"] = int(len(_grid.zgrid))
        out["z_last"] = float(_grid.zgrid[-1])
    except Exception:  # pragma: no cover - never expected in a CLI process
        return None
    for module_name, prefix in (
        ("darksirens.gw.populations.gp", "gp_znorm"),
        ("darksirens.sky.models", "sky_znorm"),
    ):
        try:
            module = importlib.import_module(module_name)
            out[f"{prefix}_hi"] = float(module._ZNORM_HI)
            out[f"{prefix}_n"] = int(module._ZNORM_N)
        except Exception:  # pragma: no cover - optional import failure
            out[f"{prefix}_hi"] = None
    return out


def build_run_fingerprint(
    opts,
    *,
    labels,
    lower_bound,
    upper_bound,
    prior_kinds=None,
    prior_overrides=None,
    fixed_parameter_values=None,
) -> dict:
    """Canonical fingerprint of everything that defines this run's target."""
    sem_opts = {}
    for key, val in sorted(vars(opts).items()):
        if key.startswith("_") or key in _NON_SEMANTIC_KEYS:
            continue
        sem_opts[key] = _jsonable(val)

    # Content identity for every option value that names an existing file.
    # This is deliberately generic: any data path added to either CLI is
    # covered without touching this module.  (A path that does not exist yet
    # -- an output target -- is skipped; loaders fail on it independently.)
    data_files = {}
    for key, val in sem_opts.items():
        for j, cand in enumerate(_iter_file_candidates(val)):
            if os.path.isfile(cand):
                slot = key if j == 0 else f"{key}[{j}]"
                data_files[slot] = _file_identity(cand)
    _add_flow_ensemble_identity(sem_opts, data_files)

    # The normalization grids are process-global, environment-driven inputs
    # to the population normalizers -- as semantic as any CLI flag.
    try:
        from darksirens.gw.populations.utils import normalization_grid_settings
        norm_grid = normalization_grid_settings().to_dict()
    except Exception:  # pragma: no cover - never expected in a CLI process
        norm_grid = None

    semantic = {
        "redshift_grid": _redshift_grid_identity(),
        "labels": [str(lab) for lab in labels],
        "lower_bound": [float(x) for x in lower_bound],
        "upper_bound": [float(x) for x in upper_bound],
        "prior_kinds": _jsonable(list(prior_kinds)) if prior_kinds is not None else None,
        "prior_overrides": _jsonable(prior_overrides) if prior_overrides else None,
        "fixed_parameter_values": (
            _jsonable(fixed_parameter_values) if fixed_parameter_values else None
        ),
        "options": sem_opts,
        "data_files": data_files,
        "normalization_grid": norm_grid,
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")
    ).hexdigest()

    try:
        from darksirens.io.settings import code_identity
        code = dict(code_identity())
    except Exception:  # pragma: no cover - identity is best-effort by design
        code = {}

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "digest": digest,
        "semantic": semantic,
        "advisory": {"code": code},
    }


def save_run_fingerprint(run_dir: str, fingerprint: dict) -> str:
    """Atomically write ``run_fingerprint.json`` into ``run_dir``."""
    path = os.path.join(run_dir, FINGERPRINT_BASENAME)
    fd, tmp = tempfile.mkstemp(
        prefix=FINGERPRINT_BASENAME + ".", suffix=".tmp", dir=run_dir
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(fingerprint, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _semantic_diff(stored, current, prefix="", out=None, limit=20):
    """Human-readable paths where two semantic blocks disagree."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in sorted(set(stored) | set(current)):
            _semantic_diff(
                stored.get(key, "<absent>"), current.get(key, "<absent>"),
                f"{prefix}.{key}" if prefix else str(key), out, limit,
            )
    elif isinstance(stored, list) and isinstance(current, list):
        if len(stored) != len(current):
            out.append(
                f"{prefix}: length {len(stored)} (checkpointed run) vs "
                f"{len(current)} (this run)"
            )
        else:
            for i, (s, c) in enumerate(zip(stored, current)):
                _semantic_diff(s, c, f"{prefix}[{i}]", out, limit)
    elif stored != current:
        out.append(f"{prefix}: {stored!r} (checkpointed run) vs {current!r} (this run)")
    return out


def check_resume_fingerprint(run_dir: str, current: dict, *, force: bool = False):
    """Gate a resume: the target run directory must match this configuration.

    Returns the stored fingerprint dict on a match, or ``None`` when the run
    directory has no (readable) fingerprint and ``force`` was given -- the
    caller should then stamp the current fingerprint so later requeues can be
    validated.  Raises :class:`ResumeFingerprintError` on any mismatch or on
    a missing/corrupt fingerprint without ``force``.
    """
    path = os.path.join(run_dir, FINGERPRINT_BASENAME)
    header = (
        f"Refusing to resume from '{run_dir}': "
    )
    footer = (
        "\nResuming would mix nested-sampling state from two different "
        "statistical targets; the resulting posterior and logZ would have no "
        "valid interpretation. Start a fresh run (--resume off), point "
        "--resume at the matching run directory, or -- only if you are "
        "certain the difference is harmless -- pass --resume_force."
    )
    if not os.path.isfile(path):
        msg = (
            header
            + "it has no run_fingerprint.json, so compatibility with this "
              "configuration cannot be verified (checkpoint created by a "
              "pre-fingerprint version of darksirens?)."
            + footer
        )
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' WITHOUT a fingerprint "
                "match; you are vouching that the configuration is identical.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg)

    try:
        with open(path) as f:
            stored = json.load(f)
    except (OSError, ValueError) as exc:
        msg = header + f"its run_fingerprint.json is unreadable ({exc})." + footer
        if force:
            warnings.warn(
                f"--resume_force: resuming '{run_dir}' with an unreadable "
                "fingerprint; you are vouching for compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise ResumeFingerprintError(msg) from exc

    if int(stored.get("schema_version", -1)) != FINGERPRINT_SCHEMA_VERSION:
        diffs = [
            f"fingerprint schema_version: {stored.get('schema_version')!r} vs "
            f"{FINGERPRINT_SCHEMA_VERSION!r}"
        ]
    elif stored.get("digest") == current.get("digest"):
        _warn_code_identity_drift(run_dir, stored, current)
        return stored
    else:
        diffs = _semantic_diff(
            stored.get("semantic") or {}, current.get("semantic") or {}
        )
        if not diffs:  # digest disagrees but the diff walker found nothing
            diffs = [f"digest: {stored.get('digest')} vs {current.get('digest')}"]

    shown = "\n".join(f"  - {d}" for d in diffs[:20])
    more = len(diffs) - 20
    if more > 0:
        shown += f"\n  ... and {more} more"
    msg = (
        header
        + "its configuration does not match this run's:\n"
        + shown
        + footer
    )
    if force:
        warnings.warn(
            f"--resume_force: resuming '{run_dir}' DESPITE a fingerprint "
            f"mismatch:\n{shown}\nThe resumed output mixes two targets and "
            "must not be used as a science result.",
            RuntimeWarning,
            stacklevel=2,
        )
        return stored
    raise ResumeFingerprintError(msg)


def _warn_code_identity_drift(run_dir, stored, current):
    """Advisory only: a requeue that straddles a deployment should proceed,
    but the log must say the code changed under the checkpoint."""
    old = ((stored.get("advisory") or {}).get("code") or {})
    new = ((current.get("advisory") or {}).get("code") or {})
    keys = ("darksirens_version", "git_sha", "gwcat_commit", "tinyns_commit")
    drift = [
        f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
        for k in keys
        if old.get(k) is not None and new.get(k) is not None and old.get(k) != new.get(k)
    ]
    if drift:
        warnings.warn(
            f"Resuming '{run_dir}' with the same configuration but different "
            "code identity (" + "; ".join(drift) + "). If the code change "
            "altered the likelihood, the resumed chain mixes two targets -- "
            "verify the change was behavior-neutral for this configuration.",
            RuntimeWarning,
            stacklevel=3,
        )
