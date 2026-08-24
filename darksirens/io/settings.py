import functools
import importlib.metadata as _metadata
import json
import os
import subprocess
import sys

import numpy as np

import darksirens
from darksirens.cli.common import _fixed_dark_energy_metadata

# JAX/healpy and the population grid are imported at CALL time only: io.results
# imports this module at module scope, and io.results must stay importable by
# tooling that only reads/writes HDF5.

# ── Run provenance ─────────────────────────────────────────────────────────────
#
# Every run artifact must say which code produced it.  Without this, deciding
# which archived logZ has to be recomputed after a physics fix comes down to
# directory mtimes — and a run whose sampling straddled a `git pull` is simply
# unattributable (issue #288).  Built here, in one place, so both the main and
# the lensing CLI (and results.hdf5) record the identical block.

_UNKNOWN = "unknown"


def _repo_root() -> str:
    """Directory holding the installed/checked-out ``darksirens`` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(darksirens.__file__)))


def _git(*args: str, cwd: str | None = None) -> str | None:
    """Run a git command in ``cwd`` (default: this package's repo), else None.

    Returns None for every failure mode a deployed install can hit: no git on
    PATH, an installed (non-checkout) tree with no ``.git``, a hung filesystem
    (bounded by ``timeout``), or a nonzero exit.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd if cwd is not None else _repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return _metadata.version(name)
    except Exception:
        return _UNKNOWN


def _package_commit(name: str) -> str:
    """Best-effort commit id for a dependency installed from git or a checkout.

    ``gwcat`` and ``tinyns`` both carry version ``0.1.0`` forever, so the
    version alone cannot say which gwcat wrote an input file or which tinyns
    estimator produced a logZerr.  pip records the install source in
    ``direct_url.json`` (PEP 610): a ``git+`` install stores the resolved
    ``commit_id``, and a local/editable directory install stores its path, whose
    git HEAD we resolve here.  ``"unknown"`` when the package came from an index.
    """
    try:
        raw = _metadata.distribution(name).read_text("direct_url.json")
        if not raw:
            return _UNKNOWN
        info = json.loads(raw)
    except Exception:
        return _UNKNOWN

    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return str(commit)

    url = str(info.get("url") or "")
    if url.startswith("file://"):
        path = url[len("file://"):]
        head = _git("rev-parse", "HEAD", cwd=path)
        if head:
            dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=path)
            return f"{head}-dirty" if dirty else head
    return url or _UNKNOWN


@functools.lru_cache(maxsize=1)
def code_identity() -> dict:
    """Identify the analysis code that produced a run.

    ``git_sha``/``git_dirty`` are ``"unknown"``/``None`` outside a checkout (a
    wheel install, an sdist, a container layer), which is the honest answer
    rather than a crash.  ``git_dirty`` counts tracked modifications only, so
    scratch files and untracked run outputs do not flag an otherwise clean
    commit.  Cached: the git calls are identical for the process's lifetime.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "darksirens_version": getattr(darksirens, "__version__", _UNKNOWN),
        "git_sha":            sha if sha else _UNKNOWN,
        # None (not False) when there is no checkout to compare against.
        "git_dirty":          None if status is None else bool(status),
        "git_branch":         _git("rev-parse", "--abbrev-ref", "HEAD") or _UNKNOWN,
        "gwcat_version":      _package_version("gwcat"),
        "gwcat_commit":       _package_commit("gwcat"),
        "tinyns_version":     _package_version("tinyns"),
        "tinyns_commit":      _package_commit("tinyns"),
    }


def environment_block() -> dict:
    """Code identity + numerics stack + invocation, for a run's artifacts.

    ``darksirens_env`` is the run's ``DARKSIRENS_*`` environment: several of
    those knobs are read once at import and change the statistical target, not
    just performance — ``DARKSIRENS_ZMAX`` sets the shared ``zgrid`` (the
    integration domain of the redshift prior, the missing-galaxy budget and the
    selection integral), and ``DARKSIRENS_GP_ZNORM_HI`` /
    ``DARKSIRENS_SKY_ZNORM_HI`` set where the GP z-normalisers freeze.  Without
    them recorded, an archived results.hdf5 cannot say which domain produced its
    logZ (the issue-#288 class of problem this block exists for).
    """
    import healpy as hp
    import jax

    block = dict(code_identity())
    block.update({
        "jax_version":    jax.__version__,
        "numpy_version":  np.__version__,
        "healpy_version": hp.__version__,
        "jax_backend":    jax.default_backend(),
        "jax_devices":    [str(dv) for dv in jax.devices()],
        "python_version": sys.version,
        "argv":           list(sys.argv),
        "darksirens_env": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("DARKSIRENS_")
        },
    })
    return block


def save_settings_json(
    opts,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    meta:                   dict,
    basename:               str = "settings.json",
) -> str:
    """Human-readable settings.json for easy inspection and re-runs.

    ``basename`` lets a resumed run record its attempt as
    ``settings.resume-<timestamp>.json`` without overwriting the original
    ``settings.json`` -- the record of the configuration that created the
    checkpoint being restored.
    """
    d: dict = {}

    for key, val in vars(opts).items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(val)
            d[key] = val
        except (TypeError, ValueError):
            d[key] = str(val)

    # Emit None explicitly so it's obvious when not set — not an empty dict
    d["fixed_parameter_values"] = fixed_parameter_values if fixed_parameter_values else None
    d["prior_overrides"]        = prior_overrides        if prior_overrides        else None
    d["fixed_cosmology"]        = bool(getattr(opts, "fix_cosmology", False))
    d["fixed_de"]               = bool(getattr(opts, "fix_de", False))
    de_meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
    d["fixed_dark_energy"]      = de_meta["fixed_dark_energy"]
    d["dark_energy_fixed_values"] = {
        label: value
        for label, value in (
            ("w0", de_meta["w0_value"]),
            ("wa", de_meta["wa_value"]),
        )
        if value is not None
    } or None
    d["w0_fixed"]               = de_meta["w0_fixed"]
    d["wa_fixed"]               = de_meta["wa_fixed"]

    d["labels"]      = list(labels)
    d["lower_bound"] = list(map(float, lower_bound))
    d["upper_bound"] = list(map(float, upper_bound))
    d.update(meta)
    from darksirens.gw.populations.utils import normalization_grid_settings
    d["normalization_grid"] = normalization_grid_settings().to_dict()

    d["environment"] = environment_block()

    path = os.path.join(run_dir, basename)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    return path



