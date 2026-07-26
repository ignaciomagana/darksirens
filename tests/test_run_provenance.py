"""Run artifacts must identify the code that produced them.

Before this block existed, deciding which archived ``logZ`` had to be recomputed
after a physics fix came down to comparing run-directory mtimes, and a run whose
sampling straddled a ``git pull`` was unattributable (issue #288).  The block is
built once in ``darksirens.io.settings`` so ``settings.json`` (both CLIs) and
``results.hdf5`` cannot drift apart.
"""

import json
import os
import subprocess

import pytest

from darksirens.io.settings import code_identity, environment_block

_IDENTITY_KEYS = {
    "darksirens_version",
    "git_sha",
    "git_dirty",
    "git_branch",
    "gwcat_version",
    "gwcat_commit",
    "tinyns_version",
    "tinyns_commit",
}


def test_code_identity_has_the_full_key_set():
    ident = code_identity()
    assert _IDENTITY_KEYS <= set(ident)
    assert ident["darksirens_version"]  # non-empty
    assert isinstance(ident["git_sha"], str)
    assert ident["git_dirty"] is None or isinstance(ident["git_dirty"], bool)


def test_code_identity_resolves_the_real_sha_inside_a_checkout():
    """In a work tree the SHA must be the real one, not the 'unknown' fallback."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        want = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git work tree")

    assert code_identity()["git_sha"] == want
    assert len(want) == 40


def test_code_identity_degrades_cleanly_without_git(monkeypatch):
    """A wheel/container install has no .git: report 'unknown', never raise."""
    import darksirens.io.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_git", lambda *a, **kw: None)
    code_identity.cache_clear()
    try:
        ident = code_identity()
    finally:
        code_identity.cache_clear()

    assert ident["git_sha"] == "unknown"
    assert ident["git_dirty"] is None
    assert ident["git_branch"] == "unknown"
    assert ident["darksirens_version"]  # still known: it ships in the package


def test_dependency_commits_are_recorded_when_pip_knows_the_source():
    """gwcat/tinyns are 0.1.0 forever; the commit is the only real identifier."""
    ident = code_identity()
    for key in ("gwcat_commit", "tinyns_commit"):
        value = ident[key]
        assert isinstance(value, str) and value
        if value != "unknown":
            # A 40-char sha (optionally '-dirty'), or the recorded install URL.
            head = value.removesuffix("-dirty")
            assert len(head) == 40 or "://" in head, value


def test_environment_block_carries_identity_stack_and_argv():
    env = environment_block()
    assert _IDENTITY_KEYS <= set(env)
    for key in ("jax_version", "numpy_version", "healpy_version",
                "jax_backend", "jax_devices", "python_version", "argv"):
        assert key in env, key
    assert isinstance(env["argv"], list)
    # Must survive both artifact encodings (json.dump for settings.json,
    # json.dumps into an HDF5 attr for results.hdf5).
    assert json.loads(json.dumps(env, default=str))["git_sha"] == env["git_sha"]


def test_both_clis_write_the_same_environment_block():
    """The lensing twin used to persist no ``environment`` key at all."""
    import argparse

    from darksirens.cli.inference_lensing import _jsonable_settings

    opts = argparse.Namespace(cluster_mode="off", nlive=60)
    lensing_env = _jsonable_settings(opts)["environment"]
    assert set(lensing_env) == set(environment_block())
