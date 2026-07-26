"""Packaging contract: every shipped directory must be a real package.

``setup.py`` discovers modules with classic
``setuptools.find_packages(include=["darksirens", "darksirens.*"])``, which
only walks directories that contain an ``__init__.py``.  A directory of
git-tracked modules without one is silently dropped from both the sdist and
the wheel, so ``pip install .`` produces an importable-looking package whose
modules are absent — while the editable/source tree keeps working because the
same directory resolves as a PEP-420 namespace portion under the checkout root.

This has now bitten twice: once for the root package (see
``darksirens/__init__.py``) and once for ``darksirens/utils`` (cosmology,
interp2d, plotting, utils — imported by the likelihood core, the priors and
five CLIs).  These tests fail loudly the next time a new subpackage is added
without its ``__init__.py``, and pin the console-script targets to callables
that keep the GPU teardown guard.
"""

import subprocess
from pathlib import Path

import pytest
from setuptools import find_packages

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "darksirens"

# Directories that legitimately hold no importable modules.
_SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}


def _tracked_py_files():
    """Git-tracked ``*.py`` paths under ``darksirens/``, or skip if not a repo."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "darksirens"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        pytest.skip("git unavailable or not a work tree")
    paths = [p for p in out.split("\0") if p.endswith(".py")]
    if not paths:  # pragma: no cover - shallow export
        pytest.skip("no tracked darksirens/*.py files (source export?)")
    return paths


def test_find_packages_covers_every_tracked_module_directory():
    discovered = set(find_packages(where=str(REPO_ROOT),
                                  include=["darksirens", "darksirens.*"]))
    missing = {}
    for rel in _tracked_py_files():
        parts = Path(rel).parent.parts
        if any(p in _SKIP_DIR_NAMES for p in parts):
            continue
        pkg = ".".join(parts)
        if pkg not in discovered:
            missing.setdefault(pkg, []).append(rel)

    assert not missing, (
        "find_packages() drops directories that hold tracked modules — "
        "`pip install .` would ship none of them. Add an __init__.py to: "
        + ", ".join(sorted(missing))
    )


def test_utils_is_a_real_package_not_a_namespace_portion():
    """Regression pin for the concrete defect (utils/ had no __init__.py)."""
    assert (PKG_ROOT / "utils" / "__init__.py").is_file()

    import darksirens.utils

    # A PEP-420 namespace portion has __file__ is None; a real package does not.
    assert darksirens.utils.__file__ is not None
    assert "darksirens.utils" in find_packages(
        where=str(REPO_ROOT), include=["darksirens", "darksirens.*"]
    )


def test_console_scripts_for_long_running_clis_keep_the_teardown_guard():
    """The two GPU CLIs must not be entered through a bare ``main``.

    ``run_cli`` hard-exits with ``os._exit(0)`` because interpreter teardown can
    block for hours in the CUDA exit handlers on a shared GPU (a finished run
    idling until the SLURM cgroup OOM-kills it).  ``python -m`` gets that via
    the ``__main__`` guard; the console scripts only get it if their entry
    point is the wrapper, not ``main``.
    """
    text = (REPO_ROOT / "setup.py").read_text()
    for script in ("darksirens_inference", "darksirens_inference_lensing"):
        line = next(
            ln.strip() for ln in text.splitlines() if ln.strip().startswith(f'"{script}=')
        )
        assert line.endswith(':console_main",'), (
            f"{script} console script must target console_main (run_cli wrapper), "
            f"got: {line}"
        )

    from darksirens.cli.inference import console_main as inference_console_main
    from darksirens.cli.inference_lensing import console_main as lensing_console_main

    assert callable(inference_console_main)
    assert callable(lensing_console_main)
