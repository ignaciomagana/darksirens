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

import re
import subprocess
import sys
import tarfile
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


def _files_setup_py_opens():
    """Paths that ``setup.py`` reads at build time (so the sdist must ship them)."""
    text = (REPO_ROOT / "setup.py").read_text()
    return sorted(set(re.findall(r'open\(os\.path\.join\(_HERE,\s*"([^"]+)"\)', text)))


def test_setup_py_reads_its_build_inputs_relative_to_itself():
    """A cwd-relative ``open()`` in setup.py breaks any build from elsewhere.

    ``python setup.py --name`` from outside the tree (and, historically, the
    unpacked-sdist rebuild) died on ``open("requirements.txt")``.  Every
    build-time read must be anchored to the setup.py directory.
    """
    text = (REPO_ROOT / "setup.py").read_text()
    bare = [m for m in re.findall(r'open\(\s*"([^"]+)"', text)]
    assert not bare, (
        "setup.py opens these build inputs relative to the CWD, not to "
        f"itself: {bare}. Wrap them in os.path.join(_HERE, ...)."
    )
    assert _files_setup_py_opens(), "no build-time reads found — did setup.py change shape?"

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "setup.py"), "--name"],
        cwd=REPO_ROOT.parent, capture_output=True, text=True,
    )
    assert out.returncode == 0, f"setup.py --name failed outside the tree:\n{out.stderr}"
    assert out.stdout.strip().splitlines()[-1] == "darksirens", out.stdout


def test_sdist_ships_every_file_setup_py_needs_to_rebuild_itself(tmp_path):
    """The generated sdist must be able to build itself (OPS-01).

    ``setup.py`` reads requirements.txt for ``install_requires``; setuptools
    does not put non-package data files into an sdist on its own, so without
    MANIFEST.in the tarball shipped to PyPI unpacked into a tree whose very
    first build step raised ``FileNotFoundError: 'requirements.txt'``.  CI only
    ever built from the checkout, where the file is present, so the wheel job
    stayed green while the source artifact was dead on arrival.
    """
    needed = _files_setup_py_opens()
    dist = tmp_path / "dist"
    egg = tmp_path / "egg"
    egg.mkdir()
    # --egg-base into tmp: setuptools REUSES a stale SOURCES.txt if the
    # checkout already carries a darksirens.egg-info from an earlier build,
    # which would let a tarball built without MANIFEST.in still list the file
    # and hide the very defect this pins.
    build = subprocess.run(
        [sys.executable, "setup.py", "-q",
         "egg_info", "--egg-base", str(egg),
         "sdist", "--formats=gztar", "--dist-dir", str(dist)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert build.returncode == 0, f"sdist build failed:\n{build.stderr[-3000:]}"
    tarballs = sorted(dist.glob("*.tar.gz"))
    assert len(tarballs) == 1, tarballs

    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()
        root = names[0].split("/")[0]
        missing = [f for f in needed if f"{root}/{f}" not in names]
        assert not missing, (
            "the sdist omits files setup.py opens at build time — the source "
            f"distribution cannot rebuild itself: {missing}. Add them to "
            "MANIFEST.in."
        )
        tf.extractall(tmp_path / "ext")

    # The reproduction itself: build metadata from the UNPACKED sdist.
    src = tmp_path / "ext" / root
    out = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=src, capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        f"the unpacked sdist cannot run its own setup.py:\n{out.stderr[-3000:]}"
    )
    assert out.stdout.strip().splitlines()[-1] == "darksirens", out.stdout


def test_chi_eff_prior_tables_agree_across_packages():
    """Standing cross-package guard (DS-11): the JAX chi_eff prior port and
    the linked gwcat.spin reference must agree to 1e-10, so a gwcat version
    bump cannot silently split the convention again.  (The detailed port
    test lives in test_flow_pe_prior.py; this one is the packaging-level
    tripwire that fires even when the flow suites are skipped.)"""
    import numpy as np

    gwcat_spin = pytest.importorskip("gwcat.spin")
    pytest.importorskip("darksirens.likelihood.flow_events")
    import jax.numpy as jnp

    from darksirens.likelihood.flow_events import (
        build_chi_eff_prior_table,
        chi_eff_prior_logpdf,
    )

    amax = 0.99
    qg, cg, tab = build_chi_eff_prior_table(amax)
    ref_prior = gwcat_spin.ChiEffPrior(amax=amax)
    np.testing.assert_array_equal(np.asarray(tab), np.asarray(ref_prior.table))

    rng = np.random.default_rng(5)
    m1 = rng.uniform(5.0, 90.0, 500)
    m2 = rng.uniform(0.1, 1.0, 500) * m1
    chi = rng.uniform(-1.05, 1.05, 500)
    ref = np.asarray(ref_prior.logprob(chi, m1, m2))
    got = np.asarray(chi_eff_prior_logpdf(
        jnp.asarray(m1 / (m1 + m2)), jnp.asarray(chi),
        jnp.asarray(qg), jnp.asarray(cg), jnp.asarray(tab),
    ))
    np.testing.assert_allclose(got, ref, atol=1e-10)
