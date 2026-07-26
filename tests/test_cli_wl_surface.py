"""CLI surface for the weak-lensing (``spectral_sirens_wl``) universe model.

PR 1 of the WL-to-lensing-CLI move: the stock ``darksirens_inference`` no
longer accepts ``--universe_model spectral_sirens_wl`` (it raises a migration
hint pointing at the lensing CLI), and ``darksirens_inference_lensing`` becomes
the sole owner of the WL universe model, gaining a tabulated WL backend.

Subprocess tests set ``PYTHONPATH`` to THIS checkout (the package is
pip-installed editable against the main checkout) and ``JAX_PLATFORMS=cpu`` so
they exercise the worktree's code without touching a GPU.  In-process tests
import the lensing CLI helpers directly.  Mirrors test_cli_multitracer.py's
subprocess style.
"""
import os
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest

from darksirens.cli import inference_lensing as lens_cli
from darksirens.likelihood.likelihood_with_clusters import (
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
)

# Repo root of THIS worktree (tests/ is one level below it).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _subprocess_env():
    """os.environ + PYTHONPATH pinned to this checkout + CPU-only JAX."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + existing if existing else "")
    return env


def _run_inference(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )


def test_inference_rejects_spectral_sirens_wl_with_migration_hint():
    """The retired --universe_model value fails at argparse time (type runs
    before choices), pointing the user at the lensing CLI, before any I/O."""
    result = _run_inference([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--universe_model", "spectral_sirens_wl",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    # argparse writes the ArgumentTypeError message to stderr.
    assert "darksirens_inference_lensing" in result.stderr
    # Must fail at parse, before any file is opened.
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_inference_help_lists_no_wl_flags():
    """--help must expose none of the moved WL flags, and the retired
    universe-model choice must be gone from the choices list."""
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0
    out = result.stdout
    for flag in (
        "--lensing_wl_model",
        "--lensing_wl_a",
        "--lensing_wl_b",
        "--lensing_wl_table_path",
        "--wl_selection",
    ):
        assert flag not in out, f"{flag} should not appear in inference --help"
    # spectral_sirens_wl is no longer an accepted --universe_model choice.
    assert "spectral_sirens_wl" not in out


def test_lensing_parser_accepts_tabulated_and_table_path():
    """The lensing CLI grows the tabulated backend + table-path flag."""
    parser = lens_cli.build_parser()
    opts = parser.parse_args([
        "--gw_path", "g.h5",
        "--gwselection_path", "s.h5",
        "--sampler", "tinyns",
        "--wl_backend", "tabulated",
        "--lensing_wl_table_path", "/some/table.h5",
    ])
    assert opts.wl_backend == "tabulated"
    assert opts.lensing_wl_table_path == "/some/table.h5"


def test_lensing_parser_wl_defaults_hold():
    """Backend defaults to lognormal, table path defaults to None."""
    parser = lens_cli.build_parser()
    opts = parser.parse_args([
        "--gw_path", "g.h5",
        "--gwselection_path", "s.h5",
        "--sampler", "tinyns",
    ])
    assert opts.wl_backend == "lognormal"
    assert opts.lensing_wl_table_path is None


def test_wl_backend_code_matches_library_constants():
    """_wl_backend_code maps the strings to the library integer codes."""
    assert lens_cli._wl_backend_code(
        types.SimpleNamespace(wl_backend="lognormal")
    ) == WL_BACKEND_LOGNORMAL
    assert lens_cli._wl_backend_code(
        types.SimpleNamespace(wl_backend="tabulated")
    ) == WL_BACKEND_TABULATED
    assert lens_cli._wl_backend_code(
        types.SimpleNamespace(wl_backend="disabled")
    ) == WL_BACKEND_DISABLED


def test_derive_universe_model_for_all_backends():
    """Both WL backends drive spectral_sirens_wl; disabled falls back."""
    assert lens_cli._derive_universe_model("lognormal") == "spectral_sirens_wl"
    assert lens_cli._derive_universe_model("tabulated") == "spectral_sirens_wl"
    assert lens_cli._derive_universe_model("disabled") == "spectral_sirens"


def _write_wl_table(path, a, b, nz=200, nmu=769, log_mu_half=2.0, z_lo=0.4):
    """Write a lognormal p_WL(mu|z) table in the loader's HDF5 layout."""
    import jax.numpy as jnp
    from darksirens.lensing.wlmagnification import make_lognormal_log_p_wl

    z_grid = np.linspace(z_lo, 3.0, nz)
    log_mu_grid = np.linspace(-log_mu_half, log_mu_half, nmu)
    ZZ, MM = np.meshgrid(z_grid, log_mu_grid, indexing="ij")
    log_p_table = np.asarray(
        make_lognormal_log_p_wl(a, b)(jnp.exp(jnp.asarray(MM)), jnp.asarray(ZZ))
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("z_grid", data=z_grid)
        f.create_dataset("log_mu_grid", data=log_mu_grid)
        f.create_dataset("log_p_table", data=log_p_table)
    return z_grid, log_mu_grid, log_p_table


def test_load_wl_table_arrays_roundtrip(tmp_path):
    """Tabulated backend reads z_grid/log_mu_grid/log_p_table from HDF5."""
    path = tmp_path / "wl_table.h5"
    z_grid, log_mu_grid, log_p_table = _write_wl_table(path, a=0.02, b=1.0)

    opts = types.SimpleNamespace(
        wl_backend="tabulated", lensing_wl_table_path=str(path)
    )
    out = lens_cli._load_wl_table_arrays(opts)
    assert set(out) == {"wl_z_grid", "wl_log_mu_grid", "wl_log_p_table"}
    np.testing.assert_allclose(np.asarray(out["wl_z_grid"]), z_grid)
    np.testing.assert_allclose(np.asarray(out["wl_log_mu_grid"]), log_mu_grid)
    np.testing.assert_allclose(np.asarray(out["wl_log_p_table"]), log_p_table)


def test_load_wl_table_arrays_rejects_unresolved_table(tmp_path):
    """A table narrower than the 16-node ln-mu quadrature is a startup error.

    The likelihood integrates the tabulated backend on a FIXED Gauss-Legendre
    grid (ln mu in (-0.6, 0.6), 16 nodes). The shipped default lognormal width
    (a=4e-3, b=1.5) is far narrower than the node spacing below z ~ 0.5, so
    int p_WL(mu|z) dmu collapses towards 0 and those events would be silently
    DELETED from the likelihood. Fail loudly instead.
    """
    path = tmp_path / "narrow_table.h5"
    _write_wl_table(path, a=4.0e-3, b=1.5, z_lo=0.1)
    opts = types.SimpleNamespace(
        wl_backend="tabulated", lensing_wl_table_path=str(path)
    )
    with pytest.raises(SystemExit, match="does not resolve"):
        lens_cli._load_wl_table_arrays(opts)


def test_load_wl_table_arrays_empty_for_non_tabulated():
    """The loader is inert (empty dict) for lognormal/disabled backends."""
    for backend in ("lognormal", "disabled"):
        opts = types.SimpleNamespace(
            wl_backend=backend, lensing_wl_table_path=None
        )
        assert lens_cli._load_wl_table_arrays(opts) == {}


def test_load_wl_table_arrays_requires_path_when_tabulated():
    """Tabulated without --lensing_wl_table_path is a clean SystemExit."""
    opts = types.SimpleNamespace(
        wl_backend="tabulated", lensing_wl_table_path=None
    )
    with pytest.raises(SystemExit):
        lens_cli._load_wl_table_arrays(opts)
