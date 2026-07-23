"""Regression tests for the redshift-grid x64 / endpoint robustness fixes.

Guards two import-time hazards:

1. The module-level ``darksirens.redshift.grid.zgrid`` must be float64 even on a
   COLD import that has not pre-enabled x64.  ``redshift/__init__`` imports
   ``grid`` before ``cosmology`` (which self-enables x64) and the package root
   ``__init__`` is side-effect-free, so without ``grid`` self-enabling x64 the
   grid froze to float32 for the process lifetime -- the trap documented in
   this directory's ``conftest.py``.  Tested in a FRESH interpreter (no
   conftest, no pre-enable) so it actually exercises the cold path.

2. The redshift grid's top node must not exceed the cosmology distance table's
   top node, or ``dV_of_z(zgrid[-1])`` interpolates out of the table and
   returns NaN (the two grids are built independently, with different node
   counts and different libraries).
"""
import subprocess
import sys
from pathlib import Path

import jax
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh(code: str) -> str:
    """Run ``code`` in a fresh interpreter rooted at the repo (no conftest)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result.stdout.strip()


def test_zgrid_is_float64_on_cold_import():
    # No x64 enabled before the import: grid.py must self-enable it so the
    # module-level array is float64, not a frozen float32.
    out = _run_fresh(
        "import darksirens.redshift.grid as g; print(g.zgrid.dtype)"
    )
    assert out.splitlines()[-1] == "float64", (
        f"zgrid dtype was {out!r} on a cold import; grid.py must enable x64 "
        "before building the grid (see conftest.py)."
    )


def test_redshift_grid_endpoint_within_distance_table():
    # The distance table is interpolated over cosmology.zgrid; a redshift grid
    # whose top node exceeds it yields NaN from dV_of_z at that node.
    import darksirens.redshift.grid as g
    import darksirens.utils.cosmology as c
    import jax.numpy as jnp

    assert float(g.zgrid[-1]) <= float(c.zgrid[-1]), (
        "redshift grid endpoint exceeds the cosmology distance-table endpoint; "
        "dV_of_z will return NaN at the top redshift node."
    )
    dv = c.dV_of_z(g.zgrid[-1], 70.0, 0.3, -1.0, 0.0)
    assert bool(jnp.isfinite(dv)), "dV_of_z(zgrid[-1]) is not finite."


def test_gw_loaders_require_x64():
    # The PE / selection loaders must fail LOUDLY (not silently downcast the
    # importance weights to float32) if x64 is off.  Forced off after import
    # inside a fresh interpreter so no global state leaks into the suite.
    out = _run_fresh(
        "import jax\n"
        "from darksirens.gw import utils\n"
        "jax.config.update('jax_enable_x64', False)\n"
        "n = 0\n"
        "for fn in (utils.load_gw_samples, utils.load_selection_samples):\n"
        "    try:\n"
        "        fn('/nonexistent/does-not-exist.h5')\n"
        "    except RuntimeError as e:\n"
        "        assert 'x64' in str(e), str(e)\n"
        "        n += 1\n"
        "    except Exception as e:\n"
        "        raise AssertionError('expected RuntimeError(x64), got ' + repr(e))\n"
        "print('guards_fired', n)"
    )
    assert out.splitlines()[-1] == "guards_fired 2"
