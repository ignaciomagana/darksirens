"""x64 must be on before the first JAX-dependent import (review finding JAX-06).

The population registry builds ``MASS_GRID`` / ``Q_GRID`` / ``CHI_GRID`` /
``M1_MESH`` and their ``lru_cache``d getters AT IMPORT TIME.  A JAX array keeps
the precision it was built with, so enabling x64 *after* those imports leaves
them float32 for the life of the process — no later ``jax.config.update`` can
retroactively promote them.  ``darksirens_analyze`` imported JAX and the
population modules before anything enabled x64 and so recomputed its
posterior-predictive densities against float32 grids.

Every assertion here has to run in a COLD SUBPROCESS: ``tests/conftest.py``
enables x64 before collection, so inside the pytest process the bug is invisible
by construction.  That is exactly why it survived.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _cold_env():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("JAX_ENABLE_X64", None)
    return env


def _cold_run(code):
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=_cold_env(), timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()


_ANALYZE_PROBE = """
import jax
before = jax.config.jax_enable_x64
import darksirens.cli.analyze  # noqa: F401
from darksirens.gw.populations import utils as pu
from darksirens.redshift import grid as rgrid
print(before)
print(jax.config.jax_enable_x64)
for name in ("MASS_GRID", "Q_GRID", "CHI_GRID", "M1_MESH"):
    print(name, getattr(pu, name).dtype)
print("get_mass_grid", pu.get_mass_grid().dtype)
print("get_q_grid", pu.get_q_grid().dtype)
print("get_chi_grid", pu.get_chi_grid().dtype)
print("zgrid", rgrid.zgrid.dtype)
"""


def test_analyze_import_leaves_population_grids_in_float64():
    # ~4 s warm; the cost is one interpreter start plus the analyze CLI's
    # matplotlib/seaborn/healpy imports.
    lines = _cold_run(_ANALYZE_PROBE)
    assert lines[0] == "False", "the probe must start from a genuinely cold x64"
    assert lines[1] == "True", "importing the analyze CLI must enable x64"
    for line in lines[2:]:
        name, dtype = line.split()
        assert dtype == "float64", f"{name} is {dtype}: x64 was enabled too late"


_LENSING_PROBE = """
import jax
import darksirens.lensing as lensing
import jax.numpy as jnp
bad = []
for name in dir(lensing):
    value = getattr(lensing, name)
    if isinstance(value, jnp.ndarray) and value.dtype == jnp.float32:
        bad.append(name)
print(",".join(bad))
"""


def test_lensing_import_builds_no_float32_device_arrays():
    """The lensing subpackage's stated design contract: no module-level caches.

    A cold ``import darksirens.lensing`` leaves x64 OFF — deliberately, since a
    library import must not flip a global runtime flag out from under an
    embedding application (``darksirens.gw.utils`` refuses to enable x64 for the
    same reason; the CLIs configure it).  That is only safe as long as the
    import itself materialises nothing, which is what this pins.
    """
    lines = _cold_run(_LENSING_PROBE)
    assert lines == [""] or lines == [], f"import-time float32 arrays: {lines}"
