"""io.results must be importable without the JAX likelihood/sampler stack.

``darksirens.io.results`` is imported by tooling that only reads/writes HDF5
(checkpointing's completion probe, external result readers), so its module
scope must stay JAX-free: the heavy imports (inference.sampling for
``_json_safe_tinyns_value``, likelihood.selection for the guard default,
io.settings' provenance jax/healpy block, darksirens.sky) are all deferred to
call time.  This regressed once -- the module top pulled the full JAX stack
while its own docstring claimed otherwise -- so pin it in a fresh interpreter
(the pytest process already has jax loaded and cannot observe the leak).
"""

import json
import os
import subprocess
import sys

import darksirens

_HEAVY_MODULES = (
    "jax",
    "jaxlib",
    "numpyro",
    "healpy",
    "darksirens.inference.sampling",
    "darksirens.likelihood.selection",
    "darksirens.gw.populations.utils",
    "darksirens.sky",
)

_PROBE = """
import json, sys
import darksirens.io.results
heavy = [m for m in {heavy!r} if m in sys.modules]
print(json.dumps(heavy))
"""


def test_io_results_imports_without_jax_or_sampler_stack():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(darksirens.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (repo_root, env.get("PYTHONPATH", "")) if p
    )
    # Safety only: jax must not be imported at all, but if the assertion is
    # about to fail we still must not touch the GPU.
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=_HEAVY_MODULES)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=repo_root,
    )
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], (
        f"importing darksirens.io.results pulled in heavy modules: {leaked}"
    )


def test_guard_default_reexport_matches_core_constants():
    """likelihood.selection keeps exporting the guard default (its historical
    home) and the value is the same object core.constants owns."""
    from darksirens.core.constants import (
        DEFAULT_MAX_LIKELIHOOD_VARIANCE as core_value,
    )
    from darksirens.likelihood.selection import (
        DEFAULT_MAX_LIKELIHOOD_VARIANCE as selection_value,
    )

    assert selection_value == core_value == 1.0
