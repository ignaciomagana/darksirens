"""The argparse-only CLI path must not drag in the runtime stack.

``python -m darksirens.cli.inference --help`` used to cost ~5.9 s and ~830 MB
because the three big CLIs imported their whole runtime dependency graph at
module scope -- jax.numpy, healpy, the population registry, the data loaders,
the likelihood factory, the prior/sampler machinery -- none of which
``build_parser()`` or the post-parse guards can reach.  The test suite spawns
those CLIs at ~70 subprocess call sites, almost all of them ``--help`` or a
post-argparse guard check against a deliberately nonexistent path, so the whole
cost was pure waste.  The heavy imports now live inside the functions that use
them (``build_latent_field.py`` is the long-standing in-repo pattern, and
``tests/test_ultra_io_results_light_import.py`` pins the same discipline for
``darksirens.io.results``, which regressed once).

Two things are NOT deferrable and are asserted as such below:

* ``configure_jax_runtime()`` stays at module scope in all three CLIs.  It
  enables x64 before any JAX array is built anywhere in the process; deferring
  it would silently freeze import-time grids at float32 (see
  ``tests/test_cold_import_precision.py`` and ``tests/test_grid_x64_import.py``).
  It imports ``jax`` itself, so plain ``jax`` in ``sys.modules`` is expected --
  what must NOT be there is the darksirens stack built on top of it.
* ``darksirens.cli.analyze`` applies the publication matplotlib style as an
  import-time side effect that its ``plot_*`` entry points rely on, so seaborn
  and pyplot stay eager there.

The probe runs in a fresh interpreter: the pytest process already has the whole
stack loaded and cannot observe the leak.
"""

import json
import os
import subprocess
import sys

import pytest

import darksirens

#: Modules no CLI may pull in merely by being imported.  ``astropy.cosmology``
#: is on the list because it drags ``astropy.modeling`` -> ``nddata`` ->
#: ``dask``; the Planck15 fiducials are pinned as literals in
#: ``darksirens.core.constants`` / ``darksirens.utils.cosmology`` instead.
_COMMON_FORBIDDEN = (
    "astropy.cosmology",
    "dask",
    "darksirens.gw.populations",
    "darksirens.gw.samples",
    "darksirens.utils.cosmology",
    "darksirens.redshift.grid",
    "darksirens.redshift.completion",
    "darksirens.inference.prior",
    "darksirens.inference.sampling",
    "darksirens.inference.data",
    "darksirens.likelihood.factory",
    "darksirens.likelihood.selection",
    "darksirens.lensing",
)

_CLI_FORBIDDEN = {
    "darksirens.cli.inference": _COMMON_FORBIDDEN + ("healpy", "seaborn",
                                                     "matplotlib.pyplot"),
    "darksirens.cli.inference_lensing": _COMMON_FORBIDDEN + ("healpy", "seaborn",
                                                             "matplotlib.pyplot"),
    # analyze keeps the plotting stack eager (import-time style side effect).
    "darksirens.cli.analyze": _COMMON_FORBIDDEN + ("healpy",),
}

_PROBE = """
import json, sys
import {module}
print(json.dumps([m for m in {forbidden!r} if m in sys.modules]))
"""


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(darksirens.__file__)))


def _run(code):
    root = _repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (root, env.get("PYTHONPATH", "")) if p
    )
    # These CLIs configure the JAX runtime at import; keep the probe off the GPU.
    env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, env=env, cwd=root,
    )


@pytest.mark.parametrize("module", sorted(_CLI_FORBIDDEN))
def test_cli_module_import_is_light(module):
    out = _run(_PROBE.format(module=module, forbidden=_CLI_FORBIDDEN[module]))
    assert out.returncode == 0, out.stderr
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], (
        f"importing {module} pulled in post-argparse modules: {leaked}"
    )


@pytest.mark.parametrize("module", sorted(_CLI_FORBIDDEN))
def test_cli_configures_jax_runtime_at_module_scope(module):
    """x64 must be on the moment the CLI module finishes importing.

    This is the constraint that stops "make the imports lazy" from becoming
    "make configure_jax_runtime() lazy too".
    """
    code = (
        f"import {module}\n"
        "import jax\n"
        "print(jax.config.jax_enable_x64)\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"


@pytest.mark.parametrize("module", sorted(_CLI_FORBIDDEN))
def test_cli_help_exits_cleanly(module):
    """``--help`` still renders (and is now reachable without the runtime stack)."""
    root = _repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (root, env.get("PYTHONPATH", "")) if p
    )
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=600, env=env, cwd=root,
    )
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout


def test_fiducial_constants_match_planck15():
    """The pinned Planck15 literals are repr-exact, dtype included.

    ``darksirens.core.constants`` and ``darksirens.utils.cosmology`` carry
    literals instead of importing ``astropy.cosmology``; this is the guard that
    keeps them honest (and would catch an Astropy release that changed
    Planck15).
    """
    import numpy as np
    from astropy.cosmology import Planck15

    from darksirens.core.constants import H0_FID, OM0_FID
    from darksirens.utils import cosmology as cosmo

    assert H0_FID == float(Planck15.H0.value)
    assert OM0_FID == float(Planck15.Om0)

    assert cosmo.H0Planck == Planck15.H0.value
    # dtype matters: H0Planck feeds the JAX distance rescaling, where a weakly
    # typed Python float promotes differently from a strong np.float64.
    assert isinstance(cosmo.H0Planck, np.float64)
    assert isinstance(Planck15.H0.value, np.float64)
    assert cosmo.Om0Planck == float(Planck15.Om0)
    assert isinstance(cosmo.Om0Planck, float)


def test_relocated_constants_are_reexported_by_their_owners():
    """The names moved into core.constants keep their canonical spellings."""
    from darksirens.core import constants as core
    from darksirens.gw import populations as pops
    from darksirens.lensing import marginal_diagnostics, pair_tag_selection, slmarks

    assert pops.FIDUCIAL_SET_LEGACY == core.FIDUCIAL_SET_LEGACY
    assert pops.FIDUCIAL_SET_IN_PRIOR == core.FIDUCIAL_SET_IN_PRIOR
    assert pops.FIDUCIAL_SETS == core.FIDUCIAL_SETS
    assert slmarks.DEFAULT_T0_SECONDS == core.DEFAULT_T0_SECONDS
    assert (pair_tag_selection.PAIR_TAG_SELECTION_MODEL_KINDS
            == core.PAIR_TAG_SELECTION_MODEL_KINDS)
    assert (marginal_diagnostics.SIS_TIME_MARK_SHARPNESS
            == core.SIS_TIME_MARK_SHARPNESS)
