"""Golden regression test for the population-model registry refactor.

The golden file ``tests/data/population_registry_golden.json`` was generated
from the PRE-refactor registry (hardcoded factories).  For every legacy model
name it pins:

    * prior lower/upper bounds,
    * the fiducial parameter vector from ``get_fixed_population_params``,
    * ``log_p_pop`` evaluated at the fiducial on a fixed probe grid.

Parameter LaTeX labels are stored for reference (used to write the migration
note) but are deliberately NOT compared: the refactor is an approved clean
break on labels.  Physics — bounds, fiducials, numerics — must match exactly.

Regenerate (only against code known to be physics-correct) with:

    DARKSIRENS_REGEN_GOLDEN=1 pytest tests/test_population_registry_golden.py
"""

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from darksirens.gw.populations import (
    get_fixed_population_params,
    pop_model_parser,
    pop_model_prior_parser,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "population_registry_golden.json"
REGEN = os.environ.get("DARKSIRENS_REGEN_GOLDEN") == "1"

try:
    import tinygp  # noqa: F401

    # Some test modules insert a tinygp STUB into sys.modules (it raises on
    # GaussianProcess construction).  Stubs have no __file__; treat them as
    # "tinygp unavailable" so GP golden entries skip instead of erroring.
    HAVE_TINYGP = getattr(tinygp, "__file__", None) is not None
except Exception:  # pragma: no cover - absent tinygp
    HAVE_TINYGP = False

_BASE_MIXTURES = [
    "powerlaw+peak",
    "brokenpowerlaw+2peaks",
    "brokenpowerlaw+3peaks",
    "brokenpowerlaw+2peaks+powerlaw",
    "twopowerlaws+peak",
    "twopowerlaws+2peaks",
    "twopowerlaws+3peaks",
]
# NOTE: the legacy mixture-coupled GP models (gp_mass, gp_mass_pairing,
# gp_mass_pairing_joint) were removed.  GP population models are now standalone,
# true-GP-prior models registered via register_model (see populations/gp.py) and
# are covered by dedicated GP validation rather than this mixture golden.
_SUFFIXES = [""]
_CUSTOM = [
    "golomb_1g",
    "golomb_1g+tail",
    "gwtc5_fiducial_brokenpowerlaw+2peaks",
    "gwtc5_brokenpowerlaw+2peaks",
    "gwtc5_fiducial_bpl2peaks",
    # Added after the golden was first recorded: the GWTC-3 Table VI preset is
    # a PUBLISHED model, so its bounds, fiducial and log_p_pop are exactly the
    # things a golden should stop from drifting.  Its entry was appended by a
    # re-record that left every pre-existing entry byte-identical -- the golden
    # still pins the pre-refactor physics for every name it already covered.
    "gwtc3_fiducial_plpeak",
]

LEGACY_NAMES = [b + s for b in _BASE_MIXTURES for s in _SUFFIXES] + _CUSTOM

# Fixed probe grid spanning the mass/spin support.
_M1 = [6.0, 10.0, 35.0, 55.0, 75.0]
_Q = [0.5, 0.9]
_CHI = [0.0, 0.2]
_Z = 0.2

_PROBES = [(m, q, _Z, c) for m in _M1 for q in _Q for c in _CHI]


def _encode_float(x: float):
    """JSON-safe encoding: non-finite values become strings."""
    if math.isfinite(x):
        return x
    return repr(x)  # 'inf', '-inf', 'nan'


def _is_gp(name: str) -> bool:
    return name.startswith("gp_")


def _snapshot(name: str) -> dict:
    with warnings.catch_warnings():
        # Post-refactor, legacy spellings resolve through a deprecation alias.
        warnings.simplefilter("ignore", DeprecationWarning)
        lows, highs, labels, _, latex = pop_model_prior_parser(name)
        fid = np.asarray(get_fixed_population_params(name), dtype=np.float64)
        log_p_pop = pop_model_parser(name)

    m1 = jnp.array([p[0] for p in _PROBES], dtype=jnp.float64)
    q = jnp.array([p[1] for p in _PROBES], dtype=jnp.float64)
    z = jnp.array([p[2] for p in _PROBES], dtype=jnp.float64)
    chi = jnp.array([p[3] for p in _PROBES], dtype=jnp.float64)
    vals = np.asarray(log_p_pop(m1, q, z, chi, jnp.array(fid)), dtype=np.float64)

    return {
        "labels": list(labels),  # reference only, not compared
        "latex": latex,  # reference only, not compared
        "lows": [float(x) for x in lows],
        "highs": [float(x) for x in highs],
        "fiducial": fid.tolist(),
        "log_p_pop": [_encode_float(v) for v in vals.tolist()],
    }


@pytest.mark.skipif(not REGEN, reason="set DARKSIRENS_REGEN_GOLDEN=1 to regenerate")
def test_regenerate_golden():
    golden = {}
    for name in LEGACY_NAMES:
        if _is_gp(name) and not HAVE_TINYGP:
            continue
        golden[name] = _snapshot(name)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1))
    assert GOLDEN_PATH.exists()


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden file missing: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text())


# Relative tolerance for log_p_pop.  This comparison used to be exact `==`,
# which made the guard permanently red — 11 of the 13 models disagree with the
# golden file by 1-4 ULP (~1e-16 relative) on jax 0.4.34 / CPU, i.e. last-bit
# XLA reassociation, not physics.  A red guard is an ignored guard, and this one
# is the only check on the campaign's fiducial population.  1e-12 is ~4 orders
# of magnitude above float64 noise and still ~10 orders below any physics change
# worth catching (a shifted prior bound or peak location moves log_p_pop in the
# 1e-2..1e0 range).  Bounds and fiducial vectors stay exact.
_LOG_P_POP_RTOL = 1e-12


def _log_p_pop_matches(got, want) -> bool:
    # Non-finite entries are stored as strings ('inf', '-inf', 'nan') by
    # _encode_float; those must still match exactly, including nan-for-nan.
    if isinstance(got, str) or isinstance(want, str):
        return got == want
    return math.isclose(got, want, rel_tol=_LOG_P_POP_RTOL, abs_tol=0.0)


@pytest.mark.skipif(REGEN, reason="regeneration run")
@pytest.mark.parametrize("name", LEGACY_NAMES)
def test_registry_matches_golden(name):
    if _is_gp(name) and not HAVE_TINYGP:
        pytest.skip("tinygp unavailable")
    golden = _load_golden()
    if name not in golden:
        pytest.skip(f"{name} not in golden file (generated without tinygp?)")
    want = golden[name]
    got = _snapshot(name)

    assert got["lows"] == want["lows"], f"{name}: prior lower bounds diverged"
    assert got["highs"] == want["highs"], f"{name}: prior upper bounds diverged"
    assert got["fiducial"] == want["fiducial"], f"{name}: fiducial vector diverged"

    for i, (g, w) in enumerate(zip(got["log_p_pop"], want["log_p_pop"])):
        assert _log_p_pop_matches(g, w), (
            f"{name}: log_p_pop diverged at probe {i} ({_PROBES[i]}): "
            f"new={g!r} vs golden={w!r}"
        )
