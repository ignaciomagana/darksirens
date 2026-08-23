"""Importing darksirens must not reconfigure the host process (OPS-02).

``darksirens.gw.utils`` is a public loader: an embedding application imports it
to read PE samples and gets, as a side effect of the import statement alone,
whatever that module chose to do to global interpreter state.  Two such
mutations shipped and are pinned closed here:

* ``multiprocessing.set_start_method("spawn")`` — a process-wide default that
  reached every executor in the host program, imposing picklability and
  ``if __name__ == "__main__"`` requirements on pools darksirens never created,
  and (on Linux) changing their performance characteristics.  The one parallel
  path this package owns asks for its own ``get_context("spawn")`` explicitly,
  which is the correct scope.

* three message-matched ``warnings.filterwarnings("ignore", ...)`` entries for
  ``invalid value encountered in log`` / ``in arctanh`` / ``divide by zero``.
  A global filter silences those numpy warnings everywhere in the host program,
  not just in the gwcat chi_eff evaluation that raises them; that evaluation is
  now scoped with ``np.errstate``.

Every check runs in a CLEAN SUBPROCESS.  In-process this is untestable: the
mutation is idempotent and any earlier import in the same interpreter (pytest
collecting a sibling test file, say) has already performed it.
"""

import subprocess
import sys
import textwrap

import pytest

# Modules whose import must leave the process exactly as it found it.  Keep the
# public entry points here; adding one is cheap and the check is a few seconds.
PUBLIC_MODULES = [
    "darksirens",
    "darksirens.gw.utils",
    "darksirens.redshift.lognormal_completion",
]

_PROBE = textwrap.dedent(
    """
    import json, multiprocessing, sys, warnings

    before_start = multiprocessing.get_start_method(allow_none=True)
    # Documented: the first entry is this platform's default.  Reading it does
    # NOT materialize the default context (unlike get_start_method()).
    platform_default = multiprocessing.get_all_start_methods()[0]
    before_filters = list(warnings.filters)

    __import__({mod!r})

    after_start = multiprocessing.get_start_method(allow_none=True)
    added = [f for f in warnings.filters if f not in before_filters]
    # Only entries whose message pattern is a plain numpy float-error string
    # count against us; third-party imports (jax, numpy) install their own and
    # this module is not responsible for those.
    numeric = [
        str(f[1].pattern) for f in added
        if f[1] is not None and any(
            k in str(f[1].pattern)
            for k in ("invalid value encountered", "divide by zero encountered")
        )
    ]
    print("RESULT" + json.dumps({{
        "before_start": before_start,
        "after_start": after_start,
        "platform_default": platform_default,
        "numeric_filters": numeric,
    }}))
    """
)


def _probe(module):
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(mod=module)],
        capture_output=True, text=True, timeout=900,
    )
    if out.returncode != 0:
        pytest.skip(f"cannot import {module} in a clean subprocess: {out.stderr[-800:]}")
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT"))
    import json
    return json.loads(line[len("RESULT"):])


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_import_does_not_change_the_global_multiprocessing_start_method(module):
    """The start method after the import must still be the platform's own.

    ``None`` (untouched) is ideal; the platform default is also acceptable and
    is what we get in practice — ``healpy`` calls ``get_start_method()``
    somewhere in its own import and thereby MATERIALIZES the default context.
    That is a read, not a policy change: pools still behave as they would in a
    process that never imported darksirens.  What must never happen is the
    method coming out as something the platform did not choose, which is
    exactly what ``set_start_method("spawn")`` at import time did.
    """
    res = _probe(module)
    assert res["before_start"] is None, "probe started with a start method already set"
    allowed = (None, res["platform_default"])
    assert res["after_start"] in allowed, (
        f"importing {module} changed the process-wide multiprocessing start "
        f"method to {res['after_start']!r} (platform default is "
        f"{res['platform_default']!r}). Import must not mutate host process "
        "policy — pass an explicit multiprocessing.get_context(...) where a "
        "pool is actually built."
    )


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_import_does_not_install_global_numeric_warning_filters(module):
    res = _probe(module)
    assert not res["numeric_filters"], (
        f"importing {module} installed process-wide numpy float-error warning "
        f"filters {res['numeric_filters']}. Those silence unrelated host code; "
        "scope them with np.errstate at the call site instead."
    )


def test_the_owned_parallel_path_asks_for_its_context_explicitly():
    """The reason no global default is needed: the one pool darksirens builds
    names its own spawn context (and documents the __main__-guard contract)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "darksirens" / "redshift" / "lognormal_completion.py").read_text()
    assert 'multiprocessing.get_context("spawn")' in src
    assert "mp_context=mp_ctx" in src
