"""Campaign entry-point scripts must be self-contained (OPS-03).

The Q-build scripts are the reproducible record of a production campaign: a
reviewer is meant to be able to run one and get the artifact back.  As shipped
they could not do that on any machine but the author's login shell.

Two defects, both silent:

* they called bare ``python``.  With the campaign conda env not first on PATH,
  that resolves to the system interpreter — measured 3.6.8 with no JAX on the
  review node — and the failure surfaces somewhere inside a builder rather than
  at the top of the log.

* they exported ``PYTHONPATH`` as a hard-coded absolute path to one particular
  checkout.  Run from a git worktree or a second clone, the script builds
  artifacts from a DIFFERENT tree than the one it lives in, and nothing says so.

Both are now handled by ``experiments/py_env.sh``, which derives the repo
root from the script's own location, resolves an interpreter that can actually
import jax (honouring ``$PYTHON``), and prints executable / Python / JAX /
precision / device before any artifact is written.  These tests pin the pattern
so the next campaign script does not reintroduce it by copy-paste.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ENV = REPO_ROOT / "experiments" / "py_env.sh"

#: The campaign entry points converted to the shared preamble.  A new one that
#: builds a production artifact belongs here.
CAMPAIGN_SCRIPTS = [
    "experiments/desi_full259/run_qbuild_v4_prod.sh",
    "experiments/field_level_plan/pr6a/build_q_v4_rb.sh",
    "experiments/desi_ingest/run_qbuild_v2_depthmap.sh",
    "experiments/desi_ingest/run_qbuild_v3_depthmap.sh",
    "experiments/desi_ingest/run_qbuild_v4_depthmap.sh",
]


@pytest.fixture(scope="module", params=CAMPAIGN_SCRIPTS)
def script(request):
    p = REPO_ROOT / request.param
    if not p.is_file():
        pytest.skip(f"{request.param} not in this checkout")
    return p


def test_py_env_preamble_exists():
    assert PY_ENV.is_file(), "experiments/py_env.sh is the shared preamble"


def test_py_env_preamble_is_actually_tracked_by_git():
    """It lived at experiments/lib/py_env.sh for one commit and was INVISIBLE.

    .gitignore carries a bare ``lib/`` (packaging boilerplate), which matches a
    directory of that name at ANY depth. Every campaign script would have
    sourced a file that never left this machine. Same failure mode as the
    darksirens/utils namespace-portion bug: the local tree works, the shipped
    one does not.
    """
    check = subprocess.run(
        ["git", "check-ignore", "-q", str(PY_ENV)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert check.returncode != 0, (
        f"{PY_ENV.relative_to(REPO_ROOT)} is matched by .gitignore — the shared "
        "preamble every campaign script sources would not be committed."
    )
    ls = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(PY_ENV.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if ls.returncode != 0:
        pytest.skip("preamble not yet committed (pre-commit working tree)")


def test_script_sources_the_shared_preamble(script):
    text = script.read_text()
    assert 'experiments/py_env.sh' in text, (
        f"{script.name} does not source the shared interpreter preamble"
    )
    assert 'while [ ! -f "$REPO_ROOT/setup.py" ]' in text, (
        f"{script.name} does not derive REPO_ROOT by walking up from its own "
        "location — it will run against whatever tree the path names."
    )


def test_script_has_no_hardcoded_checkout_path(script):
    """A path to one machine's checkout is not a campaign entry point."""
    hits = [ln.strip() for ln in script.read_text().splitlines()
            if re.search(r"(PYTHONPATH|PYTHON)\s*=\s*/", ln) and not ln.lstrip().startswith("#")]
    assert not hits, (
        f"{script.name} hard-codes an absolute checkout/interpreter path: {hits}. "
        "Derive it from the script location (py_env.sh does)."
    )


def test_script_never_calls_a_bare_interpreter(script):
    """Every step — the builder AND the little JSON reads — uses ``$PYTHON``.

    One run must not straddle two interpreters, which is exactly what a bare
    ``$(python -c ...)`` feeding a ``"$PYTHON" -m ...`` build would do.
    """
    bad = []
    for i, line in enumerate(script.read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        # a `python`/`python3` token that is not part of "$PYTHON" / PYTHONPATH
        for m in re.finditer(r"(?<![\w$\"/])python3?(?![\w])", line):
            bad.append(f"{i}: {line.strip()}")
            break
    assert not bad, (
        f"{script.name} invokes a bare interpreter: {bad}. Bare `python` was "
        "/usr/bin/python3 3.6.8 with no JAX on the review node. Use \"$PYTHON\"."
    )


def test_preamble_refuses_an_interpreter_that_cannot_import_jax():
    """Fail fast and loud, before the builder starts writing artifacts."""
    out = subprocess.run(
        ["bash", "-c",
         f'set -euo pipefail; export PYTHON=/usr/bin/python3; '
         f'REPO_ROOT={REPO_ROOT!s}; . "{PY_ENV!s}"'],
        capture_output=True, text=True, timeout=600,
    )
    if out.returncode == 0:
        pytest.skip("/usr/bin/python3 on this box can import jax")
    assert out.returncode != 0
    assert "cannot import jax" in out.stderr, out.stderr


def test_preamble_refuses_a_repo_root_that_is_not_a_checkout(tmp_path):
    out = subprocess.run(
        ["bash", "-c",
         f'set -euo pipefail; REPO_ROOT={tmp_path!s}; . "{PY_ENV!s}"'],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode != 0
    assert "does not hold setup.py" in out.stderr, out.stderr


def test_preamble_points_python_at_this_checkout_and_prints_its_env():
    """The positive path: PYTHONPATH must lead HERE, not to another clone."""
    out = subprocess.run(
        ["bash", "-c",
         f'set -euo pipefail; REPO_ROOT={REPO_ROOT!s}; . "{PY_ENV!s}"; '
         f'echo "PYTHON=$PYTHON"'],
        capture_output=True, text=True, timeout=900,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu", "PYTHON": sys.executable},
    )
    assert out.returncode == 0, out.stderr
    for field in ("[env] executable", "[env] python", "[env] jax / jaxlib",
                  "[env] x64", "[env] devices"):
        assert field in out.stdout, f"{field} missing from the banner:\n{out.stdout}"
    # The whole point: the run resolves against the tree the script lives in.
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("[env] PYTHONPATH[0]"))
    assert line.split(":", 1)[1].strip() == str(REPO_ROOT), line
    assert "[env] darksirens   : ok" in out.stdout
