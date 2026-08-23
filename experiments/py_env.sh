# shellcheck shell=bash
#
# Shared campaign-script preamble: locate an interpreter that can actually run
# this repo, and say out loud which one it is (OPS-03).
#
# The problem this replaces.  Every campaign script called bare `python` and
# exported PYTHONPATH as a hard-coded absolute path to one particular checkout.
# On a login shell whose PATH does not have the conda env in front, bare
# `python` here is /usr/bin/python3 == 3.6.8 with no JAX; a worktree or a second
# clone silently ran against the OTHER checkout's darksirens; and neither
# failure announced itself before the build had started writing artifacts.
#
# Contract for callers:
#   REPO_ROOT must already be set (the caller walks up from its own location to
#   the directory holding setup.py).  On return, PYTHON holds an absolute
#   interpreter path and PYTHONPATH points at REPO_ROOT.  Use "$PYTHON" for
#   EVERY step -- the builder AND the little `-c "import json"` reads -- so one
#   run cannot straddle two interpreters.
#
# Usage:
#   HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   REPO_ROOT="$HERE"
#   while [ ! -f "$REPO_ROOT/setup.py" ] && [ "$REPO_ROOT" != / ]; do
#     REPO_ROOT="$(dirname "$REPO_ROOT")"; done
#   . "$REPO_ROOT/experiments/py_env.sh"

if [ -z "${REPO_ROOT:-}" ] || [ ! -f "$REPO_ROOT/setup.py" ]; then
  echo "py_env.sh: REPO_ROOT is unset or does not hold setup.py (got '${REPO_ROOT:-}')." >&2
  echo "  The caller must derive it from its own location, not hard-code it." >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- pick the interpreter -------------------------------------------------
# $PYTHON wins if set, and is NOT second-guessed: an explicit choice that
# cannot import jax is an error to report, not a reason to search elsewhere.
_ds_can_run() { "$1" -c 'import jax' >/dev/null 2>&1; }

if [ -n "${PYTHON:-}" ]; then
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "py_env.sh: \$PYTHON='$PYTHON' is not executable." >&2
    exit 2
  fi
  PYTHON="$(command -v "$PYTHON")"
  if ! _ds_can_run "$PYTHON"; then
    echo "py_env.sh: \$PYTHON='$PYTHON' cannot import jax:" >&2
    "$PYTHON" -c 'import jax' 2>&1 | sed 's/^/    /' >&2
    exit 2
  fi
else
  PYTHON=""
  for _cand in python3 python; do
    _p="$(command -v "$_cand" 2>/dev/null)" || continue
    if _ds_can_run "$_p"; then PYTHON="$_p"; break; fi
  done
  if [ -z "$PYTHON" ]; then
    echo "py_env.sh: no interpreter on PATH can import jax." >&2
    echo "  tried:" >&2
    for _cand in python3 python; do
      _p="$(command -v "$_cand" 2>/dev/null)" || continue
      echo "    $_cand -> $_p ($("$_p" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo '?'))" >&2
    done
    echo "  activate the campaign env, or set PYTHON=/path/to/python." >&2
    exit 2
  fi
fi
export PYTHON

# --- say what we are about to run on --------------------------------------
# Printed BEFORE any artifact is written, so a log that starts with the wrong
# Python or the wrong device is obvious at the top rather than at the autopsy.
"$PYTHON" - <<'PY' || exit 2
import os, sys
import jax, jaxlib
# The CLIs call configure_jax_runtime(), which turns x64 on; report the
# precision the RUN will use, not the import-time default.
try:
    from darksirens.core.jax_config import configure_jax_runtime
    configure_jax_runtime()
    root = "ok"
except Exception as exc:                                   # pragma: no cover
    root = f"darksirens NOT importable: {exc!r}"
import jax.numpy as jnp
print("[env] executable   :", sys.executable)
print("[env] python       :", sys.version.split()[0])
print("[env] jax / jaxlib :", jax.__version__, "/", jaxlib.__version__)
print("[env] x64          :", bool(jax.config.jax_enable_x64),
      "(default float dtype", jnp.zeros(1).dtype, ")")
print("[env] JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))
print("[env] devices      :", ", ".join(str(d) for d in jax.devices()))
print("[env] PYTHONPATH[0]:", os.environ.get("PYTHONPATH", "").split(os.pathsep)[0])
print("[env] darksirens   :", root)
if root != "ok":
    sys.exit(2)
PY
