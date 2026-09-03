"""The recorded size of the Tier-0 gate must match the actual gate (QA-01).

``tests/fast_subset.txt`` and ``docs/source/guide/testing.md`` both stated "32 files,
214 passed, ~3 min", measured 2026-07-29.  By 2026-08-23 the manifest held 49
files and 519 tests taking ~18 minutes in CI — a 6x understatement of what a
contributor is told to expect before every commit, and of the CI budget being
spent against a 30-minute timeout.

A record that nothing checks drifts.  These tests make the file-count and
listed-file halves of the claim self-verifying, so the next person to add a
file to the manifest is told to update the header in the same commit.

Deliberately NOT checked here: the wall-clock minutes. Timing is machine- and
load-dependent and asserting on it would make the gate flaky; the header says
where each timing was measured, and re-measuring is a manual step.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "fast_subset.txt"
DOCS = REPO_ROOT / "docs" / "source" / "testing.md"


def _listed_files():
    return [ln.strip() for ln in MANIFEST.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def _recorded_file_count(text):
    """The `N files` figure from the most recent measurement block."""
    m = re.search(r"(\d+)\s+files,\s+(\d+)\s+tests collected", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def test_every_listed_file_exists():
    missing = [f for f in _listed_files() if not (REPO_ROOT / f).is_file()]
    assert not missing, f"fast_subset.txt lists files that do not exist: {missing}"


def test_manifest_has_no_duplicate_entries():
    listed = _listed_files()
    dupes = sorted({f for f in listed if listed.count(f) > 1})
    assert not dupes, f"fast_subset.txt lists these twice: {dupes}"


def test_manifest_header_records_the_current_file_count():
    n_listed = len(_listed_files())
    n_rec, _ = _recorded_file_count(MANIFEST.read_text())
    assert n_rec is not None, (
        "fast_subset.txt no longer carries an 'N files, M tests collected' "
        "measurement line — the size of this gate must stay on the record."
    )
    assert n_rec == n_listed, (
        f"fast_subset.txt lists {n_listed} files but its header records "
        f"{n_rec}. Re-measure and update the header (and docs/source/"
        "testing.md) in the commit that changes the manifest."
    )


def test_docs_record_agrees_with_the_manifest_record():
    if not DOCS.is_file():  # pragma: no cover - docs pruned from an sdist
        pytest.skip("docs/source/guide/testing.md not in this checkout")
    n_man, t_man = _recorded_file_count(MANIFEST.read_text())
    n_doc, t_doc = _recorded_file_count(DOCS.read_text())
    assert n_doc is not None, (
        "docs/source/guide/testing.md no longer states the gate size; it and "
        "fast_subset.txt are the two places the contract is published."
    )
    assert (n_doc, t_doc) == (n_man, t_man), (
        f"docs say {n_doc} files / {t_doc} tests, fast_subset.txt says "
        f"{n_man} / {t_man}. These drifted apart once already."
    )
