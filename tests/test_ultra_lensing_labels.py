"""The lensing results.hdf5 'labels' contract vs the shared analyze reader.

The main CLI archives labels as a string DATASET (io/results.py), but the
lensing CLI wrote only ``f.attrs["labels"] = json.dumps(labels)`` and its
settings.json carries no labels key.  analyze._merge_hdf5_metadata merged the
attr verbatim, so ``settings["labels"]`` was the raw JSON STRING and
``_labels_of`` iterated it per character: every label-driven summary silently
found no parameters, while the dataset fallback could never fire because the
attr had already claimed the key.

Fixed on both sides: the reader json-decodes a str-valued labels attr (only
the canonical JSON-list form), so existing archived lensing runs read
correctly; the lensing writer also archives the canonical string dataset,
keeping the JSON attr for the scripts/mock_lensing readers that parse it.
"""
import json

import h5py
import numpy as np


def _write_legacy_lensing_result(path, labels):
    """A results.hdf5 exactly as the pre-fix lensing CLI archived it:
    samples dataset + JSON-string labels attr + completion marker."""
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=np.zeros((3, len(labels))))
        f.attrs["labels"] = json.dumps(labels)
        f.attrs["result_complete"] = True


def test_analyze_decodes_the_legacy_lensing_labels_attr(tmp_path):
    """Existing archived lensing runs (attr-only) must read as a label LIST."""
    from darksirens.cli.analyze import _labels_of, load_run

    labels = ["H0", "Om0", "log10_tau_A"]
    _write_legacy_lensing_result(tmp_path / "results.hdf5", labels)

    settings, samples, _logZ, _logZerr = load_run(str(tmp_path))
    assert _labels_of(settings) == labels
    assert samples.shape == (3, len(labels))


def test_analyze_does_not_adopt_a_scalar_json_labels_attr(tmp_path):
    """The decode is guarded: only the canonical JSON-list form is trusted; a
    scalar decode falls through unchanged rather than becoming labels."""
    from darksirens.cli.analyze import _merge_hdf5_metadata

    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=np.zeros((3, 1)))
        f.attrs["labels"] = json.dumps("H0")
    with h5py.File(path, "r") as f:
        merged = _merge_hdf5_metadata({}, f)
    assert merged["labels"] == '"H0"'


def test_analyze_still_backfills_labels_from_the_dataset(tmp_path):
    """Main-CLI-style files (labels dataset, no attr) are unaffected."""
    from darksirens.cli.analyze import _labels_of, load_run

    labels = ["H0", "Om0"]
    with h5py.File(tmp_path / "results.hdf5", "w") as f:
        f.create_dataset("samples", data=np.zeros((3, len(labels))))
        f.create_dataset(
            "labels",
            data=np.array(labels, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        f.attrs["result_complete"] = True

    settings, _samples, _logZ, _logZerr = load_run(str(tmp_path))
    assert _labels_of(settings) == labels


def test_lensing_save_archives_canonical_labels_dataset(tmp_path):
    """The fixed writer emits BOTH forms, and the run dir round-trips through
    the shared reader to the original label list."""
    from tests.test_lensing_cli_defects import _run_save_phase

    from darksirens.cli.analyze import _labels_of, load_run

    attrs, _settings, _lo, _hi = _run_save_phase(
        tmp_path, extra_args=("--fix_lens_rate", "false")
    )
    labels = json.loads(attrs["labels"])  # attr kept for mock_lensing scripts
    assert isinstance(labels, list) and "H0" in labels

    with h5py.File(tmp_path / "results.hdf5", "r") as f:
        stored = [
            lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
            for lbl in f["labels"][()]
        ]
    assert stored == labels

    settings, _samples, _logZ, _logZerr = load_run(str(tmp_path))
    assert _labels_of(settings) == labels
