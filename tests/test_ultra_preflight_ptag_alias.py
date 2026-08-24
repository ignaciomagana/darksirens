"""Preflight must not report a pair-tag the loader will never read.

load_lensed_injections consumes ONLY log_p_tag_per_source/p_tag_per_source
(lensed_injections.py), and file_contract.py defines p_tag_present as
"p_tag_per_source" in f.  _check_lensed used to also accept bare
'log_p_tag'/'p_tag' dataset names, stamping p_tag_present=True into the
preflight record for a campaign whose run then silently defaults every
kept-source tag to 1 (the both-detected approximation).  These tests pin the
fail-closed behavior: a bare name without a *_per_source dataset is an error.
"""

import h5py
import numpy as np

from darksirens.lensing.preflight import _check_lensed


def _lensed_file(tmp_path, **datasets):
    path = tmp_path / "lensed_injections.h5"
    with h5py.File(path, "w") as f:
        f.attrs["Ndraw_sources"] = 10
        for name, values in datasets.items():
            f[name] = np.asarray(values, dtype=float)
    return str(path)


def _run(path):
    errors, summary = [], {}
    _check_lensed(path, errors, summary)
    return errors, summary


def test_bare_p_tag_only_is_an_error_and_not_reported_present(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path, p_tag=[0.5, 0.7]))
    assert summary["p_tag_present"] is False
    assert any(
        "'p_tag' is not read by load_lensed_injections" in e
        and "p_tag_per_source" in e
        for e in errors
    ), errors


def test_bare_log_p_tag_only_is_an_error_and_not_reported_present(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path, log_p_tag=[-0.1, -0.5]))
    assert summary["p_tag_present"] is False
    assert any(
        "'log_p_tag' is not read by load_lensed_injections" in e
        and "log_p_tag_per_source" in e
        for e in errors
    ), errors


def test_per_source_p_tag_passes_and_reports_present(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path, p_tag_per_source=[0.5, 1.0]))
    assert errors == []
    assert summary["p_tag_present"] is True


def test_bare_alias_alongside_per_source_is_not_an_error(tmp_path):
    # The loader reads the *_per_source dataset; the stray bare name is inert
    # but does not contradict the preflight record.
    errors, summary = _run(
        _lensed_file(tmp_path, p_tag_per_source=[0.5], p_tag=[0.5])
    )
    assert errors == []
    assert summary["p_tag_present"] is True


def test_no_p_tag_dataset_defaults_to_one(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path))
    assert errors == []
    assert summary["p_tag_present"] is False
    assert summary["p_tag_default"] == 1


def test_per_source_p_tag_out_of_range_still_errors(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path, p_tag_per_source=[1.5]))
    assert summary["p_tag_present"] is True
    assert any("p_tag_per_source must be finite and in [0, 1]" in e for e in errors)


def test_per_source_log_p_tag_positive_still_errors(tmp_path):
    errors, summary = _run(_lensed_file(tmp_path, log_p_tag_per_source=[0.1]))
    assert summary["p_tag_present"] is True
    assert any(
        "log_p_tag_per_source must be finite/-inf and <= 0" in e for e in errors
    )
