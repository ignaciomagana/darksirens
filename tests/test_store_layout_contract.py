"""Layout gates on the gwcat PE/selection stores (review DATA-01).

The shared contract used to validate VALUES only, so a store whose columns
disagreed on length loaded cleanly.  The dangerous case is a length-ONE
column: numpy broadcasts it over every sample instead of raising, so a PE
file with a single right ascension assigned six posterior samples six
different HEALPix pixels off one shared ra -- wrong sky rows, no error
anywhere.  A merely-short column (length 5 against 6) was no better: it
loaded and failed much later inside an unrelated broadcast.

These tests pin the adversarial layouts at the loaders AND at the lensing
preflight (which gates the same files), and pin that the well-formed
fixtures -- including one carrying the optional component-spin columns --
still load.
"""
import sys
import types

# ``darksirens.gw.utils`` imports tqdm at module import time; stub it so the
# loaders are importable without the optional progress-bar dependency.
_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

import h5py
import numpy as np
import pytest

from darksirens.gw import store_contract
from darksirens.gw.utils import load_gw_store, load_selection_store
from darksirens.lensing import file_contract

NOBS = 2
NSAMP = 3
_N = NOBS * NSAMP

_PE_DATA = {
    "ra": np.linspace(0.1, 1.0, _N),
    "dec": np.linspace(-0.5, 0.5, _N),
    "m1det": np.linspace(30.0, 40.0, _N),
    "m2det": np.linspace(24.0, 32.0, _N),
    "dL": np.linspace(400.0, 900.0, _N),
    "chieff": np.linspace(-0.2, 0.2, _N),
    "p_pe": np.linspace(1.0, 2.0, _N),
    "m1src": np.linspace(20.0, 28.0, _N),
    "m2src": np.linspace(16.0, 22.0, _N),
}

_SEL_N = 5
_SEL_DATA = {
    "m1det": np.linspace(31.0, 41.0, _SEL_N),
    "m2det": np.linspace(25.0, 33.0, _SEL_N),
    "dL": np.linspace(410.0, 910.0, _SEL_N),
    "chieff": np.linspace(-0.15, 0.15, _SEL_N),
    "ra": np.linspace(0.2, 1.1, _SEL_N),
    "dec": np.linspace(-0.4, 0.4, _SEL_N),
    "pdraw": np.linspace(1.0e-6, 5.0e-6, _SEL_N),
    "m1src": np.linspace(21.0, 29.0, _SEL_N),
    "m2src": np.linspace(17.0, 23.0, _SEL_N),
}


def _write_pe(path, *, overrides=None, attrs=None, extra=None):
    data = dict(_PE_DATA)
    data.update(overrides or {})
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-pe-2.0"
        f.attrs["spin_basis"] = "chieff"
        f.attrs["nsamp"] = NSAMP
        f.attrs["nobs"] = NOBS
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        for name, arr in data.items():
            f.create_dataset(name, data=arr)
        for name, arr in (extra or {}).items():
            f.create_dataset(name, data=arr)
    return path


def _write_selection(path, *, overrides=None, attrs=None, extra=None):
    data = dict(_SEL_DATA)
    data.update(overrides or {})
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-2.0"
        f.attrs["spin_basis"] = "chieff"
        f.attrs["ndraw"] = 1000
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = 0.99
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        for name, arr in data.items():
            f.create_dataset(name, data=arr)
        for name, arr in (extra or {}).items():
            f.create_dataset(name, data=arr)
    return path


# ----------------------------------------------------------------------------
# The well-formed fixtures must keep loading.
# ----------------------------------------------------------------------------
def test_well_formed_pe_and_selection_still_load(tmp_path):
    pe = load_gw_store(_write_pe(tmp_path / "pe.h5"))
    assert np.asarray(pe.columns["ra"]).size == _N
    sel = load_selection_store(_write_selection(tmp_path / "sel.h5"))
    assert np.asarray(sel.columns["ra"]).size == _SEL_N


def test_optional_component_spin_columns_at_the_common_length_still_load(tmp_path):
    """A chi_eff export may carry a1/a2/cost1/cost2; the loaders read them."""
    extra = {name: np.linspace(0.0, 0.9, _N)
             for name in store_contract.COMPONENT_SPIN_DATASETS}
    pe = load_gw_store(_write_pe(tmp_path / "pe_spin.h5", extra=extra))
    assert np.asarray(pe.columns["a1"]).size == _N


# ----------------------------------------------------------------------------
# Singleton broadcast: the case that never failed.
# ----------------------------------------------------------------------------
def test_pe_singleton_ra_is_refused_not_broadcast(tmp_path):
    path = _write_pe(tmp_path / "pe_singleton.h5",
                     overrides={"ra": np.array([0.25])})
    with pytest.raises(RuntimeError) as excinfo:
        load_gw_store(path)
    msg = str(excinfo.value)
    assert "ra=1" in msg
    assert "expected 6" in msg


def test_selection_singleton_ra_is_refused_not_broadcast(tmp_path):
    path = _write_selection(tmp_path / "sel_singleton.h5",
                            overrides={"ra": np.array([0.25])})
    with pytest.raises(RuntimeError) as excinfo:
        load_selection_store(path)
    assert "ra=1" in str(excinfo.value)


def test_pe_singleton_spin_column_is_refused(tmp_path):
    """The optional columns broadcast exactly like the required ones."""
    extra = {name: np.linspace(0.0, 0.9, _N)
             for name in store_contract.COMPONENT_SPIN_DATASETS}
    extra["a1"] = np.array([0.3])
    path = _write_pe(tmp_path / "pe_spin_singleton.h5", extra=extra)
    with pytest.raises(RuntimeError) as excinfo:
        load_gw_store(path)
    assert "a1=1" in str(excinfo.value)


# ----------------------------------------------------------------------------
# Short / mis-shaped columns.
# ----------------------------------------------------------------------------
def test_pe_short_ra_is_refused_at_load(tmp_path):
    path = _write_pe(tmp_path / "pe_short.h5",
                     overrides={"ra": np.linspace(0.1, 1.0, _N - 1)})
    with pytest.raises(RuntimeError, match="expected 6"):
        load_gw_store(path)


def test_pe_two_dimensional_column_is_refused(tmp_path):
    path = _write_pe(tmp_path / "pe_2d.h5",
                     overrides={"ra": _PE_DATA["ra"].reshape(NOBS, NSAMP)})
    with pytest.raises(RuntimeError, match="one-dimensional"):
        load_gw_store(path)


def test_selection_short_ra_is_refused_at_load(tmp_path):
    path = _write_selection(tmp_path / "sel_short.h5",
                            overrides={"ra": np.linspace(0.2, 1.1, _SEL_N - 1)})
    with pytest.raises(RuntimeError, match="inconsistent lengths"):
        load_selection_store(path)


# ----------------------------------------------------------------------------
# Campaign counts.
# ----------------------------------------------------------------------------
def test_zero_ndraw_is_refused(tmp_path):
    """ndraw is the selection integral's denominator; zero is not a count."""
    path = _write_selection(tmp_path / "sel_ndraw0.h5", attrs={"ndraw": 0})
    with pytest.raises(RuntimeError, match="positive count"):
        load_selection_store(path)


def test_ndraw_below_the_detected_count_is_refused(tmp_path):
    path = _write_selection(tmp_path / "sel_ndraw_small.h5",
                            attrs={"ndraw": _SEL_N - 1})
    with pytest.raises(RuntimeError, match="different campaigns"):
        load_selection_store(path)


def test_zero_nobs_is_refused(tmp_path):
    path = _write_pe(tmp_path / "pe_nobs0.h5", attrs={"nobs": 0})
    with pytest.raises(RuntimeError, match="positive count"):
        load_gw_store(path)


# ----------------------------------------------------------------------------
# Mass ordering (review PHY-09).  The pairing models normalise p(q|m1) over
# q <= 1, so a row with m2 > m1 is scored with a density from outside the
# normalisation domain -- the store must not carry one.
# ----------------------------------------------------------------------------
def test_pe_store_with_m2det_above_m1det_is_refused(tmp_path):
    path = _write_pe(tmp_path / "pe_q_gt_1.h5", overrides={
        "m1det": np.full(_N, 30.0), "m2det": np.full(_N, 36.0),
    })
    with pytest.raises(RuntimeError) as excinfo:
        load_gw_store(path)
    msg = str(excinfo.value)
    assert "'m2det' exceeds 'm1det'" in msg
    assert "1.2" in msg


def test_selection_store_with_m2src_above_m1src_is_refused(tmp_path):
    path = _write_selection(tmp_path / "sel_q_gt_1.h5", overrides={
        "m1src": np.full(_SEL_N, 20.0), "m2src": np.full(_SEL_N, 25.0),
    })
    with pytest.raises(RuntimeError, match="'m2src' exceeds 'm1src'"):
        load_selection_store(path)


def test_equal_masses_are_accepted(tmp_path):
    """q = 1 is a physical boundary, not a violation."""
    path = _write_pe(tmp_path / "pe_q_eq_1.h5", overrides={
        "m1det": np.full(_N, 30.0), "m2det": np.full(_N, 30.0),
        "m1src": np.full(_N, 20.0), "m2src": np.full(_N, 20.0),
    })
    assert np.asarray(load_gw_store(path).columns["m2det"]).size == _N


# ----------------------------------------------------------------------------
# The lensing preflight gates the same files and must be at least as strict.
# ----------------------------------------------------------------------------
def test_preflight_refuses_singleton_ra_in_a_selection_file(tmp_path):
    path = _write_selection(tmp_path / "sel_preflight.h5",
                            overrides={"ra": np.array([0.25])})
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("ra=1" in e for e in report["errors"]), report["errors"]


def test_preflight_accepts_the_well_formed_selection_file(tmp_path):
    path = _write_selection(tmp_path / "sel_preflight_ok.h5")
    report = file_contract.validate_selection_inputs(path)
    assert report["ok"], report["errors"]
