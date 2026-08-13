"""gwcat-2.0 export compatibility for the darksirens loaders.

These tests write tiny HDF5 fixtures inline (numpy/h5py only) and do NOT
depend on the gwcat package being installed. They verify that:

  * a ``gwcat-pe-2.0`` / ``gwcat-selection-2.0`` file in the ``chieff`` spin
    basis is accepted and yields byte-for-byte the same arrays as the
    equivalent legacy 1.0 file;
  * a ``component`` (non chi_eff) spin basis is rejected at the spin_basis
    gate with an actionable, re-export-suggesting error -- before the
    member/attr check -- and the extra spin datasets are ignored; and
  * an unknown format generation is still rejected.
"""
import sys
import types

# ``darksirens.gw.utils`` imports tqdm at module import time; stub it so the
# loaders are importable without the optional progress-bar dependency (mirrors
# tests/test_data_loader.py).
_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

import h5py
import numpy as np
import pytest

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.lensing import file_contract


# ----------------------------------------------------------------------------
# Deterministic sample payloads shared by the v1 and v2 fixtures.
# ----------------------------------------------------------------------------
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
    "redshift": np.linspace(0.1, 0.3, _N),
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

# Extra spin datasets carried by 2.0 selection/PE files that darksirens ignores.
_EXTRA_SPIN = ("a1", "a2", "cost1", "cost2", "chip")


def _write_pe(path, *, format_version, spin_basis=None, include_chi_eff_in_p_pe=True,
              extra_spin_datasets=False, chi_eff_in_p_pe=True, mock_data=False):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        if spin_basis is not None:
            f.attrs["spin_basis"] = spin_basis
        f.attrs["nsamp"] = NSAMP
        f.attrs["nobs"] = NOBS
        f.attrs["pe_cosmology_H0"] = 67.7
        f.attrs["pe_cosmology_Om0"] = 0.31
        if include_chi_eff_in_p_pe:
            f.attrs["chi_eff_in_p_pe"] = bool(chi_eff_in_p_pe)
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["mock_data"] = bool(mock_data)
        for name, arr in _PE_DATA.items():
            f.create_dataset(name, data=arr)
        if extra_spin_datasets:
            for name in _EXTRA_SPIN:
                f.create_dataset(name, data=np.linspace(0.0, 0.9, _N))


def _write_selection(path, *, format_version, spin_basis=None, extra_spin_datasets=False,
                     chi_eff_swap_applied=True):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        if spin_basis is not None:
            f.attrs["spin_basis"] = spin_basis
        f.attrs["ndraw"] = 1000
        if chi_eff_swap_applied is not None:
            f.attrs["chi_eff_swap_applied"] = bool(chi_eff_swap_applied)
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["cosmology_H0"] = 67.7
        f.attrs["cosmology_Om0"] = 0.31
        for name, arr in _SEL_DATA.items():
            f.create_dataset(name, data=arr)
        if extra_spin_datasets:
            for name in _EXTRA_SPIN:
                f.create_dataset(name, data=np.linspace(0.0, 0.9, _SEL_N))


# ----------------------------------------------------------------------------
# PE loader
# ----------------------------------------------------------------------------
def test_pe_v2_chieff_matches_v1_arrays(tmp_path):
    v1 = tmp_path / "pe_v1.h5"
    v2 = tmp_path / "pe_v2.h5"
    _write_pe(v1, format_version="gwcat-1.0")
    _write_pe(v2, format_version="gwcat-pe-2.0", spin_basis="chieff",
              extra_spin_datasets=True)

    out_v1 = load_gw_samples(v1)
    out_v2 = load_gw_samples(v2)

    assert out_v1[-2:] == out_v2[-2:]  # (nEvents, nsamp) scalars
    for a, b in zip(out_v1[:-2], out_v2[:-2]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_pe_v2_component_basis_rejected_at_gate(tmp_path):
    v2 = tmp_path / "pe_component.h5"
    # component basis: chi_eff_in_p_pe attr absent, extra spin datasets present.
    _write_pe(v2, format_version="gwcat-pe-2.0", spin_basis="component",
              include_chi_eff_in_p_pe=False, extra_spin_datasets=True)

    with pytest.raises(RuntimeError) as excinfo:
        load_gw_samples(v2)
    msg = str(excinfo.value)
    assert "component" in msg
    assert "chieff" in msg
    # Must be the spin_basis gate, NOT the missing-attr check.
    assert "chi_eff_in_p_pe" not in msg
    assert "spin_basis" in msg


def test_pe_unknown_format_rejected(tmp_path):
    bad = tmp_path / "pe_future.h5"
    _write_pe(bad, format_version="gwcat-pe-3.0", spin_basis="chieff")
    with pytest.raises(RuntimeError) as excinfo:
        load_gw_samples(bad)
    assert "gwcat-pe-3.0" in str(excinfo.value)


# ----------------------------------------------------------------------------
# Selection loader
# ----------------------------------------------------------------------------
def test_selection_v2_chieff_matches_v1_arrays(tmp_path):
    v1 = tmp_path / "sel_v1.h5"
    v2 = tmp_path / "sel_v2.h5"
    _write_selection(v1, format_version="gwcat-selection-1.0")
    _write_selection(v2, format_version="gwcat-selection-2.0", spin_basis="chieff",
                     extra_spin_datasets=True)

    out_v1 = load_selection_samples(v1)
    out_v2 = load_selection_samples(v2)

    assert out_v1[-1] == out_v2[-1]  # ndraw
    for a, b in zip(out_v1[:-1], out_v2[:-1]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # pdraw scale explicitly preserved (index 6 in the return tuple).
    np.testing.assert_array_equal(np.asarray(out_v2[6]), _SEL_DATA["pdraw"])


def test_selection_v2_component_basis_rejected_at_gate(tmp_path):
    v2 = tmp_path / "sel_component.h5"
    _write_selection(v2, format_version="gwcat-selection-2.0", spin_basis="component",
                     extra_spin_datasets=True)
    with pytest.raises(RuntimeError) as excinfo:
        load_selection_samples(v2)
    msg = str(excinfo.value)
    assert "component" in msg
    assert "chieff" in msg
    assert "spin_basis" in msg


def test_selection_unknown_format_rejected(tmp_path):
    bad = tmp_path / "sel_future.h5"
    _write_selection(bad, format_version="gwcat-selection-3.0", spin_basis="chieff")
    with pytest.raises(RuntimeError) as excinfo:
        load_selection_samples(bad)
    assert "gwcat-selection-3.0" in str(excinfo.value)


# ----------------------------------------------------------------------------
# file_contract selection validator
# ----------------------------------------------------------------------------
def test_file_contract_accepts_v2_chieff_selection(tmp_path):
    path = tmp_path / "sel_contract_v2.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff",
                     extra_spin_datasets=True)
    report = file_contract.validate_selection_inputs(path)
    assert report["ok"], report["errors"]
    assert report["summary"]["format_version"] == "gwcat-selection-2.0"
    assert report["summary"]["selection_kind"] == "unlensed"


def test_file_contract_rejects_v2_component_selection(tmp_path):
    path = tmp_path / "sel_contract_component.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="component",
                     extra_spin_datasets=True)
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    joined = " ".join(report["errors"])
    assert "component" in joined
    assert "spin_basis" in joined


# ----------------------------------------------------------------------------
# Spin-measure consistency between the two loaders (review finding F-077)
# ----------------------------------------------------------------------------
def test_mock_data_does_not_override_chi_eff_in_p_pe(tmp_path, monkeypatch):
    """``chi_eff_in_p_pe`` is a required attr so the FILE decides whether p_pe
    already carries the chi_eff prior.  mock_data used to short-circuit it, so a
    mock declaring chi_eff_in_p_pe=False got p_pe without the chi_eff factor
    while the selection loader folded the chi_eff draw density into pdraw --
    numerator and denominator on different spin measures.
    """
    from darksirens.gw import utils as gw_utils

    calls = []

    def fake_chi_eff_prior_logprob(chieff, m1src, m2src, amax=0.99):
        calls.append(float(amax))
        return np.full(np.shape(chieff), np.log(2.0))

    monkeypatch.setattr(gw_utils, "chi_eff_prior_logprob", fake_chi_eff_prior_logprob)

    declared = tmp_path / "pe_mock_needs_chi.h5"
    _write_pe(declared, format_version="gwcat-1.0", chi_eff_in_p_pe=False,
              mock_data=True)
    p_pe = np.asarray(load_gw_samples(declared)[6])
    assert calls == [0.99], "the declared chi_eff prior was not applied"

    # p_pe is renormalised per event, so the constant factor cancels there; the
    # observable effect is that the factor was applied at all (calls above) and
    # the weights match the with-factor reference.
    reference = tmp_path / "pe_mock_has_chi.h5"
    _write_pe(reference, format_version="gwcat-1.0", chi_eff_in_p_pe=True,
              mock_data=True)
    np.testing.assert_allclose(p_pe, np.asarray(load_gw_samples(reference)[6]))


def test_selection_without_the_chi_eff_swap_attr_is_rejected(tmp_path):
    """The selection loader used to default chi_eff_swap_applied to True, i.e.
    fail OPEN on the one attr that says which spin measure pdraw is on."""
    path = tmp_path / "sel_no_swap_attr.h5"
    _write_selection(path, format_version="gwcat-selection-1.0",
                     chi_eff_swap_applied=None)
    with pytest.raises(RuntimeError, match="chi_eff_swap_applied"):
        load_selection_samples(path)


def test_selection_declaring_no_swap_gets_the_chi_eff_draw_density(tmp_path, monkeypatch):
    from darksirens.gw import utils as gw_utils

    monkeypatch.setattr(
        gw_utils, "chi_eff_prior_logprob",
        lambda chieff, m1src, m2src, amax=0.99: np.full(np.shape(chieff), np.log(2.0)),
    )
    path = tmp_path / "sel_needs_swap.h5"
    _write_selection(path, format_version="gwcat-selection-1.0",
                     chi_eff_swap_applied=False)
    pdraw = np.asarray(load_selection_samples(path)[6])
    np.testing.assert_allclose(pdraw, 2.0 * _SEL_DATA["pdraw"])


# ----------------------------------------------------------------------------
# Sky validation (DS-01): both loaders feed ra/dec straight to hp.ang2pix,
# and gwcat legitimately writes NaN sky for semianalytic O1/O2 campaigns.
# ----------------------------------------------------------------------------
def _overwrite_dataset(path, name, values):
    with h5py.File(path, "a") as f:
        del f[name]
        f.create_dataset(name, data=values)


def test_nan_sky_rejected_pe(tmp_path):
    path = tmp_path / "pe_nan_sky.h5"
    _write_pe(path, format_version="gwcat-1.0")
    bad = _PE_DATA["dec"].copy()
    bad[2] = np.nan
    _overwrite_dataset(path, "dec", bad)
    with pytest.raises(RuntimeError, match="non-finite"):
        load_gw_samples(path)


def test_degree_sky_rejected_pe(tmp_path):
    path = tmp_path / "pe_degree_sky.h5"
    _write_pe(path, format_version="gwcat-1.0")
    _overwrite_dataset(path, "ra", np.linspace(10.0, 350.0, _N))
    with pytest.raises(RuntimeError, match=r"\[0, 2\*pi\)"):
        load_gw_samples(path)


def test_nan_sky_rejected_selection(tmp_path):
    path = tmp_path / "sel_nan_sky.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    bad = _SEL_DATA["ra"].copy()
    bad[0] = np.nan
    _overwrite_dataset(path, "ra", bad)
    with pytest.raises(RuntimeError, match="non-finite"):
        load_selection_samples(path)


def test_degree_sky_rejected_selection(tmp_path):
    path = tmp_path / "sel_degree_sky.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    _overwrite_dataset(path, "dec", np.linspace(-45.0, 45.0, _SEL_N))
    with pytest.raises(RuntimeError, match="pi/2"):
        load_selection_samples(path)


def test_polar_dec_accepted(tmp_path):
    """dec exactly at the poles is legal, and ra=0 is legal."""
    path = tmp_path / "sel_poles.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    _overwrite_dataset(path, "dec", np.linspace(-np.pi / 2, np.pi / 2, _SEL_N))
    _overwrite_dataset(path, "ra", np.linspace(0.0, 2 * np.pi - 1e-9, _SEL_N))
    load_selection_samples(path)


def test_partially_skyless_campaign_rejected(tmp_path):
    """A per-campaign sky_position_available with any False entry is refused
    by name, before the (equally fatal) NaN scan."""
    path = tmp_path / "sel_skyless.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["sky_position_available"] = np.array([False, True])
    with pytest.raises(RuntimeError, match="sky_position_available"):
        load_selection_samples(path)


def test_all_sky_available_attr_accepted(tmp_path):
    path = tmp_path / "sel_sky_ok.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["sky_position_available"] = np.array([True, True])
    load_selection_samples(path)


def test_file_contract_rejects_nan_sky_selection(tmp_path):
    path = tmp_path / "sel_nan_sky_fc.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    bad = _SEL_DATA["dec"].copy()
    bad[1] = np.nan
    _overwrite_dataset(path, "dec", bad)
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("dec" in err for err in report["errors"])


def test_file_contract_rejects_degree_sky_selection(tmp_path):
    path = tmp_path / "sel_degree_sky_fc.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    _overwrite_dataset(path, "ra", np.linspace(10.0, 350.0, _SEL_N))
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("ra" in err for err in report["errors"])


def test_file_contract_rejects_skyless_campaign(tmp_path):
    path = tmp_path / "sel_skyless_fc.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["sky_position_available"] = np.array([True, False])
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("sky_position_available" in err for err in report["errors"])
