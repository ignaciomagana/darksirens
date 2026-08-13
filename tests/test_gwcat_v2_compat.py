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


# ----------------------------------------------------------------------------
# Store record API (DS-04): the tuple loaders are thin wrappers over
# GWStore/SelectionStore and must stay bit-identical.
# ----------------------------------------------------------------------------
def test_tuple_loaders_bit_identical_to_store_api(tmp_path):
    from darksirens.gw.utils import load_gw_store, load_selection_store

    pe = tmp_path / "pe_store.h5"
    _write_pe(pe, format_version="gwcat-pe-2.0", spin_basis="chieff")
    tup = load_gw_samples(pe)
    store = load_gw_store(pe)
    for i, name in enumerate(("m1det", "m2det", "dL", "chieff", "ra", "dec")):
        np.testing.assert_array_equal(np.asarray(tup[i]), store.columns[name])
    np.testing.assert_array_equal(np.asarray(tup[6]), store.prior_wt)
    assert (tup[7], tup[8]) == (store.n_events, store.nsamp)
    assert store.format_version == "gwcat-pe-2.0"
    assert store.fit_columns == ("m1det", "q", "dL", "chieff")
    # Raw column, not the processed weight: prior_wt is normalised per event.
    np.testing.assert_array_equal(store.columns["p_pe"], _PE_DATA["p_pe"])

    sel = tmp_path / "sel_store.h5"
    _write_selection(sel, format_version="gwcat-selection-1.0")
    tup = load_selection_samples(sel)
    store = load_selection_store(sel)
    for i, name in enumerate(("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw")):
        np.testing.assert_array_equal(np.asarray(tup[i]), store.columns[name])
    np.testing.assert_array_equal(np.asarray(tup[6]), store.prior_wt)
    assert tup[7] == store.ndraw == 1000
    assert store.n_injections == _SEL_N


def test_store_processed_prior_wt_when_swap_pending(tmp_path, monkeypatch):
    """prior_wt carries the folded-in chi_eff density; columns['pdraw'] stays raw."""
    from darksirens.gw import utils as gw_utils

    monkeypatch.setattr(
        gw_utils, "chi_eff_prior_logprob",
        lambda chieff, m1src, m2src, amax=0.99: np.full(np.shape(chieff), np.log(3.0)),
    )
    sel = tmp_path / "sel_store_swap.h5"
    _write_selection(sel, format_version="gwcat-selection-1.0",
                     chi_eff_swap_applied=False)
    store = gw_utils.load_selection_store(sel)
    np.testing.assert_allclose(store.prior_wt, 3.0 * _SEL_DATA["pdraw"])
    np.testing.assert_array_equal(store.columns["pdraw"], _SEL_DATA["pdraw"])


def test_store_surfaces_event_names_and_attrs(tmp_path):
    from darksirens.gw.utils import load_gw_store

    pe = tmp_path / "pe_names.h5"
    _write_pe(pe, format_version="gwcat-1.0")
    with h5py.File(pe, "a") as f:
        f.attrs["event_names"] = np.array(["GW150914", "GW151226"], dtype=h5py.string_dtype())
    store = load_gw_store(pe)
    assert store.event_names == ("GW150914", "GW151226")
    assert store.attrs["pe_cosmology_H0"] == 67.7
    assert store.attrs["nobs"] == NOBS

    pe2 = tmp_path / "pe_no_names.h5"
    _write_pe(pe2, format_version="gwcat-1.0")
    assert load_gw_store(pe2).event_names is None


# ----------------------------------------------------------------------------
# Format 2.1 (DS-05): accepted in the chieff basis, contract_hash compared
# across the PE/selection pair, emulator path checks the PE basis.
# ----------------------------------------------------------------------------
def _add_21_contract(path, *, parameter_space="chieff", source_class=None):
    """Stamp the gwcat-2.1 contract attrs the way writers_gwcat21 does."""
    import hashlib
    import json

    pairing = {
        "parameter_space": parameter_space,
        "fit_columns": ["m1det", "q", "dL", "chieff"],
        "advisory_columns": [],
        "spin_basis_kind": "projection" if parameter_space != "component" else "bijection",
        "spin_density_exact": parameter_space == "component",
        "sky_prior_in_density": True,
        "source_class": source_class,
    }
    payload = json.dumps(pairing, sort_keys=True, separators=(",", ":"))
    with h5py.File(path, "a") as f:
        f.attrs["parameter_space"] = parameter_space
        f.attrs["fit_columns"] = np.array(pairing["fit_columns"], dtype=h5py.string_dtype())
        f.attrs["advisory_columns"] = np.array([], dtype=h5py.string_dtype())
        f.attrs["contract"] = json.dumps(pairing)
        f.attrs["contract_hash"] = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def test_pe_v21_chieff_matches_v2_arrays(tmp_path):
    v2 = tmp_path / "pe_v2c.h5"
    v21 = tmp_path / "pe_v21.h5"
    _write_pe(v2, format_version="gwcat-pe-2.0", spin_basis="chieff")
    _write_pe(v21, format_version="gwcat-pe-2.1", spin_basis="chieff")
    _add_21_contract(v21)
    out2 = load_gw_samples(v2)
    out21 = load_gw_samples(v21)
    for a, b in zip(out2[:7], out21[:7]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    from darksirens.gw.utils import load_gw_store
    store = load_gw_store(v21)
    assert store.fit_columns == ("m1det", "q", "dL", "chieff")
    assert store.attrs["contract_hash"]


def test_pe_v21_component_rejected(tmp_path):
    path = tmp_path / "pe_v21_comp.h5"
    _write_pe(path, format_version="gwcat-pe-2.1", spin_basis="component",
              extra_spin_datasets=True)
    with pytest.raises(RuntimeError, match="spin_basis='component'"):
        load_gw_samples(path)


def test_selection_v21_chieff_accepted_component_rejected(tmp_path):
    ok = tmp_path / "sel_v21.h5"
    _write_selection(ok, format_version="gwcat-selection-2.1", spin_basis="chieff")
    _add_21_contract(ok)
    out = load_selection_samples(ok)
    np.testing.assert_array_equal(np.asarray(out[6]), _SEL_DATA["pdraw"])

    bad = tmp_path / "sel_v21_comp.h5"
    _write_selection(bad, format_version="gwcat-selection-2.1",
                     spin_basis="component", extra_spin_datasets=True)
    with pytest.raises(RuntimeError, match="spin_basis='component'"):
        load_selection_samples(bad)


def test_file_contract_accepts_v21_chieff_selection(tmp_path):
    path = tmp_path / "sel_v21_fc.h5"
    _write_selection(path, format_version="gwcat-selection-2.1", spin_basis="chieff")
    _add_21_contract(path)
    report = file_contract.validate_selection_inputs(path)
    assert report["ok"], report["errors"]


def _pair_opts(gw_path, sel_path):
    from types import SimpleNamespace

    return SimpleNamespace(gw_path=str(gw_path), gwselection_path=str(sel_path),
                           pdet_flow_path=None)


def test_mismatched_contract_hash_raises(tmp_path):
    from darksirens.inference import loaders

    pe = tmp_path / "pe_pair.h5"
    sel = tmp_path / "sel_pair.h5"
    _write_pe(pe, format_version="gwcat-pe-2.1", spin_basis="chieff")
    _write_selection(sel, format_version="gwcat-selection-2.1", spin_basis="chieff")
    _add_21_contract(pe, source_class="BBH")
    _add_21_contract(sel, source_class=None)
    with pytest.raises(RuntimeError, match="contract_hash") as err:
        loaders.load_gw_and_selection_inputs(_pair_opts(pe, sel))
    assert "source_class" in str(err.value)


def test_matching_contract_hash_accepted(tmp_path):
    from darksirens.inference import loaders

    pe = tmp_path / "pe_pair_ok.h5"
    sel = tmp_path / "sel_pair_ok.h5"
    _write_pe(pe, format_version="gwcat-pe-2.1", spin_basis="chieff")
    _write_selection(sel, format_version="gwcat-selection-2.1", spin_basis="chieff")
    _add_21_contract(pe)
    _add_21_contract(sel)
    out = loaders.load_gw_and_selection_inputs(_pair_opts(pe, sel))
    assert out["nEvents"] == NOBS
    assert out["Ndraw"] == 1000
    assert out["gw_attrs"]["contract_hash"] == out["selection_attrs"]["contract_hash"]


def test_legacy_pair_without_contract_accepted(tmp_path):
    """1.0/2.0 files predate the contract; pairing them stays legal."""
    from darksirens.inference import loaders

    pe = tmp_path / "pe_legacy.h5"
    sel = tmp_path / "sel_legacy.h5"
    _write_pe(pe, format_version="gwcat-1.0")
    _write_selection(sel, format_version="gwcat-selection-1.0")
    out = loaders.load_gw_and_selection_inputs(_pair_opts(pe, sel))
    assert out["selection_attrs"] is not None
    assert out["gw_attrs"].get("contract_hash") is None


def test_emulator_rejects_non_chieff_pe():
    from darksirens.inference.loaders import _require_chieff_pe_for_emulator

    attrs = {"spin_basis": "component", "parameter_space": "component"}
    with pytest.raises(RuntimeError, match="pdet_flow_path"):
        _require_chieff_pe_for_emulator(attrs, "x.h5")
    # chieff-basis PE passes.
    _require_chieff_pe_for_emulator({"spin_basis": "chieff"}, "x.h5")


# ----------------------------------------------------------------------------
# Spin-swap validity (DS-03): a chi_eff product whose campaigns did not draw
# spins uniform-in-magnitude/isotropic carries a wrong pdraw.
# ----------------------------------------------------------------------------
def test_non_uniform_campaign_rejected_under_chieff(tmp_path):
    path = tmp_path / "sel_nonuniform.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["injected_spin_uniform_isotropic"] = np.array([True, False])
    with pytest.raises(RuntimeError, match="injected_spin_uniform_isotropic"):
        load_selection_samples(path)


def test_all_uniform_campaigns_accepted(tmp_path):
    path = tmp_path / "sel_uniform.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["injected_spin_uniform_isotropic"] = np.array([True, True])
    load_selection_samples(path)


def test_recorded_swap_violations_rejected(tmp_path):
    path = tmp_path / "sel_violations.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["spin_basis_assumption_violations"] = '["o4ab: isotropy_dev=0.642"]'
    with pytest.raises(RuntimeError, match="spin_basis_assumption_violations"):
        load_selection_samples(path)


def test_flag_downgrades_to_warning(tmp_path, capsys):
    path = tmp_path / "sel_nonuniform_ok.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["injected_spin_uniform_isotropic"] = np.array([True, False])
    out = load_selection_samples(path, allow_invalid_spin_swap=True)
    assert out[7] == 1000
    assert "allow_invalid_spin_swap" in capsys.readouterr().out


def test_legacy_file_without_the_attr_loads(tmp_path):
    """The shipped gwcat-selection-1.0 product predates the attr; the gate
    cannot read what is not there (retiring it needs the GW-24 re-export)."""
    path = tmp_path / "sel_legacy_noattr.h5"
    _write_selection(path, format_version="gwcat-selection-1.0")
    load_selection_samples(path)


def test_file_contract_rejects_non_uniform_campaign(tmp_path):
    path = tmp_path / "sel_nonuniform_fc.h5"
    _write_selection(path, format_version="gwcat-selection-2.0", spin_basis="chieff")
    with h5py.File(path, "a") as f:
        f.attrs["injected_spin_uniform_isotropic"] = np.array([False])
    report = file_contract.validate_selection_inputs(path)
    assert not report["ok"]
    assert any("injected_spin_uniform_isotropic" in e for e in report["errors"])


# ----------------------------------------------------------------------------
# -50 -> -inf convention (DS-06, mirroring gwcat GW-03): zero density is
# zero, not 2e-22; refusals, not floors.
# ----------------------------------------------------------------------------
def test_out_of_support_pe_sample_gets_zero_weight(tmp_path, monkeypatch):
    """A PE sample outside the chi_eff prior support gets p_pe = 0 (masked by
    the likelihood, still counted in n) -- not the old floor exp(-50)."""
    from darksirens.gw import utils as gw_utils

    def fake_logprob(chieff, m1src, m2src, amax=0.99):
        logp = np.full(np.shape(chieff), np.log(2.0))
        logp[0] = -np.inf
        return logp

    monkeypatch.setattr(gw_utils, "chi_eff_prior_logprob", fake_logprob)
    path = tmp_path / "pe_out_of_support.h5"
    _write_pe(path, format_version="gwcat-1.0", chi_eff_in_p_pe=False)
    p_pe = np.asarray(load_gw_samples(path)[6])
    assert p_pe[0] == 0.0
    assert np.all(p_pe[1:] > 0.0)


def test_out_of_support_injection_refused_on_swap(tmp_path, monkeypatch):
    """A DETECTED injection at zero draw density cannot be floored (weight
    ~1e21 above median) or dropped (Ndraw fixed -> mu biased low): refuse."""
    from darksirens.gw import utils as gw_utils

    def fake_logprob(chieff, m1src, m2src, amax=0.99):
        logp = np.full(np.shape(chieff), np.log(2.0))
        logp[-1] = -np.inf
        return logp

    monkeypatch.setattr(gw_utils, "chi_eff_prior_logprob", fake_logprob)
    path = tmp_path / "sel_out_of_support.h5"
    _write_selection(path, format_version="gwcat-selection-1.0",
                     chi_eff_swap_applied=False)
    with pytest.raises(RuntimeError, match="outside the chi_eff prior support"):
        load_selection_samples(path)


def test_gwcat_new_convention_api_present():
    """The import-time guard's premise: the linked gwcat exposes the GW-03
    API (support predicate + -inf logprob), so the two packages agree on
    what an out-of-support sample means."""
    gwcat_spin = pytest.importorskip("gwcat.spin")
    assert hasattr(gwcat_spin.ChiEffPrior, "support")
    logp = np.asarray(gwcat_spin.chi_eff_prior_logprob(
        np.array([0.0]), np.array([30.0]), np.array([25.0]), amax=0.99))
    assert np.isfinite(logp).all()


# ----------------------------------------------------------------------------
# PE cosmology consumption (DS-12): pe_cosmology_H0/Om0 were required attrs
# read by no consumer; the emulator cosmology was never checked against them.
# ----------------------------------------------------------------------------
def test_emulator_cosmology_must_match_pe():
    from types import SimpleNamespace

    from darksirens.inference.loaders import _require_matching_pdet_cosmology

    attrs = {"pe_cosmology_H0": 67.7, "pe_cosmology_Om0": 0.31}
    opts_ok = SimpleNamespace(pdet_cosmology="67.7,0.31")
    _require_matching_pdet_cosmology(attrs, "pe.h5", opts_ok)
    # Sub-tolerance differences are the same cosmology.
    _require_matching_pdet_cosmology(
        attrs, "pe.h5", SimpleNamespace(pdet_cosmology="67.9,0.3065"))
    with pytest.raises(RuntimeError, match="pdet_cosmology"):
        _require_matching_pdet_cosmology(
            attrs, "pe.h5", SimpleNamespace(pdet_cosmology="70.0,0.31"))
    with pytest.raises(RuntimeError, match="pdet_cosmology"):
        _require_matching_pdet_cosmology(
            attrs, "pe.h5", SimpleNamespace(pdet_cosmology="67.7,0.25"))


def test_pair_cosmology_mismatch_warns_not_raises(tmp_path, capsys):
    from darksirens.inference import loaders

    pe = tmp_path / "pe_cosmo.h5"
    sel = tmp_path / "sel_cosmo.h5"
    _write_pe(pe, format_version="gwcat-1.0")
    _write_selection(sel, format_version="gwcat-selection-1.0")
    with h5py.File(sel, "a") as f:
        f.attrs["cosmology_H0"] = 73.0
    out = loaders.load_gw_and_selection_inputs(_pair_opts(pe, sel))
    assert out["Ndraw"] == 1000
    assert "declares cosmology" in capsys.readouterr().out


def test_per_event_cosmology_flag_surfaced(tmp_path, capsys):
    from darksirens.inference import loaders

    pe = tmp_path / "pe_varies.h5"
    sel = tmp_path / "sel_varies.h5"
    _write_pe(pe, format_version="gwcat-1.0")
    _write_selection(sel, format_version="gwcat-selection-1.0")
    with h5py.File(pe, "a") as f:
        f.attrs["cosmology_per_event_varies"] = True
        f.attrs["cosmology_mode"] = "per-event"
    loaders.load_gw_and_selection_inputs(_pair_opts(pe, sel))
    assert "cosmology_per_event_varies=True" in capsys.readouterr().out


# ----------------------------------------------------------------------------
# Provenance (DS-10): store attrs into the run record, event identity
# logged, writer-commit drift surfaced.
# ----------------------------------------------------------------------------
def test_provenance_block_is_json_clean(tmp_path):
    import json

    from darksirens.inference.data import _attr_event_names, _provenance_block

    attrs = {
        "format_version": "gwcat-selection-2.1",
        "far_threshold": np.float64(1.0),
        "campaign_ndraws": np.array([73957576, 870454872]),
        "n_campaigns": np.int64(2),
        "event_names": np.array([b"GW150914", b"GW151226"]),
        "irrelevant_attr": object(),
    }
    block = _provenance_block(attrs)
    json.dumps(block)
    assert block["campaign_ndraws"] == [73957576, 870454872]
    assert block["event_names"] == ["GW150914", "GW151226"]
    assert "irrelevant_attr" not in block
    assert _provenance_block(None) is None
    assert _attr_event_names({"event_names": np.array([b"a", b"b"])}) == ["a", "b"]
    assert _attr_event_names({}) is None


def test_settings_records_store_provenance(tmp_path):
    """opts attributes are serialised into settings.json; the provenance
    blocks attached by load_all_data must survive that round trip."""
    import json
    from types import SimpleNamespace

    from darksirens.io.settings import save_settings_json

    opts = SimpleNamespace(
        universe_model="dark_sirens",
        gw_store_provenance={"format_version": "gwcat-1.0",
                             "event_names": ["GW150914"]},
        selection_store_provenance={"format_version": "gwcat-selection-2.1",
                                    "ndraw": 1000},
        gw_event_names=["GW150914"],
    )
    path = save_settings_json(
        opts, str(tmp_path), labels=["H0"], lower_bound=[20.0],
        upper_bound=[140.0], fixed_parameter_values={}, prior_overrides={},
        meta={},
    )
    with open(path) as f:
        recorded = json.load(f)
    assert recorded["gw_store_provenance"]["event_names"] == ["GW150914"]
    assert recorded["selection_store_provenance"]["ndraw"] == 1000
    assert recorded["gw_event_names"] == ["GW150914"]


def test_writer_commit_mismatch_warns(capsys):
    from darksirens.inference.loaders import _warn_writer_commit

    _warn_writer_commit({}, "x.h5")  # no attr: silent
    _warn_writer_commit({"writer_commit": "deadbeef"}, "x.h5")
    out = capsys.readouterr().out
    # The installed gwcat commit is whatever the environment has; the warning
    # fires iff it is known and differs. Either way this must not raise, and
    # a matching prefix must stay silent.
    from darksirens.io.settings import code_identity

    installed = str(code_identity().get("gwcat_commit") or "")
    if installed and installed != "unknown":
        assert ("was written by gwcat commit" in out) == (
            not installed.split("-")[0].startswith("deadbeef")
        )
        _warn_writer_commit({"writer_commit": installed}, "x.h5")
        assert "was written by gwcat" not in capsys.readouterr().out
