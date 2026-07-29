"""results.hdf5 saving robustness (library review, findings BUG-2/BUG-5).

--pdet_flow_path runs (gwselection_path left None) used to crash *after*
inference completed: writing a raw None into an hdf5 attr raises
`TypeError: Object dtype dtype('O') has no native HDF5 equivalent`. That also
means a failed attr write could leave a truncated results.hdf5 behind; the
save now goes through a temp file + os.replace so a failure can't clobber a
previous good file or leave a half-written one at the final path.
"""
from types import SimpleNamespace

import h5py
import numpy as np

from darksirens.io.results import save_results_hdf5


def _opts(**overrides):
    base = dict(
        pop_model="powerlaw", universe_model="flatLCDM", sky_model="isotropic",
        mark_model="none", mark_names=(), complete_empty_pixel_policy="error",
        sampler="tinyns", fix_cosmology=False, fix_de=False, fix_population=False,
        fix_survey=False, gw_path="gw.h5", gwselection_path="sel.h5",
        pdet_flow_path=None, survey_path="",
        nlive=10, dlogz=0.1, tinyns_sample=None, tinyns_kernel=None,
        tinyns_walks=None, tinyns_replacement_chains=None,
        tinyns_replacement_chain_schedule=None, tinyns_max_attempts=None,
        tinyns_preset="recommended", tinyns_bound=None, tinyns_step_scale=None,
        tinyns_resolved_config=None,
        nuts_warmup=0, nuts_samples=0, nuts_chains=0, nuts_target_accept=0.0,
        nuts_max_tree_depth=0, nuts_init_tries=0, nuts_init_seed_offset=0,
        seed=123,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _meta():
    return {
        "n_events": 0, "n_samp_per_event": 0, "n_draw": 0,
        "total_runtime": "0:00:00", "sampling_runtime": "0:00:00",
        "timestamp": "2026-07-21T00-00-00",
    }


def _results():
    return {"samples": np.zeros((2, 1)), "logZ": -1.0, "logZerr": 0.1}


def test_save_results_hdf5_pdet_flow_path_round_trips(tmp_path):
    """The P_det-emulator path (gwselection_path unset, pdet_flow_path set)
    must save cleanly, with gwselection_path recorded as "" (not None) and
    pdet_flow_path recorded (previously never written at all)."""
    opts = _opts(gwselection_path=None, pdet_flow_path="/some/flow.eqx")
    path = save_results_hdf5(
        _results(), str(tmp_path), ["H0"], [10.0], [100.0], {}, {}, opts, _meta()
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["gwselection_path"] == ""
        assert f.attrs["pdet_flow_path"] == "/some/flow.eqx"


def test_save_results_hdf5_gwselection_path_round_trips(tmp_path):
    """The injection-file path (gwselection_path set) is unaffected by the fix."""
    opts = _opts(gwselection_path="injections.h5", pdet_flow_path=None)
    path = save_results_hdf5(
        _results(), str(tmp_path), ["H0"], [10.0], [100.0], {}, {}, opts, _meta()
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["gwselection_path"] == "injections.h5"
        assert f.attrs["pdet_flow_path"] == ""


def test_save_results_hdf5_leaves_no_tmp_file_behind(tmp_path):
    opts = _opts()
    save_results_hdf5(
        _results(), str(tmp_path), ["H0"], [10.0], [100.0], {}, {}, opts, _meta()
    )
    assert (tmp_path / "results.hdf5").exists()
    assert not (tmp_path / "results.hdf5.tmp").exists()


def test_nlive_actual_wins_over_the_request(tmp_path):
    """P2-19: a resumed dynesty run keeps the checkpoint's live-point count
    over --nlive, so results.hdf5 must record the value the sampler ACTUALLY
    ran with, with the request preserved beside it."""
    results = _results()
    results["nlive_actual"] = 40
    save_results_hdf5(
        results, str(tmp_path), ["x"], [0.0], [1.0], {}, {},
        _opts(nlive=64, sampler="dynesty"), _meta(),
    )
    with h5py.File(tmp_path / "results.hdf5") as f:
        assert int(f.attrs["nlive"]) == 40
        assert int(f.attrs["nlive_requested"]) == 64


def test_nlive_defaults_to_the_request_without_sampler_state(tmp_path):
    save_results_hdf5(
        _results(), str(tmp_path), ["x"], [0.0], [1.0], {}, {},
        _opts(nlive=64), _meta(),
    )
    with h5py.File(tmp_path / "results.hdf5") as f:
        assert int(f.attrs["nlive"]) == 64
        assert int(f.attrs["nlive_requested"]) == 64
