import json
from types import SimpleNamespace

import h5py
import numpy as np

from darksirens.inference.sampling import normalize_tinyns_diagnostics
from darksirens.io.results import save_results_hdf5, save_tinyns_diagnostics_json


class FakeTinyNSResult:
    logz = -22.31
    logzerr = 0.07

    def diagnostics(self):
        return {
            "ncall": 200,
            "niter": 10,
            "seconds": 5.0,
            "replacement_mean_batches": 1.5,
            "replacement_failures": 2,
            "replacement_rescue_used": True,
        }

    def summary(self):
        return "converged"


class MinimalTinyNSResult:
    logz = -1.0
    logzerr = 0.2


def _opts():
    return SimpleNamespace(
        pop_model="powerlaw", universe_model="flatLCDM", sky_model="isotropic",
        mark_model="none", mark_names=(), complete_empty_pixel_policy="error",
        sampler="tinyns", fix_cosmology=False, fix_de=False, fix_population=False,
        fix_survey=False, gw_path="gw.h5", gwselection_path="sel.h5", survey_path="",
        nlive=10, dlogz=0.1, tinyns_sample=None, tinyns_kernel=None,
        tinyns_walks=None, tinyns_replacement_chains=None,
        tinyns_replacement_chain_schedule=None, tinyns_max_attempts=None,
        tinyns_preset="recommended", tinyns_bound=None, tinyns_step_scale=None,
        tinyns_resolved_config={"sample": "rwalk", "kernel": "jax"},
        nuts_warmup=0, nuts_samples=0, nuts_chains=0, nuts_target_accept=0.0,
        nuts_max_tree_depth=0, nuts_init_tries=0, nuts_init_seed_offset=0,
        seed=123,
    )


def _meta():
    return {
        "n_events": 0, "n_samp_per_event": 0, "n_draw": 0,
        "total_runtime": "0:00:00", "sampling_runtime": "0:00:00",
        "timestamp": "2026-07-03T00-00-00",
    }


def test_normalize_tinyns_diagnostics_derives_runtime_fields():
    diag = normalize_tinyns_diagnostics(FakeTinyNSResult())
    assert diag["ncall"] == 200
    assert diag["niter"] == 10
    assert diag["seconds"] == 5.0
    assert diag["ncall_per_sec"] == 40.0
    assert diag["niter_per_sec"] == 2.0
    assert diag["calls_per_iter"] == 20.0
    assert diag["replacement_mean_batches"] == 1.5
    assert diag["replacement_failures"] == 2


def test_normalize_tinyns_diagnostics_missing_fields_are_json_safe():
    diag = normalize_tinyns_diagnostics(MinimalTinyNSResult())
    assert diag["logz"] == -1.0
    assert diag["logzerr"] == 0.2
    json.dumps(diag)


def test_main_inference_save_path_writes_tinyns_diagnostics_sidecar(tmp_path):
    results = {
        "samples": np.zeros((2, 1)),
        "logZ": -1.0,
        "logZerr": 0.1,
        "tinyns_runtime_diagnostics": {
            "niter": 10, "ncall": 200, "seconds": 5.0,
            "replacement_failures": 2, "replacement_rescue_used": True,
        },
        "tinyns_diagnostics": {"niter": 10},
        "tinyns_summary": "converged",
    }
    path = save_results_hdf5(
        results, str(tmp_path), ["H0"], [10.0], [100.0], {}, {}, _opts(), _meta()
    )
    sidecar = tmp_path / "tinyns_diagnostics.json"
    assert sidecar.exists()
    with h5py.File(path, "r") as f:
        assert "tinyns_runtime_diagnostics" in f.attrs
        assert "tinyns_resolved_config" in f.attrs
        assert "tinyns_diagnostics" in f.attrs
        assert "tinyns_summary" in f.attrs
        assert f.attrs["tinyns_niter"] == 10
        assert f.attrs["tinyns_ncall"] == 200
        assert f.attrs["tinyns_replacement_failures"] == 2
        assert bool(f.attrs["tinyns_replacement_rescue_used"]) is True
    payload = json.loads(sidecar.read_text())
    assert payload["tinyns_resolved_config"]["sample"] == "rwalk"


def test_save_tinyns_diagnostics_json_sidecar(tmp_path):
    results = {
        "tinyns_runtime_diagnostics": {"niter": 10},
        "tinyns_diagnostics": {"ncall": 200},
        "tinyns_summary": "converged",
    }
    path = save_tinyns_diagnostics_json(results, str(tmp_path), _opts())
    payload = json.loads((tmp_path / "tinyns_diagnostics.json").read_text())
    assert path == str(tmp_path / "tinyns_diagnostics.json")
    assert payload["tinyns_runtime_diagnostics"]["niter"] == 10
    assert payload["tinyns_resolved_config"]["sample"] == "rwalk"
