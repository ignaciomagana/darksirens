import sys
import types


if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub


from darksirens.tool.darksirens_analyze import _should_plot_bayes_factor_matrix


def test_bayes_factor_matrix_is_skipped_for_single_run():
    assert not _should_plot_bayes_factor_matrix(["only model"], [-1.0])


def test_bayes_factor_matrix_requires_two_evidences():
    assert not _should_plot_bayes_factor_matrix(["model a", "model b"], [-1.0, None])


def test_bayes_factor_matrix_is_plotted_for_model_pair():
    assert _should_plot_bayes_factor_matrix(["model a", "model b"], [-1.0, -2.0])

import h5py
import numpy as np

from darksirens.tool.darksirens_analyze import load_run


def test_load_run_reads_current_results_hdf5_root_samples(tmp_path):
    with h5py.File(tmp_path / "results.hdf5", "w") as f:
        f.create_dataset("samples", data=np.ones((3, 2)))
        f.create_dataset("labels", data=np.array(["a", "b"], dtype=h5py.string_dtype("utf-8")))
        f.attrs["pop_model"] = "powerlaw"
        f.attrs["logZ"] = 2.0
        f.attrs["logZerr"] = 0.5

    settings, samples, logz, logzerr = load_run(str(tmp_path))

    assert samples.shape == (3, 2)
    assert settings["labels"] == ["a", "b"]
    assert settings["pop_model"] == "powerlaw"
    assert logz == 2.0
    assert logzerr == 0.5


def test_load_run_reads_grouped_results_hdf5_samples_and_evidence_aliases(tmp_path):
    with h5py.File(tmp_path / "results.hdf5", "w") as f:
        posterior = f.create_group("posterior")
        posterior.create_dataset("samples", data=np.zeros((4, 1)))
        f.attrs["log_evidence"] = 1.25
        f.attrs["log_evidence_err"] = 0.25

    _settings, samples, logz, logzerr = load_run(str(tmp_path))

    assert samples.shape == (4, 1)
    assert logz == 1.25
    assert logzerr == 0.25
