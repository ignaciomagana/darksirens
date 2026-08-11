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


from darksirens.cli.analyze import _should_plot_bayes_factor_matrix


def test_bayes_factor_matrix_is_skipped_for_single_run():
    assert not _should_plot_bayes_factor_matrix(["only model"], [-1.0])


def test_bayes_factor_matrix_requires_two_evidences():
    assert not _should_plot_bayes_factor_matrix(["model a", "model b"], [-1.0, None])


def test_bayes_factor_matrix_is_plotted_for_model_pair():
    assert _should_plot_bayes_factor_matrix(["model a", "model b"], [-1.0, -2.0])

import json

import h5py
import numpy as np
import pytest

from darksirens.cli.analyze import _build_parser, load_run


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


def test_load_run_reads_numeric_samples_npy_recovery_chain(tmp_path):
    """The crash-recovery artifact both CLIs write before results.hdf5: a bare
    numeric (nsamples, ndim) matrix, with metadata from settings.json."""
    np.save(tmp_path / "samples.npy", np.array([[1.0, 2.0], [3.0, 4.0]]))
    (tmp_path / "settings.json").write_text(
        json.dumps({"pop_model": "powerlaw", "labels": ["H0", "alpha"]})
    )

    settings, samples, logz, logzerr = load_run(str(tmp_path))

    assert samples.shape == (2, 2)
    assert settings["labels"] == ["H0", "alpha"]
    # No evidence is stored next to the chain — it must read as absent.
    assert logz is None and logzerr is None


def test_load_run_requires_opt_in_for_legacy_pickled_samples_npy(tmp_path):
    legacy = {"samples": np.zeros((5, 3)), "logZ": -7.0, "logZerr": 0.1}
    np.save(tmp_path / "samples.npy", np.array(legacy, dtype=object))

    with pytest.raises(ValueError, match="legacy pickled results dict"):
        load_run(str(tmp_path))

    with pytest.warns(RuntimeWarning, match="allow_legacy_pickle"):
        _settings, samples, logz, logzerr = load_run(
            str(tmp_path), allow_legacy_pickle=True
        )
    assert samples.shape == (5, 3)
    assert logz == -7.0 and logzerr == 0.1


def test_analyze_parser_defaults_legacy_pickle_off():
    args = _build_parser().parse_args([])
    assert args.allow_legacy_pickle is False
    assert _build_parser().parse_args(["--allow_legacy_pickle"]).allow_legacy_pickle


def test_evidence_bars_carry_the_error_on_the_DIFFERENCE():
    """The bars are log BFs against the best model, so the error must be the
    combined hypot(err_i, err_best) — the same combination the pairwise matrix
    prints — and the reference model's own bar cannot carry an error bar."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    from darksirens.cli.analyze import plot_model_evidences

    log10Zs = [-10.0, -8.0, -12.0]
    errs = [0.3, 0.4, 0.5]
    fig = plot_model_evidences(["a", "best", "c"], log10Zs, errs)
    ax = fig.axes[0]
    (segments,) = [c.get_segments() for c in ax.collections]
    heights = [np.abs(seg[1][1] - seg[0][1]) / 2.0 for seg in segments]
    want = [np.hypot(0.3, 0.4), 0.0, np.hypot(0.5, 0.4)]
    np.testing.assert_allclose(sorted(heights), sorted(want), rtol=1e-9, atol=1e-12)
    matplotlib.pyplot.close(fig)


def test_event_overlay_is_drawn_in_the_source_frame(tmp_path):
    """``p_m1`` is a source-frame density, so the rug must be m1det/(1+z), not
    m1det: at these distances the raw detector-frame median sits tens of percent
    to the right of where its event belongs."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import h5py

    from darksirens.cli.analyze import COSMO_FID, overlay_observed_events
    from darksirens.utils.cosmology import z_of_dL

    nEvents, nsamp = 2, 8
    m1det = np.repeat(np.array([40.0, 60.0]), nsamp)
    dL = np.repeat(np.array([1500.0, 4000.0]), nsamp)
    path = tmp_path / "gw.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = nEvents
        f.attrs["nsamp"] = nsamp
        f.attrs["pe_cosmology_H0"] = COSMO_FID["H0"]
        f.attrs["pe_cosmology_Om0"] = COSMO_FID["Om0"]
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.create_dataset("m1det", data=m1det)
        f.create_dataset("m2det", data=0.8 * m1det)
        f.create_dataset("m1src", data=m1det)
        f.create_dataset("m2src", data=0.8 * m1det)
        f.create_dataset("dL", data=dL)
        f.create_dataset("chieff", data=np.zeros_like(m1det))
        f.create_dataset("ra", data=np.zeros_like(m1det))
        f.create_dataset("dec", data=np.zeros_like(m1det))
        f.create_dataset("p_pe", data=np.ones_like(m1det))

    fig, ax = matplotlib.pyplot.subplots()
    overlay_observed_events(ax, {"gw_path": str(path)})
    drawn = sorted(line.get_xdata()[0] for line in ax.lines)
    z = np.asarray(z_of_dL(np.array([1500.0, 4000.0]), COSMO_FID["H0"],
                           COSMO_FID["Om0"], COSMO_FID["w0"], COSMO_FID["wa"]))
    want = sorted(np.array([40.0, 60.0]) / (1.0 + z))
    np.testing.assert_allclose(drawn, want, rtol=1e-6)
    assert drawn[-1] < 60.0 - 1e-6          # ... and NOT the detector-frame value
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("source-frame" in t for t in labels)
    matplotlib.pyplot.close(fig)
