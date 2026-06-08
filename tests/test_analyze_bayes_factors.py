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
