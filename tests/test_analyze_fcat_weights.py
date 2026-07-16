"""darksirens_analyze multitracer host-fraction output (cli/analyze.py).

The analyze CLI derives the per-catalog mixture weights w_1..w_K from the
sampled sticks fcat_2..fcat_K with the decoder's own stick-breaking
construction (catalog_sticks_to_weights) and plots/saves them.  These tests
pin the derivation used by that block and the plotting helper."""
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


import numpy as np
import matplotlib.pyplot as plt

from darksirens.cli.analyze import plot_catalog_weight_posteriors
from darksirens.inference.pop_extractor import catalog_sticks_to_weights


def test_k2_weight_columns_reproduce_stick_semantics():
    """K=2: w_2 IS the stick fcat_2 and w_1 = 1 - fcat_2 (remainder-first
    ordering), for a whole chain at once."""
    rng = np.random.default_rng(7)
    sticks = rng.uniform(0.0, 1.0, size=(256, 1))
    weights = np.asarray(catalog_sticks_to_weights(sticks))
    assert weights.shape == (256, 2)
    np.testing.assert_allclose(weights[:, 1], sticks[:, 0], rtol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)


def test_k3_weight_rows_are_simplex():
    rng = np.random.default_rng(11)
    sticks = rng.uniform(0.0, 1.0, size=(128, 2))
    weights = np.asarray(catalog_sticks_to_weights(sticks))
    assert weights.shape == (128, 3)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)


def test_plot_catalog_weight_posteriors_labels_each_catalog():
    rng = np.random.default_rng(3)
    sticks = rng.uniform(0.0, 1.0, size=(64, 2))
    weights = np.asarray(catalog_sticks_to_weights(sticks))
    fig = plot_catalog_weight_posteriors(weights, is_field=True)
    try:
        (ax,) = fig.axes
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_texts == ["$w_{1}$", "$w_{2}$", "$w_{3}$"]
        assert "host fraction" in ax.get_xlabel()
    finally:
        plt.close(fig)


def test_plot_catalog_weight_posteriors_conditional_caveat_in_xlabel():
    """A conditional-normalizer chain must NOT be labeled a host fraction."""
    rng = np.random.default_rng(5)
    weights = np.asarray(catalog_sticks_to_weights(rng.uniform(size=(32, 1))))
    fig = plot_catalog_weight_posteriors(weights, is_field=False)
    try:
        (ax,) = fig.axes
        assert "host fraction" not in ax.get_xlabel()
        assert "z-shape" in ax.get_xlabel()
    finally:
        plt.close(fig)
