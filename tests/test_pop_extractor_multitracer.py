"""make_pop_extractor must rebuild the SAME parameter space a K>=2 run
sampled (inference/pop_extractor.py).

Regression: the extractor omitted ``n_catalogs`` (and the Q-active b_miss
drop), so ``build_parameter_space`` rejected the per-catalog labels a K>=2
run legitimately carries in ``fixed_parameter_values`` (``fcat_*``,
``*_c{k}``) with a KeyError -- a concrete analyze-time crash for any such
chain."""
import numpy as np
import jax.numpy as jnp

from darksirens.inference.pop_extractor import make_pop_extractor
from darksirens.inference.prior import build_parameter_space

POP_MODEL = "powerlaw+peak"
K2_FIXED = {"fcat_2": 0.3, "log10n0_c2": -2.0}


def _space(**kwargs):
    res = build_parameter_space(
        pop_model=POP_MODEL,
        fix_population=False,
        fix_cosmology=False,
        fix_survey=False,
        universe_model="dark_sirens",
        **kwargs,
    )
    labels, pop_labels = res[0], res[4]
    return labels, pop_labels


def _assert_extractor_matches_space(settings, **space_kwargs):
    extractor = make_pop_extractor(settings)
    labels, pop_labels = _space(**space_kwargs)
    theta = jnp.arange(len(labels), dtype=jnp.float64)
    expected = np.array([float(theta[labels.index(l)]) for l in pop_labels])
    np.testing.assert_allclose(np.asarray(extractor(theta)), expected)


def test_make_pop_extractor_accepts_multitracer_fixed_labels():
    """K=2 settings with fcat_2/log10n0_c2 fixed: raised KeyError before the
    n_catalogs pass-through; must now build and extract the pop block."""
    settings = {
        "pop_model": POP_MODEL,
        "universe_model": "dark_sirens",
        "n_catalogs": 2,
        "fixed_parameter_values": dict(K2_FIXED),
    }
    _assert_extractor_matches_space(
        settings, n_catalogs=2, fixed_parameter_values=dict(K2_FIXED)
    )


def test_make_pop_extractor_threads_lss_completion_active():
    """A Q-active run drops b_miss from the sampled survey block; the
    extractor must rebuild with the same flag so the label set matches."""
    settings = {
        "pop_model": POP_MODEL,
        "universe_model": "dark_sirens",
        "n_catalogs": 1,
        "lss_completion_active": True,
    }
    extractor = make_pop_extractor(settings)
    labels, pop_labels = _space(lss_completion_active=True)
    assert "b_miss" not in labels
    theta = jnp.arange(len(labels), dtype=jnp.float64)
    expected = np.array([float(theta[labels.index(l)]) for l in pop_labels])
    np.testing.assert_allclose(np.asarray(extractor(theta)), expected)


def test_make_pop_extractor_legacy_settings_unchanged():
    """Old settings.json without the new keys (K=1, no Q flag) keep working
    and index the identical space as before."""
    settings = {"pop_model": POP_MODEL, "universe_model": "dark_sirens"}
    _assert_extractor_matches_space(settings)
