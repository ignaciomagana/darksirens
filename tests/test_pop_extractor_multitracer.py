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
    # The whole reconstructed space, not just the pop indices: the pop block
    # precedes the survey block, so a survey label the reconstruction gets
    # wrong shifts only the LATER labels and the pop-index check alone passes.
    assert list(extractor.sampled_labels) == list(labels)
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


# ---------------------------------------------------------------------------
# lss_field_mode: the LATENT b_miss is a different parameter and IS sampled
# ---------------------------------------------------------------------------
# In table mode ``b_miss`` scales the local overdensity factor and is dropped
# from the sampled space whenever that factor cannot depend on it (--use_lss
# off, or an active Q_LSS table).  In LATENT mode the same symbol is b_GW, the
# bias with which GW hosts trace the latent field; it enters through
# ``logQ = b_GW (row_fac . phi_z) - rho(...)`` at every evaluation and is
# sampled regardless (prior.py:_b_miss_rule inverts the guard there).  The
# extractor did not thread the mode, so it rebuilt a latent chain one column
# short -- b_miss missing and every label after it off by one -- and rejected a
# latent run that FIXED b_miss, quoting a table-mode inertness reason.

LATENT_SPACE = dict(
    n_catalogs=1,
    use_lss=False,          # latent mode is incompatible with --use_lss
    c_mode="aggregate",     # ... and requires an aggregate/selection c_mode
    lss_field_mode="latent",
)

LATENT_SETTINGS = {
    "pop_model": POP_MODEL,
    "universe_model": "dark_sirens",
    "n_catalogs": 1,
    "use_LSS": False,
    "c_mode": "aggregate",
    "lss_field_mode": "latent",
}


def test_latent_field_mode_keeps_b_miss_in_the_reconstructed_space():
    labels, _ = _space(**LATENT_SPACE)
    assert "b_miss" in labels, "latent mode must sample b_miss (= b_GW)"
    _assert_extractor_matches_space(dict(LATENT_SETTINGS), **LATENT_SPACE)


def test_table_mode_still_drops_b_miss_under_the_same_gates():
    """The mode must be the only thing that changes: with --use_lss off and
    table mode, b_miss stays out of the space."""
    space = dict(LATENT_SPACE, lss_field_mode="table")
    settings = dict(LATENT_SETTINGS, lss_field_mode="table")
    labels, _ = _space(**space)
    assert "b_miss" not in labels
    _assert_extractor_matches_space(settings, **space)


def test_latent_field_mode_accepts_a_fixed_b_miss():
    """Fixing b_GW in a latent run is legitimate; before the pass-through the
    extractor raised, quoting a table-mode inertness reason."""
    fixed = {"b_miss": 1.2}
    settings = dict(LATENT_SETTINGS, fixed_parameter_values=dict(fixed))
    _assert_extractor_matches_space(
        settings, **LATENT_SPACE, fixed_parameter_values=dict(fixed)
    )


def test_missing_lss_field_mode_key_defaults_to_table():
    """Archived settings.json predating the flag must keep the table space."""
    settings = dict(LATENT_SETTINGS)
    del settings["lss_field_mode"]
    _assert_extractor_matches_space(
        settings, **dict(LATENT_SPACE, lss_field_mode="table")
    )
