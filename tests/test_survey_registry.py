"""Survey-parameter registry: bit-compat, decoder agreement, and loud guards.

``darksirens.inference.prior`` declares the sampled survey block ONCE, as an
ordered registry of ``(label, lower, upper, activity rule)`` records
(``_SURVEY_BLOCK``), plus the labels that are recognised but never sampled
(``_PINNED_SURVEY_PARAMS``: ``z50``/``w``, generative-truth fields of the mock
generator, and ``alpha_miss``, pinned by its exact degeneracy with ``b_miss``).
``build_parameter_space`` derives the catalog-1 block AND every per-catalog
``_c{k}`` block from that registry alone, and the guards below quote the very
rule that did the gating -- so an error message cannot contradict the block.

This file is the acceptance net for that refactor:

1. BIT-COMPAT.  ``EXPECTED`` is the sampled label list MASTER returned for every
   cell of

       universe_model x use_lss x n_catalogs x Q_LSS activity x fix_survey

   at ``fix_population=fix_cosmology=True`` (so a cell's labels are the survey
   block plus the mixture sticks, nothing else).  It was DERIVED, not written by
   hand: master's ``build_parameter_space`` was called over the matrix and its
   (labels, lower, upper) serialised to JSON, the same loop was re-run on this
   branch, and the two were diffed cell by cell.  All 400 named-universe-model
   cells of the wider sweep -- which also varies ``fix_population`` /
   ``fix_cosmology`` and the sky/mark models -- were identical.  Regenerate the
   same way if the block ever changes on purpose.

   The one deliberate difference is deliberately NOT a cell here: a
   ``universe_model`` the registry does not recognise (``None``, i.e. a direct
   library caller naming no model) still samples the whole block, and that block
   no longer contains the three never-sampled labels.
   ``test_unregistered_universe_model_samples_whole_block`` pins the new value.

2. DECODER AGREEMENT.  ``build_parameter_decoder`` re-derives the space from
   ``opts``; it must produce the same sampled labels in every cell.  Each case
   arms the real ``expected_sampled_labels`` fail-fast net.

3. LOUD GUARDS.  Per-label config naming a survey parameter that is inert for
   the resolved configuration must fail with the TRUE reason -- including the
   two live hazards this refactor fixed: a ``z50``/``w``/``alpha_miss`` prior
   override used to be accepted in silence, and a ``b_miss`` override under
   ``--use_lss false`` was rejected while blaming a Q_LSS table that was not
   there.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        pass

    class _KernelsStub:
        class Matern52:
            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub

from darksirens.core.constants import SURVEY_PARAMS_FID_BY_NAME
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.inference.prior import (
    _PINNED_SURVEY_PARAMS,
    _SURVEY_BLOCK,
    build_parameter_space,
)

POP = "powerlaw+peak"

#: Q_LSS activity spellings: scalar for K=1, per-catalog tuple for K=2.
Q_ACTIVITY = {
    (1, "none"): False,
    (1, "all"): True,
    (2, "none"): (False, False),
    (2, "all"): (True, True),
    (2, "mixed"): (True, False),
}

#: Master's per-label prior bounds, applied to the labels of every cell below.
MASTER_BOUNDS = {
    "log10n0": (-4.0, -1.0),
    "delta": (-3.0, 3.0),
    "b_miss": (0.0, 3.0),
    "sigma_kde": (0.0, 0.05),
    "fcat_2": (0.0, 1.0),
}

#: (universe_model, use_lss, n_catalogs, q, fix_survey) -> sampled labels, as
#: returned by MASTER.  See the module docstring for the derivation.
EXPECTED = {
    ('bright_sirens', False, 1, 'all', False): [],
    ('bright_sirens', False, 1, 'all', True): [],
    ('bright_sirens', False, 1, 'none', False): [],
    ('bright_sirens', False, 1, 'none', True): [],
    ('bright_sirens', False, 2, 'all', False): ['fcat_2'],
    ('bright_sirens', False, 2, 'all', True): ['fcat_2'],
    ('bright_sirens', False, 2, 'mixed', False): ['fcat_2'],
    ('bright_sirens', False, 2, 'mixed', True): ['fcat_2'],
    ('bright_sirens', False, 2, 'none', False): ['fcat_2'],
    ('bright_sirens', False, 2, 'none', True): ['fcat_2'],
    ('bright_sirens', True, 1, 'all', False): [],
    ('bright_sirens', True, 1, 'all', True): [],
    ('bright_sirens', True, 1, 'none', False): [],
    ('bright_sirens', True, 1, 'none', True): [],
    ('bright_sirens', True, 2, 'all', False): ['fcat_2'],
    ('bright_sirens', True, 2, 'all', True): ['fcat_2'],
    ('bright_sirens', True, 2, 'mixed', False): ['fcat_2'],
    ('bright_sirens', True, 2, 'mixed', True): ['fcat_2'],
    ('bright_sirens', True, 2, 'none', False): ['fcat_2'],
    ('bright_sirens', True, 2, 'none', True): ['fcat_2'],
    ('dark_sirens_complete', False, 1, 'all', False): ['sigma_kde'],
    ('dark_sirens_complete', False, 1, 'all', True): [],
    ('dark_sirens_complete', False, 1, 'none', False): ['sigma_kde'],
    ('dark_sirens_complete', False, 1, 'none', True): [],
    ('dark_sirens_complete', False, 2, 'all', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', False, 2, 'all', True): ['fcat_2'],
    ('dark_sirens_complete', False, 2, 'mixed', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', False, 2, 'mixed', True): ['fcat_2'],
    ('dark_sirens_complete', False, 2, 'none', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', False, 2, 'none', True): ['fcat_2'],
    ('dark_sirens_complete', True, 1, 'all', False): ['sigma_kde'],
    ('dark_sirens_complete', True, 1, 'all', True): [],
    ('dark_sirens_complete', True, 1, 'none', False): ['sigma_kde'],
    ('dark_sirens_complete', True, 1, 'none', True): [],
    ('dark_sirens_complete', True, 2, 'all', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', True, 2, 'all', True): ['fcat_2'],
    ('dark_sirens_complete', True, 2, 'mixed', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', True, 2, 'mixed', True): ['fcat_2'],
    ('dark_sirens_complete', True, 2, 'none', False): ['sigma_kde', 'sigma_kde_c2', 'fcat_2'],
    ('dark_sirens_complete', True, 2, 'none', True): ['fcat_2'],
    ('dark_sirens', False, 1, 'all', False): ['log10n0', 'delta', 'sigma_kde'],
    ('dark_sirens', False, 1, 'all', True): [],
    ('dark_sirens', False, 1, 'none', False): ['log10n0', 'delta', 'sigma_kde'],
    ('dark_sirens', False, 1, 'none', True): [],
    ('dark_sirens', False, 2, 'all', False): [
        'log10n0',
        'delta',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', False, 2, 'all', True): ['fcat_2'],
    ('dark_sirens', False, 2, 'mixed', False): [
        'log10n0',
        'delta',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', False, 2, 'mixed', True): ['fcat_2'],
    ('dark_sirens', False, 2, 'none', False): [
        'log10n0',
        'delta',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', False, 2, 'none', True): ['fcat_2'],
    ('dark_sirens', True, 1, 'all', False): ['log10n0', 'delta', 'sigma_kde'],
    ('dark_sirens', True, 1, 'all', True): [],
    ('dark_sirens', True, 1, 'none', False): ['log10n0', 'delta', 'b_miss', 'sigma_kde'],
    ('dark_sirens', True, 1, 'none', True): [],
    ('dark_sirens', True, 2, 'all', False): [
        'log10n0',
        'delta',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', True, 2, 'all', True): ['fcat_2'],
    ('dark_sirens', True, 2, 'mixed', False): [
        'log10n0',
        'delta',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'b_miss_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', True, 2, 'mixed', True): ['fcat_2'],
    ('dark_sirens', True, 2, 'none', False): [
        'log10n0',
        'delta',
        'b_miss',
        'sigma_kde',
        'log10n0_c2',
        'delta_c2',
        'b_miss_c2',
        'sigma_kde_c2',
        'fcat_2',
    ],
    ('dark_sirens', True, 2, 'none', True): ['fcat_2'],
    ('spectral_sirens_wl', False, 1, 'all', False): [],
    ('spectral_sirens_wl', False, 1, 'all', True): [],
    ('spectral_sirens_wl', False, 1, 'none', False): [],
    ('spectral_sirens_wl', False, 1, 'none', True): [],
    ('spectral_sirens_wl', False, 2, 'all', False): ['fcat_2'],
    ('spectral_sirens_wl', False, 2, 'all', True): ['fcat_2'],
    ('spectral_sirens_wl', False, 2, 'mixed', False): ['fcat_2'],
    ('spectral_sirens_wl', False, 2, 'mixed', True): ['fcat_2'],
    ('spectral_sirens_wl', False, 2, 'none', False): ['fcat_2'],
    ('spectral_sirens_wl', False, 2, 'none', True): ['fcat_2'],
    ('spectral_sirens_wl', True, 1, 'all', False): [],
    ('spectral_sirens_wl', True, 1, 'all', True): [],
    ('spectral_sirens_wl', True, 1, 'none', False): [],
    ('spectral_sirens_wl', True, 1, 'none', True): [],
    ('spectral_sirens_wl', True, 2, 'all', False): ['fcat_2'],
    ('spectral_sirens_wl', True, 2, 'all', True): ['fcat_2'],
    ('spectral_sirens_wl', True, 2, 'mixed', False): ['fcat_2'],
    ('spectral_sirens_wl', True, 2, 'mixed', True): ['fcat_2'],
    ('spectral_sirens_wl', True, 2, 'none', False): ['fcat_2'],
    ('spectral_sirens_wl', True, 2, 'none', True): ['fcat_2'],
    ('spectral_sirens', False, 1, 'all', False): [],
    ('spectral_sirens', False, 1, 'all', True): [],
    ('spectral_sirens', False, 1, 'none', False): [],
    ('spectral_sirens', False, 1, 'none', True): [],
    ('spectral_sirens', False, 2, 'all', False): ['fcat_2'],
    ('spectral_sirens', False, 2, 'all', True): ['fcat_2'],
    ('spectral_sirens', False, 2, 'mixed', False): ['fcat_2'],
    ('spectral_sirens', False, 2, 'mixed', True): ['fcat_2'],
    ('spectral_sirens', False, 2, 'none', False): ['fcat_2'],
    ('spectral_sirens', False, 2, 'none', True): ['fcat_2'],
    ('spectral_sirens', True, 1, 'all', False): [],
    ('spectral_sirens', True, 1, 'all', True): [],
    ('spectral_sirens', True, 1, 'none', False): [],
    ('spectral_sirens', True, 1, 'none', True): [],
    ('spectral_sirens', True, 2, 'all', False): ['fcat_2'],
    ('spectral_sirens', True, 2, 'all', True): ['fcat_2'],
    ('spectral_sirens', True, 2, 'mixed', False): ['fcat_2'],
    ('spectral_sirens', True, 2, 'mixed', True): ['fcat_2'],
    ('spectral_sirens', True, 2, 'none', False): ['fcat_2'],
    ('spectral_sirens', True, 2, 'none', True): ['fcat_2'],
}


def _base(label):
    return label[:-3] if label.endswith("_c2") else label


def _space(universe_model, use_lss, n_catalogs, q, fix_survey, **kwargs):
    kwargs.setdefault("lss_completion_active", Q_ACTIVITY[(n_catalogs, q)])
    return build_parameter_space(
        pop_model=POP,
        fix_population=True,
        fix_cosmology=True,
        fix_survey=fix_survey,
        universe_model=universe_model,
        use_lss=use_lss,
        n_catalogs=n_catalogs,
        **kwargs,
    )


def _cell_id(cell):
    return "-".join(str(part) for part in cell)


@pytest.mark.parametrize("cell", sorted(EXPECTED, key=_cell_id), ids=_cell_id)
def test_sampled_block_is_bit_identical_to_master(cell):
    expected = EXPECTED[cell]
    res = _space(*cell)

    assert list(res[0]) == expected
    np.testing.assert_array_equal(
        np.asarray(res[1], dtype=float),
        np.asarray([MASTER_BOUNDS[_base(lbl)][0] for lbl in expected], dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(res[2], dtype=float),
        np.asarray([MASTER_BOUNDS[_base(lbl)][1] for lbl in expected], dtype=float),
    )


@pytest.mark.parametrize("cell", sorted(EXPECTED, key=_cell_id), ids=_cell_id)
def test_decoder_agrees_with_builder_in_every_cell(cell):
    """The decoder re-derives the space from opts: same labels, every cell."""
    universe_model, use_lss, n_catalogs, q, fix_survey = cell
    builder_labels = list(_space(*cell)[0])

    opts = SimpleNamespace(
        pop_model=POP,
        universe_model=universe_model,
        fix_population=True,
        fix_cosmology=True,
        fix_survey=fix_survey,
        use_LSS=use_lss,
        n_catalogs=n_catalogs,
        lss_completion_active_by_catalog=Q_ACTIVITY[(n_catalogs, q)],
        # The CLI records the sampler labels here, and a mismatch raises inside
        # build_parameter_decoder -- so this arms the real fail-fast net.
        expected_sampled_labels=tuple(builder_labels),
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )

    assert list(decoder.sampled_labels) == builder_labels


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_and_fiducial_table_cover_the_same_fields():
    """Sampleable + pinned labels == the SurveyParams fiducial table.

    The builder decides what is sampled and the decoder fills everything else
    from the fiducials; a name in only one of the two is exactly the drift the
    single registry exists to prevent (also checked at import time).
    """
    registry = {spec.label for spec in _SURVEY_BLOCK} | set(_PINNED_SURVEY_PARAMS)

    assert registry == set(SURVEY_PARAMS_FID_BY_NAME)


def test_registry_declares_the_block_order_and_bounds():
    # Selection labels (M0hat, sigma_M) appended LAST so pre-existing
    # coordinate indices never move; they sample only under c_mode="selection".
    assert [spec.label for spec in _SURVEY_BLOCK] == [
        "log10n0", "delta", "b_miss", "sigma_kde", "M0hat", "sigma_M",
    ]
    assert [(spec.lower, spec.upper) for spec in _SURVEY_BLOCK] == [
        (-4.0, -1.0), (-3.0, 3.0), (0.0, 3.0), (0.0, 0.05),
        (-23.0, -18.0), (0.05, 3.0),
    ]


def test_never_sampled_labels_are_absent_from_every_cell():
    """z50/w/alpha_miss (and their _c{k} spellings) carry no prior anywhere."""
    pinned = set(_PINNED_SURVEY_PARAMS)
    for cell in EXPECTED:
        labels = set(_space(*cell)[0])
        assert not (labels & pinned)
        assert not ({f"{name}_c2" for name in pinned} & labels)


def test_unregistered_universe_model_samples_whole_block():
    """The one deliberate change vs master: a universe_model the registry does
    not know still samples the WHOLE sampleable block, but that block no longer
    carries z50/w/alpha_miss (master returned the seven-row block here).  No run
    reaches this path -- both CLIs always pass a universe_model."""
    labels, *_ = build_parameter_space(
        POP, fix_population=True, fix_cosmology=True, fix_survey=False
    )

    assert labels == ["log10n0", "delta", "b_miss", "sigma_kde"]


# ---------------------------------------------------------------------------
# Hazard 1: an override on a never-sampled label was accepted in silence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["z50", "w", "alpha_miss"])
def test_prior_override_on_never_sampled_label_raises(label):
    with pytest.raises(ValueError) as excinfo:
        _space("dark_sirens", True, 1, "none", False,
               prior_overrides={label: [0.1, 2.0]})

    message = str(excinfo.value)
    assert f"'{label}'" in message
    assert "never sampled for any universe model" in message


@pytest.mark.parametrize("label", ["z50", "w"])
def test_never_sampled_reason_says_what_the_label_actually_is(label):
    with pytest.raises(ValueError, match="generative-truth field of the mock generator"):
        _space("dark_sirens", True, 1, "none", False,
               prior_overrides={label: [0.1, 2.0]})


def test_alpha_miss_reason_names_the_degeneracy():
    with pytest.raises(ValueError, match=r"alpha_miss\*b_miss"):
        _space("dark_sirens", True, 1, "none", False,
               prior_overrides={"alpha_miss": [0.0, 1.0]})


def test_prior_override_on_suffixed_never_sampled_label_raises():
    with pytest.raises(ValueError, match="never sampled"):
        _space("dark_sirens", True, 2, "none", False,
               prior_overrides={"z50_c2": [0.1, 2.0]})


def test_fixed_value_on_never_sampled_label_warns_and_is_still_honoured():
    """A FIXED value is demoted to a warning: pinning a generative-truth field
    is harmless (the decoder still puts the value on SurveyParams) and archived
    settings.json / the mock scripts do exactly that."""
    with pytest.warns(UserWarning, match="not a sampled parameter"):
        labels, *_ = _space("dark_sirens", True, 1, "none", False,
                            fixed_parameter_values={"z50": 1.2})
    assert "z50" not in labels

    opts = SimpleNamespace(
        pop_model=POP,
        universe_model="dark_sirens",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        use_LSS=True,
        n_catalogs=1,
    )
    with pytest.warns(UserWarning):
        decoder = build_parameter_decoder(
            opts,
            get_fixed_population_params(POP),
            fixed_parameter_values={"z50": 1.2},
        )
    _cosmo, survey, *_ = decoder.decode(np.zeros(len(decoder.sampled_labels)))

    assert float(survey.z50) == pytest.approx(1.2)


def test_unknown_survey_label_still_raises_key_error():
    with pytest.raises(KeyError, match="Unknown prior override labels"):
        _space("dark_sirens", True, 1, "none", False,
               prior_overrides={"z_50": [0.1, 2.0]})


# ---------------------------------------------------------------------------
# Hazard 2: b_miss under --use_lss off was rejected with the WRONG reason
# ---------------------------------------------------------------------------

def test_b_miss_override_with_use_lss_off_blames_use_lss_not_a_phantom_q_table():
    with pytest.raises(ValueError) as excinfo:
        _space("dark_sirens", False, 1, "none", False,
               prior_overrides={"b_miss": [0.0, 1.0]})

    message = str(excinfo.value)
    assert "--use_lss off" in message
    assert "delta_g is the all-zero" in message
    # The old guard claimed an active Q_LSS completion table that does not exist.
    assert "Q_LSS" not in message


def test_b_miss_fixed_value_with_use_lss_off_is_still_fatal():
    """Master already rejected this (with the wrong reason); only the message
    changes.  A pinned b_miss asserts a modulation the likelihood cannot apply."""
    with pytest.raises(ValueError, match="--use_lss off"):
        _space("dark_sirens", False, 1, "none", False,
               fixed_parameter_values={"b_miss": 1.0})


def test_b_miss_override_with_q_table_blames_the_q_table():
    with pytest.raises(ValueError) as excinfo:
        _space("dark_sirens", True, 1, "all", False,
               prior_overrides={"b_miss": [0.0, 1.0]})

    message = str(excinfo.value)
    assert "Q_LSS completion table" in message
    assert "catalog 1" in message
    assert "--use_lss" not in message


def test_b_miss_override_under_mixed_q_names_the_owning_catalog():
    """Q on catalog 1 only: b_miss (catalog 1) is inert, b_miss_c2 is sampled."""
    with pytest.raises(ValueError, match="catalog 1 has an active Q_LSS"):
        _space("dark_sirens", True, 2, "mixed", False,
               prior_overrides={"b_miss": [0.0, 1.0]})

    labels, lower, upper, *_ = _space(
        "dark_sirens", True, 2, "mixed", False,
        prior_overrides={"b_miss_c2": [0.5, 1.5]},
    )
    assert float(lower[labels.index("b_miss_c2")]) == 0.5
    assert float(upper[labels.index("b_miss_c2")]) == 1.5


def test_b_miss_c2_override_with_q_on_catalog_2_names_catalog_2():
    with pytest.raises(ValueError, match="catalog 2 has an active Q_LSS"):
        _space("dark_sirens", True, 2, "none", False,
               lss_completion_active=(False, True),
               prior_overrides={"b_miss_c2": [0.0, 1.0]})


# ---------------------------------------------------------------------------
# Model-level inertness: the reason names the universe model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "universe_model", ["spectral_sirens", "bright_sirens", "spectral_sirens_wl"]
)
@pytest.mark.parametrize("label", ["log10n0", "delta", "b_miss", "sigma_kde"])
def test_catalog_free_models_reject_any_survey_override(universe_model, label):
    with pytest.raises(ValueError) as excinfo:
        _space(universe_model, True, 1, "none", False,
               prior_overrides={label: [0.0, 1.0]})

    message = str(excinfo.value)
    assert f"universe_model '{universe_model}' is catalog-free" in message
    assert "samples no survey parameters at all" in message


@pytest.mark.parametrize("label", ["log10n0", "delta", "b_miss"])
def test_complete_catalog_model_rejects_completion_overrides(label):
    with pytest.raises(ValueError, match="100%-complete catalog"):
        _space("dark_sirens_complete", True, 1, "none", False,
               prior_overrides={label: [0.0, 1.0]})


def test_complete_catalog_model_still_takes_a_sigma_kde_override():
    labels, lower, upper, *_ = _space(
        "dark_sirens_complete", True, 1, "none", False,
        prior_overrides={"sigma_kde": [0.0, 0.02]},
    )

    assert labels == ["sigma_kde"]
    assert float(upper[0]) == 0.02


@pytest.mark.parametrize("label", ["log10n0", "sigma_kde"])
def test_fixed_value_on_a_model_inactive_label_warns(label):
    """Not fatal: pinning a survey parameter this model does not sample is
    harmless, and post-processing replays archived settings.json verbatim."""
    with pytest.warns(UserWarning, match="not a sampled parameter"):
        labels, *_ = _space("spectral_sirens", True, 1, "none", False,
                            fixed_parameter_values={label: 0.0})

    assert labels == []


# ---------------------------------------------------------------------------
# The decoder fills the non-sampled SurveyParams fields from the fiducials
# ---------------------------------------------------------------------------

def test_decoder_fills_unsampled_survey_fields_from_fiducials():
    opts = SimpleNamespace(
        pop_model=POP,
        universe_model="dark_sirens",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        use_LSS=True,
        n_catalogs=1,
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    coord = np.array([-3.0, 1.0, 2.0, 0.01])  # log10n0, delta, b_miss, sigma_kde

    assert list(decoder.sampled_labels) == ["log10n0", "delta", "b_miss", "sigma_kde"]
    _cosmo, survey, *_ = decoder.decode(coord)

    assert float(survey.n0) == pytest.approx(1e-3)
    assert float(survey.delta) == pytest.approx(1.0)
    assert float(survey.b_miss) == pytest.approx(2.0)
    assert float(survey.sigma_kde) == pytest.approx(0.01)
    # Never sampled -> the shared fiducials, not whatever sat at that index.
    assert float(survey.z50) == pytest.approx(SURVEY_PARAMS_FID_BY_NAME["z50"])
    assert float(survey.w) == pytest.approx(SURVEY_PARAMS_FID_BY_NAME["w"])
    assert float(survey.alpha_miss) == pytest.approx(
        SURVEY_PARAMS_FID_BY_NAME["alpha_miss"]
    )


def test_decode_mixture_fills_each_catalog_from_its_own_suffixed_labels():
    opts = SimpleNamespace(
        pop_model=POP,
        universe_model="dark_sirens",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        use_LSS=True,
        n_catalogs=2,
        lss_completion_active_by_catalog=(True, False),
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    labels = list(decoder.sampled_labels)
    coord = np.zeros(len(labels))
    coord[labels.index("log10n0")] = -3.0
    coord[labels.index("log10n0_c2")] = -2.0
    coord[labels.index("b_miss_c2")] = 2.5
    coord[labels.index("fcat_2")] = 0.5

    _cosmo, surveys, *_ = decoder.decode_mixture(coord)

    assert float(surveys[0].n0) == pytest.approx(1e-3)
    assert float(surveys[1].n0) == pytest.approx(1e-2)
    # Catalog 1 is Q-active, so its b_miss is not sampled and stays fiducial.
    assert float(surveys[0].b_miss) == pytest.approx(
        SURVEY_PARAMS_FID_BY_NAME["b_miss"]
    )
    assert float(surveys[1].b_miss) == pytest.approx(2.5)
