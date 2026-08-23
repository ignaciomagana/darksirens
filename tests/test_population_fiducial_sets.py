"""The curated fiducial sets: legacy (out-of-prior) vs in_prior_v2 (PHY-12).

Three curated compositions inherit the component blueprints' default fiducials
while overriding the priors that would contain them, so their
``--fix_population`` truth sits OUTSIDE the model's own declared prior:

    2powerlaws+peak    PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40];
                       G.mu 35 vs [50, 100]
    2powerlaws+2peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]
    2powerlaws+3peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]

A fixed run does not apply the sampling prior, so that is not by itself an
invalid density -- it becomes a defect when the fixed arm is read as nested in
the sampled model, which can then never represent its own fiducial.  These
tests pin BOTH sets: legacy stays bit-identical (archived runs and mocks), and
in_prior_v2 is required to be inside every bound.
"""

import logging

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.gw.populations import (
    FIDUCIAL_SETS,
    FIDUCIAL_SET_IN_PRIOR,
    FIDUCIAL_SET_LEGACY,
    get_fixed_population_params,
    pop_model_prior_parser,
)

#: The three curated entries whose legacy fiducials violate their own priors.
_OFFENDERS = ["2powerlaws+peak", "2powerlaws+2peaks", "2powerlaws+3peaks"]


def _fid(name, fiducials):
    return np.asarray(get_fixed_population_params(name, fiducials=fiducials))


def _bounds(name):
    lo, hi, labels, _, _ = pop_model_prior_parser(name)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), labels


@pytest.mark.parametrize("name", _OFFENDERS)
def test_legacy_fiducials_are_outside_the_declared_priors(name):
    """The defect itself, pinned: this is what the corrected set exists for."""
    fid = _fid(name, FIDUCIAL_SET_LEGACY)
    lo, hi, labels = _bounds(name)
    outside = [(labels[i], float(fid[i]), float(lo[i]), float(hi[i]))
               for i in range(len(labels))
               if not (lo[i] <= fid[i] <= hi[i])]
    assert outside, (
        f"{name}: the legacy fiducial is now INSIDE its priors. If that was "
        "deliberate, the legacy/in_prior split is obsolete -- delete it "
        "rather than leaving a dead flag."
    )


@pytest.mark.parametrize("name", _OFFENDERS)
def test_in_prior_fiducials_lie_inside_every_declared_bound(name):
    """The corrected set's whole contract."""
    fid = _fid(name, FIDUCIAL_SET_IN_PRIOR)
    lo, hi, labels = _bounds(name)
    outside = [(labels[i], float(fid[i]), float(lo[i]), float(hi[i]))
               for i in range(len(labels))
               if not (lo[i] <= fid[i] <= hi[i])]
    assert not outside, f"{name}: in_prior_v2 fiducial still outside bounds: {outside}"


@pytest.mark.parametrize("name", _OFFENDERS)
def test_in_prior_patch_moves_only_the_violating_parameters(name):
    """Nothing else in the curated tuning is touched: same length, same values
    everywhere the legacy set was already legal."""
    legacy = _fid(name, FIDUCIAL_SET_LEGACY)
    fixed = _fid(name, FIDUCIAL_SET_IN_PRIOR)
    lo, hi, labels = _bounds(name)
    assert legacy.shape == fixed.shape
    for i, label in enumerate(labels):
        legal = bool(lo[i] <= legacy[i] <= hi[i])
        if legal:
            assert fixed[i] == legacy[i], (
                f"{name}: in_prior_v2 moved {label}, which was already legal"
            )


def test_every_curated_model_is_in_prior_under_the_corrected_set():
    """No curated composition may warn under in_prior_v2 -- the build runs with
    ``on_violation='error'``, so a silent regression is impossible."""
    from darksirens.gw.populations.registry import CURATED

    for name, entry in CURATED.items():
        if entry.mass is None:
            try:
                get_fixed_population_params(name, fiducials=FIDUCIAL_SET_IN_PRIOR)
            except ValueError as exc:  # pragma: no cover - the failure message
                pytest.fail(f"{name}: {exc}")


def test_legacy_is_the_default_and_is_bit_identical(caplog):
    """The default must not move: archived fixed-population runs and every mock
    built from ``get_fixed_population_params`` are pinned to these numbers."""
    for name in _OFFENDERS + ["powerlaw+peak", "brokenpowerlaw+2peaks"]:
        with caplog.at_level(logging.WARNING):
            default = _fid(name, FIDUCIAL_SET_LEGACY)
        explicit = np.asarray(get_fixed_population_params(name))
        np.testing.assert_array_equal(default, explicit)


def test_legacy_warning_names_the_escape_hatch(caplog):
    """The warning is where an operator meets the problem, so it must say what
    to pass instead."""
    with caplog.at_level(logging.WARNING,
                         logger="darksirens.gw.populations.grammar"):
        get_fixed_population_params("2powerlaws+peak")
    text = caplog.text
    assert "outside prior" in text
    assert "in_prior_v2" in text, text
    assert "--population_fiducials" in text, text


def test_unknown_fiducial_set_is_refused():
    with pytest.raises(ValueError, match="unknown fiducial set"):
        get_fixed_population_params("2powerlaws+peak", fiducials="v3")
    assert set(FIDUCIAL_SETS) == {FIDUCIAL_SET_LEGACY, FIDUCIAL_SET_IN_PRIOR}


def test_bespoke_models_ignore_the_fiducial_set():
    """Published parameter vectors have no per-slot priors to patch."""
    for name in ("gwtc3_fiducial_plpeak", "gwtc5_fiducial_bpl2peaks"):
        np.testing.assert_array_equal(
            _fid(name, FIDUCIAL_SET_LEGACY), _fid(name, FIDUCIAL_SET_IN_PRIOR))
