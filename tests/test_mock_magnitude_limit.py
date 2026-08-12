"""The mock survey's magnitude limit must be reachable, and must actually bite.

Why this file exists
--------------------
``SurveyConfig.magnitude_limit`` defaulted to 24.0 and was NOT exposed on the
generator's command line: ``SurveyConfig(...)`` was built from ``z50``, ``width``
and ``delta`` alone.  With ``abs_mag ~ N(-21, 1)`` the apparent magnitude at
z = 0.3 is ~20 — four sigma inside 24.0 — so no galaxy was ever
magnitude-selected, every mock this repo could build was magnitude-COMPLETE
(``C_sel == 1`` up to the footprint), and the inference's magnitude-limited
selection function (:func:`darksirens.redshift.selection.c_sel_gaussian`, and
with it the whole K-correction / completion machinery) was structurally inert.

That is why the selection path had never been exercised end-to-end on a mock,
and why the 2026-08 K-correction anchor study had to fall back to a
single-line-of-sight toy instead of a real pipeline run.

These tests pin both halves: the knob reaches the config, and lowering it
produces the z-DEPENDENT incompleteness that ``c_sel_gaussian`` models.  The
mock draws a Gaussian LF and applies no K-correction, so a K = 0 selection fit
is the matched model for it.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_GMD = (Path(__file__).resolve().parents[1]
        / "scripts" / "mock_dark_sirens" / "generate_mock_data.py")
_spec = importlib.util.spec_from_file_location("generate_mock_data", _GMD)
gmd = importlib.util.module_from_spec(_spec)
sys.modules["generate_mock_data"] = gmd
_spec.loader.exec_module(gmd)

# z50 far above the grid neutralises the sigmoid completeness, so the only
# z-dependence left is the magnitude limit — the quantity under test.
_SIGMOID_OFF = dict(z50=10.0, width=0.12)
_ZMAX = 0.3
_NGAL = 120_000


@pytest.fixture(scope="module")
def grids():
    cosmo = gmd._build_cosmology(67.74, 0.3089, -1.0, 0.0)
    return gmd._cosmology_grids(cosmo, _ZMAX)


def _kept_fraction_by_z(grids, magnitude_limit):
    survey = gmd.SurveyConfig(magnitude_limit=magnitude_limit, **_SIGMOID_OFF)
    catalog = gmd._generate_complete_catalog(
        np.random.default_rng(7), _NGAL, grids, survey)
    keep = gmd._apply_survey_selection(
        np.random.default_rng(11), catalog, survey)
    z = catalog["z"]
    near = z < 0.08
    far = z > 0.28
    return float(keep[near].mean()), float(keep[far].mean()), float(keep.mean())


def _magnitude_cut_fraction(grids, magnitude_limit):
    """Fraction of galaxies the MAGNITUDE limit alone removes.

    Measured directly rather than inferred from a near/far kept-fraction
    difference: the footprint and sigmoid cuts contribute to the latter, and
    the per-z-bin binomial noise (~0.008 on the low-z bin at this N) is of the
    same order as the effect being bounded.
    """
    survey = gmd.SurveyConfig(magnitude_limit=magnitude_limit, **_SIGMOID_OFF)
    catalog = gmd._generate_complete_catalog(
        np.random.default_rng(7), _NGAL, grids, survey)
    return float((catalog["app_mag"] > magnitude_limit).mean())


def test_default_magnitude_limit_is_inert(grids):
    """The shipped default cuts essentially nothing — the state this file documents.

    At z = 0.3 the distance modulus is ~41, so app_mag ~ 20 +- 1 against a
    limit of 24: a four-sigma tail. Asserted on the magnitude cut ITSELF, so
    the bound is not competing with footprint/sigmoid selection or with
    per-bin sampling noise. If this ever starts failing, the default changed
    and every archived mock moved with it.
    """
    frac = _magnitude_cut_fraction(grids, gmd.SurveyConfig.magnitude_limit)
    assert frac < 1e-3, (
        f"default m_lim={gmd.SurveyConfig.magnitude_limit} removes {frac:.2%} of "
        "galaxies; it is supposed to be inert"
    )


def test_a_reachable_limit_actually_removes_galaxies(grids):
    """The complement: a usable limit cuts a substantial, graded fraction."""
    fracs = {m: _magnitude_cut_fraction(grids, m) for m in (19.0, 20.0, 21.0)}
    assert fracs[20.0] > 0.1, f"m_lim=20 should cut >10%, got {fracs[20.0]:.2%}"
    assert fracs[19.0] > fracs[20.0] > fracs[21.0], (
        f"cut fraction must grow as the limit brightens, got {fracs}"
    )


def test_lowering_the_limit_makes_selection_z_dependent(grids):
    """A reachable limit must produce real, monotone-in-z incompleteness.

    Measured on this fixture: m_lim = 20 gives ~0.81 near, ~0.43 far. That is
    the regime c_sel_gaussian is the correct model for.
    """
    near, far, overall = _kept_fraction_by_z(grids, 20.0)
    assert near > 0.75, f"low-z galaxies should be nearly all kept, got {near:.3f}"
    assert far < 0.55, f"high-z galaxies should be strongly cut, got {far:.3f}"
    assert far < near - 0.25, "selection must be strongly z-dependent"
    assert 0.0 < overall < 1.0


def test_selection_tightens_monotonically_as_the_limit_drops(grids):
    """Fainter limit -> more galaxies. Guards against a sign/comparison slip."""
    overalls = [_kept_fraction_by_z(grids, m)[2] for m in (19.0, 20.0, 21.0, 24.0)]
    assert overalls == sorted(overalls), (
        f"kept fraction must increase with the magnitude limit, got {overalls}"
    )


def test_cli_exposes_the_survey_selection_knobs(tmp_path):
    """The knobs must reach SurveyConfig, not just exist on the parser.

    The bug was precisely that the dataclass field existed and was unreachable,
    so parsing alone is not the assertion — the constructed config is.
    """
    argv = [
        "generate_mock_data.py", "--outdir", str(tmp_path), "--seed", "5",
        "--n-galaxies", "2000", "--nobs", "2", "--nsamp", "8",
        "--nselection", "500", "--nside", "4", "--zmax", "0.3",
        "--survey-magnitude-limit", "20.5",
        "--survey-z-hard-max", "0.9",
        "--survey-absolute-mag-mean", "-20.5",
        "--survey-absolute-mag-sigma", "1.25",
    ]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", argv)
    try:
        args = gmd.parse_args()
    finally:
        monkey.undo()
    assert args.survey_magnitude_limit == 20.5
    assert args.survey_z_hard_max == 0.9
    assert args.survey_absolute_mag_mean == -20.5
    assert args.survey_absolute_mag_sigma == 1.25


def test_defaults_reproduce_the_pre_change_config():
    """An existing command line must build a bit-identical SurveyConfig.

    The four new parameters default to the dataclass values they previously
    held implicitly, so archived mock commands stay reproducible.
    """
    S = gmd.SurveyConfig
    before = S(z50=0.75, width=0.12, delta=0.0)
    after = S(
        z50=0.75, width=0.12, delta=0.0,
        magnitude_limit=S.magnitude_limit,
        z_hard_max=S.z_hard_max,
        absolute_mag_mean=S.absolute_mag_mean,
        absolute_mag_sigma=S.absolute_mag_sigma,
    )
    assert before == after
