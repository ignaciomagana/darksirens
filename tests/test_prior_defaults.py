import sys
import types

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

from darksirens.inference.prior import build_parameter_space
from darksirens.utils.cosmology import w0PriorLower, w0PriorUpper, waPriorLower, waPriorUpper


def test_dark_sirens_b_miss_dropped_when_lss_completion_active():
    """A loaded Q_LSS completion table REPLACES the (1 + b_eff*delta_g)
    local-overdensity factor, so b_miss no longer enters the likelihood. It must
    then be dropped from the sampled block, else it is a phantom flat nuisance
    dimension that offsets logZ and invalidates Bayes-factor comparisons."""

    def _labels(lss_active):
        labels, *_ = build_parameter_space(
            "powerlaw+peak",
            fix_population=True,
            fix_cosmology=True,
            fix_survey=False,
            universe_model="dark_sirens",
            lss_completion_active=lss_active,
        )
        return set(labels)

    without_q = _labels(False)
    with_q = _labels(True)

    # Without a Q table, dark_sirens samples b_miss alongside log10n0/delta/sigma_kde.
    assert "b_miss" in without_q
    # With a Q table, only b_miss is dropped; log10n0/delta (which still feed
    # _assemble_curves) and sigma_kde stay.
    assert "b_miss" not in with_q
    assert {"log10n0", "delta", "sigma_kde"} <= with_q
    assert without_q - with_q == {"b_miss"}


def test_survey_default_priors_are_physical_and_overridable():
    """Default bounds of the whole sampleable survey block.

    No ``universe_model`` is given, so the survey registry does not recognise
    the model and the WHOLE block is sampled -- which is now exactly the four
    sampleable labels.  ``z50``/``w``/``alpha_miss`` used to appear here as
    flat rows that every real universe model filtered straight back out; they
    are SurveyParams fields (generative-truth for the mock generator, and the
    degeneracy-pinned alpha_miss) and no longer carry a prior at all.
    """
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}

    assert labels == ["log10n0", "delta", "b_miss", "sigma_kde"]
    assert bounds["log10n0"] == (-4.0, -1.0)
    assert bounds["delta"] == (-3.0, 3.0)
    assert bounds["b_miss"] == (0.0, 3.0)
    assert bounds["sigma_kde"] == (0.0, 0.05)

    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={"log10n0": [-6.0, -2.0]},
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}
    assert bounds["log10n0"] == (-6.0, -2.0)



def test_cosmology_default_priors_include_cpl_grid_supported_bounds():
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=False,
        fix_survey=True,
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}

    assert bounds["w0"] == (w0PriorLower, w0PriorUpper)
    assert bounds["wa"] == (waPriorLower, waPriorUpper)

    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=False,
        fix_survey=True,
        prior_overrides={"w0": [-1.2, -0.8], "wa": [-0.5, 0.5]},
        fixed_parameter_values={"w0": -1.0},
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}

    assert "w0" not in labels
    assert bounds["wa"] == (-0.5, 0.5)


def test_fixed_de_removes_only_dark_energy_cosmology_labels():
    labels, lower, upper, *rest = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=False,
        fix_survey=True,
        fix_de=True,
    )
    n_cosmo_eff = rest[4]

    assert "H0" in labels
    assert "Om0" in labels
    assert "w0" not in labels
    assert "wa" not in labels
    assert n_cosmo_eff == 2
    assert len(labels) == len(lower) == len(upper) == 2


def test_fixed_cosmology_supersedes_fixed_de():
    labels, lower, upper, *rest = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
        fix_de=True,
    )
    n_cosmo_eff = rest[4]

    assert labels == []
    assert len(lower) == len(upper) == 0
    assert n_cosmo_eff == 0


def test_fixed_parameter_prior_override_overlap_in_range_is_reported():
    res = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=False,
        fix_survey=True,
        prior_overrides={"H0": [60.0, 80.0]},
        fixed_parameter_values={"H0": 67.74},
    )

    labels, lower, upper = res[0], res[1], res[2]
    fixed_parameter_statuses = res[10]

    assert "H0" not in labels
    assert len(labels) == len(lower) == len(upper)
    assert fixed_parameter_statuses == {"H0": "fixed; override ignored"}


def test_fixed_parameter_prior_override_overlap_out_of_range_raises():
    with pytest.raises(
        ValueError,
        match=r"Fixed value for 'H0' \(67\.74\) is outside the overridden prior bounds \[80\.0, 90\.0\]",
    ):
        build_parameter_space(
            "powerlaw+peak",
            fix_population=True,
            fix_cosmology=False,
            fix_survey=True,
            prior_overrides={"H0": [80.0, 90.0]},
            fixed_parameter_values={"H0": 67.74},
        )


# ---------------------------------------------------------------------------
# K-catalog multitracer mixture (n_catalogs >= 2): per-catalog suffixed
# survey blocks + fcat_2..fcat_K sticks, appended before sky/mark.  K=1 (the
# default / explicit) must remain bit-identical.
# ---------------------------------------------------------------------------

def test_n_catalogs_2_dark_sirens_exposes_fcat2_with_beta_prior_kind():
    labels, lower, upper, *rest = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={},
        fixed_parameter_values={},
        universe_model="dark_sirens",
        n_catalogs=2,
    )
    prior_kinds = rest[8]

    assert "fcat_2" in labels
    idx = labels.index("fcat_2")
    assert float(lower[idx]) == 0.0
    assert float(upper[idx]) == 1.0
    assert prior_kinds[idx] == ("beta", 1.0, 1.0)


def test_n_catalogs_2_dark_sirens_only_exposes_active_suffixed_survey_labels():
    """dark_sirens samples {log10n0, delta, b_miss, sigma_kde} from the survey
    registry; the catalog-2 block must be built from the SAME registry, so
    z50_c2/w_c2/alpha_miss_c2 (never-sampled fields) cannot appear either."""
    labels, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={},
        fixed_parameter_values={},
        universe_model="dark_sirens",
        n_catalogs=2,
    )
    active_c2 = {"log10n0_c2", "delta_c2", "b_miss_c2", "sigma_kde_c2"}
    inactive_c2 = {"z50_c2", "w_c2", "alpha_miss_c2"}

    assert active_c2 <= set(labels)
    assert not (inactive_c2 & set(labels))


def test_n_catalogs_1_labels_bounds_identical_to_call_without_kwarg():
    r_default = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={},
        fixed_parameter_values={},
        universe_model="dark_sirens",
    )
    r_explicit = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={},
        fixed_parameter_values={},
        universe_model="dark_sirens",
        n_catalogs=1,
    )

    assert r_default[0] == r_explicit[0]  # labels
    np.testing.assert_array_equal(r_default[1], r_explicit[1])  # lower
    np.testing.assert_array_equal(r_default[2], r_explicit[2])  # upper
    assert r_default[11] == r_explicit[11]  # prior_kinds


def test_n_catalogs_2_prior_override_on_suffixed_survey_label():
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={"log10n0_c2": [-5.0, -3.0]},
        fixed_parameter_values={},
        universe_model="dark_sirens",
        n_catalogs=2,
    )
    idx = labels.index("log10n0_c2")
    assert float(lower[idx]) == -5.0
    assert float(upper[idx]) == -3.0
    # The unsuffixed (catalog 1) block is untouched by the c2 override.
    idx1 = labels.index("log10n0")
    assert float(lower[idx1]) == -4.0
    assert float(upper[idx1]) == -1.0


def test_n_catalogs_2_fixed_parameter_values_removes_fcat2_from_labels():
    labels, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        prior_overrides={},
        fixed_parameter_values={"fcat_2": 0.3},
        universe_model="dark_sirens",
        n_catalogs=2,
    )
    assert "fcat_2" not in labels
