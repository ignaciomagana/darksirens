import sys
import types

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


def test_survey_default_priors_are_physical_and_overridable():
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
    )
    bounds = {label: (float(lo), float(hi)) for label, lo, hi in zip(labels, lower, upper)}

    assert bounds["log10n0"] == (-4.0, -1.0)
    assert bounds["z50"] == (0.05, 4.5)
    assert bounds["w"] == (0.02, 1.5)
    assert bounds["delta"] == (-3.0, 3.0)
    assert bounds["b_miss"] == (0.0, 3.0)
    assert bounds["alpha_miss"] == (0.0, 1.0)

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
