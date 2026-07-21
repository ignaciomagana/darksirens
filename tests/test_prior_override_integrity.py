"""Regression tests for BUG-3: prior overrides on mark / suffixed labels were
silently ignored (never applied to their bounds), and a fixed+override overlap
on a non-cosmo/pop/survey/sky-base label (marks, suffixed survey, suffixed
marks, mixture sticks) raised a bare ``KeyError`` from
``validate_fixed_parameter_overrides`` instead of the intended ``ValueError``.

One shared config (``BASE_KWARGS`` below) makes every accepted label family
live at once -- a K=2 multitracer mixture with a per-catalog ``loglinear``
mark model and a ``dipole`` sky model -- so a single parametrization can cover
cosmology, population, survey (base + suffixed ``_c2``), sky, marks (base +
suffixed ``_c2``), and the ``fcat_2`` mixture stick.
"""
import re
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

BASE_KWARGS = dict(
    pop_model="powerlaw+peak",
    fix_population=False,
    fix_cosmology=False,
    fix_survey=False,
    universe_model="dark_sirens",
    sky_model="dipole",
    mark_model="loglinear",
    mark_names=("logmstar",),
    mark_names_by_catalog=(("logmstar",), ("logmstar",)),
    n_catalogs=2,
)


def _space(prior_overrides=None, fixed_parameter_values=None):
    return build_parameter_space(
        **BASE_KWARGS,
        prior_overrides=prior_overrides or {},
        fixed_parameter_values=fixed_parameter_values or {},
    )


def _bounds_of(res, label):
    labels, lower, upper = res[0], res[1], res[2]
    idx = labels.index(label)
    return (float(lower[idx]), float(upper[idx]))


# label, default bounds, override bounds, an in-range fixed value (inside the
# OVERRIDE bounds) and an out-of-range fixed value (inside the default bounds
# but outside the override bounds, so the overlap check -- not some unrelated
# default-bounds check -- is what fires).
LABEL_FAMILIES = [
    pytest.param("H0", (20.0, 120.0), (50.0, 90.0), 70.0, 95.0, id="cosmology_base"),
    pytest.param(
        r"$\alpha_{\rm PL}$", (-4.0, 6.0), (-2.0, 2.0), 0.0, 3.0, id="population"
    ),
    pytest.param("log10n0", (-4.0, -1.0), (-3.5, -2.0), -2.5, -1.5, id="survey_base"),
    pytest.param(
        "log10n0_c2", (-4.0, -1.0), (-3.5, -2.0), -2.5, -1.5, id="survey_suffixed_c2"
    ),
    pytest.param("$d_x$", (-1.0, 1.0), (-0.5, 0.5), 0.0, 0.8, id="sky"),
    pytest.param(
        "eta_logmstar", (-5.0, 5.0), (-2.0, 2.0), 0.0, 3.0, id="mark_base"
    ),
    pytest.param(
        "eta_logmstar_c2", (-5.0, 5.0), (-2.0, 2.0), 0.0, 3.0, id="mark_suffixed_c2"
    ),
    pytest.param("fcat_2", (0.0, 1.0), (0.2, 0.8), 0.5, 0.9, id="mixture_stick"),
]


@pytest.mark.parametrize(
    "label,default_bounds,override_bounds,fixed_in_range,fixed_out_of_range",
    LABEL_FAMILIES,
)
def test_override_only_moves_bounds(
    label, default_bounds, override_bounds, fixed_in_range, fixed_out_of_range
):
    """Bug (a): a prior override for a mark / suffixed label must actually
    change the sampled bounds, not just pass label validation."""
    default_res = _space()
    assert _bounds_of(default_res, label) == default_bounds

    overridden_res = _space(prior_overrides={label: list(override_bounds)})
    assert label in overridden_res[0]
    assert _bounds_of(overridden_res, label) == override_bounds


@pytest.mark.parametrize(
    "label,default_bounds,override_bounds,fixed_in_range,fixed_out_of_range",
    LABEL_FAMILIES,
)
def test_fixed_only_removes_label_unchanged_behavior(
    label, default_bounds, override_bounds, fixed_in_range, fixed_out_of_range
):
    """Fixing a label (no override) removes it from the sampled coordinates,
    same as the pre-existing cosmology-only behavior."""
    res = _space(fixed_parameter_values={label: fixed_in_range})
    assert label not in res[0]


@pytest.mark.parametrize(
    "label,default_bounds,override_bounds,fixed_in_range,fixed_out_of_range",
    LABEL_FAMILIES,
)
def test_fixed_and_override_in_range_no_exception(
    label, default_bounds, override_bounds, fixed_in_range, fixed_out_of_range
):
    """A fixed value inside the overridden bounds is accepted (fixed wins,
    override reported as ignored) -- no exception of any kind."""
    res = _space(
        prior_overrides={label: list(override_bounds)},
        fixed_parameter_values={label: fixed_in_range},
    )
    labels = res[0]
    fixed_parameter_statuses = res[10]

    assert label not in labels
    assert fixed_parameter_statuses[label] == "fixed; override ignored"


@pytest.mark.parametrize(
    "label,default_bounds,override_bounds,fixed_in_range,fixed_out_of_range",
    LABEL_FAMILIES,
)
def test_fixed_and_override_out_of_range_raises_value_error(
    label, default_bounds, override_bounds, fixed_in_range, fixed_out_of_range
):
    """Bug (b): a fixed value outside the OVERRIDDEN bounds for a mark /
    suffixed / mixture label must raise the existing descriptive ValueError,
    not a bare KeyError from an incomplete bounds map."""
    lo, hi = override_bounds
    with pytest.raises(
        ValueError,
        match=(
            rf"Fixed value for '{re.escape(label)}' "
            rf"\({float(fixed_out_of_range)}\) is outside the overridden "
            rf"prior bounds \[{lo}, {hi}\]"
        ),
    ):
        _space(
            prior_overrides={label: list(override_bounds)},
            fixed_parameter_values={label: fixed_out_of_range},
        )
