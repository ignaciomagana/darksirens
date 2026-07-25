"""Q_LSS provenance enforcement (issue #261).

A prebuilt LSS completion table Q is a fit conditioned on build-time n0, delta,
bias and cosmology. Sampling or re-fixing those does not propagate into Q — the
mismatch is absorbed into the completion field as spurious redshift structure
and biases H0, which is exactly what the builder warns about. Loading Q used to
only print a warning while the inference kept sampling log10n0/delta; it is now
fatal.

H0 is deliberately exempt (it is the measurand, and is sampled in every
dark-siren run); see darksirens/inference/q_provenance.py.
"""

import math

import pytest

from darksirens.inference.q_provenance import check_lss_completion_provenance


def _fiducials(**overrides):
    base = {
        "path": "/tmp/q.h5",
        "fiducial_H0": 67.74,
        "fiducial_Om0": 0.3075,
        "fiducial_w0": -1.0,
        "fiducial_wa": 0.0,
        "fiducial_n0": 1e-2,        # linear density; sampler works in log10
        "fiducial_delta": 0.0,
        "bias_b_miss": 1.0,
    }
    base.update(overrides)
    return base


SAFE_LABELS = ["H0", "$\\alpha$", "$m_{\\min}$", "$\\gamma$"]


# ============================================================================
# No table -> no-op
# ============================================================================

@pytest.mark.parametrize("empty", [None, {}])
def test_no_completion_table_is_a_no_op(empty):
    check_lss_completion_provenance(empty, ["H0", "log10n0", "delta"], {})


# ============================================================================
# Sampling a conditioning parameter is fatal
# ============================================================================

@pytest.mark.parametrize("param", ["log10n0", "delta", "b_miss", "Om0", "w0", "wa"])
def test_sampling_a_conditioned_parameter_raises(param):
    with pytest.raises(ValueError, match="Q_LSS provenance mismatch"):
        check_lss_completion_provenance(_fiducials(), SAFE_LABELS + [param], {})


def test_error_names_the_offending_parameter_and_the_build_value():
    with pytest.raises(ValueError) as exc:
        check_lss_completion_provenance(_fiducials(), SAFE_LABELS + ["log10n0"], {})
    msg = str(exc.value)
    assert "log10n0" in msg
    assert "-2" in msg            # log10(1e-2)
    assert "rebuilding Q" in msg  # actionable


def test_per_catalog_suffixed_labels_are_caught():
    """log10n0_c2 must gate exactly like log10n0 (K>=2 mixtures)."""
    with pytest.raises(ValueError, match="log10n0_c2"):
        check_lss_completion_provenance(_fiducials(), SAFE_LABELS + ["log10n0_c2"], {})


def test_multiple_offenders_are_all_reported():
    with pytest.raises(ValueError) as exc:
        check_lss_completion_provenance(
            _fiducials(), SAFE_LABELS + ["log10n0", "delta", "Om0"], {}
        )
    msg = str(exc.value)
    for param in ("log10n0", "delta", "Om0"):
        assert param in msg


# ============================================================================
# Pinned to the build value -> allowed
# ============================================================================

def test_all_conditioned_parameters_absent_passes():
    check_lss_completion_provenance(_fiducials(), SAFE_LABELS, {})


def test_fixed_to_the_build_value_passes():
    check_lss_completion_provenance(
        _fiducials(), SAFE_LABELS,
        {"log10n0": math.log10(1e-2), "delta": 0.0, "b_miss": 1.0,
         "Om0": 0.3075, "w0": -1.0, "wa": 0.0},
    )


@pytest.mark.parametrize("label,value", [
    ("log10n0", -1.5),
    ("delta", 0.4),
    ("b_miss", 1.3),
    ("Om0", 0.28),
    ("w0", -0.9),
    ("wa", 0.2),
])
def test_fixed_to_a_different_value_raises(label, value):
    with pytest.raises(ValueError, match="Q_LSS provenance mismatch"):
        check_lss_completion_provenance(_fiducials(), SAFE_LABELS, {label: value})


def test_tiny_float_drift_in_a_fixed_value_is_tolerated():
    """Round-tripping through settings.json must not trip the guard."""
    check_lss_completion_provenance(
        _fiducials(), SAFE_LABELS, {"log10n0": math.log10(1e-2) * (1 + 1e-12)},
    )


# ============================================================================
# H0 carve-out
# ============================================================================

def test_sampling_h0_is_allowed():
    """H0 is the measurand and is sampled in every dark-siren run; a literal
    hard-fail on it would make prebuilt Q tables unusable rather than safe."""
    check_lss_completion_provenance(_fiducials(), ["H0", "$\\gamma$"], {})


def test_h0_fixed_away_from_the_build_value_is_also_allowed():
    check_lss_completion_provenance(_fiducials(), SAFE_LABELS, {"H0": 73.0})


# ============================================================================
# Unverifiable tables
# ============================================================================

def test_legacy_table_without_stamped_fiducials_raises():
    """No stamped values -> cannot verify -> refuse, matching the loader's
    existing no-silent-fallback policy for zgrid/indexing."""
    with pytest.raises(ValueError, match="no build-time fiducials"):
        check_lss_completion_provenance({"path": "/tmp/legacy.h5"}, SAFE_LABELS, {})


def test_partially_stamped_table_checks_what_it_can():
    """Only n0 stamped: n0 is enforced, the unstamped ones cannot be compared
    to a number but are still forbidden from being sampled."""
    partial = {"path": "/tmp/q.h5", "fiducial_n0": 1e-3}
    check_lss_completion_provenance(partial, SAFE_LABELS, {})
    with pytest.raises(ValueError, match="log10n0"):
        check_lss_completion_provenance(partial, SAFE_LABELS + ["log10n0"], {})
    with pytest.raises(ValueError, match="delta"):
        check_lss_completion_provenance(partial, SAFE_LABELS + ["delta"], {})


def test_non_positive_stamped_n0_is_not_compared_as_a_log():
    """A zero/negative stamped n0 has no log10; it must not produce a NaN
    comparison that silently passes."""
    bad = {"path": "/tmp/q.h5", "fiducial_n0": 0.0, "fiducial_delta": 0.0}
    check_lss_completion_provenance(bad, SAFE_LABELS, {"log10n0": -2.0})
    with pytest.raises(ValueError, match="not stamped"):
        check_lss_completion_provenance(bad, SAFE_LABELS + ["log10n0"], {})


# ============================================================================
# Loader plumbing: the fiducials actually reach the check
# ============================================================================

def test_loader_returns_fiducials_key_even_without_a_table():
    from types import SimpleNamespace

    from darksirens.catalogs.lss import maybe_load_lss_completion
    from darksirens.redshift.grid import zgrid

    opts = SimpleNamespace(
        lss_completion=None, survey_path=None,
        universe_model="dark_sirens", lss_marginalize=False,
    )
    out = maybe_load_lss_completion(opts, zgrid=zgrid)
    assert "lss_completion_fiducials" in out
    assert out["lss_completion_fiducials"] is None
