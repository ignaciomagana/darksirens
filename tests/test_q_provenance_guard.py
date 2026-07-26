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
    """log10n0_c2 gates like log10n0 -- against CATALOG 2's OWN table.

    This test previously passed a single table and asserted that log10n0_c2
    raised. That encoded the defect: catalog 2's independent survey parameter
    was being judged against catalog 1's build values, which both rejected the
    supported "Q on catalog 1 only" setup and never verified catalog 2's table.
    The suffixed label must be checked against the suffixed catalog's table.
    """
    with pytest.raises(ValueError, match="log10n0_c2"):
        check_lss_completion_provenance(
            [None, _fiducials(path="/tmp/q2.h5")],
            SAFE_LABELS + ["log10n0_c2"], {},
        )
    # ...and with no table on catalog 2, it is none of the guard's business.
    check_lss_completion_provenance(
        [_fiducials(), None], SAFE_LABELS + ["log10n0_c2"],
        {"log10n0": math.log10(1e-2), "delta": 0.0, "b_miss": 1.0},
    )


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


# ============================================================================
# w0 / wa must actually be comparable (regression: loader dropped them)
# ============================================================================
#
# The builder stamps fiducial_w0/fiducial_wa and _Q_CONDITIONED names them, but
# the loader's forwarding whitelist omitted both, so the guard silently skipped
# the comparison for the two dark-energy parameters it claims to protect.

@pytest.mark.parametrize("param,key,build,bad", [
    ("w0", "fiducial_w0", -1.0, -0.85),
    ("wa", "fiducial_wa", 0.0, 0.4),
])
def test_fixed_w0_wa_mismatch_is_caught(param, key, build, bad):
    fid = _fiducials(**{key: build})
    check_lss_completion_provenance(fid, SAFE_LABELS, {param: build})   # matching -> ok
    with pytest.raises(ValueError, match="Q_LSS provenance mismatch"):
        check_lss_completion_provenance(fid, SAFE_LABELS, {param: bad})


def test_loader_forwards_w0_wa_fiducials():
    """The guard is only live if the loader actually hands these through."""
    import inspect

    from darksirens.catalogs import lss as lss_mod
    from darksirens.inference.q_provenance import _Q_CONDITIONED

    src = inspect.getsource(lss_mod.maybe_load_lss_completion)
    for key in _Q_CONDITIONED.values():
        assert f'"{key}"' in src, (
            f"{key} is a Q-conditioning key but maybe_load_lss_completion does "
            "not forward it, so the provenance check for it is dead code"
        )


# ============================================================================
# Per-catalog validation for K >= 2
# ============================================================================

def test_per_catalog_survey_params_use_their_own_table():
    """catalog 2's log10n0_c2 is checked against catalog 2's table, not 1's."""
    cat1 = _fiducials(fiducial_n0=1e-2)
    cat2 = _fiducials(fiducial_n0=1e-3, path="/tmp/q2.h5")
    labels = SAFE_LABELS
    # Each fixed to ITS OWN build value -> passes.
    check_lss_completion_provenance(
        [cat1, cat2], labels,
        {"log10n0": math.log10(1e-2), "log10n0_c2": math.log10(1e-3)},
    )
    # Swapping them must be caught (previously both were compared to cat1).
    with pytest.raises(ValueError, match="log10n0_c2"):
        check_lss_completion_provenance(
            [cat1, cat2], labels,
            {"log10n0": math.log10(1e-2), "log10n0_c2": math.log10(1e-2)},
        )


def test_q_on_one_catalog_only_does_not_reject_the_other():
    """The supported mixed config: catalog 1 has Q, catalog 2 does not.

    Catalog 2 keeps sampling its own log10n0_c2/delta_c2, which must NOT be
    judged against catalog 1's table.
    """
    check_lss_completion_provenance(
        [_fiducials(), None],
        SAFE_LABELS + ["log10n0_c2", "delta_c2", "b_miss_c2"],
        {"log10n0": math.log10(1e-2), "delta": 0.0, "b_miss": 1.0},
    )


def test_second_catalogs_table_is_actually_verified():
    """Catalog 2's own table must be checked, not skipped."""
    cat2 = _fiducials(fiducial_n0=1e-3, path="/tmp/q2.h5")
    with pytest.raises(ValueError, match="log10n0_c2"):
        check_lss_completion_provenance(
            [None, cat2], SAFE_LABELS + ["log10n0_c2"], {},
        )


def test_global_cosmology_checked_against_every_catalog_table():
    """Om0/w0/wa are global, so EVERY catalog's table must agree with them."""
    cat1 = _fiducials(fiducial_Om0=0.3075)
    cat2 = _fiducials(fiducial_Om0=0.28, path="/tmp/q2.h5")
    with pytest.raises(ValueError, match="catalog 2"):
        check_lss_completion_provenance([cat1, cat2], SAFE_LABELS, {"Om0": 0.3075})


def test_sampled_global_cosmology_reported_once_per_offending_table():
    with pytest.raises(ValueError) as exc:
        check_lss_completion_provenance(
            [_fiducials(), _fiducials(path="/tmp/q2.h5")], SAFE_LABELS + ["Om0"], {},
        )
    assert "Om0" in str(exc.value)


def test_bare_dict_still_accepted_for_single_catalog():
    """Back-compat: a plain dict behaves exactly as before."""
    check_lss_completion_provenance(_fiducials(), SAFE_LABELS, {})
    with pytest.raises(ValueError, match="Q_LSS provenance mismatch"):
        check_lss_completion_provenance(_fiducials(), SAFE_LABELS + ["delta"], {})


def test_all_none_sequence_is_a_no_op():
    check_lss_completion_provenance([None, None], SAFE_LABELS + ["log10n0"], {})


def test_catalog_index_parsing():
    from darksirens.inference.q_provenance import _catalog_index

    assert _catalog_index("log10n0") == 0
    assert _catalog_index("log10n0_c2") == 1
    assert _catalog_index("log10n0_c10") == 9
    assert _catalog_index("H0") == 0
