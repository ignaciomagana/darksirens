"""
Tests for the marked-host model (galaxy marks -> BBH-host efficiency h(m|eta)).

The marked path lives entirely in ``prepare_redshift_prior_state``; the per-sample
evaluator is reused.  ``mark_model="none"`` is the legacy galaxy-count model, and
``loglinear`` with ``eta=0`` + unit weights reduces to it exactly.
"""
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.redshift.prior import prepare_redshift_prior_state, eval_redshift_prior_with_state
from darksirens.marks import (
    MARK_MODEL_NAMES, mark_model_prior_parser, mark_fiducial, available_marks,
    mark_model_parser, LogLinearMarks,
)

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
SURVEY = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)


def _cat(logmstar=None, logssfr=None):
    """Two-row catalog (unit weights, on-the-fly KDE) with optional marks."""
    zg = np.full((2, 3), 100.0)
    zg[0, :2] = [0.10, 0.30]
    zg[1, :1] = [0.20]
    ng = np.array([2, 1], dtype=np.int32)
    dz = np.full((2, 3), 0.01)
    w = np.zeros((2, 3))
    w[0, :2] = 1.0
    w[1, :1] = 1.0
    fields = dict(
        apix=1.0, zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz), wgals=jnp.asarray(w),
        ngals=jnp.asarray(ng), delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=None, pixel_to_cache_idx=None, unique_pixels=None,
    )
    if logmstar is not None:
        fields["mark_logmstar"] = jnp.asarray(logmstar)
    if logssfr is not None:
        fields["mark_logssfr"] = jnp.asarray(logssfr)
    return EMCatalog(**fields)


def test_registry_and_param_specs():
    assert set(MARK_MODEL_NAMES) == {"none", "loglinear"}
    lows, highs, labels, kinds, _ = mark_model_prior_parser("loglinear", ("logmstar", "logssfr"))
    assert list(labels) == ["eta_logmstar", "eta_logssfr"]
    assert all(k[0] == "uniform" for k in kinds)
    assert mark_fiducial("loglinear", ("logmstar",)) == (0.0,)
    # none model has no sampled parameters
    lo0, hi0, lab0, _, _ = mark_model_prior_parser("none")
    assert len(lab0) == 0


def test_available_marks_detection():
    assert available_marks(_cat()) == ()
    assert available_marks(_cat(logmstar=np.zeros((2, 3)))) == ("logmstar",)
    both = _cat(logmstar=np.zeros((2, 3)), logssfr=np.zeros((2, 3)))
    assert available_marks(both) == ("logmstar", "logssfr")


def test_log_h_is_linear_in_marks():
    m = np.array([[0.5, -0.5, 9.9], [0.2, 9.9, 9.9]])  # padded slots arbitrary
    cat = _cat(logmstar=m)
    log_h = np.asarray(LogLinearMarks(("logmstar",)).log_h(cat, jnp.array([2.0])))
    assert np.allclose(log_h, 2.0 * m)


def _prior(cat, mark_model="none", eta=None, mark_names=()):
    state = prepare_redshift_prior_state(
        "dark_sirens", COSMO, SURVEY, cat,
        mark_model=mark_model,
        mark_params=(None if eta is None else jnp.asarray(eta)),
        mark_names=mark_names,
    )
    pix = jnp.zeros(NG, jnp.int32)  # row 0
    lp = eval_redshift_prior_with_state("dark_sirens", state, zgrid, pix, COSMO, SURVEY, cat)
    return np.asarray(lp)


def test_eta_zero_unit_weights_reduces_to_unmarked():
    """loglinear with eta=0 + unit weights == the legacy galaxy-count model."""
    cat = _cat(logmstar=np.array([[0.5, -0.5, 0.0], [0.2, 0.0, 0.0]]))
    base = _prior(cat, mark_model="none")
    marked0 = _prior(cat, mark_model="loglinear", eta=[0.0], mark_names=("logmstar",))
    assert np.max(np.abs(np.exp(base) - np.exp(marked0))) < 1e-9


def test_marked_prior_normalizes_per_pixel():
    cat = _cat(logmstar=np.array([[0.6, -0.4, 0.0], [0.1, 0.0, 0.0]]))
    lp = _prior(cat, mark_model="loglinear", eta=[1.5], mark_names=("logmstar",))
    integ = _trapezoid(np.exp(lp), np.asarray(zgrid))
    assert abs(integ - 1.0) < 5e-3


def test_marked_reweights_toward_high_mark_galaxy():
    """eta_M>0 raises the catalog weight of the high-logM* galaxy (at z=0.10),
    so the prior gains mass near z=0.10 relative to the low-mark galaxy (z=0.30)."""
    # row 0: gal at z=0.10 has high logM*, gal at z=0.30 has low logM*
    cat = _cat(logmstar=np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]))
    base = np.exp(_prior(cat, mark_model="loglinear", eta=[0.0], mark_names=("logmstar",)))
    up = np.exp(_prior(cat, mark_model="loglinear", eta=[2.0], mark_names=("logmstar",)))
    z = np.asarray(zgrid)
    i_hi = int(np.argmin(np.abs(z - 0.10)))   # high-mark galaxy redshift
    i_lo = int(np.argmin(np.abs(z - 0.30)))   # low-mark galaxy redshift
    # ratio at the high-mark z increases relative to the low-mark z
    assert (up[i_hi] / base[i_hi]) > (up[i_lo] / base[i_lo])


def test_missing_mark_field_errors():
    with pytest.raises(ValueError, match="mark_logmstar is None|requested"):
        _prior(_cat(), mark_model="loglinear", eta=[1.0], mark_names=("logmstar",))


# ------------------------------------------------------------
# Depth truncation: the marked amplitude must carry log_depth_mass
# ------------------------------------------------------------
# The kernels are renormalised to unit mass on [0, z_depth], so the marked
# observed amplitude paired with them is exp(log_N_host + log_depth_mass) --
# exactly the unmarked `Nobs * exp(log_depth_mass)` scaling.  Storing the raw
# log_N_host in the state over-weighted the catalog branch by 1/m_pix and broke
# the per-pixel unit normalisation.  Row 0 of ``_cat`` has one galaxy below and
# one above z_depth=0.25, so m = 1/2 there.
# A small pixel area / low n0 keeps N_miss ~ O(1): with the production-sized
# missing budget (N_miss ~ 1e9) a 2x error in an O(1) observed amplitude is
# numerically invisible.
_SURVEY_DEPTH = SurveyParams(
    n0=1e-9, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0, z_depth=0.25,
)


def _cat_small_apix(logmstar):
    return _cat(logmstar=logmstar)._replace(apix=1e-4)


def _prior_depth(cat, mark_model="none", eta=None, mark_names=(), row=0):
    state = prepare_redshift_prior_state(
        "dark_sirens", COSMO, _SURVEY_DEPTH, cat,
        mark_model=mark_model,
        mark_params=(None if eta is None else jnp.asarray(eta)),
        mark_names=mark_names,
    )
    pix = jnp.full(NG, row, jnp.int32)
    lp = eval_redshift_prior_with_state(
        "dark_sirens", state, zgrid, pix, COSMO, _SURVEY_DEPTH, cat
    )
    return np.asarray(lp)


def test_marked_amplitude_carries_depth_mass():
    """eta=0 + unit weights reduces to the unmarked model WITH a survey depth."""
    cat = _cat_small_apix(np.array([[0.5, -0.5, 0.0], [0.2, 0.0, 0.0]]))
    for row in (0, 1):
        base = _prior_depth(cat, mark_model="none", row=row)
        marked0 = _prior_depth(
            cat, mark_model="loglinear", eta=[0.0], mark_names=("logmstar",), row=row
        )
        assert np.max(np.abs(np.exp(base) - np.exp(marked0))) < 1e-9


def test_marked_prior_normalizes_per_pixel_under_depth():
    """The depth-truncated marked prior still integrates to 1 per pixel."""
    cat = _cat_small_apix(np.array([[0.6, -0.4, 0.0], [0.1, 0.0, 0.0]]))
    for row in (0, 1):
        lp = _prior_depth(
            cat, mark_model="loglinear", eta=[1.5], mark_names=("logmstar",), row=row
        )
        integ = _trapezoid(np.exp(lp), np.asarray(zgrid))
        assert abs(integ - 1.0) < 5e-3


# ------------------------------------------------------------
# Amplitude units: count x <h>_w, not the raw weighted mass
# ------------------------------------------------------------
# The catalog:missing odds are COUNT odds (module docstring of redshift/prior.py),
# so the marked amplitude must be invariant under WEIGHT -> c*WEIGHT: the raw
# Sum_i w_i h_i let a luminosity-weighted catalog (L/L_sun ~ 1e10) swamp the
# missing branch and switch the completeness correction off silently.

def _cat_weighted(logmstar, scale=1.0, vary=True):
    """``_cat`` with non-unit (optionally per-galaxy varying) weights."""
    cat = _cat(logmstar=logmstar)
    w = np.zeros((2, 3))
    w[0, :2] = [1.0, 3.0] if vary else [1.0, 1.0]
    w[1, :1] = 2.0 if vary else 1.0
    return cat._replace(wgals=jnp.asarray(w * scale))


def test_marked_prior_is_invariant_to_the_weight_scale():
    logm = np.array([[0.6, -0.4, 0.0], [0.1, 0.0, 0.0]])
    base = _prior(_cat_weighted(logm, scale=1.0), mark_model="loglinear",
                  eta=[1.5], mark_names=("logmstar",))
    scaled = _prior(_cat_weighted(logm, scale=1e10), mark_model="loglinear",
                    eta=[1.5], mark_names=("logmstar",))
    assert np.max(np.abs(np.exp(base) - np.exp(scaled))) < 1e-9


def test_eta_zero_reduces_to_unmarked_with_non_unit_weights():
    """The eta = 0 reduction must not need unit weights (only h == 1)."""
    logm = np.array([[0.5, -0.5, 0.0], [0.2, 0.0, 0.0]])
    cat = _cat_weighted(logm, scale=7.0)
    base = _prior(cat, mark_model="none")
    marked0 = _prior(cat, mark_model="loglinear", eta=[0.0],
                     mark_names=("logmstar",))
    assert np.max(np.abs(np.exp(base) - np.exp(marked0))) < 1e-9


# ------------------------------------------------------------
# mu_miss outside the catalog's redshift coverage
# ------------------------------------------------------------

def test_mu_miss_is_continuous_and_reduces_to_the_mean_outside_coverage():
    """Uninformed z-bins take the catalog-wide mean efficiency, not 1.

    With centred marks Jensen gives <h> >= 1, so the old homogeneous default made
    the missing density drop by that factor across ONE 0.125-wide bin at the
    catalog's coverage edge -- and made eta shift the catalog:missing odds even
    for marks with no redshift structure.
    """
    from darksirens.redshift.prior import _mu_miss_from_flat

    rng = np.random.default_rng(0)
    zs = rng.uniform(0.01, 0.30, 20000)          # coverage stops at z = 0.3
    h = np.exp(np.clip(2.0 * rng.normal(0.0, 1.0, zs.size), -7.0, 7.0))
    mu = np.asarray(_mu_miss_from_flat(
        jnp.asarray(zs), jnp.asarray(h), jnp.ones(zs.size)
    ))
    z = np.asarray(zgrid)
    mean_h = float(h.mean())
    assert mean_h > 3.0                          # the Jensen factor is large here
    # No cliff across the coverage edge, and far outside it mu_miss is the mean.
    assert abs(mu[np.searchsorted(z, 0.25)] / mu[np.searchsorted(z, 0.40)] - 1.0) < 0.1
    assert abs(mu[np.searchsorted(z, 3.0)] / mean_h - 1.0) < 1e-6


def test_z_independent_marks_leave_the_prior_unchanged():
    """A mark with no redshift structure must not move the prior at all.

    h is then a constant that multiplies BOTH branches (amplitude N_obs*<h>_w and
    dN_miss*mu_miss), so it cancels in p(z|pix) -- eta is a pure SHAPE parameter.
    """
    cat = _cat(logmstar=np.full((2, 3), 0.7))
    base = _prior(cat, mark_model="none")
    for eta in (0.5, 2.0, -1.0):
        marked = _prior(cat, mark_model="loglinear", eta=[eta],
                        mark_names=("logmstar",))
        assert np.max(np.abs(np.exp(base) - np.exp(marked))) < 1e-9, eta


# --- eta-liveness guard: uncentred marks saturate the log-h clip -------------


def _mark_table(mean, spread, seed=0):
    """(4, 5) mark table with 3 real galaxies per row, drawn about ``mean``."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((4, 5))
    arr[:, :3] = mean + spread * rng.normal(size=(4, 3))
    return jnp.asarray(arr)


_NGALS = jnp.asarray(np.full(4, 3, dtype=np.int32))


def test_centred_marks_pass_the_liveness_guard():
    """A z-centred table saturates only its outlier tail, so eta stays live."""
    from darksirens.marks import check_marks_centred, get_mark_model

    model = get_mark_model("loglinear", ("logmstar",))
    check_marks_centred(
        model, {"logmstar": _mark_table(0.0, 0.5)}, _NGALS, where="unit test"
    )


def test_uncentred_marks_are_rejected():
    """Raw logM* (~10.5 dex) pins every galaxy to the +-7 rail at |eta| <= 5.

    The clip then makes log h locally constant in eta over the whole catalog:
    the posterior would come back flat and the run would look converged while
    measuring nothing, which is exactly what this guard exists to stop.
    """
    from darksirens.marks import check_marks_centred, get_mark_model

    model = get_mark_model("loglinear", ("logmstar",))
    with pytest.raises(ValueError, match="marked-host model would be dead"):
        check_marks_centred(
            model, {"logmstar": _mark_table(10.5, 0.5)}, _NGALS, where="unit test"
        )


def test_liveness_guard_uses_the_joint_worst_case_over_marks():
    """Two marks each safe alone can still saturate together.

    log h = sum_k eta_k m_k, and the eta_k are independent, so the reachable
    |log h| is eta_bound * sum_k |m_k| -- not the per-mark maximum.
    """
    from darksirens.marks import check_marks_centred, get_mark_model

    one = _mark_table(1.0, 0.02, seed=1)         # 5 * 1.0 = 5 < 7 on its own
    model1 = get_mark_model("loglinear", ("logmstar",))
    check_marks_centred(model1, {"logmstar": one}, _NGALS, where="unit test")

    model2 = get_mark_model("loglinear", ("logmstar", "logssfr"))
    with pytest.raises(ValueError, match="marked-host model would be dead"):
        check_marks_centred(
            model2,
            {"logmstar": one, "logssfr": _mark_table(1.0, 0.02, seed=2)},
            _NGALS,
            where="unit test",
        )


def test_liveness_guard_ignores_padded_slots():
    """Padding sits at 0 in the mark tables but must not dilute the fraction.

    A table whose REAL galaxies are all saturated must be rejected even when
    most columns are padding -- counting the zero-filled slots as live galaxies
    would push the saturated fraction under the threshold and let it through.
    """
    from darksirens.marks import check_marks_centred, get_mark_model

    arr = np.zeros((4, 20))
    arr[:, :2] = 10.5                            # 2 real galaxies, 18 padded
    model = get_mark_model("loglinear", ("logmstar",))
    with pytest.raises(ValueError, match="marked-host model would be dead"):
        check_marks_centred(
            model,
            {"logmstar": jnp.asarray(arr)},
            jnp.asarray(np.full(4, 2, dtype=np.int32)),
            where="unit test",
        )


def test_liveness_guard_is_a_no_op_for_the_none_model():
    from darksirens.marks import check_marks_centred, get_mark_model

    check_marks_centred(get_mark_model("none"), {}, _NGALS, where="unit test")


def test_flat_field_marks_are_guarded_too():
    """mu_miss reads the flat full-sky table through the same clip."""
    from darksirens.marks import check_flat_marks_centred, get_mark_model

    model = get_mark_model("loglinear", ("logmstar",))
    check_flat_marks_centred(
        model, jnp.asarray(np.random.default_rng(3).normal(0.0, 0.5, (100, 1))),
        where="unit test",
    )
    with pytest.raises(ValueError, match="would kill mu_miss"):
        check_flat_marks_centred(
            model, jnp.asarray(np.full((100, 1), 10.5)), where="unit test"
        )
    check_flat_marks_centred(model, None, where="unit test")


def test_liveness_guard_defers_to_the_missing_mark_error():
    """A selected mark with no array is a data error, reported downstream.

    The guard must stay quiet there: complaining about the saturation of a
    table that does not exist would bury the actual (clear) missing-mark
    message under an unrelated one.
    """
    from darksirens.marks import check_marks_centred, get_mark_model

    model = get_mark_model("loglinear", ("logmstar", "logssfr"))
    check_marks_centred(
        model,
        {"logmstar": _mark_table(10.5, 0.5), "logssfr": None},
        _NGALS,
        where="unit test",
    )
