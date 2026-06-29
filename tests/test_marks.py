"""
Tests for the marked-host model (galaxy marks -> BBH-host efficiency h(m|eta)).

The marked path lives entirely in ``prepare_redshift_prior_state``; the per-sample
evaluator is reused.  ``mark_model="none"`` is the legacy galaxy-count model, and
``loglinear`` with ``eta=0`` + unit weights reduces to it exactly.
"""
import numpy as np
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
    integ = np.trapezoid(np.exp(lp), np.asarray(zgrid))
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
