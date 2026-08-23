"""The per-pixel completeness cancels the clustering signal (PHY-01).

The DEFAULT ``c_mode='per_pixel'`` estimator divides a pixel's OWN observed
density by an ISOTROPIC expected density,

    C_est(z|p) = clip[ dN_obs_s(z|p) / dN_exp_s(z), 0, 1 ],

so under the matched-kernel idealization ``dN_obs = C_sel nbar (1 + delta)`` it
estimates ``clip[C_sel (1 + delta), 0, 1]``: the observed angular clustering is
absorbed into the completeness.  With nothing modulating the missing branch
(``--use_lss`` off, no ``--lss_completion`` table) the completed density is

    dN_obs + (1 - C_est) dN_exp = C_sel nbar (1+delta) + nbar - C_sel nbar (1+delta)
                                = nbar,

i.e. the clustering the dark-siren host weighting exists to use cancels
EXACTLY, and above the clip the overdense excess is discarded outright.

These tests do two things: MEASURE the cancellation on a constant-selection
injection with a known ``delta`` (so the claim is a number, not algebra on
paper), and pin the runtime warning that names it.  The default itself is NOT
flipped here -- that is an owner decision; ``--c_mode aggregate`` plus a
mean-one Q table is the fix the warning points at.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.inference.loaders import warn_per_pixel_clustering_cancellation
from darksirens.redshift import zgrid
from darksirens.redshift.completion import _precompute_grids, completion_curves

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
SURVEY = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0,
                      b_miss=0.0, alpha_miss=1.0)

#: Constant true selection and the per-pixel overdensities injected on top.
#: Every one keeps ``C_SEL (1 + delta) < 1`` so the estimator stays BELOW its
#: clip: the cancellation is the finding, and the clip is a separate (worse)
#: failure exercised by test_above_the_clip_the_overdense_excess_is_discarded.
C_SEL = 0.8
DELTAS = np.array([-0.5, -0.2, 0.0, 0.1, 0.2])

#: The matched-kernel estimator divides by the SMOOTHED expected density while
#: the missing branch multiplies the unsmoothed one, so the completed density
#: keeps a residual ``C_est (dN_exp_s - dN_exp)``.  That is the estimator's own
#: photo-z/kernel systematic (PHY-05), NOT the cancellation under test, so the
#: injection is read off where it is negligible -- selected from the grids
#: themselves rather than by a hand-picked z range.
_SMOOTHING_RESIDUAL_MAX = 1e-3


def _injected_catalog(deltas, c_sel=C_SEL):
    """A catalog whose OBSERVED density is exactly ``c_sel nbar (1 + delta_p)``.

    The numerator is injected directly into the KDE cache -- which is what
    ``_row_C`` reads -- so the true completeness and the true contrast are
    known exactly rather than being the outcome of a mock draw.
    """
    n_pix = len(deltas)
    zg = jnp.tile(jnp.asarray([[0.2, 0.4]]), (n_pix, 1))
    base = EMCatalog(
        apix=1.0, zgals=zg, dzgals=jnp.full_like(zg, 0.02),
        wgals=jnp.ones_like(zg), ngals=jnp.full((n_pix,), 2, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(COSMO, SURVEY, base)
    kde = c_sel * (1.0 + jnp.asarray(deltas)[:, None]) * grids.dN_exp_smooth
    return base._replace(dN_obs_kde=kde), grids


def _window(grids):
    """Grid nodes where the smooth/unsmoothed expected densities agree."""
    dN_exp = np.asarray(grids.dN_exp)
    dN_exp_s = np.asarray(grids.dN_exp_smooth)
    ok = dN_exp > 0.0
    resid = np.abs(np.where(ok, dN_exp_s / np.where(ok, dN_exp, 1.0), 1.0) - 1.0)
    w = ok & (resid < _SMOOTHING_RESIDUAL_MAX)
    assert w.sum() > 50, f"only {int(w.sum())} usable grid nodes"
    return w


def test_per_pixel_completeness_absorbs_the_injected_overdensity():
    """The mechanism itself: C_est == C_sel (1 + delta), not C_sel."""
    cat, grids = _injected_catalog(DELTAS)
    curves = completion_curves(COSMO, SURVEY, cat)
    # C_eff == 1 - dN_miss/dN_exp, and with lss == 1 that IS the clipped C_est.
    C_est = np.asarray(curves.C_eff)
    w = _window(grids)
    for p, delta in enumerate(DELTAS):
        np.testing.assert_allclose(
            C_est[p][w], C_SEL * (1.0 + delta), rtol=1e-3,
            err_msg=f"pixel {p} (delta={delta})")


def test_the_completed_density_loses_the_clustering_signal():
    """MEASURED cancellation: +-50% true contrast -> ~1e-3 in the model.

    ``dN_obs + dN_miss`` is the density the host prior actually places, so this
    is the quantity that carries (or fails to carry) the angular information.
    """
    cat, grids = _injected_catalog(DELTAS)
    curves = completion_curves(COSMO, SURVEY, cat)
    completed = np.asarray(cat.dN_obs_kde) + np.asarray(curves.dN_miss)
    w = _window(grids)
    mean = completed[:, w].mean(axis=0)
    model_contrast = completed[:, w] / mean - 1.0
    true_contrast = DELTAS[:, None] * np.ones_like(model_contrast)

    worst = np.max(np.abs(model_contrast))
    assert worst < 2e-3, (
        f"per-pixel completion retained {worst:.3e} of contrast; the "
        "cancellation claim (PHY-01) no longer holds and this test's premise "
        "needs rechecking"
    )
    # ... and the suppression is at least two orders of magnitude on the
    # strongest injected pixel, which is the finding.
    assert np.max(np.abs(true_contrast)) / max(worst, 1e-30) > 1e2


def test_above_the_clip_the_overdense_excess_is_discarded():
    """C_sel = 0.8 with delta = 0.5 clips at C = 1: the model places 1.0x the
    mean where the truth is 1.5x -- worse than cancellation."""
    deltas = np.array([0.0, 0.5])
    cat, grids = _injected_catalog(deltas)
    curves = completion_curves(COSMO, SURVEY, cat)
    completed = np.asarray(cat.dN_obs_kde) + np.asarray(curves.dN_miss)
    w = _window(grids)
    dN_exp = np.asarray(grids.dN_exp)[w]
    # The overdense pixel is capped at its own observed count (C == 1 -> no
    # missing budget at all): 0.8 * 1.5 = 1.2 x nbar, against a truth of 1.5.
    ratio = completed[1][w] / dN_exp
    np.testing.assert_allclose(ratio, 1.2, rtol=2e-3)
    assert np.all(ratio < 1.5 * 0.95)


# ── the guarded warning ──────────────────────────────────────────────────────

class _Opts:
    def __init__(self, **kw):
        self.c_mode = "per_pixel"
        self.use_LSS = False
        self.lss_completion = None
        self.lss_field_mode = "table"
        self.__dict__.update(kw)


#: A footprint: 90% empty at 20 gal/pixel, where Poisson predicts exp(-20).
_FOOTPRINT = np.where(np.arange(1000) < 100, 20, 0)
#: Sparse but all-sky: empty pixels are what the mean count predicts.
_SPARSE = np.random.default_rng(0).poisson(0.3, size=1000)


def _warnings_for(opts, ngals):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_per_pixel_clustering_cancellation(opts, ngals)
    return [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]


def test_the_warning_fires_on_the_default_configuration():
    msgs = _warnings_for(_Opts(), _FOOTPRINT)
    assert len(msgs) == 1
    text = msgs[0]
    assert "CANCELS EXACTLY" in text
    assert "c_mode aggregate" in text
    assert "lss_completion" in text
    assert "bias H0" in text


def test_sparsity_is_not_a_footprint():
    """The same separation the S-3 guard uses: a genuinely sparse all-sky
    catalog is not making a claim about sky it never observed."""
    assert _warnings_for(_Opts(), _SPARSE) == []


@pytest.mark.parametrize("kw", [
    {"c_mode": "aggregate"},
    {"c_mode": "selection"},
    {"use_LSS": True},
    {"lss_completion": ["q.h5"]},
    {"lss_field_mode": "latent"},
])
def test_no_warning_once_something_modulates_the_missing_branch(kw):
    """Either the completeness is no longer per-pixel, or the missing branch
    carries an angular factor -- in both cases the cancellation argument does
    not apply."""
    assert _warnings_for(_Opts(**kw), _FOOTPRINT) == []


def test_an_empty_count_array_is_inert():
    assert _warnings_for(_Opts(), np.array([], dtype=int)) == []
