"""
Regression tests for the **traced** (jitted) redshift-prior path.

These guard the trace-safety of the LSS-completion and marked-host extensions.
Both features were only ever unit-tested through their *eager* entry points
(``completion_curves`` / ``prepare_redshift_prior_state`` called directly with a
concrete catalog), so a ``ConcretizationTypeError`` inside the jitted likelihood
went unnoticed until the smoke-test suite ran them end-to-end:

* LSS  (``--lss_completion`` radial & gp3d): ``int(em_catalog.lss_completion_indexing)``
  on a traced enum leaf in ``_resolve_lss_completion_row_tables``.
* marks (``--mark_model loglinear``):        ``float(zgrid[-1])`` in ``_mu_miss_grid``.

In real inference ``prepare_redshift_prior_state`` runs **inside**
``@jit darksiren_log_likelihood``, so here we reproduce that by wrapping the
prepare + per-sample eval in ``jax.jit`` (the eager tests in
``test_lss_completion.py`` / ``test_marks.py`` do not, which is why they passed
while the jitted path crashed).  Each test must raise on pre-fix code and return
a finite, per-pixel-normalised ``p(z|pix)`` after the fix.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from darksirens.em import zgrid
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.em.completion import build_pixel_kde_cache
from darksirens.em.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_with_state,
    DarkSirenPriorState,
    DarkSirenEnsemblePriorState,
)

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
# b_miss = 0 -> the legacy local-overdensity factor is identically 1, so the
# missing branch is driven purely by Q_LSS / the marks (matches the smoke cases).
SURVEY = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)


def _prior_catalog(**extra):
    """Two-row dark-siren catalog with a built KDE cache (as ``make_likelihood``
    produces) plus optional Q-table / mark fields, ``unique_pixels=None`` so the
    rows are already compact (``K == n_rows``), exactly the real jit path."""
    rows = [np.array([0.10, 0.12, 0.15]), np.array([0.30, 0.34])]
    n_rows, nmax = 2, 3
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    for i, r in enumerate(rows):
        zg[i, : len(r)] = r
        dz[i, : len(r)] = 0.003
        w[i, : len(r)] = 1.0
        ng[i] = len(r)
    zg, dz, w, ng = (jnp.asarray(a) for a in (zg, dz, w, ng))
    kde, idx = build_pixel_kde_cache(np.arange(n_rows, dtype=np.int32), zg, n_rows, ngals=ng)
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=kde, pixel_to_cache_idx=idx,
        unique_pixels=None, **extra,
    )


def _jit_prior_logp(cat, *, mark_model="none", eta=None, mark_names=()):
    """Build + evaluate the dark-siren redshift prior for row 0 **under jit** —
    the same prepare->eval sequence ``darksiren_log_likelihood`` runs.  Returns
    ``p(z|pix=0)`` on ``zgrid`` (raises ConcretizationTypeError on pre-fix code)."""
    pix = jnp.zeros(NG, jnp.int32)  # all-row-0 lookup

    @jax.jit
    def _run(cosmo, survey, em_catalog, mark_params):
        state = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, em_catalog,
            mark_model=mark_model, mark_params=mark_params, mark_names=mark_names,
        )
        return state, eval_redshift_prior_with_state(
            "dark_sirens", state, zgrid, pix, cosmo, survey, em_catalog
        )

    mark_params = None if eta is None else jnp.asarray(eta, dtype=float)
    state, lp = _run(COSMO, SURVEY, cat, mark_params)
    return state, np.asarray(lp)


def _assert_finite_normalized(lp):
    assert lp.shape == (NG,)
    assert np.all(np.isfinite(lp)), "log p(z|pix) has non-finite entries"
    integ = np.trapezoid(np.exp(lp), np.asarray(zgrid))
    assert abs(integ - 1.0) < 5e-3, f"p(z|pix) not normalised: ∫={integ}"


# ------------------------------------------------------------
# LSS completion (radial & gp3d both feed a deterministic, compact Q table)
# ------------------------------------------------------------

def test_lss_deterministic_table_under_jit():
    """`--lss_completion` with a compact deterministic logQ table (the form both
    the radial AND gp3d builders emit — gp3d stores the Laplace posterior-MEAN Q)
    must build the prior under jit without ConcretizationTypeError."""
    # Spatially varying logQ (mimics real clustering); shape (n_rows, NG) == compact.
    logq = jnp.asarray(0.3 * np.sin(np.linspace(0.0, 3.0, NG))[None, :] * np.ones((2, 1)))
    cat = _prior_catalog(lss_completion_logq=logq)  # lss_completion_indexing defaults to 0
    state, lp = _jit_prior_logp(cat)
    assert isinstance(state, DarkSirenPriorState)
    _assert_finite_normalized(lp)


def test_lss_ensemble_members_under_jit():
    """An LSS ensemble (members) table must also resolve under jit — exercises the
    `q_members_row` transpose + posterior-mean branch of the resolver."""
    lm = jnp.asarray(np.stack([np.full((2, NG), o) for o in (-0.2, 0.0, 0.2)]))  # (3,2,NG)
    cat = _prior_catalog(lss_completion_logq_members=lm)
    state, lp = _jit_prior_logp(cat)
    assert isinstance(state, DarkSirenEnsemblePriorState)
    _assert_finite_normalized(lp)


# ------------------------------------------------------------
# Marked-host model
# ------------------------------------------------------------

def test_marks_loglinear_under_jit():
    """`--mark_model loglinear` (z-binned `mu_miss(z|eta)`) must build the prior
    under jit; pre-fix this raised at `float(zgrid[-1])` in `_mu_miss_grid`."""
    mstar = np.array([[0.6, -0.4, 0.0], [0.1, 0.0, 0.0]])
    ssfr = np.array([[-0.3, 0.5, 0.0], [0.2, 0.0, 0.0]])
    cat = _prior_catalog(mark_logmstar=jnp.asarray(mstar), mark_logssfr=jnp.asarray(ssfr))
    state, lp = _jit_prior_logp(
        cat, mark_model="loglinear", eta=[1.5, -0.8], mark_names=("logmstar", "logssfr"),
    )
    assert isinstance(state, DarkSirenPriorState)
    _assert_finite_normalized(lp)


def test_marks_and_lss_compose_under_jit():
    """Marks + an LSS Q table together (the missing branch carries both) must
    still trace — guards the composed path `dN_miss = curves.dN_miss * mu_miss`."""
    logq = jnp.asarray(np.full((2, NG), np.log(1.3)))
    mstar = np.array([[0.6, -0.4, 0.0], [0.1, 0.0, 0.0]])
    cat = _prior_catalog(lss_completion_logq=logq, mark_logmstar=jnp.asarray(mstar))
    state, lp = _jit_prior_logp(
        cat, mark_model="loglinear", eta=[1.0], mark_names=("logmstar",),
    )
    assert isinstance(state, DarkSirenPriorState)
    _assert_finite_normalized(lp)
