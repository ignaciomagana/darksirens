"""
test_lss_marginalization.py
---------------------------
Tests for the LSS-completion ENSEMBLE marginalisation of the GW likelihood:

    logL = logsumexp_m logL(Λ; Q_m) − log M,

treating the M lognormal-completion members as Monte-Carlo draws of the
missing-galaxy field (the documented fully-Bayesian next step beyond the
deterministic posterior-mean Q).  Opt-in via ``lss_marginalize=True``.

The decisive check is self-consistency at the LIKELIHOOD level: the marginalised
value must equal the log-mean-exp of the per-member likelihoods, where each
per-member value is obtained by running the *deterministic* likelihood with that
member's Q table.  We also check the single-member reduction, that the default
(non-marginalised) path is unaffected by the presence of members, and that
requesting marginalisation without an ensemble fails loudly.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.gw.populations import get_fixed_population_params
from darksirens.likelihood.core import darksiren_log_likelihood

NG = int(zgrid.size)
_Z = np.asarray(zgrid)
_LOGQ_CLIP = 7.0
COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
# b_miss = 0 so the legacy local-overdensity factor is 1 and the missing branch
# is driven purely by Q_LSS — exactly what the members vary.  A low z50 (and
# large n0 below) keeps the catalog INcomplete over the event redshift range so
# the missing branch — hence Q — actually moves the likelihood.
SURVEY = SurveyParams(n0=1.0, z50=0.15, w=0.08, delta=0.0, b_miss=0.0, alpha_miss=1.0)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))

# z-TILTED members: a constant Q cancels in the normalised redshift prior, so the
# members carry distinct *slopes* logQ_m(z) = slope_m (z - 0.2) that reshape the
# missing-galaxy redshift distribution differently per member.
_SLOPES = np.array([-5.0, -1.5, 1.5, 5.0])
_M = len(_SLOPES)


def _member_logq(m):
    """(NG,) tilted log Q for member ``m`` over the package zgrid."""
    return _SLOPES[m] * (_Z - 0.2)


def _members_table():
    """(M, n_rows=2, NG) ensemble of member logQ tables."""
    return np.stack([np.broadcast_to(_member_logq(m), (2, NG)) for m in range(_M)])


def _dark_catalog(*, logq=None, logq_members=None):
    """Two-row dark-siren catalog (KDE cache built, compact rows) carrying either
    a single deterministic Q table or an ensemble of member tables."""
    rows = [np.array([0.10, 0.12, 0.15]), np.array([0.28, 0.32])]
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
        delta_g_pix_z=jnp.zeros((n_rows, NG)), dN_obs_kde=kde, pixel_to_cache_idx=idx,
        unique_pixels=None,
        lss_completion_logq=(None if logq is None else jnp.asarray(logq)),
        lss_completion_logq_members=(None if logq_members is None else jnp.asarray(logq_members)),
    )


def _gw(n_events, n_samp, seed, n_rows=2):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(420.0, 1500.0, total))   # z ~ 0.09–0.30
    chieff = jnp.asarray(rng.uniform(-0.2, 0.2, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.asarray(rng.integers(0, n_rows, total), dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=m2det / m1det, valid=valid)


_N_EV, _N_SAMP, _N_SEL = 4, 64, 300
_GW_PE = _gw(_N_EV, _N_SAMP, seed=0)
_GW_SEL = _gw(_N_SEL, 1, seed=10)


def _ll(cat, *, marginalize=False):
    return darksiren_log_likelihood(
        COSMO, SURVEY, POP, _GW_PE, cat, _GW_SEL, cat,
        _N_EV, _N_SAMP, float(_N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=marginalize,
    )


def test_marginalize_equals_logmeanexp_of_members():
    """logL_marg == logsumexp_m logL(Q_m) − log M (self-consistency at the
    likelihood level)."""
    ll_marg = float(_ll(_dark_catalog(logq_members=_members_table()), marginalize=True))

    per_member = jnp.asarray([
        _ll(_dark_catalog(logq=np.broadcast_to(_member_logq(m), (2, NG))))
        for m in range(_M)
    ])
    expected = float(logsumexp(per_member) - jnp.log(_M))

    assert np.isfinite(ll_marg)
    assert ll_marg == pytest.approx(expected, abs=1e-6), (
        f"marginalised {ll_marg} != logmeanexp of members {expected}"
    )
    # The members genuinely differ (otherwise the test is vacuous).
    assert float(per_member.max() - per_member.min()) > 1e-3


def test_single_member_reduces_to_deterministic():
    """One member -> marginalisation is exactly that member's deterministic ll."""
    one = np.broadcast_to(_member_logq(2), (1, 2, NG))
    ll_marg = float(_ll(_dark_catalog(logq_members=one), marginalize=True))
    ll_det = float(_ll(_dark_catalog(logq=np.broadcast_to(_member_logq(2), (2, NG)))))
    assert ll_marg == pytest.approx(ll_det, abs=1e-6)


def test_default_path_ignores_members():
    """Without lss_marginalize, an ensemble catalog gives the SAME ll as a
    deterministic catalog carrying the posterior-mean Q (members are inert)."""
    ll_ens = float(_ll(_dark_catalog(logq_members=_members_table()), marginalize=False))
    # The scalar-compat field is the posterior-MEAN Q = mean_m exp(clip(logQ_m(z))).
    q_mean = np.mean(
        [np.exp(np.clip(_member_logq(m), -_LOGQ_CLIP, _LOGQ_CLIP)) for m in range(_M)],
        axis=0,
    )  # (NG,)
    logq_mean = np.broadcast_to(np.log(q_mean), (2, NG))
    ll_mean = float(_ll(_dark_catalog(logq=logq_mean)))
    assert np.isfinite(ll_ens)
    assert ll_ens == pytest.approx(ll_mean, abs=1e-6)


def test_marginalize_without_ensemble_raises():
    """Requesting marginalisation on a catalog without members fails loudly."""
    with pytest.raises(ValueError, match="ENSEMBLE|members"):
        _ll(_dark_catalog(logq=np.zeros((2, NG))), marginalize=True)
