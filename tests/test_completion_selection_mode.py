"""c_mode="selection": grids, gating, provenance, and the H0 firewall.

The parametric completeness rides the aggregate machinery's C_bar_raw slot,
so these tests pin (a) the curve that lands there and its EXACT H0-invariance
through the full SurveyParams path, (b) the sampled-parameter gating
(M0hat/sigma_M active iff c_mode=="selection", in every universe-model
branch), (c) the traced-int hard error for the new code, and (d) the
builder-side pairing of the mode with an offline fit payload.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.core.types import (  # noqa: E402
    C_MODE_AGGREGATE_STRUCT,
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.inference.prior import build_parameter_space  # noqa: E402
from darksirens.redshift.completion import (  # noqa: E402
    _is_aggregate_c_mode,
    _is_selection_c_mode,
    _precompute_grids,
    completion_curves,
)
from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import c_sel_gaussian  # noqa: E402

THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)


def _em(n_pix=12, max_gals=3, seed=5):
    rng = np.random.default_rng(seed)
    zg = np.full((n_pix, max_gals), 100.0)
    dz = np.ones((n_pix, max_gals))
    w = np.zeros((n_pix, max_gals))
    ng = np.zeros(n_pix, dtype=int)
    for p in range(n_pix):
        n = int(rng.integers(0, max_gals))
        zs = np.sort(rng.uniform(0.02, 0.4, n))
        zg[p, :n] = zs
        dz[p, :n] = 1e-3
        w[p, :n] = 1.0
        ng[p] = n
    import healpy as hp
    apix = hp.nside2pixarea(1)
    return EMCatalog(
        apix=float(apix), zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
        pixel_to_cache_idx=None)


def _survey(**kw):
    base = dict(n0=1e-4, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=0.0)
    base.update(kw)
    return SurveyParams(**base)


def _cosmo(h0=67.74):
    return CosmoParams(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


def test_selection_grids_carry_c_sel_exactly():
    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT, **THETA)
    grids = _precompute_grids(_cosmo(), sv, _em())
    ref = c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                         THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0)
    np.testing.assert_allclose(np.asarray(grids.C_bar_raw), np.asarray(ref),
                               rtol=0, atol=1e-14)


def test_selection_c_bar_exactly_h0_invariant_through_survey_path():
    """The M*-H0 firewall, end to end through _precompute_grids."""
    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT, **THETA)
    em = _em()
    a = _precompute_grids(_cosmo(50.0), sv, em).C_bar_raw
    b = _precompute_grids(_cosmo(95.0), sv, em).C_bar_raw
    np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                               rtol=0, atol=1e-12)


def test_selection_mode_via_eager_int_and_sentinel_agree():
    em = _em()
    a = _precompute_grids(_cosmo(), _survey(c_mode=2, **THETA), em).C_bar_raw
    b = _precompute_grids(
        _cosmo(), _survey(c_mode=C_MODE_SELECTION_STRUCT, **THETA), em
    ).C_bar_raw
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_mode_decoders_are_mutually_exclusive():
    assert _is_selection_c_mode(C_MODE_SELECTION_STRUCT)
    assert not _is_selection_c_mode(C_MODE_AGGREGATE_STRUCT)
    assert not _is_selection_c_mode(None)
    assert not _is_aggregate_c_mode(C_MODE_SELECTION_STRUCT)
    assert _is_selection_c_mode(2) and not _is_selection_c_mode(1)


def test_traced_int_c_mode_hard_errors_for_selection_too():
    em = _em()

    def f(sv):
        return completion_curves(_cosmo(), sv, em).dN_miss

    with pytest.raises((ValueError, jax.errors.JaxRuntimeError),
                       match="c_mode"):
        jax.jit(f)(_survey(c_mode=2, **THETA))


def test_theta_moves_the_missing_budget():
    """The selection parameters really steer dN_miss (hence, through the
    shared pinned prior state, the selection integral)."""
    em = _em()
    base = dict(c_mode=C_MODE_SELECTION_STRUCT, **THETA)
    c0 = completion_curves(_cosmo(), _survey(**base), em)
    deeper = dict(base, M0hat=THETA["M0hat"] - 1.0)   # brighter LF: more complete
    c1 = completion_curves(_cosmo(), _survey(**deeper), em)
    n0 = float(np.sum(np.asarray(c0.N_miss)))
    n1 = float(np.sum(np.asarray(c1.N_miss)))
    assert n1 < n0 * 0.9, (n0, n1)


@pytest.mark.parametrize("universe_model", ["dark_sirens", None])
def test_selection_labels_sample_iff_selection_mode(universe_model):
    kw = dict(universe_model=universe_model, use_lss=False)
    on = build_parameter_space("powerlaw+peak", True, True, False,
                               c_mode="selection", **kw)[0]
    off = build_parameter_space("powerlaw+peak", True, True, False, **kw)[0]
    assert {"M0hat", "sigma_M"} <= set(on)
    assert not ({"M0hat", "sigma_M"} & set(off))
    # m_lim is a pinned datum: never sampled in either mode.
    assert "m_lim" not in on and "m_lim" not in off


def test_fixed_selection_param_under_counts_mode_is_fatal():
    with pytest.raises(ValueError, match="c_mode='per_pixel'"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              universe_model="dark_sirens", use_lss=False,
                              fixed_parameter_values={"sigma_M": 0.9})


def test_selection_prior_flips_kinds_and_fails_closed():
    kw = dict(universe_model="dark_sirens", use_lss=False, c_mode="selection")
    res = build_parameter_space(
        "powerlaw+peak", True, True, False,
        selection_prior={"M0hat": (-20.19, 0.02), "sigma_M": (1.0, 0.015)},
        **kw)
    kinds = dict(zip(res[0], res[11]))
    assert kinds["M0hat"] == ("normal", -20.19, 0.02)
    assert kinds["sigma_M"] == ("normal", 1.0, 0.015)
    # a prior for a non-selection label is a config error
    with pytest.raises(ValueError, match="selection_prior"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              selection_prior={"log10n0": (-5.0, 0.1)}, **kw)
    # a prior under a counts mode (labels not sampled) is a config error
    with pytest.raises(ValueError, match="not sampled"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              universe_model="dark_sirens", use_lss=False,
                              selection_prior={"M0hat": (-20.19, 0.02)})


@pytest.mark.parametrize("universe_model", ["dark_sirens_complete",
                                            "spectral_sirens",
                                            "bright_sirens"])
def test_selection_labels_inert_for_complete_and_catalog_free(universe_model):
    labels = build_parameter_space("powerlaw+peak", True, True, False,
                                   universe_model=universe_model,
                                   use_lss=False, c_mode="selection")[0]
    assert not ({"M0hat", "sigma_M"} & set(labels))


def test_selection_labels_absent_under_fix_survey():
    labels = build_parameter_space("powerlaw+peak", True, True, True,
                                   universe_model="dark_sirens",
                                   use_lss=False, c_mode="selection")[0]
    assert not ({"M0hat", "sigma_M"} & set(labels))


def test_selection_mode_k2_requires_a_fit_per_catalog():
    """K>=2 selection is supported once EVERY catalog carries its own
    anchored fit; a partially anchored mixture leaves catalog k's theta
    flat and stays fatal."""
    kw = dict(universe_model="dark_sirens", use_lss=False, n_catalogs=2,
              c_mode="selection")
    labels = build_parameter_space("powerlaw+peak", True, True, False, **kw)[0]
    assert {"M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2"} <= set(labels)
    with pytest.raises(ValueError, match="M0hat_c2"):
        build_parameter_space(
            "powerlaw+peak", True, True, False,
            selection_prior={"M0hat": (-20.2, 0.02), "sigma_M": (1.0, 0.02)},
            **kw)


def test_builder_requires_and_rejects_fit_pairing():
    from darksirens.cli.build_lognormal_completion import _require_selection_fit
    with pytest.raises(ValueError, match="needs the offline magnitude fit"):
        _require_selection_fit("selection", None)
    with pytest.raises(ValueError, match="c_mode='aggregate'"):
        _require_selection_fit("aggregate", {"m_lim": 24.0})
    _require_selection_fit("selection", {"m_lim": 24.0})   # legal
    _require_selection_fit("per_pixel", None)              # legal


# ===========================================================================
# The selection budget carries no data anchor (review F-081)
# ===========================================================================

def test_selection_budget_audit_tracks_n0_and_rejects_counts_modes():
    """Under c_mode='selection' the missing budget is proportional to n0 with
    nothing in the likelihood calibrating it, so the audit's model-vs-observed
    ratio must move linearly with n0 -- that IS the unanchored amplitude."""
    from darksirens.redshift.completion import selection_budget_audit

    em = _em()
    ng = np.asarray(em.ngals)
    kw = dict(N_obs_total=float(ng.sum()), n_occupied=int((ng > 0).sum()),
              n_pix_total=int(ng.size))
    sv = _survey(c_mode=C_MODE_SELECTION_STRUCT, **THETA)
    a = selection_budget_audit(_cosmo(), sv, em, **kw)
    b = selection_budget_audit(
        _cosmo(), sv._replace(n0=10.0 * sv.n0), em, **kw)
    assert b["selection_model_over_observed_footprint"] == pytest.approx(
        10.0 * a["selection_model_over_observed_footprint"], rel=1e-10)
    assert a["N_obs_total"] == float(ng.sum())
    assert 0.0 < a["implied_completeness"]
    # H0^-3 scaling of the budget amplitude (r ~ 1/H0, dV_c/dz ~ r^2/(H0 E)):
    c = selection_budget_audit(_cosmo(2 * 67.74), sv, em, **kw)
    assert (c["selection_model_N_obs_sky"]
            < 0.2 * a["selection_model_N_obs_sky"])
    # The counts-based estimators anchor the budget themselves -> not applicable.
    with pytest.raises(ValueError, match="c_mode='selection'"):
        selection_budget_audit(_cosmo(), _survey(), em, **kw)
