"""
test_flow_event_term.py
-----------------------
The flow-surrogate likelihood body (likelihood/flow_events.py).

The load-bearing check is the CROSS-ESTIMATOR test: mock PE samples drawn
FROM the toy flows, weighted by the analytic PE prior, are fed to the
battle-tested stored-sample likelihood (core.darksiren_log_likelihood); the
flow path must reproduce its total log-likelihood up to a hyperparameter-
independent constant (per-event PE-prior/evidence normalisations) within the
joint Monte-Carlo error.  This pins the basis conventions, Jacobians, the
population samplers, and the ensemble evaluation against the existing
estimator in one shot.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

flows_mod = pytest.importorskip("darksirens.gw.flows")

import equinox as eqx  # noqa: E402
import jax.tree_util as jtu  # noqa: E402

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.gw.populations.sampling import (
    make_mass_q_edges,
    resolve_mass_grid_bounds,
)
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.likelihood.flow_events import (
    PePriorTables,
    build_chi_eff_prior_table,
    build_flow_loglike,
    build_pe_dl_prior_table,
    chi_eff_prior_logpdf,
)
from darksirens.utils.cosmology import H0Planck, Om0Planck


# ── fixtures ────────────────────────────────────────────────────────────────


def _flow_config(key):
    return {
        "base_dist": "Normal",
        "data_dim": 4,
        "type": "spline",
        "flow_layers": 2,
        "knots": 4,
        "key": key,
        "columns": list(flows_mod.SPECTRAL_COLUMNS),
        # Center the untrained flow on a plausible "event": m1det-m2det ~ e^2,
        # m2det ~ e^3.4, dL ~ e^7 Mpc, chi_eff near 0.  Tight scales keep the
        # fake posterior well inside the powerlaw+peak population support.
        "Z_mean": [2.0, 3.4, 7.0, 0.0],
        "Z_std": [0.3, 0.1, 0.15, 0.15],
        "constraints": {
            "0": {"type": "ordered_positive", "dims": [0, 1]},
            "1": None,
            "2": {"type": "positive"},
            "3": {"type": "interval", "low": -1, "high": 1},
        },
    }


def _save_flow(path, flow, config):
    arrays, _ = eqx.partition(flow, eqx.is_array)
    leaves, _ = jtu.tree_flatten(arrays)
    np.savez(path, *[np.asarray(l) for l in leaves], config_json=json.dumps(config))


def _toy_catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _make_gw_sel(n_sel=40_000, seed=7):
    """Injections from known analytic draws, pdraw in the (m1det, q, dL) basis."""
    rng = np.random.default_rng(seed)
    m1det = np.exp(rng.uniform(np.log(5.0), np.log(150.0), n_sel))
    q = rng.uniform(0.2, 1.0, n_sel)
    dL = rng.uniform(100.0, 8000.0, n_sel)
    chieff = rng.uniform(-0.5, 0.5, n_sel)
    p_m1 = 1.0 / (m1det * np.log(150.0 / 5.0))
    p_draw = p_m1 * (1.0 / 0.8) * (1.0 / 7900.0) * (1.0 / 1.0)
    m2det = q * m1det
    return GWEvent(
        m1det=jnp.asarray(m1det),
        m2det=jnp.asarray(m2det),
        dL=jnp.asarray(dL),
        chieff=jnp.asarray(chieff),
        prior_wt=jnp.asarray(p_draw),
        pixels=jnp.zeros(n_sel, dtype=jnp.int32),
        q=jnp.asarray(q),
        valid=jnp.ones(n_sel, dtype=bool),
    )


PE_H0, PE_OM0 = 67.74, 0.3089


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("event_flows")
    for name, key in [("GW_TOY_X", 3), ("GW_TOY_Y", 11)]:
        d = root / name
        d.mkdir()
        cfg = _flow_config(key)
        _save_flow(
            d / f"{name}_flow.npz", flows_mod.create_flow_from_config(cfg), cfg
        )
    ens = flows_mod.load_flow_ensemble(root)

    log_dl, log_p_dl = build_pe_dl_prior_table(PE_H0, PE_OM0)
    chi_qg, chi_cg, chi_tab = build_chi_eff_prior_table(0.99)
    pe_tables = PePriorTables(
        log_dl_grid=jnp.asarray(log_dl),
        log_p_dl=jnp.asarray(log_p_dl),
        chi_q_grid=jnp.asarray(chi_qg),
        chi_grid=jnp.asarray(chi_cg),
        chi_table=jnp.asarray(chi_tab),
    )

    model = get_model("powerlaw+peak")
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges, q_edges = make_mass_q_edges(m1_lo, m1_hi, n_m1=256, n_q=128)

    # Population-side sampling has ESS ~ 0.1% of J for a narrow event
    # posterior (broad population target); J is large here so the flow-side
    # MC error (~1/sqrt(ESS) ~ 0.06 per event) stays inside the tolerance
    # of the cross-estimator check below.
    J = 262_144
    u_base = jax.random.uniform(jax.random.PRNGKey(0), (J, 4), dtype=jnp.float64)

    gw_sel = _make_gw_sel()
    catalog = _toy_catalog()
    Ndraw = float(gw_sel.dL.shape[0]) * 2.0

    ll_flow = build_flow_loglike(
        model=model,
        eval_logflows=flows_mod.make_ensemble_log_prob(ens),
        group_params=ens.group_params(),
        u_base=u_base,
        m1_edges=m1_edges,
        q_edges=q_edges,
        pe_tables=pe_tables,
        gw_sel=gw_sel,
        em_catalog_sel=catalog,
        Ndraw=Ndraw,
        nEvents=ens.n_flows,
    )
    theta_fid = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    return dict(
        ens=ens,
        pe_tables=pe_tables,
        ll_flow=ll_flow,
        gw_sel=gw_sel,
        catalog=catalog,
        Ndraw=Ndraw,
        theta_fid=theta_fid,
    )


def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(
        n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
    )


def _theta_variants(theta_fid):
    """Fiducial plus moderate hyperparameter excursions (alpha, mu_G, gamma)."""
    base = np.asarray(theta_fid, dtype=np.float64)
    variants = [base]
    for d_alpha, d_mu, d_gamma in [(0.7, 0.0, 0.0), (-0.7, 3.0, 0.0),
                                   (0.0, -4.0, 1.5), (0.4, 2.0, -1.5)]:
        t = base.copy()
        t[1] += d_alpha   # alpha_PL
        t[6] += d_mu      # mu_G
        t[-1] += d_gamma  # gamma
        variants.append(t)
    return [jnp.asarray(t) for t in variants]


# ── tests ───────────────────────────────────────────────────────────────────


def test_deterministic_and_finite(setup):
    ll = setup["ll_flow"]
    a = ll(_cosmo(), _survey(), setup["theta_fid"])
    b = ll(_cosmo(), _survey(), setup["theta_fid"])
    assert np.isfinite(float(a))
    assert float(a) == float(b)


def test_cross_estimator_agreement_up_to_constant(setup):
    """Flow path vs stored-PE path on identical information.

    Mock PE samples are drawn FROM the flows, so both estimators target the
    same per-event integrals; their total logL curves over hyperparameters
    must agree up to a Lambda-independent constant.  The selection term is
    bit-identical between the paths, so it drops out of the difference.
    """
    ens = setup["ens"]
    pe_tables = setup["pe_tables"]
    nsamp = 8192
    samples = np.asarray(flows_mod.ensemble_sample(ens, jax.random.key(42), nsamp))
    m1det = samples[..., 0].reshape(-1)
    m2det = samples[..., 1].reshape(-1)
    dL = samples[..., 2].reshape(-1)
    chieff = samples[..., 3].reshape(-1)
    q = m2det / m1det

    # Analytic PE prior at the samples, in the loader's (m1det, q, dL) basis:
    # p_pe = m1det * p_dL_usf(dL) * p_chi(chieff | q), per-event normalised.
    log_p_dl = np.interp(
        np.log(dL), np.asarray(pe_tables.log_dl_grid), np.asarray(pe_tables.log_p_dl)
    )
    log_p_chi = np.asarray(
        chi_eff_prior_logpdf(
            jnp.asarray(1.0 / (1.0 + q)),
            jnp.asarray(chieff),
            pe_tables.chi_q_grid,
            pe_tables.chi_grid,
            pe_tables.chi_table,
        )
    )
    p_pe = np.exp(np.log(m1det) + log_p_dl + log_p_chi)
    p_pe = p_pe.reshape(ens.n_flows, nsamp)
    p_pe = p_pe / p_pe.sum(axis=1, keepdims=True)
    p_pe = p_pe.reshape(-1)

    gw_pe = GWEvent(
        m1det=jnp.asarray(m1det),
        m2det=jnp.asarray(m2det),
        dL=jnp.asarray(dL),
        chieff=jnp.asarray(chieff),
        prior_wt=jnp.asarray(p_pe),
        pixels=jnp.zeros(m1det.size, dtype=jnp.int32),
        q=jnp.asarray(q),
        valid=jnp.ones(m1det.size, dtype=bool),
    )

    diffs = []
    for theta in _theta_variants(setup["theta_fid"]):
        ll_f = float(setup["ll_flow"](_cosmo(), _survey(), theta))
        ll_p = float(
            darksiren_log_likelihood(
                _cosmo(),
                _survey(),
                theta,
                gw_pe,
                setup["catalog"],
                setup["gw_sel"],
                setup["catalog"],
                ens.n_flows,
                nsamp,
                setup["Ndraw"],
                "powerlaw+peak",
                "spectral_sirens",
            )
        )
        assert np.isfinite(ll_f) and np.isfinite(ll_p), (ll_f, ll_p)
        diffs.append(ll_f - ll_p)

    diffs = np.asarray(diffs)
    spread = np.abs(diffs - diffs.mean()).max()
    assert spread < 0.25, (
        f"flow vs stored-PE logL curves diverge beyond MC error: diffs={diffs}"
    )


def test_extreme_theta_never_nan(setup):
    theta = np.array(setup["theta_fid"], dtype=np.float64)
    theta[3] = 50.0   # m_max at its lower prior bound
    theta[6] = 20.0   # mu_G at its lower bound
    theta[-1] = 8.0   # extreme gamma
    val = float(setup["ll_flow"](_cosmo(), _survey(), jnp.asarray(theta)))
    assert not np.isnan(val)


def test_gradient_wrt_H0_finite(setup):
    # Gradient-based samplers need the redshift-prior optimization_barrier
    # off (it has no differentiation rule) — same policy as the stored-PE
    # factory's materialization resolver.
    ens = setup["ens"]
    ll = build_flow_loglike(
        model=get_model("powerlaw+peak"),
        eval_logflows=flows_mod.make_ensemble_log_prob(ens),
        group_params=ens.group_params(),
        u_base=jax.random.uniform(jax.random.PRNGKey(0), (8192, 4), dtype=jnp.float64),
        m1_edges=make_mass_q_edges(*resolve_mass_grid_bounds(get_model("powerlaw+peak")), n_m1=128, n_q=64)[0],
        q_edges=make_mass_q_edges(*resolve_mass_grid_bounds(get_model("powerlaw+peak")), n_m1=128, n_q=64)[1],
        pe_tables=setup["pe_tables"],
        gw_sel=setup["gw_sel"],
        em_catalog_sel=setup["catalog"],
        Ndraw=setup["Ndraw"],
        nEvents=ens.n_flows,
        materialize_redshift_prior_state=False,
    )
    theta = setup["theta_fid"]

    def f(h0):
        return ll(CosmoParams(H0=h0, Om0=Om0Planck), _survey(), theta)

    g = float(jax.grad(f)(jnp.asarray(70.0)))
    assert np.isfinite(g)
