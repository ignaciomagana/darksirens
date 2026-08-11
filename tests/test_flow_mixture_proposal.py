"""
test_flow_mixture_proposal.py
-----------------------------
The defensive mixture proposal of the flow-surrogate event term (issue #260).

Before #260 every draw came from a window measured from a few thousand flow
samples, so the learned flow's tails were dropped by a HYPERPARAMETER-DEPENDENT
amount.  The fix draws a fraction ``w_full`` of each event's points from the
full population support and weights every draw by the exact mixture density
``q = (1-w) q_win + w q_full`` — both components evaluated at BOTH kinds of
draw.  The tests here cover, in order:

1. the density evaluators are exactly the samplers' own densities;
2. the mixture integrates a known integrand and its weights sum to N, while
   the windowed-only estimator misses a known amount of mass, and the
   "score each draw only under its own component" shortcut is measurably
   wrong (the mistake naive MIS implementations make);
3. on the real likelihood body, the mixture recovers the tail the window
   clips, checked against an INDEPENDENT reference (importance sampling from
   the flow itself, with the analytic change-of-variables Jacobian);
4. the PE-support tightening: draws where the chi_eff PE prior is exactly
   zero are dropped from the integral instead of being handed e^50 of weight
   by gwcat's -50 log floor;
5. the operands reach the jitted body as ARGUMENTS, not as HLO constants.
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
    build_column_cdf_tables,
    histogram_trunc_logpdf,
    m1_given_q_trunc_logpdf,
    make_mass_q_edges,
    resolve_mass_grid_bounds,
    sample_histogram_trunc,
    sample_m1_given_q_trunc,
    truncnorm_logpdf,
    truncnorm_sample,
)
from darksirens.likelihood.flow_events import (
    DEFAULT_W_FULL,
    PePriorTables,
    build_chi_eff_prior_table,
    build_flow_loglike,
    build_pe_dl_prior_table,
    chi_eff_prior_prob,
    resolve_full_draws,
)
from darksirens.redshift.grid import zgrid
from darksirens.redshift.prior import prepare_redshift_prior_state
from darksirens.utils.cosmology import H0Planck, Om0Planck, dL_of_z, z_of_dL

PE_H0, PE_OM0 = 67.74, 0.3089
C_KM_S = 299792.458


# ── 1. the density evaluators are the samplers' own densities ───────────────


def test_histogram_trunc_logpdf_matches_sampler():
    rng = np.random.default_rng(0)
    edges = jnp.asarray(np.sort(rng.uniform(0.0, 10.0, 41)))
    dens = jnp.asarray(np.exp(rng.normal(size=40)))
    u = jnp.asarray(rng.uniform(size=5000))
    for lo, hi in [(0.0, 10.0), (2.0, 6.0), (-3.0, 4.5), (7.0, 30.0)]:
        s = sample_histogram_trunc(u, edges, jnp.log(dens), lo, hi)
        got = histogram_trunc_logpdf(s.x, edges, jnp.log(dens), lo, hi)
        np.testing.assert_allclose(np.asarray(got), np.asarray(s.log_s),
                                   rtol=0, atol=1e-9)
        # And it normalises: a Riemann sum of exp(logpdf) over the window is 1.
        a = max(lo, float(edges[0]))
        b = min(hi, float(edges[-1]))
        n, h = 200_000, (b - a) / 200_000
        xs = jnp.asarray(a + h * (np.arange(n) + 0.5))
        p = jnp.exp(histogram_trunc_logpdf(xs, edges, jnp.log(dens), lo, hi))
        assert abs(float(jnp.sum(p)) * h - 1.0) < 2e-3


def test_histogram_trunc_logpdf_is_minus_inf_outside_window():
    edges = jnp.linspace(0.0, 10.0, 51)
    dens = jnp.zeros(50)
    x = jnp.asarray([-1.0, 1.9999, 2.5, 6.0001, 12.0])
    got = np.asarray(histogram_trunc_logpdf(x, edges, dens, 2.0, 6.0))
    assert np.isneginf(got[[0, 1, 3, 4]]).all()
    assert np.isfinite(got[2])


def test_truncnorm_logpdf_matches_sampler():
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.uniform(size=4000))
    for mu, sig, lo, hi in [(0.0, 0.3, -1.0, 1.0), (0.5, 0.05, -0.2, 0.4),
                            (0.0, 0.02, 0.7, 0.9)]:
        s = truncnorm_sample(u, mu, sig, lo, hi)
        got = truncnorm_logpdf(s.x, mu, sig, lo, hi)
        np.testing.assert_allclose(np.asarray(got), np.asarray(s.log_s),
                                   rtol=0, atol=1e-9)
    assert np.isneginf(float(truncnorm_logpdf(jnp.asarray(2.0), 0.0, 0.3,
                                              -1.0, 1.0)))


def test_m1_given_q_logpdf_matches_sampler():
    rng = np.random.default_rng(2)
    m1_edges = jnp.geomspace(2.0, 200.0, 65)
    log_t = jnp.asarray(rng.normal(size=(64, 8)))
    tab = build_column_cdf_tables(m1_edges, log_t)
    n = 512
    u = jnp.asarray(rng.uniform(size=n))
    cells = jnp.asarray(rng.integers(0, 8, n))
    lo = jnp.asarray(rng.uniform(2.0, 40.0, n))
    hi = lo + jnp.asarray(rng.uniform(1.0, 120.0, n))
    x, log_s = sample_m1_given_q_trunc(u, m1_edges, log_t, cells, lo, hi,
                                       tables=tab)
    got = m1_given_q_trunc_logpdf(x, m1_edges, log_t, cells, lo, hi, tables=tab)
    np.testing.assert_allclose(np.asarray(got), np.asarray(log_s),
                               rtol=0, atol=1e-9)
    # The precomputed tables must be a pure optimisation.
    x2, log_s2 = sample_m1_given_q_trunc(u, m1_edges, log_t, cells, lo, hi)
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(log_s2), np.asarray(log_s),
                               rtol=0, atol=0)


# ── 2. the mixture algebra, on an integrand whose truth is exact ────────────


def _analytic_mixture_draws(n_total, w_full, lo, hi, seed=0):
    """Deterministic-allocation mixture on a 1-D histogram proposal.

    Mirrors the estimator: ``n_full = round(w*n)`` draws from the untruncated
    grid density, the rest from the same density truncated to ``[lo, hi]``,
    and the mixture density evaluated at every draw under both components.
    """
    edges = jnp.linspace(0.0, 10.0, 401)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Broad, everywhere-positive proposal target.
    dens = jnp.exp(-0.5 * ((centers - 2.0) / 1.5) ** 2) + 0.05
    log_dens = jnp.log(dens)

    n_full = resolve_full_draws(n_total, w_full)
    n_win = n_total - n_full
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.uniform(size=n_total))

    x_win = sample_histogram_trunc(u[:n_win], edges, log_dens, lo, hi).x
    if n_full:
        x_full = sample_histogram_trunc(u[n_win:], edges, log_dens,
                                        edges[0], edges[-1]).x
        x = jnp.concatenate([x_win, x_full])
    else:
        x = x_win

    lq_win = histogram_trunc_logpdf(x, edges, log_dens, lo, hi)
    lq_full = histogram_trunc_logpdf(x, edges, log_dens, edges[0], edges[-1])
    w = n_full / n_total
    if n_full:
        log_q = jnp.logaddexp(np.log1p(-w) + lq_win, np.log(w) + lq_full)
    else:
        log_q = lq_win
    return x, log_q, lq_win, lq_full, w, n_win


def _integrand(x):
    """f(x): 90% of its mass inside [0, 4], 10% in a bump at x = 8."""
    a = 0.9 * np.exp(-0.5 * ((x - 2.0) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))
    b = 0.1 * np.exp(-0.5 * ((x - 8.0) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    return a + b


def _quad(f, a, b, n=2_000_000):
    """Midpoint rule — exact to ~1e-12 here, and free of the numpy
    ``trapz``/``trapezoid`` rename that splits this project's environments."""
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return float(f(x).sum() * h)


def _truth():
    """(total integral, integral inside the [0, 4] window) — both essentially exact."""
    return _quad(_integrand, 0.0, 10.0), _quad(_integrand, 0.0, 4.0)


def test_mixture_weights_sum_to_n_and_integrate_a_known_integrand():
    total, inside = _truth()
    x, log_q, _, _, w, _ = _analytic_mixture_draws(400_000, 0.1, 0.0, 4.0)
    f = _integrand(np.asarray(x))
    weights = f / np.exp(np.asarray(log_q))
    est = weights.mean()
    # Sum of weights ~ N: the integrand is a (nearly) normalised density, so
    # sum_j f(x_j)/q_mix(x_j) must come out at N * int f  ~=  N.
    assert abs(weights.sum() / len(weights) - total) < 0.02
    assert abs(est - total) < 0.02, (est, total)
    assert abs(total - 1.0) < 0.01           # the truth itself is ~1


def test_windowed_only_estimator_misses_the_clipped_mass():
    total, inside = _truth()
    # w_full = 0 is the pre-#260 estimator: nothing outside [0, 4] is sampled.
    x0, lq0, _, _, _, _ = _analytic_mixture_draws(400_000, 0.0, 0.0, 4.0)
    old = (_integrand(np.asarray(x0)) / np.exp(np.asarray(lq0))).mean()
    assert abs(old - inside) < 0.02, (old, inside)
    assert abs(old - total) > 0.05          # measurably biased low
    assert abs(inside / total - 0.9) < 0.02  # by the mass it clipped

    x1, lq1, _, _, _, _ = _analytic_mixture_draws(400_000, 0.1, 0.0, 4.0)
    mix = (_integrand(np.asarray(x1)) / np.exp(np.asarray(lq1))).mean()
    assert abs(mix - total) < abs(old - total) / 4


def test_scoring_each_draw_only_under_its_own_component_is_biased():
    """The classic MIS mistake: q_win must be evaluated at the FULL draws too."""
    total, _ = _truth()
    n = 400_000
    x, log_q, lq_win, lq_full, w, n_win = _analytic_mixture_draws(n, 0.1, 0.0, 4.0)
    f = _integrand(np.asarray(x))

    exact = (f / np.exp(np.asarray(log_q))).mean()
    assert abs(exact - total) < 0.02

    # "Forgot" that the windowed component also covers the full-component
    # draws: q_win treated as zero there.
    lq_win_naive = np.asarray(lq_win).copy()
    lq_win_naive[n_win:] = -np.inf
    log_q_naive = np.logaddexp(np.log1p(-w) + lq_win_naive,
                               np.log(w) + np.asarray(lq_full))
    naive = (f / np.exp(log_q_naive)).mean()
    assert abs(naive - total) > 10 * 0.02, (naive, total)


def test_mixture_density_integrates_to_one():
    """``(1-w) q_win + w q_full`` must itself be a normalised density.

    Quadrature over the whole domain, not a self-consistency identity: this
    is what catches a missing truncation normaliser in either component.
    """
    edges = jnp.linspace(0.0, 10.0, 401)
    centers = 0.5 * (edges[:-1] + edges[1:])
    log_dens = jnp.log(jnp.exp(-0.5 * ((centers - 2.0) / 1.5) ** 2) + 0.05)
    w, lo, hi = 0.1, 0.0, 4.0

    def q_mix(x):
        xs = jnp.asarray(x)
        return np.asarray(jnp.exp(jnp.logaddexp(
            np.log1p(-w) + histogram_trunc_logpdf(xs, edges, log_dens, lo, hi),
            np.log(w) + histogram_trunc_logpdf(xs, edges, log_dens,
                                               edges[0], edges[-1]),
        )))

    assert abs(_quad(q_mix, 0.0, 10.0, n=400_000) - 1.0) < 1e-4


def test_resolve_full_draws_allocation():
    assert resolve_full_draws(4096, 0.0) == 0
    assert resolve_full_draws(4096, 0.05) == 205
    assert resolve_full_draws(4096, 0.5) == 2048
    assert resolve_full_draws(10, 1e-6) == 1      # never rounds a request away
    assert resolve_full_draws(4, 0.99) == 3       # always leaves a window draw
    with pytest.raises(ValueError):
        resolve_full_draws(4096, 1.0)
    with pytest.raises(ValueError):
        resolve_full_draws(4096, -0.1)


def test_cli_default_matches_module_default():
    """--flows_wfull's default is a literal (the parser must import without
    flowjax); keep it pinned to the module constant."""
    from darksirens.cli.inference import build_parser

    assert build_parser().get_default("flows_wfull") == DEFAULT_W_FULL


# ── toy-flow fixture (shared by the likelihood-level tests) ─────────────────


def _flow_config(key, z_std=(0.3, 0.1, 0.15, 0.15)):
    return {
        "base_dist": "Normal", "data_dim": 4, "type": "spline",
        "flow_layers": 2, "knots": 4, "key": key,
        "columns": list(flows_mod.SPECTRAL_COLUMNS),
        "Z_mean": [2.0, 3.4, 7.0, 0.0], "Z_std": list(z_std),
        "constraints": {"0": {"type": "ordered_positive", "dims": [0, 1]},
                        "1": None, "2": {"type": "positive"},
                        "3": {"type": "interval", "low": -1, "high": 1}},
    }


def _save_flow(path, flow, config):
    arrays, _ = eqx.partition(flow, eqx.is_array)
    leaves, _ = jtu.tree_flatten(arrays)
    np.savez(path, *[np.asarray(l) for l in leaves],
             config_json=json.dumps(config))


def _toy_catalog():
    return EMCatalog(apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
                     wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
                     delta_g_pix_z=jnp.zeros((1, 1)), dN_obs_kde=None,
                     pixel_to_cache_idx=None)


def _make_gw_sel(n_sel=20_000, seed=7):
    rng = np.random.default_rng(seed)
    m1det = np.exp(rng.uniform(np.log(5.0), np.log(150.0), n_sel))
    q = rng.uniform(0.2, 1.0, n_sel)
    dL = rng.uniform(100.0, 8000.0, n_sel)
    chieff = rng.uniform(-0.5, 0.5, n_sel)
    p_draw = (1.0 / (m1det * np.log(30.0))) * (1.0 / 0.8) * (1.0 / 7900.0)
    return GWEvent(m1det=jnp.asarray(m1det), m2det=jnp.asarray(q * m1det),
                   dL=jnp.asarray(dL), chieff=jnp.asarray(chieff),
                   prior_wt=jnp.asarray(p_draw),
                   pixels=jnp.zeros(n_sel, dtype=jnp.int32), q=jnp.asarray(q),
                   valid=jnp.ones(n_sel, dtype=bool))


def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                        alpha_miss=0.5)


def _make_setup(root, z_std):
    for name, key in [("GW_TOY_X", 3), ("GW_TOY_Y", 11)]:
        d = root / name
        d.mkdir()
        cfg = _flow_config(key, z_std)
        _save_flow(d / f"{name}_flow.npz",
                   flows_mod.create_flow_from_config(cfg), cfg)
    ens = flows_mod.load_flow_ensemble(root)

    log_dl, log_p_dl = build_pe_dl_prior_table(PE_H0, PE_OM0)
    qg, cg, tab = build_chi_eff_prior_table(0.99)
    pe_tables = PePriorTables(log_dl_grid=jnp.asarray(log_dl),
                              log_p_dl=jnp.asarray(log_p_dl),
                              chi_q_grid=jnp.asarray(qg),
                              chi_grid=jnp.asarray(cg),
                              chi_table=jnp.asarray(tab))

    model = get_model("powerlaw+peak")
    m1_edges, q_edges = make_mass_q_edges(*resolve_mass_grid_bounds(model),
                                          n_m1=256, n_q=128)
    return dict(ens=ens, pe_tables=pe_tables, model=model, m1_edges=m1_edges,
                q_edges=q_edges, gw_sel=_make_gw_sel(), catalog=_toy_catalog(),
                theta_fid=jnp.asarray(
                    get_fixed_population_params("powerlaw+peak")))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Sharply measured toy events (the production-like regime)."""
    return _make_setup(tmp_path_factory.mktemp("mix_flows"),
                       (0.3, 0.1, 0.15, 0.15))


@pytest.fixture(scope="module")
def broad_setup(tmp_path_factory):
    """Poorly measured toy events: posteriors wide enough that the
    full-population-support component actually lands on them at finite J."""
    return _make_setup(tmp_path_factory.mktemp("mix_flows_broad"),
                       (0.5, 0.5, 0.4, 0.4))


def _build(setup, boxes, J, w_full, pe_tables=None):
    ens = setup["ens"]
    return build_flow_loglike(
        model=setup["model"],
        eval_logflows=flows_mod.make_ensemble_log_prob_per_event(ens),
        group_params=ens.group_params(),
        u_base=jax.random.uniform(jax.random.PRNGKey(0),
                                  (ens.n_flows, J, 4), dtype=jnp.float64),
        m1_edges=setup["m1_edges"], q_edges=setup["q_edges"],
        pe_tables=pe_tables or setup["pe_tables"],
        support_boxes=boxes, gw_sel=setup["gw_sel"],
        em_catalog_sel=setup["catalog"], Ndraw=40_000.0,
        nEvents=ens.n_flows, w_full=w_full)


def _reference_lnZ(setup, theta, n=400_000, seed=1234):
    """Independent reference: importance sampling FROM the flows.

    With X = (m1det, m2det, dL, chi) and theta = (m1src, q, chi, z),
    |dX/dtheta| = m1src (1+z)^2 ddL/dz, so
    Z_i = E_{X~flow_i}[ t(theta(X)) / (pi_PE(X) |dX/dtheta|) ].
    Shares no code with the windowed/mixture estimator.
    """
    ens, model, pe = setup["ens"], setup["model"], setup["pe_tables"]
    cosmo = _cosmo()
    state = prepare_redshift_prior_state("spectral_sirens", cosmo, _survey(),
                                         setup["catalog"], materialize_state=True)
    log_pvol, zg = np.asarray(state.log_pvol), np.asarray(zgrid)
    S = np.asarray(flows_mod.ensemble_sample(ens, jax.random.key(seed), n))
    m1_edges, q_edges = setup["m1_edges"], setup["q_edges"]
    lnZ, ess = [], []
    for i in range(ens.n_flows):
        m1det, m2det, dL, chi = S[i, :, 0], S[i, :, 1], S[i, :, 2], S[i, :, 3]
        z = np.asarray(z_of_dL(jnp.asarray(dL), cosmo.H0, cosmo.Om0))
        ok = np.isfinite(z)
        z = np.where(ok, z, 0.0)
        m1src, q = m1det / (1.0 + z), m2det / m1det
        ok &= (m1src >= float(m1_edges[0])) & (m1src <= float(m1_edges[-1]))
        ok &= (q >= float(q_edges[0])) & (q <= float(q_edges[-1]))
        ok &= np.abs(chi) <= 1.0
        log_t = np.asarray(
            model.log_p_massspin(jnp.asarray(m1src), jnp.asarray(q),
                                 jnp.asarray(chi), theta)
            + model.log_rate_z(jnp.asarray(z), theta)
        ) + np.interp(z, zg, log_pvol)
        p_chi = np.asarray(chi_eff_prior_prob(
            jnp.asarray(1.0 / (1.0 + q)), jnp.asarray(chi),
            pe.chi_q_grid, pe.chi_grid, pe.chi_table))
        ok &= p_chi > 0.0
        log_pi = np.interp(np.log(np.maximum(dL, 1e-300)),
                           np.asarray(pe.log_dl_grid),
                           np.asarray(pe.log_p_dl)) \
            + np.log(np.maximum(p_chi, 1e-300))
        DC = np.asarray(dL_of_z(jnp.asarray(z), cosmo.H0, cosmo.Om0)) / (1.0 + z)
        E = np.sqrt(cosmo.Om0 * (1.0 + z) ** 3 + (1.0 - cosmo.Om0))
        log_jac = (np.log(m1src) + 2.0 * np.log1p(z)
                   + np.log(DC + (1.0 + z) * (C_KM_S / cosmo.H0) / E))
        lw = np.where(ok, log_t - log_pi - log_jac, -np.inf)
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        mx = lw.max()
        w = np.exp(lw - mx)
        lnZ.append(mx + np.log(w.mean()))
        ess.append(float(w.sum() ** 2 / np.sum(w ** 2)))
    return np.asarray(lnZ), np.asarray(ess)


def _shrink(boxes, frac):
    """Pull every window toward its centre by ``frac`` of its half-width."""
    out = dict(boxes)
    for k in ("m1det", "q", "dL", "chieff", "mc_det"):
        b = np.asarray(boxes[k])
        c, h = b.mean(axis=1), 0.5 * (b[:, 1] - b[:, 0])
        out[k] = jnp.asarray(np.stack([c - frac * h, c + frac * h], axis=1))
    r = np.asarray(boxes["chi_resid"])
    out["chi_resid"] = jnp.asarray(r * frac)
    return out


# ── 3. the mixture recovers the clipped tail on the real likelihood body ────


def _tail_favoring(theta_fid):
    """Rising mass function + strong redshift evolution: upweights exactly the
    high-mass / high-z region an empirical support box clips."""
    t = np.asarray(theta_fid, dtype=np.float64).copy()
    t[1] = -6.0     # alpha
    t[3] = 180.0    # m_max
    t[11] = 6.0     # gamma
    return jnp.asarray(t)


def test_mixture_recovers_the_clipped_mass(broad_setup):
    """An under-covering window biases lnZ_i; the mixture removes the bias."""
    setup = broad_setup
    theta = setup["theta_fid"]
    ref, ref_ess = _reference_lnZ(setup, theta)
    assert (ref_ess > 10_000).all(), ref_ess       # reference must be converged

    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1),
                                            n=4096, margin=0.25)
    tight = _shrink(boxes, 0.35)

    J = 32_768
    old = np.asarray(_build(setup, tight, J, 0.0)
                     .event_diagnostics(_cosmo(), _survey(), theta)[0])
    mix = np.asarray(_build(setup, tight, J, 0.3)
                     .event_diagnostics(_cosmo(), _survey(), theta)[0])

    d_old, d_mix = old - ref, mix - ref
    # Measured here: the window loses 5.7 and 317 nats; the mixture returns
    # both events to the reference.
    assert (d_old < -1.0).all(), (old, ref)
    assert (np.abs(d_mix) < 0.6).all(), (d_old, d_mix)
    assert np.abs(d_mix).max() < 0.2 * np.abs(d_old).max(), (d_old, d_mix)


def test_windowed_bias_moves_with_the_hyperparameters(broad_setup):
    """The clipped mass is NOT an ignorable per-event constant.

    Under a tail-favoring population the windowed-only estimator loses a
    DIFFERENT amount than at the fiducial one, which is what turns the
    truncation into a posterior bias rather than a per-event offset.  The
    mixture recovers most of it; the residual is honest Monte-Carlo error
    (the full-support component runs at ESS of a few here) and is therefore
    visible to the total-variance guard, unlike the bias it replaces.
    """
    setup = broad_setup
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1),
                                            n=4096, margin=0.25)
    tight = _shrink(boxes, 0.35)
    J = 32_768

    losses = {}
    for name, theta in (("fiducial", setup["theta_fid"]),
                        ("tail", _tail_favoring(setup["theta_fid"]))):
        ref, _ = _reference_lnZ(setup, theta)
        old = np.asarray(_build(setup, tight, J, 0.0)
                         .event_diagnostics(_cosmo(), _survey(), theta)[0])
        mix = np.asarray(_build(setup, tight, J, 0.3)
                         .event_diagnostics(_cosmo(), _survey(), theta)[0])
        losses[name] = old - ref
        assert (mix > old + 1.0).all(), (name, old, mix)

    # Same events, same window, different hyperparameters -> different loss.
    shift = np.abs(losses["tail"] - losses["fiducial"])
    assert shift.max() > 1.0, losses


def test_wide_window_and_mixture_agree_with_the_reference(setup):
    """Convergence in the margin: a wide window needs no mixture, and adding
    one must not move the answer."""
    theta = setup["theta_fid"]
    ref, ref_ess = _reference_lnZ(setup, theta)
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1),
                                            n=32_768, margin=1.0)
    J = 32_768
    a = np.asarray(_build(setup, boxes, J, 0.0)
                   .event_diagnostics(_cosmo(), _survey(), theta)[0])
    b = np.asarray(_build(setup, boxes, J, 0.05)
                   .event_diagnostics(_cosmo(), _survey(), theta)[0])
    assert np.abs(a - ref).max() < 0.10, (a, ref)
    assert np.abs(b - ref).max() < 0.10, (b, ref)
    assert np.abs(b - a).max() < 0.10, (a, b)


def test_mixture_is_deterministic_and_reports_its_allocation(setup):
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1))
    ll = _build(setup, boxes, 8192, 0.05)
    assert ll.n_full_draws == resolve_full_draws(8192, 0.05)
    assert ll.w_full == ll.n_full_draws / 8192
    a = float(ll(_cosmo(), _survey(), setup["theta_fid"]))
    b = float(ll(_cosmo(), _survey(), setup["theta_fid"]))
    assert np.isfinite(a) and a == b


# ── 4. PE-support tightening ────────────────────────────────────────────────


def test_zero_chi_eff_prior_draws_are_dropped_not_floored(setup):
    """gwcat's -50 log floor must not become e^50 of importance weight.

    Two runs differing ONLY in whether the chi_eff prior table is exactly zero
    outside a narrow band or floored at e^-50 there.  With the mask, the
    zero-prior region contributes nothing; without it, those draws would carry
    an e^50 multiplier and dominate the event term.
    """
    pe = setup["pe_tables"]
    chi_grid = np.asarray(pe.chi_grid)
    table = np.asarray(pe.chi_table).copy()
    table[:, np.abs(chi_grid) > 0.3] = 0.0
    zeroed = pe._replace(chi_table=jnp.asarray(table))
    floored = pe._replace(chi_table=jnp.asarray(np.where(table > 0.0, table,
                                                         np.exp(-50.0))))

    theta = np.asarray(setup["theta_fid"], dtype=np.float64).copy()
    theta[10] = 1.0   # sigma_chi at its prior ceiling: broad chi_eff draws
    theta = jnp.asarray(theta)

    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1))
    # Open the chi_eff window so the WINDOWED component also reaches |chi|>0.3.
    wide = dict(boxes)
    n = int(np.asarray(boxes["chieff"]).shape[0])
    wide["chieff"] = jnp.tile(jnp.asarray([-1.0, 1.0]), (n, 1))
    wide["chi_ab"] = jnp.zeros((n, 2))
    wide["chi_resid"] = jnp.tile(jnp.asarray([-2.0, 2.0]), (n, 1))

    lz_masked = np.asarray(_build(setup, wide, 8192, 0.05, pe_tables=zeroed)
                           .event_diagnostics(_cosmo(), _survey(), theta)[0])
    lz_floored = np.asarray(_build(setup, wide, 8192, 0.05, pe_tables=floored)
                            .event_diagnostics(_cosmo(), _survey(), theta)[0])
    assert np.isfinite(lz_masked).all()
    # The e^50 floor inflates the term by tens of nats; the mask does not.
    assert (lz_floored - lz_masked > 20.0).all(), (lz_floored, lz_masked)


def test_draws_beyond_the_table_grid_carry_no_weight(setup):
    """|chi_eff| > amax is outside the PE prior's support, clamping or not.

    ``jnp.interp`` CLAMPS out-of-range inputs to the end of the chi_eff grid
    (gwcat's own behaviour, kept for the port), and the shipped amax=0.99
    table's boundary column is only zero to numerical accuracy -- 57/200 q rows
    are positive there, down to 1e-12 -- so ``p_chi > 0`` alone kept such draws
    and inflated their weight by 10-28 nats.  A proposal confined to
    |chi_eff| > amax must therefore integrate to exactly zero.
    """
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1))
    n = int(np.asarray(boxes["chieff"]).shape[0])
    beyond = dict(boxes)
    beyond["chieff"] = jnp.tile(jnp.asarray([0.995, 1.0]), (n, 1))
    beyond["chi_ab"] = jnp.zeros((n, 2))
    beyond["chi_resid"] = jnp.tile(jnp.asarray([-2.0, 2.0]), (n, 1))

    theta = np.asarray(setup["theta_fid"], dtype=np.float64).copy()
    theta[10] = 1.0   # sigma_chi at its prior ceiling: the window is reachable
    theta = jnp.asarray(theta)

    # w_full = 0: every draw comes from the beyond-amax window.
    lz = np.asarray(_build(setup, beyond, 4096, 0.0)
                    .event_diagnostics(_cosmo(), _survey(), theta)[0])
    assert np.isneginf(lz).all(), lz


def test_pe_bounds_from_checkpoint_config_are_honoured(tmp_path):
    """A checkpoint declaring its PE window masks draws outside it."""
    root = tmp_path / "pe_bounded"
    cfg = _flow_config(3)
    cfg["pe_bounds"] = {"luminosity_distance": [500.0, 2000.0],
                        "mass_1": [5.0, 100.0]}
    d = root / "GW_B"
    d.mkdir(parents=True)
    _save_flow(d / "GW_B_flow.npz", flows_mod.create_flow_from_config(cfg), cfg)
    ens = flows_mod.load_flow_ensemble(root)
    box = np.asarray(flows_mod.pe_support_boxes(ens))
    assert box.shape == (1, 4, 2)
    np.testing.assert_allclose(box[0, 0], [5.0, 100.0])
    np.testing.assert_allclose(box[0, 2], [500.0, 2000.0])
    assert np.isneginf(box[0, 1, 0]) and np.isposinf(box[0, 1, 1])
    assert np.isneginf(box[0, 3, 0]) and np.isposinf(box[0, 3, 1])

    # Absent from the config -> unbounded (today's behaviour, unchanged).
    boxes = flows_mod.compute_support_boxes(ens, key=jax.random.key(0), n=512)
    assert "pe_box" in boxes


def test_pe_bounds_reject_inverted_window(tmp_path):
    root = tmp_path / "pe_bad"
    cfg = _flow_config(3)
    cfg["pe_bounds"] = {"luminosity_distance": [2000.0, 500.0]}
    d = root / "GW_C"
    d.mkdir(parents=True)
    _save_flow(d / "GW_C_flow.npz", flows_mod.create_flow_from_config(cfg), cfg)
    ens = flows_mod.load_flow_ensemble(root)
    with pytest.raises(ValueError, match="upper bound must exceed"):
        flows_mod.pe_support_boxes(ens)


# ── 5. operands are jit ARGUMENTS, not HLO constants ────────────────────────


def _constant_elements(lowered):
    """Total elements of every ``stablehlo.constant`` in a lowered module.

    Walks the MLIR instead of stringifying it: the population model's own
    normalisation table is a 1.3e7-element literal, so ``as_text()`` here
    returns ~640 MB whatever the operands do.
    """
    total = 0

    def walk(op):
        nonlocal total
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    o = child.operation
                    if o.name == "stablehlo.constant":
                        shape = getattr(o.results[0].type, "shape", None)
                        if shape is not None:
                            total += int(np.prod(shape)) if len(shape) else 1
                    walk(o)

    walk(lowered.compiler_ir().operation)
    return total


def test_operands_do_not_become_hlo_constants(setup):
    """Closing over the operands embeds them as ``dense<>`` literals (PR #296).

    A/B on the SAME body: operands bound as a jit argument, versus captured by
    an enclosing jit (what the flow factory did before this change).  The
    captured version must carry the whole operand payload as constants.
    """
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1))
    ll = _build(setup, boxes, 4096, 0.05)
    args = (_cosmo(), _survey(), setup["theta_fid"])

    n_elem = sum(int(np.asarray(x).size) for x in jtu.tree_leaves(ll.operands))
    assert n_elem > 100_000, n_elem          # the test data must be big enough

    clean = _constant_elements(
        ll.jitted_body.lower(*args, ll.operands, ll.distance_table)
    )
    # Issue #305: the 21x41x31x500 comoving-distance table (13,345,500
    # elements, ~214 MB of module text as a literal) must never lower as a
    # constant here -- it rides as the trailing jit argument.
    assert clean < 13_345_500, clean
    leaky = _constant_elements(
        jax.jit(
            lambda c, s, p: ll.jitted_body(c, s, p, ll.operands, ll.distance_table)
        ).lower(*args)
    )
    assert leaky - clean > 0.9 * n_elem, (leaky, clean, n_elem)


def test_body_compiles_once_across_calls(setup):
    boxes = flows_mod.compute_support_boxes(setup["ens"], key=jax.random.key(1))
    ll = _build(setup, boxes, 2048, 0.05)
    theta = np.asarray(setup["theta_fid"], dtype=np.float64)
    for h0 in (65.0, 70.0, 75.0):
        for d_alpha in (0.0, 0.3):
            t = theta.copy()
            t[1] += d_alpha
            ll(CosmoParams(H0=h0, Om0=Om0Planck), _survey(), jnp.asarray(t))
    assert ll.jitted_body._cache_size() == 1
