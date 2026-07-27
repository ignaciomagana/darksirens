"""
test_lensed_singleton_channel.py
--------------------------------
The lensed-singleton (exactly-one-detected image) channel:

  - fcpdet: the analytic Finn-Chernoff P_det must match the mock generator's
    SNRModel exactly (the censoring factor is only exact if they agree).
  - LensedSingleImageSet loader: picks precisely the XOR-detected sources
    from the on-disk per-image layout, with correct mu_det/mu_partner.
  - compute_lensed_single_selection_term vs numpy brute force.
  - lensed_single_log_likelihood_event vs a full numpy oracle (both image
    branches, censoring included).
  - Master likelihood: SINGLETON_LENSING_MIXTURE reduces exactly to OFF at
    tau_A = 0; selection responds monotonically to tau_A; gradients wrt
    population hyperparameters stay finite (closing the review's
    pop-coupling test gap for this channel).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from scipy.special import logsumexp as sp_lse

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import (
    H0Planck, Om0Planck, dL_of_z, z_of_dL, ddL_of_z,
)
from darksirens.redshift.volume import log_volume_prior_vmap
from darksirens.lensing.fcpdet import (
    FCPdetParams, make_fc_pdet_params, pdet_fc, log_one_minus_pdet_fc,
)
from darksirens.lensing.lensed_injections import (
    save_lensed_injections, load_lensed_single_image_set, read_fc_pdet_attrs,
)
from darksirens.lensing.slmarks import make_sis_lens_params, tau_2_SIS
from darksirens.lensing.grids import make_y_grid
from darksirens.likelihood.cluster_selection import (
    compute_lensed_single_selection_term,
)
from darksirens.likelihood.cluster_likelihood import (
    lensed_single_log_likelihood_event,
)
from darksirens.gw.populations import get_fixed_population_params


def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(
        n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
    )


def _toy_catalog():
    return EMCatalog(
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )


def _toy_log_p_pop(m1src, q, z, chieff, pop_params):
    del pop_params
    return (
        -0.5 * ((m1src - 30.0) / 8.0) ** 2
        - 0.5 * ((q - 0.7) / 0.15) ** 2
        - 0.5 * ((chieff + 0.05) / 0.2) ** 2
        + 0.3 * jnp.log1p(z)
    )


def _toy_volume_prior(z, pix, catalog):
    del pix, catalog
    return log_volume_prior_vmap(z, _cosmo(), _survey())


def _synth_campaign(n_sources=300, seed=0, p_prop_scale=1.0):
    """Synthetic lensed-injection campaign with a crude detection rule that
    produces a healthy mix of none/one/both-detected sources.

    ``p_prop_scale`` rescales the (arbitrary toy-unit) proposal density; the
    master-likelihood fixture uses it to keep the lensed channel subdominant
    to the unlensed selection channel, as in a physical configuration.
    """
    rng = np.random.default_rng(seed)
    m1_src = rng.uniform(10.0, 70.0, n_sources)
    q = rng.uniform(0.3, 1.0, n_sources)
    z = rng.uniform(0.05, 2.0, n_sources)
    chieff = rng.uniform(-0.4, 0.4, n_sources)
    y = rng.uniform(0.05, 0.95, n_sources)
    p_prop_src = np.full(n_sources, p_prop_scale / (60 * 0.7 * 1.95 * 0.8))
    p_prop_y = np.full(n_sources, 1.0 / 0.9)

    mu_p = (1.0 + y) / y
    mu_m = (1.0 - y) / y
    dL_src = np.asarray(dL_of_z(jnp.asarray(z), H0Planck, Om0Planck))
    dL_p = dL_src / np.sqrt(mu_p)
    dL_m = dL_src / np.sqrt(mu_m)
    m1det = (1.0 + z) * m1_src
    det_p = (m1det > 25.0) & (dL_p < 4000.0)
    det_m = (m1det > 25.0) & (dL_m < 4000.0)

    n_img = 2 * n_sources
    per_img = dict(
        source_id=np.repeat(np.arange(n_sources, dtype=np.int32), 2),
        image_id=np.tile(np.array([0, 1], dtype=np.int32), n_sources),
        m1_src=np.repeat(m1_src, 2), q_src=np.repeat(q, 2),
        z_src=np.repeat(z, 2), chieff=np.repeat(chieff, 2),
        y_source=np.repeat(y, 2),
        p_prop_src=np.repeat(p_prop_src, 2), p_prop_y=np.repeat(p_prop_y, 2),
    )
    mu = np.empty(n_img); mu[0::2] = mu_p; mu[1::2] = mu_m
    det = np.empty(n_img, dtype=bool); det[0::2] = det_p; det[1::2] = det_m
    per_img["mu"] = mu
    per_img["detected"] = det
    truth = dict(det_p=det_p, det_m=det_m, mu_p=mu_p, mu_m=mu_m,
                 m1_src=m1_src, q=q, z=z, chieff=chieff, y=y,
                 p_prop_src=p_prop_src, p_prop_y=p_prop_y)
    return per_img, truth, n_sources


# ============================================================================
# A. fcpdet matches the generator's SNRModel exactly
# ============================================================================

def test_fc_pdet_matches_generator_snr_model():
    from scripts.mock_lensing.generate_mock_lensing import SNRModel

    model = SNRModel(rho_thr=8.0, horizon_Mpc=3000.0)
    params = make_fc_pdet_params(rho_thr=8.0, horizon_mpc=3000.0, mc_bar=1.22)
    np.testing.assert_allclose(float(params.r0), model.r0, rtol=0, atol=0)

    rng = np.random.default_rng(1)
    m1 = rng.uniform(5.0, 80.0, 500)
    q = rng.uniform(0.2, 1.0, 500)
    z = rng.uniform(0.05, 3.0, 500)
    dL = rng.uniform(100.0, 30000.0, 500)
    ours = np.asarray(pdet_fc(jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z),
                              jnp.asarray(dL), params))
    # The generator tabulates the survival function on 20001 nodes; the
    # closed-form polynomial is the exact version of the same integral.
    theirs = model.p_det(m1, q, z, dL)
    np.testing.assert_allclose(ours, theirs, atol=5e-9)
    # log(1-Pdet) consistency where 1-Pdet > 0
    lom = np.asarray(log_one_minus_pdet_fc(
        jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z), jnp.asarray(dL), params))
    mask = (1.0 - ours) > 1e-12
    np.testing.assert_allclose(
        np.exp(lom[mask]), (1.0 - ours)[mask], rtol=1e-9,
    )


def _draw_theta_fc(n, rng):
    """Inverse-CDF draws of p(Theta) = 5 Theta (4-Theta)^3 / 256 on (0, 4)."""
    from darksirens.lensing.fcpdet import _theta_cdf

    grid = np.linspace(0.0, 4.0, 200_001)
    cdf = np.asarray(_theta_cdf(jnp.asarray(grid)))
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


@pytest.mark.parametrize("x_plus,x_minus", [(1.0, 2.0), (1.5, 3.0), (0.8, 1.0)])
def test_pair_censoring_assumes_independent_image_orientations(x_plus, x_minus):
    """Pin the INDEPENDENT-orientation convention the lensing stack is built on,
    against a direct Monte Carlo of both conventions.

    ``log_one_minus_pdet_fc`` marginalises the partner image's Theta
    independently of the detected image's, and ``cluster_selection``'s pair
    efficiency is the product p_det(+) p_det(-). That matches the mock
    generator, which draws a separate uniform per image, so the mock study is
    self-consistent -- and this test locks that agreement in.

    It is NOT the physical model: the two images of one source share their
    inclination and polarization, so with ONE Theta per source
    P(both) = S(max(x_+, x_-)) and P(exactly one) = |S(x_+) - S(x_-)|. Both
    conventions are measured here so the size of the gap (up to ~2.6x on
    P(both)) is recorded in the repo. Fixing it needs a joint p(Theta_+,
    Theta_-) -- Theta bundles the shared inclination with the Earth-rotation
    antenna response, which decorrelates over the SIS day-to-month delays -- plus
    a re-rendered injection campaign; see the fcpdet module docstring.
    """
    from darksirens.lensing.fcpdet import _theta_cdf

    S_p = 1.0 - float(_theta_cdf(jnp.asarray(x_plus)))
    S_m = 1.0 - float(_theta_cdf(jnp.asarray(x_minus)))
    assert S_p > S_m          # the fainter image needs a larger Theta

    rng = np.random.default_rng(20260726)
    n = 400_000
    th_p, th_m = _draw_theta_fc(n, rng), _draw_theta_fc(n, rng)
    det_p, det_m = th_p > x_plus, th_m > x_minus

    # The convention the code implements, validated against its own MC.
    both_independent = float(np.mean(det_p & det_m))
    one_independent = float(np.mean(det_p ^ det_m))
    np.testing.assert_allclose(both_independent, S_p * S_m, atol=4e-3)
    np.testing.assert_allclose(
        one_independent, S_p * (1 - S_m) + (1 - S_p) * S_m, atol=4e-3,
    )

    # The shared-orientation alternative, also validated, and NOT what the
    # code computes. If a future change adopts it, these must be updated
    # deliberately and the injection campaign re-rendered to match.
    th = _draw_theta_fc(n, rng)
    both_shared = float(np.mean((th > x_plus) & (th > x_minus)))
    one_shared = float(np.mean((th > x_plus) ^ (th > x_minus)))
    np.testing.assert_allclose(both_shared, S_m, atol=4e-3)
    np.testing.assert_allclose(one_shared, S_p - S_m, atol=4e-3)

    assert both_shared > both_independent    # correlated pairs are commoner
    assert one_shared < one_independent      # ... at the singletons' expense
    assert both_shared / both_independent > 1.1


# ============================================================================
# A2. The 'shared_iota' joint-orientation pair model vs direct Monte Carlo
# ============================================================================

def _draw_theta_geometric(rng, n, cos_iota):
    """One image's Finn-Chernoff Theta from an isotropic antenna draw at the
    GIVEN inclination — the exact FC93 eq. 3.31 decomposition, evaluated
    through the SAME ``theta_fc_from_antenna`` the shared_iota tables and the
    generator's shared-orientation rendering use (pins the convention)."""
    from darksirens.lensing.fcpdet import theta_fc_from_antenna

    u = rng.uniform(-1.0, 1.0, n)              # detector-frame cos(theta)
    phi = rng.uniform(0.0, 2 * np.pi, n)       # detector-frame azimuth
    psi = rng.uniform(0.0, 2 * np.pi, n)       # effective polarization
    fp0 = 0.5 * (1.0 + u**2) * np.cos(2 * phi)
    fx0 = u * np.sin(2 * phi)
    fp = fp0 * np.cos(2 * psi) + fx0 * np.sin(2 * psi)
    fx = -fp0 * np.sin(2 * psi) + fx0 * np.cos(2 * psi)
    return np.asarray(theta_fc_from_antenna(
        jnp.asarray(fp), jnp.asarray(fx), jnp.asarray(cos_iota)))


def _dL_for_threshold(x, m1, q, z, params):
    """Apparent distance whose Finn-Chernoff threshold is exactly x."""
    from darksirens.lensing.fcpdet import _chirp_mass_det

    mc = float(_chirp_mass_det(jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z)))
    return (
        x * 8.0 * float(params.r0)
        * (mc / float(params.mc_bar)) ** (5.0 / 6.0) / float(params.rho_thr)
    )


@pytest.mark.parametrize("x_plus,x_minus", [(1.0, 2.0), (1.5, 3.0), (0.8, 1.0)])
def test_shared_iota_pair_model_matches_direct_mc(x_plus, x_minus):
    """The 'shared_iota' P(both) / P(exactly one) must match a direct Monte
    Carlo of the joint model (shared inclination, per-image independent
    antenna draws — the |dt| >> hours Earth-rotation limit) to <= 1%
    relative, and must land strictly BETWEEN the independent and fully-shared
    limits OF THE SAME GEOMETRIC CONVENTION.

    The limits are computed from the same Monte Carlo (not from the
    polynomial fit): the exact geometric Theta marginal deviates from the
    FC93 polynomial by up to ~2.4x in the far tail (S_geom(3) ~ 0.038 vs
    S_poly(3) ~ 0.016), so polynomial-fit limits are not valid brackets for
    the geometric joint model. 'independent'-mode code keeps the polynomial
    exactly (see test_pair_orientation_independent_mode_is_bit_unchanged).
    """
    from darksirens.lensing.fcpdet import (
        pdet_pair_both_fc, pdet_pair_exactly_one_fc, log_pmiss_partner_fc,
    )

    params = make_fc_pdet_params()
    m1, q, z = 30.0, 0.8, 0.5
    dL_p = jnp.asarray(_dL_for_threshold(x_plus, m1, q, z, params))
    dL_m = jnp.asarray(_dL_for_threshold(x_minus, m1, q, z, params))
    m1j, qj, zj = jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z)

    both_model = float(pdet_pair_both_fc(
        m1j, qj, zj, dL_p, dL_m, params, pair_orientation_mode="shared_iota"))
    one_model = float(pdet_pair_exactly_one_fc(
        m1j, qj, zj, dL_p, dL_m, params, pair_orientation_mode="shared_iota"))

    # Direct MC of the joint model: >= 4e6 sources, ONE cos(iota) per source,
    # independent antenna draws per image (arrival times decorrelated).
    rng = np.random.default_rng(20260727)
    n = 4_000_000
    cos_iota = rng.uniform(-1.0, 1.0, n)
    th_1 = _draw_theta_geometric(rng, n, cos_iota)
    th_2 = _draw_theta_geometric(rng, n, cos_iota)
    det_1, det_2 = th_1 > x_plus, th_2 > x_minus

    both_mc = float(np.mean(det_1 & det_2))
    one_mc = float(np.mean(det_1 ^ det_2))
    np.testing.assert_allclose(both_model, both_mc, rtol=1e-2)
    np.testing.assert_allclose(one_model, one_mc, rtol=1e-2)

    # Same-convention limits from the same draws: independent = product of
    # the two images' marginals; fully shared = one Theta for both images.
    both_indep = float(np.mean(det_1)) * float(np.mean(det_2))
    one_indep = (
        float(np.mean(det_1)) * float(np.mean(~det_2))
        + float(np.mean(~det_1)) * float(np.mean(det_2))
    )
    both_full = float(np.mean(det_1 & (th_1 > x_minus)))
    one_full = float(np.mean(det_1 ^ (th_1 > x_minus)))
    assert both_indep < both_model < both_full
    assert one_full < one_model < one_indep

    # Conditional censoring factor of the lensed-singleton channel:
    # P(partner missed | this image detected), against the MC conditional.
    miss_model = float(jnp.exp(log_pmiss_partner_fc(
        m1j, qj, zj, dL_p, dL_m, params, pair_orientation_mode="shared_iota")))
    miss_mc = float(np.mean(det_1 & ~det_2)) / float(np.mean(det_1))
    np.testing.assert_allclose(miss_model, miss_mc, rtol=1e-2)


def test_pair_orientation_independent_mode_is_bit_unchanged():
    """'independent' mode must be BIT-IDENTICAL to the pinned polynomial
    forms — the new mode is opt-in, the default rendering stays untouched."""
    from darksirens.lensing.fcpdet import (
        pdet_pair_both_fc, pdet_pair_exactly_one_fc, log_pmiss_partner_fc,
    )

    params = make_fc_pdet_params()
    rng = np.random.default_rng(7)
    m1 = jnp.asarray(rng.uniform(5.0, 80.0, 200))
    q = jnp.asarray(rng.uniform(0.2, 1.0, 200))
    z = jnp.asarray(rng.uniform(0.05, 3.0, 200))
    dL_d = jnp.asarray(rng.uniform(100.0, 30000.0, 200))
    dL_p = jnp.asarray(rng.uniform(100.0, 30000.0, 200))

    s_d = np.asarray(pdet_fc(m1, q, z, dL_d, params))
    s_p = np.asarray(pdet_fc(m1, q, z, dL_p, params))
    np.testing.assert_array_equal(
        np.asarray(pdet_pair_both_fc(m1, q, z, dL_d, dL_p, params)),
        s_d * s_p,
    )
    np.testing.assert_array_equal(
        np.asarray(pdet_pair_exactly_one_fc(m1, q, z, dL_d, dL_p, params)),
        s_d * (1.0 - s_p) + (1.0 - s_d) * s_p,
    )
    np.testing.assert_array_equal(
        np.asarray(log_pmiss_partner_fc(m1, q, z, dL_d, dL_p, params)),
        np.asarray(log_one_minus_pdet_fc(m1, q, z, dL_p, params)),
    )
    with pytest.raises(ValueError, match="pair_orientation_mode"):
        log_pmiss_partner_fc(
            m1, q, z, dL_d, dL_p, params, pair_orientation_mode="bogus")


def test_fc_pdet_limits_and_gradient():
    params = make_fc_pdet_params()
    # Very close/loud -> P_det ~ 1; very far -> P_det ~ 0
    close = float(pdet_fc(jnp.asarray(30.0), jnp.asarray(0.8), jnp.asarray(0.1),
                          jnp.asarray(50.0), params))
    far = float(pdet_fc(jnp.asarray(30.0), jnp.asarray(0.8), jnp.asarray(0.1),
                        jnp.asarray(1e6), params))
    assert close > 0.999 and far < 1e-6
    g = jax.grad(lambda d: pdet_fc(jnp.asarray(30.0), jnp.asarray(0.8),
                                   jnp.asarray(0.5), d, params))(jnp.asarray(2000.0))
    assert np.isfinite(np.asarray(g))


# ============================================================================
# B. Single-image loader
# ============================================================================

def test_single_image_loader_picks_exactly_one_detected(tmp_path):
    per_img, truth, n_sources = _synth_campaign()
    path = str(tmp_path / "lensed.h5")
    save_lensed_injections(
        path, n_draw_sources=n_sources,
        snr_model_attrs={"fc_rho_thr": 8.0, "fc_r0": 750.0, "fc_mc_bar": 1.22},
        **per_img,
    )
    singles = load_lensed_single_image_set(path)
    one_det = truth["det_p"] ^ truth["det_m"]
    assert singles.n_kept == int(one_det.sum()) > 0

    # mu_det is the detected image's magnification, mu_partner the other's.
    is_plus = truth["det_p"][one_det]
    np.testing.assert_allclose(
        np.asarray(singles.mu_det),
        np.where(is_plus, truth["mu_p"][one_det], truth["mu_m"][one_det]),
    )
    np.testing.assert_allclose(
        np.asarray(singles.mu_partner),
        np.where(is_plus, truth["mu_m"][one_det], truth["mu_p"][one_det]),
    )
    np.testing.assert_array_equal(np.asarray(singles.image_is_plus), is_plus)
    assert float(singles.n_draw_sources) == n_sources

    attrs = read_fc_pdet_attrs(path)
    assert attrs == {"fc_rho_thr": 8.0, "fc_r0": 750.0, "fc_mc_bar": 1.22}


# ============================================================================
# C. Selection term vs brute force
# ============================================================================

def test_lensed_single_selection_matches_brute_force(tmp_path):
    per_img, truth, n_sources = _synth_campaign(seed=3)
    path = str(tmp_path / "lensed.h5")
    save_lensed_injections(path, n_draw_sources=n_sources, **per_img)
    singles = load_lensed_single_image_set(path)

    sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0)
    log_mu, Neff, log_sigma2 = compute_lensed_single_selection_term(
        singles, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis,
        _toy_log_p_pop, _toy_volume_prior,
    )

    one = truth["det_p"] ^ truth["det_m"]
    log_pop = np.asarray(_toy_log_p_pop(
        jnp.asarray(truth["m1_src"][one]), jnp.asarray(truth["q"][one]),
        jnp.asarray(truth["z"][one]), jnp.asarray(truth["chieff"][one]), None))
    log_pz = np.asarray(_toy_volume_prior(jnp.asarray(truth["z"][one]), None, None))
    log_tau = np.log(np.asarray(tau_2_SIS(jnp.asarray(truth["z"][one]), sis)))
    log_py = np.log(2.0 * truth["y"][one])
    log_prop = np.log(truth["p_prop_src"][one]) + np.log(truth["p_prop_y"][one])
    log_w = log_pop + log_pz + log_tau + log_py - log_prop
    expected_log_mu = sp_lse(log_w) - np.log(n_sources)
    np.testing.assert_allclose(float(log_mu), expected_log_mu, rtol=1e-10)
    assert np.isfinite(float(Neff)) and float(Neff) > 0


# ============================================================================
# D. Evidence term vs numpy oracle
# ============================================================================

def test_lensed_single_event_matches_numpy_oracle():
    rng = np.random.default_rng(5)
    n_pe, n_y = 40, 16
    m1det = rng.uniform(30.0, 50.0, n_pe)
    q = rng.uniform(0.5, 0.95, n_pe)
    dL_app = rng.uniform(800.0, 2000.0, n_pe)
    chieff = rng.uniform(-0.2, 0.2, n_pe)
    prior_wt = rng.uniform(0.5, 1.5, n_pe)
    event = {
        "m1det": jnp.asarray(m1det), "q": jnp.asarray(q),
        "dL": jnp.asarray(dL_app), "chieff": jnp.asarray(chieff),
        "prior_wt": jnp.asarray(prior_wt),
        "valid": jnp.ones(n_pe, dtype=bool),
        "pixels": jnp.zeros(n_pe, dtype=jnp.int32),
    }
    sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0)
    fc = make_fc_pdet_params(rho_thr=8.0, horizon_mpc=3000.0)
    y_nodes, log_wy = make_y_grid(n_y)

    ours = float(lensed_single_log_likelihood_event(
        event, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, fc,
        _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
    ))

    # numpy oracle: both branches, full integrand incl. censoring
    yv = np.asarray(y_nodes); lwy = np.asarray(log_wy)
    mu_p = (1.0 + yv) / yv
    mu_m = (1.0 - yv) / yv
    log_py = np.log(2.0 * yv)

    def branch(mu_det, mu_partner):
        out = np.full((n_pe, n_y), -np.inf)
        for s in range(n_pe):
            for k in range(n_y):
                dL_true = dL_app[s] * np.sqrt(mu_det[k])
                z_s = float(z_of_dL(dL_true, H0Planck, Om0Planck))
                if not np.isfinite(z_s):
                    continue
                m1s = m1det[s] / (1.0 + z_s)
                lp = float(_toy_log_p_pop(m1s, q[s], z_s, chieff[s], None))
                lz = float(_toy_volume_prior(jnp.asarray([z_s]), None, None)[0])
                lt = float(np.log(np.asarray(tau_2_SIS(jnp.asarray(z_s), sis))))
                dL_par = float(dL_of_z(z_s, H0Planck, Om0Planck)) / np.sqrt(mu_partner[k])
                lmiss = float(log_one_minus_pdet_fc(
                    jnp.asarray(m1s), jnp.asarray(q[s]), jnp.asarray(z_s),
                    jnp.asarray(dL_par), fc))
                lJ = (-np.log1p(z_s)
                      - np.log(float(ddL_of_z(z_s, dL_true, H0Planck, Om0Planck)))
                      + 0.5 * np.log(mu_det[k]))
                lpe = -np.log(prior_wt[s])
                val = lp + lz + lt + lmiss + lJ + lpe + log_py[k] + lwy[k]
                out[s, k] = val if np.isfinite(val) else -np.inf
        return sp_lse(out) - np.log(n_pe)

    expected = np.logaddexp(branch(mu_p, mu_m), branch(mu_m, mu_p))
    np.testing.assert_allclose(ours, expected, rtol=1e-8)


# ============================================================================
# E. Master likelihood: reduction, response, gradients
# ============================================================================

@pytest.fixture(scope="module")
def master_fixture(tmp_path_factory):
    rng = np.random.default_rng(0)
    n_events, n_samp, n_sel = 4, 150, 400
    total = n_events * n_samp
    gw_pe = GWEvent(
        m1det=jnp.asarray(rng.uniform(20.0, 60.0, total)),
        m2det=jnp.asarray(rng.uniform(10.0, 30.0, total)),
        dL=jnp.asarray(rng.uniform(400.0, 3000.0, total)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, total)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, total)),
        pixels=jnp.zeros(total, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, total)),
        valid=jnp.ones(total, dtype=jnp.bool_),
    )
    gw_sel = GWEvent(
        m1det=jnp.asarray(rng.uniform(15.0, 70.0, n_sel)),
        m2det=jnp.asarray(rng.uniform(8.0, 35.0, n_sel)),
        dL=jnp.asarray(rng.uniform(200.0, 3000.0, n_sel)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, n_sel)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, n_sel)),
        pixels=jnp.zeros(n_sel, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, n_sel)),
        valid=jnp.ones(n_sel, dtype=jnp.bool_),
    )
    # Large p_prop_scale keeps mu^(1,lensed) well below mu^(1,unlensed) so
    # the combined-channel Neff stays above the MFG wall, as in a physical
    # configuration where lensing is a small perturbation.
    per_img, _, n_sources = _synth_campaign(seed=11, p_prop_scale=3.0e4)
    path = str(tmp_path_factory.mktemp("inj") / "lensed.h5")
    save_lensed_injections(path, n_draw_sources=n_sources, **per_img)
    singles = load_lensed_single_image_set(path)
    return {
        "cosmo": _cosmo(), "survey": _survey(),
        "gw_pe": gw_pe, "gw_sel": gw_sel, "catalog": _toy_catalog(),
        "n_events": n_events, "n_samp": n_samp, "Ndraw": 1000.0,
        "pop_params": jnp.asarray(get_fixed_population_params("powerlaw+peak")),
        "singles": singles,
        "fc": make_fc_pdet_params(rho_thr=8.0, horizon_mpc=3000.0),
    }


def _master_call(fx, *, singleton_lensing, A_tau, pop_params=None,
                 return_diagnostics=False):
    from darksirens.likelihood.likelihood_with_clusters import (
        darksiren_log_likelihood_with_clusters,
        CLUSTER_MODE_OFF,
        SINGLETON_LENSING_OFF, SINGLETON_LENSING_MIXTURE,
    )
    mode = (SINGLETON_LENSING_MIXTURE if singleton_lensing
            else SINGLETON_LENSING_OFF)
    return darksiren_log_likelihood_with_clusters(
        fx["cosmo"], fx["survey"],
        fx["pop_params"] if pop_params is None else pop_params,
        fx["gw_pe"], fx["catalog"], fx["gw_sel"], fx["catalog"],
        fx["n_events"], fx["n_samp"], fx["Ndraw"],
        singleton_indices=jnp.arange(fx["n_events"], dtype=jnp.int32),
        pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
        n_singletons=fx["n_events"], n_pairs=0,
        lensed_injections=None, pair_kdes=None,
        sis_params=make_sis_lens_params(A_tau=A_tau, n_tau=3.0),
        log_p_tag_per_source=jnp.zeros(0),
        pop_model="powerlaw+peak",
        universe_model="spectral_sirens",
        sel_batch_size=None,
        cluster_mode=CLUSTER_MODE_OFF,
        singleton_lensing=mode,
        lensed_singles=fx["singles"] if singleton_lensing else None,
        fc_pdet_params=fx["fc"] if singleton_lensing else None,
        y_nodes_single=16,
        return_diagnostics=return_diagnostics,
    )


def test_mixture_reduces_to_off_at_zero_tau(master_fixture):
    ll_off = float(_master_call(master_fixture, singleton_lensing=False, A_tau=0.0))
    ll_mix = float(_master_call(master_fixture, singleton_lensing=True, A_tau=0.0))
    np.testing.assert_allclose(ll_mix, ll_off, rtol=0, atol=1e-10)


def test_mixture_changes_likelihood_at_nonzero_tau(master_fixture):
    ll_off = float(_master_call(master_fixture, singleton_lensing=False, A_tau=5e-3))
    ll_mix = float(_master_call(master_fixture, singleton_lensing=True, A_tau=5e-3))
    assert np.isfinite(ll_mix)
    assert abs(ll_mix - ll_off) > 1e-8


def test_mixture_singleton_selection_monotonic_in_tau(master_fixture):
    # Fixture-specific sanity check: here the lensed-single term mu^(1L) grows
    # with A_tau faster than the (1 - tau_2) suppression shrinks the unlensed
    # part, so the combined mu_sel^(1) rises. This is NOT a physical guarantee
    # (the net sign depends on the lensed-vs-unlensed detection efficiency);
    # the Poisson-consistency guard is
    # test_mixture_singleton_selection_applies_tau_suppression below.
    mus = []
    for A_tau in (1e-4, 1e-3, 1e-2):
        diag = _master_call(
            master_fixture, singleton_lensing=True, A_tau=A_tau,
            return_diagnostics=True,
        )
        mus.append(float(diag["log_mu_singleton"]))
    assert mus[0] < mus[1] < mus[2]


def test_mixture_singleton_selection_applies_tau_suppression(master_fixture):
    """The (1 - tau_2) factor must suppress the UNLENSED part of mu_sel^(1).

    Poisson-consistency regression guard. The singleton selection integral is
    mu_sel^(1) = int (1 - tau_2) r_unlensed + mu^(1L); the unlensed part (the
    MIXTURE total minus the lensed-single term) must sit strictly BELOW the OFF
    selection integral, which carries no tau suppression. Buggy code that omits
    the factor makes the two exactly equal.
    """
    from darksirens.gw.populations import pop_model_parser
    from darksirens.redshift import get_redshift_prior

    fx = master_fixture
    A_tau = 1e-2  # large enough that the tau suppression is resolvable

    # Reconstruct the master's internal lensed-single selection term mu^(1L)
    # with the SAME public builders the master uses (pop_model_parser +
    # get_redshift_prior("spectral_sirens"); see likelihood_with_clusters.py
    # lines 229, 258-271, 355-358).
    log_p_pop = pop_model_parser(pop_model="powerlaw+peak")
    raw_sel = get_redshift_prior("spectral_sirens")
    cosmo, survey = fx["cosmo"], fx["survey"]

    def log_prior_z_selection(z, pix, catalog):
        return raw_sel(z, pix, cosmo, survey, catalog)

    sis = make_sis_lens_params(A_tau=A_tau, n_tau=3.0)
    log_mu_1L, _, _ = compute_lensed_single_selection_term(
        fx["singles"], cosmo, survey, fx["pop_params"], fx["catalog"],
        sis, log_p_pop, log_prior_z_selection,
    )
    log_mu_1L = float(log_mu_1L)

    log_mu_mix = float(_master_call(
        fx, singleton_lensing=True, A_tau=A_tau, return_diagnostics=True,
    )["log_mu_singleton"])
    log_mu_off = float(_master_call(
        fx, singleton_lensing=False, A_tau=A_tau, return_diagnostics=True,
    )["log_mu_singleton"])

    # Unlensed part of the MIXTURE selection = log(exp(mix) - exp(mu_1L)).
    assert log_mu_1L < log_mu_mix  # the sum exceeds either addend
    log_mu_unlensed_mix = log_mu_mix + float(
        jnp.log1p(-jnp.exp(jnp.asarray(log_mu_1L - log_mu_mix)))
    )
    assert np.isfinite(log_mu_unlensed_mix)  # reconstruction is sane
    # tau_2 > 0 strictly suppresses the unlensed selection below the OFF value
    # (which omits the factor). On the pre-fix code these are equal.
    assert log_mu_unlensed_mix < log_mu_off - 1e-6


def test_mixture_requires_channel_inputs(master_fixture):
    from darksirens.likelihood.likelihood_with_clusters import (
        darksiren_log_likelihood_with_clusters,
        CLUSTER_MODE_OFF, SINGLETON_LENSING_MIXTURE,
    )
    fx = master_fixture
    with pytest.raises(ValueError, match="singleton_lensing=MIXTURE requires"):
        darksiren_log_likelihood_with_clusters(
            fx["cosmo"], fx["survey"], fx["pop_params"],
            fx["gw_pe"], fx["catalog"], fx["gw_sel"], fx["catalog"],
            fx["n_events"], fx["n_samp"], fx["Ndraw"],
            singleton_indices=jnp.arange(fx["n_events"], dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=fx["n_events"], n_pairs=0,
            lensed_injections=None, pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=1e-3, n_tau=3.0),
            log_p_tag_per_source=jnp.zeros(0),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            cluster_mode=CLUSTER_MODE_OFF,
            singleton_lensing=SINGLETON_LENSING_MIXTURE,
        )


def test_mixture_gradient_wrt_population_finite(master_fixture):
    """d logL / d theta_pop must stay finite through the mixture channel —
    the pair/selection machinery's population coupling under a REAL
    powerlaw+peak vector (review coverage gap)."""
    fx = master_fixture

    def f(pop):
        return _master_call(fx, singleton_lensing=True, A_tau=1e-3, pop_params=pop)

    g = jax.grad(f)(fx["pop_params"])
    assert np.all(np.isfinite(np.asarray(g))), f"non-finite grad: {np.asarray(g)}"
