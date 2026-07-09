"""
test_cluster_likelihood.py
--------------------------
Unit + integration tests for commit 3:
  - darksirens/inference/pair_kde.py
  - darksirens/inference/cluster_likelihood.py

Test plan
~~~~~~~~~
A. PairKDE unit tests
   - Silverman bandwidth scales correctly with N
   - log_eval_pair_kde agrees with scipy gaussian_kde (modulo p_prop)
   - Importance correction recovers π_PE/p_prop when p_prop is non-uniform
   - JIT-compiles
B. cluster_log_likelihood_pair math tests
   - Numpy line-by-line oracle agreement
   - Branch symmetry: swapping i ↔ j gives same value
   - Magnification dependence: lower μ → higher inferred z_s → smaller log L
   - Vanishing for incompatible pairs (very different m1det in non-lensed limit)
C. JIT smoke test
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gaussian_kde

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import H0Planck, Om0Planck, dL_of_z
from darksirens.redshift.volume import log_volume_prior_vmap
from darksirens.likelihood.pair_kde import (
    PairKDE, make_pair_kde, log_eval_pair_kde, _silverman_bandwidth_diag,
    validate_pair_prior_wt,
)
from darksirens.likelihood.cluster_likelihood import (
    cluster_log_likelihood_pair, _log_jac_app_to_src,
)
from darksirens.lensing.slmarks import (
    SISLensParams, make_sis_lens_params, tau_2_SIS,
    mu_plus_minus_from_y, log_p_y_SIS,
)
from darksirens.lensing.grids import make_y_grid
from jax import lax


# ============================================================================
# Fixtures
# ============================================================================

def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(
        n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
    )


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


def _toy_log_p_pop(m1src, q, z, chieff, pop_params):
    """Same fixture population as commits 1-2."""
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


def _synth_lensed_pair(z_true=0.7, m1src_true=30.0, q_true=0.7,
                      chieff_true=0.0, y_true=0.4,
                      n_pe=300, seed=0):
    """Synthesize a clean strong-lensing pair: two events that ARE images
    of the same source. Used as ground truth in branch-symmetry checks.

    Returns (event_i_dict, event_j_dict) in apparent-frame coordinates
    with reasonable Gaussian PE samples around the true apparent values.
    """
    rng = np.random.default_rng(seed)
    H0, Om0 = H0Planck, Om0Planck
    dL_src = float(dL_of_z(jnp.asarray(z_true), H0, Om0))
    mu_p = (1.0 + y_true) / y_true
    mu_m = (1.0 - y_true) / y_true
    dL_app_i = dL_src / np.sqrt(mu_p)         # μ_+ image
    dL_app_j = dL_src / np.sqrt(mu_m)         # μ_- image
    m1det_true = (1.0 + z_true) * m1src_true

    # PE samples: tight Gaussian noise around the truth (10% PE width)
    def _sample(true_apparent_dL):
        m1det = true_apparent_dL * 0.0 + m1det_true + rng.normal(0.0, 1.0, n_pe)
        q = q_true + rng.normal(0.0, 0.05, n_pe)
        dL = true_apparent_dL + rng.normal(0.0, 0.05 * true_apparent_dL, n_pe)
        chieff = chieff_true + rng.normal(0.0, 0.03, n_pe)
        # Constant proposal density (uniform PE prior on the relevant box)
        prior_wt = np.ones(n_pe)
        valid = np.ones(n_pe, dtype=bool)
        return {
            "m1det": jnp.asarray(m1det), "q": jnp.asarray(q),
            "dL": jnp.asarray(dL), "chieff": jnp.asarray(chieff),
            "prior_wt": jnp.asarray(prior_wt), "valid": jnp.asarray(valid),
            "pixels": jnp.zeros(n_pe, dtype=jnp.int32),
        }
    return _sample(dL_app_i), _sample(dL_app_j)


# ============================================================================
# A. PairKDE unit tests
# ============================================================================

class TestPairKDE:
    def test_silverman_bandwidth_scales_with_N(self):
        """h ~ N^{-1/(d+4)} for d=4."""
        rng = np.random.default_rng(0)
        sigma = np.array([1.0, 0.1, 100.0, 0.2])
        h_ratios = []
        for N in [200, 2000]:
            samples = rng.normal(0.0, sigma, size=(N, 4))
            h = _silverman_bandwidth_diag(samples)
            # Check axis-by-axis: h_k / σ_k should match Silverman
            ratio = h / sigma
            expected = (4.0 / ((4 + 2) * N)) ** (1.0 / (4 + 4))
            np.testing.assert_allclose(ratio, expected, rtol=0.1)
            h_ratios.append(ratio[0])
        # The ratio should drop as N^{-1/8} when going from N=200 to N=2000
        expected_drop = (200.0 / 2000.0) ** (1.0 / 8.0)
        actual_drop = h_ratios[1] / h_ratios[0]
        # Tolerance reflects the ~7% sample-σ estimation noise at N=200;
        # the scaling law is exact, but its finite-sample test is not.
        np.testing.assert_allclose(actual_drop, expected_drop, rtol=0.10)

    def test_log_eval_matches_numpy_for_uniform_pprop(self):
        """When p_prop is constant, the KDE reduces to (constant) × π_PE.
        We test against a hand-written numpy KDE oracle with the same
        Silverman bandwidth, which removes any scipy-API-version pitfalls.
        """
        rng = np.random.default_rng(1)
        N = 500
        m1det = rng.normal(35.0, 3.0, N)
        q = np.clip(rng.normal(0.7, 0.05, N), 0.1, 1.0)
        dL = rng.normal(1000.0, 100.0, N)
        chieff = rng.normal(0.0, 0.1, N)
        p_prop_const = 0.5
        prior_wt = np.ones(N) * p_prop_const

        kde = make_pair_kde(m1det, q, dL, chieff, prior_wt)

        # Hand-written reference: π_PE(θ) = (1/N) Σ_t K_h(θ - θ_t).
        # The darksirens estimator returns π_PE / p_prop = π_PE / 0.5.
        samples = np.stack([m1det, q, dL, chieff], axis=-1)
        h = np.asarray(np.exp(kde.log_h))
        log_norm = -0.5 * 4 * np.log(2.0 * np.pi) - np.log(h).sum()

        queries = np.array([
            [35.0, 0.7, 1000.0, 0.0],
            [40.0, 0.6, 1200.0, 0.1],
            [30.0, 0.8, 800.0, -0.05],
        ])

        log_p_ref = np.empty(queries.shape[0])
        for i, q_pt in enumerate(queries):
            diffs_sq = np.sum(((q_pt - samples) / h) ** 2, axis=-1)
            log_kernel = -0.5 * diffs_sq
            from scipy.special import logsumexp as sp_lse
            # π_PE estimate; then multiply by 1/p_prop = 1/0.5 = 2
            log_pi_PE = log_norm + sp_lse(log_kernel) - np.log(N)
            log_p_ref[i] = log_pi_PE - np.log(p_prop_const)

        log_p_ours = log_eval_pair_kde(kde, jnp.asarray(queries))
        np.testing.assert_allclose(
            np.asarray(log_p_ours), log_p_ref, rtol=1e-10, atol=1e-12,
        )

    def test_log_eval_uses_provided_normalized_pprop_convention(self):
        """PairKDE must use loader-normalized prior_wt as provided.

        If an image has N samples with constant raw p_prop, the loader
        normalizes the per-image array to 1/N.  The KDE should therefore
        estimate π_PE / (1/N), without any additional self-normalization.
        """
        rng = np.random.default_rng(12)
        N = 400
        m1det = rng.normal(35.0, 3.0, N)
        q = np.clip(rng.normal(0.7, 0.05, N), 0.1, 1.0)
        dL = rng.normal(1000.0, 100.0, N)
        chieff = rng.normal(0.0, 0.1, N)
        prior_wt = np.ones(N) / N

        kde = make_pair_kde(m1det, q, dL, chieff, prior_wt)
        samples = np.stack([m1det, q, dL, chieff], axis=-1)
        h = np.asarray(np.exp(kde.log_h))
        log_norm = -0.5 * 4 * np.log(2.0 * np.pi) - np.log(h).sum()
        query = np.array([[35.0, 0.7, 1000.0, 0.0]])

        from scipy.special import logsumexp as sp_lse
        diffs_sq = np.sum(((query[0] - samples) / h) ** 2, axis=-1)
        log_pi_PE = log_norm + sp_lse(-0.5 * diffs_sq) - np.log(N)
        log_ref = log_pi_PE - np.log(1.0 / N)

        log_ours = np.asarray(log_eval_pair_kde(kde, jnp.asarray(query)))[0]
        np.testing.assert_allclose(log_ours, log_ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("bad_prior_wt", [np.zeros(4), np.full(4, np.nan)])
    def test_malformed_prior_wt_raises_useful_error(self, bad_prior_wt):
        with pytest.raises(ValueError, match="finite and positive"):
            validate_pair_prior_wt(bad_prior_wt, context="pair_0/image0/prior_wt")

    def test_importance_weights_correct_target(self):
        """With non-uniform p_prop, the KDE should estimate π_PE/p_prop.
        Test: draw samples from a known π_PE, assign them p_prop = π_PE
        scaled (so the *truth* is uniform), and check the KDE returns a flat
        function (modulo statistical noise) over the support.
        """
        rng = np.random.default_rng(2)
        N = 5000

        # Truth: π_PE is a single 4-D Gaussian; samples drawn from it.
        mu_true = np.array([35.0, 0.7, 1000.0, 0.0])
        sigma_true = np.array([3.0, 0.05, 100.0, 0.1])
        samples = mu_true + sigma_true * rng.normal(size=(N, 4))

        # p_prop = π_PE: each sample's prior_wt = π_PE(sample).
        log_pi = -0.5 * np.sum(((samples - mu_true) / sigma_true) ** 2, axis=-1) \
                 - 0.5 * 4 * np.log(2 * np.pi) - np.sum(np.log(sigma_true))
        prior_wt = np.exp(log_pi)

        kde = make_pair_kde(*samples.T, prior_wt=prior_wt)

        # Query at the centre and at moderate offsets; π_PE/p_prop = 1
        # everywhere within the bandwidth, so log_p ≈ 0 within statistical
        # noise of the KDE estimator.
        queries = np.array([
            mu_true,
            mu_true + sigma_true * np.array([0.5, 0.5, 0.5, 0.5]),
            mu_true + sigma_true * np.array([-0.5, -0.5, -0.5, -0.5]),
        ])
        log_p = np.asarray(log_eval_pair_kde(kde, jnp.asarray(queries)))
        # Should all be approximately equal (constant), within a factor of
        # a few from MC noise. The mean absolute deviation across queries
        # should be small.
        spread = log_p.max() - log_p.min()
        assert spread < 0.3, (
            f"KDE of constant target has range {spread:.3f} over queries — "
            f"expected < 0.3 statistical noise"
        )

    def test_jit_compiles(self):
        rng = np.random.default_rng(3)
        N = 200
        kde = make_pair_kde(
            m1det=rng.normal(35, 3, N),
            q=rng.normal(0.7, 0.05, N),
            dL_app=rng.normal(1000, 100, N),
            chieff=rng.normal(0, 0.1, N),
            prior_wt=np.ones(N),
        )
        jit_eval = jax.jit(log_eval_pair_kde)
        queries = jnp.asarray([[35.0, 0.7, 1000.0, 0.0]])
        out = jit_eval(kde, queries)
        assert out.shape == (1,)
        assert jnp.isfinite(out[0])


# ============================================================================
# B. cluster_log_likelihood_pair math tests
# ============================================================================

class TestJacobian:
    def test_log_jac_app_to_src_sign_and_formula(self):
        """log|J_app→src| = -log(1+z) - log dL'(z) + 0.5 log μ."""
        from darksirens.utils.cosmology import ddL_of_z
        H0, Om0 = H0Planck, Om0Planck
        z = jnp.array([0.3, 0.7, 1.5])
        dL_true = jnp.array([1500.0, 4000.0, 10000.0])
        mu = jnp.array([2.0, 5.0, 0.5])

        actual = _log_jac_app_to_src(z, dL_true, mu, H0, Om0)
        expected = (
            -jnp.log1p(z) - jnp.log(ddL_of_z(z, dL_true, H0, Om0))
            + 0.5 * jnp.log(mu)
        )
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-12,
        )

    def test_log_jac_unlensed_inverse(self):
        """At μ=1, log|J_app→src| should be exactly the negative of the
        commit-2 src→app Jacobian."""
        from darksirens.inference.utils import log_jacobian_m1src_q_z_to_m1det_q_dL
        H0, Om0 = H0Planck, Om0Planck
        z = jnp.array([0.5, 1.0])
        dL = jnp.array([3000.0, 6800.0])
        mu = jnp.ones_like(z)

        app_to_src = _log_jac_app_to_src(z, dL, mu, H0, Om0)
        src_to_app = log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL, H0, Om0)
        np.testing.assert_allclose(
            np.asarray(app_to_src), -np.asarray(src_to_app), rtol=1e-12,
        )


class TestPairLikelihood:

    @pytest.fixture(scope="class")
    def setup(self):
        ev_i, ev_j = _synth_lensed_pair(
            z_true=0.7, m1src_true=30.0, q_true=0.7,
            chieff_true=0.0, y_true=0.4, n_pe=200, seed=0,
        )
        kde_i = make_pair_kde(
            ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"],
            ev_i["prior_wt"],
        )
        kde_j = make_pair_kde(
            ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"],
            ev_j["prior_wt"],
        )
        y_nodes, log_wy = make_y_grid(16)
        return {
            "ev_i": ev_i, "ev_j": ev_j,
            "kde_i": kde_i, "kde_j": kde_j,
            "y_nodes": y_nodes, "log_wy": log_wy,
            "sis_params": make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
            "cosmo": _cosmo(), "survey": _survey(),
            "catalog": _toy_catalog(),
            "pop_params": jnp.array([]),
        }

    def test_pair_likelihood_finite_for_lensed_pair(self, setup):
        """For a synthetic genuine lensed pair, log L_2 must be finite."""
        ll = cluster_log_likelihood_pair(
            setup["ev_i"], setup["ev_j"],
            setup["kde_i"], setup["kde_j"],
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        assert jnp.isfinite(ll), f"log L_2 not finite: {ll}"

    def test_branch_symmetry(self, setup):
        """log L_2(i, j) == log L_2(j, i) — the estimator is symmetric in
        the two events by construction (we symmetrize over assignments
        internally)."""
        ll_ij = cluster_log_likelihood_pair(
            setup["ev_i"], setup["ev_j"],
            setup["kde_i"], setup["kde_j"],
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        ll_ji = cluster_log_likelihood_pair(
            setup["ev_j"], setup["ev_i"],
            setup["kde_j"], setup["kde_i"],
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        np.testing.assert_allclose(float(ll_ij), float(ll_ji), rtol=1e-10)

    def test_pair_disfavored_when_m1det_strongly_incompatible(self, setup):
        """An (i, j) pair with very different detector-frame masses cannot
        be SIS doubles of the same source: m1_det is INVARIANT under
        magnification (μ rescales d_L but not m_det), so the two events
        must agree on m1_det within PE width. A 10σ_PE offset must give
        a much lower L_2 than the true lensed pair.
        """
        ev_i = setup["ev_i"]
        ev_j_wrong_mass = {
            "m1det": setup["ev_j"]["m1det"] + 30.0,  # ~30 M_sun offset, ~30σ_PE
            "q": setup["ev_j"]["q"],
            "dL": setup["ev_j"]["dL"],
            "chieff": setup["ev_j"]["chieff"],
            "prior_wt": setup["ev_j"]["prior_wt"],
            "valid": setup["ev_j"]["valid"],
            "pixels": setup["ev_j"]["pixels"],
        }
        kde_j_wrong = make_pair_kde(
            ev_j_wrong_mass["m1det"], ev_j_wrong_mass["q"],
            ev_j_wrong_mass["dL"], ev_j_wrong_mass["chieff"],
            ev_j_wrong_mass["prior_wt"],
        )
        ll_good = cluster_log_likelihood_pair(
            ev_i, setup["ev_j"],
            setup["kde_i"], setup["kde_j"],
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        ll_bad = cluster_log_likelihood_pair(
            ev_i, ev_j_wrong_mass,
            setup["kde_i"], kde_j_wrong,
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        assert ll_good - ll_bad > 5.0, (
            f"m1det-incompatible pair was not strongly disfavored: "
            f"good={ll_good}, bad={ll_bad}, diff={float(ll_good - ll_bad)}"
        )

    def test_pair_disfavored_when_chieff_strongly_incompatible(self, setup):
        """Two events with very different χ_eff cannot be images of the same
        source (χ_eff is the same in apparent and source frame), so L_2
        must drop.
        """
        ev_i = setup["ev_i"]
        ev_j_wrong_chi = {
            "m1det": setup["ev_j"]["m1det"],
            "q": setup["ev_j"]["q"],
            "dL": setup["ev_j"]["dL"],
            "chieff": setup["ev_j"]["chieff"] + 0.5,  # huge offset
            "prior_wt": setup["ev_j"]["prior_wt"],
            "valid": setup["ev_j"]["valid"],
            "pixels": setup["ev_j"]["pixels"],
        }
        kde_j_wrong = make_pair_kde(
            ev_j_wrong_chi["m1det"], ev_j_wrong_chi["q"],
            ev_j_wrong_chi["dL"], ev_j_wrong_chi["chieff"],
            ev_j_wrong_chi["prior_wt"],
        )
        ll_good = cluster_log_likelihood_pair(
            ev_i, setup["ev_j"],
            setup["kde_i"], setup["kde_j"],
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        ll_bad = cluster_log_likelihood_pair(
            ev_i, ev_j_wrong_chi,
            setup["kde_i"], kde_j_wrong,
            setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
            setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
            setup["y_nodes"], setup["log_wy"],
        )
        assert ll_good - ll_bad > 5.0, (
            f"χ_eff-incompatible pair was not strongly disfavored: "
            f"good={ll_good}, bad={ll_bad}, diff={float(ll_good - ll_bad)}"
        )

    def test_jit_compiles(self, setup):
        """The pair likelihood JIT-compiles cleanly."""
        jit_fn = jax.jit(
            lambda ev_i, ev_j, kde_i, kde_j: cluster_log_likelihood_pair(
                ev_i, ev_j, kde_i, kde_j,
                setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
                setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
                setup["y_nodes"], setup["log_wy"],
            )
        )
        out = jit_fn(setup["ev_i"], setup["ev_j"], setup["kde_i"], setup["kde_j"])
        assert jnp.isfinite(out)


# ============================================================================
# C. Numpy reference oracle
# ============================================================================

class TestNumpyReference:
    """A from-scratch numpy implementation of the pair likelihood for one
    branch (i→μ_+, j→μ_-) using fewer PE samples, used as a math oracle."""

    @staticmethod
    def _ref_branch(ev_i, kde_j, sis_params, y_nodes, log_wy, cosmo):
        """Numpy implementation of branch a, returning log_branch_a."""
        from darksirens.utils.cosmology import z_of_dL, ddL_of_z, dL_of_z, dL_in_z_grid
        H0, Om0 = float(cosmo.H0), float(cosmo.Om0)
        y = np.asarray(y_nodes)
        N_y = len(y)
        N_i = ev_i["m1det"].shape[0]
        mu_p = (1.0 + y) / y
        mu_m = (1.0 - y) / y
        log_py = np.log(2.0 * y)
        log_wy_np = np.asarray(log_wy)

        log_per_cell = np.full((N_i, N_y), -np.inf)
        for s in range(N_i):
            m1d_s = float(ev_i["m1det"][s])
            q_s = float(ev_i["q"][s])
            dL_s = float(ev_i["dL"][s])
            chi_s = float(ev_i["chieff"][s])
            pw_s = float(ev_i["prior_wt"][s])
            if not (pw_s > 0.0 and bool(ev_i["valid"][s])):
                continue
            for ell in range(N_y):
                dL_true = dL_s * np.sqrt(mu_p[ell])
                if not bool(dL_in_z_grid(jnp.asarray(dL_true), H0, Om0)):
                    continue
                z_s = float(z_of_dL(jnp.asarray(dL_true), H0, Om0))
                if not np.isfinite(z_s):
                    continue
                m1src = m1d_s / (1.0 + z_s)
                # Predict apparent event-j params
                dL_src = float(dL_of_z(jnp.asarray(z_s), H0, Om0))
                dL_app_j = dL_src / np.sqrt(mu_m[ell])
                m1det_j = (1.0 + z_s) * m1src
                # Evaluate KDE_j at this point
                theta_j = jnp.asarray([m1det_j, q_s, dL_app_j, chi_s])
                log_p_j = float(log_eval_pair_kde(kde_j, theta_j[None, :])[0])
                # Population & redshift prior
                lp_pop = float(_toy_log_p_pop(
                    jnp.asarray(m1src), jnp.asarray(q_s), jnp.asarray(z_s),
                    jnp.asarray(chi_s), jnp.asarray([]),
                ))
                lp_z = float(_toy_volume_prior(
                    jnp.asarray([z_s]), jnp.asarray([0], dtype=jnp.int32), None,
                )[0])
                # Optical depth
                log_tau = float(jnp.log(tau_2_SIS(jnp.asarray(z_s), sis_params)))
                # Jacobian app→src
                ddL = float(ddL_of_z(jnp.asarray(z_s),
                                     jnp.asarray(dL_true), H0, Om0))
                log_J = -np.log1p(z_s) - np.log(ddL) + 0.5 * np.log(mu_p[ell])
                # Per-cell log integrand
                log_per_cell[s, ell] = (
                    lp_pop + lp_z + log_tau + log_p_j + log_J
                    - np.log(pw_s) + log_py[ell] + log_wy_np[ell]
                )
        # Sum
        finite = log_per_cell[np.isfinite(log_per_cell)]
        if finite.size == 0:
            return -np.inf
        a_max = finite.max()
        return float(a_max + np.log(np.sum(np.exp(finite - a_max)))) - np.log(N_i)

    def test_jax_branch_matches_numpy(self):
        """The internal _pair_branch_log_integrand vectorization must match
        the numpy reference line-by-line."""
        from darksirens.likelihood.cluster_likelihood import _pair_branch_log_integrand
        ev_i, ev_j = _synth_lensed_pair(n_pe=50, seed=42)
        kde_i = make_pair_kde(
            ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"],
        )
        kde_j = make_pair_kde(
            ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"],
        )
        y_nodes, log_wy = make_y_grid(16)
        sis_params = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        cosmo = _cosmo()
        survey = _survey()
        catalog = _toy_catalog()

        from jax.scipy.special import logsumexp

        mu_p, mu_m = mu_plus_minus_from_y(y_nodes)
        log_py = log_p_y_SIS(y_nodes)
        log_int_jax = _pair_branch_log_integrand(
            m1det_i=ev_i["m1det"], q_i=ev_i["q"], dL_app_i=ev_i["dL"],
            chieff_i=ev_i["chieff"], prior_wt_i=ev_i["prior_wt"],
            valid_i=ev_i["valid"], pix_i=ev_i["pixels"],
            mu_i=mu_p, mu_j=mu_m,
            log_py=log_py, log_wy=log_wy,
            kde_j=kde_j,
            cosmo=cosmo, survey=survey, pop_params=jnp.array([]), catalog=catalog,
            sis_params=sis_params,
            log_p_pop_fn=_toy_log_p_pop, log_prior_z_fn=_toy_volume_prior,
        )
        N_i = ev_i["m1det"].shape[0]
        log_branch_jax = float(logsumexp(log_int_jax) - jnp.log(N_i))
        log_branch_np = self._ref_branch(ev_i, kde_j, sis_params, y_nodes, log_wy, cosmo)

        np.testing.assert_allclose(log_branch_jax, log_branch_np, rtol=1e-9, atol=1e-12)


def test_pair_marks_none_matches_legacy_likelihood():
    ev_i, ev_j = _synth_lensed_pair(y_true=0.4, n_pe=120, seed=30)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    y_nodes, log_wy = make_y_grid(16)
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    base = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy)
    marked_none = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=0, delta_t_obs=jnp.asarray(1.0), sigma_delta_t=jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(marked_none), np.asarray(base), rtol=0, atol=0)


def test_true_time_delay_increases_pair_likelihood():
    y_true = 0.45
    ev_i, ev_j = _synth_lensed_pair(y_true=y_true, n_pe=160, seed=31)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    y_nodes, log_wy = make_y_grid(32)
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    true_dt = sis.T0 * y_true
    sigma = 2.0e4
    good = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=true_dt, sigma_delta_t=sigma)
    bad = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=true_dt + 8.0 * sigma, sigma_delta_t=sigma)
    assert float(good) > float(bad)


def test_incompatible_time_delay_penalizes_shuffled_wrong_pair():
    ev_i, ev_j = _synth_lensed_pair(y_true=0.30, n_pe=160, seed=32)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    y_nodes, log_wy = make_y_grid(32)
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    sigma = 1.5e4
    compatible = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=sis.T0 * 0.30, sigma_delta_t=sigma)
    shuffled = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=sis.T0 * 0.90, sigma_delta_t=sigma)
    assert float(compatible) > float(shuffled)


def _pair_value_for_batch_test(ev_i, ev_j):
    kde_i = make_pair_kde(
        np.asarray(ev_i["m1det"]), np.asarray(ev_i["q"]), np.asarray(ev_i["dL"]),
        np.asarray(ev_i["chieff"]), np.asarray(ev_i["prior_wt"]), np.asarray(ev_i["valid"]),
    )
    kde_j = make_pair_kde(
        np.asarray(ev_j["m1det"]), np.asarray(ev_j["q"]), np.asarray(ev_j["dL"]),
        np.asarray(ev_j["chieff"]), np.asarray(ev_j["prior_wt"]), np.asarray(ev_j["valid"]),
    )
    y_nodes, log_wy = make_y_grid(16)
    return cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        make_sis_lens_params(), _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
    )


def _batched_pair_sum_for_test(pair_values, n_pairs, pair_batch_size):
    """Mirror the production padded lax.scan summation used for pair batching."""
    n_batches = (n_pairs + pair_batch_size - 1) // pair_batch_size
    n_padded = n_batches * pair_batch_size

    def _scan_pair(carry, k):
        active = k < n_pairs
        ll_pair = lax.cond(
            active,
            lambda kk: pair_values[kk],
            lambda _: jnp.asarray(0.0, dtype=jnp.float64),
            k,
        )
        return carry + ll_pair, jnp.where(active, ll_pair, -jnp.inf)

    return lax.scan(_scan_pair, jnp.asarray(0.0, dtype=jnp.float64), jnp.arange(n_padded))


def test_pair_batch_scan_matches_unbatched_for_small_mock():
    pairs = [_synth_lensed_pair(n_pe=64, seed=seed) for seed in (101, 102, 103)]
    pair_values = jnp.asarray([_pair_value_for_batch_test(a, b) for a, b in pairs])
    unbatched = jnp.sum(pair_values)
    batched, per_pair = _batched_pair_sum_for_test(pair_values, n_pairs=3, pair_batch_size=2)
    np.testing.assert_allclose(np.asarray(batched), np.asarray(unbatched), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(per_pair[:3]), np.asarray(pair_values), rtol=1e-12, atol=1e-12)


def test_pair_batch_size_larger_than_n_pairs_matches_unbatched():
    pairs = [_synth_lensed_pair(n_pe=64, seed=seed) for seed in (201, 202)]
    pair_values = jnp.asarray([_pair_value_for_batch_test(a, b) for a, b in pairs])
    unbatched = jnp.sum(pair_values)
    batched, _ = _batched_pair_sum_for_test(pair_values, n_pairs=2, pair_batch_size=8)
    np.testing.assert_allclose(np.asarray(batched), np.asarray(unbatched), rtol=1e-12, atol=1e-12)


def test_pair_batch_size_one_matches_unbatched():
    pairs = [_synth_lensed_pair(n_pe=64, seed=seed) for seed in (301, 302, 303)]
    pair_values = jnp.asarray([_pair_value_for_batch_test(a, b) for a, b in pairs])
    unbatched = jnp.sum(pair_values)
    batched, _ = _batched_pair_sum_for_test(pair_values, n_pairs=3, pair_batch_size=1)
    np.testing.assert_allclose(np.asarray(batched), np.asarray(unbatched), rtol=1e-12, atol=1e-12)


def test_pair_kde_boundary_reflection_doubles_at_q_one():
    """The q<=1 reflection estimator returns exactly log(2) more than the
    unreflected kernel sum when queried at the boundary q=1 (each sample's
    mirror kernel equals its direct kernel there). This is the fix for the
    ~2x density deficit of a plain Gaussian KDE at the equal-mass boundary."""
    from scipy.special import logsumexp as sp_lse

    rng = np.random.default_rng(7)
    N = 300
    m1det = rng.normal(35.0, 3.0, N)
    q = 1.0 - np.abs(rng.normal(0.0, 0.03, N))  # pile-up near q = 1
    dL = rng.normal(1000.0, 100.0, N)
    chieff = rng.normal(0.0, 0.1, N)
    kde = make_pair_kde(m1det, q, dL, chieff, np.ones(N))

    query = np.array([[35.0, 1.0, 1000.0, 0.0]])
    samples = np.stack([m1det, q, dL, chieff], axis=-1)
    h = np.asarray(np.exp(kde.log_h))
    log_norm = -0.5 * 4 * np.log(2.0 * np.pi) - np.log(h).sum()
    diffs_sq = np.sum(((query[0] - samples) / h) ** 2, axis=-1)
    log_unreflected = log_norm + sp_lse(-0.5 * diffs_sq) - np.log(N)

    log_ours = np.asarray(log_eval_pair_kde(kde, jnp.asarray(query)))[0]
    np.testing.assert_allclose(log_ours, log_unreflected + np.log(2.0), rtol=1e-12)


def test_pair_kde_reflection_matches_mirrored_oracle_near_boundary():
    """Full oracle check: the estimator must equal a numpy KDE evaluated with
    the sample set augmented by its q-mirror about 1 (same weights, same
    bandwidth, kernel mass restricted to the direct+mirror pair)."""
    from scipy.special import logsumexp as sp_lse

    rng = np.random.default_rng(11)
    N = 250
    m1det = rng.normal(35.0, 3.0, N)
    q = np.clip(rng.uniform(0.85, 1.0, N), None, 1.0)
    dL = rng.normal(1000.0, 100.0, N)
    chieff = rng.normal(0.0, 0.1, N)
    p_prop = rng.uniform(0.5, 2.0, N)
    kde = make_pair_kde(m1det, q, dL, chieff, p_prop)

    queries = np.array([
        [35.0, 0.999, 1000.0, 0.0],
        [34.0, 0.97, 950.0, 0.05],
        [36.0, 0.90, 1100.0, -0.05],
    ])
    samples = np.stack([m1det, q, dL, chieff], axis=-1)
    mirrored = samples.copy()
    mirrored[:, 1] = 2.0 - mirrored[:, 1]
    h = np.asarray(np.exp(kde.log_h))
    log_norm = -0.5 * 4 * np.log(2.0 * np.pi) - np.log(h).sum()
    log_w = -np.log(p_prop)

    log_ref = np.empty(queries.shape[0])
    for k, q_pt in enumerate(queries):
        direct = -0.5 * np.sum(((q_pt - samples) / h) ** 2, axis=-1)
        mirror = -0.5 * np.sum(((q_pt - mirrored) / h) ** 2, axis=-1)
        log_kernel = np.logaddexp(direct, mirror)
        log_ref[k] = log_norm + sp_lse(log_kernel + log_w) - np.log(N)

    log_ours = np.asarray(log_eval_pair_kde(kde, jnp.asarray(queries)))
    np.testing.assert_allclose(log_ours, log_ref, rtol=1e-12, atol=1e-12)


class TestJacobianClamp:
    def test_log_jac_finite_when_ddL_nonpositive(self, monkeypatch):
        """Exotic CPL corners can make dL(z) non-monotonic (ddL <= 0); the
        Jacobian must clamp before the log so the value stays finite instead
        of NaN-poisoning cells the caller later masks out."""
        import darksirens.likelihood.cluster_likelihood as cl

        def bad_ddL(z, dL, H0, Om0, w0=-1.0, wa=0.0):
            return jnp.zeros_like(jnp.asarray(z)) - 1.0

        monkeypatch.setattr(cl, "ddL_of_z", bad_ddL)
        out = _log_jac_app_to_src(
            jnp.asarray(0.5), jnp.asarray(1000.0), jnp.asarray(2.0),
            H0Planck, Om0Planck,
        )
        assert np.isfinite(np.asarray(out))

    def test_log_jac_gradient_finite_through_clamp(self, monkeypatch):
        """Reverse-mode gradients through the clamped branch must be finite
        (the pre-fix log(negative) produced NaN gradients even for masked
        cells)."""
        import darksirens.likelihood.cluster_likelihood as cl

        def bad_ddL(z, dL, H0, Om0, w0=-1.0, wa=0.0):
            return z - 10.0  # negative at any physical z

        monkeypatch.setattr(cl, "ddL_of_z", bad_ddL)
        grad = jax.grad(
            lambda z: _log_jac_app_to_src(
                z, jnp.asarray(1000.0), jnp.asarray(2.0), H0Planck, Om0Planck
            )
        )(jnp.asarray(0.5))
        assert np.isfinite(np.asarray(grad))

    def test_log_jac_unchanged_for_physical_cosmology(self):
        """The clamp is inert on the physical branch."""
        from darksirens.utils.cosmology import ddL_of_z

        z = jnp.array([0.3, 0.7, 1.5])
        dL_true = jnp.array([1500.0, 4000.0, 10000.0])
        mu = jnp.array([2.0, 5.0, 0.5])
        actual = _log_jac_app_to_src(z, dL_true, mu, H0Planck, Om0Planck)
        expected = (
            -jnp.log1p(z) - jnp.log(ddL_of_z(z, dL_true, H0Planck, Om0Planck))
            + 0.5 * jnp.log(mu)
        )
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)
