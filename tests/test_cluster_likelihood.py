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
    PairKDE, PAIR_KDE_COORDS, make_pair_kde, log_eval_pair_kde,
    _silverman_bandwidth_diag, validate_pair_prior_wt,
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

    # ---- padding invariance (P2-03) -----------------------------------

    @staticmethod
    def _padded_pair(n_real=3, n_pad=2, seed=11):
        """(unpadded KDE, KDE of the SAME samples padded with garbage).

        The padding rows carry NaN/inf coordinates and NaN/zero p_prop -- the
        shapes ``stack_pair_kdes`` produces when events have unequal N_pe.
        """
        rng = np.random.default_rng(seed)
        cols = [rng.normal(35.0, 3.0, n_real), rng.normal(0.7, 0.05, n_real),
                rng.normal(1000.0, 100.0, n_real), rng.normal(0.0, 0.1, n_real)]
        prior_wt = rng.uniform(0.5, 2.0, n_real)
        kde = make_pair_kde(*cols, prior_wt=prior_wt)

        garbage = [
            np.array([np.nan, 1.0e30])[:n_pad],
            np.array([np.nan, -5.0])[:n_pad],
            np.array([np.inf, 0.0])[:n_pad],
            np.array([np.nan, 7.0])[:n_pad],
        ]
        cols_p = [np.concatenate([c, g]) for c, g in zip(cols, garbage)]
        pw_p = np.concatenate([prior_wt, np.array([np.nan, 0.0])[:n_pad]])
        valid = np.array([True] * n_real + [False] * n_pad)
        kde_pad = make_pair_kde(*cols_p, prior_wt=pw_p, valid=valid)
        return kde, kde_pad

    def test_padding_does_not_change_the_density(self):
        """Padding 3 samples to 5 must not move the density.

        The evaluator masked the padded WEIGHTS but still divided by the padded
        LENGTH, so every log density of a padded event was offset by
        log(n_valid / N_pe) = -log(5/3) here -- a per-event constant that does
        NOT cancel in the pair Bayes factor.
        """
        kde, kde_pad = self._padded_pair()
        queries = jnp.asarray([
            [35.0, 0.7, 1000.0, 0.0],
            [40.0, 0.6, 1200.0, 0.1],
            [28.0, 0.9, 700.0, -0.2],
            [35.0, 0.99, 1000.0, 0.0],     # near the q = 1 reflection boundary
        ])
        got = np.asarray(log_eval_pair_kde(kde_pad, queries))
        ref = np.asarray(log_eval_pair_kde(kde, queries))
        assert np.all(np.isfinite(got)), f"NaN/inf leaked from padding: {got}"
        np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)
        # And the shift it used to have was exactly log(n_valid / N_pe).
        assert abs(np.log(3.0 / 5.0)) > 0.5

    def test_padding_does_not_poison_gradients(self):
        """NaN coordinates in invalid rows must not reach the backward pass.

        Masking only AFTER the component logpdf leaves the NaN on the
        differentiable path, and a NaN survives a zero cotangent (mul's VJP
        scales by the stored operand) -- the reverse-mode class documented at
        ``redshift/catalog.py:_logsumexp_neginf_safe``.
        """
        _, kde_pad = self._padded_pair()
        queries = jnp.asarray([[35.0, 0.7, 1000.0, 0.0], [40.0, 0.6, 1200.0, 0.1]])
        g = jax.grad(lambda t: jnp.sum(log_eval_pair_kde(kde_pad, t)))(queries)
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite gradient: {g}"

    def test_matches_scipy_gaussian_kde_on_unpadded_data(self):
        """Absolute normalisation check against ``scipy.stats.gaussian_kde``.

        scipy uses ``cov(data) * bw**2`` as its (full) bandwidth matrix, so we
        whiten the samples to an exactly-identity empirical covariance and hand
        scipy our Silverman factor: its bandwidth matrix is then exactly
        ``diag(h_k**2)``, matching PairKDE's diagonal bandwidth.  The q column
        is shifted far from the reflection boundary at q = 1 so the mirror
        kernel is numerically absent (its argument is ~ -100 sigma).
        """
        rng = np.random.default_rng(17)
        N, d = 300, 4
        X = rng.normal(size=(N, d)) @ rng.normal(size=(d, d)) + rng.normal(size=d)
        L = np.linalg.cholesky(np.cov(X.T, ddof=1))
        U = (X - X.mean(axis=0)) @ np.linalg.inv(L).T   # empirical cov == I
        U[:, PAIR_KDE_COORDS.index("q")] -= 60.0        # away from q = 1

        factor = (4.0 / ((d + 2) * N)) ** (1.0 / (d + 4))
        kde = make_pair_kde(*U.T, prior_wt=np.ones(N))
        np.testing.assert_allclose(np.exp(np.asarray(kde.log_h)), factor, rtol=1e-10)

        sp = gaussian_kde(U.T, bw_method=factor)
        queries = U[:5] + 0.3
        ours = np.exp(np.asarray(log_eval_pair_kde(kde, jnp.asarray(queries))))
        np.testing.assert_allclose(ours, sp(queries.T), rtol=1e-8)


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
    T_OBS = 365.25 * 86400.0
    good = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=true_dt, sigma_delta_t=sigma, t_obs_window_sec=T_OBS)
    bad = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=true_dt + 8.0 * sigma, sigma_delta_t=sigma, t_obs_window_sec=T_OBS)
    assert float(good) > float(bad)


def test_incompatible_time_delay_penalizes_shuffled_wrong_pair():
    ev_i, ev_j = _synth_lensed_pair(y_true=0.30, n_pe=160, seed=32)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    y_nodes, log_wy = make_y_grid(32)
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    sigma = 1.5e4
    T_OBS = 365.25 * 86400.0
    compatible = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=sis.T0 * 0.30, sigma_delta_t=sigma, t_obs_window_sec=T_OBS)
    shuffled = cluster_log_likelihood_pair(ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy, pair_marks=1, delta_t_obs=sis.T0 * 0.90, sigma_delta_t=sigma, t_obs_window_sec=T_OBS)
    assert float(compatible) > float(shuffled)


def test_time_delay_exceeding_window_is_finite_not_annihilated():
    """A valid SIS pair whose |dt| exceeds a short observing run must yield a
    FINITE, coincidence-favoured likelihood -- not the +inf sentinel that a
    downstream isfinite mask flips to -inf, silently ANNIHILATING the pair
    (a bias toward no-lensing). p_U(dt)=2(T-dt)/T^2 -> 0 as |dt| -> T; the fix
    clamps dt just below T so the odds stays large but finite."""
    from darksirens.likelihood.cluster_likelihood import PAIR_MARKS_DELTA_COLLAPSE
    ev_i, ev_j = _synth_lensed_pair(y_true=0.5, n_pe=160, seed=37)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    y_nodes, log_wy = make_y_grid(32)
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    T_OBS = 0.3 * float(sis.T0)          # short run: window below the SIS scale
    dt = 0.5 * float(sis.T0)             # y* = 0.5 in (0,1) but dt > T_OBS
    sigma = 2.0e4
    for mark in (1, PAIR_MARKS_DELTA_COLLAPSE):
        ll = cluster_log_likelihood_pair(
            ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1),
            _toy_catalog(), sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
            pair_marks=mark, delta_t_obs=jnp.asarray(dt),
            sigma_delta_t=jnp.asarray(sigma), t_obs_window_sec=T_OBS,
        )
        assert np.isfinite(float(ll)), f"pair_marks={mark}: ll={float(ll)}"


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


def test_time_delta_collapse_matches_dense_quadrature():
    """PAIR_MARKS_DELTA_COLLAPSE must reproduce the sharp-mark y-integral
    that the Gaussian quadrature path only recovers with a DENSE grid.
    Sharpness sigma/T0 ~ 0.008 (the production configuration that broke
    the G pilot at 64 nodes)."""
    from darksirens.likelihood.cluster_likelihood import PAIR_MARKS_DELTA_COLLAPSE

    y_true = 0.62
    ev_i, ev_j = _synth_lensed_pair(y_true=y_true, n_pe=120, seed=41)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    sigma = 3600.0
    dt_obs = float(sis.T0) * y_true

    T_OBS = 365.25 * 86400.0
    y_dense, log_wy_dense = make_y_grid(8192)
    ref = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_dense, log_wy_dense,
        pair_marks=1, delta_t_obs=jnp.asarray(dt_obs), sigma_delta_t=jnp.asarray(sigma),
        t_obs_window_sec=T_OBS,
    )
    y_any, log_wy_any = make_y_grid(16)  # ignored by the collapse
    collapsed = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_any, log_wy_any,
        pair_marks=PAIR_MARKS_DELTA_COLLAPSE,
        delta_t_obs=jnp.asarray(dt_obs), sigma_delta_t=jnp.asarray(sigma),
        t_obs_window_sec=T_OBS,
    )
    assert np.isfinite(float(collapsed))
    # Agreement to the O((sigma/T0)^2) collapse error + residual quadrature.
    np.testing.assert_allclose(float(collapsed), float(ref), rtol=0, atol=5e-3)

    # And the COARSE quadrature (the failing configuration) is measurably
    # wrong — this is the bug the collapse fixes. The size of the coarse
    # error depends on where y* falls relative to the node grid (0.2 nats
    # here; multiple nats in the G pilot when the spike is straddled), so
    # assert it exceeds an order of magnitude above the collapse tolerance.
    y64, log_wy64 = make_y_grid(64)
    coarse = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y64, log_wy64,
        pair_marks=1, delta_t_obs=jnp.asarray(dt_obs), sigma_delta_t=jnp.asarray(sigma),
        t_obs_window_sec=T_OBS,
    )
    assert abs(float(coarse) - float(ref)) > 0.05


def test_time_delta_collapse_annihilates_out_of_support():
    """A pair whose |Delta t| exceeds T0 has no SIS double solution: the
    collapsed mark must return exactly -inf (analytic false-edge kill)."""
    from darksirens.likelihood.cluster_likelihood import PAIR_MARKS_DELTA_COLLAPSE

    ev_i, ev_j = _synth_lensed_pair(y_true=0.4, n_pe=80, seed=42)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    y_nodes, log_wy = make_y_grid(16)
    out = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
        pair_marks=PAIR_MARKS_DELTA_COLLAPSE,
        delta_t_obs=jnp.asarray(40.0 * float(sis.T0)),  # months-scale separation
        sigma_delta_t=jnp.asarray(3600.0),
        t_obs_window_sec=365.25 * 86400.0,
    )
    assert float(out) == float("-inf")


def test_pair_time_mark_impl_resolution():
    """auto picks the delta collapse for sharp marks and quadrature for
    broad ones; explicit flags override."""
    from types import SimpleNamespace

    from darksirens.cli.inference_lensing import (
        _resolve_pair_marks, _sl_T0_seconds, _TIME_DELTA_SHARPNESS,
    )
    from darksirens.likelihood.likelihood_with_clusters import (
        PAIR_MARKS_NONE, PAIR_MARKS_TIME, PAIR_MARKS_TIME_DELTA,
    )

    def opts(marks="time", impl="auto"):
        return SimpleNamespace(pair_marks=marks, pair_time_mark_impl=impl,
                               sl_tau_A=5e-4, sl_tau_n=3.0)

    # Express the widths RELATIVE to T0: the rule is max(sigma)/T0 vs the
    # sharpness threshold, and T0 is a configurable lens-population scale.
    T0 = _sl_T0_seconds(opts())
    sharp_sig = 0.4 * _TIME_DELTA_SHARPNESS * T0
    broad_sig = 6.0 * _TIME_DELTA_SHARPNESS * T0
    sharp = {"pair_time_sigma": np.array([sharp_sig, sharp_sig])}
    broad = {"pair_time_sigma": np.array([sharp_sig, broad_sig])}
    assert _resolve_pair_marks(opts(), sharp) == PAIR_MARKS_TIME_DELTA
    assert _resolve_pair_marks(opts(), broad) == PAIR_MARKS_TIME
    assert _resolve_pair_marks(opts(impl="quadrature"), sharp) == PAIR_MARKS_TIME
    assert _resolve_pair_marks(opts(impl="delta"), broad) == PAIR_MARKS_TIME_DELTA
    assert _resolve_pair_marks(opts(marks="none"), sharp) == PAIR_MARKS_NONE
    assert _resolve_pair_marks(opts(), {"pair_time_sigma": []}) == PAIR_MARKS_TIME


def test_time_mark_coincidence_odds_reward_true_pairs():
    """A compatible days-scale delay in a year-long run must RAISE the pair
    likelihood relative to no time marks — by exactly
    ln p_L(dt) - ln p_U(dt) up to the collapse residual — instead of taxing
    it by ~ln T0 (the G-pilot failure)."""
    from darksirens.likelihood.cluster_likelihood import PAIR_MARKS_DELTA_COLLAPSE

    y_true = 0.55
    ev_i, ev_j = _synth_lensed_pair(y_true=y_true, n_pe=120, seed=43)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    sis = make_sis_lens_params(T0_seconds=4.32e5)
    T_OBS = 365.25 * 86400.0
    dt = float(sis.T0) * y_true
    y_nodes, log_wy = make_y_grid(64)

    no_marks = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
    )
    marked = cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
        pair_marks=PAIR_MARKS_DELTA_COLLAPSE,
        delta_t_obs=jnp.asarray(dt), sigma_delta_t=jnp.asarray(3600.0),
        t_obs_window_sec=T_OBS,
    )
    gain = float(marked) - float(no_marks)
    assert gain > 0.0, f"time mark should reward the true pair, got {gain}"
    # Expected: ln[(2 y*/T0) / (2 (T-dt)/T^2)] plus the (positive) effect of
    # pinning y at its true value rather than integrating over (0,1).
    floor = float(np.log((2 * y_true / float(sis.T0)) / (2 * (T_OBS - dt) / T_OBS**2)))
    assert gain > floor - 1.0, (gain, floor)


def _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, dt, *, signed,
                         mode, sigma=3600.0, t_obs=365.25 * 86400.0):
    from darksirens.likelihood.cluster_likelihood import PAIR_MARKS_DELTA_COLLAPSE

    y_nodes, log_wy = make_y_grid(64)
    return float(cluster_log_likelihood_pair(
        ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
        sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
        pair_marks=(PAIR_MARKS_DELTA_COLLAPSE if mode == "delta" else 1),
        delta_t_obs=jnp.asarray(dt), sigma_delta_t=jnp.asarray(sigma),
        t_obs_window_sec=t_obs, pair_time_signed=signed,
    ))


def test_signed_time_mark_removes_the_pair_double_count():
    """Both branches at |dt| counts a time-marked pair TWICE.

    For SIS the type-I minimum always arrives before the type-II saddle, so the
    two image-assignment branches predict opposite signs of the observed delay
    and at most one is compatible with the data. Evaluating both at |dt| made
    L_2 exactly 2x too large whenever the branches were comparable — a spurious
    +log 2 = 0.693 nat per time-marked pair toward pairing, and
    int L_2 = 2 mu_sel^(2), which breaks the Poisson normalization the master
    likelihood assumes. Two IDENTICAL events make the branches exactly equal,
    so the double-count is measurable to machine precision.
    """
    sis = make_sis_lens_params()
    y_true = 0.55
    dt = float(sis.T0) * y_true
    ev, _ = _synth_lensed_pair(y_true=y_true, n_pe=120, seed=43)
    kde = make_pair_kde(ev["m1det"], ev["q"], ev["dL"], ev["chieff"], ev["prior_wt"])

    unsigned = _time_marked_pair_ll(ev, ev, kde, kde, sis, dt,
                                    signed=False, mode="delta")
    signed = _time_marked_pair_ll(ev, ev, kde, kde, sis, dt,
                                  signed=True, mode="delta")
    assert np.isfinite(unsigned) and np.isfinite(signed)
    np.testing.assert_allclose(unsigned - signed, np.log(2.0), atol=1e-9)


def test_signed_time_mark_enforces_the_sis_arrival_order():
    """A pairing whose arrival order contradicts SIS must collapse to the
    ordering-consistent branch, not be rewarded at the ordering-INconsistent
    branch's (much larger) value.

    The synthetic pair has event i = mu_+ (brighter). If the recorded arrival
    order says i came SECOND (dt = t_j - t_i < 0), SIS requires j to be the
    mu_+ image, i.e. only branch b survives — and branch b is strongly
    disfavoured by the magnification data.
    """
    sis = make_sis_lens_params()
    y_true = 0.55
    dt = float(sis.T0) * y_true
    ev_i, ev_j = _synth_lensed_pair(y_true=y_true, n_pe=120, seed=43)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"],
                          ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"],
                          ev_j["prior_wt"])

    for mode, sigma in (("delta", 3600.0), ("quadrature", 3.0e5)):
        kw = dict(mode=mode, sigma=sigma)
        # Ordering CONSISTENT (brighter image first): unchanged — branch b was
        # already negligible, so there is no penalty for a determined pairing.
        ok_signed = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, dt,
                                         signed=True, **kw)
        ok_unsigned = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, dt,
                                           signed=False, **kw)
        np.testing.assert_allclose(ok_signed, ok_unsigned, atol=1e-8)

        # Ordering INCONSISTENT (brighter image second): the |dt| code returns
        # the SAME value as the consistent case — it never saw the order.
        bad_unsigned = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, -dt,
                                            signed=False, **kw)
        bad_signed = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, -dt,
                                          signed=True, **kw)
        np.testing.assert_allclose(bad_unsigned, ok_unsigned, atol=1e-8)
        over_reward = bad_unsigned - bad_signed
        assert over_reward > 20.0, (mode, over_reward)


def test_unsigned_time_marks_stay_sign_insensitive():
    """pair_time_signed=False keeps the legacy behaviour bit-for-bit: the mark
    enters only through |dt|, so flipping its sign changes nothing."""
    sis = make_sis_lens_params()
    dt = float(sis.T0) * 0.4
    ev_i, ev_j = _synth_lensed_pair(y_true=0.4, n_pe=80, seed=51)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"],
                          ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"],
                          ev_j["prior_wt"])
    for mode, sigma in (("delta", 3600.0), ("quadrature", 3.0e5)):
        pos = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, dt,
                                   signed=False, mode=mode, sigma=sigma)
        neg = _time_marked_pair_ll(ev_i, ev_j, kde_i, kde_j, sis, -dt,
                                   signed=False, mode=mode, sigma=sigma)
        assert pos == neg


def test_cli_orients_time_marks_from_catalog_arrival_times():
    """The candidate builder writes abs(t_i - t_j); the CLI restores the sign
    from the observed catalog's arrival times, keeping magnitude and sigma."""
    from darksirens.cli.inference_lensing import (
        _orient_time_marks, _event_gps_times_from_catalog,
    )
    from darksirens.lensing.partitions import CandidatePair, EdgeMarks

    pairs = [
        CandidatePair(0, 1, -1.0, None, EdgeMarks(delta_t_obs=100.0,
                                                  sigma_delta_t=10.0)),
        CandidatePair(1, 2, -1.0, None, EdgeMarks(delta_t_obs=300.0,
                                                  sigma_delta_t=10.0)),
    ]
    # t = [0, 100, -200]: edge (0,1) is forward, edge (1,2) is backward.
    catalog = {"events": [{"gps_time": 0.0}, {"gps_time": 100.0},
                          {"gps_time": -200.0}]}
    times = _event_gps_times_from_catalog(catalog)
    np.testing.assert_allclose(times, [0.0, 100.0, -200.0])

    oriented, signed = _orient_time_marks(pairs, times)
    assert signed
    assert oriented[0].delta_t_obs == 100.0
    assert oriented[1].delta_t_obs == -300.0
    assert oriented[0].sigma_delta_t == 10.0 and oriented[1].sigma_delta_t == 10.0

    # No arrival times -> unchanged, and the likelihood stays in legacy mode.
    unchanged, signed_none = _orient_time_marks(pairs, None)
    assert not signed_none
    assert unchanged is pairs
    # A catalog missing any gps_time yields no times at all.
    assert _event_gps_times_from_catalog(
        {"events": [{"gps_time": 0.0}, {}]}
    ) is None


def test_time_mark_requires_observing_window():
    ev_i, ev_j = _synth_lensed_pair(y_true=0.4, n_pe=60, seed=44)
    kde_i = make_pair_kde(ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"])
    kde_j = make_pair_kde(ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"])
    sis = make_sis_lens_params()
    y_nodes, log_wy = make_y_grid(16)
    with pytest.raises(ValueError, match="t_obs_window_sec"):
        cluster_log_likelihood_pair(
            ev_i, ev_j, kde_i, kde_j, _cosmo(), _survey(), jnp.zeros(1), _toy_catalog(),
            sis, _toy_log_p_pop, _toy_volume_prior, y_nodes, log_wy,
            pair_marks=1, delta_t_obs=jnp.asarray(1e5), sigma_delta_t=jnp.asarray(3600.0),
        )
