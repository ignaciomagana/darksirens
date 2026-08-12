"""
test_lensing.py
---------------
Unit tests for darksirens.lensing.

Coverage
~~~~~~~~
1. Weak-lensing PDF:
   - flux conservation (⟨μ⟩ = 1) for the lognormal backend
   - normalization (∫ p_WL(μ|z) dμ = 1) for both backends
   - tabulated backend agrees with lognormal at the grid points
   - JIT compilation succeeds; vmap over (μ, z) works

2. Strong-lensing marks:
   - τ_2 monotonic non-decreasing, non-negative, vectorizes
   - p(y) integrates to 1 on (0, 1)
   - mu_+ − μ_- = 2 identity holds for all y ∈ (0, 1)
   - y_from_mu_plus inverts mu_plus_minus_from_y
   - τ_4 returns zero in commit 1 (inert hook)
   - Δt(y) linear in y

3. Cluster I/O:
   - empty set round-trips
   - non-empty set round-trips
   - invalid ordering raises
   - out-of-range indices raise
   - mismatched BF lengths raise

4. Grids:
   - Gauss-Legendre over a Gaussian integrand → 1 to high precision
   - Gauss-Legendre over p(y) = 2y on (0,1) → 1 to high precision
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Make package importable for in-tree testing
HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.lensing import (
    WLParams,
    make_lognormal_wl_params,
    make_tabulated_wl_params,
    log_p_wl,
    SISLensParams,
    make_sis_lens_params,
    tau_2_SIS,
    log_p_y_SIS,
    mu_plus_minus_from_y,
    y_from_mu_plus,
    delta_t_from_y,
    ClusterSet,
    load_clusters,
    save_clusters,
    make_log_mu_grid,
    make_y_grid,
)
from darksirens.lensing.slmarks import tau_4_SIS
from darksirens.lensing.clusters import make_cluster_set, empty_cluster_set


# ============================================================================
# Grids — basic numerical fidelity
# ============================================================================

class TestGrids:
    def test_log_mu_grid_integrates_gaussian_to_one(self):
        """Gauss-Legendre over a normalized Gaussian in ln μ → 1."""
        mu_nodes, log_w = make_log_mu_grid(n_nodes=32, log_mu_range=(-4.0, 4.0))
        # p(ln μ) = N(0, 1) — already normalized over R; over our finite
        # interval we expect close to 1.
        log_mu = jnp.log(mu_nodes)
        log_pdf = -0.5 * log_mu**2 - 0.5 * jnp.log(2.0 * jnp.pi)
        log_integral = logsumexp(log_pdf + log_w)
        integral = float(jnp.exp(log_integral))
        # Cuts at ±4σ trim ~6e-5 of the total mass.
        assert abs(integral - 1.0) < 1e-4

    def test_y_grid_integrates_p_y_to_one(self):
        """∫_0^1 2y dy = 1."""
        y_nodes, log_w = make_y_grid(n_nodes=16)
        log_p = jnp.log(2.0 * y_nodes)
        log_integral = logsumexp(log_p + log_w)
        integral = float(jnp.exp(log_integral))
        assert abs(integral - 1.0) < 1e-12

    def test_log_mu_grid_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            make_log_mu_grid(n_nodes=1)
        with pytest.raises(ValueError):
            make_log_mu_grid(n_nodes=8, log_mu_range=(0.5, -0.5))

    def test_y_grid_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            make_y_grid(n_nodes=1)


# ============================================================================
# Weak-lensing PDF
# ============================================================================

class TestWLLognormal:
    """Analytic checks against the canonical lognormal moments.

    The PDF is constructed so that:
        Var(ln μ) = s²(z) = a · z^b
        E(ln μ)   = -s²/2
        E(μ)      = exp(E(ln μ) + s²/2) = 1   (flux conservation)

    Below we (i) cross-check the log-PDF against scipy at sample points
    (the strongest assertion: implementation matches a trusted oracle),
    and (ii) verify numerical moments by quadrature for **wide enough**
    distributions that 64 Gauss-Legendre nodes resolve them.

    Narrow distributions (small s) are exercised by the scipy check;
    numerical moments at small s are meaningless because Gauss-Legendre
    on a fixed wide interval cannot resolve a near-δ-function.
    """

    def test_flux_conservation_when_s_large(self):
        """⟨μ⟩ = 1 verified numerically when Gauss-Legendre can resolve the PDF.

        We pick s² ∈ [0.14, 0.28] so that 200 Gauss-Legendre nodes on
        (-4, 4) cover ~7σ (tail clipping < 10⁻¹¹) and resolve the core
        at < 0.1 σ per node. The flux-conservation property is then a
        meaningful numerical assertion at the 10⁻³ level.
        """
        p = make_lognormal_wl_params(a=0.2, b=0.5)
        mu_nodes, log_w = make_log_mu_grid(n_nodes=200, log_mu_range=(-4.0, 4.0))
        for z_test in [0.5, 1.0, 2.0]:
            z = jnp.full_like(mu_nodes, z_test)
            log_pdf = log_p_wl(mu_nodes, z, p)
            # ⟨μ⟩ = ∫ μ p(μ) dμ. Quadrature is in ln μ:
            # dμ = μ d(ln μ) → integrand = μ² · p_WL(μ).
            log_integrand = 2.0 * jnp.log(mu_nodes) + log_pdf + log_w
            mean_mu = float(jnp.exp(logsumexp(log_integrand)))
            assert abs(mean_mu - 1.0) < 1e-3, (
                f"⟨μ⟩ = {mean_mu} at z={z_test} (expected 1.0)"
            )

    def test_normalization_when_s_large(self):
        """∫ p_WL(μ|z) dμ = 1, with parameters wide enough for the grid."""
        p = make_lognormal_wl_params(a=0.2, b=0.5)
        mu_nodes, log_w = make_log_mu_grid(n_nodes=200, log_mu_range=(-4.0, 4.0))
        for z_test in [0.5, 1.0, 2.0]:
            z = jnp.full_like(mu_nodes, z_test)
            log_pdf = log_p_wl(mu_nodes, z, p)
            # ∫ p(μ) dμ = ∫ p(μ) · μ d(ln μ) → integrand = μ · p_WL(μ)
            log_integrand = log_pdf + jnp.log(mu_nodes) + log_w
            integral = float(jnp.exp(logsumexp(log_integrand)))
            assert abs(integral - 1.0) < 1e-3, (
                f"∫ p_WL dμ = {integral} at z={z_test} (expected 1.0)"
            )

    def test_variance_analytic(self):
        """Var(ln μ) = a · z^b numerically when the grid resolves the PDF."""
        a, b = 0.2, 0.5
        p = make_lognormal_wl_params(a=a, b=b)
        mu_nodes, log_w = make_log_mu_grid(n_nodes=200, log_mu_range=(-4.0, 4.0))
        for z_test in [0.5, 1.0, 2.0]:
            z = jnp.full_like(mu_nodes, z_test)
            log_pdf = log_p_wl(mu_nodes, z, p)
            log_mu = jnp.log(mu_nodes)
            log_w_mu = log_pdf + jnp.log(mu_nodes) + log_w
            w = jnp.exp(log_w_mu)
            mean_lnmu = float(jnp.sum(log_mu * w))
            var_lnmu = float(jnp.sum((log_mu - mean_lnmu) ** 2 * w))
            expected_var = a * z_test ** b
            assert abs(var_lnmu - expected_var) / expected_var < 5e-3, (
                f"Var(ln μ) = {var_lnmu} vs expected {expected_var} at z={z_test}"
            )

    def test_mean_lnmu_analytic(self):
        """E(ln μ) = -s²/2 (this is the property that yields ⟨μ⟩ = 1)."""
        a, b = 0.2, 0.5
        p = make_lognormal_wl_params(a=a, b=b)
        mu_nodes, log_w = make_log_mu_grid(n_nodes=200, log_mu_range=(-4.0, 4.0))
        for z_test in [0.5, 1.0, 2.0]:
            z = jnp.full_like(mu_nodes, z_test)
            log_pdf = log_p_wl(mu_nodes, z, p)
            log_mu = jnp.log(mu_nodes)
            log_w_mu = log_pdf + jnp.log(mu_nodes) + log_w
            w = jnp.exp(log_w_mu)
            mean_lnmu = float(jnp.sum(log_mu * w))
            expected = -0.5 * a * z_test ** b
            assert abs(mean_lnmu - expected) < 5e-4, (
                f"E(ln μ) = {mean_lnmu} vs expected {expected} at z={z_test}"
            )

    def test_jit_and_vmap(self):
        """Function JIT-compiles and broadcasts (μ, z) of compatible shapes."""
        p = make_lognormal_wl_params()
        # 2D broadcast
        mu = jnp.linspace(0.5, 2.0, 5).reshape(-1, 1)
        z  = jnp.linspace(0.1, 2.0, 7).reshape(1, -1)
        # broadcast manually since log_p_wl doesn't internally broadcast scalars
        mu_b, z_b = jnp.broadcast_arrays(mu, z)
        log_pdf = log_p_wl(mu_b, z_b, p)
        assert log_pdf.shape == (5, 7)
        assert jnp.all(jnp.isfinite(log_pdf))


class TestWLTabulated:
    def test_round_trip_against_lognormal(self):
        """Build a tabulated PDF from the lognormal and verify it agrees.

        Bilinear interpolation has truncation error ~ (Δ ln μ)² / (2 s²)
        in log-space. To keep this < 1e-2 we need Δ ln μ < 0.1 s, so for
        s ~ 0.7 (which our wide test parameters produce) we use 200 grid
        nodes on (-3, 3) (Δ ≈ 0.03). Test points are inside ±1σ of the
        mean, where bilinear interpolation is accurate.
        """
        # Use parameters that give s ~ 1 on the test redshifts so the
        # grid resolves the curvature.
        p_ln = make_lognormal_wl_params(a=0.5, b=1.0)
        z_grid = jnp.linspace(0.05, 3.0, 80)
        log_mu_grid = jnp.linspace(-3.0, 3.0, 200)
        mu_grid = jnp.exp(log_mu_grid)
        ZZ, MM = jnp.meshgrid(z_grid, mu_grid, indexing="ij")
        log_table = log_p_wl(MM, ZZ, p_ln)
        p_tab = make_tabulated_wl_params(z_grid, log_mu_grid, log_table)
        # Test inside ±1σ of the mean at each z (s ≈ √(a z) ≈ 0.7 at z=1)
        z_test = jnp.array([0.5, 1.0, 1.7])
        # Stay near μ=1 (well inside the distribution core)
        mu_test = jnp.array([0.8, 1.0, 1.2])
        lp_ln = log_p_wl(mu_test, z_test, p_ln)
        lp_tab = log_p_wl(mu_test, z_test, p_tab)
        np.testing.assert_allclose(np.asarray(lp_tab), np.asarray(lp_ln), atol=2e-3)

    def test_validation_errors(self):
        z_grid = jnp.linspace(0.1, 2.0, 10)
        log_mu_grid = jnp.linspace(-1, 1, 8)
        bad_table = jnp.zeros((10, 7))  # wrong shape
        with pytest.raises(ValueError):
            make_tabulated_wl_params(z_grid, log_mu_grid, bad_table)


def test_sqrt_of_the_lognormal_variance_has_a_finite_gradient_at_zero():
    """``s = sqrt(a·z^b)`` must survive reverse-mode at ``a = 0``.

    ``d sqrt(x)/dx = inf`` at x = 0, so the unguarded chain through
    ``s2 = a·z^b`` returns ``inf * 0 = NaN`` for every gradient w.r.t. z (and
    hence w.r.t. the cosmology) in the ``a = 0`` ablation. Same defect, same
    fix as the Hermite WL kernel in ``likelihood/wl_weight.py``.
    """
    from darksirens.lensing.wlmagnification import _sqrt_grad_safe

    x = jnp.asarray([0.0, 1e-12, 0.25, 4.0])
    np.testing.assert_allclose(np.asarray(_sqrt_grad_safe(x)),
                               np.sqrt(np.asarray(x)), rtol=0, atol=0)
    # Gradient through the a -> 0 chain: s2 = a * z^b with a = 0.
    def s_of_z(z, a):
        return _sqrt_grad_safe(a * jnp.power(z, 1.5))

    g = jax.grad(s_of_z)(jnp.asarray(0.7), jnp.asarray(0.0))
    assert np.isfinite(float(g)) and float(g) == 0.0
    # And it is still the true derivative wherever s2 > 0.
    g_pos = jax.grad(s_of_z)(jnp.asarray(0.7), jnp.asarray(4e-3))
    assert float(g_pos) == pytest.approx(
        float(jax.grad(lambda z: jnp.sqrt(4e-3 * z ** 1.5))(jnp.asarray(0.7)))
    )


# ============================================================================
# Strong-lensing marks (SIS)
# ============================================================================

class TestSISLens:
    def test_tau2_monotonic_nonnegative(self):
        p = make_sis_lens_params(A_tau=5e-4, n_tau=3.0)
        z = jnp.linspace(0.0, 4.0, 50)
        tau = tau_2_SIS(z, p)
        assert jnp.all(tau >= 0.0)
        # τ ∝ z^3 strictly increasing on z>0
        diffs = jnp.diff(tau[1:])  # skip z=0
        assert jnp.all(diffs > 0)

    def test_tau2_powerlaw_form(self):
        p = make_sis_lens_params(A_tau=5e-4, n_tau=3.0)
        z = jnp.array([0.5, 1.0, 2.0])
        tau = tau_2_SIS(z, p)
        expected = 5e-4 * z**3
        np.testing.assert_allclose(np.asarray(tau), np.asarray(expected), rtol=1e-12)

    def test_tau4_inert(self):
        """Commit 1: quads always zero."""
        p = make_sis_lens_params()
        z = jnp.linspace(0.0, 4.0, 10)
        tau4 = tau_4_SIS(z, p)
        np.testing.assert_array_equal(np.asarray(tau4), np.zeros_like(np.asarray(tau4)))

    def test_p_y_normalization(self):
        """∫_0^1 p(y) dy = 1."""
        y_nodes, log_w = make_y_grid(n_nodes=32)
        log_p = log_p_y_SIS(y_nodes)
        integral = float(jnp.exp(logsumexp(log_p + log_w)))
        assert abs(integral - 1.0) < 1e-12

    def test_p_y_out_of_range_is_minus_inf(self):
        y_bad = jnp.array([-0.1, 0.0, 1.0, 1.5])
        log_p = log_p_y_SIS(y_bad)
        assert jnp.all(jnp.isinf(log_p))
        assert jnp.all(log_p < 0)

    def test_sis_image_magnification_constraint(self):
        """μ_+ − μ_− = 2 holds for all y ∈ (0, 1)."""
        y = jnp.linspace(0.001, 0.999, 100)
        mu_p, mu_m = mu_plus_minus_from_y(y)
        diff = mu_p - mu_m
        np.testing.assert_allclose(np.asarray(diff), 2.0 * np.ones_like(diff), atol=1e-12)

    def test_sis_magnifications_positive_and_ordered(self):
        """μ_+ > μ_- > 0 throughout y ∈ (0,1); μ_+ > 2 because y < 1."""
        y = jnp.linspace(0.01, 0.99, 50)
        mu_p, mu_m = mu_plus_minus_from_y(y)
        assert jnp.all(mu_p > mu_m)
        assert jnp.all(mu_m > 0)
        assert jnp.all(mu_p > 2.0)  # at y = 1, μ_+ = 2; we have y < 1

    def test_y_from_mu_plus_inverts(self):
        """y_from_mu_plus is the inverse of the y → μ_+ map."""
        y = jnp.linspace(0.01, 0.99, 50)
        mu_p, _ = mu_plus_minus_from_y(y)
        y_recovered = y_from_mu_plus(mu_p)
        np.testing.assert_allclose(np.asarray(y_recovered), np.asarray(y), atol=1e-12)

    def test_delta_t_linear_in_y(self):
        """Δt = T_0 · y."""
        p = make_sis_lens_params(T0_seconds=4.32e5)
        y = jnp.array([0.1, 0.5, 0.9])
        dt = delta_t_from_y(y, p)
        np.testing.assert_allclose(
            np.asarray(dt),
            4.32e5 * np.asarray(y),
            rtol=1e-12,
        )

    def test_default_T0_matches_the_sis_formula_in_this_cosmology(self):
        """The default T0 must equal the module's own SIS formula at the
        reference configuration, NOT a 5-day placeholder.

        T0 = 2 (1+z_L) theta_E^2 D_L D_S / (c D_LS),
        theta_E = 4 pi (sigma_v/c)^2 D_LS/D_S,
        at z_L = 0.5, z_s = 1.0, sigma_v = 200 km/s with this repo's cosmology.
        The historical 4.32e5 s (5 d) default is ~12x too small (it implies
        sigma_v ~ 107 km/s), and because the time mark reads
        y* = |dt|/T0 against the SIS support y in (0, 1), it gave an exactly
        -inf pair likelihood to every candidate separated by more than 5 days.
        """
        from darksirens.lensing.slmarks import DEFAULT_T0_SECONDS
        from darksirens.utils.cosmology import r_of_z, H0Planck, Om0Planck

        c_kms = 299792.458
        c_ms = 2.99792458e8
        mpc_m = 3.085677581491367e22
        z_L, z_s, sigma_v = 0.5, 1.0, 200.0

        r_L = float(r_of_z(z_L, H0Planck, Om0Planck))
        r_S = float(r_of_z(z_s, H0Planck, Om0Planck))
        D_L = r_L / (1.0 + z_L)
        D_S = r_S / (1.0 + z_s)
        D_LS = (r_S - r_L) / (1.0 + z_s)          # flat universe
        theta_E = 4.0 * np.pi * (sigma_v / c_kms) ** 2 * D_LS / D_S
        T0_expected = (
            2.0 * (1.0 + z_L) * theta_E ** 2 * (D_L * D_S / D_LS) * mpc_m / c_ms
        )

        assert 5.0e6 < T0_expected < 5.7e6, T0_expected
        np.testing.assert_allclose(DEFAULT_T0_SECONDS, T0_expected, rtol=2e-3)
        assert float(make_sis_lens_params().T0) == DEFAULT_T0_SECONDS
        # And the old default really was ~12x too small.
        assert 10.0 < T0_expected / 4.32e5 < 14.0

    def test_T0_is_overridable_and_sets_the_sis_support_edge(self):
        """|dt| >= T0 is outside the SIS support y in (0, 1) — the regime that
        annihilates a pair — and T0 is a constructor knob so a real-data run
        can move that edge."""
        p = make_sis_lens_params(T0_seconds=1.0e7)
        assert float(p.T0) == 1.0e7
        y_star = np.array([1.0, 5.0, 30.0, 90.0]) * 86400.0 / float(p.T0)
        assert np.all(y_star[:3] < 1.0)      # up to 30 d now inside support
        assert y_star[3] < 1.0               # 90 d too, at T0 = 1e7 s
        # ... while the historical 5-day T0 excluded everything past 5 days.
        y_star_old = np.array([5.5, 30.0, 90.0]) * 86400.0 / 4.32e5
        assert np.all(y_star_old >= 1.0)


# ============================================================================
# Cluster I/O
# ============================================================================

class TestClusters:
    def test_empty_round_trip(self, tmp_path):
        cs = empty_cluster_set()
        path = tmp_path / "clusters_empty.h5"
        save_clusters(str(path), cs, n_events=10)
        cs2 = load_clusters(str(path), n_events=10)
        assert cs2.npairs == 0
        assert cs2.nquads == 0
        assert cs2.pair_indices.shape == (0, 2)

    def test_pair_round_trip(self, tmp_path):
        pairs = np.array([[0, 1], [2, 5]], dtype=np.int32)
        bfs = np.array([2.0, 1.3])
        cs = make_cluster_set(pair_indices=pairs, pair_log_BF=bfs, n_events=10)
        path = tmp_path / "clusters.h5"
        save_clusters(str(path), cs, n_events=10, bf_threshold=0.5)
        cs2 = load_clusters(str(path))
        np.testing.assert_array_equal(np.asarray(cs2.pair_indices), pairs)
        np.testing.assert_allclose(np.asarray(cs2.pair_log_BF), bfs)
        assert cs2.npairs == 2

    def test_pair_ordering_enforced(self):
        with pytest.raises(ValueError):
            make_cluster_set(pair_indices=np.array([[3, 1]]), n_events=10)

    def test_quad_ordering_enforced(self):
        with pytest.raises(ValueError):
            make_cluster_set(quad_indices=np.array([[1, 2, 2, 4]]), n_events=10)

    def test_pair_index_out_of_range(self):
        with pytest.raises(ValueError):
            make_cluster_set(pair_indices=np.array([[0, 12]]), n_events=10)

    def test_pair_bf_length_mismatch(self):
        with pytest.raises(ValueError):
            make_cluster_set(
                pair_indices=np.array([[0, 1], [2, 3]]),
                pair_log_BF=np.array([1.0]),  # wrong length
                n_events=10,
            )

    def test_quad_round_trip(self, tmp_path):
        quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        cs = make_cluster_set(quad_indices=quads, n_events=10)
        path = tmp_path / "quads.h5"
        save_clusters(str(path), cs, n_events=10)
        cs2 = load_clusters(str(path))
        np.testing.assert_array_equal(np.asarray(cs2.quad_indices), quads)
        assert cs2.nquads == 2

    def test_load_none_returns_empty(self):
        cs = load_clusters(None)
        assert cs.npairs == 0
        assert cs.nquads == 0

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_clusters(str(tmp_path / "nonexistent.h5"))


# ============================================================================
# Cross-check: independent integral via scipy.stats
# ============================================================================

class TestIntegrationCrossCheck:
    """Cross-check WL math against scipy as an independent oracle."""

    def test_lognormal_against_scipy(self):
        """log_p_wl agrees with scipy.stats.lognorm at sample points."""
        try:
            from scipy.stats import lognorm
        except ImportError:
            pytest.skip("scipy not available")

        a, b = 0.02, 1.5
        z_test = 1.0
        s2 = a * z_test ** b
        s = np.sqrt(s2)
        m = -0.5 * s2

        # scipy: lognorm with shape s and scale exp(m) — pdf in μ
        p = make_lognormal_wl_params(a=a, b=b)
        mu_test = jnp.array([0.5, 0.8, 1.0, 1.3, 2.0])
        z = jnp.full_like(mu_test, z_test)
        ours = np.asarray(log_p_wl(mu_test, z, p))
        scipy_pdf = lognorm.pdf(np.asarray(mu_test), s=s, scale=np.exp(m))
        scipy_log_pdf = np.log(scipy_pdf)
        np.testing.assert_allclose(ours, scipy_log_pdf, atol=1e-10)


# Pretty printer for ad-hoc running
if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", "-v", __file__], check=False)
