"""
test_wl_weight.py
-----------------
Tests for commit 2: WL-marginalized per-sample log-importance-weight.

Test hierarchy
~~~~~~~~~~~~~~
1. **Exact reduction (structural)**: with ``wl_enabled=False``, the WL
   dispatcher returns bit-identical output to the standard hot path.

2. **JIT compatibility**: the closure-style WL PDF survives JIT
   compilation when passed through as a static argument.

3. **Math: small-a limit**: with very small lognormal variance, the
   WL-marginalized log-weight approaches the standard one. Tolerance
   set by the dominant systematic O(s²) ≈ a · z_s^b / 2.

4. **Math: Jacobian sign**: the (3/2) log μ Jacobian coefficient is
   what gives ⟨μ⟩ = 1 ⇒ the marginalized rate is preserved at
   leading order. Test by integrating a constant-target lognormal
   over a flat sky/mass/spin and comparing to a numpy reference.

5. **NaN / out-of-grid handling**: samples with apparent dL such that
   some μ-nodes go off-grid still yield a finite log-weight when at
   least one node remains in-grid.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import (
    z_of_dL, dL_of_z, ddL_of_z, dL_in_z_grid, H0Planck, Om0Planck,
)
from darksirens.inference.utils import log_sample_weight
from darksirens.likelihood.wl_weight import (
    log_sample_weight_wl_marginalized,
    log_sample_weight_wl_or_standard,
)
from darksirens.lensing.wlmagnification import (
    make_lognormal_log_p_wl,
    make_tabulated_log_p_wl,
    make_tabulated_wl_params,
    log_p_wl,
    wl_mu_quadrature_coverage,
    validate_wl_mu_quadrature,
)
from darksirens.lensing.grids import make_log_mu_grid, make_wl_mu_quadrature
from darksirens.redshift.volume import log_volume_prior_vmap


# ============================================================================
# Fixtures
# ============================================================================

def _toy_cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _toy_survey():
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
    """Smooth, separable, positive log-density on (m1src, q, z, chieff).

    Has non-trivial m1src dependence so that z → z(μ) coupling shows up
    in the marginalization.
    """
    del pop_params
    return (
        -0.5 * ((m1src - 30.0) / 8.0) ** 2
        - 0.5 * ((q - 0.7) / 0.15) ** 2
        - 0.5 * ((chieff + 0.05) / 0.2) ** 2
        + 0.3 * jnp.log1p(z)
    )


def _toy_volume_prior(z, pix, catalog):
    """Comoving-volume prior. Mirrors the `spectral_sirens` registry entry."""
    del pix, catalog
    cosmo = _toy_cosmo()
    survey = _toy_survey()
    return log_volume_prior_vmap(z, cosmo, survey)


def _toy_samples(n=8, seed=0):
    """A small batch of plausible per-sample inputs."""
    rng = np.random.default_rng(seed)
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, n))
    q = jnp.asarray(rng.uniform(0.4, 0.95, n))
    dL = jnp.asarray(rng.uniform(400.0, 3000.0, n))
    chieff = jnp.asarray(rng.uniform(-0.3, 0.3, n))
    pix = jnp.zeros(n, dtype=jnp.int32)
    # Per-sample positive PE proposal density
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, n))
    return m1det, q, dL, chieff, pix, prior_wt


# ============================================================================
# 1. Exact reduction (structural)
# ============================================================================

class TestReductionExact:
    """wl_enabled=False must give bit-identical output to standard."""

    def test_dispatcher_off_matches_standard(self):
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=12)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        # Standard path
        std = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
        )

        # WL dispatcher, OFF
        mu_nodes, log_w = make_log_mu_grid(16, (-0.6, 0.6))
        # log_p_wl_fn is unused when wl_enabled=False; pass a placeholder.
        log_p_wl_fn = make_lognormal_log_p_wl(0.1, 1.0)
        off = log_sample_weight_wl_or_standard(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
            wl_enabled=False,
        )
        np.testing.assert_array_equal(np.asarray(off), np.asarray(std))

    def test_dispatcher_off_under_jit(self):
        """Same equivalence holds under jit with wl_enabled as static argname."""
        from functools import partial

        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=8)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        mu_nodes, log_w = make_log_mu_grid(16, (-0.6, 0.6))
        log_p_wl_fn = make_lognormal_log_p_wl(0.1, 1.0)

        @partial(jax.jit, static_argnames=["wl_enabled"])
        def jitted(m1det, q, dL, chieff, pix, prior_wt, wl_enabled):
            return log_sample_weight_wl_or_standard(
                m1det, q, dL, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                _toy_log_p_pop, _toy_volume_prior,
                log_p_wl_fn, mu_nodes, log_w,
                wl_enabled=wl_enabled,
            )

        out_off = jitted(m1det, q, dL, chieff, pix, prior_wt, wl_enabled=False)
        out_on = jitted(m1det, q, dL, chieff, pix, prior_wt, wl_enabled=True)

        # The OFF path must equal the standard path
        std = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
        )
        np.testing.assert_allclose(np.asarray(out_off), np.asarray(std), rtol=1e-12)
        # The ON path must give finite (non-NaN) output and differ from std
        # by an amount controlled by the WL variance.
        assert jnp.all(jnp.isfinite(out_on))
        # Variance a=0.1, b=1 means s²(z=1)≈0.1, fairly large. Expect O(s²) deviations.
        diffs = np.asarray(out_on - out_off)
        assert np.max(np.abs(diffs)) > 1e-3   # WL is doing something
        assert np.max(np.abs(diffs)) < 1.0    # but not blowing up


# ============================================================================
# 2. JIT compatibility (closures, static args)
# ============================================================================

class TestJITCompat:
    def test_lognormal_closure_jit_compiles(self):
        """Closures from make_lognormal_log_p_wl JIT-compile cleanly."""
        log_p_wl_fn = make_lognormal_log_p_wl(0.05, 1.5)
        f = jax.jit(log_p_wl_fn)
        out = f(jnp.array([1.0, 1.2, 0.8]), jnp.array([1.0, 0.5, 2.0]))
        assert out.shape == (3,)
        assert jnp.all(jnp.isfinite(out))

    def test_tabulated_closure_jit_compiles(self):
        """Tabulated closure also JIT-compiles."""
        z_grid = jnp.linspace(0.1, 3.0, 40)
        log_mu_grid = jnp.linspace(-2.0, 2.0, 60)
        # Construct a simple analytic table to verify the closure builds
        ZZ, MM = jnp.meshgrid(z_grid, log_mu_grid, indexing="ij")
        log_p_table = -0.5 * (MM / 0.1) ** 2  # crude
        log_p_wl_fn = make_tabulated_log_p_wl(z_grid, log_mu_grid, log_p_table)
        f = jax.jit(log_p_wl_fn)
        out = f(jnp.array([1.0, 1.1]), jnp.array([0.5, 1.5]))
        assert out.shape == (2,)
        assert jnp.all(jnp.isfinite(out))

    def test_wl_marginalized_jit_compiles(self):
        """The marginalization function itself JITs."""
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=6)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])
        mu_nodes, log_w = make_log_mu_grid(8, (-0.5, 0.5))
        log_p_wl_fn = make_lognormal_log_p_wl(0.02, 1.0)

        jit_fn = jax.jit(lambda *a: log_sample_weight_wl_marginalized(
            *a, cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
        ))
        out = jit_fn(m1det, q, dL, chieff, pix, prior_wt)
        assert out.shape == (6,)
        assert jnp.all(jnp.isfinite(out))


# ============================================================================
# 2b. Tabulated backend: (mu, z) broadcasting + mu-quadrature coverage
# ============================================================================

def _lognormal_table(a=0.02, b=1.0, nz=200, nmu=769, log_mu_half=2.0):
    """Dense tabulation of the lognormal WL log-PDF on a (z, ln mu) grid."""
    z_grid = jnp.linspace(0.05, 3.0, nz)
    log_mu_grid = jnp.linspace(-log_mu_half, log_mu_half, nmu)
    ZZ, MM = jnp.meshgrid(z_grid, log_mu_grid, indexing="ij")
    table = make_lognormal_log_p_wl(a, b)(jnp.exp(MM), ZZ)
    return z_grid, log_mu_grid, table


class TestTabulatedBroadcast:
    """The tabulated evaluators must honour the module's broadcasting contract.

    Regression for the review finding that both tabulated evaluators flattened
    ``mu`` and ``z`` INDEPENDENTLY before ``vmap``, so any broadcast-but-not-equal
    shape pair raised ``vmap got inconsistent sizes``. The sole production caller
    (``log_sample_weight_wl_marginalized``) passes mu (Nmu,) against
    z (nsamp, Nmu), so ``--wl_backend tabulated`` crashed at the first
    likelihood evaluation.
    """

    def test_closure_broadcasts_mu_nodes_against_sample_z(self):
        z_grid, log_mu_grid, table = _lognormal_table()
        fn = make_tabulated_log_p_wl(z_grid, log_mu_grid, table)
        mu_nodes, _ = make_wl_mu_quadrature()               # (16,)
        z = jnp.full((7, mu_nodes.shape[0]), 0.8)           # (7, 16)

        out = fn(mu_nodes, z)
        assert out.shape == (7, 16)
        # Every row is the same z, so every row must be identical, and each
        # must reproduce the lognormal the table was built from.
        ref = make_lognormal_log_p_wl(0.02, 1.0)(mu_nodes, jnp.full((16,), 0.8))
        assert np.allclose(np.asarray(out), np.asarray(ref)[None, :], atol=2e-3)

    def test_closure_broadcasts_scalar_z(self):
        z_grid, log_mu_grid, table = _lognormal_table()
        fn = make_tabulated_log_p_wl(z_grid, log_mu_grid, table)
        mu_nodes, _ = make_wl_mu_quadrature()
        out = fn(mu_nodes, jnp.asarray(0.8))
        assert out.shape == (16,)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_free_dispatcher_broadcasts(self):
        """The non-closure ``log_p_wl`` path broadcasts identically."""
        z_grid, log_mu_grid, table = _lognormal_table()
        params = make_tabulated_wl_params(z_grid, log_mu_grid, table)
        mu_nodes, _ = make_wl_mu_quadrature()
        z = jnp.full((5, mu_nodes.shape[0]), 1.3)
        out = log_p_wl(mu_nodes, z, params)
        assert out.shape == (5, 16)
        fn = make_tabulated_log_p_wl(z_grid, log_mu_grid, table)
        assert np.allclose(np.asarray(out), np.asarray(fn(mu_nodes, z)))

    def test_production_entry_point_runs_with_tabulated_closure(self):
        """``log_sample_weight_wl_or_standard(..., wl_enabled=True)`` — the exact
        call the cluster/main likelihoods make — must run through the tabulated
        closure and agree with the lognormal closure the table was built from."""
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=9, seed=7)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])
        z_grid, log_mu_grid, table = _lognormal_table(a=0.02, b=1.0)
        mu_nodes, log_w = make_wl_mu_quadrature()

        common = dict(
            cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
            log_p_pop_fn=_toy_log_p_pop, log_prior_z_fn=_toy_volume_prior,
            mu_nodes=mu_nodes, log_w_nodes=log_w, wl_enabled=True,
        )
        out_tab = log_sample_weight_wl_or_standard(
            m1det, q, dL, chieff, pix, prior_wt,
            log_p_wl_fn=make_tabulated_log_p_wl(z_grid, log_mu_grid, table),
            **common,
        )
        out_ln = log_sample_weight_wl_or_standard(
            m1det, q, dL, chieff, pix, prior_wt,
            log_p_wl_fn=make_lognormal_log_p_wl(0.02, 1.0),
            **common,
        )
        assert out_tab.shape == (9,)
        assert np.all(np.isfinite(np.asarray(out_tab)))
        # Agreement is limited by the table's bilinear interpolation error.
        assert np.allclose(np.asarray(out_tab), np.asarray(out_ln), atol=2e-2)


class TestMuQuadratureCoverage:
    """The 16-node ln-mu quadrature must be VALIDATED against the table, not
    assumed: an under-resolved table makes int p_WL(mu|z) dmu collapse to ~0
    and silently deletes those events from the likelihood."""

    def test_coverage_is_unity_for_a_resolved_table(self):
        z_grid, log_mu_grid, table = _lognormal_table(a=0.02, b=1.0)
        mu_nodes, log_w = make_wl_mu_quadrature()
        cov = wl_mu_quadrature_coverage(
            mu_nodes, log_w, z_grid, log_mu_grid, table,
            z_test=jnp.asarray([0.8, 1.5, 2.5]),
        )
        assert np.allclose(np.asarray(cov), 1.0, atol=2e-2)
        # ... and the validator accepts it on the table's own z-grid.
        validate_wl_mu_quadrature(
            mu_nodes, log_w, z_grid, log_mu_grid, table,
            z_test=jnp.asarray([0.8, 1.5, 2.5]),
        )

    def test_default_lognormal_width_is_unresolved_at_low_z(self):
        """The SHIPPED default (a=4e-3, b=1.5) is far too narrow for the
        16-node grid below z ~ 0.5 — the measured under-coverage that motivates
        the hard startup check."""
        z_grid, log_mu_grid, table = _lognormal_table(a=4.0e-3, b=1.5)
        mu_nodes, log_w = make_wl_mu_quadrature()
        cov = np.asarray(wl_mu_quadrature_coverage(
            mu_nodes, log_w, z_grid, log_mu_grid, table,
            z_test=jnp.asarray([0.1, 0.2, 0.3, 1.0]),
        ))
        assert cov[0] < 1e-3       # z = 0.1: event silently deleted
        assert cov[1] < 0.2        # z = 0.2
        assert cov[2] < 0.5        # z = 0.3
        assert abs(cov[3] - 1.0) < 2e-2   # z = 1 is fine

        with pytest.raises(ValueError, match="does not resolve"):
            validate_wl_mu_quadrature(
                mu_nodes, log_w, z_grid, log_mu_grid, table,
                z_test=jnp.asarray([0.1, 0.2, 0.3, 1.0]),
            )


# ============================================================================
# 3. Math: small-a limit
# ============================================================================

class TestSmallALimit:
    """With small lognormal variance (but resolvable by the grid),
    marginalized ≈ standard at the expected O(s²) level, and the residual
    monotonically decreases as ``a`` decreases.

    **Caveat**: Gauss-Legendre quadrature breaks down when s < grid spacing.
    For the default (-0.5, 0.5) range with 64 nodes (Δ ≈ 0.016), we stay
    within a ∈ [1e-3, 3e-2] where s ∈ [0.03, 0.17] is well-resolved.
    This is documented in the wl_weight.py docstring.
    """

    def test_moderate_a_close_to_standard(self):
        """At a = 1e-2 (s²=1%, well-resolved by grid), marginalized within O(a) of standard."""
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=10, seed=1)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        std = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
        )

        a_moderate = 1e-2
        mu_nodes, log_w = make_log_mu_grid(64, (-0.5, 0.5))
        log_p_wl_fn = make_lognormal_log_p_wl(a_moderate, 0.0)  # b=0: constant s²(z)
        marg = log_sample_weight_wl_marginalized(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
        )
        # Leading correction ~ s² · curvature(log w). For smooth populations
        # the prefactor is O(1) so total < a few × a.
        diff = np.abs(np.asarray(marg) - np.asarray(std))
        assert np.all(np.isfinite(diff))
        assert np.max(diff) < 0.05, (
            f"max |marg - std| = {np.max(diff)} at a={a_moderate}, expected < 0.05"
        )

    def test_monotonic_decrease_in_resolvable_range(self):
        """In the grid-resolvable range a ∈ [1e-3, 3e-2], residual decreases
        monotonically with a."""
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=10, seed=2)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        std = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
        )

        mu_nodes, log_w = make_log_mu_grid(64, (-0.5, 0.5))
        # All three values give s² well above the grid resolution (Δ ≈ 0.016)
        a_vals = [3e-2, 1e-2, 3e-3]
        residuals = []
        for a in a_vals:
            log_p_wl_fn = make_lognormal_log_p_wl(a, 0.0)
            marg = log_sample_weight_wl_marginalized(
                m1det, q, dL, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                _toy_log_p_pop, _toy_volume_prior,
                log_p_wl_fn, mu_nodes, log_w,
            )
            residuals.append(float(np.max(np.abs(np.asarray(marg) - np.asarray(std)))))

        # Monotonic decrease: smaller a → smaller residual.
        # With factor-~3 decrease per step in a, we should comfortably see this.
        assert residuals[1] < residuals[0], (
            f"a={a_vals[1]} residual ({residuals[1]}) should be < "
            f"a={a_vals[0]} residual ({residuals[0]})"
        )
        assert residuals[2] < residuals[1], (
            f"a={a_vals[2]} residual ({residuals[2]}) should be < "
            f"a={a_vals[1]} residual ({residuals[1]})"
        )
        # Sanity: ratio of residuals should be roughly the ratio of a's.
        # For O(s²) = O(a) scaling, residuals[0]/residuals[2] ~ 10.
        # Allow factor-3 looseness from population-curvature effects.
        ratio = residuals[0] / max(residuals[2], 1e-12)
        assert 3.0 < ratio < 30.0, (
            f"residual scaling: {residuals[0]}/{residuals[2]} = {ratio} "
            f"(expected ~10 for O(s²) scaling)"
        )


# ============================================================================
# 4. Math: independent numpy reference
# ============================================================================

class TestNumpyReference:
    """A from-scratch numpy implementation of the marginalization, used as oracle."""

    @staticmethod
    def _numpy_reference(m1det, q, dL, chieff, prior_wt,
                        cosmo, survey, pop_params, catalog,
                        log_p_pop_fn, log_prior_z_fn,
                        a_wl, b_wl, mu_nodes_np, log_w_nodes_np):
        """Reference WL-marginalized log-weight implemented in numpy.

        Implements exactly the master formula from the wl_weight.py docstring,
        with no broadcasting tricks. Used to verify the JAX version line by
        line against an independent computation.
        """

        out = np.zeros_like(np.asarray(m1det))
        H0, Om0 = float(cosmo.H0), float(cosmo.Om0)
        Nmu = len(mu_nodes_np)

        for i in range(len(out)):
            dL_app = float(dL[i])
            m1d_i = float(m1det[i])
            q_i = float(q[i])
            chi_i = float(chieff[i])
            pw_i = float(prior_wt[i])

            log_integrands = np.full(Nmu, -np.inf)
            for ell in range(Nmu):
                mu_ell = float(mu_nodes_np[ell])
                dL_true_ell = dL_app * np.sqrt(mu_ell)
                in_grid = bool(dL_in_z_grid(jnp.array(dL_true_ell), H0, Om0))
                if not in_grid:
                    continue
                z_s = float(z_of_dL(jnp.array(dL_true_ell), H0, Om0))
                if not np.isfinite(z_s):
                    continue
                m1src = m1d_i / (1.0 + z_s)
                lp_pop = float(log_p_pop_fn(
                    jnp.array(m1src), jnp.array(q_i),
                    jnp.array(z_s), jnp.array(chi_i), pop_params,
                ))
                lp_vol = float(log_prior_z_fn(
                    jnp.array([z_s]), jnp.array([0], dtype=jnp.int32), catalog,
                )[0])
                # Lognormal WL log-PDF
                z_safe = max(z_s, 1e-3)
                s2 = a_wl * (z_safe ** b_wl)
                s = np.sqrt(s2)
                m = -0.5 * s2
                log_mu = np.log(mu_ell)
                from math import log, pi
                lp_wl = (-0.5 * ((log_mu - m) / s) ** 2
                         - 0.5 * np.log(2.0 * np.pi * s2)
                         - log_mu)
                # Jacobian = log(1+z_s) + log dL'(z_s)
                ddL = float(ddL_of_z(jnp.array(z_s), jnp.array(dL_true_ell), H0, Om0))
                log_J = np.log(1.0 + z_s) + np.log(ddL)

                log_integrands[ell] = (
                    float(log_w_nodes_np[ell])
                    + lp_pop + lp_vol + lp_wl
                    - log_J
                    + 1.5 * log_mu
                )

            log_marg = float(jnp.log(jnp.sum(jnp.exp(jnp.asarray(log_integrands)))))
            if pw_i > 0.0 and np.isfinite(log_marg):
                out[i] = log_marg - np.log(pw_i)
            else:
                out[i] = -np.inf
        return out

    def test_jax_matches_numpy(self):
        """The JAX vectorized impl matches the numpy line-by-line reference."""
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=6, seed=3)
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        mu_nodes, log_w = make_log_mu_grid(16, (-0.5, 0.5))
        a_wl, b_wl = 0.01, 1.0
        log_p_wl_fn = make_lognormal_log_p_wl(a_wl, b_wl)

        jax_out = log_sample_weight_wl_marginalized(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
        )
        np_out = self._numpy_reference(
            m1det, q, dL, chieff, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            a_wl, b_wl,
            np.asarray(mu_nodes), np.asarray(log_w),
        )

        np.testing.assert_allclose(np.asarray(jax_out), np_out, rtol=1e-10, atol=1e-12)


# ============================================================================
# 5. NaN / out-of-grid handling
# ============================================================================

class TestNaNHandling:
    def test_out_of_grid_mu_nodes_masked(self):
        """A sample where some μ-nodes push dL_true out of grid still works."""
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])

        # Pick a dL_app right near the top of the grid so that high μ
        # nodes push dL_true outside.
        from darksirens.utils.cosmology import dL_grid_bounds
        dL_min, dL_max = dL_grid_bounds(cosmo.H0, cosmo.Om0)
        dL_app = jnp.array([float(dL_max) * 0.9])
        m1det = jnp.array([40.0])
        q = jnp.array([0.7])
        chieff = jnp.array([0.0])
        pix = jnp.array([0], dtype=jnp.int32)
        prior_wt = jnp.array([1.0])

        # Wide μ-grid so the top nodes will exit the grid
        mu_nodes, log_w = make_log_mu_grid(16, (-1.0, 1.0))
        log_p_wl_fn = make_lognormal_log_p_wl(0.05, 1.0)
        out = log_sample_weight_wl_marginalized(
            m1det, q, dL_app, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
        )
        # Should be finite (at least the lower-μ nodes are in-grid)
        assert jnp.isfinite(out[0]), f"got non-finite out-of-grid handling: {float(out[0])}"

    def test_zero_prior_wt_returns_minus_inf(self):
        cosmo = _toy_cosmo()
        survey = _toy_survey()
        catalog = _toy_catalog()
        pop_params = jnp.array([])
        m1det = jnp.array([40.0])
        q = jnp.array([0.7])
        dL = jnp.array([1000.0])
        chieff = jnp.array([0.0])
        pix = jnp.array([0], dtype=jnp.int32)
        prior_wt = jnp.array([0.0])   # zero — should propagate to -inf

        mu_nodes, log_w = make_log_mu_grid(8, (-0.5, 0.5))
        log_p_wl_fn = make_lognormal_log_p_wl(0.05, 1.0)
        out = log_sample_weight_wl_marginalized(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_wl_fn, mu_nodes, log_w,
        )
        assert jnp.isinf(out[0]) and out[0] < 0
