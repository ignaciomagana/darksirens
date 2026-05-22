"""
test_likelihood_integration.py
------------------------------
End-to-end tests for the patched ``darksiren_log_likelihood``.

The unit tests in test_wl_weight.py cover the marginalization function
in isolation. These tests drive the full likelihood through the JIT
boundary, exercising:

  - PRIOR_REGISTRY routing for 'spectral_sirens_wl'
  - The wl_backend static-arg dispatch
  - The wl_enabled = True branch through the per-event PE scan
  - The exact-reduction property at the LIKELIHOOD level (not just at
    the sample-weight level)
"""

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

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.inference.likelihood_core import (
    darksiren_log_likelihood,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
)
from darksirens.inference import likelihood as likelihood_module
from darksirens.gw.populations import get_fixed_population_params


# ============================================================================
# Fixtures
# ============================================================================

def _make_gw_event(n_events, n_samp, seed=0):
    """Tiny GWEvent for the inference smoke test."""
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(10.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(400.0, 3000.0, total))
    chieff = jnp.asarray(rng.uniform(-0.3, 0.3, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.zeros(total, dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    q = m2det / m1det
    return GWEvent(
        m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
        prior_wt=prior_wt, pixels=pixels, q=q, valid=valid,
    )


def _make_gw_sel(n_sel, seed=10):
    """Tiny selection injection set."""
    rng = np.random.default_rng(seed)
    m1det = jnp.asarray(rng.uniform(15.0, 70.0, n_sel))
    m2det = jnp.asarray(rng.uniform(8.0, 35.0, n_sel))
    dL = jnp.asarray(rng.uniform(200.0, 3000.0, n_sel))
    chieff = jnp.asarray(rng.uniform(-0.3, 0.3, n_sel))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, n_sel))
    pixels = jnp.zeros(n_sel, dtype=jnp.int32)
    valid = jnp.ones(n_sel, dtype=jnp.bool_)
    q = m2det / m1det
    return GWEvent(
        m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
        prior_wt=prior_wt, pixels=pixels, q=q, valid=valid,
    )


def _toy_catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        sigma_kernel=0.1,
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _cosmo():
    return CosmoParams(H0=H0Planck, Om0=Om0Planck)


def _survey():
    return SurveyParams(
        n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
    )


# ============================================================================
# Tests
# ============================================================================

class TestLikelihoodIntegration:
    """Exercise the patched darksiren_log_likelihood end-to-end."""

    @pytest.fixture(scope="class")
    def fixture(self):
        """One shared fixture for speed (compilation is the bottleneck)."""
        n_events, n_samp = 4, 200
        gw_pe = _make_gw_event(n_events, n_samp, seed=0)
        gw_sel = _make_gw_sel(500, seed=10)
        catalog = _toy_catalog()
        return {
            "cosmo": _cosmo(),
            "survey": _survey(),
            "gw_pe": gw_pe,
            "gw_sel": gw_sel,
            "catalog": catalog,
            "n_events": n_events,
            "n_samp": n_samp,
            "Ndraw": 1000.0,
            "pop_params": jnp.asarray(get_fixed_population_params("powerlaw+peak")),
        }

    def _call(self, fixture, universe_model, wl_backend=WL_BACKEND_DISABLED,
              wl_a=0.0, wl_b=0.0):
        pop_params = fixture["pop_params"]
        expected_len = len(get_fixed_population_params("powerlaw+peak"))
        assert pop_params.shape[0] > 0
        assert pop_params.shape[0] == expected_len
        return darksiren_log_likelihood(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            fixture["gw_pe"], fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
            pop_model="powerlaw+peak",
            universe_model=universe_model,
            sel_batch_size=None,
            wl_backend=wl_backend,
            wl_a=wl_a, wl_b=wl_b,
        )

    def test_baseline_spectral_sirens_finite(self, fixture):
        """Baseline (no WL): likelihood evaluates to a finite number."""
        ll = self._call(fixture, "spectral_sirens")
        assert jnp.isfinite(ll), f"baseline ll not finite: {ll}"

    def test_wl_with_tiny_a_essentially_baseline(self, fixture):
        """spectral_sirens_wl with a → 0 is essentially identical to baseline.

        The Hermite-Gauss quadrature in u = (ln μ - m(z))/s(z) is exact for
        the lognormal at *any* s, so the residual at small a scales linearly
        with a (the physical O(s²) correction in the population term),
        independent of quadrature node count. At a=1e-6 the per-event
        residual is below 1e-5; accumulated over 4 events it should be
        well under 1e-4.
        """
        ll_off = self._call(fixture, "spectral_sirens")
        ll_on  = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL,
            wl_a=1e-6, wl_b=1.0,
        )
        assert jnp.isfinite(ll_off), f"ll_off not finite: {ll_off}"
        assert jnp.isfinite(ll_on),  f"ll_on  not finite: {ll_on}"
        diff = float(abs(ll_on - ll_off))
        assert diff < 1e-4, (
            f"|ll_on - ll_off| = {diff} at a=1e-6 — Hermite quadrature is "
            f"supposed to give bit-identical reduction at very small a."
        )

    def test_wl_residual_scales_linearly_in_a(self, fixture):
        """Confirms O(a) scaling of the WL-on/off log-likelihood residual.

        This is the cleanest sanity check that the math is right: the WL
        correction enters at O(s²) = O(a · z^b), so the residual must be
        proportional to a at small a. The Hermite quadrature is exact, so
        the only source of finite residual is the physical correction.
        """
        ll_off = self._call(fixture, "spectral_sirens")
        diffs = []
        a_values = [1e-4, 1e-3, 1e-2]
        for a in a_values:
            ll_on = self._call(
                fixture, "spectral_sirens_wl",
                wl_backend=WL_BACKEND_LOGNORMAL,
                wl_a=a, wl_b=1.0,
            )
            diffs.append(float(ll_on - ll_off))
        # Each step in a by 10× should scale the residual by 10×.
        ratio_01_to_00 = diffs[1] / diffs[0]
        ratio_02_to_01 = diffs[2] / diffs[1]
        # Allow ratio in [5, 15] for the leading-order scaling.
        assert 5.0 < abs(ratio_01_to_00) < 15.0, (
            f"a=1e-3 / a=1e-4 ratio = {ratio_01_to_00} (expected ~10)"
        )
        assert 5.0 < abs(ratio_02_to_01) < 15.0, (
            f"a=1e-2 / a=1e-3 ratio = {ratio_02_to_01} (expected ~10)"
        )

    def test_wl_at_moderate_a_finite_and_meaningful(self, fixture):
        """At a=1e-2, WL likelihood differs from baseline by a measurable
        amount but remains finite — verifies WL is doing something."""
        ll_off = self._call(fixture, "spectral_sirens")
        ll_on  = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL,
            wl_a=1e-2, wl_b=1.0,
        )
        assert jnp.isfinite(ll_on)
        diff = float(abs(ll_on - ll_off))
        assert 1e-4 < diff < 5.0, (
            f"|ll_on - ll_off| = {diff} at a=1e-2 — WL is either inert "
            f"or blowing up"
        )

    def test_spectral_sirens_wl_without_backend_raises(self, fixture):
        """universe_model='spectral_sirens_wl' with wl_backend=DISABLED should raise."""
        with pytest.raises(ValueError, match="wl_backend"):
            self._call(
                fixture, "spectral_sirens_wl",
                wl_backend=WL_BACKEND_DISABLED,
            )

    def test_wl_backend_without_spectral_sirens_wl_raises(self, fixture):
        """wl_backend != DISABLED with universe_model != 'spectral_sirens_wl' raises."""
        with pytest.raises(ValueError, match="universe_model"):
            self._call(
                fixture, "spectral_sirens",
                wl_backend=WL_BACKEND_LOGNORMAL,
                wl_a=0.01, wl_b=1.0,
            )


def test_make_likelihood_spectral_sirens_wl_passes_wl_args(monkeypatch):
    """Factory should translate data['wl_params'] into explicit core kwargs."""
    captured = {}

    def _fake_core(*args, **kwargs):
        captured.update(kwargs)
        return jnp.asarray(0.0)

    class _Decoder:
        pop_labels = ()
        def decode(self, coord):
            del coord
            return _cosmo(), _survey(), jnp.array([])

    monkeypatch.setattr(likelihood_module, "prepare_catalog_views", lambda *a, **k: type("C", (), dict(
        sample_to_unique_pe=jnp.zeros(2, dtype=jnp.int32),
        sample_to_unique_sel=jnp.zeros(2, dtype=jnp.int32),
        zgals_pe_catalog=jnp.zeros((1, 1)),
        dzgals_pe_catalog=jnp.ones((1, 1)),
        wgals_pe_catalog=jnp.ones((1, 1)),
        ngals_pe_catalog=jnp.ones((1,), dtype=jnp.int32),
        zgals_sel_catalog=jnp.zeros((1, 1)),
        dzgals_sel_catalog=jnp.ones((1, 1)),
        wgals_sel_catalog=jnp.ones((1, 1)),
        ngals_sel_catalog=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        sigma_kernel=0.1,
        dN_obs_kde_pe=None, pixel_to_cache_idx_pe=None, unique_pixels_pe=None,
        dN_obs_kde_sel=None, pixel_to_cache_idx_sel=None, unique_pixels_sel=None,
    ))())
    monkeypatch.setattr(likelihood_module, "build_parameter_decoder", lambda *a, **k: _Decoder())
    monkeypatch.setattr(likelihood_module, "darksiren_log_likelihood", _fake_core)

    opts = type("Opts", (), {"pop_model": "powerlaw+peak", "universe_model": "spectral_sirens_wl", "sel_batch_size": None})()
    data = dict(
        nEvents=1, nsamp=2, Ndraw=2.0, apix=1.0,
        m1det=jnp.array([30.0, 31.0]), m2det=jnp.array([20.0, 21.0]), dL=jnp.array([1000.0, 1100.0]),
        chieff=jnp.array([0.0, 0.1]), p_pe=jnp.array([1.0, 1.0]),
        m1detsels=jnp.array([30.0, 31.0]), m2detsels=jnp.array([20.0, 21.0]), dLsels=jnp.array([1000.0, 1100.0]),
        chieffsels=jnp.array([0.0, 0.1]), p_draw=jnp.array([1.0, 1.0]),
        wl_params=type("P", (), {"backend": WL_BACKEND_LOGNORMAL, "a": jnp.asarray(1e-3), "b": jnp.asarray(1.0)})(),
    )
    like = likelihood_module.make_likelihood(opts, data, pop_params_fid=())
    _ = like(jnp.array([]))
    assert captured["wl_backend"] == WL_BACKEND_LOGNORMAL
    assert float(captured["wl_a"]) == pytest.approx(1e-3)
    assert float(captured["wl_b"]) == pytest.approx(1.0)


def test_make_likelihood_spectral_sirens_wl_missing_params_raises(monkeypatch):
    """WL universe model should fail clearly when data['wl_params'] is missing."""
    monkeypatch.setattr(likelihood_module, "prepare_catalog_views", lambda *a, **k: None)
    monkeypatch.setattr(likelihood_module, "build_parameter_decoder", lambda *a, **k: None)
    opts = type("Opts", (), {"pop_model": "powerlaw+peak", "universe_model": "spectral_sirens_wl", "sel_batch_size": None})()
    data = dict(nEvents=1, nsamp=1, Ndraw=1.0, apix=1.0, wl_params=None)
    data.update({k: jnp.array([1.0]) for k in ("m1det", "m2det", "dL", "chieff", "p_pe", "m1detsels", "m2detsels", "dLsels", "chieffsels", "p_draw")})
    with pytest.raises(ValueError, match=r"data\['wl_params'\]"):
        likelihood_module.make_likelihood(opts, data, pop_params_fid=())


def test_make_likelihood_fails_fast_on_empty_population_theta(monkeypatch):
    """When pop_model expects params, empty decoded theta should fail before core."""
    monkeypatch.setattr(likelihood_module, "prepare_catalog_views", lambda *a, **k: type("C", (), dict(
        sample_to_unique_pe=jnp.zeros(2, dtype=jnp.int32),
        sample_to_unique_sel=jnp.zeros(2, dtype=jnp.int32),
        zgals_pe_catalog=jnp.zeros((1, 1)),
        dzgals_pe_catalog=jnp.ones((1, 1)),
        wgals_pe_catalog=jnp.ones((1, 1)),
        ngals_pe_catalog=jnp.ones((1,), dtype=jnp.int32),
        zgals_sel_catalog=jnp.zeros((1, 1)),
        dzgals_sel_catalog=jnp.ones((1, 1)),
        wgals_sel_catalog=jnp.ones((1, 1)),
        ngals_sel_catalog=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        sigma_kernel=0.1,
        dN_obs_kde_pe=None, pixel_to_cache_idx_pe=None, unique_pixels_pe=None,
        dN_obs_kde_sel=None, pixel_to_cache_idx_sel=None, unique_pixels_sel=None,
    ))())

    class _Decoder:
        pop_labels = ("gamma",)

        def decode(self, coord):
            del coord
            return _cosmo(), _survey(), jnp.array([])

    monkeypatch.setattr(likelihood_module, "build_parameter_decoder", lambda *a, **k: _Decoder())
    opts = type("Opts", (), {"pop_model": "powerlaw+peak", "universe_model": "spectral_sirens", "sel_batch_size": None})()
    data = dict(
        nEvents=1, nsamp=2, Ndraw=2.0, apix=1.0,
        m1det=jnp.array([30.0, 31.0]), m2det=jnp.array([20.0, 21.0]), dL=jnp.array([1000.0, 1100.0]),
        chieff=jnp.array([0.0, 0.1]), p_pe=jnp.array([1.0, 1.0]),
        m1detsels=jnp.array([30.0, 31.0]), m2detsels=jnp.array([20.0, 21.0]), dLsels=jnp.array([1000.0, 1100.0]),
        chieffsels=jnp.array([0.0, 0.1]), p_draw=jnp.array([1.0, 1.0]),
    )
    like = likelihood_module.make_likelihood(opts, data, pop_params_fid=())
    with pytest.raises(ValueError, match="Population parameter length mismatch before likelihood evaluation"):
        _ = like(jnp.array([]))


def test_powerlaw_peak_fixture_population_vector_is_non_empty():
    """Guard fixture construction: powerlaw+peak must have non-empty pop params."""
    pop_params = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    assert pop_params.size > 0
