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

import re
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

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils import cosmology
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.likelihood.core import (
    darksiren_log_likelihood,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
)
from darksirens.likelihood import factory as likelihood_module
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
        delta_g_pix_z=jnp.zeros((1, 1)),        dN_obs_kde=None,
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
              wl_a=0.0, wl_b=0.0, wl_selection=WL_SELECTION_STANDARD):
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
            wl_selection=wl_selection,
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


    def test_empty_pop_params_raises_clear_error(self, fixture):
        """Empty pop_params should fail fast with a descriptive ValueError."""
        with pytest.raises(ValueError, match=r"pop_model='powerlaw\+peak'.*pop_params\.shape=\(0,\).*Verify parameter-space construction"):
            darksiren_log_likelihood(
                fixture["cosmo"],
                fixture["survey"],
                jnp.array([]),
                fixture["gw_pe"],
                fixture["catalog"],
                fixture["gw_sel"],
                fixture["catalog"],
                fixture["n_events"],
                fixture["n_samp"],
                fixture["Ndraw"],
                pop_model="powerlaw+peak",
                universe_model="spectral_sirens",
                sel_batch_size=None,
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

    # ------------------------------------------------------------------
    # Selection-side WL marginalization (wl_selection)
    # ------------------------------------------------------------------

    def test_wl_selection_tiny_a_reduces_to_standard(self, fixture):
        """At a → 0 the Hermite-marginalized selection weight reduces to the
        standard one, so wl_selection must be inert (same argument as the PE
        term's exact-reduction test)."""
        ll_std = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL, wl_a=1e-6, wl_b=1.0,
            wl_selection=WL_SELECTION_STANDARD,
        )
        ll_sel = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL, wl_a=1e-6, wl_b=1.0,
            wl_selection=WL_SELECTION_LOGNORMAL,
        )
        assert jnp.isfinite(ll_std) and jnp.isfinite(ll_sel)
        assert float(abs(ll_sel - ll_std)) < 1e-4

    def test_wl_selection_changes_selection_at_moderate_a(self, fixture):
        """At a=1e-2 the selection integral must respond to the WL kernel:
        the PE term is identical between the two calls, so any difference is
        purely the previously-missing selection-side marginalization."""
        ll_std = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL, wl_a=1e-2, wl_b=1.0,
            wl_selection=WL_SELECTION_STANDARD,
        )
        ll_sel = self._call(
            fixture, "spectral_sirens_wl",
            wl_backend=WL_BACKEND_LOGNORMAL, wl_a=1e-2, wl_b=1.0,
            wl_selection=WL_SELECTION_LOGNORMAL,
        )
        assert jnp.isfinite(ll_sel)
        diff = float(abs(ll_sel - ll_std))
        assert 1e-8 < diff < 5.0, (
            f"|ll_sel - ll_std| = {diff} at a=1e-2 — selection-side WL is "
            f"either inert or blowing up"
        )

    def test_wl_selection_residual_scales_linearly_in_a(self, fixture):
        """The selection-side correction is O(s^2) = O(a·z^b), same as the PE
        side: the ll(sel on) - ll(sel off) residual must scale ~linearly in a."""
        diffs = []
        for a in (1e-4, 1e-3, 1e-2):
            ll_std = self._call(
                fixture, "spectral_sirens_wl",
                wl_backend=WL_BACKEND_LOGNORMAL, wl_a=a, wl_b=1.0,
                wl_selection=WL_SELECTION_STANDARD,
            )
            ll_sel = self._call(
                fixture, "spectral_sirens_wl",
                wl_backend=WL_BACKEND_LOGNORMAL, wl_a=a, wl_b=1.0,
                wl_selection=WL_SELECTION_LOGNORMAL,
            )
            diffs.append(float(ll_sel - ll_std))
        ratio_10 = diffs[1] / diffs[0]
        ratio_21 = diffs[2] / diffs[1]
        assert 4.0 < abs(ratio_10) < 20.0, (
            f"selection residuals {diffs}: a=1e-3/1e-4 ratio {ratio_10} (expected ~10)"
        )
        assert 4.0 < abs(ratio_21) < 20.0, (
            f"selection residuals {diffs}: a=1e-2/1e-3 ratio {ratio_21} (expected ~10)"
        )

    def test_wl_selection_falls_through_when_backend_disabled(self, fixture):
        """wl_selection=LOGNORMAL with a disabled backend keeps the exact
        legacy path (mirrors the cluster wrapper's fallthrough semantics)."""
        ll_base = self._call(fixture, "spectral_sirens")
        ll_fall = self._call(
            fixture, "spectral_sirens",
            wl_backend=WL_BACKEND_DISABLED,
            wl_selection=WL_SELECTION_LOGNORMAL,
        )
        assert float(abs(ll_fall - ll_base)) < 1e-12

    def test_wl_selection_lognormal_under_tabulated_backend_raises(self, fixture):
        """The tabulated backend has NO matched selection integral: silently
        downgrading to STANDARD would normalize mu(Lambda) under a different
        observation model than the per-event weights, so it must be fatal."""
        with pytest.raises(ValueError, match="tabulated backend"):
            self._call(
                fixture, "spectral_sirens_wl",
                wl_backend=WL_BACKEND_TABULATED,
                wl_selection=WL_SELECTION_LOGNORMAL,
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
            return _cosmo(), _survey(), jnp.array([]), jnp.array([]), jnp.array([])

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
        delta_g_pix_z=jnp.zeros((1, 1)),        dN_obs_kde_pe=None, pixel_to_cache_idx_pe=None, unique_pixels_pe=None,
        dN_obs_kde_sel=None, pixel_to_cache_idx_sel=None, unique_pixels_sel=None,
        lss_completion_logq=None, lss_completion_logq_members=None, lss_completion_indexing=0,
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
    # The closure is jitted, so the WL scalars reaching the core are tracers; their
    # concrete values are the factory's bound jit operands (index 4/5 = wl_a/wl_b).
    assert float(like.operands[4]) == pytest.approx(1e-3)
    assert float(like.operands[5]) == pytest.approx(1.0)
    assert captured["wl_a"].shape == () and captured["wl_b"].shape == ()
    # No --wl_selection on opts -> the legacy standard selection path.
    assert captured["wl_selection"] == WL_SELECTION_STANDARD

    # With opts.wl_selection="wl_lognormal" the factory must thread the
    # lognormal selection code through to the core.
    captured.clear()
    opts_sel = type("Opts", (), {
        "pop_model": "powerlaw+peak",
        "universe_model": "spectral_sirens_wl",
        "sel_batch_size": None,
        "wl_selection": "wl_lognormal",
    })()
    like_sel = likelihood_module.make_likelihood(opts_sel, data, pop_params_fid=())
    _ = like_sel(jnp.array([]))
    assert captured["wl_selection"] == WL_SELECTION_LOGNORMAL


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
        delta_g_pix_z=jnp.zeros((1, 1)),        dN_obs_kde_pe=None, pixel_to_cache_idx_pe=None, unique_pixels_pe=None,
        dN_obs_kde_sel=None, pixel_to_cache_idx_sel=None, unique_pixels_sel=None,
        lss_completion_logq=None, lss_completion_logq_members=None, lss_completion_indexing=0,
    ))())

    class _Decoder:
        pop_labels = ("gamma",)

        def decode(self, coord):
            del coord
            return _cosmo(), _survey(), jnp.array([]), jnp.array([]), jnp.array([])

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


# ============================================================================
# The factory closure: hoisted operands + jitted body (PERF-P2 / PERF-P5)
# ============================================================================
#
# Only ``darksiren_log_likelihood`` used to be jitted; everything the returned
# closure did around it ran EAGERLY on every sampler call, and dynesty calls that
# closure once per live point.  Two defects lived there:
#
#   P-2  the flat K=1 path rebuilt the GW containers and re-ran
#        ``pad_gw_event_to_multiple`` over the ~1e6-injection arrays on EVERY call,
#        although none of it depends on ``coord`` (the K>=2 mixture path had always
#        hoisted them -- an unpropagated twin);
#   P-5  parameter decoding (~30 individual eager device ops) stayed outside any
#        jit, measured at 7.7-13.8 ms of a 30-51 ms spectral call.
#
# The fix hoists the containers to build time and jits the body, passing the device
# operands as ARGUMENTS (never closure captures -- see the note in
# ``factory._jit_likelihood_body``: jax lowers a closed-over concrete array to a
# ``dense<>`` HLO constant, which would embed the whole static state in the module).

def _flat_spectral_run(n_sel=2048, n_events=2, nsamp=8):
    """Minimal real (un-monkeypatched) flat K=1 spectral run for the factory."""
    rng = np.random.default_rng(3)
    total = n_events * nsamp
    opts = type("Opts", (), {
        "pop_model": "powerlaw+peak",
        "universe_model": "spectral_sirens",
        "sel_batch_size": None,
        "fix_cosmology": True,
        "fix_population": True,
        "fix_survey": True,
    })()
    data = dict(
        nEvents=n_events, nsamp=nsamp, Ndraw=float(n_sel), apix=1.0,
        m1det=jnp.asarray(rng.uniform(25.0, 45.0, total)),
        m2det=jnp.asarray(rng.uniform(12.0, 24.0, total)),
        dL=jnp.asarray(rng.uniform(400.0, 2500.0, total)),
        chieff=jnp.asarray(rng.uniform(-0.2, 0.2, total)),
        p_pe=jnp.ones(total),
        m1detsels=jnp.asarray(rng.uniform(20.0, 55.0, n_sel)),
        m2detsels=jnp.asarray(rng.uniform(10.0, 28.0, n_sel)),
        dLsels=jnp.asarray(rng.uniform(300.0, 2800.0, n_sel)),
        chieffsels=jnp.asarray(rng.uniform(-0.2, 0.2, n_sel)),
        p_draw=jnp.ones(n_sel),
        pixels_pe=jnp.zeros(total, dtype=jnp.int32),
        pixels_sel=jnp.zeros(n_sel, dtype=jnp.int32),
        nx_pe=jnp.zeros(total), ny_pe=jnp.zeros(total), nz_pe=jnp.ones(total),
        nx_sel=jnp.zeros(n_sel), ny_sel=jnp.zeros(n_sel), nz_sel=jnp.ones(n_sel),
    )
    return opts, data


def test_make_likelihood_pads_selection_once_at_build_time(monkeypatch):
    """P-2: ``pad_gw_event_to_multiple`` must run ONCE per factory build, not once
    per likelihood call (it re-materialises ~11 arrays of the full injection set)."""
    calls = []
    real_pad = likelihood_module.pad_gw_event_to_multiple

    def _counting_pad(event, multiple, **kw):
        calls.append(multiple)
        return real_pad(event, multiple, **kw)

    monkeypatch.setattr(likelihood_module, "pad_gw_event_to_multiple", _counting_pad)

    opts, data = _flat_spectral_run(n_sel=2000)
    opts.sel_batch_size = 512                       # 2000 % 512 != 0 -> real padding
    like = likelihood_module.make_likelihood(
        opts, data, get_fixed_population_params(opts.pop_model))
    assert calls == [512]                           # built once, before any call
    coord = jnp.array([])
    for _ in range(3):
        like(coord)
    assert calls == [512]                           # and never again


def test_make_likelihood_binds_the_same_gw_operands_every_call():
    """P-2: the GW containers are coord-independent, so every call must see the
    IDENTICAL leaf buffers (no per-call ``jnp.ones_like`` / concatenate churn)."""
    opts, data = _flat_spectral_run()
    like = likelihood_module.make_likelihood(
        opts, data, get_fixed_population_params(opts.pop_model))
    gw_pe, _em_pe, gw_sel, _em_sel = like.operands[:4]
    leaves_before = [id(leaf) for leaf in jax.tree_util.tree_leaves((gw_pe, gw_sel))]
    like(jnp.array([]))
    like(jnp.array([]))
    gw_pe2, _e2, gw_sel2, _e3 = like.operands[:4]
    assert [id(leaf) for leaf in jax.tree_util.tree_leaves((gw_pe2, gw_sel2))] == leaves_before


def test_make_likelihood_body_is_jitted_and_compiles_once():
    """P-5: the sampler-facing closure is jitted, and repeated calls at different
    coordinates reuse ONE compilation (same shapes/dtypes every call)."""
    opts, data = _flat_spectral_run()
    opts.fix_population = False                     # give the coord a real dimension
    pop_fid = get_fixed_population_params(opts.pop_model)
    like = likelihood_module.make_likelihood(opts, data, pop_fid)
    assert hasattr(like, "jitted_body")
    like.jitted_body._clear_cache()
    decoder = likelihood_module.build_parameter_decoder(
        opts, pop_fid, fixed_parameter_values=None, wl_params=None)
    ndim = len(decoder.sampled_labels)
    assert ndim > 0
    rng = np.random.default_rng(11)
    fid = np.asarray(pop_fid, dtype=float)
    for _ in range(4):
        coord = jnp.asarray(fid[:ndim] * (1.0 + 0.01 * rng.normal(size=ndim)))
        like(coord)
    assert like.jitted_body._cache_size() == 1


def test_jit_lowers_a_closed_over_array_to_an_hlo_constant():
    """The premise behind passing the factory's operands as ARGUMENTS: a concrete
    array captured by a jitted closure is lowered to a ``dense<>`` HLO constant, not
    to a parameter, so the module text grows with the array (verified on jax 0.4.34
    at ~8 bytes/element) and the buffer is duplicated in the executable."""
    arr = jnp.arange(4096, dtype=jnp.float64)
    closed = jax.jit(lambda x: jnp.sum(arr * x)).lower(jnp.float64(2.0)).as_text()
    passed = jax.jit(lambda x, a: jnp.sum(a * x)).lower(jnp.float64(2.0), arr).as_text()
    assert len(closed) > 8 * 4096          # array embedded in the module
    assert len(passed) < 4096              # array is a parameter
    assert "tensor<4096xf64>" in passed.split("\n")[1]   # ... of @main


def test_make_likelihood_operands_are_jit_arguments_not_captures():
    """P-5 guard: the device operands reach the jitted body as ARGUMENTS (jaxpr
    invars), so the ~1e6-row GW arrays and multi-GB catalog tables are never
    embedded in the module as constants."""
    n_sel = 20000
    opts, data = _flat_spectral_run(n_sel=n_sel)
    like = likelihood_module.make_likelihood(
        opts, data, get_fixed_population_params(opts.pop_model))
    closed_jaxpr = jax.make_jaxpr(like.jitted_body)(
        jnp.array([]), like.operands, like.distance_table)
    invar_shapes = [tuple(v.aval.shape) for v in closed_jaxpr.jaxpr.invars]
    # the 5 selection physics fields + q + valid + the 3 sky components
    assert invar_shapes.count((n_sel,)) >= 5, invar_shapes
    # ... and the cosmology distance table, the largest single operand of all
    assert tuple(cosmology.rs.shape) in invar_shapes, invar_shapes
    # and nothing of the operands' size was hoisted into the closed consts
    const_sizes = [int(np.prod(c.shape)) for c in closed_jaxpr.consts
                   if hasattr(c, "shape")]
    assert all(size < n_sel for size in const_sizes), const_sizes


# ── the cosmology distance table is an operand, not a literal ──────────────────
#
# ``utils.cosmology.rs`` is 21x41x31x500 float64 (1.33e7 elements, 106.8 MB).  As
# a module global it was a closure capture of every likelihood jit and jax lowered
# it to a ``dense<>`` literal at ~16 bytes of module text per element.  MEASURED on
# an H100 NVL before the fix: 427.5 MB of lowered text for the production spectral
# likelihood (two embeddings) and 443.6 MB for a dark-siren mock (three); after,
# 0.44 MB and 16.6 MB, with first-call compile time 6.99 s -> 2.72 s and 7.15 s ->
# 2.99 s and log-likelihoods bitwise identical at 12 / 8 prior draws.

_TABLE_ELEMENTS = int(np.prod(cosmology.rs.shape))


def _dense_literal_lengths(module_text):
    """Text length of every ``dense<...>`` literal in a lowered StableHLO module.

    Literals never contain '>' -- small ones are bracketed decimal lists, large
    ones a single ``"0x...."`` hex blob -- so the non-greedy scan is exact.
    """
    return [len(m) for m in re.findall(r"dense<[^>]*>", module_text)]


def test_spectral_likelihood_lowers_without_the_distance_table_as_a_literal():
    """Regression guard for the 427 MB module: no ``dense<>`` literal in the
    lowered likelihood may be big enough to BE the distance table, and the whole
    module must stay far below what one embedding of it would cost."""
    opts, data = _flat_spectral_run(n_sel=2048)
    like = likelihood_module.make_likelihood(
        opts, data, get_fixed_population_params(opts.pop_model))
    text = like.jitted_body.lower(
        jnp.array([]), like.operands, like.distance_table).as_text()

    # A literal costs >= 8 bytes of text per f64 element (measured ~16 on jax
    # 0.4.34), so anything at or above that bound is a table-sized constant.
    one_embedding = 8 * _TABLE_ELEMENTS
    assert max(_dense_literal_lengths(text), default=0) < one_embedding
    assert len(text) < one_embedding // 10, len(text)
    # and the table is a PARAMETER of @main instead
    assert "21x41x31x500xf64" in text.split("\n")[1]


def test_two_differently_shaped_builds_share_one_process():
    """The distance table must reach every module-level jit as an ARGUMENT, not as
    a context-variable read that gets closed over.

    A ``@jit`` function that captures a tracer is not keyed on that capture, and
    jax's jaxpr cache will replay it under a later, differently-shaped outer trace
    -- ``UnexpectedTracerError`` on jax 0.4.34.  Two builds whose shapes differ
    force exactly that re-trace of ``core.darksiren_log_likelihood``."""
    pop_fid = get_fixed_population_params("powerlaw+peak")
    values = []
    for n_sel in (2048, 4096):
        opts, data = _flat_spectral_run(n_sel=n_sel)
        like = likelihood_module.make_likelihood(opts, data, pop_fid)
        values.append(float(like(jnp.array([]))))
    assert all(np.isfinite(v) for v in values), values


def test_make_likelihood_jitted_matches_fully_eager():
    """Equivalence: jitting the closure must not change the value.  ``disable_jit``
    evaluates the SAME code path op-by-op on the host, so agreement to 1e-12
    relative rules out a reassociation/short-circuit difference."""
    opts, data = _flat_spectral_run()
    like = likelihood_module.make_likelihood(
        opts, data, get_fixed_population_params(opts.pop_model))
    coord = jnp.array([])
    jitted = float(like(coord))
    with jax.disable_jit():
        eager = float(like(coord))
    assert np.isfinite(jitted)
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=0.0)


def test_make_likelihood_padded_plan_matches_single_pass():
    """Equivalence across the hoisted padding: a padded, blocked selection pass must
    reproduce the single pass (the padded rows carry ``valid=False``)."""
    opts, data = _flat_spectral_run(n_sel=2000)
    pop_fid = get_fixed_population_params(opts.pop_model)
    opts.sel_batch_size = None
    single = float(likelihood_module.make_likelihood(opts, data, pop_fid)(jnp.array([])))
    opts_blocked, _ = _flat_spectral_run(n_sel=2000)
    opts_blocked.sel_batch_size = 512               # 2000 % 512 = 464 padded rows
    blocked = float(
        likelihood_module.make_likelihood(opts_blocked, data, pop_fid)(jnp.array([])))
    np.testing.assert_allclose(blocked, single, rtol=1e-9, atol=0.0)
