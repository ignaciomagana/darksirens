"""
test_md_rate_evolution.py
-------------------------
The '@md' rate-evolution decoration: Madau-Dickinson-like peaked merger rate

    psi(z) = (1+z)^gamma / (1 + ((1+z)/(1+z_peak))^(gamma+kappa)),

applied as psi(z)/(1+z) in place of the power-law (1+z)^(gamma-1).

Covers: registry name resolution + parameter vector layout, exact math
identities (z = z_peak turnover, z_peak -> inf power-law reduction, high-z
slope -(kappa+1)), guards, prior-parser plumbing, an end-to-end likelihood
smoke, the mock-lensing generator's rate-aware redshift sampler, and the
DARKSIRENS_ZMAX grid override.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.gw.populations.registry import (
    MD_RATE_FIDUCIALS,
    get_fixed_population_params,
    get_model,
    pop_model_prior_parser,
    split_rate_decoration,
)


# ============================================================================
# Registry resolution and parameter layout
# ============================================================================

def test_split_rate_decoration():
    assert split_rate_decoration("powerlaw+peak") == ("powerlaw+peak", "powerlaw")
    assert split_rate_decoration("powerlaw+peak@md") == ("powerlaw+peak", "md")
    for bad in ("powerlaw+peak@bogus", "@md", "powerlaw+peak@"):
        with pytest.raises(ValueError, match="decoration"):
            split_rate_decoration(bad)


def test_md_model_parameter_layout():
    base = get_model("powerlaw+peak")
    md = get_model("powerlaw+peak@md")
    base_names = [s.name for s in base.param_specs]
    md_names = [s.name for s in md.param_specs]
    assert md.rate_evolution == "md"
    assert base.rate_evolution == "powerlaw"
    assert md_names[:-3] == base_names[:-1]
    assert md_names[-3:] == ["gamma", "kappa", "z_peak"]

    vb = np.asarray(get_fixed_population_params("powerlaw+peak"))
    vm = np.asarray(get_fixed_population_params("powerlaw+peak@md"))
    assert len(vm) == len(vb) + 2
    np.testing.assert_allclose(vm[:-3], vb[:-1])
    np.testing.assert_allclose(vm[-3:], MD_RATE_FIDUCIALS)


def test_md_guards():
    with pytest.raises(ValueError, match="decoration"):
        get_model("powerlaw+peak@bogus")
    # Per-component (unshared) redshift evolution is not supported for MD.
    with pytest.raises(ValueError, match="shared"):
        get_model("powerlaw+peak+powerlaw@md", shared_gamma=False)
    with pytest.raises(ValueError, match="shared_gamma"):
        get_fixed_population_params("powerlaw+peak@md", shared_gamma=False)


def test_md_prior_parser_plumbing():
    lows, highs, labels, kinds, latex = pop_model_prior_parser("powerlaw+peak@md")
    lows_b, highs_b, labels_b, kinds_b, _ = pop_model_prior_parser("powerlaw+peak")
    assert len(labels) == len(labels_b) + 2
    assert labels[-3:] == [r"$\gamma$", r"$\kappa$", r"$z_{\rm peak}$"]
    assert len(lows) == len(highs) == len(labels) == len(kinds)
    assert (lows[-2], highs[-2]) == (0.0, 10.0)      # kappa
    assert (lows[-1], highs[-1]) == (0.2, 4.0)       # z_peak
    assert "MD" in latex


# ============================================================================
# Math identities (exact, no tolerance games)
# ============================================================================

def _log_rate_md(model, mixture_theta, z, gamma, kappa, z_pk):
    theta = jnp.concatenate([
        jnp.asarray(mixture_theta), jnp.asarray([gamma, kappa, z_pk])
    ])
    m1 = jnp.asarray([35.0]); q = jnp.asarray([0.8]); chi = jnp.asarray([0.0])
    return float(model.log_p_pop(m1, q, jnp.asarray([z]), chi, theta)[0])


def _log_rate_pl(model, mixture_theta, z, gamma):
    theta = jnp.concatenate([jnp.asarray(mixture_theta), jnp.asarray([gamma])])
    m1 = jnp.asarray([35.0]); q = jnp.asarray([0.8]); chi = jnp.asarray([0.0])
    return float(model.log_p_pop(m1, q, jnp.asarray([z]), chi, theta)[0])


class TestMDIdentities:
    @pytest.fixture(scope="class")
    def models(self):
        base = get_model("powerlaw+peak")
        md = get_model("powerlaw+peak@md")
        tm = np.asarray(get_fixed_population_params("powerlaw+peak"))[:-1]
        return base, md, tm

    def test_turnover_at_z_peak_is_exactly_log2(self, models):
        """psi(z_peak) = (1+z_peak)^gamma / 2 exactly."""
        base, md, tm = models
        gamma, kappa, z_pk = 2.7, 2.9, 1.9
        lhs = _log_rate_md(md, tm, z_pk, gamma, kappa, z_pk)
        rhs = _log_rate_pl(base, tm, z_pk, gamma) - np.log(2.0)
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-12)

    def test_zpeak_to_infinity_reduces_to_power_law(self, models):
        base, md, tm = models
        gamma, kappa = 2.7, 2.9
        for z in (0.1, 1.0, 3.0):
            lhs = _log_rate_md(md, tm, z, gamma, kappa, 1.0e9)
            rhs = _log_rate_pl(base, tm, z, gamma)
            np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-9)

    def test_high_z_slope_is_minus_kappa_plus_one(self, models):
        """d log[psi/(1+z)] / d log(1+z) -> -(kappa+1) far above the peak."""
        _, md, tm = models
        gamma, kappa, z_pk = 2.7, 2.9, 1.9
        z1, z2 = 40.0, 80.0
        slope = (
            _log_rate_md(md, tm, z2, gamma, kappa, z_pk)
            - _log_rate_md(md, tm, z1, gamma, kappa, z_pk)
        ) / (np.log1p(z2) - np.log1p(z1))
        np.testing.assert_allclose(slope, -(kappa + 1.0), rtol=5e-4)

    def test_gradients_finite(self, models):
        _, md, tm = models
        def f(rate_params):
            theta = jnp.concatenate([jnp.asarray(tm), rate_params])
            return md.log_p_pop(
                jnp.asarray([35.0]), jnp.asarray([0.8]),
                jnp.asarray([2.5]), jnp.asarray([0.0]), theta,
            )[0]
        g = jax.grad(f)(jnp.asarray([2.7, 2.9, 1.9]))
        assert np.all(np.isfinite(np.asarray(g)))


# ============================================================================
# End-to-end likelihood smoke with the '@md' pop model
# ============================================================================

def test_md_likelihood_end_to_end_finite():
    from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
    from darksirens.utils.cosmology import H0Planck, Om0Planck
    from darksirens.likelihood.core import darksiren_log_likelihood

    rng = np.random.default_rng(3)
    n_events, n_samp, n_sel = 3, 100, 300

    def _event(total, seed):
        r = np.random.default_rng(seed)
        m1det = jnp.asarray(r.uniform(20.0, 60.0, total))
        m2det = jnp.asarray(r.uniform(10.0, 30.0, total))
        return GWEvent(
            m1det=m1det, m2det=m2det,
            dL=jnp.asarray(r.uniform(400.0, 3000.0, total)),
            chieff=jnp.asarray(r.uniform(-0.3, 0.3, total)),
            prior_wt=jnp.asarray(r.uniform(0.5, 1.5, total)),
            pixels=jnp.zeros(total, dtype=jnp.int32),
            q=m2det / m1det,
            valid=jnp.ones(total, dtype=jnp.bool_),
        )

    catalog = EMCatalog(
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    pop = get_fixed_population_params("powerlaw+peak@md")
    ll = darksiren_log_likelihood(
        CosmoParams(H0=H0Planck, Om0=Om0Planck),
        SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5),
        jnp.asarray(pop),
        _event(n_events * n_samp, 0), catalog,
        _event(n_sel, 10), catalog,
        n_events, n_samp, float(n_sel),
        pop_model="powerlaw+peak@md",
        universe_model="spectral_sirens",
        sel_batch_size=None,
    )
    assert jnp.isfinite(ll), f"MD-rate likelihood not finite: {ll}"


# ============================================================================
# Generator: rate-aware redshift sampler
# ============================================================================

def test_generator_md_z_cdf_matches_analytic():
    from scripts.mock_lensing import generate_mock_lensing as gen
    from darksirens.utils.cosmology import dV_of_z, H0Planck, Om0Planck

    theta_md = np.asarray(get_fixed_population_params("powerlaw+peak@md"))
    gamma, kappa, z_pk = theta_md[-3:]
    old = gen.POP_NAME
    try:
        gen.set_pop_model("powerlaw+peak@md")
        assert gen._n_rate_params() == 3
        assert gen._theta_param_order()[-3:] == ["gamma", "kappa", "z_peak"]
        zg, pdf, cdf = gen._build_z_cdf(theta_md, H0Planck, Om0Planck, nz=2000)
        dV = np.asarray(dV_of_z(jnp.asarray(zg), H0Planck, Om0Planck))
        expected = dV * np.exp(
            (gamma - 1.0) * np.log1p(zg)
            - np.logaddexp(0.0, (gamma + kappa) * (np.log1p(zg) - np.log1p(z_pk)))
        )
        expected /= _trapezoid(expected, zg)
        np.testing.assert_allclose(pdf, expected, rtol=1e-10, atol=1e-14)
        assert abs(cdf[-1] - 1.0) < 1e-12

        # Draws follow the tabulated CDF (moment check, generous tolerance).
        rng = np.random.default_rng(7)
        z, _ = gen.sample_redshift(200_000, theta_md, rng, H0Planck, Om0Planck)
        z_mean_expected = _trapezoid(zg * pdf, zg)
        np.testing.assert_allclose(z.mean(), z_mean_expected, rtol=0.02)
    finally:
        gen.set_pop_model(old)


def test_generator_rejects_non_powerlaw_peak_mixture():
    from scripts.mock_lensing import generate_mock_lensing as gen

    with pytest.raises(ValueError, match="powerlaw\\+peak"):
        gen.set_pop_model("brokenpowerlaw+2peaks@md")


# ============================================================================
# DARKSIRENS_ZMAX grid override (subprocess: import-time env var)
# ============================================================================

def test_darksirens_zmax_env_override_extends_grids():
    code = r"""
import numpy as np
from darksirens.utils import cosmology as C
from darksirens.redshift import grid as G
assert C.zMax == 8.0, C.zMax
assert G.zMax == 8.0, G.zMax
assert abs(float(C.zgrid[-1]) - 8.0) < 1e-9
assert abs(float(G.zgrid[-1]) - 8.0) < 1e-9
# Node density preserved: counts scale with the log range.
assert len(C.zgrid) >= int(round(500 * np.log(9.0) / np.log(6.0))) - 1, len(C.zgrid)
assert G.zgrid.shape[0] >= int(round(1000 * np.log(9.0) / np.log(6.0))) - 1
# Distance/inverse-distance roundtrip well beyond the default z=5 cap.
z = 7.0
dl = float(C.dL_of_z(z, C.H0Planck, C.Om0Planck))
z_back = float(C.z_of_dL(dl, C.H0Planck, C.Om0Planck))
assert abs(z_back - z) < 1e-3, (z, z_back)
print("ZMAX-OVERRIDE-OK")
"""
    env = dict(os.environ)
    env.update({
        "DARKSIRENS_ZMAX": "8",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "True",
        "CUDA_VISIBLE_DEVICES": "",
    })
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PKG_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "ZMAX-OVERRIDE-OK" in proc.stdout


def test_default_zmax_unchanged():
    from darksirens.utils import cosmology as C
    from darksirens.redshift import grid as G

    if os.environ.get("DARKSIRENS_ZMAX") not in (None, "", "5", "5.0"):
        pytest.skip("DARKSIRENS_ZMAX set in this environment")
    assert C.zMax == 5.0
    assert G.zMax == 5.0
    assert len(C.zgrid) == 500
    assert G.zgrid.shape[0] == 1000
