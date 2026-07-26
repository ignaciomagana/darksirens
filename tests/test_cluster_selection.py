"""
test_cluster_selection.py
-------------------------
Unit + integration tests for commit 4:
  - darksirens/lensing/lensed_injections.py     (IO + validation)
  - darksirens/inference/cluster_selection.py    (importance estimator + MFG correction)
  - darksirens/inference/likelihood_with_clusters.py  (master likelihood)

Test priorities (highest first)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Exact reduction: master likelihood with cluster_mode=OFF gives
   bit-identical output to commit 2's darksiren_log_likelihood.
2. combined_selection_log_correction collapses to the singleton
   form when log_mu_cluster = -inf.
3. cluster_selection μ̂ matches a from-scratch numpy oracle.
4. Both-detected mask: undetected pairs are correctly excluded.
5. Importance-sampling consistency: when proposal == population, the
   sum of weights / N_draw gives an unbiased estimate of the
   selection probability.
6. Source-level consistency check rejects malformed injection files.
7. JIT compatibility.
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck

from darksirens.redshift.volume import log_volume_prior_vmap
from darksirens.lensing.lensed_injections import (
    LensedInjectionSet, make_lensed_injection_set,
    save_lensed_injections, load_lensed_injections,
)
from darksirens.lensing.slmarks import (
    make_sis_lens_params, tau_2_SIS, mu_plus_minus_from_y,
)
from darksirens.likelihood.cluster_selection import (
    compute_cluster_selection_term,
    combined_selection_log_correction,
    _per_source_log_weight,
)
from darksirens.gw.populations import get_fixed_population_params


def test_pair_pe_loader_prior_wt_normalization_scale_invariant():
    """Pair-image prior weights are normalized per image by the loader helper."""
    from darksirens.cli.inference_lensing import _normalize_pair_image_prior_wt

    raw = np.array([0.2, 0.4, 0.8, 1.6])
    norm = _normalize_pair_image_prior_wt(raw, context="pair_0/image0/prior_wt")
    scaled = _normalize_pair_image_prior_wt(
        17.0 * raw, context="pair_0/image0/prior_wt"
    )

    np.testing.assert_allclose(norm.sum(), 1.0)
    np.testing.assert_allclose(scaled.sum(), 1.0)
    np.testing.assert_allclose(scaled, norm)


@pytest.mark.parametrize("bad_prior_wt", [np.zeros(4), np.full(4, np.nan)])
def test_pair_pe_loader_rejects_malformed_prior_wt(bad_prior_wt):
    from darksirens.cli.inference_lensing import _normalize_pair_image_prior_wt

    with pytest.raises(ValueError, match="finite and positive"):
        _normalize_pair_image_prior_wt(
            bad_prior_wt, context="pair_0/image1/prior_wt"
        )


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
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )


def _toy_log_p_pop(m1src, q, z, chieff, pop_params):
    """Same toy population as commits 1-3."""
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


def _synth_lensed_injection_campaign(
    n_sources=200, seed=0,
    det_thresh_m1det=25.0,    # crude detection: m1det must exceed threshold
    det_thresh_dL=4000.0,     # crude detection: dL must be below threshold
):
    """Generate a synthetic lensed-injection campaign.

    Each source has (m1_src, q, z, chieff, y) drawn from a known proposal,
    images computed via SIS doubles, detection flag set by a crude
    "single-event detection" criterion (apparent m1det above threshold AND
    apparent dL below threshold).

    Returns
    -------
    Per-image arrays + ground-truth detection breakdown for testing.
    """
    from darksirens.utils.cosmology import dL_of_z
    rng = np.random.default_rng(seed)

    # Source-frame draws — uniform-in-coords proposals
    m1_src = rng.uniform(10.0, 70.0, n_sources)
    q = rng.uniform(0.3, 1.0, n_sources)
    z = rng.uniform(0.05, 2.0, n_sources)
    chieff = rng.uniform(-0.4, 0.4, n_sources)
    y = rng.uniform(0.05, 0.95, n_sources)  # avoid grazing y → 0, y → 1

    # Proposal densities (uniform → constant)
    p_prop_src = np.full(n_sources, 1.0 / (60 * 0.7 * 1.95 * 0.8))   # 1/volume
    p_prop_y = np.full(n_sources, 1.0 / 0.9)                          # 1/Δy

    mu_p = (1.0 + y) / y
    mu_m = (1.0 - y) / y

    # Apparent dL for each image
    dL_src = np.asarray(dL_of_z(jnp.asarray(z), H0Planck, Om0Planck))
    dL_p = dL_src / np.sqrt(mu_p)
    dL_m = dL_src / np.sqrt(mu_m)

    # Apparent m1det — same for both images
    m1det = (1.0 + z) * m1_src

    # Detection: both images are "detected" if m1det > threshold AND dL < threshold
    det_p = (m1det > det_thresh_m1det) & (dL_p < det_thresh_dL)
    det_m = (m1det > det_thresh_m1det) & (dL_m < det_thresh_dL)

    # Build per-image flat arrays (interleaved: source 0 plus, source 0 minus, source 1 plus, ...)
    n_img = 2 * n_sources
    source_id = np.repeat(np.arange(n_sources, dtype=np.int32), 2)
    image_id = np.tile(np.array([0, 1], dtype=np.int32), n_sources)
    m1_src_img = np.repeat(m1_src, 2)
    q_img = np.repeat(q, 2)
    z_img = np.repeat(z, 2)
    chieff_img = np.repeat(chieff, 2)
    y_img = np.repeat(y, 2)
    mu_img = np.empty(n_img)
    mu_img[0::2] = mu_p
    mu_img[1::2] = mu_m
    det_img = np.empty(n_img, dtype=bool)
    det_img[0::2] = det_p
    det_img[1::2] = det_m
    p_prop_src_img = np.repeat(p_prop_src, 2)
    p_prop_y_img = np.repeat(p_prop_y, 2)

    return {
        "source_id": source_id, "image_id": image_id,
        "m1_src": m1_src_img, "q_src": q_img, "z_src": z_img,
        "chieff": chieff_img, "y_source": y_img,
        "mu": mu_img, "detected": det_img,
        "p_prop_src": p_prop_src_img, "p_prop_y": p_prop_y_img,
        "n_draw_sources": n_sources,
        "n_both_detected": int(np.sum(det_p & det_m)),
    }


# ============================================================================
# A. LensedInjectionSet IO + validation
# ============================================================================

class TestLensedInjectionSetIO:

    def test_round_trip_through_disk(self):
        camp = _synth_lensed_injection_campaign(n_sources=100, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/inj.h5"
            save_lensed_injections(path=path, **{
                k: v for k, v in camp.items()
                if k not in ("n_both_detected",)
            })
            inj = load_lensed_injections(path)
        # The loaded set should have N_kept = both-detected count
        assert inj.n_kept == camp["n_both_detected"]
        # n_draw_sources preserved
        assert float(inj.n_draw_sources) == float(camp["n_draw_sources"])

    def test_validation_rejects_inconsistent_source_fields(self):
        """If two images of the same source disagree on a source-level field
        (z_src), the loader should reject."""
        camp = _synth_lensed_injection_campaign(n_sources=10, seed=1)
        # Corrupt one row: change z_src on the μ_- image of source 3
        camp["z_src"] = camp["z_src"].copy()
        camp["z_src"][7] = 99.0   # row 7 is source 3, μ_-
        with pytest.raises(ValueError, match="z_src"):
            make_lensed_injection_set(
                **{k: v for k, v in camp.items()
                   if k not in ("n_both_detected",)}
            )

    def test_validation_rejects_odd_image_count(self):
        camp = _synth_lensed_injection_campaign(n_sources=5, seed=2)
        # Drop the last image — now we have an odd number
        odd = {k: (v[:-1] if hasattr(v, "__len__") else v) for k, v in camp.items()}
        # Don't pass n_both_detected (not a constructor arg) anyway
        with pytest.raises(ValueError, match="odd|two"):
            make_lensed_injection_set(
                **{k: v for k, v in odd.items()
                   if k not in ("n_both_detected",)}
            )


# ============================================================================
# B. Cluster selection: math correctness via numpy oracle
# ============================================================================

class TestClusterSelectionMath:

    def test_per_source_log_weight_matches_numpy(self):
        """The vectorized log w_s matches a from-scratch numpy oracle."""
        camp = _synth_lensed_injection_campaign(n_sources=80, seed=10)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        cosmo = _cosmo()
        survey = _survey()
        catalog = _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        n_kept = inj.n_kept

        log_w_jax = _per_source_log_weight(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_tag_per_source=jnp.zeros(n_kept),
        )

        # Numpy reference
        m1 = np.asarray(inj.m1_src)
        q = np.asarray(inj.q_src)
        z = np.asarray(inj.z_src)
        chi = np.asarray(inj.chieff)
        y = np.asarray(inj.y_source)
        pps = np.asarray(inj.p_prop_src)
        ppy = np.asarray(inj.p_prop_y)

        log_pop = np.asarray(_toy_log_p_pop(
            jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z),
            jnp.asarray(chi), jnp.array([]),
        ))
        log_pz = np.asarray(_toy_volume_prior(jnp.asarray(z), None, None))
        log_tau = np.asarray(jnp.log(tau_2_SIS(jnp.asarray(z), sis)))
        log_py = np.log(2.0 * y)
        log_pprop = np.log(pps) + np.log(ppy)

        log_w_np = log_pop + log_pz + log_tau + log_py - log_pprop
        np.testing.assert_allclose(
            np.asarray(log_w_jax), log_w_np, rtol=1e-12, atol=1e-14,
        )

    def test_mu_estimator_matches_brute_force(self):
        """μ̂ from compute_cluster_selection_term matches a brute-force sum."""
        camp = _synth_lensed_injection_campaign(n_sources=80, seed=11)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        cosmo, survey, catalog = _cosmo(), _survey(), _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)

        log_mu, Neff, log_sigma2 = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
        )
        # Brute force: μ̂ = (1/N_draw) Σ exp(log_w)
        log_w = _per_source_log_weight(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_tag_per_source=jnp.zeros(inj.n_kept),
        )
        w = np.exp(np.asarray(log_w))
        mu_brute = w.sum() / camp["n_draw_sources"]
        np.testing.assert_allclose(float(jnp.exp(log_mu)), mu_brute, rtol=1e-12)

        # Neff < N_kept (would be == only if all weights equal)
        assert 0.0 < float(Neff) <= inj.n_kept


    def test_pair_tag_one_reproduces_default_selection(self):
        camp = _synth_lensed_injection_campaign(n_sources=80, seed=91)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        cosmo, survey, catalog = _cosmo(), _survey(), _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        default = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
        )
        explicit = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
            log_p_tag_per_source=jnp.zeros(inj.n_kept),
        )
        np.testing.assert_allclose(
            float(default[0]), float(explicit[0]), rtol=0, atol=0
        )

    def test_constant_pair_tag_shifts_log_mu_by_log_c(self):
        camp = _synth_lensed_injection_campaign(n_sources=100, seed=92)
        c = 0.37
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"},
            p_tag_per_source=np.full(camp["n_draw_sources"], c),
        )
        inj_one = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        cosmo, survey, catalog = _cosmo(), _survey(), _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        log_mu_c, _, _ = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior, inj.log_p_tag_per_source,
        )
        log_mu_one, _, _ = compute_cluster_selection_term(
            inj_one, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
        )
        np.testing.assert_allclose(
            float(log_mu_c - log_mu_one), np.log(c), rtol=1e-12
        )

    def test_lower_pair_tag_reduces_mu_and_selection_penalty(self):
        camp = _synth_lensed_injection_campaign(n_sources=100, seed=93)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        cosmo, survey, catalog = _cosmo(), _survey(), _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        log_mu_hi, _, log_sig_hi = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior, jnp.zeros(inj.n_kept),
        )
        log_mu_lo, _, log_sig_lo = compute_cluster_selection_term(
            inj, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior, jnp.log(jnp.full(inj.n_kept, 0.25)),
        )
        assert float(log_mu_lo) < float(log_mu_hi)
        corr_hi = combined_selection_log_correction(
            jnp.log(10.0), jnp.log(1e-4), log_mu_hi, log_sig_hi, 50, 1,
        )
        corr_lo = combined_selection_log_correction(
            jnp.log(10.0), jnp.log(1e-4), log_mu_lo, log_sig_lo, 50, 1,
        )
        assert float(corr_lo) > float(corr_hi)

    def test_undetected_pairs_excluded(self):
        """If we manually flip a source's detected flag, μ̂ must change."""
        camp = _synth_lensed_injection_campaign(n_sources=80, seed=12)
        # Build "good" injection set
        inj_good = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        # Now flip ALL detected flags for source 0 (set both images to False)
        camp2 = {k: v for k, v in camp.items() if k != "n_both_detected"}
        camp2["detected"] = camp2["detected"].copy()
        camp2["detected"][0:2] = False
        inj_drop = make_lensed_injection_set(**camp2)

        cosmo, survey, catalog = _cosmo(), _survey(), _toy_catalog()
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        log_mu_good, _, _ = compute_cluster_selection_term(
            inj_good, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
        )
        log_mu_drop, _, _ = compute_cluster_selection_term(
            inj_drop, cosmo, survey, jnp.array([]), catalog, sis,
            _toy_log_p_pop, _toy_volume_prior,
        )
        # μ̂ should decrease (assuming source 0 was originally detected)
        if camp["detected"][0] and camp["detected"][1]:
            assert float(log_mu_drop) < float(log_mu_good)

    def test_jit_compiles(self):
        camp = _synth_lensed_injection_campaign(n_sources=40, seed=13)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )
        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)

        @jax.jit
        def f(inj, sis):
            return compute_cluster_selection_term(
                inj, _cosmo(), _survey(), jnp.array([]), _toy_catalog(),
                sis, _toy_log_p_pop, _toy_volume_prior,
            )
        log_mu, Neff, log_sigma2 = f(inj, sis)
        assert jnp.isfinite(log_mu)
        assert float(Neff) > 0.0


# ============================================================================
# C. Combined selection correction
# ============================================================================

class TestCombinedSelectionCorrection:

    def test_collapses_to_singleton_form_at_cluster_minus_inf(self):
        """When log_mu_cluster = -inf and log_sigma2_cluster = -inf,
        combined_selection_log_correction returns exactly the singleton
        form -N log μ + N(N+3)/(2 N_eff).

        Pick parameters so N_eff comfortably passes the 5N MFG threshold:
        log_mu = 0.5, log_sigma2 = -8.0 → N_eff ≈ e^9 ≈ 8100 > 5×10 = 50.
        """
        log_mu_1 = jnp.asarray(0.5)              # μ_1 = e^0.5
        log_sigma2_1 = jnp.asarray(-8.0)          # σ²_1 = e^-8.0
        log_mu_2 = jnp.asarray(-jnp.inf)
        log_sigma2_2 = jnp.asarray(-jnp.inf)
        N = 10
        Neff = float(jnp.exp(2.0 * log_mu_1 - log_sigma2_1))
        assert Neff > 5 * N, "Test fixture must have Neff above the MFG threshold"
        expected = -N * float(log_mu_1) + N * (3 + N) / (2 * Neff)

        got = combined_selection_log_correction(
            log_mu_1, log_sigma2_1, log_mu_2, log_sigma2_2,
            n_singletons_observed=N, n_clusters_observed=0,
        )
        np.testing.assert_allclose(float(got), expected, rtol=1e-12)

    def test_neff_threshold_gates_to_minus_inf(self):
        """When N_eff_tot ≤ 5 N_tot, correction is -inf."""
        log_mu_1 = jnp.asarray(0.0)              # μ_1 = 1
        log_sigma2_1 = jnp.asarray(0.0)           # σ²_1 = 1, so N_eff = 1
        log_mu_2 = jnp.asarray(-jnp.inf)
        log_sigma2_2 = jnp.asarray(-jnp.inf)
        N = 10  # Need N_eff > 50
        got = combined_selection_log_correction(
            log_mu_1, log_sigma2_1, log_mu_2, log_sigma2_2,
            n_singletons_observed=N, n_clusters_observed=0,
        )
        assert jnp.isinf(got) and got < 0

    def test_combined_with_actual_clusters(self):
        """With both channels active, μ_tot = μ_1 + μ_2 in linear space."""
        log_mu_1 = jnp.log(2.0)
        log_sigma2_1 = jnp.log(0.05)
        log_mu_2 = jnp.log(0.3)
        log_sigma2_2 = jnp.log(0.01)
        N_s, N_c = 5, 2
        N_tot = N_s + N_c
        # Predict by hand
        mu_tot = 2.0 + 0.3
        sigma2_tot = 0.05 + 0.01
        Neff_tot = mu_tot ** 2 / sigma2_tot
        assert Neff_tot > 5 * N_tot   # threshold OK
        expected = -N_tot * np.log(mu_tot) + N_tot * (3 + N_tot) / (2 * Neff_tot)

        got = combined_selection_log_correction(
            log_mu_1, log_sigma2_1, log_mu_2, log_sigma2_2,
            n_singletons_observed=N_s, n_clusters_observed=N_c,
        )
        np.testing.assert_allclose(float(got), expected, rtol=1e-10)


# ============================================================================
# D. Master likelihood: exact reduction at cluster_mode=OFF
# ============================================================================

class TestMasterLikelihoodReduction:
    """The strongest test: master likelihood with cluster_mode=OFF
    must give bit-identical output to commit 2's darksiren_log_likelihood.
    """

    @pytest.fixture(scope="class")
    def fixture(self):
        rng = np.random.default_rng(0)
        n_events, n_samp, n_sel = 4, 200, 500
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
        pop_params = get_fixed_population_params("powerlaw+peak")
        assert pop_params.ndim == 1
        assert pop_params.shape[0] > 0

        return {
            "cosmo": _cosmo(), "survey": _survey(),
            "gw_pe": gw_pe, "gw_sel": gw_sel,
            "catalog": _toy_catalog(),
            "n_events": n_events, "n_samp": n_samp,
            "Ndraw": 1000.0,
            "pop_params": pop_params,
        }

    def test_cluster_mode_off_empty_pop_params_raises_clear_error(self, fixture):
        """Empty pop_params should raise a clear ValueError before JAX internals."""
        from darksirens.likelihood.likelihood_with_clusters import (
            darksiren_log_likelihood_with_clusters,
            CLUSTER_MODE_OFF,
        )

        with pytest.raises(
            ValueError,
            match=(
                r"darksiren_log_likelihood_with_clusters received empty pop_params: "
                r"pop_model='powerlaw\+peak', pop_params.shape=\(0,\)"
            ),
        ):
            darksiren_log_likelihood_with_clusters(
                fixture["cosmo"], fixture["survey"], jnp.array([]),
                fixture["gw_pe"], fixture["catalog"],
                fixture["gw_sel"], fixture["catalog"],
                fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
                singleton_indices=jnp.arange(fixture["n_events"], dtype=jnp.int32),
                pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
                n_singletons=fixture["n_events"], n_pairs=0,
                lensed_injections=None,
                pair_kdes=None,
                sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
                log_p_tag_per_source=jnp.zeros(0),
                pop_model="powerlaw+peak",
                universe_model="spectral_sirens",
                sel_batch_size=None,
                cluster_mode=CLUSTER_MODE_OFF,
            )

    def test_cluster_mode_off_matches_commit2(self, fixture):
        assert fixture["pop_params"].shape[0] > 0, "fixture pop_params must be non-empty for pop_model=powerlaw+peak"
        """With cluster_mode=OFF, the cluster-aware master likelihood must
        match commit 2's darksiren_log_likelihood bit-identically."""
        from darksirens.likelihood.core import (
            darksiren_log_likelihood, WL_BACKEND_DISABLED,
        )
        from darksirens.likelihood.likelihood_with_clusters import (
            darksiren_log_likelihood_with_clusters,
            CLUSTER_MODE_OFF, WL_BACKEND_DISABLED as WL_BACKEND_DISABLED_C4,
        )
        # commit 2 path
        ll_commit2 = darksiren_log_likelihood(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            fixture["gw_pe"], fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
        )

        # commit 4 path with cluster_mode=OFF
        ll_commit4 = darksiren_log_likelihood_with_clusters(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            fixture["gw_pe"], fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
            singleton_indices=jnp.arange(fixture["n_events"], dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=fixture["n_events"], n_pairs=0,
            lensed_injections=None,
            pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
            log_p_tag_per_source=jnp.zeros(0),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
            cluster_mode=CLUSTER_MODE_OFF,
            wl_backend=WL_BACKEND_DISABLED_C4,
        )
        # Must be bit-identical
        np.testing.assert_allclose(
            float(ll_commit4), float(ll_commit2),
            rtol=1e-12, atol=1e-14,
        )


    def test_cluster_mode_off_diagnostics_have_no_pair_terms(self, fixture):
        from darksirens.likelihood.likelihood_with_clusters import (
            darksiren_likelihood_diagnostics_with_clusters,
            CLUSTER_MODE_OFF, WL_BACKEND_DISABLED as WL_BACKEND_DISABLED_C4,
        )

        diag = darksiren_likelihood_diagnostics_with_clusters(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            fixture["gw_pe"], fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
            singleton_indices=jnp.arange(fixture["n_events"], dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=fixture["n_events"], n_pairs=0,
            lensed_injections=None,
            pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
            log_p_tag_per_source=jnp.zeros(0),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
            cluster_mode=CLUSTER_MODE_OFF,
            wl_backend=WL_BACKEND_DISABLED_C4,
        )

        assert int(diag["n_pairs"]) == 0
        assert np.asarray(diag["per_pair_logL"]).shape == (0,)
        assert float(diag["pair_logL_sum"]) == pytest.approx(0.0)
        assert jnp.isfinite(diag["logL_total"])

    def test_cluster_mode_j2_runs_end_to_end(self, fixture):
        assert fixture["pop_params"].shape[0] > 0, "fixture pop_params must be non-empty for pop_model=powerlaw+peak"
        """Master likelihood with cluster_mode=J2 evaluates finite for a
        synthetic dataset with one pair and one singleton.

        Layout:
            event 0: μ_+ image, sees one source
            event 1: μ_- image, same source
            event 2: independent singleton
            event 3: independent singleton

        Pair: (0, 1). Singletons: [2, 3].
        """
        from darksirens.likelihood.likelihood_with_clusters import (
            darksiren_log_likelihood_with_clusters,
            CLUSTER_MODE_J2,
        )
        from darksirens.likelihood.pair_kde import make_pair_kde, stack_pair_kdes

        # Build a synthetic lensed pair for events (0, 1), and use existing PE
        # samples for events 2, 3 as "noise" singletons.
        from darksirens.utils.cosmology import dL_of_z
        rng = np.random.default_rng(100)
        n_pe = fixture["n_samp"]
        z_true, m1src_true, y_true = 0.7, 30.0, 0.4
        dL_src = float(dL_of_z(jnp.asarray(z_true), H0Planck, Om0Planck))
        mu_p = (1.0 + y_true) / y_true
        mu_m = (1.0 - y_true) / y_true
        dL_app_p = dL_src / np.sqrt(mu_p)
        dL_app_m = dL_src / np.sqrt(mu_m)
        m1det_true = (1.0 + z_true) * m1src_true

        # Synthesize event 0 (μ_+) and event 1 (μ_-) as the pair
        def _synth(dL_apparent):
            m1det = m1det_true + rng.normal(0, 1.0, n_pe)
            q = 0.7 + rng.normal(0, 0.05, n_pe)
            dL = dL_apparent + rng.normal(0, 0.05 * dL_apparent, n_pe)
            chieff = 0.0 + rng.normal(0, 0.03, n_pe)
            prior_wt = np.ones(n_pe)
            valid = np.ones(n_pe, dtype=bool)
            return m1det, q, dL, chieff, prior_wt, valid

        # Override events 0, 1 in the GWEvent with the synthetic pair
        gw_pe_orig = fixture["gw_pe"]
        m1det_new = np.asarray(gw_pe_orig.m1det).copy()
        q_new = np.asarray(gw_pe_orig.q).copy()
        dL_new = np.asarray(gw_pe_orig.dL).copy()
        chieff_new = np.asarray(gw_pe_orig.chieff).copy()
        prior_wt_new = np.asarray(gw_pe_orig.prior_wt).copy()
        valid_new = np.asarray(gw_pe_orig.valid).copy()

        for ev_idx, dL_app in [(0, dL_app_p), (1, dL_app_m)]:
            s_lo, s_hi = ev_idx * n_pe, (ev_idx + 1) * n_pe
            m1det, q, dL, chieff, pw, v = _synth(dL_app)
            m1det_new[s_lo:s_hi] = m1det
            q_new[s_lo:s_hi] = q
            dL_new[s_lo:s_hi] = dL
            chieff_new[s_lo:s_hi] = chieff
            prior_wt_new[s_lo:s_hi] = pw
            valid_new[s_lo:s_hi] = v

        gw_pe = GWEvent(
            m1det=jnp.asarray(m1det_new), m2det=gw_pe_orig.m2det,
            dL=jnp.asarray(dL_new), chieff=jnp.asarray(chieff_new),
            prior_wt=jnp.asarray(prior_wt_new), pixels=gw_pe_orig.pixels,
            q=jnp.asarray(q_new), valid=jnp.asarray(valid_new),
        )

        # Build per-event PairKDEs (one per global event index)
        kdes = []
        for ev_idx in range(fixture["n_events"]):
            s_lo, s_hi = ev_idx * n_pe, (ev_idx + 1) * n_pe
            kdes.append(make_pair_kde(
                m1det=m1det_new[s_lo:s_hi], q=q_new[s_lo:s_hi],
                dL_app=dL_new[s_lo:s_hi], chieff=chieff_new[s_lo:s_hi],
                prior_wt=prior_wt_new[s_lo:s_hi],
            ))
        stacked = stack_pair_kdes(kdes)

        # Build a lensed-injection set
        camp = _synth_lensed_injection_campaign(n_sources=2000, seed=999)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )

        sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0)
        ll = darksiren_log_likelihood_with_clusters(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            gw_pe, fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], n_pe, fixture["Ndraw"],
            singleton_indices=jnp.asarray([2, 3], dtype=jnp.int32),
            pair_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
            n_singletons=2, n_pairs=1,
            lensed_injections=inj,
            pair_kdes=stacked,
            sis_params=sis,
            log_p_tag_per_source=jnp.zeros(inj.n_kept),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
            cluster_mode=CLUSTER_MODE_J2,
        )
        # Just check that the master likelihood evaluates to a finite number.
        # The actual VALUE depends on injection randomness; we only assert
        # that the cluster path doesn't blow up.
        assert jnp.isfinite(ll), f"cluster_mode=J2 returned non-finite ll: {ll}"

        from darksirens.likelihood.likelihood_with_clusters import darksiren_likelihood_diagnostics_with_clusters
        diag = darksiren_likelihood_diagnostics_with_clusters(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            gw_pe, fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], n_pe, fixture["Ndraw"],
            singleton_indices=jnp.asarray([2, 3], dtype=jnp.int32),
            pair_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
            n_singletons=2, n_pairs=1,
            lensed_injections=inj,
            pair_kdes=stacked,
            sis_params=sis,
            log_p_tag_per_source=jnp.zeros(inj.n_kept),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
            cluster_mode=CLUSTER_MODE_J2,
        )
        assert jnp.isfinite(diag["logL_total"])
        assert int(diag["n_pairs"]) == 1
        assert np.asarray(diag["per_pair_logL"]).shape == (1,)
        assert np.all(np.isfinite(np.asarray(diag["per_pair_logL"])))

    def test_cluster_mode_j2_pair_outscores_swapped_singletons(self, fixture):
        assert fixture["pop_params"].shape[0] > 0, "fixture pop_params must be non-empty for pop_model=powerlaw+peak"
        """The master likelihood with a TRUE lensed pair declared as a pair
        should outscore the SAME data with those two events declared as
        singletons (and the rest declared as pair). This is a CRUCIAL
        partition-sensitivity test.

        We rebuild the data with a synthetic pair on events (0, 1) and
        compare:
            partition A: pair=(0,1), singletons=[2,3]  ← truth
            partition B: pair=(2,3), singletons=[0,1]  ← wrong

        Partition A should outscore B (when the synthetic pair is strong).
        """
        from darksirens.likelihood.likelihood_with_clusters import (
            darksiren_log_likelihood_with_clusters,
            CLUSTER_MODE_J2,
        )
        from darksirens.likelihood.pair_kde import make_pair_kde, stack_pair_kdes
        from darksirens.utils.cosmology import dL_of_z

        # Same synthetic-pair construction as above
        rng = np.random.default_rng(101)
        n_pe = fixture["n_samp"]
        z_true, m1src_true, y_true = 0.7, 30.0, 0.4
        dL_src = float(dL_of_z(jnp.asarray(z_true), H0Planck, Om0Planck))
        mu_p = (1.0 + y_true) / y_true
        mu_m = (1.0 - y_true) / y_true
        dL_app_p = dL_src / np.sqrt(mu_p)
        dL_app_m = dL_src / np.sqrt(mu_m)
        m1det_true = (1.0 + z_true) * m1src_true

        def _synth(dL_apparent):
            m1det = m1det_true + rng.normal(0, 1.0, n_pe)
            q = 0.7 + rng.normal(0, 0.05, n_pe)
            dL = dL_apparent + rng.normal(0, 0.05 * dL_apparent, n_pe)
            chieff = 0.0 + rng.normal(0, 0.03, n_pe)
            return m1det, q, dL, chieff

        gw_pe_orig = fixture["gw_pe"]
        m1det_new = np.asarray(gw_pe_orig.m1det).copy()
        q_new = np.asarray(gw_pe_orig.q).copy()
        dL_new = np.asarray(gw_pe_orig.dL).copy()
        chieff_new = np.asarray(gw_pe_orig.chieff).copy()
        prior_wt_new = np.asarray(gw_pe_orig.prior_wt).copy()
        valid_new = np.asarray(gw_pe_orig.valid).copy()

        for ev_idx, dL_app in [(0, dL_app_p), (1, dL_app_m)]:
            s_lo, s_hi = ev_idx * n_pe, (ev_idx + 1) * n_pe
            m1det, q, dL, chieff = _synth(dL_app)
            m1det_new[s_lo:s_hi] = m1det
            q_new[s_lo:s_hi] = q
            dL_new[s_lo:s_hi] = dL
            chieff_new[s_lo:s_hi] = chieff

        gw_pe = GWEvent(
            m1det=jnp.asarray(m1det_new), m2det=gw_pe_orig.m2det,
            dL=jnp.asarray(dL_new), chieff=jnp.asarray(chieff_new),
            prior_wt=jnp.asarray(prior_wt_new), pixels=gw_pe_orig.pixels,
            q=jnp.asarray(q_new), valid=jnp.asarray(valid_new),
        )

        kdes = []
        for ev_idx in range(fixture["n_events"]):
            s_lo, s_hi = ev_idx * n_pe, (ev_idx + 1) * n_pe
            kdes.append(make_pair_kde(
                m1det=m1det_new[s_lo:s_hi], q=q_new[s_lo:s_hi],
                dL_app=dL_new[s_lo:s_hi], chieff=chieff_new[s_lo:s_hi],
                prior_wt=prior_wt_new[s_lo:s_hi],
            ))
        stacked = stack_pair_kdes(kdes)

        camp = _synth_lensed_injection_campaign(n_sources=2000, seed=999)
        inj = make_lensed_injection_set(
            **{k: v for k, v in camp.items() if k != "n_both_detected"}
        )

        def _call(pair_idx, sing_idx):
            return darksiren_log_likelihood_with_clusters(
                fixture["cosmo"], fixture["survey"], fixture["pop_params"],
                gw_pe, fixture["catalog"],
                fixture["gw_sel"], fixture["catalog"],
                fixture["n_events"], n_pe, fixture["Ndraw"],
                singleton_indices=jnp.asarray(sing_idx, dtype=jnp.int32),
                pair_indices=jnp.asarray([pair_idx], dtype=jnp.int32),
                n_singletons=len(sing_idx), n_pairs=1,
                lensed_injections=inj,
                pair_kdes=stacked,
                sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
                log_p_tag_per_source=jnp.zeros(inj.n_kept),
                pop_model="powerlaw+peak",
                universe_model="spectral_sirens",
                sel_batch_size=None,
                cluster_mode=CLUSTER_MODE_J2,
            )

        ll_truth = _call([0, 1], [2, 3])
        ll_swapped = _call([2, 3], [0, 1])
        # The true partition should give a higher likelihood
        diff = float(ll_truth - ll_swapped)
        assert diff > 0.0, (
            f"True partition (0,1)-pair did not outscore wrong partition: "
            f"ll_truth={ll_truth}, ll_swapped={ll_swapped}, diff={diff}"
        )


def test_lensing_cli_threads_sl_tau_A_into_prior_midpoint_likelihood(monkeypatch):
    """Changing --sl_tau_A must change the J=2 likelihood closure inputs."""
    from types import SimpleNamespace
    from darksirens.cli import inference_lensing

    class _Decoder:
        def decode(self, coord):
            del coord
            return _cosmo(), _survey(), jnp.ones(1), None, None

    def _fake_cluster_likelihood(*args, **kwargs):
        del kwargs
        # build_cluster_likelihood passes sis_params immediately before log_p_tag.
        sis_params = args[16]
        return sis_params.A_tau

    monkeypatch.setattr(
        inference_lensing,
        "darksiren_log_likelihood_with_clusters",
        _fake_cluster_likelihood,
    )

    inp = dict(
        gw_pe=None, gw_sel=None, nEvents=2, nsamp=1, Ndraw=1.0,
        singleton_indices=jnp.asarray([], dtype=jnp.int32),
        pair_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        n_singletons=0, n_pairs=1, pair_kdes=None, lensed=SimpleNamespace(m1_src=jnp.ones(2)),
    )

    def _opts(sl_tau_A):
        return SimpleNamespace(
            sl_tau_A=sl_tau_A,
            sl_tau_n=3.0,
            cluster_mode="j2",
            wl_backend="disabled",
            wl_selection="standard",
            universe_model="spectral_sirens",
            pop_model="powerlaw+peak",
            sel_batch_size=None,
            lensing_wl_a=4e-3,
            lensing_wl_b=1.5,
        )

    midpoint = jnp.zeros(1)
    low = float(inference_lensing.build_cluster_likelihood(_opts(1.0e-4), inp, _Decoder())(midpoint))
    high = float(inference_lensing.build_cluster_likelihood(_opts(9.0e-4), inp, _Decoder())(midpoint))

    assert low == pytest.approx(1.0e-4)
    assert high == pytest.approx(9.0e-4)
    assert high != low


def test_lensing_cli_sampled_lens_params_affect_cluster_and_pair_terms(monkeypatch):
    """Sampled log10_tau_A/tau_n must be decoded into SISLensParams."""
    from types import SimpleNamespace
    from darksirens.cli import inference_lensing

    class _Decoder:
        def decode(self, coord):
            del coord
            return _cosmo(), _survey(), jnp.ones(1), None, None

    seen = []
    def _fake_cluster_likelihood(*args, **kwargs):
        del kwargs
        sis_params = args[16]
        seen.append((float(sis_params.A_tau), float(sis_params.n_tau)))
        # Mimic a log_mu_cluster/pair-likelihood dependence through tau_2.
        return jnp.log(sis_params.A_tau) + sis_params.n_tau * jnp.log(2.0)

    monkeypatch.setattr(
        inference_lensing, "darksiren_log_likelihood_with_clusters", _fake_cluster_likelihood
    )
    inp = dict(
        gw_pe=None, gw_sel=None, nEvents=2, nsamp=1, Ndraw=1.0,
        singleton_indices=jnp.asarray([], dtype=jnp.int32),
        pair_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        n_singletons=0, n_pairs=1, pair_kdes=None, lensed=SimpleNamespace(m1_src=jnp.ones(2)),
    )
    opts = SimpleNamespace(
        fix_lens_rate=False, sl_tau_A=5e-4, sl_tau_n=3.0,
        cluster_mode="j2", wl_backend="disabled", wl_selection="standard",
        universe_model="spectral_sirens", pop_model="powerlaw+peak", sel_batch_size=None,
        lensing_wl_a=4e-3, lensing_wl_b=1.5, pair_marks="none",
    )

    loglike = inference_lensing.build_cluster_likelihood(
        opts, inp, _Decoder(), ["log10_tau_A", "tau_n"], {}
    )
    low = float(loglike(jnp.asarray([-6.0, 1.0])))
    high = float(loglike(jnp.asarray([-3.0, 4.0])))

    # pytest.approx does not recurse into tuples (10**x differs from the
    # literal by one float64 ULP), so compare fields directly.
    assert [len(s) for s in seen] == [2, 2]
    assert [s[0] for s in seen] == pytest.approx([1.0e-6, 1.0e-3])
    assert [s[1] for s in seen] == pytest.approx([1.0, 4.0])
    assert high != low


def test_lensing_cli_lens_labels_appended_when_rate_sampled():
    """Local lens priors are appended only when --fix_lens_rate false."""
    from types import SimpleNamespace
    from darksirens.cli import inference_lensing

    opts = SimpleNamespace(fix_lens_rate=False)
    labels, lower, upper = inference_lensing._build_lens_parameter_space(opts, {}, {})
    assert labels == ["log10_tau_A", "tau_n"]
    assert lower.tolist() == [-7.0, 0.0]
    assert upper.tolist() == [-2.0, 6.0]

    labels, lower, upper = inference_lensing._build_lens_parameter_space(
        opts, {"tau_n": 3.0}, {"log10_tau_A": [-5.0, -3.0]}
    )
    assert labels == ["log10_tau_A"]
    assert lower.tolist() == [-5.0]
    assert upper.tolist() == [-3.0]


@pytest.fixture(scope="module")
def wl_selection_fixture():
    rng = np.random.default_rng(0)
    n_events, n_samp, n_sel = 4, 200, 500
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
    return {
        "cosmo": _cosmo(), "survey": _survey(),
        "gw_pe": gw_pe, "gw_sel": gw_sel,
        "catalog": _toy_catalog(),
        "n_events": n_events, "n_samp": n_samp,
        "Ndraw": 1000.0,
        "pop_params": get_fixed_population_params("powerlaw+peak"),
    }


def _cluster_diag_for_wl_selection(fixture, *, wl_a, wl_selection):
    from darksirens.likelihood.likelihood_with_clusters import (
        darksiren_likelihood_diagnostics_with_clusters,
        CLUSTER_MODE_OFF,
        WL_BACKEND_LOGNORMAL,
        WL_SELECTION_STANDARD,
        WL_SELECTION_LOGNORMAL,
    )

    return darksiren_likelihood_diagnostics_with_clusters(
        fixture["cosmo"], fixture["survey"], fixture["pop_params"],
        fixture["gw_pe"], fixture["catalog"],
        fixture["gw_sel"], fixture["catalog"],
        fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
        singleton_indices=jnp.arange(fixture["n_events"], dtype=jnp.int32),
        pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
        n_singletons=fixture["n_events"], n_pairs=0,
        lensed_injections=None,
        pair_kdes=None,
        sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
        log_p_tag_per_source=jnp.zeros(0),
        pop_model="powerlaw+peak",
        universe_model="spectral_sirens_wl",
        sel_batch_size=None,
        cluster_mode=CLUSTER_MODE_OFF,
        wl_backend=WL_BACKEND_LOGNORMAL,
        wl_a=wl_a,
        wl_b=1.5,
        wl_selection=(
            WL_SELECTION_LOGNORMAL if wl_selection == "wl_lognormal"
            else WL_SELECTION_STANDARD
        ),
    )


class TestWeakLensingSingletonSelection:
    def test_wl_selection_lognormal_wl_a_zero_matches_standard_log_mu(self, wl_selection_fixture):
        diag_standard = _cluster_diag_for_wl_selection(
            wl_selection_fixture, wl_a=0.0, wl_selection="standard"
        )
        diag_wl = _cluster_diag_for_wl_selection(
            wl_selection_fixture, wl_a=0.0, wl_selection="wl_lognormal"
        )

        np.testing.assert_allclose(
            float(diag_wl["log_mu_singleton"]),
            float(diag_standard["log_mu_singleton"]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_wl_selection_lognormal_changes_nonzero_wl_a_log_mu(self, wl_selection_fixture):
        diag_standard = _cluster_diag_for_wl_selection(
            wl_selection_fixture, wl_a=0.0, wl_selection="standard"
        )
        diag_wl = _cluster_diag_for_wl_selection(
            wl_selection_fixture, wl_a=0.2, wl_selection="wl_lognormal"
        )

        assert np.isfinite(float(diag_wl["log_mu_singleton"]))
        assert abs(float(diag_wl["log_mu_singleton"] - diag_standard["log_mu_singleton"])) > 1e-6

    def test_inference_lensing_records_wl_selection_in_outputs(self):
        import inspect
        from darksirens.cli import inference_lensing

        parser = inference_lensing.build_parser()
        opts = parser.parse_args([
            "--gw_path", "gw.h5",
            "--gwselection_path", "sel.h5",
            "--sampler", "tinyns",
        ])
        assert opts.wl_selection == "standard"

        # main() is decomposed into phase functions; the wl_selection attr is
        # written by the save phase and the full opts still route through
        # _jsonable_settings (unchanged `for k, v in vars(opts).items()`)
        # in the run-dir preparation phase.
        save_src = inspect.getsource(inference_lensing._save_lensing_outputs)
        assert 'f.attrs["wl_selection"] = opts.wl_selection' in save_src
        prepare_src = inspect.getsource(inference_lensing._prepare_run_dir)
        assert "_jsonable_settings(opts)" in prepare_src
        helper_src = inspect.getsource(inference_lensing._jsonable_settings)
        assert "for k, v in vars(opts).items()" in helper_src

# PR08 candidate-pair partition enumeration tests

def test_candidate_pair_validation_rejects_duplicate_and_self_pairs():
    from darksirens.lensing.partitions import validate_candidate_pairs
    import pytest

    with pytest.raises(ValueError, match="self-pair"):
        validate_candidate_pairs({
            "n_events": 2,
            "candidate_pairs": [{"i": 1, "j": 1, "log_prior_odds": 0.0}],
        })

    with pytest.raises(ValueError, match="duplicate unordered"):
        validate_candidate_pairs({
            "n_events": 3,
            "candidate_pairs": [
                {"i": 0, "j": 2, "log_prior_odds": 0.0},
                {"i": 2, "j": 0, "log_prior_odds": 1.0},
            ],
        })


def test_exact_enumeration_counts_simple_graphs():
    from darksirens.lensing.partitions import (
        enumerate_compatible_partitions,
        validate_candidate_pairs,
    )

    n_events, pairs = validate_candidate_pairs({
        "n_events": 3,
        "candidate_pairs": [
            {"i": 0, "j": 1, "log_prior_odds": 0.0},
            {"i": 1, "j": 2, "log_prior_odds": 0.0},
        ],
    })
    states = enumerate_compatible_partitions(n_events, pairs)
    assert len(states) == 3  # empty + either edge in the length-2 path
    assert sorted(state.n_pairs for state in states) == [0, 1, 1]

    n_events, pairs = validate_candidate_pairs({
        "n_events": 4,
        "candidate_pairs": [
            {"i": 0, "j": 1, "log_prior_odds": 0.0},
            {"i": 2, "j": 3, "log_prior_odds": 0.0},
        ],
    })
    states = enumerate_compatible_partitions(n_events, pairs)
    assert len(states) == 4  # empty, each edge alone, both disjoint edges
    assert max(state.n_pairs for state in states) == 2


def test_exact_enumeration_rejects_over_limit():
    from darksirens.lensing.partitions import (
        enumerate_compatible_partitions,
        validate_candidate_pairs,
    )
    import pytest

    n_events, pairs = validate_candidate_pairs({
        "n_events": 4,
        "candidate_pairs": [
            {"i": 0, "j": 1, "log_prior_odds": 0.0},
            {"i": 2, "j": 3, "log_prior_odds": 0.0},
        ],
    })
    with pytest.raises(ValueError, match="max_partitions=2"):
        enumerate_compatible_partitions(n_events, pairs, max_partitions=2)

# Lensing preflight tests

def _write_preflight_mock(tmp_path, *, n_events=3, bad_prior=False, include_time=True):
    import json
    import h5py
    import numpy as np
    from types import SimpleNamespace

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    lens = tmp_path / "lensed.h5"
    pair = tmp_path / "pair.h5"
    part = tmp_path / "partition.json"
    cand = tmp_path / "candidate_pairs.json"
    nsamp = 4
    with h5py.File(gw, "w") as f:
        f.attrs["nobs"] = n_events
        f.attrs["nsamp"] = nsamp
        f.create_dataset("m1det", data=np.ones(n_events * nsamp))
    with h5py.File(sel, "w") as f:
        f.attrs["ndraw"] = 10
    with h5py.File(lens, "w") as f:
        f.attrs["Ndraw_sources"] = 5
        f.create_dataset("p_tag_per_source", data=np.ones(5))
    with h5py.File(pair, "w") as f:
        f.attrs["npairs"] = 1
        g = f.create_group("pair_0")
        g.attrs["event_index_image0"] = 1
        g.attrs["event_index_image1"] = 2
        if include_time:
            g.attrs["delta_t_obs"] = 12.0
            g.attrs["sigma_delta_t"] = 1.0
        for name in ("image0", "image1"):
            gi = g.create_group(name)
            gi.create_dataset("m1det", data=np.arange(1, nsamp + 1.0))
            gi.create_dataset("q", data=np.full(nsamp, 0.8))
            gi.create_dataset("dL_app", data=np.arange(10, 10 + nsamp, dtype=float))
            gi.create_dataset("chieff", data=np.zeros(nsamp))
            prior = np.ones(nsamp)
            if bad_prior and name == "image1":
                prior[0] = 0.0
            gi.create_dataset("prior_wt", data=prior)
    part.write_text(json.dumps({"singleton_indices": [0], "pair_indices": [[1, 2]], "n_singletons": 1, "n_pairs": 1}))
    cand.write_text(json.dumps({"n_events": n_events, "candidate_pairs": [{"i": 1, "j": 2, "log_prior_odds": 0.0}]}))
    opts = SimpleNamespace(
        gw_path=str(gw), gwselection_path=str(sel), lensed_injections_path=str(lens),
        observed_catalog_path=None,
        pair_pe_path=str(pair), partition_path=str(part), candidate_pairs_path=str(cand),
        partition_mode="fixed", cluster_mode="j2", pair_marks="none", pair_time_sigma_sec=None,
        wl_backend="lognormal", wl_selection="standard", fix_lens_rate=True,
        sl_tau_A=5e-4, sl_tau_n=3.0, lens_prior_overrides=None, max_exact_partitions=10000,
    )
    return opts


def test_lensing_preflight_valid_tiny_mock_passes(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    result = run_lensing_preflight(opts)
    assert result["ok"], result
    assert result["summary"]["n_events"] == 3
    assert result["summary"]["n_pairs_pair_pe"] == 1


def _write_observed_catalog(path, n_events):
    import json
    path.write_text(json.dumps({
        "format_version": "observed-lensing-catalog-1.0",
        "event_indexing": "global",
        "n_events": n_events,
        "events": [
            {
                "event_index": i,
                "event_id": f"mock_event_{i:03d}",
                "kind": "singleton_or_image",
                "gps_time": 1234567890.0 + i,
                "truth_source_id": i,
                "truth_image_index": None,
                "truth_is_lensed_image": False,
            }
            for i in range(n_events)
        ],
    }))
    return path


def test_observed_catalog_valid_passes(tmp_path):
    from darksirens.lensing.observed_catalog import validate_observed_catalog_file
    meta = validate_observed_catalog_file(_write_observed_catalog(tmp_path / "observed_catalog.json", 3))
    assert meta["n_events"] == 3


def test_observed_catalog_missing_required_fields_fail(tmp_path):
    from darksirens.lensing.observed_catalog import validate_observed_catalog_file
    import json
    path = tmp_path / "bad_observed_catalog.json"
    path.write_text(json.dumps({"event_indexing": "global", "n_events": 0, "events": []}))
    with pytest.raises(ValueError, match="format_version"):
        validate_observed_catalog_file(path)


def test_observed_catalog_noncontiguous_event_indices_fail(tmp_path):
    from darksirens.lensing.observed_catalog import validate_observed_catalog_file
    import json
    path = _write_observed_catalog(tmp_path / "observed_catalog.json", 3)
    data = json.loads(path.read_text())
    data["events"][2]["event_index"] = 4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contiguous"):
        validate_observed_catalog_file(path)


def test_lensing_preflight_observed_catalog_n_events_mismatch_errors(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.observed_catalog_path = str(_write_observed_catalog(tmp_path / "observed_catalog.json", 4))
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("observed_catalog n_events=4" in e for e in result["errors"])


def test_explicit_observed_catalog_path_overrides_legacy_heuristic(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.observed_catalog_path = str(_write_observed_catalog(tmp_path / "observed_catalog.json", 3))
    result = run_lensing_preflight(opts)
    assert result["ok"], result
    assert result["summary"]["unified_observed_mode"] is True
    assert not result["summary"].get("unified_observed_mode_inferred_heuristic", False)


def test_lensing_preflight_missing_pair_pe_path_errors_for_j2(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.pair_pe_path = str(tmp_path / "missing_pair.h5")
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("pair_pe_path" in e for e in result["errors"])


def test_lensing_preflight_duplicate_fixed_partition_errors(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    import json
    opts = _write_preflight_mock(tmp_path)
    opts.partition_path = str(tmp_path / "dup_partition.json")
    (tmp_path / "dup_partition.json").write_text(json.dumps({"singleton_indices": [1], "pair_indices": [[1, 2]], "n_singletons": 1, "n_pairs": 1}))
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("more than once" in e for e in result["errors"])


def test_lensing_preflight_candidate_pairs_n_events_mismatch_errors(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    import json
    opts = _write_preflight_mock(tmp_path)
    opts.partition_mode = "marginalize_exact"
    (tmp_path / "candidate_pairs.json").write_text(json.dumps({"n_events": 4, "candidate_pairs": [{"i": 1, "j": 2, "log_prior_odds": 0.0}]}))
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("does not match gw n_events" in e for e in result["errors"])


def test_lensing_preflight_pair_marks_time_requires_delta_t_obs(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path, include_time=False)
    opts.pair_marks = "time"
    opts.pair_time_sigma_sec = 2.0
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("delta_t_obs" in e for e in result["errors"])



def test_lensing_preflight_malformed_prior_wt_errors(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path, bad_prior=True)
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("prior_wt" in e and "finite and positive" in e for e in result["errors"])

def test_lensing_preflight_only_cli_exits_zero_on_valid_mock(tmp_path):
    import json
    import subprocess
    import sys
    opts = _write_preflight_mock(tmp_path)
    out = tmp_path / "preflight.json"
    cmd = [
        sys.executable, "-m", "darksirens.cli.inference_lensing",
        "--gw_path", opts.gw_path,
        "--gwselection_path", opts.gwselection_path,
        "--lensed_injections_path", opts.lensed_injections_path,
        "--pair_pe_path", opts.pair_pe_path,
        "--partition_path", opts.partition_path,
        "--cluster_mode", "j2",
        "--sampler", "dynesty",
        "--preflight_only", "true",
        "--preflight_json", str(out),
        "--save_path", str(tmp_path / "run"),
    ]
    proc = subprocess.run(cmd, cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1]), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout
    payload = json.loads(out.read_text())
    assert payload["ok"] is True


def test_lensing_preflight_marginalized_time_marks_require_candidate_edge_marks(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.partition_mode = "marginalize_exact"
    opts.pair_marks = "time"
    result = run_lensing_preflight(opts)
    assert not result["ok"]
    assert any("missing marks.delta_t_obs/sigma_delta_t" in e for e in result["errors"])


def test_lensing_preflight_marginalized_time_marks_pass_with_edge_marks(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.partition_mode = "marginalize_exact"
    opts.pair_marks = "time"
    import json
    path = __import__('pathlib').Path(opts.candidate_pairs_path)
    data = json.loads(path.read_text())
    for pair in data["candidate_pairs"]:
        pair["marks"] = {"delta_t_obs": 1.0, "sigma_delta_t": 2.0}
    path.write_text(json.dumps(data))
    result = run_lensing_preflight(opts)
    assert result["ok"], result["errors"]


def test_lensing_preflight_warns_on_placeholder_time_marks(tmp_path):
    from darksirens.lensing.preflight import run_lensing_preflight
    opts = _write_preflight_mock(tmp_path)
    opts.partition_mode = "marginalize_exact"
    opts.pair_marks = "time"
    import json
    path = __import__('pathlib').Path(opts.candidate_pairs_path)
    data = json.loads(path.read_text())
    for pair in data["candidate_pairs"]:
        pair["marks"] = {"delta_t_obs": 1.0, "sigma_delta_t": 1.0}
    path.write_text(json.dumps(data))
    result = run_lensing_preflight(opts)
    assert result["ok"], result["errors"]
    assert result["summary"]["candidate_time_marks_placeholder"] is True
    assert any("candidate time marks look like placeholders" in w for w in result["warnings"])


def test_pair_tag_selection_model_probabilities_and_snr_monotonic():
    from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model

    model = make_pair_tag_selection_model("snr_time")
    low = model.probability(
        snr_image0=np.array([8.0]), snr_image1=np.array([8.0]), delta_t_obs=np.array([1000.0])
    )[0]
    high = model.probability(
        snr_image0=np.array([14.0]), snr_image1=np.array([14.0]), delta_t_obs=np.array([1000.0])
    )[0]
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low



def test_pair_tag_selection_model_required_fields_for_snr_ablations():
    from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model

    assert make_pair_tag_selection_model("snr_only").required_fields == ("snr_image0", "snr_image1")
    assert make_pair_tag_selection_model("snr_sky").required_fields == (
        "snr_image0",
        "snr_image1",
        "log_sky_overlap",
    )


def test_snr_sky_pair_tag_probability_omits_time_dependence():
    from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model

    model = make_pair_tag_selection_model("snr_sky")
    base = model.probability(
        snr_image0=np.array([10.0]),
        snr_image1=np.array([9.0]),
        log_sky_overlap=np.array([-1.0]),
        delta_t_obs=np.array([1.0]),
    )
    later = model.probability(
        snr_image0=np.array([10.0]),
        snr_image1=np.array([9.0]),
        log_sky_overlap=np.array([-1.0]),
        delta_t_obs=np.array([1.0e9]),
    )
    np.testing.assert_allclose(base, later)

def test_pair_tag_perturb_logit_changes_probability():
    from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model

    base = make_pair_tag_selection_model("constant", constant=0.25)
    pert = make_pair_tag_selection_model("constant", constant=0.25, perturb_logit=1.0)
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, base.probability(), pert.probability())


def test_pair_tag_probability_is_per_source_for_every_kind():
    """Every kind returns an array shaped like the supplied fields.

    The constant/none kinds used to return a 0-d array while the score-based
    kinds returned (n,), so the mock generator's
    ``tagged_pair[both] = rng.uniform(...) < p_tag[both]`` crashed with
    ``IndexError: invalid index to scalar variable`` under
    ``--pair-tag-model constant``.
    """
    from darksirens.lensing.pair_tag_selection import (
        make_pair_tag_selection_model, PAIR_TAG_SELECTION_MODEL_KINDS,
    )

    n = 5
    fields = dict(
        snr_image0=np.linspace(9.0, 20.0, n),
        snr_image1=np.linspace(8.5, 12.0, n),
        delta_t_obs=np.linspace(1.0e3, 3.0e5, n),
        log_sky_overlap=np.full(n, -0.7),
    )
    both = np.array([True, False, True, False, True])
    kinds = [k for k in PAIR_TAG_SELECTION_MODEL_KINDS if k != "file"] + ["none"]
    for kind in kinds:
        model = make_pair_tag_selection_model(kind, constant=0.6)
        p = model.probability(**fields)
        assert p.shape == (n,), f"{kind}: got shape {p.shape}"
        assert np.all((p >= 0.0) & (p <= 1.0))
        # The generator indexes with a boolean mask; must not raise.
        assert p[both].shape == (int(both.sum()),)
        lp = model.log_probability(**fields)
        assert lp.shape == (n,)
        np.testing.assert_allclose(np.exp(lp), p, rtol=1e-12)
    # No fields supplied: nothing to broadcast to, stays 0-d (the inference
    # side calls it this way and broadcasts the log-weight itself).
    assert np.shape(make_pair_tag_selection_model("constant", constant=0.6).probability()) == ()


def test_pair_tag_log_probability_handles_zero_probability():
    from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model

    model = make_pair_tag_selection_model("constant", constant=0.0)
    lp = model.log_probability(snr_image0=np.zeros(3), snr_image1=np.zeros(3))
    assert lp.shape == (3,)
    assert np.all(np.isneginf(lp))


def test_lensed_injection_loading_reads_pair_tag_fields():
    camp = _synth_lensed_injection_campaign(n_sources=8, seed=24)
    n = camp["n_draw_sources"]
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/inj.h5"
        save_lensed_injections(
            path=path,
            **{k: v for k, v in camp.items() if k not in ("n_both_detected",)},
            snr_image0=np.linspace(8, 12, n),
            snr_image1=np.linspace(7, 11, n),
            delta_t_obs=np.linspace(10, 20, n),
            log_sky_overlap=np.linspace(-2, 0, n),
            p_tag_true=np.linspace(0.2, 0.8, n),
            tagged_pair=np.arange(n) % 2 == 0,
        )
        inj = load_lensed_injections(path)
    assert hasattr(inj, "snr_image0")
    assert inj.snr_image0.shape == inj.m1_src.shape
    assert np.all(np.isfinite(np.asarray(inj.delta_t_obs)))


def test_preflight_catches_missing_snr_sky_fields_without_requiring_time(tmp_path):
    from argparse import Namespace
    from darksirens.lensing.preflight import run_lensing_preflight

    camp = _synth_lensed_injection_campaign(n_sources=4, seed=125)
    inj_path = tmp_path / "inj_snr_sky.h5"
    save_lensed_injections(
        path=str(inj_path),
        **{k: v for k, v in camp.items() if k not in ("n_both_detected",)},
        snr_image0=np.linspace(8, 12, camp["n_draw_sources"]),
        snr_image1=np.linspace(7, 11, camp["n_draw_sources"]),
    )
    opts = Namespace(
        cluster_mode="j2", partition_mode="marginalize_exact", pair_marks="none",
        edge_mark_prior_keys="", edge_mark_likelihood_keys="", gw_path=None,
        gwselection_path=str(inj_path), lensed_injections_path=str(inj_path),
        candidate_pairs_path=None, observed_catalog_path=None, pair_tag_model="snr_sky",
        pair_tag_constant=1.0, pair_tag_perturb_logit=0.0,
    )
    report = run_lensing_preflight(opts)
    assert any("log_sky_overlap" in e for e in report["errors"])
    assert not any("delta_t_obs" in e for e in report["errors"])


def test_preflight_catches_missing_snr_time_sky_fields(tmp_path):
    from argparse import Namespace
    from darksirens.lensing.preflight import run_lensing_preflight

    camp = _synth_lensed_injection_campaign(n_sources=4, seed=25)
    inj_path = tmp_path / "inj.h5"
    save_lensed_injections(
        path=str(inj_path),
        **{k: v for k, v in camp.items() if k not in ("n_both_detected",)},
    )
    opts = Namespace(
        cluster_mode="j2", partition_mode="marginalize_exact", pair_marks="none",
        edge_mark_prior_keys="", edge_mark_likelihood_keys="", gw_path=None,
        gwselection_path=str(inj_path), lensed_injections_path=str(inj_path),
        candidate_pairs_path=None, observed_catalog_path=None, pair_tag_model="snr_time_sky",
        pair_tag_constant=1.0, pair_tag_perturb_logit=0.0,
    )
    report = run_lensing_preflight(opts)
    assert any("snr_image0" in e for e in report["errors"])
    assert any("log_sky_overlap" in e for e in report["errors"])


def test_selection_correction_changes_with_pair_tag_model():
    camp = _synth_lensed_injection_campaign(n_sources=60, seed=26)
    inj1 = make_lensed_injection_set(**{k: v for k, v in camp.items() if k not in ("n_both_detected",)})
    inj2 = make_lensed_injection_set(**{k: v for k, v in camp.items() if k not in ("n_both_detected",)}, p_tag_per_source=np.full(camp["n_draw_sources"], 0.5))
    args = (_cosmo(), _survey(), jnp.zeros(1), _toy_catalog(), make_sis_lens_params(), _toy_log_p_pop, _toy_volume_prior)
    log_mu1, _, _ = compute_cluster_selection_term(inj1, *args, log_p_tag_per_source=inj1.log_p_tag_per_source)
    log_mu2, _, _ = compute_cluster_selection_term(inj2, *args, log_p_tag_per_source=inj2.log_p_tag_per_source)
    assert float(log_mu2) < float(log_mu1)
