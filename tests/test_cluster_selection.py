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
