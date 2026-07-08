"""
Tests for the LSS-conditioned lognormal completion correction.

Covers the deterministic/mean-Q consumption path in ``completion.py`` and
``prior.py``, plus the offline builder I/O.  The likelihood stays deterministic
(no field sampling); these tests only exercise fixed Q arrays.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.redshift.completion import (
    completion_curves, build_pixel_kde_cache, catalog_completion, completion_clip_diagnostics,
)
from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_with_state,
    eval_redshift_prior_members_with_state,
    DarkSirenPriorState,
    DarkSirenEnsemblePriorState,
)
from darksirens.redshift.lognormal_completion import (
    gaussian_correlation_spectrum,
    poisson_lognormal_map,
    laplace_lognormal_members,
    save_lss_completion_hdf5,
    load_lss_completion_hdf5,
)

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
# b_miss = 0 -> the legacy local-overdensity factor is identically 1.
SURVEY_B0 = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)


def _tiny_catalog(unique_pixels=None, **extra):
    """A 2-row catalog using the on-the-fly KDE fallback (dN_obs_kde=None)."""
    zg = np.full((2, 3), 100.0)
    zg[0, :2] = [0.10, 0.15]
    zg[1, :1] = [0.20]
    ng = np.array([2, 1], dtype=np.int32)
    base = dict(
        apix=1.0,
        zgals=jnp.asarray(zg),
        dzgals=jnp.asarray(np.full((2, 3), 0.01)),
        wgals=jnp.zeros((2, 3)),
        ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=None if unique_pixels is None else jnp.asarray(unique_pixels, jnp.int32),
    )
    base.update(extra)
    return EMCatalog(**base)


def test_no_q_is_legacy_and_members_none():
    """With no Q table the ensemble fields are absent (backward-compatible)."""
    curves = completion_curves(COSMO, SURVEY_B0, _tiny_catalog())
    assert curves.dN_miss.shape == (2, NG)
    assert curves.dN_miss_members is None
    assert curves.N_miss_members is None


def test_unity_q_matches_homogeneous_branch():
    """logq = 0 everywhere + legacy b_miss = 0 -> identical missing density."""
    legacy = completion_curves(COSMO, SURVEY_B0, _tiny_catalog())
    q1 = completion_curves(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG)))
    )
    assert float(jnp.max(jnp.abs(q1.dN_miss - legacy.dN_miss))) < 1e-12
    assert float(jnp.max(jnp.abs(q1.N_miss - legacy.N_miss))) < 1e-12


def test_deterministic_q_scales_missing_density():
    """Q = 2 everywhere (legacy disabled) doubles dN_miss and N_miss."""
    q1 = completion_curves(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG)))
    )
    q2 = completion_curves(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.full((2, NG), np.log(2.0)))
    )
    assert float(jnp.max(jnp.abs(q2.dN_miss - 2.0 * q1.dN_miss))) < 1e-10
    assert float(jnp.max(jnp.abs(q2.N_miss - 2.0 * q1.N_miss))) < 1e-10


def test_compact_vs_global_indexing():
    """Compact Q indexes by row; global Q indexes by unique_pixels[row]."""
    q1 = completion_curves(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG)))
    )
    # compact: row 0 -> Q=3, row 1 -> Q=5
    compact = jnp.stack([jnp.full(NG, np.log(3.0)), jnp.full(NG, np.log(5.0))])
    cc = completion_curves(
        COSMO, SURVEY_B0,
        _tiny_catalog(lss_completion_logq=compact, lss_completion_indexing=1),
    )
    # dN_miss is O(1e8) here, so compare at relative float64 precision: the
    # x*exp(log q) vs q*x orderings legitimately differ by ~1 ULP.
    np.testing.assert_allclose(np.asarray(cc.dN_miss[0]), 3.0 * np.asarray(q1.dN_miss[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(cc.dN_miss[1]), 5.0 * np.asarray(q1.dN_miss[1]), rtol=1e-12)

    # global: rows map to pixels 5 and 2; table is global-pixel indexed
    gt = np.zeros((6, NG))
    gt[5] = np.log(3.0)
    gt[2] = np.log(5.0)
    gc = completion_curves(
        COSMO, SURVEY_B0,
        _tiny_catalog(
            unique_pixels=[5, 2],
            lss_completion_logq=jnp.asarray(gt),
            lss_completion_indexing=2,
        ),
    )
    np.testing.assert_allclose(np.asarray(gc.dN_miss[0]), 3.0 * np.asarray(q1.dN_miss[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gc.dN_miss[1]), 5.0 * np.asarray(q1.dN_miss[1]), rtol=1e-12)


def test_member_shapes_and_mean():
    """logq_members (M,N_rows,N_grid) -> member densities + posterior-mean compat."""
    q1 = completion_curves(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG)))
    )
    lm = jnp.full((3, 2, NG), np.log(2.0))
    m = completion_curves(COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq_members=lm))
    assert m.dN_miss_members.shape == (3, 2, NG)
    assert m.N_miss_members.shape == (3, 2)
    # only members supplied -> deterministic dN_miss is the posterior-mean (= 2x here)
    assert float(jnp.max(jnp.abs(m.dN_miss - 2.0 * q1.dN_miss))) < 1e-10


def test_grid_mismatch_raises():
    """A Q table on a different N_grid must fail loudly (no silent interp)."""
    with pytest.raises(ValueError, match="N_grid"):
        completion_curves(
            COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG + 1)))
        )


def _prior_catalog(**extra):
    """Two rows with real galaxies (wgals>0) and a built KDE cache, suitable
    for the assembled dark-siren prior (which normalises per pixel)."""
    rows = [np.array([0.10, 0.12, 0.15]), np.array([0.30, 0.34])]
    n_rows, nmax = 2, 3
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    for i, r in enumerate(rows):
        zg[i, : len(r)] = r
        dz[i, : len(r)] = 0.003
        w[i, : len(r)] = 1.0
        ng[i] = len(r)
    zg, dz, w, ng = (jnp.asarray(a) for a in (zg, dz, w, ng))
    kde, idx = build_pixel_kde_cache(np.arange(n_rows, dtype=np.int32), zg, n_rows, ngals=ng)
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=kde, pixel_to_cache_idx=idx,
        unique_pixels=None, **extra,
    )


def test_ensemble_member_priors_normalize():
    """Each member prior p_m(z|pix) integrates to 1 over z (per pixel)."""
    offs = np.array([-0.5, -0.2, 0.2, 0.5])
    lm = jnp.asarray(np.stack([np.full((2, NG), o) for o in offs]))  # (4, 2, NG)
    cat = _prior_catalog(lss_completion_logq_members=lm)
    state = prepare_redshift_prior_state("dark_sirens", COSMO, SURVEY_B0, cat)
    assert isinstance(state, DarkSirenEnsemblePriorState)
    for row in (0, 1):
        pix = jnp.full(NG, row, jnp.int32)
        lpm = eval_redshift_prior_members_with_state(
            "dark_sirens", state, zgrid, pix, COSMO, SURVEY_B0, cat
        )
        assert lpm.shape == (4, NG)
        for m in range(4):
            integ = np.trapezoid(np.exp(np.asarray(lpm[m])), np.asarray(zgrid))
            assert abs(integ - 1.0) < 5e-3, f"member {m} row {row}: ∫={integ}"


def test_ensemble_scalar_compat_is_finite_and_normalized():
    """With members supplied, the scalar prior path still returns a finite,
    normalised p(z|pix) (the GW likelihood is unaffected by the ensemble)."""
    lm = jnp.full((3, 2, NG), np.log(1.5))
    cat = _prior_catalog(lss_completion_logq_members=lm)
    state = prepare_redshift_prior_state("dark_sirens", COSMO, SURVEY_B0, cat)
    pix = jnp.full(NG, 0, jnp.int32)
    lp = eval_redshift_prior_with_state("dark_sirens", state, zgrid, pix, COSMO, SURVEY_B0, cat)
    assert lp.shape == (NG,)
    assert np.all(np.isfinite(np.asarray(lp)))
    integ = np.trapezoid(np.exp(np.asarray(lp)), np.asarray(zgrid))
    assert abs(integ - 1.0) < 5e-3
    # non-ensemble state -> members evaluator still returns a leading axis of 1
    plain = prepare_redshift_prior_state("dark_sirens", COSMO, SURVEY_B0, _prior_catalog())
    assert isinstance(plain, DarkSirenPriorState)
    one = eval_redshift_prior_members_with_state(
        "dark_sirens", plain, zgrid, pix, COSMO, SURVEY_B0, _prior_catalog()
    )
    assert one.shape == (1, NG)


# ------------------------------------------------------------
# Offline builder (NumPy/SciPy; grid-agnostic, tested on a small grid)
# ------------------------------------------------------------

def test_spectrum_marginal_variance():
    pk = gaussian_correlation_spectrum(64, ell_grid=5.0, sigma=0.3)
    assert pk.shape == (64,)
    assert np.all(pk > 0.0)
    assert abs(float(np.mean(pk)) - 0.3 ** 2) < 1e-6  # mean(P) = c[0] = sigma^2


def test_map_recovers_unity_for_matched_counts():
    """When observed counts equal the homogeneous rate, Q -> 1 (no clustering)."""
    n = 64
    pk = gaussian_correlation_spectrum(n, ell_grid=5.0, sigma=0.2)
    zz = np.linspace(0.0, 1.0, n)
    dN_exp = 200.0 * np.exp(-0.5 * ((zz - 0.5) / 0.15) ** 2) + 1.0
    C = np.ones(n)
    N_obs = (C * dN_exp)[None, :]  # one row, counts == expected rate
    out = poisson_lognormal_map(N_obs, C, dN_exp, pk, bias=1.0, prior_strength=1.0, maxiter=300)
    assert out["logq_map"].shape == (1, n)
    assert np.all(np.isfinite(out["q_map"]))
    hi = dN_exp > 20.0  # high-count region: data dominates -> Q ~ 1
    assert float(np.max(np.abs(out["q_map"][0][hi] - 1.0))) < 0.05


def test_laplace_members_shapes_and_determinism():
    n = 64
    pk = gaussian_correlation_spectrum(n, ell_grid=5.0, sigma=0.2)
    dN_exp = 50.0 * np.ones(n)
    C = np.ones(n)
    N_obs = (C * dN_exp)[None, :]
    mp = poisson_lognormal_map(N_obs, C, dN_exp, pk, maxiter=150)
    m1 = laplace_lognormal_members(mp["s_map"], mp["lambda_map"], pk, n_members=4, seed=7)
    m2 = laplace_lognormal_members(mp["s_map"], mp["lambda_map"], pk, n_members=4, seed=7)
    assert m1["logq_members"].shape == (4, 1, n)
    assert m1["logq_mean"].shape == (1, n)
    assert np.all(np.isfinite(m1["q_members"]))
    assert np.allclose(m1["logq_members"], m2["logq_members"])  # deterministic given seed


def test_completion_hdf5_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    lq = 0.1 * rng.standard_normal((3, 5))
    lm = 0.1 * rng.standard_normal((4, 3, 5))
    z = np.linspace(0.0, 1.0, 5)
    path = str(tmp_path / "lss_completion.h5")
    save_lss_completion_hdf5(
        path, logq_map=lq, logq_members=lm, zgrid=z, indexing="compact",
        metadata={"sigma_s2": 0.04, "n_members": 4},
    )
    d = load_lss_completion_hdf5(path)
    assert np.allclose(d["logq_map"], lq)
    assert np.allclose(d["logq_members"], lm)
    assert np.allclose(d["zgrid"], z)
    assert d["indexing"] == "compact"
    assert d["model"] == "poisson_lognormal"
    assert d["completion_kind"] == "laplace_members"
    assert d["diagnostics"]["n_members"] == 4


# ------------------------------------------------------------
# Tier 1+2 hardening: Q-aware diagnostics, bounds, builder
# ------------------------------------------------------------

def test_catalog_completion_respects_Q():
    """The scalar diagnostic must use Q (not the legacy factor): Q=2 doubles the
    missing density, so the completeness fraction f changes."""
    cat0 = _tiny_catalog()  # no Q -> legacy (b_miss=0 -> factor 1)
    catq = _tiny_catalog(lss_completion_logq=jnp.full((2, NG), np.log(2.0)))
    z = jnp.asarray(0.12)
    pix = jnp.asarray(0, jnp.int32)
    f0, _, _ = catalog_completion(z, pix, COSMO, SURVEY_B0, cat0)
    fq, _, _ = catalog_completion(z, pix, COSMO, SURVEY_B0, catq)
    assert not np.isclose(float(f0), float(fq))


def test_clip_diagnostics_reports_lss_source():
    q = completion_clip_diagnostics(
        COSMO, SURVEY_B0, _tiny_catalog(lss_completion_logq=jnp.zeros((2, NG)))
    )
    assert q["lss_source"] == "Q_LSS"
    base = completion_clip_diagnostics(COSMO, SURVEY_B0, _tiny_catalog())
    assert base["lss_source"] == "legacy_delta_g"


def test_global_Q_too_few_rows_raises():
    gt = np.zeros((3, NG))  # 3 rows, but pixel ids reach 5
    cat = _tiny_catalog(
        unique_pixels=[5, 2], lss_completion_logq=jnp.asarray(gt), lss_completion_indexing=2
    )
    with pytest.raises(ValueError, match="reaches|does not cover"):
        completion_curves(COSMO, SURVEY_B0, cat)
