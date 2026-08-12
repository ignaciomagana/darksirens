"""Parametric magnitude selection: the H0 firewall, limits, and fit recovery.

The load-bearing property is EXACT H0-invariance of C_sel at fixed h-scaled
parameters: dL is exactly proportional to 1/H0 (table interp times
H0Planck/H0), so M0hat + 5 log10 h cancels the -5 log10 h inside DM(z)
identically.  If this ever degrades beyond float roundoff, the magnitude
channel has become an H0 prior (the M*-H0 backdoor) and the dark-siren
H0 posterior is contaminated.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.redshift.selection import (  # noqa: E402
    H0_REF,
    c_sel_gaussian,
    c_sel_schechter,
    fit_selection_from_mags,
    reference_absolute_mags,
)
from darksirens.utils.cosmology import distance_modulus  # noqa: E402

Z = jnp.linspace(1e-4, 0.5, 200)
THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)


def test_distance_modulus_exact_h_scaling():
    """DM(z; H0) == DM(z; 100) - 5 log10(H0/100), to float roundoff."""
    for h0 in (50.0, 67.74, 90.0, 140.0):
        dm = np.asarray(distance_modulus(Z, h0))
        dm_ref = np.asarray(distance_modulus(Z, H0_REF))
        np.testing.assert_allclose(
            dm, dm_ref - 5.0 * np.log10(h0 / H0_REF), rtol=0, atol=5e-13)


def test_c_sel_gaussian_exactly_h0_invariant():
    """The firewall: C_sel at fixed h-scaled theta is H0-independent."""
    ref = np.asarray(c_sel_gaussian(Z, H0=H0_REF, **THETA))
    for h0 in (50.0, 67.74, 90.0, 140.0):
        cur = np.asarray(c_sel_gaussian(Z, H0=h0, **THETA))
        np.testing.assert_allclose(cur, ref, rtol=0, atol=1e-12)


def test_c_sel_gaussian_limits_and_monotonicity():
    c = np.asarray(c_sel_gaussian(Z, H0=67.74, **THETA))
    assert np.all((c >= 0.0) & (c <= 1.0))
    assert c[0] > 1.0 - 1e-10          # nearby galaxies are never missed
    assert np.all(np.diff(c) <= 1e-12)  # deeper is never more complete


def test_c_sel_schechter_h0_invariant_and_bounded():
    kw = dict(m_lim=24.0, Mstar_hat=-20.5, alpha=-0.7, M_faint_offset=4.0)
    ref = np.asarray(c_sel_schechter(Z, H0=H0_REF, **kw))
    cur = np.asarray(c_sel_schechter(Z, H0=62.0, **kw))
    np.testing.assert_allclose(cur, ref, rtol=0, atol=1e-12)
    assert np.all((ref >= 0.0) & (ref <= 1.0))
    assert ref[0] > 1.0 - 1e-8
    assert ref[-1] < ref[0]


def _synthetic_truncated_sample(rng, n, h0_true, m0_true=-21.0, sig_true=1.0,
                                m_lim=22.5):
    # m_lim=22.5 keeps the truncation BITING at every tested h0 (a large
    # fraction of the LF is cut), so the naive-mean bias check below is a
    # real discriminator rather than a coin flip.
    """m = M + DM(z; h0_true) cut at m <= m_lim; z uniform-ish in volume."""
    z = rng.uniform(0.03, 0.45, size=4 * n) ** (1.0 / 1.5)
    z = z[(z > 0.02) & (z < 0.5)][:2 * n]
    M = rng.normal(m0_true, sig_true, size=z.size)
    dm = np.asarray(distance_modulus(jnp.asarray(z), h0_true))
    m = M + dm
    keep = m <= m_lim
    return m[keep][:n], z[keep][:n]


def test_fit_recovers_h_scaled_truth_independent_of_true_h0():
    """The fit constrains M0hat = M0 - 5 log10 h regardless of the true H0."""
    rng = np.random.default_rng(21)
    for h0_true in (60.0, 80.0):
        m, z = _synthetic_truncated_sample(rng, 4000, h0_true)
        fit = fit_selection_from_mags(m, z, 22.5)
        sd = np.sqrt(np.diag(fit.cov))
        m0hat_true = -21.0 - 5.0 * np.log10(h0_true / H0_REF)
        assert abs(fit.M0hat - m0hat_true) < 4.0 * sd[0], (
            f"H0_true={h0_true}: {fit.M0hat} vs {m0hat_true} (sd {sd[0]})")
        assert abs(fit.sigma_M - 1.0) < 4.0 * sd[1]
        # The truncation is doing real work: the naive (untruncated) mean of
        # the observed Mhat is biased bright of the truth.
        naive = float(np.mean(reference_absolute_mags(m, z)))
        assert naive < m0hat_true - 2.0 * sd[0]


def test_fit_and_suffstats_apply_the_z_floor():
    """Unclipped negative photo-z (deliberately written by the mock survey)
    must be DROPPED loudly, not crash the fit via NaN distance moduli or
    silently drag the MLE through DM -> -inf outliers."""
    from darksirens.redshift.selection import magnitude_suffstats

    rng = np.random.default_rng(41)
    m, z = _synthetic_truncated_sample(rng, 2000, 70.0)
    m2 = np.concatenate([m, [19.0, 18.5, 19.5]])
    z2 = np.concatenate([z, [-0.01, 1e-4, 0.0]])
    with pytest.warns(RuntimeWarning, match="dropped 3/"):
        fit = fit_selection_from_mags(m2, z2, 22.5)
    ref = fit_selection_from_mags(m, z, 22.5)
    assert abs(fit.M0hat - ref.M0hat) < 1e-9   # identical after the drop
    with pytest.warns(RuntimeWarning, match="dropped 3/"):
        magnitude_suffstats(m2, z2, 22.5)


def test_gaussian_fit_tolerance_scales_with_the_sample_size():
    """The NLL scales with N, so an ABSOLUTE fatol stops meaning anything at
    catalog size: at N ~ 1e6 the objective's own float64 resolution
    (eps * |NLL| ~ 3e-10) is coarser than the old 1e-10 tolerance.  The
    schechter twin already scales it; the DEFAULT family must too."""
    import scipy.optimize as sopt

    rng = np.random.default_rng(23)
    m, z = _synthetic_truncated_sample(rng, 600, 70.0)

    seen = {}
    real_minimize = sopt.minimize

    def spy(*args, **kwargs):
        seen.setdefault("options", dict(kwargs.get("options") or {}))
        return real_minimize(*args, **kwargs)

    sopt.minimize = spy
    try:
        fit = fit_selection_from_mags(m, z, 22.5)
    finally:
        sopt.minimize = real_minimize

    assert seen["options"]["fatol"] == pytest.approx(1e-9 * fit.n_gal)
    assert seen["options"]["fatol"] > np.finfo(float).eps * abs(
        fit.meta["nll"])


def test_fit_rejects_sample_fainter_than_the_declared_limit():
    rng = np.random.default_rng(22)
    m, z = _synthetic_truncated_sample(rng, 500, 70.0)
    with pytest.raises(ValueError, match="FAINTER than the declared m_lim"):
        fit_selection_from_mags(m, z, m_lim=float(m.max()) - 0.5)
