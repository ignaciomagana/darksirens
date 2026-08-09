"""K(z) template of the parametric magnitude selection.

Real surveys select on OBSERVED-frame magnitudes m = M0 + DM(z) + K(z) +
scatter; the selection model carries K(z) as fixed polynomial coefficients
(no constant term -- c0 is exactly degenerate with M0hat).  Load-bearing
properties pinned here: (1) the None path is bit-identical to the pre-K
behaviour; (2) the H0 firewall is untouched by K (K has no h); (3) fitting
K-corrected truth WITH the template recovers theta while ignoring K biases
M0hat by the mean K -- the failure mode real-catalog ingestion must alarm on;
(4) the fit JSON round-trips the template and pre-K files load as K = 0.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.redshift.selection import (  # noqa: E402
    H0_REF,
    SelectionFit,
    c_sel_gaussian,
    fit_selection_from_mags,
    k_of_z,
    load_selection_fit_json,
    reference_absolute_mags,
)
from darksirens.utils.cosmology import distance_modulus  # noqa: E402

Z = jnp.linspace(1e-4, 0.5, 200)
THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)
KC = (0.39, 0.11)               # K(z) = 0.39 z + 0.11 z^2


def test_none_and_empty_coeffs_are_bit_identical_to_legacy():
    ref = np.asarray(c_sel_gaussian(Z, H0=67.74, **THETA))
    for coeffs in (None, ()):
        cur = np.asarray(c_sel_gaussian(Z, H0=67.74, k_corr_coeffs=coeffs,
                                        **THETA))
        assert np.array_equal(cur, ref)
    assert k_of_z(Z, None) is None
    assert k_of_z(Z, ()) is None


def test_k_of_z_polynomial_and_no_constant_term():
    z = np.array([0.0, 0.1, 0.3])
    k = np.asarray(k_of_z(z, KC, xp=np))
    np.testing.assert_allclose(k, 0.39 * z + 0.11 * z ** 2, atol=1e-15)
    assert k[0] == 0.0             # K(0) = 0: no c0, by construction


def test_c_sel_with_kcorr_stays_h0_invariant():
    """K depends only on z -- the h-scaling firewall must be untouched."""
    ref = np.asarray(c_sel_gaussian(Z, H0=H0_REF, k_corr_coeffs=KC, **THETA))
    for h0 in (50.0, 67.74, 90.0, 140.0):
        cur = np.asarray(c_sel_gaussian(Z, H0=h0, k_corr_coeffs=KC, **THETA))
        np.testing.assert_allclose(cur, ref, rtol=0, atol=1e-12)
    # And K biting: the curve is strictly less complete than the K=0 curve
    # wherever the limit already bites (K > 0 pushes galaxies past m_lim).
    base = np.asarray(c_sel_gaussian(Z, H0=67.74, **THETA))
    assert np.all(ref <= base + 1e-15)
    assert ref[-1] < base[-1]


def _kcorr_truncated_sample(rng, n, h0_true, m0_true=-21.0, sig_true=1.0,
                            m_lim=22.5, coeffs=KC):
    """Observed m = M + DM(z; h0_true) + K(z), hard cut at m <= m_lim."""
    z = rng.uniform(0.03, 0.45, size=4 * n) ** (1.0 / 1.5)
    z = z[(z > 0.02) & (z < 0.5)][:2 * n]
    M = rng.normal(m0_true, sig_true, size=z.size)
    dm = np.asarray(distance_modulus(jnp.asarray(z), h0_true))
    m = M + dm + np.asarray(k_of_z(z, coeffs, xp=np))
    keep = m <= m_lim
    return m[keep][:n], z[keep][:n]


def test_fit_with_template_recovers_truth_and_without_is_biased():
    rng = np.random.default_rng(11)
    m, z = _kcorr_truncated_sample(rng, 4000, h0_true=70.0)
    m0hat_true = -21.0 - 5.0 * np.log10(70.0 / H0_REF)

    fit = fit_selection_from_mags(m, z, 22.5, k_corr_coeffs=KC)
    sd = np.sqrt(np.diag(fit.cov))
    assert abs(fit.M0hat - m0hat_true) < 4.0 * sd[0]
    assert abs(fit.sigma_M - 1.0) < 4.0 * sd[1]
    assert fit.k_corr_coeffs == KC

    # Ignoring K absorbs the mean K-correction of the sample into M0hat: a
    # faint-ward (positive) shift many sds beyond the honest fit's error.
    fit0 = fit_selection_from_mags(m, z, 22.5)
    mean_k = float(np.mean(np.asarray(k_of_z(z, KC, xp=np))))
    assert fit0.M0hat - fit.M0hat > 0.5 * mean_k
    assert fit0.M0hat - m0hat_true > 4.0 * sd[0]


def test_reference_mags_subtract_the_template():
    z = np.array([0.05, 0.2, 0.4])
    m = np.full_like(z, 20.0)
    delta = reference_absolute_mags(m, z) - reference_absolute_mags(
        m, z, k_corr_coeffs=KC)
    np.testing.assert_allclose(delta, np.asarray(k_of_z(z, KC, xp=np)),
                               atol=1e-12)


def test_fit_json_roundtrip_and_pre_k_files_load_as_k_zero(tmp_path):
    fit = SelectionFit(family="gaussian", m_lim=21.0, M0hat=-20.9,
                       sigma_M=0.9, cov=np.eye(2) * 1e-4, n_gal=1000,
                       k_corr_coeffs=KC)
    payload = {"format_version": "darksirens-selection-fit-1.0",
               "strata": [fit.to_jsonable()]}
    p = tmp_path / "fit.json"
    p.write_text(json.dumps(payload))
    s = load_selection_fit_json(p)
    assert s["k_corr_coeffs"] == KC

    # Pre-K payloads carry no key at all -> empty tuple (K = 0).
    del payload["strata"][0]["k_corr_coeffs"]
    p.write_text(json.dumps(payload))
    assert load_selection_fit_json(p)["k_corr_coeffs"] == ()


def test_survey_params_carries_structural_kcorr():
    from darksirens.core.types import SurveyParams

    s = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=1.0,
                     alpha_miss=0.0)
    assert s.k_corr_coeffs is None
    assert s._replace(k_corr_coeffs=KC).k_corr_coeffs == KC


def test_qtable_kcorr_firewall_mismatch_is_fatal():
    """The inference CLI must refuse a Q table whose stamped K template
    differs from the --selection_fit template (count or value)."""
    from darksirens.cli import inference as cli

    fid = {"path": "qt.h5", "selection_m_lim": 21.0,
           "selection_M0hat": -20.9, "selection_sigma_M": 0.9,
           "selection_kcorr_c1": 0.39, "selection_kcorr_c2": 0.11}

    def _opts(kcorr):
        # The firewall reads ONE per-catalog fit record per Q-table fiducial.
        return SimpleNamespace(selection_fits=[{
            "catalog": 1, "path": "fit.json",
            "theta": {"m_lim": 21.0, "M0hat": -20.9, "sigma_M": 0.9},
            "k_corr_coeffs": list(kcorr), "strata_fit": None,
            "strata_struct": None, "stratum_map": None,
            "stratum_map_sha256": None, "prior": {}}])

    with pytest.raises(SystemExit):            # count mismatch vs the stamp
        cli._check_selection_qtable_theta([fid], _opts((0.39,)))
    cli._check_selection_qtable_theta([fid], _opts((0.39, 0.11)))   # match
    with pytest.raises(SystemExit):            # value mismatch
        cli._check_selection_qtable_theta([fid], _opts((0.39, 0.12)))
