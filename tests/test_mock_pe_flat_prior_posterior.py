"""The mock's measurement family: every width is data, nothing is clipped.

The generator records ``rho_obs = rho_opt(theta) + N(0, sigma_rho)`` and derives
every other measurement width from that one recorded number.  Two properties
follow, and they are what these tests pin:

* the stored fixed-width posterior IS the exact flat-prior posterior of the
  recorded measurement -- checkable from the file alone, because every stored
  width must be recomputable bitwise from the stored ``obs_rho``;
* nothing recorded and nothing sampled is clipped.  Clipping the data makes the
  measurement model censored (a theta-dependent normalisation
  ``P(obs = boundary|theta) = 1 - Phi(...)``, so the exact posterior stops being
  a simple normal); clipping a sample puts a point mass at the boundary, which
  is not a density at all and which ``p_pe`` cannot describe.  Physical ranges
  are imposed on the PE PRIOR instead, by exact inverse-CDF truncation.

A truth-scaled width ``N(obs; m, f m)`` violated the detected-set score identity
``E[C] = E[A]`` at 11.3 sigma on the gws-agn matched mock; this family measures
1.4 sigma.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mock_dark_sirens"))

import generate_mock_data as gmd  # noqa: E402

THRESH = 8.0


def _truth(nobs, dl=1500.0, z=0.3, m1=35.0, m2=30.0, chi=0.0, dec=0.2):
    return {
        "z": np.full(nobs, z),
        "dl": np.full(nobs, dl),
        "ra": np.full(nobs, 1.0),
        "dec": np.full(nobs, dec),
        "m1": np.full(nobs, m1),
        "m2": np.full(nobs, m2),
        "chi": np.full(nobs, chi),
    }


def _observe(rng, truth, meas):
    """One measurement per row of ``truth``, in detector-frame masses."""
    zz = truth["z"]
    return gmd._measure(rng, truth["m1"] * (1.0 + zz), truth["m2"] * (1.0 + zz),
                        truth["chi"], truth["dl"], truth["ra"], truth["dec"], meas)


def _event_record(truth, obs):
    """The ``truth`` group as the generator writes it: truths plus the record."""
    return {**truth, **obs}


# ---------------------------------------------------------------------------
# T2 -- every width is a function of the RECORDED data, and of nothing else
# ---------------------------------------------------------------------------
def test_every_width_is_recomputable_from_the_recorded_snr_bitwise():
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    obs = _observe(np.random.default_rng(1), _truth(500), meas)
    rho = obs["obs_rho"]
    k = meas.snr_threshold / rho

    assert np.array_equal(obs["obs_sig_lnmc"], meas.a_mc * k)
    assert np.array_equal(obs["obs_sig_lnq"], meas.a_q * k)
    assert np.array_equal(obs["obs_sig_chieff"], meas.a_chi * k)
    rho_sigma = (gmd.SNR_REF_SIGMA / meas.snr_ref) * rho
    assert np.array_equal(
        obs["obs_sigma_ang"],
        np.deg2rad(np.clip(meas.sky_a_deg / rho_sigma, *gmd.SKY_CLIP_DEG)))
    # The RA width is formed from the declination ALREADY RECORDED, which is the
    # same expression the posterior uses.  A cos(dec_true) width would make the
    # stored posterior the posterior of a different measurement.
    assert np.array_equal(
        obs["obs_sig_ra"],
        obs["obs_sigma_ang"] / np.maximum(np.cos(obs["obs_dec"]), gmd.COS_DEC_FLOOR))


@pytest.mark.parametrize("width", ["obs_sig_lnmc", "obs_sig_lnq", "obs_sig_chieff",
                                   "obs_sigma_ang", "obs_sig_ra"])
def test_no_width_is_derived_from_the_latent_truth(width):
    """The negative control: with the truths held FIXED and only the noise
    changing, every width still moves.  A latent-dependent width would be
    identical across the two draws."""
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    truth = _truth(400)
    a = _observe(np.random.default_rng(2), truth, meas)
    b = _observe(np.random.default_rng(3), truth, meas)
    assert not np.allclose(a[width], b[width])


def test_the_ra_width_uses_the_observed_declination_not_the_true_one():
    """Near the pole the two differ by O(1), which is exactly where the sky cap
    also bites; the stored posterior would then not be the posterior of the
    recorded measurement."""
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH,
                                 sky_uncertainty_deg=10.0)
    truth = _truth(2000, dec=1.4)          # 80 deg: cos dec ~ 0.17
    obs = _observe(np.random.default_rng(4), truth, meas)
    from_true = obs["obs_sigma_ang"] / np.maximum(np.cos(truth["dec"]), gmd.COS_DEC_FLOOR)
    assert not np.allclose(obs["obs_sig_ra"], from_true)


# ---------------------------------------------------------------------------
# T1 -- nothing recorded and nothing sampled is clipped
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stressed():
    """Widths wide enough that the physical boundaries are reached often."""
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH,
                                 a_chi=0.8, sky_uncertainty_deg=30.0)
    truth = _truth(4000, chi=0.0, dec=1.4)
    obs = _observe(np.random.default_rng(5), truth, meas)
    post, _ = gmd._posterior_samples(np.random.default_rng(6),
                                     _event_record(truth, obs), 64, meas)
    return meas, truth, obs, post


def test_recorded_values_are_allowed_outside_the_physical_range(stressed):
    _, _, obs, _ = stressed
    assert (np.exp(obs["obs_lnq"]) > 1.0).any(), "no recorded q above 1"
    assert (np.abs(obs["obs_chieff"]) > 1.0).any(), "no recorded |chi_eff| above 1"
    assert (np.abs(obs["obs_dec"]) > 0.5 * np.pi).any(), "no recorded |dec| above pi/2"


def test_no_boundary_atom_in_any_recorded_channel(stressed):
    _, _, obs, _ = stressed
    assert not (np.abs(obs["obs_chieff"]) == 1.0).any()
    assert not (np.abs(obs["obs_dec"]) == 0.5 * np.pi).any()
    assert not (obs["obs_lnq"] == 0.0).any()
    assert not (obs["obs_m1det"] == 2.0).any()
    assert not (obs["obs_m2det"] == 1.0).any()


def test_no_boundary_atom_in_any_pe_sample(stressed):
    _, _, _, post = stressed
    q = post["m2det"] / post["m1det"]
    assert (q <= 1.0).all(), "the PE prior truncation at q <= 1 leaked"
    # ... and it is a PRIOR truncation, not a clip: a clip would pile every
    # rejected draw onto q = 1 exactly.
    assert not (q == 1.0).any()
    assert not (np.abs(post["chieff"]) == 1.0).any()
    assert not (np.abs(post["dec"]) == 0.5 * np.pi).any()
    assert not (post["m1det"] == 2.0).any()
    assert not (post["m2det"] == 1.0).any()


def test_truncated_prior_draws_are_the_exact_truncated_normal():
    """``_trunc_norm`` is the only truncated sampler, so pin it against the
    analytic CDF rather than against a rejection sampler."""
    rng = np.random.default_rng(7)
    x = gmd._trunc_norm(rng, 0.3, 1.2, -1.0, 1.0, 200_000)
    assert x.min() > -1.0 and x.max() < 1.0
    ref = stats.truncnorm((-1.0 - 0.3) / 1.2, (1.0 - 0.3) / 1.2, loc=0.3, scale=1.2)
    assert stats.kstest(x, ref.cdf).pvalue > 1e-3


# ---------------------------------------------------------------------------
# T4 -- the bijection: dL is DERIVED, and the round trip is the identity
# ---------------------------------------------------------------------------
def test_derived_distance_and_the_mass_bijection_round_trip():
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    truth = _truth(60)
    obs = _observe(np.random.default_rng(8), truth, meas)
    post, _ = gmd._posterior_samples(np.random.default_rng(9),
                                     _event_record(truth, obs), 200, meas)

    m1, m2, dL = post["m1det"], post["m2det"], post["dL"]
    q = m2 / m1
    mc = gmd._mc_of_m1q(m1, q)
    rho = gmd._rho_opt_of_mc_dl(mc, dL, meas.snr_ref)
    # rho IS the distance coordinate: rho * dL = 1000 snr_ref (Mc/30)^(5/6).
    want = 1000.0 * meas.snr_ref * (mc / 30.0) ** (5.0 / 6.0)
    np.testing.assert_allclose(rho * dL, want, rtol=1e-14)
    # and (m1det, m2det, dL) -> (Mc, q, rho) -> (m1det, m2det, dL) is the identity
    m1_back = gmd._m1_of_mc_q(mc, q)
    np.testing.assert_allclose(m1_back, m1, rtol=1e-14)
    np.testing.assert_allclose(q * m1_back, m2, rtol=1e-14)
    np.testing.assert_allclose(gmd._dl_of_mc_rho(mc, rho, meas.snr_ref), dL, rtol=1e-14)


def test_the_stored_point_estimates_are_the_bijection_at_the_observation():
    """``obs_m1det``/``obs_m2det``/``obs_dL`` are diagnostics; they must be the
    bijection evaluated at the recorded measurement, and the PE must not read
    them (it reads ``obs_lnmc``/``obs_lnq``/``obs_rho``)."""
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    obs = _observe(np.random.default_rng(10), _truth(300), meas)
    q_o = np.exp(obs["obs_lnq"])
    m1_o = gmd._m1_of_mc_q(np.exp(obs["obs_lnmc"]), q_o)
    np.testing.assert_array_equal(obs["obs_m1det"], m1_o)
    np.testing.assert_array_equal(obs["obs_m2det"], q_o * m1_o)
    np.testing.assert_array_equal(
        obs["obs_dL"], gmd._dl_of_mc_rho(np.exp(obs["obs_lnmc"]), obs["obs_rho"],
                                         meas.snr_ref))

    # The PE never reads them: corrupting them changes nothing.
    truth = _truth(300)
    rec = _event_record(truth, obs)
    a, _ = gmd._posterior_samples(np.random.default_rng(11), rec, 32, meas)
    rec = dict(rec)
    for key in ("obs_m1det", "obs_m2det", "obs_dL"):
        rec[key] = np.full_like(rec[key], np.nan)
    b, _ = gmd._posterior_samples(np.random.default_rng(11), rec, 32, meas)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


# ---------------------------------------------------------------------------
# T5 -- p_pe is the PE prior in darksirens' canonical basis, by two routes
# ---------------------------------------------------------------------------
def test_p_pe_equals_the_closed_form_exactly_and_a_numerical_jacobian():
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    nobs, nsamp = 12, 200
    truth = _truth(nobs)
    obs = _observe(np.random.default_rng(12), truth, meas)
    post, _ = gmd._posterior_samples(np.random.default_rng(13),
                                     _event_record(truth, obs), nsamp, meas)

    m1, m2, dL = post["m1det"], post["m2det"], post["dL"]
    q = m2 / m1
    # Route 1: the closed form rho/(dL m1det q), evaluated on the stored columns,
    # normalised to mean 1 per event (darksirens renormalises per event).
    raw = gmd._p_pe(m1, q, dL, meas).reshape(nobs, nsamp)
    want = (raw / raw.mean(axis=1, keepdims=True)).ravel()
    np.testing.assert_array_equal(post["p_pe"], want)
    assert not np.allclose(post["p_pe"], 1.0), "p_pe is still the wrong all-ones column"

    # Route 2: a numerical Jacobian |d(ln Mc, ln q, rho)/d(m1det, q, dL)| by
    # central differences.  This is what makes the derivation un-regressable.
    def y_of_x(x):
        m1_, q_, dl_ = x[..., 0], x[..., 1], x[..., 2]
        mc_ = gmd._mc_of_m1q(m1_, q_)
        return np.stack([np.log(mc_), np.log(q_),
                         gmd._rho_opt_of_mc_dl(mc_, dl_, meas.snr_ref)], axis=-1)

    x0 = np.stack([m1[:64], q[:64], dL[:64]], axis=-1)
    jac = np.empty(x0.shape + (3,))
    for k in range(3):
        h = 1e-6 * np.abs(x0[:, k])
        step = np.zeros_like(x0)
        step[:, k] = h
        jac[..., k] = (y_of_x(x0 + step) - y_of_x(x0 - step)) / (2.0 * h)[:, None]
    numeric = np.abs(np.linalg.det(jac))
    closed = gmd._p_pe(x0[:, 0], x0[:, 1], x0[:, 2], meas)
    np.testing.assert_allclose(numeric, closed, rtol=1e-5)


def test_p_pe_declares_the_canonical_basis_in_the_file(tmp_path, monkeypatch):
    import h5py
    monkeypatch.setattr(sys, "argv", [
        "generate_mock_data.py", "--outdir", str(tmp_path), "--seed", "3",
        "--n-galaxies", "3000", "--nobs", "2", "--nsamp", "16",
        "--nselection", "3000", "--nside", "8"])
    gmd.write_mock_data(gmd.parse_args())
    with h5py.File(tmp_path / "mock_gw_events.h5") as f:
        assert "m1det" in f.attrs["p_pe_basis"] and "q" in f.attrs["p_pe_basis"]
        p = f["p_pe"][...].reshape(int(f.attrs["nobs"]), -1)
        np.testing.assert_allclose(p.mean(axis=1), 1.0, rtol=1e-12)
        assert not np.allclose(p, 1.0)


# ---------------------------------------------------------------------------
# T6 -- calibration: the measurement pulls are N(0,1) and the PITs are uniform
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibration():
    meas = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)
    rng = np.random.default_rng(20260802)
    n = 40_000
    truth = {
        "z": rng.uniform(0.05, 0.4, n),
        "ra": rng.uniform(0.0, 2.0 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1.0, 1.0, n)),
        "m1": rng.uniform(20.0, 50.0, n),
        "chi": rng.normal(0.0, 0.1, n),
    }
    truth["m2"] = truth["m1"] * rng.uniform(0.4, 1.0, n)
    truth["dl"] = 4300.0 * truth["z"] * (1.0 + 0.8 * truth["z"])
    obs = _observe(rng, truth, meas)
    det = obs["obs_rho"] >= meas.snr_threshold
    assert 0.05 < det.mean() < 0.95, det.mean()
    return meas, truth, obs, det


@pytest.mark.parametrize("channel", ["lnmc", "lnq", "chieff", "dec"])
def test_measurement_pulls_are_standard_normal_on_the_detected_set(calibration, channel):
    """Detection depends on ``rho_obs`` alone, and these channels are drawn
    independently of it given their widths, so their pulls stay N(0,1) after
    selection."""
    meas, truth, obs, det = calibration
    latent = {"lnmc": np.log(gmd._mc_of_m1q(truth["m1"] * (1.0 + truth["z"]),
                                            truth["m2"] / truth["m1"])),
              "lnq": np.log(truth["m2"] / truth["m1"]),
              "chieff": truth["chi"],
              "dec": truth["dec"]}[channel]
    width = {"lnmc": "obs_sig_lnmc", "lnq": "obs_sig_lnq",
             "chieff": "obs_sig_chieff", "dec": "obs_sigma_ang"}[channel]
    pull = ((obs[f"obs_{channel}"] - latent) / obs[width])[det]
    assert abs(pull.mean()) < 5.0 / np.sqrt(pull.size)
    assert abs(pull.std() - 1.0) < 0.03
    assert stats.kstest(pull, "norm").pvalue > 1e-3


def test_the_detected_snr_pull_is_normal_TRUNCATED_at_the_threshold(calibration):
    """The un-truncated pull is NOT uniform on the detected set -- selection is
    a cut on this very channel -- so the reference has to be the truncated
    normal.  That it passes is the statement that detection is a deterministic
    function of this one recorded number."""
    meas, _, obs, det = calibration
    rho_opt = obs["snr_true"][det]
    pull = ((obs["obs_rho"] - obs["snr_true"]) / meas.sigma_rho)[det]
    a = (meas.snr_threshold - rho_opt) / meas.sigma_rho
    from scipy.special import ndtr
    pit = (ndtr(pull) - ndtr(a)) / (1.0 - ndtr(a))
    assert stats.kstest(pit, "uniform").pvalue > 1e-3
    # and the untruncated reference fails, which is what makes this a test
    assert stats.kstest(pull, "norm").pvalue < 1e-6


@pytest.mark.parametrize("channel", ["lnmc", "chieff", "dec"])
def test_posterior_pit_of_the_truth_is_uniform(calibration, channel):
    """The stored samples must be the EXACT flat-prior posterior of the recorded
    measurement, so the truth's PIT under them is uniform.

    ``ln q`` is deliberately absent: its PRIOR truncation at ``q <= 1`` is
    active for most events, and the PIT of a truth under a prior-truncated
    posterior is uniform only when the truth itself follows that prior -- which
    an improper flat prior cannot realise.  The corresponding statement for that
    channel is that the samples ARE the exact truncated normal, which
    :func:`test_every_posterior_channel_is_its_exact_analytic_posterior` checks
    directly.  The remaining channels' truncations are inert at these widths.
    """
    meas, truth, obs, det = calibration
    keep = np.where(det)[0][:400]
    sub_truth = {k: v[keep] for k, v in truth.items()}
    sub_obs = {k: v[keep] for k, v in obs.items()}
    post, _ = gmd._posterior_samples(np.random.default_rng(77),
                                     _event_record(sub_truth, sub_obs), 500, meas)
    n = keep.size
    sample = {
        "lnmc": np.log(gmd._mc_of_m1q(post["m1det"], post["m2det"] / post["m1det"])),
        "lnq": np.log(post["m2det"] / post["m1det"]),
        "chieff": post["chieff"],
        "dec": post["dec"],
    }[channel].reshape(n, -1)
    latent = {
        "lnmc": np.log(gmd._mc_of_m1q(sub_truth["m1"] * (1.0 + sub_truth["z"]),
                                      sub_truth["m2"] / sub_truth["m1"])),
        "lnq": np.log(sub_truth["m2"] / sub_truth["m1"]),
        "chieff": sub_truth["chi"],
        "dec": sub_truth["dec"],
    }[channel]
    pit = (sample < latent[:, None]).mean(axis=1)
    assert stats.kstest(pit, "uniform").pvalue > 1e-3


def test_every_posterior_channel_is_its_exact_analytic_posterior(calibration):
    """Channel by channel, against the closed form -- including the truncations,
    which are PRIOR truncations and therefore exact truncated normals about the
    OBSERVED value with the STORED width, not clipped normals."""
    meas, truth, obs, det = calibration
    keep = np.where(det)[0][:40]
    sub_truth = {k: v[keep] for k, v in truth.items()}
    sub_obs = {k: v[keep] for k, v in obs.items()}
    nsamp = 4000
    post, _ = gmd._posterior_samples(np.random.default_rng(78),
                                     _event_record(sub_truth, sub_obs), nsamp, meas)
    n = keep.size
    q = (post["m2det"] / post["m1det"]).reshape(n, -1)
    channels = {
        "lnmc": (np.log(gmd._mc_of_m1q(post["m1det"], post["m2det"] / post["m1det"])
                        ).reshape(n, -1), sub_obs["obs_lnmc"], sub_obs["obs_sig_lnmc"],
                 -np.inf, np.inf),
        "lnq": (np.log(q), sub_obs["obs_lnq"], sub_obs["obs_sig_lnq"], -np.inf, 0.0),
        "chieff": (post["chieff"].reshape(n, -1), sub_obs["obs_chieff"],
                   sub_obs["obs_sig_chieff"], -1.0, 1.0),
        "dec": (post["dec"].reshape(n, -1), sub_obs["obs_dec"],
                sub_obs["obs_sigma_ang"], -0.5 * np.pi, 0.5 * np.pi),
    }
    for name, (sample, loc, scale, lo, hi) in channels.items():
        pvals = []
        for i in range(n):
            ref = stats.truncnorm((lo - loc[i]) / scale[i], (hi - loc[i]) / scale[i],
                                  loc=loc[i], scale=scale[i])
            pvals.append(stats.kstest(sample[i], ref.cdf).pvalue)
        # pooled: the per-event p-values are themselves uniform under the null
        assert stats.kstest(pvals, "uniform").pvalue > 1e-3, name
    # the rho channel is the one whose truncation at rho > 0 is inert at 8 sigma
    rho = gmd._rho_opt_of_mc_dl(
        gmd._mc_of_m1q(post["m1det"], post["m2det"] / post["m1det"]),
        post["dL"], meas.snr_ref).reshape(n, -1)
    pvals = [stats.kstest(rho[i], stats.norm(sub_obs["obs_rho"][i],
                                             meas.sigma_rho).cdf).pvalue
             for i in range(n)]
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3


def test_posterior_requires_the_recorded_measurement():
    meas = gmd.MeasurementConfig(snr_threshold=THRESH)
    with pytest.raises(ValueError, match="missing"):
        gmd._posterior_samples(np.random.default_rng(0), _truth(2), 8, meas)
