"""
Regression tests for darksirens_skymaps_to_samples (Architecture A).

These guard against the localization-cancellation bug in which ``p_pe`` was set
to the *sampling* density (``prob[pix] * N(dL; mu, sig)``) instead of the PE
*prior*. That choice makes the per-event importance estimator

    (1/n) Σ_j p_pop(θ_j) p_z(z_j|pix_j) / [J_j p_pe(θ_j)]

independent of the skymap localisation, because in importance sampling
E_Q[f/Q] = ∫ f is independent of the proposal Q. Every event then collapses to
the same catalog-volume integral and the GW signal vanishes.

The fix has two halves; there is one test for each:

  * SAMPLING half  — dL must be drawn from the Singer et al. (2016) ansatz
    dL² · N(dL; DISTMU, DISTSIGMA), not a plain Gaussian.
    -> test_distance_samples_follow_dL2_ansatz

  * WEIGHT half    — p_pe must equal the PE prior dL² · g_m1 · g_q · g_chi,
    with NO prob[pix] and NO N(dL) factor.
    -> test_ppe_is_prior_not_sampling_density   (deterministic; the real guard)

Two behavioural tests confirm that, given a correct p_pe, the GW localisation
actually drives the per-event estimator in both distance and sky:

  * test_distance_localization_survives
  * test_sky_localization_survives

The tests need only numpy + h5py + healpy (and, if present, ligo.skymap). They
do NOT stand up the full inference pipeline, by design: the structural test is a
tight, deterministic tripwire, and the behavioural tests use the same
reweighting arithmetic the likelihood performs, reduced to the GW-localised
coordinates.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
h5py = pytest.importorskip("h5py")

CONVERTER_MODULE = "darksirens.cli.skymaps_to_samples"

# Converter CLI settings shared by the fixture and the structural assertion.
NSIDE = 16
NSAMP = 40_000
SEED = 0
M1DET_MIN = 2.0
M1DET_MAX = 250.0  # covers pop m_max ~100 up to z ~ 1.5
Q_MIN = 0.05
CHI_ABS_MAX = 0.99

MU_A, MU_B = 600.0, 1200.0  # well-separated event distances
SIGMA = 80.0


# ---------------------------------------------------------------------------
# Synthesis + invocation helpers
# ---------------------------------------------------------------------------


def _ansatz_norm(mu: float, sigma: float) -> float:
    """DISTNORM for p(d|pix) = DISTNORM · d² · N(d; mu, sigma) on d > 0."""
    d = np.linspace(0.0, mu + 12.0 * sigma, 40_000)
    z = (d - mu) / sigma
    normal_shape = np.exp(-0.5 * z * z)
    z_norm = np.trapezoid(d * d * normal_shape, d)
    return 1.0 / z_norm


def _write_toy_3d_skymap(path, nside, hot_pixels, mu, sigma):
    """Write a minimal flattened 3D HEALPix skymap (RING) with all PROB on hot_pixels."""
    npix = hp.nside2npix(nside)
    prob = np.zeros(npix, dtype=np.float64)
    prob[hot_pixels] = 1.0
    prob /= prob.sum()
    distmu = np.full(npix, mu, dtype=np.float64)
    distsigma = np.full(npix, sigma, dtype=np.float64)
    distnorm = np.full(npix, _ansatz_norm(mu, sigma), dtype=np.float64)
    hp.write_map(
        str(path),
        [prob, distmu, distsigma, distnorm],
        column_names=["PROB", "DISTMU", "DISTSIGMA", "DISTNORM"],
        overwrite=True,
        dtype=[np.float64] * 4,
        nest=False,
    )


def _run_converter(skymap_dir, output):
    mod = importlib.import_module(CONVERTER_MODULE)
    argv = [
        "darksirens_skymaps_to_samples",
        "--skymap_dir",
        str(skymap_dir),
        "--output",
        str(output),
        "--nsamp",
        str(NSAMP),
        "--seed",
        str(SEED),
        "--m1det_min",
        str(M1DET_MIN),
        "--m1det_max",
        str(M1DET_MAX),
        "--q_min",
        str(Q_MIN),
        "--chi_abs_max",
        str(CHI_ABS_MAX),
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = saved


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Build two disjoint-support toy skymaps, run the converter, return per-event arrays."""
    smdir = tmp_path_factory.mktemp("skymaps")
    out = tmp_path_factory.mktemp("out") / "gwdata.h5"

    npix = hp.nside2npix(NSIDE)
    # Disjoint angular supports so sky membership is unambiguous.
    hotA = np.arange(0, npix // 8)
    hotB = np.arange(npix // 2, npix // 2 + npix // 8)
    assert np.intersect1d(hotA, hotB).size == 0

    # Names chosen so sorted glob order is [eventA, eventB] == event index [0, 1].
    _write_toy_3d_skymap(smdir / "eventA.fits", NSIDE, hotA, MU_A, SIGMA)
    _write_toy_3d_skymap(smdir / "eventB.fits", NSIDE, hotB, MU_B, SIGMA)

    _run_converter(smdir, out)

    with h5py.File(out, "r") as f:
        nobs = int(f.attrs["nobs"])
        nsamp = int(f.attrs["nsamp"])
        data = {
            k: np.asarray(f[k])
            for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe")
        }

    assert nobs == 2 and nsamp == NSAMP

    def event(i):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        return {k: v[sl] for k, v in data.items()}

    return {
        "A": event(0),
        "B": event(1),
        "nside": NSIDE,
        "hotA": hotA,
        "hotB": hotB,
        "mu": {"A": MU_A, "B": MU_B},
        "sigma": SIGMA,
    }


def _pix_of(event, nside):
    theta = 0.5 * np.pi - event["dec"]
    return hp.ang2pix(nside, theta, event["ra"], nest=False)


# ---------------------------------------------------------------------------
# (1) WEIGHT half — deterministic structural tripwire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ev", ["A", "B"])
def test_ppe_is_prior_not_sampling_density(converted, ev):
    """p_pe must be exactly the PE prior dL² · g_m1 · g_q · g_chi.

    If anyone reintroduces a prob[pix] or N(dL; mu, sig) factor into p_pe, this
    fails immediately — that is precisely the localization-cancellation bug.
    """
    e = converted[ev]
    log_m1_width = np.log(M1DET_MAX / M1DET_MIN)
    q_width = 1.0 - Q_MIN
    chi_width = 2.0 * CHI_ABS_MAX

    expected = (
        e["dL"] ** 2
        * (1.0 / (e["m1det"] * log_m1_width))
        * (1.0 / q_width)
        * (1.0 / chi_width)
    )
    assert np.all(np.isfinite(e["p_pe"]))
    assert np.all(e["p_pe"] > 0.0)
    np.testing.assert_allclose(e["p_pe"], expected, rtol=1e-6, atol=0.0)


def test_ppe_independent_of_sky_pixel(converted):
    """p_pe / (dL² g_m1 g_q g_chi) must be a constant across disjoint sky patches."""
    log_m1_width = np.log(M1DET_MAX / M1DET_MIN)
    q_width = 1.0 - Q_MIN
    chi_width = 2.0 * CHI_ABS_MAX
    ratios = []
    for ev in ("A", "B"):
        e = converted[ev]
        base = (
            e["dL"] ** 2
            * (1.0 / (e["m1det"] * log_m1_width))
            * (1.0 / q_width)
            * (1.0 / chi_width)
        )
        ratios.append(e["p_pe"] / base)
    r = np.concatenate(ratios)
    assert np.ptp(r) < 1e-9
    np.testing.assert_allclose(r.mean(), 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# (2) SAMPLING half — dL must follow dL² · N, not a plain Gaussian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ev", ["A", "B"])
def test_distance_samples_follow_dL2_ansatz(converted, ev):
    """Sampled dL moments must match dL²·N(mu, sigma), not N(mu, sigma)."""
    e = converted[ev]
    mu, sigma = converted["mu"][ev], converted["sigma"]

    d = np.linspace(0.0, mu + 12.0 * sigma, 40_000)
    w = d * d * np.exp(-0.5 * ((d - mu) / sigma) ** 2)
    z_norm = np.trapezoid(w, d)
    mean_true = np.trapezoid(d * w, d) / z_norm
    std_true = np.sqrt(np.trapezoid(d * d * w, d) / z_norm - mean_true**2)

    mean_samp = e["dL"].mean()
    std_samp = e["dL"].std()

    # 5% tolerance accommodates grid + Monte-Carlo error at NSAMP.
    np.testing.assert_allclose(mean_samp, mean_true, rtol=0.05)
    np.testing.assert_allclose(std_samp, std_true, rtol=0.05)
    # Decisive separation from the plain-Gaussian (mean==mu) failure mode.
    assert mean_samp > mu + 0.10 * std_true


# ---------------------------------------------------------------------------
# (3) Behavioural — localisation drives the per-event estimator
# ---------------------------------------------------------------------------


def test_distance_localization_survives(converted):
    """A catalog distance feature at MU_A must favour event A over event B."""
    feature = lambda d_l: np.exp(-0.5 * ((d_l - MU_A) / 50.0) ** 2)

    def estimate(e):
        return np.mean(feature(e["dL"]) / e["p_pe"])

    e_a = estimate(converted["A"])
    e_b = estimate(converted["B"])
    assert e_a > 0.0 and e_b > 0.0
    assert e_a / e_b > 20.0


def test_sky_localization_survives(converted):
    """A catalog confined to event A's sky pixels must give event B exactly zero."""
    nside = converted["nside"]
    w_cat = np.zeros(hp.nside2npix(nside))
    w_cat[converted["hotA"]] = 1.0
    feature = lambda d_l: np.exp(-0.5 * ((d_l - MU_A) / 50.0) ** 2)

    def estimate(e):
        pix = _pix_of(e, nside)
        return np.mean(w_cat[pix] * feature(e["dL"]) / e["p_pe"])

    e_a = estimate(converted["A"])
    e_b = estimate(converted["B"])
    assert e_a > 0.0
    assert e_b == 0.0
