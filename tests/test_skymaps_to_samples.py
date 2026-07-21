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

A fourth block (SEAM) guards a second, previously-untested failure mode: the
converter advertised producing importance samples "for the existing
likelihood" but wrote a schema ``darksirens.gw.utils.load_gw_samples`` (the
loader that actually feeds the likelihood) rejected outright -- missing
``format_version``, ``m1src``/``m2src``, and the ``pe_cosmology_*`` /
``chi_eff_*`` attrs. These tests exercise the real seam, converter output ->
``load_gw_samples(...)``, instead of only inspecting the raw HDF5.

The tests need only numpy + h5py + healpy (and, if present, ligo.skymap). They
do NOT stand up the full inference pipeline, by design: the structural test is a
tight, deterministic tripwire, and the behavioural tests use the same
reweighting arithmetic the likelihood performs, reduced to the GW-localised
coordinates.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
h5py = pytest.importorskip("h5py")

# ``darksirens.gw.utils`` imports tqdm at module import time; stub it so the
# loader is importable without the optional progress-bar dependency (mirrors
# tests/test_data_loader.py and tests/test_gwcat_v2_compat.py).
_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

from darksirens.gw.utils import load_gw_samples  # noqa: E402
from darksirens.utils.cosmology import z_of_dL  # noqa: E402

CONVERTER_MODULE = "darksirens.cli.skymaps_to_samples"

# Converter CLI settings shared by the fixture and the structural assertion.
NSIDE = 16
NSAMP = 40_000
SEED = 0
M1DET_MIN = 2.0
M1DET_MAX = 250.0  # covers pop m_max ~100 up to z ~ 1.5
Q_MIN = 0.05
CHI_ABS_MAX = 0.99
PE_H0 = 70.0  # fiducial PE cosmology passed explicitly (not the Planck15 default)
PE_OM0 = 0.30

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
        "--pe_H0",
        str(PE_H0),
        "--pe_Om0",
        str(PE_OM0),
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

    # Direct-HDF5 read: still valid for the structural (raw p_pe/dL) tests
    # below, and for reading the schema attrs the loader also requires.
    with h5py.File(out, "r") as f:
        nobs = int(f.attrs["nobs"])
        nsamp = int(f.attrs["nsamp"])
        format_version = f.attrs["format_version"]
        pe_H0 = float(f.attrs["pe_cosmology_H0"])
        pe_Om0 = float(f.attrs["pe_cosmology_Om0"])
        chi_eff_in_p_pe = bool(f.attrs["chi_eff_in_p_pe"])
        chi_eff_amax = float(f.attrs["chi_eff_amax"])
        data = {
            k: np.asarray(f[k])
            for k in (
                "ra", "dec", "dL", "m1det", "m2det", "m1src", "m2src",
                "chieff", "p_pe",
            )
        }

    assert nobs == 2 and nsamp == NSAMP

    # The actual seam under test: this is the call the CLI pipeline makes
    # before the file ever reaches the likelihood. It must succeed.
    loaded = load_gw_samples(out)

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
        "nobs": nobs,
        "nsamp": nsamp,
        "raw": data,
        "loaded": loaded,
        "format_version": format_version,
        "pe_H0": pe_H0,
        "pe_Om0": pe_Om0,
        "chi_eff_in_p_pe": chi_eff_in_p_pe,
        "chi_eff_amax": chi_eff_amax,
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


# ---------------------------------------------------------------------------
# (4) SEAM — converter output must be accepted by load_gw_samples
# ---------------------------------------------------------------------------
#
# BUG-1: the converter advertised producing importance samples "for the
# existing likelihood", but load_gw_samples -- the loader that actually feeds
# it -- rejected the file outright: no recognised format_version, and no
# m1src/m2src/pe_cosmology_*/chi_eff_* members. These tests exercise the real
# seam (converter output -> load_gw_samples) rather than only the raw HDF5.


def test_format_version_is_loader_accepted(converted):
    """The emitted format_version must be one load_gw_samples recognises.

    gwcat-1.0 is the generic, non-lensing, chi_eff-basis PE contract -- the
    same one darksirens' own mock-data generator
    (scripts/mock_dark_sirens/generate_mock_data.py) uses for its GW event
    file. It is the semantically correct choice here: this converter is
    neither lensing-specific ("observed-lensing-pe-1.0") nor spin-basis
    versioned ("gwcat-pe-2.0" requires a spin_basis attr for bases this
    converter never produces).
    """
    assert converted["format_version"] == "gwcat-1.0"


def test_load_gw_samples_accepts_converter_output(converted):
    """converter output -> load_gw_samples(...) must succeed with consistent shapes.

    This is the regression for BUG-1 itself: a converter file that fails here
    can never reach the likelihood, no matter how correct its p_pe/dL sampling
    is (which the tests above already guard separately).
    """
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = converted["loaded"]

    assert nEvents == converted["nobs"] == 2
    assert nsamp == converted["nsamp"] == NSAMP
    n_total = nEvents * nsamp
    for name, arr in (
        ("m1det", m1det),
        ("m2det", m2det),
        ("dL", dL),
        ("chieff", chieff),
        ("ra", ra),
        ("dec", dec),
        ("p_pe", p_pe),
    ):
        arr = np.asarray(arr)
        assert arr.shape == (n_total,), name
        assert np.all(np.isfinite(arr)), name

    # m1det/m2det/dL/chieff/ra/dec pass through the loader unchanged.
    raw = converted["raw"]
    np.testing.assert_array_equal(np.asarray(m1det), raw["m1det"])
    np.testing.assert_array_equal(np.asarray(m2det), raw["m2det"])
    np.testing.assert_array_equal(np.asarray(dL), raw["dL"])
    np.testing.assert_array_equal(np.asarray(chieff), raw["chieff"])
    np.testing.assert_array_equal(np.asarray(ra), raw["ra"])
    np.testing.assert_array_equal(np.asarray(dec), raw["dec"])

    # p_pe: mock_data=True means the loader leaves the chi_eff prior alone
    # (no astrophysical reweight) but still renormalises per event.
    p_pe_raw = raw["p_pe"].reshape(nEvents, nsamp)
    p_pe_expected = (p_pe_raw / p_pe_raw.sum(axis=1, keepdims=True)).reshape(-1)
    np.testing.assert_allclose(np.asarray(p_pe), p_pe_expected, rtol=1e-10)


def test_m1src_m2src_round_trip_at_stated_cosmology(converted):
    """m1src/m2src must satisfy m*src * (1+z) ≈ m*det at the stated PE cosmology.

    ``z`` is recovered by inverting dL under (pe_cosmology_H0, pe_cosmology_Om0)
    with the same z_of_dL grid inversion the converter itself used -- i.e. this
    is a round-trip check, not an independent re-derivation of z.
    """
    assert converted["pe_H0"] == PE_H0
    assert converted["pe_Om0"] == PE_OM0

    raw = converted["raw"]
    z = np.asarray(z_of_dL(raw["dL"], converted["pe_H0"], converted["pe_Om0"]))
    assert np.all(np.isfinite(z))
    assert np.all(z >= 0.0)

    np.testing.assert_allclose(raw["m1src"] * (1.0 + z), raw["m1det"], rtol=1e-6)
    np.testing.assert_allclose(raw["m2src"] * (1.0 + z), raw["m2det"], rtol=1e-6)


def test_chi_eff_attrs_are_truthful(converted):
    """chi_eff_in_p_pe/chi_eff_amax must describe how p_pe was actually built.

    p_pe already includes g_chi, the flat proposal density over the sampled
    chieff range (see test_ppe_is_prior_not_sampling_density above), so
    chi_eff_in_p_pe must be True -- otherwise a non-mock loader path would
    double-count the chi_eff prior. chi_eff_amax must match the sampling range
    actually used (--chi_abs_max).
    """
    assert converted["chi_eff_in_p_pe"] is True
    assert converted["chi_eff_amax"] == pytest.approx(CHI_ABS_MAX)
