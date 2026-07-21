"""darksirens_analyze redshift-rate posterior (cli/analyze.py, BUG-4).

``redshift_rate_samples`` feeds the ``rate_dNdz.pdf`` figure. It used to accept
only per-sample H0 plus a scalar-fixed Om0 and drop any sampled w0/wa
entirely, so a run that actually sampled the dark-energy / matter-density
parameters had its rate curve SHAPE silently computed at the fiducial
cosmology (only H0's overall rescaling survived, and even that mostly cancels
under the per-sample normalisation). These tests pin the fixed behaviour: all
four cosmology coordinates (H0, Om0, w0, wa) flow through to ``dV_of_z``."""
import sys
import types

if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub

import numpy as np

from darksirens.cli.analyze import redshift_rate_samples
from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID
from darksirens.utils.cosmology import dV_of_z


def _reference_rate(pz_samples, zgrid, h0_samples, om0_samples, w0_samples, wa_samples):
    """Unvectorised per-sample reference: literally the docstring formula."""
    zg = np.asarray(zgrid)
    n = len(h0_samples)
    om0_arr = np.broadcast_to(np.asarray(om0_samples, dtype=float), (n,))
    w0_arr = np.broadcast_to(np.asarray(w0_samples, dtype=float), (n,))
    wa_arr = np.broadcast_to(np.asarray(wa_samples, dtype=float), (n,))
    out = np.empty((n, zg.size))
    for i in range(n):
        dv = np.asarray(dV_of_z(zg, float(h0_samples[i]), float(om0_arr[i]),
                                float(w0_arr[i]), float(wa_arr[i])))
        rate = np.asarray(pz_samples[i]) * dv / (1.0 + zg)
        norm = np.trapezoid(rate, zg)
        out[i] = rate / norm if norm > 0 else rate
    return out


def _synthetic_pz(n_samples, zgrid, seed):
    """Smooth positive p(z)-like curves, identical cosmology-independent shape
    inputs so any difference in the rate output is attributable purely to the
    cosmology arguments under test."""
    rng = np.random.default_rng(seed)
    z = np.asarray(zgrid)
    centers = rng.uniform(0.3, 1.2, size=n_samples)
    widths = rng.uniform(0.3, 0.6, size=n_samples)
    return np.exp(-0.5 * ((z[None, :] - centers[:, None]) / widths[:, None]) ** 2) + 0.05


class TestRedshiftRateSamplesFullCosmology:
    def setup_method(self):
        self.zgrid = np.linspace(0.0, 2.0, 40)
        self.n_samples = 25
        self.pz_samples = _synthetic_pz(self.n_samples, self.zgrid, seed=0)
        self.h0_samples = np.full(self.n_samples, 70.0)

    def test_separated_om0_w0_wa_change_the_rate_shape(self):
        """Two parameter sets with deliberately separated Om0/w0/wa (same H0,
        same p_z) must produce different dN/dz curves — pre-fix, both would be
        (near-)identical because Om0/w0/wa were dropped at the call site."""
        rate_a = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, 0.2),
            w0_samples=np.full(self.n_samples, -1.5),
            wa_samples=np.full(self.n_samples, -0.5),
        )
        rate_b = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, 0.4),
            w0_samples=np.full(self.n_samples, -0.5),
            wa_samples=np.full(self.n_samples, 0.5),
        )
        assert not np.allclose(rate_a, rate_b, rtol=1e-3, atol=1e-8)

    def test_matches_direct_per_sample_numpy_reference(self):
        """Per-sample arrays (not just scalars) for Om0/w0/wa, with H0 also
        varying per sample, must match an unvectorised numpy reference built
        directly from dV_of_z at those exact per-sample values."""
        rng = np.random.default_rng(1)
        h0_samples = rng.uniform(60.0, 80.0, size=self.n_samples)
        om0_samples = rng.uniform(0.2, 0.4, size=self.n_samples)
        w0_samples = rng.uniform(-1.5, -0.5, size=self.n_samples)
        wa_samples = rng.uniform(-0.5, 0.5, size=self.n_samples)

        got = redshift_rate_samples(
            self.pz_samples, self.zgrid, h0_samples,
            om0_samples=om0_samples, w0_samples=w0_samples, wa_samples=wa_samples,
        )
        expected = _reference_rate(
            self.pz_samples, self.zgrid, h0_samples, om0_samples, w0_samples, wa_samples,
        )
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)

    def test_scalar_om0_w0_wa_broadcast_across_samples(self):
        """Scalar Om0/w0/wa (the fixed-cosmology call-site case) broadcast to
        every sample identically to an explicit per-sample-repeated array."""
        got_scalar = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=0.25, w0_samples=-1.2, wa_samples=0.3,
        )
        got_array = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, 0.25),
            w0_samples=np.full(self.n_samples, -1.2),
            wa_samples=np.full(self.n_samples, 0.3),
        )
        np.testing.assert_allclose(got_scalar, got_array, rtol=1e-12, atol=1e-14)

    def test_h0_only_call_reproduces_fiducial_om0_w0_wa_backward_compat(self):
        """An H0-only call (the pre-fix call signature) must still work and
        must reproduce the previous behaviour: Om0/w0/wa held at fiducial."""
        got = redshift_rate_samples(self.pz_samples, self.zgrid, self.h0_samples)
        expected = _reference_rate(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, OM0_FID),
            w0_samples=np.full(self.n_samples, W0_FID),
            wa_samples=np.full(self.n_samples, WA_FID),
        )
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-10)

        # Also matches the *legacy* two-cosmology-arg call path, which relied
        # on dV_of_z's own w0/wa defaults (w0Fiducial=-1.0, waFiducial=0.0 —
        # numerically identical to W0_FID/WA_FID).
        legacy_zg = np.asarray(self.zgrid)
        legacy = np.empty_like(got)
        for i in range(self.n_samples):
            dv = np.asarray(dV_of_z(legacy_zg, float(self.h0_samples[i]), OM0_FID))
            rate = np.asarray(self.pz_samples[i]) * dv / (1.0 + legacy_zg)
            norm = np.trapezoid(rate, legacy_zg)
            legacy[i] = rate / norm if norm > 0 else rate
        np.testing.assert_allclose(got, legacy, rtol=1e-6, atol=1e-10)

    def test_output_stays_normalised_per_sample(self):
        rate = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, 0.35),
            w0_samples=np.full(self.n_samples, -0.8),
            wa_samples=np.full(self.n_samples, 0.1),
        )
        integrals = np.trapezoid(rate, self.zgrid, axis=1)
        np.testing.assert_allclose(integrals, 1.0, rtol=1e-6, atol=1e-9)
