"""darksirens_analyze redshift-rate posterior (cli/analyze.py, BUG-4).

``redshift_rate_samples`` feeds the ``rate_dNdz.pdf`` figure. Two things are
pinned here.

**Cosmology plumbing.** It used to accept only per-sample H0 plus a scalar-fixed
Om0 and drop any sampled w0/wa entirely, so a run that actually sampled the
dark-energy / matter-density parameters had its rate curve SHAPE silently
computed at the fiducial cosmology (only H0's overall rescaling survived, and
even that mostly cancels under the per-sample normalisation). All four
cosmology coordinates (H0, Om0, w0, wa) must flow through to ``dV_of_z``.

**No double-counted time dilation.** ``p_z`` ALREADY carries the 1/(1+z)
observer/source time dilation: :meth:`PopulationModel.log_rate_z` returns
``(gamma - 1) log1p(z)`` for the power-law evolution and ``log[psi(z)/(1+z)]``
for Madau-Dickinson (``gw/populations/base.py``). The observed rate is therefore
``p_z * dV_c/dz`` with NO further division; a second 1/(1+z) plots
(1+z)^(gamma-2) dV_c/dz and biases the curve low in redshift.

Note on references: ``_reference_rate`` below deliberately exercises only the
*plumbing* (it reuses the package's own ``dV_of_z``, which is not what these
tests are about) and is held to a tight tolerance. The *physics* — the rate law
and the absence of a second (1+z) — is anchored separately against astropy in
``_astropy_reference_rate``. A test suite whose only reference re-transcribes
the expression under test pins whatever that expression happens to do, correct
or not, which is exactly how the double-counted (1+z) survived its own
"regression" test.
"""
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

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest
from astropy.cosmology import Flatw0waCDM

import jax.numpy as jnp

from darksirens.cli.analyze import redshift_rate_samples
from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID
from darksirens.gw.populations.registry import (
    MD_RATE_FIDUCIALS,
    get_fixed_population_params,
    get_model,
)
from darksirens.utils.cosmology import dV_of_z


def _reference_rate(pz_samples, zgrid, h0_samples, om0_samples, w0_samples, wa_samples):
    """Unvectorised per-sample PLUMBING reference.

    Reuses ``dV_of_z`` on purpose: this checks that each sample's own
    (H0, Om0, w0, wa) reaches the volume element, not that the volume element
    or the rate law is right. No 1/(1+z) — it is already inside ``p_z``.
    """
    zg = np.asarray(zgrid)
    n = len(h0_samples)
    om0_arr = np.broadcast_to(np.asarray(om0_samples, dtype=float), (n,))
    w0_arr = np.broadcast_to(np.asarray(w0_samples, dtype=float), (n,))
    wa_arr = np.broadcast_to(np.asarray(wa_samples, dtype=float), (n,))
    out = np.empty((n, zg.size))
    for i in range(n):
        dv = np.asarray(dV_of_z(zg, float(h0_samples[i]), float(om0_arr[i]),
                                float(w0_arr[i]), float(wa_arr[i])))
        rate = np.asarray(pz_samples[i]) * dv
        norm = _trapezoid(rate, zg)
        out[i] = rate / norm if norm > 0 else rate
    return out


def _astropy_reference_rate(zgrid, gamma, H0=H0_FID, Om0=OM0_FID, w0=W0_FID, wa=WA_FID,
                            extra_time_dilation=False):
    """Independent PHYSICS reference: astropy volume x the analytic rate law.

    ``extra_time_dilation=True`` reproduces the *erroneous* curve, so tests can
    assert the two are far apart rather than merely that one of them matches.
    """
    z = np.asarray(zgrid)
    dv = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa).differential_comoving_volume(z).value
    rate = (1.0 + z) ** (gamma - 1.0) * dv
    if extra_time_dilation:
        rate = rate / (1.0 + z)
    return rate / _trapezoid(rate, z)


def _median(curve, z):
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (curve[1:] + curve[:-1]) * np.diff(z))))
    return float(np.interp(0.5 * cdf[-1], cdf, z))


def _powerlaw_pz(zgrid, gamma):
    """p_z straight from the package population model (not hand-written)."""
    model = get_model("powerlaw+peak")
    theta = get_fixed_population_params("powerlaw+peak").at[-1].set(gamma)
    return np.asarray(jnp.exp(model.log_rate_z(jnp.asarray(zgrid), theta)))


def _synthetic_pz(n_samples, zgrid, seed):
    """Smooth positive p(z)-like curves, identical cosmology-independent shape
    inputs so any difference in the rate output is attributable purely to the
    cosmology arguments under test."""
    rng = np.random.default_rng(seed)
    z = np.asarray(zgrid)
    centers = rng.uniform(0.3, 1.2, size=n_samples)
    widths = rng.uniform(0.3, 0.6, size=n_samples)
    return np.exp(-0.5 * ((z[None, :] - centers[:, None]) / widths[:, None]) ** 2) + 0.05


# ============================================================================
# Physics: the rate law, and the absence of a second (1+z)
# ============================================================================

class TestNoDoubleCountedTimeDilation:
    # Grid starts at z = 0.01, not 0: the package interpolates r(z) linearly on
    # a log-spaced z grid, so as r -> 0 the *relative* error grows (1.2e-3 at
    # z = 1e-3, 1.1e-4 at z = 0.01, ~3e-5 for z > 0.05). That floor is a
    # property of the distance table, not of the rate law under test here, and
    # the 5e-4 tolerance below sits above it while staying ~200x tighter than
    # the whole-power-of-(1+z) effect these tests discriminate.
    zgrid = np.linspace(0.01, 4.0, 600)

    @pytest.mark.parametrize("gamma", [1.0, 2.7, 4.0])
    def test_log_rate_z_already_divides_by_one_plus_z(self, gamma):
        """The model's own p_z is (1+z)^(gamma-1) == R(z)/(1+z), not R(z).

        Fit the exponent rather than restate the formula: gamma-1 means the
        time dilation is inside, gamma would mean it is not.
        """
        log_rate = np.log(_powerlaw_pz(self.zgrid, gamma))
        slope = np.polyfit(np.log1p(self.zgrid), log_rate, 1)[0]
        assert slope == pytest.approx(gamma - 1.0, abs=1e-9)

    @pytest.mark.parametrize("gamma", [1.0, 2.7, 4.0])
    def test_matches_independent_astropy_rate(self, gamma):
        """dN/dz == (1+z)^(gamma-1) dV_c/dz, referenced to astropy."""
        pz = _powerlaw_pz(self.zgrid, gamma)
        got = redshift_rate_samples(pz[None, :], self.zgrid, np.array([H0_FID]))[0]
        want = _astropy_reference_rate(self.zgrid, gamma)
        np.testing.assert_allclose(got, want, rtol=5e-4)

    @pytest.mark.parametrize("gamma", [1.0, 2.7, 4.0])
    def test_erroneous_double_division_is_clearly_rejected(self, gamma):
        """The regression guard: the (1+z)^(gamma-2) curve must NOT match.

        The two hypotheses differ by a whole power of (1+z) over z in [0, 4],
        which moves the median redshift by ~0.2-0.6, so this fails loudly if the
        divide is ever restored.
        """
        pz = _powerlaw_pz(self.zgrid, gamma)
        got = redshift_rate_samples(pz[None, :], self.zgrid, np.array([H0_FID]))[0]
        correct = _astropy_reference_rate(self.zgrid, gamma)
        doubled = _astropy_reference_rate(self.zgrid, gamma, extra_time_dilation=True)

        assert _median(doubled, self.zgrid) < _median(correct, self.zgrid) - 0.15
        assert abs(_median(got, self.zgrid) - _median(correct, self.zgrid)) < 5e-3
        assert np.max(np.abs(got - doubled)) > 0.05 * np.max(correct)

    def test_madau_dickinson_rate(self):
        """'@md' evolution: psi(z)/(1+z) * dV_c/dz, again astropy-referenced."""
        gamma, kappa, z_peak = MD_RATE_FIDUCIALS
        model = get_model("powerlaw+peak@md")
        theta = get_fixed_population_params("powerlaw+peak@md")
        theta = theta.at[-3].set(gamma).at[-2].set(kappa).at[-1].set(z_peak)
        pz = np.asarray(jnp.exp(model.log_rate_z(jnp.asarray(self.zgrid), theta)))

        got = redshift_rate_samples(pz[None, :], self.zgrid, np.array([H0_FID]))[0]
        z = self.zgrid
        psi = (1.0 + z) ** gamma / (1.0 + ((1.0 + z) / (1.0 + z_peak)) ** (gamma + kappa))
        dv = Flatw0waCDM(
            H0=H0_FID, Om0=OM0_FID, w0=W0_FID, wa=WA_FID
        ).differential_comoving_volume(z).value
        want = psi / (1.0 + z) * dv
        want = want / _trapezoid(want, z)
        np.testing.assert_allclose(got, want, rtol=5e-4)


# ============================================================================
# Cosmology plumbing
# ============================================================================

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
        must hold Om0/w0/wa at fiducial."""
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
            rate = np.asarray(self.pz_samples[i]) * dv
            norm = _trapezoid(rate, legacy_zg)
            legacy[i] = rate / norm if norm > 0 else rate
        np.testing.assert_allclose(got, legacy, rtol=1e-6, atol=1e-10)

    def test_output_stays_normalised_per_sample(self):
        rate = redshift_rate_samples(
            self.pz_samples, self.zgrid, self.h0_samples,
            om0_samples=np.full(self.n_samples, 0.35),
            w0_samples=np.full(self.n_samples, -0.8),
            wa_samples=np.full(self.n_samples, 0.1),
        )
        integrals = _trapezoid(rate, self.zgrid, axis=1)
        np.testing.assert_allclose(integrals, 1.0, rtol=1e-6, atol=1e-9)
