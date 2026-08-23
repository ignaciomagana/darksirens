"""PHY-06: the lognormal Hermite backend must integrate the STATED target.

The Hermite substitution u = (ln mu - m(z))/s(z) uses the APPARENT redshift
z_app = z(dL_app) to set (m, s), so its nodes are drawn from the proposal

    q(mu) = p_WL(mu | z_app),

while the stated target -- the one the generic ln-mu backend integrates, and
the one the module docstring writes down -- carries p_WL(mu | z_s(mu)) with
z_s(mu) = z(dL_app * sqrt(mu)).  Evaluating everything else at z_s while the
measure stays at z_app integrates the wrong density.

Reproduction (pre-fix): with a source weight arranged to be exactly 1, the
Hermite quadrature returns exactly 1.000000 at every redshift, while direct
scipy quadrature of the stated target returns

    z_app = 0.5 -> 0.999822
    z_app = 1   -> 0.999517
    z_app = 2   -> 0.998712
    z_app = 4   -> 0.996507        (a = 0.004, b = 1.5)

Those four values are the review's, and this file recomputes them from scratch
by adaptive quadrature rather than hard-coding them.  Pre-fix the kernel
returned 1.0000040, 1.0000019 and 1.0000008 at the first three.

The fix is the log-density-ratio importance weight

    log p_WL(mu | z_s(mu)) - log p_WL(mu | z_app)

added to every node.  The tests below check the pure-normalization case at the
four reference redshifts AND nonconstant source weights, where the discrepancy
is NOT bounded by the small numbers above: with a mu^3 source weight the
pre-fix kernel is 2.1% low at z_app = 2.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from darksirens.core.types import CosmoParams
from darksirens.likelihood.wl_weight import log_sample_weight_wl_lognormal_hermite
from darksirens.lensing.grids import make_hermite_u_grid
from darksirens.utils.cosmology import (
    H0Planck, Om0Planck, dL_in_z_grid, dL_of_z, z_of_dL,
)

from tests.test_wl_weight import _toy_catalog, _toy_survey

WL_A, WL_B = 0.004, 1.5
# The review's four reference redshifts.  z = 4 runs in a SUBPROCESS at the
# bottom of this file: its +12 sigma tail reaches mu = 8.4, i.e. dL_true =
# 2.9 dL(z=4), which is off the tabulated z(dL) grid at the repo default
# DARKSIRENS_ZMAX = 5.  Both sides then integrate a truncated integrand and
# neither reproduces 0.996507; raising the grid recovers it (measured 0.996519).
Z_REFERENCE = (0.5, 1.0, 2.0)
REVIEW_VALUES = {0.5: 0.999822, 1.0: 0.999517, 2.0: 0.998712, 4.0: 0.996507}
# Floor of the comparison: the reference has to round-trip z_s -> dL(z_s)
# through the interpolated cosmology grid to rebuild the source weight, which
# costs a few times 1e-6 relative and does not shrink with node count.  The
# effect under test is 1.8e-4 to 1.3e-3, forty times larger.
_HARNESS_REL = 2.0e-5

_COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
_ARGS = (_COSMO.H0, _COSMO.Om0, _COSMO.w0, _COSMO.wa)


# ---------------------------------------------------------------------------
# numpy/scipy reference: the stated target, integrated directly in mu
# ---------------------------------------------------------------------------

def _s_of_z(z):
    """Lognormal width, with the z clamp make_lognormal_log_p_wl applies."""
    return np.sqrt(WL_A * np.maximum(np.asarray(z, dtype=float), 1.0e-3) ** WL_B)


def _log_p_wl(mu, z):
    s = _s_of_z(z)
    return norm.logpdf(np.log(mu), loc=-0.5 * s**2, scale=s) - np.log(mu)


def _z_source(mu, dL_app):
    dl = dL_app * np.sqrt(mu)
    if not bool(dL_in_z_grid(jnp.asarray(dl), *_ARGS)):
        return None
    return float(z_of_dL(jnp.asarray(dl), *_ARGS))


def _direct_target_integral(dL_app, z_app, source_weight):
    """int dmu p_WL(mu | z_s(mu)) * source_weight(mu, z_s(mu)), by adaptive
    quadrature over the proposal's +-12 sigma support (out-of-grid nodes
    contribute nothing, exactly as the kernel masks them)."""
    def integrand(mu):
        z_s = _z_source(mu, dL_app)
        if z_s is None:
            return 0.0
        return float(np.exp(_log_p_wl(mu, z_s)) * source_weight(mu, z_s))

    s0 = float(_s_of_z(z_app))
    lo = float(np.exp(-0.5 * s0**2 - 12.0 * s0))
    hi = float(np.exp(-0.5 * s0**2 + 12.0 * s0))
    value, _err = quad(integrand, lo, hi, limit=800, epsabs=1e-13, epsrel=1e-12)
    return value


# ---------------------------------------------------------------------------
# driving the kernel with a chosen source weight
# ---------------------------------------------------------------------------

def _hermite_integral(dL_app, source_weight_log, n_nodes=64):
    """Run the real kernel and strip the PE proposal, so the returned number is
    the bare quadrature estimate of int dmu p_WL(...) * source_weight.

    The kernel's own integrand is

        p_pop * p_z * sqrt(mu) / [(1+z_s) dL'(z_s)],

    so the two closures below are given the job of cancelling the sqrt(mu) and
    the Jacobian, leaving exactly ``source_weight``.  Every argument the
    closures need is recoverable from z_s alone at fixed dL_app: dL_true =
    dL(z_s) because z_s = z(dL_true), hence sqrt(mu) = dL(z_s)/dL_app.
    """
    from darksirens.inference.utils import log_jacobian_m1src_q_z_to_m1det_q_dL

    def log_p_pop_fn(m1src, q, z, chieff, pop_params, spin=None):
        del m1src, q, chieff, pop_params, spin
        dL_true = dL_of_z(z, *_ARGS)
        log_sqrt_mu = jnp.log(dL_true / dL_app)
        return (
            log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL_true, *_ARGS)
            - log_sqrt_mu
            + source_weight_log(dL_true, z)
        )

    def log_prior_z_fn(z, pix, catalog):
        del pix, catalog
        return jnp.zeros_like(z)

    u_nodes, log_wH = make_hermite_u_grid(n_nodes)
    out = log_sample_weight_wl_lognormal_hermite(
        jnp.asarray([30.0]), jnp.asarray([0.8]), jnp.asarray([dL_app]),
        jnp.asarray([0.0]), jnp.zeros(1, dtype=jnp.int32), jnp.ones(1),
        _COSMO, _toy_survey(), jnp.array([]), _toy_catalog(),
        log_p_pop_fn, log_prior_z_fn,
        jnp.asarray(WL_A), jnp.asarray(WL_B), u_nodes, log_wH,
    )
    return float(jnp.exp(out[0]))


# ---------------------------------------------------------------------------
# 1. the review's reference normalizations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z_app", Z_REFERENCE)
def test_constant_source_weight_reproduces_the_direct_target_normalization(z_app):
    """Source weight identically 1: the quadrature must return the stated
    target's normalization, NOT 1."""
    dL_app = float(dL_of_z(jnp.asarray(z_app), *_ARGS))
    reference = _direct_target_integral(dL_app, z_app, lambda mu, z: 1.0)
    # The review's number, recomputed from scratch.
    assert reference == pytest.approx(REVIEW_VALUES[z_app], abs=1e-6)
    # Not 1: the proposal-integrating backend cannot see this.
    assert abs(reference - 1.0) > 1e-4

    got = _hermite_integral(dL_app, lambda dL_true, z: jnp.zeros_like(z))
    assert got == pytest.approx(reference, rel=_HARNESS_REL), (
        f"Hermite returned {got!r} at z_app={z_app}; the stated target is "
        f"{reference!r}. A return of exactly 1 means the density ratio is "
        f"missing and the backend is integrating p(mu | z_app)."
    )
    # And it moved essentially all the way there, not a fraction of the way.
    assert abs(got - reference) < 0.05 * abs(1.0 - reference)


# ---------------------------------------------------------------------------
# 2. nonconstant source weights, where the error is not bounded by 3e-3
# ---------------------------------------------------------------------------

_NONCONSTANT = {
    # steep in z: the population/volume factors the review warns about
    "z_power_4": (lambda mu, z: (1.0 + z) ** 4,
                  lambda dL_true, z: 4.0 * jnp.log1p(z)),
    # steep in mu (i.e. in dL_true): a magnification-sensitive weight
    "mu_power_3": (lambda mu, z: mu**3,
                   lambda dL_true, z: 6.0 * jnp.log(dL_true)),
    "mixed": (lambda mu, z: mu**2 * np.exp(-z),
              lambda dL_true, z: 4.0 * jnp.log(dL_true) - z),
}


@pytest.mark.parametrize("z_app", (0.5, 1.0, 2.0))
@pytest.mark.parametrize("name", sorted(_NONCONSTANT))
def test_nonconstant_source_weights_match_direct_quadrature(z_app, name):
    numpy_weight, jax_log_weight = _NONCONSTANT[name]
    dL_app = float(dL_of_z(jnp.asarray(z_app), *_ARGS))
    # mu_power_3 / mixed are written in dL_true on the JAX side; normalize the
    # numpy side to the same convention via sqrt(mu) = dL_true / dL_app.
    if name == "mu_power_3":
        numpy_weight = lambda mu, z: (dL_app * np.sqrt(mu)) ** 6  # noqa: E731
    elif name == "mixed":
        numpy_weight = lambda mu, z: (dL_app * np.sqrt(mu)) ** 4 * np.exp(-z)  # noqa: E731

    reference = _direct_target_integral(dL_app, z_app, numpy_weight)
    got = _hermite_integral(dL_app, jax_log_weight)
    # Steep weights amplify the harness's cosmology-grid round-trip; the
    # pre-fix discrepancies on these same cases are 3.1e-4 (z_power_4, z=0.5)
    # up to 2.1e-2 (mu_power_3, z=2) -- measured, and one to two orders of
    # magnitude above this bound.
    assert got == pytest.approx(reference, rel=1.0e-4), (
        f"{name} at z_app={z_app}: Hermite {got!r} vs direct target {reference!r}"
    )


def test_the_z4_reference_needs_a_taller_z_grid_than_the_default():
    """Why z = 4 is not in Z_REFERENCE: the proposal's +12 sigma tail leaves the
    tabulated z(dL) grid at DARKSIRENS_ZMAX = 5, so BOTH quadratures truncate."""
    dL_app = float(dL_of_z(jnp.asarray(4.0), *_ARGS))
    s0 = float(_s_of_z(4.0))
    mu_hi = float(np.exp(-0.5 * s0**2 + 12.0 * s0))
    assert not bool(dL_in_z_grid(jnp.asarray(dL_app * np.sqrt(mu_hi)), *_ARGS))


def test_z4_matches_the_review_value_on_a_grid_that_reaches_it():
    """The fourth reference value, in a clean process with DARKSIRENS_ZMAX=20
    (the grid extent is read at import, so it cannot be changed in-process)."""
    import os
    import subprocess

    script = f"""
import os, sys
sys.path.insert(0, {str(HERE.parent)!r})
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tests.test_wl_hermite_target_density import (
    _ARGS, _direct_target_integral, _hermite_integral)
from darksirens.utils.cosmology import dL_of_z
dL_app = float(dL_of_z(jnp.asarray(4.0), *_ARGS))
ref = _direct_target_integral(dL_app, 4.0, lambda mu, z: 1.0)
got = _hermite_integral(dL_app, lambda dL_true, z: jnp.zeros_like(z))
print("RESULT", repr(ref), repr(got))
"""
    env = dict(os.environ, DARKSIRENS_ZMAX="20", JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, cwd=str(HERE.parent), timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][-1]
    reference, got = (float(x) for x in line.split()[1:])
    assert reference == pytest.approx(REVIEW_VALUES[4.0], abs=2e-5)
    assert got == pytest.approx(reference, rel=_HARNESS_REL)
    assert abs(got - reference) < 0.05 * abs(1.0 - reference)


# ---------------------------------------------------------------------------
# 3. the ratio must not disturb the advertised wl_a = 0 reduction
# ---------------------------------------------------------------------------

class TestZeroWidthReductionUnchanged:
    """At wl_a = 0 both lognormals are degenerate at mu = 1: the ratio is
    identically 0 and both the VALUE and the reverse pass stay as advertised."""

    @staticmethod
    def _scalar_ll(H0, wl_a):
        from tests.test_wl_weight import (
            _toy_log_p_pop, _toy_samples, _toy_volume_prior,
        )

        cosmo = CosmoParams(H0=H0, Om0=Om0Planck)
        u_nodes, log_wH = make_hermite_u_grid(16)
        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=5, seed=3)
        return jnp.sum(log_sample_weight_wl_lognormal_hermite(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, _toy_survey(), jnp.array([]), _toy_catalog(),
            _toy_log_p_pop, _toy_volume_prior,
            wl_a, jnp.asarray(WL_B), u_nodes, log_wH,
        ))

    def test_value_matches_the_standard_hot_path(self):
        from darksirens.inference.utils import log_sample_weight
        from tests.test_wl_weight import (
            _toy_log_p_pop, _toy_samples, _toy_volume_prior,
        )

        m1det, q, dL, chieff, pix, prior_wt = _toy_samples(n=5, seed=3)
        standard = float(jnp.sum(log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            _COSMO, _toy_survey(), jnp.array([]), _toy_catalog(),
            _toy_log_p_pop, _toy_volume_prior,
        )))
        got = float(self._scalar_ll(jnp.asarray(H0Planck), jnp.asarray(0.0)))
        assert got == pytest.approx(standard, rel=1e-12)

    def test_cosmology_gradient_is_still_finite(self):
        g = jax.grad(self._scalar_ll)(jnp.asarray(H0Planck), jnp.asarray(0.0))
        assert np.isfinite(float(g))

    def test_gradient_in_wl_a_is_finite_at_zero(self):
        g = jax.grad(self._scalar_ll, argnums=1)(
            jnp.asarray(H0Planck), jnp.asarray(0.0)
        )
        assert np.isfinite(float(g))
