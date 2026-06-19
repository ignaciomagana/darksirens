"""
Regression test for ``run_sampler`` with zero free parameters.

When every parameter block is fixed (e.g. ``--sky_model isotropic`` with
``--fix_population --fixed_cosmology``, the null model of the sky ladder) the
parameter space has ``ndim == 0``.  The prior is then a point mass at the fixed
point, so the evidence is exact: ``Z = L(theta_fixed)`` ⇒ ``logZ = logL``.

``run_sampler`` must short-circuit and return that — for *every* sampler —
rather than handing a 0-dimensional problem to dynesty (which crashes in its
bounding-ellipsoid eigendecomposition, LAPACK ``dsyevr: il=1``), jaxns, or
emcee.  The short-circuit returns before the method dispatch, so it imports no
sampler package; this test therefore needs only jax + numpy.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.inference.sampling import run_sampler


_FIXED_LOG_L = -3.14159


def _stub_likelihood(coord):
    # The short-circuit evaluates the likelihood once at the empty coordinate
    # vector; assert that is exactly what it passes, then return the fixed-point
    # log-likelihood.
    assert np.asarray(coord).size == 0
    return jnp.asarray(_FIXED_LOG_L)


@pytest.mark.parametrize("method", ["dynesty", "numpyro", "jaxns", "emcee"])
def test_run_sampler_zero_free_params_short_circuits(method):
    # ``opts`` is intentionally empty: the ndim==0 path returns before any
    # sampler-specific option is read or any sampler package is imported, so
    # this must hold identically for every sampler (in particular dynesty,
    # which previously crashed on the 0-dimensional bounding ellipsoid).
    results = run_sampler(
        method=method,
        likelihood=_stub_likelihood,
        prior_transform=lambda u: u,   # never called when ndim == 0
        labels=[],
        lower_bound=np.zeros(0),
        upper_bound=np.zeros(0),
        opts=SimpleNamespace(),
    )

    # One posterior point (the fixed point), zero free dimensions.
    assert np.asarray(results["samples"]).shape == (1, 0)
    # Evidence is exact for a delta-function prior: logZ = logL, no MC error.
    assert results["logZ"] == pytest.approx(_FIXED_LOG_L)
    assert results["logZerr"] == 0.0
    np.testing.assert_allclose(np.asarray(results["log_likelihood"]), [_FIXED_LOG_L])
