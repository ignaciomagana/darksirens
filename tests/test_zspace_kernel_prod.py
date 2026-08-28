"""Production-readiness regressions for the z-space kernel quadrature.

Pins the two fixes that gate using ``domain='zspace'`` in production:

* the empty-window fallback: a galaxy more than ``n_sigma * sig`` ABOVE the
  truncation limit must return the exact tiny log-mass (like the CDF twin's
  ``_log_ndtr_span`` recovery), not ``-inf`` (zero below-depth mass);
* the field-convention survey-global normalizer must route through the same
  quadrature dispatch as the per-pixel numerator, so the identity its
  docstring asserts holds under the z-space opt-in too.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from darksirens.core.jax_config import configure_jax_runtime

configure_jax_runtime()

from darksirens.redshift.catalog import (  # noqa: E402
    _row_log_kernel_norms,
    _row_log_kernel_norms_zspace,
    configure_kernel_quadrature,
)


@pytest.fixture(autouse=True)
def _restore_quadrature():
    yield
    configure_kernel_quadrature(24, "cdf", 5.0)


def _flat_log_g():
    from darksirens.redshift.grid import zgrid
    return jnp.zeros_like(jnp.asarray(zgrid))


def test_zspace_above_depth_galaxy_is_finite_and_matches_cdf():
    # Galaxy far above the truncation: window is empty (span_z <= 0).
    z_hi = 0.3
    zs = jnp.asarray([0.45, 0.60])
    sig = jnp.asarray([0.02, 0.02])
    real = jnp.asarray([True, True])
    log_g = _flat_log_g()
    z = np.asarray(_row_log_kernel_norms_zspace(zs, sig, real, log_g, z_hi, 5.0))
    c = np.asarray(_row_log_kernel_norms(zs, sig, real, log_g, z_hi))
    assert np.all(np.isfinite(z)), z
    # Deep-tail masses agree with the CDF twin to quadrature precision.
    assert np.allclose(z, c, rtol=0, atol=1e-6), (z, c)


def test_zspace_in_window_rows_unchanged_by_fallback():
    # Rows with a resolvable window keep the direct spelling (the fallback
    # only replaces the previously -inf/-700 branches).
    z_hi = 0.3
    zs = jnp.asarray([0.05, 0.10, 0.20])
    sig = jnp.asarray([0.01, 0.02, 0.03])
    real = jnp.asarray([True, True, True])
    log_g = _flat_log_g()
    out = np.asarray(_row_log_kernel_norms_zspace(zs, sig, real, log_g, z_hi, 5.0))
    assert np.all(np.isfinite(out))
    # Against the CDF reference at high node count these are ~1e-6 accurate.
    configure_kernel_quadrature(48, "cdf", 5.0)
    ref = np.asarray(_row_log_kernel_norms(zs, sig, real, log_g, z_hi))
    assert np.allclose(out, ref, rtol=0, atol=1e-4), (out, ref)


def test_field_depth_mass_routes_through_quadrature_dispatch(monkeypatch):
    # The survey-global observed total must follow the configured domain:
    # spy on the zspace row function and assert it fires under the opt-in.
    import darksirens.redshift.catalog as cat
    from darksirens.redshift import completion

    calls = {"n": 0}
    orig = cat._row_log_kernel_norms_zspace

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(cat, "_row_log_kernel_norms_zspace", spy)
    configure_kernel_quadrature(8, "zspace", 5.0)
    zs = jnp.asarray([0.05, 0.10])
    sig = jnp.asarray([0.01, 0.02])
    real = jnp.asarray([True, True])
    cat._dispatch_log_kernel_norms(zs, sig, real, _flat_log_g(), z_hi=0.3)
    assert calls["n"] == 1
    # and the completion module resolves the dispatcher, not the CDF row
    # function, in its deferred import.
    src = open(completion.__file__).read()
    assert "_dispatch_log_kernel_norms" in src
    assert "log_Z_full = _dispatch_log_kernel_norms" in src
