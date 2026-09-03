"""Convergence guard for the Hermite WL importance-ratio quadrature.

The importance-ratio remediation (9ce0fc3) made the lognormal WL backend
integrate the stated target p_WL(mu | z_s(mu)) through a proposal drawn at
z_app.  The ratio's exponent grows like +u^2/2 (1 - s_app^2/s_s^2), positive
for u > 0 whenever wl_b > 0, so the effective Gauss-Hermite integrand is
super-Gaussian: at amplified --lensing_wl_a (an otherwise unbounded CLI float)
the per-sample log-weights are silently wrong, z-dependent (hence H0-coupled),
and do NOT improve with more nodes.  At the calibrated default a = 4e-3 the
rule is fine.

These tests pin the startup guard added for that hole:

* ``wl_hermite_quadrature_errors`` measures the rule against a dense reference
  quadrature of the identical masked target, through the production kernel
  algebra (shared helper);
* ``validate_wl_hermite_quadrature`` passes the calibrated default and refuses
  amplified amplitudes;
* more nodes do not rescue the refused regime (the non-convergence that makes
  a node-count bump a non-fix);
* the lensing CLI's eager WL chokepoint (``_load_wl_table_arrays``) fires the
  guard for the lognormal backend the way it already validated the tabulated
  one.  Before the fix, an amplified-a run passed startup silently.
"""
import types

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from darksirens.lensing.grids import make_hermite_u_grid
from darksirens.likelihood.wl_weight import (
    validate_wl_hermite_quadrature,
    wl_hermite_quadrature_errors,
)

TOL = 1.0e-4


def test_default_amplitude_passes_with_margin():
    """The calibrated default (a=4e-3, b=1.5) converges at 16 nodes."""
    u_nodes, log_wH = make_hermite_u_grid(16)
    _z, err = wl_hermite_quadrature_errors(4.0e-3, 1.5, u_nodes, log_wH)
    assert np.all(np.isfinite(err))
    assert err.max() <= TOL
    # validate returns the error array instead of raising
    out = validate_wl_hermite_quadrature(4.0e-3, 1.5, u_nodes, log_wH)
    assert np.all(out <= TOL)


def test_amplified_amplitude_is_refused():
    """a = 0.05 (12.5x calibrated) breaks convergence and must be refused."""
    with pytest.raises(ValueError, match="does not converge"):
        validate_wl_hermite_quadrature(0.05, 1.5)


def test_moderately_amplified_amplitude_is_refused():
    """Even 3x the calibrated amplitude (a=0.012) is out of the trusted
    regime at high z: the review measured ~1e-3 nats of per-sample error."""
    with pytest.raises(ValueError, match="does not converge"):
        validate_wl_hermite_quadrature(0.012, 1.5)


def test_more_nodes_do_not_rescue_the_amplified_regime():
    """The failure is non-convergence, not resolution: the error at a=0.05
    stays orders of magnitude over tolerance at 32 and 64 nodes, so bumping
    the node count is not a fix and the guard must not be tuned around."""
    for n_nodes in (32, 64):
        u_nodes, log_wH = make_hermite_u_grid(n_nodes)
        _z, err = wl_hermite_quadrature_errors(0.05, 1.5, u_nodes, log_wH)
        assert err.max() > 10.0 * TOL, (
            f"{n_nodes} nodes: worst error {err.max():.3e} unexpectedly small"
        )


def test_wl_a_zero_is_exact_and_passes():
    """a = 0: delta at mu=1, every node collapses, the rule is exact by
    construction (the advertised reduce-to-standard ablation must not be
    refused)."""
    u_nodes, log_wH = make_hermite_u_grid(16)
    _z, err = wl_hermite_quadrature_errors(0.0, 1.5, u_nodes, log_wH)
    assert np.all(err == 0.0)
    validate_wl_hermite_quadrature(0.0, 1.5, u_nodes, log_wH)


def test_cli_lognormal_chokepoint_fires_the_guard():
    """_load_wl_table_arrays is the lensing CLI's eager WL validation
    chokepoint (it already hard-fails unresolved tabulated tables); with the
    lognormal backend it must now refuse an amplified --lensing_wl_a as a
    clean SystemExit, and stay inert for the calibrated default."""
    from darksirens.cli import inference_lensing as lens_cli

    bad = types.SimpleNamespace(
        wl_backend="lognormal",
        lensing_wl_a=0.05,
        lensing_wl_b=1.5,
        lensing_wl_table_path=None,
    )
    with pytest.raises(SystemExit, match="does not converge"):
        lens_cli._load_wl_table_arrays(bad)

    good = types.SimpleNamespace(
        wl_backend="lognormal",
        lensing_wl_a=4.0e-3,
        lensing_wl_b=1.5,
        lensing_wl_table_path=None,
    )
    assert lens_cli._load_wl_table_arrays(good) == {}


def test_hermite_guard_probes_low_redshift_where_a_non_positive_b_diverges():
    """``wl_b <= 0`` makes s(z) grow toward z = 0; the old test grid started at
    z_app = 0.525 and passed a = 4e-3, b = -1.5 with a 4e-7 worst error while
    the kernel was off by up to 1.7e-2 nats at z = 0.1, where the events are."""
    import pytest
    from darksirens.likelihood.wl_weight import validate_wl_hermite_quadrature

    with pytest.raises(ValueError, match="does not converge"):
        validate_wl_hermite_quadrature(4e-3, -1.5)
    # the calibrated default still clears the widened grid
    err = validate_wl_hermite_quadrature(4e-3, 1.5)
    assert float(err.max()) < 1e-4


def test_negative_wl_a_is_refused_not_treated_as_no_lensing():
    """For a < 0 the kernel's width guard zeroes s but the mean -s^2/2 stays
    positive: every node sits at mu != 1 with a zero importance ratio, a
    silently wrong weight.  Only a == 0 is the exact delta at mu = 1."""
    import numpy as np
    import pytest
    from darksirens.likelihood.wl_weight import (
        validate_wl_hermite_quadrature,
        wl_hermite_quadrature_errors,
    )
    from darksirens.lensing.grids import make_hermite_u_grid

    with pytest.raises(ValueError, match="wl_a must be >= 0"):
        validate_wl_hermite_quadrature(-0.004, 1.5)
    u, lw = make_hermite_u_grid()
    z, err = wl_hermite_quadrature_errors(0.0, 1.5, u, lw)
    assert np.all(err == 0.0) and z[0] == 0.01
