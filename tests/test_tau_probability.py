"""One τ definition across every lensing likelihood channel (P1-07).

``tau_2_SIS = A · z^n`` is an optical DEPTH, and the CLI prior box reaches
corners where it exceeds one (``A = 1e-2, n = 6`` gives τ = 156 at z = 5).
Every likelihood channel consumes τ as the Bernoulli probability that the
source is multiply imaged — the unlensed branches carry ``log1p(-τ)`` — but
before this fix only the singleton terms clipped: the pair evidence and the
lensed selection terms used raw ``log(τ)``, so one parameter point carried
two incompatible "probabilities" across channels and the pair channel could
out-compete the (1-τ) suppression without bound.

``slmarks.tau_2_prob`` is now the single definition, and these tests pin
both its numerics and the source-level contract that no likelihood module
takes a raw ``log(tau_2_SIS(...))`` again.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.lensing.slmarks import (
    SISLensParams,
    TAU_PROB_MAX,
    tau_2_SIS,
    tau_2_prob,
)


def _params(A=1e-2, n=6.0):
    return SISLensParams(A_tau=A, n_tau=n, T0=60.0)


def test_prior_corner_exceeds_one_raw_but_is_a_probability_clipped():
    p = _params(A=1e-2, n=6.0)  # the CLI box corner
    z = jnp.asarray([0.5, 1.0, 2.0, 5.0])
    raw = np.asarray(tau_2_SIS(z, p))
    prob = np.asarray(tau_2_prob(z, p))
    assert raw[-1] > 100.0, "the prior corner really does exceed one"
    assert np.all(prob <= TAU_PROB_MAX)
    assert np.all(prob >= 0.0)
    # Below the clip the two agree exactly.
    below = raw < TAU_PROB_MAX
    assert np.allclose(prob[below], raw[below])
    # log1p(-tau_prob) stays finite even at the ceiling.
    assert np.all(np.isfinite(np.log1p(-prob)))


def test_tau_prob_is_zero_at_z_zero():
    p = _params()
    assert float(tau_2_prob(jnp.asarray(0.0), p)) == 0.0


def test_no_likelihood_module_uses_raw_log_tau():
    """Source-level contract: the channels must share ONE τ definition."""
    import inspect

    import darksirens.likelihood.cluster_likelihood as cl
    import darksirens.likelihood.cluster_selection as cs
    import darksirens.likelihood.likelihood_with_clusters as lwc

    for mod in (cl, cs, lwc):
        src = inspect.getsource(mod)
        assert "jnp.log(tau_2_SIS" not in src, (
            f"{mod.__name__} takes log of the UNCLIPPED optical depth; use "
            "slmarks.tau_2_prob so every channel shares one probability"
        )
        assert "jnp.clip(tau_2_SIS" not in src, (
            f"{mod.__name__} hand-rolls its own clip; use slmarks.tau_2_prob"
        )


def test_channels_agree_at_a_super_unity_corner():
    """The pair/selection weight and the singleton suppression must be built
    from the SAME number: log(τ_prob) + log1p(-τ_prob) is only consistent
    when both sides saturate together."""
    p = _params(A=1e-2, n=6.0)
    z = jnp.asarray([5.0])
    tau = float(tau_2_prob(z, p)[0])
    log_tau = float(jnp.log(tau_2_prob(z, p))[0])
    # Saturated: the lensed channel's weight tops out at log(1-1e-12) ~ 0,
    # and the unlensed suppression is finite (not -inf, not NaN).
    assert log_tau <= 0.0
    assert np.isfinite(np.log1p(-tau))
