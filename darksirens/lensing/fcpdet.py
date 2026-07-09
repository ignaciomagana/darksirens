"""
fcpdet.py
---------
JAX Finn & Chernoff (1993) detection-probability model for the
lensed-singleton (exactly-one-detected image) evidence channel.

The mock-lensing generator renders detections with

    rho = 8 Theta (r0 / dL_app) (Mc_det / Mc_bar)^(5/6),   Theta ~ p(Theta),
    p(Theta) = 5 Theta (4 - Theta)^3 / 256   on (0, 4),

so P_det(theta_src, dL_app) = P(Theta > x) with the threshold

    x = rho_thr / [ 8 (r0 / dL_app) (Mc_det / Mc_bar)^(5/6) ].

The survival function has the closed polynomial form

    P(Theta > x) = 1 - (160 x^2 - 80 x^3 + 15 x^4 - x^5) / 256,   x in (0, 4),

which is exactly differentiable — no tables, no interpolation. The
lensed-singleton evidence needs log[1 - P_det(partner image)]: the
probability that the UNDETECTED partner image really was missed. Using the
same analytic model as the generator makes the mock study's censoring factor
exact; for real data this hook is the place to swap in an injection-trained
P_det emulator.

Parameters (rho_thr, r0, mc_bar) are recorded by the generator as
``fc_rho_thr`` / ``fc_r0`` / ``fc_mc_bar`` attrs on the lensed-injection
file; ``r0 = rho_thr * horizon_Mpc / 32`` matches the generator's SNRModel.
"""

from __future__ import annotations

from typing import NamedTuple, Any

import jax.numpy as jnp


class FCPdetParams(NamedTuple):
    """Finn-Chernoff detection-model constants (traced scalars)."""
    rho_thr: Any
    r0: Any
    mc_bar: Any


def make_fc_pdet_params(
    rho_thr: float = 8.0,
    horizon_mpc: float = 3000.0,
    mc_bar: float = 1.22,
    r0: float | None = None,
) -> FCPdetParams:
    """Build FCPdetParams; ``r0`` derived from the horizon unless given."""
    if r0 is None:
        # rho = 8*4*(r0/horizon)*(Mc_bar/Mc_bar)^(5/6) = rho_thr  =>  r0
        r0 = float(rho_thr) * float(horizon_mpc) / 32.0
    return FCPdetParams(
        rho_thr=jnp.asarray(float(rho_thr)),
        r0=jnp.asarray(float(r0)),
        mc_bar=jnp.asarray(float(mc_bar)),
    )


def _chirp_mass_det(m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    m2 = q * m1_src
    mc_src = (m1_src * m2) ** 0.6 / (m1_src + m2) ** 0.2
    return mc_src * (1.0 + z)


def theta_threshold_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray, dL_app: jnp.ndarray,
    params: FCPdetParams,
) -> jnp.ndarray:
    """Threshold x such that the event is detected iff Theta > x."""
    mc = _chirp_mass_det(m1_src, q, z)
    denom = 8.0 * (params.r0 / jnp.maximum(dL_app, 1e-300)) * (
        mc / params.mc_bar
    ) ** (5.0 / 6.0)
    return params.rho_thr / jnp.maximum(denom, 1e-300)


def _theta_cdf(x: jnp.ndarray) -> jnp.ndarray:
    """CDF of Theta: (160 x^2 - 80 x^3 + 15 x^4 - x^5)/256 on (0,4), clipped."""
    xc = jnp.clip(x, 0.0, 4.0)
    cdf = (160.0 * xc**2 - 80.0 * xc**3 + 15.0 * xc**4 - xc**5) / 256.0
    return jnp.clip(cdf, 0.0, 1.0)


def pdet_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray, dL_app: jnp.ndarray,
    params: FCPdetParams,
) -> jnp.ndarray:
    """P_det = P(Theta > x): survival of the Finn-Chernoff Theta distribution."""
    return 1.0 - _theta_cdf(theta_threshold_fc(m1_src, q, z, dL_app, params))


def log_one_minus_pdet_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray, dL_app: jnp.ndarray,
    params: FCPdetParams,
) -> jnp.ndarray:
    """log[1 - P_det] = log CDF(x): the partner-censoring factor.

    -inf where the partner would certainly have been detected (x <= 0 is
    impossible for finite inputs; CDF -> 0 only as x -> 0, i.e. infinitely
    loud) — callers mask with jnp.where per the darksirens -inf convention.
    """
    cdf = _theta_cdf(theta_threshold_fc(m1_src, q, z, dL_app, params))
    return jnp.where(
        cdf > 0.0,
        jnp.log(jnp.maximum(cdf, jnp.finfo(jnp.float64).tiny)),
        -jnp.inf,
    )
