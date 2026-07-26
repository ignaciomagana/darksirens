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

Orientation independence: a DOCUMENTED LIMITATION, not a derived result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``log_one_minus_pdet_fc`` returns the Theta-MARGINALISED censoring probability
at the partner image's threshold, i.e. it treats the partner's orientation
factor as an INDEPENDENT draw from the detected image's. The two images of one
source share their inclination and polarization, so this is only valid for a
detection model (and an injection campaign) that re-randomises the orientation
per image -- which is exactly what ``scripts/mock_lensing/generate_mock_lensing.py``
does (two independent uniform draws per source, both for the singleton/double
split and for the lensed-injection campaign). The mock study is therefore
self-consistent, and ``tests/test_lensed_singleton_channel.py`` validates the
analytic factor against a direct Monte Carlo of that rendering.

It is NOT valid for a physically rendered campaign. With ONE orientation per
source and thresholds x_+ < x_- (the fainter image needs a larger Theta),

    P(both detected)     = S(x_-)                (not S(x_+) S(x_-))
    P(exactly image j)   = S(x_j) - S(x_partner) (not S(x_j)[1 - S(x_partner)])

Measured against a 4e6-sample Monte Carlo of both conventions, at
(x_+, x_-) = (1, 2) the shared-Theta P(both) is 1.58x the independent one and
P(exactly one) is 0.76x; at (1.5, 3) the ratios are 2.62x and 0.95x; at
(0.8, 1.0) they are 1.36x and 0.24x. Since the J=2 / lensed-singleton ratio is
what sets A_tau, swapping in a physically rendered campaign WITHOUT also
replacing this censoring factor mis-normalises that channel ratio by up to ~2x.

The correct model is neither of the two limits above. Finn-Chernoff Theta
bundles the inclination and polarization (shared between the images) with the
antenna response F_+(alpha, delta, t), F_x(alpha, delta, t), which at the SIS
delay scale (T_0 ~ 62 d, so delays of days to months) samples an essentially
independent Earth-rotation phase. Theta_+ and Theta_- are therefore PARTIALLY
correlated, and getting P(both) right needs the joint p(Theta_+, Theta_-) from
that decomposition plus a campaign rendered with one (iota, psi, alpha, delta)
per source and two arrival times. That is a new model component and is
deliberately not attempted here.
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

    INDEPENDENT-ORIENTATION assumption: this marginalises the partner's Theta
    independently of the detected image's, which is correct only for a campaign
    that re-randomises orientation per image (the mock generator does). See the
    module docstring for the shared-orientation forms, the measured discrepancy
    (up to ~2x on the J=2 / lensed-singleton ratio), and why the physically
    correct model is a partially-correlated one that is not implemented here.
    """
    cdf = _theta_cdf(theta_threshold_fc(m1_src, q, z, dL_app, params))
    return jnp.where(
        cdf > 0.0,
        jnp.log(jnp.maximum(cdf, jnp.finfo(jnp.float64).tiny)),
        -jnp.inf,
    )
