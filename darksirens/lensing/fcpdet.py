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

Pair-orientation modes: 'independent' (default) and 'shared_iota'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``log_one_minus_pdet_fc`` returns the Theta-MARGINALISED censoring probability
at the partner image's threshold, i.e. it treats the partner's orientation
factor as an INDEPENDENT draw from the detected image's. The two images of one
source share their inclination and polarization, so this is only valid for a
detection model (and an injection campaign) that re-randomises the orientation
per image -- which is what ``scripts/mock_lensing/generate_mock_lensing.py``
renders by DEFAULT (two independent uniform draws per source, both for the
singleton/double split and for the lensed-injection campaign). The mock study
is therefore self-consistent, and ``tests/test_lensed_singleton_channel.py``
validates the analytic factor against a direct Monte Carlo of that rendering.

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

The correct model is neither of the two limits above, because Finn-Chernoff
Theta bundles the SHARED inclination/polarization with the antenna response
F_+(alpha, delta, psi, t), F_x(alpha, delta, psi, t), which at the SIS delay
scale (T_0 ~ 62 d, so delays of days to months) samples an essentially
independent Earth-rotation phase. That partially-correlated joint model is
implemented here as ``pair_orientation_mode='shared_iota'``; see the section
below. The default everywhere remains 'independent' so that existing
independently-rendered mocks stay self-consistent — switching modes is a
deliberate, campaign-matched change (generate the mock AND run the inference
with the same mode).

The 'shared_iota' joint two-image model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convention (this module's own, from the rho formula above): Theta in (0, 4],
with the Finn & Chernoff (1993, eq. 3.31) geometric decomposition

    Theta^2 = 4 [ F_+^2 (1 + c^2)^2 + 4 F_x^2 c^2 ],      c = cos(iota),

where (F_+, F_x) are the single-detector antenna factors (|F| <= 1,
Theta = 4 at overhead/face-on, matching r0 = rho_thr*horizon/32 above), and
the marginal of Theta over isotropic (sky, psi, cos iota) is the fit
p(Theta) = 5 Theta (4 - Theta)^3 / 256 used by ``_theta_cdf``. NOTE the fit
is only ~few-percent accurate; in particular the exact geometric marginal has
S(3) ~ 0.038 vs the fit's 0.016. All 'shared_iota' quantities therefore use
the exact geometric distribution self-consistently (never mixing it with the
polynomial), while 'independent'-mode quantities keep the polynomial exactly.

The two images of one source share (iota, psi, alpha, delta); their antenna
factors are evaluated at arrival times separated by the pair's time delay.
For |dt| >> hours the Earth-rotation phase decorrelates the antenna response,
so the model treats (F_+, F_x) of the two images as INDEPENDENT draws of the
isotropically sky/psi-marginalised antenna distribution CONDITIONED on the
shared inclination: Theta_1 _|_ Theta_2 | iota (NOT unconditionally
independent). Pair quantities are then 1-D quadratures over cos iota of
products of conditional survivals S(x | c) = P(Theta > x | cos iota = c):

    P(both)           = int_0^1 dc S(x_+|c) S(x_-|c)
    P(exactly j)      = int_0^1 dc S(x_j|c) [1 - S(x_partner|c)]
    P(miss p | det d) = int dc S(x_d|c)[1 - S(x_p|c)] / int dc S(x_d|c)

S(x|c) reduces analytically: with psi uniform, (F_+, F_x) = F (cos z, sin z),
z ~ U(0, 2pi), F^2 = u^2 + (1-u^2)^2 t / 4 (u = detector-frame cos theta ~
U(0,1) by symmetry, t = cos^2 2phi ~ arcsine on (0,1)). The z-integral is
exact (an arccos), and integrating by parts in the arccos argument turns the
remaining average into a Gauss-Chebyshev sum over lookups of Q(f) = P(F <= f),
itself a smooth 1-D quadrature. The tables (built lazily, cached, ~1 s) enter
jit traces as constants; interpolation is piecewise-linear in x on a uniform
[0, 4] grid, so gradients are finite everywhere (piecewise-constant in x).

Validity domain: the conditional-independence step needs |dt| >> hours; under
the SIS delay law dt = T0*y with T0 ~ 62 d, P(dt < 1 day) ~ 3e-4 of pairs, so
the code applies the same dt-independent model to all pairs (short-delay pairs
are modestly over-decorrelated). It also drops the residual correlation from
the shared DECLINATION at a fixed-latitude detector (both images sample the
same declination-conditioned antenna envelope): measured against a real
two-arrival-time antenna Monte Carlo this residual is <~1.6% on P(both) and
<~6% on P(exactly one) at near-equal thresholds — an order of magnitude below
the 36-405% iota-correlation effect the mode does capture.
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
    that re-randomises orientation per image (the mock generator's default).
    See the module docstring for the measured discrepancy (up to ~2x on the
    J=2 / lensed-singleton ratio) and for the partially-correlated
    'shared_iota' alternative, ``log_pmiss_partner_fc``.
    """
    cdf = _theta_cdf(theta_threshold_fc(m1_src, q, z, dL_app, params))
    return jnp.where(
        cdf > 0.0,
        jnp.log(jnp.maximum(cdf, jnp.finfo(jnp.float64).tiny)),
        -jnp.inf,
    )


# ============================================================================
# JOINT-orientation ('shared_iota') two-image detection model
# ============================================================================

PAIR_ORIENTATION_MODES = ("independent", "shared_iota")


def _require_pair_orientation_mode(mode: str) -> None:
    if mode not in PAIR_ORIENTATION_MODES:
        raise ValueError(
            f"unknown pair_orientation_mode {mode!r}; "
            f"expected one of {PAIR_ORIENTATION_MODES}"
        )


def theta_fc_from_antenna(f_plus, f_cross, cos_iota):
    """Finn-Chernoff Theta from antenna factors + inclination (FC93 eq. 3.31).

    Theta^2 = 4 [F_+^2 (1 + cos^2 iota)^2 + 4 F_x^2 cos^2 iota], Theta in
    (0, 4] for |F| <= 1 — THE convention this module's rho formula uses
    (rho = 8 Theta (r0/dL)(Mc/Mcbar)^(5/6), optimal at Theta = 4). The single
    source of truth for the decomposition: the mock generator's shared-
    orientation rendering and the shared_iota tables both derive from it.
    """
    c2 = jnp.square(cos_iota)
    return 2.0 * jnp.sqrt(
        jnp.square(f_plus) * jnp.square(1.0 + c2)
        + 4.0 * jnp.square(f_cross) * c2
    )


class _SharedIotaTables(NamedTuple):
    """Conditional-survival tables at Gauss-Legendre cos-iota nodes."""
    c_weights: Any      # (n_c,) GL weights on cos iota in (0,1), sum to 1
    x_grid_max: Any     # scalar: upper edge of the uniform x grid (= 4.0)
    inv_dx: Any         # scalar: 1 / x-grid spacing
    survival: Any       # (n_c, n_x): S(x_grid[j] | c_i)


_SHARED_IOTA_TABLES: _SharedIotaTables | None = None


def _build_shared_iota_tables(
    n_c: int = 64, n_x: int = 1025, n_cheb: int = 384,
    n_f: int = 8193, n_u: int = 2049,
) -> _SharedIotaTables:
    """Tabulate S(x | c) = P(Theta > x | cos iota = c) on GL cos-iota nodes.

    Numpy, run once (lazily) and cached; the arrays enter jit traces as
    constants. See the module docstring for the reduction: the psi average
    makes (F_+, F_x) = F (cos z, sin z) with F^2 = u^2 + (1-u^2)^2 t/4; the
    z-integral is exact and an integration by parts turns S(x|c) into
    1 - mean_j Q(x / (2 sqrt(B^2 + (A^2 - B^2) tau_j))) over Chebyshev nodes
    tau_j, with A = 1 + c^2, B = 2c and Q(f) = P(F <= f). Verified against
    conditioned Monte Carlo to <= 4e-4 absolute per (c, x) and against pair-
    probability Monte Carlo to <= 0.25% relative (dominated by MC noise).
    """
    import numpy as np

    # --- Q(f): CDF of F = sqrt(F_+^2 + F_x^2) over the isotropic sky ---
    f_grid = np.linspace(0.0, 1.0, n_f)
    xu, wu = np.polynomial.legendre.leggauss(n_u)
    u = 0.5 * (xu + 1.0)                    # cos(theta) folded onto (0,1)
    wu = 0.5 * wu
    denom = np.maximum(0.25 * (1.0 - u**2) ** 2, 1e-300)
    Q = np.empty(n_f)
    for lo in range(0, n_f, 512):
        fs = f_grid[lo:lo + 512][:, None]
        arg = (fs**2 - u[None, :] ** 2) / denom[None, :]
        Pt = (2.0 / np.pi) * np.arcsin(np.sqrt(np.clip(arg, 0.0, 1.0)))
        Q[lo:lo + 512] = Pt @ wu

    # --- S(x | c) on GL cos-iota nodes (density 1 on (0,1) by symmetry) ---
    xc, wc = np.polynomial.legendre.leggauss(n_c)
    c = 0.5 * (xc + 1.0)
    c_weights = 0.5 * wc
    x_grid = np.linspace(0.0, 4.0, n_x)
    j = np.arange(1, n_cheb + 1)
    tau = 0.5 * (1.0 + np.cos((2 * j - 1) * np.pi / (2 * n_cheb)))
    survival = np.empty((n_c, n_x))
    for i, ci in enumerate(c):
        A2 = (1.0 + ci**2) ** 2
        B2 = 4.0 * ci**2
        Fj = x_grid[:, None] / (2.0 * np.sqrt(B2 + (A2 - B2) * tau[None, :]))
        Qv = np.interp(Fj, f_grid, Q, left=0.0, right=1.0)
        survival[i] = 1.0 - Qv.mean(axis=1)
    survival = np.clip(survival, 0.0, 1.0)
    survival[:, -1] = 0.0                    # exact: Theta <= 4 always

    return _SharedIotaTables(
        c_weights=jnp.asarray(c_weights),
        x_grid_max=jnp.asarray(float(x_grid[-1])),
        inv_dx=jnp.asarray((n_x - 1) / float(x_grid[-1])),
        survival=jnp.asarray(survival),
    )


def _shared_iota_tables() -> _SharedIotaTables:
    global _SHARED_IOTA_TABLES
    if _SHARED_IOTA_TABLES is None:
        _SHARED_IOTA_TABLES = _build_shared_iota_tables()
    return _SHARED_IOTA_TABLES


def _theta_survival_given_cos_iota(x: jnp.ndarray, tab: _SharedIotaTables) -> jnp.ndarray:
    """S(x | c_i) at every cos-iota node: shape (n_c,) + x.shape.

    Piecewise-linear in x on the uniform grid; x outside [0, 4] clips to the
    exact limits S = 1 / S = 0.
    """
    n_x = tab.survival.shape[1]
    xi = jnp.clip(x, 0.0, tab.x_grid_max) * tab.inv_dx
    i0 = jnp.clip(jnp.floor(xi).astype(jnp.int32), 0, n_x - 2)
    w = xi - i0
    s0 = tab.survival[:, i0]
    s1 = tab.survival[:, i0 + 1]
    return jnp.clip(s0 + (s1 - s0) * w, 0.0, 1.0)


def _pair_probs_shared_iota(x_a: jnp.ndarray, x_b: jnp.ndarray):
    """(P(both), P(only a), P(only b), P(det a)) under Theta_a _|_ Theta_b | iota.

    x_a, x_b broadcast; each result has the broadcast shape.
    """
    tab = _shared_iota_tables()
    Sa = _theta_survival_given_cos_iota(x_a, tab)
    Sb = _theta_survival_given_cos_iota(x_b, tab)
    w = tab.c_weights
    contract = lambda t: jnp.tensordot(w, t, axes=(0, 0))
    p_both = contract(Sa * Sb)
    p_only_a = contract(Sa * (1.0 - Sb))
    p_only_b = contract((1.0 - Sa) * Sb)
    p_det_a = contract(Sa * jnp.ones_like(Sb))
    return p_both, p_only_a, p_only_b, p_det_a


def pdet_pair_both_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray,
    dL_app_plus: jnp.ndarray, dL_app_minus: jnp.ndarray,
    params: FCPdetParams,
    pair_orientation_mode: str = "independent",
) -> jnp.ndarray:
    """P(both images detected) for one source seen at two apparent distances.

    'independent': the pinned product S(x_+) S(x_-) of the polynomial
    marginal (bit-identical to pdet_fc * pdet_fc). 'shared_iota': the joint-
    orientation quadrature int dc S(x_+|c) S(x_-|c) of the exact geometric
    Theta distribution (module docstring).
    """
    _require_pair_orientation_mode(pair_orientation_mode)
    if pair_orientation_mode == "independent":
        return (
            pdet_fc(m1_src, q, z, dL_app_plus, params)
            * pdet_fc(m1_src, q, z, dL_app_minus, params)
        )
    x_p = theta_threshold_fc(m1_src, q, z, dL_app_plus, params)
    x_m = theta_threshold_fc(m1_src, q, z, dL_app_minus, params)
    p_both, _, _, _ = _pair_probs_shared_iota(x_p, x_m)
    return p_both


def pdet_pair_exactly_one_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray,
    dL_app_plus: jnp.ndarray, dL_app_minus: jnp.ndarray,
    params: FCPdetParams,
    pair_orientation_mode: str = "independent",
) -> jnp.ndarray:
    """P(exactly one image detected), summed over the two image identities."""
    _require_pair_orientation_mode(pair_orientation_mode)
    if pair_orientation_mode == "independent":
        s_p = pdet_fc(m1_src, q, z, dL_app_plus, params)
        s_m = pdet_fc(m1_src, q, z, dL_app_minus, params)
        return s_p * (1.0 - s_m) + (1.0 - s_p) * s_m
    x_p = theta_threshold_fc(m1_src, q, z, dL_app_plus, params)
    x_m = theta_threshold_fc(m1_src, q, z, dL_app_minus, params)
    _, p_only_p, p_only_m, _ = _pair_probs_shared_iota(x_p, x_m)
    return p_only_p + p_only_m


def log_pmiss_partner_fc(
    m1_src: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray,
    dL_app_det: jnp.ndarray, dL_app_partner: jnp.ndarray,
    params: FCPdetParams,
    pair_orientation_mode: str = "independent",
) -> jnp.ndarray:
    """log P(partner image missed | this image detected): the pair-aware
    censoring factor of the lensed-singleton channel.

    'independent': EXACTLY ``log_one_minus_pdet_fc`` at the partner's
    apparent distance (dL_app_det unused) — the detected image carries no
    information about the partner's orientation draw.

    'shared_iota': the conditional
        log[ int dc S(x_det|c) (1 - S(x_par|c)) / int dc S(x_det|c) ],
    where the numerator/denominator conditioning enters only through the
    shared cos-iota posterior (the detected image's own antenna draw is
    independent of the partner's; per-event terms carry no P_det factor per
    the MFG convention, so the detection EVENT is all that is conditioned on).

    -inf where the partner would certainly have been detected (numerator 0
    with positive denominator) — callers mask with jnp.where per the
    darksirens -inf convention. Where the geometric model gives the DETECTED
    image zero detection probability (x_det >= its conditional support, an
    unphysical corner the integrand can still visit at extreme (theta, y)),
    the conditional is undefined and the factor falls back to the marginal
    'independent' value there, keeping the mode's integrand support identical
    to 'independent' (documented approximation; the PE-KDE term suppresses
    those corners).
    """
    _require_pair_orientation_mode(pair_orientation_mode)
    if pair_orientation_mode == "independent":
        return log_one_minus_pdet_fc(m1_src, q, z, dL_app_partner, params)
    x_det = theta_threshold_fc(m1_src, q, z, dL_app_det, params)
    x_par = theta_threshold_fc(m1_src, q, z, dL_app_partner, params)
    _, num, _, den = _pair_probs_shared_iota(x_det, x_par)
    tiny = jnp.finfo(jnp.float64).tiny
    log_cond = jnp.log(jnp.maximum(num, tiny)) - jnp.log(jnp.maximum(den, tiny))
    out = jnp.where(num > 0.0, log_cond, -jnp.inf)
    return jnp.where(
        den > 0.0,
        out,
        log_one_minus_pdet_fc(m1_src, q, z, dL_app_partner, params)
        + jnp.zeros_like(num),
    )
