"""
selection.py
------------
NF-based P_det emulator: pseudo-injection generation for the selection
integral.

Scope
-----
This module turns a trained detection-probability emulator — a normalizing
flow over the density of FOUND injections, shipped as ``PDetO4NF.npz`` by the
gw-nf package — into a pseudo-injection set consumed by the untouched
hierarchical selection machinery in ``darksirens.likelihood.selection``:

    mu(Lambda) = (1/N_draw) * sum_j  p_pop(theta_j | Lambda) / p_draw(theta_j)

The emulator checkpoint stores a 13-dimensional flow q(theta) ~ p(theta|det)
for one injection campaign together with ``xi = N_found / N_draw``.  Drawing
M samples theta_j ~ q and setting ``Ndraw = M / xi`` reproduces exactly the
importance structure of real found injections:

    E[mu_hat] = (xi/M) sum_j E_q[p_pop/p_inj] = int p_pop * (xi q / p_inj)
              = int p_pop * p_det = alpha(Lambda),

where p_inj is the campaign's ANALYTIC injection prior (GWTC-4 convention,
vendored below) and ``xi * q / p_inj`` is the emulator's pointwise p_det.
The flow density q(theta_j) itself is never needed — only ``flow.sample``
plus the analytic p_inj — and the downstream Neff bookkeeping stays
meaningful because the estimator has the same form as with real injections.

The generated ``pdraw`` follows the same convention as gwcat selection
exports (``gw/utils.py::load_selection_samples``): the injection density in
the likelihood's canonical basis ``(m1det, q, dL)`` with the 1-D chi_eff
spin-prior swap folded in (``gwcat.spin.chi_eff_prior_logprob`` times the
campaign-vs-reference component-spin ratio).  Uniform angle factors
(ra, sin dec, psi, cos iota, phi1, phi2) are identical in p_inj and the
isotropic population, cancel exactly, and are excluded.  Absolute
normalization therefore differs from an injection-HDF5 export by a
Lambda-independent constant (T_obs, campaign mixture), which cannot move
posteriors.

The flow reconstruction reuses ``darksirens.gw.flows`` (imported lazily so
this module stays importable without the flows extras).  The analytic
GWTC-4 injection-prior factors are adapted from the gw-nf package
(https://github.com/jgassert/gw-nf, MIT license, Julius Gassert):
``src/pdet/gwtc4_model.py`` and ``src/core/gw_utils.py``.

Checkpoint schema note (IMPORTANT): ``PDetO4NF.npz`` is a legacy-schema
gw-nf checkpoint.  Its stored leaves are the BARE spline flow, but the flow
was trained on per-column TRANSFORMED data, recorded in the config's
``transformations`` list — which gw-nf's current ``create_flow_from_config``
ignores (it only rebuilds the newer ``constraints`` schema), so the current
upstream loader reconstructs this checkpoint WITHOUT the physical-coordinate
wrapper and its ``log_prob``/``sample`` are in the wrong space.
``load_pdet_flow`` rebuilds the wrapper from the ``transformations`` names:
``log`` -> exp, ``angle_pi`` -> (0, pi) sigmoid interval, ``angle_2pi`` ->
(0, 2*pi), ``angle_signed_pi`` -> (-pi, pi) — fixed ranges from the NAME,
not from the column's physical range (z, a_i and the cosine columns were
boxed into the nearest angular interval at training time; the flow learned
the tighter physical support).  This reconstruction is validated empirically
against the found-injection HDF5 marginals (all 13 columns + the derived
chi_eff) in scripts/pdet_emulator_validation.py.

x64 note: the emulator checkpoint is float64; ``load_pdet_flow`` asserts
that x64 is enabled (darksirens does so globally in core/jax_config.py).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

# Column order of the 13-dim emulator flow; validated against the checkpoint
# config on load.  theta[:, i] corresponds to PDET_COLUMNS[i].
PDET_COLUMNS = (
    "m1_source",
    "m2_source",
    "redshift",
    "a1",
    "a2",
    "right_ascension",
    "sin_declination",
    "polarization",
    "cos_inclination",
    "cos_tilt1",
    "cos_tilt2",
    "phi1",
    "phi2",
)

# Injection-campaign cosmology (O3b catalog paper constants, used by the
# GWTC-4 injection pipeline and by gw-nf's p_inj).  This is the campaign
# TRUTH — deliberately distinct from darksirens' fiducial and from the
# 67.74/0.3089 override some HDF5 exports were built with.
PDET_INJ_H0 = 67.9
PDET_INJ_OM0 = 0.3065

_SPEED_OF_LIGHT_KM_S = 299792.458

# GWTC-4 primary-mass injection prior: five-segment broken power law,
# continuous across breaks, normalized on [1, 1000] M_sun.
_M1_BREAKS = np.array([1.0, 3.0, 8.0, 50.0, 200.0, 1000.0])
_M1_EXPONENTS = np.array([0.0, -4.0, -1.0, -4.0, -1.0])

# Fraction of flow draws allowed outside the injection-prior support before
# the drop is treated as a red flag (column mis-ordering / reconstruction
# drift).  The expected out-of-support mass is ~1%, dominated by a_i in
# (amax, 1] under p(a) ~ exp(-2 a^2).
_DROP_WARN_FRACTION = 0.05

# Legacy-schema per-column transformations: name -> (low, high) sigmoid
# interval, or "log" -> exp.  The ranges are fixed by the NAME (see module
# docstring); the flow learned any tighter physical support inside them.
_TRANSFORM_INTERVALS = {
    "angle_pi": (0.0, math.pi),
    "angle_2pi": (0.0, 2.0 * math.pi),
    "angle_signed_pi": (-math.pi, math.pi),
}


# ── vendored GWTC-4 injection prior (numpy, host-side) ──────────────────────


def _m1_segments():
    """(low, high, alpha, logh) of the broken power law, continuity anchored.

    logh[i] is the log-density at the LEFT edge of segment i relative to
    logh[0] = 0 at m1 = 1; continuity fixes logh[i+1] = logh[i] +
    alpha[i] * log(high[i]/low[i]).
    """
    low = _M1_BREAKS[:-1]
    high = _M1_BREAKS[1:]
    alpha = _M1_EXPONENTS
    increments = alpha * np.log(high / low)
    logh = np.concatenate([[0.0], np.cumsum(increments[:-1])])
    return low, high, alpha, logh


def _m1_log_norm():
    """log of the analytic normalization of the m1 prior over [1, 1000]."""
    low, high, alpha, logh = _m1_segments()
    height = np.exp(logh)
    # alpha == -1 segments integrate to height*low*log(high/low); the rest to
    # height*low/(alpha+1) * ((high/low)^(alpha+1) - 1).
    is_log = alpha == -1.0
    ap1 = np.where(is_log, 1.0, alpha + 1.0)  # dummy 1.0 avoids 0-division
    area_pow = height * low / ap1 * ((high / low) ** ap1 - 1.0)
    area_log = height * low * np.log(high / low)
    areas = np.where(is_log, area_log, area_pow)
    return float(np.log(np.sum(areas)))


_M1_LOG_NORM = _m1_log_norm()


def _gwtc4_log_p_m1(m1: np.ndarray) -> np.ndarray:
    """Normalized log density of the 5-segment broken-power-law m1 prior.

    Support is [1, 1000] (density 0 exactly at 1000, matching gw-nf's
    half-open segment assignment); -inf outside.
    """
    m1 = np.asarray(m1, dtype=float)
    low, high, alpha, logh = _m1_segments()
    out = np.full(m1.shape, -np.inf)
    for lo, hi, a, lh in zip(low, high, alpha, logh):
        in_seg = (m1 >= lo) & (m1 < hi)
        if np.any(in_seg):
            out[in_seg] = lh + a * (np.log(m1[in_seg]) - np.log(lo))
    return out - _M1_LOG_NORM


def _gwtc4_log_p_m2_given_m1(
    m2: np.ndarray, m1: np.ndarray, min_m2: float = 1.0
) -> np.ndarray:
    """log p(m2 | m1) = log[ 2 m2 / (m1^2 - min_m2^2) ] on [min_m2, m1]."""
    m2 = np.asarray(m2, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    m2, m1 = np.broadcast_arrays(m2, m1)
    denom = m1 * m1 - min_m2 * min_m2
    ok = (m2 >= min_m2) & (m2 <= m1) & (denom > 0)
    out = np.full(m2.shape, -np.inf)
    out[ok] = np.log(2.0 * m2[ok]) - np.log(denom[ok])
    return out


# Z_a = int_0^1 exp(-2 a^2) da = sqrt(pi)/(2 sqrt(2)) * erf(sqrt(2))
_SPIN_MAG_LOG_NORM = math.log(
    math.sqrt(math.pi) / (2.0 * math.sqrt(2.0)) * math.erf(math.sqrt(2.0))
)


def _gwtc4_log_p_spin_mag(a: np.ndarray) -> np.ndarray:
    """log p(a) = -2 a^2 - log Z_a on [0, 1]; -inf outside."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, -np.inf)
    ok = (a >= 0.0) & (a <= 1.0)
    out[ok] = -2.0 * a[ok] * a[ok] - _SPIN_MAG_LOG_NORM
    return out


def _gwtc4_log_p_cos_tilt(c: np.ndarray) -> np.ndarray:
    """log p(cos tau) = log[ 0.3 (1 + cos tau)^3 / 4 + 0.35 ] on [-1, 1]."""
    c = np.asarray(c, dtype=float)
    out = np.full(c.shape, -np.inf)
    ok = (c >= -1.0) & (c <= 1.0)
    out[ok] = np.log(0.3 * (1.0 + c[ok]) ** 3 / 4.0 + 0.35)
    return out


def build_injection_z_prior(
    H0: float = PDET_INJ_H0,
    Om0: float = PDET_INJ_OM0,
    z_max: float = 3.0,
    n: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the campaign redshift prior p(z) ~ dVc/dz on [0, z_max].

    Trapezoid-normalized on the grid, matching gw-nf's
    ``get_redshift_grid`` + ``redshift_pdf_jax`` (n = 1e6 points, NO extra
    (1+z) factor).  Uses astropy directly — the campaign cosmology, not
    darksirens' interpolated fiducial helpers.
    """
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    z_grid = np.linspace(0.0, z_max, n)
    dVdz = (
        cosmo.differential_comoving_volume(z_grid).to_value(u.Gpc**3 / u.sr)
        * 4.0
        * np.pi
    )
    pdf_grid = dVdz / np.trapezoid(dVdz, z_grid)
    return z_grid, pdf_grid


def _gwtc4_log_p_z(
    z: np.ndarray, z_grid: np.ndarray, pdf_grid: np.ndarray
) -> np.ndarray:
    """log of the interpolated redshift prior; -inf outside [0, z_max]."""
    p = np.interp(np.asarray(z, dtype=float), z_grid, pdf_grid, left=0.0, right=0.0)
    with np.errstate(divide="ignore"):
        return np.log(p)


# ── emulator checkpoint loading ─────────────────────────────────────────────


def _build_transformations_bijection(transformations, data_dim: int):
    """R^13 -> physical bijection from the legacy ``transformations`` names."""
    import flowjax.bijections as bij

    if len(transformations) != data_dim:
        raise ValueError(
            f"'transformations' has {len(transformations)} entries for "
            f"data_dim={data_dim}."
        )
    indexed = []
    for dim, name in enumerate(transformations):
        if name == "log":
            indexed.append(bij.Indexed(bij.Exp(), dim, (data_dim,)))
        elif name in _TRANSFORM_INTERVALS:
            lo, hi = _TRANSFORM_INTERVALS[name]
            b = bij.Chain([bij.Sigmoid(), bij.Affine(loc=lo, scale=hi - lo)])
            indexed.append(bij.Indexed(b, dim, (data_dim,)))
        else:
            raise ValueError(
                f"Unknown transformation {name!r} for column {dim}; expected "
                f"one of {['log', *_TRANSFORM_INTERVALS]}."
            )
    return bij.Chain(indexed)


def load_pdet_flow(path: str | Path):
    """Load and validate a P_det emulator checkpoint.

    Returns ``(flow, config)`` with ``flow.sample``/``flow.log_prob`` in
    PHYSICAL coordinates: the stored bare spline flow is re-wrapped with the
    per-column transform rebuilt from the config's legacy ``transformations``
    list (see module docstring).  The checkpoint must be a 13-dim spline flow
    with the PDET_COLUMNS layout and an ``xi`` detection-efficiency key.
    The structural (shape, dtype) check against the skeleton rebuilt with the
    installed flowjax is a hard error with no skip/load escape: this is a
    single mandatory checkpoint, and positional leaf assignment would
    otherwise silently corrupt it.
    """
    import jax
    import paramax
    from flowjax.distributions import Transformed
    from darksirens.gw.flows import (
        check_checkpoint_matches_skeleton,
        load_flow,
    )

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "The P_det emulator checkpoint requires x64; call "
            "configure_jax_runtime() (or jax.config.update("
            "'jax_enable_x64', True)) before loading."
        )

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        config = json.loads(str(data["config_json"]))

    cols = tuple(config.get("columns", ()))
    if cols != PDET_COLUMNS:
        raise ValueError(
            f"P_det emulator checkpoint '{path}' has columns {list(cols)}; "
            f"expected the 13-dim layout {list(PDET_COLUMNS)}."
        )
    if int(config.get("data_dim", -1)) != len(PDET_COLUMNS):
        raise ValueError(
            f"P_det emulator checkpoint '{path}' has data_dim="
            f"{config.get('data_dim')!r}; expected {len(PDET_COLUMNS)}."
        )
    if "xi" not in config:
        raise ValueError(
            f"P_det emulator checkpoint '{path}' is missing the 'xi' "
            "(N_found/N_draw detection efficiency) config key; without it "
            "Ndraw cannot be reconstructed."
        )
    xi = float(config["xi"])
    if not (0.0 < xi < 1.0):
        raise ValueError(
            f"P_det emulator checkpoint '{path}' has xi={xi!r}; expected a "
            "detection efficiency in (0, 1)."
        )
    if "constraints" in config or "Z_mean" in config:
        raise NotImplementedError(
            f"P_det emulator checkpoint '{path}' uses the newer gw-nf "
            "'constraints'/'Z_mean' schema; only the legacy 'transformations' "
            "schema of the shipped PDetO4NF.npz is wired here. Extend "
            "load_pdet_flow before using this checkpoint."
        )
    if "transformations" not in config:
        raise ValueError(
            f"P_det emulator checkpoint '{path}' has no 'transformations' "
            "list; cannot reconstruct the physical-coordinate wrapper."
        )
    transform = _build_transformations_bijection(
        config["transformations"], len(PDET_COLUMNS)
    )

    check_checkpoint_matches_skeleton(path)
    flow, config = load_flow(path)
    flow = Transformed(flow, paramax.non_trainable(transform))
    return flow, config


def sample_pdet_flow(flow, nsamp: int, seed: int, batch: int = 65536) -> np.ndarray:
    """Draw ``nsamp`` samples from the emulator flow, host-side, in batches.

    Returns a float64 numpy array of shape (nsamp, 13) in PDET_COLUMNS order.
    """
    import jax

    if nsamp <= 0:
        raise ValueError(f"nsamp must be positive; got {nsamp}.")
    key = jax.random.key(int(seed))
    out = []
    remaining = int(nsamp)
    while remaining > 0:
        n = min(int(batch), remaining)
        key, sub = jax.random.split(key)
        out.append(np.asarray(flow.sample(sub, (n,))))
        remaining -= n
    return np.concatenate(out, axis=0)


# ── pseudo-injection generation ─────────────────────────────────────────────


def pseudo_injections_from_pdet_flow(
    path: str | Path,
    *,
    nsamp: int,
    seed: int,
    H0: float = PDET_INJ_H0,
    Om0: float = PDET_INJ_OM0,
    chieff_amax: float = 0.99,
    batch: int = 65536,
):
    """Generate a pseudo-injection selection set from a P_det emulator.

    Draws ``nsamp`` samples from the found-injection flow, converts them to
    the likelihood's detector-frame convention under the injection-campaign
    cosmology ``(H0, Om0)``, and computes ``pdraw`` — the analytic campaign
    injection prior expressed in the canonical ``(m1det, q, dL)`` + chi_eff
    basis (see module docstring).  Samples outside the injection prior's
    support are dropped entirely (the population density is zero there, so a
    dropped detected sample is exactly a zero-weight one); ``ndraw = M / xi``
    is unchanged by drops.

    Returns the same 8-tuple as ``gw.utils.load_selection_samples``:
    ``(m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw)`` with jnp arrays and
    ``ndraw`` a python float.
    """
    try:
        from gwcat.spin import chi_eff_prior_logprob
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "gwcat is required to build the chi_eff spin-prior swap for the "
            "P_det emulator selection path; install the gwcat package."
        ) from e
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    import jax.numpy as jnp

    if not (0.0 < chieff_amax <= 1.0):
        raise ValueError(f"chieff_amax must be in (0, 1]; got {chieff_amax}.")

    flow, config = load_pdet_flow(path)
    xi = float(config["xi"])
    M = int(nsamp)
    theta = sample_pdet_flow(flow, M, seed, batch=batch)

    c = {name: theta[:, i] for i, name in enumerate(PDET_COLUMNS)}
    m1src, m2src, z = c["m1_source"], c["m2_source"], c["redshift"]
    a1, a2 = c["a1"], c["a2"]
    ra, sindec = c["right_ascension"], c["sin_declination"]
    psi, cosinc = c["polarization"], c["cos_inclination"]
    ct1, ct2 = c["cos_tilt1"], c["cos_tilt2"]
    phi1, phi2 = c["phi1"], c["phi2"]

    z_grid, z_pdf = build_injection_z_prior(H0=H0, Om0=Om0)
    z_max = float(z_grid[-1])

    # Support mask: outside any of these the campaign injection prior — and
    # with it the population density in the swapped-spin convention — is
    # exactly zero, so dropping is exact, not an approximation.  The a_i cut
    # uses the REFERENCE amax (0.99), not the campaign's 1.0: the chi_eff
    # population is defined against the amax-truncated reference conditional,
    # which has zero mass above amax.  The spline flow has full-R^13 support,
    # so a small leakage outside every box is expected.
    causes = {
        "m1_source outside [1, 1000]": (m1src >= 1.0) & (m1src <= 1000.0),
        "m2_source outside [1, m1_source]": (m2src >= 1.0) & (m2src <= m1src),
        f"redshift outside [0, {z_max:g}]": (z >= 0.0) & (z <= z_max),
        f"a_i outside [0, {chieff_amax:g}]": (
            (a1 >= 0.0) & (a1 <= chieff_amax) & (a2 >= 0.0) & (a2 <= chieff_amax)
        ),
        "|cos_tilt_i| > 1": (np.abs(ct1) <= 1.0) & (np.abs(ct2) <= 1.0),
        "angle outside box": (
            (ra >= 0.0) & (ra <= 2.0 * np.pi)
            & (np.abs(sindec) <= 1.0)
            & (psi >= 0.0) & (psi <= np.pi)
            & (np.abs(cosinc) <= 1.0)
            & (phi1 >= 0.0) & (phi1 <= 2.0 * np.pi)
            & (phi2 >= 0.0) & (phi2 <= 2.0 * np.pi)
        ),
    }
    keep = np.ones(M, dtype=bool)
    for mask in causes.values():
        keep &= mask
    n_kept = int(keep.sum())
    drop_frac = 1.0 - n_kept / M
    if n_kept == 0:
        raise RuntimeError(
            "Every emulator draw fell outside the injection-prior support; "
            "the checkpoint columns are almost certainly mis-mapped."
        )
    if drop_frac > _DROP_WARN_FRACTION:
        warnings.warn(
            f"P_det emulator: {drop_frac:.1%} of flow draws fell outside the "
            f"injection-prior support (expected ~1%, mostly a_i > "
            f"{chieff_amax:g}). This can indicate column mis-ordering or "
            "flow-reconstruction drift.",
            RuntimeWarning,
        )

    m1src, m2src, z = m1src[keep], m2src[keep], z[keep]
    a1, a2, ct1, ct2 = a1[keep], a2[keep], ct1[keep], ct2[keep]
    ra, sindec = ra[keep], sindec[keep]

    # Detector-frame conversion under the campaign cosmology (astropy,
    # host-side, once — exact, no interpolation).
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    DC = cosmo.comoving_distance(z).to_value(u.Mpc)
    dL = (1.0 + z) * DC
    E = np.sqrt(Om0 * (1.0 + z) ** 3 + (1.0 - Om0))
    dLprime = DC + (1.0 + z) * (_SPEED_OF_LIGHT_KM_S / H0) / E
    m1det = m1src * (1.0 + z)
    m2det = m2src * (1.0 + z)
    q = m2src / m1src
    chieff = (a1 * ct1 + q * a2 * ct2) / (1.0 + q)
    dec = np.arcsin(sindec)  # safe: |sindec| <= 1 enforced by the mask

    # log pdraw in the canonical (m1det, q, dL) + swapped-chi_eff basis:
    #   p_inj(m1src, m2src, z) x Jacobian  x  pi_ref(chieff | q; amax)
    #   x  [campaign 4-D spin density / amax-reference 4-D spin density].
    # Jacobian of (m1src, m2src, z) -> (m1det, q, dL):
    #   |det| = (1+z) * dL'(z) / m1src, and densities divide by it.
    # Uniform angle factors cancel against the isotropic population exactly
    # and are excluded (constant offsets cannot move posteriors).
    log_p_chi = chi_eff_prior_logprob(chieff, m1src, m2src, amax=chieff_amax)
    # Mirror the gwcat-export convention (gw/utils.py): floor the table
    # interpolation at -50 so pdraw stays strictly positive at the
    # |chieff| -> amax table edges.
    log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
    log_pdraw = (
        _gwtc4_log_p_m1(m1src)
        + _gwtc4_log_p_m2_given_m1(m2src, m1src)
        + _gwtc4_log_p_z(z, z_grid, z_pdf)
        + np.log(m1src) - np.log1p(z) - np.log(dLprime)
        + log_p_chi
        + _gwtc4_log_p_spin_mag(a1) + _gwtc4_log_p_spin_mag(a2)
        + _gwtc4_log_p_cos_tilt(ct1) + _gwtc4_log_p_cos_tilt(ct2)
        + 2.0 * (np.log(chieff_amax) + np.log(2.0))  # / [(1/amax)(1/2)]^2
    )
    pdraw = np.exp(log_pdraw)

    ndraw = float(M) / xi

    name = config.get("flow_name", Path(path).stem)
    print(
        f"    [pdet emulator] {name}  drew {M:,} samples, kept {n_kept:,} "
        f"({drop_frac:.2%} outside support)  xi={xi:.6g}  Ndraw={ndraw:,.1f}"
    )
    for cause, mask in causes.items():
        n_fail = int((~mask).sum())
        if n_fail:
            print(f"      dropped {n_fail:,}: {cause}")
    print(
        f"    p_draw: min={pdraw.min():.3e}  max={pdraw.max():.3e}  "
        f"mean={pdraw.mean():.3e}"
    )
    print(
        f"    campaign cosmology H0={H0}  Om0={Om0}  "
        f"chieff_amax={chieff_amax}  seed={seed}"
    )

    return (
        jnp.array(m1det),
        jnp.array(m2det),
        jnp.array(dL),
        jnp.array(chieff),
        jnp.array(ra),
        jnp.array(dec),
        jnp.array(pdraw),
        ndraw,
    )
