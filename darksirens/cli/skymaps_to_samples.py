from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

try:
    from ligo.skymap.io import read_sky_map
except Exception:  # pragma: no cover - optional dependency
    read_sky_map = None


# ============================================================================
# Architecture A: 3D skymap -> importance samples for the EXISTING likelihood
# ============================================================================
#
# The darksirens per-event term is
#
#     ln L_i = ln (1/n) Σ_j  p_pop(θ_j) p_z(z_j|pix_j) / [ J_j · p_pe(θ_j) ]
#
# with samples θ_j drawn from the GW *posterior* and p_pe the GW *prior*.
# There is no explicit GW-likelihood factor in the target: the GW information
# enters ONLY through (a) samples ~ posterior and (b) dividing by the prior.
#
# This converter therefore must satisfy two non-negotiable conditions:
#
#   (1) The (pixel, dL) samples must follow the TRUE skymap posterior, i.e.
#       pix ~ PROB  and  dL ~ DISTNORM·dL²·N(dL; DISTMU, DISTSIGMA)  (dL ≥ 0).
#       The dL² factor (Singer et al. 2016, ApJL 829 L15; ApJS 226 10) is the
#       conditional-distance ansatz and must be present when sampling.
#
#   (2) p_pe must be the PE PRIOR, NOT the sampling density:
#         - sky distance : π(dL) ∝ dL²   (the Euclidean/volume prior the ansatz
#                          assumes; cancels the dL² of the posterior, leaving the
#                          N(μ,σ) "marginal-likelihood" shape that drives H0).
#         - sky direction: isotropic ⇒ constant ⇒ absorbed by the per-event
#                          normalisation. The angular localisation re-enters via
#                          the pixel sampling (∝ PROB), exactly as for real PE.
#       Setting p_pe = prob[pix]·N(dL) (the previous behaviour) divides the
#       localisation back out and collapses every event to the same catalog
#       integral ∫ p_target — the GW signal disappears. That is the bug this
#       patch fixes.
#
# MASSES / SPINS (documented approximation of Architecture A)
# -----------------------------------------------------------
# A 3D skymap is marginalised over masses/spins, so those parameters are
# uninformative here: their "posterior" equals their prior. We therefore draw
# them from broad surrogate priors and set the corresponding p_pe blocks equal
# to those same priors (samples ~ prior, p_pe = prior — the correct treatment
# for an uninformative coordinate). The existing likelihood still evaluates the
# source-frame population mixture p_pop(m1src, q, χ) on these surrogates; in
# expectation the mass/spin reweighting integrates to
#
#     E[ p_pop(m1det/(1+z), q, χ) / π_surrogate ] = (1+z)·F(z; H0),
#
# and the (1+z) cancels the source→detector mass Jacobian in the likelihood.
# The factor F(z;H0) = ∫_{m1src ∈ [m1det_min/(1+z), m1det_max/(1+z)]} p_pop dm
# equals 1 — i.e. the mass marginalisation is UNBIASED across the H0 prior —
# **iff the surrogate detector-frame mass range covers the redshifted population
# support for every z in the analysis**:
#
#     m1det_min ≤ pop_m_min        and        m1det_max ≥ pop_m_max·(1 + z_max).
#
# This converter sets the default m1det_max from (pop_m_max, z_max) to guarantee
# that. If you narrow it, F(z;H0) becomes z-dependent and biases H0.
#
# Two caveats that the math does NOT let this patch remove (use Architecture B
# — a native skymap likelihood term — if they bite):
#   • The surrogate-mass reweighting adds Monte-Carlo variance to every event's
#     integral while carrying no information. log-uniform is a decent proposal
#     for a power-law-ish mass function, but raise --nsamp and check per-event
#     effective sample sizes. A lower-variance option is to draw the surrogates
#     from the *fiducial* population mixture (exact cancellation at the fiducial
#     cosmology); that needs the population model imported here and is left out
#     of this "for now" bridge.
#   • SELECTION CONSISTENCY: the mass+spin distribution assumed here (the
#     surrogate prior, which the likelihood reweights to p_pop) must be the SAME
#     fixed model used in the injection-based selection integral ξ(Λ), so the
#     mass/spin factors cancel between numerator and denominator. In practice:
#     run with the population mass+spin parameters FIXED, and ensure the
#     injection set's mass/spin draw is reweighted to that same fixed model.
#     You cannot infer the mass population from mass-marginalised skymaps.
#
# Population bound defaults below mirror darksirens.gw.populations.registry
# (M_MIN ∈ [2,10], M_MAX ∈ [50,100]); override if your model differs.
# ============================================================================

POP_M_MIN_DEFAULT = 2.0  # lower edge of the population m_min prior
POP_M_MAX_DEFAULT = 100.0  # upper edge of the population m_max prior


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _read_skymap(path: Path):
    """Read a 3D skymap, returning (prob, distmu, distsigma, nside) in RING order.

    PROB is renormalised to sum to 1 over valid pixels; pixels with non-finite
    or non-positive distance moments are removed from the sampling support.
    We force RING ordering everywhere so hp.pix2ang(..., nest=False) is correct;
    a mismatched ordering silently scrambles the sky positions (a latent bug in
    the previous version, which used ligo.skymap's default ordering but RING in
    pix2ang).
    """
    prob = distmu = distsigma = None

    if read_sky_map is not None:
        try:
            # nest=False -> ligo.skymap rasterises the MOC and returns RING order.
            out = read_sky_map(str(path), moc=False, distances=True, nest=False)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
                data = out[0]  # (prob, distmu, distsigma, distnorm)
            elif isinstance(out, tuple) and len(out) >= 4:
                data = out
            else:
                raise ValueError("unexpected ligo.skymap return structure")
            prob = np.asarray(data[0], dtype=float)
            distmu = np.asarray(data[1], dtype=float)
            distsigma = np.asarray(data[2], dtype=float)
        except TypeError:
            # Older ligo.skymap without the nest kwarg: returns RING by default.
            out = read_sky_map(str(path), moc=False, distances=True)
            data = (
                out[0]
                if (isinstance(out, tuple) and isinstance(out[0], tuple))
                else out
            )
            prob = np.asarray(data[0], dtype=float)
            distmu = np.asarray(data[1], dtype=float)
            distsigma = np.asarray(data[2], dtype=float)
        except Exception:
            prob = None  # fall through to healpy

    if prob is None:
        # healpy fallback: hp.read_map reorders the file to RING by default.
        prob, distmu, distsigma, _ = hp.read_map(str(path), field=range(4))
        prob = np.asarray(prob, dtype=float)
        distmu = np.asarray(distmu, dtype=float)
        distsigma = np.asarray(distsigma, dtype=float)

    if prob.ndim != 1:
        raise ValueError(f"{path}: PROB must be 1D (got shape {prob.shape})")
    if prob.size == 0:
        raise ValueError(f"{path}: empty map")
    if np.any(~np.isfinite(prob)) or np.any(prob < 0):
        raise ValueError(f"{path}: invalid PROB values")

    psum = prob.sum()
    if not np.isfinite(psum) or psum <= 0:
        raise ValueError(f"{path}: non-positive total PROB")
    prob = prob / psum

    bad = (~np.isfinite(distmu)) | (~np.isfinite(distsigma)) | (distsigma <= 0)
    if np.any(bad):
        prob = prob.copy()
        prob[bad] = 0.0
        s = prob.sum()
        if s <= 0:
            raise ValueError(f"{path}: no valid distance pixels")
        prob /= s

    nside = hp.npix2nside(prob.size)
    return prob, distmu, distsigma, nside


def _sample_distance_ansatz(
    mu: np.ndarray,
    sig: np.ndarray,
    rng: np.random.Generator,
    n_grid: int = 1024,
) -> np.ndarray:
    """Sample dL ~ dL²·N(dL; mu, sig) on dL ≥ 0 (the Singer 2016 ansatz).

    Vectorised numerical inverse-CDF on a per-sample grid. Robust to the
    pathological-but-allowed mu < 0 shape parameter and to sig ≪ mu.
    The DISTNORM normalisation is recomputed implicitly by normalising the CDF,
    so we never depend on the stored DISTNORM.
    """
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = mu.size
    # Upper support: comfortably past the Gaussian core; guard mu<0.
    hi = np.maximum(mu + 8.0 * sig, 8.0 * sig)
    u = np.linspace(0.0, 1.0, n_grid)[None, :]  # (1, n_grid)
    dl = u * hi[:, None]  # (n, n_grid)
    z = (dl - mu[:, None]) / sig[:, None]
    pdf = dl * dl * np.exp(-0.5 * z * z)  # ∝ dL²·N, unnormalised
    cdf = np.cumsum(pdf, axis=1)
    tot = cdf[:, -1:]
    # Degenerate rows (all-zero pdf): fall back to a uniform draw on [0, hi].
    degenerate = (tot[:, 0] <= 0) | ~np.isfinite(tot[:, 0])
    cdf = np.where(tot > 0, cdf / np.where(tot > 0, tot, 1.0), u)  # broadcast-safe
    r = rng.uniform(size=n)
    idx = np.clip((cdf < r[:, None]).sum(axis=1), 1, n_grid - 1)
    c0 = np.take_along_axis(cdf, (idx - 1)[:, None], axis=1)[:, 0]
    c1 = np.take_along_axis(cdf, idx[:, None], axis=1)[:, 0]
    d0 = np.take_along_axis(dl, (idx - 1)[:, None], axis=1)[:, 0]
    d1 = np.take_along_axis(dl, idx[:, None], axis=1)[:, 0]
    frac = np.where(c1 > c0, (r - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    out = d0 + frac * (d1 - d0)
    out = np.where(degenerate, r * hi, out)
    return np.maximum(out, 1e-6)  # strictly positive for z(dL) inversion downstream


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert 3D LVK skymaps into darksirens GW importance-sample HDF5 (Architecture A)."
    )
    p.add_argument(
        "--skymap_dir",
        required=True,
        help="Directory of per-event 3D skymap FITS files.",
    )
    p.add_argument("--output", required=True, help="Output gwdata.h5 path.")
    p.add_argument(
        "--nsamp",
        type=_positive_int,
        default=5000,
        help="Samples per event. Larger reduces surrogate-mass MC noise (default 5000).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pattern", default="*.fits*", help="Glob for skymap files.")
    # --- surrogate mass/spin proposal (uninformative; reweighted to p_pop downstream) ---
    p.add_argument(
        "--zmax",
        type=float,
        default=1.5,
        help=(
            "Max redshift of the analysis; sets the default m1det_max so the mass "
            "marginalisation is unbiased across the H0 prior (default 1.5)."
        ),
    )
    p.add_argument(
        "--pop_m_min",
        type=float,
        default=POP_M_MIN_DEFAULT,
        help="Lower edge of the population m_min prior (default 2).",
    )
    p.add_argument(
        "--pop_m_max",
        type=float,
        default=POP_M_MAX_DEFAULT,
        help="Upper edge of the population m_max prior (default 100).",
    )
    p.add_argument(
        "--m1det_min",
        type=float,
        default=None,
        help="Surrogate m1det lower bound. Default = pop_m_min.",
    )
    p.add_argument(
        "--m1det_max",
        type=float,
        default=None,
        help="Surrogate m1det upper bound. Default = pop_m_max*(1+zmax).",
    )
    p.add_argument("--q_min", type=float, default=0.05)
    p.add_argument("--chi_abs_max", type=float, default=0.99)
    p.add_argument(
        "--n_dist_grid",
        type=_positive_int,
        default=1024,
        help="Grid resolution for the dL ansatz inverse-CDF (default 1024).",
    )
    args = p.parse_args()

    # Resolve mass-range defaults that guarantee an unbiased mass marginalisation.
    m1det_min = args.pop_m_min if args.m1det_min is None else args.m1det_min
    m1det_max = (
        args.pop_m_max * (1.0 + args.zmax) if args.m1det_max is None else args.m1det_max
    )

    if not (m1det_min > 0 and m1det_max > m1det_min):
        raise ValueError("Require 0 < m1det_min < m1det_max")
    if not (0 < args.q_min <= 1.0):
        raise ValueError("Require 0 < q_min <= 1")
    if not (0 < args.chi_abs_max <= 1.0):
        raise ValueError("Require 0 < chi_abs_max <= 1")

    needed_max = args.pop_m_max * (1.0 + args.zmax)
    if m1det_max < needed_max - 1e-9:
        warnings.warn(
            f"m1det_max={m1det_max:.1f} < pop_m_max*(1+zmax)={needed_max:.1f}. The mass "
            f"marginalisation factor F(z;H0) will be z-dependent and BIAS H0. "
            f"Raise --m1det_max (or lower --zmax) to cover the redshifted population support."
        )
    if m1det_min > args.pop_m_min + 1e-9:
        warnings.warn(
            f"m1det_min={m1det_min:.2f} > pop_m_min={args.pop_m_min:.2f}; low-mass population "
            f"support is truncated at z=0, biasing H0. Lower --m1det_min to pop_m_min."
        )

    skymap_dir = Path(args.skymap_dir)
    files = sorted(skymap_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No skymaps in {skymap_dir} matching {args.pattern}")

    rng = _rng(args.seed)
    nobs, nsamp = len(files), args.nsamp

    ra = np.empty((nobs, nsamp))
    dec = np.empty((nobs, nsamp))
    dL = np.empty((nobs, nsamp))
    m1det = np.empty((nobs, nsamp))
    m2det = np.empty((nobs, nsamp))
    chieff = np.empty((nobs, nsamp))
    p_pe = np.empty((nobs, nsamp))

    log_m1_width = np.log(m1det_max / m1det_min)
    q_width = 1.0 - args.q_min
    chi_width = 2.0 * args.chi_abs_max

    for i, path in enumerate(files):
        prob, distmu, distsigma, nside = _read_skymap(path)

        # (1) posterior samples in (pixel, dL)
        pix = rng.choice(prob.size, size=nsamp, p=prob)
        theta, phi = hp.pix2ang(nside, pix, nest=False)  # RING (see _read_skymap)
        ra_i = phi
        dec_i = 0.5 * np.pi - theta
        dL_i = _sample_distance_ansatz(
            distmu[pix], distsigma[pix], rng, n_grid=args.n_dist_grid
        )

        # surrogate (uninformative) mass/spin draws
        u = rng.uniform(size=nsamp)
        m1_i = m1det_min * np.exp(u * log_m1_width)  # log-uniform in m1det
        q_i = rng.uniform(args.q_min, 1.0, size=nsamp)
        m2_i = q_i * m1_i
        chi_i = rng.uniform(-args.chi_abs_max, args.chi_abs_max, size=nsamp)

        # (2) p_pe = PE PRIOR in the (m1det, q, dL) basis.
        #     distance prior  π(dL) ∝ dL²   (cancels the ansatz dL²)
        #     sky prior       isotropic ⇒ constant ⇒ dropped (per-event norm absorbs it);
        #                      localisation enters via pix ~ PROB, not via p_pe.
        #     mass/q/spin     = the surrogate proposal densities (uninformative coords).
        g_m1 = 1.0 / (m1_i * log_m1_width)
        g_q = 1.0 / q_width
        g_chi = 1.0 / chi_width
        p_pe_i = (dL_i * dL_i) * g_m1 * g_q * g_chi

        ra[i], dec[i], dL[i] = ra_i, dec_i, dL_i
        m1det[i], m2det[i], chieff[i] = m1_i, m2_i, chi_i
        p_pe[i] = p_pe_i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["mock_data"] = (
            True  # loader uses p_pe as-is (no extra spin-prior multiply)
        )
        f.attrs["source"] = "darksirens_skymaps_to_samples"
        f.attrs["m1det_min"] = float(m1det_min)
        f.attrs["m1det_max"] = float(m1det_max)
        f.attrs["zmax"] = float(args.zmax)
        f.attrs["pe_distance_prior"] = "euclidean_dL2"
        for name, arr in (
            ("ra", ra),
            ("dec", dec),
            ("dL", dL),
            ("m1det", m1det),
            ("m2det", m2det),
            ("chieff", chieff),
            ("p_pe", p_pe),
        ):
            f.create_dataset(name, data=arr.reshape(-1))

    print(
        f"Wrote {out}: {nobs} events x {nsamp} samples "
        f"(m1det∈[{m1det_min:.1f},{m1det_max:.1f}], dL prior ∝ dL²)."
    )


if __name__ == "__main__":
    main()
