from __future__ import annotations

import argparse
import hashlib
import secrets
import warnings
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

from darksirens.utils.cosmology import H0Planck, Om0Planck, dL_grid_bounds, z_of_dL

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
# z_max here is NOT the user's declaration but the largest redshift the sampled
# distances actually reach at the top of the H0 prior (--h0_max), floored by
# --zmax: F(z;H0)=1 has to hold at every z the run can visit. This converter
# sets the default m1det_max from (pop_m_max, that z_max) to guarantee it, warns
# when the data reach beyond --zmax, and stamps the z it really guarantees as
# ``zmax`` (with ``zmax_declared`` / ``zmax_reached`` / ``h0_max`` alongside).
# If you narrow m1det_max, F(z;H0) becomes z-dependent and biases H0.
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
#
# OUTPUT SCHEMA
# -------------
# The output is written as ``format_version = "gwcat-1.0"`` (the generic,
# non-lensing, chi_eff-basis PE contract — the same one darksirens' own mock
# generator uses) so it loads directly through
# ``darksirens.gw.utils.load_gw_samples`` without a separate conversion step.
# That loader also requires source-frame masses ``m1src``/``m2src``, which a
# mass-marginalised 3D skymap does not constrain. We obtain them by inverting
# the *detector-frame* dL samples to redshift under a fiducial PE cosmology
# (``--pe_H0``/``--pe_Om0``, default Planck15) via the same ``z_of_dL`` grid
# inversion the likelihood itself uses (darksirens.utils.cosmology), then
# m1src = m1det/(1+z), m2src = m2det/(1+z). This bookkeeping does not feed
# back into p_pe or H0 inference: mock_data=True (below) makes the loader
# treat p_pe as-is and skip the chi_eff-prior reweight that would otherwise
# consume m1src/m2src, so the choice of fiducial PE cosmology is inert beyond
# populating these two required datasets.
#
# The file also carries two markers that make the SELECTION-CONSISTENCY caveat
# above machine-enforceable instead of documentation-only:
#
#   source                     = "darksirens_skymaps_to_samples"
#   requires_fixed_population  = True   (bool)
#
# darksirens_inference reads them during configuration validation and REFUSES to
# start with a free population block (--fix_population false), because the
# mass/spin coordinates in this file are surrogate draws carrying no
# information: inferring a population from them fits the surrogate proposal, and
# the mass/spin factors no longer cancel against the injection-based selection
# integral. Run with --fix_population true, or pass --allow_skymap_population to
# downgrade the refusal to a loud warning for a deliberate methodological study.
# ============================================================================

POP_M_MIN_DEFAULT = 2.0  # lower edge of the population m_min prior
POP_M_MAX_DEFAULT = 100.0  # upper edge of the population m_max prior

#: ``f.attrs["source"]`` stamped on every output file. darksirens.cli.inference
#: imports this to recognise pre-``requires_fixed_population`` products.
SOURCE_TAG = "darksirens_skymaps_to_samples"


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def _effective_seed(seed: int | None) -> int:
    """The seed actually used, materialised so it can be written down (REP-01).

    ``--seed`` defaulted to ``None``, ``default_rng(None)`` draws fresh OS
    entropy, and nothing about that entropy reached the output file.  The
    artifact was therefore unreproducible even from its own command line: two
    identical invocations produced different PE samples and neither recorded
    which one it had been.  Drawing the entropy HERE, printing it and stamping
    it into the file makes the default convenient AND replayable — rerun with
    ``--seed <effective_seed>`` and you get the same artifact back.
    """
    if seed is not None:
        return int(seed)
    # 63 bits of OS entropy: the same kind of source ``default_rng(None)`` uses,
    # but named rather than thrown away.  63 and not 128 so the value is an
    # ordinary int64 -- printable, retypable on a command line, and storable as
    # an HDF5 attribute (a 128-bit SeedSequence.entropy is none of those).
    return int(secrets.randbits(63))


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(_effective_seed(seed))


def _sha256(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _source_provenance(files):
    """Ordered basenames + content hashes of the skymaps that fed this file.

    The seed alone does not identify the artifact: the sampling order follows
    ``sorted(skymap_dir.glob(pattern))``, so the same seed over a different (or
    reordered, or edited) skymap set gives different samples.  Recording the
    ordered names AND their sha256 makes the input side of the reproduction
    checkable rather than assumed.
    """
    names = [Path(f).name for f in files]
    hashes = [_sha256(Path(f)) for f in files]
    # One hash over the ordered (name, hash) pairs: a single value a downstream
    # record can quote, and it changes if the ORDER changes even when the set
    # does not.
    digest = hashlib.sha256()
    for n, h in zip(names, hashes):
        digest.update(n.encode())
        digest.update(b"\0")
        digest.update(h.encode())
        digest.update(b"\n")
    return names, hashes, digest.hexdigest()


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


_JITTER_LEVELS = 6      # 4**6 = 4096 equal-area sub-pixels per skymap pixel


def _sample_angles_in_pixels(nside, pix_ring, rng, levels=_JITTER_LEVELS):
    """Directions drawn UNIFORMLY INSIDE each sampled skymap pixel.

    Placing every sample at its pixel's CENTRE makes the event's angular support a
    set of measure-zero points rather than the pixel's solid angle.  The
    likelihood re-derives ``pix = ang2pix(catalog_nside, ra, dec)`` from these
    angles, so as soon as the analysis resolution is FINER than the skymap's --
    a low-resolution or heavily rasterised skymap, a high-nside catalog or
    completion map -- every sample of one coarse pixel collapses into the single
    fine pixel containing its centre, ``p_z(z|pix)`` sees only ~(nside ratio)^-2
    of the galaxies actually inside the localisation region, and the sky-anisotropy
    models are evaluated on a comb of directions.

    Each sample is placed at the centre of a uniformly chosen equal-area sub-pixel
    at ``nside * 2**levels``, which is uniform in solid angle inside the parent
    pixel to that resolution (1/2**levels of a pixel's linear size).
    """
    nest = np.asarray(hp.ring2nest(nside, pix_ring), dtype=np.int64)
    nchild = 4 ** int(levels)
    child = nest * nchild + rng.integers(0, nchild, size=nest.size)
    return hp.pix2ang(nside * 2 ** int(levels), child, nest=True)


def _ansatz_grid_bounds(
    mu: np.ndarray, sig: np.ndarray, n_widths: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row [lo, hi] bracket of the dL²·N(dL; mu, sig) mass on dL > 0.

    The target's log-density is f(d) = 2 ln d − ½ (d−mu)²/σ², so its mode solves
    d² − mu·d − 2σ² = 0 and its local curvature gives a Laplace width:

        mode = (mu + √(mu² + 8σ²)) / 2 ,   w = mode·σ / √(2σ² + mode²).

    For the ordinary case mu ≫ σ this returns mode ≈ mu and w ≈ σ, i.e. the
    plain ±n_widths·σ window. It matters for the two regimes a fixed [0, mu+8σ]
    grid resolves badly:

      * σ ≪ mu (e.g. σ/mu = 1e-4): a grid spanning [0, mu] puts O(1) nodes
        inside the Gaussian core, so the inverse-CDF quantises the draw onto a
        few nodes and inflates its spread by an order of magnitude.
      * mu < 0 (allowed by the ansatz): the mass sits at d ≈ 2σ²/|mu| ≪ σ,
        many hundreds of grid steps below the first node of a [0, 8σ] grid.

    The positive root is evaluated in whichever algebraically equivalent form
    avoids cancellation for the sign of mu, and √(mu² + 8σ²) via hypot so a
    large mu/σ cannot overflow.
    """
    root = np.hypot(mu, np.sqrt(8.0) * sig)  # = sqrt(mu**2 + 8 sig**2)
    # A degenerate row (sig == 0, or mu == sig == 0) makes these 0/0; it is left
    # to propagate as NaN, which _sample_distance_ansatz reports naming the row.
    with np.errstate(divide="ignore", invalid="ignore"):
        mode = np.where(mu >= 0.0, 0.5 * (mu + root), 4.0 * sig * sig / (root - mu))
        width = mode * sig / np.hypot(np.sqrt(2.0) * sig, mode)
    lo = np.maximum(0.0, mode - n_widths * width)
    hi = mode + n_widths * width
    return lo, hi


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

    The grid weights are built in LOG space and rescaled per row by that row's
    maximum before exponentiating. Evaluating dL²·exp(−½z²) directly underflows
    every node of a row whenever the whole grid sits far in the Gaussian tail —
    exactly what mu ≪ 0 does, since the entire dL > 0 support is then |mu|/σ
    standard deviations out. The previous code detected the resulting all-zero
    row and substituted a UNIFORM draw on [0, hi]: a silent replacement of the
    event's distance posterior by a distribution with the wrong support, wrong
    shape and a mean ~4σ where the target concentrates near 0.06σ. Rescaling by
    the row maximum makes the same row perfectly well conditioned, so no
    fallback distribution is needed and none is offered: a row that is still
    degenerate is malformed input and raises.
    """
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = mu.size
    lo, hi = _ansatz_grid_bounds(mu, sig)
    u = np.linspace(0.0, 1.0, n_grid)[None, :]  # (1, n_grid)
    dl = lo[:, None] + u * (hi - lo)[:, None]  # (n, n_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (dl - mu[:, None]) / sig[:, None]
        logw = 2.0 * np.log(dl)  # dl == 0 -> -inf, kept as such
        logw -= 0.5 * z * z  # ∝ ln(dL²·N), unnormalised
    del z
    logw_max = np.max(logw, axis=1)
    # A row is unusable only if EVERY node is -inf. Any dl > 0 node has a finite
    # log-weight, so this means an empty/malformed bracket, not a hard sampling
    # problem — report it instead of substituting some other distribution.
    bad = ~np.isfinite(logw_max)
    if np.any(bad):
        j = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"distance ansatz has no valid support for row {j}: "
            f"DISTMU={mu[j]!r}, DISTSIGMA={sig[j]!r} "
            f"(grid [{lo[j]!r}, {hi[j]!r}], {n_grid} nodes). "
            "Check the skymap's distance moments."
        )
    logw -= logw_max[:, None]
    pdf = np.exp(logw, out=logw)  # rescaled weights, max 1 per row
    cdf = np.cumsum(pdf, axis=1)
    cdf /= cdf[:, -1:].copy()  # totals are >= 1 by construction (the max node is 1)
    r = rng.uniform(size=n)
    idx = np.clip((cdf < r[:, None]).sum(axis=1), 1, n_grid - 1)
    c0 = np.take_along_axis(cdf, (idx - 1)[:, None], axis=1)[:, 0]
    c1 = np.take_along_axis(cdf, idx[:, None], axis=1)[:, 0]
    d0 = np.take_along_axis(dl, (idx - 1)[:, None], axis=1)[:, 0]
    d1 = np.take_along_axis(dl, idx[:, None], axis=1)[:, 0]
    frac = np.where(c1 > c0, (r - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    out = d0 + frac * (d1 - d0)
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
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed. Omitted, a fresh 128-bit OS seed is drawn, PRINTED, and "
            "stamped into the output as attrs['seed'] — rerun with that value "
            "(and the same skymaps) to regenerate the artifact exactly."
        ),
    )
    p.add_argument("--pattern", default="*.fits*", help="Glob for skymap files.")
    # --- surrogate mass/spin proposal (uninformative; reweighted to p_pop downstream) ---
    p.add_argument(
        "--zmax",
        type=float,
        default=1.5,
        help=(
            "Declared max redshift of the analysis; a FLOOR for the default "
            "m1det_max (default 1.5). The bound actually used is the larger of "
            "this and the redshift the sampled distances themselves reach at "
            "--h0_max, so a distant event cannot silently bias H0."
        ),
    )
    p.add_argument(
        "--h0_max",
        type=float,
        default=120.0,
        help=(
            "Upper edge of the H0 prior the file will be analysed under "
            "(default 120, darksirens' own H0 prior upper bound). The most "
            "aggressive cosmology in the prior is the one that maps the sampled "
            "dL to the LARGEST redshift, so it sets the redshifted population "
            "support the surrogate mass window has to cover."
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
        help=("Surrogate m1det upper bound. Default = "
              "pop_m_max*(1+max(zmax, z reached at --h0_max))."),
    )
    p.add_argument("--q_min", type=float, default=0.05)
    p.add_argument("--chi_abs_max", type=float, default=0.99)
    p.add_argument(
        "--n_dist_grid",
        type=_positive_int,
        default=1024,
        help="Grid resolution for the dL ansatz inverse-CDF (default 1024).",
    )
    # --- fiducial cosmology for the m1src/m2src bookkeeping datasets only ---
    p.add_argument(
        "--pe_H0",
        type=float,
        default=None,
        help=(
            "Fiducial H0 (km/s/Mpc) used ONLY to invert detector-frame dL samples "
            "to redshift for the m1src/m2src datasets the loader requires. Does "
            "NOT affect p_pe or H0 inference (mock_data=True; see OUTPUT SCHEMA "
            "above). Default: Planck15."
        ),
    )
    p.add_argument(
        "--pe_Om0",
        type=float,
        default=None,
        help="Fiducial Om0 for the same dL->z inversion as --pe_H0. Default: Planck15.",
    )
    args = p.parse_args()

    # Resolve mass-range defaults that guarantee an unbiased mass marginalisation.
    # ``m1det_max`` needs the redshift the SAMPLES reach (below), so only the
    # lower edge and the shape checks can be settled here.
    m1det_min = args.pop_m_min if args.m1det_min is None else args.m1det_min
    if not (m1det_min > 0):
        raise ValueError("Require m1det_min > 0")
    if args.m1det_max is not None and not (args.m1det_max > m1det_min):
        raise ValueError("Require 0 < m1det_min < m1det_max")
    if not (0 < args.q_min <= 1.0):
        raise ValueError("Require 0 < q_min <= 1")
    if not (0 < args.chi_abs_max <= 1.0):
        raise ValueError("Require 0 < chi_abs_max <= 1")
    if not (args.h0_max > 0):
        raise ValueError("Require h0_max > 0")

    if m1det_min > args.pop_m_min + 1e-9:
        warnings.warn(
            f"m1det_min={m1det_min:.2f} > pop_m_min={args.pop_m_min:.2f}; low-mass population "
            f"support is truncated at z=0, biasing H0. Lower --m1det_min to pop_m_min."
        )

    skymap_dir = Path(args.skymap_dir)
    files = sorted(skymap_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No skymaps in {skymap_dir} matching {args.pattern}")

    seed_effective = _effective_seed(args.seed)
    rng = np.random.default_rng(seed_effective)
    src_names, src_hashes, src_digest = _source_provenance(files)
    print(
        f"[skymaps_to_samples] seed={seed_effective}"
        f"{'' if args.seed is not None else '  (generated; pass --seed to replay)'}\n"
        f"[skymaps_to_samples] {len(files)} skymaps from {skymap_dir} "
        f"matching {args.pattern!r}, sources_sha256={src_digest}"
    )
    nobs, nsamp = len(files), args.nsamp

    ra = np.empty((nobs, nsamp))
    dec = np.empty((nobs, nsamp))
    dL = np.empty((nobs, nsamp))
    m1det = np.empty((nobs, nsamp))
    m2det = np.empty((nobs, nsamp))
    m1src = np.empty((nobs, nsamp))
    m2src = np.empty((nobs, nsamp))
    chieff = np.empty((nobs, nsamp))
    p_pe = np.empty((nobs, nsamp))

    q_width = 1.0 - args.q_min
    chi_width = 2.0 * args.chi_abs_max

    # Fiducial PE cosmology for the m1src/m2src bookkeeping datasets (see the
    # OUTPUT SCHEMA note at the top of this file); inert for p_pe/H0 inference.
    pe_H0 = H0Planck if args.pe_H0 is None else args.pe_H0
    pe_Om0 = Om0Planck if args.pe_Om0 is None else args.pe_Om0
    dL_lo, dL_hi = (float(x) for x in dL_grid_bounds(pe_H0, pe_Om0))

    nside_min = None        # coarsest skymap resolution, stamped for the run to check

    # PASS 1 — sky + distance samples.  Done before the surrogate masses so the
    # mass window can be sized from the redshifts the data actually reach rather
    # than from the user's --zmax declaration.
    for i, path in enumerate(files):
        prob, distmu, distsigma, nside = _read_skymap(path)

        pix = rng.choice(prob.size, size=nsamp, p=prob)
        # RING indices (see _read_skymap), jittered inside the pixel so the
        # angular support is the pixel's AREA, not its centre.
        theta, phi = _sample_angles_in_pixels(nside, pix, rng)
        ra[i] = phi
        dec[i] = 0.5 * np.pi - theta
        nside_min = nside if nside_min is None else min(nside_min, nside)
        dL[i] = _sample_distance_ansatz(
            distmu[pix], distsigma[pix], rng, n_grid=args.n_dist_grid
        )

    # z(dL) under the fiducial PE cosmology, reusing the likelihood's own grid
    # inversion (darksirens.utils.cosmology.z_of_dL); clip to the grid support so
    # an out-of-range dL sample yields a clamped z rather than the
    # NaN-outside-grid sentinel.
    dL_clipped = np.clip(dL, dL_lo, dL_hi)
    z_fid = np.asarray(z_of_dL(dL_clipped, pe_H0, pe_Om0))

    # The redshift the analysis can actually reach: the most aggressive cosmology
    # in the H0 prior maps a given dL to the LARGEST z, so that -- not --zmax --
    # is what the surrogate mass window has to cover for F(z;H0) = 1.  (Om0 is
    # held at the fiducial: H0 is the direction the prior is broad in.)
    dL_lo_hi, dL_hi_hi = (float(x) for x in dL_grid_bounds(args.h0_max, pe_Om0))
    z_reached = float(
        np.max(np.asarray(z_of_dL(np.clip(dL, dL_lo_hi, dL_hi_hi),
                                  args.h0_max, pe_Om0)))
    )
    zmax_needed = max(float(args.zmax), z_reached)
    if z_reached > args.zmax + 1e-9:
        warnings.warn(
            f"--zmax={args.zmax:.2f} but the sampled distances reach z={z_reached:.2f} "
            f"at H0={args.h0_max:.1f} (the top of the H0 prior). The surrogate mass "
            f"window is sized from z={zmax_needed:.2f} instead, since F(z;H0)=1 -- the "
            f"converter's whole unbiasedness argument -- must hold at every z the run "
            f"can reach, not only at the declared one."
        )

    m1det_max = (
        args.pop_m_max * (1.0 + zmax_needed)
        if args.m1det_max is None
        else args.m1det_max
    )
    needed_max = args.pop_m_max * (1.0 + zmax_needed)
    if m1det_max < needed_max - 1e-9:
        warnings.warn(
            f"m1det_max={m1det_max:.1f} < pop_m_max*(1+zmax)={needed_max:.1f} at "
            f"zmax={zmax_needed:.2f}. The mass marginalisation factor F(z;H0) will be "
            f"z-dependent and BIAS H0. Raise --m1det_max (or restrict the analysis "
            f"to nearer events / a narrower H0 prior) to cover the redshifted "
            f"population support."
        )
    # The z up to which the resolved window really does cover the population
    # support -- stamped into the file instead of the user's declaration.
    zmax_guaranteed = min(zmax_needed, m1det_max / args.pop_m_max - 1.0)
    log_m1_width = np.log(m1det_max / m1det_min)

    # PASS 2 — surrogate (uninformative) mass/spin draws, per event.
    for i in range(nobs):
        u = rng.uniform(size=nsamp)
        m1_i = m1det_min * np.exp(u * log_m1_width)  # log-uniform in m1det
        q_i = rng.uniform(args.q_min, 1.0, size=nsamp)
        m2_i = q_i * m1_i
        chi_i = rng.uniform(-args.chi_abs_max, args.chi_abs_max, size=nsamp)
        m1src[i] = m1_i / (1.0 + z_fid[i])
        m2src[i] = m2_i / (1.0 + z_fid[i])

        # p_pe = PE PRIOR in the (m1det, q, dL) basis.
        #     distance prior  π(dL) ∝ dL²   (cancels the ansatz dL²)
        #     sky prior       isotropic ⇒ constant ⇒ dropped (per-event norm absorbs it);
        #                      localisation enters via pix ~ PROB, not via p_pe.
        #     mass/q/spin     = the surrogate proposal densities (uninformative coords).
        g_m1 = 1.0 / (m1_i * log_m1_width)
        g_q = 1.0 / q_width
        g_chi = 1.0 / chi_width

        m1det[i], m2det[i], chieff[i] = m1_i, m2_i, chi_i
        p_pe[i] = (dL[i] * dL[i]) * g_m1 * g_q * g_chi

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        # gwcat-1.0: the generic, non-lensing, chi_eff-basis PE contract (see
        # the OUTPUT SCHEMA note above) — makes this file load directly
        # through darksirens.gw.utils.load_gw_samples.
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["mock_data"] = (
            True  # loader uses p_pe as-is (no extra spin-prior multiply)
        )
        f.attrs["source"] = SOURCE_TAG
        # Machine-readable form of the selection-consistency requirement: the
        # mass/spin draws here are uninformative surrogates, so the population
        # (and the injection set reweighted to it) must be held FIXED. The
        # inference CLI refuses a free population block on a file carrying this.
        f.attrs["requires_fixed_population"] = True
        f.attrs["m1det_min"] = float(m1det_min)
        f.attrs["m1det_max"] = float(m1det_max)
        # zmax is the z up to which the resolved surrogate mass window really
        # does cover the redshifted population support (F(z;H0) = 1), NOT the
        # --zmax declaration: the two differ whenever the data reach further
        # than declared, or the user narrowed --m1det_max.
        f.attrs["zmax"] = float(zmax_guaranteed)
        f.attrs["zmax_declared"] = float(args.zmax)
        f.attrs["zmax_reached"] = float(z_reached)
        f.attrs["h0_max"] = float(args.h0_max)
        f.attrs["pe_distance_prior"] = "euclidean_dL2"
        # The coarsest skymap resolution behind these samples: the sky support is
        # a pixel of THIS nside (samples are drawn uniformly inside it), so an
        # analysis at a much finer catalog / sky-model nside is resolving
        # structure the skymaps never carried.
        f.attrs["skymap_nside"] = int(nside_min)
        # Fiducial cosmology used ONLY for the m1src/m2src bookkeeping above;
        # required by the loader but inert given mock_data=True.
        f.attrs["pe_cosmology_H0"] = float(pe_H0)
        f.attrs["pe_cosmology_Om0"] = float(pe_Om0)
        # p_pe already includes g_chi, the flat proposal density over the
        # sampled chieff range — the loader must NOT reapply an astrophysical
        # chi_eff prior on top of it.
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = float(args.chi_abs_max)
        # --- replay provenance (REP-01) ----------------------------------
        # Everything needed to regenerate this file byte-for-byte:
        #   darksirens_skymaps_to_samples --skymap_dir <dir> --pattern <pattern>
        #       --nsamp <nsamp> --seed <seed> ... (the sampling knobs below)
        # The seed alone is not enough — the sampling order is the sorted glob,
        # so the ordered source list and its hashes are part of the identity.
        # As a STRING: a user-supplied --seed is an unbounded Python int and
        # h5py cannot store one that exceeds int64 ("Object dtype has no native
        # HDF5 equivalent"). int(f.attrs["seed"]) round-trips either way, and a
        # provenance stamp that can fail on a legal input is not a stamp.
        f.attrs["seed"] = str(int(seed_effective))
        f.attrs["seed_was_generated"] = bool(args.seed is None)
        f.attrs["skymap_pattern"] = str(args.pattern)
        f.attrs["skymap_dir"] = str(skymap_dir)
        f.attrs["sources_sha256"] = str(src_digest)
        f.create_dataset(
            "provenance_source_names",
            data=np.array(src_names, dtype=h5py.string_dtype()))
        f.create_dataset(
            "provenance_source_sha256",
            data=np.array(src_hashes, dtype=h5py.string_dtype()))
        for name, arr in (
            ("ra", ra),
            ("dec", dec),
            ("dL", dL),
            ("m1det", m1det),
            ("m2det", m2det),
            ("m1src", m1src),
            ("m2src", m2src),
            ("chieff", chieff),
            ("p_pe", p_pe),
        ):
            f.create_dataset(name, data=arr.reshape(-1))

    from darksirens.cli.common import _ok
    _ok(
        f"Wrote {out}: {nobs} events x {nsamp} samples "
        f"(m1det∈[{m1det_min:.1f},{m1det_max:.1f}], dL prior ∝ dL², "
        f"pe_cosmology H0={pe_H0:.2f} Om0={pe_Om0:.3f})."
    )


if __name__ == "__main__":
    main()
