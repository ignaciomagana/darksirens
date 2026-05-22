from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

try:
    from ligo.skymap.io import read_sky_map
except Exception:  # pragma: no cover - optional dependency
    read_sky_map = None


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return ivalue


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _read_skymap(path: Path):
    """Read a 3D skymap, supporting both MOC and standard HEALPix FITS layouts."""
    # Preferred: ligo.skymap reader handles MOC products transparently.
    if read_sky_map is not None:
        try:
            m = read_sky_map(str(path), moc=False, distances=True)
            # ligo.skymap returns ((prob, distmu, distsigma, distnorm), meta)
            # when distances=True.
            if isinstance(m, tuple) and len(m) == 2 and isinstance(m[0], tuple) and len(m[0]) >= 4:
                prob, distmu, distsigma, _distnorm = m[0][:4]
            elif isinstance(m, tuple) and len(m) >= 4:
                # Backward/alternate shape: direct 4-array tuple.
                prob, distmu, distsigma, _distnorm = m[:4]
            else:
                raise ValueError(
                    "ligo.skymap returned unexpected map structure; expected "
                    "((prob, distmu, distsigma, distnorm), meta)"
                )
            prob = np.asarray(prob, dtype=float)
            distmu = np.asarray(distmu, dtype=float)
            distsigma = np.asarray(distsigma, dtype=float)
        except Exception:
            # Fallback for unusual files where ligo.skymap parsing fails.
            prob, distmu, distsigma, _distnorm = hp.read_map(str(path), field=range(4))
            prob = np.asarray(prob, dtype=float)
            distmu = np.asarray(distmu, dtype=float)
            distsigma = np.asarray(distsigma, dtype=float)
    else:
        # Fallback path when ligo.skymap is unavailable.
        prob, distmu, distsigma, _distnorm = hp.read_map(str(path), field=range(4))
        prob = np.asarray(prob, dtype=float)
        distmu = np.asarray(distmu, dtype=float)
        distsigma = np.asarray(distsigma, dtype=float)

    if prob.ndim != 1:
        raise ValueError(f"{path}: PROB must be 1D")
    if prob.size == 0:
        raise ValueError(f"{path}: empty map")

    if np.any(~np.isfinite(prob)) or np.any(prob < 0):
        raise ValueError(f"{path}: invalid PROB values")
    psum = prob.sum()
    if not np.isfinite(psum) or psum <= 0:
        raise ValueError(f"{path}: non-positive total PROB")
    prob = prob / psum

    bad_dist = (~np.isfinite(distmu)) | (~np.isfinite(distsigma)) | (distsigma <= 0)
    if np.any(bad_dist):
        # Remove bad pixels from sampling support.
        prob = prob.copy()
        prob[bad_dist] = 0.0
        psum = prob.sum()
        if psum <= 0:
            raise ValueError(f"{path}: no valid distance pixels")
        prob /= psum

    nside = hp.npix2nside(prob.size)
    return prob, distmu, distsigma, nside


def _sample_truncnorm_positive(mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Simple rejection sampler for dL > 0.
    out = rng.normal(mu, sigma)
    bad = out <= 0
    n_try = 0
    while np.any(bad):
        out[bad] = rng.normal(mu[bad], sigma[bad])
        bad = out <= 0
        n_try += 1
        if n_try > 50:
            out[bad] = np.maximum(mu[bad], 1e-3)
            break
    return out


def _norm_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert 3D LIGO/Virgo/KAGRA skymaps into darksirens GW sample HDF5.")
    p.add_argument("--skymap_dir", required=True, help="Directory containing per-event 3D skymap FITS files.")
    p.add_argument("--output", required=True, help="Output gwdata.h5 path.")
    p.add_argument("--nsamp", type=_positive_int, default=2000, help="Samples per event.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pattern", default="*.fits*", help="Glob pattern for skymap files.")
    p.add_argument("--m1det_min", type=float, default=3.0)
    p.add_argument("--m1det_max", type=float, default=120.0)
    p.add_argument("--q_min", type=float, default=0.05)
    p.add_argument("--chi_abs_max", type=float, default=0.99)
    args = p.parse_args()

    skymap_dir = Path(args.skymap_dir)
    files = sorted(skymap_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No skymaps found in {skymap_dir} matching {args.pattern}")

    if not (args.m1det_min > 0 and args.m1det_max > args.m1det_min):
        raise ValueError("Require 0 < m1det_min < m1det_max")
    if not (0 < args.q_min <= 1.0):
        raise ValueError("Require 0 < q_min <= 1")
    if not (0 < args.chi_abs_max <= 1.0):
        raise ValueError("Require 0 < chi_abs_max <= 1")

    rng = _rng(args.seed)

    nobs = len(files)
    nsamp = args.nsamp

    ra = np.empty((nobs, nsamp), dtype=np.float64)
    dec = np.empty((nobs, nsamp), dtype=np.float64)
    dL = np.empty((nobs, nsamp), dtype=np.float64)
    m1det = np.empty((nobs, nsamp), dtype=np.float64)
    m2det = np.empty((nobs, nsamp), dtype=np.float64)
    chieff = np.empty((nobs, nsamp), dtype=np.float64)
    p_pe = np.empty((nobs, nsamp), dtype=np.float64)

    log_m1_width = np.log(args.m1det_max / args.m1det_min)
    q_width = 1.0 - args.q_min
    chi_width = 2.0 * args.chi_abs_max

    for i, path in enumerate(files):
        prob, distmu, distsigma, nside = _read_skymap(path)

        pix = rng.choice(prob.size, size=nsamp, p=prob)
        theta, phi = hp.pix2ang(nside, pix)
        ra_i = phi
        dec_i = 0.5 * np.pi - theta

        mu = distmu[pix]
        sig = distsigma[pix]
        dL_i = _sample_truncnorm_positive(mu, sig, rng)

        # Broad, intentionally uninformative PE surrogates.
        u = rng.uniform(size=nsamp)
        m1_i = args.m1det_min * np.exp(u * log_m1_width)  # log-uniform
        q_i = rng.uniform(args.q_min, 1.0, size=nsamp)
        m2_i = q_i * m1_i
        chi_i = rng.uniform(-args.chi_abs_max, args.chi_abs_max, size=nsamp)

        # Proposal density in (m1det, q, dL) basis.  We sampled
        # dL from p(dL|pix) and pix from p(pix), so include p(pix)*p(dL|pix).
        g_m1 = 1.0 / (m1_i * log_m1_width)
        g_q = np.full(nsamp, 1.0 / q_width)
        g_chi = np.full(nsamp, 1.0 / chi_width)
        g_sky = prob[pix]
        g_dL_given_pix = _norm_pdf(dL_i, mu, sig)

        ppe_i = g_m1 * g_q * g_chi * g_sky * g_dL_given_pix

        ra[i] = ra_i
        dec[i] = dec_i
        dL[i] = dL_i
        m1det[i] = m1_i
        m2det[i] = m2_i
        chieff[i] = chi_i
        p_pe[i] = ppe_i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["mock_data"] = True
        f.attrs["source"] = "darksirens_skymaps_to_samples"
        # darksirens loaders expect flattened arrays with length nobs*nsamp.
        f.create_dataset("ra", data=ra.reshape(-1))
        f.create_dataset("dec", data=dec.reshape(-1))
        f.create_dataset("dL", data=dL.reshape(-1))
        f.create_dataset("m1det", data=m1det.reshape(-1))
        f.create_dataset("m2det", data=m2det.reshape(-1))
        f.create_dataset("chieff", data=chieff.reshape(-1))
        f.create_dataset("p_pe", data=p_pe.reshape(-1))

    print(f"Wrote {out} with {nobs} events x {nsamp} samples (flattened length {nobs*nsamp})")


if __name__ == "__main__":
    main()
