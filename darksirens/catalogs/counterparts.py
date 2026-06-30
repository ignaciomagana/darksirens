"""Bright-siren counterpart catalog construction."""

import healpy as hp
import numpy as np


def as_counterpart_array(counterpart) -> np.ndarray:
    """Return counterpart metadata as an ``(N, 3)`` float array.

    The public CLI accepts either one ``RA DEC Z`` triplet or a flattened list
    of triplets for multi-event bright-siren analyses.
    """
    arr = np.asarray(counterpart, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError("bright_sirens counterpart metadata must be RA DEC Z triplets.")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("bright_sirens counterpart metadata must have shape (N, 3).")
    return arr


def build_counterpart_catalog(counterpart, counterpart_dz, counterpart_nside, *, sky_marginalized=False) -> dict:
    """Build the synthetic full-sky catalog used by bright-siren analyses."""
    counterparts = as_counterpart_array(counterpart)
    ra_cp = counterparts[:, 0]
    dec_cp = counterparts[:, 1]
    z_cp = counterparts[:, 2]
    nside = int(counterpart_nside)
    npix = hp.nside2npix(nside)
    counterpart_pixels = hp.ang2pix(nside, np.pi / 2.0 - dec_cp, ra_cp).astype(np.int32)
    counterpart_pixel = int(counterpart_pixels[0])
    counterpart_zs = z_cp.astype(float, copy=False)
    counterpart_dzs = np.ones_like(counterpart_zs, dtype=float) * float(counterpart_dz)

    counts = np.bincount(counterpart_pixels, minlength=npix).astype(np.int32)
    max_counterparts = max(1, int(counts.max()))
    zgals = np.zeros((npix, max_counterparts), dtype=float)
    dzgals = np.ones((npix, max_counterparts), dtype=float) * float(counterpart_dz)
    wgals = np.zeros((npix, max_counterparts), dtype=float)
    ngals = counts
    offsets = np.zeros(npix, dtype=np.int32)
    for pix_i, z_i in zip(counterpart_pixels, counterpart_zs):
        j = offsets[pix_i]
        zgals[pix_i, j] = z_i
        wgals[pix_i, j] = 1.0
        offsets[pix_i] += 1

    apix = hp.nside2pixarea(nside)
    print(
        "Using bright-siren counterpart catalog: "
        f"{len(counterpart_zs)} counterpart(s), nside={nside}, "
        f"pixels={counterpart_pixels.tolist()}"
    )
    return {
        "nside": nside,
        "npix": npix,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals": ngals,
        "apix": apix,
        "counterpart_pixel": counterpart_pixel,
        "counterpart_pixels": counterpart_pixels,
        "counterpart_zs": counterpart_zs,
        "counterpart_dzs": counterpart_dzs,
        "bright_siren_sky_marginalized": bool(sky_marginalized),
    }
