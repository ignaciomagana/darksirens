"""Per-event host draws for the host-sum PE estimator.

Why this exists
---------------
The default PE term importance-samples the redshift integral with the event's
PE posterior as the proposal and EVALUATES the catalog prior at each sample::

    Zhat_i = (1/N) sum_s  p_cat(z_s) p_pop J / p_pe(s),   z_s = z(dL_s, H0)

With spectroscopic kernels (sigma_eff ~ 1e-5) and ~5e-4 galaxy spacing p_cat is
a COMB, and only the few samples that happen to land on a tooth contribute.
Which samples those are changes discontinuously as H0 slides z_s across the
teeth, so Zhat_i(H0) is deterministic but ROUGH: measured 8.9 nats between
adjacent H0 points at 0.05 spacing, with the gradient reversing sign 65.5% of
the time (a smooth curve gives ~0, white noise ~0.5).

The host sum reorders the integral.  p_cat is itself a sum over catalogued
galaxies, so::

    Z_i = sum_g wbar_g INT dz N(z; z_g, sigma_g) L_i(dL(z,H0)) p_pop J
        ~ sum_g wbar_g L_i(dL(z_g,H0)) p_pop(z_g) J(z_g,H0)      [delta limit]

Every term is a FIXED galaxy at a FIXED z_g; H0 enters only through the smooth
map dL(z_g, H0).  Nothing enters or leaves the sum and no term ever falls in a
gap, so Z_i(H0) is as smooth as L_i.  Measured sign-flip fraction: 0.008.

This module builds the host draws.  Rather than the exhaustive sum over every
catalogued galaxy -- which is ragged (median 85,602 hosts per event, max
357,909) and would need a padded per-event layout -- it DRAWS ``n_draw`` hosts
per event with probability wbar_g.  That is the same estimator with the same
smoothness (the draws are frozen across H0, see below), it matches the
exhaustive sum to 0.002-0.009 nats at n_draw = 4000, and it is RECTANGULAR:
``(nEvents, n_draw)``, which keeps every downstream shape static for XLA.

Two invariants this file exists to protect
------------------------------------------
1. COMMON RANDOM NUMBERS.  The draws are made ONCE, at load time, and reused at
   every H0.  Redrawing per likelihood call would paint fresh Monte-Carlo noise
   on every grid point and reproduce exactly the roughness the host sum removes.
2. A COSMOLOGY-RANGE-SAFE z-WINDOW.  Hosts are selected once, from the union of
   z(dL) over the CORNERS OF THE WHOLE SAMPLED COSMOLOGY BOX -- not from a
   fiducial cosmology with a hand-set pad.  A pad sized at one cosmology is
   only "probably big enough": as the sampler moves, the galaxies the integral
   needs shift, the frozen host list does not, and the host set is then wrong
   in a cosmology-dependent way near the edge -- a moving boundary, which is
   the defect the host sum exists to remove, reintroduced one level down.
   Scanning the corners makes the window a bound rather than a guess, and it
   must cover EVERY sampled cosmological parameter (H0, Om0, w0, wa): opening
   up the dark-energy sector moves z(dL) just as H0 does, so an H0-only window
   would silently under-cover such a run.
"""
from __future__ import annotations

import numpy as np


def _event_rows(sample_to_unique, event_idx, nsamp):
    """Compact catalog rows touched by one event's PE samples."""
    s = event_idx * nsamp
    return np.unique(np.asarray(sample_to_unique[s:s + nsamp], dtype=np.int64))


def cosmology_corners(cosmo_bounds):
    """Corners of the sampled cosmology box.

    ``cosmo_bounds`` maps each of H0/Om0/w0/wa to ``(lo, hi)``; a parameter
    that is fixed contributes ``lo == hi`` and so does not multiply the corner
    count.  At most 16 corners, and z(dL) is monotone in each parameter over
    the ranges these runs use, so the extremes of the window live at corners.
    """
    import itertools

    order = ("H0", "Om0", "w0", "wa")
    axes = []
    for k in order:
        lo, hi = cosmo_bounds[k]
        axes.append((lo,) if lo == hi else (lo, hi))
    return [dict(zip(order, c)) for c in itertools.product(*axes)]


def host_z_window(dL_event, cosmo_bounds, z_max=None):
    """(z_lo, z_hi) covering one event's dL range over the WHOLE cosmology box.

    The window is a bound, not a guess: every galaxy any sampled cosmology
    could need is inside it, so the host set is frozen without ever becoming
    cosmology-dependent.  See invariant 2 in the module docstring.
    """
    from astropy.cosmology import Flatw0waCDM
    import astropy.units as u

    d_lo = float(np.min(dL_event))
    d_hi = float(np.max(dL_event))
    zg = np.linspace(1e-6, 3.0, 30000)

    z_lo, z_hi = np.inf, -np.inf
    for c in cosmology_corners(cosmo_bounds):
        cos = Flatw0waCDM(H0=c["H0"], Om0=c["Om0"], w0=c["w0"], wa=c["wa"])
        dLg = cos.luminosity_distance(zg).to(u.Mpc).value
        z_lo = min(z_lo, float(np.interp(d_lo, dLg, zg)))
        z_hi = max(z_hi, float(np.interp(d_hi, dLg, zg)))
    if z_max is not None:
        z_hi = min(z_hi, float(z_max))
    return max(0.0, z_lo), z_hi


def build_host_draws(
    zgals,
    wgals,
    ngals,
    sample_to_unique,
    dL,
    cosmo_bounds,
    nEvents,
    nsamp,
    n_draw=4000,
    z_max=None,
    seed=0,
):
    """Draw ``n_draw`` hosts per event from the catalog, with weights wbar_g.

    Parameters
    ----------
    zgals, wgals : (N_rows, N_max) padded catalog arrays.
    ngals : (N_rows,) real galaxy count per row.
    sample_to_unique : (nEvents*nsamp,) PE sample -> compact catalog row.
    dL : (nEvents, nsamp) PE luminosity distances.  Distance is what the data
        measure, so the window is built from it and mapped to redshift at every
        corner of the cosmology box -- no fiducial cosmology enters.
    cosmo_bounds : {"H0": (lo, hi), "Om0": ..., "w0": ..., "wa": ...} over the
        SAMPLED range; fixed parameters pass lo == hi.
    n_draw : hosts drawn per event.  4000 matches the exhaustive sum to
        0.002-0.009 nats on this dataset.
    z_max : optional hard cap (the population z horizon).
    seed : fixes the draws.  They are frozen for the run; see invariant 1.

    Returns
    -------
    z_host : (nEvents, n_draw) float64 host redshifts.
    log_w_host : (nEvents, n_draw) float64 log of the per-draw weight,
        ``-log n_draw`` for a proper draw and ``-inf`` for an event with no
        hosts in the window (which makes that event's log Z = -inf, the same
        contract the sampled path uses for an all-masked event).
    n_host_avail : (nEvents,) number of catalogued hosts the draws came from,
        for diagnostics.
    """
    zg = np.asarray(zgals)
    wg = np.asarray(wgals)
    ng = np.asarray(ngals, dtype=np.int64)

    z_host = np.zeros((nEvents, n_draw), dtype=np.float64)
    log_w_host = np.full((nEvents, n_draw), -np.inf, dtype=np.float64)
    n_avail = np.zeros(nEvents, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for i in range(nEvents):
        rows = _event_rows(sample_to_unique, i, nsamp)
        if rows.size == 0:
            continue
        z_lo, z_hi = host_z_window(dL[i], cosmo_bounds, z_max=z_max)
        if not (z_hi > z_lo):
            continue

        zr = zg[rows]
        wr = wg[rows]
        real = np.arange(zr.shape[1])[None, :] < ng[rows][:, None]
        m = real & (zr >= z_lo) & (zr <= z_hi) & (wr > 0.0)
        if not m.any():
            continue

        z_i = zr[m]
        w_i = wr[m]
        n_avail[i] = z_i.size
        p = w_i / w_i.sum()
        # Drawn ONCE here, never per likelihood call (invariant 1).
        idx = rng.choice(z_i.size, size=n_draw, p=p)
        z_host[i] = z_i[idx]
        log_w_host[i] = -np.log(float(n_draw))

    return z_host, log_w_host, n_avail


def build_pe_likelihood_table(dL, p_pe, nEvents, nsamp, bandwidth=0.15):
    """Per-event weighted-KDE reconstruction of the PE likelihood L_i(dL).

    The host sum needs L_i at the host redshifts' distances, which is not where
    any stored sample sits -- so the likelihood must be reconstructed rather
    than read off.  Samples are drawn from the POSTERIOR ~ L_i(dL) p_pe(dL), so
    weighting each by 1/p_pe recovers the likelihood shape.

    ``bandwidth`` multiplies the weighted sample std.  It is the one genuinely
    new approximation the host sum introduces; measured insensitive over
    0.08-0.30 (a 3.75x range moved the validation residual by 0.006 nats), so
    0.15 is a mid-range default rather than a tuned value.

    Returns (nodes, log_w, inv_h) with shapes (nEvents, nsamp), (nEvents,
    nsamp), (nEvents,) -- the mixture is evaluated in the likelihood.
    """
    d = np.asarray(dL, dtype=np.float64).reshape(nEvents, nsamp)
    pw = np.asarray(p_pe, dtype=np.float64).reshape(nEvents, nsamp)

    w = np.where(pw > 0.0, 1.0 / np.maximum(pw, 1e-300), 0.0)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    mu = (w * d).sum(axis=1, keepdims=True)
    var = (w * (d - mu) ** 2).sum(axis=1)
    h = float(bandwidth) * np.sqrt(np.maximum(var, 1e-12))

    with np.errstate(divide="ignore"):
        log_w = np.log(np.where(w > 0.0, w, 0.0))
    return d, log_w, 1.0 / np.maximum(h, 1e-300)