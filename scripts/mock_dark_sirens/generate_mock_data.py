#!/usr/bin/env python3
"""Generate end-to-end mock data for the dark-sirens pipeline.

The mock is intentionally simple and transparent:

* galaxies are isotropic on the sky and uniform in comoving volume;
* GW hosts are drawn from the complete catalog, before EM incompleteness;
* BBH masses/spins/redshift evolution use a POWER LAW + PEAK model that
  matches the inference model exactly (logistic-tapered primary-mass edges via
  ``sfilter_low``/``sfilter_high``, a Gaussian peak, ``S_low(m2)``-tapered mass
  ratio), with shared beta, truncated-Gaussian chi_eff, and gamma parameters;
* GW detectability is a semi-analytic network-SNR threshold;
* the observed EM survey is produced by applying a footprint, redshift/magnitude
  limits, and a smooth redshift-dependent completeness curve.

The HDF5 files are written in the formats consumed by ``darksirens_inference``
and ``darksirens_pixelate``/``load_survey``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import Flatw0waCDM
import astropy.units as u
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit

C_KM_S = 299_792.458

import numpy as np
import jax.numpy as jnp

@dataclass(frozen=True)
class PopulationConfig:
    """Fiducial POWER LAW + PEAK with shared beta, spin, and gamma."""

    alpha: float = 3.4
    mmin: float = 5.0
    mmax: float = 85.0
    # Logistic edge-taper widths (Msun), matching the inference PowerLaw
    # component (darksirens.gw.populations sfilter_low/high).  These are part
    # of the mass-model truth, so the inference model contains it exactly.
    dm_min: float = 3.0
    dm_max: float = 10.0
    peak_fraction: float = 0.10
    peak_mu: float = 35.0
    peak_sigma: float = 4.0
    beta: float = 1.3
    chi_mu: float = 0.0
    chi_sigma: float = 0.15
    gamma: float = 0.0


@dataclass(frozen=True)
class SurveyConfig:
    """Simple EM selection model for catalog incompleteness."""

    footprint_dec_min_deg: float = -40.0
    footprint_dec_max_deg: float = 80.0
    z_hard_max: float = 1.2
    magnitude_limit: float = 24.0
    z50: float = 0.75
    width: float = 0.12
    absolute_mag_mean: float = -21.0
    absolute_mag_sigma: float = 1.0
    redshift_error_floor: float = 0.0005
    redshift_error_slope: float = 0.0015
    delta: float = 0.0


def _build_cosmology(h0: float, om0: float, w0: float, wa: float) -> Flatw0waCDM:
    return Flatw0waCDM(H0=h0 * u.km / u.s / u.Mpc, Om0=om0, w0=w0, wa=wa)


def _cosmology_grids(cosmo: Flatw0waCDM, zmax: float, ngrid: int = 20_000) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, zmax, ngrid)
    dc = cosmo.comoving_distance(z).to_value(u.Mpc)
    dl = cosmo.luminosity_distance(z).to_value(u.Mpc)
    ez = cosmo.efunc(z)
    dvc_dz = 4.0 * np.pi * (C_KM_S / cosmo.H0.value) * dc**2 / ez
    vc_cdf = cumulative_trapezoid(dvc_dz, z, initial=0.0)
    vc_cdf /= vc_cdf[-1]
    return {"z": z, "dc": dc, "dl": dl, "dvc_dz": dvc_dz, "vc_cdf": vc_cdf}


def _sample_uniform_comoving_z(rng: np.random.Generator, grids: dict[str, np.ndarray], n: int) -> np.ndarray:
    return np.interp(rng.uniform(size=n), grids["vc_cdf"], grids["z"])


def _interp_dl(z: np.ndarray, grids: dict[str, np.ndarray]) -> np.ndarray:
    return np.interp(z, grids["z"], grids["dl"])


def _sample_sky(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    ra = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_dec = rng.uniform(-1.0, 1.0, n)
    dec = np.arcsin(sin_dec)
    return ra, dec


# --- Inference-matched mass model -------------------------------------------
# The primary-mass density mirrors the inference ``powerlaw+peak`` model so the
# fitted model contains the injected truth exactly (no hard-edge vs. tapered
# mismatch).  ``_sfilter_low``/``_sfilter_high`` are plain-numpy mirrors of
# darksirens.gw.populations.utils.sfilter_low/sfilter_high (kept numpy-only so
# this generator stays importable without jax / the darksirens package;
# tests/test_mock_generator_taper.py asserts they match the jax originals).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Normalisation grids mirror the inference defaults: mass on [1, 200] Msun and
# mass ratio on (0, 1] (cf. get_mass_grid / get_q_grid).  Densities are
# normalised on these grids so the stored ``pdraw`` is exactly the density the
# samplers draw from.
_MASS_NORM_GRID = np.linspace(1.0, 200.0, 1024)
_Q_NORM_GRID = np.linspace(1.0e-3, 1.0, 512)


def _sfilter_low(m: np.ndarray, m_min: float, dm: float) -> np.ndarray:
    """Logistic low-mass taper: 0 for m<=m_min, ramps to 1 over [m_min, m_min+dm]."""
    m = np.asarray(m, dtype=float)
    delta = m - m_min
    safe_d = np.where(delta > 0.0, delta, 1.0)
    safe_dm = dm if dm > 0.0 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # delta == dm at one grid point gives 1/0 here; the where-masks below
        # set the correct boundary value, so the intermediate inf is harmless.
        expo = np.clip(safe_dm / safe_d + safe_dm / (safe_d - safe_dm), -500.0, 500.0)
        S = 1.0 / (np.exp(expo) + 1.0)
    S = np.where(m <= m_min, 0.0, S)
    S = np.where(m >= m_min + dm, 1.0, S)
    return S


def _sfilter_high(m: np.ndarray, m_max: float, dm: float) -> np.ndarray:
    """Logistic high-mass taper: 1 below m_max-dm, ramps to 0 over [m_max-dm, m_max]."""
    m = np.asarray(m, dtype=float)
    delta = m_max - m
    safe_d = np.where(delta > 0.0, delta, 1.0)
    safe_dm = dm if dm > 0.0 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # delta == dm at one grid point gives 1/0 here; the where-masks below
        # set the correct boundary value, so the intermediate inf is harmless.
        expo = np.clip(safe_dm / safe_d + safe_dm / (safe_d - safe_dm), -500.0, 500.0)
        S = 1.0 / (np.exp(expo) + 1.0)
    S = np.where(m >= m_max, 0.0, S)
    S = np.where(m <= m_max - dm, 1.0, S)
    return S


def _powerlaw_unnorm(
    m: np.ndarray, alpha: float, mmin: float, mmax: float, dm_min: float, dm_max: float
) -> np.ndarray:
    """Tapered power law ``S_low(m) * S_high(m) * m**(-alpha)`` (un-normalised)."""
    m = np.asarray(m, dtype=float)
    S = _sfilter_low(m, mmin, dm_min) * _sfilter_high(m, mmax, dm_max)
    return S * np.power(m, -alpha)


def _powerlaw_pdf(
    m: np.ndarray, alpha: float, mmin: float, mmax: float, dm_min: float, dm_max: float
) -> np.ndarray:
    """Primary-mass power law with logistic inner-edge tapers, matching the
    inference ``PowerLaw`` component; normalised by trapezoid on the mass grid."""
    norm = _trapz(
        _powerlaw_unnorm(_MASS_NORM_GRID, alpha, mmin, mmax, dm_min, dm_max),
        _MASS_NORM_GRID,
    )
    unnorm = _powerlaw_unnorm(m, alpha, mmin, mmax, dm_min, dm_max)
    return unnorm / np.where(norm > 0.0, norm, 1.0)


def _peak_pdf(m: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Plain Gaussian peak normalised on the mass grid (untruncated, as in the
    inference ``Gaussian`` component)."""
    m = np.asarray(m, dtype=float)
    norm = _trapz(np.exp(-0.5 * ((_MASS_NORM_GRID - mu) / sigma) ** 2), _MASS_NORM_GRID)
    return np.exp(-0.5 * ((m - mu) / sigma) ** 2) / norm


def _truncnorm_pdf(x: np.ndarray, mu: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    from scipy.stats import norm as normal_dist

    z_norm = sigma * (normal_dist.cdf((hi - mu) / sigma) - normal_dist.cdf((lo - mu) / sigma))
    out = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * z_norm)
    return np.where((x >= lo) & (x <= hi), out, 0.0)


def _sample_powerlaw(rng: np.random.Generator, n: int, alpha: float, mmin: float, mmax: float) -> np.ndarray:
    u = rng.uniform(size=n)
    if np.isclose(alpha, 1.0):
        return mmin * (mmax / mmin) ** u
    a = 1.0 - alpha
    return (u * (mmax**a - mmin**a) + mmin**a) ** (1.0 / a)


def _sample_truncated_gaussian(
    rng: np.random.Generator, n: int, mu: float, sigma: float, lo: float, hi: float
) -> np.ndarray:
    """Draw ``n`` samples from ``N(mu, sigma)`` truncated to ``[lo, hi]`` (rejection)."""
    out = np.empty(int(n), dtype=float)
    filled = 0
    while filled < n:
        cand = rng.normal(mu, sigma, max(2 * (n - filled), 64))
        cand = cand[(cand >= lo) & (cand <= hi)]
        take = min(len(cand), n - filled)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def _sample_tapered_powerlaw(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    """Draw ``n`` primary masses from the tapered power law via rejection against
    the hard power law (accept with probability ``S_low(m) * S_high(m)``)."""
    out = np.empty(int(n), dtype=float)
    filled = 0
    while filled < n:
        cand = _sample_powerlaw(rng, max(2 * (n - filled), 64), pop.alpha, pop.mmin, pop.mmax)
        S = _sfilter_low(cand, pop.mmin, pop.dm_min) * _sfilter_high(cand, pop.mmax, pop.dm_max)
        cand = cand[rng.uniform(size=len(cand)) < S]
        take = min(len(cand), n - filled)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def _sample_powerlaw_peak_m1(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    """Primary mass from the tapered power law + Gaussian peak mixture, matching
    the inference ``powerlaw+peak`` density (``peak_fraction`` weight in the peak)."""
    use_peak = rng.uniform(size=n) < pop.peak_fraction
    m1 = np.empty(int(n), dtype=float)
    n_peak = int(use_peak.sum())
    if n - n_peak:
        m1[~use_peak] = _sample_tapered_powerlaw(rng, n - n_peak, pop)
    if n_peak:
        m1[use_peak] = _sample_truncated_gaussian(
            rng, n_peak, pop.peak_mu, pop.peak_sigma, _MASS_NORM_GRID[0], _MASS_NORM_GRID[-1]
        )
    return m1


def _sample_q_hard(rng: np.random.Generator, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """Inverse-CDF draw from the hard power law ``q**beta`` on ``[m_min/m1, 1]``."""
    qmin = np.clip(pop.mmin / m1, 1.0e-3, 1.0)
    u = rng.uniform(size=len(m1))
    b = pop.beta
    if np.isclose(b, -1.0):
        return qmin * (1.0 / qmin) ** u
    bp1 = b + 1.0
    return (u * (1.0 - qmin**bp1) + qmin**bp1) ** (1.0 / bp1)


def _sample_q(rng: np.random.Generator, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """Mass ratio from ``p(q|m1) propto S_low(m2) q**beta`` (m2 = q*m1), matching
    the inference pairing model: rejection against the hard ``q**beta`` proposal."""
    m1 = np.asarray(m1, dtype=float)
    out = np.empty_like(m1)
    todo = np.ones(len(m1), dtype=bool)
    for _ in range(10000):
        if not todo.any():
            break
        idx = np.where(todo)[0]
        cand = _sample_q_hard(rng, m1[idx], pop)
        acc = rng.uniform(size=len(cand)) < _sfilter_low(cand * m1[idx], pop.mmin, pop.dm_min)
        out[idx[acc]] = cand[acc]
        todo[idx[acc]] = False
    if todo.any():
        # Pathological m1 ~ m_min (measure ~0): fall back to the hard draw.
        idx = np.where(todo)[0]
        out[idx] = _sample_q_hard(rng, m1[idx], pop)
    return out


def _q_pdf(q: np.ndarray, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """``p(q | m1) propto S_low(m2) q**beta`` normalised over q on ``_Q_NORM_GRID``
    (matches the inference pairing model)."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    qg = _Q_NORM_GRID
    m2g = qg[None, :] * m1[:, None]                       # (N, Nq)
    unnorm_g = (qg[None, :] ** pop.beta) * _sfilter_low(m2g, pop.mmin, pop.dm_min)
    norm = _trapz(unnorm_g, qg, axis=-1)                  # (N,)
    unnorm = (q ** pop.beta) * _sfilter_low(q * m1, pop.mmin, pop.dm_min)
    out = unnorm / np.where(norm > 0.0, norm, 1.0)
    return np.where((q > 0.0) & (q <= 1.0), out, 0.0)


def _sample_chieff(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    vals = []
    while sum(map(len, vals)) < n:
        cand = rng.normal(pop.chi_mu, pop.chi_sigma, n)
        vals.append(cand[(cand >= -1.0) & (cand <= 1.0)])
    return np.concatenate(vals)[:n]


def _mass_spin_pdf(m1: np.ndarray, q: np.ndarray, chi: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    p_pl = _powerlaw_pdf(m1, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max)
    p_pk = _peak_pdf(m1, pop.peak_mu, pop.peak_sigma)
    p_m1 = (1.0 - pop.peak_fraction) * p_pl + pop.peak_fraction * p_pk
    p_chi = _truncnorm_pdf(chi, pop.chi_mu, pop.chi_sigma, -1.0, 1.0)
    return p_m1 * _q_pdf(q, m1, pop) * p_chi


def _network_snr(m1: np.ndarray, m2: np.ndarray, z: np.ndarray, dl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mchirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mchirp_det = mchirp * (1.0 + z)
    projection = rng.beta(2.0, 5.0, size=len(np.atleast_1d(m1))) ** 0.5
    rho_ref = 11.5
    return rho_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl) * projection


def _generate_complete_catalog(
    rng: np.random.Generator,
    n_galaxies: int,
    grids: dict[str, np.ndarray],
    survey: SurveyConfig,
) -> dict[str, np.ndarray]:
    z = _sample_uniform_comoving_z(rng, grids, n_galaxies)
    ra, dec = _sample_sky(rng, n_galaxies)
    abs_mag = rng.normal(survey.absolute_mag_mean, survey.absolute_mag_sigma, n_galaxies)
    dl_pc = _interp_dl(z, grids) * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    return {"ra": ra, "dec": dec, "z": z, "abs_mag": abs_mag, "app_mag": app_mag}


def _apply_survey_selection(
    rng: np.random.Generator,
    catalog: dict[str, np.ndarray],
    survey: SurveyConfig,
) -> np.ndarray:
    dec_deg = np.rad2deg(catalog["dec"])
    footprint = (dec_deg >= survey.footprint_dec_min_deg) & (dec_deg <= survey.footprint_dec_max_deg)
    depth = (catalog["z"] <= survey.z_hard_max) & (catalog["app_mag"] <= survey.magnitude_limit)
    completeness = expit((survey.z50 - catalog["z"]) / survey.width)
    return footprint & depth & (rng.uniform(size=len(catalog["z"])) < completeness)


def _pixelate_catalog(ra: np.ndarray, dec: np.ndarray, z: np.ndarray, dz: np.ndarray, w: np.ndarray, nside: int, marks: dict | None = None) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    zgals = np.full((npix, max_gals), 100.0)
    dzgals = np.full((npix, max_gals), 1.0)
    wgals = np.zeros((npix, max_gals))
    markgals = {name: np.zeros((npix, max_gals)) for name in (marks or {})}
    offsets = np.zeros(npix, dtype=np.int32)
    for i, p in enumerate(pix):
        j = offsets[p]
        zgals[p, j] = z[i]
        dzgals[p, j] = dz[i]
        wgals[p, j] = w[i]
        for name, arr in (marks or {}).items():
            markgals[name][p, j] = arr[i]
        offsets[p] += 1
    return {"zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals": counts, **markgals}


def _draw_events_until_detected(
    rng: np.random.Generator,
    nobs: int,
    catalog: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
    sky_weight_fn=None,
    sky_g_max: float = 1.0,
    mark_weight=None,
    mark_g_max: float = 1.0,
) -> dict[str, np.ndarray]:
    """Draw detected events from the host catalog.

    When ``sky_weight_fn(nx, ny, nz, z)`` is given, the detected sources follow a
    rate-modulated 3-D field ``g(n̂, z)`` via rejection on the host direction and
    redshift (accept ∝ ``g / sky_g_max``).  The host galaxy catalog and the
    selection injection set stay isotropic, so this injects a *pure source-rate*
    over-density for validating the sky models (recoverable even in GW-only
    mode).  ``sky_g_max`` must upper-bound ``g`` over the sampled domain.

    When ``mark_weight`` (a per-catalog-galaxy host efficiency ``h(m|eta)``) is
    given, hosts are additionally accepted ∝ ``mark_weight[host]/mark_g_max`` —
    a *marked-host* preference recoverable by ``--mark_model loglinear``.
    """
    kept: list[dict[str, np.ndarray]] = []
    while sum(len(x["z"]) for x in kept) < nobs:
        ntry = max(4 * nobs, 256)
        host_idx = rng.integers(0, len(catalog["z"]), ntry)
        z = catalog["z"][host_idx]
        ra = catalog["ra"][host_idx]
        dec = catalog["dec"][host_idx]
        dl = _interp_dl(z, grids)
        m1 = _sample_powerlaw_peak_m1(rng, ntry, pop)
        q = _sample_q(rng, m1, pop)
        m2 = q * m1
        chi = _sample_chieff(rng, ntry, pop)
        snr = _network_snr(m1, m2, z, dl, rng)
        det = snr >= snr_threshold
        if sky_weight_fn is not None:
            nx = np.cos(dec) * np.cos(ra)
            ny = np.cos(dec) * np.sin(ra)
            nz = np.sin(dec)
            g = sky_weight_fn(nx, ny, nz, z)
            accept = rng.uniform(size=len(ra)) < np.clip(g / sky_g_max, 0.0, 1.0)
            det = det & accept
        if mark_weight is not None:
            hh = mark_weight[host_idx]
            det = det & (rng.uniform(size=len(host_idx)) < np.clip(hh / mark_g_max, 0.0, 1.0))
        if np.any(det):
            kept.append({k: v[det] for k, v in dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr).items()})
    out = {k: np.concatenate([x[k] for x in kept])[:nobs] for k in kept[0]}
    return out


def _posterior_samples(
    rng: np.random.Generator,
    truth: dict[str, np.ndarray],
    nsamp: int,
    dL_fractional_uncertainty: float | None = None,
    m1det_fractional_uncertainty: float = 0.08,
    m2det_fractional_uncertainty: float = 0.10,
    chieff_uncertainty: float = 0.08,
    sky_uncertainty_deg: float | None = None,
) -> dict[str, np.ndarray]:
    nobs = len(truth["z"])
    arrays = {"ra": [], "dec": [], "dL": [], "m1det": [], "m2det": [], "chieff": [], "p_pe": []}
    for i in range(nobs):
        rho = truth["snr"][i]
        frac_dl = dL_fractional_uncertainty if dL_fractional_uncertainty is not None else np.clip(1.8 / rho, 0.08, 0.35)
        dl = rng.lognormal(np.log(truth["dl"][i]) - 0.5 * frac_dl**2, frac_dl, nsamp)
        sigma_ang = np.deg2rad(sky_uncertainty_deg if sky_uncertainty_deg is not None else np.clip(35.0 / rho, 1.0, 12.0))
        dra = rng.normal(0.0, sigma_ang / max(np.cos(truth["dec"][i]), 0.1), nsamp)
        ddec = rng.normal(0.0, sigma_ang, nsamp)
        arrays["ra"].append((truth["ra"][i] + dra) % (2.0 * np.pi))
        arrays["dec"].append(np.clip(truth["dec"][i] + ddec, -0.5 * np.pi, 0.5 * np.pi))
        m1det = truth["m1"][i] * (1.0 + truth["z"][i])
        m2det = truth["m2"][i] * (1.0 + truth["z"][i])
        arrays["m1det"].append(np.clip(rng.normal(m1det, m1det_fractional_uncertainty * m1det, nsamp), 2.0, None))
        arrays["m2det"].append(np.clip(rng.normal(m2det, m2det_fractional_uncertainty * m2det, nsamp), 1.0, None))
        arrays["chieff"].append(np.clip(rng.normal(truth["chi"][i], chieff_uncertainty, nsamp), -1.0, 1.0))
        arrays["dL"].append(dl)
        arrays["p_pe"].append(np.ones(nsamp))
    return {k: np.concatenate(v) for k, v in arrays.items()}


def _draw_selection_batch(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
    m1det_range: tuple[float, float] = (2.0, 200.0),
) -> dict[str, np.ndarray | int]:
    # Selection injections are drawn from a BROAD proposal, NOT from the
    # population: uniform in detector-frame primary mass, mass ratio, and spin
    # (z already broad via dV_c/dz).  Drawing masses from the population makes the
    # proposal a narrow envelope around the fiducial, so the importance-sampling
    # estimate of mu(theta) is only well-conditioned near the truth: away from it
    # (steep mass slopes, large gamma) p_pop/pdraw blows up, the effective sample
    # size collapses, mu is under-resolved, and the recovered population is biased
    # (gamma rails).  The #94 mass-taper made the population envelope even more
    # peaked, tipping this over.  A broad proposal keeps mu well-conditioned over
    # the whole prior.  pdraw is the proposal density in the canonical
    # (m1det, q, dL) basis (with the isotropic 1/4pi sky factor, as before).
    m1lo, m1hi = m1det_range
    m1det = rng.uniform(m1lo, m1hi, ndraw)
    q = rng.uniform(0.0, 1.0, ndraw)
    chi = rng.uniform(-1.0, 1.0, ndraw)
    z = _sample_uniform_comoving_z(rng, grids, ndraw)
    ra, dec = _sample_sky(rng, ndraw)
    dl = _interp_dl(z, grids)
    m1src = m1det / (1.0 + z)
    m2src = q * m1src
    snr = _network_snr(m1src, m2src, z, dl, rng)
    det = snr >= snr_threshold

    pz = np.interp(z, grids["z"], grids["dvc_dz"]) / jnp.trapezoid(grids["dvc_dz"], grids["z"])
    ddldz = np.interp(z, grids["z"], np.gradient(grids["dl"], grids["z"]))
    # m1det is drawn directly (no source->detector Jacobian); p(dL) = p_z(z)|dz/ddL|.
    p_dL = pz / np.maximum(ddldz, 1.0e-300)
    p_draw = (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * p_dL / (4.0 * np.pi)
    p_draw = np.maximum(p_draw, 1.0e-300)

    return {
        "m1det": m1det[det],
        "m2det": (q * m1det)[det],
        "m1src": m1src[det],
        "m2src": m2src[det],
        "dL": dl[det],
        "chieff": chi[det],
        "ra": ra[det],
        "dec": dec[det],
        "pdraw": p_draw[det],
        "Ndraw": ndraw,
        "n_detected": int(det.sum()),
    }


def _selection_injections(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
    batch_size: int,
    target_detections: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | int]:
    chunks: list[dict[str, np.ndarray | int]] = []
    n_proposed = 0
    n_detected = 0
    keys = ["m1det", "m2det", "m1src", "m2src", "dL", "chieff", "ra", "dec", "pdraw"]

    while n_proposed < ndraw:
        n_batch = min(batch_size, ndraw - n_proposed)
        chunk = _draw_selection_batch(rng, n_batch, grids, pop, snr_threshold)
        chunks.append(chunk)
        n_proposed += int(chunk["Ndraw"])
        n_detected += int(chunk["n_detected"])
        if verbose:
            print(f"  selection batch: proposed={n_proposed:,}/{ndraw:,}, detected={n_detected:,}")
        if target_detections is not None and n_detected >= target_detections:
            break

    if chunks:
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    else:
        arrays = {key: np.array([], dtype=float) for key in keys}

    return {
        **arrays,
        "Ndraw": n_proposed,
        "n_detected": n_detected,
    }


def _galaxy_count_from_density(n0: float, delta: float, grids: dict[str, np.ndarray]) -> int:
    density_weighted_volume = jnp.trapezoid(grids["dvc_dz"] * (1.0 + grids["z"]) ** delta, grids["z"])
    return max(1, int(round(n0 * density_weighted_volume)))


def write_mock_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    pop = PopulationConfig()
    survey = SurveyConfig(
        z50=args.survey_z50,
        width=args.survey_width,
        delta=args.galaxy_density_delta,
    )
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cosmo = _build_cosmology(args.H0, args.Om0, args.w0, args.wa)
    zmax = float(args.zmax)
    grids = _cosmology_grids(cosmo, zmax)
    n_galaxies = (
        _galaxy_count_from_density(args.n0, args.galaxy_density_delta, grids)
        if args.n0 is not None
        else args.n_galaxies
    )
    if args.verbose and args.n0 is not None:
        print(f"Derived {n_galaxies:,} galaxies from n0={args.n0:g} Mpc^-3 over z=[0, {zmax:g}].")

    complete = _generate_complete_catalog(rng, n_galaxies, grids, survey)

    # Optional injected host-mark preference: assign per-pixel-coherent marks to
    # every galaxy and draw GW hosts ∝ h(m|eta_true) = exp(Σ_k eta_k m_k).  The
    # marks are z-independent (so the inference's z-centring is a no-op) but
    # spatially coherent per HEALPix pixel, so the preference imprints a
    # *recoverable* angular over-density of hosts in high-mark pixels.  Purely
    # per-galaxy-random marks average out per pixel and leave eta unconstrained.
    eta_true = {"mark_logmstar": float(args.mark_eta_logmstar),
                "mark_logssfr": float(args.mark_eta_logssfr)}
    with_marks = bool(args.with_marks) or any(v != 0.0 for v in eta_true.values())
    mark_weight, mark_g_max = None, 1.0
    if with_marks:
        npix_m = hp.nside2npix(args.nside)
        pix_all = hp.ang2pix(args.nside, np.pi / 2.0 - complete["dec"], complete["ra"])
        log_h = np.zeros(len(complete["z"]))
        for name, eta in eta_true.items():
            field = rng.normal(0.0, args.mark_pixel_sigma, npix_m)   # per-pixel coherent
            complete[name] = field[pix_all] + rng.normal(0.0, args.mark_scatter, len(pix_all))
            log_h += eta * complete[name]
        mark_weight = np.exp(log_h - log_h.max())     # max = 1 -> exact rejection bound
        if args.verbose:
            print(f"Injecting host-mark preference eta={eta_true} "
                  f"(pixel_sigma={args.mark_pixel_sigma}, scatter={args.mark_scatter}).")

    observed = _apply_survey_selection(rng, complete, survey)
    zerr = survey.redshift_error_floor + survey.redshift_error_slope * (1.0 + complete["z"])
    weights = np.ones(observed.sum())
    mark_obs = ({name: complete[name][observed] for name in eta_true} if with_marks else None)
    pixelated = _pixelate_catalog(
        complete["ra"][observed], complete["dec"][observed], complete["z"][observed],
        zerr[observed], weights, args.nside, marks=mark_obs,
    )

    # Build the injected 3-D source-rate field g(n̂, z) as a product of optional
    # factors: a (possibly z-evolving) dipole and a localized (ra, dec, z0) blob.
    # Selection injections stay isotropic; only the detected events are reweighted.
    sky_factors = []
    sky_g_max = 1.0
    injected_sky: dict = {}

    dip_amp = float(getattr(args, "sky_dipole_amp", 0.0))
    if dip_amp != 0.0:
        ra_d = np.deg2rad(args.sky_dipole_ra_deg)
        dec_d = np.deg2rad(args.sky_dipole_dec_deg)
        dvec = dip_amp * np.array([
            np.cos(dec_d) * np.cos(ra_d),
            np.cos(dec_d) * np.sin(ra_d),
            np.sin(dec_d),
        ])
        z_pivot = getattr(args, "sky_dipole_z_pivot", None)

        def _g_dipole(nx, ny, nz, z, dvec=dvec, z_pivot=z_pivot):
            # w(z) ramps 0→1 over [0, z_pivot] (constant dipole when z_pivot=None).
            w = 1.0 if z_pivot is None else np.clip(z / z_pivot, 0.0, 1.0)
            return 1.0 + w * (nx * dvec[0] + ny * dvec[1] + nz * dvec[2])

        sky_factors.append(_g_dipole)
        sky_g_max *= 1.0 + dip_amp
        injected_sky["dipole"] = {
            "d": [float(x) for x in dvec],
            "z_pivot": None if z_pivot is None else float(z_pivot),
        }
        if args.verbose:
            print(f"Injecting dipole d={dvec.tolist()} z_pivot={z_pivot}.")

    blob_amp = float(getattr(args, "sky_blob_amp", 0.0))
    if blob_amp != 0.0:
        ra_b = np.deg2rad(args.sky_blob_ra_deg)
        dec_b = np.deg2rad(args.sky_blob_dec_deg)
        n0 = np.array([
            np.cos(dec_b) * np.cos(ra_b),
            np.cos(dec_b) * np.sin(ra_b),
            np.sin(dec_b),
        ])
        sigma_ang = np.deg2rad(args.sky_blob_sigma_deg)
        z0, sigma_z = float(args.sky_blob_z0), float(args.sky_blob_sigma_z)

        def _g_blob(nx, ny, nz, z, n0=n0, B=blob_amp, sa=sigma_ang, z0=z0, sz=sigma_z):
            cosang = np.clip(nx * n0[0] + ny * n0[1] + nz * n0[2], -1.0, 1.0)
            ang = np.arccos(cosang)
            return 1.0 + B * np.exp(-0.5 * (ang / sa) ** 2 - 0.5 * ((z - z0) / sz) ** 2)

        sky_factors.append(_g_blob)
        sky_g_max *= 1.0 + blob_amp
        injected_sky["blob"] = {
            "amp": blob_amp, "ra_deg": args.sky_blob_ra_deg,
            "dec_deg": args.sky_blob_dec_deg, "z0": z0,
            "sigma_deg": args.sky_blob_sigma_deg, "sigma_z": sigma_z,
        }
        if args.verbose:
            print(f"Injecting 3-D blob amp={blob_amp:g} at "
                  f"(ra,dec,z0)=({args.sky_blob_ra_deg},{args.sky_blob_dec_deg},{z0}).")

    quad_amp = float(getattr(args, "sky_quadrupole_amp", 0.0))
    if quad_amp != 0.0:
        ra_q = np.deg2rad(args.sky_quadrupole_ra_deg)
        dec_q = np.deg2rad(args.sky_quadrupole_dec_deg)
        aq = np.array([
            np.cos(dec_q) * np.cos(ra_q),
            np.cos(dec_q) * np.sin(ra_q),
            np.sin(dec_q),
        ])

        def _g_quad(nx, ny, nz, z, aq=aq, Q=quad_amp):
            mu = nx * aq[0] + ny * aq[1] + nz * aq[2]      # cos(angle to axis)
            return 1.0 + Q * 0.5 * (3.0 * mu**2 - 1.0)     # axisymmetric ℓ=2 (P2)

        sky_factors.append(_g_quad)
        sky_g_max *= 1.0 + quad_amp                        # P2 max is 1 (at n̂=±â)
        injected_sky["quadrupole"] = {
            "amp": quad_amp, "ra_deg": args.sky_quadrupole_ra_deg,
            "dec_deg": args.sky_quadrupole_dec_deg,
        }
        if args.verbose:
            print(f"Injecting axisymmetric quadrupole amp={quad_amp:g} about "
                  f"(ra,dec)=({args.sky_quadrupole_ra_deg},{args.sky_quadrupole_dec_deg}).")

    if sky_factors:
        def sky_weight_fn(nx, ny, nz, z, _factors=tuple(sky_factors)):
            g = np.ones_like(np.asarray(z, dtype=float))
            for fkt in _factors:
                g = g * fkt(nx, ny, nz, z)
            return g
    else:
        sky_weight_fn = None

    truth = _draw_events_until_detected(
        rng, args.nobs, complete, grids, pop, args.snr_threshold,
        sky_weight_fn=sky_weight_fn, sky_g_max=sky_g_max,
        mark_weight=mark_weight, mark_g_max=mark_g_max,
    )
    post = _posterior_samples(
        rng,
        truth,
        args.nsamp,
        dL_fractional_uncertainty=args.dL_fractional_uncertainty,
        m1det_fractional_uncertainty=args.m1det_fractional_uncertainty,
        m2det_fractional_uncertainty=args.m2det_fractional_uncertainty,
        chieff_uncertainty=args.chieff_uncertainty,
        sky_uncertainty_deg=args.sky_uncertainty_deg,
    )
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)
    selection_target_detections = args.selection_target_detections
    if args.selection_per_observation_factor is not None:
        selection_target_detections = int(np.ceil(args.selection_per_observation_factor * args.nobs))
    sel = _selection_injections(
        rng,
        args.ndraw,
        grids,
        pop,
        args.snr_threshold,
        args.selection_batch_size,
        target_detections=selection_target_detections,
        verbose=args.verbose,
    )

    inv_pdraw = 1.0 / np.asarray(sel["pdraw"])
    selection_neff = float(inv_pdraw.sum() ** 2 / np.square(inv_pdraw).sum()) if len(inv_pdraw) else 0.0

    metadata = {
        "seed": args.seed,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": args.w0, "wa": args.wa},
        "population": asdict(pop),
        "survey": asdict(survey),
        "snr_threshold": args.snr_threshold,
        "pop_model_for_inference": "powerlaw+peak",
        "shared_beta_for_inference": True,
        "shared_spin_for_inference": True,
        "shared_gamma_for_inference": True,
        "injected_marks": ({"eta": eta_true,
                            "pixel_sigma": float(args.mark_pixel_sigma),
                            "scatter": float(args.mark_scatter)}
                           if with_marks else None),
    }

    complete_path = out / "mock_galaxy_catalog_complete.h5"
    with h5py.File(complete_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Complete isotropic, uniform-in-comoving-volume mock galaxy catalog before EM incompleteness."
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in complete.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    raw_path = out / "mock_survey_raw.h5"
    with h5py.File(raw_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Observed mock survey after footprint, magnitude, redshift, and completeness cuts."
        f.attrs["metadata_json"] = json.dumps(metadata)
        f.create_dataset("TARGET_RA", data=np.rad2deg(complete["ra"][observed]), compression="gzip", shuffle=True)
        f.create_dataset("TARGET_DEC", data=np.rad2deg(complete["dec"][observed]), compression="gzip", shuffle=True)
        f.create_dataset("Z", data=complete["z"][observed], compression="gzip", shuffle=True)
        f.create_dataset("ZERR", data=zerr[observed], compression="gzip", shuffle=True)
        f.create_dataset("WEIGHT", data=weights, compression="gzip", shuffle=True)

    pixel_path = out / f"catalog_pixelated_nside_{args.nside}.h5"
    with h5py.File(pixel_path, "w") as f:
        f.attrs["nside"] = int(args.nside)
        f.attrs["mock_data"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in pixelated.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    gw_path = out / "mock_gw_events.h5"
    with h5py.File(gw_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(args.nobs)
        f.attrs["nsamp"] = int(args.nsamp)
        f.attrs["pe_cosmology_H0"] = float(args.H0)
        f.attrs["pe_cosmology_Om0"] = float(args.Om0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["injected_sky_dipole"] = json.dumps(
            injected_sky.get("dipole", {}).get("d")  # back-compat: dipole vector or None
        )
        f.attrs["injected_sky"] = json.dumps(injected_sky or None)
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in post.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)
        truth_group = f.create_group("truth")
        for key, val in truth.items():
            truth_group.create_dataset(key, data=val)

    sel_path = out / "mock_gw_selection.h5"
    with h5py.File(sel_path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])
        f.attrs["Neff"] = selection_neff
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["cosmology_H0"] = float(args.H0)
        f.attrs["cosmology_Om0"] = float(args.Om0)
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key in ["m1det", "m2det", "m1src", "m2src", "dL", "chieff", "ra", "dec", "pdraw"]:
            f.create_dataset(key, data=sel[key], compression="gzip", shuffle=True)

    print("Mock dark-sirens data written:")
    print(f"  complete catalog : {complete_path} ({n_galaxies:,} galaxies)")
    print(f"  observed survey  : {raw_path} ({observed.sum():,} galaxies retained)")
    print(f"  pixelated survey : {pixel_path} (nside={args.nside})")
    print(f"  GW posteriors    : {gw_path} ({args.nobs} events x {args.nsamp} samples)")
    print(f"  GW selection     : {sel_path} ({sel['n_detected']:,}/{sel['Ndraw']:,} detected injections, Neff={selection_neff:.1f})")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/mock_dark_sirens", help="Output directory for HDF5 products.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-galaxies", type=_positive_int, default=None, help="Number of complete-catalog galaxies to draw when --n0 is omitted.")
    parser.add_argument("--n0", type=_positive_float, default=None, help="Comoving galaxy density in Mpc^-3; overrides --n-galaxies when provided.")
    parser.add_argument("--nobs", type=_positive_int, default=8)
    parser.add_argument("--nsamp", type=_positive_int, default=512)
    parser.add_argument("--ndraw", type=_positive_int, default=80_000)
    parser.add_argument("--nside", type=_positive_int, default=16)
    parser.add_argument("--zmax", type=_positive_float, default=0.08)
    parser.add_argument("--H0", type=_positive_float, default=67.74)
    parser.add_argument("--Om0", type=_positive_float, default=0.3075)
    parser.add_argument("--w0", type=float, default=-1.0, help="CPL dark-energy equation-of-state value today.")
    parser.add_argument("--wa", type=float, default=0.0, help="CPL dark-energy evolution parameter.")
    parser.add_argument("--snr-threshold", type=_positive_float, default=8.0)
    parser.add_argument("--survey-z50", type=float, default=SurveyConfig.z50)
    parser.add_argument("--survey-width", type=_positive_float, default=SurveyConfig.width)
    parser.add_argument("--galaxy-density-delta", type=float, default=SurveyConfig.delta)
    parser.add_argument("--selection-batch-size", type=_positive_int, default=50_000)
    selection_targets = parser.add_mutually_exclusive_group()
    selection_targets.add_argument("--selection-target-detections", type=_positive_int, default=None)
    selection_targets.add_argument("--selection-per-observation-factor", type=_positive_float, default=None)
    parser.add_argument("--dL-fractional-uncertainty", type=_positive_float, default=None)
    parser.add_argument("--m1det-fractional-uncertainty", type=_positive_float, default=0.08)
    parser.add_argument("--m2det-fractional-uncertainty", type=_positive_float, default=0.10)
    parser.add_argument("--chieff-uncertainty", type=_positive_float, default=0.08)
    parser.add_argument("--sky-uncertainty-deg", type=_positive_float, default=None)
    parser.add_argument("--sky-dipole-amp", type=float, default=0.0,
                        help="Inject a source-rate dipole |d| in [0,1) into the detected events "
                             "(g(n)=1+n.d); 0 (default) leaves the sky isotropic.")
    parser.add_argument("--sky-dipole-ra-deg", type=float, default=0.0,
                        help="Right ascension (deg) of the injected sky-dipole direction.")
    parser.add_argument("--sky-dipole-dec-deg", type=float, default=0.0,
                        help="Declination (deg) of the injected sky-dipole direction.")
    parser.add_argument("--sky-dipole-z-pivot", type=_positive_float, default=None,
                        help="If set, the dipole amplitude ramps as min(z/z_pivot, 1) "
                             "(z-evolving dipole = a 3-D structure); unset = constant dipole.")
    parser.add_argument("--sky-blob-amp", type=float, default=0.0,
                        help="Inject a localized 3-D over-density g=1+B*exp(-ang^2/2sig^2 "
                             "-(z-z0)^2/2sigz^2) into the detected events; 0 = none.")
    parser.add_argument("--sky-blob-ra-deg", type=float, default=0.0,
                        help="Right ascension (deg) of the injected 3-D blob centre.")
    parser.add_argument("--sky-blob-dec-deg", type=float, default=0.0,
                        help="Declination (deg) of the injected 3-D blob centre.")
    parser.add_argument("--sky-blob-z0", type=_positive_float, default=0.5,
                        help="Redshift centre z0 of the injected 3-D blob.")
    parser.add_argument("--sky-blob-sigma-deg", type=_positive_float, default=15.0,
                        help="Angular width (deg) of the injected 3-D blob.")
    parser.add_argument("--sky-blob-sigma-z", type=_positive_float, default=0.1,
                        help="Redshift width of the injected 3-D blob.")
    parser.add_argument("--sky-quadrupole-amp", type=float, default=0.0,
                        help="Inject an axisymmetric quadrupole g=1+Q*(3mu^2-1)/2 "
                             "(mu=cos angle to the axis); 0 = none. Tests the l=2 channel.")
    parser.add_argument("--sky-quadrupole-ra-deg", type=float, default=0.0,
                        help="Right ascension (deg) of the quadrupole symmetry axis.")
    parser.add_argument("--sky-quadrupole-dec-deg", type=float, default=0.0,
                        help="Declination (deg) of the quadrupole symmetry axis.")
    parser.add_argument("--with-marks", action="store_true",
                        help="Write per-galaxy marks (mark_logmstar, mark_logssfr) to the "
                             "pixelated catalog so --mark_model loglinear can read them. "
                             "Auto-enabled when any --mark-eta-* is nonzero.")
    parser.add_argument("--mark-eta-logmstar", type=float, default=0.0,
                        help="Injected TRUE host-preference coefficient eta for logmstar: "
                             "GW hosts are drawn proportional to exp(eta*mark). 0 = none. "
                             "Recoverable by --mark_model loglinear --marks logmstar.")
    parser.add_argument("--mark-eta-logssfr", type=float, default=0.0,
                        help="Injected TRUE host-preference coefficient eta for logssfr.")
    parser.add_argument("--mark-pixel-sigma", type=float, default=1.0,
                        help="Std of the per-HEALPix-pixel coherent mark field — the spatial "
                             "structure that makes the injected eta recoverable (survives the "
                             "inference's per-z-bin mark centring).")
    parser.add_argument("--mark-scatter", type=float, default=0.3,
                        help="Std of the within-pixel per-galaxy mark scatter.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.n0 is None and args.n_galaxies is None:
        args.n0 = 1.0e-3
    return args


if __name__ == "__main__":
    write_mock_data(parse_args())
