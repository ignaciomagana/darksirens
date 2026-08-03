#!/usr/bin/env python3
"""Generate end-to-end mock data for the dark-sirens pipeline.

The mock is intentionally simple and transparent:

* galaxies are isotropic on the sky and uniform in comoving volume;
* GW hosts are drawn from the complete catalog, before EM incompleteness;
* BBH masses/spins/redshift evolution use a POWER LAW + PEAK model that
  matches the inference model exactly (logistic-tapered primary-mass edges via
  ``sfilter_low``/``sfilter_high``, a Gaussian peak, ``S_low(m2)``-tapered mass
  ratio), with shared beta, truncated-Gaussian chi_eff, and gamma parameters;
* GW detectability is a threshold on a RECORDED signal-to-noise ratio,
  ``rho_obs = rho_opt(theta) + N(0, sigma_rho) >= rho_th``, and every other
  measurement width is a function of that same recorded number -- the
  all-observable measurement family described under "THE MEASUREMENT FAMILY"
  below.  The same measurement the threshold saw is what the posterior samples
  condition on, and no recorded value or posterior sample is ever clipped;
* the observed EM survey is produced by applying a footprint, redshift/magnitude
  limits, and a smooth redshift-dependent completeness curve, and it sees a
  REALISED photo-z ``z_obs`` rather than the true redshift it declares an error
  on.

The HDF5 files are written in the formats consumed by ``darksirens_inference``
and ``darksirens_pixelate``/``load_survey``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
from astropy.cosmology import Flatw0waCDM
import astropy.units as u
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit

C_KM_S = 299_792.458

from functools import partial

import jax
# Distances/redshifts (and the dV_c/dz weights behind pdraw) need double
# precision, matching the rest of darksirens; enable x64 before any jax op.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

SELECTION_PROPOSALS = ("population", "uniform", "population+uniform")

@dataclass(frozen=True)
class PopulationConfig:
    """Registry-fixed POWER LAW + PEAK with shared beta, spin, and gamma.

    Keep these defaults aligned with
    ``get_fixed_population_params("powerlaw+peak")`` so mock generation and
    ``--fix_population true`` inference use the same truth without a JSON
    override.  The curated registry has ``$v_1$=0.10``, i.e. final mixture
    fractions ``w_PL=0.10`` and ``w_G=0.90`` under stick breaking.
    """

    alpha: float = 2.3
    mmin: float = 5.0
    mmax: float = 80.0
    # Logistic edge-taper widths (Msun), matching the inference PowerLaw
    # component (darksirens.gw.populations sfilter_low/high).  These are part
    # of the mass-model truth, so the inference model contains it exactly.
    dm_min: float = 3.0
    dm_max: float = 10.0
    peak_fraction: float = 0.90
    peak_mu: float = 35.0
    peak_sigma: float = 5.0
    beta: float = 1.0
    chi_mu: float = 0.0
    chi_sigma: float = 0.10
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
_trapz = _trapezoid

# Normalisation grids mirror the inference defaults EXACTLY: mass on [1, 200]
# Msun with N_MASS=500 nodes and mass ratio on [0, 1] with N_Q=200 nodes, i.e.
# the same linspaces darksirens.gw.populations.utils.get_mass_grid() /
# get_q_grid() build from the default NormalizationGridSettings (n_mass=500,
# n_q=200, m_lo=1, m_hi=200, q_lo=0, q_hi=1).  Densities are normalised on these
# grids so the stored ``pdraw`` is exactly the density the samplers draw from AND
# is bit-for-bit the same trapezoid the inference uses -- the mock's mass/pair
# component densities then match the inference model to machine precision (cf.
# tests/test_mock_generator_taper.py::test_pairing_matches_inference_component_densities).
_MASS_NORM_GRID = np.linspace(1.0, 200.0, 500)
_Q_NORM_GRID = np.linspace(0.0, 1.0, 200)

# Fallback pairing secondary-mass taper for the Gaussian-peak component.  The
# inference pairs every mass component SEPARATELY: a component without an
# ``m_min_spec`` (the Gaussian peak) falls back to (M_LO=1.0, dm=0.01) in
# MixtureModel.component_densities (darksirens/gw/populations/base.py:366,
# darksirens/gw/populations/utils.py:14), i.e. an essentially un-tapered m2
# floor at 1 Msun -- NOT the power law's (m_min=5, dm_min=3).  The mock must use
# the same per-component pairing so p_inference/p_draw stays O(1).
_PAIR_M_LO = 1.0
_PAIR_DM = 0.01


def _sfilter_low(m: np.ndarray, m_min, dm) -> np.ndarray:
    """Logistic low-mass taper: 0 for m<=m_min, ramps to 1 over [m_min, m_min+dm].

    ``m_min`` and ``dm`` may be scalars or per-element arrays (broadcasting with
    ``m``); the per-lane form is used by the per-component pairing sampler, which
    applies (m_min=5, dm=3) to power-law lanes and (m_min=1, dm=0.01) to peak
    lanes in the same vectorised call.
    """
    m = np.asarray(m, dtype=float)
    delta = m - m_min
    safe_d = np.where(delta > 0.0, delta, 1.0)
    safe_dm = np.where(dm > 0.0, dm, 1.0)
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


def _sample_powerlaw_peak_m1(
    rng: np.random.Generator, n: int, pop: PopulationConfig, return_component: bool = False
):
    """Primary mass from the tapered power law + Gaussian peak mixture, matching
    the inference ``powerlaw+peak`` density (``peak_fraction`` weight in the peak).

    With ``return_component=True`` also returns the boolean ``use_peak`` lane mask
    (True where the primary was drawn from the Gaussian peak).  The mask selects
    the per-component pairing taper in :func:`_sample_q` so the joint (m1, q) draw
    matches the inference's per-component pairing (peak lanes pair against the
    (m_min=1, dm=0.01) fallback, power-law lanes against (m_min, dm_min))."""
    use_peak = rng.uniform(size=n) < pop.peak_fraction
    m1 = np.empty(int(n), dtype=float)
    n_peak = int(use_peak.sum())
    if n - n_peak:
        m1[~use_peak] = _sample_tapered_powerlaw(rng, n - n_peak, pop)
    if n_peak:
        m1[use_peak] = _sample_truncated_gaussian(
            rng, n_peak, pop.peak_mu, pop.peak_sigma, _MASS_NORM_GRID[0], _MASS_NORM_GRID[-1]
        )
    if return_component:
        return m1, use_peak
    return m1


def _sample_q_hard(rng: np.random.Generator, m1: np.ndarray, pop: PopulationConfig, m_min=None) -> np.ndarray:
    """Inverse-CDF draw from the hard power law ``q**beta`` on ``[m_min/m1, 1]``.

    ``m_min`` defaults to ``pop.mmin`` (the power-law pairing floor); pass a
    per-lane array (e.g. ``_PAIR_M_LO`` for peak lanes) to widen the proposal
    support down to ``m2 = m_min`` so the rejection sampler covers the whole
    per-component pairing support."""
    if m_min is None:
        m_min = pop.mmin
    qmin = np.clip(m_min / m1, 1.0e-3, 1.0)
    u = rng.uniform(size=len(m1))
    b = pop.beta
    if np.isclose(b, -1.0):
        return qmin * (1.0 / qmin) ** u
    bp1 = b + 1.0
    return (u * (1.0 - qmin**bp1) + qmin**bp1) ** (1.0 / bp1)


def _sample_q(rng: np.random.Generator, m1: np.ndarray, pop: PopulationConfig, use_peak=None) -> np.ndarray:
    """Mass ratio from ``p(q|m1) propto S_low(m2) q**beta`` (m2 = q*m1), matching
    the inference pairing model: rejection against the hard ``q**beta`` proposal.

    ``use_peak`` (optional per-lane boolean mask, from
    :func:`_sample_powerlaw_peak_m1`) selects the per-component pairing taper:
    peak lanes use (``_PAIR_M_LO``, ``_PAIR_DM``) = (1, 0.01) and power-law lanes
    use (``pop.mmin``, ``pop.dm_min``), exactly as the inference pairs each mass
    component separately.  ``use_peak=None`` reproduces the historical all-power-
    law behaviour (every lane tapered at ``pop.mmin``)."""
    m1 = np.asarray(m1, dtype=float)
    out = np.empty_like(m1)
    if use_peak is None:
        m_min_lane = np.full(m1.shape, pop.mmin)
        dm_lane = np.full(m1.shape, pop.dm_min)
    else:
        use_peak = np.asarray(use_peak, dtype=bool)
        m_min_lane = np.where(use_peak, _PAIR_M_LO, pop.mmin)
        dm_lane = np.where(use_peak, _PAIR_DM, pop.dm_min)
    todo = np.ones(len(m1), dtype=bool)
    for _ in range(10000):
        if not todo.any():
            break
        idx = np.where(todo)[0]
        cand = _sample_q_hard(rng, m1[idx], pop, m_min=m_min_lane[idx])
        acc = rng.uniform(size=len(cand)) < _sfilter_low(cand * m1[idx], m_min_lane[idx], dm_lane[idx])
        out[idx[acc]] = cand[acc]
        todo[idx[acc]] = False
    if todo.any():
        # Pathological m1 ~ m_min (measure ~0): fall back to the hard draw.
        idx = np.where(todo)[0]
        out[idx] = _sample_q_hard(rng, m1[idx], pop, m_min=m_min_lane[idx])
    return out


def _pair_pdf(q: np.ndarray, m1: np.ndarray, m_min: float, dm: float, beta: float) -> np.ndarray:
    """Normalised conditional pairing density ``p(q | m1)`` for ONE mass component.

    Replicates the inference ``PairingModel.__call__`` /
    ``PowerLawPairing._eval_unnorm`` exactly (darksirens/gw/populations/base.py:205-228,
    parametric.py:661-673):

    * unnormalised ``q**beta * S_low(q*m1; m_min, dm)`` with a HARD zero for
      ``m2 = q*m1 < m_min`` and the ``q>0`` safe-power guard;
    * per-``m1`` normalisation by trapezoid over ``_Q_NORM_GRID`` (== get_q_grid())
      with the row-maximum factored out of the quadrature.  Factoring the row max
      is a backward-pass conditioning trick in the inference; here it only changes
      the association order, so the forward density equals ``unnorm / trapz(unnorm)``
      up to ULP while staying bit-comparable to the inference quadrature.

    ``m_min``/``dm`` are the pairing floor for the component: (``pop.mmin``,
    ``pop.dm_min``) for the power law, (``_PAIR_M_LO``, ``_PAIR_DM``) for the peak.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    qg = _Q_NORM_GRID
    m2g = qg[None, :] * m1[:, None]                       # (N, Nq)
    # Row grid of the unnormalised pairing (safe q>0 power + hard m2<m_min zero).
    safe_qg = np.where(qg > 0.0, qg, 1.0)
    pg = np.where(qg > 0.0, safe_qg ** beta, 0.0)[None, :] * _sfilter_low(m2g, m_min, dm)
    pg = np.where(m2g < m_min, 0.0, pg)
    scale = np.max(pg, axis=-1, keepdims=True)            # row max (base.py:221)
    scale_s = np.where(scale > 0.0, scale, 1.0)
    n_sc = _trapz(pg / scale_s, qg, axis=-1)              # (N,) scaled norm
    scale_m = scale_s[:, 0]
    # Same unnormalised form at the requested q.
    safe_q = np.where(q > 0.0, q, 1.0)
    p = np.where(q > 0.0, safe_q ** beta, 0.0) * _sfilter_low(q * m1, m_min, dm)
    p = np.where(q * m1 < m_min, 0.0, p)
    out = np.where(n_sc > 0.0, (p / scale_m) / np.where(n_sc > 0.0, n_sc, 1.0), 0.0)
    return np.where((q > 0.0) & (q <= 1.0), out, 0.0)


def _q_pdf(q: np.ndarray, m1: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """``p(q | m1) propto S_low(m2) q**beta`` for the POWER-LAW pairing floor
    (``pop.mmin``, ``pop.dm_min``); thin wrapper over :func:`_pair_pdf` kept for
    backward compatibility with the pinned sampler/normalisation tests."""
    return _pair_pdf(q, m1, pop.mmin, pop.dm_min, pop.beta)


def _sample_chieff(rng: np.random.Generator, n: int, pop: PopulationConfig) -> np.ndarray:
    vals = []
    while sum(map(len, vals)) < n:
        cand = rng.normal(pop.chi_mu, pop.chi_sigma, n)
        vals.append(cand[(cand >= -1.0) & (cand <= 1.0)])
    return np.concatenate(vals)[:n]


def _mass_spin_pdf(m1: np.ndarray, q: np.ndarray, chi: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """Joint source-frame mass-ratio-spin density ``p(m1, q, chi)``, matching the
    inference ``powerlaw+peak`` mixture's per-component pairing EXACTLY.

    The inference (MixtureModel.component_densities, base.py:324-378) pairs each
    mass component with its OWN secondary-mass floor: the power law with
    (``m_min``, ``dm_min``) and the Gaussian peak with the fallback
    (``M_LO``=1, dm=0.01).  The mixture is therefore

        p(m1,q) = w_PL p_PL(m1) pair(q|m1; m_min, dm_min)
                + w_G  p_G (m1) pair(q|m1; 1.0,  0.01),

    NOT a single tapered pairing applied to the whole primary mixture (the old
    ``p_m1 * _q_pdf`` form, which mis-tapered the peak's secondaries at m_min=5
    and drove p_inference/p_draw -> e^31 for injections with m2 just above 5).
    """
    p_pl = _powerlaw_pdf(m1, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max)
    p_pk = _peak_pdf(m1, pop.peak_mu, pop.peak_sigma)
    pair_pl = _pair_pdf(q, m1, pop.mmin, pop.dm_min, pop.beta)
    pair_pk = _pair_pdf(q, m1, _PAIR_M_LO, _PAIR_DM, pop.beta)
    p_chi = _truncnorm_pdf(chi, pop.chi_mu, pop.chi_sigma, -1.0, 1.0)
    p_mass_pair = (1.0 - pop.peak_fraction) * p_pl * pair_pl + pop.peak_fraction * p_pk * pair_pk
    return p_mass_pair * p_chi


# =============================================================================
# THE MEASUREMENT FAMILY -- all-observable, sequential, no latent-dependent width
# =============================================================================
# Every measurement width is a function of the RECORDED signal-to-noise ratio and
# of nothing else.  This is the family of Fishbach, Holz & Farr (2018)
# arXiv:1805.10270 eqs. 29-31, in the form released as ``GWMockCat`` (Farah,
# Edelman, Zevin, Fishbach, Maria Ezquiaga, Farr & Holz 2023, ApJ 955, 107,
# arXiv:2301.00834, App. A; git.ligo.org/amanda.farah/GWMockCat, CC0):
#
#     rho_obs = rho_opt(theta) + N(0, sigma_rho)      detection: rho_obs >= rho_th
#     sigma_x = A_x * (rho_th / rho_obs)              for every other channel
#
# and the measurement basis is (ln Mc_det, ln q, rho, chi_eff, ra, dec).  Because
# there is no projection latent, ``rho_opt`` is an exact function of (Mc_det, dL),
# so that basis is a BIJECTION of (m1det, m2det, dL, chi_eff, ra, dec) and dL is
# DERIVED from (Mc_det, rho) rather than measured on its own.
#
# Why the SNR has to be the distance coordinate.  Recording ``rho_obs`` AND
# measuring ``dL`` through its own channel would leave the factor
# ``N(rho_obs; rho_opt(theta), sigma_rho)`` in the true likelihood -- a
# theta-dependent term the consuming likelihood, which only ever sees
# (m1det, q, dL, chi_eff, sky) samples and ``p_pe``, cannot represent.  GWMockCat
# does not do it either: it samples (Mc, eta, rho, Theta) and derives
# ``dL = Theta rho_opt/rho`` (``transforms.py::redshift``).
#
# Why not a truth-scaled width.  ``N(obs; m, f m)`` carries a theta-dependent
# normalisation ``1/(f m)``, so its flat-prior posterior is skewed by
# construction and the ensemble mean of a non-linear functional of it need not
# equal the functional at the truth.  Measured on the gws-agn matched-mock
# campaign with the EXACT per-event posterior of that family, the detected-set
# score identity E[C] = E[A] was violated at 11.3 sigma
# (``(C - A)_pop = -1.274e-3 +- 0.113e-3``); under this family the same statistic
# is ``+4.44e-4 +- 3.19e-4``, i.e. 1.4 sigma.  The same mechanism, in the sky
# channel alone, was worth ``-0.49 +- 0.08 km/s/Mpc`` in recovered H0.
#
# Fishbach & Holz (2020) arXiv:1905.12669 App. B add a ``0.2 z/(1+z)`` term to
# the chirp-mass width.  It is deliberately NOT adopted: it is a function of the
# source's LATENT redshift, which is the defect above.  GWMockCat drops it too.
MEASUREMENT_FAMILY = "all-observable-v3"
P_PE_BASIS = ("rho/(dL m1det q): the PE prior, flat in "
              "(ln Mc_det, ln q, rho, chi_eff), expressed in darksirens' "
              "canonical (m1det, q, dL, chi_eff) integration basis; "
              "normalised to mean 1 per event")
SNR_REF_DEFAULT = 11.5
# The sky-width amplitude scale.  ``sigma_ang = clip(SKY_A_DEG / rho_sigma, ...)``
# with ``rho_sigma`` the amplitude on THIS reference, which is not necessarily the
# reference the detection statistic uses.  Converting the recorded ``rho_obs``
# from the detection scale to the sky scale is one factor,
# ``rho_sigma = (SNR_REF_SIGMA / snr_ref) * rho_obs``; at the default
# ``--snr-ref 11.5`` the factor is 1 and the width is the historical
# ``clip(35 deg / rho_obs, 1, 12)``.  At the gws-agn campaign's detection scale
# ``snr_ref = 6.278`` it is 1.83165, i.e. ``clip(19.1069 deg / rho_obs, 1, 12)``,
# which realises the same width distribution.  Carry the factor explicitly or the
# sky widths silently move by 1.83x.  (Delta Omega ~ rho^-2: Fairhurst 2009,
# arXiv:0908.2356; Berry et al. 2015, ApJ 804, 114, arXiv:1411.6934.)
SNR_REF_SIGMA = 11.5
SKY_A_DEG = 35.0
SKY_CLIP_DEG = (1.0, 12.0)
# GWMockCat ``uncert_default``: sigma_rho = 1 additive on the SNR, and the width
# of each other channel at rho_obs = the SNR threshold.
SIGMA_RHO_DEFAULT = 1.0        # uncert_default["snr"]     (Fishbach+2018 eq. 29)
A_MC_DEFAULT = 0.08            # uncert_default["mc"]
A_CHI_DEFAULT = 0.20           # GWMockCat parser.py --Xeff_uncert
# GWMockCat measures the SYMMETRIC mass ratio, sigma_eta = 0.022 (8/rho_obs), on
# [0, 0.25].  That observation's truncated-normal normalisation depends on eta
# itself, and with a beta=1 pairing against a 35 Msun peak the detected eta
# median sits 0.17 sigma from the boundary -- so the censoring would be active
# for the MEDIAN event.  ``ln q`` with an unbounded Gaussian and a prior
# truncation at q <= 1 is the same physics with no theta-dependent
# normalisation anywhere.  Converting, ``d eta/d ln q = q(1-q)/(1+q)^3`` gives
# ``A_Q(q) = 0.022 (1+q)^3 / (q (1-q))`` = 0.63 at q = 0.75; the independent
# GW150914 anchor (q = 0.86 +0.14/-0.21 at network rho ~ 24; Abbott et al. 2016,
# PRL 116, 061102, arXiv:1602.03840) gives 0.20 * 24/8 = 0.60, which is adopted.
A_Q_DEFAULT = 0.60
COS_DEC_FLOOR = 0.1            # the RA-width floor, unchanged
CHIEFF_RANGE = (-1.0, 1.0)     # PE PRIOR support -- never a clip on the DATA


@dataclass(frozen=True)
class MeasurementConfig:
    """Every constant of the measurement family, in one object.

    Passed identically to the event loop, the selection injections and the
    posterior sampler, so the detection rule and the PE cannot drift apart.
    """

    snr_ref: float = SNR_REF_DEFAULT
    snr_threshold: float = 8.0
    sigma_rho: float = SIGMA_RHO_DEFAULT
    a_mc: float = A_MC_DEFAULT
    a_q: float = A_Q_DEFAULT
    a_chi: float = A_CHI_DEFAULT
    sky_a_deg: float = SKY_A_DEG
    # A data-independent constant sky width, in degrees.  ``None`` uses the
    # SNR-derived width; an explicit value is passed through unchanged and is
    # still a function of data (a constant is), so the fixed-width sky
    # posterior stays exact.
    sky_uncertainty_deg: float | None = None


def _snr_from_detector_frame(m1det, m2det, dl, snr_ref: float = SNR_REF_DEFAULT):
    """The optimal (projection-free, noise-free) SNR from DETECTOR-frame masses.

    Detector-frame masses are what lets this be evaluated on OBSERVED
    quantities, where no redshift is available."""
    mchirp_det = (m1det * m2det) ** (3.0 / 5.0) / (m1det + m2det) ** (1.0 / 5.0)
    return snr_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)


# --- the bijection (Mc_det, q, rho) <-> (m1det, m2det, dL) -------------------
def _mc_of_m1q(m1det, q):
    """Detector-frame chirp mass, ``Mc = m1 q^(3/5) / (1+q)^(1/5)``."""
    return m1det * q ** 0.6 / (1.0 + q) ** 0.2


def _m1_of_mc_q(mc_det, q):
    """Inverse of :func:`_mc_of_m1q`."""
    return mc_det * (1.0 + q) ** 0.2 / q ** 0.6


def _rho_opt_of_mc_dl(mc_det, dl, snr_ref: float = SNR_REF_DEFAULT):
    """:func:`_snr_from_detector_frame` written in the chirp mass, so the
    bijection is explicit."""
    return snr_ref * (mc_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)


def _dl_of_mc_rho(mc_det, rho, snr_ref: float = SNR_REF_DEFAULT):
    """The inverse: the distance at which a source of chirp mass ``mc_det`` has
    optimal SNR ``rho``.  This is what makes rho the distance coordinate."""
    return 1000.0 * snr_ref * (mc_det / 30.0) ** (5.0 / 6.0) / rho


def _sigma_ang_from_rho_obs(rho_obs, meas: MeasurementConfig):
    """Sky width from the RECORDED SNR alone (see ``SNR_REF_SIGMA``)."""
    if meas.sky_uncertainty_deg is not None:
        return np.deg2rad(np.full(np.shape(rho_obs), float(meas.sky_uncertainty_deg)))
    rho_sigma = (SNR_REF_SIGMA / meas.snr_ref) * np.asarray(rho_obs, dtype=float)
    return np.deg2rad(np.clip(meas.sky_a_deg / rho_sigma, *SKY_CLIP_DEG))


def _measurement_widths(rho_obs, meas: MeasurementConfig) -> dict[str, np.ndarray]:
    """Every measurement width, from ``rho_obs`` and nothing else.

    This is the property that makes the stored fixed-width posterior the EXACT
    flat-prior posterior of the recorded measurement, and it is checkable from
    the file alone: each stored width must be recomputable, bitwise, from the
    stored ``obs_rho``.
    """
    rho = np.asarray(rho_obs, dtype=float)
    k = meas.snr_threshold / rho
    return {"sig_lnmc": meas.a_mc * k,
            "sig_lnq": meas.a_q * k,
            "sig_chieff": meas.a_chi * k,
            "sigma_ang": _sigma_ang_from_rho_obs(rho, meas)}


def _trunc_norm(rng, loc, scale, lo, hi, size=None):
    """Exact truncated-normal draws by inverse CDF, ``Phi^-1(Phi(a) + u(Phi(b)-Phi(a)))``.

    ``lo``/``hi`` may be +-inf.  This is the ONLY truncated sampler in the
    generator, and it is used exclusively on the PE PRIOR -- never to censor a
    recorded value.  Clipping a *sample* would put a point mass at the boundary,
    which is not a density at all and which ``p_pe`` cannot describe.
    """
    from scipy.special import ndtr, ndtri
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    a = ndtr((np.asarray(lo, dtype=float) - loc) / scale)
    b = ndtr((np.asarray(hi, dtype=float) - loc) / scale)
    u = rng.random(size if size is not None else np.broadcast(loc, scale).shape)
    return loc + scale * ndtri(np.clip(a + u * (b - a), 1e-300, 1.0 - 1e-16))


def _measure(rng, m1det, m2det, chi, dl, ra, dec, meas: MeasurementConfig, *,
             need_sky: bool = True):
    """ONE measurement of a source.  The draw order IS the model; do not reorder.

    1. ``rho_obs = rho_opt(theta) + N(0, sigma_rho)`` -- the detection statistic,
       and the only channel whose width is not derived from data.
    2. every other width, from ``rho_obs`` alone.
    3. ``ln Mc_obs``, ``ln q_obs``, ``chi_eff_obs``, then ``dec_obs``, then
       ``ra_obs``.  Declination is measured BEFORE right ascension and the RA
       width is formed from the declination ALREADY RECORDED,
       ``sigma_ra = sigma_ang / max(cos dec_obs, 0.1)`` -- the same expression
       the posterior uses, so the stored posterior is the exact posterior of the
       recorded measurement.  A ``cos(dec_true)`` width would be O(1) wrong near
       the poles, exactly where the 12 deg cap also bites.

    NOTHING recorded here is clipped or truncated.  Clipping the data makes the
    measurement model censored -- the likelihood acquires a theta-dependent
    normalisation ``P(obs = boundary | theta) = 1 - Phi(...)`` and the exact
    flat-prior posterior stops being a simple normal.  Physical ranges
    (``q <= 1``, ``rho > 0``, ``|chi_eff| <= 1``, ``|dec| <= pi/2``) are
    statements about the PARAMETER and are imposed on the PE PRIOR instead (see
    :func:`_posterior_samples`).  A recorded value is therefore allowed to lie
    outside the physical range, and about 2 % of an isotropic sky sits within
    the 12 deg sky cap of a pole, so censored declinations were not a corner
    case.

    ``need_sky=False`` draws only ``rho_obs`` -- which is the whole detection
    rule -- and is what the selection injections use.
    """
    n = np.asarray(dl, dtype=float).shape[0]
    m1det = np.asarray(m1det, dtype=float)
    m2det = np.asarray(m2det, dtype=float)
    q_true = m2det / m1det
    mc_det = _mc_of_m1q(m1det, q_true)
    rho_true = _rho_opt_of_mc_dl(mc_det, dl, meas.snr_ref)
    rho_obs = rho_true + meas.sigma_rho * rng.normal(size=n)
    out = {"obs_rho": rho_obs,
           "obs_snr": rho_obs,          # alias: the stored detection statistic
           "snr_true": rho_true,
           "obs_sigma_rho": np.full(n, float(meas.sigma_rho))}
    if not need_sky:
        return out
    # ``rho_obs`` can be <= 0 for a proposal that is never detected, which would
    # make the widths infinite.  Form them on a floored copy: every row that can
    # be detected has rho_obs >= the threshold, so no stored width is affected
    # and each one stays bitwise recomputable from the stored ``obs_rho``.
    w = _measurement_widths(np.maximum(rho_obs, 1.0e-3), meas)
    out["obs_sig_lnmc"] = w["sig_lnmc"]
    out["obs_sig_lnq"] = w["sig_lnq"]
    out["obs_sig_chieff"] = w["sig_chieff"]
    out["obs_sigma_ang"] = w["sigma_ang"]
    out["obs_lnmc"] = np.log(mc_det) + w["sig_lnmc"] * rng.normal(size=n)
    out["obs_lnq"] = np.log(q_true) + w["sig_lnq"] * rng.normal(size=n)
    out["obs_chieff"] = np.asarray(chi, dtype=float) + w["sig_chieff"] * rng.normal(size=n)
    obs_dec = np.asarray(dec, dtype=float) + w["sigma_ang"] * rng.normal(size=n)
    sig_ra = w["sigma_ang"] / np.maximum(np.cos(obs_dec), COS_DEC_FLOOR)
    out["obs_dec"] = obs_dec
    out["obs_sig_ra"] = sig_ra
    out["obs_ra"] = (np.asarray(ra, dtype=float) + sig_ra * rng.normal(size=n)) % (2.0 * np.pi)
    # Derived point estimates -- DIAGNOSTIC ONLY; the PE never reads them.  For a
    # proposal whose rho_obs is near zero the widths blow up and these overflow;
    # those rows are never detected and nothing downstream reads them, so the
    # overflow is ignored rather than clipped (clipping would misrepresent what
    # was recorded).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q_o = np.exp(out["obs_lnq"])
        mc_o = np.exp(out["obs_lnmc"])
        m1_o = _m1_of_mc_q(mc_o, q_o)
        out["obs_m1det"] = m1_o
        out["obs_m2det"] = q_o * m1_o
        out["obs_dL"] = _dl_of_mc_rho(mc_o, np.maximum(rho_obs, 1.0e-3), meas.snr_ref)
    return out


def _detect_on_observation(rng, m1src, m2src, z, dl, ra, dec, chi,
                           meas: MeasurementConfig, *, need_sky: bool = True):
    """Detection as a DETERMINISTIC FUNCTION OF ONE RECORDED NUMBER.

    Draws one measurement per source and thresholds its recorded ``rho_obs``.
    Returns ``(detected, observation)``; the caller must hand that same
    observation to the posterior, which is the whole point.

    A population likelihood evaluates
    ``prod_i [int p(d_i|theta) p(theta|Lambda) dtheta] / mu^N``, which is the
    correct detected-set likelihood only when detection is a deterministic
    function of the data, so that ``1[det(d_i)] = 1`` for observed events and
    drops out of the numerator.  Thresholding a latent amplitude instead leaves
    an extra ``P(det|theta)`` INSIDE each event's integral, i.e. the mock is not
    drawn from the model the inference assumes.
    """
    obs = _measure(rng, m1src * (1.0 + z), m2src * (1.0 + z), chi, dl, ra, dec,
                   meas, need_sky=need_sky)
    return obs["obs_rho"] >= meas.snr_threshold, obs


def _p_pe(m1det, q, dL, meas: MeasurementConfig):
    """The PE PRIOR density in darksirens' canonical ``(m1det, q, dL, chi_eff)``
    basis -- the number the consuming likelihood divides each sample by.

    ``darksirens/inference/utils.py`` ("Integration variables") fixes the basis:
    posterior samples and detected injections are integrated in
    ``(m1det, q, dL, chi_eff, sky pixel)``, and any proposal density divided out
    "must be expressed per unit m1det, per unit q, per Mpc of dL".

    The PE prior is flat in ``y = (ln Mc_det, ln q, rho, chi_eff)``.  With
    ``x = (m1det, q, dL, chi_eff)`` and ``A = 3/(5q) - 1/(5(1+q))``,

        d(ln Mc, ln q, rho)/d(m1det, q, dL) =
            [ 1/m1det          A            0       ]
            [ 0                1/q          0       ]
            [ (5/6) rho/m1det  (5/6) rho A  -rho/dL ]

    whose determinant is ``-(rho/dL)/(m1det q)``, so

        p_pe  ~  rho / (dL m1det q)  ~  Mc_det^(5/6) / (dL^2 m1det q).

    ``chi_eff`` maps identically and the sky prior is flat in (ra, dec), so
    neither contributes a factor.  darksirens renormalises ``p_pe`` per event,
    so only the shape matters; the stored column is normalised to mean 1 per
    event.
    """
    mc_det = _mc_of_m1q(m1det, q)
    rho = _rho_opt_of_mc_dl(mc_det, dL, meas.snr_ref)
    return rho / (dL * m1det * q)


def _generate_complete_catalog(
    rng: np.random.Generator,
    n_galaxies: int,
    grids: dict[str, np.ndarray],
    survey: SurveyConfig,
) -> dict[str, np.ndarray]:
    """The complete catalog, with TWO redshift columns.

    ``z`` is the true redshift: it drives the host draw, the event truth and
    every generative step.  ``z_obs = z + N(0, zerr(z))`` is what the survey --
    and therefore the likelihood -- actually sees, with ``zerr`` the very error
    the survey block declares.

    Realising the declared error is not cosmetic.  Copying the true redshifts
    into the survey block while declaring a photo-z width makes the likelihood
    smooth a comb that carries no error, so darksirens' per-galaxy kernel
    ``g(z) N(z; z_g, sigma_g)/Z(z_g)`` is NOT the Bayesian posterior for the
    host's true redshift.  Measured on the gws-agn matched mock, that internal
    inconsistency was ``+6.383e-4 +- 0.836e-4``, i.e. 7.6 sigma; realising the
    error left ``-5.49e-5 +- 9.19e-5``, 0.60 sigma.

    ``z_obs`` may be negative for a galaxy at ``z ~ 0`` and is deliberately NOT
    clipped: clipping re-introduces a censored observation, which is the same
    defect in a different channel.  The realised count is reported instead.
    """
    z = _sample_uniform_comoving_z(rng, grids, n_galaxies)
    ra, dec = _sample_sky(rng, n_galaxies)
    abs_mag = rng.normal(survey.absolute_mag_mean, survey.absolute_mag_sigma, n_galaxies)
    dl_pc = _interp_dl(z, grids) * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    zerr = _catalog_zerr(z, survey)
    z_obs = z + zerr * rng.normal(size=n_galaxies)
    return {"ra": ra, "dec": dec, "z": z, "z_obs": z_obs,
            "abs_mag": abs_mag, "app_mag": app_mag}


def _catalog_zerr(z: np.ndarray, survey: SurveyConfig) -> np.ndarray:
    """The declared -- and now realised -- photo-z width at redshift ``z``."""
    return survey.redshift_error_floor + survey.redshift_error_slope * (1.0 + np.asarray(z, dtype=float))


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
    """Pack the survey into fixed-width HEALPix rows, padded at ``z = 100``.

    The three pixelated arrays are float64 EXPLICITLY, and the writer asserts
    it.  ``darksirens.redshift.completion._kde_dndz_obs`` builds the truncated
    kernel mass in the catalog's storage dtype and clamps it at ``1e-300``,
    while the kernel itself is promoted to the float64 z grid.  For a padded
    slot the mass underflows to exactly 0 -- and ``1e-300`` is NOT representable
    in float32, so the clamp cannot rescue it, ``0/0 = NaN``, the
    ``* real_gal`` mask cannot remove a NaN, and every pixel row carrying any
    padding comes back all-NaN.  The NaN reaches the survey-global normaliser
    and the likelihood returns -inf in every cell of every grid.  Cheap
    insurance against a failure mode that has cost a full dataset
    regeneration once.
    """
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    zgals = np.full((npix, max_gals), 100.0, dtype=np.float64)
    dzgals = np.full((npix, max_gals), 1.0, dtype=np.float64)
    wgals = np.zeros((npix, max_gals), dtype=np.float64)
    markgals = {name: np.zeros((npix, max_gals), dtype=np.float64) for name in (marks or {})}
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
    meas: MeasurementConfig | None = None,
    rejected_keep: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Draw detected events from the host catalog.

    One measurement is drawn per source and the threshold is applied to its
    recorded ``rho_obs``; that same measurement is carried forward under
    ``obs_*`` keys so ``_posterior_samples`` conditions on the very data the
    threshold saw.  See :func:`_detect_on_observation` for why detection has to
    be a function of the data, and :func:`_measure` for the measurement family.

    ``snr_threshold`` must equal ``meas.snr_threshold``; it is kept as an
    explicit argument because it is the generator's headline knob.

    The detected fraction of this rule is NOT the detected fraction of a
    true-parameter cut at the same ``--snr-ref``: dropping the projection latent
    raises it by about 5.75x (measured on the deep mock of the gws-agn
    matched-mock experiment), so ``--snr-ref`` needs recalibration if a detected
    population is to be held comparable to a pre-existing one.  For that mock
    ``--snr-ref 6.278`` reproduced the old arm's detected fraction.

    ``rejected_keep`` records up to that many proposals that FAILED the SNR
    threshold, with their own ``obs_rho`` and ``snr_true``.  Every stored
    rejection must fail the threshold on its own recorded number, which is what
    closes the loop on "detection is a deterministic function of the data";
    together with the detected set it also exhibits the Malmquist scatter a
    noise-free statistic cannot produce (detections whose TRUE SNR is below
    threshold, and rejections whose TRUE SNR is above it -- both exactly zero
    for a true-parameter cut).

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
    meas = meas or MeasurementConfig(snr_threshold=snr_threshold)
    if float(meas.snr_threshold) != float(snr_threshold):
        raise ValueError("snr_threshold and meas.snr_threshold disagree: "
                         f"{snr_threshold} vs {meas.snr_threshold}")
    kept: list[dict[str, np.ndarray]] = []
    rejected: list[dict[str, np.ndarray]] = []
    n_rejected = 0
    while sum(len(x["z"]) for x in kept) < nobs:
        ntry = max(4 * nobs, 256)
        host_idx = rng.integers(0, len(catalog["z"]), ntry)
        z = catalog["z"][host_idx]
        ra = catalog["ra"][host_idx]
        dec = catalog["dec"][host_idx]
        dl = _interp_dl(z, grids)
        m1, use_peak = _sample_powerlaw_peak_m1(rng, ntry, pop, return_component=True)
        q = _sample_q(rng, m1, pop, use_peak=use_peak)
        m2 = q * m1
        chi = _sample_chieff(rng, ntry, pop)
        det, obs = _detect_on_observation(rng, m1, m2, z, dl, ra, dec, chi, meas)
        snr = obs["obs_snr"]
        if rejected_keep and n_rejected < rejected_keep:
            # Rejected by the SNR RULE specifically -- before the rate/sky/mark
            # acceptance factors below, which are a different rule.
            miss = ~det
            take = min(int(miss.sum()), rejected_keep - n_rejected)
            if take:
                idx = np.where(miss)[0][:take]
                rejected.append({"obs_rho": obs["obs_rho"][idx],
                                 "snr_true": obs["snr_true"][idx],
                                 "z": z[idx]})
                n_rejected += take
        # Inject the rate evolution: GW mergers per host galaxy ∝ (1+z)**(gamma-1)
        # — the source-frame rate (1+z)**gamma times the 1/(1+z) clock-dilation
        # factor.  Host galaxies themselves trace the comoving volume dV_c/dz
        # (catalog draw), so weighting host acceptance by (1+z)**(gamma-1) makes
        # the detected events follow p(z) ∝ dV_c/dz (1+z)**(gamma-1), i.e. exactly
        # the inference population redshift model, so the injected ``gamma`` is the
        # value the inference recovers (the bare dV_c/dz draw is gamma=1, not 0).
        zmax_grid = float(grids["z"][-1])
        rate_gmax = max(1.0, (1.0 + zmax_grid) ** (pop.gamma - 1.0))
        rate_acc = (1.0 + z) ** (pop.gamma - 1.0) / rate_gmax
        det = det & (rng.uniform(size=len(z)) < rate_acc)
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
            block = dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr)
            # Carry the measurement forward so the posterior conditions on the
            # very data the threshold was applied to.
            block.update(obs)
            kept.append({k: v[det] for k, v in block.items()})
    out = {k: np.concatenate([x[k] for x in kept])[:nobs] for k in kept[0]}
    rej = ({k: np.concatenate([x[k] for x in rejected])[:rejected_keep]
            for k in rejected[0]} if rejected else
           {k: np.array([], dtype=float) for k in ("obs_rho", "snr_true", "z")})
    return out, rej


def _posterior_samples(
    rng: np.random.Generator,
    truth: dict[str, np.ndarray],
    nsamp: int,
    meas: MeasurementConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """EXACT flat-prior posterior draws GIVEN the recorded measurement.

    The mock's per-event term is an importance average over these samples with
    ``p_pe`` as the sampling density, so the samples must be draws from the
    posterior of an ACTUAL measurement under the prior ``p_pe`` declares, and
    that measurement must be the one the detection threshold acted on.  No new
    noise is drawn here: the ``obs_*`` columns ``_draw_events_until_detected``
    recorded are read back, which is what makes the selection a function of the
    data this posterior conditions on.

    The PE prior is flat in ``(ln Mc_det, ln q, rho, chi_eff, ra, dec)`` on the
    physical support ``q <= 1, rho > 0, |chi_eff| <= 1, |dec| <= pi/2``.  Every
    channel's likelihood is an UNBOUNDED Gaussian in exactly the variable the
    prior is flat in, so every posterior is a (possibly prior-truncated) normal
    about the OBSERVED value with the STORED width -- no shift, no skew:

        ln Mc  ~ N(ln Mc_obs, sig_lnmc)
        ln q   ~ N(ln q_obs,  sig_lnq )   truncated to ln q <= 0
        rho    ~ N(rho_obs,   sigma_rho)  truncated to rho > 0 (inert at 8 sigma)
        chi_eff~ N(chi_obs,   sig_chieff) truncated to [-1, 1]
        dec    ~ N(dec_obs,   sigma_ang)  truncated to [-pi/2, pi/2]
        ra     ~ wrapped N(ra_obs, sig_ra)

    Every truncation is a PRIOR truncation drawn by exact inverse CDF
    (:func:`_trunc_norm`); no sample is clipped, because a clip would put a
    point mass at the boundary, which is not a density and which ``p_pe``
    cannot describe.  The samples are then mapped through the bijection of
    :func:`_mc_of_m1q` / :func:`_dl_of_mc_rho` to the storage basis, so
    ``q <= 1`` holds for EVERY sample by construction -- an independent
    ``(m1det, m2det)`` measurement put 18.4 % of its samples in the ``q > 1``
    region the population prior sets to zero.

    ``sigma`` is treated as known per event on both the measurement and the
    posterior side, which keeps the conjugacy exact; real parameter estimation
    infers the width from the data.  This is the standard mock-PE idealisation
    (``GWMockCat`` makes it too).

    Returns
    -------
    samples : dict
        Flat ``nobs * nsamp`` arrays in the gwcat PE layout, plus ``p_pe``.
    observations : dict
        The per-event recorded measurement the posteriors condition on, echoed
        back so the writer can assert the detection side and the PE side hold
        the same numbers.
    """
    missing = [k for k in ("obs_rho", "obs_lnmc", "obs_lnq", "obs_chieff",
                           "obs_ra", "obs_dec", "obs_sigma_rho", "obs_sig_lnmc",
                           "obs_sig_lnq", "obs_sig_chieff", "obs_sigma_ang",
                           "obs_sig_ra") if k not in truth]
    if missing:
        raise ValueError(
            "the posterior must condition on the measurement the detection "
            f"threshold was applied to; missing {missing}. Draw the events with "
            "_draw_events_until_detected.")
    nobs = len(truth["z"])
    keys = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    obs_keys = ("obs_rho", "obs_lnmc", "obs_lnq", "obs_chieff", "obs_ra", "obs_dec")
    observations = {k: np.asarray(truth[k], dtype=float).copy() for k in obs_keys}
    half_pi = 0.5 * np.pi
    for i in range(nobs):
        s_mc = float(truth["obs_sig_lnmc"][i])
        s_q = float(truth["obs_sig_lnq"][i])
        s_ch = float(truth["obs_sig_chieff"][i])
        s_an = float(truth["obs_sigma_ang"][i])
        s_ra = float(truth["obs_sig_ra"][i])
        s_rho = float(truth["obs_sigma_rho"][i])

        lnmc = float(truth["obs_lnmc"][i]) + s_mc * rng.normal(size=nsamp)
        lnq = _trunc_norm(rng, float(truth["obs_lnq"][i]), s_q, -np.inf, 0.0, nsamp)
        rho = _trunc_norm(rng, float(truth["obs_rho"][i]), s_rho, 0.0, np.inf, nsamp)
        chi = _trunc_norm(rng, float(truth["obs_chieff"][i]), s_ch,
                          CHIEFF_RANGE[0], CHIEFF_RANGE[1], nsamp)
        dec = _trunc_norm(rng, float(truth["obs_dec"][i]), s_an, -half_pi, half_pi, nsamp)
        ra = (float(truth["obs_ra"][i]) + s_ra * rng.normal(size=nsamp)) % (2.0 * np.pi)

        mc = np.exp(lnmc)
        q = np.exp(lnq)
        m1 = _m1_of_mc_q(mc, q)
        m2 = q * m1
        dL = _dl_of_mc_rho(mc, rho, meas.snr_ref)
        # Evaluated on the STORED columns, not on the sampling intermediates, so
        # the file is exactly self-consistent: a reader recomputing p_pe from
        # (m1det, m2det, dL) reproduces the stored column bit for bit.
        raw = _p_pe(m1, m2 / m1, dL, meas)

        arrays["ra"].append(ra)
        arrays["dec"].append(dec)
        arrays["dL"].append(dL)
        arrays["m1det"].append(m1)
        arrays["m2det"].append(m2)
        arrays["chieff"].append(chi)
        arrays["p_pe"].append(raw / raw.mean())
    return {k: np.concatenate(v) for k, v in arrays.items()}, observations


_M1DET_RANGE: tuple[float, float] = (2.0, 200.0)
SELECTION_KEYS = ["m1det", "m2det", "m1src", "m2src", "dL", "chieff", "ra", "dec", "pdraw"]


def _selection_pdraw(
    proposal: str,
    m1src: np.ndarray,
    q: np.ndarray,
    chi: np.ndarray,
    z: np.ndarray,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    m1det_range: tuple[float, float] = _M1DET_RANGE,
) -> np.ndarray:
    """Proposal density of the selection draws in the likelihood's canonical
    coordinates (m1det, q, chi, dL, sky).

    ``population`` draws masses/spins FROM the fiducial population, so pdraw is the
    population density with the (1+z) m1src->m1det Jacobian.  ``uniform`` draws a
    broad, population-parameter-independent proposal (uniform m1det/q/chi), so its
    mass-spin factor is flat -- this is the proposal that keeps the importance-
    sampling estimate of mu(theta) well-conditioned across the whole prior (high
    Neff), at the cost of detecting a smaller fraction of the draws.

    ``population+uniform`` is a DEFENSIVE mixture proposal: each draw is a
    Bernoulli(0.9) choice of the population branch else the uniform branch, so the
    proposal density of EVERY row is 0.9 p_population + 0.1 p_uniform.  The 0.1
    uniform floor bounds the importance weight p_inference/p_draw (no e^31 tails,
    healthy Neff even under a stress population) while the 0.9 population mass
    keeps the detected fraction high.
    """
    pz = np.interp(z, grids["z"], grids["dvc_dz"]) / _trapz(grids["dvc_dz"], grids["z"])
    ddldz = np.interp(z, grids["z"], np.gradient(grids["dl"], grids["z"]))
    m1lo, m1hi = m1det_range

    def _p_uniform():
        # m1det drawn directly (no source->detector Jacobian); p(dL)=p_z(z)|dz/ddL|.
        p_dL = pz / np.maximum(ddldz, 1.0e-300)
        return (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * p_dL / (4.0 * np.pi)

    def _p_population():
        # m1det = (1+z) m1src, dL = dL(z): Jacobian (m1src,q,z)->(m1det,q,dL) is (1+z) dL'(z).
        jac = ddldz * (1.0 + z)
        return _mass_spin_pdf(m1src, q, chi, pop) * pz / np.maximum(jac, 1.0e-300) / (4.0 * np.pi)

    if proposal == "uniform":
        p_draw = _p_uniform()
    elif proposal == "population":
        p_draw = _p_population()
    elif proposal == "population+uniform":
        p_draw = 0.9 * _p_population() + 0.1 * _p_uniform()
    else:
        raise ValueError(f"Unknown proposal {proposal!r}; expected one of {SELECTION_PROPOSALS}")
    return np.maximum(p_draw, 1.0e-300)


def _draw_selection_batch(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
    proposal: str = "population",
    m1det_range: tuple[float, float] = _M1DET_RANGE,
    meas: MeasurementConfig | None = None,
) -> dict[str, np.ndarray | int]:
    """Numpy reference selection draw (the JAX path in ``_selection_injections``
    mirrors this exactly; tests pin this implementation).

    The detection rule is the events': one ``rho_obs`` drawn per injection and
    thresholded.  mu(theta) is the probability that a source at theta passes the
    SAME rule the data passed.  Stored coordinates and ``pdraw`` stay the TRUE
    parameters -- mu is an integral over true parameters, and only the detection
    decision sees the noise.  This is what a real injection campaign does: true
    injected parameters, recovery decided on noisy data.

    Because ``P(det|theta) = Phi((rho_opt(theta) - rho_th)/sigma_rho)`` in closed
    form, this rule also has an exact oracle a test can check the draw against."""
    meas = meas or MeasurementConfig(snr_threshold=snr_threshold)
    z = _sample_uniform_comoving_z(rng, grids, ndraw)
    ra, dec = _sample_sky(rng, ndraw)
    dl = _interp_dl(z, grids)
    if proposal == "uniform":
        m1lo, m1hi = m1det_range
        m1det = rng.uniform(m1lo, m1hi, ndraw)
        q = rng.uniform(0.0, 1.0, ndraw)
        chi = rng.uniform(-1.0, 1.0, ndraw)
        m1src = m1det / (1.0 + z)
    elif proposal == "population":
        m1src, use_peak = _sample_powerlaw_peak_m1(rng, ndraw, pop, return_component=True)
        q = _sample_q(rng, m1src, pop, use_peak=use_peak)
        chi = _sample_chieff(rng, ndraw, pop)
    elif proposal == "population+uniform":
        # DEFENSIVE mixture: per draw, Bernoulli(0.9) -> population branch else
        # uniform branch.  Draw both branches for every lane and select; pdraw is
        # the mixture density 0.9 p_pop + 0.1 p_unif for ALL rows (_selection_pdraw).
        m1lo, m1hi = m1det_range
        use_pop = rng.uniform(size=ndraw) < 0.9
        m1_pop, use_peak = _sample_powerlaw_peak_m1(rng, ndraw, pop, return_component=True)
        q_pop = _sample_q(rng, m1_pop, pop, use_peak=use_peak)
        chi_pop = _sample_chieff(rng, ndraw, pop)
        m1det_u = rng.uniform(m1lo, m1hi, ndraw)
        q_u = rng.uniform(0.0, 1.0, ndraw)
        chi_u = rng.uniform(-1.0, 1.0, ndraw)
        m1src_u = m1det_u / (1.0 + z)
        m1src = np.where(use_pop, m1_pop, m1src_u)
        q = np.where(use_pop, q_pop, q_u)
        chi = np.where(use_pop, chi_pop, chi_u)
    else:
        raise ValueError(f"Unknown proposal {proposal!r}; expected one of {SELECTION_PROPOSALS}")
    m2src = q * m1src
    # need_sky=False: the detection rule is the recorded rho_obs alone, so the
    # sky/mass/spin channels of the measurement are never drawn for injections.
    det, _obs = _detect_on_observation(rng, m1src, m2src, z, dl, ra, dec, chi,
                                       meas, need_sky=False)

    p_draw = _selection_pdraw(proposal, m1src, q, chi, z, grids, pop, m1det_range)

    return {
        "m1det": (m1src * (1.0 + z))[det],
        "m2det": (q * m1src * (1.0 + z))[det],
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


# --- JIT/vmap-accelerated selection-injection kernels ------------------------
# ``_draw_selection_batch`` (above) is the readable numpy reference and is what
# tests pin.  Production runs draw tens of millions of injections to clear the
# SNR cut, so the hot loop below runs the *same* draw + detection on the XLA
# device: a per-injection kernel ``vmap``-ed over the batch and ``jit``-compiled
# once, fed in fixed-size chunks (``--nbatches``) so device memory stays bounded.
# Only the SAMPLING + detection run on device; pdraw is evaluated by the exact
# numpy density (``_selection_pdraw``) on the small DETECTED subset, so the
# stored pdraw is byte-for-byte the reference value (no density reimplementation).
# jax.random supplies the draws (seeded from ``rng``); the selection set is the
# last consumer of ``rng`` in both generators, so nothing else shifts.
_REJECTION_MAXITER = 10_000


def _jax_sfilter_low(m, m_min, dm):
    # m_min/dm may be traced per-lane arrays (peak vs power-law pairing floor),
    # so select the safe dm with jnp.where rather than a Python ``if``.
    delta = m - m_min
    safe_d = jnp.where(delta > 0.0, delta, 1.0)
    safe_dm = jnp.where(dm > 0.0, dm, 1.0)
    expo = jnp.clip(safe_dm / safe_d + safe_dm / (safe_d - safe_dm), -500.0, 500.0)
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m <= m_min, 0.0, S)
    return jnp.where(m >= m_min + dm, 1.0, S)


def _jax_sfilter_high(m, m_max, dm):
    delta = m_max - m
    safe_d = jnp.where(delta > 0.0, delta, 1.0)
    safe_dm = dm if dm > 0.0 else 1.0
    expo = jnp.clip(safe_dm / safe_d + safe_dm / (safe_d - safe_dm), -500.0, 500.0)
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m >= m_max, 0.0, S)
    return jnp.where(m <= m_max - dm, 1.0, S)


def _make_population_mass_spin_sampler(pop: PopulationConfig):
    """Build scalar (per-injection) JAX samplers for m1src, q|m1, chi mirroring the
    numpy population samplers: tapered-power-law + Gaussian-peak m1, S_low(m2)-tapered
    q (both by rejection via a per-lane ``while_loop``), truncated-Gaussian chi."""
    alpha, mmin, mmax = float(pop.alpha), float(pop.mmin), float(pop.mmax)
    dm_min, dm_max = float(pop.dm_min), float(pop.dm_max)
    pf, mu, sigma = float(pop.peak_fraction), float(pop.peak_mu), float(pop.peak_sigma)
    beta = float(pop.beta)
    chi_mu, chi_sigma = float(pop.chi_mu), float(pop.chi_sigma)
    lo_m, hi_m = float(_MASS_NORM_GRID[0]), float(_MASS_NORM_GRID[-1])
    pair_m_lo, pair_dm = float(_PAIR_M_LO), float(_PAIR_DM)
    a = 1.0 - alpha
    alpha_is_one = bool(np.isclose(alpha, 1.0))
    bp1 = beta + 1.0
    beta_is_minus_one = bool(np.isclose(beta, -1.0))

    def _hard_powerlaw(u):
        if alpha_is_one:
            return mmin * (mmax / mmin) ** u
        return (u * (mmax ** a - mmin ** a) + mmin ** a) ** (1.0 / a)

    def _hard_q(u, qmin):
        if beta_is_minus_one:
            return qmin * (1.0 / qmin) ** u
        return (u * (1.0 - qmin ** bp1) + qmin ** bp1) ** (1.0 / bp1)

    def sample_m1(key):
        ku, kpl, kpk = jax.random.split(key, 3)
        use_peak = jax.random.uniform(ku) < pf

        def cond(st):
            _, _, acc, it = st
            return jnp.logical_and(jnp.logical_not(acc), it < _REJECTION_MAXITER)

        def body(st):
            k, m, _, it = st
            k, k1, k2 = jax.random.split(k, 3)
            cand = _hard_powerlaw(jax.random.uniform(k1))
            S = _jax_sfilter_low(cand, mmin, dm_min) * _jax_sfilter_high(cand, mmax, dm_max)
            acc = jax.random.uniform(k2) < S
            return k, jnp.where(acc, cand, m), acc, it + 1

        init = (kpl, jnp.asarray(mmin), jnp.asarray(False), jnp.asarray(0, jnp.int32))
        _, m_pl, _, _ = lax.while_loop(cond, body, init)
        tn = jax.random.truncated_normal(kpk, (lo_m - mu) / sigma, (hi_m - mu) / sigma)
        m_pk = mu + sigma * tn
        # Return the peak/power-law lane flag so sample_q can pick the matching
        # per-component pairing floor (mirrors _sample_powerlaw_peak_m1's mask).
        return jnp.where(use_peak, m_pk, m_pl), use_peak

    def sample_q(key, m1, use_peak):
        # Per-component pairing floor: peak lanes -> (1.0, 0.01), power-law lanes
        # -> (m_min, dm_min).  qmin uses the component m_min so the proposal
        # support reaches m2 = m_min (peak lanes must cover m2 in [1, m_min]).
        m_min_lane = jnp.where(use_peak, pair_m_lo, mmin)
        dm_lane = jnp.where(use_peak, pair_dm, dm_min)
        qmin = jnp.clip(m_min_lane / m1, 1.0e-3, 1.0)
        kinit, kloop = jax.random.split(key, 2)
        q0 = _hard_q(jax.random.uniform(kinit), qmin)   # fallback for the (measure-0) no-accept case

        def cond(st):
            _, _, acc, it = st
            return jnp.logical_and(jnp.logical_not(acc), it < _REJECTION_MAXITER)

        def body(st):
            k, qv, _, it = st
            k, k1, k2 = jax.random.split(k, 3)
            cand = _hard_q(jax.random.uniform(k1), qmin)
            acc = jax.random.uniform(k2) < _jax_sfilter_low(cand * m1, m_min_lane, dm_lane)
            return k, jnp.where(acc, cand, qv), acc, it + 1

        init = (kloop, q0, jnp.asarray(False), jnp.asarray(0, jnp.int32))
        _, qf, _, _ = lax.while_loop(cond, body, init)
        return qf

    def sample_chi(key):
        tn = jax.random.truncated_normal(key, (-1.0 - chi_mu) / chi_sigma, (1.0 - chi_mu) / chi_sigma)
        return chi_mu + chi_sigma * tn

    return sample_m1, sample_q, sample_chi


def _make_selection_kernel(grids: dict[str, np.ndarray], pop: PopulationConfig,
                           snr_threshold: float, proposal: str,
                           m1det_range: tuple[float, float] = _M1DET_RANGE,
                           extra_detect=None, meas: MeasurementConfig | None = None,
                           snr_fn=None):
    """Return ``jit(vmap(sample_one))`` mapping a batch of PRNG keys to per-injection
    (m1src, q, chi, z, ra, dec, dL, det).  ``extra_detect(key, state) -> bool`` adds a
    further per-injection cut (e.g. the bright-siren EM selection).

    The detection statistic mirrors the numpy reference ``_draw_selection_batch``
    exactly: ``rho_obs = rho_opt(theta) + N(0, sigma_rho)``, thresholded, while
    the returned coordinates stay the TRUE parameters.  ``snr_fn(key, state) ->
    snr`` overrides that statistic for callers whose events use a different
    detection rule (the bright-siren mock still uses a projection latent); the
    override replaces the statistic rather than adding to it, so events and
    injections cannot end up on different rules."""
    meas = meas or MeasurementConfig(snr_threshold=snr_threshold)
    z_grid = jnp.asarray(grids["z"])
    dl_grid = jnp.asarray(grids["dl"])
    vc_cdf = jnp.asarray(grids["vc_cdf"])
    m1lo, m1hi = m1det_range
    uses_population = proposal in ("population", "population+uniform")
    samplers = _make_population_mass_spin_sampler(pop) if uses_population else None
    snr_ref = float(meas.snr_ref)
    sigma_rho = float(meas.sigma_rho)

    def sample_one(key):
        ks = jax.random.split(key, 7)
        z = jnp.interp(jax.random.uniform(ks[0]), vc_cdf, z_grid)
        ra = jax.random.uniform(ks[1], maxval=2.0 * jnp.pi)
        dec = jnp.arcsin(jax.random.uniform(ks[2], minval=-1.0, maxval=1.0))
        dl = jnp.interp(z, z_grid, dl_grid)
        if proposal == "uniform":
            m1det = jax.random.uniform(ks[3], minval=m1lo, maxval=m1hi)
            q = jax.random.uniform(ks[4])
            chi = jax.random.uniform(ks[5], minval=-1.0, maxval=1.0)
            m1src = m1det / (1.0 + z)
        else:
            sample_m1, sample_q, sample_chi = samplers
            m1src, use_peak = sample_m1(ks[3])
            q = sample_q(ks[4], m1src, use_peak)
            chi = sample_chi(ks[5])
            if proposal == "population+uniform":
                # DEFENSIVE mixture: Bernoulli(0.9) picks the population draw else
                # a uniform (m1det, q, chi) draw.  Extra randoms come from a folded
                # key so the population/uniform-only paths keep their key layout.
                ku = jax.random.split(jax.random.fold_in(key, 202), 4)
                use_pop = jax.random.uniform(ku[0]) < 0.9
                m1det_u = jax.random.uniform(ku[1], minval=m1lo, maxval=m1hi)
                q_u = jax.random.uniform(ku[2])
                chi_u = jax.random.uniform(ku[3], minval=-1.0, maxval=1.0)
                m1src_u = m1det_u / (1.0 + z)
                m1src = jnp.where(use_pop, m1src, m1src_u)
                q = jnp.where(use_pop, q, q_u)
                chi = jnp.where(use_pop, chi, chi_u)
        m2src = q * m1src
        state = {"m1src": m1src, "q": q, "chi": chi, "z": z, "ra": ra, "dec": dec, "dL": dl}
        if snr_fn is None:
            mchirp_det = (m1src * m2src) ** 0.6 / (m1src + m2src) ** 0.2 * (1.0 + z)
            rho_opt = snr_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)
            snr = rho_opt + sigma_rho * jax.random.normal(ks[6])
        else:
            snr = snr_fn(ks[6], state)
        det = snr >= snr_threshold
        if extra_detect is not None:
            det = jnp.logical_and(det, extra_detect(key, state))
        return {**state, "det": det}

    return jax.jit(jax.vmap(sample_one))


def _run_selection_chunks(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    proposal: str,
    batch_size: int,
    kernel,
    target_detections: int | None = None,
    verbose: bool = False,
    label: str = "selection",
    m1det_range: tuple[float, float] = _M1DET_RANGE,
) -> dict[str, np.ndarray | int]:
    """Drive ``kernel`` (a jitted vmapped sampler) over fixed-size chunks, mask to
    detected on the host, and evaluate the exact numpy pdraw on the detected subset."""
    key0 = jax.random.PRNGKey(int(rng.integers(0, 2**63 - 1)))
    chunks: list[dict[str, np.ndarray]] = []
    n_proposed = 0
    n_detected = 0
    ci = 0
    while n_proposed < ndraw:
        n_batch = int(min(batch_size, ndraw - n_proposed))
        subkeys = jax.random.split(jax.random.fold_in(key0, ci), n_batch)
        ci += 1
        out = kernel(subkeys)
        det = np.asarray(out["det"])
        z = np.asarray(out["z"])[det]
        m1src = np.asarray(out["m1src"])[det]
        q = np.asarray(out["q"])[det]
        chi = np.asarray(out["chi"])[det]
        chunks.append({
            "m1det": m1src * (1.0 + z),
            "m2det": q * m1src * (1.0 + z),
            "m1src": m1src,
            "m2src": q * m1src,
            "dL": np.asarray(out["dL"])[det],
            "chieff": chi,
            "ra": np.asarray(out["ra"])[det],
            "dec": np.asarray(out["dec"])[det],
            "pdraw": _selection_pdraw(proposal, m1src, q, chi, z, grids, pop, m1det_range),
        })
        n_proposed += n_batch
        n_detected += int(det.sum())
        if verbose:
            print(f"  {label} batch: proposed={n_proposed:,}/{ndraw:,}, detected={n_detected:,}")
        if target_detections is not None and n_detected >= target_detections:
            break

    if chunks:
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in SELECTION_KEYS}
    else:
        arrays = {key: np.array([], dtype=float) for key in SELECTION_KEYS}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected}


def _selection_injections(
    rng: np.random.Generator,
    ndraw: int,
    grids: dict[str, np.ndarray],
    pop: PopulationConfig,
    snr_threshold: float,
    batch_size: int,
    target_detections: int | None = None,
    verbose: bool = False,
    proposal: str = "population",
    m1det_range: tuple[float, float] = _M1DET_RANGE,
    meas: MeasurementConfig | None = None,
) -> dict[str, np.ndarray | int]:
    kernel = _make_selection_kernel(grids, pop, float(snr_threshold), proposal,
                                    m1det_range, meas=meas)
    return _run_selection_chunks(
        rng, ndraw, grids, pop, proposal, batch_size, kernel,
        target_detections=target_detections, verbose=verbose,
        label="selection", m1det_range=m1det_range,
    )


def _galaxy_count_from_density(n0: float, delta: float, grids: dict[str, np.ndarray]) -> int:
    density_weighted_volume = jnp.trapezoid(grids["dvc_dz"] * (1.0 + grids["z"]) ** delta, grids["z"])
    return max(1, int(round(n0 * density_weighted_volume)))


def write_mock_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    pop = PopulationConfig(gamma=args.gamma)
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
    # The survey sees the REALISED photo-z, and declares the width of the model
    # at the value it recorded: dz = zerr(z_obs).  The true redshift never
    # leaves the complete catalog.
    z_obs_survey = complete["z_obs"][observed]
    zerr_survey = _catalog_zerr(z_obs_survey, survey)
    n_negative_z_obs = int((z_obs_survey < 0.0).sum())
    weights = np.ones(observed.sum())
    mark_obs = ({name: complete[name][observed] for name in eta_true} if with_marks else None)
    pixelated = _pixelate_catalog(
        complete["ra"][observed], complete["dec"][observed], z_obs_survey,
        zerr_survey, weights, args.nside, marks=mark_obs,
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

    # The measurement model, shared by the detection rule, the selection
    # injections and the posterior: passing one object keeps them from drifting
    # apart.
    meas = MeasurementConfig(
        snr_ref=float(args.snr_ref),
        snr_threshold=float(args.snr_threshold),
        sigma_rho=float(args.snr_uncertainty),
        a_mc=float(args.lnmc_uncertainty),
        a_q=float(args.lnq_uncertainty),
        a_chi=float(args.chieff_uncertainty),
        sky_uncertainty_deg=args.sky_uncertainty_deg,
    )
    truth, rejected = _draw_events_until_detected(
        rng, args.nobs, complete, grids, pop, args.snr_threshold,
        sky_weight_fn=sky_weight_fn, sky_g_max=sky_g_max,
        mark_weight=mark_weight, mark_g_max=mark_g_max,
        meas=meas, rejected_keep=int(args.record_rejected),
    )
    post, pe_observations = _posterior_samples(rng, truth, args.nsamp, meas)
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)
    selection_target_detections = args.selection_target_detections
    if args.selection_per_observation_factor is not None:
        selection_target_detections = int(np.ceil(args.selection_per_observation_factor * args.nobs))
    # Chunk the nselection draws: explicit --selection-batch-size wins (legacy),
    # otherwise split into --nbatches equal chunks (default 1 = one kernel call).
    selection_batch_size = (
        args.selection_batch_size
        if args.selection_batch_size is not None
        else int(np.ceil(args.ndraw / args.nbatches))
    )
    sel = _selection_injections(
        rng,
        args.ndraw,
        grids,
        pop,
        args.snr_threshold,
        selection_batch_size,
        target_detections=selection_target_detections,
        verbose=args.verbose,
        proposal=args.proposal,
        meas=meas,
    )

    inv_pdraw = 1.0 / np.asarray(sel["pdraw"])
    selection_neff = float(inv_pdraw.sum() ** 2 / np.square(inv_pdraw).sum()) if len(inv_pdraw) else 0.0

    metadata = {
        "seed": args.seed,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": args.w0, "wa": args.wa},
        "population": asdict(pop),
        "survey": asdict(survey),
        "snr_threshold": args.snr_threshold,
        "selection_proposal": args.proposal,
        "snr_ref": args.snr_ref,
        # Every constant of the measurement family, so a file is self-describing
        # and every stored width is recomputable from the file alone.
        "measurement": {
            "family": MEASUREMENT_FAMILY,
            "basis": "(ln Mc_det, ln q, rho, chi_eff, ra, dec)",
            "detection": "rho_obs >= snr_threshold, rho_obs = rho_opt(theta) + N(0, sigma_rho)",
            "width_law": "sigma_x = a_x * (snr_threshold / rho_obs)",
            "snr_ref": float(meas.snr_ref),
            "snr_threshold": float(meas.snr_threshold),
            "sigma_rho": float(meas.sigma_rho),
            "a_mc": float(meas.a_mc),
            "a_q": float(meas.a_q),
            "a_chi": float(meas.a_chi),
            "sky_a_deg": float(meas.sky_a_deg),
            "sky_clip_deg": list(SKY_CLIP_DEG),
            "snr_ref_sigma": float(SNR_REF_SIGMA),
            "cos_dec_floor": float(COS_DEC_FLOOR),
            "sky_uncertainty_deg": (None if meas.sky_uncertainty_deg is None
                                    else float(meas.sky_uncertainty_deg)),
            "p_pe_basis": P_PE_BASIS,
        },
        # The declared photo-z error is REALISED: the catalog carries z (true)
        # and z_obs, and the survey block pixelates on z_obs.  z_obs is not
        # clipped, so the realised count of negative entries is recorded here.
        "catalog_photoz": {
            "realised": True,
            "model": "z_obs = z + N(0, redshift_error_floor + redshift_error_slope (1+z))",
            "survey_declares": "dz = zerr(z_obs)",
            "n_negative_z_obs_in_survey": n_negative_z_obs,
            "n_survey_galaxies": int(observed.sum()),
        },
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
        # Z is the OBSERVED redshift and ZERR its declared error: an EM survey
        # table cannot carry a true redshift.  The true one stays in the
        # complete catalog, which is the generator's own record.
        f.create_dataset("Z", data=z_obs_survey, compression="gzip", shuffle=True)
        f.create_dataset("ZERR", data=zerr_survey, compression="gzip", shuffle=True)
        f.create_dataset("WEIGHT", data=weights, compression="gzip", shuffle=True)

    pixel_path = out / f"catalog_pixelated_nside_{args.nside}.h5"
    with h5py.File(pixel_path, "w") as f:
        f.attrs["nside"] = int(args.nside)
        f.attrs["mock_data"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in pixelated.items():
            if key != "ngals":
                assert val.dtype == np.float64, (
                    f"pixelated {key!r} is {val.dtype}, not float64: "
                    "_kde_dndz_obs clamps its truncated-kernel mass at 1e-300, "
                    "which is not representable below float64, so every padded "
                    "row would evaluate 0/0 = NaN (see _pixelate_catalog)")
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
        f.attrs["measurement_family"] = MEASUREMENT_FAMILY
        f.attrs["p_pe_basis"] = P_PE_BASIS
        f.attrs["snr_ref"] = float(args.snr_ref)
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["sigma_rho"] = float(meas.sigma_rho)
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
        # The measurement already sits in ``truth`` (the detection step recorded
        # it and the posterior conditioned on that same record), so those keys
        # must not be written twice -- and if both sides hold a value it must be
        # the same one.
        for key, val in pe_observations.items():
            if key in truth:
                assert np.array_equal(np.asarray(truth[key]), np.asarray(val)), (
                    f"truth[{key!r}] disagrees with the observation the posterior "
                    "conditioned on; detection and PE have drifted apart")
                continue
            truth_group.create_dataset(key, data=val)
        # A sample of proposals the SNR rule REJECTED, with their own recorded
        # rho_obs and their true amplitude.  Every row must fail the threshold
        # on its own recorded number; together with the detected set these give
        # the two Malmquist observables that are exactly zero for a
        # true-parameter cut and strictly positive for a data-space one.
        if len(rejected["obs_rho"]):
            rej_group = f.create_group("rejected")
            for key, val in rejected.items():
                rej_group.create_dataset(key, data=val)

    sel_path = out / "mock_gw_selection.h5"
    with h5py.File(sel_path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])
        f.attrs["Neff"] = selection_neff
        f.attrs["selection_proposal"] = args.proposal
        f.attrs["measurement_family"] = MEASUREMENT_FAMILY
        f.attrs["snr_ref"] = float(args.snr_ref)
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["sigma_rho"] = float(meas.sigma_rho)
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
    print(f"  observed survey  : {raw_path} ({observed.sum():,} galaxies retained, "
          f"photo-z realised, {n_negative_z_obs} with z_obs < 0 left unclipped)")
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


# Flags whose behaviour the all-observable measurement family removed.  argparse's
# bare "unrecognized arguments" would not tell a caller WHY, and these appear in
# saved command lines and job scripts, so each one gets a pointer.
_REMOVED_FLAGS = {
    "--detection-data":
        "the threshold always acts on the recorded rho_obs now; there is no "
        "true-parameter detection rule to select. Recalibrate --snr-ref instead.",
    "--pe-centering":
        "posterior samples are always the exact flat-prior posterior of the "
        "recorded measurement; truth-centred clouds are gone.",
    "--dL-fractional-uncertainty":
        "dL is no longer measured on its own -- it is DERIVED from (Mc_det, rho), "
        "so the distance precision follows from --snr-uncertainty and "
        "--lnmc-uncertainty (sigma_ln dL = sqrt((5/6 sigma_lnMc)^2 + "
        "(sigma_rho/rho)^2)).",
    "--m1det-fractional-uncertainty":
        "component masses are no longer measured independently; the mass channel "
        "is (ln Mc_det, ln q). Use --lnmc-uncertainty / --lnq-uncertainty.",
    "--m2det-fractional-uncertainty":
        "component masses are no longer measured independently; the mass channel "
        "is (ln Mc_det, ln q). Use --lnmc-uncertainty / --lnq-uncertainty.",
}


def _reject_removed_flags(argv: list[str]) -> None:
    for arg in argv:
        name = arg.split("=", 1)[0]
        if name in _REMOVED_FLAGS:
            raise SystemExit(
                f"{name} was removed with the all-observable measurement family: "
                f"{_REMOVED_FLAGS[name]}")


def parse_args() -> argparse.Namespace:
    _reject_removed_flags(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/mock_dark_sirens", help="Output directory for HDF5 products.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-galaxies", type=_positive_int, default=None, help="Number of complete-catalog galaxies to draw when --n0 is omitted.")
    parser.add_argument("--n0", type=_positive_float, default=None, help="Comoving galaxy density in Mpc^-3; overrides --n-galaxies when provided.")
    parser.add_argument("--nobs", type=_positive_int, default=8)
    parser.add_argument("--nsamp", type=_positive_int, default=512)
    parser.add_argument("--nselection", "--ndraw", dest="ndraw", type=_positive_int, default=80_000,
                        help="Total number of selection injections to PROPOSE (the detected "
                             "subset is what gets stored). Without an early-stop target, exactly "
                             "this many are drawn. --ndraw is a legacy alias.")
    parser.add_argument("--nbatches", type=_positive_int, default=1,
                        help="Split the --nselection draws into this many equal chunks for the "
                             "jit/vmap kernel; chunking bounds device memory. Default 1 draws all "
                             "injections in a single kernel call (best on GPU).")
    parser.add_argument("--proposal", choices=SELECTION_PROPOSALS, default="population",
                        help="Selection-injection proposal. 'population' (default) draws "
                             "masses/spins from the fiducial population (matches the inference "
                             "selection integral; smaller detected Neff). 'uniform' draws a broad, "
                             "population-independent proposal (uniform m1det/q/chi) that keeps the "
                             "importance-sampling Neff high across the whole prior. "
                             "'population+uniform' is a defensive Bernoulli(0.9) mixture of the two "
                             "(pdraw = 0.9 p_population + 0.1 p_uniform for every row): the 0.1 "
                             "uniform floor bounds p_inference/p_draw (no heavy weight tails / Neff "
                             "collapse) while keeping most draws population-like.")
    parser.add_argument("--nside", type=_positive_int, default=16)
    parser.add_argument("--zmax", type=_positive_float, default=0.08)
    parser.add_argument("--H0", type=_positive_float, default=67.74)
    parser.add_argument("--Om0", type=_positive_float, default=0.3075)
    parser.add_argument("--w0", type=float, default=-1.0, help="CPL dark-energy equation-of-state value today.")
    parser.add_argument("--wa", type=float, default=0.0, help="CPL dark-energy evolution parameter.")
    parser.add_argument("--gamma", type=float, default=PopulationConfig.gamma,
                        help="Injected rate-evolution index: GW rate ∝ (1+z)**gamma "
                             "(events drawn ∝ dV_c/dz (1+z)**(gamma-1)). Default matches "
                             "the inference fiducial; recoverable only with enough redshift "
                             "lever arm (raise --zmax).")
    parser.add_argument("--snr-threshold", type=_positive_float, default=8.0)
    parser.add_argument("--survey-z50", type=float, default=SurveyConfig.z50)
    parser.add_argument("--survey-width", type=_positive_float, default=SurveyConfig.width)
    parser.add_argument("--galaxy-density-delta", type=float, default=SurveyConfig.delta)
    parser.add_argument("--selection-batch-size", type=_positive_int, default=None,
                        help="Explicit chunk size (legacy). If set, it takes precedence over "
                             "--nbatches; otherwise the chunk size is --nselection / --nbatches.")
    selection_targets = parser.add_mutually_exclusive_group()
    selection_targets.add_argument("--selection-target-detections", type=_positive_int, default=None)
    selection_targets.add_argument("--selection-per-observation-factor", type=_positive_float, default=None)
    parser.add_argument("--snr-ref", type=_positive_float, default=SNR_REF_DEFAULT,
                        help=("Amplitude scale of the detection statistic: "
                              "rho_opt = snr_ref (Mc_det/30)^(5/6) (1000 Mpc/dL). "
                              "The threshold acts on rho_obs = rho_opt + N(0, "
                              "--snr-uncertainty), with NO projection latent -- a "
                              "detection decision that depends on a variable absent "
                              "from the data leaves an extra P(det|theta) inside each "
                              "event's integral. That raises the detected fraction "
                              "~5.75x relative to a true-parameter cut at the same "
                              "scale, so recalibrate this if a detected population "
                              "must stay comparable to an older mock (snr_ref = 6.278 "
                              "reproduced the old fraction on the gws-agn deep mock)."))
    parser.add_argument("--snr-uncertainty", type=_positive_float,
                        default=SIGMA_RHO_DEFAULT,
                        help=("sigma_rho, the additive SNR noise: rho_obs = rho_opt + "
                              "N(0, sigma_rho). GWMockCat's uncert_default['snr'] = 1."))
    parser.add_argument("--lnmc-uncertainty", type=_positive_float, default=A_MC_DEFAULT,
                        help=("Width of ln Mc_det at rho_obs = --snr-threshold; the "
                              "realised width is this times (snr_threshold/rho_obs). "
                              "GWMockCat's uncert_default['mc'] = 0.08."))
    parser.add_argument("--lnq-uncertainty", type=_positive_float, default=A_Q_DEFAULT,
                        help=("Width of ln q at rho_obs = --snr-threshold. 0.60 is the "
                              "GW150914 anchor; see the A_Q calibration note in the "
                              "measurement-family header."))
    parser.add_argument("--chieff-uncertainty", type=_positive_float, default=A_CHI_DEFAULT,
                        help=("Width of chi_eff at rho_obs = --snr-threshold (the width "
                              "now scales as 1/rho_obs rather than being constant). "
                              "GWMockCat's --Xeff_uncert = 0.20."))
    parser.add_argument("--sky-uncertainty-deg", type=_positive_float, default=None,
                        help=("Data-independent constant sky width, in degrees. Omit to "
                              "derive it from the recorded SNR, "
                              "clip(35 deg / ((11.5/snr_ref) rho_obs), 1, 12)."))
    parser.add_argument("--record-rejected", type=int, default=0,
                        help=("Record up to this many proposals REJECTED by the SNR "
                              "threshold, with their own rho_obs and true amplitude, "
                              "under 'rejected/' in the events file. 0 (default) writes "
                              "none. Every stored row must fail the threshold on its own "
                              "recorded number."))
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
