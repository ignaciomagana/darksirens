#!/usr/bin/env python
"""
generate_mock_lensing.py
========================
Standalone strong-lensing mock-data generator for ``darksirens_inference_lensing``.

This is a single-file consolidation of the former multi-step ``SIM_LENSING``
pipeline (population + lensing marks, Finn-Chernoff selection, the two injection
campaigns, the PE-posterior model, and the assembler).  It mirrors the structure
of ``scripts/mock_dark_sirens/generate_mock_data.py`` and writes every output in
the **current** loader schema (``gwcat-1.0`` / ``gwcat-selection-1.0``) — no
legacy formats.

Everything that defines a density is IMPORTED from ``darksirens`` (population,
cosmology, lensing marks), never re-derived here, so the simulator and the
likelihood share one source of truth.

Physics
-------
* Population: (m1, q, chi_eff) by rejection sampling against the imported
  ``MixtureModel.mixture`` density; z from ``dV_c/dz * (1+z)^(gamma-1)`` by
  inverse-CDF on the imported cosmology grid.
* Lensing marks: per source, Bernoulli(``tau_2_SIS(z)``) -> J=2 double else
  singleton; singleton gets a lognormal weak-lensing magnification; doubles draw
  ``y ~ 2y`` and the imported SIS image magnifications ``(mu_+, mu_-)``.
* Selection: smooth orientation-averaged Finn & Chernoff (1993) detection
  probability at each object's weak-/strong-lensed apparent distance.
* PE: ``nsamp`` posterior samples per detected object around a noisy realisation
  of the truth with SNR-scaled widths and the canonical m1det-dL correlation.

Outputs (in ``--outdir``)
-------------------------
  mock_gw_pe.h5             singleton PE          (format_version="gwcat-1.0")
  mock_pair_metadata.h5     pair metadata/marks  (no duplicated image PE)
  mock_gw_selection.h5      unlensed injections   (format_version="gwcat-selection-1.0")
  mock_lensed_injections.h5 lensed J=2 injections (lensing.lensed_injections schema)
  partition.json            TRUE (singleton_indices, pair_indices)
  truth.json / manifest.json  ground truth + inventory (informational)

Example
-------
  python scripts/mock_lensing/generate_mock_lensing.py \
      --outdir data/mock_lensing --seed 7 --nsamp 1000 \
      --n-sing-keep 200 --n-pair-keep 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import shutil
from pathlib import Path

# Make ``darksirens`` importable when this script is run directly from any cwd
# (the repo is used in-place, not necessarily pip-installed): add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import h5py
from scipy.stats import norm as _norm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---- imported truth: population ----
from darksirens.gw.populations.registry import get_fixed_population_params, get_model

# ---- imported truth: cosmology ----
from darksirens.utils.cosmology import (
    dL_of_z, z_of_dL, dV_of_z, ddL_of_z, H0Planck, Om0Planck, zMax,
)

# ---- imported truth: lensing ----
from darksirens.lensing.slmarks import (
    tau_2_SIS, mu_plus_minus_from_y, make_sis_lens_params, delta_t_from_y,
    DEFAULT_T0_SECONDS,
)
from darksirens.lensing.wlmagnification import make_lognormal_wl_params
from darksirens.lensing.lensed_injections import save_lensed_injections
from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model
from darksirens.lensing.observed_catalog import write_observed_pe_attrs
from scripts.mock_lensing.build_candidate_pairs_from_observed import build_candidate_pairs, _log_sky_overlap

POP_NAME = "powerlaw+peak"
_MIXTURE_PARAM_ORDER = [
    "v1", "alpha", "mmin", "mmax", "dmmin", "dmmax",
    "muG", "sigG", "beta", "muchi", "sigchi",
]
# Backward-compatible default (power-law rate); truth.json records the
# rate-aware order via _theta_param_order().
THETA_PARAM_ORDER = _MIXTURE_PARAM_ORDER + ["gamma"]


def set_pop_model(pop_name: str) -> None:
    """Select the generator population model (main() calls this from --pop_model).

    The mixture part must stay powerlaw+peak (the analytic rejection proposal
    is tailored to it); the '@md' decoration swaps the redshift evolution for
    the Madau-Dickinson-like peaked rate with (gamma, kappa, z_peak).
    """
    from darksirens.gw.populations.registry import split_rate_decoration
    base, _ = split_rate_decoration(pop_name)
    if base != "powerlaw+peak":
        raise ValueError(
            f"generate_mock_lensing supports the powerlaw+peak mixture only "
            f"(got {pop_name!r}); the rejection proposal is tailored to it."
        )
    global POP_NAME
    POP_NAME = pop_name


def _rate_evolution() -> str:
    from darksirens.gw.populations.registry import split_rate_decoration
    return split_rate_decoration(POP_NAME)[1]


def _n_rate_params() -> int:
    return 3 if _rate_evolution() == "md" else 1


def _theta_param_order() -> list:
    if _rate_evolution() == "md":
        return _MIXTURE_PARAM_ORDER + ["gamma", "kappa", "z_peak"]
    return _MIXTURE_PARAM_ORDER + ["gamma"]


# ============================================================
# Truth container
# ============================================================
def make_truth(seed, H0, Om0, sis, wl):
    """Pin all ground-truth hyperparameters by importing the fiducials."""
    theta = np.asarray(get_fixed_population_params(POP_NAME))
    n_rate = _n_rate_params()
    truth = dict(
        pop_name=POP_NAME,
        rate_evolution=_rate_evolution(),
        theta=theta,                       # mixture params + rate params
        gamma=float(theta[-n_rate]),
        H0=float(H0), Om0=float(Om0), zMax=float(zMax),
        tau_A=float(sis.A_tau), tau_n=float(sis.n_tau), T0_sec=float(sis.T0),
        wl_a=float(wl.a), wl_b=float(wl.b),
        seed=int(seed),
    )
    if n_rate == 3:
        truth["kappa"] = float(theta[-2])
        truth["z_peak"] = float(theta[-1])
    return truth


# ============================================================
# STEP 1a - mass / mass-ratio / spin via rejection sampling
# ============================================================
def _mixture_density(m1, q, chi, theta):
    """Imported mass*pairing*spin density (no z factor). Vectorised."""
    model = get_model(POP_NAME)
    tm = jnp.asarray(theta[:-_n_rate_params()])   # drop rate params
    return np.asarray(model.mixture(jnp.asarray(m1), jnp.asarray(q),
                                    jnp.asarray(chi), tm))


def _analytic_proposal_density(m1, q, chi, theta):
    """Density of the tailored (m1, q, chi) proposal — the SAME formulas the
    draws in :func:`_analytic_proposal` come from, kept standalone so the
    matched injection campaign can evaluate it at arbitrary points (mixture
    pdraw). Vectorised; zero outside support.

    Clip-atom caveat: draws clip the Gaussian peak and chi into their boxes
    instead of redrawing; the leaked mass is <~5e-6 for generator-range
    parameters, so the density treats the components as un-atomized pdfs.
    """
    v1, alpha, mmin, mmax, dmmin, dmmax, muG, sigG, beta, muchi, sigchi = \
        [float(x) for x in theta[:11]]
    lo, hi = mmin - dmmin, mmax + dmmax
    w_peak = max(min(1.0 - v1, 0.95), 0.05)
    a = 1.0 - alpha

    m1 = np.asarray(m1, dtype=float)
    q = np.asarray(q, dtype=float)
    chi = np.asarray(chi, dtype=float)

    if abs(a) > 1e-8:
        pl_norm = (hi**a - lo**a) / a
    else:
        pl_norm = np.log(hi / lo)
    m1_safe = np.where(m1 > 0, m1, 1.0)
    g_pl = np.where((m1 >= lo) & (m1 <= hi), m1_safe**(-alpha) / pl_norm, 0.0)
    sg = 1.5 * sigG
    g_pk = np.exp(-0.5 * ((m1 - muG) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
    g_m1 = w_peak * g_pk + (1 - w_peak) * g_pl

    bp1 = beta + 1.0
    q_safe = np.where(q > 0, q, 1.0)
    g_q = bp1 * q_safe**beta if abs(bp1) > 1e-8 else 1.0 / q_safe
    g_q = np.where((q > 0) & (q <= 1.0), g_q, 0.0)

    sgc = 1.3 * sigchi
    Z = _norm.cdf((1 - muchi) / sgc) - _norm.cdf((-1 - muchi) / sgc)
    g_chi = np.exp(-0.5 * ((chi - muchi) / sgc) ** 2) / (np.sqrt(2 * np.pi) * sgc * Z)
    g_chi = np.where(np.abs(chi) <= 0.999, g_chi, 0.0)

    return g_m1 * g_q * g_chi


def _analytic_proposal(n, theta, rng):
    """Tailored proposal close to the imported mixture, for high acceptance.

    Returns draws AND the proposal density g(m1,q,chi) for the IS correction.
    """
    v1, alpha, mmin, mmax, dmmin, dmmax, muG, sigG, beta, muchi, sigchi = \
        [float(x) for x in theta[:11]]     # mixture params; rate tail unused here

    # --- m1: PL + Gaussian mixture (broadened slightly past the tapers) ---
    lo, hi = mmin - dmmin, mmax + dmmax
    w_peak = max(min(1.0 - v1, 0.95), 0.05)   # rough; IS weight corrects exactly
    use_peak = rng.uniform(size=n) < w_peak
    u = rng.uniform(size=n)
    a = 1.0 - alpha
    m_pl = (u * (hi**a - lo**a) + lo**a) ** (1.0 / a) if abs(a) > 1e-8 \
        else lo * (hi / lo) ** u
    m_pk = rng.normal(muG, 1.5 * sigG, n)      # broadened peak
    m1 = np.where(use_peak, m_pk, m_pl)
    m1 = np.clip(m1, lo, hi)

    # --- q: q ~ U^(1/(beta+1)) gives pdf prop q^beta on (0,1] ---
    bp1 = beta + 1.0
    uq = rng.uniform(size=n)
    q = uq ** (1.0 / bp1) if abs(bp1) > 1e-8 else np.exp(np.log(uq))
    q = np.clip(q, 1e-3, 1.0)

    # --- chi: truncated Gaussian(mu_chi, sigma_chi) ---
    chi = rng.normal(muchi, 1.3 * sigchi, n)   # slightly broadened
    chi = np.clip(chi, -0.999, 0.999)

    g = _analytic_proposal_density(m1, q, chi, theta)
    return m1, q, chi, g


def sample_masses_spins(n, theta, rng, *, batch=100_000, max_iter=200):
    """Rejection-sample (m1,q,chi) from the imported mixture using a tailored
    analytic proposal g (adaptive envelope, inflated 1.3x)."""
    out_m1, out_q, out_chi = [], [], []
    M = None
    kept = 0
    for _ in range(max_iter):
        m1, q, chi, g = _analytic_proposal(batch, theta, rng)
        p = _mixture_density(m1, q, chi, theta)
        p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
        g = np.where(g > 0, g, np.inf)
        w = p / g
        if M is None or np.nanmax(w) > M:
            M = 1.3 * np.nanmax(w)
        u = rng.uniform(0.0, M, batch)
        acc = u < w
        out_m1.append(m1[acc]); out_q.append(q[acc]); out_chi.append(chi[acc])
        kept += int(acc.sum())
        if kept >= n:
            break
    m1 = np.concatenate(out_m1)[:n]
    q = np.concatenate(out_q)[:n]
    chi = np.concatenate(out_chi)[:n]
    if len(m1) < n:
        raise RuntimeError(f"rejection sampling under-filled: {len(m1)}/{n}")
    return m1, q, chi


# ============================================================
# STEP 1b - redshift via inverse-CDF of dV/dz (1+z)^(gamma-1)
# ============================================================
def _build_z_cdf(theta, H0, Om0, nz=4000):
    """Tabulate the normalized source-frame redshift PDF and its CDF using the
    IMPORTED dV_of_z so the cosmology matches the likelihood.

    Rate evolution follows the generator population model: power law
    (1+z)^(gamma-1), or for '@md' the Madau-Dickinson-like psi(z)/(1+z) with
    the SAME (unnormalised) form as PopulationModel.log_p_pop, so mock truth
    and inference share one definition.
    """
    zg = np.linspace(1e-4, float(zMax), nz)
    dV = np.asarray(dV_of_z(jnp.asarray(zg), H0, Om0))
    if _rate_evolution() == "md":
        gamma, kappa, z_pk = [float(x) for x in theta[-3:]]
        log_rate = (gamma - 1.0) * np.log1p(zg) - np.logaddexp(
            0.0, (gamma + kappa) * (np.log1p(zg) - np.log1p(z_pk))
        )
        pdf = dV * np.exp(log_rate)
    else:
        gamma = float(theta[-1])
        pdf = dV * (1.0 + zg) ** (gamma - 1.0)
    pdf = np.where(np.isfinite(pdf) & (pdf > 0), pdf, 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return zg, pdf / np.trapezoid(pdf, zg), cdf


def sample_redshift(n, theta, rng, H0, Om0):
    zg, pdf, cdf = _build_z_cdf(theta, H0, Om0)
    u = rng.uniform(0.0, 1.0, n)
    z = np.interp(u, cdf, zg)
    return z, (zg, pdf)


# ============================================================
# STEP 2 - lensing marks
# ============================================================
def assign_marks(z, sis, wl, rng):
    """Per source: Bernoulli(tau_2(z)) -> double; else singleton + WL mu."""
    n = len(z)
    tau = np.asarray(tau_2_SIS(jnp.asarray(z), sis))
    tau = np.clip(tau, 0.0, 1.0)
    is_double = rng.uniform(0.0, 1.0, n) < tau

    # singletons: WL magnification, ln mu ~ N(-s^2/2, s), s^2(z) = a z^b
    s2 = wl.a * np.power(np.maximum(z, 1e-3), wl.b)
    s = np.sqrt(s2)
    ln_mu = rng.normal(-0.5 * s2, s)
    mu_wl = np.exp(ln_mu)
    mu = np.where(~is_double, mu_wl, np.nan)

    # doubles: y ~ 2y on (0,1) via y = sqrt(u); imported (mu_+, mu_-)
    u = rng.uniform(0.0, 1.0, n)
    y_all = np.sqrt(u)
    mp_all, mm_all = mu_plus_minus_from_y(jnp.asarray(y_all))
    mp_all = np.asarray(mp_all); mm_all = np.asarray(mm_all)
    y = np.where(is_double, y_all, np.nan)
    mu_plus = np.where(is_double, mp_all, np.nan)
    mu_minus = np.where(is_double, mm_all, np.nan)

    return dict(is_double=is_double, mu=mu, y=y,
                mu_plus=mu_plus, mu_minus=mu_minus, tau_at_z=tau)


def generate_step12(n_universe, seed, H0, Om0, sis, wl):
    """Population + marks driver (former sim_step12)."""
    truth = make_truth(seed, H0, Om0, sis, wl)
    rng = np.random.default_rng(seed)
    theta = truth["theta"]

    m1, q, chi = sample_masses_spins(n_universe, theta, rng)
    z, (zg, zpdf) = sample_redshift(n_universe, theta, rng, H0, Om0)
    marks = assign_marks(z, sis, wl, rng)

    dL_src = np.asarray(dL_of_z(jnp.asarray(z), H0, Om0))
    m1det = (1.0 + z) * m1
    ra_true = rng.uniform(0.0, 2 * np.pi, n_universe)
    dec_true = np.arcsin(rng.uniform(-1.0, 1.0, n_universe))
    src = dict(m1=m1, q=q, chi=chi, z=z, dL_src=dL_src, m1det=m1det,
               ra_true=ra_true, dec_true=dec_true)
    return dict(truth=truth, src=src, marks=marks, z_grid=zg, z_pdf=zpdf)


# ============================================================
# STEP 3 - Finn & Chernoff (1993) smooth selection
# ============================================================
def _theta_pdf(theta):
    theta = np.asarray(theta)
    out = 5.0 * theta * (4.0 - theta) ** 3 / 256.0
    return np.where((theta > 0) & (theta < 4.0), out, 0.0)


def _build_theta_sf(n=20001):
    tg = np.linspace(0.0, 4.0, n)
    pdf = _theta_pdf(tg)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(tg))])
    cdf /= cdf[-1]
    return tg, np.clip(1.0 - cdf, 0.0, 1.0)


_THETA_GRID, _THETA_SF = _build_theta_sf()


def theta_survival(x):
    """P(Theta > x) by interpolating the tabulated survival function."""
    return np.interp(np.asarray(x), _THETA_GRID, _THETA_SF, left=1.0, right=0.0)


def chirp_mass_det(m1_src, q, z):
    m1 = np.asarray(m1_src); m2 = q * m1
    mc_src = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return mc_src * (1.0 + z)


class SNRModel:
    """Finn-Chernoff SNR detection model with a single horizon constant r0."""

    def __init__(self, rho_thr=8.0, horizon_Mpc=3000.0, mc_bar=1.22):
        self.rho_thr = float(rho_thr)
        self.mc_bar = float(mc_bar)
        # rho = 8 Theta (r0/dL)(Mc/Mc_bar)^5/6; Theta=4, Mc=Mc_bar, dL=horizon
        # -> rho=rho_thr => r0 = rho_thr*horizon/32.
        self.r0 = self.rho_thr * horizon_Mpc / (8.0 * 4.0)

    def expected_snr_optimal(self, m1_src, q, z, dL_app):
        mc = chirp_mass_det(m1_src, q, z)
        return 8.0 * 4.0 * (self.r0 / np.asarray(dL_app)) * (mc / self.mc_bar) ** (5.0 / 6.0)

    def theta_threshold(self, m1_src, q, z, dL_app):
        mc = chirp_mass_det(m1_src, q, z)
        denom = 8.0 * (self.r0 / np.asarray(dL_app)) * (mc / self.mc_bar) ** (5.0 / 6.0)
        return self.rho_thr / np.maximum(denom, 1e-300)

    def p_det(self, m1_src, q, z, dL_app):
        return theta_survival(self.theta_threshold(m1_src, q, z, dL_app))


def apply_selection_singletons(src, marks, model, rng):
    z = src["z"]; m1 = src["m1"]; q = src["q"]; dL_src = src["dL_src"]
    sing = ~marks["is_double"]
    dL_app = dL_src / np.sqrt(np.where(sing, marks["mu"], 1.0))
    pdet = np.where(sing, model.p_det(m1, q, z, dL_app), 0.0)
    detected = sing & (rng.uniform(0.0, 1.0, len(z)) < pdet)
    return detected, dL_app, pdet


def apply_selection_doubles(src, marks, model, rng):
    z = src["z"]; m1 = src["m1"]; q = src["q"]; dL_src = src["dL_src"]
    dbl = marks["is_double"]
    dL_app_p = dL_src / np.sqrt(np.where(dbl, marks["mu_plus"], 1.0))
    dL_app_m = dL_src / np.sqrt(np.where(dbl, marks["mu_minus"], 1.0))
    pdet_p = np.where(dbl, model.p_det(m1, q, z, dL_app_p), 0.0)
    pdet_m = np.where(dbl, model.p_det(m1, q, z, dL_app_m), 0.0)
    det_p = dbl & (rng.uniform(0.0, 1.0, len(z)) < pdet_p)
    det_m = dbl & (rng.uniform(0.0, 1.0, len(z)) < pdet_m)
    return dict(det_plus=det_p, det_minus=det_m, both_detected=det_p & det_m)


# ============================================================
# STEP 4 - injection campaigns
# ============================================================
def _broad_z_table(H0, Om0, nz=4000):
    """z proposal table of the broad campaign: pdf ∝ dV_c/dz on (0, zMax].
    Returns (zg, cdf, Znorm); the exact density at arbitrary z is
    dV_of_z(z)/Znorm (matching the historical pdraw convention exactly)."""
    zg = np.linspace(1e-4, float(zMax), nz)
    dV = np.asarray(dV_of_z(jnp.asarray(zg), H0, Om0))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dV[1:] + dV[:-1]) * np.diff(zg))])
    Znorm = cdf[-1]; cdf = cdf / Znorm
    return zg, cdf, Znorm


def _detectability_tilted_z_table(theta, model, H0, Om0, nz=4000, n_msc=4096):
    """z proposal table for the matched campaign: pdf ∝ p_z_astro(z) x
    p̄_det(z), where p̄_det(z) is the Finn-Chernoff detection probability
    averaged over fiducial-shaped (m1, q, chi) draws. This concentrates
    injection draws where detected sources actually live — near-equal
    importance weights at the truth, so Neff tracks the detected count
    instead of collapsing (the flat proposal's Neff was ~10-20% of N_det
    and fell an order of magnitude short of the GWTC-4/5 selection
    variance criterion at 280 events).

    Uses a fixed-seed private rng for the (m1,q,chi) average so the table
    is deterministic and consumes nothing from the campaign stream.
    """
    zg, pdf_astro, _ = _build_z_cdf(theta, H0, Om0, nz=nz)
    msc_rng = np.random.default_rng(714_2026)
    m1, q, chi, _g = _analytic_proposal(n_msc, theta, msc_rng)
    dLg = np.asarray(dL_of_z(jnp.asarray(zg), H0, Om0))
    pbar = np.empty(nz)
    for i in range(nz):
        pbar[i] = float(np.mean(np.asarray(
            model.p_det(m1, q, np.full(n_msc, zg[i]), np.full(n_msc, dLg[i]))
        )))
    pdf = pdf_astro * pbar
    pdf = np.where(np.isfinite(pdf) & (pdf > 0), pdf, 0.0)
    norm = np.trapezoid(pdf, zg)
    pdf = pdf / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf = cdf / cdf[-1]
    return zg, pdf, cdf


MATCHED_FRACTION = 0.85


def generate_unlensed_injections(n_draw, model, rng, H0, Om0, *,
                                 m1det_range=(3.0, 200.0), out_path=None,
                                 proposal="broad", theta=None):
    """Unlensed singleton injections in the detector-frame proposal basis,
    written in the ``gwcat-selection-1.0`` schema.

    proposal="broad" (default): flat m1det x flat q x flat chieff x dV_c/dz —
    the historical campaign; byte-identical output for a given rng state.

    proposal="matched": defensive two-branch mixture — MATCHED_FRACTION of
    draws use the fiducial-shaped (m1, q, chi) proposal with a
    detectability-tilted z, the rest use the broad proposal (full prior
    support). pdraw is the exact mixture density in the canonical
    (m1det, q, dL, chieff) basis for every sample, so the estimator stays
    unbiased while near-truth weights are near-equal (Neff ~ N_det).
    """
    m1lo, m1hi = m1det_range
    zg_b, cdf_b, Znorm_b = _broad_z_table(H0, Om0)

    if proposal == "broad":
        m1det = rng.uniform(m1lo, m1hi, n_draw)
        q = rng.uniform(0.0, 1.0, n_draw)
        chieff = rng.uniform(-1.0, 1.0, n_draw)
        z = np.interp(rng.uniform(0, 1, n_draw), cdf_b, zg_b)
        pz_of_z = np.asarray(dV_of_z(jnp.asarray(z), H0, Om0)) / Znorm_b

        dL = np.asarray(dL_of_z(jnp.asarray(z), H0, Om0))
        ddL = np.asarray(ddL_of_z(jnp.asarray(z), jnp.asarray(dL), H0, Om0))

        # proposal density in the canonical (m1det, q, dL) basis
        p_draw = (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * (pz_of_z / ddL)
    elif proposal == "matched":
        if theta is None:
            raise ValueError("proposal='matched' requires the truth theta.")
        zg_t, pdf_t, cdf_t = _detectability_tilted_z_table(theta, model, H0, Om0)

        nA = int(round(MATCHED_FRACTION * n_draw))
        nC = n_draw - nA
        # branch A: fiducial-shaped source frame + tilted z
        m1A, qA, chiA, _gA = _analytic_proposal(nA, theta, rng)
        zA = np.interp(rng.uniform(0, 1, nA), cdf_t, zg_t)
        m1detA = m1A * (1.0 + zA)
        # branch C: broad (full support)
        m1detC = rng.uniform(m1lo, m1hi, nC)
        qC = rng.uniform(0.0, 1.0, nC)
        chiC = rng.uniform(-1.0, 1.0, nC)
        zC = np.interp(rng.uniform(0, 1, nC), cdf_b, zg_b)

        m1det = np.concatenate([m1detA, m1detC])
        q = np.concatenate([qA, qC])
        chieff = np.concatenate([chiA, chiC])
        z = np.concatenate([zA, zC])

        dL = np.asarray(dL_of_z(jnp.asarray(z), H0, Om0))
        ddL = np.asarray(ddL_of_z(jnp.asarray(z), jnp.asarray(dL), H0, Om0))

        # exact mixture density at EVERY sample, canonical (m1det, q, dL) basis
        m1_src_all = m1det / (1.0 + z)
        gA_msc = _analytic_proposal_density(m1_src_all, q, chieff, theta)
        pdfA_z = np.interp(z, zg_t, pdf_t)
        pA = gA_msc / (1.0 + z) * (pdfA_z / ddL)
        pdfC_z = np.asarray(dV_of_z(jnp.asarray(z), H0, Om0)) / Znorm_b
        inC = (m1det >= m1lo) & (m1det <= m1hi) & (q >= 0.0) & (q <= 1.0) \
            & (np.abs(chieff) <= 1.0)
        pC = np.where(inC, (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * (pdfC_z / ddL), 0.0)
        p_draw = MATCHED_FRACTION * pA + (1.0 - MATCHED_FRACTION) * pC
    else:
        raise ValueError(f"unknown injection proposal {proposal!r}")

    # selection: Finn-Chernoff p_det at the (unlensed) apparent distance dL
    m1_src = m1det / (1.0 + z)
    detected = rng.uniform(0, 1, n_draw) < model.p_det(m1_src, q, z, dL)

    m2det = q * m1det
    ra = rng.uniform(0, 2 * np.pi, n_draw)
    dec = np.arcsin(rng.uniform(-1, 1, n_draw))
    # source-frame component masses (required by the gwcat-selection schema)
    m1src = m1det / (1.0 + z)
    m2src = m2det / (1.0 + z)

    d = {k: v[detected] for k, v in dict(
        m1det=m1det, m2det=m2det, m1src=m1src, m2src=m2src, dL=dL,
        chieff=chieff, ra=ra, dec=dec, pdraw=p_draw,
    ).items()}

    if out_path:
        with h5py.File(out_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["mock_data"] = True
            f.attrs["ndraw"] = int(n_draw)
            f.attrs["chi_eff_swap_applied"] = True
            f.attrs["chi_eff_amax"] = 0.99
            f.attrs["cosmology_H0"] = float(H0)
            f.attrs["cosmology_Om0"] = float(Om0)
            f.attrs["pop_model"] = POP_NAME
            f.attrs["rho_thr"] = float(model.rho_thr)
            f.attrs["injection_proposal"] = str(proposal)
            if proposal == "matched":
                f.attrs["matched_fraction"] = float(MATCHED_FRACTION)
            for key in ("m1det", "m2det", "m1src", "m2src", "dL",
                        "chieff", "ra", "dec", "pdraw"):
                f.create_dataset(key, data=d[key])
    return d, int(detected.sum())


def generate_lensed_injections(n_draw_sources, model, rng, H0, Om0, *,
                               m1src_range=(3.0, 120.0), out_path=None,
                               pair_tag_model="none", pair_tag_prob=1.0,
                               sis=None):
    """Lensed J=2 injections in the SOURCE-frame proposal basis, written via the
    lensing per-image schema (``save_lensed_injections``)."""
    m1lo, m1hi = m1src_range
    m1_src = rng.uniform(m1lo, m1hi, n_draw_sources)
    q = rng.uniform(0.0, 1.0, n_draw_sources)
    chieff = rng.uniform(-1.0, 1.0, n_draw_sources)
    y = rng.uniform(0.0, 1.0, n_draw_sources)            # p_prop_y = 1 (weight 2y)

    zg = np.linspace(1e-4, float(zMax), 4000)
    dV = np.asarray(dV_of_z(jnp.asarray(zg), H0, Om0))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dV[1:] + dV[:-1]) * np.diff(zg))])
    Znorm = cdf[-1]; cdf /= cdf[-1]
    z = np.interp(rng.uniform(0, 1, n_draw_sources), cdf, zg)
    pz_of_z = np.asarray(dV_of_z(jnp.asarray(z), H0, Om0)) / Znorm

    p_prop_src = (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * pz_of_z
    p_prop_y = np.ones(n_draw_sources)

    mu_p, mu_m = mu_plus_minus_from_y(jnp.asarray(y))
    mu_p = np.asarray(mu_p); mu_m = np.asarray(mu_m)
    dL_src = np.asarray(dL_of_z(jnp.asarray(z), H0, Om0))
    pdet_p = model.p_det(m1_src, q, z, dL_src / np.sqrt(mu_p))
    pdet_m = model.p_det(m1_src, q, z, dL_src / np.sqrt(mu_m))
    det_p = rng.uniform(0, 1, n_draw_sources) < pdet_p
    det_m = rng.uniform(0, 1, n_draw_sources) < pdet_m
    both = det_p & det_m

    snr_p = model.expected_snr_optimal(m1_src, q, z, dL_src / np.sqrt(mu_p))
    snr_m = model.expected_snr_optimal(m1_src, q, z, dL_src / np.sqrt(mu_m))
    # Render with the SAME SISLensParams the observed pairs use: T0 is now
    # configurable, so a bare make_sis_lens_params() here would put the
    # injection campaign's delays on a different time-delay scale from the
    # observed catalog's.
    sis_render = sis if sis is not None else make_sis_lens_params()
    true_dt = np.asarray(delta_t_from_y(jnp.asarray(y), sis_render), dtype=float)
    log_sky_overlap = np.log(np.clip((np.minimum(snr_p, snr_m) / np.maximum(snr_p, snr_m)) ** 2, 1e-12, 1.0))
    normalized_model = "snr_time" if pair_tag_model == "min_snr_proxy" else pair_tag_model
    if normalized_model == "none":
        p_tag = np.ones(n_draw_sources, dtype=float)
    elif normalized_model in ("constant", "snr_time", "snr_time_sky"):
        tag_model = make_pair_tag_selection_model(normalized_model, constant=pair_tag_prob)
        p_tag = tag_model.probability(snr_image0=snr_p, snr_image1=snr_m, delta_t_obs=true_dt, log_sky_overlap=log_sky_overlap)
    else:
        raise ValueError(f"unknown pair_tag_model: {pair_tag_model}")
    tagged_pair = np.zeros(n_draw_sources, dtype=bool)
    tagged_pair[both] = rng.uniform(0, 1, int(both.sum())) < p_tag[both]

    # per-IMAGE flat arrays (2 rows per source, interleaved +/-)
    N = n_draw_sources
    source_id = np.repeat(np.arange(N, dtype=np.int32), 2)
    image_id = np.tile(np.array([0, 1], dtype=np.int32), N)

    def interleave(a_plus, a_minus):
        out = np.empty(2 * N, dtype=np.asarray(a_plus).dtype)
        out[0::2] = a_plus; out[1::2] = a_minus
        return out

    if out_path:
        save_lensed_injections(
            out_path,
            source_id=source_id, image_id=image_id,
            m1_src=np.repeat(m1_src, 2), q_src=np.repeat(q, 2),
            z_src=np.repeat(z, 2), chieff=np.repeat(chieff, 2),
            y_source=np.repeat(y, 2), mu=interleave(mu_p, mu_m),
            detected=interleave(det_p, det_m),
            p_prop_src=np.repeat(p_prop_src, 2), p_prop_y=np.repeat(p_prop_y, 2),
            n_draw_sources=int(N), p_tag_per_source=p_tag,
            snr_image0=snr_p, snr_image1=snr_m, delta_t_obs=true_dt, true_delta_t=true_dt,
            log_sky_overlap=log_sky_overlap, p_tag_true=p_tag, tagged_pair=tagged_pair,
            snr_model_attrs={
                "fc_rho_thr": model.rho_thr,
                "fc_r0": model.r0,
                "fc_mc_bar": model.mc_bar,
            },
        )
    return dict(n_sources=N, n_both=int(both.sum()),
                pair_tag_model=pair_tag_model, pair_tag_prob=float(pair_tag_prob),
                mean_p_tag_both=float(np.mean(p_tag[both])) if np.any(both) else None), int(both.sum())


# ============================================================
# STEP 5 - mock PE posteriors
# ============================================================
def _pe_prior_density(m1det, q, dL):
    """Canonical PE sampling prior in the (m1det, q, dL) basis: p ∝ m1det dL^2."""
    return m1det * dL ** 2


def _draw_posterior_samples(m1det_true, q_true, dL_app_true, chieff_true,
                            rho, nsamp, rng):
    """nsamp posterior samples around a NOISY realisation of the truth with
    SNR-scaled widths and an m1det-dL correlation (PIT-calibrated)."""
    f_dL = float(np.clip(1.8 / max(rho, 1.0), 0.02, 0.40))
    f_m1 = float(np.clip(0.6 / max(rho, 1.0), 0.02, 0.25))
    s_chi = float(np.clip(0.5 / max(rho, 1.0), 0.02, 0.30))
    f_q = float(np.clip(0.5 / max(rho, 1.0), 0.02, 0.25))
    rho_corr = -0.4

    cov = np.array([[f_m1**2, rho_corr * f_m1 * f_dL],
                    [rho_corr * f_m1 * f_dL, f_dL**2]])
    L = np.linalg.cholesky(cov)

    c = L @ rng.standard_normal(2)             # noisy observed center
    m1det_obs = m1det_true * np.exp(c[0])
    dL_obs = dL_app_true * np.exp(c[1])
    q_obs = q_true + f_q * q_true * rng.standard_normal()
    chi_obs = chieff_true + s_chi * rng.standard_normal()

    dlog = L @ rng.standard_normal((2, nsamp))  # scatter around the center
    m1det = m1det_obs * np.exp(dlog[0])
    dL = dL_obs * np.exp(dlog[1])
    q = np.clip(q_obs + f_q * q_true * rng.standard_normal(nsamp), 1e-3, 1.0)
    chieff = np.clip(chi_obs + s_chi * rng.standard_normal(nsamp), -0.999, 0.999)
    return m1det, q, dL, chieff


def _wrap_ra(ra):
    return float(np.mod(ra, 2 * np.pi))


def _clip_dec(dec):
    eps = 1e-9
    return float(np.clip(dec, -0.5 * np.pi + eps, 0.5 * np.pi - eps))


def make_event_pe(m1_src, q, z, chieff, mu, model, nsamp, rng, H0, Om0, *,
                  ra_true=None, dec_true=None, sky_sigma_floor_rad=0.01):
    """One detected object's PE samples given its TRUE source params and mu."""
    dL_src = float(np.asarray(dL_of_z(jnp.asarray(z), H0, Om0)))
    dL_app_true = dL_src / np.sqrt(mu)
    m1det_true = (1.0 + z) * m1_src

    rho_opt = float(model.expected_snr_optimal(
        np.array([m1_src]), np.array([q]), np.array([z]), np.array([dL_app_true]))[0])
    rho_eff = max(rho_opt / 2.5, 6.0)          # crude network-averaged proxy

    m1det, qs, dL, chi = _draw_posterior_samples(
        m1det_true, q, dL_app_true, chieff, rho_eff, nsamp, rng)
    m2det = qs * m1det
    if ra_true is None:
        ra_true = rng.uniform(0, 2 * np.pi)
    if dec_true is None:
        dec_true = np.arcsin(rng.uniform(-1, 1))
    sky_sigma_rad = float(max(sky_sigma_floor_rad, np.clip(2.5 / max(rho_eff, 1.0), 0.01, 0.40)))
    dec_mean = _clip_dec(float(dec_true) + rng.normal(0.0, sky_sigma_rad))
    # Tangent-plane approximation: scale RA noise by cos(dec) to keep angular noise comparable.
    ra_mean = _wrap_ra(float(ra_true) + rng.normal(0.0, sky_sigma_rad / max(np.cos(dec_mean), 0.1)))
    ra = np.full(nsamp, ra_mean)
    dec = np.full(nsamp, dec_mean)
    p_pe = _pe_prior_density(m1det, qs, dL)
    return dict(m1det=m1det, m2det=m2det, dL=dL, chieff=chi, ra=ra, dec=dec, p_pe=p_pe,
                ra_mean=float(ra_mean), dec_mean=float(dec_mean), sky_sigma_rad=sky_sigma_rad,
                ra_true=float(ra_true), dec_true=float(dec_true))


def _src_masses_from_dL(m1det, m2det, dL, H0, Om0):
    """Naive source-frame component masses for the gwcat schema: m = mdet/(1+z),
    z inferred from each sample's dL (NaN -> 0 so the datasets stay finite; the
    loader skips the chi_eff swap for mock_data so these are not used downstream)."""
    z = np.asarray(z_of_dL(jnp.asarray(dL), H0, Om0))
    z = np.nan_to_num(z, nan=0.0)
    return m1det / (1.0 + z), m2det / (1.0 + z)


def write_gw_pe_file(events, path, nsamp, H0, Om0):
    """Singleton PE in the ``gwcat-1.0`` schema (event-major flatten)."""
    nobs = len(events)

    def stack(key):
        return np.concatenate([e[key] for e in events])

    m1det = stack("m1det"); m2det = stack("m2det"); dL = stack("dL")
    m1src, m2src = _src_masses_from_dL(m1det, m2det, dL, H0, Om0)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["pe_cosmology_H0"] = float(H0)
        f.attrs["pe_cosmology_Om0"] = float(Om0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["pop_model"] = POP_NAME
        for key, val in dict(
            m1det=m1det, m2det=m2det, m1src=m1src, m2src=m2src, dL=dL,
            chieff=stack("chieff"), ra=stack("ra"), dec=stack("dec"),
            p_pe=stack("p_pe"),
        ).items():
            f.create_dataset(key, data=val)
        for key in ("ra_mean", "dec_mean", "sky_sigma_rad", "ra_true", "dec_true"):
            if all(key in e for e in events):
                f.create_dataset(
                    key, data=np.asarray([e[key] for e in events], dtype=float)
                )
    return nobs



def write_pair_metadata_file(pairs, path, pair_indices=None):
    """Metadata-only HDF5 for candidate-edge/pair marks; contains no image PE samples."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "lensing-pair-metadata-1.0"
        f.attrs["npairs"] = len(pairs)
        f.attrs["mock_data"] = True
        for k, pair in enumerate(pairs):
            g = f.create_group(f"pair_{k}")
            if pair_indices is not None:
                g.attrs["event_index_image0"] = int(pair_indices[k][0])
                g.attrs["event_index_image1"] = int(pair_indices[k][1])
                g.attrs["i"] = int(pair_indices[k][0])
                g.attrs["j"] = int(pair_indices[k][1])
            g.attrs["label"] = "true"
            g.attrs["truth_is_lensed_pair"] = True
            g.attrs["delta_t_obs"] = float(pair["delta_t_obs"])
            g.attrs["sigma_delta_t"] = float(pair["sigma_delta_t"])
            g.attrs["true_y"] = float(pair["true_y"])
            g.attrs["true_mu0"] = float(pair.get("true_mu0", np.nan))
            g.attrs["true_mu1"] = float(pair.get("true_mu1", np.nan))
            g.attrs["true_source_id"] = int(pair.get("source_index", -1))
    return len(pairs)

def write_pair_pe_file(pairs, path, nsamp, pair_indices=None):
    """Pair PE per image (apparent-frame coords) — read directly by the lensing
    tool's ``load_inputs``."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "lensed-pair-pe-1.0"
        f.attrs["npairs"] = len(pairs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["prior_weight_convention"] = (
            "raw_pe_prior_density_written; loader_normalizes_per_image"
        )
        f.attrs["mock_data"] = True
        f.attrs["time_delay_metadata"] = True
        for k, pair in enumerate(pairs):
            img0, img1 = pair["images"]
            g = f.create_group(f"pair_{k}")
            if pair_indices is not None:
                g.attrs["event_index_image0"] = int(pair_indices[k][0])
                g.attrs["event_index_image1"] = int(pair_indices[k][1])
            g.attrs["source_index"] = int(pair.get("source_index", -1))
            g.attrs["pair_index"] = int(k)
            g.attrs["delta_t_obs"] = float(pair["delta_t_obs"])
            g.attrs["sigma_delta_t"] = float(pair["sigma_delta_t"])
            g.attrs["true_y"] = float(pair["true_y"])
            g.attrs["true_delta_t"] = float(pair["true_delta_t"])
            g.attrs["t_obs_image0"] = float(pair["t_obs_image0"])
            g.attrs["t_obs_image1"] = float(pair["t_obs_image1"])
            for name, img in (("image0", img0), ("image1", img1)):
                gi = g.create_group(name)
                gi.create_dataset("m1det", data=img["m1det"])
                gi.create_dataset("q", data=img["m2det"] / img["m1det"])
                gi.create_dataset("dL_app", data=img["dL"])
                gi.create_dataset("chieff", data=img["chieff"])
                gi.create_dataset("prior_wt", data=img["p_pe"])
    return len(pairs)


# ============================================================
# STEP 6 - assemble
# ============================================================
def assemble(out_dir, *, n_universe, seed, nsamp, n_sing_keep, n_pair_keep,
             conditioning, max_sing_keep, max_pair_keep,
             rho_thr, horizon_Mpc, n_unlensed_inj, n_lensed_inj,
             H0, Om0, sis, wl, pair_tag_model="none", pair_tag_prob=1.0,
             n_wrong_candidate_pairs=0, candidate_pair_log_prior_odds=0.0,
             wrong_candidate_log_prior_odds=-5.0, time_delay_sigma_sec=3600.0,
             write_unified_observed_catalog=True, candidate_time_marks=True,
             include_lensed_singletons=False,
             observation_times="placeholder", t_obs_days=365.25,
             build_candidate_pairs_from_observed=False,
             validation_sample_log10_tau_A=False, validation_log10_tau_A_prior=(-7.0, -2.0),
             write_legacy_pair_pe=False,
             injection_proposal="broad"):
    os.makedirs(out_dir, exist_ok=True)
    truth = make_truth(seed, H0, Om0, sis, wl)
    truth.update(rho_thr=rho_thr, horizon_Mpc=horizon_Mpc,
                 pair_tag_model=pair_tag_model, pair_tag_prob=pair_tag_prob,
                 n_sources_universe=n_universe, nsamp=nsamp,
                 conditioning=conditioning,
                 selection_model="Finn-Chernoff orientation-averaged p_det",
                 pe_prior_convention="p_pe proportional to m1det * dL^2 in the (m1det, q, dL) basis",
                 time_delay_sigma_sec=float(time_delay_sigma_sec))

    rng = np.random.default_rng(seed)
    model = SNRModel(rho_thr=rho_thr, horizon_Mpc=horizon_Mpc)

    # ---- population + marks + selection ----
    d = generate_step12(n_universe, seed, H0, Om0, sis, wl)
    src, marks = d["src"], d["marks"]
    det_s, _, _ = apply_selection_singletons(src, marks, model, rng)
    dbl = apply_selection_doubles(src, marks, model, rng)

    sing_candidates = np.where(det_s)[0]
    pair_candidates = np.where(dbl["both_detected"])[0]
    one_det = dbl["det_plus"] ^ dbl["det_minus"]
    lensed_sing_candidates = np.where(one_det)[0]
    n_singletons_detected_total = int(sing_candidates.size)
    n_pairs_both_detected_total = int(pair_candidates.size)
    n_lensed_singletons_detected_total = int(lensed_sing_candidates.size)

    # Realistic observation protocol (--include-lensed-singletons true):
    # a strongly lensed source with exactly one detected image cannot be
    # distinguished from an ordinary singleton, so it joins the singleton
    # pool BEFORE the shuffle/caps — preserving the natural lensed fraction
    # among observed singletons. Default (false) keeps the legacy protocol
    # that drops these sources entirely (consistent with the OFF likelihood).
    sing_is_lensed = np.zeros(sing_candidates.size, dtype=bool)
    if include_lensed_singletons:
        sing_candidates = np.concatenate([sing_candidates, lensed_sing_candidates])
        sing_is_lensed = np.concatenate(
            [sing_is_lensed, np.ones(lensed_sing_candidates.size, dtype=bool)]
        )

    # Randomize before any truncation so the kept catalog is an unbiased subset
    # of the detected mock rather than the first sources in simulation order.
    perm = rng.permutation(sing_candidates.size)
    sing_candidates = sing_candidates[perm]
    sing_is_lensed = sing_is_lensed[perm]
    rng.shuffle(pair_candidates)

    caps_applied = {"singletons": False, "pairs": False}
    if conditioning == "fixed_counts":
        if n_sing_keep > n_singletons_detected_total:
            warnings.warn(
                f"requested --n-sing-keep={n_sing_keep} but only "
                f"{n_singletons_detected_total} singleton sources were detected; "
                "keeping all available singleton detections",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_pair_keep > n_pairs_both_detected_total:
            warnings.warn(
                f"requested --n-pair-keep={n_pair_keep} but only "
                f"{n_pairs_both_detected_total} lensed pairs had both images detected; "
                "keeping all available detected pairs",
                RuntimeWarning,
                stacklevel=2,
            )
        sing_idx = sing_candidates[:n_sing_keep]
        sing_idx_is_lensed = sing_is_lensed[:n_sing_keep]
        pair_src_idx = pair_candidates[:n_pair_keep]
    elif conditioning == "poisson_counts":
        sing_idx = sing_candidates
        sing_idx_is_lensed = sing_is_lensed
        pair_src_idx = pair_candidates
        if max_sing_keep is not None and sing_idx.size > max_sing_keep:
            sing_idx = sing_idx[:max_sing_keep]
            sing_idx_is_lensed = sing_idx_is_lensed[:max_sing_keep]
            caps_applied["singletons"] = True
        if max_pair_keep is not None and pair_src_idx.size > max_pair_keep:
            pair_src_idx = pair_src_idx[:max_pair_keep]
            caps_applied["pairs"] = True
    else:  # argparse enforces this; keep defensive guard for direct callers.
        raise ValueError(f"unknown conditioning mode: {conditioning}")

    # ---- PE: singletons + pairs ----
    def _observed_singleton_mu(i, is_lensed):
        if is_lensed:
            return (marks["mu_plus"][i] if dbl["det_plus"][i]
                    else marks["mu_minus"][i])
        return marks["mu"][i]

    events = [make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                            _observed_singleton_mu(i, lens), model, nsamp, rng, H0, Om0,
                            ra_true=src["ra_true"][i], dec_true=src["dec_true"][i])
              for i, lens in zip(sing_idx, sing_idx_is_lensed)]
    write_gw_pe_file(events, os.path.join(out_dir, "mock_gw_pe.h5"), nsamp, H0, Om0)

    # Observation times. "placeholder" keeps the legacy base+index catalog
    # times (1 s spacing — NOT physical; time marks then exist only for truth
    # pairs via metadata). "uniform" draws each source's arrival uniformly
    # over t_obs_days, so EVERY candidate edge carries a physical
    # |Delta t_gps| — required for pair_marks=time on graphs with false
    # edges (case G at scale).
    if observation_times not in ("placeholder", "uniform"):
        raise ValueError(f"unknown observation_times mode: {observation_times!r}")
    uniform_times = observation_times == "uniform"
    t_obs_window_sec = float(t_obs_days) * 86400.0
    singleton_arrival = (
        rng.uniform(0.0, t_obs_window_sec, len(sing_idx)) if uniform_times else None
    )

    pairs = []
    for i in pair_src_idx:
        img0 = make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                             marks["mu_plus"][i], model, nsamp, rng, H0, Om0,
                             ra_true=src["ra_true"][i], dec_true=src["dec_true"][i])
        img1 = make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                             marks["mu_minus"][i], model, nsamp, rng, H0, Om0,
                             ra_true=src["ra_true"][i], dec_true=src["dec_true"][i])
        true_dt = float(delta_t_from_y(jnp.asarray(marks["y"][i]), sis))
        dt_obs = float(true_dt + rng.normal(0.0, time_delay_sigma_sec))
        # Arrival convention: catalog times are the MEASURED arrivals, so the
        # noisy delay lives in the arrival difference and the builder's
        # |Delta t_gps| reproduces delta_t_obs exactly for truth pairs.
        t0 = float(rng.uniform(0.0, t_obs_window_sec)) if uniform_times else 0.0
        pairs.append(dict(
            images=(img0, img1), true_y=float(marks["y"][i]),
            true_delta_t=true_dt, delta_t_obs=dt_obs,
            sigma_delta_t=float(time_delay_sigma_sec),
            t_obs_image0=t0, t_obs_image1=t0 + dt_obs,
            true_mu0=float(marks["mu_plus"][i]), true_mu1=float(marks["mu_minus"][i]),
            source_index=int(i),
        ))

    # ---- unified observed catalog / truth partition indices ----
    S, P = len(events), len(pairs)
    pair_indices = [[S + 2 * k, S + 2 * k + 1] for k in range(P)]
    observed_events = list(events)
    for pair in pairs:
        observed_events.extend(pair["images"])

    observed_pe_path = os.path.join(out_dir, "mock_observed_gw_pe.h5")
    contract_observed_pe_path = os.path.join(out_dir, "observed_gw_pe.h5")
    observed_catalog_path = os.path.join(out_dir, "observed_catalog.json")
    if write_unified_observed_catalog:
        write_gw_pe_file(observed_events, observed_pe_path, nsamp, H0, Om0)

    write_pair_metadata_file(pairs, os.path.join(out_dir, "mock_pair_metadata.h5"), pair_indices=pair_indices)
    if (not write_unified_observed_catalog) or write_legacy_pair_pe:
        write_pair_pe_file(pairs, os.path.join(out_dir, "mock_pair_pe.h5"), nsamp, pair_indices=pair_indices)

    # ---- injection campaigns ----
    generate_unlensed_injections(n_unlensed_inj, model, rng, H0, Om0,
                                 out_path=os.path.join(out_dir, "mock_gw_selection.h5"),
                                 proposal=injection_proposal,
                                 theta=truth["theta"])
    lensed_inj_summary, _ = generate_lensed_injections(
        n_lensed_inj, model, rng, H0, Om0,
        out_path=os.path.join(out_dir, "mock_lensed_injections.h5"),
        pair_tag_model=pair_tag_model, pair_tag_prob=pair_tag_prob,
        sis=sis,
    )

    # ---- partition (TRUE): [singletons 0..S-1, then pair images S+2k, S+2k+1] ----
    partition = dict(
        n_singletons=S, n_pairs=P,
        singleton_indices=list(range(S)),
        pair_indices=pair_indices,
        truth=dict(
            singleton_source_indices=[int(i) for i in sing_idx],
            pair_source_indices=[int(i) for i in pair_src_idx],
            pair_time_delays=[dict(source_index=p["source_index"], true_y=p["true_y"], true_delta_t=p["true_delta_t"], delta_t_obs=p["delta_t_obs"], sigma_delta_t=p["sigma_delta_t"]) for p in pairs],
            pair_partition=pair_indices,
        ),
        note="event order: singletons [0..S-1], then pair images [S+2k, S+2k+1]",
    )
    with open(os.path.join(out_dir, "partition.json"), "w") as f:
        json.dump(partition, f, indent=2)

    event_records = []
    schema_events = []
    base_gps_time = 1234567890.0
    for event_index, (source_index, is_lensed) in enumerate(
        zip(sing_idx, sing_idx_is_lensed)
    ):
        image_index = (
            (0 if dbl["det_plus"][source_index] else 1) if is_lensed else None
        )
        event_records.append(dict(
            event_index=int(event_index),
            kind="lensed_single_image" if is_lensed else "singleton",
            source_index=int(source_index),
            pair_index=None,
            image_index=image_index,
            truth_partner_event_index=None,
        ))
        gps = (
            base_gps_time + singleton_arrival[event_index]
            if uniform_times else base_gps_time + event_index
        )
        schema_events.append(dict(
            event_index=int(event_index),
            event_id=f"mock_event_{event_index:03d}",
            kind="singleton_or_image",
            gps_time=float(gps),
            truth_source_id=int(source_index),
            truth_image_index=image_index,
            truth_is_lensed_image=bool(is_lensed),
            ra_mean=float(events[event_index]["ra_mean"]),
            dec_mean=float(events[event_index]["dec_mean"]),
            sky_sigma_rad=float(events[event_index]["sky_sigma_rad"]),
            ra_true=float(events[event_index]["ra_true"]),
            dec_true=float(events[event_index]["dec_true"]),
        ))
    for k, (pair, (i0, i1)) in enumerate(zip(pairs, pair_indices)):
        event_records.extend([
            dict(event_index=int(i0), kind="lensed_image", source_index=int(pair["source_index"]),
                 pair_index=int(k), image_index=0, truth_partner_event_index=int(i1)),
            dict(event_index=int(i1), kind="lensed_image", source_index=int(pair["source_index"]),
                 pair_index=int(k), image_index=1, truth_partner_event_index=int(i0)),
        ])
        for image_index, event_index in enumerate((i0, i1)):
            gps = (
                base_gps_time + (pair["t_obs_image0"] if image_index == 0
                                 else pair["t_obs_image1"])
                if uniform_times else base_gps_time + event_index
            )
            schema_events.append(dict(
                event_index=int(event_index),
                event_id=f"mock_event_{event_index:03d}",
                kind="singleton_or_image",
                gps_time=float(gps),
                truth_source_id=int(pair["source_index"]),
                truth_image_index=int(image_index),
                truth_is_lensed_image=True,
                ra_mean=float(observed_events[event_index]["ra_mean"]),
                dec_mean=float(observed_events[event_index]["dec_mean"]),
                sky_sigma_rad=float(observed_events[event_index]["sky_sigma_rad"]),
                ra_true=float(observed_events[event_index]["ra_true"]),
                dec_true=float(observed_events[event_index]["dec_true"]),
            ))
    observed_catalog = dict(
        format_version="observed-lensing-catalog-1.0",
        event_indexing="global",
        observation_times=str(observation_times),
        t_obs_days=float(t_obs_days),
        time_delay_sigma_sec=float(time_delay_sigma_sec),
        n_events=int(S + 2 * P),
        events=schema_events,
        event_order="singletons first, then lensed image pairs",
        event_records=event_records,
        truth_partition=dict(
            singleton_indices=partition["singleton_indices"],
            pair_indices=partition["pair_indices"],
            n_singletons=S, n_pairs=P,
        ),
    )
    if write_unified_observed_catalog:
        with open(observed_catalog_path, "w") as f:
            json.dump(observed_catalog, f, indent=2)
        write_observed_pe_attrs(
            observed_pe_path,
            n_events=int(S + 2 * P),
            catalog_path=observed_catalog_path,
            source="mock_lensing",
        )
        shutil.copyfile(observed_pe_path, contract_observed_pe_path)

    # ---- candidate pairs for exact partition marginalization ----
    true_edges = []
    for k in range(P):
        edge = {
            "i": int(S + 2 * k),
            "j": int(S + 2 * k + 1),
            "log_prior_odds": float(candidate_pair_log_prior_odds),
            "label": "true",
        }
        marks_edge = {
            "log_sky_overlap": float(
                _log_sky_overlap(
                    observed_events[edge["i"]],
                    observed_events[edge["j"]],
                    sigma_floor_rad=1e-3,
                )
            ),
        }
        if candidate_time_marks:
            marks_edge.update({
                "delta_t_obs": float(pairs[k]["delta_t_obs"]),
                "sigma_delta_t": float(pairs[k]["sigma_delta_t"]),
            })
        edge["marks"] = marks_edge
        true_edges.append(edge)
    used_edges = {tuple(edge[x] for x in ("i", "j")) for edge in true_edges}
    wrong_edges = []
    n_events_total = S + 2 * P
    max_wrong_edges = n_events_total * (n_events_total - 1) // 2 - len(used_edges)
    if n_wrong_candidate_pairs > max_wrong_edges:
        raise ValueError(
            f"requested {n_wrong_candidate_pairs} wrong candidate pairs but only "
            f"{max_wrong_edges} are available"
        )
    while len(wrong_edges) < n_wrong_candidate_pairs:
        i, j = sorted(rng.choice(n_events_total, size=2, replace=False).astype(int).tolist())
        edge = (i, j)
        if edge in used_edges:
            continue
        used_edges.add(edge)
        wrong_edges.append({
            "i": i, "j": j,
            "log_prior_odds": float(wrong_candidate_log_prior_odds),
            "label": "wrong",
            "marks": {
                "log_sky_overlap": float(
                    _log_sky_overlap(
                        observed_events[i], observed_events[j], sigma_floor_rad=1e-3
                    )
                )
            },
        })
    candidate_pairs = {
        "n_events": n_events_total,
        "candidate_pairs": true_edges + wrong_edges,
    }
    candidate_pairs["format_version"] = "candidate-pairs-1.0"
    candidate_pairs["pairs"] = candidate_pairs["candidate_pairs"]
    # Flat constant log_prior_odds above: no mark values are folded in.
    candidate_pairs["folded_mark_keys"] = []
    with open(os.path.join(out_dir, "candidate_pairs.json"), "w") as f:
        json.dump(candidate_pairs, f, indent=2)
    if build_candidate_pairs_from_observed:
        observed_built = build_candidate_pairs(
            gw_path=observed_pe_path,
            observed_catalog_path=observed_catalog_path,
            truth_path=os.path.join(out_dir, "truth.json"),
            max_edges_per_event=max(1, min(8, n_events_total - 1)),
            max_total_edges=max(1, n_events_total * max(1, min(8, n_events_total - 1)) // 2),
            time_window_sec=float("inf"),
            mass_distance_top_k=0,
            include_time_marks=bool(candidate_time_marks),
            include_truth_labels=True,
            include_sky_marks=True,
            sky_overlap_weight=0.0,
            sky_sigma_floor_rad=1e-3,
            seed=int(seed),
        )
        with open(os.path.join(out_dir, "candidate_pairs.json"), "w") as f:
            json.dump(observed_built, f, indent=2)

    # ---- consolidated selection contract (links legacy component files) ----
    with h5py.File(os.path.join(out_dir, "selection_inputs.h5"), "w") as f:
        f.attrs["format_version"] = "lensing-selection-inputs-1.0"
        f.attrs["source"] = "mock_lensing"
        f.attrs["unlensed_path"] = "mock_gw_selection.h5"
        f.attrs["lensed_path"] = "mock_lensed_injections.h5"
        f["unlensed"] = h5py.ExternalLink("mock_gw_selection.h5", "/")
        f["lensed"] = h5py.ExternalLink("mock_lensed_injections.h5", "/")

    # ---- truth.json (informational) ----
    truth_out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in truth.items()}
    truth_out["theta_param_order"] = _theta_param_order()
    truth_out.update(
        n_singletons_detected_total=n_singletons_detected_total,
        n_pairs_both_detected_total=n_pairs_both_detected_total,
        n_lensed_singletons_detected_total=n_lensed_singletons_detected_total,
        include_lensed_singletons=bool(include_lensed_singletons),
        observation_times=str(observation_times),
        t_obs_days=float(t_obs_days),
        n_lensed_singletons_kept=int(np.sum(sing_idx_is_lensed)),
        n_singletons_kept=S,
        n_pairs_kept=P,
        caps_applied=caps_applied,
        pair_partition_truth=partition["truth"],
        pair_time_delay_truth=partition["truth"]["pair_time_delays"],
    )
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth_out, f, indent=2)


    tiny_recovery_command = (
        "darksirens_inference_lensing "
        f"--gw_path {os.path.join(out_dir, 'mock_gw_pe.h5')} "
        f"--gwselection_path {os.path.join(out_dir, 'mock_gw_selection.h5')} "
        f"--lensed_injections_path {os.path.join(out_dir, 'mock_lensed_injections.h5')} "
        f"--pair_metadata_path {os.path.join(out_dir, 'mock_pair_metadata.h5')} "
        f"--partition_path {os.path.join(out_dir, 'partition.json')} "
        "--cluster_mode j2 --wl_backend lognormal --fix_cosmology true --fix_survey true "
        "--sampler tinyns --nlive 32 --max_samples 256 --pe_max_per_pair 16 "
        f"--sl_tau_A {float(sis.A_tau)} --sl_tau_n {float(sis.n_tau)} "
        f"--sl_T0_sec {float(sis.T0)}"
    )
    if validation_sample_log10_tau_A:
        lo, hi = (float(validation_log10_tau_A_prior[0]), float(validation_log10_tau_A_prior[1]))
        fixed_json = json.dumps({"tau_n": float(sis.n_tau)})
        override_json = json.dumps({"log10_tau_A": [lo, hi]})
        tiny_recovery_command += (
            " --fix_lens_rate false "
            f"--fixed_parameter_values '{fixed_json}' "
            f"--lens_prior_overrides '{override_json}'"
        )

    # ---- manifest.json (informational) ----
    manifest = dict(
        description="Standalone strong-lensing mock for darksirens_inference_lensing "
                    f"(population {POP_NAME} + SIS strong lensing + lognormal WL).",
        files={
            "mock_gw_pe.h5": "legacy singleton PE; gwcat-1.0; load via load_gw_samples",
            "mock_observed_gw_pe.h5": "legacy name for unified observed-event PE",
            "observed_gw_pe.h5": "contract unified observed-event PE; observed-lensing-pe-1.0; global event indices",
            "observed_catalog.json": "metadata for unified observed event-index system",
            "mock_pair_metadata.h5": "metadata-only pair/candidate-edge marks; lensing-pair-metadata-1.0",
            "mock_pair_pe.h5": "legacy split-pair PE per image; lensed-pair-pe-1.0 (written only in legacy mode or with --write-legacy-pair-pe true)",
            "selection_inputs.h5": "contract consolidated selection inputs linking singleton and lensed injection campaigns",
            "mock_gw_selection.h5": "legacy unlensed injections; gwcat-selection-1.0; load_selection_samples",
            "mock_lensed_injections.h5": "lensed J=2 injections; load_lensed_injections; includes p_tag_per_source pair-tag metadata",
            "partition.json": "TRUE partition (singleton_indices, pair_indices, source-index truth)",
            "candidate_pairs.json": "candidate-pair graph for --partition_mode marginalize_exact",
        },
        lensed_injection_schema=dict(
            pair_tag_dataset="p_tag_per_source",
            pair_tag_model=pair_tag_model,
            pair_tag_prob=float(pair_tag_prob),
            note="min_snr_proxy is a deterministic mock-only proxy based on the weaker image SNR strength",
        ),
        pair_pe_schema=dict(
            format_version="lensed-pair-pe-1.0",
            layout="root/pair_{k}/image{0,1}/datasets",
            attrs=[
                "format_version", "npairs", "nsamp",
                "prior_weight_convention", "mock_data",
            ],
            coordinates=["m1det", "q", "dL_app", "chieff"],
            time_delay={"per_pair_attrs": ["delta_t_obs", "sigma_delta_t", "true_y", "true_delta_t", "t_obs_image0", "t_obs_image1"]},
            prior_wt=(
                "raw PE prior-density values are written in each image; "
                "darksirens_inference_lensing validates finite positive values "
                "and normalizes prior_wt per image at load time"
            ),
        ),
        counts=dict(
            n_sources_universe=n_universe,
            n_singletons_detected_total=n_singletons_detected_total, n_singletons_kept=S,
            n_pairs_both_detected_total=n_pairs_both_detected_total, n_pairs_kept=P,
            conditioning=conditioning, caps_applied=caps_applied,
            lensed_injection_pair_tag=lensed_inj_summary,
            nsamp=nsamp, n_unlensed_injections=n_unlensed_inj,
            n_lensed_injection_sources=n_lensed_inj,
            n_wrong_candidate_pairs=int(n_wrong_candidate_pairs),
            unified_observed_catalog=bool(write_unified_observed_catalog),
            candidate_pairs_from_observed=bool(build_candidate_pairs_from_observed),
        ),
        model=dict(pop_name=POP_NAME, rho_thr=rho_thr, horizon_Mpc=horizon_Mpc,
                   selection_model="Finn-Chernoff orientation-averaged p_det",
                   pe_prior_convention="p_pe proportional to m1det * dL^2 in the (m1det, q, dL) basis",
                   cosmology=f"H0={H0}, Om0={Om0}",
                   tau_A=float(sis.A_tau), tau_n=float(sis.n_tau),
                   T0_sec=float(sis.T0),
                   wl_a=float(wl.a), wl_b=float(wl.b)),
        validation=dict(
            sample_log10_tau_A=bool(validation_sample_log10_tau_A),
            tiny_recovery_command=tiny_recovery_command,
        ),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="data/mock_lensing", help="output directory")
    p.add_argument("--pop_model", "--pop-model", dest="pop_model",
                   choices=("powerlaw+peak", "powerlaw+peak@md"),
                   default="powerlaw+peak",
                   help="generator population: powerlaw+peak with a power-law "
                        "(1+z)^(gamma-1) rate, or '@md' for the Madau-Dickinson "
                        "peaked rate (gamma, kappa, z_peak)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--nsamp", type=int, default=1000, help="PE samples per object")
    p.add_argument("--n-universe", type=int, default=120_000, help="source draws")
    p.add_argument("--n-sing-keep", type=int, default=200, help="singletons kept")
    p.add_argument("--n-pair-keep", type=int, default=40, help="pairs kept")
    p.add_argument("--conditioning", choices=("fixed_counts", "poisson_counts"),
                   default="fixed_counts",
                   help="fixed_counts preserves exact requested debug counts; "
                        "poisson_counts keeps the stochastic detected counts")
    p.add_argument("--max-sing-keep", type=int, default=None,
                   help="optional singleton cap for --conditioning poisson_counts")
    p.add_argument("--max-pair-keep", type=int, default=None,
                   help="optional pair cap for --conditioning poisson_counts")
    p.add_argument("--n-unlensed-inj", type=int, default=200_000)
    p.add_argument("--injection-proposal", choices=("broad", "matched"),
                   default="broad",
                   help="unlensed injection campaign proposal; 'matched' uses "
                        "a fiducial-shaped + detectability-tilted mixture so "
                        "selection Neff tracks the detected count (required "
                        "for paper-scale catalogs under the selection "
                        "variance guard)")
    p.add_argument("--n-lensed-inj", type=int, default=300_000, help="lensed source draws")
    p.add_argument("--rho-thr", type=float, default=8.0, help="network SNR threshold")
    p.add_argument("--horizon-mpc", type=float, default=3000.0, help="sets r0")
    p.add_argument("--H0", type=float, default=float(H0Planck))
    p.add_argument("--Om0", type=float, default=float(Om0Planck))
    p.add_argument("--tau-A", type=float, default=5.0e-4, help="SIS optical-depth amplitude")
    p.add_argument("--tau-n", type=float, default=3.0, help="SIS optical-depth z-power")
    p.add_argument("--T0-sec", type=float, default=DEFAULT_T0_SECONDS,
                   help="SIS time-delay scale T0 in seconds (Delta t = T0 * y); default "
                        f"{DEFAULT_T0_SECONDS:.3g} s (~62 d) at z_L=0.5, z_s=1, sigma_v=200 km/s")
    p.add_argument("--wl-a", type=float, default=4.0e-3, help="WL lognormal variance amplitude")
    p.add_argument("--wl-b", type=float, default=1.5, help="WL lognormal variance z-power")
    p.add_argument("--pair-tag-model", choices=("none", "constant", "min_snr_proxy", "snr_time", "snr_time_sky"),
                   default="none", help="mock-only pair-tag selection model for lensed injections")
    p.add_argument("--pair-tag-prob", type=float, default=1.0,
                   help="pair-tag probability used by --pair-tag-model constant")
    p.add_argument("--n-wrong-candidate-pairs", "--candidate-extra-wrong-pairs", dest="n_wrong_candidate_pairs", type=int, default=0,
                   help="number of shuffled non-truth candidate edges to add to candidate_pairs.json")
    p.add_argument("--candidate-random-pairs", dest="n_wrong_candidate_pairs", type=int,
                   help="alias for --candidate-extra-wrong-pairs; random non-truth candidate edges")
    p.add_argument("--write-legacy-pair-pe", choices=("true", "false"), default="false",
                   help="write duplicated image PE to mock_pair_pe.h5; default false for unified observed mode")
    p.add_argument("--write-unified-observed-catalog", choices=("true", "false"), default="true",
                   help="write mock_observed_gw_pe.h5 and observed_catalog.json using one observed-event index system")
    p.add_argument("--candidate-pair-log-prior-odds", type=float, default=0.0,
                   help="log prior odds assigned to true candidate edges")
    p.add_argument("--wrong-candidate-log-prior-odds", type=float, default=-5.0,
                   help="log prior odds assigned to shuffled wrong candidate edges")
    p.add_argument("--candidate-time-marks", choices=("true", "false"), default="true",
                   help="write edge-level time-delay marks for true candidate edges")
    p.add_argument("--include-lensed-singletons", "--include_lensed_singletons",
                   dest="include_lensed_singletons",
                   choices=("true", "false"), default="false",
                   help="true adds strongly lensed sources with exactly one detected "
                        "image to the observed singleton pool (realistic protocol; "
                        "requires the sl_mixture singleton channel at inference); "
                        "false keeps the legacy protocol that drops them")
    p.add_argument("--build_candidate_pairs_from_observed", "--build-candidate-pairs-from-observed",
                   choices=("true", "false"), default="false",
                   help="overwrite candidate_pairs.json with a graph scored from observed metadata/posteriors")
    p.add_argument("--time-delay-sigma-sec", type=float, default=3600.0,
                   help="Gaussian sigma for observed SIS pair time delays, in seconds")
    p.add_argument("--observation-times", "--observation_times",
                   dest="observation_times",
                   choices=("placeholder", "uniform"), default="placeholder",
                   help="placeholder keeps legacy base+index catalog times; uniform draws "
                        "each source's arrival over --t-obs-days so every candidate edge "
                        "carries a physical |Delta t| (required for pair_marks=time with "
                        "false edges)")
    p.add_argument("--t-obs-days", "--t_obs_days", dest="t_obs_days",
                   type=float, default=365.25,
                   help="observing-run length in days for --observation-times uniform")
    p.add_argument("--validation-sample-log10-tau-A", action="store_true",
                   help="write/print a tiny validation command that samples log10_tau_A while fixing tau_n")
    p.add_argument("--validation-log10-tau-A-prior", type=float, nargs=2, default=(-7.0, -2.0),
                   metavar=("LO", "HI"), help="prior for the optional validation log10_tau_A recovery command")
    return p.parse_args()


def main():
    args = parse_args()
    set_pop_model(args.pop_model)
    sis = make_sis_lens_params(A_tau=args.tau_A, n_tau=args.tau_n,
                              T0_seconds=args.T0_sec)
    wl = make_lognormal_wl_params(a=args.wl_a, b=args.wl_b)
    manifest = assemble(
        args.outdir, n_universe=args.n_universe, seed=args.seed, nsamp=args.nsamp,
        n_sing_keep=args.n_sing_keep, n_pair_keep=args.n_pair_keep,
        conditioning=args.conditioning, max_sing_keep=args.max_sing_keep,
        max_pair_keep=args.max_pair_keep,
        rho_thr=args.rho_thr, horizon_Mpc=args.horizon_mpc,
        n_unlensed_inj=args.n_unlensed_inj, n_lensed_inj=args.n_lensed_inj,
        injection_proposal=args.injection_proposal,
        H0=args.H0, Om0=args.Om0, sis=sis, wl=wl,
        pair_tag_model=args.pair_tag_model, pair_tag_prob=args.pair_tag_prob,
        n_wrong_candidate_pairs=args.n_wrong_candidate_pairs,
        candidate_pair_log_prior_odds=args.candidate_pair_log_prior_odds,
        wrong_candidate_log_prior_odds=args.wrong_candidate_log_prior_odds,
        time_delay_sigma_sec=args.time_delay_sigma_sec,
        write_unified_observed_catalog=args.write_unified_observed_catalog.lower() == "true",
        candidate_time_marks=args.candidate_time_marks.lower() == "true",
        include_lensed_singletons=args.include_lensed_singletons.lower() == "true",
        observation_times=args.observation_times,
        t_obs_days=args.t_obs_days,
        build_candidate_pairs_from_observed=args.build_candidate_pairs_from_observed.lower() == "true",
        validation_sample_log10_tau_A=args.validation_sample_log10_tau_A,
        validation_log10_tau_A_prior=args.validation_log10_tau_A_prior,
        write_legacy_pair_pe=args.write_legacy_pair_pe.lower() == "true",
    )
    c = manifest["counts"]
    print(json.dumps(c, indent=2))
    print(f"partition: {c['n_singletons_kept']} singletons, {c['n_pairs_kept']} pairs")
    if args.validation_sample_log10_tau_A:
        print("tiny recovery command (samples log10_tau_A, fixes tau_n):")
        print(manifest["validation"]["tiny_recovery_command"])
    print(f"written to: {args.outdir}")


if __name__ == "__main__":
    main()
