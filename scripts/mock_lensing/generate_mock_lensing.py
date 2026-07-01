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
  mock_pair_pe.h5           paired-image PE       (per-pair groups; read directly)
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
    tau_2_SIS, mu_plus_minus_from_y, make_sis_lens_params,
)
from darksirens.lensing.wlmagnification import make_lognormal_wl_params
from darksirens.lensing.lensed_injections import save_lensed_injections

POP_NAME = "powerlaw+peak"
THETA_PARAM_ORDER = [
    "v1", "alpha", "mmin", "mmax", "dmmin", "dmmax",
    "muG", "sigG", "beta", "muchi", "sigchi", "gamma",
]


# ============================================================
# Truth container
# ============================================================
def make_truth(seed, H0, Om0, sis, wl):
    """Pin all ground-truth hyperparameters by importing the fiducials."""
    theta = np.asarray(get_fixed_population_params(POP_NAME))
    return dict(
        pop_name=POP_NAME,
        theta=theta,                       # 12-vector
        gamma=float(theta[-1]),
        H0=float(H0), Om0=float(Om0), zMax=float(zMax),
        tau_A=float(sis.A_tau), tau_n=float(sis.n_tau),
        wl_a=float(wl.a), wl_b=float(wl.b),
        seed=int(seed),
    )


# ============================================================
# STEP 1a - mass / mass-ratio / spin via rejection sampling
# ============================================================
def _mixture_density(m1, q, chi, theta):
    """Imported mass*pairing*spin density (no z factor). Vectorised."""
    model = get_model(POP_NAME)
    tm = jnp.asarray(theta[:-1])           # drop gamma
    return np.asarray(model.mixture(jnp.asarray(m1), jnp.asarray(q),
                                    jnp.asarray(chi), tm))


def _analytic_proposal(n, theta, rng):
    """Tailored proposal close to the imported mixture, for high acceptance.

    Returns draws AND the proposal density g(m1,q,chi) for the IS correction.
    """
    v1, alpha, mmin, mmax, dmmin, dmmax, muG, sigG, beta, muchi, sigchi, gamma = \
        [float(x) for x in theta]

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

    def _pl_pdf(m):
        if abs(a) > 1e-8:
            norm = (hi**a - lo**a) / a
        else:
            norm = np.log(hi / lo)
        return np.where((m >= lo) & (m <= hi), m**(-alpha) / norm, 0.0)

    def _pk_pdf(m):
        sg = 1.5 * sigG
        return np.exp(-0.5 * ((m - muG) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)

    g_m1 = w_peak * _pk_pdf(m1) + (1 - w_peak) * _pl_pdf(m1)

    # --- q: q ~ U^(1/(beta+1)) gives pdf prop q^beta on (0,1] ---
    bp1 = beta + 1.0
    uq = rng.uniform(size=n)
    q = uq ** (1.0 / bp1) if abs(bp1) > 1e-8 else np.exp(np.log(uq))
    q = np.clip(q, 1e-3, 1.0)
    g_q = bp1 * q**beta if abs(bp1) > 1e-8 else 1.0 / q
    g_q = np.where((q > 0) & (q <= 1.0), g_q, 0.0)

    # --- chi: truncated Gaussian(mu_chi, sigma_chi) ---
    chi = rng.normal(muchi, 1.3 * sigchi, n)   # slightly broadened
    chi = np.clip(chi, -0.999, 0.999)
    sgc = 1.3 * sigchi
    Z = _norm.cdf((1 - muchi) / sgc) - _norm.cdf((-1 - muchi) / sgc)
    g_chi = np.exp(-0.5 * ((chi - muchi) / sgc) ** 2) / (np.sqrt(2 * np.pi) * sgc * Z)

    g = g_m1 * g_q * g_chi
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
    IMPORTED dV_of_z so the cosmology matches the likelihood."""
    gamma = float(theta[-1])
    zg = np.linspace(1e-4, float(zMax), nz)
    dV = np.asarray(dV_of_z(jnp.asarray(zg), H0, Om0))
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
    src = dict(m1=m1, q=q, chi=chi, z=z, dL_src=dL_src, m1det=m1det)
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
def generate_unlensed_injections(n_draw, model, rng, H0, Om0, *,
                                 m1det_range=(3.0, 200.0), out_path=None):
    """Unlensed singleton injections in the detector-frame proposal basis,
    written in the ``gwcat-selection-1.0`` schema."""
    m1lo, m1hi = m1det_range
    m1det = rng.uniform(m1lo, m1hi, n_draw)
    q = rng.uniform(0.0, 1.0, n_draw)
    chieff = rng.uniform(-1.0, 1.0, n_draw)

    # z from p_z(z) ∝ dV_c/dz on (0, zMax]; inverse-CDF
    zg = np.linspace(1e-4, float(zMax), 4000)
    dV = np.asarray(dV_of_z(jnp.asarray(zg), H0, Om0))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dV[1:] + dV[:-1]) * np.diff(zg))])
    Znorm = cdf[-1]; cdf /= cdf[-1]
    z = np.interp(rng.uniform(0, 1, n_draw), cdf, zg)
    pz_of_z = np.asarray(dV_of_z(jnp.asarray(z), H0, Om0)) / Znorm

    dL = np.asarray(dL_of_z(jnp.asarray(z), H0, Om0))
    ddL = np.asarray(ddL_of_z(jnp.asarray(z), jnp.asarray(dL), H0, Om0))

    # proposal density in the canonical (m1det, q, dL) basis (physical scale kept)
    p_draw = (1.0 / (m1hi - m1lo)) * 1.0 * 0.5 * (pz_of_z / ddL)

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
            for key in ("m1det", "m2det", "m1src", "m2src", "dL",
                        "chieff", "ra", "dec", "pdraw"):
                f.create_dataset(key, data=d[key])
    return d, int(detected.sum())


def generate_lensed_injections(n_draw_sources, model, rng, H0, Om0, *,
                               m1src_range=(3.0, 120.0), out_path=None,
                               pair_tag_model="none", pair_tag_prob=1.0):
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

    if pair_tag_model == "none":
        p_tag = np.ones(n_draw_sources, dtype=float)
    elif pair_tag_model == "constant":
        if not (0.0 <= pair_tag_prob <= 1.0):
            raise ValueError("pair_tag_prob must be in [0, 1]")
        p_tag = np.ones(n_draw_sources, dtype=float)
        p_tag[both] = float(pair_tag_prob)
    elif pair_tag_model == "min_snr_proxy":
        # Mock-only deterministic proxy: higher minimum image SNR strength gives
        # a higher chance that a pair-identification statistic would tag the pair.
        strength_p = model.expected_snr_optimal(m1_src, q, z, dL_src / np.sqrt(mu_p)) / model.rho_thr
        strength_m = model.expected_snr_optimal(m1_src, q, z, dL_src / np.sqrt(mu_m)) / model.rho_thr
        min_strength = np.minimum(strength_p, strength_m)
        p_tag = np.ones(n_draw_sources, dtype=float)
        p_tag[both] = np.clip(0.10 + 0.45 * (min_strength[both] - 1.0), 0.05, 1.0)
    else:
        raise ValueError(f"unknown pair_tag_model: {pair_tag_model}")

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


def make_event_pe(m1_src, q, z, chieff, mu, model, nsamp, rng, H0, Om0):
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
    ra = np.full(nsamp, rng.uniform(0, 2 * np.pi))
    dec = np.full(nsamp, np.arcsin(rng.uniform(-1, 1)))
    p_pe = _pe_prior_density(m1det, qs, dL)
    return dict(m1det=m1det, m2det=m2det, dL=dL, chieff=chi, ra=ra, dec=dec, p_pe=p_pe)


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
    return nobs


def write_pair_pe_file(pairs, path, nsamp):
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
        for k, (img0, img1) in enumerate(pairs):
            g = f.create_group(f"pair_{k}")
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
             wrong_candidate_log_prior_odds=-5.0):
    os.makedirs(out_dir, exist_ok=True)
    truth = make_truth(seed, H0, Om0, sis, wl)
    truth.update(rho_thr=rho_thr, horizon_Mpc=horizon_Mpc,
                 pair_tag_model=pair_tag_model, pair_tag_prob=pair_tag_prob,
                 n_sources_universe=n_universe, nsamp=nsamp,
                 conditioning=conditioning,
                 selection_model="Finn-Chernoff orientation-averaged p_det",
                 pe_prior_convention="p_pe proportional to m1det * dL^2 in the (m1det, q, dL) basis")

    rng = np.random.default_rng(seed)
    model = SNRModel(rho_thr=rho_thr, horizon_Mpc=horizon_Mpc)

    # ---- population + marks + selection ----
    d = generate_step12(n_universe, seed, H0, Om0, sis, wl)
    src, marks = d["src"], d["marks"]
    det_s, _, _ = apply_selection_singletons(src, marks, model, rng)
    dbl = apply_selection_doubles(src, marks, model, rng)

    sing_candidates = np.where(det_s)[0]
    pair_candidates = np.where(dbl["both_detected"])[0]
    n_singletons_detected_total = int(sing_candidates.size)
    n_pairs_both_detected_total = int(pair_candidates.size)

    # Randomize before any truncation so the kept catalog is an unbiased subset
    # of the detected mock rather than the first sources in simulation order.
    rng.shuffle(sing_candidates)
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
        pair_src_idx = pair_candidates[:n_pair_keep]
    elif conditioning == "poisson_counts":
        sing_idx = sing_candidates
        pair_src_idx = pair_candidates
        if max_sing_keep is not None and sing_idx.size > max_sing_keep:
            sing_idx = sing_idx[:max_sing_keep]
            caps_applied["singletons"] = True
        if max_pair_keep is not None and pair_src_idx.size > max_pair_keep:
            pair_src_idx = pair_src_idx[:max_pair_keep]
            caps_applied["pairs"] = True
    else:  # argparse enforces this; keep defensive guard for direct callers.
        raise ValueError(f"unknown conditioning mode: {conditioning}")

    # ---- PE: singletons + pairs ----
    events = [make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                            marks["mu"][i], model, nsamp, rng, H0, Om0)
              for i in sing_idx]
    write_gw_pe_file(events, os.path.join(out_dir, "mock_gw_pe.h5"), nsamp, H0, Om0)

    pairs = []
    for i in pair_src_idx:
        img0 = make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                             marks["mu_plus"][i], model, nsamp, rng, H0, Om0)
        img1 = make_event_pe(src["m1"][i], src["q"][i], src["z"][i], src["chi"][i],
                             marks["mu_minus"][i], model, nsamp, rng, H0, Om0)
        pairs.append((img0, img1))
    write_pair_pe_file(pairs, os.path.join(out_dir, "mock_pair_pe.h5"), nsamp)

    # ---- injection campaigns ----
    generate_unlensed_injections(n_unlensed_inj, model, rng, H0, Om0,
                                 out_path=os.path.join(out_dir, "mock_gw_selection.h5"))
    lensed_inj_summary, _ = generate_lensed_injections(
        n_lensed_inj, model, rng, H0, Om0,
        out_path=os.path.join(out_dir, "mock_lensed_injections.h5"),
        pair_tag_model=pair_tag_model, pair_tag_prob=pair_tag_prob,
    )

    # ---- partition (TRUE): [singletons 0..S-1, then pair images S+2k, S+2k+1] ----
    S, P = len(events), len(pairs)
    partition = dict(
        n_singletons=S, n_pairs=P,
        singleton_indices=list(range(S)),
        pair_indices=[[S + 2 * k, S + 2 * k + 1] for k in range(P)],
        truth=dict(
            singleton_source_indices=[int(i) for i in sing_idx],
            pair_source_indices=[int(i) for i in pair_src_idx],
            pair_partition=[[S + 2 * k, S + 2 * k + 1] for k in range(P)],
        ),
        note="event order: singletons [0..S-1], then pair images [S+2k, S+2k+1]",
    )
    with open(os.path.join(out_dir, "partition.json"), "w") as f:
        json.dump(partition, f, indent=2)

    # ---- candidate pairs for exact partition marginalization ----
    true_edges = [
        {
            "i": int(S + 2 * k),
            "j": int(S + 2 * k + 1),
            "log_prior_odds": float(candidate_pair_log_prior_odds),
            "label": "true",
        }
        for k in range(P)
    ]
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
        })
    candidate_pairs = {
        "n_events": n_events_total,
        "candidate_pairs": true_edges + wrong_edges,
    }
    with open(os.path.join(out_dir, "candidate_pairs.json"), "w") as f:
        json.dump(candidate_pairs, f, indent=2)

    # ---- truth.json (informational) ----
    truth_out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in truth.items()}
    truth_out["theta_param_order"] = THETA_PARAM_ORDER
    truth_out.update(
        n_singletons_detected_total=n_singletons_detected_total,
        n_pairs_both_detected_total=n_pairs_both_detected_total,
        n_singletons_kept=S,
        n_pairs_kept=P,
        caps_applied=caps_applied,
        pair_partition_truth=partition["truth"],
    )
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth_out, f, indent=2)

    # ---- manifest.json (informational) ----
    manifest = dict(
        description="Standalone strong-lensing mock for darksirens_inference_lensing "
                    f"(population {POP_NAME} + SIS strong lensing + lognormal WL).",
        files={
            "mock_gw_pe.h5": "singleton PE; gwcat-1.0; load via load_gw_samples",
            "mock_pair_pe.h5": "pair PE per image; lensed-pair-pe-1.0",
            "mock_gw_selection.h5": "unlensed injections; gwcat-selection-1.0; load_selection_samples",
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
        ),
        model=dict(pop_name=POP_NAME, rho_thr=rho_thr, horizon_Mpc=horizon_Mpc,
                   selection_model="Finn-Chernoff orientation-averaged p_det",
                   pe_prior_convention="p_pe proportional to m1det * dL^2 in the (m1det, q, dL) basis",
                   cosmology=f"H0={H0}, Om0={Om0}",
                   tau_A=float(sis.A_tau), tau_n=float(sis.n_tau),
                   wl_a=float(wl.a), wl_b=float(wl.b)),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="data/mock_lensing", help="output directory")
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
    p.add_argument("--n-lensed-inj", type=int, default=300_000, help="lensed source draws")
    p.add_argument("--rho-thr", type=float, default=8.0, help="network SNR threshold")
    p.add_argument("--horizon-mpc", type=float, default=3000.0, help="sets r0")
    p.add_argument("--H0", type=float, default=float(H0Planck))
    p.add_argument("--Om0", type=float, default=float(Om0Planck))
    p.add_argument("--tau-A", type=float, default=5.0e-4, help="SIS optical-depth amplitude")
    p.add_argument("--tau-n", type=float, default=3.0, help="SIS optical-depth z-power")
    p.add_argument("--wl-a", type=float, default=4.0e-3, help="WL lognormal variance amplitude")
    p.add_argument("--wl-b", type=float, default=1.5, help="WL lognormal variance z-power")
    p.add_argument("--pair-tag-model", choices=("none", "constant", "min_snr_proxy"),
                   default="none", help="mock-only pair-tag selection model for lensed injections")
    p.add_argument("--pair-tag-prob", type=float, default=1.0,
                   help="pair-tag probability used by --pair-tag-model constant")
    p.add_argument("--n-wrong-candidate-pairs", type=int, default=0,
                   help="number of shuffled non-truth candidate edges to add to candidate_pairs.json")
    p.add_argument("--candidate-pair-log-prior-odds", type=float, default=0.0,
                   help="log prior odds assigned to true candidate edges")
    p.add_argument("--wrong-candidate-log-prior-odds", type=float, default=-5.0,
                   help="log prior odds assigned to shuffled wrong candidate edges")
    return p.parse_args()


def main():
    args = parse_args()
    sis = make_sis_lens_params(A_tau=args.tau_A, n_tau=args.tau_n)
    wl = make_lognormal_wl_params(a=args.wl_a, b=args.wl_b)
    manifest = assemble(
        args.outdir, n_universe=args.n_universe, seed=args.seed, nsamp=args.nsamp,
        n_sing_keep=args.n_sing_keep, n_pair_keep=args.n_pair_keep,
        conditioning=args.conditioning, max_sing_keep=args.max_sing_keep,
        max_pair_keep=args.max_pair_keep,
        rho_thr=args.rho_thr, horizon_Mpc=args.horizon_mpc,
        n_unlensed_inj=args.n_unlensed_inj, n_lensed_inj=args.n_lensed_inj,
        H0=args.H0, Om0=args.Om0, sis=sis, wl=wl,
        pair_tag_model=args.pair_tag_model, pair_tag_prob=args.pair_tag_prob,
        n_wrong_candidate_pairs=args.n_wrong_candidate_pairs,
        candidate_pair_log_prior_odds=args.candidate_pair_log_prior_odds,
        wrong_candidate_log_prior_odds=args.wrong_candidate_log_prior_odds,
    )
    c = manifest["counts"]
    print(json.dumps(c, indent=2))
    print(f"partition: {c['n_singletons_kept']} singletons, {c['n_pairs_kept']} pairs")
    print(f"written to: {args.outdir}")


if __name__ == "__main__":
    main()
