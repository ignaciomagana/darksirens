import importlib
import importlib.util
import multiprocessing as mp
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Respect the start method if the embedding application already selected one.
    pass

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

import astropy
import numpy as np
import healpy as hp

import h5py
import astropy.units as u

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
_TQDM_MODULE = sys.modules.get("tqdm")
if _TQDM_MODULE is not None and hasattr(_TQDM_MODULE, "tqdm"):
    tqdm = _TQDM_MODULE.tqdm
elif importlib.util.find_spec("tqdm") is not None:
    tqdm = importlib.import_module("tqdm").tqdm
else:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _NoOpProgress:
                def update(self, *_args, **_kwargs):
                    return None

                def close(self):
                    return None

                def set_postfix(self, *_args, **_kwargs):
                    return None

            return _NoOpProgress()
        return iterable

from argparse import ArgumentParser
import glob

from darksirens.utils.cosmology import *

from gwcat.spin import chi_eff_prior_logprob

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in arctanh")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

def load_gw_samples(gw_path):
    """
    Load GW posterior samples from an HDF5 file and return flattened arrays
    of length (nEvents * nsamp).

    Parameters
    ----------
    gw_path : str
        Path to HDF5 file containing PE samples.

    Returns
    -------
    m1det, m2det, dL, chieff, ra, dec, p_pe : jnp.ndarray
        Flattened arrays of length (nEvents * nsamp). ``m1det`` and
        ``m2det`` are detector-frame masses in solar masses, ``dL`` is in
        Mpc, sky angles are radians, and ``chieff`` is dimensionless.
        After all transformations in this loader, ``p_pe`` is the
        per-event-normalised PE proposal density in the likelihood's
        canonical sample basis ``(m1det, q, dL)`` with
        ``q = m2det / m1det``.  Thus its native units before the
        per-event normalisation are inverse solar mass per unit mass-ratio
        per Mpc, times any spin/sky factors included by the input file.  If
        an input file provides a density in ``(m1det, m2det, dL)``, it must
        be converted to the ``q`` basis by multiplying by
        ``|dm2det/dq| = m1det`` before being stored as ``p_pe``.
    nEvents : int
        Number of GW events.
    """

    with h5py.File(gw_path, "r") as f:
        nsamp = int(f.attrs["nsamp"])
        nEvents = int(f.attrs["nobs"])

        # Load arrays as NumPy first (safer for reshaping)
        ra     = np.array(f["ra"])
        dec    = np.array(f["dec"])
        m1det  = np.array(f["m1det"])
        m2det  = np.array(f["m2det"])
        dL     = np.array(f["dL"]) * u.Mpc
        dL     = dL.value  # convert to float Mpc
        chieff = np.array(f["chieff"]) if "chieff" in f else np.zeros(dL.shape)
        p_pe = np.array(f["p_pe"]) if "p_pe" in f else np.ones(dL.shape)
        
        is_mock = bool(f.attrs.get("mock_data", False))

        # gwcat-1.0 files store the PE cosmology and pre-computed source masses
        _pe_H0 = float(f.attrs["pe_cosmology_H0"]) if "pe_cosmology_H0" in f.attrs else None
        _pe_Om0 = float(f.attrs["pe_cosmology_Om0"]) if "pe_cosmology_Om0" in f.attrs else None
        _pe_w0 = float(f.attrs["pe_cosmology_w0"]) if "pe_cosmology_w0" in f.attrs else None
        _pe_wa = float(f.attrs["pe_cosmology_wa"]) if "pe_cosmology_wa" in f.attrs else None
        _has_src = "m1src" in f and "m2src" in f
        if _has_src:
            m1source = np.array(f["m1src"])
            m2source = np.array(f["m2src"])

        _chi_in_ppe = bool(f.attrs.get("chi_eff_in_p_pe", False))
        _chi_amax = float(f.attrs.get("chi_eff_amax", 0.99))
    
    # Source masses: use pre-stored if available (gwcat-1.0), else compute
    if not _has_src:
        H0_ref = _pe_H0 if _pe_H0 is not None else Planck15.H0.value
        Om0_ref = _pe_Om0 if _pe_Om0 is not None else Planck15.Om0
        w0_ref = _pe_w0 if _pe_w0 is not None else -1.0
        wa_ref = _pe_wa if _pe_wa is not None else 0.0
        redshift = z_of_dL(dL, H0_ref, Om0_ref, w0_ref, wa_ref)
        m1source = m1det/(1+redshift)
        m2source = m2det/(1+redshift)
    
    # ------------------------------------------------------------
    # p_pe handling
    # ------------------------------------------------------------
    if is_mock:
        print("This is using mock data.")
    elif not _chi_in_ppe:
        # chi_eff not yet in p_pe — apply it now
        logp_chi = chi_eff_prior_logprob(chieff, m1source, m2source, amax=_chi_amax)
        p_pe = p_pe * np.exp(np.clip(logp_chi, -50.0, None))

    # Normalise per event so that each event's importance weights are
    # independent.  The per-event marginal likelihood is
    #   (1/nsamp) Σ_j  p_pop(θ_j) / p_pe(θ_j)
    # and dividing by the per-event sum makes the effective weights
    # dimensionless while preserving the correct relative scale within
    # each event.  Global normalisation (over nEvents*nsamp) would
    # introduce a factor of nEvents into every per-event sum, biasing
    # log μ and therefore the posterior on H0.
    p_pe = p_pe.reshape(nEvents, nsamp)
    p_pe = p_pe / p_pe.sum(axis=1, keepdims=True)
    p_pe = p_pe.flatten()
    
    # Convert to jnp in requested order.  m2det is retained so
    # make_gw_event can form q, but p_pe is already in the (m1det, q, dL)
    # proposal-density basis used by the likelihood.
    return (
        jnp.array(m1det),
        jnp.array(m2det),
        jnp.array(dL),
        jnp.array(chieff),
        jnp.array(ra),
        jnp.array(dec),
        jnp.array(p_pe),
        nEvents,
        nsamp
    )


def load_selection_samples(
    file,
    far_threshold=1.0,
    rng=None,
):
    """
    Return (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) for detected injections.

    Integration convention after all transformations:

    - ``m1det`` and ``m2det`` are detector-frame masses in solar masses.
    - ``dL`` is luminosity distance in Mpc.
    - ``ra`` and ``dec`` are radians, and ``chieff`` is dimensionless.
    - ``pdraw``/``p_draw`` is the physical injection proposal density in the
      likelihood's canonical basis ``(m1det, q, dL)`` with
      ``q = m2det / m1det``.  Its units are inverse solar mass per unit
      mass-ratio per Mpc per year, times any spin/sky factors included by
      the draw distribution.  Unlike ``p_pe``, selection densities are not
      normalised after loading because their absolute scale enters the
      expected-detection integral.

    - ndraw is the total number of generated injections (accepted + rejected).
    - If nsamp is not None, a subsample of detected injections is drawn with
      proper importance weighting so that ndraw stays the same and pdraw
      is corrected for the sampling probability.

    Parameters
    ----------
    file : str
        Path to injection file.
    nsamp : int or None
        Number of detected injections to return. If None, return all detected.
    far_threshold : float
        FAR threshold (per year) for detection.
    rng : np.random.Generator or None
        RNG for subsampling. If None, a new default_rng() is created.

    Returns
    -------
    m1detsels : jnp.ndarray
    m2detsels : jnp.ndarray
    dLsels    : jnp.ndarray
    chieffsels: jnp.ndarray
    rasels    : jnp.ndarray
    decsels   : jnp.ndarray
    pdraw_sel : jnp.ndarray
    ndraw     : int
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, "r") as f:

        # Branch: gwcat pre-processed selection file (format_version="gwcat-selection-1.0")
        # Everything is already in (m1det, q, dL) basis except the 1-D chi_eff
        # swap, which we apply here.
        if f.attrs.get("format_version", "") == "gwcat-selection-1.0":
            m1detsels  = np.array(f["m1det"])
            m2detsels  = np.array(f["m2det"])
            dLsels     = np.array(f["dL"])
            chieffsels = np.array(f["chieff"])
            rasels     = np.array(f["ra"])
            decsels    = np.array(f["dec"])
            pdraw_sel  = np.array(f["pdraw"])
            m1src_sel  = np.array(f["m1src"])
            m2src_sel  = np.array(f["m2src"])
            ndraw      = int(f.attrs["ndraw"])

            # Apply 1-D chi_eff spin-prior swap if not already done
            if not f.attrs.get("chi_eff_swap_applied", True):
                _amax = float(f.attrs.get("chi_eff_amax", 0.99))
                log_p_chi = chi_eff_prior_logprob(chieffsels, m1src_sel, m2src_sel, amax=_amax)
                safe_log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
                pdraw_sel = pdraw_sel * np.exp(safe_log_p_chi)

            n_det = len(pdraw_sel)
            print(f"    [gwcat selection] {n_det:,} detected injections  "
                  f"Ndraw={ndraw:,}  "
                  f"H0={f.attrs.get('cosmology_H0', '?')}  "
                  f"Om0={f.attrs.get('cosmology_Om0', '?')}")
            print(f"    p_draw: min={pdraw_sel.min():.3e}  "
                  f"max={pdraw_sel.max():.3e}  "
                  f"mean={pdraw_sel.mean():.3e}")

            return (
                jnp.array(m1detsels),
                jnp.array(m2detsels),
                jnp.array(dLsels),
                jnp.array(chieffsels),
                jnp.array(rasels),
                jnp.array(decsels),
                jnp.array(pdraw_sel),
                ndraw,
            )

        # Branch 0 — mock selection file (generate_selection.py)
        if f.attrs.get("mock_data", False):
            print("This is using mock selection.")
            m1detsels  = np.array(f["m1detsels"][:])
            m2detsels  = np.array(f["m2detsels"][:])
            dLsels     = np.array(f["dLsels"][:])
            chieffsels = np.array(f["chieffsels"][:])
            rasels     = np.array(f["rasels"][:])
            decsels    = np.array(f["decsels"][:])
            pdraw_sel  = np.array(f["p_draw"][:])
            ndraw      = int(f.attrs["Ndraw"])

            n_det = len(m1detsels)
            print(f"    [mock selection] {n_det:,} injections  "
                  f"Ndraw={ndraw:,}  "
                  f"pop_model={f.attrs.get('pop_model', 'unknown')}")
            print(f"    p_draw: min={pdraw_sel.min():.3e}  "
                  f"max={pdraw_sel.max():.3e}  "
                  f"mean={pdraw_sel.mean():.3e}")

            return (
                jnp.array(m1detsels),
                jnp.array(m2detsels),
                jnp.array(dLsels),
                jnp.array(chieffsels),
                jnp.array(rasels),
                jnp.array(decsels),
                jnp.array(pdraw_sel),
                ndraw,
            )
        # ------------------------------------------------------------
        # Branch 1: "events/..." format (all current LVK injection sets)
        # ------------------------------------------------------------
        elif "events" in f:
            m1src_all = np.array(f["events"]["mass1_source"][:])
            m2src_all = np.array(f["events"]["mass2_source"][:])
            dL_all    = np.array(f["events"]["luminosity_distance"][:])
            ra_all    = np.array(f["events"]["right_ascension"][:])
            dec_all   = np.array(f["events"]["declination"][:])
            
            # Extract all spin components needed for the 6D analytical prior
            s1x_all   = np.array(f["events"]["spin1x"][:])
            s1y_all   = np.array(f["events"]["spin1y"][:])
            s1z_all   = np.array(f["events"]["spin1z"][:])
            s2x_all   = np.array(f["events"]["spin2x"][:])
            s2y_all   = np.array(f["events"]["spin2y"][:])
            s2z_all   = np.array(f["events"]["spin2z"][:])
            
            chieff_all = (m1src_all*s1z_all + m2src_all*s2z_all)/(m1src_all + m2src_all)
            
            H0Planck = Planck15.H0.value
            Om0Planck = Planck15.Om0
            w0Planck = -1.0
            waPlanck = 0.0

            z_all = z_of_dL(dL_all, H0Planck, Om0Planck, w0Planck, waPlanck)
            m1det_all = m1src_all * (1.0 + z_all)
            m2det_all = m2src_all * (1.0 + z_all)

            weights = np.array(f["events"]["weights"][:])

            # Extract joint probability
            ln_pdraw_joint = np.array(
                f["events"]["lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"][:]
            )
            
            # Analytically compute the 6D spin log-probability
            a1 = np.sqrt(s1x_all**2 + s1y_all**2 + s1z_all**2)
            a2 = np.sqrt(s2x_all**2 + s2y_all**2 + s2z_all**2)
            
            # LVK standard prior: p(s1, s2) = 1 / (16 * pi^2 * a1^2 * a2^2 * a_max^2)
            ln_pdraw_spin = -np.log(16.0 * np.pi**2 * a1**2 * a2**2 * 0.99**2)
            
            # Safely calculate 1D chi_eff draw probability
            log_p_chi = chi_eff_prior_logprob(chieff_all, m1src_all, m2src_all, amax=0.99)
            safe_log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
            
            # Swap the 6D spin probability for the 1D chi_eff probability in log-space
            ln_pdraw_effective = ln_pdraw_joint - ln_pdraw_spin + safe_log_p_chi

            # Convert from native source-frame component-mass draw
            # coordinates (m1src, m2src, z) to the canonical likelihood
            # basis (m1det, q, dL).  The absolute inverse Jacobian is
            # m1det / [(1+z)^2 * d(dL)/dz].
            pdraw_all = (
                np.exp(ln_pdraw_effective)
                * m1det_all
                / (1.0 + z_all) ** 2
                / ddL_of_z(z_all, dL_all, H0Planck, Om0Planck, w0Planck, waPlanck)
            )

            raw_searches = f.attrs["searches"]
            if hasattr(raw_searches, 'tolist'):
                raw_searches = raw_searches.tolist()
            searches = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in (raw_searches if hasattr(raw_searches, '__iter__')
                                  and not isinstance(raw_searches, (str, bytes))
                                  else [raw_searches])]

            far_all = np.min(
                [np.array(f["events"]["%s_far" % s][:]) for s in searches],
                axis=0,
            )

            ndraw = int(f.attrs["total_generated"])

            T = f.attrs["total_analysis_time"] / (3600.0 * 24.0 * 365.25)
            pdraw_all /= T
            pdraw_all /= weights

            detected = far_all < far_threshold

        else:
            if "injections" in f:
                raise RuntimeError(
                    "This file uses the legacy O3 'injections/' format, which "
                    "is no longer supported. Use the cumulative GWTC-5 injection "
                    "sets (Zenodo 19500052) which cover O1–O4b in the modern "
                    "'events/' format, or pre-process with gwcat.SelectionSet."
                )
            raise RuntimeError("Unrecognized injection file format: "
                               "no 'mock_data' attr or 'events' group.")
    # ------------------------------------------------------------
    # Restrict to detected injections
    # ------------------------------------------------------------
    m1detsels  = m1det_all[detected]
    m2detsels  = m2det_all[detected]
    chieffsels = chieff_all[detected]
    dLsels     = dL_all[detected]
    rasels     = ra_all[detected]
    decsels    = dec_all[detected]
    pdraw_sel  = pdraw_all[detected]

    # The selection integral requires
    #
    #   μ = (1/N_draw) Σ_det  p_pop(d_i|λ) / p_draw(d_i)
    #
    # so p_draw must retain its physical scale (per unit volume per unit
    # mass per unit time).  The previous code normalised pdraw_wt to
    # sum=1, which removed that scale and rendered log μ parameter-
    # independent — effectively turning the selection correction into a
    # constant that cannot track changes in the population model.
    #
    # We keep pdraw_sel as-is (already in physical units in the
    # (m1det, q, dL) basis after the Jacobian and time corrections applied
    # above) and do not renormalise.
    pdraw_wt = pdraw_sel
    print(f"    Selection samples: Ndet={len(pdraw_wt)}, Ndraw={ndraw}, "
          f"mean(p_draw)={pdraw_wt.mean():.3e}")

    # Convert to jnp in requested order.  m2det is retained for q
    # construction, but p_draw is already in the (m1det, q, dL) basis.
    return (
        jnp.array(m1detsels),
        jnp.array(m2detsels),
        jnp.array(dLsels),
        jnp.array(chieffsels),
        jnp.array(rasels),
        jnp.array(decsels),
        jnp.array(pdraw_wt),
        ndraw
    )