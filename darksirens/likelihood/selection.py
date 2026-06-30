"""
selection.py
------------
Hierarchical selection integral for gravitational-wave population inference.

Physical picture
~~~~~~~~~~~~~~~~
The observed GW event rate depends on which signals pass the detection
threshold.  To avoid biasing the population inference, we must correct
for this selection effect via Thrane & Talbot (2019) / Farr (2019):

    log L_sel = -N_obs * log μ  +  N_obs(N_obs + 3) / (2 N_eff)

where μ is the expected number of detections per unit time under the
proposed population model, estimated as a Monte Carlo average over
injection samples:

    μ = (1/N_draw) Σ_{det inj}  p_pop(d_i|λ) / p_draw(d_i)

and N_eff is the effective sample size of the selection integral
(Farr 2019, arXiv:1904.10879, eq. 15):

    N_eff = μ² / Var(μ)   ≈   [Σ w_i]² / Σ w_i²

The coefficient N_obs(N_obs+3)/(2 N_eff) is the leading-order correction
from the uncertainty in μ on the log-likelihood (see the derivation in
the appendix of Talbot & Golomb 2023, arXiv:2209.02209, eq. A9).  This
is *not* the simpler N_obs²/(2 N_eff) term from the basic Farr (2019)
expansion; the extra factor of 3 arises from the next-order term in the
Taylor expansion of log μ around its mean.

Reliability criterion
~~~~~~~~~~~~~~~~~~~~~
Farr (2019) recommends discarding proposals where N_eff < 4 N_obs
(equivalently returning -inf).  We use the slightly more conservative
threshold of 5 N_obs following Vitale et al. (2022).

References
~~~~~~~~~~
- Farr, W.M. (2019). arXiv:1904.10879
- Thrane & Talbot (2019). PASA 36, e010
- Talbot & Golomb (2023). arXiv:2209.02209
- Vitale et al. (2022). arXiv:2007.05579
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.utils.utils import logdiffexp
from darksirens.core.types import GWEvent, EMCatalog
from darksirens.likelihood.events import pad_gw_event_to_multiple


# ============================================================
# Core estimators (testable in isolation)
# ============================================================

def _lse_to_log_mu_neff(
    lse: jnp.ndarray,
    lse2: jnp.ndarray,
    Ndraw: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert logsumexp aggregates to (log_mu, N_eff, log_sigma2).
 
    Parameters
    ----------
    lse  : logsumexp(log_weights)
    lse2 : logsumexp(2 * log_weights)
    Ndraw : total number of generated injections
 
    Returns
    -------
    log_mu : scalar
    Neff   : scalar  (0.0 when log_mu = -inf; never NaN)
    log_sigma2 : scalar  (log Monte-Carlo variance of the selection integral μ;
        consumed by the strong-lensing cluster-selection combiner. May be
        -inf/NaN when log_mu = -inf, where Neff = 0 already forces the
        too-sparse -inf selection correction downstream.)
    """
    log_Ndraw  = jnp.log(Ndraw)
    log_mu     = lse  - log_Ndraw
    log_s2     = lse2 - 2.0 * log_Ndraw
 
    # Var(μ) estimator in log-space
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - log_Ndraw)
 
    # Guard: when log_mu = -inf (all weights −∞), both log_mu and
    # log_sigma2 are -inf.  The subtraction 2*(-inf) - (-inf) = NaN.
    # We instead set Neff = 0.0, which triggers too_sparse → return -inf.
    finite_mu = jnp.isfinite(log_mu)
    Neff = jnp.where(
        finite_mu,
        jnp.exp(2.0 * log_mu - log_sigma2),
        0.0,
    )
 
    return log_mu, Neff, log_sigma2


def selection_log_correction(
    log_mu: jnp.ndarray,
    Neff: jnp.ndarray,
    nEvents: int,
) -> jnp.ndarray:
    """
    Log selection correction term (Farr 2019 / Talbot & Golomb 2023).

    Returns ``-inf`` when N_eff < 5 * N_obs (Vitale et al. 2022 criterion),
    indicating the injection set is too sparse for a reliable estimate.

    The correction is:

        -N_obs * log μ  +  N_obs * (N_obs + 3) / (2 * N_eff)

    The first term is the standard Poisson selection factor.  The second
    is the leading uncertainty correction from Taylor-expanding log μ.

    Parameters
    ----------
    log_mu : log of the selection integral estimate
    Neff   : effective sample size of the selection integral
    nEvents : number of observed GW events

    Returns
    -------
    Scalar log-likelihood contribution from the selection term.
    """
    too_sparse = Neff <= 5 * nEvents
    correction = (
        -nEvents * log_mu
        + nEvents * (3 + nEvents) / (2.0 * Neff)
    )
    return jnp.where(too_sparse, -jnp.inf, correction)


# ============================================================
# Full selection term (batched or unbatched)
# ============================================================

def compute_selection_term(
    gw_sel: GWEvent,
    em_catalog_sel: EMCatalog,
    log_weight_fn,
    Ndraw: float,
    nEvents: int,
    sel_batch_size: int | None = None,
    sky_log_weight_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Estimate log μ, N_eff and log_sigma2 from the injection set.

    Returns the 3-tuple ``(log_mu, Neff, log_sigma2)`` (the third element is the
    log Monte-Carlo variance of μ, used by the strong-lensing cluster path; the
    standard single-event likelihood ignores it).

    Parameters
    ----------
    gw_sel : GWEvent
        Injection samples (detected).  If ``sel_batch_size`` is set and the
        length is not divisible by the batch size, the event is padded with
        explicit invalid sentinel rows before scanning.
    em_catalog_sel : EMCatalog
        EM catalog sliced to the injection sky positions.
    log_weight_fn : callable(m1det, q, dL, chieff, pix, prior_wt, catalog) → array
        Per-sample log importance weight.  Must broadcast over the batch
        dimension.  Typically a closure from ``likelihood.py`` that captures
        cosmo, survey, pop_params, and the finite-guard for log_prior_z.
    Ndraw : float
        Total number of generated injections (detected + missed).
    nEvents : int
        Number of observed GW events (for the N_eff reliability check).
    sel_batch_size : int or None
        If not None, process injections in chunks via ``lax.scan`` to
        limit peak GPU memory.  Non-divisible inputs are padded internally;
        padded rows have ``valid == False`` and contribute zero weight.
    sky_log_weight_fn : callable(nx, ny, nz, dL) → array or None
        Optional sky factor ``log g(n̂, z)`` added to each injection's log
        importance weight (the same factor applied to the PE term), so the
        selection integral reweights ``μ`` consistently.  ``dL`` is passed so the
        closure can derive the redshift ``z`` for 3-D ``g(n̂, z)`` models with the
        same cosmology as the PE term.  ``None`` (default) leaves the integral
        sky-agnostic — the isotropic / legacy behaviour.

    Returns
    -------
    log_mu : scalar — log of the selection integral estimate
    Neff   : scalar — effective sample size
    """
    def _batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b, valid_b, nx_b, ny_b, nz_b):
        ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        if sky_log_weight_fn is not None:
            ldw = ldw + sky_log_weight_fn(nx_b, ny_b, nz_b, dL_b)
        valid = valid_b & (pwt_b > 0.0)
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        # --- Unbatched: process all injections at once ---
        lse, lse2 = _batch_lse(
            gw_sel.dL,
            gw_sel.m1det,
            gw_sel.q,
            gw_sel.chieff,
            gw_sel.pixels,
            gw_sel.prior_wt,
            gw_sel.valid,
            gw_sel.nx,
            gw_sel.ny,
            gw_sel.nz,
        )
    else:
        # --- Batched via lax.scan ---
        # Peak GPU memory: O(sel_batch_size × N_grid) instead of O(N_sel × N_grid).
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)
        N_sel = gw_sel.dL.shape[0]
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            if sky_log_weight_fn is not None:
                nx_b, ny_b, nz_b = sl(gw_sel.nx), sl(gw_sel.ny), sl(gw_sel.nz)
            else:
                nx_b = ny_b = nz_b = None
            lse_b, lse2_b = _batch_lse(
                sl(gw_sel.dL),
                sl(gw_sel.m1det),
                sl(gw_sel.q),
                sl(gw_sel.chieff),
                sl(gw_sel.pixels),
                sl(gw_sel.prior_wt),
                sl(gw_sel.valid),
                nx_b,
                ny_b,
                nz_b,
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(
            _scan_fn, None, jnp.arange(N_batches)
        )
        # Combine per-batch logsumexp values: logsumexp is additive
        # across disjoint index sets.
        lse  = logsumexp(lse_all)
        lse2 = logsumexp(lse2_all)

    return _lse_to_log_mu_neff(lse, lse2, Ndraw)
