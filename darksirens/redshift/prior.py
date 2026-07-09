"""
prior.py
--------
Redshift prior assembly for dark-siren and spectral-siren cosmological
inference with gravitational waves.

Physical picture
~~~~~~~~~~~~~~~~
We want p(z | pix, Θ) — the probability that a GW source at sky
position ``pix`` has redshift z, given cosmological parameters Θ.

Four regimes are supported:

1. ``"spectral_sirens"``
   GW data only.  Prior is the comoving volume element dV_c/dz.

2. ``"bright_sirens"``
   Counterpart-informed inference; see ``_log_prior_bright_sirens``.

3. ``"dark_sirens_complete"``
   EM catalog assumed 100 % complete: p(z | pix) = p_cat(z | pix), with
   an explicit empty-pixel policy (``zero`` default, ``volume`` fallback).

4. ``"dark_sirens"``  (default)
   Incomplete catalog.  The prior is additive in galaxy *densities*
   (counts per unit z), the in/out-of-catalog decomposition of
   Gray et al. 2020 (arXiv:1908.06050) / Gair et al. 2023 (AJ 166, 22):

       p(z | pix) = [ N_obs(pix) * p_cat(z | pix)  +  dN_miss(z | pix) ]
                    / [ N_obs(pix) + N_miss(pix) ]

   with p_cat the per-pixel *normalised* weighted-kernel catalog shape
   (catalog.py), N_obs the observed real-galaxy count in the pixel,
   dN_miss = (1 - C(z)) * dN_exp(z) * LSS the missing-galaxy density
   (completion.py), and N_miss = ∫ dN_miss dz.  Writing the catalog term
   as N_obs * p_cat assigns the missing population the pixel-mean
   observed weight (⟨w⟩_miss = Σ_i w_i / N_obs): the catalog:missing
   odds are the count odds N_obs : N_miss, while the within-catalog
   shape follows the weights.  This is conservative under luminosity
   weighting (true missing galaxies are typically fainter, so if
   anything the missing branch is over-weighted).

   The prior integrates to 1 per pixel by construction — no z-dependent
   mixture weight, no per-pixel normalisation residue for the selection
   term to (fail to) absorb.  Empty pixels route to the missing branch
   automatically (N_obs = 0).

Two-phase evaluation (performance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Everything except the sample redshift depends only on (pixel, Θ).  The
state API splits the work accordingly:

    state = prepare_redshift_prior_state(model, cosmo, survey, em_catalog)
    log_p = eval_redshift_prior_with_state(model, state, z, pix,
                                           cosmo, survey, em_catalog)

``prepare`` computes, once per parameter proposal: the log galaxy
measure grid, per-galaxy kernel weights/widths (catalog.py), and the
per-row completion curves dN_miss(zgrid), N_miss (completion.py) —
O(N_rows × N_grid).  ``eval`` then costs O(N_max_gals) per sample: one
row gather, one Gaussian logpdf per galaxy, two 1-D interpolations.
The likelihood builds the state once per proposal and captures it in
its per-sample closures, so neither the per-event ``lax.scan`` nor the
selection batching recomputes it.

The historical one-shot registry functions (``PRIOR_REGISTRY``) remain
for checks, diagnostics, and tests; they build the state internally on
every call.
"""

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from typing import NamedTuple, Any

from darksirens.redshift.volume import log_volume_prior_vmap, _precompute_volume_grid
from darksirens.redshift.catalog import (
    catalog_kernel_state,
    marked_catalog_kernel_state,
    eval_log_catalog_prior_state,
    CatalogKernelState,
)
from darksirens.redshift.completion import (
    completion_curves,
    field_global_log_Z,
    log_galaxy_measure_grid,
)

from darksirens.redshift.grid import zgrid


COMPLETE_EMPTY_PIXEL_POLICY_ZERO = 0
COMPLETE_EMPTY_PIXEL_POLICY_VOLUME = 1


def _materialize(state):
    """Force XLA to materialize a prior state instead of fusing its
    construction into per-sample consumers.

    Without this barrier, gathering ``state.dN_miss[pix, idx]`` per sample
    invites XLA to recompute the producing (N_rows × N_grid) curves inside
    the sample loop — measured ~10x slowdown of the one-shot prior at 1e5
    samples.  The barrier is a value identity.
    """
    return lax.optimization_barrier(state)


def _maybe_materialize(state, enabled: bool = True):
    """Return ``state`` with the optimization barrier only when requested."""
    return _materialize(state) if enabled else state


# ------------------------------------------------------------
# Prior states (per-proposal precomputations)
# ------------------------------------------------------------

class SpectralPriorState(NamedTuple):
    log_pvol: Any  # (N_grid,) log normalised comoving-volume prior


class CompletePriorState(NamedTuple):
    kernels: Any   # CatalogKernelState
    row_has: Any   # (N_rows,) bool — row contains at least one real galaxy
    log_pvol: Any  # (N_grid,) volume fallback for the ``volume`` policy


class DarkSirenPriorState(NamedTuple):
    kernels: Any   # CatalogKernelState
    log_Nobs: Any  # (N_rows,) log real-galaxy count (-inf for empty rows)
    dN_miss: Any   # (N_rows, N_grid) missing-galaxy density
    log_Z: Any     # (N_rows,) log[N_obs + N_miss]  (per-pixel "conditional" normalizer)
    # FIELD-convention sky weighting: survey-GLOBAL log normalizer
    # log Sum_all-pixels [N_obs + N_miss].  Populated (as a JAX scalar) only when
    # ``catalog_sky_weighting == "field"``; kept at the constant ``0.0`` and never
    # read in the default "conditional" mode, so the treedef is stable and the
    # conditional evaluator is bit-identical to the pre-existing code.
    log_Z_global: Any = 0.0


class DarkSirenEnsemblePriorState(NamedTuple):
    """Dark-siren state with a fixed LSS-completion ensemble (Q_LSS^(m)).

    ``dN_miss`` / ``log_Z`` are the **posterior-mean (or deterministic)**
    scalar-compatibility fields — identical to what a plain
    :class:`DarkSirenPriorState` would carry — so ``eval_redshift_prior_with_state``
    (and hence the GW likelihood) behaves exactly as the mean prior and is
    unaffected by the ensemble.  The member fields drive only the Bayesian
    redshift-prior **diagnostic** (``eval_redshift_prior_members_with_state``).
    """
    kernels: Any           # CatalogKernelState
    log_Nobs: Any          # (N_rows,) log real-galaxy count (-inf for empty rows)
    dN_miss: Any           # (N_rows, N_grid) scalar-compat missing-galaxy density
    log_Z: Any             # (N_rows,) scalar-compat log[N_obs + N_miss]
    dN_miss_members: Any   # (M, N_rows, N_grid) per-member missing-galaxy density
    log_Z_members: Any     # (M, N_rows) per-member log[N_obs + N_miss^m]


def _row_counts(em_catalog: EMCatalog) -> jnp.ndarray:
    if em_catalog.ngals is not None:
        return jnp.asarray(em_catalog.ngals, dtype=jnp.float64)
    return jnp.sum(em_catalog.wgals > 0.0, axis=-1).astype(jnp.float64)


# Symmetric clip on log h(m|eta) so wide eta cannot blow up the host efficiency
# (mirrors the lognormal-completion log-Q clip).
_LOG_H_CLIP: float = 7.0
#: coarse z-bins for the empirical missing-galaxy efficiency mu_miss(z|eta).
_MU_MISS_NBINS: int = 40


def _mu_miss_grid(em_catalog: EMCatalog, log_h: jnp.ndarray) -> jnp.ndarray:
    """Level-B missing-galaxy host efficiency ``mu_miss(z|eta) = E_obs[h | z]``.

    The deterministic estimator of the *expected* host efficiency of the
    unobserved galaxies along the line of sight: the z-binned mean of
    ``h = exp(log_h)`` over the catalog's **observed** galaxies, interpolated to
    ``zgrid``.  Empty/out-of-range z-bins default to 1 (homogeneous), so the
    missing branch is only modulated where the catalog carries mark information.
    No galaxies are invented — this reuses the observed marks (consistent with
    the deterministic-likelihood principle of the LSS completion).
    """
    zs = jnp.asarray(em_catalog.zgals).reshape(-1)
    h = jnp.exp(jnp.asarray(log_h)).reshape(-1)
    if em_catalog.ngals is not None:
        n_max = em_catalog.zgals.shape[1]
        real = (jnp.arange(n_max)[None, :] < jnp.asarray(em_catalog.ngals)[:, None]).reshape(-1)
    else:
        real = (jnp.asarray(em_catalog.wgals) > 0.0).reshape(-1)
    real = real.astype(h.dtype)

    # Keep ``z_hi`` as a JAX scalar (do NOT ``float()`` it): ``_mu_miss_grid`` runs
    # inside the jitted likelihood, where the module ``zgrid`` constant is captured
    # as a tracer, so a concrete ``float()`` raises ConcretizationTypeError.
    # ``jnp.linspace`` accepts an array ``stop`` because ``num`` is static.
    z_hi = zgrid[-1]
    edges = jnp.linspace(0.0, z_hi, _MU_MISS_NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = jnp.clip(jnp.searchsorted(edges, zs, side="right") - 1, 0, _MU_MISS_NBINS - 1)
    sum_h = jnp.zeros(_MU_MISS_NBINS).at[b].add(h * real)
    cnt = jnp.zeros(_MU_MISS_NBINS).at[b].add(real)
    mu_bin = jnp.where(cnt > 0.0, sum_h / jnp.where(cnt > 0.0, cnt, 1.0), 1.0)
    return jnp.maximum(jnp.interp(zgrid, centers, mu_bin), 0.0)


def prepare_redshift_prior_state(
    model: str,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    mark_model: str = "none",
    mark_params=None,
    mark_names=(),
    *,
    materialize_state: bool = True,
    catalog_sky_weighting: str = "conditional",
):
    """Build the per-proposal state for ``model``.  O(N_rows × N_grid).

    ``mark_model`` (+ sampled ``mark_params`` ``eta`` and the resolved
    ``mark_names``) optionally activates the marked-host model for
    ``dark_sirens`` (:mod:`darksirens.marks`): catalog galaxies are reweighted by
    a BBH-host efficiency ``h(m|eta)``.  ``mark_model="none"`` (default) is the
    legacy galaxy-count host model, bit-for-bit.

    ``catalog_sky_weighting`` selects the ``dark_sirens`` redshift-prior
    normalization convention (static; only consulted for ``model ==
    "dark_sirens"``): ``"conditional"`` (default) normalizes each pixel by its
    own ``Z[pix] = N_obs + N_miss`` (bit-identical legacy behaviour);
    ``"field"`` instead stores a survey-GLOBAL ``log_Z_global`` so the K-catalog
    mixture weight measures the host FRACTION (number-density / sky-clustering
    contrast), the estimand fix motivated by the gws-agn campaign.  ``"field"``
    is gated to the plain galaxy-count host model with the legacy dummy
    overdensity (no marks, no Q_LSS ensemble/table).

    NOTE (K=1): the global constant ``log_Z_global`` cancels between the PE
    and selection terms of a single-catalog likelihood, so at K=1 field mode's
    only effect is removing the per-pixel ``Z[pix]`` normalization; the global
    normalizer bites only for K>=2 mixtures, where each catalog's ``Z_k``
    enters its mixture branch non-cancelling. Do not expect K=1 field runs to
    constrain n0 through ``log_Z_global``.
    """
    if model == "spectral_sirens":
        # NaN-guarded log: the volume grid is EXACTLY zero at z = 0, and a
        # naked log makes d log_pvol[0]/dH0 = (1/0) * 0 = NaN in reverse mode.
        # Any sample interpolating into the first z-cell (low-z injections!)
        # then poisons d logL/dH0 wholesale -- this is what broke dark-siren
        # NumPyro NUTS. log(tiny) instead of -inf is numerically identical
        # downstream (exp underflows to exactly zero weight).
        vol = _precompute_volume_grid(cosmo)
        state = SpectralPriorState(
            log_pvol=jnp.log(jnp.maximum(vol, jnp.finfo(vol.dtype).tiny)))
        return _maybe_materialize(state, materialize_state)

    if model == "bright_sirens":
        return None  # per-event counterpart logic needs the live catalog

    if model == "dark_sirens_complete":
        kernels = catalog_kernel_state(cosmo, survey, em_catalog, volume_weighted=True)
        row_has = _row_counts(em_catalog) > 0.0
        vol = _precompute_volume_grid(cosmo)
        log_pvol = jnp.log(jnp.maximum(vol, jnp.finfo(vol.dtype).tiny))
        state = CompletePriorState(kernels=kernels, row_has=row_has, log_pvol=log_pvol)
        return _maybe_materialize(state, materialize_state)

    if model == "dark_sirens":
        is_field = catalog_sky_weighting == "field"
        if catalog_sky_weighting not in ("conditional", "field"):
            raise ValueError(
                "catalog_sky_weighting must be 'conditional' or 'field', got "
                f"{catalog_sky_weighting!r}."
            )
        if is_field:
            # Field convention is the plain galaxy-count host model with the
            # legacy dummy overdensity: no marks, no Q_LSS table/ensemble.  These
            # are static (structure) checks, so they resolve once per trace.
            if mark_model is not None and mark_model != "none":
                raise NotImplementedError(
                    "catalog_sky_weighting='field' is not supported with a marked-host "
                    "model (mark_model); use 'conditional'."
                )
            if any(
                getattr(em_catalog, name) is not None
                for name in (
                    "lss_completion_logq",
                    "lss_completion_q",
                    "lss_completion_logq_members",
                    "lss_completion_q_members",
                )
            ):
                raise NotImplementedError(
                    "catalog_sky_weighting='field' is not supported with an "
                    "LSS-completion Q_LSS table/ensemble; use 'conditional'."
                )
            if em_catalog.field_dN_obs_s is None:
                raise ValueError(
                    "catalog_sky_weighting='field' requires the survey-global field "
                    "normalization inputs (field_dN_obs_s / field_n_empty / "
                    "field_N_obs_total). Build them via "
                    "darksirens.redshift.completion.build_field_normalization_inputs "
                    "from the FULL-sky catalog (the union/full-catalog path)."
                )

        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
        curves = completion_curves(cosmo, survey, em_catalog)

        if mark_model is not None and mark_model != "none":
            # Marked-host model: galaxies reweighted by h(m|eta).  Built entirely
            # here (the per-sample evaluator and DarkSirenPriorState are reused),
            # because dN_host_obs = N_host_obs * p_host(z) with p_host normalised.
            # Composes with Q_LSS: curves.dN_miss already carries any deterministic
            # (or posterior-mean) Q_LSS.  The Level-B missing branch multiplies it
            # by mu_miss(z|eta) = E_obs[h|z].
            from darksirens.marks import mark_model_parser
            log_h = jnp.clip(
                mark_model_parser(mark_model, mark_names)(em_catalog, mark_params),
                -_LOG_H_CLIP, _LOG_H_CLIP,
            )
            kernels, log_N_host = marked_catalog_kernel_state(
                cosmo, survey, em_catalog, log_h, log_g_grid=log_g_grid
            )
            mu_miss = _mu_miss_grid(em_catalog, log_h)                  # (N_grid,)
            dN_miss = curves.dN_miss * mu_miss[None, :]                 # (N_rows, N_grid)
            N_host_miss = jnp.trapezoid(dN_miss, zgrid, axis=-1)        # (N_rows,)
            N_host_obs = jnp.where(jnp.isfinite(log_N_host), jnp.exp(log_N_host), 0.0)
            Z = N_host_obs + N_host_miss
            log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1e-300)), 0.0)
            state = DarkSirenPriorState(
                kernels=kernels, log_Nobs=log_N_host, dN_miss=dN_miss, log_Z=log_Z
            )
            return _maybe_materialize(state, materialize_state)

        kernels = catalog_kernel_state(cosmo, survey, em_catalog, log_g_grid=log_g_grid)
        Nobs = _row_counts(em_catalog)
        log_Nobs = jnp.where(Nobs > 0.0, jnp.log(jnp.maximum(Nobs, 1e-300)), -jnp.inf)
        # Scalar-compatibility normalisation: curves.dN_miss / N_miss already
        # carry the deterministic Q (or the posterior-mean Q when only an
        # ensemble was supplied), so this matches the legacy behaviour exactly.
        Z = Nobs + curves.N_miss
        log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1e-300)), 0.0)
        if is_field:
            # Survey-GLOBAL normalizer log Sum_all-pixels[N_obs + N_miss].  The
            # per-pixel numerator (log_Nobs, dN_miss) is UNCHANGED; only the
            # denominator becomes global, turning the mixture weight into a host
            # fraction.  Field mode is gated (above) to the no-ensemble path.
            log_Z_global = field_global_log_Z(cosmo, survey, em_catalog)
            state = DarkSirenPriorState(
                kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss,
                log_Z=log_Z, log_Z_global=log_Z_global,
            )
            return _maybe_materialize(state, materialize_state)
        if curves.dN_miss_members is None:
            state = DarkSirenPriorState(
                kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss, log_Z=log_Z
            )
            return _maybe_materialize(state, materialize_state)
        # Fixed LSS-completion ensemble present -> add per-member fields for the
        # Bayesian redshift-prior diagnostic (the scalar fields above are unchanged).
        Z_members = Nobs[None, :] + curves.N_miss_members          # (M, N_rows)
        log_Z_members = jnp.where(
            Z_members > 0.0, jnp.log(jnp.maximum(Z_members, 1e-300)), 0.0
        )
        state = DarkSirenEnsemblePriorState(
            kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss, log_Z=log_Z,
            dN_miss_members=curves.dN_miss_members, log_Z_members=log_Z_members,
        )
        return _maybe_materialize(state, materialize_state)

    raise ValueError(f"Unknown redshift prior model '{model}'.")


# ------------------------------------------------------------
# Per-sample evaluators
# ------------------------------------------------------------

def _interp_row(table_row_lo, table_row_hi, t):
    """Linear interpolation given precomputed bracketing values and weight."""
    return table_row_lo + t * (table_row_hi - table_row_lo)


def _grid_bracket(z):
    """Index and weight bracketing ``z`` on ``zgrid`` (endpoint-clamped).

    NaN z propagates to a NaN weight, which downstream positivity checks
    turn into -inf (never probability 1).
    """
    idx = jnp.clip(jnp.searchsorted(zgrid, z, side="right") - 1, 0, zgrid.size - 2)
    t = (z - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx])
    return idx, jnp.clip(t, 0.0, 1.0)


def _eval_dark_scalar(
    z, pix, state: DarkSirenPriorState, em_catalog: EMCatalog,
    catalog_sky_weighting: str = "conditional",
):
    log_p_cat = eval_log_catalog_prior_state(z, pix, state.kernels, em_catalog)
    # NaN (out-of-grid z, degenerate kernels) must mean "impossible", never p=1.
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)

    # Two-element gather instead of jnp.interp(z, zgrid, dN_miss[pix]):
    # gathering the full (N_grid,) row per sample costs ~8 kB of memory
    # traffic per sample and dominates the evaluator on CPU/GPU alike.
    idx, t = _grid_bracket(z)
    miss = _interp_row(state.dN_miss[pix, idx], state.dN_miss[pix, idx + 1], t)
    log_miss = jnp.where(miss > 0.0, jnp.log(jnp.maximum(miss, 1e-300)), -jnp.inf)

    numerator = jnp.logaddexp(state.log_Nobs[pix] + log_p_cat, log_miss)
    # FIELD convention: normalize the per-pixel numerator by the survey-GLOBAL
    # Z (static branch on the mode string) instead of the per-pixel Z[pix].
    if catalog_sky_weighting == "field":
        return numerator - state.log_Z_global
    return numerator - state.log_Z[pix]


def _eval_complete_scalar(
    z, pix, state: CompletePriorState, survey: SurveyParams, em_catalog: EMCatalog
):
    log_p_cat = eval_log_catalog_prior_state(z, pix, state.kernels, em_catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    log_p_vol = jnp.interp(z, zgrid, state.log_pvol)
    empty_value = jnp.where(
        survey.complete_empty_pixel_policy == COMPLETE_EMPTY_PIXEL_POLICY_VOLUME,
        log_p_vol,
        -jnp.inf,
    )
    return jnp.where(state.row_has[pix], log_p_cat, empty_value)


def eval_redshift_prior_with_state(
    model: str,
    state,
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    catalog_sky_weighting: str = "conditional",
) -> jnp.ndarray:
    """Vectorised log p(z | pix) using a prepared state.  O(N_max) / sample.

    ``catalog_sky_weighting`` ("conditional" default) selects the ``dark_sirens``
    normalization convention; it is a static string branch and is inert for every
    other ``model``.
    """
    if model == "spectral_sirens":
        return vmap(lambda z_i: jnp.interp(z_i, zgrid, state.log_pvol))(z)

    if model == "bright_sirens":
        return _log_prior_bright_sirens(z, pix, cosmo, survey, em_catalog)

    if model == "dark_sirens_complete":
        return vmap(
            lambda z_i, p_i: _eval_complete_scalar(z_i, p_i, state, survey, em_catalog)
        )(z, pix)

    if model == "dark_sirens":
        return vmap(
            lambda z_i, p_i: _eval_dark_scalar(
                z_i, p_i, state, em_catalog, catalog_sky_weighting
            )
        )(z, pix)

    raise ValueError(f"Unknown redshift prior model '{model}'.")


def _eval_dark_member_scalar(z, pix, m, state: "DarkSirenEnsemblePriorState", em_catalog):
    """log p_m(z | pix) for LSS-completion ensemble member ``m`` (diagnostic).

        p_m(z|pix) = [N_obs(pix) p_cat(z|pix) + dN_miss^m(z|pix)]
                     / [N_obs(pix) + N_miss^m(pix)].
    """
    log_p_cat = eval_log_catalog_prior_state(z, pix, state.kernels, em_catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    idx, t = _grid_bracket(z)
    miss = _interp_row(
        state.dN_miss_members[m, pix, idx],
        state.dN_miss_members[m, pix, idx + 1],
        t,
    )
    log_miss = jnp.where(miss > 0.0, jnp.log(jnp.maximum(miss, 1e-300)), -jnp.inf)
    return jnp.logaddexp(state.log_Nobs[pix] + log_p_cat, log_miss) - state.log_Z_members[m, pix]


def eval_redshift_prior_members_with_state(
    model: str,
    state,
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """Per-member log p_m(z | pix) for a fixed LSS-completion ensemble.

    Returns shape ``(M, len(z))``.  This is a **diagnostic** path (the Bayesian
    marginalised prior is ``p_Bayes(z|pix) = mean_m exp(log p_m)`` with each
    member normalised individually) and is never used inside the GW likelihood.
    For non-ensemble states (or other models) it returns shape ``(1, len(z))``
    using the scalar (posterior-mean) prior, so callers can always index a
    leading member axis.
    """
    z = jnp.asarray(z)
    pix = jnp.asarray(pix)
    if model == "dark_sirens" and isinstance(state, DarkSirenEnsemblePriorState):
        M = int(state.dN_miss_members.shape[0])

        def _per_member(m):
            return vmap(
                lambda z_i, p_i: _eval_dark_member_scalar(z_i, p_i, m, state, em_catalog)
            )(z, pix)

        return vmap(_per_member)(jnp.arange(M, dtype=jnp.int32))  # (M, len(z))

    lp = eval_redshift_prior_with_state(model, state, z, pix, cosmo, survey, em_catalog)
    return jnp.reshape(lp, (1, -1))


# ------------------------------------------------------------
# One-shot prior implementations (registry; checks/tests/back-compat)
# ------------------------------------------------------------

@jit
def _log_prior_spectral_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """GW-only prior: normalised comoving volume element."""
    return log_volume_prior_vmap(z, cosmo, survey)


@jit
def _log_prior_complete_catalog(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior under the complete-catalog assumption:
    p(z | pix) = p_cat(z | pix), with the empty-pixel policy from
    ``survey.complete_empty_pixel_policy`` (0 = zero/-inf, 1 = volume
    fallback).  Empty pixels are identified from real-galaxy counts, not
    from numerical underflow of p_cat.
    """
    state = prepare_redshift_prior_state("dark_sirens_complete", cosmo, survey, em_catalog)
    return eval_redshift_prior_with_state(
        "dark_sirens_complete", state, z, pix, cosmo, survey, em_catalog
    )


@jit
def _log_prior_bright_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Bright-siren counterpart redshift likelihood with an optional sky gate.

    Multi-event bright-siren analyses store one counterpart redshift and one
    counterpart sky pixel per GW event; the likelihood sets
    ``em_catalog.active_counterpart_index`` before evaluating each event.
    For backward compatibility, if the per-event counterpart arrays are
    absent this falls back to the historical one-object synthetic catalog
    prior.
    """
    from jax.scipy.stats import norm
    from darksirens.redshift.catalog import log_catalog_prior_vmap  # local import avoids circular

    if em_catalog.counterpart_zs is not None:
        idx = jnp.asarray(em_catalog.active_counterpart_index, dtype=jnp.int32)
        counterpart_z = jnp.take(em_catalog.counterpart_zs, idx)
        counterpart_dz = jnp.take(em_catalog.counterpart_dzs, idx)
        counterpart_pixel = jnp.take(em_catalog.counterpart_pixels, idx)

        if em_catalog.unique_pixels is None:
            global_pix = pix
        else:
            global_pix = jnp.take(em_catalog.unique_pixels, pix)

        sky_marginalized = jnp.asarray(em_catalog.bright_siren_sky_marginalized)
        in_counterpart_pixel = global_pix == counterpart_pixel
        log_p_cp = norm.logpdf(z, counterpart_z, counterpart_dz)
        return jnp.where(sky_marginalized | in_counterpart_pixel, log_p_cp, -jnp.inf)

    counterpart_pixel = em_catalog.counterpart_pixel
    if counterpart_pixel is None:
        if em_catalog.ngals is None:
            counterpart_pixel = 0
        elif em_catalog.unique_pixels is None:
            counterpart_pixel = jnp.argmax(em_catalog.ngals > 0)
        else:
            counterpart_pixel = em_catalog.unique_pixels[jnp.argmax(em_catalog.ngals > 0)]

    if em_catalog.unique_pixels is None:
        global_pix = pix
        counterpart_row = counterpart_pixel
    else:
        global_pix = jnp.take(em_catalog.unique_pixels, pix)
        counterpart_row = jnp.argmax(em_catalog.unique_pixels == counterpart_pixel)

    sky_marginalized = jnp.asarray(em_catalog.bright_siren_sky_marginalized)
    prior_pix = jnp.where(sky_marginalized, counterpart_row, pix)
    log_p_cp = log_catalog_prior_vmap(z, prior_pix, cosmo, survey, em_catalog)

    in_counterpart_pixel = global_pix == counterpart_pixel
    return jnp.where(sky_marginalized | in_counterpart_pixel, log_p_cp, -jnp.inf)


@jit
def _log_prior_dark_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """
    Dark-siren prior with catalog completion (the general case):

        p(z|pix) = [N_obs * p_cat(z|pix) + dN_miss(z|pix)] / (N_obs + N_miss).

    Exactly normalised per pixel.  One-shot signature: builds the
    per-proposal state internally; the likelihood uses the state API
    directly to avoid rebuilding it per event / per selection batch.
    """
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, em_catalog)
    return eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, em_catalog
    )


# ------------------------------------------------------------
# Registry and factory
# ------------------------------------------------------------

#: Maps model name → compiled prior function.
#: Signature: f(z, pix, cosmo, survey, em_catalog) → log_prior (array).
PRIOR_REGISTRY: dict = {
    "spectral_sirens":      _log_prior_spectral_sirens,
    # Weak-lensing universe model reuses the spectral-sirens redshift prior; the
    # WL magnification marginalization is applied in the likelihood, not here.
    "spectral_sirens_wl":   _log_prior_spectral_sirens,
    "bright_sirens":        _log_prior_bright_sirens,
    "dark_sirens_complete": _log_prior_complete_catalog,
    "dark_sirens":          _log_prior_dark_sirens,
}


def get_redshift_prior(model: str):
    """Return the compiled one-shot log-prior function for ``model``."""
    if model not in PRIOR_REGISTRY:
        available = ", ".join(f'"{k}"' for k in PRIOR_REGISTRY)
        raise ValueError(
            f"Unknown redshift prior model '{model}'. Available: {available}."
        )
    return PRIOR_REGISTRY[model]