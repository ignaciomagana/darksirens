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
from jax import lax, vmap

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import threads_distance_table
from typing import NamedTuple, Any

from darksirens.redshift.volume import log_volume_prior_vmap, _precompute_volume_grid
from darksirens.redshift.catalog import (
    catalog_kernel_state,
    marked_catalog_kernel_state,
    eval_log_catalog_prior_state,
)
from darksirens.redshift.completion import (
    completion_curves,
    field_global_log_Z,
    field_global_log_Z_marked,
    field_global_log_Z_marked_members,
    field_global_log_Z_members,
    log_galaxy_measure_grid,
    member_N_miss_integrals,
    _member_q_eff_from_logq,
    _resolve_member_logq_row,
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
    # FIELD-convention sky weighting (complete model): per-row log N_obs and the
    # survey-GLOBAL log Sum_pix N_obs.  Unlike dark_sirens, the complete-model
    # field normalizer is theta-INDEPENDENT (Z = Sum_pix N_obs, no missing-galaxy
    # budget), so an empty pixel carries zero count weight -> -inf REGARDLESS of
    # ``complete_empty_pixel_policy``.  Populated ALWAYS (both are cheap) and read
    # only under ``catalog_sky_weighting == "field"``, so the conditional
    # evaluator is bit-identical to the pre-existing code and the treedef is
    # stable (no per-mode leaf structure change).
    log_Nobs: Any = -jnp.inf         # (N_rows,) log real-galaxy count (-inf for empty rows)
    log_N_obs_total: Any = 0.0       # scalar log Sum_all-pixels N_obs


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
    unaffected by the ensemble.  The member fields drive the Bayesian
    redshift-prior **diagnostic** (``eval_redshift_prior_members_with_state``) and
    the ``lss_marginalize`` member vmap.

    ``base_miss`` is the member-INDEPENDENT missing-density base ``(N_rows,
    N_grid)`` (``(1 - C) dN_exp`` with the ``z_depth`` relaxation baked in, times
    ``mu_miss`` in the marked model); member ``m``'s density is reconstructed on
    the fly at query redshifts as ``base_miss * Q_eff_m`` -- so the state stores
    ``O(N_rows x N_grid)`` here, NOT the ``O(M x N_rows x N_grid)`` cube.  The
    per-member log-Q lives on the catalog (a resident data constant, gathered two
    grid nodes at a time by ``eval_dark_member_completion``), never in this state.
    """
    kernels: Any           # CatalogKernelState
    log_Nobs: Any          # (N_rows,) log real-galaxy count (-inf for empty rows)
    dN_miss: Any           # (N_rows, N_grid) scalar-compat missing-galaxy density
    log_Z: Any             # (N_rows,) scalar-compat log[N_obs + N_miss]
    base_miss: Any         # (N_rows, N_grid) member-independent missing-density base
    log_Z_members: Any     # (M, N_rows) per-member log[N_obs + N_miss^m]
    # FIELD-convention survey-global normalizers (None under conditional):
    # the scalar-compat (posterior-mean-Q) global Z and the per-member ones.
    log_Z_global: Any = None          # scalar log Z(theta) with the mean Q
    log_Z_global_members: Any = None  # (M,) per-member log Z_m(theta)


def _row_counts(em_catalog: EMCatalog) -> jnp.ndarray:
    if em_catalog.ngals is not None:
        return jnp.asarray(em_catalog.ngals, dtype=jnp.float64)
    return jnp.sum(em_catalog.wgals > 0.0, axis=-1).astype(jnp.float64)


# Symmetric clip on log h(m|eta) so wide eta cannot blow up the host efficiency
# (mirrors the lognormal-completion log-Q clip).
_LOG_H_CLIP: float = 7.0
#: coarse z-bins for the empirical missing-galaxy efficiency mu_miss(z|eta).
_MU_MISS_NBINS: int = 40


def _mu_miss_from_flat(zs: jnp.ndarray, h: jnp.ndarray, real: jnp.ndarray) -> jnp.ndarray:
    """Shared core of the mu_miss estimator on FLAT per-galaxy arrays.

    ``real`` is a {0,1} weight selecting genuine galaxies (1 everywhere for
    the field-normalizer flat full-sky inputs, which are pre-masked).
    """
    # Keep ``z_hi`` as a JAX scalar (do NOT ``float()`` it): this runs inside
    # the jitted likelihood, where the module ``zgrid`` constant is captured
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
    return _mu_miss_from_flat(zs, h, real)


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
    "dark_sirens"``).  ``"field"`` is the JOINT catalog host-density estimand
    and the CLI default (all K): it stores a survey-GLOBAL ``log_Z_global`` and
    normalizes the per-pixel numerator by it, so RELATIVE angular host density
    is preserved -- a pixel with 100 candidate hosts carries ~100x the angular
    weight of a pixel with 1 -- and for a K-catalog mixture the weight measures
    the host FRACTION (number-density / sky-clustering contrast).  ``"field"``
    is gated to the plain galaxy-count host model with the legacy dummy
    overdensity (no marks, no Q_LSS ensemble/table).  ``"conditional"`` is the
    radial-only LEGACY estimand: it normalizes each pixel by its OWN ``Z[pix] =
    N_obs + N_miss``, so every pixel integrates to unit mass and relative
    angular host density is DISCARDED (bit-identical legacy behaviour; kept for
    reproducing older single-catalog runs).  It is also this builder's SIGNATURE
    default -- the safe fallback for programmatic callers that do not supply the
    field-normalization inputs; the CLI resolves the user-facing default to
    ``"field"`` (see ``_resolve_catalog_sky_weighting``).

    NOTE (K=1): the survey-global constant ``log_Z_global`` cancels between the
    PE and selection terms of a single-catalog likelihood, so at K=1 the
    NET difference field makes versus conditional is removing the per-pixel
    ``Z[pix]`` normalization -- i.e. restoring the relative angular host
    weighting.  Its dedicated number-density channel (log10n0 entering through
    ``log_Z_global``) therefore vanishes at K=1; log10n0 keeps only the weak
    identification it has in both modes, through the completeness ratio
    ``C(z; n0)`` and the missing-branch amplitude ``dN_miss`` in the per-pixel
    numerator, and otherwise marginalizes against its prior.  The global
    normalizer bites for log10n0 only at K>=2, where each catalog's ``Z_k``
    enters its mixture branch non-cancelling.
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
        if catalog_sky_weighting not in ("conditional", "field"):
            raise ValueError(
                "catalog_sky_weighting must be 'conditional' or 'field', got "
                f"{catalog_sky_weighting!r}."
            )
        # Unit-mass kernels (volume_weighted=False): the complete-catalog host
        # prior is the catalog's own (weighted) redshift distribution.  The
        # catalog COUNTS already track candidate hosts per redshift shell, so
        # additionally weighting each galaxy by g(z_i) = dV_c/dz (1+z)^delta
        # (volume_weighted=True, d58af14) double-counts the volume element:
        # any catalog with dN/dz ∝ dV_c/dz (volume-limited, or a uniform-in-
        # comoving-volume null) gets an effective host prior ∝ (dV_c/dz)^2.
        # Measured on a uniform-in-V_c null catalog, p_cat/(dV/dz) has
        # log-log slope +0.94 with volume weighting and 0.00 without (see
        # tests/test_complete_catalog_volume_convention.py).
        kernels = catalog_kernel_state(cosmo, survey, em_catalog, volume_weighted=False)
        Nobs = _row_counts(em_catalog)
        row_has = Nobs > 0.0
        vol = _precompute_volume_grid(cosmo)
        log_pvol = jnp.log(jnp.maximum(vol, jnp.finfo(vol.dtype).tiny))
        # FIELD convention (complete model, field mode only): the prior is a
        # survey FIELD density weighting each occupied pixel by its share
        # N_obs[pix] / N_obs_total of the total observed count.  The normalizer
        # Z = Sum_all-pixels N_obs is theta-independent, so log_N_obs_total must be
        # the FULL-sky total (``em_catalog.field_N_obs_total``, built by PR-3's
        # build_field_normalization_inputs), not the compacted-row sum.  Both
        # leaves are populated ALWAYS so the treedef is mode-independent; in the
        # conditional path neither is read (bit-identical legacy behaviour).
        log_Nobs = jnp.where(Nobs > 0.0, jnp.log(jnp.maximum(Nobs, 1e-300)), -jnp.inf)
        if catalog_sky_weighting == "field":
            if em_catalog.field_N_obs_total is None:
                raise ValueError(
                    "catalog_sky_weighting='field' requires the survey-global "
                    "field_N_obs_total on the (full-sky) complete-model catalog; "
                    "build it via "
                    "darksirens.redshift.completion.build_field_normalization_inputs."
                )
            log_N_obs_total = jnp.log(
                jnp.maximum(
                    jnp.asarray(em_catalog.field_N_obs_total, dtype=Nobs.dtype), 1e-300
                )
            )
        else:
            # Never read in conditional mode; a finite in-catalog total keeps the
            # scalar leaf well-defined without requiring the field inputs.
            log_N_obs_total = jnp.log(jnp.maximum(jnp.sum(Nobs), 1e-300))
        state = CompletePriorState(
            kernels=kernels, row_has=row_has, log_pvol=log_pvol,
            log_Nobs=log_Nobs, log_N_obs_total=log_N_obs_total,
        )
        return _maybe_materialize(state, materialize_state)

    if model == "dark_sirens":
        is_field = catalog_sky_weighting == "field"
        if catalog_sky_weighting not in ("conditional", "field"):
            raise ValueError(
                "catalog_sky_weighting must be 'conditional' or 'field', got "
                f"{catalog_sky_weighting!r}."
            )
        if is_field:
            # Field-convention scope: static (pytree-structure) checks, resolved
            # once per trace.  The missing-galaxy budget modulations (Q_LSS
            # table, per-pixel delta_g) and the marked-host model ARE supported
            # provided the matching survey-global field_* inputs are present,
            # so the numerator and the global normalizer carry the SAME budget
            # (field_global_log_Z / field_global_log_Z_marked).
            if (mark_model is not None and mark_model != "none"
                    and (em_catalog.field_mark_values is None
                         or em_catalog.field_mark_z is None
                         or em_catalog.field_mark_w is None)):
                raise ValueError(
                    "catalog_sky_weighting='field' with a marked-host model "
                    "requires the flat FULL-SKY mark inputs (field_mark_z / "
                    "field_mark_w / field_mark_values) built via "
                    "darksirens.redshift.completion.build_field_mark_inputs -- "
                    "mu_miss and the observed marked mass must be "
                    "view-independent so PE and selection share one global Z."
                )
            has_q_members = any(
                getattr(em_catalog, name) is not None
                for name in (
                    "lss_completion_logq_members",
                    "lss_completion_q_members",
                )
            )
            if has_q_members and em_catalog.field_lss_q_members is None:
                raise ValueError(
                    "catalog_sky_weighting='field' with a Q_LSS ENSEMBLE "
                    "requires the per-member survey-global Q rows "
                    "(field_lss_q_members / field_lss_q_empty_sum_members) "
                    "built from the GLOBAL ensemble via "
                    "darksirens.redshift.completion.build_field_lss_q_member_inputs."
                )
            has_q_table = any(
                getattr(em_catalog, name) is not None
                for name in ("lss_completion_logq", "lss_completion_q")
            )
            if has_q_table and em_catalog.field_lss_q is None:
                raise ValueError(
                    "catalog_sky_weighting='field' with a deterministic Q_LSS "
                    "table requires the survey-global Q rows "
                    "(field_lss_q / field_lss_q_empty_sum) built from the "
                    "GLOBAL table via "
                    "darksirens.redshift.completion.build_field_lss_q_inputs -- "
                    "otherwise the global normalizer would carry a different "
                    "missing-galaxy budget than the numerator."
                )
            has_real_delta_g = (
                em_catalog.delta_g_pix_z is not None
                and em_catalog.delta_g_pix_z.shape[0] != 1
            )
            if has_real_delta_g and em_catalog.field_delta_g is None:
                raise ValueError(
                    "catalog_sky_weighting='field' with a per-pixel delta_g "
                    "overdensity requires the survey-global delta_g rows "
                    "(field_delta_g) built via "
                    "darksirens.redshift.completion.build_field_delta_g_inputs."
                )
            if em_catalog.field_dN_obs_s is None:
                raise ValueError(
                    "catalog_sky_weighting='field' requires the survey-global field "
                    "normalization inputs (field_dN_obs_s / field_n_empty / "
                    "field_N_obs_total). Build them via "
                    "darksirens.redshift.completion.build_field_normalization_inputs "
                    "from the FULL-sky catalog (the union/full-catalog path)."
                )
            if survey.z_depth is not None and (
                em_catalog.field_depth_z is None
                or em_catalog.field_depth_dz is None
                or em_catalog.field_depth_c is None
            ):
                raise ValueError(
                    "catalog_sky_weighting='field' with a survey depth "
                    f"(z_depth={survey.z_depth!r}) requires the flat FULL-SKY "
                    "depth inputs (field_depth_z / field_depth_dz / "
                    "field_depth_c) built via "
                    "darksirens.redshift.completion.build_field_depth_inputs -- "
                    "the global normalizer's OBSERVED term must carry the same "
                    "depth scaling as the numerator (Sum_pix N_obs,pix * m_pix), "
                    "or every above-depth catalogued galaxy is counted twice."
                )

        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
        curves = completion_curves(cosmo, survey, em_catalog)

        if mark_model is not None and mark_model != "none":
            # Marked-host model: galaxies reweighted by h(m|eta).  Built entirely
            # here (the per-sample evaluator and the prior states are reused),
            # because dN_host_obs = N_host_obs * p_host(z) with p_host normalised.
            # Composes with Q_LSS: curves.dN_miss already carries any deterministic
            # (or posterior-mean) Q_LSS.  The Level-B missing branch multiplies it
            # by mu_miss(z|eta) = E_obs[h|z].  With a Q_LSS ENSEMBLE the scalar
            # fields stay the posterior-mean composition and per-member marked
            # missing densities are added (a DarkSirenEnsemblePriorState) for the
            # lss_marginalize member vmap, exactly mirroring the unmarked ensemble
            # branch below.
            from darksirens.marks import mark_model_parser
            log_h = jnp.clip(
                mark_model_parser(mark_model, mark_names)(em_catalog, mark_params),
                -_LOG_H_CLIP, _LOG_H_CLIP,
            )
            kernels, log_N_host = marked_catalog_kernel_state(
                cosmo, survey, em_catalog, log_h, log_g_grid=log_g_grid,
                z_depth=survey.z_depth,
            )
            has_flat_marks = (
                em_catalog.field_mark_values is not None
                and em_catalog.field_mark_z is not None
            )
            if is_field or has_flat_marks:
                # mu_miss and the observed marked mass come from the FULL-SKY
                # flat marks (field_mark_*), so the PE and selection states
                # share ONE global normalizer for the same (theta, eta) and the
                # numerator's missing budget matches it.  Field mode REQUIRES
                # these inputs (checked above); the conditional path uses them
                # whenever they are attached, because mu_miss is a survey-level
                # host-efficiency curve and NOT a property of whichever pixels
                # the current view happens to hold -- the view-level estimator
                # below gives the PE and selection seams different modulations
                # when their views differ (see require_view_independent_mu_miss
                # in likelihood/core.py, which refuses that combination).
                from darksirens.marks import mark_model_flat_parser
                log_h_flat = jnp.clip(
                    mark_model_flat_parser(mark_model, mark_names)(
                        em_catalog.field_mark_values, mark_params
                    ),
                    -_LOG_H_CLIP, _LOG_H_CLIP,
                )
                mu_miss = _mu_miss_from_flat(
                    jnp.asarray(em_catalog.field_mark_z),
                    jnp.exp(log_h_flat),
                    jnp.ones_like(log_h_flat),
                )
            else:
                mu_miss = _mu_miss_grid(em_catalog, log_h)              # (N_grid,)
            dN_miss = curves.dN_miss * mu_miss[None, :]                 # (N_rows, N_grid)
            N_host_miss = jnp.trapezoid(dN_miss, zgrid, axis=-1)        # (N_rows,)
            # DEPTH-TRUNCATED CATALOG: the marked kernels were renormalised to
            # unit mass on [0, z_depth], so the amplitude paired with them is
            # the marked mass actually below the depth — the exact marked twin
            # of the unmarked `Nobs * exp(log_depth_mass)` scaling below.  The
            # SAME depth-scaled amplitude is stored on the state as `log_Nobs`
            # (the numerator's observed amplitude), not just used for Z: keeping
            # the raw `log_N_host` there over-weighted the catalog branch by
            # 1/m_pix and broke the per-pixel unit normalisation.
            log_N_host_depth = jnp.where(
                jnp.isfinite(log_N_host),
                log_N_host + kernels.log_depth_mass,
                -jnp.inf,
            )
            N_host_obs = jnp.where(
                jnp.isfinite(log_N_host_depth), jnp.exp(log_N_host_depth), 0.0
            )
            Z = N_host_obs + N_host_miss
            log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1e-300)), 0.0)
            if is_field:
                log_Z_global = field_global_log_Z_marked(
                    cosmo, survey, em_catalog, mu_miss, log_h_flat
                )
            if curves.base_miss is not None:
                # Marked LSS-completion ENSEMBLE (mirrors the unmarked ensemble
                # design below): KEEP the scalar posterior-mean-Q fields above and
                # ADD the member-INDEPENDENT marked base curve + the compact
                # per-member normalizers so the lss_marginalize member vmap
                # reconstructs each member's marked density on the fly
                # (``base_miss * Q_eff_m``), never storing the ``(M, N_rows,
                # N_grid)`` cube.  ``mu_miss`` is pixel-independent (N_grid,) and
                # folds into ``base_miss`` exactly as it composed with the scalar
                # posterior-mean dN_miss; the marked kernels / log_N_host are
                # member-INDEPENDENT.  ``member_N_miss_integrals`` streams the
                # (M, N_rows) missing-mass integrals over the members (chunked +
                # rematerialised in reverse mode) from the SAME marked base.  With
                # lss_marginalize OFF only the scalar fields are read, so the
                # likelihood value is unchanged (only the state TYPE changes).
                base_miss_marked = curves.base_miss * mu_miss[None, :]         # (N_rows,N_grid)
                N_host_miss_members = member_N_miss_integrals(
                    base_miss_marked, em_catalog, survey
                )                                                              # (M,N_rows)
                Z_members = N_host_obs[None, :] + N_host_miss_members
                log_Z_members = jnp.where(
                    Z_members > 0.0, jnp.log(jnp.maximum(Z_members, 1e-300)), 0.0
                )
                if is_field:
                    state = DarkSirenEnsemblePriorState(
                        kernels=kernels, log_Nobs=log_N_host_depth, dN_miss=dN_miss,
                        log_Z=log_Z,
                        base_miss=base_miss_marked,
                        log_Z_members=log_Z_members,
                        log_Z_global=log_Z_global,
                        log_Z_global_members=field_global_log_Z_marked_members(
                            cosmo, survey, em_catalog, mu_miss, log_h_flat
                        ),
                    )
                    return _maybe_materialize(state, materialize_state)
                state = DarkSirenEnsemblePriorState(
                    kernels=kernels, log_Nobs=log_N_host_depth, dN_miss=dN_miss,
                    log_Z=log_Z,
                    base_miss=base_miss_marked,
                    log_Z_members=log_Z_members,
                )
                return _maybe_materialize(state, materialize_state)
            if is_field:
                state = DarkSirenPriorState(
                    kernels=kernels, log_Nobs=log_N_host_depth, dN_miss=dN_miss,
                    log_Z=log_Z, log_Z_global=log_Z_global,
                )
                return _maybe_materialize(state, materialize_state)
            state = DarkSirenPriorState(
                kernels=kernels, log_Nobs=log_N_host_depth, dN_miss=dN_miss, log_Z=log_Z
            )
            return _maybe_materialize(state, materialize_state)

        kernels = catalog_kernel_state(
            cosmo, survey, em_catalog, log_g_grid=log_g_grid,
            z_depth=survey.z_depth,
        )
        Nobs = _row_counts(em_catalog)
        # DEPTH-TRUNCATED CATALOG: kernels.log_kw has been renormalised to unit
        # mass on [0, z_depth], so the amplitude paired with it must be the
        # number of galaxies actually catalogued BELOW the depth, not the full
        # row count.  Using the full count relocates every above-depth galaxy's
        # host probability onto the below-depth redshifts while _assemble_curves
        # simultaneously counts those galaxies in the missing branch
        # (dN_miss == dN_exp above the depth) -- a double count that measured
        # 2.6x on a 10-galaxy row with z_depth=0.3.  Exactly a no-op (m = 1)
        # when the catalog holds nothing above the depth, which is what
        # z_depth asserts in the first place.
        Nobs = Nobs * jnp.exp(kernels.log_depth_mass)
        log_Nobs = jnp.where(Nobs > 0.0, jnp.log(jnp.maximum(Nobs, 1e-300)), -jnp.inf)
        # Scalar-compatibility normalisation: curves.dN_miss / N_miss already
        # carry the deterministic Q (or the posterior-mean Q when only an
        # ensemble was supplied), so this matches the legacy behaviour exactly.
        Z = Nobs + curves.N_miss
        log_Z = jnp.where(Z > 0.0, jnp.log(jnp.maximum(Z, 1e-300)), 0.0)
        if curves.base_miss is None:
            if is_field:
                # Survey-GLOBAL normalizer log Sum_all-pixels[N_obs + N_miss].
                # The per-pixel numerator (log_Nobs, dN_miss) is UNCHANGED; only
                # the denominator becomes global, turning the mixture weight
                # into a host fraction.  field_global_log_Z carries the SAME
                # budget modulation (Q_LSS / delta_g) as curves.dN_miss via the
                # field_* rows.
                log_Z_global = field_global_log_Z(cosmo, survey, em_catalog)
                state = DarkSirenPriorState(
                    kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss,
                    log_Z=log_Z, log_Z_global=log_Z_global,
                )
                return _maybe_materialize(state, materialize_state)
            state = DarkSirenPriorState(
                kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss, log_Z=log_Z
            )
            return _maybe_materialize(state, materialize_state)
        # Fixed LSS-completion ensemble present -> add per-member fields for the
        # Bayesian redshift-prior diagnostic and the lss_marginalize member vmap
        # (the scalar fields above are unchanged: posterior-mean Q).
        Z_members = Nobs[None, :] + curves.N_miss_members          # (M, N_rows)
        log_Z_members = jnp.where(
            Z_members > 0.0, jnp.log(jnp.maximum(Z_members, 1e-300)), 0.0
        )
        if is_field:
            # Per-member survey-global normalizers: member m's mixture weight
            # divides by its OWN Z_m(theta), the field analogue of log_Z_members.
            state = DarkSirenEnsemblePriorState(
                kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss,
                log_Z=log_Z,
                base_miss=curves.base_miss,
                log_Z_members=log_Z_members,
                log_Z_global=field_global_log_Z(cosmo, survey, em_catalog),
                log_Z_global_members=field_global_log_Z_members(
                    cosmo, survey, em_catalog
                ),
            )
            return _maybe_materialize(state, materialize_state)
        state = DarkSirenEnsemblePriorState(
            kernels=kernels, log_Nobs=log_Nobs, dN_miss=curves.dN_miss, log_Z=log_Z,
            base_miss=curves.base_miss, log_Z_members=log_Z_members,
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


# ------------------------------------------------------------
# Factored dark-siren evaluator (member-marginalization hot path)
# ------------------------------------------------------------
# ``_eval_dark_scalar`` fuses the member-INDEPENDENT observed-catalog work (the
# O(N_max_gals) catalog KDE + the zgrid bracket) with the member-DEPENDENT
# missing-galaxy completion (a 2-point gather + one logaddexp).  When the GW
# likelihood marginalizes over an LSS-completion ensemble it evaluates the SAME
# prior for every member, so vmapping the whole scalar over the member axis
# redoes the KDE M times.  These two helpers split that work at the exact seam
# so the KDE is done ONCE and only the completion is vmapped.  Together they are
# op-for-op identical to ``_eval_dark_scalar`` (``eval_dark_member_completion``
# consumes ``eval_dark_obs_bracket``'s output), so composing them reproduces the
# monolithic evaluator per member up to floating-point re-association only.


def eval_dark_obs_bracket(z, pix, state: DarkSirenPriorState, em_catalog: EMCatalog):
    """Member-INDEPENDENT observed part of the dark-siren redshift prior.

    Vectorized over samples like :func:`eval_redshift_prior_with_state`.
    Reproduces ``_eval_dark_scalar`` up to (and including) the zgrid bracket:
    the per-sample catalog KDE ``log p_cat`` (same ``nan_to_num``) and the
    ``_grid_bracket`` index/weight.  Returns ``(A_obs, idx, t)`` with
    ``A_obs = state.log_Nobs[pix] + log p_cat`` -- the piece that depends only
    on the (Q-independent) kernels and ``log_Nobs`` broadcast across members.
    """
    def _one(z_i, pix_i):
        log_p_cat = eval_log_catalog_prior_state(z_i, pix_i, state.kernels, em_catalog)
        # NaN (out-of-grid z, degenerate kernels) must mean "impossible", never p=1.
        log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
        idx, t = _grid_bracket(z_i)
        return state.log_Nobs[pix_i] + log_p_cat, idx, t

    return vmap(_one)(z, pix)


def eval_dark_member_completion(
    A_obs, idx, t, pix, base_miss, member_logq, member_is_log,
    log_Z_row_m_or_global, z_depth, is_field: bool,
):
    """Member-DEPENDENT completion of the dark-siren redshift prior (cube-free).

    Given the member-independent observed part ``A_obs`` and grid bracket
    ``(idx, t)`` from :func:`eval_dark_obs_bracket`, reconstruct member ``m``'s
    missing-galaxy density at the TWO bracket nodes ON THE FLY from the SHARED
    member-independent base ``base_miss`` ``(N_rows, N_grid)`` and this member's
    row-aligned RAW log-Q table ``member_logq`` ``(N_rows, N_grid)`` -- gathering
    only ``base_miss[pix, idx:idx+2]`` and ``member_logq[pix, idx:idx+2]``, never
    the full ``(M, N_rows, N_grid)`` cube.  The member factor is
    ``Q_eff = exp(clip(logQ, ±_LOGQ_CLIP))`` relaxed to 1 beyond ``z_depth`` (a
    concrete Python float or ``None``), and

        miss     = interp( base_miss * Q_eff  @ (idx, idx+1), t )
        log_miss = log(max(miss, tiny))   where miss > 0 else -inf
        log p_z  = logaddexp(A_obs, log_miss) - normalizer,

    which is BIT-IDENTICAL at grid nodes to the pre-existing cube evaluation
    (``base_miss[·, j] = (1 - C) dN_exp`` below the depth, ``= dN_exp`` above,
    and ``Q_eff`` is ``Q_m`` below / ``1`` above; their product is the old
    ``dN_miss_m[·, j]`` node-for-node).  The normalizer is the per-pixel
    ``log_Z`` rows ``(N_rows,)`` under the conditional convention, or the scalar
    survey-global ``log_Z_global_m`` under the field convention.  Written for an
    arbitrary leading sample shape (advanced indexing over ``pix``/``idx``), so
    vmapping over the stacked member axis of ``member_logq`` (and the per-member
    ``log_Z`` leaf) is trivial; ``base_miss`` is member-INDEPENDENT and passed
    unbatched.  ``member_is_log`` / ``is_field`` are static Python bools.
    """
    b_lo = base_miss[pix, idx]
    b_hi = base_miss[pix, idx + 1]
    lq_lo = member_logq[pix, idx]
    lq_hi = member_logq[pix, idx + 1]
    if z_depth is None:
        depth_lo = depth_hi = None
    else:
        depth_lo = zgrid[idx] <= z_depth
        depth_hi = zgrid[idx + 1] <= z_depth
    q_lo = _member_q_eff_from_logq(lq_lo, depth_lo, member_is_log)
    q_hi = _member_q_eff_from_logq(lq_hi, depth_hi, member_is_log)
    miss = _interp_row(b_lo * q_lo, b_hi * q_hi, t)
    log_miss = jnp.where(miss > 0.0, jnp.log(jnp.maximum(miss, 1e-300)), -jnp.inf)
    numerator = jnp.logaddexp(A_obs, log_miss)
    if is_field:
        return numerator - log_Z_row_m_or_global
    return numerator - log_Z_row_m_or_global[pix]


def _eval_complete_scalar(
    z, pix, state: CompletePriorState, survey: SurveyParams, em_catalog: EMCatalog,
    catalog_sky_weighting: str = "conditional",
):
    log_p_cat = eval_log_catalog_prior_state(z, pix, state.kernels, em_catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    if catalog_sky_weighting == "field":
        # FIELD convention: survey field density weighting each occupied pixel by
        # its share N_obs[pix] / N_obs_total of the total observed count:
        #   log p(z, pix) = log p_cat(z|pix) + log N_obs[pix] - log N_obs_total.
        # The normalizer Z = Sum_pix N_obs is theta-independent, so an EMPTY pixel
        # carries zero count weight -> -inf REGARDLESS of
        # complete_empty_pixel_policy (the volume fallback is a conditional-mode
        # per-pixel construct with no place in the global field density).
        field_val = log_p_cat + state.log_Nobs[pix] - state.log_N_obs_total
        return jnp.where(state.row_has[pix], field_val, -jnp.inf)
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

    ``catalog_sky_weighting`` selects the ``dark_sirens`` normalization
    convention; it is a static string branch and is inert for every other
    ``model``.  The signature default ``"conditional"`` is the safe fallback for
    callers that supply no field-normalization inputs; the CLI resolves the
    user-facing default to ``"field"`` (the joint catalog host-density estimand).
    """
    if model == "spectral_sirens":
        return vmap(lambda z_i: jnp.interp(z_i, zgrid, state.log_pvol))(z)

    if model == "bright_sirens":
        return _log_prior_bright_sirens(z, pix, cosmo, survey, em_catalog)

    if model == "dark_sirens_complete":
        return vmap(
            lambda z_i, p_i: _eval_complete_scalar(
                z_i, p_i, state, survey, em_catalog, catalog_sky_weighting
            )
        )(z, pix)

    if model == "dark_sirens":
        return vmap(
            lambda z_i, p_i: _eval_dark_scalar(
                z_i, p_i, state, em_catalog, catalog_sky_weighting
            )
        )(z, pix)

    raise ValueError(f"Unknown redshift prior model '{model}'.")


def _eval_dark_member_scalar(
    z, pix, m, member_logq_all, member_is_log,
    state: "DarkSirenEnsemblePriorState", survey: SurveyParams, em_catalog,
):
    """log p_m(z | pix) for LSS-completion ensemble member ``m`` (diagnostic).

        p_m(z|pix) = [N_obs(pix) p_cat(z|pix) + dN_miss^m(z|pix)]
                     / [N_obs(pix) + N_miss^m(pix)].

    ``dN_miss^m`` is reconstructed on the fly at the two bracket nodes from the
    shared ``state.base_miss`` and member ``m``'s row of the RAW log-Q table
    ``member_logq_all`` ``(M, N_rows, N_grid)`` (a resident data constant; the
    ``[m]`` slice is a gather, not an ``(M, N_rows, N_grid)`` copy), identically
    to :func:`eval_dark_member_completion`.
    """
    log_p_cat = eval_log_catalog_prior_state(z, pix, state.kernels, em_catalog)
    log_p_cat = jnp.nan_to_num(log_p_cat, nan=-jnp.inf, neginf=-jnp.inf)
    idx, t = _grid_bracket(z)
    b_lo = state.base_miss[pix, idx]
    b_hi = state.base_miss[pix, idx + 1]
    lq_lo = member_logq_all[m, pix, idx]
    lq_hi = member_logq_all[m, pix, idx + 1]
    if survey.z_depth is None:
        depth_lo = depth_hi = None
    else:
        depth_lo = zgrid[idx] <= survey.z_depth
        depth_hi = zgrid[idx + 1] <= survey.z_depth
    q_lo = _member_q_eff_from_logq(lq_lo, depth_lo, member_is_log)
    q_hi = _member_q_eff_from_logq(lq_hi, depth_hi, member_is_log)
    miss = _interp_row(b_lo * q_lo, b_hi * q_hi, t)
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
        M = int(state.log_Z_members.shape[0])
        # Resolve the row-aligned RAW member log-Q ONCE (a view of the resident
        # data constant on the hot path); each member's density is reconstructed
        # from state.base_miss * Q_eff_m at the query brackets (no cube).
        member_logq_all, member_is_log = _resolve_member_logq_row(em_catalog)

        def _per_member(m):
            return vmap(
                lambda z_i, p_i: _eval_dark_member_scalar(
                    z_i, p_i, m, member_logq_all, member_is_log,
                    state, survey, em_catalog,
                )
            )(z, pix)

        return vmap(_per_member)(jnp.arange(M, dtype=jnp.int32))  # (M, len(z))

    lp = eval_redshift_prior_with_state(model, state, z, pix, cosmo, survey, em_catalog)
    return jnp.reshape(lp, (1, -1))


# ------------------------------------------------------------
# One-shot prior implementations (registry; checks/tests/back-compat)
# ------------------------------------------------------------
#
# ``threads_distance_table`` rather than ``@jit``: these reach the 106.8 MB
# comoving-distance table (through ``log_volume_prior`` / the completion curves),
# and the cluster likelihood resolves them through ``get_redshift_prior``, i.e.
# from INSIDE an enclosing trace.  A plain ``@jit`` would close over the caller's
# table tracer, which JAX neither keys its tracing cache on nor tolerates being
# replayed from a dead trace.  See ``utils.cosmology``.

@threads_distance_table()
def _log_prior_spectral_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
) -> jnp.ndarray:
    """GW-only prior: normalised comoving volume element."""
    return log_volume_prior_vmap(z, cosmo, survey)


@threads_distance_table()
def _log_prior_complete_catalog(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
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


@threads_distance_table()
def _log_prior_bright_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
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


@threads_distance_table()
def _log_prior_dark_sirens(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
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