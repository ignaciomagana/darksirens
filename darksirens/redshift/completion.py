"""
completion.py
-------------
Catalog completion model: characterises the missing-galaxy distribution
for pixels / redshifts where the EM survey is incomplete.

Model
-----
Galaxy number density follows n(z) = n0 (1+z)^delta, so the expected
count per redshift shell in a pixel of solid angle ``apix`` is

    dN_exp/dz = n0 * apix * dV_c/dz * (1+z)^delta  =  n0 * apix * g(z),

with ``g(z) = dV_c/dz * (1+z)^delta`` the *galaxy measure*.  The exponent
is delta, not (delta-1); merger-rate evolution is handled by the
population model and must not appear here.

Completeness is the matched-kernel differential ratio

    C(z|pix) = clip( dN_obs_s(z|pix) / dN_exp_s(z) , 0, 1 ),

where *both* sides are smoothed by the same linear operator: a Gaussian
kernel of width ``_SIGMA_SMOOTH``, truncated to [0, zmax] and
renormalised per source point.  The numerator is the per-galaxy KDE
(each observed galaxy contributes a unit-mass truncated kernel); the
denominator applies the identical operator to dN_exp/dz via quadrature
on ``zgrid``.  When the survey carries a concrete ``z_depth`` the
expected side is truncated at the depth before smoothing
(``S @ (dN_exp · 1[z <= z_depth])``), matching the support of the
observed side, whose sources all lie below the depth.  Because the
operator is linear and shared, a constant true completeness passes
through the ratio exactly, including at the z = 0 boundary and at the
depth edge.  There is no parametric roll-off: the ratio itself is
the completeness estimator (survey depth shows up as the data-driven
decline of dN_obs_s).

That exactness holds for the *observed* redshifts as given: the
numerator smooths the per-galaxy POINT ESTIMATES ``zgals`` and ignores
their uncertainties ``dzgals``, so for a photometric catalog the
observed side is effectively convolved with ``sqrt(_SIGMA_SMOOTH^2 +
dz^2)`` while the denominator keeps ``_SIGMA_SMOOTH``.  Wherever
``dN_exp/dz`` has curvature the ratio then picks up a bias of order
``dz^2/2 * (dN_exp)''/dN_exp`` — a few percent in C for ``dz ~
_SIGMA_SMOOTH``, which is an O(50%) *relative* error on the ``(1 - C)``
missing branch of a nearly complete pixel, and of opposite sign either
side of the ``dV_c/dz`` turnover.  The estimator is therefore matched
only for ``dz << _SIGMA_SMOOTH`` (spectroscopic depth);
``completion_clip_diagnostics`` measures ``max(dz)/_SIGMA_SMOOTH`` per
run and warns when it is not small.  Fixing it properly needs a
per-galaxy kernel on BOTH sides (a dz-mixture denominator), not a wider
numerator alone.

The estimator above is the PER-PIXEL mode (``survey.c_mode == 0``, the
legacy default).  Its numerator is per-pixel while its denominator is
isotropic, so C really estimates ``C_sel(z) * (1 + delta_obs(pix, z))``
clipped to [0, 1]: the observed angular clustering is absorbed into the
completeness, ``(1 - C)`` anti-tracks the true missing density, and
overdense complete pixels clip at 1, silently discarding their excess.
The opt-in AGGREGATE mode (``survey.c_mode == 1``) instead forms ONE
sky-aggregate curve per proposal,

    Cbar(z) = clip( Sum_p dN_obs_s(z|p) / (N_pix_total * dN_exp_s(z)), 0, 1 ),
    N_pix_total = round(4 pi / apix)   (occupied AND empty pixels),

broadcast to every pixel in place of the per-pixel C -- a radial-only
budget ("C says HOW MUCH is missing"), with all angular structure carried
by the mean-one Q field ("Q says WHERE it goes").  The field-convention
global normalizer uses the SAME curve, so numerator and normalizer carry
one budget.  ``z_depth`` semantics are unchanged (Cbar := 0 beyond the
depth).  ``c_mode`` is resolved at trace time (structural, never sampled:
``None`` per_pixel, the leaf-less ``AggregateCMode`` sentinel aggregate;
see ``_is_aggregate_c_mode``).  Under ``c_mode="selection"`` a second
structural flag, ``survey.selection_family`` (``None`` gaussian, the
leaf-less ``SchechterSelectionFamily`` sentinel schechter; see
``_decode_selection_family``), picks which luminosity function forms the
parametric curve -- both land in the same ``C_bar_raw`` slot.  Known
caveat, deliberately not addressed here: Cbar is proportional to 1/n0
through a single GLOBAL clip, so near full completeness the likelihood
can be kinked in n0 (one clip boundary for the whole sky instead of
per-pixel boundaries that engage pixel by pixel) -- flagged for the
inference stage, watch sampler behaviour in n0.

Under ``c_mode == 2`` (``"selection"``) the completeness is the
parametric ``C_sel(z; m_lim, M0hat/Mstar_hat, ...)``, which by
construction never sees a galaxy count.  The missing density is then

    dN_miss = (1 - C_sel(z)) n0 apix dV_c/dz (1+z)^delta,

strictly proportional to ``n0`` and (since ``r`` is proportional to
``1/H0`` and ``dV_c/dz = c r^2 / (H0 E)``) to ``H0^-3``, while the
observed branch ``N_obs`` is a pure count.  The mixture weight
``N_miss : N_obs`` -- how much of the host probability is the catalog,
i.e. the strength of the catalog's H0 information -- is therefore set by
the sampled ``10^log10n0 / H0^3`` with NOTHING in the likelihood
calibrating it (``log10n0`` is sampled flat over three decades), and
unlike the counts-based modes there is no ``C <= 1`` clip to force
``dN_miss -> 0`` where the catalog is complete.  Nothing enforces the
internal consistency ``dN_obs_s(z) ~ C_sel(z) n0 apix g(z)``, so the
completeness of a selection run is prior-driven.  Audit it with
:func:`selection_budget_audit` (the ``--validate_completion`` dry run
stamps its numbers and warns on a gross mismatch).

The missing-galaxy *density* (count units, per unit z) is

    dN_miss(z|pix) = (1 - C(z|pix)) * dN_exp(z) * max(1 + b_eff * delta_g(pix,z), 0),

with ``b_eff = alpha_miss * b_miss``.  (``alpha_miss`` and ``b_miss``
enter the model only through this product — they are exactly degenerate —
so only ``b_miss`` is sampled and ``alpha_miss`` defaults to 1.)
``compute_lss_overdensity`` builds ``delta_g`` with an exactly zero
full-sky mean, so ``Sum_p (1 + b_eff delta_g_p) == N_pix`` and the factor
is a pure REDISTRIBUTION of the budget set by ``(1 - C)`` and ``n0``.
The ``max(..., 0)`` floor breaks that identity wherever
``b_eff delta_g < -1`` (reachable for ``b_miss > 1``, since
``delta_g >= -1``): the total missing count then EXCEEDS
``(1 - C) N_exp N_pix``, i.e. the floor inflates the physical budget and
with it the inferred completeness and the observed-vs-missing odds that
carry the dark-siren information.  The Q path removes exactly this
failure mode by construction (``lognormal_completion.renormalize_q_mean_one``,
which measured +55% budget inflation from an unrenormalized monopole);
the legacy ``delta_g`` factor has no counterpart, so
``completion_clip_diagnostics`` warns whenever the floor engages at all.
``dN_miss`` carries the same (1+z)^delta evolution as dN_exp, and is the
quantity the assembled redshift prior adds to the catalog counts; see
``darksirens/redshift/prior.py``.

Per-pixel KDE cache
-------------------
``_kde_dndz_obs`` builds an (N_grid,) smoothed observed density for one
pixel.  ``build_pixel_kde_cache`` precomputes it for every unique pixel
appearing in the PE and selection sample sets, once at startup, stored
as ``EMCatalog.dN_obs_kde`` with the lookup ``EMCatalog.pixel_to_cache_idx``
indexed by the *same key space the cache was built with* (global HEALPix
pixel in the production union path; see
``darksirens/likelihood/catalog_views.py``).

Public API
----------
build_pixel_kde_cache(unique_pixels, zgals, n_pix_catalog, wgals=None, ngals=None)
catalog_completion(z, pix, cosmo, survey, em_catalog)        -> (f, p_miss, C_eff)
catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)   -> vectorised
completion_curves(cosmo, survey, em_catalog)                 -> per-row curve bundle
compute_lss_overdensity(zgals, nside, wgals=None, ngals=None)
log_galaxy_measure_grid(cosmo, survey)                       -> log g(zgrid)

The scalar/vmap entry points recompute the per-row curves per call and
are intended for checks and diagnostics.  Hot paths (the likelihood)
should use ``completion_curves`` once per parameter proposal via
``darksirens.redshift.prior.prepare_redshift_prior_state``.
"""

from __future__ import annotations

import contextvars
import warnings
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import ndtr
from typing import NamedTuple

from darksirens.utils.cosmology import dV_of_z
from darksirens.core.types import (
    AggregateCMode,
    CosmoParams,
    EMCatalog,
    LegacyLSSFloor,
    SchechterSelectionFamily,
    SelectionCMode,
    SurveyParams,
)

from darksirens.redshift.grid import zgrid


# Gaussian kernel width for the completeness ratio (both sides).
_SIGMA_SMOOTH: float = 0.05

# Completeness estimator mode (SurveyParams.c_mode; string names in
# darksirens.core.constants.C_MODES).  Resolved once per trace -- a
# Python-level branch exactly like ``z_depth``.
C_MODE_PER_PIXEL: int = 0   # legacy default: per-pixel matched-kernel ratio
C_MODE_AGGREGATE: int = 1   # one sky-aggregate Cbar(z) broadcast to all pixels
C_MODE_SELECTION: int = 2   # parametric C_sel(z; m_lim, M0hat, sigma_M)


def _is_aggregate_c_mode(c_mode) -> bool:
    """Trace-safe decode of ``SurveyParams.c_mode`` -> aggregate?

    The likelihood's module-level jit (``darksiren_log_likelihood``) takes the
    survey as a pytree ARGUMENT, so any int leaf crosses that boundary as a
    value-unreadable tracer.  BOTH modes therefore ride the ``z_depth``
    structural pattern: per_pixel is carried as ``None`` (the container
    default) and aggregate as the leaf-less
    :data:`darksirens.core.types.C_MODE_AGGREGATE_STRUCT` sentinel (an empty
    NamedTuple -- pure pytree structure), which the parameter decoder stamps
    on every production survey.  A concrete int is decoded by value (so eager
    callers may pass 0 or 1 directly), but an int that DID cross a jit
    boundary is unreadable here and is a HARD ERROR: assuming either mode
    would silently swap the completeness estimator -- and with it the entire
    clustering budget -- for a plain plumbing mistake (a concrete
    ``c_mode=0`` traced through jit used to be evaluated as aggregate).
    """
    return _decode_c_mode(c_mode) == C_MODE_AGGREGATE


def _is_selection_c_mode(c_mode) -> bool:
    """Trace-safe decode of ``SurveyParams.c_mode`` -> parametric selection?

    Same structural contract as :func:`_is_aggregate_c_mode`; the selection
    mode rides :data:`darksirens.core.types.C_MODE_SELECTION_STRUCT`.
    """
    return _decode_c_mode(c_mode) == C_MODE_SELECTION


def _validated_c_mode(code: int) -> int:
    """An int ``c_mode`` must name one of the three legal estimators.

    Anything else (a stray 3, a -1, a ``True`` that decodes to 1) used to fall
    through: ``_is_selection_c_mode`` and ``_is_aggregate_c_mode`` both read
    False, ``C_bar_raw`` stayed ``None`` and the run silently used the legacy
    per-pixel estimator -- exactly the silent estimator swap the surrounding
    hard errors exist to prevent.
    """
    legal = (C_MODE_PER_PIXEL, C_MODE_AGGREGATE, C_MODE_SELECTION)
    if code not in legal:
        from darksirens.core.constants import C_MODES

        raise ValueError(
            f"SurveyParams.c_mode={code} is not a completeness estimator code; "
            f"the legal set is {dict(C_MODES)}. An unrecognised code would "
            "silently fall through to the legacy per-pixel estimator, swapping "
            "the entire clustering budget for a plumbing mistake. Carry the "
            "mode structurally across jit boundaries (None / "
            "C_MODE_AGGREGATE_STRUCT / C_MODE_SELECTION_STRUCT)."
        )
    return code


def _decode_c_mode(c_mode) -> int:
    """Decode any legal ``c_mode`` spelling to its int code; hard-error on
    a traced int (guessing would silently swap the completeness estimator
    -- and with it the entire clustering budget) or an out-of-range code."""
    if c_mode is None:
        return C_MODE_PER_PIXEL
    if isinstance(c_mode, AggregateCMode):
        return C_MODE_AGGREGATE
    if isinstance(c_mode, SelectionCMode):
        return C_MODE_SELECTION
    if isinstance(c_mode, (bool, np.bool_)):
        # bool is an int subclass, so True would decode to aggregate.
        raise ValueError(
            f"SurveyParams.c_mode={c_mode!r} is a bool, not a completeness "
            "estimator code (True would decode to aggregate). Use None / "
            "C_MODE_AGGREGATE_STRUCT / C_MODE_SELECTION_STRUCT."
        )
    if isinstance(c_mode, (int, np.integer)):
        return _validated_c_mode(int(c_mode))
    try:
        return _validated_c_mode(int(c_mode))    # concrete 0-d array
    except jax.errors.ConcretizationTypeError as exc:
        raise ValueError(
            "SurveyParams.c_mode reached a jit boundary as a traced int "
            "leaf, so its value cannot be read at trace time and the "
            "completeness estimator cannot be selected (guessing would "
            "silently swap the clustering budget). Carry c_mode "
            "STRUCTURALLY across jit boundaries instead: None (or a "
            "concrete int 0, eagerly) for per_pixel, "
            "darksirens.core.types.C_MODE_AGGREGATE_STRUCT (or a concrete "
            "int 1, eagerly) for aggregate, and "
            "darksirens.core.types.C_MODE_SELECTION_STRUCT (or a concrete "
            "int 2, eagerly) for the parametric selection mode. The "
            "parameter decoder "
            "(darksirens.inference.parameters._survey_params) performs this "
            "normalisation for production runs."
        ) from exc


#: Luminosity-function families of the parametric selection curve; the theta
#: tables live in :mod:`darksirens.redshift.selection`.
_SELECTION_FAMILIES = ("gaussian", "schechter")


def _decode_selection_family(selection_family) -> str:
    """Trace-safe decode of ``SurveyParams.selection_family`` -> family name.

    Structural like ``c_mode``: ``None`` (gaussian, the legacy default) or the
    leaf-less :data:`darksirens.core.types.SELECTION_FAMILY_SCHECHTER_STRUCT`
    sentinel; eager strings are accepted for host-side callers.  Anything else
    -- notably a traced leaf -- is a HARD ERROR: guessing the family would
    silently swap the entire completeness curve.
    """
    if selection_family is None:
        return "gaussian"
    if isinstance(selection_family, SchechterSelectionFamily):
        return "schechter"
    if isinstance(selection_family, str) and selection_family in _SELECTION_FAMILIES:
        return selection_family
    raise ValueError(
        "SurveyParams.selection_family could not be decoded at trace time "
        f"(got {type(selection_family).__name__}); guessing would silently "
        "swap the entire C_sel(z) luminosity-function shape. Carry the family "
        "STRUCTURALLY: None for the gaussian default, "
        "darksirens.core.types.SELECTION_FAMILY_SCHECHTER_STRUCT for "
        "schechter (the eager strings 'gaussian'/'schechter' are accepted "
        "host-side only). The parameter decoder "
        "(darksirens.inference.parameters._survey_params) performs this "
        "normalisation for production runs."
    )


def _is_legacy_lss_floor(lss_floor) -> bool:
    """Trace-safe decode of ``SurveyParams.lss_floor`` -> unrenormalized legacy?

    Structural like ``c_mode``: ``None`` (the conserving default) or the
    leaf-less :data:`darksirens.core.types.LSS_FLOOR_LEGACY_STRUCT` sentinel;
    the eager strings ``"conserve"`` / ``"legacy"`` are accepted host-side.
    Anything else -- notably a bool or int leaf that crossed a jit boundary as
    a value-unreadable tracer -- is a HARD ERROR, for the same reason
    :func:`_is_aggregate_c_mode` refuses one: guessing would silently change
    the TOTAL missing-galaxy budget, which is the quantity carrying the
    dark-siren information.
    """
    if lss_floor is None or lss_floor == "conserve":
        return False
    if isinstance(lss_floor, LegacyLSSFloor) or lss_floor == "legacy":
        return True
    raise ValueError(
        "SurveyParams.lss_floor could not be decoded at trace time (got "
        f"{type(lss_floor).__name__}); guessing would silently change the "
        "total missing-galaxy budget. Carry it STRUCTURALLY: None for the "
        "conserving default, darksirens.core.types.LSS_FLOOR_LEGACY_STRUCT "
        "for the unrenormalized legacy floor (the eager strings 'conserve'/"
        "'legacy' are accepted host-side only)."
    )


def legacy_lss_floor_normalizer(survey: SurveyParams, em_catalog: EMCatalog):
    r"""Full-sky mean of ``max(1 + b_eff delta_g, 0)`` at every z, or ``None``.

    ``compute_lss_overdensity`` centres ``delta_g`` on the FULL sky (occupied
    pixels carry the deviation from the occupied-pixel mean, empty pixels carry
    exactly zero), so

        Sum_p (1 + b_eff delta_g_p(z)) == N_pix   for every z, every b_eff,

    i.e. the unfloored factor is a pure REDISTRIBUTION of the missing budget
    that ``(1 - C)`` and ``n0`` set.  ``max(..., 0)`` breaks that identity
    wherever ``b_eff delta_g < -1`` -- reachable for ``b_miss > 1``, since
    ``delta_g >= -1`` by construction and ``b_miss`` samples over [0, 3].  Two
    equal pixels with ``delta_g = (-1, +1)`` and ``b_eff = 2`` floor to
    ``(0, 3)``, mean 1.5: a 50% INFLATION of the missing budget, 100% at
    ``b_eff = 3``.  That is not a spatial reshuffle -- it changes the total
    catalog-versus-missing odds, and with them the inferred completeness and
    the strength of the catalog's H0 information.

    Dividing the floored factor by this normalizer restores
    ``Sum_p lss_p(z) == N_pix`` exactly, for every parameter draw and every
    redshift slice, while leaving the ANGULAR pattern the floor produced
    untouched.  (The Q path has carried the same renormalization since
    ``lognormal_completion.renormalize_q_mean_one``, which measured +55% budget
    inflation from an unrenormalized monopole; this is the ``delta_g`` path's
    missing counterpart.)

    Returns ``None`` -- meaning "no renormalization, the factor is already
    mean one" -- when:

    * the legacy floor was explicitly requested (``survey.lss_floor`` is the
      legacy sentinel), or
    * ``delta_g_pix_z`` is the ``(1, N_grid)`` broadcast dummy, which is the
      canonical spelling of "no per-pixel LSS" everywhere in the package
      (``prior.py`` and ``likelihood/core.py`` both gate ``use_lss`` on
      ``shape[0] != 1``).  Its every pixel shares one row, so the sky mean IS
      the factor and dividing would erase the field rather than conserve it.

    Written as ``1 + <relu(-(1 + b_eff delta_g))>_p`` rather than as
    ``<max(1 + b_eff delta_g, 0)>_p``.  The two are equal analytically
    (``max(y, 0) = y + relu(-y)`` and ``<1 + b_eff delta_g>_p == 1`` exactly),
    but the relu form is EXACTLY 1.0 whenever nothing floors -- the deficit is
    then the mean of an all-zero array -- where the direct mean lands 1 ulp
    away and would perturb the last bit of every existing run that never
    engaged the floor.  It is also the cheaper of the two.

    Cost is one fused elementwise-relu + mean over the resident
    ``(N_pix, N_grid)`` table, once per proposal (not once per row).
    """
    if _is_legacy_lss_floor(survey.lss_floor):
        return None
    dg = em_catalog.delta_g_pix_z
    if dg is None or dg.shape[0] == 1:
        return None
    b_eff = survey.alpha_miss * survey.b_miss
    deficit = jnp.mean(
        jnp.maximum(-(1.0 + b_eff * jnp.asarray(dg)), 0.0), axis=0)
    return 1.0 + deficit


_ZMAX: float = float(np.asarray(zgrid)[-1])
_SQRT2PI: float = float(np.sqrt(2.0 * np.pi))


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a (possibly non-uniform) grid."""
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


_TRAPW_NP = _trapezoid_weights(np.asarray(zgrid, dtype=np.float64))
_TRAPW = jnp.asarray(_TRAPW_NP)


def _truncated_kernel_mass_np(z0: np.ndarray, sigma: float) -> np.ndarray:
    """Mass of N(.; z0, sigma) inside [0, zmax] (NumPy, import time)."""
    from scipy.special import ndtr as _ndtr
    return _ndtr((_ZMAX - z0) / sigma) - _ndtr(-z0 / sigma)


def _build_smoothing_operator() -> jnp.ndarray:
    """
    Linear operator S such that (S @ f) approximates

        (S f)(z_g) = ∫_0^zmax  N(z_g; z', σ) / M(z')  f(z') dz'

    via trapezoid quadrature on ``zgrid``, where M(z') is the truncated
    kernel mass.  This is the *same* per-source-point kernel used by the
    observed-galaxy KDE, so ratios of S-smoothed quantities are exact for
    constant completeness, boundary included.
    """
    z = np.asarray(zgrid, dtype=np.float64)
    M = np.maximum(_truncated_kernel_mass_np(z, _SIGMA_SMOOTH), 1e-300)
    pdf = np.exp(-0.5 * ((z[:, None] - z[None, :]) / _SIGMA_SMOOTH) ** 2)
    pdf /= _SQRT2PI * _SIGMA_SMOOTH
    S = (pdf / M[None, :]) * _TRAPW_NP[None, :]
    return jnp.asarray(S)


# (N_grid, N_grid) ~8 MB float64; built once at import.
_S_EXP: jnp.ndarray = _build_smoothing_operator()

# Jit-boundary threading for _S_EXP, mirroring utils.cosmology's distance
# table (issue #305's residual): a module-global concrete jax.Array captured
# by a jit lowers as a dense<1000x1000xf64> HLO CONSTANT — ~16 MB of module
# text per specialization — instead of a parameter.  Callers that jit a body
# reaching _precompute_grids (likelihood/factory._jit_likelihood_body) pass
# the operator as a jit argument and bind it for the trace; every deep call
# site keeps reading it ambiently, so no signatures change below this frame.
_ACTIVE_S_EXP = contextvars.ContextVar("darksirens_smoothing_operator")


def smoothing_operator():
    """The active expected-counts smoothing operator (default: `_S_EXP`)."""
    return _ACTIVE_S_EXP.get(_S_EXP)


@contextmanager
def bound_smoothing_operator(operator):
    """Make ``operator`` the active smoothing operator for the enclosed scope.

    ``None`` is a no-op so callers can thread an unresolved argument through
    without special-casing (same contract as ``bound_distance_table``).
    """
    if operator is None:
        yield
        return
    token = _ACTIVE_S_EXP.set(operator)
    try:
        yield
    finally:
        _ACTIVE_S_EXP.reset(token)


# Ride the SAME jit boundaries as the distance table: every
# threads_distance_table-decorated function now resolves the active operator
# at its call site and rebinds it inside its own trace, so a nested boundary
# reached from inside an enclosing trace (the factory's outer jit binds a
# TRACER here) passes the tracer as an argument instead of leaking it.
from darksirens.utils.cosmology import register_ambient_jit_channel

register_ambient_jit_channel(smoothing_operator, bound_smoothing_operator)


# ------------------------------------------------------------
# Shared galaxy measure g(z) = dV/dz * (1+z)^delta
# ------------------------------------------------------------

def log_galaxy_measure_grid(cosmo: CosmoParams, survey: SurveyParams) -> jnp.ndarray:
    """
    log g(zgrid) with g(z) = dV_c/dz(z) * (1+z)^delta.

    g(0) = 0 (dV vanishes); the log is floored so that interpolation near
    z = 0 stays finite (the prior density correctly -> 0 there).
    Depends only on (cosmo, survey.delta) — inside a JIT trace JAX hoists
    this out of any vmap over samples.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    g = dV_of_z(zgrid, H0, Om0, w0, wa) * (1.0 + zgrid) ** survey.delta
    return jnp.log(jnp.maximum(g, 1e-300))


# ------------------------------------------------------------
# Per-pixel observed dN/dz via boundary-corrected Gaussian KDE
# ------------------------------------------------------------

def _kde_dndz_obs(
    pix: int,
    zgals: jnp.ndarray,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Smoothed observed number density dN_obs_s/dz for pixel ``pix`` on ``zgrid``.

    Each *real* galaxy contributes a unit-mass Gaussian of width
    ``_SIGMA_SMOOTH`` truncated and renormalised on [0, zmax]:

        k(z; z_i) = N(z; z_i, σ) / M(z_i).

    The estimate is Σ_i k(z; z_i) — properly normalised (counts per unit
    z) and boundary-corrected, matching the operator applied to the
    expected counts.  Raw *counts* are used (no luminosity weights): this
    keeps the numerator the direct counterpart of n0·apix·g(z).
    ``wgals``/``ngals`` serve only to mask padded slots.

    The kernel width is the FIXED ``_SIGMA_SMOOTH`` for every galaxy —
    ``dzgals`` deliberately does not enter, because the denominator applies
    one shared operator and a per-galaxy numerator width would unmatch the
    ratio (module docstring: the completeness estimator is matched only for
    ``dz << _SIGMA_SMOOTH``; the run diagnostics report the ratio).  This is
    NOT the catalog's own redshift kernel — see
    ``darksirens.redshift.catalog``, whose observed-host kernel does use
    ``sigma_eff = sqrt(dz^2 + sigma_kde^2)``.

    Safe to vmap over ``pix`` with ``in_axes=(0, None, None, None)``.
    """
    # Accept a 1D ngals array passed positionally as the third argument.
    if ngals is None and wgals is not None and wgals.ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "_kde_dndz_obs requires either wgals or ngals to mask padded galaxies"
        )

    zs = zgals[pix]  # (N_max_gals,)
    if ngals is not None:
        real_gal = jnp.arange(zs.shape[0]) < ngals[pix]
    else:
        real_gal = wgals[pix] > 0

    mass = ndtr((_ZMAX - zs) / _SIGMA_SMOOTH) - ndtr(-zs / _SIGMA_SMOOTH)
    mass = jnp.maximum(mass, 1e-300)
    pdf = jnp.exp(-0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2)
    pdf = pdf / (_SQRT2PI * _SIGMA_SMOOTH)
    kern = (pdf / mass[None, :]) * real_gal[None, :].astype(pdf.dtype)
    return kern.sum(axis=1)  # (N_grid,)


# ------------------------------------------------------------
# Pixel KDE cache — precomputed at startup
# ------------------------------------------------------------

def build_pixel_kde_cache(
    unique_pixels: np.ndarray,
    zgals: jnp.ndarray,
    n_pix_catalog: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
    batch_size: int = 512,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute ``_kde_dndz_obs`` for all unique pixels.  Call once in
    ``make_likelihood`` before the JIT closure is built.

    Computation is chunked (``batch_size`` pixels per JIT call) to bound
    peak memory at high nside.

    Returns
    -------
    dN_obs_kde : (N_unique, N_grid) — smoothed observed densities.
    pixel_to_cache_idx : (n_pix_catalog,) int32 — lookup from the index
        space of ``unique_pixels`` to rows of ``dN_obs_kde``.  Whatever
        key space ``unique_pixels`` lives in (global HEALPix pixels or
        compact rows) is the key space of this lookup; callers must index
        it consistently (see catalog_views.py for the compact-row case).
    """
    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "build_pixel_kde_cache requires either wgals or ngals to mask padded galaxies"
        )

    zgals_jax = jnp.asarray(zgals)
    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    pix_idx = np.asarray(unique_pixels)
    n_unique = int(pix_idx.size)

    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_kde = np.empty((n_unique, zgrid.size), dtype=np.float64)
    for start in range(0, n_unique, batch_size):
        stop = min(start + batch_size, n_unique)
        dN_obs_kde[start:stop] = np.asarray(
            _batch_kde(
                jnp.asarray(pix_idx[start:stop], dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )
    dN_obs_kde = jnp.asarray(dN_obs_kde)

    pixel_to_cache_idx = np.zeros(n_pix_catalog, dtype=np.int32)
    for i, p in enumerate(unique_pixels):
        pixel_to_cache_idx[int(p)] = i

    return dN_obs_kde, jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32)


# ------------------------------------------------------------
# FIELD-convention sky-weighting: survey-global normalization inputs
# ------------------------------------------------------------

class FieldNormalizationInputs(NamedTuple):
    """Survey-global ingredients for the FIELD-convention normalizer.

    ``dN_obs_s / n_empty / N_obs_total`` are the legacy trio;
    ``occupied_pixels`` keys the per-pixel budget modulations
    (:func:`build_field_lss_q_inputs`, :func:`build_field_delta_g_inputs`) to
    the same row order as ``dN_obs_s``.
    """

    dN_obs_s: jnp.ndarray          # (n_occupied, N_grid) float32
    n_empty: int
    N_obs_total: float
    occupied_pixels: np.ndarray    # (n_occupied,) int32, host-side


def build_field_normalization_inputs(
    full_z: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
    batch_size: int = 4096,
) -> FieldNormalizationInputs:
    """Host-side precompute for the FIELD-convention global normalizer.

    Builds the survey-global ingredients consumed by :func:`field_global_log_Z`
    to normalize the catalog redshift prior by the GLOBAL count
    ``Z(theta) = Sum_all-pixels [N_obs,pix + N_miss,pix(theta)]`` instead of the
    per-pixel ``Z[pix]``.  Called ONCE at startup (before compaction /
    ``--drop_full_catalog``), over the FULL-sky catalog rows so empty pixels are
    counted.

    The observed density is the SAME smoothed per-galaxy KDE the in-likelihood
    completion uses (``_kde_dndz_obs``, raw counts, matched truncated kernel),
    so the per-pixel completeness ``C`` computed in ``field_global_log_Z``
    reproduces exactly what ``_assemble_curves`` computes per occupied pixel.

    Parameters
    ----------
    full_z : (N_pix, N_max_gals)
        Full-sky padded galaxy redshifts (one row per HEALPix pixel).
    full_w : (N_pix, N_max_gals) or None
        Padded galaxy base weights (used only to mask padded slots when
        ``full_n`` is absent).
    full_n : (N_pix,) or None
        Real-galaxy count per pixel.  Preferred padding mask and the source of
        ``N_obs_total``; when ``None`` falls back to ``full_w > 0``.

    Returns
    -------
    FieldNormalizationInputs
        ``dN_obs_s`` (n_occupied, N_grid) float32 smoothed observed density for
        occupied pixels; ``n_empty`` galaxy-free pixel count; ``N_obs_total``
        total observed real-galaxy count; ``occupied_pixels`` (n_occupied,)
        int32 global pixel ids keying the per-pixel budget modulations.
    """
    full_z = np.asarray(full_z)
    n_pix_total = int(full_z.shape[0])

    if full_n is not None:
        ngals_np = np.asarray(full_n).reshape(-1).astype(np.int64)
        occupied = ngals_np > 0
        N_obs_total = float(ngals_np.sum())
    elif full_w is not None:
        full_w_np = np.asarray(full_w)
        per_pix = (full_w_np > 0.0).sum(axis=-1)
        occupied = per_pix > 0
        N_obs_total = float(per_pix.sum())
    else:
        raise ValueError(
            "build_field_normalization_inputs requires either full_w or full_n "
            "to mask padded galaxies and count observed galaxies."
        )

    occupied_pixels = np.nonzero(occupied)[0].astype(np.int32)
    n_occ = int(occupied_pixels.size)
    n_empty = int(n_pix_total - n_occ)

    zgals_jax = jnp.asarray(full_z)
    wgals_jax = None if full_w is None else jnp.asarray(full_w)
    ngals_jax = None if full_n is None else jnp.asarray(full_n)

    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_s = np.empty((n_occ, zgrid.size), dtype=np.float64)
    for start in range(0, n_occ, batch_size):
        stop = min(start + batch_size, n_occ)
        dN_obs_s[start:stop] = np.asarray(
            _batch_kde(
                jnp.asarray(occupied_pixels[start:stop], dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )

    # Store as float32: for a full-sky (n_occupied, N_grid) table this halves the
    # device footprint; it only feeds the GLOBAL normalizer (a survey-scale
    # constant) so the f32 rounding is immaterial to the estimand.
    field_dN_obs_s = jnp.asarray(dN_obs_s, dtype=jnp.float32)
    return FieldNormalizationInputs(
        dN_obs_s=field_dN_obs_s,
        n_empty=n_empty,
        N_obs_total=N_obs_total,
        occupied_pixels=occupied_pixels,
    )


class FieldDepthInputs(NamedTuple):
    """Flat FULL-SKY per-galaxy inputs for the DEPTH-truncated field normalizer.

    Only needed when ``survey.z_depth`` is set: the global observed term then has
    to be the depth-scaled count ``Sum_pix N_obs,pix * m_pix(theta)`` rather than
    the raw ``N_obs_total`` (see :func:`field_observed_global_total`), and
    ``m_pix`` depends on theta through the galaxy measure ``g(z)``, so the
    per-galaxy kernel geometry must survive to the likelihood.
    """

    z: jnp.ndarray     # (N_gal_total,) float64 galaxy redshifts
    dz: jnp.ndarray    # (N_gal_total,) float64 per-galaxy redshift sigma
    c: jnp.ndarray     # (N_gal_total,) float64 N_obs,pix * w_i / Sum_{j in pix} w_j


def build_field_depth_inputs(
    full_z: jnp.ndarray,
    full_dz: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
) -> FieldDepthInputs:
    """Host-side precompute for the DEPTH-consistent FIELD observed total.

    With a survey depth the per-pixel numerator's observed branch integrates to
    ``N_obs,pix * m_pix(theta)`` (``prior.py``'s ``Nobs * exp(log_depth_mass)``),
    where

        m_pix = Sum_{i in pix} (w_i / W_pix) * Z_i^depth / Z_i

    is the catalog mixture's mass below the depth (``W_pix = Sum_j w_j``,
    ``Z_i^depth / Z_i`` the truncated/full GL kernel norms of
    :func:`darksirens.redshift.catalog._row_log_kernel_norms`).  The survey-global
    normalizer must carry the SAME convention, so it needs

        Sum_pix N_obs,pix * m_pix = Sum_{i, full sky} c_i * Z_i^depth / Z_i,
        c_i = N_obs,pix(i) * w_i / W_pix(i),

    a FLAT per-galaxy reduction (the pixel structure cancels).  ``c_i`` is
    theta-INDEPENDENT and precomputed here; the ratio is theta-dependent (through
    ``g(z)``, and through ``survey.sigma_kde`` in ``sigma_eff``) and is evaluated
    per proposal by :func:`_field_depth_weighted_mass`.

    Parameters mirror :func:`build_field_normalization_inputs` (FULL-sky padded
    rows, one per HEALPix pixel) plus ``full_dz``, the padded per-galaxy sigma.
    Real (non-padded) slots are selected by ``full_n`` (fallback ``full_w > 0``),
    in the SAME row-major order as :func:`build_field_mark_inputs`, so the marked
    normalizer can pair these ``z``/``dz`` with its own ``field_mark_w``.

    Returns
    -------
    FieldDepthInputs
        ``(z, dz, c)``, each ``(N_gal_total,)`` float64.  Kept at full precision
        (not the f32 of ``dN_obs_s``): the ratio is exponentially sensitive to
        ``(z_depth - z_i) / sigma_i``, and f64 keeps the global term bit-consistent
        with the per-pixel numerator it must match.
    """
    z_np = np.asarray(full_z)
    dz_np = np.asarray(full_dz)
    if dz_np.shape != z_np.shape:
        raise ValueError(
            "build_field_depth_inputs: full_dz must have the same (N_pix, N_max) "
            f"shape as full_z; got {dz_np.shape} vs {z_np.shape}."
        )
    if full_n is not None:
        n_np = np.asarray(full_n).reshape(-1).astype(np.int64)
        real = np.arange(z_np.shape[1])[None, :] < n_np[:, None]
    elif full_w is not None:
        real = np.asarray(full_w) > 0.0
    else:
        raise ValueError(
            "build_field_depth_inputs requires full_n or full_w to mask padded "
            "galaxy slots."
        )

    w_np = (
        np.asarray(full_w, dtype=np.float64) if full_w is not None
        else real.astype(np.float64)
    )
    w_np = np.where(real, w_np, 0.0)
    # ``N_obs,pix`` is the real-galaxy COUNT (prior._row_counts), never the weight
    # sum; ``W_pix`` is the weight sum the mixture normalises by.
    n_obs_pix = (
        np.asarray(full_n).reshape(-1).astype(np.float64) if full_n is not None
        else real.sum(axis=-1).astype(np.float64)
    )
    W_pix = w_np.sum(axis=-1)
    W_safe = np.where(W_pix > 0.0, W_pix, 1.0)
    c_np = w_np * (n_obs_pix / W_safe)[:, None]

    return FieldDepthInputs(
        z=jnp.asarray(z_np[real], dtype=jnp.float64),
        dz=jnp.asarray(dz_np[real], dtype=jnp.float64),
        c=jnp.asarray(c_np[real], dtype=jnp.float64),
    )


def build_field_lss_q_inputs(
    logq_map: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Field-normalizer Q_LSS modulation rows from a GLOBAL log-Q table.

    ``logq_map`` must be globally pixel-indexed with exactly ``n_pix_total``
    rows (the field normalizer sums over the FULL sky, so a compact per-view
    table cannot supply the empty-pixel budget).  Returns

    - ``field_lss_q``: (n_occupied, N_grid) float32 LINEAR Q rows aligned with
      ``FieldNormalizationInputs.occupied_pixels`` / ``field_dN_obs_s``;
    - ``field_lss_q_empty_sum``: (N_grid,) float64 ``Sum_{p empty} Q_p(z)`` —
      a data constant (Q is loaded data, never theta-dependent), the
      empty-pixel budget curve (empty pixels have C == 0).

    ``logQ`` is clipped to ``±_LOGQ_CLIP`` before exponentiating, EXACTLY as the
    per-pixel numerator's :func:`_to_q` does: the shipped tables are not bounded
    by the builder's clip (it is applied BEFORE the per-z mean-one renorm, which
    then shifts every row of a bin by its log-monopole — see
    ``cli/build_lognormal_completion._stamp_post_renorm_railing``), so an
    unclipped normalizer would integrate a LARGER missing budget than the
    numerator on railed cells and break the field convention's
    ``Sum_pix integral p_field dz == 1`` identity.
    """
    logq_np = np.asarray(logq_map, dtype=np.float64)
    if logq_np.ndim != 2 or logq_np.shape[0] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_inputs requires a GLOBAL (n_pix_total, N_grid) "
            f"log-Q table; got shape {logq_np.shape} for n_pix_total="
            f"{int(n_pix_total)}. The field normalizer sums the full sky, so a "
            "compact per-view Q table cannot be used -- rebuild Q over the "
            "full nside."
        )
    if logq_np.shape[1] != int(zgrid.size):
        raise ValueError(
            f"log-Q table has {logq_np.shape[1]} grid nodes but zgrid has "
            f"{int(zgrid.size)}."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    q = np.exp(np.clip(logq_np, -_LOGQ_CLIP, _LOGQ_CLIP))
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[occ] = True
    q_occ = jnp.asarray(q[occ], dtype=jnp.float32)
    q_empty_sum = jnp.asarray(q[~occ_mask].sum(axis=0), dtype=jnp.float64)
    return q_occ, q_empty_sum


def build_field_lss_q_member_inputs(
    logq_members: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-member field-normalizer Q rows from a GLOBAL (M, n_pix, N_grid)
    log-Q ensemble: the member analogue of :func:`build_field_lss_q_inputs`.

    Returns ``(field_lss_q_members, field_lss_q_empty_sum_members)`` —
    (M, n_occupied, N_grid) float32 and (M, N_grid) float64.  Clipped to
    ``±_LOGQ_CLIP`` like the deterministic rows and the numerator's
    :func:`_member_q_eff_from_logq`.
    """
    logq_np = np.asarray(logq_members, dtype=np.float64)
    if logq_np.ndim != 3 or logq_np.shape[1] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_member_inputs requires a GLOBAL "
            "(M, n_pix_total, N_grid) log-Q ensemble; got shape "
            f"{logq_np.shape} for n_pix_total={int(n_pix_total)}."
        )
    if logq_np.shape[2] != int(zgrid.size):
        raise ValueError(
            f"log-Q ensemble has {logq_np.shape[2]} grid nodes but zgrid has "
            f"{int(zgrid.size)}."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    q = np.exp(np.clip(logq_np, -_LOGQ_CLIP, _LOGQ_CLIP))
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[occ] = True
    q_occ = jnp.asarray(q[:, occ, :], dtype=jnp.float32)
    q_empty_sum = jnp.asarray(q[:, ~occ_mask, :].sum(axis=1), dtype=jnp.float64)
    return q_occ, q_empty_sum


def _q_fp_empty_sum(logq_np, occ_mask, f_p_np, chunk: int = 4096):
    """``Sum_{p empty} f_p Q_p(z)`` from a global log-Q block, chunked.

    Shared core of the deterministic and per-member builders below.  Chunked
    over pixels because the full-sky ``exp`` the plain empty-pixel budget
    already pays is a (n_pix, N_grid) float64 array -- ~0.4 GB at nside 64 on
    the production grid -- and this must not double it.
    """
    empty = np.flatnonzero(~occ_mask)
    acc = np.zeros(logq_np.shape[-1], dtype=np.float64)
    for lo in range(0, empty.size, chunk):
        idx = empty[lo:lo + chunk]
        q = np.exp(np.clip(logq_np[..., idx, :], -_LOGQ_CLIP, _LOGQ_CLIP))
        acc = acc + (f_p_np[idx][..., :, None] * q).sum(axis=-2)
    return acc


def build_field_lss_q_fp_empty_sum(
    logq_map: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
    f_p: np.ndarray,
) -> jnp.ndarray:
    """``Sum_{p empty} f_p Q_p(z)`` -- the f_p-weighted empty-pixel Q budget.

    The twin of :func:`build_field_lss_q_inputs`'s ``field_lss_q_empty_sum``,
    and the datum that makes the per-pixel selection fraction admissible
    ALONGSIDE a Q table.  Without it the field normalizer charges every empty
    pixel ``(1 - Cbar) Q_p(z)``, i.e. it models sky the survey never observed
    as ``Cbar``-COMPLETE; with it the budget is
    ``Sum_empty Q_p (1 - f_p Cbar)``, so an off-footprint pixel (``f_p = 0``)
    is correctly counted as entirely missing.

    On a footprint-limited survey the difference is not a correction term: at
    DESI's 26,702 deg^2 of coverage roughly a third of the sky is empty, and
    charging it ``Cbar`` completeness removes that fraction of the missing-host
    budget from the normalizer while the numerator keeps it.

    Same ``±_LOGQ_CLIP`` convention as every other consumer of a log-Q table.
    """
    logq_np = np.asarray(logq_map, dtype=np.float64)
    if logq_np.ndim != 2 or logq_np.shape[0] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_fp_empty_sum requires a GLOBAL "
            f"(n_pix_total, N_grid) log-Q table; got shape {logq_np.shape} "
            f"for n_pix_total={int(n_pix_total)}.")
    f_p_np = np.asarray(f_p, dtype=np.float64).reshape(-1)
    if f_p_np.shape[0] != int(n_pix_total):
        raise ValueError(
            f"build_field_lss_q_fp_empty_sum: f_p has {f_p_np.shape[0]} "
            f"pixels but the Q table has {int(n_pix_total)}.")
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)] = True
    return jnp.asarray(_q_fp_empty_sum(logq_np, occ_mask, f_p_np),
                       dtype=jnp.float64)


def build_field_lss_q_fp_empty_sum_members(
    logq_members: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
    f_p: np.ndarray,
) -> jnp.ndarray:
    """(M, N_grid) per-member twin of :func:`build_field_lss_q_fp_empty_sum`.

    A member ensemble whose normalizer reused the DETERMINISTIC f_p-weighted
    budget would marginalize inconsistent estimands, exactly as the plain
    per-member empty budget must not be reused across members.
    """
    logq_np = np.asarray(logq_members, dtype=np.float64)
    if logq_np.ndim != 3 or logq_np.shape[1] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_fp_empty_sum_members requires a GLOBAL "
            f"(M, n_pix_total, N_grid) log-Q ensemble; got shape "
            f"{logq_np.shape} for n_pix_total={int(n_pix_total)}.")
    f_p_np = np.asarray(f_p, dtype=np.float64).reshape(-1)
    if f_p_np.shape[0] != int(n_pix_total):
        raise ValueError(
            f"build_field_lss_q_fp_empty_sum_members: f_p has "
            f"{f_p_np.shape[0]} pixels but the Q ensemble has "
            f"{int(n_pix_total)}.")
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)] = True
    return jnp.asarray(_q_fp_empty_sum(logq_np, occ_mask, f_p_np),
                       dtype=jnp.float64)


def build_field_delta_g_inputs(
    delta_g_pix_z: jnp.ndarray,
    occupied_pixels: np.ndarray,
) -> jnp.ndarray:
    """Field-normalizer overdensity rows: ``delta_g`` gathered to occupied pixels.

    ``delta_g_pix_z`` must be the REAL per-pixel table (first axis == n_pix;
    the (1, N_grid) dummy means "no modulation" and must not reach here).
    Empty pixels carry ``delta_g == 0`` by construction
    (:func:`compute_lss_overdensity`), so their budget factor is exactly 1 and
    only the occupied rows are stored.
    """
    dg = np.asarray(delta_g_pix_z)
    if dg.ndim != 2 or dg.shape[0] <= 1:
        raise ValueError(
            "build_field_delta_g_inputs requires the real per-pixel "
            f"delta_g table; got shape {dg.shape} (the (1, N_grid) dummy "
            "means no modulation)."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    if occ.size and int(occ.max()) >= dg.shape[0]:
        raise ValueError(
            f"delta_g table has {dg.shape[0]} pixel rows but an occupied "
            f"pixel index reaches {int(occ.max())}."
        )
    return jnp.asarray(dg[occ], dtype=jnp.float32)


# ------------------------------------------------------------
# Precomputed grid bundle (pixel-independent, per cosmo/survey)
# ------------------------------------------------------------

class _CompletionGrids(NamedTuple):
    """Pixel-independent grids computed once per likelihood evaluation.

    ``C_bar_raw`` is the UNCLIPPED sky-aggregate completeness ratio, populated
    only in aggregate mode (``survey.c_mode == C_MODE_AGGREGATE``); ``None``
    (the per-pixel default) keeps the pytree structure -- and therefore every
    downstream ``is not None`` branch -- static, so consumers dispatch on it at
    trace time without touching ``survey.c_mode`` again.  Kept raw (one clip
    site would hide the clip fraction from the diagnostics); consumers clip to
    [0, 1] where they use it.
    """
    log_g: jnp.ndarray          # (N_grid,) log galaxy measure
    dN_exp: jnp.ndarray         # (N_grid,) n0 * apix * g(z)
    dN_exp_smooth: jnp.ndarray  # (N_grid,) S-smoothed expected counts
    # (N_grid,) aggregate/selection curve, OR (S, N_grid) stratified-selection
    # stack (indexed by EMCatalog.pixel_stratum_map), OR None (per-pixel).
    # The ndim is static under jit, so consumers branch on it at trace time.
    C_bar_raw: jnp.ndarray = None
    # (N_grid,) full-sky mean of the FLOORED legacy local-overdensity factor,
    # or None (no per-pixel delta_g, or the legacy unrenormalized floor was
    # asked for).  Pixel-independent by construction and depends on the draw
    # only through b_eff, so it belongs here: computed ONCE per proposal, not
    # once per catalog row.  See :func:`legacy_lss_floor_normalizer`.
    lss_floor_norm: jnp.ndarray = None


def _aggregate_dN_obs_sum(em_catalog: EMCatalog) -> jnp.ndarray:
    """Survey-total smoothed observed density ``Sum_p dN_obs_s(z|p)``, (N_grid,).

    The aggregate numerator is a DATA CONSTANT (theta-independent) summed over
    every pixel of the survey -- empty pixels contribute exactly zero, so any
    row set covering all OCCUPIED pixels suffices.  Reuses the caches that
    already exist rather than building a second one:

    * ``field_dN_obs_s`` (field-convention rows, one per occupied pixel over
      the FULL sky) when present -- the survey-complete row set by
      construction;
    * otherwise the catalog rows themselves, which are the full global pixel
      set only when ``unique_pixels is None`` (legacy full catalogs / the
      eager diagnostic path): the ``dN_obs_kde`` cache if built, else the
      on-the-fly per-row KDE.

    A COMPACT catalog without field inputs is refused: its rows are only the
    PE-union-selection pixels, so occupied pixels outside the union would be
    silently dropped from the numerator and bias Cbar low.  All branches are
    pytree-structure (``None``-ness) decisions, resolved at trace time.
    """
    if em_catalog.field_dN_obs_s is not None:
        return jnp.sum(
            jnp.asarray(em_catalog.field_dN_obs_s, dtype=zgrid.dtype), axis=0
        )
    if em_catalog.unique_pixels is not None:
        raise ValueError(
            "c_mode='aggregate' on a COMPACT catalog requires the field-"
            "convention observed rows (field_dN_obs_s, built over the FULL sky "
            "by build_field_normalization_inputs): the compact rows cover only "
            "the PE-union-selection pixels, so summing them would drop every "
            "occupied pixel outside the union and bias Cbar low. Run with "
            "--catalog_sky_weighting field (the default), which populates them."
        )
    if em_catalog.dN_obs_kde is not None:
        return jnp.sum(em_catalog.dN_obs_kde, axis=0)
    n_rows = em_catalog.zgals.shape[0]
    rows = jnp.arange(n_rows, dtype=jnp.int32)
    kde = vmap(_kde_dndz_obs, in_axes=(0, None, None, None))(
        rows, em_catalog.zgals, em_catalog.wgals, em_catalog.ngals
    )
    return jnp.sum(kde, axis=0)


def _aggregate_sky_norm(em_catalog: EMCatalog, n_pix_total):
    """Sky normalization of the aggregate ``Cbar`` ratio (pixel units).

    Returns ``n_pix_total`` on the legacy path (no ``f_p``), and the
    f_p-weighted covered sky ``Sum_{all p} f_p`` when a per-pixel selection
    fraction is active.  The choice is what makes the survey-total budget
    close: consumers spend ``Sum_p C_p dN_exp`` of the expectation on the
    observed side, with ``C_p = f_p Cbar`` under f_p and ``C_p = Cbar``
    without it, so the denominator must be exactly ``Sum_p (C_p / Cbar)`` for
    that to equal the observed sum ``dN_obs_sum``.

    ``f_p_total_sum`` is a data constant covering the WHOLE sky (occupied and
    empty pixels alike), not a per-view row sum: the compact PE/selection
    views carry only the union pixels, and the field normalizer's occupied
    rows only the occupied ones, so neither could supply it.  A run that
    populates ``f_p`` without it is refused rather than silently normalized
    by the full sphere -- that is the double-count this function exists to
    prevent, and it is invisible in every diagnostic (Cbar simply reads low
    by the covered fraction).

    Structure-only dispatch (``None``-ness), so it resolves at trace time and
    the no-``f_p`` path stays bit-identical.
    """
    has_fp = (em_catalog.f_p_rows is not None
              or em_catalog.field_f_p_occ is not None)
    if not has_fp:
        return n_pix_total
    if em_catalog.f_p_total_sum is None:
        raise ValueError(
            "c_mode='aggregate' with a per-pixel selection fraction requires "
            "EMCatalog.f_p_total_sum (Sum_{all sky p} f_p): the aggregate "
            "numerator sums observed counts over the footprint only, so "
            "normalizing it by the full sphere would make Cbar the all-sky "
            "mean <f_p> C_true and the consumers' C_p = f_p Cbar would apply "
            "the mask loss TWICE -- a fraction (1 - <f_p>) of the catalogued "
            "galaxies would stay in the missing budget on top of their own "
            "observed counts. Build it from the full-sky f_p map "
            "(darksirens.likelihood.catalog_views.prepare_catalog_views)."
        )
    f_tot = jnp.asarray(em_catalog.f_p_total_sum, dtype=zgrid.dtype)
    # A completely uncovered map (Sum f_p == 0) has no observed counts either,
    # so the ratio is 0/0; the inert 1 keeps Cbar at exactly 0 there instead
    # of NaN, matching the guarded dN_exp_smooth denominator alongside it.
    return jnp.where(f_tot > 0.0, f_tot, 1.0)


def _check_stratum_labels(em_catalog: EMCatalog, n_strata: int) -> None:
    """Bound-check the stratum labels against the curve stack (host-side only).

    A JAX out-of-bounds gather CLAMPS instead of raising, so a pixel labelled
    ``s >= S`` would silently carry stratum ``S-1``'s completeness curve for the
    whole run.  The production loader already refuses a map whose labels do not
    span exactly ``[0, S)`` (``cli/inference._resolve_selection_fits``), so this
    is the module's own belt for library/diagnostic callers -- and it can only
    run where the map is concrete: inside a trace the labels are unreadable and
    the check is skipped.
    """
    smap = em_catalog.pixel_stratum_map
    if smap is None:
        raise ValueError(
            "stratified selection (SurveyParams.selection_strata) needs "
            "EMCatalog.pixel_stratum_map to assign pixels to strata."
        )
    try:
        hi = int(np.asarray(smap).max())
        lo = int(np.asarray(smap).min())
    except (jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError):
        return
    if lo < 0 or hi >= int(n_strata):
        raise ValueError(
            f"EMCatalog.pixel_stratum_map labels span [{lo}, {hi}] but "
            f"SurveyParams.selection_strata carries {int(n_strata)} strata: "
            "every pixel needs a label in [0, S). An out-of-range label is not "
            "an error at gather time -- JAX clamps it -- so the pixel would "
            "silently be assigned another stratum's completeness curve."
        )


def _check_empty_budget_arity(shape, n_strata: int, name: str) -> None:
    """The per-stratum empty-pixel budget must have exactly S rows.

    ``V_empty = sum_s (1 - Cbar_s) * budget_s`` BROADCASTS: a budget with one
    row (a stratum map that happens to label every pixel 0, while the fit
    carries S curves) would apply that single empty-pixel count to EVERY
    stratum curve and over-count the empty-sky missing budget S-fold, with no
    error raised.  Shapes are static, so this costs nothing under jit.
    """
    if int(shape[0]) != int(n_strata):
        raise ValueError(
            f"EMCatalog.{name} has {int(shape[0])} stratum rows but "
            f"SurveyParams.selection_strata carries {int(n_strata)}: the "
            "global normalizer sums (1 - Cbar_s) * budget_s over strata, so a "
            "mismatched arity silently broadcasts one stratum's empty-pixel "
            "budget onto every curve. Rebuild the per-stratum budgets from the "
            "same stratum map the fit was defined on."
        )


def _precompute_grids(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> _CompletionGrids:
    """Pixel-independent completion grids for one (cosmo, survey) proposal.

    The completeness denominator must see the SAME support as the numerator:
    the observed-galaxy KDE only ever receives sources below the survey depth
    (nothing past ``z_depth`` is catalogued), so with a concrete ``z_depth``
    the expected side is truncated at the depth BEFORE smoothing,

        dN_exp_smooth = S @ (dN_exp · 1[z <= z_depth]).

    Without the truncation the denominator keeps mass above the depth that the
    numerator can never match, so C dips within ~2 sigma_smooth below the edge
    and ``(1 - C) dN_exp`` spikes there (the closure experiment measured
    exactly this artifact).  With it, a constant true completeness passes
    through the ratio exactly up to the edge — the depth counterpart of the
    z = 0 boundary treatment already built into the operator.  This is the
    ONLY site that forms ``dN_exp_smooth``; the field-mode recompute
    (``_field_missing_curve``) and the offline Q builder consume these grids,
    so they inherit the same truncated denominator.

    ``z_depth`` is a concrete Python float (or ``None``) at trace time — never
    a sampled/traced value — so this is a Python-level branch resolved once
    per trace; the ``None`` path builds no mask and takes the ORIGINAL
    expression untouched (bit-identical legacy behaviour, pinned by
    tests/test_completion_depth.py::test_z_depth_none_bit_identical).

    ``survey.c_mode`` (structural at trace time -- ``None``/0 legacy
    per-pixel, the ``C_MODE_AGGREGATE_STRUCT`` sentinel / eager 1 aggregate;
    see :func:`_is_aggregate_c_mode` for how the flag survives nested jit
    boundaries) selects the completeness estimator.  In
    AGGREGATE mode this is the single formation site of the sky-aggregate
    ratio

        Cbar(z) = Sum_p dN_obs_s(z|p) / (A_sky * dN_exp_smooth(z)),
        A_sky   = round(4 pi / apix)   (occupied AND empty pixels),

    or, when a per-pixel selection fraction is active, ``A_sky = Sum_p f_p``
    (the f_p-weighted covered sky) so that Cbar is the TRUE per-covered-sky
    completeness and the consumers' ``C_p = f_p Cbar`` carries the mask loss
    exactly once -- see :func:`_aggregate_sky_norm`.  Stored UNCLIPPED on the
    grids bundle (consumers clip to [0, 1]); every C consumer (the per-row
    curves, the field normalizer, the clip diagnostics)
    then broadcasts this one curve in place of the per-pixel ratio, so the
    numerator and the global normalizer carry the same budget by
    construction.  The numerator is a data constant; the theta dependence is
    the shared denominator, so Cbar is proportional to 1/n0 through ONE
    global clip -- the likelihood can be kinked in n0 near full completeness
    (module docstring caveat; flagged for the inference stage, not addressed
    here).  ``c_mode == C_MODE_PER_PIXEL`` (the default) leaves the bundle's
    ``C_bar_raw`` at ``None`` and every downstream path bit-identical.
    """
    log_g = log_galaxy_measure_grid(cosmo, survey)
    dN_exp = survey.n0 * em_catalog.apix * jnp.exp(log_g)
    if survey.z_depth is None:
        dN_exp_smooth = smoothing_operator() @ dN_exp
    else:
        depth_mask = zgrid <= survey.z_depth
        dN_exp_smooth = smoothing_operator() @ jnp.where(depth_mask, dN_exp, 0.0)
        # Beyond the depth the ratio estimator is never data: every consumer
        # discards C there (the depth prior relaxes to C := 0, lss := 1).  But
        # the truncated smooth UNDERFLOWS to ~0 a few sigma past the edge, and
        # a live ``obs / tiny`` division there feeds inf local Jacobians into
        # the zero cotangents of the downstream clip/where -- 0 * inf = NaN in
        # reverse mode (the standard JAX where-trap; the zero cotangent does
        # not sanitise the discarded branch).  Pin the denominator to an inert
        # 1 beyond the depth: the discarded primal stays finite and its
        # gradient is exactly zero (data numerator / constant denominator).
        dN_exp_smooth = jnp.where(depth_mask, dN_exp_smooth, 1.0)

    C_bar_raw = None
    if _is_selection_c_mode(survey.c_mode):
        # Parametric magnitude-limited selection: no counts enter the budget
        # at all.  The h-scaled magnitude center (M0hat / Mstar_hat) means the
        # +5 log10 h restored inside the curve cancels the -5 log10 h inside
        # DM(z) EXACTLY (dL is exactly proportional to 1/H0), so the curve
        # carries no H0 information -- the M*-H0 firewall, pinned by
        # tests/test_completion_selection_mode.py and
        # tests/test_selection_schechter.py.  Both families return values in
        # [0, 1], so the consumers' clip is a no-op, and beyond a concrete
        # z_depth they relax C := 0 exactly as in the other modes.  Formed
        # here, in the single C_bar_raw slot, so every consumer (_row_C,
        # _field_missing_curve, diagnostics) works unchanged.  Deferred import:
        # selection.py is a leaf module (utils.cosmology only), imported here
        # rather than at module top to keep completion's import graph flat.
        family = _decode_selection_family(survey.selection_family)
        if family == "schechter":
            # C_sel(z) = Gamma(a, x_lim(z)) / Gamma(a, x_faint), a = alpha + 1,
            # faint cutoff M_faint = Mstar_hat + M_faint_offset (a magnitude
            # DIFFERENCE, so it carries no h either).
            if survey.selection_strata is not None:
                raise NotImplementedError(
                    "stratified selection is gaussian-only: the per-stratum "
                    "curve stack is built from the (dM0hat, sigma_ratio) "
                    "common-mode decomposition, which has no schechter "
                    "counterpart. Fit and run the strata separately, or use "
                    "the gaussian family.")
            if survey.k_corr_coeffs:
                raise NotImplementedError(
                    "the schechter selection curve carries no K(z) template "
                    f"(got k_corr_coeffs={tuple(survey.k_corr_coeffs)}): "
                    "c_sel_schechter would silently DROP it, so the run would "
                    "apply a different selection model than the fit. Use the "
                    "gaussian family for K-corrected surveys.")
            from darksirens.redshift.selection import c_sel_schechter
            C_bar_raw = c_sel_schechter(
                zgrid, survey.m_lim, survey.Mstar_hat, survey.alpha,
                survey.M_faint_offset, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        else:
            from darksirens.redshift.selection import c_sel_gaussian
            # k_corr_coeffs is STRUCTURAL (hashable tuple or None, never
            # traced), so the K(z) branch inside c_sel_gaussian is a
            # Python-level decision and the None path stays bit-identical to
            # the pre-K behaviour.
            if survey.selection_strata is not None:
                # Stratified selection: one curve per stratum, stacked
                # (S, N_grid).  The SAMPLED (M0hat, sigma_M) are the common
                # mode; each stratum's STRUCTURAL (m_lim_s, dM0hat_s,
                # sigma_ratio_s) offsets are fixed data from the offline fit
                # (see SurveyParams.selection_strata).  S is small (2-4) and
                # static, so a Python loop traces S curves.
                C_bar_raw = jnp.stack([
                    c_sel_gaussian(
                        zgrid, m_lim_s, survey.M0hat + dm0_s,
                        survey.sigma_M * sig_ratio_s,
                        cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa,
                        k_corr_coeffs=survey.k_corr_coeffs)
                    for (m_lim_s, dm0_s, sig_ratio_s) in survey.selection_strata
                ])
                _check_stratum_labels(em_catalog, len(survey.selection_strata))
            else:
                C_bar_raw = c_sel_gaussian(
                    zgrid, survey.m_lim, survey.M0hat, survey.sigma_M,
                    cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa,
                    k_corr_coeffs=survey.k_corr_coeffs)
    elif _is_aggregate_c_mode(survey.c_mode):
        dN_obs_sum = _aggregate_dN_obs_sum(em_catalog)      # data constant
        n_pix_total = jnp.round(4.0 * jnp.pi / em_catalog.apix)
        # Same guarded denominator as _row_C (an exact-zero smooth is inert);
        # beyond a concrete z_depth the smooth is already pinned to 1 above,
        # and every consumer discards Cbar there anyway (C := 0 relax).
        dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
        # Sky normalization of the aggregate ratio.  The numerator sums the
        # observed density over the FOOTPRINT only (empty/uncovered pixels
        # contribute exactly zero), so the denominator must count the sky the
        # survey could actually observe.  Without f_p that is the whole
        # sphere, because C is then applied unweighted to every pixel.  With
        # f_p it is the f_p-weighted covered sky Sum_p f_p: consumers form
        # C_p = f_p * Cbar, so normalizing by n_pix_total instead would make
        # Cbar = <f_p>_allsky * C_true and C_p = f_p <f_p> C_true -- the mask
        # loss applied twice, leaving (1 - <f_p>) of the catalogued galaxies
        # in the missing budget alongside their own observed counts.
        sky_norm = _aggregate_sky_norm(em_catalog, n_pix_total)
        C_bar_raw = dN_obs_sum / (sky_norm * dN_exp_safe)
    return _CompletionGrids(
        log_g=log_g, dN_exp=dN_exp, dN_exp_smooth=dN_exp_smooth,
        C_bar_raw=C_bar_raw,
        lss_floor_norm=legacy_lss_floor_normalizer(survey, em_catalog),
    )


# ------------------------------------------------------------
# Per-row completion curves (the unit the state API vmaps over)
# ------------------------------------------------------------

class CompletionCurves(NamedTuple):
    """Per-catalog-row completion outputs on ``zgrid``.

    ``base_miss`` / ``N_miss_members`` are populated only when the catalog
    carries an LSS-completion *ensemble* (``lss_completion_*_members``); they
    support the Bayesian redshift-prior diagnostic and the ``lss_marginalize``
    member vmap, and default to ``None`` (the scalar API and the deterministic GW
    likelihood ignore them).

    ``base_miss`` is the member-INDEPENDENT missing-density base ``(1 - C)
    dN_exp`` (with the ``z_depth`` relaxation to ``dN_exp`` beyond the depth
    already applied), so member ``m``'s missing density is exactly
    ``base_miss * Q_eff_m`` with ``Q_eff_m == 1`` beyond the depth -- the full
    ``(M, N_rows, N_grid)`` member cube is NEVER materialised; it is reconstructed
    two grid nodes at a time at query redshifts (see
    :func:`darksirens.redshift.prior.eval_dark_member_completion`).
    ``N_miss_members`` is the grid-integrated per-member missing mass ``(M,
    N_rows)`` -- a compact scalar table computed by the streamed
    :func:`member_N_miss_integrals` (no cube).
    """
    f: jnp.ndarray        # (N_rows,) scalar number-weighted completeness (diagnostic)
    dN_miss: jnp.ndarray  # (N_rows, N_grid) missing-galaxy density [counts / unit z]
    C_eff: jnp.ndarray    # (N_rows, N_grid) effective completeness (diagnostic)
    N_miss: jnp.ndarray   # (N_rows,) integral of dN_miss
    base_miss: jnp.ndarray = None        # (N_rows, N_grid) member-independent base | None
    N_miss_members: jnp.ndarray = None   # (M, N_rows) | None
    # LATENT mode only (field-level PR-5): the per-member budget normalizer
    # ``rho_m(z; C, b_GW)`` of PLAN eq. (2), ``(M, N_grid)``.  It is returned
    # from HERE -- rather than recomputed by the likelihood -- so the
    # completeness curve that enters ``rho`` is the SAME object, from the same
    # ``_precompute_grids`` call, that ``base_miss`` was built from.  PLAN §4.2
    # is explicit that the numerator and the normalizer must carry one budget
    # from one formation site; rebuilding the grids at the seam would create a
    # second one, and the two would diverge under any change that touched C.
    latent_rho: jnp.ndarray = None       # (M, N_grid) | None


# ------------------------------------------------------------
# LSS-conditioned lognormal completion factor Q_LSS (optional)
# ------------------------------------------------------------

#: log Q is clipped to this symmetric range before exponentiating, so that a
#: heavy lognormal tail cannot blow up the missing-galaxy density.  TABLE path
#: only: the shipped Q_LSS tables are empirically fit and are NOT bounded by
#: their builder's own clip (it is applied before the per-z mean-one renorm),
#: so this bound is what keeps the numerator's and the normalizer's missing
#: budgets equal on a railed cell.
_LOGQ_CLIP: float = 7.0

#: The LATENT seam's bound on ``logQ`` -- a numerical-safety rail, NOT a tail
#: tamer, and it must never bind on a physical row.  The latent field is not an
#: empirically fit table: ``rho`` is DEFINED so that the consumed missing budget
#: ``sum_p (1 - f_p C(z)) Q_p(z)`` equals its ``Q == 1`` value exactly (PLAN
#: eq. 4).  That identity BOUNDS the positive tail on its own --
#: ``Q_p <= (sum_q w_q) / w_p <= P_F / min_p w_p``, i.e. ``logQ <= 13.4`` at the
#: DESI footprint (``P_F = 30470``, ``min w = 1 - max f_p = 0.044``) -- so a
#: ``+/-7`` rail does not bound a tail there, it DELETES budget, and the deficit
#: grows with ``b_GW``.  ``60`` sits ~4x above that structural bound while
#: ``exp(60) = 1.1e26`` stays ~280 orders under f64 overflow (and 12 under f32),
#: so the product with the missing-density base cannot become ``inf``.  The
#: negative rail is inert by the same identity: a floored ``Q = e^-60``
#: contributes ``< 1e-26`` of the budget, against ``e^-7 = 9e-4`` today.
_LATENT_LOGQ_CLIP: float = 60.0


def _check_lss_grid(arr):
    """Static ``N_grid == zgrid.size`` check for a Q_LSS table (no interpolation)."""
    if arr.shape[-1] != int(zgrid.size):
        raise ValueError(
            f"LSS completion table has N_grid={arr.shape[-1]} but the package "
            f"zgrid has size {int(zgrid.size)}; the offline builder must use the "
            "same redshift grid (no silent interpolation)."
        )


def _row_align_lss(table, row_axis, em_catalog: EMCatalog, n_rows: int):
    """Align a Q_LSS table's ``row_axis`` to catalog row order.

    Compact-vs-global is decided by STATIC shapes only (``K`` and ``n_rows`` are
    concrete even under a trace, so nothing here concretises a tracer).  In the
    jitted likelihood the table is ALWAYS already compact -- the factory slices
    global->compact before the jit -- so ``K == n_rows`` and it is returned
    untouched (the hot path; the returned object is a VIEW of the resident data
    constant, so no ``(M, N_rows, N_grid)`` copy is made).  ``K != n_rows`` is
    reachable ONLY on the eager path (a full global table handed straight to
    ``completion_curves`` by a test or diagnostic), never under jit, so
    ``unique_pixels`` is concrete here and the coverage check is safe.
    """
    K = table.shape[row_axis]
    if K == n_rows:
        return table
    if em_catalog.unique_pixels is None:
        idx = jnp.arange(n_rows, dtype=jnp.int32)
    else:
        idx = jnp.asarray(em_catalog.unique_pixels, dtype=jnp.int32)
    if int(idx.max()) >= K:
        raise ValueError(
            f"global LSS completion table has {K} rows but a catalog pixel index "
            f"reaches {int(idx.max())}; the table does not cover all catalog pixels "
            "(rebuild Q over the full nside, or pass a compact table)."
        )
    return jnp.take(table, idx, axis=row_axis)


def _to_q(logq_arr, q_arr):
    """Clip-and-exp a Q_LSS table: ``exp(clip(logQ, ±_LOGQ_CLIP))``.

    Accepts either a log table (``logq_arr``) or a linear table (``q_arr``, then
    log it first).  Used for the (small, ``(N_rows, N_grid)``) DETERMINISTIC Q
    only -- the member ensemble defers this to the 2-node gathers / streamed chunk
    so no ``(M, N_rows, N_grid)`` intermediate is built.
    """
    if logq_arr is not None:
        lq = jnp.clip(jnp.asarray(logq_arr, dtype=float), -_LOGQ_CLIP, _LOGQ_CLIP)
    elif q_arr is not None:
        lq = jnp.clip(
            jnp.log(jnp.maximum(jnp.asarray(q_arr, dtype=float), 1e-300)),
            -_LOGQ_CLIP, _LOGQ_CLIP,
        )
    else:
        return None
    return jnp.exp(lq)


def _member_q_eff_from_logq(logq_arr, depth_mask, member_is_log: bool,
                            clip: float = _LOGQ_CLIP):
    """One member's completion factor ``Q_eff = exp(clip(logQ, ±clip))``,
    relaxed to ``1`` beyond ``survey.z_depth``.

    ``logq_arr`` is a RAW (un-clipped, un-exponentiated) log table when
    ``member_is_log`` else a linear-Q table (then log it first).  Bit-identical at
    grid nodes to the pre-existing whole-table ``_to_q`` followed by the
    ``_completion_member_row`` depth relaxation, but applied to whatever slice /
    2-node gather the caller passes (never the full cube).  ``depth_mask`` is a
    static ``(N_grid,)`` (or gathered ``(...,)``) boolean, or ``None`` when
    ``z_depth is None`` (Q_eff is then just ``exp(clip(logQ))`` everywhere).

    ``clip`` is a STATIC Python float and defaults to the table path's
    ``_LOGQ_CLIP``.  The latent seam passes ``_LATENT_LOGQ_CLIP`` instead: its
    ``logQ`` is generated against a normalizer that already conserves the
    consumed budget, so the table bound would truncate that budget rather than
    tame a tail (see the two constants).  Keying this off ``member_is_log``
    would be wrong -- a LOADED log-Q table is ``member_is_log=True`` too.
    """
    if not member_is_log:
        logq_arr = jnp.log(jnp.maximum(logq_arr, 1e-300))
    q = jnp.exp(jnp.clip(logq_arr, -float(clip), float(clip)))
    if depth_mask is None:
        return q
    return jnp.where(depth_mask, q, 1.0)


def _resolve_member_logq_row(em_catalog: EMCatalog):
    """Row-aligned RAW member log-Q table ``(M, N_rows, N_grid)`` + ``is_log`` flag.

    Returns the ensemble table WITHOUT clipping or exponentiating -- both are
    deferred to the 2-node gathers in the evaluator and to the per-member chunk in
    :func:`member_N_miss_integrals` -- so no ``(M, N_rows, N_grid)`` intermediate
    is ever built on the hot jit path (there the table is already compact and
    row-aligned, ``K == N_rows``, and is returned untouched, a view of the
    resident data constant).  ``member_is_log`` is ``True`` for a log-Q table and
    ``False`` for a linear-Q table (then the consumer takes ``log`` at the gathered
    nodes).  Returns ``(None, True)`` when the catalog carries no ensemble.
    """
    logq_m = em_catalog.lss_completion_logq_members
    q_m = em_catalog.lss_completion_q_members
    if logq_m is None and q_m is None:
        return None, True
    n_rows = em_catalog.zgals.shape[0]  # static int (shape) -- safe under trace
    if logq_m is not None:
        tab, is_log = jnp.asarray(logq_m, dtype=float), True
    else:
        tab, is_log = jnp.asarray(q_m, dtype=float), False
    _check_lss_grid(tab)
    return _row_align_lss(tab, 1, em_catalog, n_rows), is_log


def _member_posterior_mean_q(logq_members_row, member_is_log: bool):
    """Posterior-MEAN Q ``(N_rows, N_grid)`` = ``mean_m exp(clip(logQ_m))``.

    Streamed with ``lax.scan`` over the M members so the ``(M, N_rows, N_grid)``
    exponentiated cube is never materialised; bit-identical at grid nodes to the
    old ``jnp.mean(exp(clip(logQ_members)), axis=member)``.  Q is theta-INDEPENDENT
    (it is loaded data), so this carries no reverse-mode cost.  No depth relaxation
    here -- the deterministic path applies ``z_depth`` downstream in
    ``_assemble_curves`` exactly as before.
    """
    M = logq_members_row.shape[0]

    def _body(acc, logq_m):
        return acc + _member_q_eff_from_logq(logq_m, None, member_is_log), None

    total, _ = lax.scan(
        _body, jnp.zeros(logq_members_row.shape[1:], dtype=float), logq_members_row
    )
    return total / M


def member_N_miss_integrals(base_miss, em_catalog: EMCatalog, survey: SurveyParams):
    """Streamed per-member missing-mass integrals ``N_miss_members`` ``(M, N_rows)``.

    ``N_miss[m, row] = integral base_miss[row] * Q_eff_m[row] dz`` on ``zgrid``,
    computed WITHOUT ever materialising the ``(M, N_rows, N_grid)`` member density
    cube: a ``lax.scan`` over the M ensemble members forms ONE member's ``(N_rows,
    N_grid)`` density from the SHARED ``base_miss`` and that member's row-aligned
    log-Q, integrates it to ``(N_rows,)``, and discards it.  The scan body is
    ``jax.checkpoint``-wrapped so the reverse pass REMATERIALISES each member's
    density instead of stacking ``(M, N_rows, N_grid)`` residuals -- keeping BOTH
    the forward and backward peak at ``O(N_rows x N_grid) + O(M x N_rows)``, the
    crux that lets ``--lss_marginalize`` scale.  ``base_miss`` already carries the
    ``survey.z_depth`` relaxation (``== dN_exp`` beyond the depth) and ``Q_eff`` is
    forced to ``1`` there, so the integrand is bit-identical at grid nodes to the
    pre-existing ``_completion_member_row`` cube.
    """
    logq_members_row, member_is_log = _resolve_member_logq_row(em_catalog)
    depth_mask = None if survey.z_depth is None else (zgrid <= survey.z_depth)

    def _body(carry, logq_m):                 # logq_m: (N_rows, N_grid)
        q_eff = _member_q_eff_from_logq(logq_m, depth_mask, member_is_log)
        dN_m = base_miss * q_eff              # (N_rows, N_grid) -- transient, discarded
        return carry, jnp.trapezoid(dN_m, zgrid, axis=-1)  # (N_rows,)

    _, N = lax.scan(jax.checkpoint(_body), None, logq_members_row)  # (M, N_rows)
    return N


# ------------------------------------------------------------
# LATENT completion: Q is GENERATED, not loaded (field-level PR-5)
# ------------------------------------------------------------
#
# Every consumer below already streams the ensemble ONE member at a time
# (``member_N_miss_integrals``, ``_member_posterior_mean_q``,
# ``field_global_log_Z_members``), precisely so the ``(M, N_rows, N_grid)``
# cube is never materialised.  Latent mode changes only what the scan body
# READS: instead of gathering member ``m``'s slice of a resident table it
# generates that member's rows from the compact ``(M_draw, n_fit + 1, M_z)``
# row-factor leaf.  The memory profile is therefore unchanged -- one
# ``(N_rows, N_grid)`` transient, formed and discarded -- and at DESI scale the
# table those scans would otherwise need does not exist (1.06 GB at
# ``M_draw = 8``, and it is theta-dependent, so it could not be a data constant
# even if it fit).


def latent_enabled(em_catalog: EMCatalog) -> bool:
    """Static (pytree-STRUCTURE) test for latent mode."""
    return em_catalog.latent_row_fac is not None


def latent_n_members(em_catalog: EMCatalog) -> int:
    return int(em_catalog.latent_row_fac.shape[0])


def latent_b_gw(survey: SurveyParams):
    """``b_GW`` in latent mode.

    PLAN §4.3 inverts the ``b_miss`` guard: in table mode ``b_miss`` scales the
    local-overdensity factor through ``alpha_miss * b_miss`` and is not
    identified; in latent mode it IS ``b_GW``, the bias with which GW hosts
    trace the field, and it is genuinely (if weakly) identified by the events.
    ``alpha_miss`` does not enter -- it is the delta_g path's calibration and
    has no referent here.
    """
    return survey.b_miss


def _latent_C_curve(grids: "_CompletionGrids"):
    """The SCALAR sky-aggregate completeness ``C(z; theta)`` the moments assume.

    PLAN eq. (2) factors the budget normalizer through two ``z``-curves,
    ``A(z; b) = sum_p e^{bf}`` and ``B(z; b) = sum_p f_p e^{bf}``, which closes
    the identity in closed form only if the completeness enters as ``f_p``
    times ONE sky-independent curve.  A per-pixel ``C`` (``c_mode=per_pixel``)
    or a stratified stack does not factor that way -- the sum would need its
    own moment per distinct curve -- so both are refused here rather than
    silently evaluated against the wrong normalizer.  This is PLAN §4.4
    guard 6 stated at the point where the assumption is actually used.
    """
    if grids.C_bar_raw is None:
        raise NotImplementedError(
            "lss_field_mode='latent' requires an aggregate/selection c_mode: "
            "the budget normalizer rho = log[(A - C B)/(P_F - C F_F)] assumes "
            "ONE sky-aggregate completeness curve, and a per-pixel C does not "
            "factor through the sky moments (PLAN §4.4 guard 6). "
            "Use --c-mode aggregate or selection with --per_pixel_completeness."
        )
    if grids.C_bar_raw.ndim == 2:
        raise NotImplementedError(
            "lss_field_mode='latent' does not support stratified selection: "
            "each stratum's C(z) would need its own (A, B) moment pair, and "
            "the shipped anchor artifact carries one. Refused rather than "
            "evaluated against the wrong stratum's normalizer."
        )
    return jnp.clip(grids.C_bar_raw, 0.0, 1.0)


def latent_support_mask(em_catalog: EMCatalog, survey: SurveyParams):
    """The z nodes at which the seam MODELS the field -- PR-8's one accessor.

    ``None`` (no ``latent_support`` leaf) is the pre-PR-8 contract: the field
    lives at ``z <= survey.z_depth`` and nowhere else, so the mask is the depth
    mask and callers that relax ``Q := 1`` above the depth are relaxing it
    exactly where ``logQ`` is already bit-zero.  With an ``amp(z)`` anchor the
    leaf is present, the support reaches above the depth, and BOTH facts have
    to travel together: ``rho`` must be applied wherever the field is nonzero
    (else the seam injects an un-normalized monopole above the depth, breaking
    PLAN eq. (4) exactly where 99.99% of the budget lives), and the downstream
    depth relaxation must NOT fire there (else the assumed modulation is
    discarded and the scan measures nothing).
    """
    if em_catalog.latent_support is not None:
        return jnp.asarray(em_catalog.latent_support)
    return None if survey.z_depth is None else (zgrid <= survey.z_depth)


def latent_rho_member(em_catalog: EMCatalog, survey: SurveyParams, C_curve, m):
    """``rho_m(z)`` on the FULL ``zgrid`` -- PLAN eq. (2), zeroed off support.

    ``C_curve`` is the SCALAR sky-aggregate completeness ``C(z; theta)`` (the
    ``f_p`` structure is already inside ``B``).  ``O(n_b + N_grid)`` per member
    per proposal -- the entire per-proposal cost of the latent normalizer.

    PR-8: when the support reaches ABOVE the depth, the ``C`` that enters here
    must be the one the consumption actually uses, and above the depth that is
    ``C := 0`` -- ``base_miss`` relaxes to ``dN_exp`` and ``_field_missing_curve``
    to the total pixel count there.  Normalizing against the raw ``c_sel``
    curve instead would conserve a budget nothing consumes, and eq. (4) would
    close on paper while the likelihood integrated something else.  Below the
    depth the two are the same curve, so the pre-PR-8 branch is untouched.
    """
    from darksirens.likelihood.latent_q import rho_from_moments

    below = latent_support_mask(em_catalog, survey)
    if em_catalog.latent_support is not None and survey.z_depth is not None:
        C_curve = jnp.where(zgrid <= survey.z_depth, C_curve, 0.0)
    return rho_from_moments(
        em_catalog.latent_A[m], em_catalog.latent_B[m], C_curve,
        latent_b_gw(survey), em_catalog.latent_b_nodes,
        em_catalog.latent_P_F, em_catalog.latent_F_F, below)


def latent_member_logq_rows(em_catalog: EMCatalog, survey: SurveyParams,
                            C_curve, m, *, field_rows: bool = False):
    """One member's ``(N_rows, N_grid)`` log-Q block, generated.

    ``field_rows`` selects the row set: catalog rows (the numerator) or the
    ``field_dN_obs_s`` occupied rows (the survey-global normalizer).  Both
    carry their own footprint map, because they are different row orders over
    different pixel sets.
    """
    row_map = (em_catalog.latent_field_row_map if field_rows
               else em_catalog.latent_row_map)
    on_fp = (em_catalog.latent_field_on_fp if field_rows
             else em_catalog.latent_on_fp)
    rho = latent_rho_member(em_catalog, survey, C_curve, m)
    phi_z = jnp.asarray(em_catalog.latent_phi_z)
    rows = jnp.asarray(em_catalog.latent_row_fac)[m][jnp.asarray(row_map)]
    field = rows.astype(phi_z.dtype) @ phi_z.T          # (N_rows, N_grid)
    lq = jnp.asarray(latent_b_gw(survey), dtype=field.dtype) * field - rho[None, :]
    return jnp.where(jnp.asarray(on_fp)[:, None], lq, 0.0)


def latent_member_N_miss_integrals(base_miss, C_curve, em_catalog: EMCatalog,
                                   survey: SurveyParams):
    """Latent twin of :func:`member_N_miss_integrals` -- same scan, same peak.

    ``jax.checkpoint`` on the body for the same reason: the reverse pass
    rematerialises each member's block instead of stacking ``(M, N_rows,
    N_grid)`` residuals.  Unlike the table path this integrand IS
    theta-dependent (through ``C_curve`` and ``b_GW``), so the checkpoint is
    load-bearing here rather than merely tidy.
    """
    # The relaxation mask is the SEAM'S SUPPORT (PR-8), which is the depth mask
    # on every pre-PR-8 anchor -- same expression, same array -- and reaches
    # above the depth on an amp(z) one, where relaxing ``Q := 1`` would throw
    # away the assumed modulation this integral exists to carry.
    depth_mask = latent_support_mask(em_catalog, survey)

    def _body(carry, m):
        lq = latent_member_logq_rows(em_catalog, survey, C_curve, m)
        q_eff = _member_q_eff_from_logq(lq, depth_mask, True,
                                        _LATENT_LOGQ_CLIP)
        dN_m = base_miss * q_eff                        # transient, discarded
        return carry, jnp.trapezoid(dN_m, zgrid, axis=-1)

    _, N = lax.scan(jax.checkpoint(_body), None,
                    jnp.arange(latent_n_members(em_catalog)))
    return N                                            # (M, N_rows)


def latent_posterior_mean_q(em_catalog: EMCatalog, survey: SurveyParams,
                            C_curve, *, field_rows: bool = False):
    """Posterior-MEAN ``Q`` ``(N_rows, N_grid)`` = ``mean_m exp(clip(logQ_m))``.

    The latent twin of :func:`_member_posterior_mean_q`, feeding the SCALAR
    (deterministic-Q) path so that a latent run without ``--lss_marginalize``
    still evaluates a well-defined prior.  Streamed, no cube.

    Note this is the mean of ``Q``, not ``Q`` of the mean field -- the two
    differ, and PLAN §4.2 (``rem:noncommute``) is explicit that renormalising
    each member does not commute with renormalising the mean.  The member
    average is the estimand; this scalar path is the fallback.
    """
    M = latent_n_members(em_catalog)
    depth_mask = latent_support_mask(em_catalog, survey)     # PR-8; see above
    n_rows = ((em_catalog.latent_field_row_map if field_rows
               else em_catalog.latent_row_map)).shape[0]

    def _body(acc, m):
        lq = latent_member_logq_rows(em_catalog, survey, C_curve, m,
                                     field_rows=field_rows)
        return acc + _member_q_eff_from_logq(
            lq, depth_mask, True, _LATENT_LOGQ_CLIP), None

    total, _ = lax.scan(
        _body, jnp.zeros((n_rows, zgrid.size), dtype=float), jnp.arange(M))
    return total / M


def _resolve_lss_completion_row_tables(em_catalog: EMCatalog):
    """Resolve the (optional) Q_LSS tables to **row order** for the vmap.

    Trace-safe consumer: this runs INSIDE the jitted likelihood (via
    :func:`completion_curves` <- ``prepare_redshift_prior_state`` <-
    ``darksiren_log_likelihood``), so every ``em_catalog`` field is a tracer and no
    concrete ``int()``/``.max()`` may touch one.  The indexing-enum decode, the
    global->compact row gather and the pixel-coverage validation are already done
    EAGERLY (host-side, before the jit) in the factory, which delivers a compact,
    row-aligned table.  Here we only consume it with static shapes.

    Returns ``(q_row, logq_members_row)``:

    * ``q_row``            — ``(N_rows, N_grid)`` DETERMINISTIC Q (already clipped
      + exponentiated), or ``None``.
    * ``logq_members_row`` — ``(M, N_rows, N_grid)`` RAW ensemble LOG-Q (row-major,
      NOT clipped/exponentiated -- deferred to the gathers / streamed chunk so no
      cube is built), or ``None``.

    Resolution: prefer ``logq`` over ``q`` for the deterministic table; clip to
    ``[-_LOGQ_CLIP, _LOGQ_CLIP]`` then exponentiate; validate ``N_grid ==
    zgrid.size``.  If ONLY an ensemble is supplied, ``q_row`` is the posterior-mean
    Q (streamed, no cube) so the scalar prior path still runs.  Q is **not**
    renormalised -- it is a physical density ratio.  Callers that need the member
    ``is_log`` flag call :func:`_resolve_member_logq_row` directly.
    """
    logq = em_catalog.lss_completion_logq
    q = em_catalog.lss_completion_q
    logq_members_row, member_is_log = _resolve_member_logq_row(em_catalog)
    if logq is None and q is None and logq_members_row is None:
        return None, None  # no Q table -> legacy local-overdensity path

    n_rows = em_catalog.zgals.shape[0]  # static int (shape) — safe under trace

    q_det = _to_q(logq, q)  # (K, N_grid) | None
    q_row = None
    if q_det is not None:
        _check_lss_grid(q_det)
        q_row = _row_align_lss(q_det, 0, em_catalog, n_rows)   # (N_rows, N_grid)
    elif logq_members_row is not None:
        # Only an ensemble supplied: the deterministic branch uses the
        # posterior-mean Q, streamed so the exponentiated member cube is not built.
        q_row = _member_posterior_mean_q(logq_members_row, member_is_log)

    return q_row, logq_members_row


def _row_C(row, grids: _CompletionGrids, em_catalog: EMCatalog):
    """Differential completeness ``C(z)`` for one catalog row (shared core).

    ``row`` is the compact catalog row index (== global HEALPix pixel for legacy
    full catalogs).  Returns ``(C, global_pix)``; depends on ``(row, Θ)`` only,
    never on a sample redshift.

    AGGREGATE mode (``grids.C_bar_raw is not None``, a static pytree-structure
    branch): the one sky-aggregate curve replaces the per-pixel ratio for
    EVERY row -- the row's own observed KDE is not consulted, so the observed
    angular clustering cannot enter the missing budget (it belongs to Q).

    Per-pixel selection fraction (field-level PR-2): with
    ``em_catalog.f_p_rows`` populated (a static pytree-structure branch;
    ``None`` is bit-identical), the row's completeness is
    ``C_p(z) = f_p * C(z)`` — survey mask-loss thinning multiplying the
    survey curve.  The loader admits ``f_p`` only for the aggregate/selection
    ``c_mode``s (a per-pixel count-derived ``C`` already contains the mask
    loss, so multiplying would double-count it).
    """
    global_pix = row if em_catalog.unique_pixels is None else em_catalog.unique_pixels[row]

    if grids.C_bar_raw is not None:
        if grids.C_bar_raw.ndim == 2:
            # Stratified selection: this pixel's stratum picks its curve.
            stratum = em_catalog.pixel_stratum_map[global_pix]
            C = jnp.clip(grids.C_bar_raw[stratum], 0.0, 1.0)
        else:
            C = jnp.clip(grids.C_bar_raw, 0.0, 1.0)
        if em_catalog.f_p_rows is not None:
            C = em_catalog.f_p_rows[row] * C
        return C, global_pix

    # --- observed density: O(1) cache lookup, or on-the-fly fallback ---
    if em_catalog.dN_obs_kde is not None:
        # The KDE cache is built row-for-row with THIS catalog's ``unique_pixels``
        # (every builder -- build_pixel_kde_cache, catalog_views, the CLI dry-run
        # -- aligns ``dN_obs_kde[k]`` with row ``k``), so the historical
        # ``pixel_to_cache_idx[global_pix]`` round-trip is exactly the identity
        # ``k -> k``.  Index the cache directly by row and skip the dense
        # ``pixel_to_cache_idx`` lookup entirely (12*nside^2 on the flat union
        # path).  Pinned by tests/test_cache_row_indexing.py.
        dN_obs = em_catalog.dN_obs_kde[row]
    else:
        dN_obs = _kde_dndz_obs(row, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals)

    # --- differential completeness: matched-kernel ratio, no roll-off ---
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
    C = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0)
    return C, global_pix


def _assemble_curves(C, lss, grids: _CompletionGrids, survey: SurveyParams):
    """Assemble the per-row completion outputs from ``C`` and the rate factor ``lss``.

    ``survey.z_depth`` encodes prior knowledge that the EM survey does not
    catalog galaxies past its own depth: beyond ``z_depth`` completeness is
    exactly zero, so *every* modeled host there is uncatalogued (missing), and
    the LSS overdensity ``delta_g`` (measured from the survey and undefined
    outside its coverage) relaxes to the mean, ``lss -> 1``.  The missing
    density therefore relaxes to the full expected count,

        dN_miss(z) = (1 - C) dN_exp lss   for z <= z_depth,
                   = dN_exp               for z >  z_depth,

    i.e. the source prior above the depth in an empty pixel is the plain
    volumetric x population shape ``dN_exp`` -- NOT zero.  ``z_depth`` is prior
    knowledge about completeness, not an analysis cutoff: ``N_miss`` is
    integrated over the FULL grid, and the diagnostics follow (``C_eff`` is 0
    beyond the depth because ``dN_miss == dN_exp`` there, and ``f`` uses the
    full-grid ``N_exp``).

    ``z_depth`` is a concrete Python float (or ``None``) at trace time --
    never a sampled/traced value -- so the branch below is a Python-level
    ``if``, resolved once per trace, not a ``jnp.where``/``lax.cond``.

    ``z_depth is None`` takes the ORIGINAL expression untouched (no mask is
    even constructed), which is the legacy full-grid budget and is
    guaranteed bit-identical to the pre-existing behaviour.
    """
    dN_miss = (1.0 - C) * grids.dN_exp * lss

    if survey.z_depth is None:
        # Legacy full-grid missing budget -- EXACTLY the pre-existing expression.
        N_miss = jnp.trapezoid(dN_miss, zgrid)

        dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
        C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
        N_exp = jnp.trapezoid(grids.dN_exp, zgrid)
        f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
        return f, dN_miss, C_eff, N_miss

    # Beyond the depth C := 0 and lss := 1, so the missing density relaxes to
    # the full expected count dN_exp (hosts there are missing, not nonexistent).
    depth_mask = zgrid <= survey.z_depth
    dN_miss = jnp.where(depth_mask, dN_miss, grids.dN_exp)
    N_miss = jnp.trapezoid(dN_miss, zgrid)

    # Identical downstream to the None branch -- the only difference is the
    # relaxed dN_miss above.  C_eff is 0 wherever dN_miss == dN_exp (beyond the
    # depth), and f uses the full-grid N_exp.
    dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(grids.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
    return f, dN_miss, C_eff, N_miss


def _completion_curves_row(
    row: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Legacy completion curves for one row (local-overdensity LSS factor).

    Used whenever no ``Q_LSS`` table is supplied — physically identical to the
    pre-existing behaviour: ``lss = max(1 + alpha_miss*b_miss*delta_g(pix,z), 0)``.
    """
    C, global_pix = _row_C(row, grids, em_catalog)
    delta_g_pix_z = em_catalog.delta_g_pix_z
    b_eff = survey.alpha_miss * survey.b_miss
    if delta_g_pix_z.shape[0] == 1:        # static shape -> trace-time branch
        delta_g_z = delta_g_pix_z[0]
    else:
        delta_g_z = delta_g_pix_z[global_pix]
    lss = jnp.maximum(1.0 + b_eff * delta_g_z, 0.0)
    # Number conservation: the floor breaks delta_g's mean-one identity and
    # INFLATES the missing budget (see legacy_lss_floor_normalizer).  The
    # normalizer is a static-pytree branch and is exactly 1.0 wherever the
    # floor never engaged, so the division is bit-inert there.
    if grids.lss_floor_norm is not None:
        lss = lss / grids.lss_floor_norm
    return _assemble_curves(C, lss, grids, survey)


def _completion_curves_row_q(
    row: int,
    q_row: jnp.ndarray,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Completion curves for one row using a precomputed Q_LSS factor.

    ``q_row`` (``(N_grid,)``, already row-aligned) **replaces** the legacy
    local-overdensity factor: ``dN_miss = (1 - C) dN_exp Q_LSS``.
    """
    C, _ = _row_C(row, grids, em_catalog)
    return _assemble_curves(C, q_row, grids, survey)


def _base_miss_row(
    row: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Member-INDEPENDENT missing-density base for one row: ``(1 - C) dN_exp``.

    Mirrors ``_assemble_curves``'s ``dN_miss`` with the LSS rate factor set to 1
    (Q_LSS is applied per member as ``base_miss * Q_eff_m``).  The
    ``survey.z_depth`` relaxation is baked in EXACTLY as in ``_assemble_curves``
    (Python-level branch on the concrete ``z_depth``; ``None`` builds no mask):
    beyond the depth C := 0 and lss := 1, so the base relaxes to the full expected
    count ``dN_exp`` -- and because ``Q_eff`` is forced to 1 there too, every
    member's density is ``dN_exp`` beyond the depth, matching the pre-existing
    per-member cube node-for-node.
    """
    C, _ = _row_C(row, grids, em_catalog)
    base = (1.0 - C) * grids.dN_exp
    if survey.z_depth is not None:
        depth_mask = zgrid <= survey.z_depth
        base = jnp.where(depth_mask, base, grids.dN_exp)
    return base


_completion_curves_rows_vmap = vmap(
    _completion_curves_row, in_axes=(0, None, None, None), out_axes=(0, 0, 0, 0)
)
_completion_curves_rows_q_vmap = vmap(
    _completion_curves_row_q, in_axes=(0, 0, None, None, None), out_axes=(0, 0, 0, 0)
)
_base_miss_rows_vmap = vmap(
    _base_miss_row, in_axes=(0, None, None, None), out_axes=0
)


def completion_curves(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> CompletionCurves:
    """All per-row completion curves for one parameter proposal.

    Eager (host-side); called once per proposal by
    :func:`darksirens.redshift.prior.prepare_redshift_prior_state`.  When the catalog
    carries an LSS-conditioned lognormal completion table the per-row factor is
    ``Q_LSS`` (replacing the legacy ``max(1 + b_eff delta_g, 0)``); with an
    ensemble it additionally returns the member-INDEPENDENT ``base_miss`` curve and
    the compact per-member ``N_miss_members`` scalar table (both cube-free) so the
    member marginalisation / diagnostic can reconstruct each member's density on
    the fly (``base_miss * Q_eff_m``) without ever storing ``(M, N_rows, N_grid)``.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    n_rows = em_catalog.zgals.shape[0]
    rows = jnp.arange(n_rows, dtype=jnp.int32)

    q_row, logq_members_row = _resolve_lss_completion_row_tables(em_catalog)

    # LATENT mode: Q is generated from the anchor artifact rather than loaded.
    # Static pytree-structure branch; ``latent_row_fac is None`` leaves every
    # line below textually and numerically as it was.
    latent = latent_enabled(em_catalog)
    C_curve = None
    if latent:
        if q_row is not None or logq_members_row is not None:
            raise NotImplementedError(
                "lss_field_mode='latent' generates Q from the latent field, so "
                "a loaded Q_LSS table (--lss_completion / lss_completion_q*) is "
                "refused: the two are different completion models and silently "
                "preferring one would make the run's estimand unreadable "
                "(PLAN §4.4 guard 6)."
            )
        C_curve = _latent_C_curve(grids)
        q_row = latent_posterior_mean_q(em_catalog, survey, C_curve)

    if q_row is None:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_vmap(rows, grids, survey, em_catalog)
    else:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_q_vmap(
            rows, q_row, grids, survey, em_catalog
        )

    base_miss = None
    N_miss_members = None
    latent_rho = None
    if latent:
        base_miss = _base_miss_rows_vmap(rows, grids, survey, em_catalog)
        N_miss_members = latent_member_N_miss_integrals(
            base_miss, C_curve, em_catalog, survey)                         # (M, N_rows)
        latent_rho = jnp.stack([
            latent_rho_member(em_catalog, survey, C_curve, m)
            for m in range(latent_n_members(em_catalog))
        ])                                                                  # (M, N_grid)
    elif logq_members_row is not None:
        base_miss = _base_miss_rows_vmap(rows, grids, survey, em_catalog)   # (N_rows, N_grid)
        N_miss_members = member_N_miss_integrals(base_miss, em_catalog, survey)  # (M, N_rows)

    return CompletionCurves(
        f=f, dN_miss=dN_miss, C_eff=C_eff, N_miss=N_miss,
        base_miss=base_miss, N_miss_members=N_miss_members,
        latent_rho=latent_rho,
    )


# ------------------------------------------------------------
# FIELD-convention global normalizer Z(theta)
# ------------------------------------------------------------

#: Galaxy-axis chunk for the flat below-depth mass reduction: (chunk, N_GL_nodes)
#: intermediates, so 32k galaxies is ~6 MB at 24 nodes / f64.
_FIELD_DEPTH_CHUNK: int = 1 << 15


def _field_depth_weighted_mass(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    weights: jnp.ndarray,
    chunk_size: int = _FIELD_DEPTH_CHUNK,
) -> jnp.ndarray:
    """``Sum_i weights_i * Z_i^depth / Z_i`` over the FLAT FULL-SKY galaxies.

    The survey-global companion of ``kernels.log_depth_mass``: the per-galaxy
    ratio of the below-depth to the full GL kernel norm, reusing
    :func:`darksirens.redshift.catalog._row_log_kernel_norms` verbatim (SAME nodes,
    SAME truncation, SAME ``sigma_eff`` floor) so the global observed term is
    term-for-term the full-sky sum of the per-pixel numerator's depth-scaled count.

    ``weights`` is ``(N_gal_total,)``, row-aligned with ``field_depth_z``:
    ``field_depth_c`` for the plain galaxy-count host model, ``w_i h_i`` for the
    marked one.  Differentiable in theta (cosmology + delta through ``g(z)``,
    ``sigma_kde`` through ``sigma_eff``); chunked with ``lax.scan`` so the
    ``(N_gal, N_nodes)`` quadrature intermediate never materialises whole.
    """
    # Deferred import: catalog.py imports log_galaxy_measure_grid from this module.
    from darksirens.redshift.catalog import (
        SIGMA_EFF_FLOOR,
        _row_log_kernel_norms,
    )

    z_flat = em_catalog.field_depth_z
    dz_flat = em_catalog.field_depth_dz
    if z_flat is None or dz_flat is None:
        raise ValueError(
            "catalog_sky_weighting='field' with survey.z_depth set requires the "
            "flat FULL-SKY depth inputs (field_depth_z / field_depth_dz / "
            "field_depth_c) built via "
            "darksirens.redshift.completion.build_field_depth_inputs -- without "
            "them the global normalizer would count every above-depth catalogued "
            "galaxy twice (once in N_obs_total, again in the missing budget) "
            "while the per-pixel numerator counts it once."
        )
    z_flat = jnp.asarray(z_flat, dtype=zgrid.dtype)
    dz_flat = jnp.asarray(dz_flat, dtype=zgrid.dtype)
    w_flat = jnp.asarray(weights, dtype=zgrid.dtype)
    if dz_flat.shape != z_flat.shape or w_flat.shape != z_flat.shape:
        raise ValueError(
            "field depth reduction: field_depth_z / field_depth_dz and the "
            f"weights must be row-aligned (N_gal_total,); got {z_flat.shape}, "
            f"{dz_flat.shape}, {w_flat.shape}. All three flat FULL-SKY arrays "
            "must come from the SAME real-galaxy mask (build_field_depth_inputs "
            "and build_field_mark_inputs use the same row-major order)."
        )

    log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    n_gal = int(z_flat.shape[0])
    chunk = int(chunk_size) if chunk_size and chunk_size > 0 else n_gal
    chunk = max(1, min(chunk, max(n_gal, 1)))
    pad = (-n_gal) % chunk
    n_pad = n_gal + pad
    z_pad = jnp.pad(z_flat, (0, pad))
    dz_pad = jnp.pad(dz_flat, (0, pad))
    w_pad = jnp.pad(w_flat, (0, pad))
    valid = jnp.arange(n_pad) < n_gal
    n_chunks = n_pad // chunk

    def _body(acc, xs):
        z_c, dz_c, w_c, val_c = xs
        sig = jnp.maximum(
            jnp.sqrt(dz_c ** 2 + survey.sigma_kde ** 2), SIGMA_EFF_FLOOR
        )
        log_Z_full = _row_log_kernel_norms(z_c, sig, val_c, log_g_grid)
        log_Z_depth = _row_log_kernel_norms(
            z_c, sig, val_c, log_g_grid, z_hi=survey.z_depth
        )
        ratio = jnp.where(val_c, jnp.exp(log_Z_depth - log_Z_full), 0.0)
        return acc + jnp.sum(w_c * ratio), None

    total, _ = lax.scan(
        _body,
        jnp.zeros((), dtype=zgrid.dtype),
        (
            z_pad.reshape(n_chunks, chunk),
            dz_pad.reshape(n_chunks, chunk),
            w_pad.reshape(n_chunks, chunk),
            valid.reshape(n_chunks, chunk),
        ),
    )
    return total


def field_observed_global_total(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """Survey-GLOBAL observed host count in the NUMERATOR's depth convention.

    ``survey.z_depth is None``: the raw full-sky galaxy count
    ``field_N_obs_total`` (theta-independent; the legacy path, bit-identical).

    ``survey.z_depth`` set: ``Sum_pix N_obs,pix * m_pix(theta)``, the depth-scaled
    count that the per-pixel numerator uses (``prior.py``'s ``Nobs *
    exp(kernels.log_depth_mass)``).  Using the raw total there counts every
    above-depth catalogued galaxy TWICE -- once in ``N_obs_total``, again in the
    missing budget, which ``_field_missing_curve`` relaxes to the full ``dN_exp``
    beyond the depth.  On a 12-pixel full-sky fixture (9 occupied pixels, 6
    galaxies each, half above ``z_depth=0.3``) the double count broke the
    normalization identity ``Sum_pix integral p_field dz == 1``, driving the sky
    mass to 0.64 at n0 = 1e-11 (the well-observed regime, where the observed count
    is comparable to the missing budget).  At K=1 the whole ``log_Z_global``
    cancels -- it enters the N per-event PE terms and the ``-N log mu`` selection
    term the same number of times, whether or not it depends on theta -- so
    single-catalog likelihood VALUES are bit-unchanged by this correction; at K>=2
    each catalog's ``log_Z_global`` sits inside its own mixture branch and does
    NOT cancel, so the double count became a spurious theta-dependent tilt on the
    sampled host fractions.
    """
    if survey.z_depth is None:
        return jnp.asarray(em_catalog.field_N_obs_total, dtype=zgrid.dtype)
    if em_catalog.field_depth_c is None:
        raise ValueError(
            "catalog_sky_weighting='field' with survey.z_depth set requires "
            "field_depth_c (the per-galaxy N_obs,pix * w_i / W_pix coefficients) "
            "built via darksirens.redshift.completion.build_field_depth_inputs."
        )
    return _field_depth_weighted_mass(
        cosmo, survey, em_catalog, jnp.asarray(em_catalog.field_depth_c)
    )


def field_marked_observed_global_total(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_h_flat: jnp.ndarray,
) -> jnp.ndarray:
    """Survey-GLOBAL observed MARKED mass in the numerator's depth convention.

    ``Sum_{i, full sky} w_hat_i h_i`` with no depth; with a depth, each galaxy's
    marked mass is scaled by its below-depth kernel ratio, which is exactly the
    marked amplitude ``exp(log_N_host + log_depth_mass)`` summed over the full sky
    (the per-pixel marked mixture normalisation cancels).  ``w_hat`` is
    ``field_mark_w``'s COUNT-renormalised weight (``build_field_mark_inputs``), so
    this reproduces the numerator's ``N_obs,pix * <h>_w,pix`` convention and is
    invariant under a global rescaling of the catalog WEIGHT column.
    """
    w_flat = jnp.asarray(em_catalog.field_mark_w, dtype=zgrid.dtype)
    h_flat = jnp.exp(jnp.asarray(log_h_flat, dtype=zgrid.dtype))
    if survey.z_depth is None:
        return jnp.sum(w_flat * h_flat)
    return _field_depth_weighted_mass(cosmo, survey, em_catalog, w_flat * h_flat)


def field_global_log_Z(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    chunk_size: int = 4096,
    N_obs_total=None,
    latent_q_rows=None,
) -> jnp.ndarray:
    """log of the survey-GLOBAL FIELD normalizer for one parameter proposal.

    ``Z(theta) = Sum_all-pixels [ N_obs,pix + N_miss,pix(theta) ]`` with, per
    pixel, ``N_miss = integral (1 - C) dN_exp lss_p`` -- the SAME missing-galaxy
    budget modulation as the per-pixel numerator:

        lss_p = 1                                (legacy; no modulation inputs)
        lss_p = Q_p(z)                           (``field_lss_q`` rows;
                                                  empty-pixel budget is the data
                                                  constant ``field_lss_q_empty_sum``)
        lss_p = max(1 + b_eff*delta_g_p(z), 0)   (``field_delta_g`` rows;
                                                  empty pixels carry delta_g == 0
                                                  by construction, so lss = 1)

    The reduction accumulates the (N_grid,) field missing CURVE
    ``V(z; theta) = Sum_occ (1 - C_p)(z) lss_p(z) + V_empty(z)`` and integrates
    ``dN_exp * V`` once -- the curve form is what lets a z-dependent factor
    (Q, delta_g, and the marked-host ``mu_miss``) modulate the budget before
    the quadrature.  ``C`` reuses ``_row_C``'s exact recipe --
    ``clip(dN_obs_s / where(dN_exp_smooth > 0, dN_exp_smooth, 1), 0, 1)`` -- and
    ``dN_exp = survey.n0 * apix * g(z)`` is the identical grid used per pixel, so
    the reduction is consistent with ``_assemble_curves`` term by term.  In
    AGGREGATE mode (``survey.c_mode``) ``C`` is instead the SAME sky-aggregate
    ``Cbar`` curve the per-row numerator uses -- one formation site,
    ``_precompute_grids`` -- for occupied and empty pixels alike, so the global
    normalizer and the numerator carry one budget.  The
    ``survey.z_depth`` completeness prior is applied EXACTLY as in
    ``_assemble_curves`` (Python-level branch on the concrete ``z_depth``;
    ``None`` takes the untouched full-grid expression): beyond the depth every
    pixel has C == 0 and lss == 1, so ``V`` relaxes to the total pixel count and
    the survey-global missing curve there is the full ``dN_exp`` per pixel.  The
    observed term follows the SAME depth convention as the numerator (see
    :func:`field_observed_global_total`): with a depth it is the depth-scaled
    count ``Sum_pix N_obs,pix * m_pix(theta)``, so an above-depth catalogued
    galaxy is represented ONCE, by the missing branch.

    Fully differentiable in ``theta`` (n0, cosmology, delta, and b_miss through
    the delta_g mode): the frozen inputs are ``field_dN_obs_s`` and the
    modulation rows (data constants).  The occupied-pixel reduction is chunked
    with ``lax.scan`` to bound peak memory at high nside.

    ``N_obs_total`` may be passed precomputed: it is member-INDEPENDENT, so
    :func:`field_global_log_Z_members` hoists the depth reduction out of its
    member vmap instead of repeating it M times.
    """
    if latent_q_rows is None and latent_enabled(em_catalog):
        # Scalar (deterministic-Q) latent path: the posterior-MEAN Q on the
        # occupied rows, mirroring what ``_resolve_lss_completion_row_tables``
        # hands the per-row numerator.  Explicit rather than implicit: without
        # this the latent branch of ``_field_missing_curve`` would never fire
        # here and the scalar normalizer would silently be the Q == 1 budget
        # while the numerator carried the mean Q.
        latent_q_rows = latent_posterior_mean_q(
            em_catalog, survey,
            _latent_C_curve(_precompute_grids(cosmo, survey, em_catalog)),
            field_rows=True)

    V_total, dN_exp = _field_missing_curve(
        cosmo, survey, em_catalog, chunk_size=chunk_size,
        latent_q_rows=latent_q_rows,
    )
    if N_obs_total is None:
        N_obs_total = field_observed_global_total(cosmo, survey, em_catalog)
    N_obs_total = jnp.asarray(N_obs_total, dtype=dN_exp.dtype)

    N_miss_total = jnp.trapezoid(dN_exp * V_total, zgrid)

    Z = N_obs_total + N_miss_total
    return jnp.log(jnp.maximum(Z, 1e-300))


def _field_missing_curve(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    chunk_size: int = 4096,
    latent_q_rows=None,
):
    """The (N_grid,) field missing curve ``V(z; theta)`` plus its companions.

    ``V = Sum_occ (1 - C_p)(z) lss_p(z) + V_empty(z)`` -- everything of the
    survey-global missing budget EXCEPT the ``dN_exp`` quadrature, so callers
    can insert a z-dependent factor (the marked-host ``mu_miss``) before
    integrating.  Beyond ``survey.z_depth`` the curve is relaxed to the total
    pixel count (every pixel has C == 0 and lss == 1 there), so the beyond-depth
    completeness prior is already baked in and callers integrate ``dN_exp * V``
    over the FULL grid.  The completeness denominator comes from
    ``_precompute_grids`` and therefore carries the depth-truncated smoothing
    ``S @ (dN_exp · 1[z <= z_depth])`` — identical to the per-pixel numerator
    path, so C matches ``_row_C`` at the depth edge too.  In AGGREGATE mode
    (``survey.c_mode``) the grids carry the sky-aggregate ``Cbar`` and it
    replaces the per-pixel ratio for every pixel, occupied AND empty (empty
    pixels have ``C == Cbar`` too, not 0), so the global normalizer carries
    exactly the budget the per-row numerator does -- the same single curve
    from the same formation site.  Returns ``(V_total, dN_exp)``.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    dN_exp = grids.dN_exp                                    # (N_grid,) theta-dependent
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
    # AGGREGATE mode: static pytree-structure branch (mirrors _row_C).
    aggregate = grids.C_bar_raw is not None
    # Stratified selection carries an (S, N_grid) stack; ndim is static.
    stratified = aggregate and grids.C_bar_raw.ndim == 2
    C_bar = jnp.clip(grids.C_bar_raw, 0.0, 1.0) if aggregate else None
    if stratified:
        # S is static (the curve stack's leading axis), so the label bound and
        # the empty-pixel budget arity are both checkable at trace time.
        n_strata = int(grids.C_bar_raw.shape[0])
        _check_stratum_labels(em_catalog, n_strata)
        # Per-occupied-row stratum, aligned with field_dN_obs_s rows.
        strat_occ = jnp.asarray(em_catalog.pixel_stratum_map)[
            jnp.asarray(em_catalog.field_occupied_pixels)]

    field_obs = jnp.asarray(em_catalog.field_dN_obs_s)       # (n_occ, N_grid) f32 constant
    n_occ = int(field_obs.shape[0])                          # static
    n_empty = jnp.asarray(em_catalog.field_n_empty, dtype=dN_exp.dtype)

    # Static (pytree-structure) modulation dispatch, mirroring the numerator's
    # rule: a Q table REPLACES the local-overdensity factor.
    # LATENT: ``latent_q_rows`` (n_occ, N_grid) is member m's GENERATED Q on the
    # occupied rows, supplied by the caller.  It enters exactly where a loaded
    # Q table would, so the budget arithmetic below is shared rather than
    # duplicated -- the only differences are that the empty-pixel budget comes
    # from the ``f_p`` formula (empty pixels are off-footprint, so Q == 1 there
    # by the seam's own convention), where the table path reads the built
    # ``field_lss_q_fp_empty_sum``.
    latent = latent_q_rows is not None
    has_q = latent or em_catalog.field_lss_q is not None
    has_dg = em_catalog.field_delta_g is not None
    if has_q and has_dg:
        raise NotImplementedError(
            "field_global_log_Z: field_lss_q and field_delta_g are mutually "
            "exclusive (Q_LSS replaces the local-overdensity factor, matching "
            "the per-pixel numerator)."
        )
    if has_q and not latent and em_catalog.field_lss_q_empty_sum is None:
        raise ValueError(
            "field_global_log_Z: field_lss_q requires field_lss_q_empty_sum "
            "(the empty-pixel Q budget); build both via "
            "build_field_lss_q_inputs."
        )
    if latent:
        mod_rows = jnp.asarray(latent_q_rows)                # (n_occ, N_grid)
    elif has_q:
        mod_rows = jnp.asarray(em_catalog.field_lss_q)       # (n_occ, N_grid) f32
    elif has_dg:
        mod_rows = jnp.asarray(em_catalog.field_delta_g)     # (n_occ, N_grid) f32
    else:
        mod_rows = jnp.zeros((n_occ, 1), dtype=jnp.float32)  # inert placeholder
    b_eff = survey.alpha_miss * survey.b_miss                # traced (delta_g mode)

    # Per-pixel selection fraction (field-level PR-2): C -> f_p * C on the
    # occupied rows, and the empty-pixel budget below becomes
    # V_empty - C * Sum_empty f_p (lss-weighted; see the branch).  Static
    # pytree-structure branch; None is bit-identical.  The loader restricts
    # f_p to aggregate/selection c_modes without strata or a Q ENSEMBLE (their
    # empty budgets would need f_p-weighted twins); a deterministic Q table is
    # admitted and carries ``field_lss_q_fp_empty_sum``.
    has_fp = em_catalog.field_f_p_occ is not None
    if has_fp and (has_dg or stratified or not aggregate):
        raise NotImplementedError(
            "_field_missing_curve: field_f_p_occ (per-pixel selection "
            "fraction) is only supported under aggregate/selection c_modes "
            "without delta_g rows or stratified selection (their empty-pixel "
            "budgets would need f_p-weighted twins)."
        )
    if (has_fp and not latent and em_catalog.field_lss_q_members is not None
            and em_catalog.field_lss_q_fp_empty_sum_members is None):
        raise ValueError(
            "_field_missing_curve: a Q ENSEMBLE with field_f_p_occ requires "
            "field_lss_q_fp_empty_sum_members (the per-member twin of "
            "field_lss_q_fp_empty_sum); without it member m's normalizer "
            "would carry the DETERMINISTIC f_p-weighted empty-pixel budget "
            "while its numerator carries member m's Q. Build it via "
            "build_field_lss_q_fp_empty_sum_members."
        )
    if has_fp and has_q and not latent and (
            em_catalog.field_lss_q_fp_empty_sum is None):
        raise ValueError(
            "_field_missing_curve: a Q table with field_f_p_occ requires "
            "field_lss_q_fp_empty_sum (Sum_{p empty} f_p Q_p(z)); build it "
            "via build_field_lss_q_fp_empty_sum. Without it the empty-pixel "
            "budget would model unobserved sky as Cbar-complete."
        )
    if latent and not has_fp:
        raise ValueError(
            "lss_field_mode='latent' requires the per-pixel selection "
            "fraction on the field side (field_f_p_occ / "
            "field_f_p_empty_sum): the sky moments B(z; b) = sum_p f_p e^{bf} "
            "carry f_p, so a normalizer built without it would consume a "
            "different budget than the one rho conserves (PLAN §4.4 guard 6)."
        )
    if has_fp and em_catalog.field_f_p_empty_sum is None:
        raise ValueError(
            "_field_missing_curve: field_f_p_occ requires "
            "field_f_p_empty_sum (Sum_{p empty} f_p) for the empty-pixel "
            "budget."
        )
    f_occ_rows = (jnp.asarray(em_catalog.field_f_p_occ)
                  if has_fp else jnp.zeros((n_occ,), dtype=jnp.float32))

    # Legacy-floor number conservation, delta_g mode only (Q carries its own
    # mean-one renormalization, and the latent path generates a conserved Q).
    dg_norm = grids.lss_floor_norm if has_dg else None

    def _row_V(obs_row, mod_row, strat_row, f_row):
        obs_row = obs_row.astype(dN_exp.dtype)
        if stratified:
            C = C_bar[strat_row]     # this row's stratum curve
        elif aggregate:
            C = C_bar        # one survey curve; the row's own KDE is unused
        else:
            C = jnp.clip(obs_row / dN_exp_safe, 0.0, 1.0)
        if has_fp:
            C = f_row.astype(dN_exp.dtype) * C
        if has_q:
            lss = mod_row.astype(dN_exp.dtype)
        elif has_dg:
            lss = jnp.maximum(1.0 + b_eff * mod_row.astype(dN_exp.dtype), 0.0)
            # Same number-conservation renormalization the per-pixel numerator
            # applies (_completion_curves_row): the global normalizer has to
            # carry the SAME missing budget as the numerator or the two sides
            # divide different totals.  Static pytree-structure branch, and
            # exactly 1.0 wherever the floor never engaged.
            if dg_norm is not None:
                lss = lss / dg_norm
        else:
            lss = 1.0
        return (1.0 - C) * lss                               # (N_grid,)

    # Chunked scan over occupied rows: pad to a whole number of ``chunk_size``
    # blocks and mask the padding so padded (all-zero) rows -- which would
    # otherwise read as empty pixels with C == 0 -- contribute nothing.
    pad = (-n_occ) % chunk_size
    n_pad = n_occ + pad
    obs_pad = jnp.pad(field_obs, ((0, pad), (0, 0)))
    mod_pad = jnp.pad(mod_rows, ((0, pad), (0, 0)))
    strat_rows = (strat_occ.astype(jnp.int32) if stratified
                  else jnp.zeros(n_occ, dtype=jnp.int32))
    strat_pad = jnp.pad(strat_rows, (0, pad))
    valid = jnp.arange(n_pad) < n_occ
    n_chunks = n_pad // chunk_size if chunk_size > 0 else 0
    obs_chunks = obs_pad.reshape(n_chunks, chunk_size, zgrid.size)
    mod_chunks = mod_pad.reshape(n_chunks, chunk_size, mod_rows.shape[1])
    strat_chunks = strat_pad.reshape(n_chunks, chunk_size)
    valid_chunks = valid.reshape(n_chunks, chunk_size)

    f_pad_rows = jnp.pad(f_occ_rows, (0, pad))
    f_chunks = f_pad_rows.reshape(n_chunks, chunk_size)

    def _body(acc, xs):
        obs_c, mod_c, strat_c, val_c, f_c = xs
        Vc = vmap(_row_V)(obs_c, mod_c, strat_c, f_c)        # (chunk, N_grid)
        Vc = jnp.where(val_c[:, None], Vc, 0.0)
        return acc + jnp.sum(Vc, axis=0), None

    V_occ, _ = lax.scan(
        _body,
        jnp.zeros(zgrid.size, dtype=dN_exp.dtype),
        (obs_chunks, mod_chunks, strat_chunks, valid_chunks, f_chunks),
    )

    # Empty pixels: per-pixel C == 0, so their budget curve is lss_p itself --
    # n_empty for the legacy/delta_g modes (delta_g == 0 on empty pixels), the
    # data constant Sum_empty Q_p(z) for the Q mode.  In aggregate mode empty
    # pixels carry C == Cbar like everyone else, so their curve is
    # (1 - Cbar) * lss_p.
    if stratified:
        # Per-stratum decomposition: empty pixels in stratum s carry the
        # stratum's own curve, so V_empty = Sum_s (1 - Cbar_s) * budget_s with
        # budget_s = n_empty_s (no Q) or Sum_{empty in s} Q_p(z) (Q mode).
        # Collapsing this to one curve would put the WRONG completeness on
        # every empty pixel outside the reference stratum -- the numerator and
        # normalizer would then carry different budgets.
        if has_q:
            if em_catalog.field_lss_q_empty_sum_strata is None:
                raise ValueError(
                    "stratified selection with a Q table needs "
                    "EMCatalog.field_lss_q_empty_sum_strata (per-stratum "
                    "empty-pixel Q budgets); build them alongside "
                    "field_lss_q_empty_sum from the stratum map."
                )
            budget_s = jnp.asarray(em_catalog.field_lss_q_empty_sum_strata,
                                   dtype=dN_exp.dtype)      # (S, N_grid)
            _check_empty_budget_arity(
                budget_s.shape, n_strata, "field_lss_q_empty_sum_strata")
        else:
            if em_catalog.empty_stratum_counts is None:
                raise ValueError(
                    "stratified selection needs "
                    "EMCatalog.empty_stratum_counts (per-stratum empty-pixel "
                    "counts) for the global normalizer."
                )
            budget_s = jnp.asarray(em_catalog.empty_stratum_counts,
                                   dtype=dN_exp.dtype)[:, None]  # (S, 1)
            _check_empty_budget_arity(
                budget_s.shape, n_strata, "empty_stratum_counts")
        V_empty = jnp.sum((1.0 - C_bar) * budget_s, axis=0)
    else:
        if has_q and not latent:
            V_empty = jnp.asarray(em_catalog.field_lss_q_empty_sum,
                                  dtype=dN_exp.dtype)
            # The weight f_p multiplies inside the empty sum, so the f_p-aware
            # budget is Sum_empty Q_p (1 - f_p C) = Sum_empty Q_p
            # - C Sum_empty f_p Q_p -- NOT n_empty - C Sum_empty f_p.
            fp_weight = (jnp.asarray(em_catalog.field_lss_q_fp_empty_sum,
                                     dtype=dN_exp.dtype)
                         if has_fp else None)
        else:
            # LATENT: every empty pixel is outside the fitted footprint (the
            # footprint is fitted TO the counts), so Q == 1 there and the
            # budget is the plain f_p formula below -- no per-member empty
            # budget exists or is needed, which is the structural reason
            # PLAN eq. (4)'s off-footprint block is conserved trivially.
            V_empty = n_empty
            fp_weight = (jnp.asarray(em_catalog.field_f_p_empty_sum,
                                     dtype=dN_exp.dtype)
                         if has_fp else None)
        if has_fp:
            # Sum_{p empty} lss_p (1 - f_p C(z)) = V_empty - C(z) * fp_weight,
            # with (V_empty, fp_weight) = (n_empty, Sum_empty f_p) without a Q
            # table and (Sum_empty Q_p, Sum_empty f_p Q_p) with one.  S-3: the
            # ``elif aggregate`` branch below is what modelled unobserved sky
            # as Cbar-complete, and f_p is what supplies the zero.
            V_empty = V_empty - C_bar * fp_weight
        elif aggregate:
            V_empty = (1.0 - C_bar) * V_empty

    if dg_norm is not None:
        # Empty pixels carry delta_g == 0 by construction, so their UNFLOORED
        # factor is exactly 1 -- and after the renormalization it is 1/norm(z),
        # the same divisor every occupied row got.  Without this the empty
        # block would keep the pre-renormalization weight and the two halves of
        # the global budget would disagree.
        V_empty = V_empty / dg_norm

    V_total = V_occ + V_empty
    # ``z_depth`` is concrete at trace time -> Python-level branch (mirrors
    # ``_assemble_curves``); ``None`` never constructs a mask.  Beyond the depth
    # every pixel (occupied and empty) has C == 0 and lss == 1, so the global
    # missing curve relaxes to the total pixel count: the full expected
    # population is uncatalogued there, not nonexistent.
    #
    # PR-8 does NOT change this line, and that is a statement about the model
    # rather than an omission.  With an amp(z) anchor ``lss != 1`` above the
    # depth, but PLAN eq. (4) makes ``sum_{p in F} Q_p = P_F`` at ``C = 0`` and
    # every off-footprint pixel carries ``Q == 1``, so the sum this branch
    # replaces is ALREADY the total pixel count -- the amp(z) field moves
    # placement, never the survey-global budget (§4.2's gauge fixing).  Keeping
    # the relaxation therefore also keeps the normalizer member-INDEPENDENT
    # bit-exactly (pin P13), instead of letting each member's float error in a
    # ~1e5-term sum enter ``-259 log mu``.
    if survey.z_depth is not None:
        depth_mask = zgrid <= survey.z_depth
        n_pix_total = n_occ + n_empty
        V_total = jnp.where(depth_mask, V_total, n_pix_total)
    return V_total, dN_exp


def _member_fp_empty_rows(em_catalog: EMCatalog):
    """Per-member ``Sum_{p empty} f_p Q_p(z)``, or ``None`` when unused.

    ``None`` whenever the run has no ``f_p`` on the field side, or no Q ENSEMBLE
    to pair it with (the budget never reaches :func:`_field_missing_curve`, so
    the member vmap broadcasts it).  A
    run that HAS ``f_p`` and a Q ensemble but not its per-member twin is
    rejected rather than silently reusing the deterministic budget -- the same
    hazard :func:`_member_empty_strata_rows` guards one axis over.
    """
    if em_catalog.field_f_p_occ is None or em_catalog.field_lss_q_members is None:
        return None
    rows = em_catalog.field_lss_q_fp_empty_sum_members
    if rows is None:
        if em_catalog.field_lss_q_fp_empty_sum is None:
            return None
        raise ValueError(
            "a Q ENSEMBLE with the per-pixel selection fraction needs "
            "EMCatalog.field_lss_q_fp_empty_sum_members (the per-member twin "
            "of field_lss_q_fp_empty_sum); build it via "
            "build_field_lss_q_fp_empty_sum_members, or the per-member "
            "normalizers would reuse the deterministic f_p-weighted "
            "empty-pixel budget."
        )
    return jnp.asarray(rows)


def _member_empty_strata_rows(survey: SurveyParams, em_catalog: EMCatalog):
    """Per-member per-stratum empty-pixel Q budget, or ``None`` when unused.

    ``None`` whenever the run is not stratified (the strata budget never reaches
    :func:`_field_missing_curve`, so the member vmap broadcasts it) or the
    stratified normalizer takes its ``empty_stratum_counts`` branch.  A
    stratified Q-ensemble run that supplies the DETERMINISTIC strata budget but
    NOT its per-member twin is rejected: the stratified branch reads only the
    strata array, so member m's normalizer would carry the deterministic
    empty-pixel budget while member m's numerator carries member m's Q --
    marginalizing over inconsistent estimands.
    """
    if survey.selection_strata is None:
        return None
    rows = em_catalog.field_lss_q_empty_sum_strata_members
    if rows is None:
        if em_catalog.field_lss_q_empty_sum_strata is None:
            return None
        raise ValueError(
            "stratified selection with a Q ENSEMBLE needs "
            "EMCatalog.field_lss_q_empty_sum_strata_members (the per-member "
            "twin of field_lss_q_empty_sum_strata); build it alongside "
            "field_lss_q_empty_sum_members from the stratum map, or the "
            "per-member normalizers would reuse the deterministic "
            "empty-pixel budget."
        )
    rows = jnp.asarray(rows)
    # Same broadcast hazard as the deterministic budget, one axis over: the
    # stratum axis of the (M, S, N_grid) member stack must span exactly S.
    _check_empty_budget_arity(
        rows.shape[1:], len(survey.selection_strata),
        "field_lss_q_empty_sum_strata_members",
    )
    return rows


def _replace_member_q(em_catalog, q_m, q_empty_m, q_strata_m,
                      q_fp_empty_m=None):
    """Install ensemble member m's Q rows / empty-pixel budgets on the catalog.

    ``q_fp_empty_m`` is member m's ``Sum_{p empty} f_p Q_p(z)``.  It must be
    swapped in for the same reason the plain empty budget is: member m's
    normalizer carrying the DETERMINISTIC f_p-weighted budget while its
    numerator carries member m's Q marginalizes inconsistent estimands.  Left
    ``None`` when the run has no ``f_p`` (then the field is already ``None`` on
    the catalog and the substitution is a no-op).
    """
    cat_m = em_catalog._replace(
        field_lss_q=q_m, field_lss_q_empty_sum=q_empty_m
    )
    if q_fp_empty_m is not None:
        cat_m = cat_m._replace(field_lss_q_fp_empty_sum=q_fp_empty_m)
    if q_strata_m is None:
        return cat_m
    return cat_m._replace(field_lss_q_empty_sum_strata=q_strata_m)


def field_global_log_Z_members(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """Per-member survey-GLOBAL normalizers ``log Z_m(theta)``, shape (M,).

    One :func:`field_global_log_Z` evaluation per Q-ensemble member, sharing
    everything except the member's Q rows / empty-pixel budget
    (``field_lss_q_members`` / ``field_lss_q_empty_sum_members``).  Consumed by
    the lss_marginalize member vmap in the likelihood core (each member state
    carries its own global normalizer under the field convention).

    The observed term (``field_observed_global_total``, including the depth
    reduction) is member-INDEPENDENT and is hoisted OUT of the member vmap.

    Under STRATIFIED selection the per-stratum empty-pixel budget
    (``field_lss_q_empty_sum_strata``) is what ``_field_missing_curve`` reads, so
    its per-member twin must be swapped in too (see
    :func:`_member_empty_strata_rows`).
    """
    if latent_enabled(em_catalog):
        # LATENT: each member's occupied-row Q is generated, installed, and
        # discarded one at a time (``lax.map``, not ``vmap``), so the peak stays
        # at ONE (n_occ, N_grid) block instead of the (M, n_occ, N_grid) stack
        # the table path can afford only because that stack is a data constant.
        #
        # A prediction worth knowing when reading these numbers: by PLAN eq.
        # (4) the occupied-row budget sums to its Q == 1 value exactly, and
        # every empty pixel is off-footprint, so the survey-GLOBAL normalizer
        # is member-INDEPENDENT in latent mode.  That is the gauge fixing of
        # §4.2 ("C and n0 own the budget, Q owns placement"), not a coincidence
        # -- and it is pinned by tests/test_latent_seam.py.
        grids = _precompute_grids(cosmo, survey, em_catalog)
        C_curve = _latent_C_curve(grids)
        depth_mask = latent_support_mask(em_catalog, survey)  # PR-8; see above
        N_obs_total_l = field_observed_global_total(cosmo, survey, em_catalog)

        def _one_latent(m):
            lq = latent_member_logq_rows(em_catalog, survey, C_curve, m,
                                         field_rows=True)
            q_m = _member_q_eff_from_logq(lq, depth_mask, True,
                                          _LATENT_LOGQ_CLIP)
            return field_global_log_Z(
                cosmo, survey, em_catalog, N_obs_total=N_obs_total_l,
                latent_q_rows=q_m)

        return lax.map(_one_latent, jnp.arange(latent_n_members(em_catalog)))

    q_members = em_catalog.field_lss_q_members
    q_empty_members = em_catalog.field_lss_q_empty_sum_members
    if q_members is None or q_empty_members is None:
        raise ValueError(
            "field_global_log_Z_members requires field_lss_q_members and "
            "field_lss_q_empty_sum_members; build them via "
            "build_field_lss_q_member_inputs."
        )
    strata_members = _member_empty_strata_rows(survey, em_catalog)
    fp_empty_members = _member_fp_empty_rows(em_catalog)
    N_obs_total = field_observed_global_total(cosmo, survey, em_catalog)

    def _one(q_m, q_empty_m, q_strata_m, q_fp_empty_m):
        cat_m = _replace_member_q(em_catalog, q_m, q_empty_m, q_strata_m,
                                  q_fp_empty_m)
        return field_global_log_Z(
            cosmo, survey, cat_m, N_obs_total=N_obs_total
        )

    return vmap(_one, in_axes=(0, 0,
                               None if strata_members is None else 0,
                               None if fp_empty_members is None else 0))(
        jnp.asarray(q_members), jnp.asarray(q_empty_members), strata_members,
        fp_empty_members
    )


def field_global_log_Z_marked(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    mu_miss: jnp.ndarray,
    log_h_flat: jnp.ndarray,
    S_obs=None,
) -> jnp.ndarray:
    """Marked-host survey-GLOBAL normalizer.

    ``Z(theta, eta) = Sum_{i in obs, full sky} w_i h(m_i | eta) r_i(theta)
                      + integral mu_miss(z; eta) dN_exp(z) V(z; theta) dz``

    -- the marked analogue of :func:`field_global_log_Z`: the observed term is
    the full-sky marked mass (the marked kernel state rescales each pixel's
    catalog mass from N_obs to Sum w_i h_i, so the global total must follow),
    and the missing budget carries the SAME ``mu_miss(z | eta)`` factor as the
    numerator's ``dN_miss``.  ``mu_miss`` and ``log_h_flat`` must be built from
    the FULL-SKY flat marks (``field_mark_*``), so the PE and selection states
    produce the SAME Z for the same (theta, eta) and the constants cancel
    structurally between the two likelihood seams.

    ``r_i(theta)`` is the below-depth kernel-mass ratio, 1 when no
    ``survey.z_depth`` is set: with a depth the numerator's marked amplitude is
    ``exp(log_N_host + log_depth_mass)``, so the global marked mass must be
    depth-scaled the same way (see :func:`field_marked_observed_global_total`).
    ``S_obs`` may be passed precomputed -- it is member-INDEPENDENT, so
    :func:`field_global_log_Z_marked_members` hoists it out of the member vmap.
    """
    V_total, dN_exp = _field_missing_curve(cosmo, survey, em_catalog)

    if S_obs is None:
        S_obs = field_marked_observed_global_total(
            cosmo, survey, em_catalog, log_h_flat
        )
    S_obs = jnp.asarray(S_obs, dtype=dN_exp.dtype)

    # ``V_total`` already carries the beyond-depth relaxation (C == 0, lss == 1),
    # where ``mu_miss`` defaults to the catalog-wide mean host efficiency (no
    # observed galaxies to inform it locally), so the integrand is taken over the
    # FULL grid.
    integrand = jnp.asarray(mu_miss, dtype=dN_exp.dtype) * dN_exp * V_total
    N_miss_total = jnp.trapezoid(integrand, zgrid)

    Z = S_obs + N_miss_total
    return jnp.log(jnp.maximum(Z, 1e-300))


def field_global_log_Z_marked_members(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    mu_miss: jnp.ndarray,
    log_h_flat: jnp.ndarray,
) -> jnp.ndarray:
    """Per-member MARKED survey-GLOBAL normalizers ``log Z_m(theta, eta)``, (M,).

    The marked analogue of :func:`field_global_log_Z_members`: one
    :func:`field_global_log_Z_marked` evaluation per Q-ensemble member, swapping
    only the member's Q rows / empty-pixel budget (``field_lss_q_members`` /
    ``field_lss_q_empty_sum_members``, plus the per-stratum twin under
    stratified selection) into the missing curve.  The observed
    marked mass (with its depth scaling) and the ``mu_miss(z | eta)`` factor are
    member-INDEPENDENT (full-sky flat marks), so they are computed ONCE here and
    shared verbatim across members -- reusing ``field_global_log_Z_marked``
    guarantees the observed term and mu_miss integrand are op-for-op identical to
    the scalar marked path.
    Consumed by the lss_marginalize member vmap in the likelihood core (each
    member state carries its own marked global normalizer under field weighting).
    """
    q_members = em_catalog.field_lss_q_members
    q_empty_members = em_catalog.field_lss_q_empty_sum_members
    if q_members is None or q_empty_members is None:
        raise ValueError(
            "field_global_log_Z_marked_members requires field_lss_q_members and "
            "field_lss_q_empty_sum_members; build them via "
            "build_field_lss_q_member_inputs."
        )
    strata_members = _member_empty_strata_rows(survey, em_catalog)
    S_obs = field_marked_observed_global_total(
        cosmo, survey, em_catalog, log_h_flat
    )

    def _one(q_m, q_empty_m, q_strata_m):
        cat_m = _replace_member_q(em_catalog, q_m, q_empty_m, q_strata_m)
        return field_global_log_Z_marked(
            cosmo, survey, cat_m, mu_miss, log_h_flat, S_obs=S_obs
        )

    return vmap(_one, in_axes=(0, 0, None if strata_members is None else 0))(
        jnp.asarray(q_members), jnp.asarray(q_empty_members), strata_members
    )


def build_field_mark_inputs(
    full_z: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
    mark_tables: dict,
    mark_names: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flat FULL-SKY per-galaxy inputs for the MARKED field normalizer.

    ``mark_tables`` maps mark name -> (N_pix, N_max) z-centred mark array (the
    same ``attach_mark_inputs`` products the numerator uses); real (non-padded)
    slots are selected by ``full_n`` (fallback ``full_w > 0``).  Returns
    ``(field_mark_z, field_mark_w, field_mark_values)`` — (N_gal,), (N_gal,),
    and (N_gal, n_marks) float32 with columns ordered by ``mark_names``.

    ``field_mark_w`` carries the COUNT-renormalised weights
    ``w_hat_i = N_obs,pix * w_i / Σ_{j in pix} w_j`` (identically 1 for a
    unit-weight catalog), so the flat sum ``Σ_i w_hat_i h_i`` that
    :func:`field_marked_observed_global_total` forms is exactly the numerator's
    per-pixel amplitude ``N_obs,pix * <h>_w,pix`` summed over the sky (see
    :func:`darksirens.redshift.catalog._row_marked_kernel_state`) -- the raw
    weights would make the global observed budget scale with the arbitrary
    WEIGHT normalisation while the missing budget stayed a count.
    """
    z_np = np.asarray(full_z)
    if full_n is not None:
        n_np = np.asarray(full_n).reshape(-1).astype(np.int64)
        real = np.arange(z_np.shape[1])[None, :] < n_np[:, None]
    elif full_w is not None:
        real = np.asarray(full_w) > 0.0
    else:
        raise ValueError(
            "build_field_mark_inputs requires full_n or full_w to mask padded "
            "galaxy slots."
        )
    z_flat = jnp.asarray(z_np[real], dtype=jnp.float32)
    if full_w is None:
        w_hat = np.ones(int(real.sum()))
    else:
        w_np = np.where(real, np.asarray(full_w, dtype=np.float64), 0.0)
        w_sum = w_np.sum(axis=1)                       # (N_pix,)
        n_obs = real.sum(axis=1).astype(np.float64)    # (N_pix,)
        scale = np.where(w_sum > 0.0, n_obs / np.where(w_sum > 0.0, w_sum, 1.0), 0.0)
        w_hat = (w_np * scale[:, None])[real]
    w_flat = jnp.asarray(w_hat, dtype=jnp.float32)
    missing = [name for name in mark_names if mark_tables.get(name) is None]
    if missing:
        raise ValueError(
            f"build_field_mark_inputs: marks {missing} not present in "
            "mark_tables; the field normalizer needs every selected mark over "
            "the full sky."
        )
    values = np.stack(
        [np.asarray(mark_tables[name])[real] for name in mark_names], axis=1
    )
    return z_flat, w_flat, jnp.asarray(values, dtype=jnp.float32)


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def _completion_clip_fractions_for_pixel(
    pix: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    q_row=None,
) -> dict[str, float]:
    """Clipping fractions on ``zgrid`` for one catalog row.

    ``C_iso_clipped_fraction`` is the raw-ratio clip and
    ``C_eff_clipped_fraction`` the effective-completeness clip (both always
    valid).  ``rho_miss_eff_clipped_fraction`` depends on the missing-galaxy
    rate factor: for the legacy local-overdensity model it is the fraction where
    ``1 + b_eff*delta_g < 0``; for an LSS-conditioned ``Q_LSS`` table (``q_row``
    supplied, the row-aligned ``(N_grid,)`` factor) ``Q >= 0`` always, so it is
    instead the fraction where ``logQ`` hit the ``±_LOGQ_CLIP`` bound.
    ``lss_source`` records which factor was used.
    """
    global_pix = pix if em_catalog.unique_pixels is None else em_catalog.unique_pixels[pix]

    if grids.C_bar_raw is not None:
        # AGGREGATE mode: the raw ratio is the one sky-aggregate curve (same
        # for every pixel), so the reported C-clip fraction is the fraction of
        # the grid where the GLOBAL clip engages -- exactly the n0-kink
        # exposure flagged in the module docstring.  Stratified selection
        # reports this pixel's stratum curve.
        if grids.C_bar_raw.ndim == 2:
            C_raw = grids.C_bar_raw[em_catalog.pixel_stratum_map[global_pix]]
        else:
            C_raw = grids.C_bar_raw
    else:
        if em_catalog.dN_obs_kde is not None:
            # Cache row == catalog row by construction (see _row_C); index directly.
            dN_obs = em_catalog.dN_obs_kde[pix]
        else:
            dN_obs = _kde_dndz_obs(
                pix, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals
            )

        dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
        C_raw = dN_obs / dN_exp_safe
    C_clipped_mask = (C_raw < 0.0) | (C_raw > 1.0)
    C = jnp.clip(C_raw, 0.0, 1.0)

    if q_row is not None:
        lss = q_row[pix]                       # row-aligned Q_LSS factor (>= 0)
        lss_source = "Q_LSS"
        q_hi = float(jnp.exp(_LOGQ_CLIP)) * (1.0 - 1e-6)
        q_lo = float(jnp.exp(-_LOGQ_CLIP)) * (1.0 + 1e-6)
        lss_clipped_mask = (lss >= q_hi) | (lss <= q_lo)
    else:
        b_eff = survey.alpha_miss * survey.b_miss
        if em_catalog.delta_g_pix_z.shape[0] == 1:
            delta_g_z = em_catalog.delta_g_pix_z[0]
        else:
            delta_g_z = em_catalog.delta_g_pix_z[global_pix]
        lss_raw = 1.0 + b_eff * delta_g_z
        lss = jnp.maximum(lss_raw, 0.0)
        # Report the factor the LIKELIHOOD actually forms: the clip FRACTION is
        # unchanged by the renormalization (a divisor cannot make a floored
        # cell unfloored), but C_eff and the missing density are, and those are
        # the numbers this diagnostic is read for.
        if grids.lss_floor_norm is not None:
            lss = lss / grids.lss_floor_norm
            lss_source = "legacy_delta_g_conserved"
        else:
            lss_source = "legacy_delta_g"
        lss_clipped_mask = lss_raw < 0.0

    dN_miss = (1.0 - C) * grids.dN_exp * lss
    dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
    C_eff_raw = 1.0 - dN_miss / dN_exp_pos
    C_eff_clipped_mask = (C_eff_raw < 0.0) | (C_eff_raw > 1.0)

    return {
        "C_iso_clipped_fraction": float(jnp.mean(C_clipped_mask)),
        "C_eff_clipped_fraction": float(jnp.mean(C_eff_clipped_mask)),
        "rho_miss_eff_clipped_fraction": float(jnp.mean(lss_clipped_mask)),
        "lss_source": lss_source,
    }


def completion_clip_diagnostics(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    pixels: np.ndarray | None = None,
    max_pixels: int = 64,
) -> dict[str, object]:
    """Summarise completion clipping over a representative pixel set."""
    grids = _precompute_grids(cosmo, survey, em_catalog)
    # Q_LSS-aware: resolve the (optional) completion table once and report the
    # Q-clip instead of the (inapplicable) delta_g-negativity when it is present.
    q_row, _ = _resolve_lss_completion_row_tables(em_catalog)

    # LATENT MODE IS NOT MEASURABLE HERE, and silence was the wrong answer.
    # There is no Q TABLE in latent mode -- Q is generated in-likelihood -- so
    # ``q_row`` is None and this function used to fall through to the delta_g
    # branch, reporting ``lss_source = "legacy_delta_g"`` with a clipped
    # fraction computed from the all-zero delta_g placeholder (``--use_lss`` is
    # refused in latent mode).  That reads as a clean zero, which is the one
    # answer it cannot support: the generated logQ IS clipped to +-_LOGQ_CLIP,
    # and on the production anchor the clipped footprint-voxel fraction is
    # 0.085% at b_GW = 1, 1.8% at 2, 4.6% at 3 and 44.5% at 4 -- the top of the
    # artifact's own Chebyshev range.  So the one diagnostic built to catch this
    # could not fire.  Refuse instead of reporting a number that means nothing;
    # measuring it properly needs the seam's generated rows, which this
    # table-oriented function does not have.
    if latent_enabled(em_catalog):
        raise NotImplementedError(
            "completion_clip_diagnostics cannot measure latent mode: Q is "
            "generated in-likelihood, so there is no table to inspect and the "
            "delta_g fallback would report a meaningless zero from the inert "
            "placeholder. The generated logQ IS clipped (measured on the "
            "production anchor: 0.085% of footprint voxels at b_GW = 1 rising to "
            "44.5% at b_GW = 4), so a zero here would be actively misleading. "
            "Use the seam's own rows (latent_member_logq_rows) to measure it.")

    if pixels is None:
        if em_catalog.unique_pixels is not None:
            pixels_np = np.arange(np.asarray(em_catalog.unique_pixels).size, dtype=np.int32)
        else:
            pixels_np = np.arange(np.asarray(em_catalog.zgals).shape[0], dtype=np.int32)
    else:
        pixels_np = np.asarray(pixels, dtype=np.int32).reshape(-1)

    if max_pixels is not None and max_pixels > 0:
        pixels_np = pixels_np[:max_pixels]

    per_pixel = []
    for pix in pixels_np:
        fractions = _completion_clip_fractions_for_pixel(
            int(pix), grids, survey, em_catalog, q_row=q_row
        )
        fractions["pixel"] = int(pix)
        if em_catalog.unique_pixels is not None:
            fractions["global_pixel"] = int(np.asarray(em_catalog.unique_pixels)[int(pix)])
        per_pixel.append(fractions)

    fields = [
        "C_iso_clipped_fraction",
        "C_eff_clipped_fraction",
        "rho_miss_eff_clipped_fraction",
    ]
    summary: dict[str, object] = {
        "n_zgrid": int(zgrid.size),
        "z_min": float(zgrid[0]),
        "z_max": float(zgrid[-1]),
        "n_pixels_checked": int(len(per_pixel)),
        "lss_source": ("Q_LSS" if q_row is not None
                       else ("legacy_delta_g_conserved"
                             if grids.lss_floor_norm is not None
                             else "legacy_delta_g")),
        "per_pixel": per_pixel,
    }
    for field in fields:
        vals = np.array([item[field] for item in per_pixel], dtype=float)
        summary[f"mean_{field}"] = float(vals.mean()) if vals.size else 0.0
        summary[f"max_{field}"] = float(vals.max()) if vals.size else 0.0

    _conserved = any(item.get("lss_source") == "legacy_delta_g_conserved"
                     for item in per_pixel)
    if (q_row is None and not _conserved
            and summary["max_rho_miss_eff_clipped_fraction"] > 0.0):
        # The legacy factor's floor is not a numerical guard here: it changes
        # the BUDGET.  delta_g is mean-zero over the sky, so the unfloored
        # factor sums to N_pix exactly (pure redistribution); every floored
        # cell adds missing galaxies that (1 - C) and n0 never authorised.
        warnings.warn(
            "legacy delta_g missing-galaxy factor is floored at zero on "
            f"{summary['max_rho_miss_eff_clipped_fraction']:.1%} of the grid "
            f"(worst pixel of {summary['n_pixels_checked']} checked): "
            "1 + alpha_miss*b_miss*delta_g < 0 there, which breaks the "
            "mean-one property of delta_g and INFLATES the total missing "
            "budget instead of only redistributing it (the Q path renormalizes "
            "for exactly this reason). This run asked for the LEGACY "
            "unrenormalized floor (SurveyParams.lss_floor / "
            "--lss_floor legacy); the default renormalizes the floored factor "
            "to full-sky mean one at every z and reports "
            "lss_source='legacy_delta_g_conserved'. Drop the legacy flag, "
            "lower b_miss's prior ceiling, or use an LSS completion table.",
            RuntimeWarning,
            stacklevel=2,
        )
    summary.update(_kernel_match_diagnostics(em_catalog))
    return summary


def selection_budget_audit(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    N_obs_total: float,
    n_occupied: int,
    n_pix_total: int,
) -> dict[str, float]:
    """Audit the ``c_mode='selection'`` missing budget against the actual counts.

    Under the counts-based estimators the missing density is data-anchored:
    ``(1 - C) dN_exp ~ dN_exp - dN_obs_s`` where ``0 < C < 1``, and the clip at
    ``C = 1`` forces it to zero where the survey is complete.  Under
    ``c_mode='selection'`` no count enters the completeness at all, so the budget
    amplitude is exactly ``(1 - C_sel) n0 apix dV_c/dz (1+z)^delta`` -- linear in
    ``n0``, proportional to ``H0^-3``, and identified only by the ``log10n0``
    prior (module docstring).  The consistency the likelihood never imposes is

        N_obs_total  ~  Sum_p integral C_sel(z) dN_exp(z) dz,

    which this evaluates at the run's (theta, n0, delta, cosmology): a
    ``model / observed`` ratio far from 1 means the sampled budget amplitude is
    inconsistent with the catalog it is completing, and the resulting host
    fractions -- and therefore the H0 posterior width -- are prior-driven.

    ``n_occupied`` is the footprint (pixels with at least one galaxy) and
    ``n_pix_total`` the whole sky; both predictions are reported because the
    truth sits between them for a sparsely populated footprint.  Returns a flat
    float dict (JSON-friendly for the run provenance).
    """
    if not _is_selection_c_mode(survey.c_mode):
        raise ValueError(
            "selection_budget_audit applies to c_mode='selection' only: the "
            "counts-based estimators anchor the budget to dN_obs by "
            "construction."
        )
    grids = _precompute_grids(cosmo, survey, em_catalog)
    C = np.clip(np.asarray(grids.C_bar_raw, dtype=float), 0.0, 1.0)
    dN_exp = np.asarray(grids.dN_exp, dtype=float)
    zg = np.asarray(zgrid, dtype=float)
    # jnp.trapezoid to match every other quadrature in this module (numpy 1.26
    # has no np.trapezoid and np.trapz is deprecated in numpy 2).
    obs_per_pix = np.asarray(jnp.trapezoid(C * dN_exp, zg, axis=-1))  # () or (S,)
    exp_per_pix = float(jnp.trapezoid(dN_exp, zg))
    smap = (None if em_catalog.pixel_stratum_map is None
            else np.asarray(em_catalog.pixel_stratum_map).reshape(-1))
    if C.ndim == 2 and smap is not None:
        # Stratified: weight each stratum's curve by its share of the sky, and of
        # the footprint when the occupied set is known (else scale by the
        # footprint fraction, which is exact for a stratum-independent mask).
        w_all = np.bincount(smap, minlength=C.shape[0]).astype(float)
        if em_catalog.field_occupied_pixels is not None:
            occ = np.asarray(em_catalog.field_occupied_pixels).reshape(-1)
            w_occ = np.bincount(smap[occ], minlength=C.shape[0]).astype(float)
        else:
            w_occ = w_all * (float(n_occupied) / max(float(n_pix_total), 1.0))
        model_footprint = float(np.sum(w_occ * obs_per_pix))
        model_sky = float(np.sum(w_all * obs_per_pix))
    else:
        # One curve (or a stratified stack with no routing available: then the
        # stratum mean is the only defensible summary).
        per_pix = float(np.mean(obs_per_pix))
        model_footprint = float(n_occupied) * per_pix
        model_sky = float(n_pix_total) * per_pix
    N_obs_total = float(N_obs_total)
    N_exp_sky = float(n_pix_total) * exp_per_pix
    return {
        "N_obs_total": N_obs_total,
        "selection_model_N_obs_footprint": model_footprint,
        "selection_model_N_obs_sky": model_sky,
        "selection_model_over_observed_footprint": (
            model_footprint / N_obs_total if N_obs_total > 0 else float("inf")),
        # The completeness the catalog+model actually imply, for the record.
        "implied_completeness": (N_obs_total / N_exp_sky if N_exp_sky > 0
                                 else float("nan")),
    }


def _kernel_match_diagnostics(em_catalog: EMCatalog) -> dict[str, float]:
    """How well the completeness numerator's fixed kernel matches the catalog.

    The ratio ``C = dN_obs_s / dN_exp_s`` is exact for a constant true
    completeness only because ONE operator smooths both sides — but the
    numerator smooths the redshift POINT ESTIMATES, so a catalog whose
    ``dzgals`` are comparable to ``_SIGMA_SMOOTH`` is effectively smoothed with
    ``sqrt(_SIGMA_SMOOTH^2 + dz^2)`` on the observed side only (module
    docstring).  Report the ratio and warn when it is not small: the residual
    bias in C is a few percent there, which is an O(50%) relative error on the
    ``(1 - C)`` missing branch of a nearly complete pixel.
    """
    dz = np.asarray(em_catalog.dzgals, dtype=float)
    ng = np.asarray(em_catalog.ngals).reshape(-1)
    real = np.arange(dz.shape[-1])[None, :] < ng[:, None]
    dz_max = float(np.max(dz[real])) if bool(real.any()) else 0.0
    dz_mean = float(np.mean(dz[real])) if bool(real.any()) else 0.0
    ratio = dz_max / _SIGMA_SMOOTH
    if ratio > 0.5:
        warnings.warn(
            f"completeness kernel mismatch: the catalog's largest per-galaxy "
            f"redshift sigma is {dz_max:.4g} = {ratio:.2f} x the completeness "
            f"smoothing width _SIGMA_SMOOTH = {_SIGMA_SMOOTH:g}. The "
            "matched-kernel ratio C = dN_obs_s / dN_exp_s smooths the observed "
            "POINT ESTIMATES, so the observed side carries the extra dz "
            "convolution and the denominator does not: C is biased by a few "
            "percent wherever dN_exp/dz has curvature, i.e. by O(50%) "
            "relatively on the (1 - C) missing branch of a nearly complete "
            "pixel. Treat the completion of this catalog as approximate.",
            RuntimeWarning,
            stacklevel=3,
        )
    return {
        "sigma_smooth": float(_SIGMA_SMOOTH),
        "max_dz": dz_max,
        "mean_dz": dz_mean,
        "max_dz_over_sigma_smooth": ratio,
    }


# ------------------------------------------------------------
# Public scalar / vmapped API (checks & diagnostics; slow path)
# ------------------------------------------------------------

def _curves_scalar(z, pix, curves: CompletionCurves):
    """Interpolate ``(f, p_miss(z), C_eff(z))`` from precomputed per-row curves."""
    nm = jnp.where(curves.N_miss[pix] > 0.0, curves.N_miss[pix], 1.0)
    p_miss = jnp.interp(z, zgrid, curves.dN_miss[pix] / nm)
    C_eff = jnp.interp(z, zgrid, curves.C_eff[pix])
    return curves.f[pix], p_miss, C_eff


def catalog_completion(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Characterise catalog incompleteness at a single (z, pix) point.

    **Q_LSS-aware**: delegates to :func:`completion_curves` (the single source of
    truth that applies any LSS-conditioned completion table), then interpolates
    the per-row curve at ``z``.  Eager / diagnostic slow path (NOT jitted — the
    hot likelihood uses ``completion_curves`` via ``darksirens.redshift.prior``).

    Returns
    -------
    f : float — scalar number-weighted completeness fraction (diagnostic).
    p_miss : float — normalised missing-galaxy PDF at z.
    C : float — effective completeness C_eff(z|pix).
    """
    curves = completion_curves(cosmo, survey, em_catalog)
    return _curves_scalar(z, jnp.asarray(pix), curves)


def catalog_completion_vmap(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Vectorised ``catalog_completion`` over arrays of (z, pix) pairs.

    Q_LSS-aware (delegates to ``completion_curves``); eager / diagnostic slow
    path.  Hot paths use ``completion_curves`` via the state API in
    ``darksirens.redshift.prior``.
    """
    curves = completion_curves(cosmo, survey, em_catalog)
    z = jnp.asarray(z)
    pix = jnp.asarray(pix)
    return vmap(lambda z_i, p_i: _curves_scalar(z_i, p_i, curves))(z, pix)


# ------------------------------------------------------------
# LSS overdensity field
# ------------------------------------------------------------

def compute_lss_overdensity(
    zgals: jnp.ndarray,
    nside: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
    batch_size: int = 4096,
) -> jnp.ndarray:
    """
    Pre-compute the LSS overdensity field delta_g(pix, z) on ``zgrid``.

    Call once at startup; store the result in ``EMCatalog.delta_g_pix_z``.

    Construction
    ------------
    The mean density at each z is taken over *occupied* pixels only
    (pixels with at least one real galaxy), so survey footprints do not
    dilute the mean.  Pixels with no galaxies carry delta_g = 0: the
    observed catalog cannot distinguish a void from an unobserved region,
    and zero keeps the missing-galaxy density isotropic there (the
    completeness ratio already handles the incompleteness itself).  For
    the same reason delta_g -> 0 wherever the mean density vanishes
    (beyond the catalog depth).  No per-pixel mean subtraction is applied
    downstream; this construction is already centred.

    Caveat: delta_g is estimated from the *observed* galaxies, so in
    partially incomplete pixels it mixes true structure with the local
    selection.  ``b_miss`` absorbs the amplitude but not the z-dependence
    of that mixing.

    Computation is chunked (``batch_size`` pixels per JIT call) to bound
    peak memory at high nside.
    """
    import healpy as hp
    n_pix = hp.nside2npix(nside)

    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "compute_lss_overdensity requires either wgals or ngals to mask padded galaxies"
        )

    zgals_jax = jnp.asarray(zgals)
    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    _batch = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    kde = np.empty((n_pix, zgrid.size), dtype=np.float64)
    for start in range(0, n_pix, batch_size):
        stop = min(start + batch_size, n_pix)
        kde[start:stop] = np.asarray(
            _batch(
                jnp.arange(start, stop, dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )

    if ngals_jax is not None:
        occupied = np.asarray(ngals_jax) > 0
    else:
        occupied = np.asarray(jnp.any(wgals_jax > 0, axis=-1))
    occupied = occupied.astype(bool)[:n_pix]

    n_occ = max(int(occupied.sum()), 1)
    mean_density = kde[occupied].sum(axis=0) / n_occ if occupied.any() else np.zeros(zgrid.size)
    mean_safe = np.where(mean_density > 0.0, mean_density, 1.0)

    delta_g = (kde - mean_density[None, :]) / mean_safe[None, :]
    delta_g[~occupied] = 0.0
    delta_g[:, mean_density <= 0.0] = 0.0
    return jnp.asarray(delta_g)
