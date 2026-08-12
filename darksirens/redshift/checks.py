"""
checks.py
---------
Numerical correctness checks for the redshift prior.

None of the functions here are JIT-compiled.  They are diagnostic tools
intended to be called once at startup (or during testing) to catch
normalisation bugs early, before they silently corrupt inference results.

Why normalisation matters
~~~~~~~~~~~~~~~~~~~~~~~~~
Every quantity in this package is constructed to be normalised by
design.  In particular the assembled dark-sirens prior

    p(z | pix) = [N_obs * p_cat(z|pix) + dN_miss(z|pix)] / (N_obs + N_miss)

integrates to 1 per pixel exactly (up to grid quadrature of the narrow
catalog kernels).  A deviation here signals an implementation bug, not a
modelling tolerance: per-pixel normalisation errors vary across the sky
and cannot be absorbed by the selection-correction term.

Typical usage
~~~~~~~~~~~~~
    from darksirens.redshift.checks import run_all_checks

    run_all_checks(
        cosmo, survey, em_catalog,
        test_pixels=jnp.array([0, 100, 500]),
        verbose=True,
        raise_on_failure=True,
    )

Individual check functions are also exported for targeted use in unit
tests.
"""

from __future__ import annotations

import numpy as np

# numpy 2 renamed trapz -> trapezoid; the validated env floor is numpy 1.26,
# which only has trapz.  Support both (requirements admit >=1.26,<3).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

from darksirens.redshift.grid import zgrid
from darksirens.redshift.completion import catalog_completion_vmap
from .catalog import catalog_kernel_state, eval_log_catalog_prior_state
from .volume import log_volume_prior
from darksirens.redshift.prior import PRIOR_REGISTRY


#: Registry models whose prior is normalised at EVERY pixel, i.e. the ones a
#: per-pixel normalisation check answers for.  ``bright_sirens`` is deliberately
#: absent: it is ``-inf`` off the counterpart pixel BY CONSTRUCTION (a
#: counterpart localises the host to one row), so it integrates to 0 -- and
#: fails -- at every other test pixel; check it with ``test_pixels`` restricted
#: to the counterpart pixel.  ``spectral_sirens_wl`` is absent because it is the
#: same function as ``spectral_sirens`` (the WL marginalisation lives in the
#: likelihood), so checking it twice only doubles the runtime.
DEFAULT_CHECK_MODELS = ("spectral_sirens", "dark_sirens_complete", "dark_sirens")


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _integrate(log_p: np.ndarray, z: np.ndarray) -> float:
    """Trapezoidal integral of exp(log_p) over z; NaN in, NaN out.

    ``-inf`` is a legitimately zero density and exponentiates to 0.  NaN is
    NOT: it is the failure mode these checks exist to catch, so it must not be
    laundered into a density of 0 -- that made a NaN-poisoned prior integrate to
    slightly LESS than 1 and PASS the tolerance.  Returning NaN makes
    ``_check_result``'s ``abs(value - 1) <= atol`` False, i.e. a loud FAIL.
    """
    log_p = np.asarray(log_p, dtype=float)
    if np.isnan(log_p).any():
        return float("nan")
    p = np.exp(np.where(np.isneginf(log_p), -np.inf, log_p))
    return float(_trapezoid(p, z))


def _log_p_cat_grid(cosmo, survey, em_catalog, pix) -> np.ndarray:
    """``log p_cat(z | pix)`` on ``zgrid``, honouring ``survey.z_depth``.

    Goes through the STATE API the production likelihood uses: the one-shot
    ``log_catalog_prior`` ignores ``z_depth`` entirely, so under a survey depth
    it would integrate a different density than the run does (the truncated,
    below-depth-renormalised mixture).  With ``z_depth=None`` the two agree.
    """
    from jax import vmap

    state = catalog_kernel_state(cosmo, survey, em_catalog,
                                z_depth=survey.z_depth)
    pix_arr = jnp.full_like(zgrid, int(pix), dtype=jnp.int32)
    log_p = vmap(eval_log_catalog_prior_state, in_axes=(0, 0, None, None))(
        zgrid, pix_arr, state, em_catalog
    )
    return np.array(log_p)


def _check_result(label: str, value: float, atol: float, verbose: bool) -> bool:
    """
    Compare `value` to 1.0 within `atol`.  Print a pass/fail line and
    return True if the check passes, False otherwise.
    """
    passed = abs(value - 1.0) <= atol
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: integral = {value:.6f}  (target 1.0 ± {atol})")
    return passed


# ------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------

def check_volume_prior(
    cosmo: CosmoParams,
    survey: SurveyParams,
    *,
    atol: float = 0.01,
    verbose: bool = True,
) -> bool:
    """
    Verify that the volume prior integrates to 1 over zgrid.

    This should always pass unless there is a bug in the normalisation
    step inside ``log_volume_prior``.

    Parameters
    ----------
    cosmo, survey : CosmoParams, SurveyParams
    atol : float
        Absolute tolerance around 1.0.  Default 0.01 (1 %).
    verbose : bool
        Print a pass/fail line.

    Returns
    -------
    bool
        True if the check passes.
    """
    log_p = np.array([
        float(log_volume_prior(z, cosmo, survey)) for z in zgrid
    ])
    integral = _integrate(log_p, np.array(zgrid))
    return _check_result("volume prior", integral, atol, verbose)


def check_p_miss_normalization(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    test_pixels: jnp.ndarray,
    *,
    atol: float = 0.02,
    verbose: bool = True,
) -> dict[int, bool]:
    """
    Verify that the missing-galaxy PDF p_miss(z | pix) integrates to 1
    for each test pixel.

    p_miss is explicitly normalised inside ``catalog_completion``, so this
    should always pass.  A failure indicates a bug in the normalisation
    step or that the missing density is zero everywhere for that pixel
    (survey is effectively complete).

    Parameters
    ----------
    cosmo, survey, em_catalog : as in ``catalog_completion``
    test_pixels : jnp.ndarray, shape (K,)
        Pixel indices to test.
    atol : float
        Absolute tolerance around 1.0.  Default 0.02 (2 %).
    verbose : bool

    Returns
    -------
    dict mapping pixel index → bool (True = pass).
    """
    results = {}
    for pix in np.array(test_pixels):
        pix_arr  = jnp.full_like(zgrid, int(pix), dtype=jnp.int32)
        _, p_miss, _ = catalog_completion_vmap(
            zgrid, pix_arr, cosmo, survey, em_catalog
        )
        integral = _integrate(np.log(np.array(p_miss) + 1e-300), np.array(zgrid))
        passed = _check_result(f"p_miss  pix={int(pix):6d}", integral, atol, verbose)
        results[int(pix)] = passed
    return results


def check_C_eff_bounds(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    test_pixels: jnp.ndarray,
    *,
    verbose: bool = True,
) -> dict[int, bool]:
    """
    Verify that the completeness curve C_eff(z | pix) lies in [0, 1]
    for all z in zgrid for each test pixel.

    C_eff is clipped to [0, 1] inside the completion model, so this
    should always pass.  A failure would indicate that the clip is being
    bypassed or that interpolation is producing out-of-range values.

    Parameters
    ----------
    cosmo, survey, em_catalog : as in ``catalog_completion``
    test_pixels : jnp.ndarray, shape (K,)
    verbose : bool

    Returns
    -------
    dict mapping pixel index → bool (True = pass).
    """
    results = {}
    for pix in np.array(test_pixels):
        pix_arr  = jnp.full_like(zgrid, int(pix), dtype=jnp.int32)
        _, _, C_z = catalog_completion_vmap(
            zgrid, pix_arr, cosmo, survey, em_catalog
        )
        C = np.array(C_z)
        lo, hi = float(C.min()), float(C.max())
        passed = (lo >= -1e-6) and (hi <= 1.0 + 1e-6)
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"  [{status}] C_eff bounds pix={int(pix):6d}: "
                f"min={lo:.6f}  max={hi:.6f}  (expected [0, 1])"
            )
        results[int(pix)] = passed
    return results


def check_catalog_prior_normalization(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    test_pixels: jnp.ndarray,
    *,
    atol: float = 0.05,
    verbose: bool = True,
) -> dict[int, bool]:
    """
    Verify that p_cat(z | pix) integrates to ~1 over zgrid for each
    test pixel.

    p_cat is a weighted Gaussian mixture normalised to 1 over all z
    (-∞, +∞).  The integral over zgrid will be slightly less than 1 if
    galaxies sit near the edge of the grid; a value well below 1 signals
    that either the galaxy redshifts are outside zgrid or the kernel
    widths are too narrow.  A ``survey.z_depth`` is honoured (the mixture is
    then normalised on ``[0, z_depth]``, so the target is still 1).

    Parameters
    ----------
    cosmo, survey, em_catalog : as in ``log_catalog_prior``
    test_pixels : jnp.ndarray, shape (K,)
    atol : float
        Absolute tolerance around 1.0.  Default 0.05 (5 %).
    verbose : bool

    Returns
    -------
    dict mapping pixel index → bool (True = pass).
    """
    results = {}
    for pix in np.array(test_pixels):
        log_p = _log_p_cat_grid(cosmo, survey, em_catalog, int(pix))
        integral = _integrate(log_p, np.array(zgrid))
        passed = _check_result(
            f"p_cat   pix={int(pix):6d}", integral, atol, verbose
        )
        results[int(pix)] = passed
    return results


def check_prior_normalization(
    model: str,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    test_pixels: jnp.ndarray,
    *,
    atol: float = 0.02,
    verbose: bool = True,
) -> dict[int, bool]:
    """
    Verify that the assembled redshift prior integrates to ~1 over zgrid
    for each test pixel.

    For ``"dark_sirens"`` the prior is additive in densities:

        p(z|pix) = [N_obs * p_cat(z|pix) + dN_miss(z|pix)] / (N_obs + N_miss)

    and integrates to 1 per pixel by construction.  The residual is grid
    quadrature error from the narrow catalog kernels, so the default
    tolerance is tight (2 %); any larger deviation indicates an
    implementation bug.

    ``"bright_sirens"`` is normalised only at the counterpart pixel -- it is
    ``-inf`` everywhere else by construction -- so pass ``test_pixels`` holding
    that pixel alone (``em_catalog.counterpart_pixel``, or the row index of the
    first non-empty pixel when it is ``None``).  It is excluded from
    :data:`DEFAULT_CHECK_MODELS` for that reason.

    Parameters
    ----------
    model : str
        One of the keys in ``PRIOR_REGISTRY``.
    cosmo, survey, em_catalog : as above.
    test_pixels : jnp.ndarray, shape (K,)
    atol : float
        Absolute tolerance around 1.0.  Default 0.02 (2 %).
    verbose : bool

    Returns
    -------
    dict mapping pixel index → bool (True = pass).
    """
    if model not in PRIOR_REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. Available: {list(PRIOR_REGISTRY)}."
        )
    log_prior_fn = PRIOR_REGISTRY[model]

    results = {}
    for pix in np.array(test_pixels):
        pix_arr = jnp.full_like(zgrid, int(pix), dtype=jnp.int32)
        log_p   = np.array(
            log_prior_fn(zgrid, pix_arr, cosmo, survey, em_catalog)
        )
        integral = _integrate(log_p, np.array(zgrid))
        passed = _check_result(
            f"prior({model}) pix={int(pix):6d}", integral, atol, verbose
        )
        results[int(pix)] = passed
    return results


# ------------------------------------------------------------
# Omnibus runner
# ------------------------------------------------------------

def run_all_checks(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    test_pixels: jnp.ndarray,
    *,
    models: list[str] | None = None,
    atol_volume: float = 0.01,
    atol_p_miss: float = 0.02,
    atol_p_cat: float = 0.05,
    atol_prior: float = 0.02,
    verbose: bool = True,
    raise_on_failure: bool = False,
) -> dict[str, bool | dict]:
    """
    Run all normalisation and bounds checks and return a summary.

    Intended to be called once at the start of an inference run:

        from darksirens.redshift.checks import run_all_checks
        run_all_checks(cosmo, survey, em_catalog,
                       test_pixels=jnp.array([0, 100, 500]),
                       raise_on_failure=True)

    Parameters
    ----------
    cosmo, survey, em_catalog : model inputs.
    test_pixels : jnp.ndarray
        Pixel indices to use for per-pixel checks.
    models : list[str] or None
        Which prior models to check.  Defaults to
        :data:`DEFAULT_CHECK_MODELS` -- the registry entries that ARE normalised
        at every pixel.  ``bright_sirens`` must be requested explicitly, with
        ``test_pixels`` restricted to the counterpart pixel (see
        :func:`check_prior_normalization`).
    atol_volume, atol_p_miss, atol_p_cat, atol_prior : float
        Per-check tolerances (see individual check functions).
    verbose : bool
        Print pass/fail lines for each check.
    raise_on_failure : bool
        If True, raise ``RuntimeError`` after running all checks if any
        failed.  Recommended for CI and startup validation.

    Returns
    -------
    dict
        Nested summary: key → True (all passed) / False / per-pixel dict.
        Top-level key ``"all_passed"`` is True iff every check passed.
    """
    if models is None:
        models = list(DEFAULT_CHECK_MODELS)

    summary: dict = {}
    all_passed = True

    if verbose:
        print("\n" + "=" * 60)
        print("  redshift_prior: normalisation checks")
        print("=" * 60)

    # 1. Volume prior
    if verbose:
        print("\n-- Volume prior --")
    passed = check_volume_prior(cosmo, survey, atol=atol_volume, verbose=verbose)
    summary["volume_prior"] = passed
    all_passed = all_passed and passed

    # 2. p_miss normalisation
    if verbose:
        print("\n-- p_miss normalisation --")
    r = check_p_miss_normalization(
        cosmo, survey, em_catalog, test_pixels,
        atol=atol_p_miss, verbose=verbose,
    )
    summary["p_miss"] = r
    all_passed = all_passed and all(r.values())

    # 3. C_eff bounds
    if verbose:
        print("\n-- C_eff ∈ [0, 1] --")
    r = check_C_eff_bounds(
        cosmo, survey, em_catalog, test_pixels, verbose=verbose,
    )
    summary["C_eff_bounds"] = r
    all_passed = all_passed and all(r.values())

    # 4. Catalog prior normalisation
    if verbose:
        print("\n-- p_cat normalisation --")
    r = check_catalog_prior_normalization(
        cosmo, survey, em_catalog, test_pixels,
        atol=atol_p_cat, verbose=verbose,
    )
    summary["p_cat"] = r
    all_passed = all_passed and all(r.values())

    # 5. Full assembled prior normalisation (per model)
    for model in models:
        if verbose:
            print(f"\n-- Assembled prior: {model} --")
        r = check_prior_normalization(
            model, cosmo, survey, em_catalog, test_pixels,
            atol=atol_prior, verbose=verbose,
        )
        summary[f"prior_{model}"] = r
        all_passed = all_passed and all(r.values())

    summary["all_passed"] = all_passed

    if verbose:
        print("\n" + "=" * 60)
        outcome = "ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED"
        print(f"  {outcome}")
        print("=" * 60 + "\n")

    if raise_on_failure and not all_passed:
        failed = [k for k, v in summary.items()
                  if k != "all_passed" and (
                      v is False or
                      (isinstance(v, dict) and not all(v.values()))
                  )]
        raise RuntimeError(
            f"redshift_prior normalisation checks failed: {failed}\n"
            "Run with verbose=True for details."
        )

    return summary