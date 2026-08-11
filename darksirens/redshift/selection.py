"""Parametric magnitude-limited selection functions and their offline fit.

The differential completeness of a magnitude-limited survey is a property of
the SELECTION, not of the galaxy field: conditioned on detection at redshift
``z``, the apparent-magnitude distribution is the truncated luminosity
function regardless of how clustered the galaxies are (the thinning theorem).
This module provides

* analytic selection curves ``C_sel(z; theta)`` for a Gaussian and a Schechter
  luminosity function (JAX, evaluated in-likelihood by
  ``darksirens.redshift.completion`` under ``c_mode="selection"``), and
* an OFFLINE maximum-likelihood fit of ``theta`` from the survey's per-galaxy
  apparent magnitudes (numpy/scipy; consumed by ``darksirens_fit_selection``),
  whose Laplace covariance becomes the Gaussian prior on the sampled ``theta``.

h-scaling convention (the H0 firewall).  Absolute magnitudes are carried as
``M0hat = M0 - 5 log10 h`` (h = H0/100).  Because the tabulated luminosity
distance is EXACTLY proportional to 1/H0 (:func:`darksirens.utils.cosmology.
distance_modulus`), the combination ``M0 + DM(z)`` = ``M0hat + DM(z; H0=100)``
is H0-independent, so a selection curve built from ``m_lim - M0 - DM(z)``
carries no H0 information: magnitudes constrain the LF shape and the
completeness budget, never the Hubble constant.  The offline fit likewise
works in reference absolute magnitudes ``Mhat_i = m_i - DM(z_i; H0=100)``,
which equal ``M0hat + scatter`` independent of the (unknown) true H0.

What the firewall does NOT cover: Om0, w0, wa.  The cancellation above is exact
for H0 alone, because ``dL`` is exactly proportional to ``1/H0``.  The remaining
background parameters change the SHAPE of ``DM(z)``, and only its z-INDEPENDENT
part is absorbable by the fitted zero point, so the offline fit's background is a
provenance datum, not a free choice: it is stamped into ``meta`` and checked
against the package fiducial when a fit is loaded
(:func:`_validate_fit_background`).  The residual that remains once the fit and
the run share that fiducial is the genuinely z-dependent piece swept out by
SAMPLING Om0/w0/wa: at H0 = 100, ``DM`` moves by 0.012 mag at z = 0.05 and
0.110 mag at z = 0.5 between Om0 = 0.25 and 0.40 (0.021 / 0.134 mag between
w0 = -0.8 and -1.2), i.e. ~0.1 mag of non-absorbable shape across a
z = 0.05-0.5 catalog, against a Laplace sd of ~0.02 mag at catalog scale.  A
selection-mode run that samples the dark-energy parameters over their full prior
therefore carries a completeness zero point anchored at the fiducial rather than
re-fitted per proposal; removing it needs the fit sample's redshift distribution
persisted and the mean ``DM`` offset applied inside the curves, not a wider
prior.

Strata.  Real compilations have direction-dependent depth; the fit accepts a
stratum label per galaxy and returns one ``theta`` per stratum sharing the LF
shape convention.  The mock program uses a single stratum.

K-corrections.  Real surveys select on OBSERVED-frame apparent magnitude
``m_obs = M0 + DM(z) + K(z) + scatter`` while ``M0`` is rest-frame, so the
selection curve and the fit both accept a fixed K(z) template as polynomial
coefficients ``k_corr_coeffs = (c1, c2, ...)`` meaning ``K(z) = sum_j c_j
z^j``.  There is deliberately NO constant term: a c0 is exactly degenerate
with ``M0hat`` (both shift the mean magnitude at every z), so the template
is pinned to ``K(0) = 0``.  K depends only on z and a fixed reference color
-- no h enters -- so the H0 firewall below is untouched.  ``None``/empty
coefficients reproduce the K = 0 behaviour bit-identically.

Luminosity-function families.  ``gaussian`` (the mock program) models the
absolute magnitudes as an upper-truncated Gaussian; ``schechter`` (the
real-catalog family) models them as a Schechter LF whose completeness
denominator is integrated down to ``M_faint = Mstar_hat + M_faint_offset``.
The Schechter family carries NO K(z) template -- the pinned
:func:`c_sel_schechter` takes none, so a K-corrected Schechter fit could not
be consumed in-likelihood.

Where the faint cutoff does and does not act.  ``C_sel(z)`` is a RATIO of
upper incomplete gammas, ``Gamma(a, x_lim(z)) / Gamma(a, x_faint)`` with
``a = alpha + 1``; both arguments are strictly positive, so both integrals
converge for every real ``alpha``.  The cutoff exists to make the DENOMINATOR
-- the count of "all galaxies" -- finite when ``alpha <= -1``; the detected
side never needed it, because detection itself truncates at ``M_lim(z)``.
The offline fit uses exactly that: it normalizes each galaxy against its own
detection limit (plus an optional PARAMETER-FREE absolute-magnitude cut), so
its support edge never moves with the fitted ``Mstar_hat`` and the MLE stays
a regular interior optimum instead of an order statistic.  The corollary is
stated in :func:`_fit_schechter_truncated` and stamped into the fit's
``meta``: ``M_faint_offset`` is a pure PROTOCOL constant that the magnitudes
cannot constrain, while it multiplies the whole out-of-catalog budget.

Not covered here (documented limits): photometric-error convolution of the
magnitude likelihood, per-galaxy (color-dependent) K-corrections, extinction
-- the catalog is expected to carry dereddened magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc, gammaln, ndtr

from darksirens.utils.cosmology import (
    Om0Planck,
    distance_modulus,
    w0Fiducial,
    waFiducial,
)

#: Reference Hubble constant of the h-scaled magnitude convention.
H0_REF = 100.0

#: Faint-end integration cutoff of the Schechter family, as an offset from
#: M*: ``M_faint = Mstar_hat + M_faint_offset``.  A magnitude DIFFERENCE, so it
#: carries no h and never touches the H0 firewall.  5.0 mag = 0.01 L*.  It is a
#: PROTOCOL constant, not a fitted parameter: it defines what counts as "a
#: galaxy" and therefore fixes the completeness denominator, so the fit, the
#: Q-table base and the likelihood must all carry the SAME value (stamped and
#: hard-checked).
M_FAINT_OFFSET_DEFAULT = 5.0

#: Floor of the legal ``M_faint_offset`` range.  NEGATIVE offsets are legal:
#: they declare a BRIGHT-truncated modelled population (``M_faint`` bright-ward
#: of M*, e.g. a mock whose luminosities are drawn above a cut ``x_cut L*``
#: with ``x_cut = 10^{-0.4 M_faint_offset} > 1``) -- the denominator
#: ``Gamma(alpha+1, x_faint)`` is finite for every real offset because
#: ``x_faint = 10^{-0.4 offset}`` is strictly positive either way.  The floor
#: is NUMERICAL, not physical: by ``x_faint ~ 630`` (offset = -7) the
#: ``e^{-x_faint}`` factor is within a decade of the f64 underflow wall at
#: ``x ~ 745`` and :func:`_upper_gamma_scaled`'s recurrence difference has
#: shed ~``log10(x_faint)`` leading digits, so past it the curve would
#: degrade silently instead of failing.
_M_FAINT_OFFSET_MIN = -7.0


def _validate_m_faint_offset(value, where):
    """Shared legality check of the protocol constant (see the floor above)."""
    v = float(value)
    if not np.isfinite(v) or v <= _M_FAINT_OFFSET_MIN:
        raise ValueError(
            f"{where}: M_faint_offset={value} must be finite and greater "
            f"than {_M_FAINT_OFFSET_MIN}. Negative offsets are legal -- they "
            "declare a BRIGHT-truncated modelled population (M_faint = "
            "Mstar_hat + M_faint_offset bright-ward of M*) -- but below the "
            "floor the completeness denominator Gamma(alpha+1, "
            "10^(-0.4 offset)) underflows f64 and the curve would degrade "
            "silently.")
    return v

#: Per-family theta carried by a selection-fit stratum, in the order the
#: Q-table provenance check and the ``selection_<name>`` stamps use.  Single
#: source of truth for cli/build_lognormal_completion.py, cli/inference.py and
#: catalogs/lss.py.
SELECTION_THETA_FIELDS = {
    "gaussian": ("m_lim", "M0hat", "sigma_M"),
    "schechter": ("m_lim", "Mstar_hat", "alpha", "M_faint_offset"),
}

#: The SAMPLED coordinates of each family, in the row/column order of
#: :attr:`SelectionFit.cov`.  The inference CLI reads the marginal Laplace sds
#: off the diagonal in exactly this order.
SELECTION_SAMPLED_FIELDS = {
    "gaussian": ("M0hat", "sigma_M"),
    "schechter": ("Mstar_hat", "alpha"),
}

SELECTION_FAMILIES = tuple(SELECTION_THETA_FIELDS)      # ("gaussian", "schechter")

#: Faint-end slope floor.  NOT the physics: ``Gamma(a, x_lim)/Gamma(a, x_faint)``
#: is finite for every real ``a = alpha + 1`` because both arguments are
#: strictly positive.  It is the SPELLING that has a domain -- ``gammaincc``
#: implements the REGULARIZED ratio, defined only for a positive first argument
#: -- and :func:`_upper_gamma_scaled` buys back one unit of it with the
#: ``Gamma(a+1, x) = a Gamma(a, x) + x^a e^-x`` recurrence, so the family
#: carries ``alpha > -2``.  That covers every measured galaxy faint-end slope
#: (2MASS K -1.02, SDSS r -1.05, GLADE B -1.21) with margin.  Enforced by the
#: offline fit, by the JSON loader, by the sampled ``alpha`` bounds
#: (inference/prior.py, which refuses a --prior_overrides floor at or below it)
#: and, as the last wall, by :func:`c_sel_schechter` itself.
_ALPHA_MIN = -2.0

#: ``|alpha + 1|`` below which the recurrence's REMOVABLE singularity is nudged
#: aside.  ``Gamma(a, x)`` is entire in ``a`` at fixed ``x > 0``, so ``C_sel``
#: is analytic through ``alpha = -1``; only the ``a Gamma(a, x)`` spelling
#: reads 0/0 there.  1e-6 keeps ~10 digits in the recurrence's difference and
#: moves the curve by O(1e-6) -- orders below any completeness the likelihood
#: resolves -- so the alpha ~ -1.0 slopes of real K/r-band LFs evaluate
#: normally instead of hitting a wall.
_A_NEAR_ZERO = 1e-6

#: Fraction of the fit sample whose own detection limit already reaches past
#: the modelled faint cutoff, above which an undeclared ``m_faint_cut`` is
#: worth a warning: over that sub-sample the fit's LF has no faint edge while
#: the completion's does.
_DEEP_WARN = 0.02


def m0_absolute(M0hat, H0):
    """Absolute magnitude from its h-scaled form: ``M0 = M0hat + 5 log10 h``."""
    return M0hat + 5.0 * jnp.log10(H0 / H0_REF)


def k_of_z(z, k_corr_coeffs, xp=jnp):
    """Polynomial K-correction template ``K(z) = sum_j c_j z^j`` (no c0).

    ``k_corr_coeffs`` is ``(c1, c2, ...)``; ``None`` or empty means K = 0.
    ``xp`` selects the array module so the same helper serves the JAX
    likelihood path and the numpy offline fit.
    """
    if not k_corr_coeffs:
        return None
    z = xp.asarray(z)
    out = xp.zeros_like(z)
    for c in reversed(tuple(k_corr_coeffs)):
        out = z * (c + out)
    return out


def c_sel_gaussian(z, m_lim, M0hat, sigma_M, H0, Om0=Om0Planck,
                   w0=w0Fiducial, wa=waFiducial, k_corr_coeffs=None):
    """Gaussian-LF selection ``P(m <= m_lim | z) = Phi((m_lim - M0 - DM - K)/sigma)``.

    ``M0hat`` is h-scaled; the ``+5 log10 h`` restored here cancels the
    ``-5 log10 h`` inside ``DM`` exactly, so the returned curve is
    H0-invariant to float precision (pinned by
    tests/test_selection_function.py).  At ``z -> 0`` the modulus diverges
    negatively and ``C -> 1``: a magnitude limit misses nothing nearby.
    ``k_corr_coeffs`` (structural, never traced) adds the fixed K(z)
    template; ``None`` is the K = 0 legacy path, bit-identical.
    """
    dm = distance_modulus(z, H0, Om0, w0, wa)
    M0 = m0_absolute(M0hat, H0)
    kz = k_of_z(z, k_corr_coeffs)
    if kz is not None:
        dm = dm + kz
    return ndtr((m_lim - M0 - dm) / sigma_M)


def _a_off_zero(a, xp=jnp):
    """Nudge ``a = alpha + 1`` off the removable singularity at ``a = 0``.

    Idempotent, so a caller that needs both ``a Gamma(a, x)`` and ``a`` itself
    (the offline fit divides one by the other) gets the SAME nudged value from
    two independent calls.  See :data:`_A_NEAR_ZERO` for why nudging rather
    than walling is the honest treatment.
    """
    return xp.where(xp.abs(a) < _A_NEAR_ZERO, _A_NEAR_ZERO, a)


def _upper_gamma_scaled(a, x, xp=jnp):
    """``a * Gamma(a, x)`` -- the UNREGULARIZED upper incomplete gamma, pole
    factored out, valid for ``a > -1`` (i.e. ``alpha > -2``).

    ``Gamma(a+1, x) = a Gamma(a, x) + x^a e^{-x}`` exactly, so

        a * Gamma(a, x) = Gamma(a+1) Q(a+1, x) - x^a e^{-x}

    is evaluated entirely inside ``gammaincc``'s ``a + 1 > 0`` domain even
    where the TARGET ``a`` is zero or negative -- which is where every real
    galaxy faint-end slope lives.  The leading ``a`` is deliberately left in
    the result: :func:`c_sel_schechter` forms a ratio at fixed ``a``, in which
    it cancels exactly, so the curve has no pole at ``alpha = -1`` at all,
    only the removable singularity :func:`_a_off_zero` steps around.  ``xp``
    selects the array module (the :func:`k_of_z` idiom) so the JAX curve and
    the numpy offline fit share one implementation and one convention.
    """
    if xp is np:
        from scipy.special import gammaincc as _gammaincc, gammaln as _gammaln
    else:
        _gammaincc, _gammaln = gammaincc, gammaln
    a = _a_off_zero(a, xp)
    return (xp.exp(_gammaln(a + 1.0)) * _gammaincc(a + 1.0, x)
            - x ** a * xp.exp(-x))


def c_sel_schechter(z, m_lim, Mstar_hat, alpha, M_faint_offset, H0,
                    Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Schechter-LF selection: a ratio of UPPER INCOMPLETE GAMMAS.

    ``C_sel(z) = Gamma(alpha+1, x_lim(z)) / Gamma(alpha+1, x_faint)`` with
    ``x = L/L* = 10^{-0.4 (M - M*)}``, ``M_lim(z) = m_lim - DM(z)`` and the
    faint-end integration cutoff ``M_faint = M* + M_faint_offset``.  Both
    arguments are strictly positive, so BOTH integrals converge for every real
    ``alpha``: the cutoff is what keeps the denominator -- the count of every
    galaxy the completeness answers for -- finite once ``alpha <= -1``, and
    the numerator never needed it.  Written through
    :func:`_upper_gamma_scaled`, whose ``Gamma(a)`` factors cancel in the
    ratio, so the curve carries the ``alpha ~ -1.0 to -1.3`` slopes real
    catalogs measure.  ``Mstar_hat`` is h-scaled like ``M0hat``.

    The result is clipped into [0, 1]: ``x_lim < x_faint`` means the survey
    reaches FAINTER than the modelled population and is complete there.

    ``M_faint_offset`` may be NEGATIVE: that declares a BRIGHT-truncated
    modelled population (``M_faint`` bright-ward of M*, i.e. luminosities
    drawn above ``x_cut = 10^{-0.4 M_faint_offset} > 1`` in ``L*`` units --
    the mock-with-a-luminosity-cut case).  Nothing in the ratio changes:
    ``x_faint`` stays strictly positive, the denominator counts the truncated
    population, and the clip handles the (now larger) complete regime.  The
    legality floor is :data:`_M_FAINT_OFFSET_MIN` (f64 underflow of
    ``e^{-x_faint}``), enforced by the fit/loader/CLI walls, not here.

    Below ``alpha = _ALPHA_MIN`` the curve is NaN, not a number: that is the
    edge of the recurrence's one lift, and a silent wrong answer there would
    read as an identically COMPLETE survey -- the one failure mode with no
    visible symptom.  NaN propagates to ``logL = -inf`` the way an off-grid
    cosmology does.  This is the last wall, not the first: the offline fit,
    the JSON loader and the sampled ``alpha`` bounds all refuse
    ``alpha <= _ALPHA_MIN`` up front.
    """
    dm = distance_modulus(z, H0, Om0, w0, wa)
    Mstar = m0_absolute(Mstar_hat, H0)
    x_lim = 10.0 ** (-0.4 * (m_lim - dm - Mstar))
    x_faint = 10.0 ** (-0.4 * M_faint_offset)
    a = alpha + 1.0
    num = _upper_gamma_scaled(a, x_lim)
    den = _upper_gamma_scaled(a, x_faint)
    ratio = jnp.clip(num / den, 0.0, 1.0)
    return jnp.where(alpha > _ALPHA_MIN, ratio, jnp.nan)


def reference_absolute_mags(m, z, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                            k_corr_coeffs=None):
    """``Mhat_i = m_i - DM(z_i; H0=100) - K(z_i)`` -- h-scaled absolute mags.

    Independent of the true H0: with ``m = M0 + scatter + DM(z; H0_true) +
    K(z)``, ``Mhat = M0 - 5 log10 h_true + scatter = M0hat + scatter``
    exactly (K cancels its own template, contributing no h).
    """
    dm = np.asarray(distance_modulus(jnp.asarray(z, dtype=float), H0_REF,
                                     Om0, w0, wa))
    kz = k_of_z(z, k_corr_coeffs, xp=np)
    if kz is not None:
        dm = dm + kz
    return np.asarray(m, dtype=float) - dm


def magnitude_suffstats(m, z, m_lim, n_bins=64, Om0=Om0Planck,
                        w0=w0Fiducial, wa=waFiducial, k_corr_coeffs=None):
    """Compress the magnitude sample into per-truncation-bin statistics.

    The truncated-Gaussian likelihood has a PER-GALAXY truncation
    ``T_i = m_lim - DM(z_i; H0=100)``, so the naive (N, sum x, sum x^2)
    reduction is not exact.  Binning galaxies by T restores a finite
    representation: within a bin the truncation is approximated by the
    bin-mean T, and the joint likelihood becomes a sum of per-bin terms in
    (N_b, sum Mhat_b, sum Mhat_b^2, T_b) -- exact in the fine-bin limit
    (the approximation error is O(bin width^2) in log Phi's curvature).

    This is the cheap exact-enough carrier of the FULL magnitude likelihood
    for the joint-term cross-check (the Gaussian-prior path is the default;
    see the module docstring).
    """
    m, z = _apply_z_floor(m, z, "magnitude_suffstats")
    Mhat = reference_absolute_mags(m, z, Om0, w0, wa, k_corr_coeffs)
    T = np.asarray(m_lim, dtype=float) - (np.asarray(m, dtype=float) - Mhat)
    edges = np.linspace(T.min() - 1e-9, T.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(T, edges) - 1, 0, n_bins - 1)
    stats = {"n": np.zeros(n_bins), "sum": np.zeros(n_bins),
             "sumsq": np.zeros(n_bins), "T": np.zeros(n_bins)}
    np.add.at(stats["n"], idx, 1.0)
    np.add.at(stats["sum"], idx, Mhat)
    np.add.at(stats["sumsq"], idx, Mhat ** 2)
    np.add.at(stats["T"], idx, T)
    keep = stats["n"] > 0
    return {k: v[keep] for k, v in stats.items()} | {
        "T": stats["T"][keep] / stats["n"][keep]}


def magnitude_loglike_from_stats(M0hat, sigma_M, stats):
    """Truncated-Gaussian magnitude log-likelihood from the binned statistics.

    ``sum_b [ -N_b log sigma - (S2_b - 2 mu S1_b + N_b mu^2)/(2 sigma^2)
              - N_b log Phi((T_b - mu)/sigma) ]``  (+ const).
    JAX-differentiable in (M0hat, sigma_M); usable as an explicit joint term.
    """
    n, s1, s2, T = (jnp.asarray(stats[k]) for k in ("n", "sum", "sumsq", "T"))
    mu, sig = M0hat, sigma_M
    quad = (s2 - 2.0 * mu * s1 + n * mu ** 2) / (2.0 * sig ** 2)
    log_trunc = jnp.log(jnp.maximum(ndtr((T - mu) / sig), 1e-300))
    return jnp.sum(-n * jnp.log(sig) - quad - n * log_trunc)


#: Selection-fit JSON formats this module reads: 1.0 is the single-stratum
#: original; 1.1 adds multi-entry ``strata`` (written by
#: ``darksirens_fit_selection --strata`` when more than one stratum is fit --
#: a single-stratum output stays byte-compatible 1.0).
_SELECTION_FIT_FORMATS = ("darksirens-selection-fit-1.0",
                          "darksirens-selection-fit-1.1")


#: Keys every stratum of a given family must carry: the family tag, the fixed
#: truncation datum, that family's own theta and the Laplace covariance.
_STRATUM_REQUIRED = {
    "gaussian": ("family", "m_lim", "M0hat", "sigma_M", "cov"),
    "schechter": ("family", "m_lim", "Mstar_hat", "alpha", "M_faint_offset",
                  "cov"),
}


#: Tolerance of the loaded fit's background-cosmology provenance check.  The
#: stamp is a float64 round-trip of the fiducial constant, so anything above
#: rounding is a different background.
_BACKGROUND_ATOL = 1e-9

#: The background the consumed curves are anchored at: the sampled cosmology's
#: prior centre, and the default the offline fit runs at.
_FIT_BACKGROUND = (("Om0", Om0Planck), ("w0", w0Fiducial), ("wa", waFiducial))


def _validate_fit_background(where, meta):
    """Refuse a fit whose stamped background is not the run's fiducial.

    The h-scaled zero point absorbs H0 EXACTLY (module docstring), but Om0/w0/wa
    change the shape of ``DM(z)``, and only the z-independent part of that change
    is absorbable by the fitted ``M0hat``/``Mstar_hat``.  A fit measured at
    another background is therefore not the fit this run consumes -- ``c_sel_*``
    is evaluated at the run's cosmology while the zero point stays anchored where
    it was measured -- and nothing downstream reads the stamp, so the mismatch
    would be silent.  Fits written before the stamp existed carry no keys and
    are accepted unchanged.
    """
    for key, fiducial in _FIT_BACKGROUND:
        if key not in meta:
            continue
        value = float(meta[key])
        if abs(value - float(fiducial)) > _BACKGROUND_ATOL:
            raise ValueError(
                f"{where}: the fit was measured at {key}={value} but this "
                f"package's background is {key}={float(fiducial)}. The h-scaled "
                "zero point absorbs H0 exactly, NOT Om0/w0/wa: they change the "
                "shape of DM(z), so the fitted M0hat/Mstar_hat does not carry "
                "over (~0.1 mag of non-absorbable residual across a "
                "z = 0.05-0.5 catalog, several times the fit's own Laplace sd). "
                "Refit the magnitudes at the fiducial background.")


def _validate_sampled_prior_numerics(where, family, cov, sigma_M):
    """Numeric legality of the quantities that BECOME the sampled theta prior.

    ``cov`` is consumed POSITIONALLY -- the inference CLI reads the marginal
    Laplace sds off its diagonal in ``SELECTION_SAMPLED_FIELDS[family]`` order --
    so a wrong-shape matrix silently mis-maps the priors, a negative diagonal
    entry yields a NaN prior sd, and a 1x1 matrix raises an opaque IndexError far
    from the file that caused it.  ``sigma_M`` divides the gaussian selection
    curve: 0 turns it into a step function and a negative value INVERTS it (a
    survey MORE complete at high z), with no error anywhere downstream.  The fit
    path is safe by construction (it optimises log sigma and rejects a non-PD
    Hessian); this wall is for hand-edited, merged or externally generated JSON.
    """
    n = len(SELECTION_SAMPLED_FIELDS[family])
    fields = list(SELECTION_SAMPLED_FIELDS[family])
    C = np.asarray(cov, dtype=float)
    if C.shape != (n, n) or not np.all(np.isfinite(C)):
        raise ValueError(
            f"{where}: cov must be a finite ({n}, {n}) matrix in {fields} "
            f"order (the sampled theta of the {family!r} family); got shape "
            f"{C.shape}.")
    scale = max(1.0, float(np.max(np.abs(C))))
    if (not np.allclose(C, C.T, rtol=0.0, atol=1e-10 * scale)
            or float(np.min(np.linalg.eigvalsh(C))) <= 0.0):
        raise ValueError(
            f"{where}: cov is not symmetric positive definite; it becomes the "
            f"Gaussian prior on {fields}, whose sds are the square roots of its "
            "diagonal.")
    if family == "gaussian" and not float(sigma_M) > 0.0:
        raise ValueError(
            f"{where}: sigma_M={sigma_M} must be positive -- c_sel_gaussian "
            "divides by it, so 0 makes the selection curve a step function and "
            "a negative width inverts it (completeness RISING with redshift).")


def _validate_stratum(path, s):
    s = dict(s)
    if "family" not in s:
        raise ValueError(f"{path}: stratum missing required key 'family'.")
    family = s["family"]
    if family not in _STRATUM_REQUIRED:
        raise NotImplementedError(
            f"{path}: family {family!r}; consumers support "
            f"{list(SELECTION_FAMILIES)}.")
    for key in _STRATUM_REQUIRED[family]:
        if key not in s:
            raise ValueError(f"{path}: stratum missing required key {key!r}.")
    _validate_fit_background(str(path), s.get("meta") or {})
    _validate_sampled_prior_numerics(str(path), family, s["cov"],
                                    s.get("sigma_M"))
    # Optional K(z) template; absent in pre-K fit files -> K = 0.
    s["k_corr_coeffs"] = tuple(float(c) for c in s.get("k_corr_coeffs") or ())
    if family == "schechter":
        if s["k_corr_coeffs"]:
            raise NotImplementedError(
                f"{path}: a schechter fit carries a K(z) template "
                f"{list(s['k_corr_coeffs'])}, but the pinned c_sel_schechter "
                "takes no k_corr_coeffs, so the template could not be applied "
                "in-likelihood; refit the gaussian family with "
                "--k_corr_coeffs, or extend c_sel_schechter (and its "
                "H0-invariance pin) first.")
        if float(s["alpha"]) <= _ALPHA_MIN:
            raise ValueError(
                f"{path}: alpha={float(s['alpha'])} is at or below the "
                f"{_ALPHA_MIN} floor; c_sel_schechter reaches a <= 0 with ONE "
                "recurrence step off gammaincc's positive-argument domain, so "
                "alpha + 2 > 0 is the edge of the spelling (every measured "
                "galaxy faint-end slope is well inside it).")
        _validate_m_faint_offset(s["M_faint_offset"], str(path))
    return s


def _load_selection_fit_payload(path):
    import json

    with open(path) as f:
        payload = json.load(f)
    fmt = payload.get("format_version")
    if fmt not in _SELECTION_FIT_FORMATS:
        raise ValueError(
            f"{path}: unknown selection-fit format {fmt!r} (expected one of "
            f"{_SELECTION_FIT_FORMATS} from darksirens_fit_selection).")
    return payload


def load_selection_fit_json(path):
    """Load and validate a ``darksirens_fit_selection`` JSON; return the
    single-stratum theta dict ``{family, m_lim, M0hat, sigma_M, cov, ...}``.

    Multi-stratum payloads are rejected HERE by contract: the single-survey
    builder and the K=1 likelihood carry one theta.  Multi-stratum consumers
    use :func:`load_selection_fit_strata`.
    """
    payload = _load_selection_fit_payload(path)
    strata = payload.get("strata") or []
    if len(strata) != 1:
        raise NotImplementedError(
            f"{path}: {len(strata)} strata; this consumer carries exactly one "
            "selection stratum (use load_selection_fit_strata for the "
            "stratified path).")
    return _validate_stratum(path, strata[0])


def load_selection_fit_strata(path):
    """Load a selection-fit JSON as a LIST of validated stratum dicts.

    Accepts both the 1.0 single-stratum format and the 1.1 multi-stratum
    format; every entry carries the same keys as
    :func:`load_selection_fit_json`'s return.  Strata are returned in file
    order (the fit CLI writes them sorted by stratum label).
    """
    payload = _load_selection_fit_payload(path)
    strata = payload.get("strata") or []
    if not strata:
        raise ValueError(f"{path}: no strata in selection-fit payload.")
    out = [_validate_stratum(path, s) for s in strata]
    families = {s["family"] for s in out}
    if len(families) > 1:
        raise NotImplementedError(
            f"{path}: strata mix the {sorted(families)} luminosity-function "
            "families; one fit describes ONE population model, and the "
            "in-likelihood curve stack is built from a single family.")
    if len(out) > 1 and families != {"gaussian"}:
        raise NotImplementedError(
            f"{path}: multi-stratum fits are gaussian-only for now (got "
            f"{len(out)} strata of family {sorted(families)[0]!r}); the "
            "stratified in-likelihood curve stack "
            "(SurveyParams.selection_strata) is built from the gaussian "
            "common-mode + offset decomposition, which has no schechter "
            "counterpart. Fit the strata separately, or fit the whole "
            "catalog as one schechter stratum.")
    return out


@dataclass
class SelectionFit:
    """One stratum's fitted selection parameters (gaussian or schechter LF).

    The FIELD ORDER of the gaussian block is frozen (positional construction
    is part of the API); the schechter fields are appended with defaults and
    the per-family required set is enforced in :meth:`__post_init__` -- the
    same contract :func:`_validate_stratum` applies to a loaded payload.
    """

    family: str
    m_lim: float
    M0hat: float | None = None
    sigma_M: float | None = None
    #: (2, 2) Laplace covariance in ``SELECTION_SAMPLED_FIELDS[family]`` order.
    cov: np.ndarray | None = None
    n_gal: int = 0
    stratum: str = "all"
    k_corr_coeffs: tuple = ()
    meta: dict = field(default_factory=dict)
    Mstar_hat: float | None = None
    alpha: float | None = None
    M_faint_offset: float | None = None

    def __post_init__(self):
        if self.family not in SELECTION_THETA_FIELDS:
            raise NotImplementedError(
                f"unknown selection family {self.family!r}; supported: "
                f"{list(SELECTION_FAMILIES)}.")
        if self.cov is None:
            raise ValueError(
                f"SelectionFit(family={self.family!r}) needs the Laplace "
                "covariance of "
                f"{list(SELECTION_SAMPLED_FIELDS[self.family])}; it is what "
                "becomes the sampled theta prior.")
        required = set(SELECTION_THETA_FIELDS[self.family]) | set(
            SELECTION_SAMPLED_FIELDS[self.family])
        for name in sorted(required):
            if getattr(self, name) is None:
                raise ValueError(
                    f"SelectionFit(family={self.family!r}) is missing {name!r}"
                    f" (the family carries "
                    f"{list(SELECTION_THETA_FIELDS[self.family])}).")
        _validate_sampled_prior_numerics(
            f"SelectionFit(family={self.family!r})", self.family, self.cov,
            self.sigma_M)
        if self.family == "schechter":
            if float(self.alpha) <= _ALPHA_MIN:
                raise ValueError(
                    f"alpha={float(self.alpha)} is at or below the "
                    f"{_ALPHA_MIN} floor: c_sel_schechter's upper-gamma "
                    "recurrence carries alpha + 2 > 0.")
            _validate_m_faint_offset(self.M_faint_offset, "SelectionFit")
            if tuple(self.k_corr_coeffs):
                raise ValueError(
                    "the schechter family carries no K(z) template (the "
                    "pinned c_sel_schechter takes no k_corr_coeffs), so a "
                    "K-corrected schechter fit could not be consumed "
                    "in-likelihood.")

    def to_jsonable(self) -> dict:
        """Payload dict: ``family``, ``m_lim``, the family's own theta, then
        the shared tail.  For the gaussian family this reproduces the
        pre-schechter key order exactly (pinned by a test)."""
        out = {"family": self.family, "m_lim": self.m_lim}
        for name in SELECTION_THETA_FIELDS[self.family][1:]:      # m_lim first
            out[name] = getattr(self, name)
        out.update({
            "cov": np.asarray(self.cov).tolist(), "n_gal": self.n_gal,
            "stratum": self.stratum,
            "k_corr_coeffs": [float(c) for c in self.k_corr_coeffs],
            "meta": dict(self.meta),
        })
        return out


#: Redshift floor for the magnitude fit and suff-stats: recorded survey
#: redshifts are PHOTOMETRIC and the mock generator deliberately leaves
#: negative realizations unclipped, so z <= 0 gives NaN distance moduli
#: (crashing the fit) and tiny positive z gives DM -> -inf outliers that
#: silently drag the MLE.  Galaxies below the floor are DROPPED and counted;
#: the un-modeled photo-z scatter at low z is a documented limitation.
Z_FLOOR = 0.01


def _apply_z_floor(m, z, where):
    m = np.asarray(m, dtype=float)
    z = np.asarray(z, dtype=float)
    keep = np.isfinite(m) & np.isfinite(z) & (z >= Z_FLOOR)
    n_drop = int((~keep).sum())
    if n_drop:
        import warnings

        warnings.warn(
            f"{where}: dropped {n_drop}/{m.size} galaxies below the "
            f"z >= {Z_FLOOR} floor (photometric redshifts near/below zero "
            "have no usable distance modulus).", RuntimeWarning)
    return m[keep], z[keep]


def fit_selection_from_mags(m, z, m_lim, *, family="gaussian",
                            M_faint_offset=M_FAINT_OFFSET_DEFAULT,
                            m_faint_cut=None,
                            Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                            stratum="all", k_corr_coeffs=None):
    """Truncated-LF maximum likelihood for one stratum (offline, numpy/scipy).

    The data are per-galaxy apparent magnitudes ``m_i`` at redshifts ``z_i``
    with a KNOWN hard limit ``m_i <= m_lim`` (the truncation datum of the
    selection protocol -- not a fitted parameter: inside the selection curve
    only the combination ``m_lim - M0hat`` / ``m_lim - Mstar_hat`` is
    identified, so fitting both is an exact flat direction).  In reference
    absolute magnitudes the per-galaxy truncation is
    ``T_i = m_lim - DM(z_i; H0=100)`` and the model is

    * ``family="gaussian"``: an upper-truncated Gaussian,

          L(theta) = prod_i  phi((Mhat_i - M0hat)/sigma) / sigma
                             / Phi((T_i - M0hat)/sigma) ,

    * ``family="schechter"``: a Schechter LF normalized against each galaxy's
      own detection limit, plus the optional parameter-free absolute-magnitude
      cut ``m_faint_cut`` (h-scaled, applied to ``Mhat``).  See
      :func:`_fit_schechter_truncated` for the closed-form normalization and
      for why ``M_faint_offset`` -- the completeness DENOMINATOR's protocol
      constant -- is not and cannot be fitted from magnitudes.

    Returns the MLE and the Laplace covariance from the numerical Hessian at
    the optimum (finite differences of the exact gradient-free objective).
    The z-shape of the sample never enters: this likelihood is exactly
    independent of the galaxy density field (thinning), which is what makes
    the fitted selection clustering-safe.
    """
    if family not in SELECTION_FAMILIES:
        raise NotImplementedError(
            f"unknown selection family {family!r}; the offline fit implements "
            f"{list(SELECTION_FAMILIES)}.")
    if family == "schechter":
        if k_corr_coeffs:
            raise NotImplementedError(
                "the pinned c_sel_schechter carries no K(z) template, so a "
                "K-corrected Schechter fit could not be consumed "
                "in-likelihood; fit the gaussian family with "
                "--k_corr_coeffs, or extend c_sel_schechter (and its "
                "H0-invariance pin) first.")
        _validate_m_faint_offset(M_faint_offset, "fit_selection_from_mags")
        if float(M_faint_offset) < 0.0 and m_faint_cut is None:
            # A bright-truncated protocol is a POPULATION claim: no galaxies
            # exist faint-ward of M_faint < M*.  Fitting such a sample with
            # an un-cut faint end does not fail -- the MLE quietly inverts
            # the slope (alpha driven positive, M* dragged faint) to explain
            # the missing faint counts, so the mismatch must be refused, not
            # warned about.
            raise ValueError(
                f"M_faint_offset={float(M_faint_offset)} declares a BRIGHT-"
                "truncated modelled population (M_faint = Mstar_hat + "
                "M_faint_offset bright-ward of M*: nothing exists past the "
                "cut), but no m_faint_cut was declared, so the fit would "
                "model an LF with no faint edge against a sample that has "
                "one -- the MLE then inverts the faint-end slope instead of "
                "failing. Pass --m_faint_cut at the population's edge (the "
                "h-scaled absolute magnitude of the truncation).")
    elif m_faint_cut is not None:
        raise NotImplementedError(
            f"m_faint_cut={float(m_faint_cut)} is a schechter-family option: "
            "the gaussian LF is already normalizable faint-ward, so its fit "
            "has no absolute-magnitude cut to declare.")

    m = np.asarray(m, dtype=float)
    z = np.asarray(z, dtype=float)
    if m.shape != z.shape or m.ndim != 1:
        raise ValueError("need matching 1-D m, z arrays")
    m, z = _apply_z_floor(m, z, "fit_selection_from_mags")
    if m.size < 10:
        raise ValueError("need at least 10 galaxies above the z floor")
    over = m > m_lim + 1e-9
    if over.any():
        raise ValueError(
            f"{int(over.sum())} galaxies are FAINTER than the declared "
            f"m_lim={m_lim}: the truncation datum does not describe this "
            "sample (wrong m_lim, or the survey is not magnitude-limited).")

    Mhat = reference_absolute_mags(m, z, Om0, w0, wa, k_corr_coeffs)
    T = m_lim - (m - Mhat)          # = m_lim - DM(z; H0_REF) - K(z), per galaxy

    if family == "schechter":
        if m_faint_cut is not None:
            # A PARAMETER-FREE cut, so the sample it defines does not move with
            # the fitted theta: this is the whole reason the MLE below is a
            # regular interior optimum rather than an order statistic.
            keep = Mhat <= float(m_faint_cut)
            n_cut = int((~keep).sum())
            if int(keep.sum()) < 10:
                raise ValueError(
                    f"m_faint_cut={float(m_faint_cut)} leaves "
                    f"{int(keep.sum())} of {Mhat.size} galaxies (need at least "
                    "10): the cut is brighter than essentially the whole "
                    "sample.")
            Mhat, T, z = Mhat[keep], T[keep], z[keep]
        else:
            n_cut = 0
        return _fit_schechter_truncated(
            Mhat, T, z, m_lim, float(M_faint_offset),
            None if m_faint_cut is None else float(m_faint_cut),
            n_cut=n_cut, stratum=stratum, Om0=Om0, w0=w0, wa=wa)
    return _fit_gaussian_truncated(
        Mhat, T, m_lim, stratum=stratum, k_corr_coeffs=k_corr_coeffs,
        Om0=Om0, w0=w0, wa=wa)


def _fit_gaussian_truncated(Mhat, T, m_lim, *, stratum, k_corr_coeffs,
                            Om0, w0, wa):
    """Upper-truncated-Gaussian MLE + Laplace covariance in (M0hat, sigma_M)."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    def nll(theta):
        mu, log_sig = theta
        sig = np.exp(log_sig)
        resid = (Mhat - mu) / sig
        log_trunc = norm.logcdf((T - mu) / sig)
        return -np.sum(norm.logpdf(resid) - np.log(sig) - log_trunc)

    theta0 = np.array([np.median(Mhat), np.log(max(np.std(Mhat), 1e-3))])
    # fatol is ABSOLUTE while this NLL scales with N, so at catalog size the old
    # 1e-10 sat BELOW the objective's own float64 resolution (eps * |NLL| ~ 2.8e-10
    # at N = 1e6) and the stop was the accidental bit-identity of the simplex
    # values rather than a meaningful criterion.  Scaled as in the schechter twin;
    # measured at N = 1e6 / 2e6 it reaches the identical MLE (8 decimals) in ~40%
    # fewer evaluations.  Theta precision is xatol's job.
    res = minimize(nll, theta0, method="Nelder-Mead",
                   options={"xatol": 1e-8,
                            "fatol": 1e-9 * max(1.0, float(Mhat.size)),
                            "maxiter": 20000})
    if not res.success:
        raise RuntimeError(f"selection fit did not converge: {res.message}")
    mu_hat, log_sig_hat = res.x
    sig_hat = float(np.exp(log_sig_hat))

    # Laplace covariance in (M0hat, sigma_M): numerical Hessian of the NLL
    # reparametrized to (mu, sigma) so the prior is Gaussian in the sampled
    # coordinates.
    def nll_sig(theta):
        return nll(np.array([theta[0], np.log(theta[1])]))

    x0 = np.array([mu_hat, sig_hat])
    h = np.array([1e-4, 1e-4])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * h[i]
            ej = np.eye(2)[j] * h[j]
            H[i, j] = (nll_sig(x0 + ei + ej) - nll_sig(x0 + ei - ej)
                       - nll_sig(x0 - ei + ej) + nll_sig(x0 - ei - ej)
                       ) / (4.0 * h[i] * h[j])
    cov = np.linalg.inv(H)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        raise RuntimeError("selection-fit Hessian is not positive definite")

    return SelectionFit(
        family="gaussian", m_lim=float(m_lim), M0hat=float(mu_hat),
        sigma_M=sig_hat, cov=cov, n_gal=int(Mhat.size), stratum=str(stratum),
        k_corr_coeffs=tuple(float(c) for c in (k_corr_coeffs or ())),
        meta={"Om0": float(Om0), "w0": float(w0), "wa": float(wa),
              "H0_ref": H0_REF, "nll": float(res.fun)},
    )


def _fit_schechter_truncated(Mhat, T, z, m_lim, M_faint_offset, m_faint_cut,
                             *, n_cut, stratum, Om0, w0, wa):
    """Detection-truncated Schechter MLE + Laplace covariance in (Mstar_hat, alpha).

    Work in reference absolute magnitudes ``Mhat_i = m_i - DM(z_i; H0=100)``
    (K refused: the pinned curve carries no template).  With
    ``x(M) = 10^{-0.4 (M - Mstar_hat)} = L/L*`` and ``a = alpha + 1``, the
    Schechter LF in magnitude space is

        f(M) ∝ x(M)^{alpha+1} exp(-x(M)) ,
        x_i = 10^{-0.4 (Mhat_i - Mstar_hat)}

    Galaxy ``i`` enters the catalog only if ``Mhat_i <= T_i = m_lim -
    DM(z_i; H0=100)``, and -- when the caller declares one -- only if
    ``Mhat_i <= m_faint_cut``.  Both edges are PARAMETER-FREE numbers, so the
    per-galaxy support ``M <= U_i = min(T_i, m_faint_cut)`` does not move with
    the fitted theta and the MLE is a regular interior optimum.

    KNOWN LIMITATION (photo-z scatter vs a sharp edge): both edges are
    NOISELESS-observable assumptions on ``Mhat``, but a photometric redshift
    propagates ``sigma_z`` into ``Mhat`` through the distance modulus.  For
    the detection edge this only inflates the fitted widths slightly (the
    gaussian family's documented behaviour); for ``m_faint_cut`` the fit
    DEGRADES once ``sd(Mhat_obs - Mhat_true)`` is comparable to the
    truncation depth ``|M_faint - M*|``: galaxies scatter across the edge,
    the cut removes the ones that scattered faint-ward, and the edge-biased
    survivors pull (Mstar_hat, alpha) coherently -- measured at +3.2 / +4.3
    sd on a mock whose scatter was 92% of a 0.094 mag truncation, mispricing
    the transition-zone missing budget (1 - C_sel) by up to ~2.5x.  Fit on
    spectroscopic (or, for mocks, TRUE) redshifts when the truncation is
    shallow; the principled photo-z treatment is a scatter-convolved
    normalization, not a widened cut (which trades the sharp bias for an
    unpropagatable contamination bias).  In ``x`` the
    support is ``x >= x_{U,i} = max(x_lim,i, x_cut)`` and the normalization is
    closed-form:

        ∫_{-inf}^{U_i} f(M) dM = Gamma(a, x_{U,i}) / (0.4 ln10)

    with ``Gamma(a, .)`` the UNREGULARIZED upper incomplete gamma, finite for
    every real ``a`` because ``x_{U,i} > 0`` (:func:`_upper_gamma_scaled`
    carries it).  Hence

        log p(Mhat_i | Mstar_hat, alpha)
            = log(0.4 ln10) + a log x_i - x_i - log Gamma(a, x_{U,i})
            with  log x_i = 0.4 ln10 (Mstar_hat - Mhat_i)

        NLL(Mstar_hat, alpha) = - sum_i log p(Mhat_i | .)

    What is deliberately ABSENT: the model's faint cutoff ``M_faint =
    Mstar_hat + M_faint_offset``.  Putting it in the fit would make the
    support edge a function of the parameter being fitted, whose MLE is then
    an order statistic pinned to the faintest galaxy -- and it would refuse
    every catalog that reaches past 10^{-0.4 M_faint_offset} L*, which is most
    of them at low z.  The cutoff belongs to the completeness DENOMINATOR, not
    to the detected-magnitude distribution, so it is carried as the protocol
    constant it is and stamped unfitted into ``meta``:

        meta["m_faint_offset_constrained"] = False

    together with the missing-galaxy budget it buys at +/-1 mag, because that
    budget multiplies the entire out-of-catalog weight downstream.  Declare
    ``m_faint_cut`` to fit only the part of the sample the modelled population
    is meant to describe; it is an analysis cut on the FIT, independent of
    ``M_faint_offset``, and the two agree only if the caller makes them.

    The z-distribution never enters (thinning), so the fit is clustering-safe
    exactly as the gaussian one is.
    """
    from scipy.optimize import minimize

    ln10 = np.log(10.0)
    n_gal = int(Mhat.size)
    log_pref = np.log(0.4 * ln10)

    # --- sufficient statistics: the per-galaxy magnitude terms are EXACT ---
    #
    #   sum_i a log x_i = a 0.4 ln10 (N Mstar - sum_i Mhat_i)
    #   sum_i x_i       = 10^{0.4 Mstar} * sum_i 10^{-0.4 Mhat_i}
    #
    # so only the per-galaxy normalization ``log Gamma(a, x(U_i))`` with the
    # PARAMETER-FREE support edge ``U_i = min(T_i, m_faint_cut)`` needs
    # compression: ``max(x_lim,i, x_cut) = 10^{-0.4 (U_i - Mstar)}`` exactly
    # (both edges scale with the same ``10^{0.4 Mstar}``), and ``U`` is binned
    # the way :func:`magnitude_suffstats` bins ``T`` for the gaussian family
    # (bin-mean edge per bin; error ``O(bin width^2)`` in log Phi curvature,
    # here ``O(bin width^2)`` in log Gamma curvature -- far below the Laplace
    # sd at 64 bins over a <~ 1 mag U range).  The cost per NLL evaluation
    # drops from N to n_bins ``gammaincc`` calls: the difference between
    # seconds and hours at the N ~ 10^6 of real catalogs (measured >2 h
    # unconverged at 8e5 galaxies on the per-galaxy spelling).
    S1 = float(np.sum(Mhat))
    S_L = float(np.sum(10.0 ** (-0.4 * Mhat)))
    U = T if m_faint_cut is None else np.minimum(T, float(m_faint_cut))
    n_bins = 64
    _lo, _hi = float(np.min(U)), float(np.max(U))
    if _hi - _lo < 1e-12:
        # Degenerate support (every galaxy shares one edge, e.g. a deep
        # survey where the faint cut dominates everywhere): exact, one bin.
        U_b = np.array([_lo])
        N_b = np.array([float(n_gal)])
    else:
        _edges = np.linspace(_lo - 1e-9, _hi + 1e-9, n_bins + 1)
        _idx = np.clip(np.digitize(U, _edges) - 1, 0, n_bins - 1)
        N_b = np.zeros(n_bins)
        _U_sum = np.zeros(n_bins)
        np.add.at(N_b, _idx, 1.0)
        np.add.at(_U_sum, _idx, U)
        _keep = N_b > 0
        U_b = _U_sum[_keep] / N_b[_keep]
        N_b = N_b[_keep]

    def _norm(Mstar, a):
        """``Gamma(a, x(U_b))`` on the bin-mean support edges.

        ``_upper_gamma_scaled`` returns ``a Gamma(a, x)``; the same nudged
        ``a`` divides it back out, so the pair is exact at every ``a`` and
        merely loses the leading digits of the recurrence's difference as
        ``a -> 0``.
        """
        a_s = _a_off_zero(a, np)
        x_u = 10.0 ** (-0.4 * (U_b - Mstar))
        return _upper_gamma_scaled(a_s, x_u, xp=np) / a_s

    def nll(theta):
        # Optimizer coordinates (Mstar_hat, u) with alpha = exp(u) + _ALPHA_MIN,
        # so the curve's domain holds by construction and the optimizer cannot
        # walk out of it (the trick the gaussian fit uses for sigma > 0).
        Mstar, u = float(theta[0]), float(theta[1])
        a = np.exp(u) + _ALPHA_MIN + 1.0
        gam = _norm(Mstar, a)
        if not np.all(np.isfinite(gam)) or np.any(gam <= 0.0):
            return np.inf
        val = -(a * 0.4 * ln10 * (n_gal * Mstar - S1)
                - 10.0 ** (0.4 * Mstar) * S_L
                - float(np.sum(N_b * np.log(gam)))
                + n_gal * log_pref)
        return val if np.isfinite(val) else np.inf

    Mstar0 = float(np.percentile(Mhat, 10.0))          # the bright decile scale
    u0 = np.log(-_ALPHA_MIN - 0.7)                     # alpha0 = -0.7
    # fatol is ABSOLUTE while the NLL scales with N: at N ~ 10^6 the float64
    # resolution of the objective (eps * |NLL| ~ 1e-10) sits ON the old
    # 1e-10 tolerance, so Nelder-Mead could never satisfy it and ground to
    # maxiter. Scale it with N; theta precision is xatol's job (1e-8 mag,
    # orders below any Laplace sd).
    res = minimize(nll, np.array([Mstar0, u0]), method="Nelder-Mead",
                   options={"xatol": 1e-8,
                            "fatol": 1e-9 * max(1.0, float(n_gal)),
                            "maxiter": 20000})
    if not res.success:
        raise RuntimeError(f"selection fit did not converge: {res.message}")
    Mstar_hat = float(res.x[0])
    alpha_hat = float(np.exp(res.x[1]) + _ALPHA_MIN)
    if alpha_hat <= _ALPHA_MIN:
        raise RuntimeError(
            f"schechter fit reached alpha={alpha_hat:.4f}, at or below the "
            f"{_ALPHA_MIN} floor: c_sel_schechter reaches a <= 0 with one "
            "recurrence step off gammaincc's positive-argument domain, so "
            "alpha + 2 > 0 is the edge of the spelling. A sample this steep "
            "is not a faint-end slope any survey measures -- check m_lim and "
            "the redshifts before widening anything.")
    gam_hat = _norm(Mstar_hat, alpha_hat + 1.0)
    if np.any(gam_hat <= 0.0):
        raise RuntimeError(
            f"{int(np.sum(N_b[gam_hat <= 0.0]))} galaxies have an "
            "UNDERFLOWING normalization Gamma(alpha+1, x_lim) = 0 at the "
            f"optimum: the declared m_lim={float(m_lim)} puts them so far "
            "into the bright tail that the truncated Schechter assigns them "
            "no probability mass. The declared limit cannot describe this "
            "sample.")

    # Laplace covariance in the REPORTED coordinates (Mstar_hat, alpha): the
    # same reparametrized-Hessian structure as the gaussian's nll_sig.
    def nll_alpha(p):
        return nll(np.array([p[0], np.log(p[1] - _ALPHA_MIN)]))

    x0 = np.array([Mstar_hat, alpha_hat])
    h = np.array([1e-4, 1e-3])          # alpha's per-galaxy curvature is ~10x weaker
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * h[i]
            ej = np.eye(2)[j] * h[j]
            H[i, j] = (nll_alpha(x0 + ei + ej) - nll_alpha(x0 + ei - ej)
                       - nll_alpha(x0 - ei + ej) + nll_alpha(x0 - ei - ej)
                       ) / (4.0 * h[i] * h[j])
    cov = np.linalg.inv(H)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        raise RuntimeError("selection-fit Hessian is not positive definite")

    diag = _faint_end_diagnostics(Mstar_hat, alpha_hat, M_faint_offset, m_lim,
                                 Mhat, T, z, Om0, w0, wa)
    if m_faint_cut is None and diag["frac_complete_at_m_faint"] > _DEEP_WARN:
        import warnings

        # Not fatal: whether the real population ends at M_faint is a modelling
        # claim, not something these magnitudes can settle.  But an undeclared
        # cut here means the fit models an LF with NO faint edge over a
        # sub-sample where the completion assumes one, which biases M* bright
        # and alpha shallow (both by many sd at these fractions).
        warnings.warn(
            f"schechter fit: {diag['frac_complete_at_m_faint']:.1%} of the "
            f"sample is complete past M_faint = {diag['m_faint_implied']:.3f} "
            "(the survey sees fainter than the modelled population there), and "
            "no m_faint_cut was declared, so the fit assumed an LF with no "
            "faint edge over that sub-sample. Pass m_faint_cut (a "
            "PARAMETER-FREE h-scaled absolute magnitude, e.g. the implied "
            "M_faint) to fit only the population the completeness answers "
            "for.", RuntimeWarning)

    return SelectionFit(
        family="schechter", m_lim=float(m_lim), Mstar_hat=Mstar_hat,
        alpha=alpha_hat, M_faint_offset=float(M_faint_offset), cov=cov,
        n_gal=n_gal, stratum=str(stratum), k_corr_coeffs=(),
        meta={"Om0": float(Om0), "w0": float(w0), "wa": float(wa),
              "H0_ref": H0_REF, "nll": float(res.fun),
              "m_faint_cut": (None if m_faint_cut is None
                              else float(m_faint_cut)),
              "n_gal_cut_faintward": int(n_cut), **diag},
    )


def _faint_end_diagnostics(Mstar_hat, alpha, M_faint_offset, m_lim, Mhat, T, z,
                          Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                          d_offset=1.0):
    """What the magnitudes say -- and cannot say -- about the faint cutoff.

    ``M_faint_offset`` never enters :func:`_fit_schechter_truncated`'s
    likelihood, yet it sets the ``Gamma(a, x_faint)`` denominator of the
    consumed curve and therefore the whole missing-galaxy budget.  This
    records that asymmetry in numbers a reviewer can act on:

    * ``m_faint_offset_constrained``: always False, stated rather than implied;
    * ``n_gal_faintward_of_m_faint``: galaxies the catalog holds that the
      modelled population does not contain (a cutoff too bright for this
      survey);
    * ``frac_complete_at_m_faint``: fraction of the sample whose own detection
      limit already reaches past the cutoff, i.e. where ``C_sel = 1`` and the
      cutoff is doing all the work;
    * ``missing_budget_vs_offset``: ``1 - C_sel(z_med)`` at the declared offset
      and at +/- ``d_offset`` mag, evaluated through the consumed
      :func:`c_sel_schechter` itself, so the protocol's leverage on the budget
      is visible next to the fitted theta.
    """
    m_faint = Mstar_hat + M_faint_offset
    z_med = float(np.median(np.asarray(z, dtype=float)))
    budget = {}
    for d in (-d_offset, 0.0, d_offset):
        off = float(M_faint_offset) + d
        if off <= _M_FAINT_OFFSET_MIN:
            continue
        c = float(c_sel_schechter(z_med, m_lim, Mstar_hat, alpha, off, H0_REF,
                                  Om0, w0, wa))
        budget[f"{off:.2f}"] = 1.0 - c
    return {
        "m_faint_offset_constrained": False,
        "m_faint_implied": float(m_faint),
        "n_gal_faintward_of_m_faint": int(np.sum(np.asarray(Mhat) > m_faint)),
        "frac_complete_at_m_faint": float(np.mean(np.asarray(T) > m_faint)),
        "z_budget_ref": z_med,
        "missing_budget_vs_offset": budget,
    }
