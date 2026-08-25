"""
slmarks.py
----------
Strong-lensing optical depth and mark distribution for the SIS model.

Mathematical conventions
~~~~~~~~~~~~~~~~~~~~~~~~
For a singular isothermal sphere (SIS) lens with Einstein angle θ_E and
dimensionless source impact ``y = β / θ_E ∈ (0, 1)``, two images form
with magnitudes

    μ_+ = (1 + y) / y          (type-I minimum, parity +1)
    μ_- = (1 - y) / y          (type-II saddle, parity -1)

so the two image magnifications satisfy the rigid constraint

    μ_+ − μ_− = 2

(NOT μ_+ + μ_- = 2, which is a common confusion; see Schneider, Kochanek
& Wambsganss 2006, eq. 8.65). The joint PDF of (μ_+, μ_-) is therefore
a delta function on a 1D manifold, and the natural integration variable
for any cluster likelihood is ``y`` itself.

Source-plane density of impacts: assuming a uniform spatial density of
sources behind the lens, the probability that the impact is within the
Einstein ring at parameter ``y`` is the source-plane area ∝ y², giving

    p(y) = 2 y,   y ∈ (0, 1).

Time delay (Fermat principle for SIS):

    Δt(y, z_L, z_s) = 2 (1 + z_L) θ_E² y · D_L D_S / (c · D_LS)
                    = T_0(z_L, z_s) · y

where ``T_0`` carries all the cosmology and lens-redshift dependence
through angular-diameter distances. We still treat T_0 as an externally
supplied scalar (see ``DEFAULT_T0_SECONDS`` for the calibration and
``make_sis_lens_params`` for how to override it); making it a
sampled/marginalised function ``T_0(z_L, z_s, σ_v | H_0, Ω_m, w_0, w_a)``
is the proper fix and is not done here.

Optical depth τ_2(z_s)
~~~~~~~~~~~~~~~~~~~~~~
For an SIS lens population characterized by a comoving number density
``n_*`` and a velocity-dispersion function ``Φ(σ_v)``, the strong-
lensing optical depth scales with the source comoving distance ``D_C``
as a power law to good approximation,

    τ_2(z_s) ≈ F_* · [D_C(z_s) / D_H]^n_τ,

with ``F_*`` a population efficiency (≈ 0.013 from a Schechter VDF
calibrated to SDSS early-type galaxies; Mitchell+05) and ``n_τ ≈ 3``
for low-z, asymptoting more slowly at high z. For an MVP without
cosmology coupling, we adopt the simple analytic surrogate

    τ_2(z_s) = A · z_s^n_τ                   (commit-1 default)

with ``A = 5e-4``, ``n_τ = 3`` giving order-of-magnitude agreement with
Hilbert+08 / Wierda+21 around z_s ≈ 1. The user is encouraged to override
with a tabulated optical depth in production runs.

References
~~~~~~~~~~
- Schneider, Kochanek & Wambsganss (2006), "Gravitational Lensing:
  Strong, Weak and Micro", Saas-Fee Advanced Course 33.
- Mitchell, Keeton, Frieman, Sheth (2005), ApJ 622, 81.
- Hilbert, White, Hartlap, Schneider (2008), MNRAS 386, 1845.
- Wierda, Wempe, Hannuksela, Koopmans, Van Den Broeck (2021), ApJ 921, 154.
"""

from __future__ import annotations

from typing import NamedTuple, Any

import jax.numpy as jnp
from jax import jit


# ============================================================================
# SIS lens-population container
# ============================================================================

# Fiducial SIS time-delay scale, in seconds.
#
# Derived from this module's own formula (see the header) with THIS repo's
# cosmology (``utils.cosmology.r_of_z``, Planck15 defaults) at the reference
# lens/source configuration z_L = 0.5, z_s = 1.0, σ_v = 200 km/s:
#
#     θ_E   = 4π (σ_v/c)² D_LS/D_S            = 0.493 arcsec
#     T_0   = 2 (1 + z_L) θ_E² D_L D_S/(c D_LS) = 5.36e6 s = 62 days
#
# Neighbouring configurations: 5.06e6 s at z_L = 0.3; 7.09e6 s at z_s = 1.5;
# 1.31e7 s at σ_v = 250 km/s.  The historical default of 4.32e5 s (5 days) is
# ~12x too small — it corresponds to σ_v ≈ 107 km/s, i.e. a dwarf-galaxy lens,
# not the σ_v ~ 200 km/s early-type population the optical depth is calibrated
# to.  That matters because the time mark converts the observed delay to an
# impact parameter as y* = |Δt|/T_0 and the SIS support is y ∈ (0, 1): with the
# old default, EVERY candidate pair separated by more than 5 days received an
# exactly -inf pair log-likelihood (see ``likelihood.cluster_likelihood``
# PAIR_MARKS_DELTA_COLLAPSE), and GWTC-style lensing candidates are weeks to
# months apart.
#
# This is a scalar stand-in for a distribution.  The physically complete model
# marginalises T_0 over the lens redshift and the velocity-dispersion function
# (or samples it as a hyperparameter); that is deliberately NOT done here.
# The value itself lives in the dependency-free darksirens.core.constants so the
# lensing CLI's argparse default does not have to import this package; this stays
# the canonical spelling.
from darksirens.core.constants import DEFAULT_T0_SECONDS  # noqa: E402,F401


class SISLensParams(NamedTuple):
    """SIS lens-population parameters.

    Fields
    ------
    A_tau, n_tau
        Optical-depth amplitude and power-law slope: τ_2(z) = A · z^n.
    T0
        Time-delay scaling constant (seconds): Δt = T_0 · y. A single
        z-averaged scalar, not a function of (z_L, z_s, σ_v) — see
        ``DEFAULT_T0_SECONDS`` for the calibration and the caveat.
    """
    A_tau: Any
    n_tau: Any
    T0: Any


def make_sis_lens_params(
    A_tau: float = 5.0e-4,
    n_tau: float = 3.0,
    T0_seconds: float = DEFAULT_T0_SECONDS,
) -> SISLensParams:
    """Default SIS lens-population parameters.

    Defaults
    --------
    A_tau, n_tau
        τ_2(z) = 5e-4 · z^3.  At z = 1 → τ ≈ 5e-4 (consistent with
        galaxy-scale SIS lensing optical depth in the literature).
    T0_seconds
        ``DEFAULT_T0_SECONDS`` = 5.36e6 s ≈ 62 days: the SIS time-delay
        scale at z_L = 0.5, z_s = 1.0, σ_v = 200 km/s under this repo's
        cosmology. Override for a different lens population (the lensing
        CLI exposes ``--sl_T0_sec``); a candidate pair with
        |Δt| ≥ T0_seconds falls outside the SIS support y ∈ (0, 1) and its
        time-marked pair likelihood is exactly -inf.
    """
    return SISLensParams(
        A_tau=jnp.asarray(A_tau, dtype=jnp.float64),
        n_tau=jnp.asarray(n_tau, dtype=jnp.float64),
        T0=jnp.asarray(T0_seconds, dtype=jnp.float64),
    )


# ============================================================================
# Optical depth
# ============================================================================

@jit
def tau_2_SIS(z: jnp.ndarray, p: SISLensParams) -> jnp.ndarray:
    """Strong-lensing optical depth for J = 2 images.

    Returns ``A · z^n``. Vectorised over z.

    Notes
    -----
    The output is **always non-negative** (no clipping required for
    ``z ≥ 0``).  Calling with ``z < 0`` returns NaN for non-integer
    ``n_tau``; the cluster likelihood is responsible for never passing
    negative redshifts.

    ``A · z^n`` is an optical DEPTH and exceeds one at prior corners (the
    CLI's box allows ``A = 1e-2, n = 6``, i.e. τ = 156 at z = 5).  Every
    likelihood channel consumes it as the Bernoulli PROBABILITY that the
    source is multiply imaged — use :func:`tau_2_prob` there, never a raw
    ``log(tau_2_SIS(...))``.
    """
    return p.A_tau * jnp.power(z, p.n_tau)


# One definition of "τ as a probability" for every likelihood channel.  The
# ceiling stays strictly below 1 so the unlensed branch's log1p(-τ) is finite.
TAU_PROB_MAX = 1.0 - 1e-12


@jit
def tau_2_prob(z: jnp.ndarray, p: SISLensParams) -> jnp.ndarray:
    """τ₂ consumed as a probability: ``clip(tau_2_SIS, 0, 1 - 1e-12)``.

    The pair evidence, the lensed selection terms, and the singleton
    (un)lensed branches all weight by "P(multiply imaged | z_s)" and its
    complement.  Before this helper the singleton terms clipped while the
    pair/selection terms used the raw power law, so ONE parameter point
    carried two incompatible "probabilities" across channels — inflating the
    pair evidence relative to the (1-τ) suppression and biasing A_tau/n_tau
    wherever the prior box reaches τ > 1.  Clipping is a plateau, not a
    reparameterization: points with τ(z) ≥ 1 remain in the prior but can no
    longer out-compete τ = 1, which is the most a probability can say.
    """
    return jnp.clip(tau_2_SIS(z, p), 0.0, TAU_PROB_MAX)


# ============================================================================
# Mark distribution in the natural variable y
# ============================================================================

@jit
def log_p_y_SIS(y: jnp.ndarray) -> jnp.ndarray:
    """Log of the source-plane impact PDF, p(y) = 2 y on (0, 1).

    Returns ``-inf`` for y outside (0, 1). Vectorised.
    """
    valid = (y > 0.0) & (y < 1.0)
    log_p = jnp.log(2.0) + jnp.log(jnp.where(valid, y, 1.0))
    return jnp.where(valid, log_p, -jnp.inf)


@jit
def mu_plus_minus_from_y(y: jnp.ndarray) -> tuple:
    """Image magnifications (μ_+, μ_-) from impact parameter y.

    Both are positive magnitudes for y ∈ (0, 1).
    """
    inv_y = 1.0 / y
    mu_plus = (1.0 + y) * inv_y
    mu_minus = (1.0 - y) * inv_y
    return mu_plus, mu_minus


@jit
def y_from_mu_plus(mu_plus: jnp.ndarray) -> jnp.ndarray:
    """Inverse map: y = 1 / (μ_+ - 1).

    Valid for μ_+ > 2 (corresponds to y < 1). Returns NaN otherwise to
    flag domain errors; callers should pre-filter.
    """
    return 1.0 / (mu_plus - 1.0)


@jit
def delta_t_from_y(y: jnp.ndarray, p: SISLensParams) -> jnp.ndarray:
    """SIS time delay: Δt = T_0 · y, in seconds.

    Uses the constant ``p.T0``; a cosmology-and-lens-redshift-dependent
    prefactor (marginalised over the lens redshift and the velocity-dispersion
    function) is the physically complete form — see ``DEFAULT_T0_SECONDS``.
    """
    return p.T0 * y


# ============================================================================
# Hooks for J = 4 (quads) — wired but inert
# ============================================================================

@jit
def tau_4_SIS(z: jnp.ndarray, p: SISLensParams) -> jnp.ndarray:
    """Quad optical depth. Inert in commit 1 (returns zero).

    Real quads require an SIE / elliptical-potential model; a future
    extension will tabulate τ_4 and the four-image mark distribution
    from a lens-population Monte Carlo.
    """
    return jnp.zeros_like(z, dtype=jnp.float64)
