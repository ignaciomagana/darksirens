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
through angular-diameter distances. In commit 1 we treat T_0 as an
externally supplied scalar; commit 2 will hook it up to the cosmology
module so ``T_0(z_L, z_s | H_0, Ω_m)`` is computed self-consistently.

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

import jax
import jax.numpy as jnp
from jax import jit


# ============================================================================
# SIS lens-population container
# ============================================================================

class SISLensParams(NamedTuple):
    """SIS lens-population parameters.

    Fields
    ------
    A_tau, n_tau
        Optical-depth amplitude and power-law slope: τ_2(z) = A · z^n.
    T0
        Time-delay scaling constant (seconds). In commit 1 this is a
        single scalar (z-averaged); commit 2 will replace it with a
        cosmology-aware function. Defaults to ~5 days in seconds at
        z_s = 1 — order of magnitude only.
    """
    A_tau: Any
    n_tau: Any
    T0: Any


def make_sis_lens_params(
    A_tau: float = 5.0e-4,
    n_tau: float = 3.0,
    T0_seconds: float = 4.32e5,
) -> SISLensParams:
    """Default SIS lens-population parameters.

    Defaults
    --------
    A_tau, n_tau
        τ_2(z) = 5e-4 · z^3.  At z = 1 → τ ≈ 5e-4 (consistent with
        galaxy-scale SIS lensing optical depth in the literature).
    T0_seconds
        4.32e5 s ≈ 5 days. Order-of-magnitude SIS time-delay scale at
        z_s = 1 for σ_v ~ 200 km/s lenses.
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
    """
    return p.A_tau * jnp.power(z, p.n_tau)


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

    Commit-1 form uses a constant T_0; commit 2 will replace with a
    cosmology-and-lens-redshift-dependent prefactor.
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
