"""
grids.py
--------
Quadrature grids for marginalizing over lensing magnification.

We provide two builders:

- ``make_log_mu_grid(n_nodes, log_mu_range)``: Gauss-Legendre nodes and
  log-weights for integration in ``ln μ`` on a finite interval, suitable
  for the weak-lensing marginalization (lognormal has most mass on
  ln μ ∈ [-0.5, 0.5]).

- ``make_y_grid(n_nodes)``: Gauss-Legendre nodes in y ∈ (0, 1) for the
  SIS strong-lensing marginalization. p(y) = 2y, so we don't need
  ultra-fine sampling; ~32 nodes captures the integral to machine
  precision for any smooth integrand.

The returned log-weights already include the Jacobian of the
substitution from the canonical (-1, 1) interval, so the user can
combine them directly with the log-integrand:

    log_integral ≈ logsumexp(log_integrand + log_weights)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _gauss_legendre_on_interval(
    n_nodes: int,
    a: float,
    b: float,
) -> tuple:
    """Gauss-Legendre nodes and weights on the interval (a, b).

    Returns (x_nodes, w_nodes). The weights w_i include the Jacobian
    (b - a) / 2, so ``∫_a^b f(x) dx ≈ Σ_i w_i f(x_i)``.
    """
    x_canon, w_canon = np.polynomial.legendre.leggauss(n_nodes)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    x = half * x_canon + mid
    w = half * w_canon
    return x, w


def make_log_mu_grid(
    n_nodes: int = 16,
    log_mu_range: tuple = (-0.6, 0.6),
) -> tuple:
    """Gauss-Legendre nodes in ``ln μ`` for weak-lensing marginalization.

    Parameters
    ----------
    n_nodes
        Number of quadrature nodes. 16 is overkill for a lognormal with
        s² ~ 10⁻², comfortable for s² ~ 10⁻¹.
    log_mu_range
        (log_mu_min, log_mu_max). The default (-0.6, 0.6) covers
        μ ∈ (0.55, 1.82), which holds essentially all the mass for
        s² ≤ 0.04 (i.e. up to z ~ 5 under the default lognormal).

    Returns
    -------
    mu_nodes : (n_nodes,) jnp.ndarray
        Magnification values μ_ℓ = exp(ln μ_ℓ).
    log_weights : (n_nodes,) jnp.ndarray
        Log of the Gauss-Legendre weights, including the (b-a)/2 Jacobian
        for the ln-μ substitution. Use as

            ∫ f(μ) dμ = ∫ f(e^x) e^x dx
                      ≈ Σ_ℓ w_ℓ · f(μ_ℓ) · μ_ℓ
                      = Σ_ℓ exp(log_weights_ℓ + log(f(μ_ℓ)) + ln μ_ℓ)

        The extra ``ln μ`` comes from ``dμ = e^x dx``; the caller is
        responsible for adding it (we do NOT bake it into log_weights,
        because for log-space PDFs we typically express ``p(μ) dμ`` as
        ``p(ln μ) d(ln μ)`` already, in which case the extra μ cancels).
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be ≥ 2, got {n_nodes}.")
    a, b = log_mu_range
    if not (b > a):
        raise ValueError(f"log_mu_range must be strictly increasing, got ({a}, {b}).")

    log_mu_x, log_mu_w = _gauss_legendre_on_interval(n_nodes, a, b)
    mu_nodes = np.exp(log_mu_x)
    # The weights are positive on a finite interval — log is finite.
    log_w = np.log(log_mu_w)
    return jnp.asarray(mu_nodes, dtype=jnp.float64), jnp.asarray(log_w, dtype=jnp.float64)


def make_y_grid(n_nodes: int = 32) -> tuple:
    """Gauss-Legendre nodes in ``y ∈ (0, 1)`` for SIS marginalization.

    The interval excludes the boundaries; Gauss-Legendre nodes are open,
    so no special handling required.

    Returns
    -------
    y_nodes : (n_nodes,) jnp.ndarray
    log_weights : (n_nodes,) jnp.ndarray
        log of the Gauss-Legendre weights with the (1 - 0)/2 = 1/2
        Jacobian baked in.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be ≥ 2, got {n_nodes}.")
    y_x, y_w = _gauss_legendre_on_interval(n_nodes, 0.0, 1.0)
    log_w = np.log(y_w)
    return jnp.asarray(y_x, dtype=jnp.float64), jnp.asarray(log_w, dtype=jnp.float64)
