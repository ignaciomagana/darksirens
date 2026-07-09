"""
pair_kde.py
-----------
Precomputed per-event Gaussian KDE for the cluster-pair likelihood.

For the J=2 cluster likelihood (commit 3), each candidate pair (d_i, d_j)
is evaluated by:
  1. Mapping PE samples of event i through the magnification μ_+(y) to
     predict the *apparent* parameters that event j *would* show under
     image assignment σ.
  2. Evaluating a Gaussian KDE of event-j's apparent-frame PE samples
     at those predicted apparent points.

The KDE bandwidths are set by Silverman's rule in 4-D from per-event
sample standard deviations. These quantities are constants for fixed
input data, so we precompute them at load time and pass them through
JIT as traced arrays — keeping the hot path branch-free.

Coordinates
~~~~~~~~~~~
The KDE lives in the *apparent* 4-D coordinate system used throughout
the inference:

    θ_app = (m_1_det, q, d_L_app, χ_eff).

This matches the coordinates of every other PE-using piece of darksirens
and avoids any detector→source Jacobian gymnastics at evaluation time.

The Jacobian of the apparent→source map at fixed μ is applied at the
likelihood level (cluster_likelihood.py), exactly as in commit 2's
wl_weight.py.
"""

from __future__ import annotations

from typing import NamedTuple, Any
import numpy as np
import jax.numpy as jnp


# Coordinate axes carried through the KDE in canonical order.
PAIR_KDE_COORDS = ("m1det", "q", "dL_app", "chieff")
_D = 4   # KDE dimensionality (fixed for this analysis)


def validate_pair_prior_wt(
    prior_wt: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    context: str = "PairKDE prior_wt",
) -> np.ndarray:
    """Validate per-sample PE proposal-density weights for one pair image.

    Pair PE uses the same convention as singleton PE: every structurally valid
    sample must carry a finite, strictly positive proposal-density value.  The
    pair-PE loader is responsible for any per-image normalization convention;
    this helper only rejects malformed weights before they can silently become
    ``-inf``/``nan`` in the KDE.
    """
    prior_wt = np.asarray(prior_wt, dtype=np.float64)
    if valid is None:
        valid = np.ones(prior_wt.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    if prior_wt.shape != valid.shape:
        raise ValueError(
            f"{context}: prior_wt shape {prior_wt.shape} does not match "
            f"valid shape {valid.shape}."
        )
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(f"{context}: no valid PE samples.")
    good = np.isfinite(prior_wt) & (prior_wt > 0.0)
    if not bool(np.all(good[valid])):
        n_bad = int(np.count_nonzero(valid & ~good))
        raise ValueError(
            f"{context}: all valid prior_wt values must be finite and positive; "
            f"found {n_bad} malformed sample(s)."
        )
    return prior_wt


class PairKDE(NamedTuple):
    """Per-event precomputed Gaussian KDE for the cluster-pair likelihood.

    Fields
    ------
    samples : (N_pe, 4) float64
        Apparent-frame PE samples in the canonical coordinates
        ``(m1det, q, dL_app, chieff)``. Order matches the parent GWEvent's
        sample ordering.
    log_weights : (N_pe,) float64
        Per-sample log weights ``log(1 / p_prop)`` for importance correction.
        These are passed through to the kernel sum so the resulting density
        is the posterior-equivalent KDE, not the proposal-density KDE.
    log_h : (4,) float64
        Log of the Silverman bandwidths in each coordinate. We store the log
        to avoid repeated division at evaluation time.
    log_norm : scalar float64
        Log of the Gaussian-kernel normalization constant
        ``-0.5 * d * log(2π) - sum(log_h)``, ready to add into the
        log-evaluated density.
    valid : (N_pe,) bool
        Validity mask carried alongside samples (matches GWEvent.valid).
        Padded entries get -inf weight in the kernel sum.

    Notes
    -----
    - The bandwidth is *diagonal*. A full-covariance KDE would need a
      4×4 inverse per pair, which is wasteful for N~2000 and a smooth
      population posterior. If you need full covariance (e.g. for
      strongly correlated PE products), this container can be extended
      to carry the precomputed inverse covariance + log-determinant.
    - All arrays are float64 to match darksirens' jax_enable_x64 setting.
    - Building a PairKDE is **not** JIT-compatible — it uses sample
      statistics. Call ``make_pair_kde`` once per event at data-load
      time, not inside the likelihood.
    """
    samples: Any
    log_weights: Any
    log_h: Any
    log_norm: Any
    valid: Any


def _silverman_bandwidth_diag(
    samples: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Silverman's rule for diagonal Gaussian KDE in d dimensions.

    h_k = (4 / ((d + 2) N))^(1/(d+4)) · σ_k

    σ_k is the unweighted standard deviation along axis k over valid
    samples. N is the count of valid samples.
    """
    n, d = samples.shape
    if valid is None:
        valid = np.ones(n, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(
            f"PairKDE: need at least 2 valid samples, got {n_valid}."
        )
    s = samples[valid]
    sigma = s.std(axis=0, ddof=1)
    factor = (4.0 / ((d + 2) * n_valid)) ** (1.0 / (d + 4))
    h = factor * sigma
    h = np.where(h > 0.0, h, 1.0e-8)
    return h


def make_pair_kde(
    m1det: np.ndarray,
    q: np.ndarray,
    dL_app: np.ndarray,
    chieff: np.ndarray,
    prior_wt: np.ndarray,
    valid: np.ndarray | None = None,
) -> PairKDE:
    """Build a PairKDE for one event.

    Parameters
    ----------
    m1det, q, dL_app, chieff
        Per-sample apparent-frame PE samples. All shape (N_pe,). Drawn
        from the per-event posterior π_PE.
    prior_wt
        Per-sample PE proposal density values (NOT log). These are the
        per-sample p_prop values from the LALInference output.
    valid
        Optional structural mask (matches GWEvent.valid). Defaults to all
        True. Padded samples are excluded from the bandwidth fit and
        masked out at evaluation time.

    Returns
    -------
    PairKDE

    Notes
    -----
    The Silverman bandwidth uses the *unweighted* posterior sample
    statistics. The importance weights 1/p_prop enter the KDE *value*
    (they correct π_PE → π_PE/p_prop at evaluation time), but they
    should not enter the bandwidth — the bandwidth is a property of the
    posterior's spatial scale, not of the proposal density.
    """
    m1det = np.asarray(m1det, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dL_app = np.asarray(dL_app, dtype=np.float64)
    chieff = np.asarray(chieff, dtype=np.float64)
    prior_wt = np.asarray(prior_wt, dtype=np.float64)
    samples = np.stack([m1det, q, dL_app, chieff], axis=-1)
    n = samples.shape[0]
    if valid is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    prior_wt = validate_pair_prior_wt(prior_wt, valid, context="make_pair_kde prior_wt")

    # Bandwidth from UNWEIGHTED posterior samples
    h = _silverman_bandwidth_diag(samples, valid=valid)

    # KDE evaluation weights: 1/p_prop for the importance correction
    log_w = np.where(
        valid & (prior_wt > 0.0),
        -np.log(prior_wt),
        -np.inf,
    )
    log_h = np.log(h)
    log_norm = -0.5 * _D * np.log(2.0 * np.pi) - log_h.sum()

    return PairKDE(
        samples=jnp.asarray(samples),
        log_weights=jnp.asarray(log_w),
        log_h=jnp.asarray(log_h),
        log_norm=jnp.asarray(log_norm),
        valid=jnp.asarray(valid),
    )


def stack_pair_kdes(kdes: list) -> PairKDE:
    """Stack a Python list of nEvents PairKDE objects into a single
    PairKDE whose leaves have a leading event axis.

    Returns a PairKDE with fields:
        samples       : (nEvents, N_pe, 4)
        log_weights   : (nEvents, N_pe)
        log_h         : (nEvents, 4)
        log_norm      : (nEvents,)
        valid         : (nEvents, N_pe)

    Use ``lax.dynamic_index_in_dim(stacked.<field>, i, axis=0)`` to
    extract event i's KDE inside JIT-compiled code.

    All input KDEs must have identical ``N_pe`` (pad upstream if needed).
    """
    if len(kdes) == 0:
        raise ValueError("stack_pair_kdes: empty input list")
    n_pe = kdes[0].samples.shape[0]
    for k, kde in enumerate(kdes):
        if kde.samples.shape[0] != n_pe:
            raise ValueError(
                f"stack_pair_kdes: PairKDE {k} has N_pe={kde.samples.shape[0]}, "
                f"expected {n_pe}. Pad PE arrays upstream so all events have "
                f"the same length."
            )
    return PairKDE(
        samples=jnp.stack([k.samples for k in kdes], axis=0),
        log_weights=jnp.stack([k.log_weights for k in kdes], axis=0),
        log_h=jnp.stack([k.log_h for k in kdes], axis=0),
        log_norm=jnp.stack([k.log_norm for k in kdes], axis=0),
        valid=jnp.stack([k.valid for k in kdes], axis=0),
    )


def _slice_event_kde_inside_jit(stacked: PairKDE, event_idx) -> PairKDE:
    """Internal: extract one event's PairKDE from a stacked container.

    Used by the master likelihood's per-pair loop. ``event_idx`` may be
    a traced scalar.
    """
    from jax import lax
    return PairKDE(
        samples=lax.dynamic_index_in_dim(stacked.samples, event_idx, axis=0, keepdims=False),
        log_weights=lax.dynamic_index_in_dim(stacked.log_weights, event_idx, axis=0, keepdims=False),
        log_h=lax.dynamic_index_in_dim(stacked.log_h, event_idx, axis=0, keepdims=False),
        log_norm=lax.dynamic_index_in_dim(stacked.log_norm, event_idx, axis=0, keepdims=False),
        valid=lax.dynamic_index_in_dim(stacked.valid, event_idx, axis=0, keepdims=False),
    )


def log_eval_pair_kde(
    kde: PairKDE,
    theta_app: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate ``log p̂(θ_app) = log [π_PE(θ) / p_prop(θ)]`` from the
    precomputed KDE.

    Parameters
    ----------
    kde : PairKDE
    theta_app : (..., 4)
        Query points in apparent-frame coordinates. Last axis must match
        ``PAIR_KDE_COORDS``.

    Returns
    -------
    log_p : (...,) array
        log of the importance-weighted KDE density estimator. The
        ESTIMAND is ``π_PE(θ) / p_prop(θ)``, the apparent-frame
        PE-evidence density that appears in the cluster-pair likelihood.

    Why this normalization
    ----------------------
    PE samples are drawn from the posterior π_PE. The MC importance
    estimator:

        p̂(θ) = (1/N) Σ_t K_h(θ - θ_t) / p_prop(θ_t)

    has expectation ∫ K_h(θ-θ') · π_PE(θ')/p_prop(θ') dθ' → π_PE(θ)/p_prop(θ),
    which is exactly the apparent-frame data likelihood p(d|θ) (up to the
    overall PE evidence Z_PE, which cancels in the Bayes factor).

    Do NOT self-normalize by Σ w_t: that would estimate π_PE(θ) instead
    of π_PE(θ)/p_prop(θ), giving the wrong target. The standard
    darksirens singleton sample-weight formula confirms this — it has
    -log(prior_wt) per sample, NOT -log(prior_wt) + log(Σ 1/prior_wt).

    Boundary correction at q = 1
    ----------------------------
    q has a hard physical boundary at 1 where equal-mass posteriors pile
    up. A plain Gaussian kernel leaks half its mass past the boundary
    there, underestimating the density by up to 2× as q → 1. We use the
    standard reflection estimator: each sample also contributes a mirror
    kernel about q = 1 (at 2 - q_t), which restores unit kernel mass on
    q ≤ 1 and is exponentially negligible away from the boundary. No
    change to log_norm is needed.
    """
    from jax.scipy.special import logsumexp

    q_axis = PAIR_KDE_COORDS.index("q")
    diff = (theta_app[..., None, :] - kde.samples[None, :, :]) / jnp.exp(kde.log_h)
    sq = diff * diff                                              # (..., N, 4)
    sq_rest = jnp.sum(sq, axis=-1) - sq[..., q_axis]              # (..., N)
    # Mirror image of each sample's q about the boundary: 2 - q_t.
    diff_q_ref = (
        theta_app[..., q_axis][..., None] + kde.samples[None, :, q_axis] - 2.0
    ) / jnp.exp(kde.log_h[q_axis])
    log_kernel_q = jnp.logaddexp(
        -0.5 * sq[..., q_axis], -0.5 * diff_q_ref * diff_q_ref
    )
    log_kernel = -0.5 * sq_rest + log_kernel_q                    # (..., N)

    log_w = kde.log_weights                                       # (N,)
    log_sum = logsumexp(log_kernel + log_w, axis=-1)              # (...,)
    log_N = jnp.log(jnp.asarray(kde.samples.shape[0], dtype=jnp.float64))
    return kde.log_norm + log_sum - log_N
