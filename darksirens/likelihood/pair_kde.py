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

Bandwidth
~~~~~~~~~
Silverman's rule is AMISE-optimal for DENSITY ESTIMATION, and at d = 4 its
exponent 1/(d+4) = 1/8 makes it almost N-independent: h = 0.45 σ_k at
N = 400 (the ``--pe_max_per_pair 400`` production setting), 0.37 σ_k at
N = 2000.  The estimand is therefore not π_PE/p_prop but its convolution
with a ~0.4 σ Gaussian in every coordinate, which INFLATES the tails: for a
Gaussian marginal the smoothed/true density ratio at separation Δ is

    (σ/√(σ² + h²)) · exp[(Δ²/2) h²/(σ²(σ² + h²))]

= 0.91 at Δ = 0 and 1.9× at Δ = 3 σ for h = 0.45 σ, per coordinate.  Only the
LENSED branch uses a KDE (cluster_likelihood.py's pair integrand); the
singleton/unlensed weight is an unsmoothed PE sum, so the smoothing bias does
NOT cancel in the pair Bayes factor and it favours widely separated (false)
pairs.  Note WHY it cannot be made to cancel: the pair branch evaluates a
density POINTWISE, where the kernel inflates the tails, while the singleton
branch integrates the population model against the event's own samples, which
a narrow kernel leaves almost unchanged.  The two are not two evaluations of
one estimator, so smoothing both sides does not remove the bandwidth
dependence — it has to be attacked in the kernel itself.

What we do about it: the kernel is scaled to the sample COVARIANCE, not to the
marginal widths (``_whitening_transform``).  σ_k above is a marginal width, and
PE posteriors are thin correlated ridges in (m1det, q, dL_app, χ_eff), so the
old diagonal kernel smoothed across the ridge by many times the posterior's
real extent there — the dominant source of the tail inflation.  Silverman's
rule is unchanged; it is simply applied in coordinates where its unit-variance
premise holds.

``make_pair_kde(..., bandwidth_scale=...)`` remains the sensitivity handle:
rebuild a candidate pair at 1.0 and 0.5 and compare the pair log Bayes factor —
a shift of order nats means the kernel, not the physics, is setting the answer.
The default is left at the published rule; changing it is a validation
exercise, not a code fix.
"""

from __future__ import annotations

from typing import NamedTuple, Any
import numpy as np
import jax.numpy as jnp


# Coordinate axes carried through the KDE in canonical order.
PAIR_KDE_COORDS = ("m1det", "q", "dL_app", "chieff")
_D = 4   # KDE dimensionality (fixed for this analysis)

#: Bandwidth used for a coordinate with zero sample spread. See
#: :func:`_silverman_bandwidth_diag` — this is a documented degenerate-dimension
#: policy, not a numerical epsilon.
DEGENERATE_H = 1.0e-8


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
        sample ordering. Kept for provenance and diagnostics; the kernel sum
        runs on ``samples_w``.
    samples_w : (N_pe, 4) float64
        The same samples in WHITENED coordinates, ``u_t = L^-1 (θ_t - μ)``.
        The kernel is isotropic here, so the evaluation loop stays a
        coordinate-by-coordinate diagonal reduction (no (..., N, 4) temporary).
    mean : (4,) float64
        Sample mean μ used by the whitening transform.
    l_inv : (4, 4) float64
        Inverse Cholesky factor ``L^-1`` of the sample covariance, applied to
        QUERY points at evaluation time.
    log_weights : (N_pe,) float64
        Per-sample log weights ``log(1 / p_prop)`` for importance correction.
        These are passed through to the kernel sum so the resulting density
        is the posterior-equivalent KDE, not the proposal-density KDE.
    log_h : (4,) float64
        Log of the Silverman bandwidths in WHITENED coordinates. Whitening
        makes the sample covariance the identity, so all four entries are the
        same Silverman factor unless a coordinate was degenerate. Stored as
        a log, and per-coordinate, so the degenerate-dimension policy below
        and the stacking helpers keep working unchanged.
    log_norm : scalar float64
        Log of the Gaussian-kernel normalization constant
        ``-0.5 * d * log(2π) - sum(log_h) - log|det L|``, ready to add into
        the log-evaluated density. The ``log|det L|`` term is the Jacobian of
        the whitening transform.
    valid : (N_pe,) bool
        Validity mask carried alongside samples (matches GWEvent.valid).
        Padded entries are excluded from BOTH the kernel sum and the 1/N
        normalization (see :func:`log_eval_pair_kde`), so padding a KDE to a
        longer common ``N_pe`` leaves its density unchanged.

    Notes
    -----
    - The bandwidth is full-covariance, implemented as a diagonal rule in
      whitened coordinates: ``H = h_w² C`` for the sample covariance ``C``.
      See :func:`_whitening_transform` for why the marginal-width diagonal
      rule this replaced inflated the tails.
    - All arrays are float64 to match darksirens' jax_enable_x64 setting.
    - Building a PairKDE is **not** JIT-compatible — it uses sample
      statistics. Call ``make_pair_kde`` once per event at data-load
      time, not inside the likelihood.
    """
    samples: Any
    samples_w: Any
    mean: Any
    l_inv: Any
    log_weights: Any
    log_h: Any
    log_norm: Any
    valid: Any


def _silverman_bandwidth_diag(
    samples: np.ndarray,
    valid: np.ndarray | None = None,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Silverman's rule for diagonal Gaussian KDE in d dimensions.

    h_k = scale · (4 / ((d + 2) N))^(1/(d+4)) · σ_k

    σ_k is the unweighted standard deviation along axis k over valid
    samples. N is the count of valid samples. ``bandwidth_scale`` multiplies
    the rule (see the module docstring's "Bandwidth" note for why one would
    want it below 1).

    Degenerate coordinates
    ~~~~~~~~~~~~~~~~~~~~~~
    A coordinate with σ_k = 0 (χ_eff pinned by a zero-spin PE run, a single
    surviving q value after aggressive downsampling, a mock with one parameter
    fixed) cannot support a KDE, and the ``DEGENERATE_H`` floor below is a
    PHYSICS decision dressed as a numerical safety net: it makes the kernel a
    delta in that coordinate — any query displaced by more than ~1e-8 gets zero
    density, annihilating the whole pair branch — and it adds
    ``-log(1e-8) = +18.4`` nats to ``log_norm``, a per-event offset that does NOT
    cancel in the pair Bayes factor (the unlensed branch uses no KDE).  It is
    kept so a synthetic PE product still runs, but it warns and names the
    coordinate: the caller should drop that dimension from the pair kernel or
    supply real samples.
    """
    n, d = samples.shape
    if valid is None:
        valid = np.ones(n, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(
            f"PairKDE: need at least 2 valid samples, got {n_valid}."
        )
    if not (bandwidth_scale > 0.0):
        raise ValueError(
            f"PairKDE: bandwidth_scale must be positive; got {bandwidth_scale}."
        )
    s = samples[valid]
    sigma = s.std(axis=0, ddof=1)
    factor = (4.0 / ((d + 2) * n_valid)) ** (1.0 / (d + 4))
    h = float(bandwidth_scale) * factor * sigma
    degenerate = ~(h > 0.0)
    if degenerate.any():
        import warnings

        names = [
            PAIR_KDE_COORDS[k] if k < len(PAIR_KDE_COORDS) else str(k)
            for k in np.flatnonzero(degenerate)
        ]
        warnings.warn(
            f"PairKDE: zero sample spread in coordinate(s) {names} over "
            f"{n_valid} valid PE samples; falling back to a delta-like "
            f"bandwidth h={DEGENERATE_H:g} there. The kernel then rejects any "
            "query displaced from the constant value, and log_norm carries "
            f"{-np.log(DEGENERATE_H):.1f} nats per degenerate coordinate that do "
            "NOT cancel in the pair Bayes factor. Drop the dimension or supply "
            "real samples."
        )
    h = np.where(degenerate, DEGENERATE_H, h)
    return h


def _silverman_factor(n_valid: int, d: int = _D) -> float:
    """Silverman's N-dependent factor ``(4 / ((d + 2) N))^(1/(d+4))``.

    In whitened coordinates every marginal has unit variance, so this factor
    IS the bandwidth: ``h_w = scale · factor``.
    """
    return float((4.0 / ((d + 2) * n_valid)) ** (1.0 / (d + 4)))


def _whitening_transform(samples: np.ndarray, valid: np.ndarray):
    """Cholesky whitening of the valid PE samples.

    Returns ``(mean, l_inv, log_det_l, degenerate)``: the sample mean, the
    inverse Cholesky factor of the sample covariance, ``log|det L|``, and the
    per-coordinate degenerate mask.

    Why whiten
    ~~~~~~~~~~
    The diagonal rule this replaced set ``h_k = factor · σ_k`` with ``σ_k`` the
    MARGINAL standard deviation.  PE posteriors in ``(m1det, q, dL_app, χ_eff)``
    are strongly correlated — chirp mass is measured far better than either
    component mass, and dL_app tracks m1det through the redshift — so the
    posterior is a thin ridge whose width ACROSS the ridge is a small fraction
    of any marginal width.  A kernel scaled to the marginals therefore smears
    the density by many times the posterior's actual extent in the tight
    directions, and since the pair branch reads this density POINTWISE (at the
    point event i predicts for event j, cluster_likelihood.py) that shows up
    directly as inflated tail density: the finding measured 1.9x per coordinate
    at 3σ, i.e. ~14x and +2.7 nats of spurious lensed-pair evidence for a false
    pair separated by 3σ in all four coordinates.  Scaling the kernel to the
    sample COVARIANCE instead makes the smoothing anisotropic in the same way
    the posterior is, so it no longer spills across the ridge.  Silverman's rule
    is untouched — it is applied in the coordinates where its unit-variance
    assumption actually holds.

    Degenerate coordinates
    ~~~~~~~~~~~~~~~~~~~~~~
    A coordinate with zero spread makes the covariance singular. It keeps the
    documented :data:`DEGENERATE_H` policy: its variance is floored to
    ``DEGENERATE_H²`` and its cross-covariances zeroed, so it stays independent
    and delta-like exactly as under the diagonal rule, and the remaining
    coordinates whiten normally.
    """
    s = samples[valid]
    n_valid, d = s.shape
    mean = s.mean(axis=0)
    cov = np.cov(s, rowvar=False, ddof=1).reshape(d, d)
    sigma = s.std(axis=0, ddof=1)
    degenerate = ~(sigma > 0.0)
    if degenerate.any():
        cov[degenerate, :] = 0.0
        cov[:, degenerate] = 0.0
        cov[degenerate, degenerate] = DEGENERATE_H ** 2
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        # Exactly collinear coordinates (e.g. q reconstructed from m1det and a
        # fixed m2det in a synthetic PE product) leave the covariance singular
        # with no zero-variance coordinate to flag.  Fall back to the marginal
        # diagonal rather than failing the run, and say so: the fallback is the
        # tail-inflating behaviour this transform exists to remove.
        import warnings

        warnings.warn(
            "PairKDE: sample covariance is not positive definite "
            f"({exc}); falling back to the MARGINAL diagonal bandwidth for "
            "this event. Its pair density will carry the tail inflation that "
            "whitening removes (review F-034) -- check the PE product for "
            "collinear coordinates."
        )
        L = np.diag(np.where(degenerate, DEGENERATE_H, sigma))
    l_inv = np.linalg.inv(L)
    log_det_l = float(np.sum(np.log(np.diag(L))))
    return mean, l_inv, log_det_l, degenerate


def make_pair_kde(
    m1det: np.ndarray,
    q: np.ndarray,
    dL_app: np.ndarray,
    chieff: np.ndarray,
    prior_wt: np.ndarray,
    valid: np.ndarray | None = None,
    bandwidth_scale: float = 1.0,
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
    bandwidth_scale
        Multiplier on Silverman's bandwidth (default 1.0, i.e. the rule as
        published). Lowering it trades variance for the tail bias quantified
        in the module docstring's "Bandwidth" note; rebuilding a candidate
        pair's KDEs at 1.0 and 0.5 and comparing the pair log Bayes factor is
        the sensitivity check that note asks for.

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

    if not (bandwidth_scale > 0.0):
        raise ValueError(
            f"PairKDE: bandwidth_scale must be positive; got {bandwidth_scale}."
        )
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f"PairKDE: need at least 2 valid samples, got {n_valid}.")

    # Bandwidth from UNWEIGHTED posterior samples, in whitened coordinates.
    mean, l_inv, log_det_l, degenerate = _whitening_transform(samples, valid)
    if degenerate.any():
        import warnings

        names = [
            PAIR_KDE_COORDS[k] if k < len(PAIR_KDE_COORDS) else str(k)
            for k in np.flatnonzero(degenerate)
        ]
        warnings.warn(
            f"PairKDE: zero sample spread in coordinate(s) {names} over "
            f"{n_valid} valid PE samples; falling back to a delta-like "
            f"bandwidth h={DEGENERATE_H:g} there. The kernel then rejects any "
            "query displaced from the constant value, and log_norm carries "
            f"{-np.log(DEGENERATE_H):.1f} nats per degenerate coordinate that do "
            "NOT cancel in the pair Bayes factor. Drop the dimension or supply "
            "real samples."
        )
    # Whitened coordinates have unit variance in every direction, so Silverman's
    # sigma_k is 1 and the rule collapses to its N-dependent factor alone.
    h = np.full(_D, float(bandwidth_scale) * _silverman_factor(n_valid))
    # Padded rows commonly carry NaN coordinates; zero them BEFORE the matmul
    # so the whitened table has no NaN to mask later (the evaluator masks too,
    # but a NaN that never exists cannot leak into a gradient).
    samples_w = (np.where(valid[:, None], samples, 0.0) - mean) @ l_inv.T
    samples_w = np.where(valid[:, None], samples_w, 0.0)

    # KDE evaluation weights: 1/p_prop for the importance correction.
    # Padding slots commonly carry 0/NaN p_prop, so substitute a dummy before
    # the log rather than taking log(0) and discarding the warning afterwards.
    usable = valid & np.isfinite(prior_wt) & (prior_wt > 0.0)
    log_w = np.where(usable, -np.log(np.where(usable, prior_wt, 1.0)), -np.inf)
    log_h = np.log(h)
    # H = h_w^2 C, so |det H|^(1/2) = h_w^d |det L| and the Jacobian of the
    # whitening transform enters the normalization as log|det L|.
    log_norm = -0.5 * _D * np.log(2.0 * np.pi) - log_h.sum() - log_det_l

    return PairKDE(
        samples=jnp.asarray(samples),
        samples_w=jnp.asarray(samples_w),
        mean=jnp.asarray(mean),
        l_inv=jnp.asarray(l_inv),
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
        samples_w=jnp.stack([k.samples_w for k in kdes], axis=0),
        mean=jnp.stack([k.mean for k in kdes], axis=0),
        l_inv=jnp.stack([k.l_inv for k in kdes], axis=0),
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
        samples_w=lax.dynamic_index_in_dim(stacked.samples_w, event_idx, axis=0, keepdims=False),
        mean=lax.dynamic_index_in_dim(stacked.mean, event_idx, axis=0, keepdims=False),
        l_inv=lax.dynamic_index_in_dim(stacked.l_inv, event_idx, axis=0, keepdims=False),
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

    Padding
    -------
    ``N`` in the 1/N above is the number of VALID samples, not the padded
    array length: ``stack_pair_kdes`` requires a common ``N_pe``, so events
    with fewer PE samples arrive padded, and normalizing by the padded length
    would shift every log density of a padded event by ``log(n_valid/N_pe)``
    (exactly -log(5/3) for 3 samples padded to 5) — a per-event constant that
    does NOT cancel in the pair Bayes factor.  Padded rows' COORDINATES are
    also sanitized to a finite dummy before the kernel arithmetic: they are
    commonly NaN, and ``NaN + (-inf) = NaN`` poisons the logsumexp forward,
    while masking only afterwards still poisons the BACKWARD pass (both
    branches of a ``where`` are differentiated, and a NaN survives a zero
    cotangent — the reverse-mode class documented at
    ``redshift/catalog.py:_logsumexp_neginf_safe``).

    Boundary correction at q = 1
    ----------------------------
    q has a hard physical boundary at 1 where equal-mass posteriors pile
    up. A plain Gaussian kernel leaks half its mass past the boundary
    there, underestimating the density by up to 2× as q → 1. We use the
    standard reflection estimator, which restores unit kernel mass on q ≤ 1
    and is exponentially negligible away from the boundary. No change to
    log_norm is needed.

    The reflection is applied to the QUERY, not to the sample. Under the old
    diagonal kernel the two were interchangeable — only the squared q
    difference entered, and |q - (2 - q_t)| = |(2 - q) - q_t| — but the kernel
    is now full-covariance, and only the query reflection stays exact:

        ∫_{q ≤ 1} K_C(Rθ - θ_t) dθ = ∫_{q ≥ 1} K_C(θ - θ_t) dθ,

    so the two terms sum to the kernel's total mass for ANY covariance C.
    Reflecting the sample instead would require also flipping the sign of q's
    cross-covariances to preserve that identity, and silently loses unit mass
    if you do not.
    """
    from jax.scipy.special import logsumexp

    q_axis = PAIR_KDE_COORDS.index("q")
    valid = jnp.asarray(kde.valid, dtype=bool)                    # (N,)
    # Sanitize BEFORE any arithmetic (see "Padding" above): invalid rows carry
    # no NaN/inf onto the differentiable path, so neither the forward
    # logsumexp nor its transpose can be poisoned by padding.
    samples = jnp.where(valid[:, None], kde.samples_w, 0.0)       # (N, 4) whitened
    log_w = jnp.where(valid, kde.log_weights, 0.0)                # (N,)

    # Whiten the query, and its mirror image about the q = 1 boundary.  The
    # kernel is isotropic in these coordinates, so the reduction below stays
    # diagonal and never materializes a (..., N, 4) temporary.
    theta_ref = theta_app.at[..., q_axis].set(2.0 - theta_app[..., q_axis])
    u = (theta_app - kde.mean) @ kde.l_inv.T                      # (..., 4)
    u_ref = (theta_ref - kde.mean) @ kde.l_inv.T                  # (..., 4)

    # Accumulate the squared distance coordinate BY coordinate.  A (..., N, 4)
    # difference array is 4x the (..., N) reduction it feeds -- at the caller's
    # (N_pe_i, N_y, 4) query with N_pe = 400, N_y = 64 that is 327 MB per
    # temporary -- and building ``sq_rest`` as ``sum(sq) - sq[q_axis]`` also
    # cancelled the other three coordinates away whenever the q term dominated
    # the sum by many orders of magnitude.
    h = jnp.exp(kde.log_h)                                        # (4,)
    sq = jnp.zeros(())
    sq_ref = jnp.zeros(())
    for k in range(_D):
        d_k = (u[..., k][..., None] - samples[:, k]) / h[k]
        sq = sq + d_k * d_k                                       # (..., N)
        d_k_ref = (u_ref[..., k][..., None] - samples[:, k]) / h[k]
        sq_ref = sq_ref + d_k_ref * d_k_ref                       # (..., N)
    log_kernel = jnp.logaddexp(-0.5 * sq, -0.5 * sq_ref)          # (..., N)

    terms = jnp.where(valid, log_kernel + log_w, -jnp.inf)        # (..., N)
    # All--inf-safe reduction (cf. ``_logsumexp_neginf_safe``): the -1e30
    # sentinel underflows to exactly zero weight, so a row with any valid
    # sample is bit-identical to a plain logsumexp, while an all-invalid row
    # returns exactly -inf with finite gradients instead of a 0/0 softmax.
    finite = jnp.isfinite(terms)
    safe = jnp.where(finite, terms, -1e30)
    log_sum = jnp.where(
        jnp.any(finite, axis=-1), logsumexp(safe, axis=-1), -jnp.inf
    )                                                             # (...,)
    # Normalize by the number of VALID samples, not the padded length.
    n_valid = jnp.sum(valid.astype(jnp.float64))
    log_N = jnp.log(jnp.maximum(n_valid, 1.0))
    return kde.log_norm + log_sum - log_N
