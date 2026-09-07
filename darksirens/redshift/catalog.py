"""
catalog.py
----------
EM catalog redshift prior: p_cat(z | pix), the *shape* of the host
probability over the observed galaxies in a pixel.

Each real galaxy i contributes a kernel that is the normalised
volumetric photo-z posterior

    p(z | gal i) = N(z; z_i, sigma_eff,i) * g(z) / Z_i,
    Z_i          = ∫_0^zmax N(z'; z_i, sigma_eff,i) g(z') dz',

with the galaxy measure g(z) = dV_c/dz * (1+z)^delta and

    sigma_eff,i = max( sqrt(sigma_cat,i^2 + sigma_kde^2), 1e-4 ).

The Gaussian is the photo-z/instrumental likelihood broadened by the
LSS kernel sigma_kde (variances add for Gaussian overlap); g(z) is the
volumetric interim prior on the galaxy's true redshift (Gray et al.
2020, arXiv:1908.06050).  Because Z_i normalises each kernel to unit
mass on [0, zmax], every galaxy carries exactly its base weight:

    p_cat(z | pix) = Σ_i  w~_i * p(z | gal i),     w~_i = w_i / Σ_j w_j,

which integrates to 1 per pixel.  Crucially the volumetric prior tilts
each kernel but does NOT rescale a galaxy's total host probability —
multiplying the mixture weights by dV(z_i) (a previous implementation)
makes a galaxy more probable as a host merely for being far away.

Z_i is evaluated by Gauss–Legendre quadrature in the Gaussian quantile
variable (exact treatment of the [0, zmax] truncation; robust for any
sigma_eff from the 1e-4 floor up to broad photo-z), with g interpolated
from a per-proposal grid.  The sigma_eff floor (~30 km/s) only protects
numerics when spectroscopic errors underflow; it is far below any
physical redshift uncertainty.

Hot paths precompute the per-galaxy quantities once per parameter
proposal via ``catalog_kernel_state`` and evaluate per sample with
``eval_log_catalog_prior_state``; the scalar ``log_catalog_prior`` keeps
the historical signature for the complete-catalog and bright-siren
models and for tests.

When Om0, w0, wa, delta and sigma_kde are all FIXED — the production
configuration — the quadrature is not per-proposal work at all: g(z)
is then exactly (H0_ref/H0)^3 g(z; H0_ref), so Z_i and log_kw are a
build-time constant plus the scalar shift 3 ln(H0/H0_ref) and log p_cat
carries no H0 information.  ``PinnedKernelQuadrature`` is that constant,
built once by the likelihood factory and re-verified in the graph on
every proposal.
"""

from __future__ import annotations

import weakref

import numpy as np

from darksirens.utils.utils import array_shape
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from jax.scipy.stats import norm
from typing import NamedTuple, Any

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

from darksirens.redshift.grid import log_interp_zgrid, zgrid
from darksirens.utils.cosmology import H0Planck, threads_distance_table
from .completion import log_galaxy_measure_grid

_ZMAX: float = float(np.asarray(zgrid)[-1])
_HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))

#: Numerical floor on sigma_eff [redshift units]; ~30 km/s, well below any
#: physical photo-z or peculiar-velocity scale.  Protects against
#: spectroscopic dzgals -> 0 with sigma_kde = 0.
SIGMA_EFF_FLOOR: float = 1e-4

# Gauss–Legendre nodes/weights on [0, 1] for the kernel normalisation.
# The default 24 nodes are conservative for broad photo-z kernels; for
# spectroscopic catalogs (sigma_eff ~ 1e-3, g(z) locally smooth) far fewer
# nodes are exact to likelihood precision and the quadrature dominates the
# per-proposal cost of wide-sky runs, so the count is configurable.
#
# Two quadrature domains are available:
#
#   ``'cdf'`` (default, historical) — GL in the Gaussian CDF variable
#   u = Phi((z - z_i)/sig).  The change of variables removes the Gaussian
#   weight, leaving only g(z) for GL to integrate: mathematically optimal,
#   but requires per-galaxy ``ndtri`` evaluations (expensive).
#
#   ``'zspace'`` — GL directly in redshift on [z_i - n_sigma*sig,
#   z_i + n_sigma*sig] clipped to [0, z_hi].  Avoids ``ndtri`` entirely:
#   the integrand is ``N(z; z_i, sig) g(z)``, evaluated with cheap
#   ``exp(-0.5 d^2)`` instead.  Benchmarked 2–3x faster per row than
#   ``'cdf'`` at the same node count.  Node guidance (measured against an
#   exact analytic reference on the DESI nside-64 production catalog, a
#   MIXED spectro+photo catalog with sig_eff spanning 3e-3 to 0.1):
#   16 nodes at n_sigma=5 gives per-galaxy |dlog Z| median 9e-7 / max 6e-5
#   (75x tighter than cdf-24 coherently over 259 events) and is the first
#   node count that beats cdf-24; 12 is a wash; 8 at n_sigma=5 is ~9e-3
#   per galaxy -- 130x WORSE than cdf-24.  n_sigma=4 pins an -8e-5
#   truncation bias no node count can remove; use n_sigma >= 5.
_GL_NODES = 24
_GL_DOMAIN = 'cdf'
_GL_NSIGMA = 5.0
_GL_X = None
_GL_W = None


def configure_kernel_quadrature(n_nodes: int = 24, domain: str = 'cdf',
                                n_sigma: float = 5.0):
    """Set the Gauss–Legendre scheme for the kernel normalisation Z_i.

    Trace-time configuration: call BEFORE the first likelihood evaluation
    (reconfiguring after a jit trace does not affect the compiled graph).

    ``n_nodes``
        Node count.  24 (default) is safe for any kernel width; 8 is
        validated for spectroscopic catalogs across the sampled sigma_kde
        prior (see tests/test_catalog_row_chunk.py); 4 is only safe when
        sigma_kde is FIXED near zero — its error grows ~20x by
        sigma_kde = 0.05.

    ``domain``
        ``'cdf'`` (default) — GL in CDF space; ``'zspace'`` — GL in
        redshift space (avoids ``ndtri``, ~2x faster, see module header).

    ``n_sigma``
        Half-width of the z-space integration range in units of sigma_eff
        (only used when ``domain='zspace'``; default 5.0).  Values >= 4
        lose < 1e-4 of the Gaussian tail mass.
    """
    global _GL_NODES, _GL_DOMAIN, _GL_NSIGMA, _GL_X, _GL_W
    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise ValueError(f"kernel quadrature needs >= 2 nodes, got {n_nodes}")
    if domain not in ('cdf', 'zspace'):
        raise ValueError(f"domain must be 'cdf' or 'zspace', got {domain!r}")
    _GL_NODES = n_nodes
    _GL_DOMAIN = domain
    _GL_NSIGMA = float(n_sigma)
    _glx, _glw = np.polynomial.legendre.leggauss(_GL_NODES)
    _GL_X = jnp.asarray(0.5 * (_glx + 1.0))
    _GL_W = jnp.asarray(0.5 * _glw)


configure_kernel_quadrature(_GL_NODES)


def _row_real_mask(zs, ws, ngal):
    """Real-galaxy mask for one padded row."""
    if ngal is not None:
        return jnp.arange(zs.shape[0]) < ngal
    return ws > 0


# Row-chunked mapping over catalog rows.  The per-row kernel-norm quadrature
# materialises (N_max_gals, K) intermediates; a plain vmap over all rows holds
# (N_rows, N_max_gals, K) at once, which OOMs for wide-sky runs (e.g. a
# 49k-row x 2113-galaxy DESI view requests ~80 GB).  Above the element
# threshold the vmap is wrapped in ``lax.map`` over fixed-size row chunks:
# identical per-row arithmetic (no cross-row ops), bounded peak memory.
_ROW_CHUNK_AUTO_THRESHOLD: int = 2**25   # N_rows * N_max_gals elements
_ROW_CHUNK_SIZE: int = 512
_ROW_CHUNK_MODE = "auto"                 # "auto" | None | int (for tests)


def configure_catalog_row_chunk(mode="auto"):
    """Set row-chunking for kernel-state builds: "auto", None (off), or an int.

    Trace-time configuration: call BEFORE the first likelihood evaluation.
    Reconfiguring after a function has been jit-traced has no effect on the
    already-compiled graph (the mode is read when the trace is built).
    """
    global _ROW_CHUNK_MODE
    if mode is not None and mode != "auto":
        mode = int(mode)
        if mode < 1:
            raise ValueError(f"row chunk must be >= 1, got {mode}")
    _ROW_CHUNK_MODE = mode


def _resolve_row_chunk(n_rows: int, n_max: int) -> int | None:
    if _ROW_CHUNK_MODE is None:
        return None
    if _ROW_CHUNK_MODE != "auto":
        return int(_ROW_CHUNK_MODE)
    if n_rows * n_max > _ROW_CHUNK_AUTO_THRESHOLD:
        return _ROW_CHUNK_SIZE
    return None


# ------------------------------------------------------------
# Windowed per-sample catalog KDE (issue #280)
# ------------------------------------------------------------
# ``eval_log_catalog_prior_state`` historically gathered the whole padded row
# and evaluated a Gaussian for every galaxy in it, per sample: under vmap over
# samples that materialises (N_samp, N_max) transients per event even though at
# sigma_eff ~ 0.02 the kernel support is a few percent of the row's redshift
# range.  When the catalog rows are z-sorted (the load-time invariant
# established by ``darksirens.catalogs.io.sort_survey_rows_by_z``), ONE binary
# search locates the sample's insertion index in the row and only a
# STATIC-size window of the W index-nearest galaxies, centred there, is
# gathered and evaluated (``_sorted_row_window_start``).
#
# W is static (jit-compatible) and is what carries the accuracy contract: sized
# by ``recommended_kde_window`` -- twice the largest one-sided count of galaxies
# within n_sigma * max_row(sigma_eff) at the widest sigma_kde the run can reach,
# which is what the likelihood factory does through ``auto_kde_window`` -- the
# centred window provably holds every galaxy within n_sigma widths of any
# sample, so the evaluator never truncates inside the contract.  A sample beyond
# the row's support straddles the nearest galaxies (far-tail evaluations stay
# accurate to a few hundred nats below the row's kernel maximum, then degrade
# and floor at -inf at the backend's exp underflow edge -- see the one-pass
# branch in ``eval_log_catalog_prior_state``); windows with zero real galaxies go
# through the same reduction as empty full rows (log_kw is sanitized to -1e30 at
# state build time, so the reduction has finite gradients).
_KDE_WINDOW_SIZE = 1024                # static window size W; None = full row
_KDE_WINDOW_NSIGMA: float = 8.0        # sizing multiplier: n_sigma * max(sig_eff)


def configure_catalog_kde_window(size=1024, n_sigma=8.0):
    """Configure the windowed per-sample catalog-KDE evaluator.

    Trace-time configuration: call BEFORE the first likelihood evaluation
    (reconfiguring after a jit trace does not affect the compiled graph).

    ``size`` is the static window length W (galaxies gathered per sample);
    ``None`` disables windowing entirely — the full-row escape hatch for A/B
    validation.  ``n_sigma`` is the SIZING multiplier: the window must hold
    every galaxy within ``n_sigma * max_row(sigma_eff)`` of a sample
    (:func:`recommended_kde_window` / :func:`auto_kde_window` size it so).
    The likelihood factory sizes the window from the bound catalogs and
    threads it statically (``CatalogKernelState.kde_window``); this
    process-global size is the fallback for direct callers.  A fixed W that is
    smaller than the data-sized one truncates: the window is CENTRED on the
    sample's insertion index (one binary search), so it holds the W
    index-nearest galaxies and nothing repositions it to fit a block --
    exactness is entirely the sizing's job.  Measured:
    W=1024 held max |delta log p_cat| < 1e-6 against the full-row evaluator on
    a 2113-galaxy spectroscopic row across the sampled sigma_kde prior
    [0, 0.05] (tests/test_catalog_kde_window.py) but moved it by 0.17 nats on
    average on a DESI-like mixed spectro+photo row of the same length.

    Windowing additionally requires rows verified z-sorted (see
    ``_rows_sorted_for_windowing``); catalogs loaded with
    ``load_survey(..., sort_rows_by_z=False)`` or built ad hoc fall back to
    the full-row path automatically.
    """
    global _KDE_WINDOW_SIZE, _KDE_WINDOW_NSIGMA
    if size is not None:
        size = int(size)
        if size < 2:
            raise ValueError(f"KDE window size must be >= 2, got {size}")
    n_sigma = float(n_sigma)
    if not n_sigma > 0.0:
        raise ValueError(f"KDE window n_sigma must be > 0, got {n_sigma}")
    _KDE_WINDOW_SIZE = size
    _KDE_WINDOW_NSIGMA = n_sigma


def recommended_kde_window(zgals, ngals, dzgals, sigma_kde_max, n_sigma=6.0):
    """Data-driven window size for :func:`configure_catalog_kde_window`.

    The smallest static ``W`` such that a window of ``W`` consecutive galaxies
    CENTRED on a sample's insertion index in a z-sorted row (the rule
    :func:`_sorted_row_window_start` applies) holds every galaxy within
    ``n_sigma * max_row(sigma_eff)`` of the sample, for every row and every
    sample position, at the WIDEST kernel the sampled prior admits
    (``sigma_kde_max``; the prior upper bound in
    ``darksirens/inference/prior.py``, 0.05 by default): TWICE the largest
    number of galaxies any interval of length ``n_sigma * sigma_max`` (one
    side of the sample) contains.  NOT capped at the row's real count: on a
    ragged catalog a window as long as a SHORT row is not that row -- centred
    on a sample past the row's support it starts at ``n - W//2 > 0`` and
    drops the row's front half (measured 0.53 nats on a 300-galaxy photo-z
    row beside a 2000-galaxy one).  A result longer than every row simply
    sends the evaluator down its exact full-row path.  At the default
    ``n_sigma=6`` the kernel mass a covered sample can miss is < 2e-9 per
    galaxy, comfortably inside the 1e-6 |delta log p_cat| validation bar.
    Host-side numpy diagnostic -- run once per catalog when sizing W, not in
    the hot path.  Raises ``ValueError`` unless ``ngals`` carries exactly one
    count per ``zgals`` row: the scan visits rows in count order rather than in
    storage order, so a short ``ngals`` would size the window from a prefix of
    the catalog instead of failing.

    The one-sided rule is what lets the evaluator locate a window with ONE
    binary search (the insertion index) instead of three (the block's two
    edges and the centre): for an evenly populated row it is the same count
    as the block itself; only a sample sitting right beside a dense clump pays
    for the clump on both sides.
    """
    z = np.asarray(zgals)
    ng = np.asarray(ngals)
    dz = np.asarray(dzgals)
    if z.ndim != 2:
        raise ValueError("zgals must be (N_rows, N_max)")
    if ng.ndim != 1 or ng.shape[0] != z.shape[0]:
        # The scan no longer walks range(z.shape[0]), so a short ``ngals`` would
        # silently size the window from a PREFIX of the catalog instead of
        # raising: refuse it here, as the auto_kde_window widths check does.
        raise ValueError(
            f"ngals must be (N_rows,) matching zgals: got {ng.shape} "
            f"for zgals {z.shape}"
        )
    worst = 0
    # Rows in DESCENDING count order with an exact early exit.  A row with n
    # real galaxies has right[j] <= n - j and left[j] <= j + 1, so one_sided
    # <= n and the row contributes at most 2n to the max: once ``worst`` has
    # reached 2n for the largest remaining n, no unvisited row can raise it.
    # This is a pure visitation-order prune -- the returned max is unchanged.
    order = np.argsort(-np.asarray(ng, dtype=np.int64), kind="stable")
    for r in order:
        n = int(ng[r])
        if n < 1:
            break                      # descending: every remaining row is empty
        if 2 * n <= worst:
            break                      # every remaining row has n' <= n
        zr = np.sort(z[r, :n])
        sig_max = float(
            np.max(np.sqrt(dz[r, :n] ** 2 + float(sigma_kde_max) ** 2))
        )
        width = float(n_sigma) * max(sig_max, SIGMA_EFF_FLOOR)      # one side
        idx = np.arange(n)
        right = np.searchsorted(zr, zr + width, side="right") - idx   # [z_j, z_j + w]
        left = idx - np.searchsorted(zr, zr - width, side="left") + 1  # [z_j - w, z_j]
        one_sided = int(max(np.max(right), np.max(left)))
        worst = max(worst, 2 * one_sided)
    return worst


# Verified-sorted verdicts keyed by id(zgals); each entry pins nothing (a
# weakref) and is revalidated if the id was recycled by a different array.
_SORTED_ROWS_CACHE: dict = {}

# Build-time attestation for jit-ARGUMENT catalogs.  The production likelihood
# (darksiren_log_likelihood, a module-level jit) receives every EMCatalog as a
# traced argument, so inside its trace _rows_sorted_for_windowing cannot verify
# the concrete data and the evaluator would fall back to the full row — i.e.
# the windowed hot path would NEVER engage where it matters most.  The
# likelihood factory holds the concrete arrays at build time: it verifies every
# catalog view it binds and arms this flag, which lets the evaluator window
# TRACED catalogs too.  Process-global, last-attestation-wins; a factory build
# with ANY unsorted view disarms it.  The concrete-array check still runs first
# and never consults this flag, so eager/closure callers and the bitwise
# unsorted-fallback contract are unaffected.
#
# The arming is additionally keyed to the attested views' ROW SHAPES — the only
# property of a catalog the evaluator can read off a tracer — so one build's
# attestation cannot spill onto an unrelated view (a diagnostic call through
# PRIOR_REGISTRY with an ad-hoc or re-sliced catalog, which would be windowed
# without ever having been verified).  Two tracers of the SAME shape remain
# indistinguishable inside a trace, so the contract is still "attest every view
# you bind".
_ROWS_SORTED_ATTESTED: bool = False
_ATTESTED_ROW_SHAPES: frozenset = frozenset()


def attest_rows_sorted_for_windowing(*em_catalogs) -> bool:
    """Verify the z-sort invariant on CONCRETE catalogs and arm windowing for
    the traced (jit-argument) evaluator path.

    Call with every catalog view the likelihood will bind (PE, selection, and
    all mixture views).  Returns the armed verdict: True only if every catalog
    with per-galaxy rows verifiably satisfies the invariant.  Catalog views
    without rows (``zgals is None``) are ignored rather than disarming.  Only the
    row shapes attested here can be windowed through a jit boundary.
    """
    global _ROWS_SORTED_ATTESTED, _ATTESTED_ROW_SHAPES
    verdict = True
    shapes = set()
    for cat in em_catalogs:
        zgals = getattr(cat, "zgals", None)
        ngals = getattr(cat, "ngals", None)
        if zgals is None or ngals is None:
            continue
        if not _rows_sorted_for_windowing(zgals, ngals):
            verdict = False
            break
        shapes.add(tuple(zgals.shape))
    _ROWS_SORTED_ATTESTED = verdict
    _ATTESTED_ROW_SHAPES = frozenset(shapes) if verdict else frozenset()
    return verdict


def _traced_rows_attested(zgals) -> bool:
    """True iff ``zgals`` is a TRACER whose row shape was attested at build time.

    Traced catalogs (jit arguments — the production likelihood path) cannot be
    verified in the evaluator; the factory attests them while the arrays are
    concrete (:func:`attest_rows_sorted_for_windowing`).  Requiring the attested
    ``(N_rows, N_max)`` keeps that arming from covering a view nobody checked.

    This is a PERFORMANCE gate only, and deliberately so: the flags it reads are
    mutable process globals and therefore NOT part of any jit cache key, so a
    function compiled while they were armed can be replayed after they are
    disarmed.  Correctness on the windowed branch is enforced IN THE GRAPH by
    ``CatalogKernelState.rows_sorted`` (see :func:`_rows_sorted_traced`), which
    is computed from the arrays the compiled graph actually receives.
    """
    return (
        _ROWS_SORTED_ATTESTED
        and isinstance(zgals, jax.core.Tracer)
        and tuple(zgals.shape) in _ATTESTED_ROW_SHAPES
    )


def _rows_sorted_traced(zgals, ngals):
    """In-GRAPH counterpart of :func:`_rows_sorted_for_windowing`.

    Returns a scalar boolean *array* (a tracer when the catalog is a jit
    argument) asserting the same invariant on the same data: every row's real
    prefix ``[0, ngals[row])`` is non-decreasing in z.  ``None`` when the shapes
    cannot carry the invariant, which leaves the guard disabled.

    Cost: one chunked pass over ``zgals`` per state build, alongside the
    24-node Gauss-Legendre kernel normalisation the same build already runs over
    every galaxy -- a few percent of a build that is itself once per proposal.
    When the arrays are concrete and already verified the expression folds to a
    literal ``True`` and the guard disappears from the graph entirely.
    """
    if zgals is None or ngals is None:
        return None
    if getattr(zgals, "ndim", 0) != 2 or getattr(ngals, "ndim", 0) != 1:
        return None
    if ngals.shape[0] != zgals.shape[0] or zgals.shape[1] < 2:
        return None

    def _row(zs, ng):
        cols = jnp.arange(1, zs.shape[0])
        return jnp.all((jnp.diff(zs) >= 0) | (cols >= ng))

    return jnp.all(_map_rows(_row, (zgals, ngals)))


@jax.tree_util.register_pytree_node_class
class StaticInt:
    """A Python int carried through a pytree as AUXILIARY DATA, not as a leaf.

    ``CatalogKernelState`` is a NamedTuple, so every field is a pytree leaf:
    ``lax.optimization_barrier`` (the prior state's materialize barrier) and
    ``jax.jit`` turn a plain Python int field into a traced scalar, which can
    no longer size a slice or be compared with a shape.  Flattening to no
    children keeps the value concrete on every path, and it is part of the
    treedef -- hence of every jit cache key -- so two states built for
    different windows never share a compiled graph.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = int(value)

    def tree_flatten(self):
        return (), self.value

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)

    def __int__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, StaticInt):
            return self.value == other.value
        if isinstance(other, (int, np.integer)):
            return self.value == int(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"StaticInt({self.value})"


def _effective_kde_window(kde_window):
    """The static window a state was built for: its own ``kde_window`` when the
    caller pinned one (the likelihood factory sizes it from the data), else the
    process-global :func:`configure_catalog_kde_window` setting."""
    if kde_window is None:
        return _KDE_WINDOW_SIZE
    return int(kde_window)


def _static_window(kde_window):
    """Box a caller's window for the state (``None`` stays ``None``).

    Same validity rule as :func:`configure_catalog_kde_window`: a window is at
    least 2 galaxies (``_sorted_row_window_start`` centres it on the insertion
    index, so W = 1 could exclude the single nearest galaxy).
    """
    if kde_window is None:
        return None
    boxed = kde_window if isinstance(kde_window, StaticInt) else StaticInt(kde_window)
    if int(boxed) < 2:
        raise ValueError(f"KDE window size must be >= 2, got {int(boxed)}")
    return boxed


def auto_kde_window(catalogs, sigma_kde_max, n_sigma=None, granule=64):
    """Data-sized static window for a set of CONCRETE catalog views.

    The largest number of galaxies any row of any view holds inside an interval
    of length ``2 n_sigma max_row(sigma_eff)`` at the WIDEST kernel the run can
    reach (:func:`recommended_kde_window` at ``sigma_kde_max``), plus one, rounded
    up to a multiple of ``granule``.  Every sample's in-range block then fits the
    window, so the evaluator never truncates: it evaluates exactly the galaxies
    within ``n_sigma`` row-max widths of the sample (the contract
    :func:`configure_catalog_kde_window` documents), and no others beyond a
    ``granule`` of nearest neighbours whose contribution is below
    ``exp(-n_sigma^2 / 2)`` relative.  A view whose rows are no longer than the
    result takes the full-row path by the evaluator's own length test.

    Returns ``None`` when no view carries per-galaxy rows.  ``n_sigma`` defaults
    to the configured evaluator half-width multiplier.
    """
    if n_sigma is None:
        n_sigma = _KDE_WINDOW_NSIGMA
    worst = None
    seen_ids = set()
    pinned = []                            # keep scanned arrays alive so no
                                           # freed view can recycle an id()
    for cat in catalogs:
        zgals = getattr(cat, "zgals", None)
        ngals = getattr(cat, "ngals", None)
        dzgals = getattr(cat, "dzgals", None)
        if zgals is None or getattr(zgals, "ndim", 0) != 2:
            continue                       # no per-galaxy rows: nothing to size
        if dzgals is None:
            raise ValueError(
                "auto_kde_window: a catalog view carries per-galaxy rows "
                "(zgals) but no widths (dzgals); the window cannot be sized "
                "for it and it would be evaluated under another view's window."
            )
        # Aliased views (the flat-union path binds the SAME arrays to the PE
        # and selection views) scan identically; the result is a max over
        # views, so a repeat contributes nothing.  Identity only -- these are
        # hundreds of MB.  The dzgals refusal above already fired for THIS
        # view, so dedup cannot swallow a mis-specified one.
        view_key = (id(zgals), id(ngals), id(dzgals))
        if view_key in seen_ids:
            continue
        seen_ids.add(view_key)
        pinned.append((zgals, ngals, dzgals))
        if ngals is None:                  # no counts: every slot is a galaxy
            ngals = np.full(zgals.shape[0], zgals.shape[1], dtype=np.int64)
        need = recommended_kde_window(
            np.asarray(zgals), np.asarray(ngals), np.asarray(dzgals),
            float(sigma_kde_max), n_sigma=float(n_sigma),
        ) + 1
        worst = need if worst is None else max(worst, need)
    if worst is None:
        return None
    granule = max(int(granule), 1)
    return int(-(-worst // granule) * granule)


def _resolve_rows_sorted_guard(zgals, ngals, kde_window=None):
    """The ``CatalogKernelState.rows_sorted`` guard for this catalog, or None.

    Emitted ONLY for TRACED catalogs, and only while windowing is enabled at
    all: that is exactly the path where the trace-time verdict comes from the
    mutable attestation globals and can therefore be replayed on unattested
    data.  A concrete catalog is verified from its own arrays at trace time
    (:func:`_rows_sorted_for_windowing`), so it needs no runtime node, and with
    ``configure_catalog_kde_window(size=None)`` there is no windowed branch to
    guard.  The row length is deliberately NOT part of the gate: if the state
    outlives a window reconfiguration, the guard must already be in it.  When
    the evaluator ends up not windowing, the node is unused and XLA drops it.
    """
    if _effective_kde_window(kde_window) is None or zgals is None or ngals is None:
        return None
    if not isinstance(zgals, jax.core.Tracer):
        return None
    return _rows_sorted_traced(zgals, ngals)


def _rows_sorted_for_windowing(zgals, ngals) -> bool:
    """True iff ``zgals``/``ngals`` are CONCRETE arrays whose rows verifiably
    satisfy the z-sort invariant (real prefix non-decreasing).

    This is the trace-time gate for the windowed evaluator: it never trusts a
    flag — the invariant is checked (once per array; verdicts cached by id +
    weakref identity) on the actual data.  Traced arrays return False, so any
    call path that feeds the catalog through a jit boundary as an argument
    keeps the bit-identical full-row path.
    """
    if zgals is None or ngals is None:
        return False
    key = id(zgals)
    entry = _SORTED_ROWS_CACHE.get(key)
    if entry is not None:
        ref, verdict = entry
        if ref() is zgals:
            return verdict
        del _SORTED_ROWS_CACHE[key]
    try:
        z = np.asarray(zgals)
        ng = np.asarray(ngals)
    except Exception:
        return False  # traced (or otherwise non-concrete) arrays
    if z.ndim != 2 or ng.ndim != 1 or ng.shape[0] != z.shape[0]:
        return False
    cols = np.arange(1, z.shape[1])[None, :]
    verdict = bool(np.all((np.diff(z, axis=1) >= 0) | (cols >= ng[:, None])))
    try:
        ref = weakref.ref(zgals)
    except TypeError:
        return verdict  # unlikely: not weakref-able -> verify every trace
    _SORTED_ROWS_CACHE[key] = (ref, verdict)
    return verdict


def _sorted_row_window_start(zgals, pix, z, n_real, window):
    """Start index of the W-galaxy window in row ``pix`` of z-sorted ``zgals``.

    ONE binary search, over the real prefix ``[0, n_real)`` only (scalar
    gathers per iteration; the padded tail is never consulted, so trailing
    padding values need no sentinel), for the insertion index ``i_z`` of the
    sample; the window is the ``W`` index-nearest galaxies centred there,
    ``clip(i_z - W//2, 0, N_max - W)``.  Index order IS z order, so on each
    side the excluded galaxies are z-farther than every included one, and a
    sample beyond the row's support straddles the nearest galaxies so far-tail
    evaluations stay finite.

    Exactness is the WINDOW SIZE's job, not this rule's: with ``W`` sized by
    :func:`recommended_kde_window` (twice the largest one-sided count of
    galaxies within ``n_sigma`` row-max widths, which is what
    :func:`auto_kde_window` hands the likelihood factory) the centred window
    provably covers every galaxy within ``n_sigma`` widths of ANY sample, so
    nothing inside the truncation contract is ever excluded.  The earlier
    three-search rule (block edges plus centre, then a fit test) bought
    coverage for a marginally smaller ``W`` at three dependent search chains
    per sample; MEASURED on CPU the searches were ~19% of the windowed
    evaluation, and on a GPU dependent scalar gathers are latency-bound.
    """
    n_max = zgals.shape[1]
    # Fixed iteration count: each step halves [lo, hi); log2(n_max)+1 is
    # enough to reach lo == hi from any initial span <= n_max.
    n_iter = int(np.ceil(np.log2(max(int(n_max), 2)))) + 1
    lo0 = jnp.zeros((), dtype=jnp.int32)
    hi0 = jnp.asarray(n_real, dtype=jnp.int32)

    def _body(_, lh):
        lo, hi = lh
        active = lo < hi
        mid = (lo + hi) // 2
        go_right = active & (zgals[pix, mid] < z)
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(active & ~go_right, mid, hi)
        return lo, hi

    i_z, _ = lax.fori_loop(0, n_iter, _body, (lo0, hi0))
    return jnp.clip(i_z - window // 2, 0, n_max - window)


def _map_rows(row_fn, args: tuple):
    """vmap ``row_fn`` over the leading (row) axis of every array in ``args``,
    chunking with ``lax.map`` when the catalog view is large (see above).

    Zero-padded rows evaluate through the same masked per-row path (empty
    ``real`` mask) and are sliced off the result, so outputs are identical to
    the unchunked vmap row-for-row.
    """
    n_rows = args[0].shape[0]
    n_max = args[0].shape[1] if args[0].ndim > 1 else 1
    chunk = _resolve_row_chunk(n_rows, n_max)
    if chunk is None or chunk >= n_rows:
        return vmap(row_fn)(*args)

    n_pad = (-n_rows) % chunk
    def _prep(a):
        if n_pad:
            pad = jnp.zeros((n_pad,) + a.shape[1:], dtype=a.dtype)
            a = jnp.concatenate([a, pad], axis=0)
        return a.reshape((n_rows + n_pad) // chunk, chunk, *a.shape[1:])
    chunked = tuple(_prep(a) for a in args)
    out = lax.map(lambda ch: vmap(row_fn)(*ch), chunked)
    def _post(o):
        return o.reshape(-1, *o.shape[2:])[:n_rows]
    if isinstance(out, tuple):
        return tuple(_post(o) for o in out)
    return _post(out)


def _log_ndtr_span(lo, hi):
    """``log(Phi(hi) - Phi(lo))`` without the f64 underflow of the difference.

    ``ndtr`` returns exactly 0 past ~39 sigma, so the plain difference
    underflows for any truncation that deep into a tail; ``log_ndtr`` stays
    finite for arbitrarily deep ones.  The ``-1e-16`` clamp keeps the result
    finite (and its gradient defined) when the two limits coincide.
    """
    log_lo, log_hi = log_ndtr(lo), log_ndtr(hi)
    return log_hi + jnp.log(-jnp.expm1(jnp.minimum(log_lo - log_hi, -1e-16)))


def _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=_ZMAX):
    """
    log Z_i for one row: Z_i = ∫_0^{z_hi} N(z; z_i, sig_i) g(z) dz.

    Substituting u = Phi((z - z_i)/sig_i) maps the truncated integral to
    ∫_a^b g(z_i + sig_i * Phi^{-1}(u)) du with a = Phi(-z_i/sig),
    b = Phi((z_hi - z_i)/sig): truncation handled exactly, integrand
    smooth, Gauss–Legendre converges fast for any sigma.

    ``z_hi`` is the upper truncation limit (default the grid ``zMax``).
    Passing ``survey.z_depth`` yields the below-depth kernel mass Z_i^depth used
    to renormalise the catalog prior when a survey depth is set (see
    :func:`_renorm_log_kw_below_depth`).
    """
    a = ndtr(-zs / sig_eff)
    b = ndtr((z_hi - zs) / sig_eff)
    span = b - a                                            # (N_max,)
    u = a[..., None] + span[..., None] * _GL_X              # (N_max, K)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    z_node = jnp.clip(zs[..., None] + sig_eff[..., None] * ndtri(u), 0.0, z_hi)
    g = jnp.exp(
        log_interp_zgrid(z_node.reshape(-1), log_g_grid)
    ).reshape(z_node.shape)
    Zg = (g * _GL_W).sum(axis=-1)                           # (N_max,)
    Z = span * Zg
    # ``span`` underflows to exactly 0 in f64 once BOTH ndtr tails do (the
    # truncation limit >~39 sigma_eff from the galaxy), which is routine on the
    # depth path (``z_hi = z_depth``) for a REAL galaxy above the depth -- with
    # spectroscopic sig_eff at the floor it is the rule, not the exception.
    # Those rows are recovered in log space, where the truncated mass is exact:
    # returning the padding fallback (0.0) instead would declare UNIT kernel
    # mass below the truncation for a galaxy that has essentially none, and the
    # depth-mass consumer would pick up that galaxy's whole weight.  Rows with a
    # resolvable ``span`` keep the direct (bit-identical) spelling.
    ok = Z > 0.0
    log_Z = jnp.where(
        ok,
        jnp.log(jnp.where(ok, Z, 1.0)),
        _log_ndtr_span(-zs / sig_eff, (z_hi - zs) / sig_eff)
        + jnp.log(jnp.maximum(Zg, 1e-300)),
    )
    return jnp.where(real, log_Z, 0.0)


def _row_log_kernel_norms_zspace(zs, sig_eff, real, log_g_grid,
                                 z_hi=_ZMAX, n_sigma=5.0):
    """
    log Z_i for one row via z-space Gauss–Legendre: no ``ndtri``.

    GL nodes are placed directly in redshift on
    [max(0, z_i - n_sigma*sig_i), min(z_hi, z_i + n_sigma*sig_i)]
    and the integrand N(z; z_i, sig_i) g(z) is evaluated with a plain
    ``exp(-0.5 d^2)`` — avoiding the expensive ``ndtri`` of the CDF-domain
    path.  The truncation at +/-n_sigma loses < ``2 Phi(-n_sigma)`` of the
    Gaussian tail mass (6e-5 at n_sigma=4, 3e-7 at n_sigma=5).

    Accuracy: the CDF-domain path is mathematically optimal for wide
    kernels (the change of variables removes the Gaussian weight, leaving
    only smooth g(z) for GL).  The z-space path integrates the peaked
    Gaussian × g product, which needs adequate node density per sigma to
    resolve.  At 8–12 nodes with n_sigma=4–5 and sigma_eff ~ 0.01 (the
    spectroscopic regime), the z-space path is both faster (no ndtri) and
    more accurate per node than the CDF-domain path at the same count.
    """
    z_lo = jnp.maximum(0.0, zs - n_sigma * sig_eff)
    z_hi_eff = jnp.minimum(z_hi, zs + n_sigma * sig_eff)
    span_z = z_hi_eff - z_lo                                   # (N_max,)
    z_node = z_lo[..., None] + span_z[..., None] * _GL_X       # (N_max, K)
    d = (z_node - zs[..., None]) / sig_eff[..., None]
    log_gauss = (-0.5 * d * d
                 - jnp.log(sig_eff[..., None])
                 - 0.5 * jnp.log(2.0 * jnp.pi))
    log_g = log_interp_zgrid(
        z_node.reshape(-1), log_g_grid
    ).reshape(z_node.shape)
    integrand = jnp.exp(log_gauss + log_g)
    Z = span_z * (integrand * _GL_W).sum(axis=-1)              # (N_max,)
    # A galaxy more than n_sigma*sig ABOVE the truncation has span_z <= 0 (no
    # window at all), and a window that barely clips the [0, z_hi] boundary can
    # underflow Z to exactly 0.  Both previously returned -700 / -inf --
    # declaring ZERO truncated mass where the CDF twin returns the exact tiny
    # value.  Recover both in log space exactly as _row_log_kernel_norms does:
    # the true Gaussian mass on [0, z_hi] via _log_ndtr_span, times g at the
    # point where that mass concentrates (the nearer boundary).  Rows with a
    # resolvable Z keep the direct (bit-identical) spelling.
    z_star = jnp.clip(zs, 0.0, z_hi)
    log_g_star = log_interp_zgrid(z_star, log_g_grid)
    fallback = _log_ndtr_span(-zs / sig_eff, (z_hi - zs) / sig_eff) + log_g_star
    ok = Z > 0.0
    log_Z = jnp.where(ok, jnp.log(jnp.where(ok, Z, 1.0)), fallback)
    return jnp.where(real, log_Z, 0.0)


def _dispatch_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=_ZMAX):
    """Route to CDF-domain or z-space GL based on the process-global config."""
    if _GL_DOMAIN == 'zspace':
        return _row_log_kernel_norms_zspace(
            zs, sig_eff, real, log_g_grid, z_hi, _GL_NSIGMA)
    return _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi)


def _renorm_log_kw_below_depth(
    log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
):
    """Renormalise per-galaxy kernel weights so the depth-truncated catalog
    prior ``p_cat(z | pix)`` stays a proper density.

    A magnitude-limited survey catalogs nothing past ``z_depth``, so the
    per-sample evaluator (:func:`eval_log_catalog_prior_state`) zeroes
    ``p_cat`` there.  Without a correction the mixture would then integrate to
    its below-depth mass ``m < 1``, breaking the additive-density prior's
    per-pixel unit normalisation (and, under the field convention, the match
    between the numerator's catalog mass and the ``N_obs`` in the global
    normalizer).  The mixture is therefore divided by

        m = ∫_0^{z_depth} p_cat(z|pix) dz = Σ_i exp(log_kw_i) · Z_i^depth,

    with ``Z_i^depth`` the same GL kernel norm truncated at ``z_depth``.  Empty
    rows (``log_kw ≡ -inf``) carry an inert ``m`` and are returned unchanged.

    ``m`` is ALSO returned, and the caller MUST scale the row's observed count
    by it.  Renormalising the shape without rescaling the amplitude would leave
    the observed branch integrating to the full ``N_obs`` below the depth, i.e.
    it would relocate the host probability of every above-depth catalogued
    galaxy onto the below-depth redshifts -- while ``_assemble_curves`` is
    simultaneously counting those same galaxies in the missing branch, which
    relaxes to the FULL ``dN_exp`` above the depth.  Measured on a 10-galaxy row
    with 4 galaxies below ``z_depth=0.3``, that placed 10.00 hosts below the
    depth instead of 3.81 -- a 2.6x over-weighting of exactly the redshifts that
    drive the dark-siren H0 constraint.  With ``N_obs -> N_obs * m`` the
    observed branch integrates to the number of galaxies actually catalogued
    below the depth, and the above-depth ones are represented once, by
    ``dN_miss = dN_exp`` (the stated intent: hosts beyond the depth are
    *missing*, not nonexistent).
    """
    log_Z_depth = _dispatch_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=z_depth)
    log_m = jnp.where(
        has_galaxies, _logsumexp_neginf_safe(log_kw + log_Z_depth), 0.0
    )
    return jnp.where(real, log_kw - log_m, -jnp.inf), log_m


def _row_kernel_state(
    zs, dzs, ws, ngal, sigma_kde, log_g_grid, volume_weighted=False, z_depth=None
):
    """
    Per-galaxy kernel quantities for one row, under one of two host-weight
    conventions for the galaxy measure g(z) = dV_c/dz * (1+z)^delta:

    - ``volume_weighted=False`` (incomplete-catalog default): each galaxy's
      total host probability is its base weight w~_i; g only tilts the kernel
      shape, divided back out by Z_i = ∫ N(z;z_i,sig) g(z) dz so each kernel has
      unit mass.  The g(z) front factor is reapplied per sample in the evaluator.

    - ``volume_weighted=True`` (complete-catalog): each galaxy's host
      probability scales with the comoving volume at its redshift, weight
      w_i * g(z_i), with a plain N(z;z_i,sig) kernel (no Z_i, no front g(z)).
      The catalog is the full universe, so the host rate must track the number
      of candidate hosts per redshift shell.

    Returns ``(log_kw, sig_eff, log_depth_mass)``; the evaluator's g(z) handling
    is selected by the same ``volume_weighted`` flag carried on
    :class:`CatalogKernelState`.  ``log_depth_mass`` is 0.0 (mass 1) whenever no
    depth truncation applies.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)
    # The 1e-300 floor is a numerical BACKSTOP only (it keeps padding slots from
    # producing log(0) = -inf * 0 = NaN inside the traced reduction).  A real
    # galaxy with w <= 0 is a data error, not something to floor: it would carry
    # a ~-690 log-weight while still counting in ``ngal``, so the observed count
    # and the mixture would measure different galaxy sets.  Such weights are
    # rejected at the data-entry boundary by ``darksirens_pixelate``.
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)

    if volume_weighted:
        log_w = log_w + jnp.where(real, log_interp_zgrid(zs, log_g_grid), 0.0)

    lse = logsumexp(log_w)
    has_galaxies = jnp.isfinite(lse)
    log_w_norm = jnp.where(real, log_w - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    if volume_weighted:
        log_kw = log_w_norm
    else:
        log_Z = _dispatch_log_kernel_norms(zs, sig_eff, real, log_g_grid)
        log_kw = jnp.where(real, log_w_norm - log_Z, -jnp.inf)
        # ``z_depth`` (concrete Python float or None; never traced) renormalises
        # the mixture to unit mass on [0, z_depth] so the depth-truncated prior
        # stays proper.  ``None`` skips it entirely (bit-identical legacy path).
        # ``log_depth_mass`` is the mass the mixture had below the depth BEFORE
        # renormalising; the caller must scale the row's observed count by it,
        # or the galaxies above the depth are counted twice (see
        # :func:`_renorm_log_kw_below_depth`).
        if z_depth is not None:
            log_kw, log_depth_mass = _renorm_log_kw_below_depth(
                log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
            )
    return log_kw, sig_eff, log_depth_mass


class CatalogKernelState(NamedTuple):
    """Per-galaxy kernel quantities for all catalog rows (one proposal)."""
    log_g_grid: jnp.ndarray  # (N_grid,)
    log_kw: jnp.ndarray      # (N_rows, N_max)
    sig_eff: jnp.ndarray     # (N_rows, N_max)
    log_sig_eff: jnp.ndarray = None  # (N_rows, N_max) — precomputed log(sig_eff)
    volume_weighted: bool = False
    #: (N_rows, N_max) ``log_kw - log(sig_eff) - log sqrt(2 pi)``: everything of
    #: a galaxy's log kernel term that does not depend on the sample redshift,
    #: fused at state-build time so the per-sample evaluator gathers THREE
    #: arrays (z_i, sigma_i, this) instead of four and adds one term instead of
    #: three.  Padding slots carry the same ``-1e30`` sentinel as ``log_kw``.
    #: Every state builder populates it; ``None`` (a hand-built state) selects
    #: the historical four-gather arithmetic, which agrees with the fused form
    #: to the last ulp (~1e-13 relative on the golden likelihoods, not
    #: bit-identical: the constant is summed in a different order).
    log_kw_eff: Any = None
    #: Static window length this state was built for (a :class:`StaticInt`,
    #: carried as pytree aux data so it stays concrete under jit and the
    #: materialize barrier; or None); the evaluator uses it in place of the
    #: process-global ``configure_catalog_kde_window`` size when set.  The
    #: likelihood factory sizes it from the bound catalogs
    #: (:func:`auto_kde_window`) so one process may hold likelihoods over
    #: differently dense catalogs without one build's window truncating
    #: another's rows.
    kde_window: Any = None
    #: (N_rows,) log of the mixture mass that lay below ``z_depth`` BEFORE the
    #: shape was renormalised.  Callers scale the row's observed galaxy count by
    #: it so the depth-truncated catalog branch integrates to the number of
    #: galaxies actually catalogued below the depth.  All-zero when
    #: ``z_depth is None``.
    log_depth_mass: Any = 0.0
    # ``survey.z_depth`` (concrete Python float or None; never traced): when set,
    # ``log_kw`` is already renormalised to unit mass on [0, z_depth] and the
    # per-sample evaluator zeroes p_cat beyond it.  ``None`` means no depth
    # truncation (the legacy, bit-identical path).
    z_depth: Any = None
    #: (N_rows,) per-row max sigma_eff, the traced input to the windowed
    #: evaluator's half-width ``n_sigma * sig_eff_row_max[pix]``.  ``None``
    #: (e.g. a state built by hand) disables windowing for that state.
    sig_eff_row_max: Any = None
    #: Scalar boolean verdict on the z-sort invariant, computed FROM THE ARRAYS
    #: THEMSELVES (traced when the catalog is a jit argument).  This is what
    #: makes the windowed branch self-verifying: the trace-time decision to emit
    #: it comes from process globals (``_KDE_WINDOW_SIZE``,
    #: ``attest_rows_sorted_for_windowing``) that are NOT part of any jit cache
    #: key, so a compiled windowed graph can be replayed on data nobody
    #: attested.  Carrying the verdict in the GRAPH means such a replay yields
    #: NaN instead of a plausible wrong number.  ``None`` (a hand-built state)
    #: leaves the branch unguarded, exactly as before.
    rows_sorted: Any = None
    #: (N_rows,) bool: True where the row had NO finite kernel weight before
    #: the -1e30 sanitize (an empty pixel / all-padding row).  The evaluators
    #: use it to restore the exact -inf output the sanitize would otherwise
    #: turn into a finite ~-1e30 value.  ``None`` (a hand-built state) keeps
    #: plain logsumexp semantics.
    row_empty: Any = None
    #: (N_rows,) per-row maximum of the SANITIZED ``log_kw_eff``, 0.0 on an
    #: all-padding row.  It is the OFFSET the per-sample reduction subtracts,
    #: which is what removes the ``logsumexp`` amax pass: the Gaussian exponent
    #: ``-0.5 u^2`` is <= 0 and ``log_kw_eff_i - rowmax <= 0`` by construction,
    #: so no term can overflow and the data-dependent maximum is not needed.
    #: Populated by every builder alongside ``log_kw_eff``; ``None`` (a
    #: hand-built state) selects the historical two-pass ``logsumexp``.
    log_kw_eff_rowmax: Any = None
    #: (N_rows, N_max) ``1 / sig_eff``, formed once at state-build time so the
    #: per-sample evaluator multiplies instead of dividing.  ``sig_eff`` is
    #: theta-invariant once ``sigma_kde`` is fixed, and for an unpinned state
    #: the reciprocal is formed inside the traced build so the derivative with
    #: respect to ``sigma_kde`` still flows.  Padding slots carry 0.0, so their
    #: deviate is exactly zero and their exponent is the bare ``-1e30``
    #: sentinel.  ``None`` (a hand-built state) selects the historical
    #: division.
    inv_sig_eff: Any = None
    #: Scalar boolean verdict of the H0-pin probe (see
    #: :class:`PinnedKernelQuadrature`), or ``None`` for an unpinned state.
    #: Consumed by :func:`kernel_pin_poison` at the prior's NORMALIZER seam,
    #: NOT here: ``log_kw``/``log_p_cat`` are downstream of
    #: ``nan_to_num(nan=-inf)`` and of several ``where(x > 0, ..., -inf)``
    #: amplitude filters, any of which turns a poisoned value into "this pixel
    #: has no catalog hosts" -- a plausible number, which is exactly what the
    #: probe exists to prevent.
    pin_ok: Any = None


# ------------------------------------------------------------
# H0-pinned kernel quadrature
# ------------------------------------------------------------
#: Reference H0 [km/s/Mpc] the pinned quadrature is evaluated at.  The
#: comoving-distance table's own scale, so the reference build is the one
#: ``r_of_z`` is tabulated at (``r(z; H0) = r_tab(z) * H0Planck / H0``).
KERNEL_PIN_H0_REF: float = float(H0Planck)

#: Rows rebuilt from the LIVE proposal on every pinned state build.  8 rows of
#: 49,152 is 0.016% of the build the pin removes.
KERNEL_PIN_PROBE_ROWS: int = 8

#: Absolute tolerance on ``|log_kw_live - (log_kw_ref + shift)|`` over the probe
#: rows.  MEASURED residual on real DESI nside-64 rows over the full sampled
#: prior H0 in [20, 140] is 1.07e-14 (f64 rounding of an O(10) quantity); the
#: smallest premise violation measured is 3.9e-2 (Om0 0.3089 -> 0.35).  1e-9
#: sits five orders above the noise and seven below the signal.
KERNEL_PIN_TOL: float = 1e-9

#: Values at or below this are the ``-1e30`` padding sentinel, not a kernel
#: weight, and are excluded from the probe comparison.
_KERNEL_PIN_SENTINEL_CUT: float = -1e29


class PinnedKernelQuadrature(NamedTuple):
    """``catalog_kernel_state`` evaluated ONCE at ``H0_ref``, plus its probe.

    Valid iff the galaxy measure is EXACTLY H0-separable and the kernel widths
    are run constants:

        r(z; H0)  = r_tab(z; Om0, w0, wa) * (H0Planck / H0)  [cosmology.r_of_z]
        dV_c/dz   = c r^2 / (H0 E)                           [cosmology.dV_of_z]
        g(z)      = dV_c/dz * (1 + z)^delta
                  = (H0_ref / H0)^3 * g(z; H0_ref)    iff Om0, w0, wa, delta FIXED
        sig_eff_i = max(sqrt(dz_i^2 + sigma_kde^2), 1e-4)    iff sigma_kde FIXED

    The Gauss-Legendre nodes ``z_ik = clip(z_i + sig_i Phi^-1(a_i + (b_i - a_i)
    x_k), 0, z_hi)`` then carry NO theta at all, and the only theta-dependent
    factor in ``Z_i = int N(z; z_i, sig_i) g(z) dz`` is the scalar
    ``(H0_ref / H0)^3``.  Hence, EXACTLY:

        log Z_i(H0)        = log Z_i(H0_ref) - 3 ln(H0 / H0_ref)
        log_kw(H0)         = log_kw(H0_ref)  + 3 ln(H0 / H0_ref)
        log_depth_mass(H0) = log_depth_mass(H0_ref)      (the shifts cancel in
                                                          Sum_i kw_i Z_i^depth)

    and ``log p_cat`` is H0-INDEPENDENT: the ``+3 ln h`` on ``log_kw`` cancels
    the ``-3 ln h`` the evaluator's ``log_g_front`` carries.  MEASURED on real
    DESI nside-64 rows over the full sampled prior H0 in [20, 140] at
    ``H0_ref = 67.74``: ``max |log_kw(H0) - 3 ln(H0/H0_ref) - log_kw(H0_ref)| =
    1.07e-14``, ``max |dlog_depth_mass| = 1.78e-15``, ``max |dlog p_cat| =
    7.11e-15``.  NEGATIVE controls: Om0 0.3089 -> 0.35 makes the shift
    row-dependent with spread 3.9e-2; sigma_kde 0.003 -> 0.03 moves ``log_kw``
    by up to 1.85.

    SELF-VERIFYING, for exactly the reason ``rows_sorted`` is.  The decision to
    emit this branch is a build-time Python test on the run's SAMPLED-LABEL set
    (:func:`darksirens.likelihood.factory.kernel_pin_admissible`), and while the
    pin's PRESENCE is pytree structure -- part of every jit cache key -- the
    premise it was built under is not, so a graph compiled for one build could
    be replayed on a pin built under a violated premise.  ``probe_rows`` are
    therefore rebuilt from THIS proposal's ``log_g_grid``, ``sigma_kde`` and
    ``z_depth`` by the same ``_row_kernel_state`` the pin came from, and must
    reproduce ``log_kw[probe_rows]`` to ``KERNEL_PIN_TOL`` -- the array actually
    served, not a stored copy of it, so a corrupted pin fails the probe as
    loudly as a violated premise.  The verdict rides on
    ``CatalogKernelState.pin_ok``; see :func:`kernel_pin_poison` for where it
    is spent.

    ``log_kw`` is the SANITIZED array (padding at ``-1e30``): ``-1e30 + shift``
    is still exactly ``-1e30`` in f64, so padding stays inert and the probe
    excludes those slots (the raw ``_row_kernel_state`` output has ``-inf``
    there).
    """
    H0_ref: Any                     # scalar, the reference the pin was built at
    log_kw: jnp.ndarray             # (N_rows, N_max) at H0_ref, depth-renormalised
    sig_eff: jnp.ndarray            # (N_rows, N_max)  theta-invariant
    log_sig_eff: jnp.ndarray        # (N_rows, N_max)  theta-invariant
    sig_eff_row_max: jnp.ndarray    # (N_rows,)        theta-invariant
    log_depth_mass: Any             # (N_rows,)        H0-invariant
    row_empty: jnp.ndarray          # (N_rows,)        theta-invariant
    probe_rows: jnp.ndarray         # (P,) int32 rows rebuilt every proposal
    log_kw_eff: Any = None          # (N_rows, N_max) at H0_ref; shifts like log_kw
    #: (N_rows,) row max of ``log_kw_eff`` at ``H0_ref``.  The shift is UNIFORM
    #: over a row, so the row max shifts by exactly the same scalar as
    #: ``log_kw_eff`` itself and no reduction is repeated per proposal.
    log_kw_eff_rowmax: Any = None
    inv_sig_eff: Any = None         # (N_rows, N_max)  theta-invariant


def _spread_probe_rows(zgals, ngals, n_probe: int) -> jnp.ndarray:
    """``n_probe`` OCCUPIED catalog rows spread evenly over the view.

    Host-side (the pin is built from concrete arrays).  Occupied rows are what
    the probe can actually compare: an all-padding row's live ``log_kw`` is
    ``-inf`` everywhere, which the sentinel mask drops, so probing empty rows
    would leave the guard vacuously true.
    """
    n_rows = int(array_shape(zgals)[0])
    occ = (np.arange(n_rows) if ngals is None
           else np.flatnonzero(np.asarray(ngals) > 0))
    if occ.size == 0:
        occ = np.arange(n_rows)
    n_probe = max(1, min(int(n_probe), int(occ.size)))
    sel = np.unique(np.linspace(0, occ.size - 1, n_probe).round().astype(np.int64))
    return jnp.asarray(occ[sel].astype(np.int32))


def _real_row_max(sig_eff, log_kw_raw):
    """Per-row max ``sigma_eff`` over the REAL galaxies (finite raw ``log_kw``).

    ``n_sigma`` times this is the half-width the static window is sized to
    cover (:func:`recommended_kde_window` computes the same row maximum on the
    host).  The evaluator no longer reads the value -- its window is centred by
    index and its exactness rests on the sizing -- but the state keeps it as the
    per-row diagnostic of that contract and as the ``use_window`` marker.  Taken
    over the whole padded row it would be the PADDING width (``dzgals`` pads at
    1.0), which is how the traced half-width it once fed made every ragged row
    "not fit" its window.  Empty rows report 0.
    """
    real = jnp.isfinite(log_kw_raw)
    return jnp.max(jnp.where(real, sig_eff, 0.0), axis=1)


def _fused_log_kw_eff(log_kw_safe, sig_eff):
    """``log_kw - log(sig_eff) - log sqrt(2 pi)`` with the ``-1e30`` padding
    sentinel preserved EXACTLY (``-1e30 - O(10)`` already rounds to ``-1e30``
    in f64, but the select makes the contract explicit rather than incidental)."""
    live = log_kw_safe > _KERNEL_PIN_SENTINEL_CUT
    return jnp.where(
        live, log_kw_safe - jnp.log(sig_eff) - _HALF_LOG_2PI, -1e30
    )


def _inv_sig_eff(log_kw_eff, sig_eff):
    """``1 / sig_eff`` on real galaxies, 0.0 on the ``-1e30`` padding slots.

    Zeroing padding makes the per-sample deviate ``u = (z - z_pad) * 0`` exactly
    zero there, so the padded exponent is the bare sentinel and no padding
    convention (``dzgals`` pads at 1.0 today, but a hand-built view may pad at 0,
    where ``sig_eff`` floors at ``SIGMA_EFF_FLOOR`` and ``u`` would reach 1e6)
    can put a large number in front of it.  The term is ``exp(-1e30) = 0``
    either way; this only removes the overflow question.
    """
    return jnp.where(log_kw_eff > _KERNEL_PIN_SENTINEL_CUT, 1.0 / sig_eff, 0.0)


def _log_kw_eff_rowmax(log_kw_eff):
    """Per-row maximum of the SANITIZED ``log_kw_eff``, clamped to 0.0 on an
    all-padding row.

    This is the build-time offset the one-pass per-sample reduction subtracts
    (see :attr:`CatalogKernelState.log_kw_eff_rowmax`).  Rows with at least one
    real galaxy report that galaxy's ``log_kw_eff``; a row whose every slot is
    the ``-1e30`` sentinel would otherwise hand the evaluator an offset of
    ``-1e30``, which makes ``log_kw_eff - offset`` exactly ``0`` for padding and
    returns a finite ``-1e30 + log N_max`` instead of ``-inf``.  Clamping to 0.0
    keeps the sum at ``exp(-1e30) = 0`` and hence the answer at ``-inf`` -- which
    ``CatalogKernelState.row_empty`` overrides with the same value anyway, so the
    clamp is belt-and-braces rather than the contract.
    """
    rm = jnp.max(log_kw_eff, axis=1)
    return jnp.where(rm > _KERNEL_PIN_SENTINEL_CUT, rm, 0.0)


def _pinned_kernel_state(cosmo, survey, em_catalog, pinned, log_g_grid, z_depth,
                         kde_window=None):
    """This proposal's kernel state from the pin: one scalar shift + the probe.

    The whole per-galaxy Gauss-Legendre quadrature was done once at
    ``pinned.H0_ref``; this proposal only adds ``3 ln(H0/H0_ref)`` to
    ``log_kw``.  ``sig_eff`` / ``log_sig_eff`` / ``sig_eff_row_max`` /
    ``row_empty`` carry no theta and ``log_depth_mass`` is H0-invariant, so all
    five are returned unchanged.  ``log_g_grid`` is the LIVE proposal's grid --
    the evaluator's front factor reads it, and the H0 dependence only cancels in
    the product with ``log_kw`` -- and ``rows_sorted`` is likewise rebuilt from
    the arrays THIS graph receives (the pin is built from concrete arrays, where
    that guard is a Python verdict and no node is emitted).
    """
    shift = 3.0 * (jnp.log(cosmo.H0) - jnp.log(pinned.H0_ref))
    pr = pinned.probe_rows
    ngals = em_catalog.ngals
    if ngals is not None:
        probe_kw, _, _ = _map_rows(
            lambda zs, dzs, ws, ng: _row_kernel_state(
                zs, dzs, ws, ng, survey.sigma_kde, log_g_grid, False, z_depth),
            (em_catalog.zgals[pr], em_catalog.dzgals[pr],
             em_catalog.wgals[pr], ngals[pr]),
        )
    else:
        probe_kw, _, _ = _map_rows(
            lambda zs, dzs, ws: _row_kernel_state(
                zs, dzs, ws, None, survey.sigma_kde, log_g_grid, False, z_depth),
            (em_catalog.zgals[pr], em_catalog.dzgals[pr], em_catalog.wgals[pr]),
        )
    ref_kw = pinned.log_kw[pr]
    want = ref_kw + shift
    # Compare only slots that carry a kernel weight in BOTH: the pin is
    # sanitized (padding -1e30) and the live rebuild is raw (padding -inf).
    compared = jnp.isfinite(probe_kw) & (ref_kw > _KERNEL_PIN_SENTINEL_CUT)
    ok = jnp.all(
        jnp.where(compared, jnp.abs(probe_kw - want), 0.0) <= KERNEL_PIN_TOL
    )
    log_kw_eff = getattr(pinned, "log_kw_eff", None)
    if log_kw_eff is None:
        log_kw_eff = _fused_log_kw_eff(pinned.log_kw, pinned.sig_eff)
    rowmax = getattr(pinned, "log_kw_eff_rowmax", None)
    if rowmax is None:
        rowmax = _log_kw_eff_rowmax(log_kw_eff)
    inv_sig_eff = getattr(pinned, "inv_sig_eff", None)
    if inv_sig_eff is None:
        inv_sig_eff = _inv_sig_eff(log_kw_eff, pinned.sig_eff)
    # ``-1e30 + shift`` is exactly ``-1e30`` in f64 (|shift| < 10), so the
    # padding sentinel survives the shift on both fused and unfused arrays.
    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=pinned.log_kw + shift,
        sig_eff=pinned.sig_eff, log_sig_eff=pinned.log_sig_eff,
        volume_weighted=False, z_depth=z_depth,
        log_depth_mass=pinned.log_depth_mass,
        sig_eff_row_max=pinned.sig_eff_row_max,
        rows_sorted=_resolve_rows_sorted_guard(
            em_catalog.zgals, em_catalog.ngals, kde_window),
        row_empty=pinned.row_empty,
        pin_ok=ok,
        log_kw_eff=log_kw_eff + shift,
        # The shift is one scalar added to the WHOLE array, so the row maximum
        # shifts by exactly the same scalar: adding it here (a (N_rows,) add)
        # keeps ``log_kw_eff - rowmax <= 0`` structurally at every H0 rather
        # than merely <= 3 ln(140 / H0Planck) = 2.18.
        log_kw_eff_rowmax=rowmax + shift,
        inv_sig_eff=inv_sig_eff,
        kde_window=_static_window(kde_window),
    )


def kernel_pin_poison(state):
    """``NaN`` when ``state``'s H0 pin failed its probe, ``None`` when unpinned.

    Add it to the prior's NORMALIZER (``log_Z`` / ``log_Z_global``), which is the
    first expression downstream of the kernels with no NaN filter in front of it:
    ``_eval_dark_scalar`` runs ``nan_to_num(log_p_cat, nan=-inf)`` and every
    amplitude between the state and the likelihood passes a ``where(x > 0, ...,
    -inf)``, so a NaN placed on ``log_kw``, ``log_depth_mass`` or ``N_obs`` is
    silently rewritten as "this pixel has no catalog hosts" and the run returns a
    plausible wrong number.  Through the normalizer the poison reaches every
    sample's log prior as NaN, and the run's log-likelihood as ``-inf`` (the
    likelihood core's final ``where(isfinite(ll), ll, -inf)``).  ``None``
    (unpinned) means the caller emits no op at all, so the legacy path stays
    bit-identical.
    """
    ok = getattr(state, "pin_ok", None)
    if ok is None:
        return None
    return jnp.where(ok, 0.0, jnp.nan)


def catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_g_grid: jnp.ndarray | None = None,
    volume_weighted: bool = False,
    z_depth=None,
    pinned: "PinnedKernelQuadrature | None" = None,
    kde_window=None,
) -> CatalogKernelState:
    """Precompute per-galaxy kernel quantities once per parameter proposal.

    ``z_depth`` (a concrete Python float or ``None``) is the survey depth beyond
    which the catalog asserts nothing: when set, each row's ``log_kw`` is
    renormalised to unit mass on ``[0, z_depth]`` and stored on the returned
    state so the evaluator zeroes ``p_cat`` there.  It is threaded ONLY from the
    incomplete ``dark_sirens`` prior; the complete-catalog and bright-siren paths
    pass ``None`` (they model the full universe, so there is no depth to bound).

    ``pinned`` short-circuits the whole quadrature to the closed-form H0 shift
    (:class:`PinnedKernelQuadrature`).  ``volume_weighted`` kernels never form
    ``Z_i`` at all, so the pin is inert for them.

    ``kde_window`` (a concrete Python int or ``None``, never traced) pins the
    static per-sample window this state is evaluated with; ``None`` defers to
    the process-global :func:`configure_catalog_kde_window` size.
    """
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    if pinned is not None and not volume_weighted:
        return _pinned_kernel_state(
            cosmo, survey, em_catalog, pinned, log_g_grid, z_depth, kde_window)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, ng: _row_kernel_state(
                zs, dzs, ws, ng, survey.sigma_kde, log_g_grid, volume_weighted,
                z_depth,
            ),
            (zgals, dzgals, wgals, ngals),
        )
    else:
        log_kw, sig_eff, log_depth_mass = _map_rows(
            lambda zs, dzs, ws: _row_kernel_state(
                zs, dzs, ws, None, survey.sigma_kde, log_g_grid, volume_weighted,
                z_depth,
            ),
            (zgals, dzgals, wgals),
        )

    row_empty = ~jnp.any(jnp.isfinite(log_kw), axis=-1)
    log_kw_safe = jnp.where(jnp.isfinite(log_kw), log_kw, -1e30)
    log_kw_eff = _fused_log_kw_eff(log_kw_safe, sig_eff)
    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=log_kw_safe, sig_eff=sig_eff,
        log_sig_eff=jnp.log(sig_eff),
        volume_weighted=volume_weighted, z_depth=z_depth,
        log_depth_mass=log_depth_mass,
        sig_eff_row_max=_real_row_max(sig_eff, log_kw),
        rows_sorted=_resolve_rows_sorted_guard(zgals, ngals, kde_window),
        row_empty=row_empty,
        log_kw_eff=log_kw_eff,
        log_kw_eff_rowmax=_log_kw_eff_rowmax(log_kw_eff),
        inv_sig_eff=_inv_sig_eff(log_kw_eff, sig_eff),
        kde_window=_static_window(kde_window),
    )


def build_pinned_kernel_quadrature(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    z_depth=None,
    n_probe: int = KERNEL_PIN_PROBE_ROWS,
) -> PinnedKernelQuadrature:
    """Evaluate the kernel state ONCE at ``KERNEL_PIN_H0_REF`` and box it.

    Call from the likelihood factory with the CONCRETE (pre-jit) catalog arrays
    and the run's reference cosmology; ``cosmo.H0`` is overridden with the
    reference, and every other field must be the run's FIXED value (that is what
    :func:`darksirens.likelihood.factory.kernel_pin_admissible` establishes).

    The build goes through :func:`catalog_kernel_state` under the PROCESS-GLOBAL
    quadrature configuration, unchanged: the pin must be bit-for-bit what the
    live config would compute at ``H0_ref``, or the in-graph probe -- which
    rebuilds under the live config -- would compare two different quadratures
    and fire on a scheme difference rather than on a premise violation.

    That matters more than it looks, because the pin FREEZES whatever scheme the
    run configured: the shipped ``cdf``-24 kernel norm carries a ~5e-5 relative
    error that the survey-global observed count (~1.6e5 galaxies) amplifies into
    an H0-CORRELATED +/-8 nats on the production likelihood (measured against a
    ``zspace``-24/n_sigma-6 reference; the cdf error follows 4723/K^2).  A run
    that wants the accurate quadrature sets it ON THE RUN
    (``--kernel_gl_domain zspace --kernel_gl_nodes 24``, applied by
    ``_configure_performance_grids`` before the likelihood is built) -- which
    the pin then makes free, since it runs once per run instead of once per
    proposal.  Hard-coding a different scheme here would break the probe.
    """
    ref = cosmo._replace(H0=jnp.asarray(KERNEL_PIN_H0_REF, dtype=zgrid.dtype))
    state = catalog_kernel_state(ref, survey, em_catalog, z_depth=z_depth)
    probe_rows = _spread_probe_rows(em_catalog.zgals, em_catalog.ngals, n_probe)
    return PinnedKernelQuadrature(
        H0_ref=ref.H0,
        log_kw=state.log_kw,
        sig_eff=state.sig_eff,
        log_sig_eff=state.log_sig_eff,
        sig_eff_row_max=state.sig_eff_row_max,
        log_depth_mass=state.log_depth_mass,
        row_empty=state.row_empty,
        probe_rows=probe_rows,
        log_kw_eff=state.log_kw_eff,
        log_kw_eff_rowmax=state.log_kw_eff_rowmax,
        inv_sig_eff=state.inv_sig_eff,
    )


# ------------------------------------------------------------
# Marked-host kernel state (galaxy marks -> BBH-host efficiency)
# ------------------------------------------------------------

def _row_marked_kernel_state(
    zs, dzs, ws, log_h_row, ngal, sigma_kde, log_g_grid, z_depth=None
):
    """Per-galaxy kernel quantities for one row using host-efficiency weights.

    Identical to :func:`_row_kernel_state` but the per-pixel-normalised weight is
    ``w_i·h_i`` (host efficiency ``h_i = exp(log_h_row_i)``) instead of ``w_i``,
    and the row's marked total ``log_N_host = log Σ_i w_i h_i`` is returned (it
    replaces the integer count in the assembled prior).  With ``log_h_row ≡ 0``
    and unit weights this reduces to :func:`_row_kernel_state`.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)

    # 1e-300: numerical backstop for padding slots only; real galaxies with
    # w <= 0 are rejected by ``darksirens_pixelate`` (see _row_kernel_state).
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    log_wh = jnp.where(real, log_w + log_h_row, -jnp.inf)   # log(w_i h_i)
    lse = logsumexp(log_wh)                                 # log Σ_i w_i h_i
    has_galaxies = jnp.isfinite(lse)
    log_wh_norm = jnp.where(real, log_wh - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    log_Z = _dispatch_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_wh_norm - log_Z, -jnp.inf)
    # Depth renormalisation mirrors the unmarked twin: the marked mixture SHAPE
    # is renormalised to unit mass on [0, z_depth] and the below-depth mass is
    # returned so the caller scales the row's marked amplitude by it — the
    # marked count Σ_i w_i h_i alone would keep every above-depth galaxy's host
    # probability while _assemble_curves counts those galaxies again in the
    # missing branch (the double count 349b717 removed from the unmarked path).
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)
    if z_depth is not None:
        log_kw, log_depth_mass = _renorm_log_kw_below_depth(
            log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
        )
    # AMPLITUDE CONVENTION (redshift/prior.py's module docstring): the
    # catalog:missing odds are COUNT odds, so the marked amplitude is the
    # observed COUNT times the weighted-mean host efficiency,
    #
    #     N_host = N_obs * <h>_w = N_obs * Σ_i w_i h_i / Σ_i w_i,
    #
    # not the raw Σ_i w_i h_i.  The raw mass carries the arbitrary scale of the
    # WEIGHT column (a luminosity-weighted catalog with L/L_sun ~ 1e10 made the
    # observed branch dominate the missing one by ~1e10 and silently switched the
    # completeness correction off), and the missing branch it is paired against
    # -- (1 - C) dN_exp * mu_miss with mu_miss = E_obs[h|z] -- carries no weight
    # factor at all.  This form is invariant under w -> c*w and is EXACTLY the
    # unmarked count at h == 1, so eta = 0 reduces to the galaxy-count model for
    # any weights.  ``field_marked_observed_global_total`` uses the same
    # convention through ``build_field_mark_inputs``' count-renormalised weights.
    log_N_obs = jnp.log(jnp.maximum(jnp.sum(real.astype(sig_eff.dtype)), 1e-300))
    log_w_tot = logsumexp(log_w)
    log_N_host = jnp.where(has_galaxies, log_N_obs + lse - log_w_tot, -jnp.inf)
    return log_kw, sig_eff, log_N_host, log_depth_mass


def marked_catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_h: jnp.ndarray,
    log_g_grid: jnp.ndarray | None = None,
    z_depth=None,
    kde_window=None,
):
    """Marked per-galaxy kernel state + per-row marked total ``log_N_host``.

    ``log_h`` is ``(N_rows, N_max_gals)`` per-galaxy log host efficiency (from
    :mod:`darksirens.marks`).  Returns ``(CatalogKernelState, log_N_host)`` where
    the state's ``log_kw`` carries the marked (per-pixel-normalised) weights, so
    the existing per-sample evaluator is reused unchanged.  ``z_depth`` behaves
    exactly as in :func:`catalog_kernel_state`: the marked mixture is
    renormalised to unit mass on ``[0, z_depth]`` and the state's
    ``log_depth_mass`` carries the below-depth mass, by which the caller must
    scale the marked amplitude ``exp(log_N_host)`` (see
    :func:`_renorm_log_kw_below_depth`).
    """
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff, log_N_host, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, lh, ng: _row_marked_kernel_state(
                zs, dzs, ws, lh, ng, survey.sigma_kde, log_g_grid, z_depth
            ),
            (zgals, dzgals, wgals, log_h, ngals),
        )
    else:
        log_kw, sig_eff, log_N_host, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, lh: _row_marked_kernel_state(
                zs, dzs, ws, lh, None, survey.sigma_kde, log_g_grid, z_depth
            ),
            (zgals, dzgals, wgals, log_h),
        )

    row_empty = ~jnp.any(jnp.isfinite(log_kw), axis=-1)
    log_kw_safe = jnp.where(jnp.isfinite(log_kw), log_kw, -1e30)
    log_kw_eff = _fused_log_kw_eff(log_kw_safe, sig_eff)
    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=log_kw_safe, sig_eff=sig_eff,
        log_sig_eff=jnp.log(sig_eff),
        log_depth_mass=log_depth_mass, z_depth=z_depth,
        sig_eff_row_max=_real_row_max(sig_eff, log_kw),
        rows_sorted=_resolve_rows_sorted_guard(zgals, ngals, kde_window),
        row_empty=row_empty,
        log_kw_eff=log_kw_eff,
        log_kw_eff_rowmax=_log_kw_eff_rowmax(log_kw_eff),
        inv_sig_eff=_inv_sig_eff(log_kw_eff, sig_eff),
        kde_window=_static_window(kde_window),
    ), log_N_host


def _logsumexp_neginf_safe(terms):
    """logsumexp returning exactly -inf for an all--inf row WITHOUT the NaN
    backward pass of the plain reduction: softmax of all--inf is 0/0 = NaN,
    and that NaN survives multiplication by a ZERO upstream cotangent (mul's
    VJP scales by the stored NaN), poisoning every parameter's gradient —
    this is what broke NumPyro NUTS for dark sirens (empty catalog pixels).
    Rows with any finite entry are bit-identical to plain logsumexp: the
    sanitized padding's weight exp(-1e30 - max) underflows to exactly zero.
    """
    finite = jnp.isfinite(terms)
    safe = jnp.where(finite, terms, -1e30)
    return jnp.where(jnp.any(finite), logsumexp(safe), -jnp.inf)


def eval_log_catalog_prior_state(
    z: float,
    pix: int,
    state: CatalogKernelState,
    em_catalog: EMCatalog,
) -> float:
    """
    log p_cat(z | pix) using a precomputed ``CatalogKernelState``.

    O(W) per sample on the windowed hot path (rows z-sorted, see
    :func:`configure_catalog_kde_window`): a binary search over the row's real
    prefix, one fused 2-D windowed gather per array, one Gaussian logpdf per
    windowed galaxy, one reduction.  Falls back to the historical O(N_max)
    full-row evaluation when windowing is disabled, the rows are not
    verifiably sorted, or the row is not longer than the window.

    The reduction is ONE pass: it subtracts the row's build-time maximum
    (``CatalogKernelState.log_kw_eff_rowmax``) instead of the sample's own, so
    the ``logsumexp`` amax pass -- and with it the (N_samples, N_max) f64
    intermediate XLA materialises between the two passes -- disappears.  The
    price is an underflow edge below the row maximum -- 708.4 nats on the XLA
    CPU backend, 744.4 on CUDA, with a degradation band just above it; see the
    branch itself.

    Self-verifying windowed branch.  Whether to EMIT the windowed code is a
    trace-time Python decision that reads mutable process globals
    (``_KDE_WINDOW_SIZE``, ``_ROWS_SORTED_ATTESTED``), none of which is part of
    a jit cache key -- so a callable compiled while a sorted view was attested
    can be replayed later on an unsorted view of the SAME SHAPE and silently
    return the windowed answer for data the window is invalid on (MEASURED: a
    (1, 20) row at window 8 shifted log p_cat by 2.5e-3 nats on a permutation of
    the same galaxies, and worse on realistic rows).  The emitted branch
    therefore carries ``state.rows_sorted`` -- the invariant evaluated on the
    arrays the compiled graph actually receives -- and collapses to NaN when it
    is violated, so such a replay is loud instead of plausible.  Attest every
    view you bind (:func:`attest_rows_sorted_for_windowing`), or disable
    windowing with ``configure_catalog_kde_window(size=None)``.
    """
    window = _effective_kde_window(getattr(state, "kde_window", None))
    use_window = (
        window is not None
        and state.sig_eff_row_max is not None
        and em_catalog.ngals is not None
        and getattr(em_catalog.zgals, "ndim", 0) == 2
        and em_catalog.zgals.shape[1] > window
        and (
            _rows_sorted_for_windowing(em_catalog.zgals, em_catalog.ngals)
            # Traced catalogs (jit arguments — the production likelihood path)
            # cannot be verified here; the factory attests them at build time
            # with the concrete arrays (attest_rows_sorted_for_windowing), and
            # only the row shapes it attested are armed.
            or _traced_rows_attested(em_catalog.zgals)
        )
    )
    # ONE int32 row index for every use below (the windowed dynamic slice, the
    # full-row gathers, the row-max offset and the empty-row select).
    pix_i = jnp.asarray(pix, dtype=jnp.int32)
    if use_window:
        n_real = em_catalog.ngals[pix_i]
        start = _sorted_row_window_start(
            em_catalog.zgals, pix_i, z, n_real, window
        )

        # Fused 2-D windowed gathers (contiguous dynamic slices; no index
        # vector, never the full row).  Padded slots carry log_kw = -1e30
        # (sanitized at state build time), so plain logsumexp is gradient-safe
        # even for all-padding windows.
        def _gather(a):
            return lax.dynamic_slice(a, (pix_i, start), (1, window))[0]
    else:
        def _gather(a):
            return a[pix_i]

    log_kw_eff = getattr(state, "log_kw_eff", None)
    rowmax = getattr(state, "log_kw_eff_rowmax", None)
    inv_sig_eff = getattr(state, "inv_sig_eff", None)
    zs = _gather(em_catalog.zgals)
    if (log_kw_eff is not None
            and rowmax is not None and inv_sig_eff is not None):
        # ONE-PASS reduction with a BUILD-TIME offset.  ``logsumexp`` is two
        # reductions over the row -- an amax, then exp+sum -- and the gathered
        # per-galaxy vector cannot be produced once and consumed twice inside a
        # single XLA fusion, so the compiler materialises the whole
        # (N_samples, N_max) f64 exponent array between them (MEASURED on the
        # 259-event DESI nside-64 production run: 14.6 GB per sample set,
        # written by one fusion and re-read by the next, which also re-gathers
        # ``log_kw_eff``).  The row's maximum is a BUILD-TIME constant of the
        # catalog, so subtracting it instead removes the first pass entirely:
        # ``log_kw_eff_i - rowmax <= 0`` and ``-0.5 u^2 <= 0``, hence every
        # exponent is <= 0 and nothing can overflow.  What is left is one
        # register-resident input fusion.  MEASURED on an H100 NVL, PROD_ARGS,
        # median of 20 warm calls: 59.2 -> 32.2 ms, peak device memory
        # 22.62 -> 12.24 GB.  The division is fused out for the same reason
        # (it is what forced XLA to break the fusion), one multiply by the
        # build-time ``1 / sig_eff`` instead; both halves are needed --
        # MEASURED, either one alone buys 5.7 ms and leaves the 14.6 GB
        # intermediate in place.
        #
        # Exactness: sum_i exp(x_i - M) is the same real number for ANY offset
        # M, so this is re-association only (MEASURED max |dlog p_cat| 1.1e-13
        # per sample, aggregate log-likelihood bit-identical at 8 prior
        # coordinates).  Nothing here carries a sampled label -- the offset is
        # a catalog constant and ``sig_eff`` is theta-invariant once
        # ``sigma_kde`` is fixed -- so no H0-correlated component is possible.
        # The one behaviour change is a finite OFFSET BUDGET, and it is graded
        # rather than binary.  ``exp`` of a large negative argument underflows
        # at the backend's edge: the XLA CPU backend FLUSHES subnormal results,
        # so the budget there is ln(smallest normal f64) = 708.40 nats, while
        # CUDA keeps subnormals and floors at ln(smallest subnormal) = 744.44.
        # Just above the edge the value is still finite but no longer exact --
        # MEASURED on an H100 NVL with a one-galaxy row: relative drift 4e-15
        # at 720 nats of deficit, 2e-12 at 725 (the first point past the
        # campaign's ulp-level bar), 3.4e-4 (0.25 nats) at 744, -inf from 745.
        # On XLA CPU the same band appears by a different mechanism -- whole
        # kernel terms flushed to zero, since the offset is the ROW's maximum
        # and not the sample's -- and MEASURED 5.9 nats of error on a finite
        # value at 707.5-708.2 nats of deficit on a synthetic 1001-galaxy row
        # whose weights span 1 nat.  None of it is observable downstream: a
        # sample 708 nats under its row's peak carries weight ``exp(-708) = 0``
        # in the ``logaddexp`` the caller weighs it in, whatever value it
        # returns.  And the production configuration stays 530 nats clear of
        # the nearer (CPU) edge -- worst headroom actually used below
        # ``z_depth``: 178 nats of the 708 available, over 51,959 PE + 52,594
        # selection samples at 5 values of H0; ``test_kde_window_auto.py``
        # pins the edge against the live backend.
        u = (z - zs) * _gather(inv_sig_eff)
        m = rowmax[pix_i]
        s = jnp.sum(jnp.exp(_gather(log_kw_eff) - m - 0.5 * u * u))
        # The double ``where`` is the gradient-safe spelling: a naked
        # ``log(s)`` returns NaN cotangents when a row underflows to s = 0,
        # which is exactly what broke NumPyro NUTS on empty catalog pixels.
        log_mix = m + jnp.where(s > 0, jnp.log(jnp.where(s > 0, s, 1.0)),
                                -jnp.inf)
    elif log_kw_eff is not None:
        # A state built before the offset leaves existed: three gathers, the
        # sample-independent part fused at build time, and the historical
        # two-pass ``logsumexp``.
        d = (z - zs) / _gather(state.sig_eff)
        log_mix = logsumexp(_gather(log_kw_eff) - 0.5 * d * d)
    else:
        # Hand-built state without the fused array: the historical arithmetic.
        sig = _gather(state.sig_eff)
        d = (z - zs) / sig
        log_kw = _gather(state.log_kw)
        log_sig = (_gather(state.log_sig_eff) if state.log_sig_eff is not None
                   else jnp.log(sig))
        log_gauss = -0.5 * d * d - log_sig - _HALF_LOG_2PI
        log_mix = logsumexp(log_kw + log_gauss)
    # The build-time -1e30 sanitize makes plain logsumexp gradient-safe but
    # would return a finite ~-1e30 for an empty pixel; restore the exact -inf
    # contract with one per-row scalar select (zero gradient on empty rows,
    # bit-identical elsewhere -- the sanitized padding underflows to 0 weight).
    if state.row_empty is not None:
        log_mix = jnp.where(state.row_empty[pix_i], -jnp.inf, log_mix)
    # Volume-weighted (complete-catalog) kernels already carry g(z_i) in their
    # weights, so no front g(z); otherwise reapply the per-sample galaxy measure
    # g(z) that Z_i divided out per kernel.  ``volume_weighted`` is a static bool.
    log_g_front = jnp.where(state.volume_weighted, 0.0,
                            log_interp_zgrid(z, state.log_g_grid))
    log_p_cat = log_g_front + log_mix
    # Depth truncation: a magnitude-limited survey catalogs nothing past
    # ``z_depth``, so p_cat asserts nothing there -- zero it (the missing branch
    # supplies the full expected population beyond the depth).  ``z_depth`` is a
    # concrete Python float or ``None`` (never traced), so this is a Python-level
    # branch resolved once per trace; ``None`` leaves p_cat untouched (legacy).
    # ``log_kw`` is already renormalised to unit mass on [0, z_depth], so p_cat
    # stays a proper density.
    if state.z_depth is not None:
        log_p_cat = jnp.where(z <= state.z_depth, log_p_cat, -jnp.inf)
    if use_window and state.rows_sorted is not None:
        # The windowed answer is valid ONLY on z-sorted rows.  ``rows_sorted``
        # is evaluated on the arrays this graph receives (a compile-time literal
        # for concrete catalogs, so the select folds away), which is what a
        # process global read at trace time can never be.
        log_p_cat = jnp.where(state.rows_sorted, log_p_cat, jnp.nan)
    return log_p_cat


def _log_catalog_prior_impl(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> float:
    """Un-jitted body shared by the scalar and vector boundaries below.

    Deliberately NOT decorated: it reaches the 106.8 MB comoving-distance
    table through ``log_galaxy_measure_grid``, so its only jit boundaries must
    be :func:`threads_distance_table` ones (see ``utils.cosmology``).  Keeping
    the body plain lets ``log_catalog_prior_vmap`` vmap it directly instead of
    nesting a scalar jit inside the vector one.
    """
    zs = em_catalog.zgals[pix]
    dzs = em_catalog.dzgals[pix]
    ws = em_catalog.wgals[pix]
    ngal = None if em_catalog.ngals is None else em_catalog.ngals[pix]

    log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    # Unit-mass kernel convention, matching the dark_sirens_complete prior
    # state (prepare_redshift_prior_state): catalog counts already track
    # hosts per redshift shell, so volume-weighted (True) kernels would
    # double-count dV_c/dz for any catalog whose dN/dz follows the volume.
    log_kw, sig_eff, _log_depth_mass = _row_kernel_state(
        zs, dzs, ws, ngal, survey.sigma_kde, log_g_grid, False
    )
    log_g_z = log_interp_zgrid(z, log_g_grid)
    return log_g_z + _logsumexp_neginf_safe(log_kw + norm.logpdf(z, zs, sig_eff))


# ``threads_distance_table`` rather than ``@jit``, on BOTH boundaries.  These
# reach the 106.8 MB comoving-distance table through ``log_galaxy_measure_grid``
# -> ``dV_of_z`` -> ``r_of_z``, and ``_log_prior_bright_sirens`` resolves the
# vector one from INSIDE an enclosing traced boundary.  A plain ``@jit``
# MEASURED (jax 0.4.34):
#
#   * lowering the scalar function serialised 229.7 MB of module text -- the
#     table as a dense<> HLO CONSTANT rather than a parameter, rebuilt,
#     re-serialised and re-parsed by XLA on every compilation;
#   * the public legacy bright-siren path succeeded for z shape (1,) and then
#     raised ``UnexpectedTracerError`` for z shape (2,) in the SAME process,
#     because the plain jit closed over the enclosing trace's table tracer and
#     JAX replayed the cached jaxpr from that now-dead trace.
#
# ``tests/test_catalog_prior_distance_table.py`` pins both properties.

@threads_distance_table()
def log_catalog_prior(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
) -> float:
    r"""
    Log of the EM-catalog redshift prior at redshift z for catalog row pix.

    Historical scalar signature; builds the per-row kernel state on the
    fly.  Under ``vmap`` over samples, the (cosmo, survey)-only pieces
    (the log g grid) are hoisted automatically; the per-row pieces are
    recomputed per sample, so hot paths should use
    ``catalog_kernel_state`` + ``eval_log_catalog_prior_state`` instead.

    Empty rows return -inf (no host candidates), never NaN.

    ``distance_table`` is the comoving-distance grid; leaving it ``None``
    resolves whatever table is active for the current trace.  It is a jit
    ARGUMENT, never a closure capture -- see ``utils.cosmology``.
    """
    return _log_catalog_prior_impl(z, pix, cosmo, survey, em_catalog)


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.  ONE distance-aware
# boundary over the plain body, not a jit wrapped around a jit.
@threads_distance_table()
def log_catalog_prior_vmap(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    distance_table=None,
) -> jnp.ndarray:
    """``log_catalog_prior`` over matched ``(z, pix)`` arrays."""
    return vmap(_log_catalog_prior_impl, in_axes=(0, 0, None, None, None), out_axes=0)(
        z, pix, cosmo, survey, em_catalog
    )
