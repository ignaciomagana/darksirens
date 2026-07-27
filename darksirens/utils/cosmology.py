"""JAX-compatible flat-CPL distance and volume utilities.

The likelihood evaluates cosmological factors for every posterior sample and
selection injection, so these helpers avoid calling Astropy inside JIT-compiled
code.  At import time the module builds a four-dimensional interpolation table
of comoving distance as a function of redshift, ``Om0``, ``w0``, and ``wa`` at
the Planck-2015 ``H0``.  Runtime functions rescale that table by
``H0Planck / H0`` and expose JIT-able distance, inverse-distance, and
differential-volume operations.

Distances are in Mpc, ``H0`` is in km/s/Mpc, and redshift is dimensionless.  The
interpolation grid covers the prior range used by :mod:`darksirens.inference`
with guard bands so that sampler proposals at the edge of the allowed prior do
not silently extrapolate.
"""

import contextlib
import contextvars
import functools
import inspect
import os

import astropy.constants as constants
import jax
import numpy as np
from astropy.cosmology import Flatw0waCDM, Planck15
from jax import jit
from jax import numpy as jnp

from darksirens.utils.interp2d import interpnd, interpnd_scalar_head

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

zMax = float(os.environ.get("DARKSIRENS_ZMAX", 5))
"""Maximum redshift covered by the precomputed interpolation grid.

Default 5 (unchanged numerics). Override with the DARKSIRENS_ZMAX environment
variable (read once at import) for high-redshift studies — e.g. Madau-Dickinson
rate inference with strongly lensed sources whose true z exceeds 5. The z-grid
node count scales with the log range so low-z interpolation accuracy is
preserved; darksirens.redshift.grid reads the same variable, keeping the two
grids consistent."""

H0Planck = Planck15.H0.value
"""Planck-2015 Hubble constant used to build the reference distance grid."""

Om0Planck = float(Planck15.Om0)
"""Planck-2015 matter density used as the center of the interpolation grid."""

w0Fiducial = -1.0
"""Fiducial CPL present-day dark-energy equation-of-state parameter."""

waFiducial = 0.0
"""Fiducial CPL dark-energy evolution parameter."""

speed_of_light = constants.c.to("km/s").value
"""Speed of light in km/s, matching the units of ``H0``."""

# The default prior in inference/prior.py covers [Om0Planck-0.1,
# Om0Planck+0.1].  The interpolation grid must extend strictly beyond those
# bounds so that sampler proposals at or near the prior boundary never
# extrapolate outside the grid.  The CPL ranges below use the same convention:
# an intended prior range plus a guard pad on both sides.
_OM0_PRIOR_HALF_WIDTH = 0.1
_OM0_GRID_PAD = 0.05
_W0_PRIOR_HALF_WIDTH = 1.0
_W0_GRID_PAD = 0.25
_WA_PRIOR_HALF_WIDTH = 2.0
_WA_GRID_PAD = 0.5

Om0PriorLower = Om0Planck - _OM0_PRIOR_HALF_WIDTH
Om0PriorUpper = Om0Planck + _OM0_PRIOR_HALF_WIDTH
w0PriorLower = w0Fiducial - _W0_PRIOR_HALF_WIDTH
w0PriorUpper = w0Fiducial + _W0_PRIOR_HALF_WIDTH
waPriorLower = waFiducial - _WA_PRIOR_HALF_WIDTH
waPriorUpper = waFiducial + _WA_PRIOR_HALF_WIDTH

# Node count scales with the log-z range (500 at the default zMax=5) so a
# raised DARKSIRENS_ZMAX keeps the same low-z node density.
_ZGRID_NODES = max(500, int(round(500 * np.log(zMax + 1.0) / np.log(6.0))))
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax + 1), _ZGRID_NODES))

# ── Node counts: allocated by measured interpolation error, not by habit ───────
#
# Multilinear interpolation is exact AT the nodes and worst at cell midpoints,
# with error ~ (cell width)^2, so the accuracy budget should be spent where the
# midpoint error actually is.  Measured against astropy's Flatw0waCDM over
# z in [0.01, 4.5] (max relative error in dL):
#
#                        nodes   cell     midpoint err
#     on-node floor        --     --        5.4e-5      <- set by the z axis
#     Om0                  31    0.010      5.5e-5      <- AT the floor
#     w0                   21    0.125      5.1e-4
#     wa                   21    0.250      2.8e-4
#     worst over the physical (w0 < -0.3) grid           2.96e-3
#
# The Om0 axis was resolved ~10x finer than it needed to be: its midpoint error
# had already reached the floor imposed by interpolating r(z) linearly in z, so
# extra Om0 nodes bought nothing while w0 and wa carried ~5x more error each.
# Reallocating to 21/41/31 gives
#
#     Om0 21 (0.015), w0 41 (0.0625), wa 31 (0.167):
#         w0 midpoint 1.18e-4, wa midpoint 1.17e-4, worst 9.74e-4  (3x better)
#
# for a 54.7 -> 106.8 MB table.  The separable interpolation path never touches
# the full table (it contracts these three axes into a 500-node curve first), so
# the size increase costs residency, not per-query bandwidth.
#
# NOTE these errors are ZERO at the nodes and maximal at midpoints, so they are
# a ripple on the likelihood surface with the grid period, not a smooth bias.
# The fiducial Om0, w0 = -1 and wa = 0 all remain EXACTLY on nodes under both
# the old and new counts, and the interpolated output at fiducial cosmology is
# bitwise identical between them — so H0-only and fixed-cosmology runs (which
# is most of them, including the golden regression) are unaffected.  Only runs
# that actually sample Om0/w0/wa see a change, and there the change is a
# reduction in interpolation error.
#
# The residual 5.4e-5 floor is the z axis and cannot be reduced from here; it
# would need a finer zgrid or a higher-order rule in z.

Om0grid = jnp.linspace(
    Om0PriorLower - _OM0_GRID_PAD,
    Om0PriorUpper + _OM0_GRID_PAD,
    21,
)
"""Matter-density coordinates of the distance interpolation grid."""

w0grid = jnp.linspace(
    w0PriorLower - _W0_GRID_PAD,
    w0PriorUpper + _W0_GRID_PAD,
    41,
)
"""CPL ``w0`` coordinates of the distance interpolation grid."""

wagrid = jnp.linspace(
    waPriorLower - _WA_GRID_PAD,
    waPriorUpper + _WA_GRID_PAD,
    31,
)
"""CPL ``wa`` coordinates of the distance interpolation grid."""

# Keep an Astropy cosmology object attached to the grid construction so that the
# tabulated model is explicitly the same flat CPL family Astropy calls
# ``Flatw0waCDM``.  The grid values themselves are produced by the equivalent
# analytic CPL expansion law below; this avoids thousands of slow Astropy
# distance integrations at import time while preserving JAX-free grid setup.
_reference_cosmology = Flatw0waCDM(
    H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial
)


def _cpl_E_numpy(z, Om0, w0, wa):
    """NumPy version of the flat CPL expansion law used during grid setup."""
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * np.exp(-3.0 * wa * z / one_plus_z)
    )
    return np.sqrt(Om0 * one_plus_z**3 + dark_energy)


def _cumulative_trapezoid(y, x):
    """Return cumulative trapezoid integrals with zero initial value."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _build_cpl_distance_grid():
    """Build ``r(Om0, w0, wa, z)`` in Mpc for the flat CPL grid."""
    # The Astropy object above documents the model family mirrored by this table.
    r_grid = np.empty(
        (len(Om0grid), len(w0grid), len(wagrid), len(zgrid)), dtype=np.float64
    )
    z = np.asarray(zgrid, dtype=np.float64)
    for i, Om0 in enumerate(np.asarray(Om0grid)):
        for j, w0 in enumerate(np.asarray(w0grid)):
            for k, wa in enumerate(np.asarray(wagrid)):
                inv_E = 1.0 / _cpl_E_numpy(z, float(Om0), float(w0), float(wa))
                r_grid[i, j, k, :] = (
                    speed_of_light / H0Planck
                ) * _cumulative_trapezoid(inv_E, z)
    return r_grid


zgrid = jnp.array(zgrid)
"""Redshift coordinates of the distance interpolation grid."""

rs = jnp.asarray(_build_cpl_distance_grid())
"""Comoving-distance table indexed by ``(Om0grid, w0grid, wagrid, zgrid)`` in Mpc."""


# ── The table travels as a jit ARGUMENT, never as a closure capture ────────────
#
# ``jax.jit`` lowers a closed-over concrete ``jax.Array`` to a ``dense<>`` HLO
# CONSTANT rather than to a parameter (jax 0.4.34), and a float64 literal costs
# ~16 bytes of module text per element.  ``rs`` is 21x41x31x500 = 1.33e7 elements
# / 106.8 MB, so every jit that reached it through a module-global closure carried
# ~214 MB of literal per embedding.  MEASURED before this change: the production
# spectral likelihood lowered to 427.5 MB of module text (two embeddings) and the
# dark-siren mock to 443.6 MB (three) -- text that has to be built, serialised and
# parsed by XLA on every compilation, and whose buffers are duplicated in the
# executable even though the table is already device-resident.
#
# The fix is the pattern PR #296 applied to the likelihood operands: thread the
# array through as an ARGUMENT.  Doing that by hand would mean adding a ``table``
# parameter to ~45 call sites spread over a dozen modules (likelihood.core,
# wl_weight, cluster_likelihood, inference.utils, redshift.*, sky.models, ...)
# plus every intermediate signature between them, so instead the table is bound
# DYNAMICALLY for the extent of a trace and re-materialised as an explicit
# argument at each jit boundary by :func:`threads_distance_table`:
#
#   * the public entry points take the table as a real (keyword-only) parameter
#     whose default is resolved OUTSIDE the jit, so an eager call hands XLA a
#     device-resident parameter instead of a literal;
#   * inside the jit the decorator re-binds the context variable to that jit's OWN
#     tracer, so the deep call sites resolve a value that belongs to the trace
#     they are in.
#
# The re-binding at every boundary is not cosmetic.  A ``jax.jit`` function that
# closes over a tracer is not cached by its closure, and JAX's jaxpr-tracing cache
# will happily replay a jaxpr whose consts are tracers from a DEAD trace --
# reproduced on jax 0.4.34 as ``UnexpectedTracerError`` the first time the same
# module-level jit is re-traced under a differently-shaped outer jit.  Any NEW
# module-level jit that reaches the table must therefore be declared with
# :func:`threads_distance_table` as well; a plain ``@jit`` that only reads the
# context variable is the bug this note exists to prevent.
#
# The 1-D coordinate grids (``zgrid``, ``Om0grid``, ``w0grid``, ``wagrid``: 593
# elements together, ~10 KB of text) are deliberately left as closures -- they are
# five orders of magnitude below the table and threading them would buy nothing.

_ACTIVE_DISTANCE_TABLE = contextvars.ContextVar(
    "darksirens_active_distance_table", default=None
)


def distance_table():
    """Return the comoving-distance table in force for the current trace.

    Outside any :func:`bound_distance_table` scope this is the module-level
    :data:`rs`; inside one it is the caller-supplied array (typically a tracer
    standing for a jit argument).
    """
    active = _ACTIVE_DISTANCE_TABLE.get()
    return rs if active is None else active


def resolve_distance_table(table=None):
    """``table`` if it was given, else the currently active table."""
    return distance_table() if table is None else table


@contextlib.contextmanager
def bound_distance_table(table):
    """Make ``table`` the active comoving-distance table for the enclosed scope.

    ``None`` is a no-op, so callers can pass an unresolved argument through
    without special-casing it.  Used at jit boundaries (see
    :func:`threads_distance_table`) to hand the deep cosmology call sites the
    tracer that belongs to their own trace.
    """
    if table is None:
        yield
        return
    token = _ACTIVE_DISTANCE_TABLE.set(table)
    try:
        yield
    finally:
        _ACTIVE_DISTANCE_TABLE.reset(token)


def threads_distance_table(**jit_kwargs):
    """Jit ``fn`` with its ``distance_table`` parameter as a real jit ARGUMENT.

    Replaces ``@partial(jax.jit, **jit_kwargs)`` on any function that reaches the
    comoving-distance table.  The decorated function must declare a trailing
    ``distance_table=None`` parameter; its body is otherwise untouched.

    Two things happen around the original function:

    ``outside`` the jit
        the public wrapper resolves ``distance_table=None`` to the currently
        active table (:func:`distance_table`) and passes it explicitly, so the
        array is a parameter of the lowered module instead of a literal;
    ``inside`` the jit
        the table -- now this trace's own tracer -- is installed as the active
        table, so nested cosmology calls and every deeper consumer pick it up
        without carrying it in their signatures.

    The jitted callable is exposed as ``.jitted`` for tests and diagnostics that
    want ``lower()`` / ``_cache_size()``.
    """

    def decorate(fn):
        signature = inspect.signature(fn)
        if "distance_table" not in signature.parameters:
            raise TypeError(
                f"{fn.__qualname__} must declare a 'distance_table' parameter to "
                "be decorated with threads_distance_table"
            )

        @functools.wraps(fn)
        def _bind_then_call(*args, **kwargs):
            table = signature.bind(*args, **kwargs).arguments.get("distance_table")
            with bound_distance_table(table):
                return fn(*args, **kwargs)

        jitted = jax.jit(_bind_then_call, **jit_kwargs)

        @functools.wraps(fn)
        def public(*args, distance_table=None, **kwargs):
            return jitted(
                *args, distance_table=resolve_distance_table(distance_table), **kwargs
            )

        public.jitted = jitted
        return public

    return decorate


@jit
def E(z, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return the dimensionless CPL expansion rate ``H(z) / H0``.

    The expression assumes a spatially flat CPL cosmology with matter density
    ``Om0`` and dark-energy density ``1 - Om0``.  Inputs may be scalars or JAX
    arrays and are broadcast by JAX.
    """
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * jnp.exp(-3.0 * wa * z / one_plus_z)
    )
    return jnp.sqrt(Om0 * one_plus_z**3 + dark_energy)


@threads_distance_table()
def r_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial, distance_table=None):
    """Return line-of-sight comoving distance at redshift ``z``.

    The distance is interpolated from the precomputed ``(Om0, w0, wa, z)`` grid
    and rescaled from the Planck reference Hubble constant to the requested
    ``H0``.  Values outside the tabulated grid are returned as ``NaN`` rather
    than extrapolated.

    ``Om0``/``w0``/``wa`` are scalars on every likelihood path (they come from
    a :class:`~darksirens.core.types.CosmoParams`, one cosmology per sample),
    while ``z`` carries the sample axis.  In that case the three cosmology axes
    are contracted into a 1-D ``r(z)`` curve BEFORE the redshift lookup, so the
    per-point gathers hit a 4 KB curve instead of the 54.7 MB table — the same
    number up to floating-point reassociation (~1e-15 relative), measured
    0.95x/1.64x/1.95x at N = 1e5/1e6/4e6 on an H100 NVL (so: a wash on small
    queries, ~1.6-2x on selection-integral-sized ones).  A non-scalar
    cosmology coordinate (e.g. a vmap that maps over Om0 directly) falls back to
    the general ``interpnd`` path.  ``jnp.ndim`` is a trace-time shape query, so
    the branch is resolved at compile time and never appears in the graph.

    ``distance_table`` is the comoving-distance grid to interpolate; leaving it
    ``None`` picks up :func:`distance_table` (the module-level :data:`rs`, or
    whatever an enclosing :func:`bound_distance_table` installed).  It is a jit
    ARGUMENT, never a closure capture -- see the note above :data:`rs`.
    """
    if jnp.ndim(Om0) == 0 and jnp.ndim(w0) == 0 and jnp.ndim(wa) == 0:
        interpolate = interpnd_scalar_head
    else:
        interpolate = interpnd
    return interpolate(
        (Om0, w0, wa, z),
        (Om0grid, w0grid, wagrid, zgrid),
        distance_table,
        fill_value=jnp.nan,
    ) * (H0Planck / H0)


@threads_distance_table()
def dL_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial, distance_table=None):
    """Return luminosity distance for redshift ``z`` in Mpc."""
    return (1 + z) * r_of_z(z, H0, Om0, w0, wa)


@threads_distance_table()
def dL_grid_bounds(H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                   distance_table=None):
    """Return the luminosity-distance support covered by ``zgrid``.

    The pair ``(dL_min, dL_max)`` is useful for masking posterior samples before
    inverse interpolation.  The lower bound is zero because the redshift grid
    starts at ``z = 0``.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    return dL_grid[0], dL_grid[-1]


@threads_distance_table()
def dL_in_z_grid(dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                 distance_table=None):
    """Return a boolean mask for distances supported by the redshift grid."""
    dL_min, dL_max = dL_grid_bounds(H0, Om0, w0, wa)
    return (dL >= dL_min) & (dL <= dL_max)


@threads_distance_table()
def z_of_dL(dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
            distance_table=None):
    """Invert luminosity distance to redshift by one-dimensional interpolation.

    Unsupported distances are returned as ``NaN`` rather than extrapolated.  The
    likelihood converts those values to zero or ``-inf`` contributions depending
    on the surrounding calculation, which prevents out-of-grid samples from
    looking artificially valid.
    """
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    z = jnp.interp(dL, dL_grid, zgrid)
    return jnp.where(in_grid, z, jnp.nan)


@threads_distance_table()
def dV_of_z(z, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
            distance_table=None):
    """Return the differential comoving-volume factor per steradian.

    The returned value is ``c * r(z)^2 / (H0 * E(z))`` in Mpc^3.  Angular pixel
    areas are applied by the electromagnetic prior code when a prior density per
    HEALPix pixel is required.
    """
    return speed_of_light * r_of_z(z, H0, Om0, w0, wa) ** 2 / (H0 * E(z, Om0, w0, wa))


@jit
def ddL_of_z(z, dL, H0, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Return ``d dL / dz`` for the flat-CPL luminosity-distance relation.

    The derivative is used when transforming redshift densities into
    luminosity-distance densities.  ``dL`` is accepted as an argument so callers
    that already computed the luminosity distance can avoid recomputing it.
    """
    return dL / (1 + z) + speed_of_light * (1 + z) / (H0 * E(z, Om0, w0, wa))
