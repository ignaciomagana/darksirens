"""Shared redshift grid for redshift priors and completion models."""

import math
import os

import jax
import jax.numpy as jnp

# Enable x64 BEFORE building the module-level ``zgrid`` below.  ``darksirens.
# redshift`` imports this module FIRST (before ``cosmology``, which self-enables
# x64), and the package root ``__init__`` is deliberately side-effect-free, so a
# cold ``import darksirens.redshift`` outside the CLI would otherwise freeze
# ``zgrid`` to float32 for the process lifetime -- silently degrading every
# downstream redshift/volume/completion computation (the import-order trap
# documented in tests/conftest.py).  Mirrors ``cosmology.py``.
jax.config.update("jax_enable_x64", True)

# Log-spaced from z~0 to zMax, giving 1000 points at the default zMax=5.
# expm1(linspace(log(1), log(zMax+1))) maps [0, log(zMax+1)] → [0, zMax].
# DARKSIRENS_ZMAX overrides the cap (read once at import; the cosmology
# module reads the same variable so the two grids stay consistent); the node
# count scales with the log range to preserve low-z density.
zMax: float = float(os.environ.get("DARKSIRENS_ZMAX", 5.0))
_ZGRID_NODES = max(1000, int(round(1000 * math.log(zMax + 1.0) / math.log(6.0))))
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), _ZGRID_NODES))
# The cosmology module tabulates ITS grid with numpy's expm1/log while the
# grid above uses jnp's (XLA libm): at some DARKSIRENS_ZMAX values the two
# endpoints differ by one ulp (e.g. 0.52, 0.65) and this grid's last node
# lands ABOVE the distance table, which the out-of-range mask turns into a
# NaN r(zMax) that poisons every consumer of chi(zgrid). Pin the endpoint to
# the numpy value whenever it is lower (numpy specifically -- math's libm
# differs from numpy's at exactly the pathological values); bit-inert at the
# default zMax=5 and at every zMax where the two agree.
import numpy as _np

zgrid = zgrid.at[-1].set(
    jnp.minimum(zgrid[-1], float(_np.expm1(_np.log(zMax + 1.0)))))

# ``zgrid`` is expm1 of a UNIFORM grid in log(1+z), so the bracketing node of
# any z is available in closed form as floor(log1p(z)/_DLOG) -- no binary
# search.  ``jnp.interp`` instead calls ``searchsorted``, whose default 'scan'
# method is a serial ceil(log2(N))-level lax.scan with a dependent gather per
# level; that chain is the most-executed primitive in the redshift stack (the
# per-proposal O(N_rows x N_max x 24) Gauss-Legendre reduction in
# ``catalog._row_log_kernel_norms`` alone). See :func:`_interp_zgrid`.
_DLOG = math.log(zMax + 1.0) / (_ZGRID_NODES - 1)
# ``jnp.interp``'s guard against a zero-width cell, reproduced verbatim.
_INTERP_DX0_EPS = float(_np.spacing(_np.finfo(_np.dtype(zgrid.dtype)).eps))
# Escape hatch: ``DARKSIRENS_INTERP_SEARCHSORTED=1`` restores the ``jnp.interp``
# path.  The closed-form index reproduces ``searchsorted(side='right')`` exactly
# and the interpolation is spelled identically, so the two paths are pointwise
# bit-identical in isolation; fused into a larger graph, dropping the scan lets
# XLA contract the surrounding arithmetic differently, which moves the last ulp
# (~1e-12 relative) of downstream reductions.  The hatch is what makes that
# A/B'able in one process (tests/test_grid_interp_closed_form.py).
_USE_SEARCHSORTED = os.environ.get("DARKSIRENS_INTERP_SEARCHSORTED") == "1"


def zgrid_upper_index(z):
    """Index ``i`` of the UPPER node of the ``zgrid`` cell bracketing ``z``.

    Returns exactly ``clip(searchsorted(zgrid, z, side='right'), 1, n - 1)``
    -- the shared index behind :func:`_interp_zgrid` and
    ``redshift.prior._grid_bracket`` -- without the binary search.  ``zgrid``
    is ``expm1`` of a UNIFORM grid in log(1+z) (see the construction above), so
    ``v = log1p(z)/_DLOG`` IS the node coordinate: measured against the
    tabulated nodes, ``|v - k| <= 2.3e-13`` at the 1086 nodes and ``v`` sits
    strictly inside ``(k, k+1)`` in between, so ``floor(v)`` is the true index
    except within ~2e-13 of a node, where it is off by exactly one.  The two
    one-step corrections below compare ``z`` against the EXACT tabulated node
    values and recover that case; an off-by-two would need a ``log1p`` relative
    error of order ``_DLOG/log1p(z) ~ 1e-3``, thirteen orders of magnitude of
    headroom, which is why the CPU verification carries over to XLA's libm.
    The uniform-log(1+z) construction, not any node-agreement assertion, is the
    admissibility precondition; it holds for every ``DARKSIRENS_ZMAX``
    including the pinned-last-node cases (0.52, 0.65).

    ``z >= zgrid[-1]`` is pinned to ``n - 1`` for ``z = +inf`` (where
    ``floor(inf/_DLOG)`` overflows int32 to the negative clip bound).  ``NaN``
    still lands on ``1`` where ``searchsorted`` lands on ``n - 1``; every
    consumer folds the index into a weight that is ``NaN`` on BOTH paths, so
    the divergence is value-inert (see ``prior._grid_bracket``).
    """
    x = jnp.asarray(z)
    x = x.astype(jnp.result_type(x.dtype, zgrid.dtype))
    n = zgrid.shape[0]
    if _USE_SEARCHSORTED:
        return jnp.clip(jnp.searchsorted(zgrid, x, side="right"), 1, n - 1)
    i = jnp.clip(
        jnp.floor(jnp.log1p(jnp.maximum(x, 0.0)) / _DLOG).astype(jnp.int32) + 1,
        1, n - 1)
    i = i - ((i > 1) & (x < zgrid[i - 1]))
    i = i + ((i < n - 1) & (x >= zgrid[i]))
    return jnp.where(x >= zgrid[-1], n - 1, i)


def _interp_zgrid(z, log_grid):
    """``jnp.interp(z, zgrid, log_grid)`` with the node index in closed form.

    Bit-identical to ``jnp.interp`` pointwise: the index is the same one
    ``searchsorted(zgrid, z, side='right')`` returns (the one-step correction
    below absorbs both the last-ulp disagreement between libm's ``log1p`` and
    the tabulated node, and the pinned last node above), and the
    interpolation, the ``dx0`` guard and the out-of-range fills are the same
    expressions jax 0.4.34's ``_interp`` uses.
    """
    x = jnp.asarray(z)
    x = x.astype(jnp.result_type(x.dtype, zgrid.dtype))
    fp = jnp.asarray(log_grid)
    i = zgrid_upper_index(x)
    df = fp[i] - fp[i - 1]
    dx = zgrid[i] - zgrid[i - 1]
    delta = x - zgrid[i - 1]
    dx0 = jnp.abs(dx) <= _INTERP_DX0_EPS
    f = jnp.where(dx0, fp[i - 1],
                  fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df)
    f = jnp.where(x < zgrid[0], fp[0], f)
    return jnp.where(x > zgrid[-1], fp[-1], f)


def log_interp_zgrid(z, log_grid):
    """Interpolate a LOG-valued grid on ``zgrid``, with the z -> 0 power law.

    ``zgrid[0]`` is exactly 0, where ``dV_c/dz`` vanishes, so node 0 of any log
    grid built from the volume element is a FLOOR SENTINEL (``log(tiny)`` =
    -708, ``log(1e-300)`` = -690.8) rather than a value.  Linear interpolation of
    the LOG array across the first cell then ramps ~300 decades from that
    sentinel up to the physical value at ``zgrid[1]``: at z = 9e-4 it
    underestimates the density by e^-339, i.e. deletes it (a nearby galaxy's host
    weight, or a PE sample's volume prior).  Since ``dV_c/dz ∝ z^2`` as z -> 0,
    take the power law off node 1 below the first node -- exact to O(z) there,
    and the ``maximum(z, tiny)`` keeps the value and its gradient finite at
    z = 0, which is what the sentinel was introduced for.
    """
    z1 = zgrid[1]
    below = log_grid[1] + 2.0 * jnp.log(
        jnp.maximum(z, jnp.finfo(zgrid.dtype).tiny) / z1)
    above = (jnp.interp(z, zgrid, log_grid) if _USE_SEARCHSORTED
             else _interp_zgrid(z, log_grid))
    return jnp.where(z < z1, below, above)
