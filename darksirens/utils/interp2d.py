from typing import Iterable, Optional, Sequence

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

__all__ = ["interp2d", "interpnd", "interpnd_scalar_head", "CartesianGrid"]
Array = jnp.ndarray


def interp2d(
    x: Array,
    y: Array,
    xp: Array,
    yp: Array,
    zp: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.

    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`

    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z


def _axis_weights(coord: Array, grid: Array):
    """Bracketing indices, interpolation weight and out-of-range mask for one axis.

    Shared by :func:`interpnd` and :func:`interpnd_scalar_head` so both paths
    handle NaN and out-of-range coordinates identically.

    Sanitize BEFORE computing weights: out-of-range (or NaN) queries are
    answered by fill_value anyway, but if the weight itself is computed from
    the raw coordinate the backward pass multiplies cotangents by non-finite
    intermediates (mul's VJP scales by the stored operand), so a single
    NaN/inf here poisons the WHOLE gradient even though the forward value is
    correctly masked. Clamp for the arithmetic; keep the mask from the raw
    coordinate.
    """
    bad = jnp.isnan(coord)
    coord_c = jnp.where(bad, grid[0], jnp.clip(coord, grid[0], grid[-1]))
    upper = jnp.clip(jnp.searchsorted(grid, coord_c, side="right"), 1, len(grid) - 1)
    lower = upper - 1
    denom = grid[upper] - grid[lower]
    weight = (coord_c - grid[lower]) / denom
    oob = bad | (coord < grid[0]) | (coord > grid[-1])
    return lower, upper, weight, oob


def _validate_grid(coords, points, values):
    if len(coords) != len(points):
        raise ValueError("coords and points must have the same length")
    if values.ndim != len(points):
        raise ValueError("values must have one dimension per interpolation axis")
    for axis, grid in enumerate(points):
        if grid.ndim != 1:
            raise ValueError("all interpolation point arrays must be 1D")
        if values.shape[axis] != grid.shape[0]:
            raise ValueError("values shape must match the interpolation grids")


def interpnd(
    coords: Sequence[Array],
    points: Sequence[Array],
    values: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """Multilinear interpolation on a rectilinear grid.

    Unlike :class:`CartesianGrid`, each axis may be irregularly spaced.  The
    coordinate arrays are broadcast together, so callers can mix scalar model
    parameters with vector-valued coordinates.  When ``fill_value`` is provided,
    any point outside at least one grid axis is filled instead of extrapolated.

    When every coordinate except the last is a scalar,
    :func:`interpnd_scalar_head` computes the same thing with far less memory
    traffic — see its docstring.
    """
    _validate_grid(coords, points, values)

    coords = jnp.broadcast_arrays(*coords)
    lower_indices = []
    upper_indices = []
    weights = []
    oob = jnp.zeros(coords[0].shape, dtype=bool)

    for coord, grid in zip(coords, points):
        lower, upper, weight, axis_oob = _axis_weights(coord, grid)
        lower_indices.append(lower)
        upper_indices.append(upper)
        weights.append(weight)
        oob = oob | axis_oob

    interpolated = jnp.zeros(coords[0].shape, dtype=values.dtype)
    for corner in range(1 << len(points)):
        corner_indices = []
        corner_weight = jnp.ones(coords[0].shape, dtype=values.dtype)
        for axis in range(len(points)):
            if corner & (1 << axis):
                corner_indices.append(upper_indices[axis])
                corner_weight = corner_weight * weights[axis]
            else:
                corner_indices.append(lower_indices[axis])
                corner_weight = corner_weight * (1.0 - weights[axis])
        interpolated = interpolated + corner_weight * values[tuple(corner_indices)]

    if fill_value is not None:
        interpolated = jnp.where(oob, fill_value, interpolated)

    return interpolated


def interpnd_scalar_head(
    coords: Sequence[Array],
    points: Sequence[Array],
    values: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """:func:`interpnd` for the case where all axes but the LAST are scalars.

    Multilinear interpolation is a tensor product, so the axes may be contracted
    in any order.  :func:`interpnd` broadcasts every coordinate to the query
    shape and gathers ``2**ndim`` corners out of the full table — for a query of
    N points that is ``2**ndim`` gathers of N elements each, scattered across
    the whole (potentially very large) ``values`` array.

    When the leading coordinates are scalars, contracting THEM first collapses
    ``values`` to a single 1-D curve over the last axis, and only that curve is
    indexed per query point.  For the cosmology distance table this replaces 16
    gathers into a 54.7 MB array with 8 slices of a fixed row plus 2 gathers
    into a 4 KB curve.  Measured on an H100 NVL (kernel-time dominated): 0.95x
    at N=1e5, 1.64x at N=1e6, 1.95x at N=4e6, and 1.53x for the inference-shaped
    vmap of 512 cosmologies x 2000 redshifts.  The win needs enough query points
    to amortise building the curve — below ~1e5 it is a wash, which is why the
    fallback path is kept rather than removed.

    The result is the same number up to floating-point reassociation (the corner
    sum is grouped differently); agreement is ~1e-15 relative in float64. NaN and
    out-of-range handling is identical because both paths share
    :func:`_axis_weights`.

    Raises ``ValueError`` if any leading coordinate is not a scalar — callers
    that cannot guarantee that should use :func:`interpnd`.
    """
    _validate_grid(coords, points, values)
    if len(coords) < 2:
        raise ValueError("interpnd_scalar_head needs at least two axes")

    head, tail = coords[:-1], coords[-1]
    head_grids, tail_grid = points[:-1], points[-1]
    for axis, coord in enumerate(head):
        if jnp.ndim(coord) != 0:
            raise ValueError(
                f"interpnd_scalar_head requires scalar leading coordinates; "
                f"axis {axis} has ndim {jnp.ndim(coord)}. Use interpnd instead."
            )

    n_head = len(head)
    lowers, uppers, weights = [], [], []
    head_oob = jnp.asarray(False)
    for coord, grid in zip(head, head_grids):
        lower, upper, weight, axis_oob = _axis_weights(coord, grid)
        lowers.append(lower)
        uppers.append(upper)
        weights.append(weight)
        head_oob = head_oob | axis_oob

    # Contract the scalar axes into a 1-D curve over the trailing axis.
    curve = jnp.zeros(values.shape[-1], dtype=values.dtype)
    for corner in range(1 << n_head):
        corner_indices = []
        corner_weight = jnp.asarray(1.0, dtype=values.dtype)
        for axis in range(n_head):
            if corner & (1 << axis):
                corner_indices.append(uppers[axis])
                corner_weight = corner_weight * weights[axis]
            else:
                corner_indices.append(lowers[axis])
                corner_weight = corner_weight * (1.0 - weights[axis])
        curve = curve + corner_weight * values[tuple(corner_indices)]

    lower, upper, weight, tail_oob = _axis_weights(tail, tail_grid)
    interpolated = curve[lower] * (1.0 - weight) + curve[upper] * weight

    if fill_value is not None:
        interpolated = jnp.where(head_oob | tail_oob, fill_value, interpolated)

    return interpolated


class CartesianGrid:
    """
    Linear Multivariate Cartesian Grid interpolation in arbitrary dimensions. Based
    on ``map_coordinates``.

    Notes:
        Translated directly from https://github.com/JohannesBuchner/regulargrid/ to jax.
    """

    values: Array
    """
    Values to interpolate.
    """
    limits: Iterable[Iterable[float]]
    """
    Limits along each dimension of ``values``.
    """

    def __init__(
        self,
        limits: Iterable[Iterable[float]],
        values: Array,
        mode: str = "constant",
        cval: float = jnp.nan,
    ):
        """
        Initializer.

        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        super().__init__()
        self.values = values
        self.limits = limits
        self.mode = mode
        self.cval = cval

    def __call__(self, *coords) -> Array:
        """
        Perform interpolation.

        Args:
            coords: point at which to interpolate. These will be broadcasted if
                they are not the same shape.

        Returns:
            Interpolated values, with extrapolation handled according to ``mode``.
        """
        # transform coords into pixel values
        coords = jnp.broadcast_arrays(*coords)
        # coords = jnp.asarray(coords)
        coords = [
            (c - lo) * (n - 1) / (hi - lo)
            for (lo, hi), c, n in zip(self.limits, coords, self.values.shape)
        ]
        return map_coordinates(
            self.values, coords, mode=self.mode, cval=self.cval, order=1
        )
