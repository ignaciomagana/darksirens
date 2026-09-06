import numpy as np
from jax import jit
from jax import numpy as jnp
from jax.scipy.special import logsumexp


def array_shape(a):
    """Shape of an array-like WITHOUT forcing a device->host copy.

    ``np.asarray(x).shape`` on a device-resident ``jax.Array`` materialises the
    whole table on the host (676 MB per full-sky galaxy table on the production
    DESI nside-64 catalog, ~0.17 s at 4 GB/s) purely to read a static integer
    that the ``jax.Array`` already carries.  Anything exposing ``.shape`` --
    ``jax.Array``, ``np.ndarray``, ``h5py`` datasets -- answers from metadata;
    only genuine sequences fall through to ``np.asarray``.  Exactly
    bit-identical: no value is read, and the shape is the same object either
    way.
    """
    shape = getattr(a, "shape", None)
    if shape is None:
        return np.asarray(a).shape
    return tuple(int(d) for d in shape)

@jit
def logdiffexp(x, y):
    """Stable log(exp(x) - exp(y)) for y <= x. If y > x, result is undefined; return -inf.

    CAUTION -- reverse mode: the forward value at ``y == x`` is correctly
    ``-inf``, but the derivative runs through ``log1p(-exp(0)) = log1p(-1)``
    whose slope is ``1/0``, so the cotangent is non-finite (NaN once it meets a
    zero upstream cotangent).  A downstream ``jnp.where`` cannot erase it: both
    branches of a ``where`` are differentiated.  Any estimator that can reach
    ``y == x`` (e.g. an exactly-zero Monte-Carlo variance from uniform
    importance weights) must therefore carry the ratio directly rather than
    differentiate this function -- see
    ``likelihood/selection.py:_lse_to_log_mu_neff``.
    """
    return jnp.where(y <= x, x + jnp.log1p(-jnp.exp(y - x)), -jnp.inf)


def logsumexp_neginf_safe(terms, axis=None):
    """logsumexp returning exactly -inf for an all--inf input WITHOUT the NaN
    backward pass of the plain reduction: softmax of all--inf is 0/0 = NaN,
    and that NaN survives multiplication by a ZERO upstream cotangent (mul's
    VJP scales by the stored NaN), poisoning every parameter's gradient.

    Inputs with any finite entry are bit-identical to plain ``logsumexp``: the
    sanitized padding's weight ``exp(-1e30 - max)`` underflows to exactly zero.
    Non-finite entries that are NOT ``-inf`` (NaN, ``+inf``) are left alone, so
    a genuinely poisoned weight still propagates and is caught by the callers'
    finiteness guards instead of being silently dropped.
    """
    neg_inf = jnp.isneginf(terms)
    safe = jnp.where(neg_inf, -1e30, terms)
    live = jnp.any(~neg_inf, axis=axis)
    return jnp.where(live, logsumexp(safe, axis=axis), -jnp.inf)
