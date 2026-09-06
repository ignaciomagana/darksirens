"""``array_shape`` reads metadata, never the values.

The startup path asks several full-sky galaxy tables only for a static
dimension.  Doing that with ``np.asarray(x).shape`` materialises the whole
device table on the host (676 MB per table on the production DESI nside-64
catalog); ``.shape`` answers from metadata.  These tests pin the behaviour
that makes the substitution safe -- same answer for every array-like -- and
the property that makes it faster: a ``jax.Array`` that has never been pulled
to the host still has no host copy afterwards.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from darksirens.utils.utils import array_shape


@pytest.mark.parametrize("factory", [
    lambda: np.zeros((5, 3)),
    lambda: jnp.zeros((5, 3)),
    lambda: [[0.0, 0.0, 0.0]] * 5,
    lambda: ((0.0, 0.0, 0.0),) * 5,
])
def test_matches_np_asarray_shape(factory):
    a = factory()
    assert array_shape(a) == np.asarray(a).shape == (5, 3)


def test_scalar_and_1d():
    assert array_shape(np.float64(1.0)) == ()
    assert array_shape(jnp.arange(7)) == (7,)
    assert array_shape(3.0) == ()


def test_does_not_materialise_a_jax_array_on_the_host():
    a = jnp.arange(1024, dtype=jnp.float64).reshape(32, 32)
    assert getattr(a, "_npy_value", None) is None
    assert array_shape(a) == (32, 32)
    assert getattr(a, "_npy_value", None) is None, (
        "array_shape must not trigger a device->host transfer"
    )
