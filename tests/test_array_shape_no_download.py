"""``array_shape`` reads metadata; the converted call sites never convert.

The startup path asks several full-sky galaxy tables only for a static
dimension.  Doing that with ``np.asarray(x).shape`` runs the whole table
through numpy -- on a GPU-resident ``jax.Array`` that is a real
device->host copy of 676 MB per table on the production DESI nside-64
catalog.  ``array_shape`` answers from ``.shape`` instead.

Pinning that is backend-sensitive, so this file pins it three ways and only
ONE of them depends on the backend:

* ``_Shim`` -- an array-like that records every ``__array__`` call.  Works
  identically on CPU and GPU and fails against the pre-PR
  ``np.asarray(a).shape`` body.
* call-site probes -- ``np.asarray`` is wrapped for the duration of the call
  and asserted never to be handed the big table.  Also backend-independent.
* ``_npy_value`` -- the real jax host-copy cache.  ``np.asarray`` on a
  jax.Array populates it on GPU but NOT on CPU (a CPU array is already host
  resident, so numpy takes the buffer protocol and never enters
  ``ArrayImpl.__array__``), so this one is skipped on the CPU backend and is
  the reason the other two exist.

Plus a source pin on the converted expressions themselves, so a revert at any
one site is caught even where no cheap behavioural probe exists.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.utils.utils import array_shape


class _Shim:
    """Array-like exposing ``.shape``; counts host conversions."""

    def __init__(self, shape, fill=0.0):
        self.shape = tuple(shape)
        self.n_array_calls = 0
        self._fill = fill

    def __array__(self, dtype=None, copy=None):
        self.n_array_calls += 1
        return np.full(self.shape, self._fill, dtype=dtype or np.float64)


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


def test_shape_is_read_without_any_host_conversion():
    """The property the helper exists for, on every backend.

    Reverting ``array_shape`` to ``np.asarray(a).shape`` makes this fail:
    ``__array__`` is called once instead of never.
    """
    shim = _Shim((4096, 2158))
    assert array_shape(shim) == (4096, 2158)
    assert shim.n_array_calls == 0, (
        "array_shape must read .shape, not convert the array"
    )


def test_shape_entries_are_plain_ints():
    """Callers index static shapes; a numpy/weak-typed dim would leak."""
    for d in array_shape(jnp.zeros((5, 3))):
        assert type(d) is int


def test_falls_back_for_a_shapeless_sequence():
    assert array_shape([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) == (3, 2)


def test_a_traced_array_returns_its_abstract_shape():
    """Documented departure from ``np.asarray``, which raises on a tracer.

    Call sites that used the raise as their traced-array detector must keep
    their own check; this pins the behaviour so that contract stays visible.
    """
    seen = {}

    @jax.jit
    def f(x):
        seen["shape"] = array_shape(x)
        return x.sum()

    f(jnp.zeros((7, 3)))
    assert seen["shape"] == (7, 3)


@pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="np.asarray on a CPU jax.Array takes the buffer protocol and never "
           "populates _npy_value, so this probe is vacuous on CPU; the _Shim "
           "and call-site tests above cover the property backend-independently",
)
def test_does_not_materialise_a_jax_array_on_the_host():
    a = jnp.arange(1024, dtype=jnp.float64).reshape(32, 32)
    assert getattr(a, "_npy_value", None) is None
    assert array_shape(a) == (32, 32)
    assert getattr(a, "_npy_value", None) is None, (
        "array_shape must not trigger a device->host transfer"
    )


class _AsarrayWatch:
    """Wrap ``np.asarray`` and record whether ``target`` was ever converted."""

    def __init__(self, monkeypatch, target):
        self.target = target
        self.hits = 0
        real = np.asarray

        def wrapper(a, *args, **kwargs):
            if a is self.target:
                self.hits += 1
            return real(a, *args, **kwargs)

        monkeypatch.setattr(np, "asarray", wrapper)


def test_build_field_normalization_inputs_never_converts_full_z(monkeypatch):
    """The site that pays the biggest table: only its row count is wanted.

    ``full_z`` is handed straight to ``jnp.asarray`` afterwards, so converting
    it here would also force a second device copy on the way back.
    """
    from darksirens.redshift import completion

    full_z = jnp.asarray(
        np.linspace(0.01, 0.5, 6 * 4, dtype=np.float64).reshape(6, 4)
    )
    full_n = np.array([4, 0, 2, 0, 4, 1], dtype=np.int64)

    watch = _AsarrayWatch(monkeypatch, full_z)
    out = completion.build_field_normalization_inputs(full_z, None, full_n)

    assert watch.hits == 0, "full_z must not be converted for its row count"
    assert out.n_empty == 2
    assert out.N_obs_total == 11.0
    assert out.occupied_pixels.tolist() == [0, 2, 4, 5]


def test_spread_probe_rows_never_converts_zgals(monkeypatch):
    from darksirens.redshift import catalog as rc

    zgals = jnp.asarray(np.zeros((8, 3), dtype=np.float64))
    ngals = np.array([3, 0, 3, 3, 0, 3, 3, 3], dtype=np.int64)

    watch = _AsarrayWatch(monkeypatch, zgals)
    rows = rc._spread_probe_rows(zgals, ngals, 4)

    assert watch.hits == 0, "zgals must not be converted for its row count"
    assert np.asarray(rows).tolist() == [0, 3, 5, 7]


# (file, exact source snippet) for every site converted away from
# ``np.asarray(...).shape``.  A revert at any one of them fails here even
# where no cheap behavioural probe exists.
_CONVERTED_SITES = [
    ("darksirens/redshift/completion.py", "n_pix_total = int(array_shape(full_z)[0])"),
    ("darksirens/redshift/completion.py",
     "n_gal = int(array_shape(em_catalog.field_depth_z)[0])"),
    ("darksirens/redshift/completion.py",
     "np.arange(array_shape(em_catalog.zgals)[0], dtype=np.int32)"),
    ("darksirens/redshift/catalog.py", "n_rows = int(array_shape(zgals)[0])"),
    ("darksirens/likelihood/factory.py", "int(array_shape(c.zgals)[1])"),
    ("darksirens/likelihood/catalog_views.py",
     "array_shape(full_z)[0] if full_z is not None else 0"),
    ("darksirens/likelihood/catalog_views.py",
     'data.get("n_pix_catalog", array_shape(full_z)[0])'),
    ("darksirens/likelihood/catalog_views.py",
     "n_union_rows = int(array_shape(pe_view.zgals)[0])"),
    ("darksirens/likelihood/catalog_views.py",
     "n_pe_rows = int(array_shape(pe_view.zgals)[0])"),
    ("darksirens/likelihood/catalog_views.py",
     "n_sel_rows = int(array_shape(sel_view.zgals)[0])"),
]

_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relpath,snippet", _CONVERTED_SITES)
def test_converted_site_still_reads_the_shape(relpath, snippet):
    src = (_ROOT / relpath).read_text()
    assert snippet in src, (
        f"{relpath} no longer reads this shape through array_shape; "
        "reverting to np.asarray(...).shape reintroduces a full host copy "
        "of a full-sky galaxy table"
    )
