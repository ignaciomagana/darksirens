"""The Q_LSS table / ensemble must be COMPACTED BEFORE it reaches the device.

``catalog_views`` deliberately carries the global ``(n_pix, n_grid)`` log-Q table
and the ``(M, n_pix, n_grid)`` member ensemble HOST-side and unbarriered --
"likelihood.py slices it to the per-view union pixels so only the compact block
reaches the device".  The factory used to do the opposite: ``jnp.asarray(full)``
first, ``full_j[:, up]`` afterwards, which transfers the whole table and then holds
it live alongside the compact slice.

The win is PEAK DEVICE MEMORY and nothing else -- wall time gets WORSE.  Measured
at the DESI ensemble shape (M=8, n_pix=49152, N_grid=1086 f64, half the pixels in
the union; jit warm, host pages pre-touched, fresh process per arm): H100 NVL
0.31 s at 6.833 GB peak the old way against 0.58 s at 3.416 GB with the gather in
numpy first, and on CPU 0.32 s -> 0.74 s.  So: half the device peak, ~2x the wall
time of a step that runs once per likelihood build, and ~1.7 GB of host memory
moved in as the device memory moved out (numpy holds full + compact at once).

A table the caller already put on the DEVICE
(``darksirens/cli/diagnose_lognormal_completion.py``) must keep the on-device
gather -- pulling it back to the host to slice would be strictly worse -- so the
seam is residency-guarded, not unconditional.

The two-sided bounds check is load-bearing rather than cosmetic: JAX silently
CLAMPS an out-of-range gather index while numpy WRAPS a negative one, so a corrupt
pixel array would mean different things on the two backends.
"""
import ast
import inspect

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import darksirens.likelihood.factory as factory
from darksirens.likelihood.factory import _gather_pixel_rows

NPIX, NG, NMEM = 16, 5, 3
UP = np.array([1, 4, 4, 9, 15], dtype=np.int64)   # duplicates are legal


def _table():
    rng = np.random.default_rng(0)
    return rng.normal(size=(NPIX, NG))


def _ensemble():
    rng = np.random.default_rng(1)
    return rng.normal(size=(NMEM, NPIX, NG))


def _q(full, up=UP):
    return _gather_pixel_rows(full, up, what="table", unit="rows")


def _qm(full, up=UP):
    return _gather_pixel_rows(full, up, what="ensemble", unit="pixels")


# ── the transfer seam ────────────────────────────────────────────────────────

@pytest.mark.parametrize("make,gather", [(_table, _q), (_ensemble, _qm)])
def test_host_table_is_gathered_before_it_reaches_the_device(make, gather):
    """A HOST table must come back as a HOST array: the full table is never a
    device operand, only the compact block the caller then barriers."""
    out = gather(make())
    assert isinstance(out, np.ndarray)
    assert not hasattr(out, "devices")             # the duck-typed device marker
    assert out.shape[-2] == UP.size


@pytest.mark.parametrize("make,gather", [(_table, _q), (_ensemble, _qm)])
def test_host_gather_is_bit_identical_to_the_old_device_gather(make, gather):
    """Numerics: a gather is value-preserving and commutes with the dtype cast."""
    full = make()
    old = np.asarray(jnp.asarray(full)[..., jnp.asarray(UP, dtype=jnp.int32), :])
    new = np.asarray(gather(full))
    assert new.dtype == old.dtype == np.float64
    assert np.array_equal(new, old)


@pytest.mark.parametrize("make,gather", [(_table, _q), (_ensemble, _qm)])
def test_device_table_keeps_the_on_device_gather(make, gather):
    """A caller who already paid for the transfer must NOT be round-tripped back
    to the host to slice (``cli/diagnose_lognormal_completion.py``)."""
    full = make()
    out = gather(jnp.asarray(full))
    assert hasattr(out, "devices")                 # still a jax.Array
    assert np.array_equal(np.asarray(out), np.asarray(_gather_pixel_rows(
        full, UP, what="table", unit="rows")))


# ── the two-sided bounds check ───────────────────────────────────────────────

@pytest.mark.parametrize("device", [False, True])
def test_out_of_range_pixel_index_is_refused(device):
    full = _table()
    up = np.array([0, NPIX], dtype=np.int64)
    with pytest.raises(ValueError, match=r"table has 16 rows .*reaches 16"):
        _q(jnp.asarray(full) if device else full, up)


@pytest.mark.parametrize("device", [False, True])
def test_negative_pixel_index_is_refused_not_silently_reinterpreted(device):
    """numpy WRAPS a negative index and JAX CLAMPS it, so the same corrupt pixel
    array would mean two different things once the gather moved to the host."""
    full = _ensemble()
    up = np.array([0, -1], dtype=np.int64)
    with pytest.raises(ValueError, match=r"ensemble has 16 pixels .*reaches -1"):
        _qm(jnp.asarray(full) if device else full, up)


# ── structural guard: every compaction helper routes through the seam ────────

def test_every_lss_compaction_helper_gathers_before_the_transfer():
    """There are FOUR of these helpers -- the flat pair and the per-catalog pair --
    and a new one is easy to write in the old order, which is invisible in every
    value test (the gather is bit-identical either way) and only shows up as
    device memory.  Pin it structurally instead.
    """
    src = inspect.getsource(factory)
    tree = ast.parse(src)
    helpers = {
        node.name: ast.get_source_segment(src, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_compact_lss_")
    }
    assert set(helpers) == {
        "_compact_lss_q", "_compact_lss_members",
        "_compact_lss_q_for", "_compact_lss_members_for",
    }, f"unexpected LSS compaction helpers: {sorted(helpers)}"
    for name, body in helpers.items():
        assert "_gather_pixel_rows(" in body, (
            f"{name} does not compact through _gather_pixel_rows"
        )
        assert "jnp.asarray(full)" not in body, (
            f"{name} transfers the FULL table before slicing it"
        )
