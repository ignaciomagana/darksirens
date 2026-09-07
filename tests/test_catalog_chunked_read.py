"""Raw-chunk threaded reads of the survey catalog (``darksirens.catalogs.io``).

:func:`read_dataset_chunked` reimplements the (shuffle, deflate) HDF5 filter
pipeline so the inflate work can run off the GIL in a thread pool.  Every test
here pins the same contract: the array it returns is byte-for-byte the array
``np.asarray(dset)`` returns, on the fast path and on every fallback.

Byte identity ALONE does not pin that contract, because the fallback is itself
byte-identical: a fast path that raises on every chunk degrades silently to a
plain h5py read and every naive identity assertion still passes.  So each test
also states WHICH path ran, by spying on :func:`~darksirens.catalogs.io.
_read_dataset_h5py` -- the single funnel every refusal and every fallback
returns through.  Without that, a defect that disables the fast path
(mis-sized edge chunks, a deleted ``filter_mask`` guard, a broken gate) costs
the whole 2.2x with a green suite.
"""

import os
import threading
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import pytest

import darksirens.catalogs.io as catalogs_io
from darksirens.catalogs.io import (
    _unshuffle_into,
    chunked_read_admissible,
    chunked_read_enabled,
    load_survey,
    read_dataset_chunked,
    usable_cpu_count,
)


@pytest.fixture(autouse=True)
def _no_size_floor(monkeypatch):
    """Let the fixtures (kilobytes, not gigabytes) reach the threaded path.

    The production floor (64 MB) exists so tiny datasets skip the raw-chunk
    setup; the size test below pins it explicitly instead.
    """
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_MIN_BYTES", 0)


@pytest.fixture
def h5py_reads(monkeypatch):
    """Count trips through the plain-h5py funnel, i.e. fallbacks.

    Returns a zero-length list that grows by one dataset name per fallback,
    so a test can assert the fast path ran (empty) or that it did not.
    """
    seen = []
    original = catalogs_io._read_dataset_h5py

    def spy(dset):
        seen.append(dset.name)
        return original(dset)

    monkeypatch.setattr(catalogs_io, "_read_dataset_h5py", spy)
    return seen


def _write(path, name, data, **kwargs):
    with h5py.File(path, "a") as f:
        f.create_dataset(name, data=data, **kwargs)


def _rng_table(shape, dtype=np.float64, seed=7):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(0, 10_000, size=shape).astype(dtype)


def _read_both_ways(path, name, workers=4):
    with h5py.File(path, "r") as f:
        ref = np.asarray(f[name])
        got = read_dataset_chunked(f[name], workers)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert got.tobytes() == ref.tobytes()
    return ref, got


def _assert_reads_identically(path, name, h5py_reads, workers=4):
    """Byte-identical AND decoded by the raw-chunk path, not by h5py."""
    ref, got = _read_both_ways(path, name, workers=workers)
    assert h5py_reads == [], f"fell back to h5py instead of decoding: {h5py_reads}"
    return ref, got


def _assert_falls_back_identically(path, name, h5py_reads, workers=4):
    """Byte-identical BECAUSE the raw-chunk path was refused/abandoned."""
    ref, got = _read_both_ways(path, name, workers=workers)
    assert h5py_reads, "expected a fallback to the plain h5py read, got none"
    return ref, got


# ---------------------------------------------------------------------------
# Bit identity on the fast path
# ---------------------------------------------------------------------------


def test_partial_edge_chunks_are_decompressed_and_sliced(tmp_path, h5py_reads):
    """Both axes end in a partial chunk, as the production catalog does.

    (100, 47) over (32, 9) chunks is 100 = 3*32 + 4 rows and 47 = 5*9 + 2
    columns, i.e. the same shape of edge case as the production catalog's
    1719 = 63*27 + 18: HDF5 still stores a FULL 32x9 chunk there, so the
    reader has to inflate all of it and drop the padding.  Sizing the
    inflate buffer to the visible edge region instead is the classic bug;
    it raises, and only the no-fallback assertion catches that.
    """
    path = tmp_path / "edge.h5"
    data = _rng_table((100, 47))
    _write(path, "x", data, chunks=(32, 9), compression="gzip", shuffle=True)
    ref, got = _assert_reads_identically(path, "x", h5py_reads)
    assert np.array_equal(got, data)
    assert np.array_equal(got, ref)


def test_exact_chunk_grid_and_one_dimensional(tmp_path, h5py_reads):
    path = tmp_path / "exact.h5"
    _write(
        path, "grid", _rng_table((64, 24)),
        chunks=(16, 8), compression="gzip", shuffle=True,
    )
    _write(
        path, "line", _rng_table((1000,), dtype=np.int64),
        chunks=(128,), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "grid", h5py_reads)
    _assert_reads_identically(path, "line", h5py_reads)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
def test_identity_across_itemsizes(tmp_path, h5py_reads, dtype):
    """The un-shuffle is an itemsize-wide byte transpose; pin 4- and 8-byte."""
    path = tmp_path / f"dt_{np.dtype(dtype).name}.h5"
    _write(
        path, "x", _rng_table((37, 29), dtype=dtype),
        chunks=(8, 7), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "x", h5py_reads)


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_identity_is_worker_count_independent(tmp_path, h5py_reads, workers):
    path = tmp_path / "w.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "x", h5py_reads, workers=workers)


class _CountingPool(ThreadPoolExecutor):
    """Executor that records the deepest submitted-but-unfinished queue.

    The calling thread reads raw chunk bytes about ten times faster than the
    pool inflates them, so without a bound the submit loop queues the whole
    COMPRESSED dataset; the sleep here makes that gap unmissable at test
    size.  ``peak`` is the largest in-flight count seen at submit time.
    """

    peak = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._submitted = 0
        self._finished = 0
        type(self).peak = 0

    def submit(self, fn, *args, **kwargs):
        def slow(*inner_args, **inner_kwargs):
            time.sleep(0.002)
            return fn(*inner_args, **inner_kwargs)

        future = super().submit(slow, *args, **kwargs)
        with self._lock:
            self._submitted += 1
            type(self).peak = max(type(self).peak, self._submitted - self._finished)

        def _done(_):
            with self._lock:
                self._finished += 1

        future.add_done_callback(_done)
        return future


def test_read_ahead_is_bounded_but_still_reads_every_chunk(
    tmp_path, monkeypatch, h5py_reads
):
    """More chunks than the in-flight cap, so the submit loop has to drain.

    (256, 12) over (8, 4) chunks is 32*3 = 96 chunks against a cap of
    ``_CHUNK_READ_QUEUE_DEPTH * workers`` = 2; the bound exists so the
    executor queue cannot grow to hold the whole compressed dataset, which
    is unbounded host memory on a large one.  Both halves are pinned: the
    queue stays shallow, and every chunk still lands.
    """
    path = tmp_path / "deep.h5"
    data = _rng_table((256, 12))
    _write(path, "x", data, chunks=(8, 4), compression="gzip", shuffle=True)
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_QUEUE_DEPTH", 1)
    monkeypatch.setattr(catalogs_io, "ThreadPoolExecutor", _CountingPool)
    ref, got = _assert_reads_identically(path, "x", h5py_reads, workers=2)
    n_chunks = 32 * 3
    assert _CountingPool.peak <= 2 * catalogs_io._CHUNK_READ_QUEUE_DEPTH * 2
    assert _CountingPool.peak < n_chunks // 4
    assert np.array_equal(got, data)
    assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# The filter pipeline: equality, not presence
# ---------------------------------------------------------------------------


def test_filter_pipeline_reports_ids_in_application_order(tmp_path):
    path = tmp_path / "pipe.h5"
    _write(
        path, "x", _rng_table((32, 8)),
        chunks=(8, 4), compression="gzip", shuffle=True,
    )
    with h5py.File(path, "r") as f:
        assert catalogs_io._filter_pipeline(f["x"]) == (
            h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE,
        )
        assert catalogs_io._CHUNK_READ_PIPELINE == (
            h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE,
        )


def _build_with_pipeline(path, shape, chunks, order, dtype=np.dtype("i4")):
    """Create a chunked dataset with an explicitly ordered filter pipeline.

    ``h5py.Dataset.create_dataset`` cannot express order or a third-party
    filter, so this goes through the low-level DCPL the way the file writer
    that produced such a catalog would have.
    """
    truth = np.random.default_rng(9).integers(
        -2**30, 2**30, size=shape
    ).astype(dtype)
    with h5py.File(path, "w") as f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk(chunks)
        for step in order:
            if step == "shuffle":
                dcpl.set_shuffle()
            elif step == "deflate":
                dcpl.set_deflate(4)
            elif step == "nbit":
                dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0)
        assert dcpl.get_nfilters() == len(order)
        sid = h5py.h5s.create_simple(shape)
        tid = h5py.h5t.py_create(dtype, logical=True)
        dset = h5py.Dataset(h5py.h5d.create(f.id, b"x", tid, sid, dcpl=dcpl))
        dset[...] = truth
    return truth


def test_third_filter_in_the_pipeline_is_refused(tmp_path, h5py_reads):
    """(shuffle, nbit, deflate) passes every PRESENCE check and decodes wrong.

    h5py reports compression == 'gzip', shuffle is True, fletcher32 is False
    and scaleoffset is None -- the four properties the gate used to consult
    -- so the whole presence-based gate admits it.  Nothing downstream saves
    it either: every chunk's filter_mask is 0, and the inflated buffer is
    exactly one chunk of bytes, so neither the mask guard nor the size guard
    in :func:`_unshuffle_into` fires and no exception is raised.  Refusing on
    pipeline EQUALITY is the only thing that makes the byte-identity claim
    true, here and for any third-party filter id this module never heard of.

    No byte comparison: h5py's OWN repeated reads of this dataset disagree
    with each other (nbit at default precision does not round-trip an i4),
    which is itself the reason a two-filter reader must not touch it.
    """
    path = tmp_path / "nbit.h5"
    _build_with_pipeline(path, (60, 29), (16, 9), ["shuffle", "nbit", "deflate"])
    with h5py.File(path, "r") as f:
        dset = f["x"]
        assert dset.compression == "gzip"
        assert bool(dset.shuffle) is True
        assert bool(dset.fletcher32) is False
        assert dset.scaleoffset is None
        assert catalogs_io._filter_pipeline(dset) == (
            h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_NBIT,
            h5py.h5z.FILTER_DEFLATE,
        )
        assert chunked_read_admissible(dset) is False
        # Nothing below the gate would have objected: mask clear, and the
        # inflated chunk is exactly the right byte count to un-shuffle.
        filter_mask, raw = dset.id.read_direct_chunk((0, 0))
        assert filter_mask == 0
        _unshuffle_into(np.empty((16, 9), dtype=dset.dtype), zlib.decompress(raw))
        read_dataset_chunked(dset, 4)
    assert h5py_reads == ["/x"]


def test_reversed_pipeline_order_is_refused(tmp_path, h5py_reads):
    """(deflate, shuffle) is not (shuffle, deflate) and must be refused here.

    The presence properties cannot tell the two apart at all; only the
    ordered pipeline can.
    """
    path = tmp_path / "rev.h5"
    _build_with_pipeline(path, (60, 29), (16, 9), ["deflate", "shuffle"])
    with h5py.File(path, "r") as f:
        dset = f["x"]
        assert dset.compression == "gzip" and bool(dset.shuffle) is True
        assert catalogs_io._filter_pipeline(dset) == (
            h5py.h5z.FILTER_DEFLATE, h5py.h5z.FILTER_SHUFFLE,
        )
        assert chunked_read_admissible(dset) is False
    _assert_falls_back_identically(path, "x", h5py_reads)


# ---------------------------------------------------------------------------
# The admissibility predicate, and every fallback it gates
# ---------------------------------------------------------------------------


def test_admissible_only_for_chunked_shuffled_gzip(tmp_path):
    path = tmp_path / "adm.h5"
    table = _rng_table((40, 17))
    _write(path, "ok", table, chunks=(8, 5), compression="gzip", shuffle=True)
    _write(path, "contiguous", table)
    _write(path, "no_shuffle", table, chunks=(8, 5), compression="gzip")
    _write(path, "uncompressed", table, chunks=(8, 5), shuffle=True)
    _write(path, "lzf", table, chunks=(8, 5), compression="lzf", shuffle=True)
    _write(
        path, "checksummed", table,
        chunks=(8, 5), compression="gzip", shuffle=True, fletcher32=True,
    )
    _write(
        path, "scaled", table,
        chunks=(8, 5), compression="gzip", shuffle=True, scaleoffset=4,
    )
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["ok"]) is True
        for name in (
            "contiguous", "no_shuffle", "uncompressed", "lzf", "checksummed",
            "scaled",
        ):
            assert chunked_read_admissible(f[name]) is False, name


def test_non_numeric_dtype_is_refused(tmp_path, h5py_reads):
    """A compound chunk is not a dense block of itemsize-byte elements."""
    path = tmp_path / "compound.h5"
    dtype = np.dtype([("z", "f8"), ("tag", "i4")])
    table = np.zeros(64, dtype=dtype)
    table["z"] = np.linspace(0.0, 1.0, 64)
    table["tag"] = np.arange(64)
    _write(path, "x", table, chunks=(16,), compression="gzip", shuffle=True)
    with h5py.File(path, "r") as f:
        assert f["x"].dtype.kind == "V"
        assert chunked_read_admissible(f["x"]) is False
    _assert_falls_back_identically(path, "x", h5py_reads)


def test_virtual_dataset_is_refused(tmp_path, h5py_reads):
    """A VDS's chunks live in other files; h5py reports chunks is None.

    The gate's explicit ``is_virtual`` term is therefore redundant with the
    ``chunks is None`` term beside it and cannot be pinned on its own -- it
    is kept as belt-and-braces that documents intent.  What this test pins
    is the refusal itself, whichever term delivers it.
    """
    source = tmp_path / "src.h5"
    table = _rng_table((40, 17))
    _write(source, "x", table, chunks=(8, 5), compression="gzip", shuffle=True)
    path = tmp_path / "vds.h5"
    layout = h5py.VirtualLayout(shape=table.shape, dtype=table.dtype)
    layout[...] = h5py.VirtualSource(str(source), "x", shape=table.shape)
    with h5py.File(path, "w") as f:
        f.create_virtual_dataset("x", layout)
    with h5py.File(path, "r") as f:
        assert bool(f["x"].is_virtual) is True
        assert chunked_read_admissible(f["x"]) is False
    _assert_falls_back_identically(path, "x", h5py_reads)


def test_size_floor_refuses_small_datasets(tmp_path, monkeypatch, h5py_reads):
    path = tmp_path / "small.h5"
    _write(
        path, "x", _rng_table((40, 17)),
        chunks=(8, 5), compression="gzip", shuffle=True,
    )
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_MIN_BYTES", 1 << 30)
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["x"]) is False
    _assert_falls_back_identically(path, "x", h5py_reads)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},                                                    # contiguous
        {"chunks": (8, 5), "compression": "gzip"},              # no shuffle
        {"chunks": (8, 5), "shuffle": True},                    # no deflate
        {"chunks": (8, 5), "compression": "lzf", "shuffle": True},
        {
            "chunks": (8, 5), "compression": "gzip", "shuffle": True,
            "fletcher32": True,
        },
        {
            "chunks": (8, 5), "compression": "gzip", "shuffle": True,
            "scaleoffset": 4,
        },
    ],
)
def test_refused_pipelines_fall_back_to_h5py(tmp_path, h5py_reads, kwargs):
    path = tmp_path / "fb.h5"
    _write(path, "x", _rng_table((40, 17)), **kwargs)
    _assert_falls_back_identically(path, "x", h5py_reads)


# ---------------------------------------------------------------------------
# Worker floor and the escape hatch
# ---------------------------------------------------------------------------


def test_fewer_than_two_workers_refuses_the_threaded_path(tmp_path, h5py_reads):
    """One worker is 1.4x SLOWER than the h5py read, so it is refused.

    With no inflate to overlap against, the un-shuffle and the assemble copy
    are pure added Python: measured 6.41 s against 4.53 s on the production
    catalog.  The floor is a correctness-neutral performance gate, so it
    must actually fire.
    """
    path = tmp_path / "solo.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["x"]) is True
    _assert_falls_back_identically(path, "x", h5py_reads, workers=1)
    assert catalogs_io._CHUNK_READ_MIN_WORKERS == 2


def test_default_workers_follow_the_cpuset_not_the_host(tmp_path, monkeypatch):
    """``os.cpu_count()`` ignores a cpuset; ``sched_getaffinity`` does not.

    A one-CPU allocation must end up on the h5py path, and a wide one must
    cap at :data:`_CHUNK_READ_WORKERS` rather than at the host core count.
    """
    seen = []
    real = catalogs_io.read_dataset_chunked

    def record(dset, workers=None):
        if workers is None:
            workers = min(
                catalogs_io._CHUNK_READ_WORKERS, catalogs_io.usable_cpu_count()
            )
        seen.append(workers)
        return real(dset, workers)

    path = tmp_path / "cpuset.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    monkeypatch.setattr(catalogs_io, "usable_cpu_count", lambda: 1)
    with h5py.File(path, "r") as f:
        assert record(f["x"]).tobytes() == np.asarray(f["x"]).tobytes()
    monkeypatch.setattr(catalogs_io, "usable_cpu_count", lambda: 256)
    with h5py.File(path, "r") as f:
        assert record(f["x"]).tobytes() == np.asarray(f["x"]).tobytes()
    assert seen == [1, catalogs_io._CHUNK_READ_WORKERS]
    assert usable_cpu_count() >= 1


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="no CPU affinity control"
)
def test_usable_cpu_count_reads_the_affinity_mask(tmp_path, h5py_reads):
    """The real function, against a real cpuset -- not a monkeypatched stub.

    ``os.cpu_count()`` reports the host's cores whatever mask the process
    carries, so on a one-CPU allocation it hands four workers to a read that
    is 1.4x slower with them.  Narrow this process's own affinity and the
    threaded path must disappear.
    """
    original = os.sched_getaffinity(0)
    if len(original) < 2:  # pragma: no cover - already pinned
        pytest.skip("process is already pinned to a single CPU")
    path = tmp_path / "pinned.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    one = {sorted(original)[0]}
    try:
        os.sched_setaffinity(0, one)
        assert usable_cpu_count() == 1
        assert os.cpu_count() > 1  # the number the old code used
        with h5py.File(path, "r") as f:
            assert chunked_read_admissible(f["x"]) is True
            ref = np.asarray(f["x"])
            got = read_dataset_chunked(f["x"])
    finally:
        os.sched_setaffinity(0, original)
    assert h5py_reads == ["/x"]
    assert got.tobytes() == ref.tobytes()
    assert usable_cpu_count() == len(original)


def test_env_switch_forces_the_plain_h5py_read(tmp_path, monkeypatch, h5py_reads):
    path = tmp_path / "hatch.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    monkeypatch.setenv(catalogs_io._CHUNK_READ_ENV, "0")
    assert chunked_read_enabled() is False
    _assert_falls_back_identically(path, "x", h5py_reads)
    monkeypatch.setenv(catalogs_io._CHUNK_READ_ENV, "1")
    assert chunked_read_enabled() is True
    del h5py_reads[:]
    _assert_reads_identically(path, "x", h5py_reads)


# ---------------------------------------------------------------------------
# Fallbacks taken after the gate admitted the dataset -- these warn
# ---------------------------------------------------------------------------


def test_chunk_with_filters_skipped_falls_back(tmp_path, h5py_reads):
    """A non-zero ``filter_mask`` means the stored bytes are not pipeline output.

    HDF5 sets it when it stored a chunk with part of the pipeline skipped, so
    inflating that chunk would be wrong; the reader must abandon the raw path
    for the whole dataset and let h5py do it.  The warning is matched on the
    guard's own message, so deleting the guard and letting zlib fail instead
    is a different (and, for a mask that skipped only SOME filters, wrong)
    outcome and fails here.
    """
    path = tmp_path / "mask.h5"
    data = _rng_table((32, 10))
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "x", data=data, chunks=(16, 5), compression="gzip", shuffle=True,
        )
        raw = np.ascontiguousarray(data[:16, :5]).tobytes()
        dset.id.write_direct_chunk((0, 0), raw, filter_mask=0xFFFFFFFF)
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["x"]) is True
        ref = np.asarray(f["x"])
        with pytest.warns(RuntimeWarning, match="filters skipped"):
            got = read_dataset_chunked(f["x"], 4)
    assert h5py_reads == ["/x"]
    assert got.tobytes() == ref.tobytes()
    assert np.array_equal(got[:16, :5], data[:16, :5])


@pytest.mark.parametrize("written", [0, 1])
def test_partially_allocated_dataset_is_refused_by_the_gate(
    tmp_path, h5py_reads, written
):
    """A chunk that was never written has no stored bytes to read.

    ``read_direct_chunk`` on one does not fail in any ONE way -- h5py sizes
    its destination buffer from a storage size HDF5 never set, so the same
    unallocated chunk raises ``OSError``, ``MemoryError`` or ``SystemError``
    from run to run (all three observed on this h5py 3.12.1 / HDF5 1.14.4).
    Gating on the chunk count makes the refusal deterministic and silent,
    and is what lets the reader's own ``except`` stay narrow.
    """
    path = tmp_path / f"sparse_{written}.h5"
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "x", shape=(32, 10), dtype=np.float64,
            chunks=(16, 5), compression="gzip", shuffle=True, fillvalue=1.25,
        )
        if written:
            dset[:16, :5] = 2.5
    with h5py.File(path, "r") as f:
        assert catalogs_io._n_chunks(f["x"]) == 4
        assert f["x"].id.get_num_chunks() == written
        assert catalogs_io._fully_allocated(f["x"]) is False
        assert chunked_read_admissible(f["x"]) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a gate refusal must not warn
        _, got = _assert_falls_back_identically(path, "x", h5py_reads)
    assert np.all(got[16:, :] == 1.25)


def test_fully_allocated_is_true_for_a_written_dataset(tmp_path):
    path = tmp_path / "dense.h5"
    _write(
        path, "x", _rng_table((32, 10)),
        chunks=(16, 5), compression="gzip", shuffle=True,
    )
    with h5py.File(path, "r") as f:
        assert f["x"].id.get_num_chunks() == catalogs_io._n_chunks(f["x"]) == 4
        assert catalogs_io._fully_allocated(f["x"]) is True


# ---------------------------------------------------------------------------
# The shuffle inverse itself
# ---------------------------------------------------------------------------


def test_unshuffle_inverts_the_hdf5_filter(tmp_path):
    """Against HDF5's own shuffle output, not against a reimplementation."""
    path = tmp_path / "sh.h5"
    data = _rng_table((16, 5))
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "x", data=data, chunks=(16, 5), compression="gzip", shuffle=True,
        )
        filter_mask, raw = dset.id.read_direct_chunk((0, 0))
    assert filter_mask == 0
    out = np.empty((16, 5), dtype=np.float64)
    _unshuffle_into(out, zlib.decompress(raw))
    assert out.tobytes() == data.tobytes()


@pytest.mark.parametrize("dtype", ["u1", "i4", "f8"])
def test_unshuffle_destination_is_always_whole_elements(dtype):
    """HDF5's trailing-partial-element rule is unreachable through this API.

    The destination is a typed array, so its byte count is a whole multiple
    of ``itemsize`` for every dtype -- which is why the reader carries no
    leftover-bytes branch.
    """
    out = np.empty((7, 3), dtype=dtype)
    dst_bytes = out.reshape(-1).view(np.uint8).size
    assert dst_bytes % out.dtype.itemsize == 0
    body = np.arange(dst_bytes, dtype=np.uint8).reshape(-1, out.dtype.itemsize)
    _unshuffle_into(out, body.T.tobytes())
    assert out.reshape(-1).view(np.uint8).tolist() == body.reshape(-1).tolist()


def test_unshuffle_rejects_a_wrong_sized_buffer():
    out = np.empty((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="wrong byte count"):
        _unshuffle_into(out, b"\x00" * 8)


# ---------------------------------------------------------------------------
# load_survey itself
# ---------------------------------------------------------------------------


def test_load_survey_matches_a_plain_h5py_read(tmp_path):
    """End to end: the four datasets load_survey reads, byte for byte."""
    path = tmp_path / "survey.h5"
    npix, maxgals = 96, 23
    rng = np.random.default_rng(3)
    ngals = rng.integers(0, maxgals + 1, size=npix)
    zgals = np.sort(rng.uniform(0.01, 0.3, size=(npix, maxgals)), axis=1)
    dzgals = rng.uniform(1e-3, 1e-2, size=(npix, maxgals))
    wgals = rng.uniform(0.5, 1.5, size=(npix, maxgals))
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 4
        f.attrs["z_depth"] = 0.3
        for name, arr in (
            ("zgals", zgals), ("dzgals", dzgals), ("wgals", wgals),
        ):
            f.create_dataset(
                name, data=arr, chunks=(32, 7), compression="gzip", shuffle=True,
            )
        f.create_dataset(
            "ngals", data=ngals, chunks=(32,), compression="gzip", shuffle=True,
        )

    nside, ng, zg, dz, wg, z_depth = load_survey(
        path, to_device=False, sort_rows_by_z=False
    )
    with h5py.File(path, "r") as f:
        assert np.asarray(zg).tobytes() == np.asarray(f["zgals"]).tobytes()
        assert np.asarray(dz).tobytes() == np.asarray(f["dzgals"]).tobytes()
        assert np.asarray(wg).tobytes() == np.asarray(f["wgals"]).tobytes()
        assert np.asarray(ng).tobytes() == np.asarray(f["ngals"]).tobytes()
    assert int(nside) == 4 and z_depth == pytest.approx(0.3)


def test_load_survey_takes_the_threaded_path_when_it_pays(tmp_path, h5py_reads):
    """Wide cpuset, no size floor: all four datasets decode off the GIL."""
    path = tmp_path / "survey_fast.h5"
    npix, maxgals = 96, 23
    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 4
        for name in ("zgals", "dzgals", "wgals"):
            f.create_dataset(
                name, data=rng.uniform(0.01, 0.3, size=(npix, maxgals)),
                chunks=(32, 7), compression="gzip", shuffle=True,
            )
        f.create_dataset(
            "ngals", data=rng.integers(0, maxgals + 1, size=npix),
            chunks=(32,), compression="gzip", shuffle=True,
        )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(catalogs_io, "usable_cpu_count", lambda: 8)
        load_survey(path, to_device=False, sort_rows_by_z=False)
    assert h5py_reads == []


def test_load_survey_sorted_output_is_read_path_independent(tmp_path, monkeypatch):
    """The z-sorted arrays are identical whether the read is threaded or not."""
    path = tmp_path / "survey_sorted.h5"
    npix, maxgals = 64, 19
    rng = np.random.default_rng(11)
    ngals = rng.integers(1, maxgals + 1, size=npix)
    zgals = rng.uniform(0.01, 0.3, size=(npix, maxgals))
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 4
        for name, arr in (
            ("zgals", zgals),
            ("dzgals", rng.uniform(1e-3, 1e-2, size=(npix, maxgals))),
            ("wgals", rng.uniform(0.5, 1.5, size=(npix, maxgals))),
        ):
            f.create_dataset(
                name, data=arr, chunks=(16, 6), compression="gzip", shuffle=True,
            )
        f.create_dataset("ngals", data=ngals)

    monkeypatch.setattr(catalogs_io, "usable_cpu_count", lambda: 8)
    threaded = load_survey(path, to_device=False)
    monkeypatch.setenv(catalogs_io._CHUNK_READ_ENV, "0")
    sequential = load_survey(path, to_device=False)
    for a, b in zip(threaded[1:5], sequential[1:5]):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
