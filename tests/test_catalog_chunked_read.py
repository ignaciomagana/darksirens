"""Raw-chunk threaded reads of the survey catalog (``darksirens.catalogs.io``).

:func:`read_dataset_chunked` reimplements the (shuffle, deflate) HDF5 filter
pipeline so the inflate work can run off the GIL in a thread pool.  Every test
here pins the same contract: the array it returns is byte-for-byte the array
``np.asarray(dset)`` returns, on the fast path and on every fallback.
"""

import numpy as np
import h5py
import pytest

import darksirens.catalogs.io as catalogs_io
from darksirens.catalogs.io import (
    chunked_read_admissible,
    load_survey,
    read_dataset_chunked,
)


@pytest.fixture(autouse=True)
def _no_size_floor(monkeypatch):
    """Let the fixtures (kilobytes, not gigabytes) reach the threaded path.

    The production floor (64 MB) exists so tiny datasets skip the raw-chunk
    setup; the size test below pins it explicitly instead.
    """
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_MIN_BYTES", 0)


def _write(path, name, data, **kwargs):
    with h5py.File(path, "a") as f:
        f.create_dataset(name, data=data, **kwargs)


def _rng_table(shape, dtype=np.float64, seed=7):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(0, 10_000, size=shape).astype(dtype)


def _assert_reads_identically(path, name, workers=4):
    with h5py.File(path, "r") as f:
        ref = np.asarray(f[name])
        got = read_dataset_chunked(f[name], workers)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert got.tobytes() == ref.tobytes()
    return ref, got


# ---------------------------------------------------------------------------
# Bit identity on the fast path
# ---------------------------------------------------------------------------


def test_partial_edge_chunks_are_decompressed_and_sliced(tmp_path):
    """Both axes end in a partial chunk, as the production catalog does.

    (100, 47) over (32, 9) chunks is 100 = 3*32 + 4 rows and 47 = 5*9 + 2
    columns, i.e. the same shape of edge case as the production catalog's
    1719 = 63*27 + 18: HDF5 still stores a FULL 32x9 chunk there, so the
    reader has to inflate all of it and drop the padding.
    """
    path = tmp_path / "edge.h5"
    data = _rng_table((100, 47))
    _write(path, "x", data, chunks=(32, 9), compression="gzip", shuffle=True)
    ref, got = _assert_reads_identically(path, "x")
    assert np.array_equal(got, data)
    assert np.array_equal(got, ref)


def test_exact_chunk_grid_and_one_dimensional(tmp_path):
    path = tmp_path / "exact.h5"
    _write(
        path, "grid", _rng_table((64, 24)),
        chunks=(16, 8), compression="gzip", shuffle=True,
    )
    _write(
        path, "line", _rng_table((1000,), dtype=np.int64),
        chunks=(128,), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "grid")
    _assert_reads_identically(path, "line")


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
def test_identity_across_itemsizes(tmp_path, dtype):
    """The un-shuffle is an itemsize-wide byte transpose; pin 4- and 8-byte."""
    path = tmp_path / f"dt_{np.dtype(dtype).name}.h5"
    _write(
        path, "x", _rng_table((37, 29), dtype=dtype),
        chunks=(8, 7), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "x")


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_identity_is_worker_count_independent(tmp_path, workers):
    path = tmp_path / "w.h5"
    _write(
        path, "x", _rng_table((100, 47)),
        chunks=(32, 9), compression="gzip", shuffle=True,
    )
    _assert_reads_identically(path, "x", workers=workers)


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
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["ok"]) is True
        for name in (
            "contiguous", "no_shuffle", "uncompressed", "lzf", "checksummed",
        ):
            assert chunked_read_admissible(f[name]) is False, name


def test_size_floor_refuses_small_datasets(tmp_path, monkeypatch):
    path = tmp_path / "small.h5"
    _write(
        path, "x", _rng_table((40, 17)),
        chunks=(8, 5), compression="gzip", shuffle=True,
    )
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_MIN_BYTES", 1 << 30)
    with h5py.File(path, "r") as f:
        assert chunked_read_admissible(f["x"]) is False
    _assert_reads_identically(path, "x")  # still byte-identical, via h5py


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
    ],
)
def test_refused_pipelines_fall_back_to_h5py(tmp_path, kwargs):
    path = tmp_path / "fb.h5"
    _write(path, "x", _rng_table((40, 17)), **kwargs)
    _assert_reads_identically(path, "x")


def test_chunk_with_filters_skipped_falls_back(tmp_path):
    """A non-zero ``filter_mask`` means the stored bytes are not pipeline output.

    HDF5 sets it when it stored a chunk with part of the pipeline skipped, so
    inflating that chunk would be wrong; the reader must abandon the raw path
    for the whole dataset and let h5py do it.
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
        got = read_dataset_chunked(f["x"], 4)
    assert got.tobytes() == ref.tobytes()
    assert np.array_equal(got[:16, :5], data[:16, :5])


def test_unallocated_chunk_falls_back(tmp_path):
    """Nothing written: ``read_direct_chunk`` raises, the read still succeeds."""
    path = tmp_path / "sparse.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "x", shape=(32, 10), dtype=np.float64,
            chunks=(16, 5), compression="gzip", shuffle=True, fillvalue=1.25,
        )
    ref, got = _assert_reads_identically(path, "x")
    assert np.all(got == 1.25)


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

    threaded = load_survey(path, to_device=False)
    monkeypatch.setattr(catalogs_io, "_CHUNK_READ_MIN_BYTES", 1 << 30)
    sequential = load_survey(path, to_device=False)
    for a, b in zip(threaded[1:5], sequential[1:5]):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
