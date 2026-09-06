"""Survey catalog I/O helpers.

This module loads pixelated survey HDF5 files and optional per-galaxy mark
datasets. Redshift grids live in :mod:`darksirens.redshift.grid`.
"""

import os
import warnings
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import h5py


#: Raised by :func:`sort_survey_rows_by_z` (both implementations) when the
#: real-galaxy prefix is not ascending after sorting; ``cli/pixelate.py``
#: refers to this message.
ROW_Z_SORT_INVARIANT_ERROR = (
    "row z-sort invariant violated after sorting (NaN redshifts or "
    "real galaxies outside the ngal prefix?)"
)


def _row_z_sort_order(zgals, ngals):
    """Per-row permutation sorting the real-galaxy prefix by ascending z.

    Padded slots (index >= ngal) keep their positions after the real prefix
    (stable argsort on a key of +inf), so ``arange < ngal`` remains the real
    mask and padding values are untouched.  Deterministic: two callers reading
    the same file datasets derive bit-identical permutations, which is how
    :func:`load_survey_marks` stays co-indexed with :func:`load_survey`
    without any cross-call plumbing.
    """
    z = np.asarray(zgals)
    ng = np.asarray(ngals)
    real = np.arange(z.shape[1])[None, :] < ng[:, None]
    key = np.where(real, z, np.inf)
    return np.argsort(key, axis=1, kind="stable")


def _row_z_sort_order_device(zgals, ngals):
    """Device twin of :func:`_row_z_sort_order` (same permutation, exactly).

    ``jnp.argsort(..., stable=True)`` is the same stable sort on the same
    ``+inf``-padded key, so ties -- including every padding slot -- break by
    column index in both implementations and the two permutations are equal
    element for element (verified on the full production catalog).  The order
    is returned as ``int32``: the padded width is a few thousand columns, and
    halving the index table halves the transient device footprint of the
    load.
    """
    z = jnp.asarray(zgals)
    ng = jnp.asarray(ngals)
    real = jnp.arange(z.shape[1])[None, :] < ng[:, None]
    key = jnp.where(real, z, jnp.inf)
    return jnp.argsort(key, axis=1, stable=True).astype(jnp.int32)


def device_row_sort_admissible():
    """Is the device row sort the FASTER implementation here, and is it exact?

    ``to_device`` answers "will the caller upload these arrays", not "is there
    an accelerator to upload them to".  Two things have to hold before the
    device implementation is taken:

    * a non-CPU backend.  On a CPU-only install XLA-CPU's ``argsort`` is
      ~3.2x SLOWER than ``np.argsort(kind="stable")`` on production-shaped
      rows (8,192 x 1,719 float64, x64 on, backend pre-warmed: 0.59 s numpy
      vs 1.86 s XLA-CPU; on the full 49,152 x 1,719 production catalog
      2.79 s vs 10.0 s, i.e. +7.2 s per ``load_survey``).  The loaders pass
      ``to_device=True`` on a CPU box too, so without this test the change
      would buy seconds on the GPU by paying more of them on the CPU.
    * x64 enabled.  With it off ``jnp`` builds the sort key in float32 while
      :func:`_row_z_sort_order` (still used by :func:`load_survey_marks` and
      :func:`load_survey_galprops`) builds it in float64 numpy; the two
      permutations then differ on near-ties and marks silently stop being
      co-indexed with the galaxy arrays.  Falling back is the conservative
      answer -- it is exactly the pre-existing behaviour -- rather than
      raising on a call that used to work.

    Both are cheap: ``load_survey(..., to_device=True)`` initializes the
    backend a few lines later anyway.
    """
    return jax.default_backend() != "cpu" and bool(jax.config.jax_enable_x64)


def sort_survey_rows_by_z(zgals, dzgals, wgals, ngals, extras=(), to_device=False):
    """Sort every padded catalog row by galaxy redshift, coherently.

    Applies ONE per-row permutation (from :func:`_row_z_sort_order`) to every
    co-indexed per-galaxy array: ``zgals`` / ``dzgals`` / ``wgals`` and any
    ``extras`` (e.g. mark tables).  ``ngals`` is per-row and returned as-is.

    This establishes the row z-sort invariant required by the windowed
    catalog-KDE evaluator (``darksirens.redshift.catalog``): within each row
    the first ``ngal`` entries are non-decreasing in z, real galaxies form a
    contiguous prefix, and padding follows.  The invariant is hard-asserted
    here, at construction.

    ``to_device=True`` runs the identical operation on the accelerator
    (:func:`_sort_survey_rows_by_z_device`) and returns device arrays: the
    caller is about to upload these arrays anyway, so the sort rides along
    instead of costing seconds of single-threaded host work first.  It does
    so only when :func:`device_row_sort_admissible` agrees -- there is a
    non-CPU backend and x64 is on -- because on a CPU-only install the XLA
    path is the SLOWER of the two.  The default host path stays for callers
    that must compact on the host before any transfer
    (``load_survey(..., to_device=False)``) and for every CPU-only run.

    Returns ``(zgals, dzgals, wgals, ngals, extras)`` with extras a tuple
    (``None`` entries pass through).
    """
    if to_device and device_row_sort_admissible():
        return _sort_survey_rows_by_z_device(zgals, dzgals, wgals, ngals, extras)
    order = _row_z_sort_order(zgals, ngals)

    def _take(a):
        return np.take_along_axis(np.asarray(a), order, axis=1)

    z_s = _take(zgals)
    ng = np.asarray(ngals)
    # Sort invariant, asserted where it is constructed: real prefix ascending.
    cols = np.arange(1, z_s.shape[1])[None, :]
    ok = (np.diff(z_s, axis=1) >= 0) | (cols >= ng[:, None])
    if not bool(np.all(ok)):
        raise AssertionError(ROW_Z_SORT_INVARIANT_ERROR)
    extras_s = tuple(None if e is None else _take(e) for e in extras)
    return z_s, _take(dzgals), _take(wgals), ng, extras_s


def _sort_survey_rows_by_z_device(zgals, dzgals, wgals, ngals, extras=()):
    """Device implementation of :func:`sort_survey_rows_by_z`.

    Same permutation, same outputs, bit for bit -- only the machine differs:
    the (npix, maxgals) key, the stable argsort and the gathers run on the
    accelerator instead of one host core, and the invariant is a device
    reduction with a single scalar transfer instead of a full-width
    ``np.diff`` on the host.

    Each source array is uploaded and gathered one at a time, and the gather
    is blocked on before the next upload starts, so the raw buffer is
    reclaimable as soon as its gather is done: the transient device footprint
    stays near two copies of one array plus the int32 order, rather than
    holding every raw and gathered array at once.
    """
    ng = jnp.asarray(ngals)
    z_d = jnp.asarray(zgals)
    if z_d.shape[0] == 0:
        # Degenerate shape (no rows): the gather op rejects a zero-row
        # operand, while the host path returns cleanly.  Keep the two
        # implementations interchangeable.
        def _up(a):
            return None if a is None else jnp.asarray(a)

        extras_up = tuple(_up(e) for e in extras)
        return z_d, _up(dzgals), _up(wgals), ng, extras_up
    order = _row_z_sort_order_device(z_d, ng)

    def _take(a):
        out = jnp.take_along_axis(jnp.asarray(a), order, axis=1)
        # Block before the next upload: the raw buffer is dead at this point,
        # so the allocator can reuse it instead of stacking a second one.
        out.block_until_ready()
        return out

    z_s = jnp.take_along_axis(z_d, order, axis=1)
    z_s.block_until_ready()
    del z_d
    # Sort invariant, asserted where it is constructed: real prefix ascending.
    cols = jnp.arange(1, z_s.shape[1])[None, :]
    ok = jnp.all((jnp.diff(z_s, axis=1) >= 0) | (cols >= ng[:, None]))
    if not bool(ok):
        raise AssertionError(ROW_Z_SORT_INVARIANT_ERROR)
    extras_s = tuple(None if e is None else _take(e) for e in extras)
    return z_s, _take(dzgals), _take(wgals), ng, extras_s


#: Worker count for :func:`read_dataset_chunked`.  Four, not more: only
#: ``zlib.decompress`` releases the GIL, while the byte un-shuffle and the
#: assemble copy (48% of the per-chunk work on the production catalog) hold
#: it, so the Amdahl ceiling is 2.08x and it is reached at four workers.
#: Measured on the production catalog (49,152 x 1,719 f64, gzip-4 + shuffle,
#: (768, 27) chunks), warm page cache, median of three: sequential h5py
#: 4.68 s, 2 workers 3.60 s, 4 workers 1.93 s, 6 workers 2.63 s, 8 workers
#: 4.27 s, 16 workers 3.48 s.  Past ~4 workers GIL convoying eats the win,
#: so raising this is a regression, not a bigger gain.
_CHUNK_READ_WORKERS = 4

#: Below this many workers the threaded read is SLOWER than plain h5py and
#: :func:`read_dataset_chunked` refuses outright.  It is not that the win
#: merely shrinks: with one worker the pipeline pays the un-shuffle and the
#: assemble copy in Python with no inflate to overlap them against, measured
#: on the production catalog at 6.41 s against 4.53 s for the h5py read it
#: replaces -- a 1.4x REGRESSION.  Two usable CPUs already win (3.34 s
#: against 4.42 s on a ``taskset -c 20,21`` cpuset), so two is the floor.
_CHUNK_READ_MIN_WORKERS = 2

#: Datasets smaller than this read faster sequentially than the raw-chunk
#: machinery costs to set up.  The four production catalog datasets are
#: 0.68 GB (zgals/dzgals/wgals) and 0.4 MB (ngals) decompressed, so the
#: threshold selects exactly the three that dominate the read.
_CHUNK_READ_MIN_BYTES = 64 << 20

#: Cap on submitted-but-not-yet-inflated chunks, in multiples of the worker
#: count.  The calling thread reads raw chunk bytes ~10x faster than the pool
#: inflates them (0.17 s of ``read_direct_chunk`` against 1.5 s of ``zlib``
#: on the production zgals), so an unbounded submit loop holds essentially
#: the whole COMPRESSED dataset in the executor queue -- a measured +0.14 GB
#: of host RSS on the production catalog, and unbounded in general.  Four
#: deep per worker keeps every worker fed while capping the queue at a few
#: chunks, and costs nothing: reading the four production datasets measures
#: 2.01 s capped against 2.10 s unbounded (medians of three), with peak host
#: RSS 0.77 GB capped, 0.91 GB unbounded, 0.78 GB for the h5py read.
_CHUNK_READ_QUEUE_DEPTH = 4

#: The one HDF5 filter pipeline :func:`read_dataset_chunked` knows how to
#: invert, as filter ids in APPLICATION order: shuffle, then deflate.
_CHUNK_READ_PIPELINE = (h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE)

#: Set to ``"0"`` to force the plain h5py read (the escape hatch for a
#: machine where the threaded read is not a win -- a cgroup CPU *quota*, say,
#: which :func:`usable_cpu_count` cannot see).  Results are byte-identical
#: either way; this only chooses which code assembles the same bytes.
_CHUNK_READ_ENV = "DARKSIRENS_CATALOG_CHUNKED_READ"


def _read_dataset_h5py(dset):
    """Plain ``np.asarray(dset)``: the fallback every refusal here returns.

    A named funnel, not a wrapper for its own sake -- it is the single point
    the tests spy on to tell "byte-identical because the fast path is right"
    from "byte-identical because the fast path silently bailed out", which
    every identity assertion in this module would otherwise conflate.
    """
    return np.asarray(dset)


def chunked_read_enabled():
    """Is the raw-chunk read path switched on for this process?

    ``DARKSIRENS_CATALOG_CHUNKED_READ=0`` turns it off; anything else (or
    unset) leaves it on.  Read per call rather than at import, so a caller
    or a test can flip it without reloading the module.
    """
    return os.environ.get(_CHUNK_READ_ENV, "1") != "0"


def usable_cpu_count():
    """CPUs this PROCESS may run on, not the ones the box has.

    ``os.cpu_count()`` reports the host's cores and ignores the cpuset a
    batch allocation or a container pins the process to, so sizing a thread
    pool with it hands four workers to a one-CPU allocation -- where the
    threaded read is a measured 1.4x regression on the very phase it exists
    to speed up.  ``sched_getaffinity`` is the cpuset-aware count.  It does
    not see a cgroup CPU *quota* (fractional bandwidth rather than a mask),
    which is what :data:`_CHUNK_READ_ENV` is for.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def _filter_pipeline(dset):
    """The dataset's HDF5 filter ids, in application order.

    Read from the dataset creation property list, which is the only place
    the pipeline is stated exactly.  ``dset.compression`` / ``dset.shuffle``
    are convenience properties answering "is this filter PRESENT", so they
    cannot see a third filter in the pipeline and cannot see the order.
    """
    plist = dset.id.get_create_plist()
    return tuple(plist.get_filter(i)[0] for i in range(plist.get_nfilters()))


def _n_chunks(dset):
    """How many chunks the dataset grid has, allocated or not."""
    n = 1
    for extent, chunk in zip(dset.shape, dset.chunks):
        n *= -(-extent // chunk)
    return n


def _fully_allocated(dset):
    """Does every chunk of the grid actually have storage in the file?

    A chunk that was never written has none, and ``read_direct_chunk`` on it
    does not fail in any one way: h5py sizes the destination buffer from a
    storage size HDF5 never set, so the same unallocated chunk raises
    ``OSError``, ``MemoryError`` or ``SystemError`` from one run to the next.
    Gating on the chunk count instead turns that lottery into one cheap,
    deterministic refusal (a sub-millisecond ``H5Dget_num_chunks`` on the
    4096-chunk production datasets) and lets the read's own error handling
    stay narrow enough to expose a real defect.
    """
    return dset.id.get_num_chunks() == _n_chunks(dset)


def chunked_read_admissible(dset):
    """Is ``dset`` one this module may read through raw chunks?

    :func:`read_dataset_chunked` reimplements exactly one HDF5 filter
    pipeline -- shuffle followed by deflate -- so it must refuse every
    dataset whose bytes on disk mean anything else.  What makes that true is
    pipeline EQUALITY against :data:`_CHUNK_READ_PIPELINE`, not a set of
    presence checks on ``dset.compression`` / ``dset.shuffle`` /
    ``dset.fletcher32`` / ``dset.scaleoffset``: those properties report
    whether a filter is in the pipeline and say nothing about what ELSE is
    in it or in what ORDER.  A legal ``(shuffle, nbit, deflate)`` dataset
    passes every presence check, inflates to a chunk of exactly the right
    byte count, and then decodes to different bytes on every element; a
    reversed ``(deflate, shuffle)`` dataset is indistinguishable from the
    pipeline handled here.  One equality comparison refuses both, and with
    them fletcher32, scale-offset, lzf and any unknown third-party filter
    id -- deliberately, rather than by accident.

    Also refused (the caller reads each with plain h5py, which handles all
    of them):

    * contiguous datasets (``chunks is None``): there are no chunks to read.
      A virtual dataset is one of these -- h5py reports ``chunks is None``
      for the ``H5D_VIRTUAL`` layout -- so the explicit ``is_virtual`` term
      below is belt-and-braces documenting intent, not a distinct gate;
    * non-numeric dtypes (object/vlen/compound), where a chunk is not a
      dense block of ``itemsize``-byte elements;
    * partially allocated datasets (:func:`_fully_allocated`), where some
      chunk was never written and has no stored bytes to read;
    * anything below :data:`_CHUNK_READ_MIN_BYTES`, where the win is smaller
      than the setup.

    A dataset that passes decompresses to exactly the bytes h5py would
    return -- the same two filters, inverted in the same order.
    """
    if dset.chunks is None or bool(dset.is_virtual):
        return False
    if dset.dtype.kind not in "fiub":
        return False
    try:
        pipeline = _filter_pipeline(dset)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return False
    if pipeline != _CHUNK_READ_PIPELINE:
        return False
    if dset.nbytes < _CHUNK_READ_MIN_BYTES:
        return False
    return _fully_allocated(dset)


def _unshuffle_into(out, raw):
    """Invert the HDF5 shuffle filter from ``raw`` into ``out``.

    Shuffle stores the ``k``-th byte of every element contiguously: for
    ``n`` elements of ``itemsize`` bytes the filtered buffer is the
    ``(itemsize, n)`` byte transpose of the element bytes.  Transposing back
    straight into the destination costs one copy instead of materialising
    the un-shuffled buffer first.

    HDF5's filter also copies through any trailing bytes that do not fill a
    whole element, and that rule is deliberately NOT coded for: ``out`` is a
    typed array, so its byte count is a whole multiple of ``itemsize`` by
    construction and there are never leftover bytes.  (An earlier revision
    carried the branch; it was unreachable from any caller and untestable
    through this signature.)
    """
    itemsize = out.dtype.itemsize
    dst = out.reshape(-1).view(np.uint8)
    src = np.frombuffer(raw, dtype=np.uint8)
    if src.size != dst.size:
        raise ValueError("shuffled chunk has the wrong byte count")
    n = dst.size // itemsize
    dst.reshape(n, itemsize)[:] = src.reshape(itemsize, n).T


def _chunk_starts(shape, chunks):
    """Every chunk origin of the dataset grid, in C order."""
    starts = [range(0, shape[ax], chunks[ax]) for ax in range(len(shape))]
    out = [()]
    for axis_starts in starts:
        out = [prev + (s,) for prev in out for s in axis_starts]
    return out


def read_dataset_chunked(dset, workers=None):
    """Read ``dset`` by decompressing its raw chunks in a thread pool.

    Byte-for-byte the result of ``np.asarray(dset)``: the same stored chunks
    are inflated with ``zlib`` and un-shuffled with the same filter
    definitions HDF5 uses, then written into disjoint slices of one
    preallocated array.  Nothing numeric is computed -- these are the same
    bytes, assembled by hand -- so the caller's arrays are unchanged.

    The point is that h5py cannot be parallelised (HDF5 serializes, and 4
    threads on 4 file handles measure no faster than one), while this can:
    the raw chunk bytes come off disk on the calling thread and
    ``zlib.decompress`` releases the GIL.  On the production catalog the
    read of zgals/dzgals/wgals drops 4.6 s -> 2.1 s (2.2x), which is the
    GIL ceiling for the split of work here (see :data:`_CHUNK_READ_WORKERS`).

    ``workers`` defaults to :data:`_CHUNK_READ_WORKERS` capped by
    :func:`usable_cpu_count`, i.e. by the process's cpuset rather than by
    the host's core count.

    Falls back to :func:`_read_dataset_h5py` -- and so stays correct rather
    than fast -- whenever the threaded read is not wanted or the raw bytes
    are not what this function knows how to invert:

    * fewer than :data:`_CHUNK_READ_MIN_WORKERS` workers, where the threaded
      read is measurably SLOWER than the h5py read it replaces;
    * ``DARKSIRENS_CATALOG_CHUNKED_READ=0`` (:func:`chunked_read_enabled`);
    * a dataset :func:`chunked_read_admissible` refuses;
    * a chunk whose ``filter_mask`` is non-zero: HDF5 stored that chunk with
      part of the pipeline skipped, so the bytes are not (shuffle, deflate)
      output -- and because the mask is per filter it does not even say the
      chunk is stored raw, so the whole dataset falls back rather than this
      one chunk being copied through;
    * any I/O or zlib error on the way.

    Those last two warn (``RuntimeWarning``): the fast path was admitted and
    then failed, which is worth naming rather than silently losing the 2.2x.
    The refusals above them are by design, and silent.
    """
    if workers is None:
        workers = min(_CHUNK_READ_WORKERS, usable_cpu_count())
    workers = int(workers)
    if workers < _CHUNK_READ_MIN_WORKERS or not chunked_read_enabled():
        return _read_dataset_h5py(dset)
    if not chunked_read_admissible(dset):
        return _read_dataset_h5py(dset)
    shape = dset.shape
    chunks = dset.chunks
    dtype = dset.dtype
    try:
        out = np.empty(shape, dtype=dtype)
        dsid = dset.id

        def _decompress_into(start, raw):
            buf = np.empty(chunks, dtype=dtype)
            _unshuffle_into(buf, zlib.decompress(raw))
            stop = tuple(
                min(start[ax] + chunks[ax], shape[ax]) for ax in range(len(shape))
            )
            dst = tuple(slice(start[ax], stop[ax]) for ax in range(len(shape)))
            # The stored chunk is always full-size; an edge chunk's padding
            # past the dataset bounds is decompressed and dropped here.
            src = tuple(slice(0, stop[ax] - start[ax]) for ax in range(len(shape)))
            out[dst] = buf[src]

        queue_cap = max(1, _CHUNK_READ_QUEUE_DEPTH * workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = deque()
            for start in _chunk_starts(shape, chunks):
                filter_mask, raw = dsid.read_direct_chunk(start)
                if filter_mask != 0:
                    raise ValueError("chunk stored with filters skipped")
                in_flight.append(pool.submit(_decompress_into, start, raw))
                # Drop this thread's reference and drain back under the cap:
                # the reader outruns the inflaters ~10x, so an unbounded
                # submit loop would queue the whole compressed dataset.
                del raw
                while len(in_flight) >= queue_cap:
                    in_flight.popleft().result()
            while in_flight:
                in_flight.popleft().result()
    except (OSError, zlib.error, ValueError) as exc:
        # Narrow on purpose: a genuine logic error in the fast path must
        # surface as a failure, not masquerade as a correct-but-slow read.
        # These three are what a bad FILE produces -- HDF5 I/O, a corrupt
        # deflate stream, a chunk whose bytes are not pipeline output.  The
        # one condition that used to raise something else (an unallocated
        # chunk) is refused by the gate now, so nothing legitimate reaches
        # here as a TypeError or an AttributeError.
        warnings.warn(
            f"raw-chunk read of {dset.name!r} failed ({type(exc).__name__}: "
            f"{exc}); falling back to the plain h5py read -- same bytes, "
            f"slower. Set {_CHUNK_READ_ENV}=0 to skip the raw-chunk path.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _read_dataset_h5py(dset)
    return out


def load_survey(survey_path, to_device=True, sort_rows_by_z=True):
    """Load the pixelated survey. ``to_device=False`` keeps the dense full-sky
    arrays on the host so callers can compact before transferring to device.

    ``z_depth`` is the optional per-survey redshift depth written by
    ``darksirens_pixelate --z_depth`` (``f.attrs['z_depth']``): the redshift
    beyond which the survey catalogs no galaxies, used as the completeness prior
    (completeness is zero beyond it; hosts there are missing, not nonexistent --
    :mod:`darksirens.redshift.completion`).  ``None`` when the attribute is
    absent (completeness estimated over the full grid; legacy path).

    The four galaxy datasets are read through :func:`read_dataset_chunked`,
    which decompresses their raw HDF5 chunks in a small thread pool and
    returns byte-identical arrays (it falls back to plain h5py for any
    dataset it cannot invert bit for bit, and on a cpuset too small for the
    threaded read to pay).  ``DARKSIRENS_CATALOG_CHUNKED_READ=0`` forces the
    plain h5py read; the arrays are the same bytes either way.

    ``sort_rows_by_z`` (default True) applies :func:`sort_survey_rows_by_z`,
    establishing the per-row z-sort invariant the windowed catalog-KDE hot
    path requires.  It runs on the device when ``to_device=True`` and there
    is an accelerator to run it on (:func:`device_row_sort_admissible`; the
    arrays are bound for the device anyway) and on the host otherwise, so
    callers that compact before transferring, and every CPU-only run, still
    never touch the accelerator.  Per-galaxy mark datasets loaded via
    :func:`load_survey_marks` derive the SAME permutation from the file, so
    they stay co-indexed.  Pass False to keep the raw file
    order (disables KDE windowing downstream; row-reduction results are
    identical either way up to floating-point summation order).
    """
    asarray = jnp.asarray if to_device else np.asarray
    with h5py.File(survey_path, 'r') as f:
        nside = f.attrs['nside']
        zgals = read_dataset_chunked(f['zgals'])
        ngals = read_dataset_chunked(f['ngals'])
        dzgals = read_dataset_chunked(f['dzgals'])
        wgals = read_dataset_chunked(f['wgals'])
        z_depth = float(f.attrs['z_depth']) if 'z_depth' in f.attrs else None
    if sort_rows_by_z:
        zgals, dzgals, wgals, ngals, _ = sort_survey_rows_by_z(
            zgals, dzgals, wgals, ngals, to_device=to_device
        )
    return (
        nside, asarray(ngals), asarray(zgals), asarray(dzgals), asarray(wgals),
        z_depth,
    )


#: Per-galaxy "mark" datasets optionally written by ``darksirens_pixelate``
#: (padded ``(npix, maxgals)`` arrays), keyed by the EMCatalog field name.
MARK_DATASETS = ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color")

#: Per-galaxy PROPERTY datasets (padded like marks) that are NOT marks: they
#: never enter the sampled likelihood and never enroll in the marked-host
#: model -- they exist for OFFLINE consumers only (the selection-function fit
#: and the Q-table builder).  Kept in a separate registry precisely so a
#: catalog carrying magnitudes cannot silently become a "marked" catalog.
#: Padding value is 0.0 (an absurd apparent magnitude): every reader MUST mask
#: real slots via ``ngals`` (``arange < ngal``), never by value.
#: ``gal_stratum`` carries an integer stratum label per galaxy (imaging side
#: or m_th stratum) as a padded float table; -pad 0.0 is NOT a label, mask by
#: ``ngals`` like every galprop.
GALPROP_DATASETS = ("gal_app_mag", "gal_stratum")


def load_survey_marks(survey_path, datasets=None, sort_rows_by_z=True):
    """Load per-galaxy mark datasets present in the pixelated survey file.

    Returns ``{dataset_name: (npix, maxgals) ndarray}`` for whichever of
    :data:`MARK_DATASETS` exist (empty dict if none).  These are the *raw* marks;
    z-centering happens at load (``inference/data.py``).

    ``datasets``, if given, restricts the read to that subset of
    :data:`MARK_DATASETS` (e.g. a K>=2 mixture catalog's requested marks only)
    so unrequested mark tables are never pulled off disk.  ``None`` (default)
    loads every mark dataset present, for callers that need to auto-detect
    which marks a catalog provides.

    ``sort_rows_by_z`` (default True, matching :func:`load_survey`) permutes
    each mark row through the SAME per-row z-sort permutation ``load_survey``
    applies, re-derived deterministically from this file's ``zgals``/``ngals``
    (stable argsort of identical inputs), so marks stay co-indexed with the
    sorted galaxy arrays.  Pass False when pairing with
    ``load_survey(..., sort_rows_by_z=False)``.
    """
    wanted = MARK_DATASETS if datasets is None else tuple(datasets)
    out = {}
    with h5py.File(survey_path, 'r') as f:
        for ds in wanted:
            if ds in f:
                out[ds] = np.asarray(f[ds])
        if sort_rows_by_z and out:
            order = _row_z_sort_order(
                np.asarray(f['zgals']), np.asarray(f['ngals'])
            )
            out = {
                ds: np.take_along_axis(arr, order, axis=1)
                for ds, arr in out.items()
            }
    return out


def load_survey_galprops(survey_path, datasets=None, sort_rows_by_z=True):
    """Load per-galaxy property datasets (:data:`GALPROP_DATASETS`).

    Same padded layout and row z-sort permutation as
    :func:`load_survey_marks` (the permutation is re-derived from this file's
    ``zgals``/``ngals``, so properties stay co-indexed with the sorted galaxy
    arrays) -- but from the separate non-mark registry, so requesting
    galaxy properties can never enroll a catalog into the marked-host model.
    Real slots must be masked via ``ngals``; the 0.0 padding is not a value.
    """
    wanted = GALPROP_DATASETS if datasets is None else tuple(datasets)
    return load_survey_marks(
        survey_path, datasets=wanted, sort_rows_by_z=sort_rows_by_z)
