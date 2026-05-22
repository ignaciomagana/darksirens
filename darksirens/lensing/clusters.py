"""
clusters.py
-----------
Container and HDF5 I/O for pre-identified candidate multi-image clusters.

The cluster set is supplied by the user from an external pairwise-Bayes-
factor analysis (the standard LVK lensing-search output, or any equivalent
selection). The inference treats the set as fixed; the partition is not
re-summed inside the sampler.

HDF5 schema
~~~~~~~~~~~
Required datasets (any missing dataset is treated as empty):

    /pairs               (Npairs, 2)   int   — event indices into gw_pe
    /quads               (Nquads, 4)   int   — event indices into gw_pe
    /pair_log_BF         (Npairs,)     float — log10 lensing Bayes factor
    /quad_log_BF         (Nquads,)     float — log10 lensing Bayes factor

Required attributes:

    nEvents              int — total number of events in the PE catalogue.
                         Used to validate the event indices.

Optional attributes:

    bf_threshold         float — threshold log10 BF used to select
                         candidates. Stored for provenance only.
"""

from __future__ import annotations

from typing import NamedTuple, Any, Optional
from pathlib import Path

import numpy as np
import jax.numpy as jnp

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


class ClusterSet(NamedTuple):
    """JAX-friendly container of candidate lensed clusters.

    Fields
    ------
    pair_indices : (npairs, 2) int32
        Event-index pairs (i, j) for candidate doubles. Empty when no
        candidates. ``i < j`` enforced at load time so each unordered
        pair appears exactly once.
    quad_indices : (nquads, 4) int32
        Event-index tuples for candidate quads.
    pair_log_BF : (npairs,) float64
        log10 Bayes factor for the lensed-vs-unlensed pair hypothesis.
        Carried for diagnostics; not used inside the likelihood.
    quad_log_BF : (nquads,) float64
        Same, for quads.
    npairs, nquads : int
        Counts (also derivable from shapes; carried as Python ints for
        ``static_argnames`` ergonomics).
    """
    pair_indices: Any
    quad_indices: Any
    pair_log_BF: Any
    quad_log_BF: Any
    npairs: int
    nquads: int


# ============================================================================
# Constructors
# ============================================================================

def empty_cluster_set() -> ClusterSet:
    """Empty cluster set (no lensed candidates)."""
    return ClusterSet(
        pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
        quad_indices=jnp.zeros((0, 4), dtype=jnp.int32),
        pair_log_BF=jnp.zeros((0,), dtype=jnp.float64),
        quad_log_BF=jnp.zeros((0,), dtype=jnp.float64),
        npairs=0,
        nquads=0,
    )


def make_cluster_set(
    pair_indices: Optional[np.ndarray] = None,
    quad_indices: Optional[np.ndarray] = None,
    pair_log_BF: Optional[np.ndarray] = None,
    quad_log_BF: Optional[np.ndarray] = None,
    n_events: Optional[int] = None,
) -> ClusterSet:
    """Construct a ClusterSet from arrays, with validation.

    Parameters
    ----------
    pair_indices, quad_indices
        Event-index arrays. None or empty arrays produce empty fields.
    pair_log_BF, quad_log_BF
        Bayes-factor arrays. Defaults to zeros of matching length.
    n_events
        If supplied, validate that all indices are in ``[0, n_events)``.
    """
    if pair_indices is None or len(pair_indices) == 0:
        pair_indices = np.zeros((0, 2), dtype=np.int32)
    else:
        pair_indices = np.asarray(pair_indices, dtype=np.int32)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError(
                f"pair_indices must have shape (N, 2), got {pair_indices.shape}."
            )

    if quad_indices is None or len(quad_indices) == 0:
        quad_indices = np.zeros((0, 4), dtype=np.int32)
    else:
        quad_indices = np.asarray(quad_indices, dtype=np.int32)
        if quad_indices.ndim != 2 or quad_indices.shape[1] != 4:
            raise ValueError(
                f"quad_indices must have shape (N, 4), got {quad_indices.shape}."
            )

    npairs = pair_indices.shape[0]
    nquads = quad_indices.shape[0]

    if pair_log_BF is None:
        pair_log_BF = np.zeros(npairs, dtype=np.float64)
    else:
        pair_log_BF = np.asarray(pair_log_BF, dtype=np.float64)
        if pair_log_BF.shape != (npairs,):
            raise ValueError(
                f"pair_log_BF shape {pair_log_BF.shape} != pair count ({npairs},)."
            )

    if quad_log_BF is None:
        quad_log_BF = np.zeros(nquads, dtype=np.float64)
    else:
        quad_log_BF = np.asarray(quad_log_BF, dtype=np.float64)
        if quad_log_BF.shape != (nquads,):
            raise ValueError(
                f"quad_log_BF shape {quad_log_BF.shape} != quad count ({nquads},)."
            )

    # Validation: ordering and bounds
    if npairs > 0:
        if np.any(pair_indices[:, 0] >= pair_indices[:, 1]):
            raise ValueError(
                "pair_indices must satisfy i < j (each unordered pair listed once)."
            )
    if nquads > 0:
        # Quad indices should be strictly increasing within each row.
        diffs = np.diff(quad_indices, axis=1)
        if np.any(diffs <= 0):
            raise ValueError(
                "quad_indices must be strictly increasing along the row (each unordered quad listed once)."
            )

    if n_events is not None:
        if npairs > 0 and (pair_indices.min() < 0 or pair_indices.max() >= n_events):
            raise ValueError(
                f"pair_indices out of range [0, {n_events}): "
                f"min={pair_indices.min()}, max={pair_indices.max()}."
            )
        if nquads > 0 and (quad_indices.min() < 0 or quad_indices.max() >= n_events):
            raise ValueError(
                f"quad_indices out of range [0, {n_events}): "
                f"min={quad_indices.min()}, max={quad_indices.max()}."
            )

    return ClusterSet(
        pair_indices=jnp.asarray(pair_indices, dtype=jnp.int32),
        quad_indices=jnp.asarray(quad_indices, dtype=jnp.int32),
        pair_log_BF=jnp.asarray(pair_log_BF, dtype=jnp.float64),
        quad_log_BF=jnp.asarray(quad_log_BF, dtype=jnp.float64),
        npairs=int(npairs),
        nquads=int(nquads),
    )


# ============================================================================
# HDF5 I/O
# ============================================================================

def load_clusters(path: Optional[str], n_events: Optional[int] = None) -> ClusterSet:
    """Load a cluster set from HDF5, or return an empty set if path is None.

    Parameters
    ----------
    path
        Path to the HDF5 file. If ``None`` or the file is missing,
        returns an empty cluster set.
    n_events
        If supplied, validate that all event indices are in range.
    """
    if path is None:
        return empty_cluster_set()

    if not _HAS_H5PY:
        raise ImportError(
            "h5py is required for load_clusters. Install via `pip install h5py`."
        )

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cluster file not found: {path}")

    with h5py.File(str(path), "r") as f:
        pairs = np.array(f["pairs"]) if "pairs" in f else None
        quads = np.array(f["quads"]) if "quads" in f else None
        pair_bf = np.array(f["pair_log_BF"]) if "pair_log_BF" in f else None
        quad_bf = np.array(f["quad_log_BF"]) if "quad_log_BF" in f else None
        # Use file's nEvents attr if not overridden
        if n_events is None and "nEvents" in f.attrs:
            n_events = int(f.attrs["nEvents"])

    return make_cluster_set(
        pair_indices=pairs,
        quad_indices=quads,
        pair_log_BF=pair_bf,
        quad_log_BF=quad_bf,
        n_events=n_events,
    )


def save_clusters(
    path: str,
    cluster_set: ClusterSet,
    n_events: Optional[int] = None,
    bf_threshold: Optional[float] = None,
) -> None:
    """Write a cluster set to HDF5. Round-trips with ``load_clusters``."""
    if not _HAS_H5PY:
        raise ImportError(
            "h5py is required for save_clusters. Install via `pip install h5py`."
        )

    with h5py.File(path, "w") as f:
        f.create_dataset("pairs", data=np.asarray(cluster_set.pair_indices))
        f.create_dataset("quads", data=np.asarray(cluster_set.quad_indices))
        f.create_dataset("pair_log_BF", data=np.asarray(cluster_set.pair_log_BF))
        f.create_dataset("quad_log_BF", data=np.asarray(cluster_set.quad_log_BF))
        if n_events is not None:
            f.attrs["nEvents"] = int(n_events)
        if bf_threshold is not None:
            f.attrs["bf_threshold"] = float(bf_threshold)
