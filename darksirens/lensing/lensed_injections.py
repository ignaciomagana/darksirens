"""
lensed_injections.py
--------------------
Container and loader for pre-rendered lensed-injection sets used by the
cluster selection integral (commit 4).

Format (α from commit 4 design choices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A lensed-injection campaign generates N_src source-frame draws and, for
each, computes both SIS images and renders each through the detection
pipeline. The on-disk representation is a flat (per-image) HDF5:

  source_id      (N_img,) int32     groups images by underlying source
  image_id       (N_img,) int32     0 = μ_+ (type-I), 1 = μ_- (type-II)
  m1_src         (N_img,)           detector-frame source mass [M☉]
                                    (constant within a source_id group)
  q_src          (N_img,)           mass ratio (same for both images)
  z_src          (N_img,)           source redshift (same for both images)
  chieff         (N_img,)           effective spin (same for both images)
  y_source       (N_img,)           SIS impact parameter ∈ (0, 1)
                                    (same for both images of a source)
  mu             (N_img,)           image magnification (different per image)
  detected       (N_img,) bool      single-event detection flag for THIS image
  p_prop_src     (N_img,)           source proposal density evaluated at θ_src
                                    (same for both images of a source)
  p_prop_y       (N_img,)           y proposal density evaluated at y_source
                                    (same for both images of a source)

  attrs: Ndraw_sources              total source-frame draws (detected + undetected)

This is the natural product of an injection campaign that draws sources
from p_prop_src · p_prop_y, computes both images, and renders each.
Storing one row per image (not per source) makes the file format trivial
to merge across batches and avoids any nontrivial 2-D indexing.

Why ``p_prop_src`` and ``p_prop_y`` are per-image
-------------------------------------------------
They are redundant (same value for both images of a source) but storing
them per-image lets the consumer flatten and re-pair without juggling
two array shapes. Memory cost is negligible (~10 MB per million images).

What the cluster selection integral uses
----------------------------------------
``compute_cluster_selection_term`` keeps only sources where BOTH images
are detected:

    mu_sel^(2)(λ, Θ) ≈ (1 / N_draw_sources) · Σ_{s: both detected}
        w_s(λ, Θ)

with

    w_s = p_pop(θ_src) · p_z(z_src) · τ_2(z_src) · p(y_source)
          / (p_prop_src · p_prop_y).

See ``cluster_selection.py`` for the implementation.
"""

from __future__ import annotations

from typing import NamedTuple, Any, Optional
import numpy as np
import jax.numpy as jnp
import h5py


class LensedInjectionSet(NamedTuple):
    """Per-image lensed injection arrays + bookkeeping.

    Convention: paired arrays are flattened to one entry per IMAGE. The
    cluster selection integral re-pairs them by ``source_id``. To avoid
    any costly group-by during the JIT'd hot path, we precompute the
    "both-detected" mask at load time and re-pack the kept-source data
    into per-source arrays.
    """
    # Per-source (pre-paired) arrays — what the JIT'd hot path actually uses.
    # These arrays are length N_kept_sources, one row per source where
    # both images were detected.
    source_id_kept: Any           # (N_kept,) int32
    m1_src: Any                   # (N_kept,)
    q_src: Any                    # (N_kept,)
    z_src: Any                    # (N_kept,)
    chieff: Any                   # (N_kept,)
    y_source: Any                 # (N_kept,)
    mu_plus: Any                  # (N_kept,) — μ_+ for kept source
    mu_minus: Any                 # (N_kept,) — μ_- for kept source
    p_prop_src: Any               # (N_kept,) — source proposal density
    p_prop_y: Any                 # (N_kept,) — y proposal density
    valid: Any                    # (N_kept,) bool — structural mask for padding
    # Bookkeeping
    n_draw_sources: Any           # scalar — total source-frame draws

    @property
    def n_kept(self):
        return int(self.source_id_kept.shape[0])


def _make_lensed_injection_arrays_per_source(
    source_id: np.ndarray,
    image_id: np.ndarray,
    m1_src: np.ndarray,
    q_src: np.ndarray,
    z_src: np.ndarray,
    chieff: np.ndarray,
    y_source: np.ndarray,
    mu: np.ndarray,
    detected: np.ndarray,
    p_prop_src: np.ndarray,
    p_prop_y: np.ndarray,
):
    """Group per-image arrays into per-source arrays for paired sources where
    BOTH images are detected. Returns the seven per-source arrays in the
    canonical (μ_+, μ_-) order.

    Validation
    ----------
    - Each source_id must appear exactly twice (one μ_+, one μ_-).
    - image_id ∈ {0, 1} must be {0: μ_+, 1: μ_-}.
    - All source-level fields (m1_src, q_src, z_src, chieff, y_source,
      p_prop_src, p_prop_y) must agree between the two images of a source.

    A "kept source" is one where BOTH images have detected==True. The
    returned arrays are length N_kept (not N_sources or N_img).
    """
    source_id = np.asarray(source_id)
    image_id = np.asarray(image_id)
    n_img = len(source_id)
    if n_img % 2 != 0:
        raise ValueError(
            f"lensed injection set has odd number of images ({n_img}); "
            f"each source must have exactly two."
        )

    # Group: stable sort by (source_id, image_id) so each source occupies
    # two contiguous rows with image_id=0 first.
    order = np.lexsort((image_id, source_id))
    sid = source_id[order]
    iid = image_id[order]
    if not np.all(sid[0::2] == sid[1::2]):
        raise ValueError(
            "Each source_id must appear exactly twice. Found mismatched "
            "consecutive source_ids after sort."
        )
    if not (np.all(iid[0::2] == 0) and np.all(iid[1::2] == 1)):
        raise ValueError(
            "image_id must be in {0, 1} with 0 → μ_+ and 1 → μ_- per source. "
            "After sorting, expected (0, 1, 0, 1, ...) but got otherwise."
        )

    def _take_plus(arr): return np.asarray(arr)[order][0::2]
    def _take_minus(arr): return np.asarray(arr)[order][1::2]

    # Source-level fields: take the μ_+ value (which by validation equals μ_-)
    m1_src_psrc = _take_plus(m1_src)
    q_src_psrc = _take_plus(q_src)
    z_src_psrc = _take_plus(z_src)
    chieff_psrc = _take_plus(chieff)
    y_source_psrc = _take_plus(y_source)
    p_prop_src_psrc = _take_plus(p_prop_src)
    p_prop_y_psrc = _take_plus(p_prop_y)

    # Validate source-level consistency between paired images
    def _check_consistency(arr, name):
        plus = _take_plus(arr)
        minus = _take_minus(arr)
        if not np.allclose(plus, minus, rtol=1e-10, atol=1e-12):
            n_bad = int(np.sum(~np.isclose(plus, minus, rtol=1e-10, atol=1e-12)))
            raise ValueError(
                f"Source-level field '{name}' inconsistent between μ_+ and "
                f"μ_- images for {n_bad} sources. Check injection writer."
            )
    for arr, name in [
        (m1_src, "m1_src"), (q_src, "q_src"), (z_src, "z_src"),
        (chieff, "chieff"), (y_source, "y_source"),
        (p_prop_src, "p_prop_src"), (p_prop_y, "p_prop_y"),
    ]:
        _check_consistency(arr, name)

    # Image-level fields: keep both
    mu_plus = _take_plus(mu)
    mu_minus = _take_minus(mu)
    det_plus = _take_plus(detected).astype(bool)
    det_minus = _take_minus(detected).astype(bool)

    # Apply both-detected mask
    both_det = det_plus & det_minus
    n_kept = int(both_det.sum())

    # Source IDs of kept sources (from the μ_+ row of each kept pair)
    sid_plus_kept = _take_plus(source_id)[both_det]

    return {
        "source_id": sid_plus_kept.astype(np.int32),
        "m1_src": m1_src_psrc[both_det],
        "q_src": q_src_psrc[both_det],
        "z_src": z_src_psrc[both_det],
        "chieff": chieff_psrc[both_det],
        "y_source": y_source_psrc[both_det],
        "mu_plus": mu_plus[both_det],
        "mu_minus": mu_minus[both_det],
        "p_prop_src": p_prop_src_psrc[both_det],
        "p_prop_y": p_prop_y_psrc[both_det],
        "n_kept": n_kept,
    }


def make_lensed_injection_set(
    source_id: np.ndarray,
    image_id: np.ndarray,
    m1_src: np.ndarray,
    q_src: np.ndarray,
    z_src: np.ndarray,
    chieff: np.ndarray,
    y_source: np.ndarray,
    mu: np.ndarray,
    detected: np.ndarray,
    p_prop_src: np.ndarray,
    p_prop_y: np.ndarray,
    n_draw_sources: int,
) -> LensedInjectionSet:
    """Construct a LensedInjectionSet from raw per-image arrays.

    Parameters
    ----------
    Per-image arrays, see module docstring.
    n_draw_sources
        TOTAL source-frame draws in the campaign, including those where
        neither or only one image was detected. This is the normalization
        denominator for the importance-sampled selection integral —
        getting it wrong rescales the inferred rate by a constant factor.
    """
    grouped = _make_lensed_injection_arrays_per_source(
        source_id=source_id, image_id=image_id,
        m1_src=m1_src, q_src=q_src, z_src=z_src, chieff=chieff,
        y_source=y_source, mu=mu, detected=detected,
        p_prop_src=p_prop_src, p_prop_y=p_prop_y,
    )
    n_kept = grouped["n_kept"]
    valid = np.ones(n_kept, dtype=bool)

    return LensedInjectionSet(
        source_id_kept=jnp.asarray(grouped["source_id"], dtype=jnp.int32),
        m1_src=jnp.asarray(grouped["m1_src"]),
        q_src=jnp.asarray(grouped["q_src"]),
        z_src=jnp.asarray(grouped["z_src"]),
        chieff=jnp.asarray(grouped["chieff"]),
        y_source=jnp.asarray(grouped["y_source"]),
        mu_plus=jnp.asarray(grouped["mu_plus"]),
        mu_minus=jnp.asarray(grouped["mu_minus"]),
        p_prop_src=jnp.asarray(grouped["p_prop_src"]),
        p_prop_y=jnp.asarray(grouped["p_prop_y"]),
        valid=jnp.asarray(valid),
        n_draw_sources=jnp.asarray(n_draw_sources, dtype=jnp.float64),
    )


def load_lensed_injections(path: str) -> LensedInjectionSet:
    """Load a LensedInjectionSet from an HDF5 file.

    Expected layout
    ---------------
        /source_id    (N_img,) int32
        /image_id     (N_img,) int32     0 = μ_+, 1 = μ_-
        /m1_src       (N_img,)
        /q_src        (N_img,)
        /z_src        (N_img,)
        /chieff       (N_img,)
        /y_source     (N_img,)
        /mu           (N_img,)
        /detected     (N_img,) bool
        /p_prop_src   (N_img,)
        /p_prop_y     (N_img,)

        attrs/n_draw_sources  int
    """
    with h5py.File(path, "r") as f:
        out = make_lensed_injection_set(
            source_id=f["source_id"][:],
            image_id=f["image_id"][:],
            m1_src=f["m1_src"][:],
            q_src=f["q_src"][:],
            z_src=f["z_src"][:],
            chieff=f["chieff"][:],
            y_source=f["y_source"][:],
            mu=f["mu"][:],
            detected=f["detected"][:],
            p_prop_src=f["p_prop_src"][:],
            p_prop_y=f["p_prop_y"][:],
            n_draw_sources=int(f.attrs["n_draw_sources"]),
        )
    return out


def save_lensed_injections(
    path: str,
    source_id: np.ndarray,
    image_id: np.ndarray,
    m1_src: np.ndarray,
    q_src: np.ndarray,
    z_src: np.ndarray,
    chieff: np.ndarray,
    y_source: np.ndarray,
    mu: np.ndarray,
    detected: np.ndarray,
    p_prop_src: np.ndarray,
    p_prop_y: np.ndarray,
    n_draw_sources: int,
) -> None:
    """Save raw per-image arrays to disk in the canonical HDF5 layout."""
    with h5py.File(path, "w") as f:
        f.create_dataset("source_id", data=np.asarray(source_id, dtype=np.int32))
        f.create_dataset("image_id", data=np.asarray(image_id, dtype=np.int32))
        f.create_dataset("m1_src", data=np.asarray(m1_src))
        f.create_dataset("q_src", data=np.asarray(q_src))
        f.create_dataset("z_src", data=np.asarray(z_src))
        f.create_dataset("chieff", data=np.asarray(chieff))
        f.create_dataset("y_source", data=np.asarray(y_source))
        f.create_dataset("mu", data=np.asarray(mu))
        f.create_dataset("detected", data=np.asarray(detected, dtype=bool))
        f.create_dataset("p_prop_src", data=np.asarray(p_prop_src))
        f.create_dataset("p_prop_y", data=np.asarray(p_prop_y))
        f.attrs["n_draw_sources"] = int(n_draw_sources)
