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


# Campaigns predating the two-image orientation flag were all rendered with
# independent per-image draws, so a missing attr means 'independent'.
DEFAULT_PAIR_ORIENTATION_MODE = "independent"


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
    log_p_tag_per_source: Any      # (N_kept,) — log pair-tag probability
    snr_image0: Any                # (N_kept,) optional pair-tag field; nan if absent
    snr_image1: Any                # (N_kept,) optional pair-tag field; nan if absent
    delta_t_obs: Any               # (N_kept,) optional pair-tag field; nan if absent
    log_sky_overlap: Any           # (N_kept,) optional pair-tag field; nan if absent
    p_tag_true: Any                # (N_kept,) optional injected tag probability; nan if absent
    tagged_pair: Any               # (N_kept,) optional injected tag boolean; false if absent
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
    log_p_tag_per_source: Optional[np.ndarray] = None,
    p_tag_per_source: Optional[np.ndarray] = None,
    snr_image0: Optional[np.ndarray] = None,
    snr_image1: Optional[np.ndarray] = None,
    delta_t_obs: Optional[np.ndarray] = None,
    true_delta_t: Optional[np.ndarray] = None,
    log_sky_overlap: Optional[np.ndarray] = None,
    p_tag_true: Optional[np.ndarray] = None,
    tagged_pair: Optional[np.ndarray] = None,
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

    def _check_consistency(arr, name):
        plus = _take_plus(arr)
        minus = _take_minus(arr)
        if not np.allclose(plus, minus, rtol=1e-10, atol=1e-12):
            n_bad = int(np.sum(~np.isclose(plus, minus, rtol=1e-10, atol=1e-12)))
            raise ValueError(
                f"Source-level field '{name}' inconsistent between μ_+ and "
                f"μ_- images for {n_bad} sources. Check injection writer."
            )

    # Source-level fields: take the μ_+ value (which by validation equals μ_-)
    m1_src_psrc = _take_plus(m1_src)
    q_src_psrc = _take_plus(q_src)
    z_src_psrc = _take_plus(z_src)
    chieff_psrc = _take_plus(chieff)
    y_source_psrc = _take_plus(y_source)
    p_prop_src_psrc = _take_plus(p_prop_src)
    p_prop_y_psrc = _take_plus(p_prop_y)

    if log_p_tag_per_source is not None and p_tag_per_source is not None:
        raise ValueError("Pass only one of log_p_tag_per_source or p_tag_per_source")
    if p_tag_per_source is not None:
        p_tag_arr = np.asarray(p_tag_per_source, dtype=float)
        if np.any((p_tag_arr < 0.0) | (p_tag_arr > 1.0)):
            raise ValueError("p_tag_per_source must be in [0, 1]")
        log_p_tag_arr = np.full_like(p_tag_arr, -np.inf, dtype=float)
        positive = p_tag_arr > 0.0
        log_p_tag_arr[positive] = np.log(p_tag_arr[positive])
    elif log_p_tag_per_source is not None:
        log_p_tag_arr = np.asarray(log_p_tag_per_source, dtype=float)
    else:
        log_p_tag_arr = None

    if log_p_tag_arr is None:
        log_p_tag_psrc = np.zeros_like(m1_src_psrc, dtype=float)
    elif log_p_tag_arr.shape[0] == n_img:
        _check_consistency(log_p_tag_arr, "log_p_tag_per_source")
        log_p_tag_psrc = _take_plus(log_p_tag_arr)
    elif log_p_tag_arr.shape[0] == n_img // 2:
        # Per-source layout follows sorted source_id order. This is what the
        # mock generator writes; old files omit this dataset and default to 1.
        sid_plus = _take_plus(source_id)
        if not np.array_equal(sid_plus, np.arange(n_img // 2)):
            raise ValueError(
                "per-source p_tag datasets require source_id values 0..N_sources-1"
            )
        log_p_tag_psrc = log_p_tag_arr
    else:
        raise ValueError(
            "log_p_tag_per_source/p_tag_per_source must have length N_sources "
            "or N_img"
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

    def _optional_source_array(arr, *, dtype=float, default=np.nan):
        if arr is None:
            return np.full(n_kept, default, dtype=dtype)
        arr = np.asarray(arr, dtype=dtype)
        if arr.shape[0] == n_img:
            arr = _take_plus(arr)
        elif arr.shape[0] != n_img // 2:
            raise ValueError("optional lensed-injection pair-tag fields must have length N_sources or N_img")
        return arr[both_det]

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
        "log_p_tag_per_source": log_p_tag_psrc[both_det],
        "snr_image0": _optional_source_array(snr_image0),
        "snr_image1": _optional_source_array(snr_image1),
        "delta_t_obs": _optional_source_array(delta_t_obs if delta_t_obs is not None else true_delta_t),
        "log_sky_overlap": _optional_source_array(log_sky_overlap),
        "p_tag_true": _optional_source_array(p_tag_true),
        "tagged_pair": _optional_source_array(tagged_pair, dtype=bool, default=False),
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
    log_p_tag_per_source: Optional[np.ndarray] = None,
    p_tag_per_source: Optional[np.ndarray] = None,
    snr_image0: Optional[np.ndarray] = None,
    snr_image1: Optional[np.ndarray] = None,
    delta_t_obs: Optional[np.ndarray] = None,
    true_delta_t: Optional[np.ndarray] = None,
    log_sky_overlap: Optional[np.ndarray] = None,
    p_tag_true: Optional[np.ndarray] = None,
    tagged_pair: Optional[np.ndarray] = None,
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
        log_p_tag_per_source=log_p_tag_per_source,
        p_tag_per_source=p_tag_per_source,
        snr_image0=snr_image0, snr_image1=snr_image1,
        delta_t_obs=delta_t_obs, true_delta_t=true_delta_t,
        log_sky_overlap=log_sky_overlap, p_tag_true=p_tag_true, tagged_pair=tagged_pair,
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
        log_p_tag_per_source=jnp.asarray(grouped["log_p_tag_per_source"]),
        snr_image0=jnp.asarray(grouped["snr_image0"]),
        snr_image1=jnp.asarray(grouped["snr_image1"]),
        delta_t_obs=jnp.asarray(grouped["delta_t_obs"]),
        log_sky_overlap=jnp.asarray(grouped["log_sky_overlap"]),
        p_tag_true=jnp.asarray(grouped["p_tag_true"]),
        tagged_pair=jnp.asarray(grouped["tagged_pair"]),
        valid=jnp.asarray(valid),
        n_draw_sources=jnp.asarray(n_draw_sources, dtype=jnp.float64),
    )



def _read_n_draw_sources(attrs):
    """Total source-frame draws, accepting either stamped attribute name.

    The module docstring documents ``Ndraw_sources`` and ``preflight._check_lensed``
    validates under either spelling, but the loaders read ``n_draw_sources``
    unconditionally -- so a campaign file written to the documented schema passed
    preflight and then raised KeyError here.  Accept both, and say which names
    were expected when neither is present.
    """
    for key in ("n_draw_sources", "Ndraw_sources"):
        if key in attrs:
            return float(attrs[key])
    raise KeyError(
        "lensed-injection file has neither 'n_draw_sources' nor "
        "'Ndraw_sources' in its attrs; one is required as the source-frame "
        "draw normalisation."
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
        log_p_tag = None
        p_tag = None
        if "log_p_tag_per_source" in f:
            log_p_tag = f["log_p_tag_per_source"][:]
        elif "p_tag_per_source" in f:
            p_tag = f["p_tag_per_source"][:]
        optional = {name: (f[name][:] if name in f else None) for name in ("snr_image0", "snr_image1", "delta_t_obs", "true_delta_t", "log_sky_overlap", "p_tag_true", "tagged_pair")}
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
            n_draw_sources=int(_read_n_draw_sources(f.attrs)),
            log_p_tag_per_source=log_p_tag,
            p_tag_per_source=p_tag,
            **optional,
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
    log_p_tag_per_source: Optional[np.ndarray] = None,
    p_tag_per_source: Optional[np.ndarray] = None,
    snr_image0: Optional[np.ndarray] = None,
    snr_image1: Optional[np.ndarray] = None,
    delta_t_obs: Optional[np.ndarray] = None,
    true_delta_t: Optional[np.ndarray] = None,
    log_sky_overlap: Optional[np.ndarray] = None,
    p_tag_true: Optional[np.ndarray] = None,
    tagged_pair: Optional[np.ndarray] = None,
    snr_model_attrs: Optional[dict] = None,
) -> None:
    """Save raw per-image arrays to disk in the canonical HDF5 layout.

    ``snr_model_attrs`` optionally records the generator's analytic
    detection-model constants (e.g. fc_rho_thr / fc_r0 / fc_mc_bar) as file
    attrs so the lensed-singleton evidence channel can reproduce the exact
    partner-censoring factor (see darksirens.lensing.fcpdet).
    """
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
        if log_p_tag_per_source is not None and p_tag_per_source is not None:
            raise ValueError("Pass only one of log_p_tag_per_source or p_tag_per_source")
        if log_p_tag_per_source is not None:
            f.create_dataset("log_p_tag_per_source", data=np.asarray(log_p_tag_per_source))
            f.attrs["pair_tag_dataset"] = "log_p_tag_per_source"
        elif p_tag_per_source is not None:
            f.create_dataset("p_tag_per_source", data=np.asarray(p_tag_per_source))
            f.attrs["pair_tag_dataset"] = "p_tag_per_source"
        for name, arr in (("snr_image0", snr_image0), ("snr_image1", snr_image1), ("delta_t_obs", delta_t_obs), ("true_delta_t", true_delta_t), ("log_sky_overlap", log_sky_overlap), ("p_tag_true", p_tag_true), ("tagged_pair", tagged_pair)):
            if arr is not None:
                f.create_dataset(name, data=np.asarray(arr, dtype=bool) if name == "tagged_pair" else np.asarray(arr))
        f.attrs["n_draw_sources"] = int(n_draw_sources)
        for key, value in (snr_model_attrs or {}).items():
            f.attrs[key] = float(value)


# ============================================================================
# Exactly-one-detected subset: the lensed-singleton selection channel
# ============================================================================

class LensedSingleImageSet(NamedTuple):
    """Per-source arrays for sources where EXACTLY ONE image was detected.

    Built from the same on-disk per-image layout as LensedInjectionSet —
    the single-image channel is purely a load-time regrouping. ``mu_det``
    is the detected image's magnification, ``mu_partner`` the undetected
    partner's; ``image_is_plus`` records which image was seen.
    """
    m1_src: Any                   # (N_kept,)
    q_src: Any                    # (N_kept,)
    z_src: Any                    # (N_kept,)
    chieff: Any                   # (N_kept,)
    y_source: Any                 # (N_kept,)
    mu_det: Any                   # (N_kept,)
    mu_partner: Any               # (N_kept,)
    image_is_plus: Any            # (N_kept,) bool
    p_prop_src: Any               # (N_kept,)
    p_prop_y: Any                 # (N_kept,)
    valid: Any                    # (N_kept,) bool
    n_draw_sources: Any           # scalar

    @property
    def n_kept(self):
        return int(self.m1_src.shape[0])


def read_pair_orientation_mode(path: str) -> str:
    """The two-image orientation convention the campaign was RENDERED with.

    Single source of truth for the file attr and its legacy default, shared by
    the lensing CLI's startup gate and by preflight so the two cannot disagree
    about what an attr-less campaign means.
    """
    with h5py.File(path, "r") as f:
        return str(
            f.attrs.get("pair_orientation_mode", DEFAULT_PAIR_ORIENTATION_MODE)
        )


def pair_orientation_mismatch_message(path, campaign_mode, run_mode) -> str:
    """Why a campaign/runtime orientation mismatch is not a cosmetic warning."""
    return (
        f"--pair_orientation_mode {run_mode!r} but the lensed-injection campaign "
        f"{path!r} was rendered with {campaign_mode!r}. The pair efficiencies "
        "(the campaign's rendered detection flags) and the analytic "
        "lensed-singleton censoring factor then follow DIFFERENT orientation "
        "conventions, so the J=2/lensed-singleton channel ratio that sets A_tau "
        "is mis-normalised by up to ~2.6x (see darksirens.lensing.fcpdet). "
        "Re-render the campaign with --pair-orientation-mode "
        f"{run_mode!r}, or run with --pair_orientation_mode {campaign_mode!r}."
    )


def load_lensed_single_image_set(
    path: str, *, return_campaign_attrs: bool = False
):
    """Load the exactly-one-detected per-source subset from an injection file.

    With ``return_campaign_attrs`` the loader also returns the campaign's
    rendering conventions as a plain dict (currently
    ``pair_orientation_mode``).  They are deliberately NOT fields of
    ``LensedSingleImageSet``: that NamedTuple is passed to
    ``darksiren_log_likelihood_with_clusters`` as a traced JIT argument, and a
    string leaf in the pytree is not a valid JAX type.  The subset view used to
    drop the attr entirely, which is how a campaign/runtime orientation
    mismatch could reach the likelihood unnoticed.
    """
    with h5py.File(path, "r") as f:
        source_id = np.asarray(f["source_id"][:])
        image_id = np.asarray(f["image_id"][:])
        order = np.lexsort((image_id, source_id))
        sid = source_id[order]
        iid = image_id[order]
        if len(sid) % 2 != 0 or not np.all(sid[0::2] == sid[1::2]):
            raise ValueError("Each source_id must appear exactly twice.")
        if not (np.all(iid[0::2] == 0) and np.all(iid[1::2] == 1)):
            raise ValueError("image_id must be {0: mu_+, 1: mu_-} per source.")

        def plus(name):
            return np.asarray(f[name][:])[order][0::2]

        def minus(name):
            return np.asarray(f[name][:])[order][1::2]

        det_p = plus("detected").astype(bool)
        det_m = minus("detected").astype(bool)
        one_det = det_p ^ det_m
        image_is_plus = det_p[one_det]
        mu_p = plus("mu")[one_det]
        mu_m = minus("mu")[one_det]
        campaign_attrs = {
            "pair_orientation_mode": str(
                f.attrs.get(
                    "pair_orientation_mode", DEFAULT_PAIR_ORIENTATION_MODE
                )
            ),
        }
        singles = LensedSingleImageSet(
            m1_src=jnp.asarray(plus("m1_src")[one_det]),
            q_src=jnp.asarray(plus("q_src")[one_det]),
            z_src=jnp.asarray(plus("z_src")[one_det]),
            chieff=jnp.asarray(plus("chieff")[one_det]),
            y_source=jnp.asarray(plus("y_source")[one_det]),
            mu_det=jnp.asarray(np.where(image_is_plus, mu_p, mu_m)),
            mu_partner=jnp.asarray(np.where(image_is_plus, mu_m, mu_p)),
            image_is_plus=jnp.asarray(image_is_plus),
            p_prop_src=jnp.asarray(plus("p_prop_src")[one_det]),
            p_prop_y=jnp.asarray(plus("p_prop_y")[one_det]),
            valid=jnp.ones(int(one_det.sum()), dtype=bool),
            n_draw_sources=jnp.asarray(_read_n_draw_sources(f.attrs), dtype=jnp.float64),
        )
    return (singles, campaign_attrs) if return_campaign_attrs else singles


def read_fc_pdet_attrs(path: str) -> dict:
    """Read the generator's Finn-Chernoff detection-model attrs, if present."""
    with h5py.File(path, "r") as f:
        out = {}
        for key in ("fc_rho_thr", "fc_r0", "fc_mc_bar"):
            if key in f.attrs:
                out[key] = float(f.attrs[key])
        return out
