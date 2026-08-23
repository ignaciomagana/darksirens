#!/usr/bin/env python3
"""darksirens_analyze — post-process and plot one or more inference runs.

Reads the current ``results.hdf5`` save format (falling back to the numeric
``samples.npy`` crash-recovery chain), recomputes posterior-predictive
population distributions, and makes publication-quality figures:

  * 1-D posterior-predictive spectra p(m1), p(m2), p(q), p(z), p(chi)
  * 2-D joint p(m1, m2)
  * cosmology posteriors (H0 / Om0 / w0 / wa) with the injected value marked
  * the redshift distribution / detection rate dN/dz
  * a compact GP-latent summary (for GP / gppop models)
  * model comparison: relative evidences + pairwise Bayes factors

The population density is evaluated on a flattened, equal-shape coordinate grid
so the recompute works for every model type, including the GP / binned-GP
(gppop) models (which require equal-shape arguments to ``log_p_pop``).
"""
# ── JAX runtime configuration (before ANY JAX-dependent import) ───────────────
# x64 has to be on before the first JAX-dependent import, not after it: the
# population registry builds MASS_GRID / Q_GRID / CHI_GRID / M1_MESH and their
# lru_cached getters AT IMPORT TIME, and those arrays keep whatever precision
# was configured when they were built.  Importing jax (and the population
# modules) first and enabling x64 later left every one of them float32 for the
# life of the process -- MEASURED in a clean subprocess: x64 False before the
# import, True after it, and get_mass_grid() still float32 -- which violates the
# package's x64 contract and cost ~1e-8 in recomputed log densities.  The
# inference CLIs already configure first; this one now matches them.  The
# pytest conftest enables x64 before collection, so no test could have caught it.
from darksirens.core.jax_config import configure_jax_runtime  # noqa: E402

configure_jax_runtime()

import os
import json
import argparse
import warnings

import numpy as np
import h5py
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import lax

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from darksirens.cli.common import _banner, _section, _row, _end, _ok, _warn
from darksirens.gw.populations import pop_model_parser
from darksirens.inference.pop_extractor import make_pop_extractor
from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID
from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.plotting import (
    set_publication_style,
    latent_indices,
    make_latent_summary,
)

set_publication_style()
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
_PALETTE = sns.color_palette('colorblind')

COSMO_FID = {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID}
COSMO_TEX = {
    "H0": r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]",
    "Om0": r"$\Omega_m$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}


# ------------------------------------------------------------
# I/O — current HDF5 format with a legacy NPY fallback
# ------------------------------------------------------------
def _json_safe_hdf5_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_hdf5_attr(item) for item in value.tolist()]
    return value


def _decode_labels(raw):
    return [lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl) for lbl in raw]


def _load_settings(run_dir):
    settings_path = os.path.join(run_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return json.load(f)
    return {}


def _merge_hdf5_metadata(settings, h5_file):
    """Backfill settings from results.hdf5 attrs/datasets when missing from JSON."""
    merged = dict(settings)

    for key, value in h5_file.attrs.items():
        if key in {"environment", "prior_overrides"} and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        merged.setdefault(key, _json_safe_hdf5_attr(value))

    if "labels" not in merged and "labels" in h5_file:
        merged["labels"] = _decode_labels(h5_file["labels"][()])
    if "lower_bound" not in merged and "lower_bound" in h5_file:
        merged["lower_bound"] = h5_file["lower_bound"][()].astype(float).tolist()
    if "upper_bound" not in merged and "upper_bound" in h5_file:
        merged["upper_bound"] = h5_file["upper_bound"][()].astype(float).tolist()

    if (
        not merged.get("fixed_parameter_values")
        and "fixed_labels" in h5_file
        and "fixed_values" in h5_file
    ):
        fixed_labels = _decode_labels(h5_file["fixed_labels"][()])
        fixed_values = h5_file["fixed_values"][()]
        merged["fixed_parameter_values"] = {
            label: float(value) for label, value in zip(fixed_labels, fixed_values)
        }

    return merged


def _load_run_hdf5(run_dir):
    path = os.path.join(run_dir, "results.hdf5")
    settings = _load_settings(run_dir)

    with h5py.File(path, "r") as f:
        # Samples: a top-level dataset (current layout), else nested under a
        # "posterior" group (grouped layout).
        if "samples" in f:
            samples = f["samples"][()]
        elif "posterior" in f and "samples" in f["posterior"]:
            samples = f["posterior"]["samples"][()]
        else:
            raise KeyError(
                f"{path} does not contain a 'samples' dataset "
                "(checked the root and the 'posterior' group)"
            )
        settings = _merge_hdf5_metadata(settings, f)
        # Evidence: canonical attr names with documented aliases.
        logZ = f.attrs.get("logZ", f.attrs.get("log_evidence", None))
        logZerr = f.attrs.get("logZerr", f.attrs.get("log_evidence_err", None))

    if logZ is not None:
        logZ = float(logZ)
    if logZerr is not None:
        logZerr = float(logZerr)
        if np.isnan(logZerr):
            logZerr = None

    return settings, samples, logZ, logZerr


def _load_run_npy(run_dir, allow_legacy_pickle=False):
    """Read the ``samples.npy`` crash-recovery chain (or a legacy pickled dict).

    Both inference CLIs write ``samples.npy`` *before* results.hdf5 as a bare
    numeric (nsamples, ndim) recovery matrix, so that is the normal path here:
    labels/bounds come from settings.json and there is no evidence to report.
    Very old runs instead stored a pickled results dict; reading one deserializes
    arbitrary Python, so it takes the explicit ``--allow_legacy_pickle`` opt-in.
    """
    path = os.path.join(run_dir, "samples.npy")
    settings = _load_settings(run_dir)
    try:
        samples = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "allow_pickle=False" not in str(exc):
            raise
        if not allow_legacy_pickle:
            raise ValueError(
                f"{path} is a legacy pickled results dict, not the numeric "
                "recovery matrix written by the current CLIs. Reading it "
                "executes arbitrary pickle; pass --allow_legacy_pickle to "
                "darksirens_analyze if you trust this file."
            ) from exc
        warnings.warn(
            f"Unpickling legacy {path} because --allow_legacy_pickle was "
            "passed; this executes arbitrary code from the file.",
            RuntimeWarning,
            stacklevel=2,
        )
        results = np.load(path, allow_pickle=True).item()
        return settings, results["samples"], results.get("logZ"), results.get("logZerr")

    samples = np.asarray(samples)
    if samples.dtype.kind not in ("i", "u", "f") or samples.ndim != 2:
        raise ValueError(
            f"{path} holds a {samples.ndim}-D array of dtype {samples.dtype!s}; "
            "expected the numeric (nsamples, ndim) recovery matrix."
        )
    # No evidence is stored alongside the chain: report it as absent rather
    # than letting a caller mistake a stale/zero value for a real logZ.
    return settings, samples, None, None


def load_run(run_dir, allow_legacy_pickle=False):
    """Load a completed inference run (current HDF5, else samples.npy)."""
    if os.path.exists(os.path.join(run_dir, "results.hdf5")):
        return _load_run_hdf5(run_dir)
    if os.path.exists(os.path.join(run_dir, "samples.npy")):
        return _load_run_npy(run_dir, allow_legacy_pickle=allow_legacy_pickle)
    raise FileNotFoundError(
        f"No inference results found in {run_dir!r}; expected 'results.hdf5' "
        "(current format) or 'samples.npy' (chain-only recovery / legacy format)."
    )


def _labels_of(settings):
    return [str(lbl) for lbl in settings.get("labels", [])]


def _column(samples, labels, name):
    """Posterior samples of parameter ``name`` (or None if it was fixed)."""
    if name in labels:
        return np.asarray(samples)[:, labels.index(name)]
    return None


# ------------------------------------------------------------
# Vectorized batched map ("jmap"): jit + vmap + automatic batch size
# ------------------------------------------------------------
# probe_device_memory_bytes now lives in darksirens.likelihood.block_sizing
# (shared with the CLI block-size auto-sizer); re-exported here so existing
# imports (e.g. tests/test_analyze_ppd_chunking.py) keep resolving unchanged.
from darksirens.likelihood.block_sizing import probe_device_memory_bytes


def plan_ppd_sizing(nsamples, n_outer, nz, nchi, n_nodes, max_mem_bytes,
                    safe_frac=0.4, batch_size=None, grid_chunk=None,
                    dtype_bytes=8, n_live=16, batch_cap=64, slab_floor=8,
                    full_grid_max=200_000):
    """Choose ``(batch_size, slab_rows)`` for the slab-accumulation evaluator.

    The density is integrated over ``(z, chi)`` in slabs of ``slab_rows`` rows of
    the flattened ``(m1, q)`` outer plane (``n_outer = nm*nq``); the per-step
    working set is ``~ batch * slab_rows * nz * nchi * dtype * (n_nodes +
    n_live)`` (the density slab + the GP kernel ``(pts x n_nodes)`` + a few
    trapezoid buffers).  The marginals are tiny running accumulators, so the
    full 4-D grid is never materialised and the memory model is a small,
    *predictable* constant — unlike the legacy full-grid path, whose many
    simultaneously-live grid-sized reduction buffers are what OOMs.

    Returns ``slab_rows=None`` for small grids (``grid_points <=
    full_grid_max``) *whose full plane also fits the budget* so the caller takes
    the cheaper full-grid path; otherwise a concrete slab size.  The size test
    matters for the GP models: at ``n_nodes = 6400`` even a 65k-point grid needs
    215 GB at ``batch = 64``, so a grid-size-only shortcut would OOM on exactly
    the model class this planner exists for.  ``batch_size`` / ``grid_chunk``
    overrides are honoured (``grid_chunk`` sets ``slab_rows`` directly).
    """
    budget = max(1.0, float(safe_frac) * float(max_mem_bytes))
    n_outer = max(1, int(n_outer))
    plane = max(1, int(nz) * int(nchi))
    grid_points = n_outer * plane
    nn = max(0, int(n_nodes))

    if batch_size is not None:
        b = max(1, int(batch_size))
    else:
        b = max(1, min(batch_cap, int(nsamples)))

    if grid_chunk is not None:
        if int(grid_chunk) >= n_outer:
            # Clamping silently to n_outer IS the full-grid path, i.e. chunking
            # off -- the opposite of what a user passing this flag wants.  It is
            # the usual symptom of reading the value as flattened 4-D grid points
            # (nm*nq*nz*nchi) rather than outer-plane rows (nm*nq).
            _warn(f"--grid_chunk={int(grid_chunk)} >= nm*nq={n_outer}: one slab "
                  "covers the whole (m1, q) plane, so chunking is disabled. The "
                  "flag counts ROWS of that plane, not flattened 4-D grid points.")
        return b, max(1, min(int(grid_chunk), n_outer))

    def _slab_for(bb):
        return int(budget // max(bb * plane * dtype_bytes * (nn + n_live), 1))

    c = _slab_for(b)
    while c < slab_floor and b > 1:          # GP eval term too costly at this batch
        b = max(1, b // 2)
        c = _slab_for(b)
    if grid_points <= full_grid_max and c >= n_outer:
        return b, None                       # small grid AND full plane in budget
    c = max(1, min(c, n_outer))
    return b, c


def gp_node_count(pop_model):
    """Total inducing-node count behind a bound ``log_p_pop`` (0 for parametric).

    ``JointGPPopulation`` / ``BinnedGPPopulation`` expose one flat node block as
    ``M``; the additive family (``gp_separable``, ``gp4d_additive``) carries one
    block per ANOVA term in ``_term_meta`` and has no top-level ``M``, so sum
    those — reading ``M`` alone silently sizes the PPD slab as if the per-query
    kernel tensor did not exist (~18x too large for ``gp4d_additive``).
    """
    model = getattr(pop_model, "__self__", None)
    if model is None:
        return 0
    M = int(getattr(model, "M", 0) or 0)
    if M:
        return M
    return int(sum(int(t["M"]) for t in getattr(model, "_term_meta", ()) or ()))


def batched_map(fn, samples, batch_size):
    """Apply ``fn`` to each row of ``samples`` via jit(vmap), batched.

    Pads the final batch so every call shares one compiled shape, then trims.
    Returns a pytree (matching ``fn``'s output) stacked over all samples.
    """
    samples = jnp.asarray(samples)
    ns = samples.shape[0]
    pad = (-ns) % batch_size
    if pad:
        samples = jnp.concatenate([samples, jnp.repeat(samples[-1:], pad, axis=0)], axis=0)

    vfn = jax.jit(jax.vmap(fn))
    # Each batch is pulled to the host as it completes and the stack is built with
    # numpy: keeping the per-batch outputs on device and concatenating there held
    # three simultaneous device copies of the FULL per-sample stack at the seam
    # (p_m1m2 alone is nsamples*nm*nm*8 bytes -- ~6.5 GB for a 50k-sample chain at
    # --nm 128, ~20 GB tripled), none of which plan_ppd_sizing's per-step slab
    # budget accounts for.
    outs = [
        jax.tree_util.tree_map(np.asarray, vfn(samples[i:i + batch_size]))
        for i in tqdm(range(0, samples.shape[0], batch_size),
                      desc="Posterior-predictive batches")
    ]
    stacked = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *outs)
    return jax.tree_util.tree_map(lambda a: a[:ns], stacked)


# ------------------------------------------------------------
# Posterior-predictive engine (flattened grid → works for all models)
# ------------------------------------------------------------
def make_single_theta_predictive(pop_model, settings, mgrid, qgrid, zgrid, chigrid,
                                  grid_chunk=None):
    """Build a jit'd per-sample evaluator returning the PPD marginals + p(m1,m2).

    The density is integrated over ``(z, chi)`` to three small arrays — the
    inner integrals

        I_c  = ∫ p dchi        shape (nm, nq, nz)
        I_zc = ∫∫ p dz dchi     shape (nm, nq)
        I_z  = ∫ p dz          shape (nm, nq, nchi)

    from which every output marginal is a trivial outer ``(m1, q)`` integral.
    All six outputs are bit-equivalent to the legacy full-tensor marginalization
    (same integration order, up to floating-point reassociation).

    ``grid_chunk`` bounds peak memory: when set, the ``(m1, q)`` outer plane is
    processed in slabs of ``grid_chunk`` rows via ``lax.scan`` (mirroring
    ``inference/selection.py``), so the density slab ``(slab, nz, nchi)`` and the
    GP kernel ``(slab*nz*nchi, n_nodes)`` are the only large transients — the
    full 4-D grid is never held.  ``grid_chunk=None`` keeps the simple full-grid
    path (used for small grids, where it is cheap).
    """
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)
    chigrid = jnp.asarray(chigrid)
    nm, nq, nz, nchi = mgrid.size, qgrid.size, zgrid.size, chigrid.size

    # Flattened (m1, q) outer plane: row r = i_m*nq + i_q.
    n_outer = nm * nq
    m1_row = jnp.repeat(mgrid, nq)
    q_row = jnp.tile(qgrid, nm)

    if grid_chunk is not None:
        C = max(1, int(grid_chunk))
        n_slabs = (n_outer + C - 1) // C
        Pc = n_slabs * C
        pad = Pc - n_outer
        if pad:
            _ext = lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], pad)])
            m1_row_p, q_row_p = _ext(m1_row), _ext(q_row)
        else:
            m1_row_p, q_row_p = m1_row, q_row

    # q→m2 interpolation geometry for the 2-D joint.
    q_eval = mgrid[None, :] / mgrid[:, None]
    valid = (q_eval >= qgrid[0]) & (q_eval <= qgrid[-1])
    jac = 1.0 / mgrid[:, None]

    pop_extractor = make_pop_extractor(settings)

    def _slab_integrals(m1c, qc, pop_theta):
        """Integrate the density over (z, chi) for a slab of outer rows.

        ``m1c``/``qc`` are (S,) outer-row coordinates; returns
        ``(I_c (S,nz), I_zc (S,), I_z (S,nchi))``.  Built from the unbatched grid
        constants and broadcast against the sampled ``pop_theta`` — the outer
        vmap tracks the sample axis through the fixed-shape reshapes.
        """
        S = m1c.shape[0]
        shp = (S, nz, nchi)
        m1f = jnp.broadcast_to(m1c[:, None, None], shp).reshape(S * nz * nchi)
        qf = jnp.broadcast_to(qc[:, None, None], shp).reshape(S * nz * nchi)
        zf = jnp.broadcast_to(zgrid[None, :, None], shp).reshape(S * nz * nchi)
        chif = jnp.broadcast_to(chigrid[None, None, :], shp).reshape(S * nz * nchi)
        p = jnp.exp(pop_model(m1f, qf, zf, chif, pop_theta)).reshape(shp)
        I_c = jnp.trapezoid(p, chigrid, axis=2)      # (S, nz)
        I_zc = jnp.trapezoid(I_c, zgrid, axis=1)     # (S,)
        I_z = jnp.trapezoid(p, zgrid, axis=1)        # (S, nchi)
        return I_c, I_zc, I_z

    @jax.jit
    def single_theta(theta):
        pop_theta = pop_extractor(theta)
        if grid_chunk is None:
            I_c, I_zc, I_z = _slab_integrals(m1_row, q_row, pop_theta)
        else:
            def _body(_carry, k):
                s = k * C
                sl = lambda a: lax.dynamic_slice_in_dim(a, s, C)
                return _carry, _slab_integrals(sl(m1_row_p), sl(q_row_p), pop_theta)
            _, (I_c_s, I_zc_s, I_z_s) = lax.scan(_body, None, jnp.arange(n_slabs))
            # Concatenate slabs and trim padding (fixed-shape reshapes keep the
            # outer vmap valid).
            I_c = I_c_s.reshape(Pc, nz)[:n_outer]
            I_zc = I_zc_s.reshape(Pc)[:n_outer]
            I_z = I_z_s.reshape(Pc, nchi)[:n_outer]
        I_c = I_c.reshape(nm, nq, nz)
        I_zc = I_zc.reshape(nm, nq)
        I_z = I_z.reshape(nm, nq, nchi)

        # 1-D marginals: outer (m1, q) integrals of the inner (z, chi) integrals.
        # Every normalisation is guarded like the 2-D joint below: a sample whose
        # density integrates to exactly 0 on this grid (a constraint-violating
        # parametric draw, an underflowed GP field, or a --mmin/--mmax/--zmax
        # window that misses the model support) would otherwise give 0/0 = NaN,
        # and jnp.median/jnp.percentile in summarize_ppd propagate NaN — one bad
        # draw out of thousands would blank the whole credible band, silently.
        def _normed(p, grid):
            """Normalise a 1-D marginal, leaving an all-zero curve alone."""
            z = jnp.trapezoid(p, grid)
            return p / jnp.where(z > 0, z, 1.0)

        p_m1q = I_zc                                          # ∫∫ p dz dchi
        p_m1 = _normed(jnp.trapezoid(p_m1q, qgrid, axis=1), mgrid)
        p_q = _normed(jnp.trapezoid(p_m1q, mgrid, axis=0), qgrid)
        p_z = _normed(jnp.trapezoid(jnp.trapezoid(I_c, qgrid, axis=1), mgrid, axis=0),
                      zgrid)
        p_chi = _normed(jnp.trapezoid(jnp.trapezoid(I_z, qgrid, axis=1), mgrid, axis=0),
                        chigrid)

        # 2-D joint p(m1, m2): map q→m2 on p(m1, q).
        p_interp = jax.vmap(
            lambda row, qev: jnp.interp(qev, qgrid, row, left=0.0, right=0.0)
        )(p_m1q, q_eval)
        p_m1m2 = p_interp * jac * valid
        norm_2d = jnp.trapezoid(jnp.trapezoid(p_m1m2, mgrid, axis=0), mgrid, axis=0)
        p_m1m2 = jnp.where(norm_2d > 0, p_m1m2 / norm_2d, p_m1m2)
        p_m2 = _normed(jnp.trapezoid(p_m1m2, mgrid, axis=0), mgrid)

        return p_m1, p_m2, p_q, p_z, p_chi, p_m1m2

    return single_theta


def posterior_predictive(pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
                         batch_size=None, grid_chunk=None, max_mem_gb=None,
                         mem_safe_frac=0.4):
    """Return per-sample (p_m1, p_m2, p_q, p_z, p_chi, p_m1m2) stacks.

    Sample-batch and grid-chunk sizes are chosen automatically from probed
    device memory (``probe_device_memory_bytes`` / ``plan_ppd_sizing``) so the
    recompute does not OOM for any model — in particular the GP / binned-GP
    models, whose per-query kernel tensor scales with the number of inducing
    nodes.  ``batch_size`` / ``grid_chunk`` / ``max_mem_gb`` override the
    automatic choices.
    """
    samples = jnp.asarray(samples)
    nm, nq, nz, nchi = mgrid.size, qgrid.size, zgrid.size, chigrid.size
    n_outer = int(nm * nq)
    grid_points = int(nm * nq * nz * nchi)
    # GP inducing-node count (drives the (pts x M) eval cost); 0 for parametric.
    n_nodes = gp_node_count(pop_model)

    if max_mem_gb is not None:
        max_mem_bytes, mem_src = float(max_mem_gb) * 1e9, "cli:--max_mem_gb"
    elif os.environ.get("DARKSIRENS_ANALYZE_MAX_MEM_GB"):
        max_mem_bytes = float(os.environ["DARKSIRENS_ANALYZE_MAX_MEM_GB"]) * 1e9
        mem_src = "env:DARKSIRENS_ANALYZE_MAX_MEM_GB"
    else:
        max_mem_bytes, mem_src = probe_device_memory_bytes()

    batch_size, grid_chunk = plan_ppd_sizing(
        samples.shape[0], n_outer, nz, nchi, n_nodes, max_mem_bytes,
        safe_frac=mem_safe_frac, batch_size=batch_size, grid_chunk=grid_chunk,
    )
    slab_pts = (grid_chunk if grid_chunk is not None else n_outer) * nz * nchi
    est_peak_gb = batch_size * slab_pts * 8 * (n_nodes + 16) / 1e9
    # The accumulated per-sample outputs are a second, independent cost: they live
    # on the HOST (batched_map moves each batch off device), but they are what the
    # run actually has to fit in RAM, so report them rather than leaving the
    # printed peak an order of magnitude below the real footprint.
    out_gb = samples.shape[0] * (2 * nm + nq + nz + nchi + nm * nm) * 8 / 1e9
    print(
        f"  │  [ppd] grid={grid_points:,} pts  GP_nodes={n_nodes}  "
        f"mem={max_mem_bytes / 1e9:.1f}GB ({mem_src})  batch={batch_size}  "
        f"slab_rows={grid_chunk if grid_chunk is not None else 'full'}  "
        f"~peak={est_peak_gb:.1f}GB device + {out_gb:.1f}GB host (outputs)"
    )

    single_theta = make_single_theta_predictive(
        pop_model, settings, mgrid, qgrid, zgrid, chigrid, grid_chunk=grid_chunk
    )
    return batched_map(single_theta, samples, batch_size)


def summarize_ppd(ppd_samples, limits=(5, 95)):
    lo, hi = limits
    median = jnp.median(ppd_samples, axis=0)
    lower = jnp.percentile(ppd_samples, lo, axis=0)
    upper = jnp.percentile(ppd_samples, hi, axis=0)
    return np.asarray(median), np.asarray(lower), np.asarray(upper)


def redshift_rate_samples(pz_samples, zgrid, h0_samples,
                          om0_samples=OM0_FID, w0_samples=W0_FID, wa_samples=WA_FID):
    """dN/dz ∝ p_z(z) · (dV_c/dz)(z; H0, Om0, w0, wa), normalised per sample.

    NO explicit 1/(1+z) here: the detector-frame time dilation is ALREADY inside
    ``p_z``.  ``p_z`` is the posterior-predictive marginal of the population
    density, and :meth:`PopulationModel.log_rate_z` returns ``(gamma - 1) *
    log1p(z)`` — i.e. the merger-rate density R(z) ∝ (1+z)^gamma divided by
    (1+z) — for the ``powerlaw`` evolution, and ``log[psi(z)/(1+z)]`` for
    ``md``.  So ``p_z ∝ R(z)/(1+z)`` and the observed rate is simply
    ``p_z · dV_c/dz``.  Dividing again would plot (1+z)^(gamma-2) dV_c/dz and
    bias the curve low in redshift (median z shifts by -7% to -22% over
    gamma = 4 to 1).

    ``om0_samples`` / ``w0_samples`` / ``wa_samples`` accept either a per-sample
    array (same length as ``h0_samples``) or a scalar broadcast across every
    sample (the fiducial defaults, kept for backward compatibility with
    H0-only callers). All four cosmology coordinates are vmapped through the
    full ``dV_of_z`` so a sampled Om0/w0/wa actually reshapes the rate curve,
    not just its (normalization-cancelling) H0 rescaling.
    """
    zg = jnp.asarray(zgrid)
    n = jnp.asarray(h0_samples).shape[0]
    om0_samples = jnp.broadcast_to(jnp.asarray(om0_samples), (n,))
    w0_samples = jnp.broadcast_to(jnp.asarray(w0_samples), (n,))
    wa_samples = jnp.broadcast_to(jnp.asarray(wa_samples), (n,))

    def one(pz, h0, om0, w0, wa):
        rate = pz * dV_of_z(zg, h0, om0, w0, wa)
        norm = jnp.trapezoid(rate, zg)
        return rate / jnp.where(norm > 0, norm, 1.0)

    return np.asarray(jax.vmap(one)(
        jnp.asarray(pz_samples), jnp.asarray(h0_samples), om0_samples, w0_samples, wa_samples))


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_1d_spectrum(xgrid, summaries, labels, xlabel, ylabel,
                     xlim=None, ylim=None, logy=True, figsize=(16, 9)):
    xgrid = np.asarray(xgrid)
    fig, ax = plt.subplots(figsize=figsize)
    for i, (median, lower, upper) in enumerate(summaries):
        color = _PALETTE[i % len(_PALETTE)]
        ax.fill_between(xgrid, lower, upper, alpha=0.18, color=color, lw=0, label=labels[i])
        ax.plot(xgrid, median, color=color, lw=2.5)
        ax.plot(xgrid, lower, color=color, lw=0.8, alpha=0.6)
        ax.plot(xgrid, upper, color=color, lw=0.8, alpha=0.6)

    ax.set_xlim(*(xlim or (xgrid.min(), xgrid.max())))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, frameon=False)
    fig.tight_layout()
    return fig


def plot_2d_mass(mgrid, p_m1m2_per_model, labels, figsize=None):
    """Median joint p(m1, m2) as filled contours, one panel per model.

    ``p_m1m2_per_model`` holds one ALREADY-MEDIANED ``(nm, nm)`` array per model:
    the median is all this plot uses, and keeping the full
    ``(nsamples, nm, nm)`` stack per model resident instead costs
    ``nsamples * nm**2 * 8`` bytes per model (~6.5 GB for a 50k-sample chain at
    --nm 128) for nothing.
    """
    mgrid = np.asarray(mgrid)
    n = len(labels)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    figsize = figsize or (6.0 * ncol, 5.5 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    M1, M2 = np.meshgrid(mgrid, mgrid, indexing="ij")
    for k, (label, med) in enumerate(zip(labels, p_m1m2_per_model)):
        ax = axes[k // ncol][k % ncol]
        masked = np.where(M2 <= M1, np.asarray(med), np.nan)
        levels = np.linspace(np.nanmax(masked) * 1e-3, np.nanmax(masked), 12) \
            if np.nanmax(masked) > 0 else None
        cf = ax.contourf(M1, M2, masked, levels=levels, cmap="viridis")
        ax.plot([mgrid.min(), mgrid.max()], [mgrid.min(), mgrid.max()], 'w--', lw=1, alpha=0.6)
        ax.set_xlabel(r"$m_1$ [$M_\odot$]", fontsize=22)
        ax.set_ylabel(r"$m_2$ [$M_\odot$]", fontsize=22)
        ax.set_title(label, fontsize=20)
        fig.colorbar(cf, ax=ax, label=r"$p(m_1, m_2)$")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    return fig


def plot_cosmology_posterior(cosmo_samples_per_model, labels, params, figsize=None):
    """Overlay 1-D cosmology posteriors per model; mark the injected value."""
    params = [p for p in params if any(p in cs and cs[p] is not None
                                       for cs in cosmo_samples_per_model)]
    if not params:
        return None
    n = len(params)
    figsize = figsize or (6.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    for a, param in enumerate(params):
        ax = axes[0][a]
        for i, cs in enumerate(cosmo_samples_per_model):
            vals = cs.get(param)
            if vals is None:
                continue
            color = _PALETTE[i % len(_PALETTE)]
            ax.hist(np.asarray(vals), bins=40, density=True, histtype="stepfilled",
                    alpha=0.25, color=color)
            ax.hist(np.asarray(vals), bins=40, density=True, histtype="step",
                    color=color, lw=2.0, label=labels[i])
        if param in COSMO_FID:
            ax.axvline(COSMO_FID[param], color="k", ls="--", lw=1.5, label="injected")
        ax.set_xlabel(COSMO_TEX.get(param, param), fontsize=24)
        ax.set_ylabel("posterior density", fontsize=22)
        ax.tick_params(labelsize=18)
        if a == 0:
            ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    return fig


def plot_catalog_weight_posteriors(weights, is_field=True, figsize=None):
    """Overlay the per-catalog mixture-weight posteriors ``w_1 .. w_K``.

    ``weights`` is the ``(n_samples, K)`` output of
    :func:`darksirens.inference.pop_extractor.catalog_sticks_to_weights`.
    Under the FIELD sky-weighting convention ``w_k`` is catalog ``k``'s GW
    host fraction (``w_2`` = ``f_agn`` for the canonical GAL+AGN K=2 run).
    """
    weights = np.asarray(weights)
    n_cat = weights.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=figsize or (7.0, 5.0))
    for k in range(n_cat):
        color = _PALETTE[k % len(_PALETTE)]
        ax.hist(weights[:, k], bins=40, range=(0.0, 1.0), density=True,
                histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(weights[:, k], bins=40, range=(0.0, 1.0), density=True,
                histtype="step", color=color, lw=2.0, label=f"$w_{{{k + 1}}}$")
    xlabel = ("catalog host fraction $w_k$" if is_field
              else "catalog weight $w_k$ (conditional: z-shape preference)")
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel("posterior density", fontsize=22)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    return fig


def plot_model_evidences(labels, log10Zs, log10Zerrs, figsize=(10, 6)):
    log10Zs = np.asarray(log10Zs, dtype=float)
    errs = np.asarray([0.0 if e is None else e for e in log10Zerrs], dtype=float)
    # The bars are log Bayes factors AGAINST THE BEST MODEL, so the uncertainty is
    # the one on the difference — hypot(err_i, err_best), the same combination
    # plot_bayes_factor_matrix uses — not the single-model error on log10 Z.  The
    # reference model's own bar is exactly 0 and carries no error.
    best = int(np.argmax(log10Zs))
    delta = log10Zs - log10Zs[best]
    delta_err = np.hypot(errs, errs[best])
    delta_err[best] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(labels))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    ax.bar(xs, delta, yerr=delta_err, color=colors, alpha=0.85, capsize=5)
    ax.axhline(0.0, color="black", lw=1.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=16)
    ax.set_ylabel(r"$\Delta \log_{10} Z$ (relative to best)", fontsize=20)
    ax.set_title("Model comparison", fontsize=22)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    return fig


def print_bayes_factors(labels, log10Zs):
    """Print pairwise Bayes-factor rows (framed by the caller's section)."""
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if log10Zs[i] is not None and log10Zs[j] is not None:
                _row(f"{labels[i]} vs {labels[j]}",
                     f"log10 BF = {log10Zs[i] - log10Zs[j]:.3f}", width=40)


def _should_plot_bayes_factor_matrix(labels, log10Zs):
    """Whether a pairwise Bayes-factor matrix is meaningful.

    Requires at least two models AND at least two present evidences (a pairwise
    matrix needs >=2 comparable ``logZ`` values; ``None`` entries are runs with
    no evidence estimate).
    """
    return len(labels) >= 2 and sum(z is not None for z in log10Zs) >= 2


def plot_bayes_factor_matrix(labels, log10Zs, log10Zerrs, figsize=(10, 10),
                             cmap_name="coolwarm"):
    n = len(labels)
    bf = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if log10Zs[i] is not None and log10Zs[j] is not None:
                bf[i, j] = log10Zs[i] - log10Zs[j]

    norm = matplotlib.colors.Normalize(vmin=np.nanmin(bf), vmax=np.nanmax(bf))
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    errs = [0.0 if e is None else e for e in log10Zerrs]
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:
                ax.text(0.5, 0.5, labels[i], ha="center", va="center", fontsize=11, weight="bold")
            elif not np.isnan(bf[i, j]):
                ax.set_facecolor(cmap(norm(bf[i, j])))
                ax.text(0.5, 0.58, f"{bf[i, j]:.2f}", ha="center", va="center", fontsize=12)
                ax.text(0.5, 0.30, f"±{np.hypot(errs[i], errs[j]):.2f}",
                        ha="center", va="center", fontsize=9)
            else:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)

    fig.subplots_adjust(right=0.85, top=0.92, wspace=0.1, hspace=0.1)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(
        r"$\log_{10}$ Bayes factor (model $i - j$)", fontsize=14)
    fig.suptitle(r"Pairwise $\log_{10}$ Bayes factors", fontsize=18, y=0.98)
    return fig


def overlay_observed_events(ax, settings, cosmo=None):
    """Best-effort faint rug of observed SOURCE-frame m1 medians on p(m1).

    The plotted ``p_m1`` is a SOURCE-frame density (the likelihood forms
    ``m1src = m1det/(1+z)`` before calling ``log_p_pop``), so rugging the raw
    detector-frame medians put every tick a factor ``(1+z)`` -- 20-70% over this
    pipeline's redshift range -- to the right of where its event belongs, which
    reads as a high-mass misfit that is purely a frame error.  Each sample is
    converted with ``z_of_dL`` under ``cosmo`` (the fiducial cosmology by default;
    pass the run's posterior medians to use those instead) BEFORE the median, and
    the legend names the frame and that conversion.
    """
    try:
        from darksirens.gw.samples import load_gw_samples
        from darksirens.utils.cosmology import z_of_dL
        gw_path = settings.get("gw_path")
        if not gw_path or not os.path.exists(gw_path):
            return
        out = load_gw_samples(gw_path)
        m1det = np.asarray(out[0])
        dL = np.asarray(out[2])
        nEvents, nsamp = int(out[-2]), int(out[-1])
        c = {**COSMO_FID, **(cosmo or {})}
        z = np.asarray(z_of_dL(dL, c["H0"], c["Om0"], c["w0"], c["wa"]))
        m1src = m1det / (1.0 + z)
        med = np.median(m1src.reshape(nEvents, nsamp), axis=1)
        for k, v in enumerate(med):
            ax.axvline(v, color="0.3", alpha=0.18, lw=0.8,
                       label="observed (source-frame $m_1$, fiducial cosmology)"
                             if k == 0 else None)
        ax.legend(fontsize=16, frameon=False)
    except Exception as exc:  # noqa: BLE001 — overlay is best-effort
        _warn(f"event overlay skipped: {exc}")


# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dirs", nargs="+", default=["."],
                   help="One or more run directories; defaults to the current directory.")
    p.add_argument("--mmin", type=float, default=1.0)
    p.add_argument("--mmax", type=float, default=100.0)
    p.add_argument("--nm", type=int, default=128)
    p.add_argument("--nq", type=int, default=48)
    p.add_argument("--nz", type=int, default=32)
    p.add_argument("--nchi", type=int, default=24)
    p.add_argument("--zmax", type=float, default=2.0)
    p.add_argument("--chimin", type=float, default=-1.0)
    p.add_argument("--chimax", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Per-sample batch size; auto-chosen from device memory if unset.")
    p.add_argument("--grid_chunk", type=int, default=None,
                   help="Rows of the flattened (m1, q) outer plane per density-evaluation "
                        "slab (nm*nq rows in total, NOT flattened 4-D grid points); "
                        "auto-chosen from device memory if unset (bounds the GP kernel "
                        "tensor). A value >= nm*nq disables chunking.")
    p.add_argument("--max_mem_gb", type=float, default=None,
                   help="Memory budget in GB for posterior-predictive sizing; probes the "
                        "JAX device if unset (env DARKSIRENS_ANALYZE_MAX_MEM_GB also works).")
    p.add_argument("--mem_safe_frac", type=float, default=0.4,
                   help="Fraction of the memory budget the posterior-predictive peak may use.")
    p.add_argument("--cred_lo", type=float, default=5.0)
    p.add_argument("--cred_hi", type=float, default=95.0)
    p.add_argument("--overlay_events", action="store_true",
                   help="Overlay observed detector-frame m1 medians on p(m1).")
    p.add_argument("--sky_nside", type=int, default=16,
                   help="HEALPix nside for the sphere_gp posterior sky map.")
    p.add_argument("--outdir", default=".", help="Directory for output figures.")
    p.add_argument("--allow_legacy_pickle", action="store_true",
                   help="Permit reading a very old pickled-dict samples.npy. This "
                        "deserializes arbitrary Python from the file (arbitrary code "
                        "execution); only use it on files you trust.")
    return p


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)

    print()
    _banner("DARK SIRENS ANALYZE")
    print()

    mgrid = np.linspace(args.mmin, args.mmax, args.nm)
    qgrid = np.linspace(0.01, 1.0, args.nq)
    zgrid = np.linspace(0.0, args.zmax, args.nz)
    chigrid = np.linspace(args.chimin, args.chimax, args.nchi)
    limits = (args.cred_lo, args.cred_hi)

    labels, logZs, logZerrs = [], [], []
    spec = {k: [] for k in ("m1", "m2", "q", "z", "chi")}
    p_m1m2_per_model, cosmo_per_model, rate_per_model = [], [], []
    first_settings = None

    for run_dir in args.run_dirs:
        _section(f"Model  ·  {run_dir}")
        settings, samples, logZ, logZerr = load_run(
            run_dir, allow_legacy_pickle=args.allow_legacy_pickle
        )
        first_settings = first_settings or settings
        run_labels = _labels_of(settings)

        pop_model = pop_model_parser(
            settings["pop_model"],
            shared_beta=bool(settings.get("shared_beta", True)),
            shared_spin=bool(settings.get("shared_spin", True)),
            shared_gamma=bool(settings.get("shared_gamma", True)),
        )

        log10Z = (logZ / np.log(10.0)) if logZ is not None else None
        log10Zerr = (logZerr / np.log(10.0)) if (logZ is not None and logZerr is not None) else None
        logZs.append(log10Z)
        logZerrs.append(log10Zerr)

        p_m1, p_m2, p_q, p_z, p_chi, p_m1m2 = posterior_predictive(
            pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
            batch_size=args.batch_size,
            grid_chunk=args.grid_chunk,
            max_mem_gb=args.max_mem_gb,
            mem_safe_frac=args.mem_safe_frac,
        )
        spec["m1"].append(summarize_ppd(p_m1, limits))
        spec["m2"].append(summarize_ppd(p_m2, limits))
        spec["q"].append(summarize_ppd(p_q, limits))
        spec["z"].append(summarize_ppd(p_z, limits))
        spec["chi"].append(summarize_ppd(p_chi, limits))
        # Only the median enters the 2-D figure; keep that, not every sample's
        # (nm, nm) plane for every model.
        p_m1m2_per_model.append(np.median(np.asarray(p_m1m2), axis=0))

        # Cosmology posteriors (only the sampled ones).
        cosmo_per_model.append({
            name: _column(samples, run_labels, name) for name in COSMO_FID
        })

        # Redshift distribution / detection rate dN/dz, using the run's full
        # cosmology (H0, Om0, w0, wa) so the curve's SHAPE — not just an
        # H0 rescaling that mostly cancels in the per-sample normalisation —
        # reflects what was actually sampled/fixed. Per parameter: the sampled
        # column if present, else this run's individually-fixed value
        # (``fixed_parameter_values``, recovered from settings.json or the
        # results.hdf5 ``fixed_labels``/``fixed_values`` datasets), else the
        # fiducial — mirroring the decode-time fallback in
        # ``inference/parameters.py``.
        fixed_cosmo = settings.get("fixed_parameter_values") or {}

        def _cosmo_samples(name):
            col = _column(samples, run_labels, name)
            if col is not None:
                return col
            # noqa comment: ruff cannot see that this closure only runs inside
            # the loop iteration that bound p_z above (review-confirmed false
            # positive, potato_review P3-03).
            return np.full(p_z.shape[0], fixed_cosmo.get(name, COSMO_FID[name]))  # noqa: F821

        h0_samples = _cosmo_samples("H0")
        om0_samples = _cosmo_samples("Om0")
        w0_samples = _cosmo_samples("w0")
        wa_samples = _cosmo_samples("wa")
        rate_per_model.append(summarize_ppd(
            redshift_rate_samples(p_z, zgrid, h0_samples, om0_samples, w0_samples, wa_samples),
            limits))

        # GP-latent caterpillar (only for models with latents).
        if run_labels and latent_indices(run_labels):
            fig = make_latent_summary(samples, run_labels)
            if fig is not None:
                tag = os.path.basename(os.path.normpath(run_dir)) or "run"
                fig.savefig(out(f"latents_{tag}.pdf"), bbox_inches="tight", dpi=300)
                plt.close(fig)

        # Sky-anisotropy summaries — only when an anisotropic sky model was run
        # (gated on the ``sky_model`` flag stored in the run's hdf5/settings).
        sky_model = str(settings.get("sky_model", "isotropic"))
        if sky_model != "isotropic" and run_labels:
            tag = os.path.basename(os.path.normpath(run_dir)) or "run"
            from darksirens.sky.analyze import (
                summarize_dipole_posterior,
                plot_dipole_posterior,
                plot_sphere_gp_map,
                summarize_multipole_posterior,
                plot_multipole_cl,
            )
            try:
                if sky_model == "dipole":
                    summ = summarize_dipole_posterior(samples, run_labels)
                    amp = summ["amplitude_quantiles"]
                    d = summ["mean_direction_deg"]
                    print(
                        f"  │  sky dipole |d| (5/50/95%) = "
                        f"{amp[0.05]:.3f}/{amp[0.5]:.3f}/{amp[0.95]:.3f}; "
                        f"dir (ra,dec)deg = ({d['ra']:.1f}, {d['dec']:.1f}); "
                        f"P(|d|<0.05) = {summ['P_amp_lt_0.05']:.2f}"
                    )
                    fig = plot_dipole_posterior(samples, run_labels)
                    fig.savefig(out(f"sky_dipole_{tag}.pdf"), bbox_inches="tight", dpi=300)
                    plt.close(fig)
                elif sky_model in ("sphere_gp", "sphere_gp_z", "overdensity_gp"):
                    # Angular model → one map; 3-D models → a few z-shell maps.
                    z_slices = [None] if sky_model == "sphere_gp" else [0.1, 0.5, 1.0]
                    for zs in z_slices:
                        fig = plot_sphere_gp_map(
                            samples, run_labels, nside=args.sky_nside,
                            sky_model=sky_model, z_slice=zs,
                        )
                        suffix = "" if zs is None else f"_z{zs:.2f}".replace(".", "p")
                        fig.savefig(out(f"sky_gp_map_{tag}{suffix}.pdf"),
                                    bbox_inches="tight", dpi=300)
                        plt.close(fig)
                elif sky_model in ("multipole", "multipole_l3"):
                    summ = summarize_multipole_posterior(samples, run_labels, sky_model)
                    cl = summ["C_ell_quantiles"]
                    print("  │  sky multipole C_l (median): " + ", ".join(
                        f"l={l}:{cl[l][0.5]:.3g}" for l in sorted(cl)))
                    fig = plot_multipole_cl(samples, run_labels, sky_model)
                    fig.savefig(out(f"sky_multipole_cl_{tag}.pdf"),
                                bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    fig = plot_sphere_gp_map(samples, run_labels, nside=args.sky_nside,
                                             sky_model=sky_model)
                    fig.savefig(out(f"sky_gp_map_{tag}.pdf"), bbox_inches="tight", dpi=300)
                    plt.close(fig)
            except (KeyError, ImportError) as exc:
                _warn(f"skipped sky plot for {tag}: {exc}")

        # Multitracer host-fraction summaries — derive the per-catalog mixture
        # weights w_1..w_K from the sampled sticks fcat_2..fcat_K with the
        # decoder's own stick-breaking construction, so the interpretable
        # estimand appears in the outputs without manual post-processing.
        n_catalogs = int(settings.get("n_catalogs", 1) or 1)
        fcat_cols = [f"fcat_{m}" for m in range(2, n_catalogs + 1)]
        if (n_catalogs >= 2 and run_labels
                and all(c in run_labels for c in fcat_cols)):
            from darksirens.inference.pop_extractor import catalog_sticks_to_weights
            tag = os.path.basename(os.path.normpath(run_dir)) or "run"
            sticks = np.stack(
                [_column(samples, run_labels, c) for c in fcat_cols], axis=1
            )
            weights = np.asarray(catalog_sticks_to_weights(sticks))
            is_field = (
                str(settings.get("catalog_sky_weighting", "conditional")) == "field"
            )
            qs = np.percentile(weights, [5.0, 50.0, 95.0], axis=0)
            name = "host fraction" if is_field else "weight"
            for k in range(n_catalogs):
                print(f"  │  catalog {name} w_{k + 1} (5/50/95%) = "
                      f"{qs[0, k]:.3f}/{qs[1, k]:.3f}/{qs[2, k]:.3f}")
            if not is_field:
                _warn("this run used the conditional normalizer, so w_k is a "
                      "per-pixel z-shape preference weight, NOT a host "
                      "fraction (field convention).")
            fig = plot_catalog_weight_posteriors(weights, is_field=is_field)
            fig.savefig(out(f"catalog_weights_{tag}.pdf"),
                        bbox_inches="tight", dpi=300)
            plt.close(fig)
            np.save(out(f"catalog_weights_{tag}.npy"), weights)

        labels.append(settings.get("model_name", os.path.basename(os.path.normpath(run_dir))))
        _ok(f"Posterior predictive done:  {np.asarray(samples).shape[0]:,} samples")
        _end()
        del p_m1, p_m2, p_q, p_z, p_chi, p_m1m2
        jax.clear_caches()

    # ---- 1-D posterior-predictive spectra ----
    specs = [
        ("m1", mgrid, r"$m_1$ [$M_\odot$]", r"$p(m_1)$ [$M_\odot^{-1}$]", None, (1e-5, 1.0)),
        ("m2", mgrid, r"$m_2$ [$M_\odot$]", r"$p(m_2)$ [$M_\odot^{-1}$]", None, (1e-5, 1.0)),
        ("q",  qgrid, r"$q$", r"$p(q)$", (0.0, 1.0), (1e-6, 1e1)),
        ("z",  zgrid, r"$z$", r"$p(z)$", (0.0, args.zmax), (1e-2, 1e1)),
        ("chi", chigrid, r"$\chi_\mathrm{eff}$", r"$p(\chi_\mathrm{eff})$",
         (args.chimin, args.chimax), (1e-2, 1e1)),
    ]
    for key, xg, xl, yl, xlim, ylim in specs:
        fig = plot_1d_spectrum(xg, spec[key], labels, xl, yl, xlim=xlim, ylim=ylim)
        if key == "m1" and args.overlay_events and first_settings is not None:
            overlay_observed_events(fig.axes[0], first_settings)
        fig.savefig(out(f"p{key}_all_models.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    # ---- 2-D joint p(m1, m2) ----
    fig = plot_2d_mass(mgrid, p_m1m2_per_model, labels)
    fig.savefig(out("pm1m2_all_models.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- cosmology posteriors ----
    fig = plot_cosmology_posterior(cosmo_per_model, labels, list(COSMO_FID))
    if fig is not None:
        fig.savefig(out("cosmology_posterior.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        print()
        _warn("Cosmology was fixed in all runs; skipping cosmology posterior plot.")

    # ---- redshift distribution / rate ----
    fig = plot_1d_spectrum(zgrid, rate_per_model, labels, r"$z$",
                           r"$\mathrm{d}N/\mathrm{d}z$  (normalised)",
                           xlim=(0.0, args.zmax), ylim=None, logy=False)
    fig.savefig(out("rate_dNdz.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- model comparison ----
    if any(z is not None for z in logZs):
        # Only runs with an evidence estimate enter the bar chart: coercing a
        # missing logZ to 0.0 would plot that run as the (fake) best model,
        # since real log10 Z values are large and negative.
        with_z = [(lbl, z, ze) for lbl, z, ze in zip(labels, logZs, logZerrs)
                  if z is not None]
        fig = plot_model_evidences([lbl for lbl, _, _ in with_z],
                                   [z for _, z, _ in with_z],
                                   [ze for _, _, ze in with_z])
        fig.savefig(out("model_evidences.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        _section("Model comparison")
        _row("Model", "log10 Z", width=40)
        for label, z, ze in zip(labels, logZs, logZerrs):
            _row(label,
                 "no evidence estimate" if z is None else f"{z:.3f} ± "
                 + (f"{ze:.3f}" if ze is not None else "n/a"),
                 width=40)
        print("  │")
        print_bayes_factors(labels, logZs)
        _end()
        if _should_plot_bayes_factor_matrix(labels, logZs):
            fig = plot_bayes_factor_matrix(labels, logZs, logZerrs)
            fig.savefig(out("bayes_factors.pdf"), bbox_inches="tight", dpi=300)
            plt.close(fig)
    else:
        print()
        _warn("No evidence information found in any run; skipping model comparison.")

    print()
    _ok(f"Figures written to {os.path.abspath(args.outdir)}")
    print()


if __name__ == "__main__":
    main()
