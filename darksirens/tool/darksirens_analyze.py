#!/usr/bin/env python3
"""darksirens_analyze — post-process and plot one or more inference runs.

Reads the current ``results.hdf5`` save format (legacy ``samples.npy`` is still
accepted), recomputes posterior-predictive population distributions, and makes
publication-quality figures:

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
import os
import json
import argparse

import numpy as np
import h5py
from tqdm import tqdm

import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from darksirens.gw.populations import pop_model_parser
from darksirens.inference.pop_extractor import make_pop_extractor
from darksirens.inference.parameters import H0_FID, OM0_FID, W0_FID, WA_FID
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
        if "samples" not in f:
            raise KeyError(f"{path} does not contain a 'samples' dataset")
        samples = f["samples"][()]
        settings = _merge_hdf5_metadata(settings, f)
        logZ = f.attrs.get("logZ", None)
        logZerr = f.attrs.get("logZerr", None)

    if logZ is not None:
        logZ = float(logZ)
    if logZerr is not None:
        logZerr = float(logZerr)
        if np.isnan(logZerr):
            logZerr = None

    return settings, samples, logZ, logZerr


def _load_run_npy(run_dir):
    """Legacy loader for the deprecated samples.npy format."""
    settings = _load_settings(run_dir)
    results = np.load(os.path.join(run_dir, "samples.npy"), allow_pickle=True).item()
    return settings, results["samples"], results.get("logZ"), results.get("logZerr")


def load_run(run_dir):
    """Load a completed inference run (current HDF5, else legacy NPY)."""
    if os.path.exists(os.path.join(run_dir, "results.hdf5")):
        return _load_run_hdf5(run_dir)
    if os.path.exists(os.path.join(run_dir, "samples.npy")):
        return _load_run_npy(run_dir)
    raise FileNotFoundError(
        f"No inference results found in {run_dir!r}; expected 'results.hdf5' "
        "(current format) or 'samples.npy' (legacy format)."
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
def auto_batch_size(nsamples, grid_points, target_bytes=2.0e9, dtype_bytes=8, cap=256):
    """Largest batch whose per-sample grid stays under ``target_bytes``."""
    per_sample = float(grid_points) * dtype_bytes
    if per_sample <= 0:
        return min(nsamples, cap)
    return int(min(nsamples, max(1, int(target_bytes // per_sample)), cap))


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
    outs = [
        vfn(samples[i:i + batch_size])
        for i in tqdm(range(0, samples.shape[0], batch_size),
                      desc="Posterior-predictive batches")
    ]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)
    return jax.tree_util.tree_map(lambda a: a[:ns], stacked)


# ------------------------------------------------------------
# Posterior-predictive engine (flattened grid → works for all models)
# ------------------------------------------------------------
def make_single_theta_predictive(pop_model, settings, mgrid, qgrid, zgrid, chigrid):
    """Build a jit'd per-sample evaluator returning the PPD marginals + p(m1,m2).

    The population density is evaluated on a flattened, equal-shape coordinate
    grid (every coordinate is a 1-D array of length nm*nq*nz*nchi), so the
    call satisfies the equal-shape requirement of the GP / binned-GP models'
    ``log_p_pop`` as well as the natural broadcasting of the parametric models.
    """
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)
    chigrid = jnp.asarray(chigrid)
    nm, nq, nz, nchi = mgrid.size, qgrid.size, zgrid.size, chigrid.size

    M1, Q, Z, CHI = jnp.meshgrid(mgrid, qgrid, zgrid, chigrid, indexing="ij")
    m1f, qf, zf, chif = M1.ravel(), Q.ravel(), Z.ravel(), CHI.ravel()

    # q→m2 interpolation geometry for the 2-D joint.
    q_eval = mgrid[None, :] / mgrid[:, None]
    valid = (q_eval >= qgrid[0]) & (q_eval <= qgrid[-1])
    jac = 1.0 / mgrid[:, None]

    pop_extractor = make_pop_extractor(settings)

    @jax.jit
    def single_theta(theta):
        pop_theta = pop_extractor(theta)
        logp = pop_model(m1f, qf, zf, chif, pop_theta)
        p = jnp.exp(logp).reshape(nm, nq, nz, nchi)

        # 1-D marginals (integrate out the other three axes).
        p_m1 = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2), qgrid, axis=1)
        p_m1 /= jnp.trapezoid(p_m1, mgrid)
        p_q = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2), mgrid, axis=0)
        p_q /= jnp.trapezoid(p_q, qgrid)
        p_z = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), qgrid, axis=1), mgrid, axis=0)
        p_z /= jnp.trapezoid(p_z, zgrid)
        p_chi = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(p, zgrid, axis=2), qgrid, axis=1), mgrid, axis=0)
        p_chi /= jnp.trapezoid(p_chi, chigrid)

        # 2-D joint p(m1, m2): marginalize chi, z → p(m1, q), map q→m2.
        p_m1q = jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2)
        p_interp = jax.vmap(
            lambda row, qev: jnp.interp(qev, qgrid, row, left=0.0, right=0.0)
        )(p_m1q, q_eval)
        p_m1m2 = p_interp * jac * valid
        norm_2d = jnp.trapezoid(jnp.trapezoid(p_m1m2, mgrid, axis=0), mgrid, axis=0)
        p_m1m2 = jnp.where(norm_2d > 0, p_m1m2 / norm_2d, p_m1m2)
        p_m2 = jnp.trapezoid(p_m1m2, mgrid, axis=0)
        p_m2 /= jnp.trapezoid(p_m2, mgrid)

        return p_m1, p_m2, p_q, p_z, p_chi, p_m1m2

    return single_theta


def posterior_predictive(pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid,
                         batch_size=None):
    """Return per-sample (p_m1, p_m2, p_q, p_z, p_chi, p_m1m2) stacks."""
    samples = jnp.asarray(samples)
    grid_points = mgrid.size * qgrid.size * zgrid.size * chigrid.size
    if batch_size is None:
        batch_size = auto_batch_size(samples.shape[0], grid_points)

    single_theta = make_single_theta_predictive(pop_model, settings, mgrid, qgrid, zgrid, chigrid)
    return batched_map(single_theta, samples, batch_size)


def summarize_ppd(ppd_samples, limits=(5, 95)):
    lo, hi = limits
    median = jnp.median(ppd_samples, axis=0)
    lower = jnp.percentile(ppd_samples, lo, axis=0)
    upper = jnp.percentile(ppd_samples, hi, axis=0)
    return np.asarray(median), np.asarray(lower), np.asarray(upper)


def redshift_rate_samples(pz_samples, zgrid, h0_samples, om0=OM0_FID):
    """dN/dz ∝ p_z(z) · (dV_c/dz)(z; H0) / (1+z), normalised per sample."""
    zg = jnp.asarray(zgrid)

    def one(pz, h0):
        rate = pz * dV_of_z(zg, h0, om0) / (1.0 + zg)
        norm = jnp.trapezoid(rate, zg)
        return rate / jnp.where(norm > 0, norm, 1.0)

    return np.asarray(jax.vmap(one)(jnp.asarray(pz_samples), jnp.asarray(h0_samples)))


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
    """Median joint p(m1, m2) as filled contours, one panel per model."""
    mgrid = np.asarray(mgrid)
    n = len(labels)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    figsize = figsize or (6.0 * ncol, 5.5 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    M1, M2 = np.meshgrid(mgrid, mgrid, indexing="ij")
    for k, (label, p2d) in enumerate(zip(labels, p_m1m2_per_model)):
        ax = axes[k // ncol][k % ncol]
        med = np.median(np.asarray(p2d), axis=0)
        masked = np.where(M2 <= M1, med, np.nan)
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


def plot_model_evidences(labels, log10Zs, log10Zerrs, figsize=(10, 6)):
    log10Zs = np.asarray(log10Zs, dtype=float)
    errs = np.asarray([0.0 if e is None else e for e in log10Zerrs], dtype=float)
    delta = log10Zs - np.max(log10Zs)

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(labels))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    ax.bar(xs, delta, yerr=errs, color=colors, alpha=0.85, capsize=5)
    ax.axhline(0.0, color="black", lw=1.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=16)
    ax.set_ylabel(r"$\Delta \log_{10} Z$ (relative to best)", fontsize=20)
    ax.set_title("Model comparison", fontsize=22)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    return fig


def print_bayes_factors(labels, log10Zs):
    print("\n=== Pairwise Bayes factors (log10) ===")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if log10Zs[i] is not None and log10Zs[j] is not None:
                print(f"{labels[i]} vs {labels[j]}:  log10 BF = {log10Zs[i] - log10Zs[j]:.3f}")


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


def overlay_observed_events(ax, settings):
    """Best-effort faint rug of observed detector-frame m1 medians on p(m1)."""
    try:
        from darksirens.gw.utils import load_gw_samples
        gw_path = settings.get("gw_path")
        if not gw_path or not os.path.exists(gw_path):
            return
        out = load_gw_samples(gw_path)
        m1det, nEvents, nsamp = np.asarray(out[0]), int(out[-2]), int(out[-1])
        med = np.median(m1det.reshape(nEvents, nsamp), axis=1)
        for k, v in enumerate(med):
            ax.axvline(v, color="0.3", alpha=0.18, lw=0.8,
                       label="observed (det-frame $m_1$)" if k == 0 else None)
        ax.legend(fontsize=16, frameon=False)
    except Exception as exc:  # noqa: BLE001 — overlay is best-effort
        print(f"  [info] event overlay skipped: {exc}")


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
                   help="Per-sample batch size; auto-chosen from grid memory if unset. "
                        "Smooth-GP models may need a small value (e.g. 1-4).")
    p.add_argument("--cred_lo", type=float, default=5.0)
    p.add_argument("--cred_hi", type=float, default=95.0)
    p.add_argument("--overlay_events", action="store_true",
                   help="Overlay observed detector-frame m1 medians on p(m1).")
    p.add_argument("--outdir", default=".", help="Directory for output figures.")
    return p


def main():
    args = _build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)

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
        print(f"\n=== Processing model: {run_dir} ===")
        settings, samples, logZ, logZerr = load_run(run_dir)
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
        )
        spec["m1"].append(summarize_ppd(p_m1, limits))
        spec["m2"].append(summarize_ppd(p_m2, limits))
        spec["q"].append(summarize_ppd(p_q, limits))
        spec["z"].append(summarize_ppd(p_z, limits))
        spec["chi"].append(summarize_ppd(p_chi, limits))
        p_m1m2_per_model.append(np.asarray(p_m1m2))

        # Cosmology posteriors (only the sampled ones).
        cosmo_per_model.append({
            name: _column(samples, run_labels, name) for name in COSMO_FID
        })

        # Redshift distribution / detection rate dN/dz, using sampled H0 if present.
        h0_col = _column(samples, run_labels, "H0")
        h0_samples = h0_col if h0_col is not None else np.full(p_z.shape[0], H0_FID)
        rate_per_model.append(summarize_ppd(
            redshift_rate_samples(p_z, zgrid, h0_samples), limits))

        # GP-latent caterpillar (only for models with latents).
        if run_labels and latent_indices(run_labels):
            fig = make_latent_summary(samples, run_labels)
            if fig is not None:
                tag = os.path.basename(os.path.normpath(run_dir)) or "run"
                fig.savefig(out(f"latents_{tag}.pdf"), bbox_inches="tight", dpi=300)
                plt.close(fig)

        labels.append(settings.get("model_name", os.path.basename(os.path.normpath(run_dir))))
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
        print("\nCosmology was fixed in all runs; skipping cosmology posterior plot.")

    # ---- redshift distribution / rate ----
    fig = plot_1d_spectrum(zgrid, rate_per_model, labels, r"$z$",
                           r"$\mathrm{d}N/\mathrm{d}z$  (normalised)",
                           xlim=(0.0, args.zmax), ylim=None, logy=False)
    fig.savefig(out("rate_dNdz.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- model comparison ----
    if any(z is not None for z in logZs):
        fig = plot_model_evidences(labels, [0.0 if z is None else z for z in logZs], logZerrs)
        fig.savefig(out("model_evidences.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print("\n=== Model evidences (log10 Z) ===")
        for label, z, ze in zip(labels, logZs, logZerrs):
            print(f"{label:24s} log10Z = {z} ± {ze}")
        print_bayes_factors(labels, logZs)
        if len(labels) >= 2 and sum(z is not None for z in logZs) >= 2:
            fig = plot_bayes_factor_matrix(labels, logZs, logZerrs)
            fig.savefig(out("bayes_factors.pdf"), bbox_inches="tight", dpi=300)
            plt.close(fig)
    else:
        print("\nNo evidence information found in any run; skipping model comparison.")

    print(f"\nFigures written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
