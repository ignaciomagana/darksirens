#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import h5py
from tqdm import tqdm

import jax
import jax.numpy as jnp

from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser, get_fixed_population_params
from darksirens.inference.pop_extractor import make_pop_extractor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c = sns.color_palette('colorblind')


# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
def _json_safe_hdf5_attr(value):
    """Convert HDF5 attribute values into plain JSON-like Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_hdf5_attr(item) for item in value.tolist()]
    return value


def _load_settings(run_dir):
    settings_path = os.path.join(run_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return json.load(f)
    return {}


def _merge_hdf5_metadata(settings, h5_file):
    """Backfill settings from results.hdf5 attrs/datasets when available."""
    merged = dict(settings)

    for key, value in h5_file.attrs.items():
        if key in {"environment", "prior_overrides"} and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        merged.setdefault(key, _json_safe_hdf5_attr(value))

    if "labels" not in merged and "labels" in h5_file:
        merged["labels"] = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in h5_file["labels"][()]
        ]

    if "lower_bound" not in merged and "lower_bound" in h5_file:
        merged["lower_bound"] = h5_file["lower_bound"][()].astype(float).tolist()
    if "upper_bound" not in merged and "upper_bound" in h5_file:
        merged["upper_bound"] = h5_file["upper_bound"][()].astype(float).tolist()

    if (
        not merged.get("fixed_parameter_values")
        and "fixed_labels" in h5_file
        and "fixed_values" in h5_file
    ):
        fixed_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in h5_file["fixed_labels"][()]
        ]
        fixed_values = h5_file["fixed_values"][()]
        merged["fixed_parameter_values"] = {
            label: float(value) for label, value in zip(fixed_labels, fixed_values)
        }

    return merged



def _find_hdf5_samples_dataset(h5_file):
    """Return the dataset path for posterior samples in supported HDF5 layouts."""
    candidates = (
        "samples",
        "posterior/samples",
        "results/samples",
        "posterior_samples",
    )
    for candidate in candidates:
        if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
            return candidate
    return None


def _first_hdf5_attr(h5_file, *names):
    """Return the first available attribute from common evidence-name aliases."""
    for name in names:
        if name in h5_file.attrs:
            return h5_file.attrs[name]
    return None

def _load_run_hdf5(run_dir):
    path = os.path.join(run_dir, "results.hdf5")
    settings = _load_settings(run_dir)

    with h5py.File(path, "r") as f:
        samples_path = _find_hdf5_samples_dataset(f)
        if samples_path is None:
            raise KeyError(
                f"{path} does not contain posterior samples; expected a 'samples' "
                "dataset at the file root or in a posterior/results group"
            )

        samples = f[samples_path][()]
        settings = _merge_hdf5_metadata(settings, f)
        logZ = _first_hdf5_attr(f, "logZ", "log_evidence", "logz")
        logZerr = _first_hdf5_attr(f, "logZerr", "log_evidence_err", "logzerr")

    if logZ is not None:
        logZ = float(logZ)
    if logZerr is not None:
        logZerr = float(logZerr)
        if np.isnan(logZerr):
            logZerr = None

    return settings, samples, logZ, logZerr


def _load_run_npy(run_dir):
    settings = _load_settings(run_dir)
    path = os.path.join(run_dir, "samples.npy")
    results = np.load(path, allow_pickle=True).item()

    samples = results["samples"]
    logZ = results.get("logZ", None)
    logZerr = results.get("logZerr", None)

    return settings, samples, logZ, logZerr


def load_run(run_dir):
    """Load a completed inference run from the current HDF5 or legacy NPY format."""
    hdf5_path = os.path.join(run_dir, "results.hdf5")
    npy_path = os.path.join(run_dir, "samples.npy")

    if os.path.exists(hdf5_path):
        return _load_run_hdf5(run_dir)
    if os.path.exists(npy_path):
        return _load_run_npy(run_dir)

    raise FileNotFoundError(
        f"No inference results found in {run_dir!r}; expected 'results.hdf5' "
        "(current format) or 'samples.npy' (legacy format)."
    )


# ------------------------------------------------------------
# Evidence plotting
# ------------------------------------------------------------
def plot_model_evidences(labels, logZs, logZerrs, figsize=(10, 6)):
    # 1. Convert to numpy arrays for calculation
    logZs = np.array(logZs)
    logZerrs = np.array(logZerrs)
    
    # 2. Calculate relative evidence (Delta log10 Z)
    # The "best" model is the one with the maximum logZ
    best_logZ = np.max(logZs)
    delta_logZs = logZs - best_logZ

    fig, ax = plt.subplots(figsize=figsize)

    xs = np.arange(len(labels))
    
    # Using a color cycle if 'c' isn't explicitly passed
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    # 3. Plot the delta values
    bars = ax.bar(xs, delta_logZs, yerr=logZerrs, color=colors, alpha=0.8, capsize=5)

    # 4. Formatting
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=18)
    
    # Update label to show it is relative
    ax.set_ylabel(r"$\Delta \log_{10} Z$ (Relative to Best)", fontsize=22)
    ax.set_title("Model Comparison (Relative Evidence)", fontsize=26)
    ax.tick_params(labelsize=18)
    
    # Add a horizontal dashed line at 0 for the reference model
    ax.axhline(0, color='black', lw=1.5, ls='--')

    # Optional: Add text labels on top/bottom of bars to show exact delta values
    for bar, val in zip(bars, delta_logZs):
        yval = bar.get_height()
        # Offset text slightly below the bar
        ax.text(bar.get_x() + bar.get_width()/2, yval - (0.05 * abs(np.min(delta_logZs))), 
                f'{val:.2f}', ha='center', va='top', fontsize=14, fontweight='bold')

    fig.tight_layout()
    return fig


def print_bayes_factors(labels, logZs):
    print("\n=== Bayes Factors (log BF) ===")
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if logZs[i] is not None and logZs[j] is not None:
                print(f"{labels[i]} vs {labels[j]}:  log10 BF = {logZs[i] - logZs[j]:.3f}")


def _should_plot_bayes_factor_matrix(labels, logZs):
    """Return True when there is at least one pair of models to compare."""
    if len(labels) < 2:
        return False
    return sum(z is not None for z in logZs) >= 2


import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_bayes_factor_matrix_pairwise(labels, log10Zs, log10Zerrs, figsize=(10, 10), cmap_name="coolwarm"):
    n = len(labels)
    # Use layout=None and we will handle the colorbar axis manually
    fig, axes = plt.subplots(n, n, figsize=figsize)

    # Compute all pairwise BF values for normalization
    bf_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if log10Zs[i] is not None and log10Zs[j] is not None:
                bf_matrix[i, j] = log10Zs[i] - log10Zs[j]

    vmin = np.nanmin(bf_matrix)
    vmax = np.nanmax(bf_matrix)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                ax.text(0.5, 0.5, labels[i], ha="center", va="center", fontsize=12, weight='bold')
            elif log10Zs[i] is not None and log10Zs[j] is not None:
                bf = bf_matrix[i, j]
                bf_err = np.sqrt(log10Zerrs[i]**2 + log10Zerrs[j]**2)
                ax.set_facecolor(cmap(norm(bf)))
                ax.text(0.5, 0.55, f"{bf:.2f}", ha="center", va="center", fontsize=12)
                ax.text(0.5, 0.30, f"±{bf_err:.2f}", ha="center", va="center", fontsize=9)
            else:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)

            ax.set_xticks([])
            ax.set_yticks([])

    # --- THE FIX ---
    # 1. Create a "divider" based on the last column of axes
    # This ensures the colorbar matches the height of the grid
    divider = make_axes_locatable(axes[0, -1]) 
    
    # Actually, to cover the whole height of the grid, we'll use a more manual 'cax' approach
    # We'll shrink the grid slightly to make a dedicated home for the colorbar.
    fig.subplots_adjust(right=0.85) # Make room on the right
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # [left, bottom, width, height]

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log_{10}$ Bayes Factor (Model $i$ $-$ Model $j$)", fontsize=14)

    fig.suptitle(r"Pairwise $\log_{10}$ Bayes Factors", fontsize=18, y=0.98)
    
    # We use subplots_adjust instead of tight_layout to keep our manual cax in place
    fig.subplots_adjust(top=0.92, wspace=0.1, hspace=0.1)

    return fig


# ------------------------------------------------------------
# JAX posterior predictive engine (batched, per-sample PPD)
# ------------------------------------------------------------
def make_single_theta_predictive(pop_model, settings, mgrid, qgrid, zgrid, chigrid):
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)
    chigrid = jnp.asarray(chigrid)

    nm = mgrid.size

    # Expand to 4D: (m1, q, z, chi)
    m1_grid  = mgrid[:, None, None, None]
    q_grid   = qgrid[None, :, None, None]
    z_grid   = zgrid[None, None, :, None]
    chi_grid = chigrid[None, None, None, :]

    m2_vals = mgrid
    q_eval = m2_vals[None, :] / mgrid[:, None]
    valid = (q_eval >= qgrid[0]) & (q_eval <= qgrid[-1])
    jac = 1.0 / mgrid[:, None]

    # Initialize dynamic parameter extractor
    pop_extractor = make_pop_extractor(settings)

    @jax.jit
    def single_theta(theta):
        # Extract purely population parameters
        pop_theta = pop_extractor(theta)
        
        # pop_model must now accept chi_grid and the filtered pop_theta
        logp = pop_model(m1_grid, q_grid, z_grid, chi_grid, pop_theta)
        p = jnp.exp(logp)

        # -----------------------------------------------------
        # 1D Marginalizations (Integrate out 3 dimensions)
        # -----------------------------------------------------
        # p(m1): integrate over chi (axis 3), z (axis 2), q (axis 1)
        p_m1 = jnp.trapezoid(
            jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2), qgrid, axis=1
        )
        p_m1 /= jnp.trapezoid(p_m1, mgrid)

        # p(q): integrate over chi (axis 3), z (axis 2), m1 (axis 0)
        p_q = jnp.trapezoid(
            jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2), mgrid, axis=0
        )
        p_q /= jnp.trapezoid(p_q, qgrid)

        # p(z): integrate over chi (axis 3), q (axis 1), m1 (axis 0)
        p_z = jnp.trapezoid(
            jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), qgrid, axis=1), mgrid, axis=0
        )
        p_z /= jnp.trapezoid(p_z, zgrid)

        # p(chi): integrate over z (axis 2), q (axis 1), m1 (axis 0)
        p_chi = jnp.trapezoid(
            jnp.trapezoid(jnp.trapezoid(p, zgrid, axis=2), qgrid, axis=1), mgrid, axis=0
        )
        p_chi /= jnp.trapezoid(p_chi, chigrid)

        # -----------------------------------------------------
        # 2D Joint Mass Distribution p(m1, m2)
        # -----------------------------------------------------
        # First, marginalize over chi and z to get p(m1, q)
        p_m1q = jnp.trapezoid(jnp.trapezoid(p, chigrid, axis=3), zgrid, axis=2)

        # Interpolate q onto m2 grid
        p_interp = jax.vmap(
            lambda row, qev: jnp.interp(qev, qgrid, row, left=0.0, right=0.0)
        )(p_m1q, q_eval)

        p_m1m2 = p_interp * jac * valid

        norm_2d = jnp.trapezoid(
            jnp.trapezoid(p_m1m2, mgrid, axis=0), mgrid, axis=0
        )
        p_m1m2 = jnp.where(norm_2d > 0, p_m1m2 / norm_2d, p_m1m2)

        p_m2 = jnp.trapezoid(p_m1m2, mgrid, axis=0)
        p_m2 /= jnp.trapezoid(p_m2, mgrid)

        return p_m1, p_m2, p_q, p_z, p_chi, p_m1m2

    return single_theta


def auto_batch_size(nsamples, nm, nq, nz, nchi, target_bytes=2e9):
    # Now calculating bytes for a 4D array per sample
    per_sample_bytes = nm * nq * nz * nchi * 8.0
    if per_sample_bytes == 0:
        return min(nsamples, 64)

    max_batch = int(target_bytes // per_sample_bytes)
    max_batch = max(1, min(max_batch, 256))
    return min(nsamples, max_batch)


def posterior_predictive_mass_distributions_jax(
    pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid, batch_size=None
):
    samples = jnp.asarray(samples)
    mgrid = jnp.asarray(mgrid)
    qgrid = jnp.asarray(qgrid)
    zgrid = jnp.asarray(zgrid)
    chigrid = jnp.asarray(chigrid)

    ns = samples.shape[0]
    nm = mgrid.size
    nq = qgrid.size
    nz = zgrid.size
    nchi = chigrid.size

    if batch_size is None:
        batch_size = auto_batch_size(ns, nm, nq, nz, nchi)

    single_theta = make_single_theta_predictive(pop_model, settings, mgrid, qgrid, zgrid, chigrid)

    pm1_list   = []
    pm2_list   = []
    pq_list    = []
    pz_list    = []
    pchi_list  = []

    n_batches = (ns + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Posterior predictive batches"):
        start = i * batch_size
        end = min((i + 1) * batch_size, ns)
        batch = samples[start:end]

        p1_batch, p2_batch, pq_batch, pz_batch, pchi_batch, p2d_batch = jax.vmap(single_theta)(batch)

        pm1_list.append(p1_batch)
        pm2_list.append(p2_batch)
        pq_list.append(pq_batch)
        pz_list.append(pz_batch)
        pchi_list.append(pchi_batch)

    pm1_samples   = jnp.concatenate(pm1_list,   axis=0)
    pm2_samples   = jnp.concatenate(pm2_list,   axis=0)
    pq_samples    = jnp.concatenate(pq_list,    axis=0)
    pz_samples    = jnp.concatenate(pz_list,    axis=0)
    pchi_samples  = jnp.concatenate(pchi_list,  axis=0)

    return pm1_samples, pm2_samples, pq_samples, pz_samples, pchi_samples


# ------------------------------------------------------------
# PPD summarization + plotting
# ------------------------------------------------------------
def summarize_ppd(ppd_samples, limits=(5, 95)):
    lo, hi = limits
    median = jnp.median(ppd_samples, axis=0)
    lower = jnp.percentile(ppd_samples, lo, axis=0)
    upper = jnp.percentile(ppd_samples, hi, axis=0)
    return median, lower, upper


def plot_parameter_corner(samples_list, labels, parameter_labels_list, max_parameters=6, figsize_per_panel=2.8):
    """Plot 1D/2D posterior marginals for sampled parameters shared by all runs."""
    if not samples_list or not parameter_labels_list:
        return None

    common = list(parameter_labels_list[0])
    for labs in parameter_labels_list[1:]:
        labset = set(labs)
        common = [lab for lab in common if lab in labset]
    if not common:
        return None

    chosen = common[:max_parameters]
    ndim = len(chosen)
    fig, axes = plt.subplots(
        ndim, ndim, figsize=(figsize_per_panel * ndim, figsize_per_panel * ndim), squeeze=False
    )

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue
            for i, (samples, run_labels) in enumerate(zip(samples_list, parameter_labels_list)):
                color = c[i % len(c)]
                idx_x = run_labels.index(chosen[col])
                x = np.asarray(samples[:, idx_x], dtype=float)
                if row == col:
                    ax.hist(x, bins=40, density=True, histtype="step", color=color, lw=1.8)
                else:
                    idx_y = run_labels.index(chosen[row])
                    y = np.asarray(samples[:, idx_y], dtype=float)
                    ax.scatter(x, y, s=4, alpha=0.12, color=color, rasterized=True)
            if row == ndim - 1:
                ax.set_xlabel(chosen[col], fontsize=12)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(chosen[row], fontsize=12)
            elif col != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9)

    handles = [Line2D([0], [0], color=c[i % len(c)], lw=2) for i in range(len(labels))]
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=12)
    fig.suptitle("Posterior parameter marginals", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.94, 0.98))
    return fig


def plot_ppd_moment_summary(metric_names, median_matrix, labels, ylabel, figsize=(10, 6)):
    """Plot scalar summaries of posterior-predictive distributions across runs."""
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(metric_names))
    width = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        offset = (i - 0.5 * (len(labels) - 1)) * width
        ax.bar(xs + offset, median_matrix[i], width=width, label=label, color=c[i % len(c)], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_1d_spectrum(
    xgrid,
    ppd_list,
    labels,
    limits=(5, 95),
    xlabel="x",
    ylabel="p(x)",
    xlim=None,
    ylim=None,
    colors=None,
    figsize=(24, 10),
    logy=True,
):
    if colors is None:
        import seaborn as sns
        colors = sns.color_palette('colorblind')

    xgrid = np.asarray(xgrid)
    fig, ax = plt.subplots(figsize=figsize)

    means = []
    for i, ppd in enumerate(ppd_list):
        # We assume ppd is a tuple of (median, lower, upper)
        median, lower, upper = ppd 
        color = colors[i % len(colors)]

        ax.fill_between(
            xgrid, np.asarray(lower), np.asarray(upper),
            alpha=0.15,
            color=color,
            label=labels[i],
            lw=0
        )

        ax.plot(xgrid, np.asarray(lower), color=color, lw=0.8, alpha=0.6)
        ax.plot(xgrid, np.asarray(upper), color=color, lw=0.8, alpha=0.6)
        
        # Optional: plotting the median to represent the curve's center
        ax.plot(xgrid, np.asarray(median), color=color, lw=2.5)

        means.append(np.asarray(median))

    if xlim is None:
        xlim = (xgrid.min(), xgrid.max())
    if ylim is None:
        mean_arr = np.vstack(means)
        ymin = max(1e-6, float(mean_arr.min()))
        ymax = float(mean_arr.max()) * 1.5
        ylim = (ymin, ymax)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel, fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.legend(fontsize=24, frameon=False)
    ax.tick_params(labelsize=22)

    if logy:
        ax.set_yscale("log")

    fig.tight_layout()
    return fig

# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=["."],
        help="One or more run directories (PL1, PL2, Bump, etc.); defaults to the current directory"
    )
    parser.add_argument("--mmin", type=float, default=1.0)
    parser.add_argument("--mmax", type=float, default=100.0)
    parser.add_argument("--nm", type=int, default=300)
    parser.add_argument("--nq", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    
    # New arguments for chieff
    parser.add_argument("--nchi", type=int, default=50)
    parser.add_argument("--chimin", type=float, default=-1.0)
    parser.add_argument("--chimax", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cred_lo", type=float, default=5.0)
    parser.add_argument("--cred_hi", type=float, default=95.0)
    args = parser.parse_args()

    mgrid = np.linspace(args.mmin, args.mmax, args.nm)
    qgrid = np.linspace(0.01, 1.0, args.nq)
    zgrid = np.linspace(0.0, 2.0, args.nz)
    chigrid = np.linspace(args.chimin, args.chimax, args.nchi)

    pm1_list   = []
    pm2_list   = []
    pq_list    = []
    pz_list    = []
    pchi_list  = []
    sample_list = []
    parameter_labels_list = []
    ppd_mean_rows = []
    labels     = []
    logZs      = []
    logZerrs   = []
    
    limits = (args.cred_lo, args.cred_hi)
    
    for run_dir in args.run_dirs:
        print(f"\n=== Processing model: {run_dir} ===")

        settings, samples, logZ, logZerr = load_run(run_dir)
        pop_model = pop_model_parser(
            settings["pop_model"],
            shared_beta=bool(settings.get("shared_beta", True)),
            shared_spin=bool(settings.get("shared_spin", True)),
            shared_gamma=bool(settings.get("shared_gamma", True)),
        )

        # Convert to log10 if available; preserve missing evidence as None.
        if logZ is not None:
            log10Z = logZ / np.log(10.0)
            log10Zerr = logZerr / np.log(10.0) if logZerr is not None else 0.0
        else:
            log10Z = None
            log10Zerr = None

        logZs.append(log10Z)
        logZerrs.append(log10Zerr)

        pm1_samples, pm2_samples, pq_samples, pz_samples, pchi_samples = (
            posterior_predictive_mass_distributions_jax(
                pop_model, settings, samples, mgrid, qgrid, zgrid, chigrid, batch_size=args.batch_size
            )
        )

        # Summaries
        pm1_med, pm1_lo, pm1_hi = summarize_ppd(pm1_samples, limits)
        pm2_med, pm2_lo, pm2_hi = summarize_ppd(pm2_samples, limits)
        pq_med,  pq_lo,  pq_hi  = summarize_ppd(pq_samples,  limits)
        pz_med,  pz_lo,  pz_hi  = summarize_ppd(pz_samples,  limits)
        pchi_med, pchi_lo, pchi_hi = summarize_ppd(pchi_samples, limits)

        parameter_labels = list(settings.get("labels", []))
        if parameter_labels and len(parameter_labels) == samples.shape[1]:
            sample_list.append(np.asarray(samples))
            parameter_labels_list.append(parameter_labels)

        ppd_mean_rows.append([
            float(jnp.trapezoid(mgrid * pm1_med, mgrid)),
            float(jnp.trapezoid(mgrid * pm2_med, mgrid)),
            float(jnp.trapezoid(qgrid * pq_med, qgrid)),
            float(jnp.trapezoid(zgrid * pz_med, zgrid)),
            float(jnp.trapezoid(chigrid * pchi_med, chigrid)),
        ])

        # Store summaries as tuples for plotting
        pm1_list.append((pm1_med, pm1_lo, pm1_hi))
        pm2_list.append((pm2_med, pm2_lo, pm2_hi))
        pq_list.append((pq_med, pq_lo, pq_hi))
        pz_list.append((pz_med, pz_lo, pz_hi))
        pchi_list.append((pchi_med, pchi_lo, pchi_hi))

        # Free GPU memory
        del pm1_samples, pm2_samples, pq_samples, pz_samples, pchi_samples
        jax.clear_caches()

        model_label = settings.get("model_name", os.path.basename(run_dir))
        labels.append(model_label)

    # p(m1)
    fig_m1 = plot_1d_spectrum(
        mgrid,
        pm1_list,
        labels,
        limits,
        xlabel=r"$m_1$ [$M_\odot$]",
        ylabel=r"$p(m_1)$ [$M_\odot^{-1}$]",
        logy=True,
        ylim=(1e-5, 1.0),
    )
    fig_m1.savefig("pm1_all_models.pdf")

    # p(m2)
    fig_m2 = plot_1d_spectrum(
        mgrid,
        pm2_list,
        labels,
        limits,
        xlabel=r"$m_2$ [$M_\odot$]",
        ylabel=r"$p(m_2)$ [$M_\odot^{-1}$]",
        logy=True,
        ylim=(1e-5, 1.0),
    )
    fig_m2.savefig("pm2_all_models.pdf")

    # p(q)
    fig_q = plot_1d_spectrum(
        qgrid,
        pq_list,
        labels,
        limits,
        xlabel=r"$q$",
        ylabel=r"$p(q)$",
        xlim=(0.0, 1.0),
        ylim=(1e-6, 10.0),
        logy=True,
    )
    fig_q.savefig("pq_all_models.pdf")

    # p(z)
    fig_z = plot_1d_spectrum(
        zgrid,
        pz_list,
        labels,
        limits,
        xlabel=r"$z$",
        ylabel=r"$p(z)$",
        xlim=(0.0, 2.0),
        ylim=(1e-2, 10.0),
        logy=True,
    )
    fig_z.savefig("pz_all_models.pdf")
    
    # p(chi)
    fig_chi = plot_1d_spectrum(
        chigrid, pchi_list, labels, limits,
        xlabel=r"$\chi_\mathrm{eff}$",
        ylabel=r"$p(\chi_\mathrm{eff})$",
        xlim=(args.chimin, args.chimax), ylim=(1e-2, 10.0), logy=True
    )
    fig_chi.savefig("pchi_all_models.pdf")

    fig_corner = plot_parameter_corner(sample_list, labels, parameter_labels_list)
    if fig_corner is not None:
        fig_corner.savefig("posterior_parameter_corner.pdf")

    if ppd_mean_rows:
        fig_moments = plot_ppd_moment_summary(
            [r"$E[m_1]$", r"$E[m_2]$", r"$E[q]$", r"$E[z]$", r"$E[\chi_\mathrm{eff}]$"],
            np.asarray(ppd_mean_rows), labels, "Posterior-predictive median mean"
        )
        fig_moments.savefig("ppd_median_means.pdf")

    # ------------------------------------------------------------
    # Evidence comparison (separate figure)
    # ------------------------------------------------------------
    if any(z is not None for z in logZs):
        evidence_items = [(label, z, ze or 0.0) for label, z, ze in zip(labels, logZs, logZerrs) if z is not None]
        fig_ev = plot_model_evidences(
            [item[0] for item in evidence_items],
            [item[1] for item in evidence_items],
            [item[2] for item in evidence_items],
        )
        fig_ev.savefig("model_evidences.pdf")

        print("\n=== Model Evidences ===")
        for label, z, ze in zip(labels, logZs, logZerrs):
            if z is None:
                print(f"{label:20s} log10Z = unavailable")
            else:
                print(f"{label:20s} log10Z = {z} ± {ze}")

        print_bayes_factors(labels, logZs)
    else:
        print("\nNo evidence information found in any run.")
        
    # ------------------------------------------------------------
    # Bayes factor matrices
    # ------------------------------------------------------------
    if _should_plot_bayes_factor_matrix(labels, logZs):
        # Full pairwise matrix
        fig_pair = plot_bayes_factor_matrix_pairwise(labels, logZs, logZerrs)
        fig_pair.savefig("bayes_factors_pairwise.pdf")
        print("Saved bayes_factors_pairwise.pdf")

    elif len(labels) < 2:
        print("\nSkipping Bayes factor matrix: at least two run directories are required.")

    else:
        print("\nNo evidence available to compute Bayes factors.")


if __name__ == "__main__":
    main()