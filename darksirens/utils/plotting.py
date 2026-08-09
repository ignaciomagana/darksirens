"""Shared publication-quality plotting helpers.

Provides a single style entry point, a parameter-label classifier (used to make
corner plots readable for high-dimensional models), a high-dimensional-aware
corner plot, and a compact summary for Gaussian-process latent parameters.
"""

import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared publication style
# ---------------------------------------------------------------------------
def set_publication_style():
    """Apply the package's publication styling (serif/CM fonts, seaborn ticks)."""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_context('talk')
    sns.set_style('ticks')


# ---------------------------------------------------------------------------
# Parameter-label classification
# ---------------------------------------------------------------------------
# Cosmology and survey labels are stored as plain identifiers; population
# parameters carry LaTeX labels (e.g. r"$\alpha$"); GP/binned-GP latents are
# the whitened r"$\xi_i$" parameters (name "xi_i").
COSMOLOGY_LABELS = ("H0", "Om0", "w0", "wa")
SURVEY_LABELS = ("log10n0", "z50", "w", "delta", "b_miss", "alpha_miss",
                 "sigma_kde", "m_lim", "M0hat", "sigma_M",
                 "Mstar_hat", "alpha", "M_faint_offset")


def is_latent_label(label):
    """True if ``label`` is a GP whitened latent (``xi_i`` / ``$\\xi_i$``)."""
    s = str(label).strip().strip("$")
    return s.startswith("xi_") or s.startswith(r"\xi") or (r"\xi" in str(label))


def classify_label(label):
    """Return one of 'cosmology', 'survey', 'latent', 'population' for ``label``."""
    if label in COSMOLOGY_LABELS:
        return "cosmology"
    if is_latent_label(label):
        return "latent"
    if label in SURVEY_LABELS:
        return "survey"
    return "population"


def classify_labels(labels):
    """Classify a list of labels (see :func:`classify_label`)."""
    return [classify_label(lbl) for lbl in labels]


def headline_indices(labels):
    """Indices of the non-nuisance ('headline') parameters: everything but latents."""
    return [i for i, lbl in enumerate(labels) if classify_label(lbl) != "latent"]


def latent_indices(labels):
    """Indices of the GP whitened-latent parameters."""
    return [i for i, lbl in enumerate(labels) if classify_label(lbl) == "latent"]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def get_bounds(data):
    """Return (median, +error, -error) using the 90% credible interval."""
    data = np.asarray(data)
    med = np.median(data)
    upper_lim = np.percentile(data, 95)
    lower_lim = np.percentile(data, 5)
    return med, upper_lim - med, med - lower_lim


# ---------------------------------------------------------------------------
# Corner plot (high-dimensional-aware)
# ---------------------------------------------------------------------------
def make_production_corner(samples, labels, color=None, figsize=None,
                           bins=20, hist_alpha=0.7, custom_bounds=None,
                           max_params=14, exclude_latents=True):
    """High-quality corner plot with readable ticks.

    For high-dimensional models (e.g. GP / ``gppop`` with dozens of ``xi_i``
    latents) a full N×N grid is illegible.  When ``exclude_latents`` is set and
    the dimension exceeds ``max_params``, only the 'headline' parameters
    (cosmology + physical population hyperparameters + survey) are shown; use
    :func:`make_latent_summary` for the latents.
    """
    set_publication_style()
    samples = np.asarray(samples)
    labels = list(labels)

    if color is None:
        color = sns.color_palette('colorblind')[4]

    # Subset to the readable parameter set when the model is high-dimensional.
    if exclude_latents and samples.shape[1] > max_params:
        idx = headline_indices(labels)
        if not idx:                       # degenerate: nothing but latents
            idx = list(range(samples.shape[1]))
    else:
        idx = list(range(samples.shape[1]))

    samples = samples[:, idx]
    labels = [labels[i] for i in idx]
    if custom_bounds is not None:
        custom_bounds = [custom_bounds[i] for i in idx]

    ndim = samples.shape[1]
    if figsize is None:
        side = float(max(6.0, min(28.0, 2.2 * ndim)))
        figsize = (side, side)

    fig = plt.figure(figsize=figsize)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", color])

    for i in range(ndim):
        # --- 1D histograms (diagonal) ---
        ax = fig.add_subplot(ndim, ndim, int(1 + (ndim + 1) * i))

        data_i = samples[:, i]
        p_bounds_i = custom_bounds[i] if custom_bounds else (np.min(data_i), np.max(data_i))
        edges = np.linspace(p_bounds_i[0], p_bounds_i[1], bins)

        ax.hist(data_i, bins=edges, color=color, alpha=hist_alpha, density=True,
                zorder=0, rasterized=True)
        ax.hist(data_i, bins=edges, histtype='step', color='black', density=True, zorder=2)

        ax.grid(True, dashes=(1, 3))
        ax.set_xlim(p_bounds_i)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(2))

        med, up, lo = get_bounds(data_i)
        ax.axvline(med + up, color=color, ls='--')
        ax.axvline(med - lo, color=color, ls='--')
        ax.set_title(fr"${med:.2f}^{{+{up:.2f}}}_{{-{lo:.2f}}}$", fontsize=20)

        if i != 0:
            ax.set_yticklabels([])
        if i == ndim - 1:
            ax.set_xlabel(labels[i], fontsize=24)
        else:
            ax.set_xticklabels([])

        # --- 2D hexbins (lower triangle) ---
        for j in range(i + 1, ndim):
            ax_2d = fig.add_subplot(ndim, ndim, int(1 + i + j * ndim))

            data_j = samples[:, j]
            p_bounds_j = custom_bounds[j] if custom_bounds else (np.min(data_j), np.max(data_j))

            ax_2d.hexbin(data_i, data_j,
                         cmap=cmap, mincnt=1, gridsize=bins, rasterized=True,
                         extent=(p_bounds_i[0], p_bounds_i[1], p_bounds_j[0], p_bounds_j[1]),
                         linewidths=(0,), zorder=0)

            ax_2d.set_xlim(p_bounds_i)
            ax_2d.set_ylim(p_bounds_j)
            ax_2d.grid(True, dashes=(1, 3))
            ax_2d.xaxis.set_major_locator(ticker.MaxNLocator(2, prune=None))
            ax_2d.yaxis.set_major_locator(ticker.MaxNLocator(2, prune=None))

            if i == 0:
                ax_2d.set_ylabel(labels[j], fontsize=24)
            else:
                ax_2d.set_yticklabels([])

            if j == ndim - 1:
                ax_2d.set_xlabel(labels[i], fontsize=24)
            else:
                ax_2d.set_xticklabels([])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    return fig


# ---------------------------------------------------------------------------
# GP-latent summary (caterpillar)
# ---------------------------------------------------------------------------
def make_latent_summary(samples, labels, color=None, figsize=None):
    """Compact caterpillar of the GP whitened-latent marginals.

    Returns ``None`` if the model has no latents.  Each latent is shown as its
    posterior median with a 90% credible interval; the whitened-prior mean
    (``xi = 0``) is marked, so latents whose interval excludes 0 are the ones
    the data constrains away from the prior.
    """
    samples = np.asarray(samples)
    idx = latent_indices(labels)
    if not idx:
        return None

    set_publication_style()
    sub = samples[:, idx]
    med = np.median(sub, axis=0)
    lo = np.percentile(sub, 5, axis=0)
    hi = np.percentile(sub, 95, axis=0)
    n = len(idx)
    y = np.arange(n)

    if color is None:
        color = sns.color_palette('colorblind')[0]
    if figsize is None:
        figsize = (7.0, max(4.0, 0.22 * n))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color='0.6', ls='--', lw=1.0, zorder=0)
    ax.errorbar(med, y, xerr=np.vstack([med - lo, hi - med]),
                fmt='o', ms=4, color=color, ecolor=color,
                elinewidth=1.0, capsize=2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"GP latent $\xi_i$  (median, 90% CI)", fontsize=18)
    ax.grid(True, axis='x', dashes=(1, 3))
    fig.tight_layout()
    return fig
