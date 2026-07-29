"""
analyze.py
----------
Post-processing helpers for sky-anisotropy runs: summarise the dipole
posterior (amplitude + preferred direction) and build a posterior sky map for
the ``sphere_gp`` model.  Kept dependency-light — the dipole summary is pure
NumPy; the GP map imports ``jax``/``healpy`` lazily, only when called.

The isotropy-vs-anisotropy comparison itself needs no special code: the
samplers already report ``logZ`` per run, so the Bayes factor is
``logZ(anisotropic) - logZ(isotropic)``.
"""

from __future__ import annotations

import numpy as np

from .registry import sky_model_prior_parser, get_sky_model


def _sky_column_indices(labels, sky_model):
    """Return ``{ascii_name: column_index}`` for the sky parameters in ``labels``.

    ``labels`` are the (LaTeX) sampler labels saved with a run; we match them to
    the sky block by regenerating the model's ``(label, name)`` pairs.
    """
    model = get_sky_model(sky_model)
    label_to_name = {s.label: s.name for s in model.param_specs}
    out = {}
    for col, lbl in enumerate(labels):
        if lbl in label_to_name:
            out[label_to_name[lbl]] = col
    return out


def summarize_dipole_posterior(samples, labels, quantiles=(0.05, 0.5, 0.95)):
    """Summarise the dipole amplitude and direction from posterior ``samples``.

    Parameters
    ----------
    samples : (N, ndim) array of posterior draws.
    labels  : list of parameter labels aligned to ``samples`` columns.
    quantiles : reported quantiles for the amplitude and direction.

    Returns
    -------
    dict with the amplitude ``|d|`` quantiles, the per-component quantiles, the
    posterior-mean preferred direction ``(ra_deg, dec_deg)``, and the fraction
    of posterior mass consistent with isotropy proxied by ``P(|d| < 0.05)``.
    """
    idx = _sky_column_indices(labels, "dipole")
    missing = {"sky_dx", "sky_dy", "sky_dz"} - set(idx)
    if missing:
        raise KeyError(f"dipole parameters not found in labels: {sorted(missing)}")

    samples = np.asarray(samples)
    dx = samples[:, idx["sky_dx"]]
    dy = samples[:, idx["sky_dy"]]
    dz = samples[:, idx["sky_dz"]]
    amp = np.sqrt(dx**2 + dy**2 + dz**2)

    # Posterior-mean direction (mean vector, then normalise).
    dbar = np.array([dx.mean(), dy.mean(), dz.mean()])
    norm = np.linalg.norm(dbar)
    if norm > 0:
        nhat = dbar / norm
        ra_deg = float(np.rad2deg(np.arctan2(nhat[1], nhat[0])) % 360.0)
        dec_deg = float(np.rad2deg(np.arcsin(np.clip(nhat[2], -1.0, 1.0))))
    else:
        ra_deg = dec_deg = float("nan")

    return {
        "amplitude_quantiles": {q: float(np.quantile(amp, q)) for q in quantiles},
        "component_quantiles": {
            "dx": {q: float(np.quantile(dx, q)) for q in quantiles},
            "dy": {q: float(np.quantile(dy, q)) for q in quantiles},
            "dz": {q: float(np.quantile(dz, q)) for q in quantiles},
        },
        "mean_direction_deg": {"ra": ra_deg, "dec": dec_deg},
        "P_amp_lt_0.05": float(np.mean(amp < 0.05)),
    }


def summarize_multipole_posterior(samples, labels, sky_model="multipole",
                                  quantiles=(0.05, 0.5, 0.95)):
    """Summarise a multipole run: the angular power spectrum
    ``C_ℓ = (1 / (2ℓ+1)) Σ_m a_lm²`` (per ℓ) and the per-coefficient quantiles.

    The ``1/(2ℓ+1)`` is the conventional CMB/LSS normalisation — ``C_ℓ`` is the
    power PER MODE, so it is the quantity that equals the isotropic variance of
    the ``a_lm`` and is comparable across ℓ (``Σ_m a_lm²`` alone is the TOTAL
    multipole power and grows with the ``2ℓ+1`` mode count even for a
    scale-invariant field).  This function previously reported the total and
    labelled it ``C_ℓ``.

    Returns a dict with ``C_ell_quantiles`` (``{ℓ: {q: value}}``),
    ``coeff_quantiles`` (``{name: {q: value}}``), and ``C_ell_samples``
    (``{ℓ: (N,) array}``) for downstream plotting.
    """
    idx = _sky_column_indices(labels, sky_model)
    model = get_sky_model(sky_model)
    names = [s.name for s in model.param_specs]
    missing = set(names) - set(idx)
    if missing:
        raise KeyError(f"{sky_model} parameters not found in labels: {sorted(missing)}")

    samples = np.asarray(samples)
    by_l = {}
    for n in names:                       # name format: sky_a_l{ℓ}_m{m}
        parts = n.split("_")
        ell = int(parts[2][1:])
        by_l.setdefault(ell, []).append(idx[n])

    cl_quant, cl_samples = {}, {}
    for ell, cols in sorted(by_l.items()):
        # C_ℓ per posterior sample: power PER MODE, i.e. divided by the
        # 2ℓ+1 mode multiplicity (the conventional definition).
        cl = np.sum(samples[:, cols] ** 2, axis=1) / (2.0 * ell + 1.0)
        cl_samples[ell] = cl
        cl_quant[ell] = {q: float(np.quantile(cl, q)) for q in quantiles}
    coeff_quant = {
        n: {q: float(np.quantile(samples[:, idx[n]], q)) for q in quantiles}
        for n in names
    }
    return {"C_ell_quantiles": cl_quant, "coeff_quantiles": coeff_quant,
            "C_ell_samples": cl_samples}


def sphere_gp_posterior_map(samples, labels, nside=16, max_draws=200,
                            sky_model="sphere_gp", z_slice=None):
    """Posterior-mean sky density ``g(n̂)`` (or ``g(n̂, z_slice)`` for the 3-D
    models) of a GP-sky run as a HEALPix map (RING ordering), for
    ``healpy.mollview``.

    Averages ``g`` over up to ``max_draws`` posterior samples evaluated at the
    HEALPix pixel centres.  ``z_slice=None`` evaluates at z=0 (angular models
    ignore z); for ``sphere_gp_z`` / ``overdensity_gp`` pass a redshift to map a
    distance shell.  Imports ``jax``/``healpy`` lazily.
    """
    import healpy as hp
    import jax.numpy as jnp

    idx = _sky_column_indices(labels, sky_model)
    model = get_sky_model(sky_model)
    names = [s.name for s in model.param_specs]
    missing = set(names) - set(idx)
    if missing:
        raise KeyError(f"{sky_model} parameters not found in labels: {sorted(missing)}")
    cols = [idx[n] for n in names]

    samples = np.asarray(samples)
    if samples.shape[0] > max_draws:
        sel = np.linspace(0, samples.shape[0] - 1, max_draws).astype(int)
        samples = samples[sel]

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    nx = jnp.asarray(np.sin(theta) * np.cos(phi))
    ny = jnp.asarray(np.sin(theta) * np.sin(phi))
    nz = jnp.asarray(np.cos(theta))

    z_arr = jnp.zeros(npix) if z_slice is None else jnp.full(npix, float(z_slice))
    acc = np.zeros(npix)
    for row in samples:
        theta_sky = jnp.asarray(row[cols])
        acc += np.asarray(jnp.exp(model.log_g_sky(nx, ny, nz, z_arr, theta_sky)))
    return acc / len(samples)


# --------------------------------------------------------------------------
# Figures (matplotlib / healpy imported lazily)
# --------------------------------------------------------------------------

def plot_dipole_posterior(samples, labels, figsize=(11.0, 4.5)):
    """Two-panel dipole figure: amplitude ``|d|`` posterior and a Mollweide
    scatter of the per-sample preferred direction.  Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    idx = _sky_column_indices(labels, "dipole")
    missing = {"sky_dx", "sky_dy", "sky_dz"} - set(idx)
    if missing:
        raise KeyError(f"dipole parameters not found in labels: {sorted(missing)}")

    samples = np.asarray(samples)
    dx = samples[:, idx["sky_dx"]]
    dy = samples[:, idx["sky_dy"]]
    dz = samples[:, idx["sky_dz"]]
    amp = np.sqrt(dx**2 + dy**2 + dz**2)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(amp, bins=40, density=True, color="C0", alpha=0.8)
    med = float(np.median(amp))
    ax1.axvline(med, color="k", lw=1.5, label=f"median |d| = {med:.3f}")
    ax1.axvline(0.0, color="0.4", ls="--", lw=1.0, label="isotropy (|d|=0)")
    ax1.set_xlabel(r"dipole amplitude $|d|$")
    ax1.set_ylabel("posterior density")
    ax1.legend(fontsize=8)

    # Per-sample direction (only where the amplitude is resolved).
    good = amp > 1e-6
    ra = np.arctan2(dy[good], dx[good])  # already in (-pi, pi] for Mollweide
    dec = np.arcsin(np.clip(dz[good] / amp[good], -1.0, 1.0))
    ax2 = fig.add_subplot(1, 2, 2, projection="mollweide")
    ax2.scatter(ra, dec, s=2, alpha=0.05, color="C0", rasterized=True)
    summ = summarize_dipole_posterior(samples, labels)
    ra_m = np.deg2rad(((summ["mean_direction_deg"]["ra"] + 180.0) % 360.0) - 180.0)
    dec_m = np.deg2rad(summ["mean_direction_deg"]["dec"])
    ax2.scatter([ra_m], [dec_m], marker="*", s=160, color="crimson",
                edgecolor="k", zorder=5, label="mean direction")
    ax2.grid(True)
    ax2.set_title("preferred direction (posterior)", fontsize=9)
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_sphere_gp_map(samples, labels, nside=16, max_draws=200,
                       sky_model="sphere_gp", z_slice=None):
    """Mollweide HEALPix map of the posterior-mean sky density ``g(n̂)`` (or
    ``g(n̂, z_slice)`` for the 3-D models).  Returns a matplotlib Figure.  Imports
    ``healpy`` lazily."""
    import matplotlib.pyplot as plt
    import healpy as hp

    m = sphere_gp_posterior_map(samples, labels, nside=nside, max_draws=max_draws,
                                sky_model=sky_model, z_slice=z_slice)
    zlabel = "" if z_slice is None else rf" at $z={float(z_slice):.2f}$"
    fig = plt.figure(figsize=(8.0, 5.0))
    hp.mollview(
        m,
        fig=fig.number,
        title=rf"Posterior-mean sky density $g(\hat n)${zlabel}",
        unit=r"$g(\hat n)$",
        cmap="RdBu_r",
        min=float(np.min(m)),
        max=float(np.max(m)),
    )
    hp.graticule()
    return fig


def plot_multipole_cl(samples, labels, sky_model="multipole", figsize=(7.0, 4.5)):
    """Angular power spectrum ``C_ℓ = (1/(2ℓ+1)) Σ_m a_lm²`` (posterior median +
    5–95%) per multipole ℓ.  Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    summ = summarize_multipole_posterior(samples, labels, sky_model=sky_model)
    cl = summ["C_ell_quantiles"]
    ells = sorted(cl)
    med = [cl[l][0.5] for l in ells]
    lo = [cl[l][0.5] - cl[l][0.05] for l in ells]
    hi = [cl[l][0.95] - cl[l][0.5] for l in ells]

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(ells, med, yerr=[lo, hi], fmt="o", capsize=5, color="C0")
    ax.set_xticks(ells)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$C_\ell = \frac{1}{2\ell+1}\sum_m a_{\ell m}^2$")
    ax.set_title("Angular power spectrum (posterior median, 5–95%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
