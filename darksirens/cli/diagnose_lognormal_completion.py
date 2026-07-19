"""
darksirens_diagnose_lognormal_completion
----------------------------------------
Diagnostic plots for an LSS-conditioned lognormal completion file, for a single
pixel: the completion factor ``Q_LSS(p,z)``, the missing-galaxy density
``dN_miss``, and the assembled redshift prior ``p(z|p)`` — each comparing the
homogeneous reference, the MAP, and (if an ensemble is present) the
posterior-mean with a 16-84% band.

The Bayesian marginalised prior is ``p_Bayes(z|p) = mean_m p_m(z|p)`` with each
member normalised **individually** (via
``eval_redshift_prior_members_with_state``), not by averaging Q then normalising.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.catalogs.io import load_survey
from darksirens.core.types import EMCatalog, SurveyParams
from darksirens.redshift.completion import completion_curves
from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_with_state,
    eval_redshift_prior_members_with_state,
)
from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
from darksirens.cli.build_lognormal_completion import _fiducial_cosmo_survey

_INDEXING = {"compact": 1, "global": 2}


def _full_catalog(catalog_path, **lss):
    import healpy as hp
    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    return EMCatalog(
        apix=float(hp.nside2pixarea(int(nside))),
        zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals), wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals), delta_g_pix_z=jnp.zeros((1, int(zgrid.size))),
        dN_obs_kde=None, pixel_to_cache_idx=None, unique_pixels=None, **lss,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose an LSS lognormal completion file for one pixel.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--lss-completion", required=True)
    p.add_argument("--pixel", type=int, required=True, help="Catalog row / global HEALPix pixel index.")
    p.add_argument("--outdir", default=".")
    opts = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(opts.outdir, exist_ok=True)
    cosmo, survey = _fiducial_cosmo_survey()
    # b_miss = 0 -> the legacy LSS factor is 1, i.e. the homogeneous reference.
    survey_homog = survey._replace(b_miss=0.0)
    loaded = load_lss_completion_hdf5(opts.lss_completion)
    idx_enum = _INDEXING.get(str(loaded.get("indexing", "global")), 0)
    logq_map = loaded["logq_map"]
    logq_members = loaded["logq_members"]
    row = int(opts.pixel)
    z = np.asarray(zgrid)
    pix = jnp.full(z.size, row, jnp.int32)

    # --- catalogs: homogeneous, MAP (deterministic Q), ensemble ---
    cat_homog = _full_catalog(opts.catalog)
    cat_map = _full_catalog(opts.catalog, lss_completion_logq=jnp.asarray(logq_map),
                            lss_completion_indexing=idx_enum)
    curves_homog = completion_curves(cosmo, survey_homog, cat_homog)
    curves_map = completion_curves(cosmo, survey, cat_map)

    Q_map = np.exp(np.asarray(logq_map)[row])
    dN_homog = np.asarray(curves_homog.dN_miss[row])
    dN_map = np.asarray(curves_map.dN_miss[row])

    have_members = logq_members is not None
    if have_members:
        cat_ens = _full_catalog(opts.catalog, lss_completion_logq_members=jnp.asarray(logq_members),
                                lss_completion_indexing=idx_enum)
        curves_ens = completion_curves(cosmo, survey, cat_ens)
        Q_mem = np.exp(np.asarray(logq_members)[:, row, :])          # (M, NG)
        dN_mem = np.asarray(curves_ens.dN_miss_members[:, row, :])   # (M, NG)
        Q_mean, Q_lo, Q_hi = Q_mem.mean(0), np.percentile(Q_mem, 16, 0), np.percentile(Q_mem, 84, 0)
        dN_mean = dN_mem.mean(0)
        dN_lo, dN_hi = np.percentile(dN_mem, 16, 0), np.percentile(dN_mem, 84, 0)

    # --- redshift priors ---
    lp_homog = np.exp(np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", prepare_redshift_prior_state("dark_sirens", cosmo, survey_homog, cat_homog),
        zgrid, pix, cosmo, survey_homog, cat_homog)))
    lp_map = np.exp(np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat_map),
        zgrid, pix, cosmo, survey, cat_map)))
    if have_members:
        state_ens = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat_ens)
        lpm = np.asarray(eval_redshift_prior_members_with_state(
            "dark_sirens", state_ens, zgrid, pix, cosmo, survey, cat_ens))  # (M, NG)
        pm = np.exp(lpm)
        p_bayes = pm.mean(0)
        p_lo, p_hi = np.percentile(pm, 16, 0), np.percentile(pm, 84, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.axhline(1.0, color="k", ls=":", lw=1, label="homogeneous (Q=1)")
    ax.plot(z, Q_map, color="C1", label="MAP")
    if have_members:
        ax.fill_between(z, Q_lo, Q_hi, color="C0", alpha=0.25, label="16-84%")
        ax.plot(z, Q_mean, color="C0", label="posterior mean")
    ax.set(xlabel="z", ylabel=r"$Q_{\rm LSS}(p,z)$", title=f"Completion factor (pixel {row})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(z, dN_homog, color="k", ls=":", label="homogeneous")
    ax.plot(z, dN_map, color="C1", label="MAP")
    if have_members:
        ax.fill_between(z, dN_lo, dN_hi, color="C0", alpha=0.25, label="16-84%")
        ax.plot(z, dN_mean, color="C0", label="posterior mean")
    ax.set(xlabel="z", ylabel=r"$dN_{\rm miss}/dz$", title="Missing-galaxy density")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(z, lp_homog, color="k", ls=":", label="homogeneous")
    ax.plot(z, lp_map, color="C1", label="MAP")
    if have_members:
        ax.fill_between(z, p_lo, p_hi, color="C0", alpha=0.25, label="member 16-84%")
        ax.plot(z, p_bayes, color="C0", label=r"$p_{\rm Bayes}=\langle p_m\rangle$")
    ax.set(xlabel="z", ylabel=r"$p(z\,|\,p)$", title="Redshift prior")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(opts.outdir, f"lss_completion_pixel{row}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    from darksirens.cli.common import _ok
    _ok(f"lss_completion_pixel{row}.pdf  →  {out}")


if __name__ == "__main__":
    main()
