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

The cosmology/survey are rebuilt from the FILE's own build stamps (fiducial
cosmology, ``n0``, ``delta``, bias, ``c_mode`` and -- for a selection base --
the fitted theta/family/K(z)/strata), never from the builder defaults: Q is the
fit residual to a specific completeness base, so validating it against a
different base can show a plausible-looking Q while the consumed prior is a
different function.  A table missing the stamps is refused rather than
diagnosed at defaults.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.catalogs.io import load_survey
from darksirens.core.types import (
    C_MODE_AGGREGATE_STRUCT,
    C_MODE_SELECTION_STRUCT,
    SELECTION_FAMILY_SCHECHTER_STRUCT,
    CosmoParams,
    EMCatalog,
)
from darksirens.redshift.completion import completion_curves
from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_with_state,
    eval_redshift_prior_members_with_state,
)
from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
from darksirens.cli.build_lognormal_completion import (
    _fiducial_cosmo_survey,
    _load_stratum_map,
    _stratum_map_sha256,
)
from darksirens.redshift.selection import SELECTION_THETA_FIELDS

_INDEXING = {"compact": 1, "global": 2}


def _stamp(diag, key):
    """A required build stamp; a missing one is fatal (see the module docstring)."""
    if key not in diag:
        raise SystemExit(
            f"FATAL: the completion file carries no '{key}' build stamp, so the "
            "completeness base it was fit residual to cannot be reconstructed "
            "and the plotted reference/prior would be a different function than "
            "the one inference consumes. Rebuild the table with the current "
            "darksirens_build_lognormal_completion."
        )
    return diag[key]


def _survey_from_stamps(loaded, catalog_z_depth, stratum_map_path=None):
    """Rebuild ``(cosmo, survey, stratum_map)`` from the table's build stamps."""
    diag = loaded.get("diagnostics") or {}
    if not isinstance(diag, dict):
        raise SystemExit("FATAL: the completion file's diagnostics are not a "
                         "readable dict; rebuild the table.")
    n0 = float(_stamp(diag, "fiducial_n0"))
    cosmo = CosmoParams(
        H0=float(_stamp(diag, "fiducial_H0")),
        Om0=float(_stamp(diag, "fiducial_Om0")),
        w0=float(_stamp(diag, "fiducial_w0")),
        wa=float(_stamp(diag, "fiducial_wa")),
    )
    # The depth the base was truncated at; fall back to the catalog attr for
    # tables built before the stamp existed.
    z_depth = diag.get("fiducial_z_depth", catalog_z_depth)
    _c, survey = _fiducial_cosmo_survey(
        log10n0=np.log10(n0), delta=float(_stamp(diag, "fiducial_delta")),
        z_depth=z_depth)
    survey = survey._replace(b_miss=float(_stamp(diag, "bias_b_miss")))

    c_mode = str(loaded.get("c_mode") or "per_pixel")
    stratum_map = None
    if c_mode == "aggregate":
        survey = survey._replace(c_mode=C_MODE_AGGREGATE_STRUCT)
    elif c_mode == "selection":
        family = str(diag.get("selection_family") or "gaussian")
        theta = {name: float(_stamp(diag, f"selection_{name}"))
                 for name in SELECTION_THETA_FIELDS[family]}
        kcorr = []
        j = 1
        while f"selection_kcorr_c{j}" in diag:
            kcorr.append(float(diag[f"selection_kcorr_c{j}"]))
            j += 1
        n_strata = int(diag.get("selection_n_strata", 0) or 0)
        strata = None
        if n_strata > 1:
            # Stratified base: the per-pixel curve is routed by the stratum map
            # the table was built with, so the map (verified by hash) is
            # required -- without it every pixel would be diagnosed against the
            # reference stratum's completeness.
            if stratum_map_path is None:
                raise SystemExit(
                    f"FATAL: this table was built on a {n_strata}-stratum "
                    "selection base; pass --stratum-map (the same map, sha256 "
                    f"{diag.get('selection_stratum_map_sha256')}) so each "
                    "pixel is diagnosed against its own C_sel curve.")
            sha = _stratum_map_sha256(stratum_map_path)
            if sha != str(diag.get("selection_stratum_map_sha256")):
                raise SystemExit(
                    "FATAL: --stratum-map sha256 does not match the table's "
                    "stamp; the per-pixel base routing must be identical.")
            # The likelihood's convention: the sampled (M0hat, sigma_M) are the
            # common mode and each stratum carries its own offsets.
            strata = tuple(
                (float(diag[f"selection_s{j}_m_lim"]),
                 float(diag[f"selection_s{j}_M0hat"]) - theta["M0hat"],
                 float(diag[f"selection_s{j}_sigma_M"]) / theta["sigma_M"])
                for j in range(n_strata))
        survey = survey._replace(
            c_mode=C_MODE_SELECTION_STRUCT,
            selection_family=(SELECTION_FAMILY_SCHECHTER_STRUCT
                              if family == "schechter" else None),
            k_corr_coeffs=(tuple(kcorr) or None),
            selection_strata=strata,
            **theta,
        )
        if strata is not None:
            nside_map = int(round((len(np.asarray(loaded["logq_map"])) / 12) ** 0.5))
            stratum_map = _load_stratum_map(stratum_map_path, nside_map, n_strata)
    elif c_mode != "per_pixel":
        raise SystemExit(f"FATAL: unknown c_mode stamp {c_mode!r} in the "
                         "completion file.")
    return cosmo, survey, c_mode, stratum_map


def _full_catalog(catalog_path, stratum_map=None, **lss):
    import healpy as hp
    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    if stratum_map is not None:
        lss["pixel_stratum_map"] = jnp.asarray(stratum_map, dtype=jnp.int32)
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
    p.add_argument("--stratum-map", default=None,
                   help="Stratum map the table was built with (required for a "
                        "multi-stratum selection base; verified by sha256).")
    opts = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(opts.outdir, exist_ok=True)
    loaded = load_lss_completion_hdf5(opts.lss_completion)
    # The completeness base is the FILE's, not the builder defaults: Q is a fit
    # residual to it, so diagnosing against another base validates nothing.
    *_ignored, catalog_z_depth = load_survey(opts.catalog)
    cosmo, survey, c_mode, stratum_map = _survey_from_stamps(
        loaded, catalog_z_depth, stratum_map_path=opts.stratum_map)
    from darksirens.cli.common import _row as _print_row, _section
    _section("Build fiducials (from the completion file)")
    _print_row("Cosmology", f"H0={cosmo.H0:.4f} Om0={cosmo.Om0:.4f} "
                            f"w0={cosmo.w0:.4f} wa={cosmo.wa:.4f}")
    _print_row("Density", f"n0={survey.n0:.6g} Mpc^-3  delta={survey.delta:.4f}")
    _print_row("Field bias", f"b_miss={survey.b_miss:.4f}")
    _print_row("c_mode", c_mode)
    _print_row("z_depth", "none" if survey.z_depth is None
                          else f"{float(survey.z_depth):.4f}")
    if c_mode == "selection":
        _print_row("Selection theta_hat",
                   "  ".join(f"{k}={float(getattr(survey, k)):.4f}"
                             for k in SELECTION_THETA_FIELDS[
                                 "schechter" if survey.selection_family
                                 is not None else "gaussian"]))
    # b_miss = 0 -> the legacy LSS factor is 1, i.e. the homogeneous reference.
    survey_homog = survey._replace(b_miss=0.0)
    idx_enum = _INDEXING.get(str(loaded.get("indexing", "global")), 0)
    logq_map = loaded["logq_map"]
    logq_members = loaded["logq_members"]
    row = int(opts.pixel)
    z = np.asarray(zgrid)
    pix = jnp.full(z.size, row, jnp.int32)

    # --- catalogs: homogeneous, MAP (deterministic Q), ensemble ---
    cat_homog = _full_catalog(opts.catalog, stratum_map=stratum_map)
    cat_map = _full_catalog(opts.catalog, stratum_map=stratum_map,
                            lss_completion_logq=jnp.asarray(logq_map),
                            lss_completion_indexing=idx_enum)
    curves_homog = completion_curves(cosmo, survey_homog, cat_homog)
    curves_map = completion_curves(cosmo, survey, cat_map)

    Q_map = np.exp(np.asarray(logq_map)[row])
    dN_homog = np.asarray(curves_homog.dN_miss[row])
    dN_map = np.asarray(curves_map.dN_miss[row])

    have_members = logq_members is not None
    if have_members:
        cat_ens = _full_catalog(opts.catalog, stratum_map=stratum_map,
                                lss_completion_logq_members=jnp.asarray(logq_members),
                                lss_completion_indexing=idx_enum)
        curves_ens = completion_curves(cosmo, survey, cat_ens)
        # The (M, N_rows, N_grid) member cube is no longer materialised; each
        # member's density is base_miss * Q_eff_m, reconstructed here for the plot
        # (Q_eff = exp(clip(logQ, ±_LOGQ_CLIP)), relaxed to 1 beyond z_depth --
        # node-for-node identical to the old cube row).
        from darksirens.redshift.completion import _LOGQ_CLIP
        base_row = np.asarray(curves_ens.base_miss[row])             # (NG,)
        Q_eff = np.exp(np.clip(np.asarray(logq_members)[:, row, :], -_LOGQ_CLIP, _LOGQ_CLIP))
        if survey.z_depth is not None:
            _below = np.asarray(zgrid) <= survey.z_depth
            Q_eff = np.where(_below[None, :], Q_eff, 1.0)
        Q_mem = np.exp(np.asarray(logq_members)[:, row, :])          # (M, NG)
        dN_mem = base_row[None, :] * Q_eff                           # (M, NG)
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
