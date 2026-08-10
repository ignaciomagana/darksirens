"""
darksirens_build_lognormal_completion
-------------------------------------
**Offline** preprocessor that builds an LSS-conditioned lognormal completion
file ``Q_LSS(p, z)`` from a pixelated survey catalog, for later consumption by
the dark-siren redshift prior (``darksirens_inference --lss_completion``).

It must NOT be run inside the GW likelihood — it is a one-off build step.

Pipeline
--------
1. Load the survey catalog (``darksirens.catalogs.io.load_survey``).
2. Bin observed galaxies onto the package ``zgrid`` per pixel  -> ``N_obs``.
3. Compute the matched-kernel completeness ``C`` and homogeneous expected
   counts ``dN_exp`` with the existing completion machinery
   (:func:`darksirens.redshift.completion._precompute_grids`,
   :func:`darksirens.redshift.completion._kde_dndz_obs`) under a *fiducial*
   cosmology/survey (the same fiducials the inference dry-run uses).
   ``--c-mode aggregate`` swaps the per-pixel ``C`` for the ONE sky-aggregate
   ``Cbar`` (the inference ``SurveyParams.c_mode=1`` estimator) and extends
   the fit to EMPTY pixels as N_obs = 0 rows — the void information; the
   choice is stamped in the output attrs and hard-checked at load against
   the consuming survey's ``c_mode``.
4. Build a per-pixel 1-D Gaussian-correlation power spectrum from the **fixed**
   SurveyParams/CosmoParams hyperparameters (correlation length in Mpc mapped to
   grid units via the comoving-distance grid; field amplitude ``lss_sigma``;
   bias ``b_miss``).  ``--lss-corr-length-mpc`` / ``--lss-sigma`` override the
   fiducials AT BUILD TIME only — the values are stamped in the diagnostics and
   are never sampled or marginalised in the inference (asserted by
   tests/test_lss_completion_gp3d.py::test_lss_corr_length_ang_not_sampled).
   The gp3d inducing grid is guarded: a build whose radial node spacing in
   ``zeta = log1p z`` exceeds the GP lengthscale ``ls_z`` is a HARD ERROR
   (:func:`_gp3d_resolution_guard`), because a low-rank GP with node spacing
   much larger than the lengthscale collapses to the prior (Burt et al. 2019,
   arXiv:1903.03571; the shipped 50 Mpc default measured a fitted-vs-truth
   logQ slope of 0.04 on the closure experiment).
5. Run the solver and save an HDF5 completion file (same table contract either
   way): ``--mode radial`` (default) -> independent per-pixel 1-D
   :func:`poisson_lognormal_map`; ``--mode gp3d`` -> ONE low-rank
   Poisson-lognormal field over occupied (pixel x z) voxels reusing the
   (sphere x z) GP, so empty pixels borrow angularly from their neighbours.
6. Per-z mean-one budget renormalization of the output tables (MAP and each
   member independently) under ``w = (1 - C) dN_exp`` over the fitted
   footprint (:func:`renormalize_q_mean_one`), so Q only redistributes the
   missing budget (default ON; ``--no-budget-renorm`` to skip; the choice and
   the removed monopole are stamped in the HDF5 attrs).
"""
from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from darksirens.cli.common import _banner, _section, _row, _end, _ok, _warn
from darksirens.redshift import zgrid
from darksirens.catalogs.io import load_survey
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import r_of_z, H0Planck, Om0Planck, w0Fiducial, waFiducial
from darksirens.redshift.completion import _precompute_grids, _kde_dndz_obs
from darksirens.redshift.selection import SELECTION_THETA_FIELDS
from darksirens.redshift.lognormal_completion import (
    gaussian_correlation_spectrum,
    poisson_lognormal_map,
    laplace_lognormal_members,
    renormalize_q_mean_one,
    save_lss_completion_hdf5,
    lowrank_inducing_nodes,
    build_lowrank_operator,
    poisson_lognormal_gp3d_map,
    laplace_lognormal_gp3d_members,
    eval_logq_gp3d,
)


def _fiducial_cosmo_survey(log10n0=None, delta=None):
    """Fiducial cosmology + survey (matching the inference dry-run defaults).

    ``log10n0``/``delta`` override the expected-density normalisation
    ``dN_exp = n0 * apix * dV_c/dz * (1+z)^delta`` that the Poisson-lognormal
    fit is conditioned on.  Calibrate ``log10n0`` to the survey's observed
    comoving density: a mis-set n0 cannot be separated from completeness and
    is absorbed into Q with spurious redshift structure (which then biases
    any downstream H0 inference — see working/GATES.md V2/V3).
    """
    cosmo = CosmoParams(H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)
    survey = SurveyParams(
        n0=10.0 ** (float(log10n0) if log10n0 is not None else -2.0),
        z50=1.0, w=0.5,
        delta=float(delta) if delta is not None else 0.0,
        b_miss=1.0, alpha_miss=1.0, sigma_kde=0.0,
    )  # lss_corr_length_mpc / lss_sigma take their container defaults
    return cosmo, survey


def _apply_lss_overrides(survey, *, lss_corr_length_mpc=None, lss_sigma=None):
    """Build-time GP hyperparameter overrides (CLI knobs, NEVER sampled).

    ``None`` keeps the SurveyParams fiducial (backward-compatible default);
    the values actually used are stamped in the output diagnostics via the
    (possibly replaced) ``survey`` container.
    """
    if lss_corr_length_mpc is not None:
        survey = survey._replace(lss_corr_length_mpc=float(lss_corr_length_mpc))
    if lss_sigma is not None:
        survey = survey._replace(lss_sigma=float(lss_sigma))
    return survey


def _selection_cbar_fine(selection_fit, cosmo):
    """``C_sel(zgrid; theta_hat)`` for the parametric completeness base.

    Evaluated at the builder's fiducial cosmology; since ``M0hat``
    (``Mstar_hat`` for the schechter family) is h-scaled, the curve is exactly
    H0-invariant either way, so the fiducial choice carries no H0 imprint
    (Om0/w0/wa enter weakly through the distance shape, same footing as every
    other fixed build fiducial).  The Q table is conditioned on this FIXED
    theta_hat while the likelihood samples theta around it -- the same
    first-order convention as the fixed-fiducial n0/delta, guarded at load by
    the theta_hat-vs-prior-center provenance check.
    """
    family = str(selection_fit.get("family", "gaussian"))
    if family == "schechter":
        from darksirens.redshift.selection import c_sel_schechter

        return np.asarray(c_sel_schechter(
            jnp.asarray(zgrid), float(selection_fit["m_lim"]),
            float(selection_fit["Mstar_hat"]), float(selection_fit["alpha"]),
            float(selection_fit["M_faint_offset"]),
            cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), dtype=float)

    from darksirens.redshift.selection import c_sel_gaussian

    return np.asarray(c_sel_gaussian(
        jnp.asarray(zgrid), float(selection_fit["m_lim"]),
        float(selection_fit["M0hat"]), float(selection_fit["sigma_M"]),
        cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa,
        k_corr_coeffs=tuple(selection_fit.get("k_corr_coeffs") or ())),
        dtype=float)


def _selection_theta_stamp(selection_fit):
    """Q-table provenance stamp of the fit's FIXED theta, family-driven.

    ONE source for both builder paths; the key set is
    ``SELECTION_THETA_FIELDS[family]`` (so a gaussian table stamps exactly the
    pre-schechter keys) plus the family tag itself, which the inference CLI
    compares against the run's family before comparing any theta.
    """
    family = str(selection_fit.get("family", "gaussian"))
    return {"selection_family": family,
            **{f"selection_{k}": float(selection_fit[k])
               for k in SELECTION_THETA_FIELDS[family]}}


def _selection_cbar_fine_strata(selection_strata, cosmo):
    """Per-stratum ``C_sel(zgrid; theta_hat_s)`` stack, (S, n_grid).

    Each stratum's curve is evaluated at its OWN fitted theta (the builder
    needs no common-mode decomposition -- that convention lives in the
    sampled likelihood); the K(z) template is shared across strata (enforced
    at the CLI).
    """
    return np.stack([_selection_cbar_fine(s, cosmo) for s in selection_strata])


def _load_stratum_map(path, nside, n_strata):
    """Load and validate a full-sky RING stratum map at the survey nside."""
    import h5py

    with h5py.File(path, "r") as f:
        if "stratum_map" not in f:
            raise ValueError(f"{path}: no 'stratum_map' dataset.")
        smap = np.asarray(f["stratum_map"], dtype=np.int64)
    n_pix = 12 * int(nside) ** 2
    if smap.shape[0] != n_pix:
        raise ValueError(
            f"{path}: stratum map covers {smap.shape[0]} pixels but the "
            f"survey nside={nside} sky has {n_pix}; regenerate the map at "
            "the survey nside (RING).")
    if smap.min() < 0 or int(smap.max()) + 1 != int(n_strata):
        raise ValueError(
            f"{path}: labels span [{int(smap.min())}, {int(smap.max())}] but "
            f"the selection fit carries {n_strata} strata; every pixel needs "
            "a label in [0, S).")
    return smap


def _stratum_map_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _selection_strata_stamp(selection_strata, stratum_map_sha):
    """Provenance stamps for a stratified selection base.

    Flat scalar keys per stratum (the float whitelist machinery carries
    them) plus the stratum-map file hash: the inference hard-errors when a
    stratified run's fit strata or map differ from what the table was built
    with -- the per-pixel base rows would silently disagree otherwise.
    """
    out = {"selection_n_strata": float(len(selection_strata)),
           "selection_stratum_map_sha256": str(stratum_map_sha)}
    for j, s in enumerate(selection_strata):
        out[f"selection_s{j}_m_lim"] = float(s["m_lim"])
        out[f"selection_s{j}_M0hat"] = float(s["M0hat"])
        out[f"selection_s{j}_sigma_M"] = float(s["sigma_M"])
    return out


def _stamp_post_renorm_railing(diagnostics, logq_fit, logq_members_fit=None):
    """Post-renorm clip-railing provenance (deferred review MINOR).

    The solvers clip ``|logQ| <= logq_clip`` BEFORE the per-z mean-one budget
    renormalization; the renorm then SHIFTS every row of a z-bin by that bin's
    log-monopole, so cells that were sitting on the clip wall move off it (or
    past it) and the shipped table's extremes are no longer the clip value.
    A table whose post-renorm cells still rail at/beyond the clip is a table
    whose solve saturated -- the clip, not the data, chose those Q values --
    and that must be auditable from the file alone.  Stamped over the FITTED
    rows only (unfitted rows are exactly logQ = 0 and would dilute the
    fraction).
    """
    clip = float(diagnostics.get("logq_clip", 7.0))
    lq = np.asarray(logq_fit, dtype=float)
    n_tot = int(lq.size)
    n_rail = int(np.sum(np.abs(lq) >= clip)) if n_tot else 0
    diagnostics["post_renorm_logq_rail_count"] = n_rail
    diagnostics["post_renorm_logq_rail_frac"] = (
        n_rail / n_tot if n_tot else 0.0)
    diagnostics["post_renorm_logq_abs_max"] = (
        float(np.abs(lq).max()) if n_tot else 0.0)
    if logq_members_fit is not None:
        lm = np.asarray(logq_members_fit, dtype=float)
        n_m = int(lm.size)
        diagnostics["post_renorm_logq_rail_frac_members"] = (
            int(np.sum(np.abs(lm) >= clip)) / n_m if n_m else 0.0)


def _selection_kcorr_stamp(selection_fit):
    """Per-coefficient provenance keys for the fit's fixed K(z) template.

    Stamped as scalar ``selection_kcorr_c{j}`` attrs (j from 1) so the float
    whitelist machinery carries them; absent keys mean K = 0 (pre-K tables
    load unchanged).  The inference CLI compares them elementwise against the
    --selection_fit template.
    """
    coeffs = tuple(selection_fit.get("k_corr_coeffs") or ())
    return {f"selection_kcorr_c{j}": float(c)
            for j, c in enumerate(coeffs, start=1)}


def _require_selection_fit(c_mode, selection_fit):
    """Fail-closed pairing of ``c_mode='selection'`` with a fit payload."""
    if c_mode == "selection" and selection_fit is None:
        raise ValueError(
            "c_mode='selection' needs the offline magnitude fit "
            "(--selection-fit selection_fit.json, written by "
            "darksirens_fit_selection): the parametric base "
            "C_sel(z; theta_hat) dN_exp has no counts-based fallback.")
    if c_mode != "selection" and selection_fit is not None:
        raise ValueError(
            f"--selection-fit given but c_mode='{c_mode}': the fit "
            "parameterizes the c_mode='selection' base only; a mismatch "
            "would stamp provenance the consumer check cannot honor.")


def _gp3d_resolution_guard(*, z_node_hi, n_z_nodes, ls_z, n_sph_nodes, ls_sph):
    """Resolution gate on the gp3d inducing grid; returns the zeta node spacing.

    A low-rank GP whose inducing-node spacing exceeds the kernel lengthscale
    cannot represent the posterior and collapses to the prior (Burt et al.
    2019, arXiv:1903.03571) — silently, with ``converged=True``: the shipped
    50 Mpc fiducial was ~30x under-resolved and measured a fitted-vs-truth
    logQ slope of 0.04 on the closure experiment.  So the radial side is a
    HARD ERROR: spacing ``log1p(z_node_hi)/(M_z - 1)`` must not exceed
    ``ls_z`` (the Mpc correlation length mapped to zeta units at the
    count-weighted z_ref).  The sphere side only WARNS — the Fibonacci
    spacing ``~ sqrt(4 pi / M_sph)`` (chordal, matching the chordal
    ``ls_sph``) is a cruder estimate and the angular fiducial is not the
    measured failure mode.
    """
    if int(n_z_nodes) < 2:
        raise ValueError(
            f"--gp3d-nz-nodes must be >= 2 (got {int(n_z_nodes)}): a single "
            "radial inducing node cannot resolve any redshift structure."
        )
    zeta_hi = float(np.log1p(float(z_node_hi)))
    spacing = zeta_hi / (int(n_z_nodes) - 1)
    if spacing > float(ls_z):
        min_nodes = int(np.ceil(zeta_hi / float(ls_z))) + 1
        raise ValueError(
            f"gp3d radial inducing grid is under-resolved: node spacing in "
            f"zeta = log1p(z) is {spacing:.4g} ({int(n_z_nodes)} nodes up to "
            f"z_node_hi = {float(z_node_hi):.4g}) but the GP lengthscale is "
            f"ls_z = {float(ls_z):.4g} (lss_corr_length_mpc mapped to zeta at "
            f"the count-weighted z_ref). A low-rank GP with node spacing > "
            f"lengthscale collapses to the prior (Burt et al. 2019) while "
            f"still reporting convergence. Pass --gp3d-nz-nodes >= "
            f"{min_nodes}, or raise --lss-corr-length-mpc / lower "
            f"--gp3d-z-node-hi until spacing <= ls_z."
        )
    d_sph = float(np.sqrt(4.0 * np.pi / max(int(n_sph_nodes), 1)))
    if d_sph > float(ls_sph):
        _warn(
            f"gp3d sphere inducing grid may be under-resolved: Fibonacci node "
            f"spacing ~ sqrt(4pi/M_sph) = {d_sph:.3g} exceeds the chordal "
            f"angular lengthscale ls_sph = {float(ls_sph):.3g}; raise "
            f"--gp3d-nsph-nodes (>= "
            f"{int(np.ceil(4.0 * np.pi / float(ls_sph) ** 2))}) or the "
            f"angular correlation length."
        )
    return spacing


def _zgrid_bin_edges() -> np.ndarray:
    """Bin edges around the (non-uniform) ``zgrid`` points (midpoints)."""
    z = np.asarray(zgrid, dtype=float)
    mids = 0.5 * (z[:-1] + z[1:])
    return np.concatenate([[z[0]], mids, [z[-1]]])


def _rebin_counts_to_uniform(counts_z, chi, chi_u):
    """Reassign per-zgrid-bin counts to the uniform-chi bin each point falls in
    (conserves the total, unlike interpolation)."""
    n = chi_u.size
    dchi_u = (chi_u[1] - chi_u[0]) if n > 1 else 1.0
    idx = np.clip(np.round((chi - chi_u[0]) / dchi_u).astype(int), 0, n - 1)
    out = np.zeros(n, dtype=float)
    np.add.at(out, idx, counts_z)
    return out


def _build_completion_radial(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    prior_strength: float = 1.0,
    maxiter: int = 200000,
    workers: int = 1,
    log10n0=None,
    delta=None,
    budget_renorm: bool = True,
    lss_corr_length_mpc=None,
    lss_sigma=None,
    c_mode: str = "per_pixel",
    selection_fit=None,
    selection_strata=None,
    stratum_map=None,
    stratum_map_sha=None,
):
    """Radial (per-pixel, independent 1-D) MAP + optional ensemble log Q tables.

    The original LSS completion builder: an independent 1-D Poisson-lognormal
    field per occupied pixel on a uniform comoving-distance grid (no angular
    coupling).  See :func:`_build_completion_gp3d` for the 3-D upgrade.

    ``budget_renorm`` (default ON) applies the per-z mean-one budget
    renormalization (:func:`renormalize_q_mean_one`) to the output tables over
    the fitted rows, so Q only redistributes the missing budget.

    ``lss_corr_length_mpc`` / ``lss_sigma`` override the fixed SurveyParams GP
    hyperparameters at build time (``None`` keeps the fiducials).

    ``c_mode`` selects the completeness base the fit is residual to:
    ``"per_pixel"`` (legacy default, bit-identical -- per-pixel matched-kernel
    C, occupied rows only) or ``"aggregate"`` (ONE sky-aggregate ``Cbar``
    matching the in-likelihood ``SurveyParams.c_mode=1`` estimator, and the
    fit INCLUDES EMPTY PIXELS as N_obs = 0 rows against the nonzero base
    ``Cbar dN_exp`` -- the void information; Q then targets the FULL observed
    overdensity, not the sub-smoothing residual).
    """
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

    cosmo, survey = _fiducial_cosmo_survey(log10n0=log10n0, delta=delta)
    survey = _apply_lss_overrides(
        survey, lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma)
    em = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, n_grid)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, em)
    dN_exp_density = np.asarray(grids.dN_exp, dtype=float)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth, dtype=float)
    safe_smooth = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    edges = _zgrid_bin_edges()
    ngals_np = np.asarray(ngals).astype(int)
    zgals_np = np.asarray(zgals, dtype=float)

    # Grid-aware P(k): solve on a UNIFORM comoving-distance grid so the Gaussian
    # correlation length is constant in Mpc (zgrid is log-spaced ⇒ Δχ varies).
    chi = np.asarray(r_of_z(jnp.asarray(zgrid), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), dtype=float)
    chi_u = np.linspace(float(chi[0]), float(chi[-1]), n_grid)
    dchi_u = float(chi_u[1] - chi_u[0]) if n_grid > 1 else 1.0
    ell_grid = float(survey.lss_corr_length_mpc) / max(dchi_u, 1e-6)  # now constant in Mpc
    pk = gaussian_correlation_spectrum(n_grid, ell_grid, float(survey.lss_sigma))

    # Expected counts per uniform-chi bin: INTEGRATE the per-unit-z density in z
    # between the bin-edge redshifts (cumulative trapezoid on the fine grid,
    # differenced at the chi-bin edges; chi(z) is monotone so the edges map
    # 1:1). The previous point-evaluation `interp(chi_u, chi, density) * dchi_u`
    # multiplied a per-unit-z density by a bin width in Mpc — missing the
    # dz/dchi Jacobian (~ c/H(z) ~ 4e3) and inflating the homogeneous
    # expectation ~1e3x, which crushed the field and railed every radial table
    # at the -7 clip (library review P0.1). Bin edges are the chi-node
    # midpoints, matching the round-to-nearest convention of
    # `_rebin_counts_to_uniform` used for N_obs.
    zg = np.asarray(zgrid, dtype=float)
    dz_fine = np.diff(zg)
    edges_chi = np.concatenate([[chi_u[0]], 0.5 * (chi_u[:-1] + chi_u[1:]), [chi_u[-1]]])

    def _bin_integral(density_fine):
        cum = np.concatenate([
            [0.0], np.cumsum(0.5 * (density_fine[1:] + density_fine[:-1]) * dz_fine)
        ])
        return np.diff(np.interp(edges_chi, chi, cum))

    dN_exp_count_u = _bin_integral(dN_exp_density)               # (n_grid,)
    exp_safe = np.where(dN_exp_count_u > 0.0, dN_exp_count_u, 1.0)

    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)
    if c_mode in ("aggregate", "selection"):
        # SKY-UNIFORM base: ONE completeness curve for every pixel, and the
        # fit covers EVERY pixel: empty pixels enter as N_obs = 0 rows against
        # the nonzero base C * dN_exp -- that zero IS the void information the
        # occupied-only fit never sees (deep voids read Q < 1 instead of the
        # homogeneous Q = 1).  The occupied-only compact indexing of the
        # legacy branch is untouched; the pixel list is extended explicitly.
        #   aggregate:  Cbar(z) = clip(Sum_p dN_obs_s(z|p)
        #                              / (N_pix_total * dN_exp_smooth), 0, 1),
        #               N_pix_total = round(4 pi / apix)  (occupied AND empty),
        #               mirroring the in-likelihood SurveyParams.c_mode=1
        #               estimator;
        #   selection:  the PARAMETRIC C_sel(z; theta_hat) from the offline
        #               magnitude fit -- no counts enter the budget at all
        #               (mirrors SurveyParams.c_mode=2).
        fit = np.arange(n_pix)
        n_fit = int(fit.size)
        n_pix_total = int(np.round(4.0 * np.pi / apix))
        if c_mode == "selection" and selection_strata is not None:
            # STRATIFIED base: one C_sel curve per stratum, routed per pixel
            # by the stratum map -- mirrors the in-likelihood
            # SurveyParams.selection_strata consumption, so the table's fixed
            # base carries the same per-pixel budget the numerator does.
            Cfine_s = _selection_cbar_fine_strata(selection_strata, cosmo)
            Cu_s = np.stack([
                np.clip(_bin_integral(cf * dN_exp_density) / exp_safe, 0.0, 1.0)
                for cf in Cfine_s])
            C_u = Cu_s[stratum_map[fit]]
            w_budget = ((1.0 - Cfine_s) * dN_exp_density)[stratum_map[fit]]
        elif c_mode == "selection":
            Cbar_fine = _selection_cbar_fine(selection_fit, cosmo)
        else:
            dN_obs_sum = np.zeros(n_grid, dtype=float)
            for r in occ:
                dN_obs_sum += np.asarray(
                    _kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
            Cbar_fine = np.clip(dN_obs_sum / (n_pix_total * safe_smooth), 0.0, 1.0)
        if not (c_mode == "selection" and selection_strata is not None):
            # Same expectation-weighted bin average as the per-pixel branch
            # (C_bin = int(C * dN_exp) / int(dN_exp), the existing Jacobian
            # treatment), so rate_base = Cbar_u * dN_exp_count_u is the
            # bin-integrated expected OBSERVED counts, same footing as N_obs.
            Cbar_u = np.clip(
                _bin_integral(Cbar_fine * dN_exp_density) / exp_safe, 0.0, 1.0)
            C_u = np.tile(Cbar_u, (n_fit, 1))
            # One shared budget-weight row (1 - Cbar) dN_exp for EVERY pixel:
            # the renormalization footprint is the whole fitted sky.
            w_budget = np.tile((1.0 - Cbar_fine) * dN_exp_density, (n_fit, 1))
        N_obs_u = np.zeros((n_fit, n_grid), dtype=float)
        for i, r in enumerate(fit):
            if ngals_np[r] > 0:
                zs = zgals_np[r, : ngals_np[r]]
                counts_z, _ = np.histogram(zs, bins=edges)
                N_obs_u[i] = _rebin_counts_to_uniform(counts_z, chi, chi_u)
    else:
        # Build only OCCUPIED pixels (DESI footprints are mostly empty ⇒ huge
        # speedup); empty pixels get logQ = 0 (Q = 1, homogeneous) by the
        # zero-init below.
        fit = occ
        n_fit = n_occ
        C_u = np.empty((n_occ, n_grid), dtype=float)
        N_obs_u = np.zeros((n_occ, n_grid), dtype=float)
        w_budget = np.empty((n_occ, n_grid), dtype=float)
        for i, r in enumerate(occ):
            dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
            # Expectation-weighted bin completeness: C_bin = int(C * dN_exp) / int(dN_exp),
            # so the solver's rate_base = C_u * dN_exp_count_u is exactly the
            # bin-integrated expected OBSERVED counts — the same footing as N_obs.
            C_fine = np.clip(dN_obs_s / safe_smooth, 0.0, 1.0)
            prod = C_fine * dN_exp_density
            C_u[i] = np.clip(_bin_integral(prod) / exp_safe, 0.0, 1.0)
            # This pixel's missing-budget weight on the OUTPUT zgrid at the build
            # fiducial: w_p(z) = (1 - C_p(z)) * dN_exp(z) — the same
            # (1 - C) dN_exp the likelihood multiplies Q into (dN_miss).
            w_budget[i] = (1.0 - C_fine) * dN_exp_density
            zs = zgals_np[r, : ngals_np[r]]
            counts_z, _ = np.histogram(zs, bins=edges)
            N_obs_u[i] = _rebin_counts_to_uniform(counts_z, chi, chi_u)

    bias = float(survey.b_miss)
    mp = poisson_lognormal_map(
        N_obs_u, C_u, dN_exp_count_u, pk,
        bias=bias, prior_strength=prior_strength, maxiter=maxiter,
        workers=workers,
    )
    # Map logQ back from uniform-χ to zgrid and scatter the fitted rows into
    # the full (n_pix, n_grid) table (unfitted rows stay logQ = 0; in
    # aggregate mode every row is fitted).
    logq_map = np.zeros((n_pix, n_grid), dtype=float)
    for i, r in enumerate(fit):
        logq_map[r] = np.interp(chi, chi_u, mp["logq_map"][i])

    diagnostics = dict(mp["diagnostics"])
    diagnostics.update({
        # "n_occupied" is the FITTED-row count (what n_converged is judged
        # against): the occupied pixels in per_pixel mode, every pixel in
        # aggregate mode (empty ones fit as N_obs = 0 rows).
        "nside": int(nside), "n_pix": n_pix, "n_occupied": n_fit,
        "c_mode": c_mode,
        "ell_grid_uniform_chi": ell_grid, "dchi_uniform_mpc": dchi_u,
        "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
        "lss_sigma": float(survey.lss_sigma),
        # Fixed fiducials Q was built at (inference varies these — see the warning
        # printed at load): cosmology, density/evolution, and the field bias.
        "fiducial_H0": float(cosmo.H0), "fiducial_Om0": float(cosmo.Om0),
        "fiducial_w0": float(cosmo.w0), "fiducial_wa": float(cosmo.wa),
        "fiducial_n0": float(survey.n0), "fiducial_delta": float(survey.delta),
        "bias_b_miss": float(survey.b_miss),
    })
    if c_mode in ("aggregate", "selection"):
        diagnostics.update({"n_pix_total": n_pix_total, "n_occupied_data": n_occ})
    if c_mode == "selection":
        # theta_hat provenance: the consumer hard-errors when the table's
        # build theta does not match the --selection_fit prior center fed to
        # the inference (a stale table would carry the WRONG fixed base).
        diagnostics.update({
            **_selection_theta_stamp(selection_fit),
            **_selection_kcorr_stamp(selection_fit),
            **(_selection_strata_stamp(selection_strata, stratum_map_sha)
               if selection_strata is not None else {}),
        })

    logq_members = None
    if n_members and n_members > 0:
        members = laplace_lognormal_members(
            mp["s_map"], mp["lambda_map"], pk,
            n_members=int(n_members), bias=bias, prior_strength=prior_strength, seed=int(seed),
        )
        lm_u = members["logq_members"]                       # (M, n_fit, n_grid) on χ_u
        M = int(n_members)
        logq_members = np.zeros((M, n_pix, n_grid), dtype=float)
        for i, r in enumerate(fit):
            for m in range(M):
                logq_members[m, r] = np.interp(chi, chi_u, lm_u[m, i])
        diagnostics.update(members["diagnostics"])

    # Per-z mean-one budget renormalization over the FITTED FOOTPRINT (the
    # occupied rows in per_pixel mode, the whole sky in aggregate mode): the
    # Laplace E[Q] carries a per-z monopole (var_post is largest where data
    # are sparse -> spatially varying Jensen bias, measured +55% budget
    # inflation), which would RESCALE the missing budget the likelihood forms
    # as (1 - C) dN_exp Q.  After this, the total budget with Q equals the
    # homogeneous budget identically at the build fiducial; each member is
    # renormalized independently (placement uncertainty only, zero budget
    # uncertainty).  Unfitted rows stay logQ = 0 (Q = 1) exactly.
    if budget_renorm:
        logq_map[fit], log_mono = renormalize_q_mean_one(logq_map[fit], w_budget)
        if logq_members is not None:
            logq_members[:, fit], _ = renormalize_q_mean_one(
                logq_members[:, fit], w_budget)
        diagnostics["budget_monopole_logq"] = log_mono
        _stamp_post_renorm_railing(
            diagnostics, logq_map[fit],
            logq_members[:, fit] if logq_members is not None else None)
    diagnostics["budget_renormalized"] = bool(budget_renorm)

    return logq_map, logq_members, diagnostics


class _GP3DSurveyAssembly(NamedTuple):
    """Per-survey gp3d voxel inputs on the SHARED coarse solve grid ``z_s``.

    Everything :func:`_build_completion_gp3d` (K=1) and
    :func:`darksirens.cli.build_joint_lognormal_completion.build_joint_completion`
    (K>=1) need to stack a survey into the low-rank Poisson-lognormal field: the
    occupied-voxel geometry (``n_hat_occ``), the all-pixel output directions
    (``n_hat_all``), the coarse-bin observed counts (``N_obs_vox``) and
    bin-integrated expected observed base (``base_vox``), and the per-survey
    coarse-z count profile (``counts_z``) that the joint ``z_ref`` weighting sums
    over all surveys.  ``w_budget`` is the fine-zgrid missing-budget weight
    ``(1 - C_p) dN_exp`` of the fitted rows, consumed by the per-z mean-one
    Q renormalization of the OUTPUT table (not by the solve).

    ``occ`` / ``n_occ`` / ``n_hat_occ`` are the FITTED rows: the occupied RING
    pixels for the legacy per-pixel base, EVERY pixel (empties as N_obs = 0
    rows) for the aggregate base (``c_mode="aggregate"``).
    """
    nside: int
    n_pix: int
    apix: float
    occ: np.ndarray          # (n_occ,) fitted RING pixel indices
    n_occ: int
    n_hat_all: np.ndarray    # (n_pix, 3) all-pixel output directions
    n_hat_occ: np.ndarray    # (n_occ, 3) occupied-voxel directions
    N_obs_vox: np.ndarray    # (n_occ, G_s) coarse-bin histogram counts
    base_vox: np.ndarray     # (n_occ, G_s) bin-integrated expected observed counts
    counts_z: np.ndarray     # (G_s,) coarse-z count profile (for z_ref weighting)
    w_budget: np.ndarray     # (n_occ, n_grid) fine-zgrid (1 - C) dN_exp weights


def _gp3d_base_diagnostics(cosmo, survey, *, nside, n_pix, n_occ, amp, ls_sph,
                           bias, mode="gp3d"):
    """The gp3d diagnostics block every build (single or joint) records.

    ``mode`` is ``"gp3d"`` for the single-survey builder and ``"gp3d_joint"``
    for the joint multi-survey builder; the fiducial keys are what the inference
    loader prints at Q load time (build-time cosmology/density/bias).
    """
    return {
        "mode": mode, "nside": int(nside), "n_pix": int(n_pix),
        "n_occupied": int(n_occ),
        "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
        "lss_sigma": amp, "lss_corr_length_ang": ls_sph,
        # Fixed fiducials Q was built at (inference varies these; see load warning).
        "fiducial_H0": float(cosmo.H0), "fiducial_Om0": float(cosmo.Om0),
        "fiducial_w0": float(cosmo.w0), "fiducial_wa": float(cosmo.wa),
        "fiducial_n0": float(survey.n0), "fiducial_delta": float(survey.delta),
        "bias_b_miss": bias,
    }


def _assemble_gp3d_survey(catalog_path, *, cosmo, survey, z_s, edges_s,
                          c_mode: str = "per_pixel", selection_fit=None,
                          selection_strata=None, stratum_map=None):
    """Load a survey and assemble its gp3d voxel inputs on the shared ``z_s`` grid.

    Extracted verbatim from the single-survey ``_build_completion_gp3d`` body so
    the K=1 driver and the JOINT multi-survey builder share ONE assembly path
    (bit-identical at K=1 — same operation order).  ``z_s``/``edges_s`` are the
    SHARED coarse solve grid; they depend only on the package ``zgrid`` and
    ``gp3d_nz_solve`` (never on the survey), so every survey in a joint fit lands
    on the same radial bins.  ``survey`` carries this survey's own fiducial n0 /
    delta (its expected-density normalisation); the field bias ``b_miss`` does
    NOT enter the assembly (``_precompute_grids`` depends only on n0/delta/cosmo),
    so a per-survey bias is applied later by scaling the design matrix.

    ``c_mode="aggregate"`` swaps the per-pixel completeness base for the ONE
    sky-aggregate ``Cbar dN_exp`` and extends the voxel list to EVERY pixel
    (empty pixels as N_obs = 0 rows -- the void information); the returned
    ``occ``/``n_occ``/``n_hat_occ`` then describe the FITTED rows (the whole
    sky).  The default keeps the legacy occupied-only assembly bit-identical
    (the joint multi-survey builder never passes it).
    """
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

    em = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, n_grid)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, em)
    dN_exp_density = np.asarray(grids.dN_exp, dtype=float)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth, dtype=float)
    safe_smooth = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    zgrid_np = np.asarray(zgrid, dtype=float)
    ngals_np = np.asarray(ngals).astype(int)
    zgals_np = np.asarray(zgals, dtype=float)

    # All-pixel output directions (RING ordering — matches the catalog pixelation
    # and the inference event->pixel mapping).
    n_hat_all = np.asarray(
        hp.pix2vec(int(nside), np.arange(n_pix), nest=False), dtype=float).T  # (n_pix, 3)

    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)

    # Expected OBSERVED counts per coarse z-bin: integrate C(z) * dN_exp(z) over
    # each bin on the fine grid, on the same footing as the histogram that fills
    # N_obs_vox over the same bin. Point-evaluating C at the coarse node instead
    # is catastrophic wherever the KDE support and the bin contents disagree
    # (e.g. the z=0 node of a low-z-complete catalog: C(0) ~ KDE tail ~ 0 while
    # the bin holds real galaxies, so the fit demands field values beyond
    # field_clip and the solve saturates).
    G_s = int(np.asarray(z_s).size)
    dz_fine = np.diff(zgrid_np)

    def _coarse_bin_integral(density_fine):
        cum = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (density_fine[1:] + density_fine[:-1]) * dz_fine),
        ])
        return np.diff(np.interp(edges_s, zgrid_np, cum))

    if c_mode in ("aggregate", "selection"):
        # SKY-UNIFORM base (mirrors the radial builder and the in-likelihood
        # SurveyParams.c_mode=1/2 estimators): one shared completeness curve
        # -- the counts aggregate Cbar or the parametric C_sel(z; theta_hat)
        # -- and the voxel list extends to EVERY pixel so empty pixels enter
        # as N_obs = 0 rows against the nonzero base: the void information.
        fit = np.arange(n_pix)
        n_fit = int(fit.size)
        n_pix_total = int(np.round(4.0 * np.pi / apix))
        if c_mode == "selection" and selection_strata is not None:
            # STRATIFIED base: per-stratum C_sel curves routed per pixel by
            # the stratum map (mirrors the radial branch and the in-likelihood
            # SurveyParams.selection_strata consumption).
            Cfine_s = _selection_cbar_fine_strata(selection_strata, cosmo)
            base_s = np.stack([
                _coarse_bin_integral(cf * dN_exp_density) for cf in Cfine_s])
            base_vox = base_s[stratum_map[fit]]
            w_budget = ((1.0 - Cfine_s) * dN_exp_density)[stratum_map[fit]]
        else:
            if c_mode == "selection":
                Cbar_fine = _selection_cbar_fine(selection_fit, cosmo)
            else:
                dN_obs_sum = np.zeros(n_grid, dtype=float)
                for r in occ:
                    dN_obs_sum += np.asarray(
                        _kde_dndz_obs(int(r), em.zgals, ngals=em.ngals),
                        dtype=float)
                Cbar_fine = np.clip(
                    dN_obs_sum / (n_pix_total * safe_smooth), 0.0, 1.0)
            base_row = _coarse_bin_integral(Cbar_fine * dN_exp_density)
            base_vox = np.tile(base_row, (n_fit, 1))
            w_budget = np.tile((1.0 - Cbar_fine) * dN_exp_density, (n_fit, 1))
        N_obs_vox = np.zeros((n_fit, G_s), dtype=float)
        for i, r in enumerate(fit):
            if ngals_np[r] > 0:
                zs = zgals_np[r, : ngals_np[r]]
                counts_s, _ = np.histogram(zs, bins=edges_s)
                N_obs_vox[i] = counts_s
        n_hat_fit = n_hat_all
    else:
        fit = occ
        n_fit = n_occ
        n_hat_fit = np.asarray(
            hp.pix2vec(int(nside), occ, nest=False), dtype=float).T  # (n_occ,3)
        N_obs_vox = np.zeros((n_occ, G_s), dtype=float)
        base_vox = np.empty((n_occ, G_s), dtype=float)
        w_budget = np.empty((n_occ, n_grid), dtype=float)
        for i, r in enumerate(occ):
            dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
            C_fine = np.clip(dN_obs_s / safe_smooth, 0.0, 1.0)
            base_vox[i] = _coarse_bin_integral(C_fine * dN_exp_density)
            # Fine-zgrid missing-budget weight (1 - C) dN_exp of this row, for the
            # per-z mean-one renormalization of the OUTPUT table (same footing as
            # the likelihood's dN_miss = (1 - C) dN_exp Q).
            w_budget[i] = (1.0 - C_fine) * dN_exp_density
            zs = zgals_np[r, : ngals_np[r]]
            counts_s, _ = np.histogram(zs, bins=edges_s)
            N_obs_vox[i] = counts_s

    counts_z = N_obs_vox.sum(axis=0)
    return _GP3DSurveyAssembly(
        nside=int(nside), n_pix=n_pix, apix=apix, occ=fit, n_occ=n_fit,
        n_hat_all=n_hat_all, n_hat_occ=n_hat_fit,
        N_obs_vox=N_obs_vox, base_vox=base_vox, counts_z=counts_z,
        w_budget=w_budget,
    )


def _count_weighted_zref_lsz(counts_per_survey, z_s, cosmo, corr_length_mpc):
    """Count-weighted reference redshift over one-or-many surveys, and the
    ``zeta = log1p(z)``-units GP length ``ls_z`` at that redshift.

    Map the fixed radial Mpc length into the kernel's zeta units at a
    count-weighted reference redshift ``z_ref = sum(z_s * N) / sum(N)`` summed
    over ALL surveys' coarse-z count profiles: ``ls_z = L_mpc * dzeta/dchi
    |_{z_ref}``.  K=1 reduces to the single-survey arithmetic EXACTLY (one
    summand added to a zero accumulator is a no-op in float; same formula, same
    operation order).
    """
    z_s = np.asarray(z_s, dtype=float)
    counts_z = np.zeros(int(z_s.size), dtype=float)
    for c in counts_per_survey:
        counts_z = counts_z + np.asarray(c, dtype=float)
    z_ref = (float(np.sum(z_s * counts_z) / np.sum(counts_z))
             if np.sum(counts_z) > 0 else 1.0)
    eps = 1e-3
    zlo, zhi = max(z_ref - eps, 0.0), z_ref + eps
    dchi_dz = float(
        (r_of_z(jnp.asarray(zhi), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
         - r_of_z(jnp.asarray(zlo), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))
        / (zhi - zlo)
    )
    ls_z = float(corr_length_mpc) / max((1.0 + z_ref) * dchi_dz, 1e-6)
    return z_ref, ls_z


def _build_completion_gp3d(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
    log10n0=None,
    delta=None,
    budget_renorm: bool = True,
    lss_corr_length_mpc=None,
    lss_sigma=None,
    gp3d_nz_nodes: int = 6,
    gp3d_nsph_nodes: int = 32,
    gp3d_z_node_hi=None,
    c_mode: str = "per_pixel",
    selection_fit=None,
    selection_strata=None,
    stratum_map=None,
    stratum_map_sha=None,
):
    """3-D angular-coupling MAP + optional ensemble log Q tables.

    Solves ONE low-rank Poisson-lognormal field over the occupied (pixel x z)
    voxels using the (sphere x z) GP, so empty pixels borrow angularly from their
    neighbours and pixels far from any data read as exactly Q = 1.  Output is the
    SAME global ``(n_pix, n_grid)`` log Q table contract as the radial builder, so
    the inference side is unchanged.

    The per-survey voxel assembly (:func:`_assemble_gp3d_survey`) and the
    ``z_ref``/``ls_z`` map (:func:`_count_weighted_zref_lsz`) are the SAME helpers
    the joint multi-survey builder stacks over, so this K=1 path is one instance
    of the joint fit.

    ``lss_corr_length_mpc`` / ``lss_sigma`` override the fixed SurveyParams GP
    hyperparameters; ``gp3d_nz_nodes`` / ``gp3d_nsph_nodes`` size the inducing
    grid (defaults keep the historical 6 x 32).  ``gp3d_z_node_hi`` sets the
    top redshift of the radial inducing nodes; the default ``None`` resolves to
    the PACKAGE ``zgrid[-1]`` — a BEHAVIOR CHANGE from the historical hardwired
    3.0, which left ``zgrid[-1] > 3`` covered only by prior extrapolation.  The
    inducing grid must resolve ``ls_z`` (:func:`_gp3d_resolution_guard`, hard
    error) or the field silently collapses to the prior.

    ``c_mode="aggregate"`` fits against the sky-aggregate base ``Cbar dN_exp``
    over EVERY pixel (see :func:`_assemble_gp3d_survey`); the default keeps
    the legacy occupied-only per-pixel base bit-identical.
    """
    M_SPH, M_Z = int(gp3d_nsph_nodes), int(gp3d_nz_nodes)

    cosmo, survey = _fiducial_cosmo_survey(log10n0=log10n0, delta=delta)
    if lss_corr_length_ang is not None:
        survey = survey._replace(lss_corr_length_ang=float(lss_corr_length_ang))
    survey = _apply_lss_overrides(
        survey, lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma)

    bias = float(survey.b_miss)
    amp = float(survey.lss_sigma)
    ls_sph = float(survey.lss_corr_length_ang)
    zgrid_np = np.asarray(zgrid, dtype=float)
    n_grid = int(zgrid.size)
    # Radial inducing-node range: default is the package zgrid max so the nodes
    # span the full output grid (see the docstring behavior-change note).
    z_node_hi = (float(zgrid_np[-1]) if gp3d_z_node_hi is None
                 else float(gp3d_z_node_hi))

    # Coarse SOLVE z-grid: the field is low-rank in z (only M_z nodes), so a fine
    # solve grid adds no resolvable radial signal.  Output is on the full zgrid.
    # Survey-independent (depends only on zgrid and gp3d_nz_solve), so the joint
    # builder shares it across surveys.
    z_s = np.linspace(0.0, float(zgrid_np[-1]), int(gp3d_nz_solve))
    edges_s = np.concatenate([[z_s[0]], 0.5 * (z_s[:-1] + z_s[1:]), [z_s[-1]]])
    zeta_s = np.log1p(z_s)
    G_s = int(z_s.size)

    survey_data = _assemble_gp3d_survey(
        catalog_path, cosmo=cosmo, survey=survey, z_s=z_s, edges_s=edges_s,
        c_mode=c_mode, selection_fit=selection_fit,
        selection_strata=selection_strata, stratum_map=stratum_map)
    nside, n_pix, n_occ = survey_data.nside, survey_data.n_pix, survey_data.n_occ

    def _base_diag(extra):
        d = _gp3d_base_diagnostics(
            cosmo, survey, nside=nside, n_pix=n_pix, n_occ=n_occ,
            amp=amp, ls_sph=ls_sph, bias=bias)
        d["c_mode"] = c_mode
        if c_mode == "selection":
            d.update({
                **_selection_theta_stamp(selection_fit),
                **_selection_kcorr_stamp(selection_fit),
                **(_selection_strata_stamp(selection_strata, stratum_map_sha)
                   if selection_strata is not None else {}),
            })
        d.update(extra)
        return d

    # Empty catalog: nothing to solve, Q == 1 everywhere.  The homogeneous
    # table is trivially mean-one, so the budget stamp is honest as-is.
    if n_occ == 0:
        _warn("no occupied pixels — writing Q = 1 (logQ = 0) everywhere.")
        logq_map = np.zeros((n_pix, n_grid), dtype=float)
        logq_members = (np.zeros((int(n_members), n_pix, n_grid), dtype=float)
                        if n_members and n_members > 0 else None)
        extra = {"budget_renormalized": bool(budget_renorm)}
        if budget_renorm:
            extra["budget_monopole_logq"] = np.zeros(n_grid, dtype=float)
        return logq_map, logq_members, _base_diag(extra)

    z_ref, ls_z = _count_weighted_zref_lsz(
        [survey_data.counts_z], z_s, cosmo, survey.lss_corr_length_mpc)

    # HARD gate before any solve: an under-resolved inducing grid produces a
    # prior-collapsed field that still stamps converged=True downstream.
    zeta_spacing = _gp3d_resolution_guard(
        z_node_hi=z_node_hi, n_z_nodes=M_Z, ls_z=ls_z,
        n_sph_nodes=M_SPH, ls_sph=ls_sph)

    # Low-rank operator over the occupied voxels.
    Zn, Zz = lowrank_inducing_nodes(
        n_inducing_sphere=M_SPH, n_inducing_z=M_Z, z_node_hi=z_node_hi)
    M = int(np.asarray(Zn).shape[0])
    X_n = np.repeat(survey_data.n_hat_occ, G_s, axis=0)                    # (n_occ*G_s, 3)
    X_z = np.tile(zeta_s, n_occ)                                           # (n_occ*G_s,)
    Phi, L = build_lowrank_operator(Zn, Zz, X_n, X_z, amp=amp, ls_sph=ls_sph, ls_z=ls_z)

    mp = poisson_lognormal_gp3d_map(
        survey_data.N_obs_vox.ravel(), survey_data.base_vox.ravel(), Phi, bias=bias)

    # Global all-pixel output table on the full package zgrid (chunked over
    # pixels): the Laplace posterior-mean E[Q] (empty pixels -> Q=1, near-data
    # pixels borrow), so H_chol is passed.
    logq_map = np.asarray(eval_logq_gp3d(
        mp["xi_map"], Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
        n_hat_out=survey_data.n_hat_all, z_out=zgrid_np, bias=bias,
        pix_chunk=int(gp3d_pix_chunk), L=L, H_chol=mp["H_chol"],
    ), dtype=float)
    # Non-finite cells are COUNTED here and gated in main(): substituting
    # Q = 1 silently is indistinguishable from real homogeneity downstream,
    # so it only happens under the explicit, stamped --allow-unconverged.
    n_nonfinite_map = int(np.sum(~np.isfinite(logq_map)))

    sig2 = np.asarray(mp["sigma2_vox"], dtype=float)
    diagnostics = _base_diag({
        "M_sph": M_SPH, "M_z": M_Z, "M": M, "gp3d_nz_solve": G_s,
        "gp3d_z_node_hi": z_node_hi, "zeta_node_spacing": zeta_spacing,
        "ls_z_zeta": ls_z, "z_ref": z_ref, "ls_sph_chordal": ls_sph,
        "sigma2_vox_min": float(sig2.min()), "sigma2_vox_max": float(sig2.max()),
        "n_iter": int(mp["diagnostics"]["n_iter"]),
        "converged": bool(mp["diagnostics"]["converged"]),
        "grad_inf": float(mp["diagnostics"]["grad_inf"]),
        "n_nonfinite_map": n_nonfinite_map,
    })

    logq_members = None
    n_nonfinite_members = 0
    if n_members and n_members > 0:
        xi_mem = laplace_lognormal_gp3d_members(
            mp["xi_map"], mp["H_chol"], n_members=int(n_members), seed=int(seed))
        logq_members = np.asarray(eval_logq_gp3d(
            xi_mem, Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
            n_hat_out=survey_data.n_hat_all, z_out=zgrid_np, bias=bias,
            pix_chunk=int(gp3d_pix_chunk), L=L,
        ), dtype=float)
        n_nonfinite_members = int(np.sum(~np.isfinite(logq_members)))
        diagnostics["n_nonfinite_members"] = n_nonfinite_members
        diagnostics["n_members"] = int(n_members)

    # Per-z mean-one budget renormalization over the FITTED FOOTPRINT (the
    # occupied rows; in aggregate mode every pixel is fitted, so the footprint
    # is the whole sky), same convention as the radial builder.  Unfitted
    # pixels keep their posterior-mean value: far pixels read exactly Q = 1
    # (homogeneous) and must not absorb the footprint's monopole; the
    # borrowing halo just outside the footprint keeps its (small) angular
    # tail unrenormalized.  Skipped when the eval produced non-finite cells
    # (only reachable as a stamped --allow-unconverged research ablation): a
    # NaN entering the per-z weighted mean would smear across every occupied
    # row of that bin.
    do_renorm = bool(budget_renorm) and (n_nonfinite_map + n_nonfinite_members) == 0
    if do_renorm:
        occ = survey_data.occ
        logq_map[occ], log_mono = renormalize_q_mean_one(
            logq_map[occ], survey_data.w_budget)
        if logq_members is not None:
            logq_members[:, occ], _ = renormalize_q_mean_one(
                logq_members[:, occ], survey_data.w_budget)
        diagnostics["budget_monopole_logq"] = log_mono
        _stamp_post_renorm_railing(
            diagnostics, logq_map[occ],
            logq_members[:, occ] if logq_members is not None else None)
    diagnostics["budget_renormalized"] = bool(do_renorm)

    return logq_map, logq_members, diagnostics


def build_completion(
    catalog_path: str,
    *,
    mode: str = "radial",
    n_members: int = 32,
    seed: int = 1234,
    prior_strength: float = 1.0,
    maxiter: int = 200000,
    workers: int = 1,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
    log10n0=None,
    delta=None,
    budget_renorm: bool = True,
    lss_corr_length_mpc=None,
    lss_sigma=None,
    gp3d_nz_nodes: int = 6,
    gp3d_nsph_nodes: int = 32,
    gp3d_z_node_hi=None,
    c_mode: str = "per_pixel",
    selection_fit=None,
    selection_strata=None,
    stratum_map=None,
    stratum_map_sha=None,
):
    """Build the log Q completion tables from a survey catalog.

    ``mode="radial"`` (default) -> independent per-pixel 1-D Poisson-lognormal
    (:func:`_build_completion_radial`).  ``mode="gp3d"`` -> the 3-D
    angular-coupling low-rank field (:func:`_build_completion_gp3d`).  Both return
    ``(logq_map, logq_members, diagnostics)`` with the SAME global table contract,
    and both apply the per-z mean-one budget renormalization by default
    (``budget_renorm``; the removed monopole is carried in the diagnostics).

    ``lss_corr_length_mpc`` / ``lss_sigma`` override the fixed GP
    hyperparameters in BOTH modes; the ``gp3d_*_nodes`` / ``gp3d_z_node_hi``
    inducing-grid knobs apply to ``mode="gp3d"`` only, which hard-errors when
    the grid cannot resolve ``ls_z`` (:func:`_gp3d_resolution_guard`).

    ``c_mode`` selects the completeness base of the fit in BOTH modes:
    ``"per_pixel"`` (legacy default, bit-identical) or ``"aggregate"`` (the
    sky-aggregate ``Cbar`` base with empty pixels fit as N_obs = 0 rows).  The
    table must be consumed under the SAME ``SurveyParams.c_mode`` -- the two
    targets differ by the entire clustering signal -- so the mode is stamped
    in the HDF5 attrs and hard-checked at load.
    """
    if c_mode not in ("per_pixel", "aggregate", "selection"):
        raise ValueError(
            f"c_mode must be 'per_pixel', 'aggregate' or 'selection', "
            f"got {c_mode!r}.")
    _require_selection_fit(c_mode, selection_fit)
    if mode == "radial":
        return _build_completion_radial(
            catalog_path, n_members=n_members, seed=seed,
            prior_strength=prior_strength, maxiter=maxiter, workers=workers,
            log10n0=log10n0, delta=delta, budget_renorm=budget_renorm,
            lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma,
            c_mode=c_mode, selection_fit=selection_fit,
            selection_strata=selection_strata, stratum_map=stratum_map,
            stratum_map_sha=stratum_map_sha,
        )
    if mode == "gp3d":
        return _build_completion_gp3d(
            catalog_path, n_members=n_members, seed=seed,
            gp3d_nz_solve=gp3d_nz_solve, gp3d_pix_chunk=gp3d_pix_chunk,
            lss_corr_length_ang=lss_corr_length_ang,
            log10n0=log10n0, delta=delta, budget_renorm=budget_renorm,
            lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma,
            gp3d_nz_nodes=gp3d_nz_nodes, gp3d_nsph_nodes=gp3d_nsph_nodes,
            gp3d_z_node_hi=gp3d_z_node_hi,
            c_mode=c_mode, selection_fit=selection_fit,
            selection_strata=selection_strata, stratum_map=stratum_map,
            stratum_map_sha=stratum_map_sha,
        )
    raise ValueError(f"Unknown build mode {mode!r}; expected 'radial' or 'gp3d'.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build an LSS-conditioned lognormal completion file (offline)."
    )
    p.add_argument("--catalog", required=True, help="Pixelated survey catalog HDF5 (load_survey schema).")
    p.add_argument("--out", required=True, help="Output completion HDF5 path.")
    p.add_argument("--n-members", type=int, default=32,
                   help="Number of Laplace/FFT-diagonal ensemble members (0 = MAP only).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prior-strength", type=float, default=1.0)
    # Each occupied pixel is an n_grid-dimensional (1000 on the package zgrid)
    # L-BFGS-B MAP solve; 300 iterations stops ~5x short of the fixed point on a
    # DESI-like nside=16 catalog (measured 0/2580 converged, |grad|_inf ~ 21,
    # max|dlogQ| 1.36 vs the converged field) and 10000 still capped near-empty
    # pixels ~40% short (logQ 0.556 vs 0.924 at the fixed point).  L-BFGS-B
    # self-terminates once converged, so the high cap only costs on solves that
    # genuinely need it — and an unconverged build now FAILS instead of saving.
    p.add_argument("--maxiter", type=int, default=200000,
                   help=("Max L-BFGS-B iterations per pixel MAP solve (default "
                         "200000; converged solves self-terminate long before "
                         "this). If the cap binds, the build fails rather than "
                         "silently under-relaxing logQ toward 0 (Q -> 1)."))
    p.add_argument("--workers", type=int, default=1,
                   help=("Parallel processes for the per-pixel MAP solves "
                         "(1 = serial). The solves are independent, so this is "
                         "exact and only changes wall time; at nside=128 (~69k "
                         "occupied pixels) serial is ~42 h and 16-way is ~3 h. "
                         "--mode radial ONLY; ignored by --mode gp3d, which "
                         "solves one coupled field and chunks via "
                         "--gp3d-pix-chunk instead."))
    p.add_argument("--mode", choices=["radial", "gp3d"], default="radial",
                   help="Completion model: 'radial' (default; independent per-pixel "
                        "1-D Poisson-lognormal) or 'gp3d' (3-D angular-coupling "
                        "low-rank field — empty pixels borrow from neighbours).")
    p.add_argument("--gp3d-nz-solve", type=int, default=32,
                   help="(gp3d) Number of coarse redshift points for the solve grid.")
    p.add_argument("--gp3d-pix-chunk", type=int, default=512,
                   help="(gp3d) Pixel chunk size for the all-pixel field evaluation.")
    p.add_argument("--lss-corr-length-ang", type=float, default=None,
                   help="(gp3d) Override the fixed angular (chordal) correlation "
                        "length; default is the SurveyParams fiducial.")
    p.add_argument("--lss-corr-length-mpc", type=float, default=None,
                   help="Override the fixed radial GP correlation length [Mpc] "
                        "(both modes; default is the SurveyParams fiducial, "
                        "50). Build-time only, never sampled. In gp3d mode the "
                        "inducing grid must resolve the mapped ls_z or the "
                        "build hard-errors.")
    p.add_argument("--lss-sigma", type=float, default=None,
                   help="Override the fixed GP field amplitude (both modes; "
                        "default is the SurveyParams fiducial, 1.0). "
                        "Build-time only, never sampled.")
    p.add_argument("--gp3d-nz-nodes", type=int, default=6,
                   help="(gp3d) Number of radial inducing nodes M_z (default 6, "
                        "the historical value). Node spacing in zeta = log1p(z) "
                        "must not exceed ls_z (hard error otherwise).")
    p.add_argument("--gp3d-nsph-nodes", type=int, default=32,
                   help="(gp3d) Number of Fibonacci sphere inducing nodes M_sph "
                        "(default 32, the historical value). A spacing "
                        "sqrt(4pi/M_sph) above the chordal angular lengthscale "
                        "draws a warning.")
    p.add_argument("--gp3d-z-node-hi", type=float, default=None,
                   help="(gp3d) Top redshift of the radial inducing nodes. "
                        "Default: the package zgrid max (zgrid[-1]) — a "
                        "BEHAVIOR CHANGE from the historical hardwired 3.0, "
                        "which left zgrid[-1] > 3 covered only by prior "
                        "extrapolation.")
    p.add_argument("--c-mode", choices=["per_pixel", "aggregate", "selection"],
                   default="per_pixel",
                   help="Completeness base the fit is residual to (both modes). "
                        "'per_pixel' (default): legacy per-pixel matched-kernel "
                        "C, occupied pixels only. 'aggregate': ONE sky-aggregate "
                        "Cbar (matching the inference c_mode=aggregate "
                        "estimator), with EMPTY pixels included as N_obs=0 rows "
                        "(voids are informative). 'selection': the PARAMETRIC "
                        "C_sel(z; theta_hat) from --selection-fit (no counts in "
                        "the budget; empty pixels included like aggregate). "
                        "Stamped in the HDF5 attrs; the inference hard-errors "
                        "when the table's c_mode does not match the survey's.")
    p.add_argument("--selection-fit", default=None,
                   help="selection_fit.json from darksirens_fit_selection "
                        "(required by, and only legal with, --c-mode "
                        "selection). theta_hat is stamped into the table "
                        "provenance; the inference hard-errors when it does "
                        "not match the --selection_fit prior center it runs "
                        "with (a stale table carries the wrong fixed base).")
    p.add_argument("--log10n0", type=float, default=None,
                   help="Override log10 of the expected comoving galaxy density "
                        "[Mpc^-3] the fit is conditioned on (default -2.0). "
                        "CALIBRATE this to the catalog (N / (f_sky * V_c)); a "
                        "mis-set n0 is absorbed into Q as spurious z-structure.")
    p.add_argument("--delta", type=float, default=None,
                   help="Override the expected-density evolution exponent "
                        "(1+z)^delta (default 0.0).")
    p.add_argument("--stratum-map", default=None,
                   help="HDF5 file with a full-sky 'stratum_map' dataset "
                        "(RING, at the catalog's nside). Required by, and "
                        "only legal with, a MULTI-stratum --selection-fit "
                        "(darksirens-selection-fit-1.1): each pixel's C_sel "
                        "base row uses its own stratum's theta_hat, and the "
                        "map's sha256 + per-stratum thetas are stamped so "
                        "the inference can verify the table against the "
                        "run's fit and map.")
    p.add_argument("--indexing", choices=["compact", "global"], default="global",
                   help="How the inference should index the Q rows. Both build "
                        "modes emit a full global (n_pix, n_grid) table, so "
                        "'global' is the only stamp this builder writes; "
                        "'compact' is accepted for script compatibility but "
                        "forced back to 'global' with a warning.")
    p.add_argument("--allow-unconverged", action="store_true", default=False,
                   help="Save the completion even when solves are unconverged "
                        "or produced non-finite cells (substituted with Q=1). "
                        "The override is stamped in the file's diagnostics and "
                        "warned about at load time; research ablations only, "
                        "never production.")
    p.add_argument("--no-budget-renorm", action="store_true", default=False,
                   help="Skip the per-z mean-one budget renormalization of Q "
                        "under the (1-C)*dN_exp weights (default ON). Without "
                        "it Q's per-z monopole RESCALES the missing budget "
                        "(measured +55%% Jensen inflation for radial tables) "
                        "instead of only redistributing it. The choice is "
                        "stamped in the file; research ablations only.")
    opts = p.parse_args(argv)

    print()
    _banner("LSS LOGNORMAL COMPLETION")
    print()

    indexing = opts.indexing
    if indexing != "global":
        # BOTH builders scatter occupied rows into the full (n_pix, n_grid)
        # table. A 'compact' stamp makes the inference factory treat the rows
        # as already union-pixel-aligned and skip the global gather — wrong
        # rows (or a traced-index crash) at likelihood time.
        _warn("this builder emits a global all-pixel table; forcing "
              "indexing='global' (a 'compact' stamp would misindex Q rows "
              "at inference).")
        indexing = "global"

    selection_fit = None
    selection_strata = None
    stratum_map = None
    stratum_map_sha = None
    if opts.selection_fit is not None:
        from darksirens.redshift.selection import load_selection_fit_strata

        strata = load_selection_fit_strata(opts.selection_fit)
        selection_fit = strata[0]       # reference stratum (stamps, banner)
        if len(strata) > 1:
            if opts.stratum_map is None:
                raise ValueError(
                    f"--selection-fit carries {len(strata)} strata but no "
                    "--stratum-map assigns pixels to them; pass the map the "
                    "strata were defined on.")
            for s in strata[1:]:
                if tuple(s["k_corr_coeffs"]) != tuple(strata[0]["k_corr_coeffs"]):
                    raise ValueError(
                        f"per-stratum K(z) templates differ inside "
                        f"{opts.selection_fit}; strata must share one "
                        "template (refit with a common --k_corr_coeffs).")
            selection_strata = strata
            import h5py

            with h5py.File(opts.catalog, "r") as f:
                _nside_cat = int(f.attrs["nside"])
            stratum_map = _load_stratum_map(
                opts.stratum_map, _nside_cat, len(strata))
            stratum_map_sha = _stratum_map_sha256(opts.stratum_map)
        elif opts.stratum_map is not None:
            raise ValueError(
                "--stratum-map given but the --selection-fit carries a "
                "single stratum; drop the map or refit with --strata.")
    elif getattr(opts, "stratum_map", None) is not None:
        raise ValueError("--stratum-map without --selection-fit.")

    _section(f"Building  [{opts.mode}]")
    _row("Catalog", opts.catalog)
    _row("Members", opts.n_members)
    _row("Seed", opts.seed)
    if selection_fit is not None:
        # Family-driven: a schechter fit carries no M0hat/sigma_M at all.
        _family = str(selection_fit.get("family", "gaussian"))
        _row(f"Selection theta_hat [{_family}]",
             "  ".join(f"{k}={float(selection_fit[k]):.4f}"
                       for k in SELECTION_THETA_FIELDS[_family]))
        if selection_strata is not None:
            _row("Strata",
                 f"{len(selection_strata)} (per-pixel base via "
                 f"{opts.stratum_map})")
    logq_map, logq_members, diagnostics = build_completion(
        opts.catalog, mode=opts.mode, n_members=opts.n_members, seed=opts.seed,
        prior_strength=opts.prior_strength, maxiter=opts.maxiter,
        workers=opts.workers,
        gp3d_nz_solve=opts.gp3d_nz_solve, gp3d_pix_chunk=opts.gp3d_pix_chunk,
        lss_corr_length_ang=opts.lss_corr_length_ang,
        log10n0=opts.log10n0, delta=opts.delta,
        budget_renorm=not opts.no_budget_renorm,
        lss_corr_length_mpc=opts.lss_corr_length_mpc, lss_sigma=opts.lss_sigma,
        gp3d_nz_nodes=opts.gp3d_nz_nodes, gp3d_nsph_nodes=opts.gp3d_nsph_nodes,
        gp3d_z_node_hi=opts.gp3d_z_node_hi,
        c_mode=opts.c_mode, selection_fit=selection_fit,
        selection_strata=selection_strata, stratum_map=stratum_map,
        stratum_map_sha=stratum_map_sha,
    )
    _ok(f"MAP logq_map shape {logq_map.shape}; "
        f"members {'none' if logq_members is None else logq_members.shape}")
    if opts.mode == "gp3d":
        _ok(f"gp3d: converged={diagnostics.get('converged')} "
            f"n_iter={diagnostics.get('n_iter')} grad_inf={diagnostics.get('grad_inf'):.2e} "
            f"ls_z={diagnostics.get('ls_z_zeta'):.4f} z_ref={diagnostics.get('z_ref'):.3f}")
    elif opts.mode == "radial":
        n_conv = diagnostics.get("n_converged")
        n_occ = diagnostics.get("n_occupied")
        if n_conv is not None and n_occ:
            frac = n_conv / n_occ if n_occ else float("nan")
            line = f"radial: {n_conv}/{n_occ} occupied-pixel solves converged ({frac:.1%})"
            if n_conv == n_occ:
                _ok(line)
            else:
                _warn(line + f" — raise --maxiter (currently {opts.maxiter}); "
                             "an unconverged solve under-relaxes logQ toward 0 "
                             "(Q -> 1), silently weakening the LSS completion")

    # Fail-closed gate: an unconverged or non-finite completion is a science
    # artifact downstream inference cannot distinguish from a solved field, so
    # it is never written unless --allow-unconverged explicitly accepts it,
    # and the override is stamped for the loader to warn about.
    problems = []
    if opts.mode == "radial":
        n_conv = diagnostics.get("n_converged")
        n_occ = diagnostics.get("n_occupied")
        if n_occ and (n_conv is None or int(n_conv) < int(n_occ)):
            problems.append(
                f"{int(n_occ) - int(n_conv or 0)}/{int(n_occ)} occupied-pixel "
                f"solves unconverged (last optimizer message: "
                f"{diagnostics.get('last_failure_message')!r})"
            )
    elif not diagnostics.get("converged", False):
        problems.append(
            f"gp3d solve unconverged (grad_inf={diagnostics.get('grad_inf')})"
        )
    n_bad = int(diagnostics.get("n_nonfinite_map") or 0) + int(
        diagnostics.get("n_nonfinite_members") or 0)
    if n_bad:
        problems.append(f"{n_bad} non-finite logQ cells")
    diagnostics["allow_unconverged"] = bool(opts.allow_unconverged)
    if problems and not opts.allow_unconverged:
        raise SystemExit(
            "FATAL: refusing to write '%s': %s. Raise --maxiter (currently "
            "%d), or pass --allow-unconverged only for a stamped research "
            "ablation. No artifact was written." % (
                opts.out, "; ".join(problems), opts.maxiter)
        )
    if n_bad:
        _warn(f"substituting Q = 1 for {n_bad} non-finite cells "
              "(--allow-unconverged).")
        logq_map = np.where(np.isfinite(logq_map), logq_map, 0.0)
        if logq_members is not None:
            logq_members = np.where(
                np.isfinite(logq_members), logq_members, 0.0)
        diagnostics["n_nonfinite_substituted"] = n_bad
    # The removed budget monopole goes into its DEDICATED attr (with the
    # boolean stamp), not the JSON diagnostics blob — the loader treats an
    # absent stamp as a legacy (non-renormalized) table and warns.
    budget_monopole = diagnostics.pop("budget_monopole_logq", None)
    save_lss_completion_hdf5(
        opts.out, logq_map=logq_map, logq_members=logq_members,
        zgrid=np.asarray(zgrid), indexing=indexing, metadata=diagnostics,
        budget_renormalized=diagnostics.get("budget_renormalized"),
        budget_monopole_logq=budget_monopole,
        c_mode=opts.c_mode,
    )
    _ok(f"completion  →  {opts.out}")
    _end()


if __name__ == "__main__":
    main()
