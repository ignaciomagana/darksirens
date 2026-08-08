"""
darksirens_build_joint_lognormal_completion
--------------------------------------------
**Offline** preprocessor that infers ONE latent LSS field from K survey
catalogs JOINTLY and emits K per-survey ``Q_LSS(p, z)`` ensembles whose member
``m`` is the SAME field realization, stamped with one shared
``realization_set_id``.  This makes a K>=2 ``darksirens_inference
--lss_marginalize`` run legitimate WITHOUT
``--allow_unverified_shared_lss_members``: member ``m`` of every catalog samples
the same missing-galaxy field.

It must NOT be run inside the GW likelihood — it is a one-off build step.

Model (gp3d only — the joint mode)
----------------------------------
The joint posterior over the M whitened GP latents ``xi`` stacks the K surveys'
occupied-voxel Poisson blocks::

    J(xi) = 0.5 ||xi||^2 + sum_k sum_{v in V_k} [ lam_kv - N_kv log lam_kv ].

**Per-survey bias b_k is absorbed into the design matrix**: build each survey's
low-rank operator ``Phi_k`` with the SHARED hyperparameters (amp = lss_sigma,
ls_sph = lss_corr_length_ang, ls_z from the JOINT count-weighted z_ref), then
pass ``Phi'_k = b_k * Phi_k`` and ``bias=1.0`` into ONE stacked
:func:`poisson_lognormal_gp3d_map`.  Because the solver defaults
``sigma2_vox = sum(Phi'^2, axis=1) = b_k^2 sigma^2_vox``, the mean-one shift is
automatically per-survey correct and ``H = I + Phi'^T diag(lam) Phi'`` is the
correct joint Laplace precision — the library solver
(:func:`poisson_lognormal_gp3d_map` / :func:`laplace_lognormal_gp3d_members` /
:func:`eval_logq_gp3d`) is used UNCHANGED.

One :func:`laplace_lognormal_gp3d_members` call draws the SHARED ``xi_m``
ensemble; each survey's per-member and posterior-mean ``logQ_k`` are evaluated
with ``bias=b_k`` at that survey's own directions (arbitrary nside/mask/depth),
so members are matched across the K files by construction.

**Radial mode is rejected** with an explanatory error: the radial completion
fits each pixel's 1-D field INDEPENDENTLY (no shared latent field even within
one survey), so "matched members" across surveys would be RNG-position
coincidence, not one physical shared realization.
"""
from __future__ import annotations

import argparse
import hashlib
import uuid

import numpy as np

from darksirens.cli.common import _banner, _section, _row, _end, _ok, _warn
from darksirens.redshift import zgrid
from darksirens.redshift.lognormal_completion import (
    lowrank_inducing_nodes,
    build_lowrank_operator,
    poisson_lognormal_gp3d_map,
    laplace_lognormal_gp3d_members,
    eval_logq_gp3d,
    save_lss_completion_hdf5,
)
from darksirens.cli.build_lognormal_completion import (
    _fiducial_cosmo_survey,
    _apply_lss_overrides,
    _assemble_gp3d_survey,
    _count_weighted_zref_lsz,
    _gp3d_base_diagnostics,
    _gp3d_resolution_guard,
)

# The joint builder's inducing grid is FIXED (no per-run node knobs): the
# historical 32 x 6 (sphere x z) nodes up to z = 3.  The grid must still
# resolve the shared ls_z (hard gate below); the CLI remedy is the shared
# radial correlation length.
M_SPH, M_Z = 32, 6
Z_NODE_HI = 3.0


def _broadcast_arity(values, K, name, *, default):
    """Resolve a 1-or-K per-survey option to a length-K list.

    ``None`` -> ``[default]`` -> broadcast; one value -> broadcast; K values ->
    as given; anything else is a clear error.
    """
    if values is None:
        values = [default]
    values = list(values)
    if len(values) == 1:
        return [values[0]] * int(K)
    if len(values) == int(K):
        return values
    raise ValueError(
        f"--{name} takes 1 or K={K} values (one shared value, or one per "
        f"catalog); got {len(values)}: {values}."
    )


def _joint_diag_keys(k, K, catalog_paths, biases, seed):
    """The joint-provenance diagnostics stamped identically (bar the 0-based
    ``joint_survey_index``) in every one of the K files."""
    return {
        "joint_n_surveys": int(K),
        "joint_survey_index": int(k),
        "joint_catalogs": [str(p) for p in catalog_paths],
        "joint_biases": [float(b) for b in biases],
        "joint_seed": int(seed),
    }


def build_joint_completion(
    catalog_paths,
    out_paths,
    *,
    n_members: int = 32,
    seed: int = 1234,
    biases=None,
    log10n0s=None,
    deltas=None,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
    lss_corr_length_mpc=None,
    lss_sigma=None,
    realization_set_id=None,
    allow_unconverged: bool = False,
):
    """Jointly infer ONE LSS field from K catalogs and write K matched Q files.

    Fail-closed contract: an unconverged shared solve, or non-finite logQ
    output, aborts BEFORE any file is written (all K tables are computed
    first, then saved) unless ``allow_unconverged`` explicitly accepts and
    stamps the degraded artifact.

    Returns a list of per-file dicts (ordered as ``catalog_paths``) carrying the
    written ``out_path``, the shared ``realization_set_id``, the per-file
    ``diagnostics``, and the ``logq_map`` / ``logq_members`` arrays.
    """
    catalog_paths = list(catalog_paths)
    out_paths = list(out_paths)
    K = len(catalog_paths)
    if K == 0:
        raise ValueError("build_joint_completion needs at least one catalog.")
    if len(out_paths) != K:
        raise ValueError(
            f"--outs ({len(out_paths)}) must have the same length as --catalogs "
            f"({K}); one output file per input catalog, positionally aligned."
        )
    biases = _broadcast_arity(biases, K, "bias", default=1.0)
    log10n0s = _broadcast_arity(log10n0s, K, "log10n0", default=-2.0)
    deltas = _broadcast_arity(deltas, K, "delta", default=0.0)

    zgrid_np = np.asarray(zgrid, dtype=float)
    n_grid = int(zgrid.size)

    # SHARED coarse solve grid (survey-independent) and SHARED hyperparameters:
    # one latent field ==> one amp / ls_sph / set of inducing nodes across all K.
    z_s = np.linspace(0.0, float(zgrid_np[-1]), int(gp3d_nz_solve))
    edges_s = np.concatenate([[z_s[0]], 0.5 * (z_s[:-1] + z_s[1:]), [z_s[-1]]])
    zeta_s = np.log1p(z_s)
    G_s = int(z_s.size)

    cosmo, ref_survey = _fiducial_cosmo_survey()
    # SHARED GP hyperparameter overrides (one latent field => one set); mirrors
    # the single-survey builder's build-time-only knobs, never sampled.
    ref_survey = _apply_lss_overrides(
        ref_survey, lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma)
    amp = float(ref_survey.lss_sigma)
    ls_sph = (float(lss_corr_length_ang) if lss_corr_length_ang is not None
              else float(ref_survey.lss_corr_length_ang))
    corr_length_mpc = float(ref_survey.lss_corr_length_mpc)

    # Per-survey assembly (its own fiducial n0/delta) on the shared coarse grid.
    surveys = []
    assemblies = []
    for path, l10n0, dlt in zip(catalog_paths, log10n0s, deltas):
        _c, survey_k = _fiducial_cosmo_survey(log10n0=l10n0, delta=dlt)
        if lss_corr_length_ang is not None:
            survey_k = survey_k._replace(lss_corr_length_ang=float(lss_corr_length_ang))
        survey_k = _apply_lss_overrides(
            survey_k, lss_corr_length_mpc=lss_corr_length_mpc, lss_sigma=lss_sigma)
        surveys.append(survey_k)
        assemblies.append(_assemble_gp3d_survey(
            path, cosmo=cosmo, survey=survey_k, z_s=z_s, edges_s=edges_s))

    rid = realization_set_id if realization_set_id is not None else uuid.uuid4().hex
    make_members = bool(n_members and n_members > 0)
    total_occ = int(sum(a.n_occ for a in assemblies))

    # Empty joint fit (no occupied voxels anywhere): Q = 1 for every survey,
    # still sharing the id (mirrors the single-survey empty shortcut). No shared
    # field is drawn, so joint_member_xi_sha256 is omitted.
    if total_occ == 0:
        _warn("no occupied pixels in ANY catalog — writing Q = 1 "
              "(logQ = 0) everywhere for all surveys.")
        results = []
        for k in range(K):
            a, survey_k, b_k = assemblies[k], surveys[k], float(biases[k])
            logq_map = np.zeros((a.n_pix, n_grid), dtype=float)
            logq_members = (np.zeros((int(n_members), a.n_pix, n_grid), dtype=float)
                            if make_members else None)
            diag = _gp3d_base_diagnostics(
                cosmo, survey_k, nside=a.nside, n_pix=a.n_pix, n_occ=a.n_occ,
                amp=amp, ls_sph=ls_sph, bias=b_k, mode="gp3d_joint")
            diag.update(_joint_diag_keys(k, K, catalog_paths, biases, seed))
            diag["z_ref"] = 1.0
            if make_members:
                diag["n_members"] = int(n_members)
            save_lss_completion_hdf5(
                out_paths[k], logq_map=logq_map, logq_members=logq_members,
                zgrid=zgrid_np, indexing="global", metadata=diag,
                realization_set_id=rid)
            results.append(dict(
                out_path=out_paths[k], realization_set_id=rid, diagnostics=diag,
                logq_map=logq_map, logq_members=logq_members, n_occupied=a.n_occ))
        return results

    # Joint count-weighted z_ref over ALL surveys -> one shared ls_z.
    z_ref, ls_z = _count_weighted_zref_lsz(
        [a.counts_z for a in assemblies], z_s, cosmo, corr_length_mpc)

    # HARD gate before any solve -- the SAME S0c guard the single-survey
    # builder runs on the identical configuration: a low-rank GP whose
    # inducing-node spacing in zeta exceeds ls_z collapses to the prior while
    # still stamping converged=True (Burt et al. 2019; the shipped 50 Mpc
    # fiducial is ~30x under-resolved for the fixed 6-node grid and measured a
    # fitted-vs-truth logQ slope of 0.04).  The joint inducing grid is fixed
    # (M_SPH x M_Z up to Z_NODE_HI), so the message is re-anchored to the one
    # knob this CLI has.
    try:
        zeta_spacing = _gp3d_resolution_guard(
            z_node_hi=Z_NODE_HI, n_z_nodes=M_Z, ls_z=ls_z,
            n_sph_nodes=M_SPH, ls_sph=ls_sph)
    except ValueError as exc:
        raise ValueError(
            f"{exc} [JOINT builder: the inducing grid is fixed at "
            f"M_sph={M_SPH} x M_z={M_Z} up to z_node_hi={Z_NODE_HI:g}, so "
            "the only remedy here is raising --lss-corr-length-mpc until the "
            "node spacing resolves ls_z.]"
        ) from None

    # SHARED inducing nodes; one Cholesky L (identical hyperparams => identical
    # L). Stack the occupied surveys' voxel blocks with the bias-scaled design
    # Phi'_k = b_k * Phi_k and solve ONE joint Poisson-lognormal MAP with the
    # library solver UNCHANGED (bias=1.0; its default sigma2_vox = sum(Phi'^2)
    # is the per-survey-correct b_k^2 sigma^2_vox mean-one shift).
    Zn, Zz = lowrank_inducing_nodes(
        n_inducing_sphere=M_SPH, n_inducing_z=M_Z, z_node_hi=Z_NODE_HI)
    M = int(np.asarray(Zn).shape[0])

    N_blocks, base_blocks, phi_blocks = [], [], []
    L_shared = None
    for a, b_k in zip(assemblies, biases):
        if a.n_occ == 0:
            continue  # empty survey contributes no voxels (still gets a Q map below)
        X_n = np.repeat(a.n_hat_occ, G_s, axis=0)                     # (n_occ*G_s, 3)
        X_z = np.tile(zeta_s, a.n_occ)                                # (n_occ*G_s,)
        Phi_k, L_k = build_lowrank_operator(
            Zn, Zz, X_n, X_z, amp=amp, ls_sph=ls_sph, ls_z=ls_z)
        if L_shared is None:
            L_shared = L_k
        phi_blocks.append(float(b_k) * np.asarray(Phi_k, dtype=float))
        N_blocks.append(np.asarray(a.N_obs_vox, dtype=float).ravel())
        base_blocks.append(np.asarray(a.base_vox, dtype=float).ravel())

    N_obs = np.concatenate(N_blocks)
    base = np.concatenate(base_blocks)
    Phi_stack = np.vstack(phi_blocks)

    mp = poisson_lognormal_gp3d_map(N_obs, base, Phi_stack, bias=1.0)
    if not bool(mp["diagnostics"]["converged"]) and not allow_unconverged:
        raise ValueError(
            "joint gp3d solve did not converge (grad_inf="
            f"{mp['diagnostics']['grad_inf']:.2e} after "
            f"{mp['diagnostics']['n_iter']} iterations); refusing to write "
            "any Q file — an unconverged completion silently under-relaxes "
            "logQ toward 0 (Q -> 1) and weakens the LSS correction. Pass "
            "--allow-unconverged only for a stamped research ablation."
        )

    # ONE shared Laplace ensemble in the whitened latents: xi_m matched across
    # all K files by construction.
    xi_members = None
    member_sha = None
    if make_members:
        xi_members = laplace_lognormal_gp3d_members(
            mp["xi_map"], mp["H_chol"], n_members=int(n_members), seed=int(seed))
        member_sha = hashlib.sha256(
            np.ascontiguousarray(xi_members).tobytes()).hexdigest()

    sig2 = np.asarray(mp["sigma2_vox"], dtype=float)
    # Compute ALL K tables before saving ANY of them: a failure mid-set must
    # not leave a partial (and therefore unmatched) family of Q files.
    pending = []
    for k in range(K):
        a, survey_k, b_k = assemblies[k], surveys[k], float(biases[k])

        # Posterior-mean E[Q_k] over ALL this survey's pixels: empty surveys and
        # far pixels read Q = 1; pixels angularly near ANY survey's structure
        # borrow the shared field. bias=b_k applies the per-survey modulation on
        # the unscaled Phi_k that eval builds internally; H_chol is the JOINT
        # precision, so the shrinkage is the joint posterior's.
        logq_map = np.asarray(eval_logq_gp3d(
            mp["xi_map"], Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
            n_hat_out=a.n_hat_all, z_out=zgrid_np, bias=b_k,
            pix_chunk=int(gp3d_pix_chunk), L=L_shared, H_chol=mp["H_chol"],
        ), dtype=float)
        n_bad = int(np.sum(~np.isfinite(logq_map)))

        logq_members = None
        if make_members:
            logq_members = np.asarray(eval_logq_gp3d(
                xi_members, Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
                n_hat_out=a.n_hat_all, z_out=zgrid_np, bias=b_k,
                pix_chunk=int(gp3d_pix_chunk), L=L_shared,
            ), dtype=float)
            n_bad += int(np.sum(~np.isfinite(logq_members)))

        if n_bad and not allow_unconverged:
            raise ValueError(
                f"{n_bad} non-finite logQ entries for '{out_paths[k]}'; "
                "refusing to write any Q file — a substituted Q=1 cell is "
                "indistinguishable from real homogeneity downstream. Pass "
                "--allow-unconverged only for a stamped research ablation."
            )
        if n_bad:
            logq_map = np.where(np.isfinite(logq_map), logq_map, 0.0)
            if logq_members is not None:
                logq_members = np.where(
                    np.isfinite(logq_members), logq_members, 0.0)

        diag = _gp3d_base_diagnostics(
            cosmo, survey_k, nside=a.nside, n_pix=a.n_pix, n_occ=a.n_occ,
            amp=amp, ls_sph=ls_sph, bias=b_k, mode="gp3d_joint")
        diag.update({
            "M_sph": M_SPH, "M_z": M_Z, "M": M, "gp3d_nz_solve": G_s,
            "gp3d_z_node_hi": Z_NODE_HI, "zeta_node_spacing": zeta_spacing,
            "ls_z_zeta": ls_z, "z_ref": z_ref, "ls_sph_chordal": ls_sph,
            "sigma2_vox_min": float(sig2.min()), "sigma2_vox_max": float(sig2.max()),
            "n_iter": int(mp["diagnostics"]["n_iter"]),
            "converged": bool(mp["diagnostics"]["converged"]),
            "grad_inf": float(mp["diagnostics"]["grad_inf"]),
            # New-style convergence stamp: records that the fail-closed gate
            # ran, and whether the operator overrode it (see the loader's
            # _warn_if_unconverged for how these are consumed).
            "allow_unconverged": bool(allow_unconverged),
            "n_nonfinite_substituted": int(n_bad),
        })
        diag.update(_joint_diag_keys(k, K, catalog_paths, biases, seed))
        if make_members:
            diag["n_members"] = int(n_members)
            # sha256 of the SHARED stacked xi_m draws — identical in all K files;
            # the cross-file field-level member-order provenance hash.
            diag["joint_member_xi_sha256"] = member_sha

        pending.append(dict(
            out_path=out_paths[k], diagnostics=diag, logq_map=logq_map,
            logq_members=logq_members, n_occupied=a.n_occ))

    results = []
    for entry in pending:
        save_lss_completion_hdf5(
            entry["out_path"], logq_map=entry["logq_map"],
            logq_members=entry["logq_members"],
            zgrid=zgrid_np, indexing="global", metadata=entry["diagnostics"],
            realization_set_id=rid)
        results.append(dict(entry, realization_set_id=rid))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build JOINT multi-survey LSS lognormal completion files "
                    "(offline; gp3d only). One shared latent field over K "
                    "catalogs -> K matched Q ensembles sharing one "
                    "realization_set_id for K>=2 --lss_marginalize."
    )
    p.add_argument("--catalogs", nargs="+", required=True, metavar="PATH",
                   help="K pixelated survey catalogs (load_survey schema).")
    p.add_argument("--outs", nargs="+", required=True, metavar="PATH",
                   help="K output completion HDF5 paths, positionally aligned "
                        "with --catalogs (same count).")
    p.add_argument("--n-members", type=int, default=32,
                   help="Number of matched Laplace ensemble members (0 = MAP only).")
    p.add_argument("--seed", type=int, default=1234,
                   help="Seed for the ONE shared Laplace member draw set.")
    p.add_argument("--bias", nargs="+", type=float, default=None, metavar="B",
                   help="Per-survey field bias b_k (1 shared value or K values; "
                        "default 1.0). Absorbed into the design matrix "
                        "(Phi'_k = b_k * Phi_k).")
    p.add_argument("--log10n0", nargs="+", type=float, default=None, metavar="L",
                   help="Per-survey log10 expected comoving density [Mpc^-3] the "
                        "fit is conditioned on (1 or K values; default -2.0). "
                        "CALIBRATE per catalog (N / (f_sky * V_c)).")
    p.add_argument("--delta", nargs="+", type=float, default=None, metavar="D",
                   help="Per-survey expected-density evolution exponent (1+z)^delta "
                        "(1 or K values; default 0.0).")
    p.add_argument("--gp3d-nz-solve", type=int, default=32,
                   help="Number of coarse redshift points for the shared solve grid.")
    p.add_argument("--gp3d-pix-chunk", type=int, default=512,
                   help="Pixel chunk size for the per-survey all-pixel evaluation.")
    p.add_argument("--lss-corr-length-ang", type=float, default=None,
                   help="Override the shared angular (chordal) correlation length; "
                        "default is the SurveyParams fiducial.")
    p.add_argument("--lss-corr-length-mpc", type=float, default=None,
                   help="Override the shared radial GP correlation length [Mpc]; "
                        "default is the SurveyParams fiducial (50). Build-time "
                        "only, never sampled. The FIXED joint inducing grid "
                        "(32 x 6 nodes up to z = 3) must resolve the mapped "
                        "ls_z or the build hard-errors (the 50 Mpc fiducial "
                        "does not).")
    p.add_argument("--lss-sigma", type=float, default=None,
                   help="Override the shared GP field amplitude; default is the "
                        "SurveyParams fiducial (1.0). Build-time only, never "
                        "sampled.")
    p.add_argument("--allow-unconverged", action="store_true", default=False,
                   help="Save the Q files even when the shared solve is "
                        "unconverged or produced non-finite cells (substituted "
                        "with Q=1). The override is stamped in the files' "
                        "diagnostics and warned about at load time; research "
                        "ablations only, never production.")
    p.add_argument("--realization-set-id", type=str, default=None,
                   help="Shared realization_set_id stamped on all K files "
                        "(default: a fresh uuid4 shared across the K outputs).")
    p.add_argument("--mode", choices=["gp3d", "radial"], default="gp3d",
                   help="Joint completion model. Only 'gp3d' is supported; "
                        "'radial' is rejected (no shared field to match members).")
    opts = p.parse_args(argv)

    if opts.mode == "radial":
        p.error(
            "--mode radial is not supported by the JOINT builder: the radial "
            "completion fits each pixel's 1-D field INDEPENDENTLY (no shared "
            "latent field even within one survey), so 'matched members' across "
            "surveys would be RNG-position coincidence, not one shared LSS "
            "realization. Use --mode gp3d (the joint mode), or build each survey "
            "separately with darksirens_build_lognormal_completion --mode radial."
        )
    if len(opts.outs) != len(opts.catalogs):
        p.error(
            f"--outs ({len(opts.outs)}) must match --catalogs "
            f"({len(opts.catalogs)}) in length; one output per input catalog."
        )

    print()
    _banner("JOINT LSS LOGNORMAL COMPLETION")
    print()

    _section(f"Building  [gp3d, {len(opts.catalogs)} catalogs]")
    for c, o in zip(opts.catalogs, opts.outs):
        _row(c, f"→  {o}", width=0)
    try:
        results = build_joint_completion(
            opts.catalogs, opts.outs,
            n_members=opts.n_members, seed=opts.seed,
            biases=opts.bias, log10n0s=opts.log10n0, deltas=opts.delta,
            gp3d_nz_solve=opts.gp3d_nz_solve, gp3d_pix_chunk=opts.gp3d_pix_chunk,
            lss_corr_length_ang=opts.lss_corr_length_ang,
            lss_corr_length_mpc=opts.lss_corr_length_mpc,
            lss_sigma=opts.lss_sigma,
            realization_set_id=opts.realization_set_id,
            allow_unconverged=opts.allow_unconverged,
        )
    except ValueError as exc:
        raise SystemExit(f"FATAL: {exc}") from None
    rid = results[0]["realization_set_id"] if results else None
    d0 = results[0]["diagnostics"] if results else {}
    _ok(f"shared realization_set_id={rid}")
    if "converged" in d0:
        _ok(f"joint gp3d: converged={d0.get('converged')} "
            f"n_iter={d0.get('n_iter')} grad_inf={d0.get('grad_inf'):.2e} "
            f"ls_z={d0.get('ls_z_zeta'):.4f} z_ref={d0.get('z_ref'):.3f}")
    for r in results:
        m = r["logq_members"]
        _ok(f"{r['out_path']}: logq_map {r['logq_map'].shape}; "
            f"members {'none' if m is None else m.shape}; "
            f"n_occupied={r['n_occupied']}")
    _end()
    return results


if __name__ == "__main__":
    main()
