#!/usr/bin/env python
"""Build the latent-field ANCHOR artifact (field-level PR-4).

One offline solve of the shell-total-conditioned count channel at
``theta_ref`` plus everything the seam consumes online, written to one HDF5
group ``/latent_field``:

    xi_hat            (M,)                 count-channel MAP
    H_chol            (M, M)               lower Cholesky of the Fisher H
    sensitivity_S     (M, n_theta)         d xi_hat / d theta  (IFT columns)
    sensitivity_labels                     the theta names, in column order
    g_members         (M_draw, M)          antithetic standard-normal draws
    eps_members       (M_draw,)            antithetic scalar normals of the
                                           b_gal rank-1 stream (zeros if the
                                           inflation is off)
    Xi_members        (M_draw, M_sph, M_z) xi_hat + L^{-T} g + s_b eps v,
                                           reshaped
    row_fac           (M_draw, n_fit, M_z) f32 member row factors (the seam's
                                           static leaf: Phi_s[fit] @ Xi_m)
    A_moments, B_moments (M_draw, n_b, N_z_sub)  PLAN eq. (2) sky moments
    dA_moments, dB_moments (M_draw, n_b, N_z_sub, n_theta) projected
                                           theta-derivatives (PR-6b/P18 use)
    b_nodes           (n_b,)               Chebyshev b_GW grid
    z_sub             (N_z_sub,)           consumption-grid nodes <= z_depth
                                           (<= z_node_hi with an amp(z) profile)
    P_F, F_F                               eq. (2) constants (footprint)
    fit_pixels        (n_fit,)             footprint pixel ids
    completeness      (n_fit,)             f_p rows
    counts            (n_fit, G_s)         integer shell counts
    z_count_edges     (G_s + 1,)           stamped shell edges (never ZMAX)
    shell_response    (G_s, N_fine)        the frozen W at theta_ref
    basis_meta / theta_ref / sha256        provenance (guard 1)

**The draw covariance is PLAN §3.4's, not `H^{-1}`.**  Members are drawn from
``H^{-1} + s_b^2 v v^T`` with ``v = d xi_hat/d b`` (the ``b_gal`` column of
``sensitivity_S``) and ``s_b`` the measured profile curvature of the count
channel in ``b_gal``; ``--no-b-gal-dispersion`` restores the pre-S-2
``H^{-1}``-only convention for comparison.  The convention, ``s_b``, its
profile/floor decomposition and the implied member-spread inflation are all
stamped into the artifact, so a reader can tell which one an anchor carries.
Turning the inflation on CHANGES THE STAMPED sha256 of an anchor (the members,
and therefore ``A``/``B``/``row_fac``, move); it does not change guard 1's
CONTENT digest, which covers the field and not the draws.  The two-build
reproducibility gate is unaffected: everything, including the new scalar
stream, is a deterministic function of ``--seed``.

**``--amp-hi`` builds a SENSITIVITY-SCAN artifact, not a better anchor
(field-level PR-8).**  ``amp`` is pinned at 1 wherever the counts constrain the
field (PLAN §4.3: ``(b, xi)`` and ``(amp, xi)`` enter only through ``b*xi`` and
``b*amp``, so ``b_gal`` is the sole clustering amplitude there).  Above
``--z-depth`` there are no counts at all, so ``--amp-hi`` is an ASSUMPTION and
the width it produces is a pure function of it.  Anchors built this way exist to
be scanned -- "H0 shift under an assumed amp(z)" -- and never to be marginalized
into a headline posterior (OWNER DECISION 7).  ``--amp-hi 0`` is the shipped
convention (``Q == 1`` above the depth) stated explicitly.

**K >= 2 (field-level PR-7): one field, K catalogs of selection against it.**
With ``--tracer-labels`` naming a per-galaxy tracer index in ``--survey``, the
counts are partitioned into K DISJOINT arrays (OWNER DECISION 9 -- a label is
single-valued, so the partition is structural, not checked), the solve becomes
the stacked objective ``J = 0.5||xi||^2 - sum_k log p_count,k`` over ONE ``xi``,
and the artifact gains

    /latent_field/tracers/<k>/counts, completeness, A_moments, B_moments,
                              shell_response
                            attrs: name, b_gal, P_F, F_F,
                                   selection_fit, m_lim, sigma_z, theta_ref
    /latent_field attrs: n_tracer, tracer_names, tracer_bias,
                         tracer_overlap_policy, tracer_theta_ref,
                         bias_profile_cov, bias_prior_log_sd,
                         bias_cov_convention

on top of the shared block, which does not change shape: there is still one
``xi_hat``, one ``H_chol``, one member ensemble.  That layout IS PLAN §4.4's
shared-realization theorem written down -- "member m of every catalog is the
same realization" is true here because there is no per-catalog index on ``xi``
for a ``realization_set_id`` to disagree about, so the check is retired for want
of a referent rather than satisfied.  ``sensitivity_S`` gains K bias columns in
place of one (K >= 2 ADDS COLUMNS, PLAN §3.4), the moment tables are per tracer
because eq. (2)'s two reductions run over that tracer's footprint and weight its
own ``f_p``, and the rank-1 draw-covariance inflation becomes rank K with a
CORRELATED ``C_b`` -- the off-diagonal being the shared field itself.

**One field, K SELECTIONS -- including the shell response.**  A tracer's counts
are a shell multinomial whose WITHIN-SHELL weighting is that catalog's own
magnitude selection ``C(z; m_lim, M0hat, sigma_M, k_corr)`` convolved with its
own photo-z kernel; that weighting is exactly what ``W`` is.  So ``W`` is per
tracer (``--tracer-selection-fit``, ``--tracer-sigma-z``, defaulting to the
shared ``--selection-fit`` and to ``SIGMA_Z_DEFAULT``), and ``M0hat`` and
``sigma_M`` become K sensitivity columns each -- catalog k's own theta, built
from catalog k's own operator, which is PLAN §3.4's sentence rather than a
paraphrase of it.  ``delta`` and ``Om0`` stay one column apiece: they describe
the field and the cosmology, which the K catalogs share.

**A K = 1 build is byte-identical through all of that, digest included.**  The
tracer switches are dropped from the stamped configuration when ``K == 1`` and
the operator handed to the solve is a bare ``CountOperator``, so the K = 1 path
evaluates the pre-PR-7 expression tree.  Measured: a rebuild of the PR-6a
closure world (nside 16, 1854 pixels, rank 320) is byte-identical as a FILE to
the same build at commit ``03c5882``, sha256 ``662d1d7f...`` on both sides
(``experiments/field_level_plan/pr7/REPORT.md``).

The build is MINUTES at production rank (PLAN §3.4): the solve is ~13
Fisher-scoring trips over an M x M system, and the heavy reductions are two
staged contractions.  Everything is deterministic at fixed ``--seed``; the
sha256 covers the arrays and the configuration, so two same-seed builds are
byte-comparable (the PR-4 reproducibility gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

#: Population-average photo-z kernel width used when no per-tracer value is
#: given.  It is a CONSTANT and not a flag: turning it into one would put a
#: non-``None`` key into the stamped configuration of every K = 1 anchor ever
#: built and move its digest while moving not one array in it (the same reason
#: the tracer switches below are additive).  ``--tracer-sigma-z`` defaults to
#: this value per tracer, so a K >= 2 build that does not pass it evaluates the
#: same kernel this constant always meant.
SIGMA_Z_DEFAULT = 0.023


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--selection-fit", required=True)
    ap.add_argument("--n0-calibration", required=True)
    ap.add_argument("--per-pixel-completeness", required=True,
                    help="depth map (guard 6: latent requires f_p)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--z-node-hi", type=float, default=None,
                    help="upper end of the radial inducing-node range "
                         "(default: --z-depth, the pre-PR-8 behaviour). Set it "
                         "ABOVE the depth only together with --amp-hi: nodes "
                         "above the fitted depth exist so the assumed amp(z) "
                         "field has somewhere to live, and --m-z must grow "
                         "with them (resolution guard).")
    ap.add_argument("--amp-hi", type=float, default=None,
                    help="PR-8 sensitivity scan: the ASSUMED clustering "
                         "amplitude above --z-depth, relative to the fitted "
                         "region (where PLAN 4.3 pins amp = 1). Omitted, or "
                         "0.0, reproduces the shipped 'Q == 1 above z_depth' "
                         "convention exactly. Any value > 0 makes this anchor "
                         "a SENSITIVITY-SCAN artifact: nothing in this "
                         "pipeline constrains amp(z) above the depth, so a "
                         "posterior from it is a statement about the assumption "
                         "(OWNER DECISION 7).")
    ap.add_argument("--amp-kind", default="step", choices=("step", "growth"),
                    help="'step': amp_hi above the depth (what the PR-8 table "
                         "quotes). 'growth': amp_hi * D(z)/D(z_depth).")
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--ls-sph", type=float, default=0.2)
    ap.add_argument("--ls-z", type=float, default=0.039,
                    help="zeta units (guard 2: Mpc is unrepresentable here)")
    ap.add_argument("--m-sph", type=int, default=315)
    ap.add_argument("--m-z", type=int, default=8)
    ap.add_argument("--b-gal", type=float, default=1.0)
    # ---- K >= 2 (field-level PR-7).  All three default to the K = 1 build.
    #
    # Deliberately ADDITIVE argument names rather than turning --b-gal and
    # --per-pixel-completeness into repeatable options: those two are stamped
    # into ``cfg`` and thence into the artifact's sha256, and changing a scalar
    # into a one-element list would move the stamped digest of every K = 1
    # anchor ever built while changing not one array in it.  The K = 1 build is
    # byte-identical through PR-7, digest included -- ``cfg`` drops all three
    # keys below (and --bias-prior-log-sd) when K == 1, so even the command-line
    # half of the stamp is unmoved (see the write block).
    ap.add_argument("--tracer-labels", default=None,
                    help="dataset name in --survey holding a per-galaxy "
                         "tracer index, shaped like zgals. Its presence is "
                         "what makes the build K >= 2. Because a label is "
                         "single-valued, the K count arrays are a DISJOINT "
                         "partition of the catalog by construction (OWNER "
                         "DECISION 9), never by a check.")
    ap.add_argument("--tracer-completeness", action="append", default=None,
                    help="per-tracer depth map, repeated K times in tracer-"
                         "index order. Each tracer has its own f_p because it "
                         "has its own selection; omit to reuse "
                         "--per-pixel-completeness for every tracer.")
    ap.add_argument("--tracer-b-gal", action="append", type=float,
                    default=None,
                    help="per-tracer b_k, repeated K times in tracer-index "
                         "order (omit to reuse --b-gal for every tracer).")
    ap.add_argument("--tracer-selection-fit", action="append", default=None,
                    help="per-tracer selection fit JSON, repeated K times in "
                         "tracer-index order (omit to reuse --selection-fit "
                         "for every tracer). Tracer k's magnitude selection "
                         "C(z; m_lim, M0hat, sigma_M, k_corr) is what weights "
                         "its galaxies WITHIN a shell, so it enters that "
                         "tracer's shell response W_k and nothing else's; a "
                         "shared W would give every tracer catalog 0's "
                         "within-shell weighting while its f_p and b_k were "
                         "its own.")
    ap.add_argument("--tracer-sigma-z", action="append", type=float,
                    default=None,
                    help="per-tracer photo-z kernel width, repeated K times in "
                         "tracer-index order (omit to reuse "
                         f"{SIGMA_Z_DEFAULT} for every tracer). It convolves "
                         "the MODEL inside W_k, so like the selection fit it "
                         "is a property of the catalog, not of the field.")
    ap.add_argument("--bias-prior-log-sd", type=float, default=None,
                    help="K >= 2 only: sd of the log-normal prior on each "
                         "tracer's b_k, centred on the anchor's own b_k "
                         "(default: --s-b-floor-frac, i.e. the same 5% the "
                         "K = 1 systematics floor states). The count channel "
                         "does not identify the bias AMPLITUDE -- only the "
                         "ratio -- so this is what makes the (K, K) draw-"
                         "covariance inflation a proper posterior rather than "
                         "the inverse of an indefinite matrix.")
    ap.add_argument("--tracer-names", default=None,
                    help="comma-separated tracer labels, in tracer-index "
                         "order (default t0,t1,...).")
    ap.add_argument("--no-b-gal-dispersion", action="store_true",
                    default=False,
                    help="draw members from H^{-1} alone (PR-6a's convention) "
                         "instead of PLAN §3.4's H^{-1} + s_b^2 v v^T; "
                         "reproduces a pre-S-2 anchor for comparison")
    ap.add_argument("--s-b-floor-frac", type=float, default=None,
                    help="systematics floor on s_b as a fraction of b_gal "
                         "(default: latent_counts.B_GAL_SYSTEMATIC_FLOOR_FRAC "
                         "= 0.05, PLAN §3.4 v4's 'stated systematics floor')")
    ap.add_argument("--m-draw", type=int, default=8)
    ap.add_argument("--n-b-nodes", type=int, default=33)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--h0", type=float, default=67.74)
    ap.add_argument("--om0", type=float, required=True)
    ap.add_argument("--seed", type=int, default=22)
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    import h5py
    import healpy as hp
    import jax.numpy as jnp

    from darksirens.catalogs.depth_map import load_selection_fraction
    from darksirens.redshift.grid import zgrid
    from darksirens.redshift.latent_counts import (
        B_GAL_SYSTEMATIC_FLOOR_FRAC, MultiTracerCountOperator, TracerCounts,
        b_gal_profile_sigma, bias_profile_curvature_log, bias_profile_grad,
        bias_profile_hessian, check_disjoint_tracers,
        count_map_solve, counts_from_catalog, counts_from_catalog_by_tracer,
        dgrad_db, dgrad_db_by_tracer, gradient, laplace_draws,
        laplace_draws_multitracer, make_count_operator,
        make_multi_count_operator, multi_gradient, sensitivity)

    def _grad_of(xi, o):
        """``gradient`` for a bare operator, ``multi_gradient`` for a stack.

        Written as an explicit two-branch helper and not as a single dispatch
        call, so that the K = 1 build provably evaluates ``gradient`` -- the
        pre-PR-7 expression -- and the sensitivity columns it produces are bit
        identical to the ones the shipped anchors carry.
        """
        if isinstance(o, MultiTracerCountOperator):
            return multi_gradient(xi, o)
        return gradient(xi, o)
    from darksirens.redshift.latent_field import (
        build_latent_basis, row_factor, shell_response, sky_constant_coeffs,
        sky_moments, sky_moments_by_tracer)
    from darksirens.redshift.selection import (
        c_sel_gaussian, load_selection_fit_json)

    CLIGHT = 299792.458
    sel = load_selection_fit_json(args.selection_fit)
    cal = json.load(open(args.n0_calibration))
    theta_ref = dict(M0hat=float(sel["M0hat"]),
                     sigma_M=float(sel["sigma_M"]),
                     delta=float(cal["delta"]), Om0=args.om0)

    # ------------------------------------------------------------- guards
    # PR-8 splits two numbers that were one through PR-7: ``z_depth`` is the
    # FITTED depth (where the counts are, where PLAN §4.3 pins amp = 1) and
    # ``z_node_hi`` is how far the radial inducing nodes reach.  Defaulting the
    # second to the first reproduces every pre-PR-8 build exactly.
    z_node_hi = float(args.z_depth if args.z_node_hi is None else args.z_node_hi)
    if z_node_hi < args.z_depth:
        raise SystemExit(
            f"--z-node-hi {z_node_hi} is below --z-depth {args.z_depth}: the "
            "nodes would not span the region the counts constrain, so part of "
            "the fitted field would be an extrapolation of its own basis.")
    if z_node_hi > args.z_depth and args.amp_hi is None:
        raise SystemExit(
            f"--z-node-hi {z_node_hi} reaches above --z-depth {args.z_depth} "
            "but no --amp-hi was given. Nodes above the depth are only "
            "meaningful with an ASSUMED amplitude (PR-8): without one the "
            "seam still zeroes the field above the depth, and the extra nodes "
            "would cost rank and change the fitted basis for nothing. Pass "
            "--amp-hi (0.0 is the legacy convention, stated explicitly).")
    if args.amp_hi is not None and args.amp_hi > 0.0 \
            and z_node_hi <= args.z_depth:
        raise SystemExit(
            f"--amp-hi {args.amp_hi} assumes a field ABOVE z_depth "
            f"{args.z_depth}, but the nodes stop at {z_node_hi}. The basis rows "
            "decay to zero within ~2 ls_z of the last node, so the assumed "
            "amplitude would be multiplied into a field that is already zero "
            "and the scan would measure nothing. Raise --z-node-hi (and --m-z "
            "with it).")
    d_sph = np.sqrt(4 * np.pi / args.m_sph)
    if d_sph > args.ls_sph:
        raise SystemExit(
            f"resolution guard (HARD in latent mode): sphere node spacing "
            f"{d_sph:.3f} > ls_sph {args.ls_sph}; raise --m-sph.")
    # The spacing guard is over the NODE range, which is what determines it --
    # identical to the pre-PR-8 expression whenever the two coincide, and the
    # binding constraint on --m-z once they do not (it is also the consumer's
    # copy in ``likelihood/factory._latent_guard_resolution``, so an anchor that
    # passed here cannot fail there).
    dz_node = np.log1p(z_node_hi) / (args.m_z - 1)
    if dz_node > args.ls_z:
        raise SystemExit(
            f"resolution guard: zeta node spacing {dz_node:.4f} > ls_z "
            f"{args.ls_z}; raise --m-z.")

    # ---------------------------------------------------------------- data
    with h5py.File(args.survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
        tracer_labels = (f[args.tracer_labels][...]
                         if args.tracer_labels is not None else None)
    occ = np.where(ng > 0)[0]
    edges = np.linspace(0.0, args.z_depth, args.n_shells + 1)
    counts = counts_from_catalog(zg, ng, occ, edges)
    gal_ok = counts.sum(0) >= 1e4
    pix_ok = (counts > 0).sum(0) >= 500
    if not (gal_ok & pix_ok).all():
        raise SystemExit(
            f"occupancy guard 7: shells "
            f"{np.where(~(gal_ok & pix_ok))[0].tolist()} under-occupied; "
            f"reduce --n-shells.")

    # ---------------------------------------------------------- K >= 2 split
    # ``counts`` above is the PARENT catalog's, and it stays the parent's: at
    # K >= 2 it is what the disjointness budget is checked against and what the
    # occupancy guard was already evaluated on.  The per-tracer arrays below
    # partition it.
    n_tracer = 1
    counts_k = [counts]
    if tracer_labels is not None:
        lab = np.asarray(tracer_labels)
        if lab.shape != zg.shape:
            raise SystemExit(
                f"--tracer-labels dataset {args.tracer_labels!r} has shape "
                f"{lab.shape}, expected zgals' {zg.shape}: the label is "
                f"PER GALAXY, which is what makes the partition disjoint by "
                f"construction rather than by assertion (OWNER DECISION 9).")
        valid = np.arange(lab.shape[1])[None, :] < np.asarray(ng)[:, None]
        if not valid.any():
            raise SystemExit("--tracer-labels: the catalog is empty.")
        lab_v = lab[valid].astype(int)
        if lab_v.min() < 0:
            raise SystemExit(
                f"--tracer-labels holds a negative label ({lab_v.min()}); "
                f"labels are tracer INDICES 0..K-1.")
        n_tracer = int(lab_v.max()) + 1
        counts_k = counts_from_catalog_by_tracer(
            zg, ng, occ, edges, lab, n_tracer)
        for k, ck in enumerate(counts_k):
            g_ok = ck.sum(0) >= 1e4
            p_ok = (ck > 0).sum(0) >= 500
            if not (g_ok & p_ok).all():
                raise SystemExit(
                    f"occupancy guard 7 on tracer {k}: shells "
                    f"{np.where(~(g_ok & p_ok))[0].tolist()} under-occupied. "
                    f"The guard is PER TRACER because the count channel is: "
                    f"each tracer contributes its own shell multinomial, and a "
                    f"shell that is empty for tracer k contributes nothing to "
                    f"xi from tracer k however well the parent fills it.")
    # float32 ROUND-TRIP, deliberately.  ``likelihood/catalog_views.py`` stores
    # the run's ``f_p_rows`` / ``field_f_p_occ`` as float32, and those are what
    # both integrators consume; forming ``(A, B, F_F)`` from the float64 map
    # would normalize eq. (4) against a completeness the likelihood never sees.
    # Measured on the shipped anchor: the mismatch leaves the budget identity
    # closing at 1.6e-9 (b_GW = 0.37) to 3.25e-8 (b_GW = 3.77), growing with
    # b_GW as e^{b f} predicts, against 8.9e-16 when the two agree.  Exactly the
    # defect PR-5 fixed on the OTHER eq. (2) input (moments from the f64 draws
    # rather than the stored f32 row factors) -- same class, one input over, and
    # small enough to sit inside factory's own 1e-6 f_p guard, so nothing
    # refuses such a run.  PLAN 4.2 claims eq. (4) is exact; this is what makes
    # that true rather than true-to-1e-8.
    def _load_fp(path):
        return np.maximum(
            load_selection_fraction(path, nside).f_p[occ],
            1e-3).astype(np.float32).astype(np.float64)

    f_p = _load_fp(args.per_pixel_completeness)
    print(f"[data] {occ.size} pixels, {int(counts.sum())} galaxies, "
          f"{args.n_shells} shells", flush=True)

    # Per-tracer depth maps and biases.  The FOOTPRINT rows are shared (``occ``,
    # the parent's occupied pixels) while ``f_p`` and ``b_k`` are per tracer:
    # the tracers live on one sky and one basis -- that is the shared-realization
    # property -- but each carries its own selection.  A pixel where tracer k has
    # no galaxies is still a legitimate row of tracer k's multinomial (it
    # predicts a small pi there), so restricting each tracer to its own occupied
    # subset would DISCARD information rather than avoid an error.
    #
    # ``sel_k`` / ``sigz_k`` / ``theta_k`` are the SELECTION half of the same
    # statement, and they were the half PR-7 left shared.  Tracer k's counts
    # are a multinomial over shells whose within-shell weighting is
    # ``C(z; m_lim_k, M0hat_k, sigma_M_k)`` convolved with tracer k's own
    # photo-z kernel -- that is what the shell response ``W_k`` is -- so a
    # single ``W`` gives every tracer catalog 0's magnitude selection while its
    # ``f_p`` and its ``b_k`` are its own.  Nothing about the field is per
    # tracer here: there is still one ``xi``, one basis and one set of shell
    # edges (PLAN §4.4).  ``W_k`` is per tracer because SELECTION is.
    fp_k = [f_p]
    b_k = [float(args.b_gal)]
    names_k = ["t0"]
    sel_k = [sel]
    sigz_k = [float(SIGMA_Z_DEFAULT)]
    theta_k = [theta_ref]
    if n_tracer > 1:
        cpaths = list(args.tracer_completeness or [])
        if not cpaths:
            cpaths = [args.per_pixel_completeness] * n_tracer
        if len(cpaths) != n_tracer:
            raise SystemExit(
                f"--tracer-completeness given {len(cpaths)} times for "
                f"{n_tracer} tracers (labels 0..{n_tracer - 1}).")
        bs = list(args.tracer_b_gal or [])
        if not bs:
            bs = [float(args.b_gal)] * n_tracer
        if len(bs) != n_tracer:
            raise SystemExit(
                f"--tracer-b-gal given {len(bs)} times for {n_tracer} "
                f"tracers.")
        names_k = ([s.strip() for s in args.tracer_names.split(",")]
                   if args.tracer_names else
                   [f"t{k}" for k in range(n_tracer)])
        if len(names_k) != n_tracer:
            raise SystemExit(
                f"--tracer-names has {len(names_k)} entries for {n_tracer} "
                f"tracers.")
        spaths = list(args.tracer_selection_fit or [])
        if not spaths:
            spaths = [args.selection_fit] * n_tracer
        if len(spaths) != n_tracer:
            raise SystemExit(
                f"--tracer-selection-fit given {len(spaths)} times for "
                f"{n_tracer} tracers.")
        sigs = list(args.tracer_sigma_z or [])
        if not sigs:
            sigs = [float(SIGMA_Z_DEFAULT)] * n_tracer
        if len(sigs) != n_tracer:
            raise SystemExit(
                f"--tracer-sigma-z given {len(sigs)} times for {n_tracer} "
                f"tracers.")
        fp_k = [_load_fp(p) for p in cpaths]
        b_k = [float(x) for x in bs]
        sel_k = [load_selection_fit_json(p) for p in spaths]
        sigz_k = [float(s) for s in sigs]
        # delta and Om0 are SHARED: one evolution index and one cosmology
        # describe the field the K catalogs select against.  M0hat and sigma_M
        # are per catalog, because they came out of that catalog's own fit.
        theta_k = [dict(M0hat=float(s["M0hat"]), sigma_M=float(s["sigma_M"]),
                        delta=float(cal["delta"]), Om0=args.om0)
                   for s in sel_k]
        print(f"[tracers] K = {n_tracer}: "
              + ", ".join(f"{nm}(N={int(c.sum())}, b={b:.3f}, "
                          f"<f_p>={float(fp.mean()):.3f})"
                          for nm, c, b, fp in zip(names_k, counts_k, b_k, fp_k)),
              flush=True)
        print("[tracers] selection: "
              + ", ".join(
                  f"{nm}(m_lim={float(s['m_lim']):.3f}, "
                  f"M0hat={float(s['M0hat']):.4f}, "
                  f"sigma_M={float(s['sigma_M']):.4f}, sigma_z={sz:g})"
                  for nm, s, sz in zip(names_k, sel_k, sigz_k)), flush=True)

    # ---------------------------------------------------------------- basis
    z_fine = np.linspace(1e-4, args.z_depth, 400)
    # The CONSUMPTION rows.  Without an amp(z) profile they stop at the fitted
    # depth, which is the whole grid the seam ever evaluates (``Q == 1`` above,
    # PLAN §4.2).  With one, they run to the top of the node range instead: the
    # moments ``(A, B)`` are what ``rho`` normalizes the budget against, and a
    # field that is nonzero above the depth needs its normalizer there too, or
    # the seam would inject an un-normalized monopole over 99.99% of the
    # missing budget.  ``z_sub`` stays a PREFIX of ``zgrid`` either way, which
    # is what the loader's padding assumes.
    z_top = float(args.z_depth if args.amp_hi is None else z_node_hi)
    z_sub = np.asarray(zgrid[zgrid <= z_top])
    amp_kw = ({} if args.amp_hi is None else
              dict(amp_hi=float(args.amp_hi), amp_kind=str(args.amp_kind),
                   amp_z_depth=float(args.z_depth), amp_Om0=float(args.om0)))
    vec = np.column_stack(hp.pix2vec(nside, occ))
    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=args.m_sph,
        n_inducing_z=args.m_z, z_node_hi=z_node_hi,
        ls_sph=args.ls_sph, ls_z=args.ls_z, zeta_fine=np.log1p(z_fine),
        **amp_kw)
    if amp_kw:
        print(f"[amp] PR-8 SENSITIVITY-SCAN anchor: amp(z <= {args.z_depth}) "
              f"= 1 (PLAN 4.3), amp(z > {args.z_depth}) = {args.amp_hi} "
              f"[{args.amp_kind}]; nodes to z = {z_node_hi}, "
              f"{z_sub.size} consumption rows. Nothing in this pipeline "
              f"constrains amp above the depth -- a posterior from this "
              f"anchor is a statement about that number.", flush=True)

    def _dvdz_shape(z, Om0):
        E = np.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
        dc = np.concatenate([[0.0], np.cumsum(
            0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))])
        return dc ** 2 / E

    def _base_fn_for(m_lim_k, kcorr_k):
        """``base(z; theta)`` closed over ONE catalog's magnitude selection.

        At K = 1 the closure variables are ``--selection-fit``'s own ``m_lim``
        and k-corrections, so the expression evaluated is the pre-PR-7 one,
        operation for operation.
        """
        def base_fn(theta):
            def _f(z):
                Cz = np.asarray(c_sel_gaussian(
                    jnp.asarray(z), m_lim_k, theta["M0hat"], theta["sigma_M"],
                    args.h0, theta["Om0"], k_corr_coeffs=kcorr_k))
                return (Cz * (1 + z) ** theta["delta"]
                        * _dvdz_shape(z, theta["Om0"]) + 1e-300)
            return _f
        return base_fn

    def _sigma_z_for(width):
        return lambda z: float(width) * np.ones_like(z)

    base_fn_k = [_base_fn_for(float(s["m_lim"]), tuple(s["k_corr_coeffs"]))
                 for s in sel_k]
    sigma_z_k = [_sigma_z_for(s) for s in sigz_k]

    # ``tracers`` is the K-element list even at K = 1, but ``op_at`` returns a
    # BARE CountOperator there and a MultiTracerCountOperator only at K >= 2.
    # That is not cosmetic tidiness: everything downstream (count_map_solve,
    # gradient, the objective) dispatches on the type and evaluates the
    # single-tracer expression tree for a bare operator, so the K = 1 build is
    # bit-identical to the pre-PR-7 builder rather than merely equal to it.
    tracers = [TracerCounts(pix=occ, counts=c, completeness=fp, bias=b,
                            label=nm)
               for c, fp, b, nm in zip(counts_k, fp_k, b_k, names_k)]
    if n_tracer > 1:
        disj = check_disjoint_tracers(tracers, parent_counts=counts)
        print(f"[tracers] disjointness verified={disj['verified']}, "
              f"max over-count {disj['max_excess']:.6g} galaxies in any "
              f"(pixel, shell) cell", flush=True)

    def op_at(thetas):
        """``(operator, [W_0..W_{K-1}])`` at a LIST of K per-catalog thetas.

        One shell response per tracer, because the within-shell weighting is
        tracer k's own magnitude selection and its own photo-z kernel.
        ``make_multi_count_operator`` has always taken "one response or a
        length-K sequence"; PR-7 built the sequence-free case and handed the
        same ``W`` to all K, which silently gave tracers 1..K-1 tracer 0's
        selection.  At K = 1 the list has one entry and the bare
        ``CountOperator`` branch below evaluates the pre-PR-7 expression tree.
        """
        Ws = [shell_response(edges, z_fine, sigma_z_k[j], base_fn_k[j](th))
              for j, th in enumerate(thetas)]
        if n_tracer == 1:
            return make_count_operator(basis.phi_sph, basis.phi_z_fine, Ws[0],
                                       tracers[0]), Ws
        return make_multi_count_operator(basis.phi_sph, basis.phi_z_fine, Ws,
                                         tracers), Ws

    print("[solve] anchor ...", flush=True)
    op, W_k_ref = op_at(theta_k)
    sol = count_map_solve(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    print(f"[solve] grad_inf={float(sol['grad_inf']):.2e}", flush=True)
    if float(sol["grad_inf"]) > 1e-8:
        raise SystemExit("P6 gate: anchor solve did not converge to 1e-8.")

    # --------------------------------------------------------- sensitivity
    # K >= 2 ADDS COLUMNS and nothing else (PLAN §3.4).  ``sensitivity`` is
    # linear in the stacked block and never inspects it, so no line of it
    # changes; what changes is the BLOCK handed to it.
    #
    # PLAN §3.4, quoted in ``sensitivity``'s own docstring: "catalog k's own
    # theta contributes its own column built from its own operator".  ``M0hat``
    # and ``sigma_M`` came out of catalog k's OWN selection fit and enter only
    # catalog k's ``W_k``, so at K >= 2 they are K columns each, and bumping
    # ``M0hat[k]`` rebuilds tracer k's response alone.  ``delta`` and ``Om0``
    # stay ONE column apiece: they describe the field's evolution and the
    # cosmology, which the K catalogs share, so bumping them rebuilds all K
    # responses at once and the stacked gradient's difference is their true
    # total derivative.  At K = 1 the column list is exactly
    # ``[M0hat, sigma_M, delta, Om0]`` in exactly that order, and the bump is
    # the same arithmetic on the same single theta dict.
    per_cat_theta = ("M0hat", "sigma_M")
    shared_theta = ("delta", "Om0")
    steps = dict(M0hat=1e-3, sigma_M=1e-3, delta=1e-2, Om0=1e-3)
    if n_tracer == 1:
        cols = ([(nme, 0) for nme in per_cat_theta]
                + [(nme, None) for nme in shared_theta])
        theta_labels = list(per_cat_theta) + list(shared_theta)
    else:
        cols = ([(nme, kk) for nme in per_cat_theta
                 for kk in range(n_tracer)]
                + [(nme, None) for nme in shared_theta])
        theta_labels = ([f"{nme}[{names_k[kk]}]" for nme in per_cat_theta
                         for kk in range(n_tracer)] + list(shared_theta))

    def _bump(nme, which, h):
        ts = [dict(t) for t in theta_k]
        if which is None:
            for t in ts:
                t[nme] += h
        else:
            ts[which][nme] += h
        return ts

    n_b_cols = 1 if n_tracer == 1 else n_tracer
    dgrad = np.zeros((op.rank, len(cols) + n_b_cols))
    for j, (nme, which) in enumerate(cols):
        h = steps[nme]
        dgrad[:, j] = (np.asarray(_grad_of(xi_hat,
                                           op_at(_bump(nme, which, +h))[0]))
                       - np.asarray(_grad_of(xi_hat,
                                             op_at(_bump(nme, which, -h))[0]))) \
            / (2 * h)
    if n_tracer == 1:
        dgrad[:, -1] = np.asarray(dgrad_db(xi_hat, op))
        labels = theta_labels + ["b_gal"]
    else:
        dgrad[:, -n_tracer:] = np.asarray(dgrad_db_by_tracer(xi_hat, op))
        labels = theta_labels + [f"b_gal[{nm}]" for nm in names_k]
    S = np.asarray(sensitivity(xi_hat, L, jnp.asarray(dgrad)))

    # ------------------------------------------------ b_gal dispersion (S-2)
    # PLAN §3.4: b_gal is FIXED in the solve and its uncertainty is carried by
    # a rank-1 inflation of the DRAW covariance,
    #     Cov(xi) = H^{-1} + s_b^2 v v^T,   v = d xi_hat/d b,
    # with s_b the profile curvature [-d^2 log p_count/db^2]^{-1/2} at the
    # anchor ([v4, §0.5 finding 11]: NOT a 20% prior width -- §4.3 pins
    # amp = 1, so the counts measure b_gal, and a free dial would let Tier-B's
    # "latent-on CI >= table CI" gate be chosen rather than measured).
    #
    # ``v`` is the b_gal column of ``S`` that the loop above ALREADY built --
    # the same IFT construction against the same H_chol -- so it is read out
    # here, not recomputed.  Through PR-6a this column was built, stored, and
    # never consumed: every member ensemble shipped so far was drawn from
    # H^{-1} alone and is under-dispersed by exactly the term below (closure
    # finding S-2).
    #
    # At K >= 2 the scalar ``s_b`` becomes a (K, K) COVARIANCE of the biases and
    # the rank-1 inflation becomes rank K: Cov(xi) = H^-1 + V C_b V^T with
    # V = S[:, b-block].  The rank-K form is forced, not stylistic: the
    # off-diagonal of C_b is (dgrad/db_j)^T H^-1 (dgrad/db_k), so it exists ONLY
    # because the tracers share one field, and inflating each b_k independently
    # would draw members from the DECOUPLED model while the counts were fitted
    # with the shared one -- the exact mismatch PLAN §0.5 finding 12 is about.
    #
    # **The K >= 2 uncertainty is a POSTERIOR under a stated log-bias prior, and
    # that is a different object from K = 1's floor.  Saying so is the point.**
    # The count channel is invariant under xi -> xi/s, b -> s b for one common
    # s (only the ridge sees it, and it decreases), so the bias AMPLITUDE is not
    # identified while the RATIO is -- measured on this operator, the log-bias
    # curvature at the anchor has eigenvalues of both signs, the negative one
    # along the amplitude.  A "profile covariance" quoted there would be a
    # matrix inverse of an indefinite matrix, i.e. a number with the right units
    # and no meaning.  So the amplitude's width comes from an explicit
    # log-normal prior of width --bias-prior-log-sd, defaulting to the SAME 5%
    # that B_GAL_SYSTEMATIC_FLOOR_FRAC states at K = 1 and for the same reason
    # ("5% is the order at which linear bias is quotable at all"); the RATIO's
    # width still comes from the counts, and Tier E measures that it is
    # essentially insensitive to the prior over a factor 16 in width
    # (experiments/field_level_plan/pr7).  At K = 1 the floor only ever RAISES
    # s_b; here the prior supplies a width where the counts supply none.  The
    # prior is centred on the anchor's own b_k, so it says "b_k is known to 5%
    # around the value this artifact was built at", never "b_k is 1".
    fl_frac = (B_GAL_SYSTEMATIC_FLOOR_FRAC if args.s_b_floor_frac is None
               else float(args.s_b_floor_frac))
    cov_b = None
    bias_prior_sd = None
    if n_tracer == 1:
        prof = b_gal_profile_sigma(xi_hat, op, dgrad_b=dgrad[:, -1],
                                   v_b=S[:, -1],
                                   systematic_floor_frac=fl_frac)
        s_b = float(prof["s_b"])
    else:
        bias_prior_sd = float(fl_frac if args.bias_prior_log_sd is None
                              else args.bias_prior_log_sd)
        b_arr = jnp.asarray(b_k, dtype=jnp.float64)
        prior = (np.log(np.asarray(b_k, dtype=float)), bias_prior_sd)
        g_b = bias_profile_grad(xi_hat, op, log_b_prior=prior)
        H_b = bias_profile_hessian(xi_hat, op, H_chol=L, log_b_prior=prior)
        H_u = np.asarray(bias_profile_curvature_log(g_b, H_b, b_arr))
        evals = np.linalg.eigvalsh(H_u)
        if evals.min() <= 0.0:
            raise SystemExit(
                f"K = {n_tracer} bias covariance is not positive definite: the "
                f"log-bias curvature at the anchor has eigenvalues "
                f"{evals.tolist()} even with a prior of width "
                f"{bias_prior_sd:g} nat. The count channel does not identify "
                f"the bias amplitude (only the ratio), so the prior is what "
                f"makes the amplitude proper; tighten --bias-prior-log-sd, or "
                f"pass --no-b-gal-dispersion to ship an anchor whose members "
                f"carry no bias dispersion at all (and which stamps that).")
        cov_u = np.linalg.inv(H_u)
        cov_b = np.diag(np.asarray(b_k)) @ cov_u @ np.diag(np.asarray(b_k))
        sd_u = np.sqrt(np.diag(cov_u))
        corr = float(cov_u[0, 1] / (sd_u[0] * sd_u[1]))
        prof = dict(s_b=float(np.max(np.sqrt(np.diag(cov_b)))),
                    s_b_stat=float(np.max(np.sqrt(np.diag(cov_b)))),
                    s_b_floor=float(bias_prior_sd),
                    floor_active=False,
                    curvature_profile=float(evals.min()),
                    curvature_conditional=float(evals.max()))
        s_b = float(prof["s_b"])
        print(f"[b_gal] K = {n_tracer}: log-bias prior sd {bias_prior_sd:g}; "
              f"posterior sd in log b {sd_u.tolist()}, corr(0,1) {corr:.6f} "
              f"(the off-diagonal IS the shared field); log-ratio sd "
              f"{float(np.sqrt(cov_u[0, 0] + cov_u[1, 1] - 2 * cov_u[0, 1])):.6e}",
              flush=True)
    inflate = not args.no_b_gal_dispersion
    # The spread the inflation adds, reported TWO ways because one number
    # alone misreads it.  (i) On the whole member norm: ||L^{-T} g|| has
    # E||.||^2 = tr H^{-1} over all M modes, and the rank-1 term adds
    # s_b^2 ||v||^2 to that trace -- a small factor by construction, since one
    # direction is being added to M of them.  (ii) ALONG v, which is where the
    # term actually lives: Var(v.xi) goes from v^T H^{-1} v to
    # v^T H^{-1} v + s_b^2 (v.v)^2.  That is the honest size of S-2, and it is
    # the direction the count channel's amplitude response lives in.
    Hinv = np.linalg.inv(np.asarray(L) @ np.asarray(L).T)
    v_b = S[:, -1]
    tr_hinv = float(np.trace(Hinv))
    if n_tracer == 1:
        rank1_tr = float(s_b ** 2 * np.dot(v_b, v_b))
    else:
        V_blk = S[:, -n_tracer:]
        rank1_tr = float(np.trace(V_blk @ cov_b @ V_blk.T))
    infl_factor = float(np.sqrt((tr_hinv + rank1_tr) / tr_hinv))
    var_v = float(v_b @ Hinv @ v_b)
    infl_v = float(np.sqrt((var_v + s_b ** 2 * float(v_b @ v_b) ** 2) / var_v))
    print(f"[b_gal] s_b = {s_b:.6e} "
          f"(profile {prof['s_b_stat']:.6e}, floor {prof['s_b_floor']:.6e}, "
          f"{'FLOOR' if prof['floor_active'] else 'PROFILE'} binding); "
          f"curvature profile {prof['curvature_profile']:.6e} vs conditional "
          f"{prof['curvature_conditional']:.6e}", flush=True)
    print(f"[b_gal] rank-1 inflation {'ON' if inflate else 'OFF'}: "
          f"tr H^-1 = {tr_hinv:.6e}, s_b^2||v||^2 = {rank1_tr:.6e}, "
          f"member-spread factor "
          f"{'' if inflate else 'WOULD BE '}{infl_factor:.6f} overall, "
          f"{infl_v:.4f} ALONG v", flush=True)

    # --------------------------------------------------------- members
    # ``g_members`` and ``Xi_members`` are DIFFERENT objects; the builder used
    # to write the members under both names, so anything trusting the header's
    # "antithetic standard-normal draws" and skipping the ``- xi_hat`` step
    # silently picked up ``a.xi_hat`` (found by the PR-5b prediction: eight
    # spurious same-sign offsets with a spread 7x too small).
    if n_tracer == 1:
        draws, g_members, eps_members = laplace_draws(
            xi_hat, L, args.m_draw, jax.random.PRNGKey(args.seed),
            return_g=True, return_eps=True,
            s_b=(s_b if inflate else None),
            v_b=(S[:, -1] if inflate else None))
    elif inflate:
        draws, g_members, eps_members = laplace_draws_multitracer(
            xi_hat, L, args.m_draw, jax.random.PRNGKey(args.seed),
            cov_b=cov_b, V_b=S[:, -n_tracer:], return_g=True, return_eps=True)
    else:
        draws, g_members, eps_members = laplace_draws(
            xi_hat, L, args.m_draw, jax.random.PRNGKey(args.seed),
            return_g=True, return_eps=True)
    Xi_members = np.asarray(draws).reshape(args.m_draw, args.m_sph, args.m_z)
    row_fac = np.stack([
        np.asarray(row_factor(basis, d)).astype(np.float32) for d in draws])

    # --------------------------------------------------------- sky moments
    # Chebyshev nodes on [0, b_max] for the b_GW interpolation grid (P9).
    k = np.arange(args.n_b_nodes)
    b_nodes = 0.5 * args.b_max * (1 - np.cos(np.pi * k / (args.n_b_nodes - 1)))
    # Build the moments from the STORED f32 row factors, not from the f64
    # draws: eq. (4)'s budget identity is exact only if the moments and the
    # seam evaluate the same field, and the seam consumes ``row_fac``.  With
    # the f64 draws the identity closes to 2.7e-7 relative at the production
    # corner instead of 2e-15 (measured, field-level PR-5).
    A, B = sky_moments(basis, np.asarray(draws), b_nodes, f_p,
                       row_fac=row_fac)
    A, B = np.asarray(A), np.asarray(B)
    P_F, F_F = sky_constant_coeffs(f_p)

    # PER-TRACER moment tables (PLAN §2.2 / PR-7).  Eq. (2)'s two reductions
    # depend on the tracer through its footprint and its f_p, so K tracers need
    # K tables even though they share one xi and one b_GW grid; this is also
    # what makes the per-catalog log Z_k of completion.field_global_log_Z --
    # which CANCELS at K = 1 and does not at K >= 2 -- closed-form online in the
    # scalar c = C(z; theta). At K = 1 the block below is skipped entirely and
    # the artifact is byte-identical.
    A_k = B_k = P_k = F_k = None
    if n_tracer > 1:
        A_k, B_k, P_k, F_k = sky_moments_by_tracer(
            basis, np.asarray(draws), b_nodes, fp_k,
            row_fac_by_tracer=[row_fac] * n_tracer)
        A_k, B_k = np.asarray(A_k), np.asarray(B_k)

    # projected theta-derivatives of the moments (PR-6b/P18; PLAN §1.7):
    # dA_m/dtheta_j = b sum_p e^{b f_m} f'_{m,j}(p, z), f' = Phi rows . S_j
    # computed per member via the same row-factor contraction.
    n_th = S.shape[1]
    dA = np.zeros(A.shape + (n_th,), dtype=np.float32)
    dB = np.zeros_like(dA)
    phi_z_out = np.asarray(basis.phi_z_out)
    proj = np.asarray(basis.proj_sph)
    fprime = np.stack([
        (proj @ S[:, j].reshape(args.m_sph, args.m_z)) @ phi_z_out.T
        for j in range(n_th)]).astype(np.float32)           # (n_th, n_fit, Nz)
    for m in range(args.m_draw):
        # Same f32 row factors the moments and the seam use (see above).
        f_m = row_fac[m].astype(np.float64) @ phi_z_out.T   # (n_fit, Nz)
        for i, b in enumerate(b_nodes):
            e = np.exp(b * f_m)
            dA[m, i] = b * np.einsum("pz,jpz->zj", e, fprime)
            dB[m, i] = b * np.einsum("p,pz,jpz->zj", f_p, e, fprime)

    # ---------------------------------------------------------------- write
    # ``s_b`` and the inflation switch go into the STAMPED digest: two anchors
    # that differ only in their draw covariance are different artifacts, and
    # the stamp is the only digest that can say so (guard 1's CONTENT digest
    # covers the field -- geometry, f_p, W, edges, counts, theta_ref, b_gal --
    # and deliberately not the draws, so it is unchanged by this feature; see
    # factory.latent_artifact_fingerprint).
    cfg = dict(vars(args), theta_ref=theta_ref, labels=labels,
               jitter=dict(basis.meta),
               s_b=(s_b if inflate else 0.0),
               b_gal_dispersion=bool(inflate))
    # The sha identifies the artifact CONTENT + configuration; the output
    # path is identity-irrelevant (the reproducibility gate compares two
    # same-seed builds written to different paths).
    cfg.pop("out", None)
    # PR-8's three switches are dropped from the stamp on a build that used
    # none of them, for the same reason PR-7 drops its tracer keys: a K = 1,
    # no-amp rebuild must reproduce the digest of the anchor it rebuilds, and
    # a key whose value is None changes the JSON while changing no array. When
    # a profile IS applied, all three stay in -- two anchors differing only in
    # the assumed amp(z) generate different Q over 99.99% of the missing budget
    # and must not share a digest (``basis.meta`` carries them into ``jitter``
    # as well, which is what guard 1's CONTENT digest sees).
    if args.amp_hi is None and args.z_node_hi is None:
        for _k in ("amp_hi", "amp_kind", "z_node_hi"):
            cfg.pop(_k, None)
    # PR-7's tracer switches, dropped on a K = 1 build for the reason the PR-8
    # block above gives: a K = 1 rebuild must reproduce the digest of the anchor
    # it rebuilds, byte for byte, and a key whose value is None changes the JSON
    # while changing no array in the file. That is also why the per-tracer
    # SELECTION switches are popped here rather than defaulted into the K = 1
    # stamp. At K >= 2 they all stay in (and ``n_tracer`` joins them), because
    # two anchors that partition the same catalog differently, or select
    # against it differently, are different artifacts.
    if n_tracer == 1:
        for _k in ("tracer_labels", "tracer_completeness", "tracer_b_gal",
                   "tracer_names", "bias_prior_log_sd",
                   "tracer_selection_fit", "tracer_sigma_z"):
            cfg.pop(_k, None)
    else:
        cfg["n_tracer"] = int(n_tracer)
        cfg["tracer_bias"] = [float(x) for x in b_k]
        cfg["bias_prior_log_sd"] = float(bias_prior_sd)
        cfg["tracer_theta_ref"] = theta_k
        cfg["tracer_sigma_z_used"] = [float(s) for s in sigz_k]
        cfg["tracer_m_lim"] = [float(s["m_lim"]) for s in sel_k]
    sha = hashlib.sha256()
    for arr in (np.asarray(xi_hat), np.asarray(L), S, counts, f_p,
                np.asarray(W_k_ref[0]), A, B, b_nodes, z_sub, edges):
        sha.update(np.ascontiguousarray(arr).tobytes())
    # The per-tracer arrays join the stamp only when they exist, so the K = 1
    # digest is the pre-PR-7 one.  Tracer 0's response is already in the tuple
    # above (it is the K = 1 ``W``); tracers 1.. join here, so two K >= 2
    # anchors whose catalogs differ only in their magnitude selection cannot
    # share a digest.
    if n_tracer > 1:
        for arr in (counts_k + fp_k + [A_k, B_k, P_k, F_k]
                    + [np.asarray(w) for w in W_k_ref[1:]]):
            sha.update(np.ascontiguousarray(np.asarray(arr)).tobytes())
    sha.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    digest = sha.hexdigest()

    out = Path(args.out)
    with h5py.File(out, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset("xi_hat", data=np.asarray(xi_hat))
        g.create_dataset("H_chol", data=np.asarray(L))
        g.create_dataset("sensitivity_S", data=S)
        g.attrs["sensitivity_labels"] = json.dumps(labels)
        g.create_dataset("g_members", data=np.asarray(g_members))
        # The rank-1 stream, stored for the same reason ``g_members`` is: a
        # reader who wants to rebuild the members must have BOTH sources, and
        # ``eps`` is a different object from ``g`` (one scalar per member, not
        # M).  With the inflation off it is exact zeros, which is itself the
        # record that the anchor carries no b_gal dispersion.
        g.create_dataset("eps_members", data=np.asarray(eps_members))
        g.create_dataset("Xi_members", data=Xi_members)
        g.create_dataset("row_fac", data=row_fac)
        g.create_dataset("A_moments", data=A)
        g.create_dataset("B_moments", data=B)
        g.create_dataset("dA_moments", data=dA)
        g.create_dataset("dB_moments", data=dB)
        g.create_dataset("b_nodes", data=b_nodes)
        g.create_dataset("z_sub", data=z_sub)
        g.create_dataset("fit_pixels", data=occ.astype(np.int32))
        g.create_dataset("completeness", data=f_p)
        g.create_dataset("counts", data=counts)
        g.create_dataset("z_count_edges", data=edges)
        # Tracer 0's response.  At K = 1 that is THE response; at K >= 2 the
        # other K-1 live in the tracer subgroups, because the within-shell
        # weighting is each catalog's own magnitude selection.
        g.create_dataset("shell_response", data=np.asarray(W_k_ref[0]))
        g.attrs["P_F"] = P_F
        g.attrs["F_F"] = F_F
        g.attrs["theta_ref"] = json.dumps(theta_ref)
        g.attrs["basis_meta"] = json.dumps(dict(basis.meta))
        g.attrs["nside"] = nside
        g.attrs["b_gal"] = args.b_gal
        # Which draw-covariance convention this anchor was built under (PLAN
        # §3.4).  Stated as a formula rather than a boolean alone so an
        # artifact read years from now says what it means without the CLI.
        g.attrs["draw_covariance"] = (
            "H^{-1} + s_b^2 v v^T, v = sensitivity_S[:, 'b_gal']"
            if inflate else "H^{-1}")
        g.attrs["b_gal_dispersion"] = bool(inflate)
        g.attrs["s_b"] = float(s_b if inflate else 0.0)
        g.attrs["s_b_profile"] = float(prof["s_b_stat"])
        g.attrs["s_b_floor"] = float(prof["s_b_floor"])
        g.attrs["s_b_floor_frac"] = float(fl_frac)
        g.attrs["s_b_floor_active"] = bool(prof["floor_active"])
        g.attrs["b_gal_curvature_profile"] = float(prof["curvature_profile"])
        g.attrs["b_gal_curvature_conditional"] = float(
            prof["curvature_conditional"])
        g.attrs["b_gal_spread_inflation"] = float(infl_factor if inflate
                                                  else 1.0)
        g.attrs["b_gal_spread_inflation_along_v"] = float(infl_v if inflate
                                                          else 1.0)
        g.attrs["seed"] = args.seed
        g.attrs["grad_inf"] = float(sol["grad_inf"])
        # ---- K >= 2 (PR-7).  A SUBGROUP, deliberately: everything above is
        # the shared object (one xi_hat, one H_chol, one member ensemble, one
        # sensitivity block) and everything below is per tracer.  The layout is
        # the shared-realization theorem written down -- there is one field in
        # this file and K catalogs' worth of selection against it, and no way to
        # express a per-tracer field even by accident. Absent entirely at K = 1,
        # so a K = 1 artifact is byte-identical to a pre-PR-7 one.
        if n_tracer > 1:
            g.attrs["n_tracer"] = int(n_tracer)
            g.attrs["tracer_names"] = json.dumps(names_k)
            g.attrs["tracer_bias"] = np.asarray(b_k, dtype=float)
            g.attrs["tracer_overlap_policy"] = (
                "disjoint (OWNER DECISION 9); the K count arrays are a "
                "partition of the parent catalog by per-galaxy label, so "
                "disjointness is structural rather than checked")
            g.attrs["bias_profile_cov"] = json.dumps(
                np.asarray(cov_b).tolist())
            g.attrs["bias_prior_log_sd"] = float(bias_prior_sd)
            g.attrs["bias_cov_convention"] = (
                "posterior in log b under a log-normal prior of sd "
                "bias_prior_log_sd centred on each tracer's own b_gal; the "
                "count channel identifies the RATIO b_j/b_k, not the "
                "amplitude (PLAN 4.3), so the prior is what makes the "
                "amplitude direction proper")
            g.attrs["draw_covariance"] = (
                "H^{-1} + V C_b V^T, V = sensitivity_S[:, b_gal block], "
                "C_b = bias_profile_cov" if inflate else "H^{-1}")
            g.attrs["tracer_theta_ref"] = json.dumps(theta_k)
            tg = g.create_group("tracers")
            for k, nm in enumerate(names_k):
                t = tg.create_group(str(k))
                t.attrs["name"] = nm
                t.attrs["b_gal"] = float(b_k[k])
                t.attrs["P_F"] = float(P_k[k])
                t.attrs["F_F"] = float(F_k[k])
                # This tracer's own selection: the fit it came from, the
                # magnitude limit and photo-z width that build its W_k, and the
                # theta its W_k was frozen at.  Without these the artifact
                # cannot say which catalog's selection weighted which counts.
                t.attrs["selection_fit"] = str(
                    (args.tracer_selection_fit or [args.selection_fit] *
                     n_tracer)[k])
                t.attrs["m_lim"] = float(sel_k[k]["m_lim"])
                t.attrs["sigma_z"] = float(sigz_k[k])
                t.attrs["theta_ref"] = json.dumps(theta_k[k])
                t.create_dataset("counts", data=counts_k[k])
                t.create_dataset("completeness", data=fp_k[k])
                t.create_dataset("A_moments", data=A_k[k])
                t.create_dataset("B_moments", data=B_k[k])
                t.create_dataset("shell_response",
                                 data=np.asarray(W_k_ref[k]))
        g.attrs["sha256"] = digest
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
    print(f"[artifact] sha256 = {digest}")
    print(f"[artifact] wrote {out}")


if __name__ == "__main__":
    main()
