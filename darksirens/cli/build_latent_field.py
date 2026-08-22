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
from pathlib import Path

import numpy as np


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
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--ls-sph", type=float, default=0.2)
    ap.add_argument("--ls-z", type=float, default=0.039,
                    help="zeta units (guard 2: Mpc is unrepresentable here)")
    ap.add_argument("--m-sph", type=int, default=315)
    ap.add_argument("--m-z", type=int, default=8)
    ap.add_argument("--b-gal", type=float, default=1.0)
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
        B_GAL_SYSTEMATIC_FLOOR_FRAC, TracerCounts, b_gal_profile_sigma,
        count_map_solve, counts_from_catalog, dgrad_db, gradient,
        laplace_draws, make_count_operator, sensitivity)
    from darksirens.redshift.latent_field import (
        build_latent_basis, row_factor, shell_response, sky_constant_coeffs,
        sky_moments)
    from darksirens.redshift.selection import (
        c_sel_gaussian, load_selection_fit_json)

    sel = load_selection_fit_json(args.selection_fit)
    cal = json.load(open(args.n0_calibration))
    theta_ref = dict(M0hat=float(sel["M0hat"]),
                     sigma_M=float(sel["sigma_M"]),
                     delta=float(cal["delta"]), Om0=args.om0)
    m_lim = float(sel["m_lim"])
    kcorr = tuple(sel["k_corr_coeffs"])

    # ------------------------------------------------------------- guards
    d_sph = np.sqrt(4 * np.pi / args.m_sph)
    if d_sph > args.ls_sph:
        raise SystemExit(
            f"resolution guard (HARD in latent mode): sphere node spacing "
            f"{d_sph:.3f} > ls_sph {args.ls_sph}; raise --m-sph.")
    dz_node = np.log1p(args.z_depth) / (args.m_z - 1)
    if dz_node > args.ls_z:
        raise SystemExit(
            f"resolution guard: zeta node spacing {dz_node:.4f} > ls_z "
            f"{args.ls_z}; raise --m-z.")

    # ---------------------------------------------------------------- data
    with h5py.File(args.survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
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
    f_p = np.maximum(
        load_selection_fraction(args.per_pixel_completeness, nside).f_p[occ],
        1e-3).astype(np.float32).astype(np.float64)
    print(f"[data] {occ.size} pixels, {int(counts.sum())} galaxies, "
          f"{args.n_shells} shells", flush=True)

    # ---------------------------------------------------------------- basis
    z_fine = np.linspace(1e-4, args.z_depth, 400)
    z_sub = np.asarray(zgrid[zgrid <= args.z_depth])
    vec = np.column_stack(hp.pix2vec(nside, occ))
    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=args.m_sph,
        n_inducing_z=args.m_z, z_node_hi=args.z_depth,
        ls_sph=args.ls_sph, ls_z=args.ls_z, zeta_fine=np.log1p(z_fine))

    def _dvdz_shape(z, Om0):
        E = np.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
        dc = np.concatenate([[0.0], np.cumsum(
            0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))])
        return dc ** 2 / E

    def base_fn(theta):
        def _f(z):
            Cz = np.asarray(c_sel_gaussian(
                jnp.asarray(z), m_lim, theta["M0hat"], theta["sigma_M"],
                args.h0, theta["Om0"], k_corr_coeffs=kcorr))
            return (Cz * (1 + z) ** theta["delta"]
                    * _dvdz_shape(z, theta["Om0"]) + 1e-300)
        return _f

    sigma_z = lambda z: 0.023 * np.ones_like(z)

    def op_at(theta):
        W = shell_response(edges, z_fine, sigma_z, base_fn(theta))
        tracer = TracerCounts(pix=occ, counts=counts, completeness=f_p,
                              bias=args.b_gal)
        return make_count_operator(basis.phi_sph, basis.phi_z_fine, W,
                                   tracer), W

    print("[solve] anchor ...", flush=True)
    op, W_ref = op_at(theta_ref)
    sol = count_map_solve(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    print(f"[solve] grad_inf={float(sol['grad_inf']):.2e}", flush=True)
    if float(sol["grad_inf"]) > 1e-8:
        raise SystemExit("P6 gate: anchor solve did not converge to 1e-8.")

    # --------------------------------------------------------- sensitivity
    names = ["M0hat", "sigma_M", "delta", "Om0"]
    steps = dict(M0hat=1e-3, sigma_M=1e-3, delta=1e-2, Om0=1e-3)
    dgrad = np.zeros((op.rank, len(names) + 1))
    for j, nme in enumerate(names):
        tp = dict(theta_ref); tp[nme] += steps[nme]
        tm = dict(theta_ref); tm[nme] -= steps[nme]
        dgrad[:, j] = (np.asarray(gradient(xi_hat, op_at(tp)[0]))
                       - np.asarray(gradient(xi_hat, op_at(tm)[0]))) \
            / (2 * steps[nme])
    dgrad[:, -1] = np.asarray(dgrad_db(xi_hat, op))
    S = np.asarray(sensitivity(xi_hat, L, jnp.asarray(dgrad)))
    labels = names + ["b_gal"]

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
    fl_frac = (B_GAL_SYSTEMATIC_FLOOR_FRAC if args.s_b_floor_frac is None
               else float(args.s_b_floor_frac))
    prof = b_gal_profile_sigma(xi_hat, op, dgrad_b=dgrad[:, -1],
                               v_b=S[:, -1], systematic_floor_frac=fl_frac)
    s_b = float(prof["s_b"])
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
    rank1_tr = float(s_b ** 2 * np.dot(v_b, v_b))
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
    draws, g_members, eps_members = laplace_draws(
        xi_hat, L, args.m_draw, jax.random.PRNGKey(args.seed), return_g=True,
        return_eps=True,
        s_b=(s_b if inflate else None), v_b=(S[:, -1] if inflate else None))
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
    sha = hashlib.sha256()
    for arr in (np.asarray(xi_hat), np.asarray(L), S, counts, f_p,
                np.asarray(W_ref), A, B, b_nodes, z_sub, edges):
        sha.update(np.ascontiguousarray(arr).tobytes())
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
        g.create_dataset("shell_response", data=np.asarray(W_ref))
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
        g.attrs["sha256"] = digest
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
    print(f"[artifact] sha256 = {digest}")
    print(f"[artifact] wrote {out}")


if __name__ == "__main__":
    main()
