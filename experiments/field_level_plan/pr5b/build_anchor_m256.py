"""Build the ``M_draw = 256`` anchor PR-5b measures on -- and the MAP twin.

PLAN §6.5 item 1 asks for "the ``ll_m`` vector at ``M_draw = 256``".  The
shipped anchor (``darksirens/cli/build_latent_field.py``, PR-4) is built at
``M_draw = 8``, and the number of members is baked into the artifact
(``row_fac``, ``(A, B)``, the draws), so the campaign needs its own build.

Why this is a script in ``experiments/`` and not ``--m-draw 256`` on the CLI
--------------------------------------------------------------------------
PR-5b ships no production code (PLAN §7: "the deliverable is a report plus
two pins"), so ``darksirens/cli/build_latent_field.py`` is not modified.
That matters here because the CLI's cost does NOT scale benignly in
``M_draw``: its projected theta-derivative block

    for m in range(m_draw):                      #  256
        for i, b in enumerate(b_nodes):          #   33
            e = np.exp(b * f_m)                  #  (30470, 147) = 4.5 M
            dA[m, i] = b * np.einsum("pz,jpz->zj", e, fprime)
            dB[m, i] = b * np.einsum("p,pz,jpz->zj", f_p, e, fprime)

is a PYTHON loop over (member x b-node): 264 passes at ``M_draw = 8``,
**8448** at 256, each an exp + two five-way einsums over the full 30470-row
footprint.  Measured at M=8 below and extrapolated in the log.

``dA``/``dB`` are PR-6b (theta-coupling) objects.  PR-5b does not read them:
the seam's per-proposal path is ``rho`` from ``(A, B)`` plus the ``row_fac``
gather (``completion.latent_member_logq_rows``), and ``moments_at`` /
``theta_shift`` -- the only consumers of ``dA``/``dB`` -- are the rung-1
entry points that K9 already demoted to an inertness flag (PLAN §0.5, "K9's
benign branch fires -- PR-6a is the deliverable").  So this build SKIPS them
and writes them with a ZERO-WIDTH theta axis, ``(M_draw, n_b, N_z_sub, 0)``.

That shape is deliberate and is the honest encoding of "not built":

* ``latent_q.load_latent_plan`` reads ``dA_moments``/``dB_moments``
  unconditionally and pads them to the full grid, so the datasets must
  EXIST; a zero-width axis satisfies that at zero bytes.
* writing zeros at the real width ``n_theta = 5`` would instead be a silent
  lie -- a PR-6b consumer would read a well-shaped table of zeros and
  compute ``dtheta``-corrections of exactly zero, i.e. it would see rung 1
  as inert when in fact rung 1 was never built.  A zero-width axis makes any
  such consumer fail on shape, immediately, which is the behaviour this
  codebase asks for everywhere else.
* it also saves 734 MB of device memory: the loader promotes ``dA``/``dB``
  to f64 and pads them to the FULL 1086-node grid, i.e.
  ``2 x 256 x 33 x 1086 x 5 x 8 B``, for arrays nothing on this path reads.

The attribute ``pr5b_dA_dB_skipped = True`` records it in the artifact.

Two artifacts are written
-------------------------
``latent_anchor_m256.h5``   the 256-member ensemble -- ``sigma``, ESS, P14.
``latent_anchor_map_m1.h5`` ONE "member" that is ``xi_hat`` itself (``g = 0``).
    This is not a draw; it is the P17 arm (b) reference point.  PLAN §6.5
    item 5 / §1.6 Limit III state the closed form as
    ``LSE_m ll_m - log M - ll(xi_hat) -> 0.5 a^T H^{-1} a = 0.5 sigma^2``,
    so ``ll(xi_hat)`` is a REQUIRED measurement, not a diagnostic, and the
    only way to get it through the shipped seam is an artifact whose single
    member is the MAP field.

Gates run here (all HARD)
-------------------------
G1  ``xi_hat``, ``H_chol``, ``counts``, ``f_p``, ``W`` are bit-identical to
    the M_draw=8 anchor.  The count-channel solve does not see ``m_draw``,
    so any difference would mean the two artifacts describe different
    posteriors and the closed-form prediction (computed on the M=8 anchor's
    ``H_chol``) would not be a prediction for this ensemble.
G2  P6 convergence, ``grad_inf <= 1e-8`` (the builder's own gate).
G3  ANTITHETIC (PLAN §6.5 item 3): ``laplace_draws`` builds
    ``g = [g_half, -g_half]``, so member ``k`` and member ``k + M/2`` are the
    pair -- NOT ``2k``/``2k+1``.  Verified exactly here, and it is the reason
    ``run_member_spread.py`` takes ANTITHETICALLY BALANCED prefixes rather
    than naive ones for the ``M in {4, ..., 128}`` sub-series.
G4  The M=256 build's first 8 members are NOT the M=8 anchor's 8 members
    (``jax.random.normal(key, (n_draw//2, M))`` is a different draw for a
    different shape).  Asserted, so the report cannot claim a nesting that
    does not exist.
G5  Occupancy guard 7 and both resolution guards, as in the CLI.

Run from ``experiments/desi_full259`` (its ``common.py`` pins ZMAX = 6.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import latent_harness as H
from latent_harness import jax, jnp, np

CLIGHT = 299792.458


def _sizes(args, n_sub, n_fit):
    return (f"row_fac {args.m_draw * n_fit * args.m_z * 4 / 1e6:.1f} MB, "
            f"(A,B) {2 * args.m_draw * args.n_b_nodes * n_sub * 8 / 1e6:.1f} MB")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m-draw", type=int, default=256)
    ap.add_argument("--moment-chunk", type=int, default=16,
                    help="members per sky_moments call; the vmap materializes "
                         "(chunk, n_fit, N_z_sub) f64 twice (36 MB/member at "
                         "production rank), so 256 at once is 18 GB")
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--ls-sph", type=float, default=0.2)
    ap.add_argument("--ls-z", type=float, default=0.039)
    ap.add_argument("--m-sph", type=int, default=315)
    ap.add_argument("--m-z", type=int, default=8)
    ap.add_argument("--b-gal", type=float, default=1.0)
    ap.add_argument("--n-b-nodes", type=int, default=33)
    ap.add_argument("--b-max", type=float, default=4.0)
    ap.add_argument("--h0", type=float, default=67.74)
    ap.add_argument("--om0", type=float, default=0.3089)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--time-dadb", action="store_true",
                    help="time ONE (member, b) dA/dB pass and extrapolate, so "
                         "the skip decision is measured rather than asserted")
    args = ap.parse_args(argv)

    import h5py
    import healpy as hp

    from darksirens.catalogs.depth_map import load_selection_fraction
    from darksirens.redshift.grid import zgrid
    from darksirens.redshift.latent_counts import (
        TracerCounts, count_map_solve, counts_from_catalog, dgrad_db,
        gradient, laplace_draws, make_count_operator, sensitivity)
    from darksirens.redshift.latent_field import (
        build_latent_basis, row_factor, shell_response, sky_constant_coeffs,
        sky_moments)
    from darksirens.redshift.selection import (
        c_sel_gaussian, load_selection_fit_json)

    C = H.C
    survey_path = str(C.SURVEY_N64)
    sel = load_selection_fit_json(str(C.FIT_JSON))
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    theta_ref = dict(M0hat=float(sel["M0hat"]), sigma_M=float(sel["sigma_M"]),
                     delta=float(cal["delta"]), Om0=args.om0)
    m_lim = float(sel["m_lim"])
    kcorr = tuple(sel["k_corr_coeffs"])
    depth_map = str(C.INGEST_DATA / "mth_map_nside128.h5")

    # ------------------------------------------------------------ G5 guards
    d_sph = np.sqrt(4 * np.pi / args.m_sph)
    if d_sph > args.ls_sph:
        raise SystemExit(f"resolution guard: {d_sph:.3f} > {args.ls_sph}")
    dz_node = np.log1p(args.z_depth) / (args.m_z - 1)
    if dz_node > args.ls_z:
        raise SystemExit(f"resolution guard: {dz_node:.4f} > {args.ls_z}")

    with h5py.File(survey_path) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
    occ = np.where(ng > 0)[0]
    edges = np.linspace(0.0, args.z_depth, args.n_shells + 1)
    counts = counts_from_catalog(zg, ng, occ, edges)
    gal_ok = counts.sum(0) >= 1e4
    pix_ok = (counts > 0).sum(0) >= 500
    if not (gal_ok & pix_ok).all():
        raise SystemExit("occupancy guard 7")
    f_p = np.maximum(load_selection_fraction(depth_map, nside).f_p[occ], 1e-3)
    print(f"[data] {occ.size} pixels, {int(counts.sum())} galaxies", flush=True)

    # ------------------------------------------------------------- basis
    z_fine = np.linspace(1e-4, args.z_depth, 400)
    z_sub = np.asarray(zgrid[zgrid <= args.z_depth])
    vec = np.column_stack(hp.pix2vec(nside, occ))
    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=args.m_sph,
        n_inducing_z=args.m_z, z_node_hi=args.z_depth,
        ls_sph=args.ls_sph, ls_z=args.ls_z, zeta_fine=np.log1p(z_fine))
    n_sub, n_fit = z_sub.size, occ.size
    print(f"[basis] N_z_sub={n_sub} n_fit={n_fit}  " + _sizes(args, n_sub, n_fit),
          flush=True)

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

    sigma_z = lambda z: 0.023 * np.ones_like(z)  # noqa: E731

    def op_at(theta):
        W = shell_response(edges, z_fine, sigma_z, base_fn(theta))
        tracer = TracerCounts(pix=occ, counts=counts, completeness=f_p,
                              bias=args.b_gal)
        return make_count_operator(basis.phi_sph, basis.phi_z_fine, W,
                                   tracer), W

    t0 = time.time()
    op, W_ref = op_at(theta_ref)
    sol = count_map_solve(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    grad_inf = float(sol["grad_inf"])
    print(f"[solve] grad_inf={grad_inf:.3e}  ({time.time() - t0:.1f} s)",
          flush=True)
    if grad_inf > 1e-8:                                              # G2
        raise SystemExit("P6 gate: anchor solve did not converge to 1e-8.")

    # ------------------------------------------------------- G1: same solve
    ref_path = H.resolve_anchor_m8()
    with h5py.File(ref_path) as f:
        g = f["latent_field"]
        ref = {k: g[k][...] for k in ("xi_hat", "H_chol", "counts",
                                      "completeness", "shell_response")}
        ref_sha = str(g.attrs["sha256"])
    g1 = {
        "xi_hat": np.abs(np.asarray(xi_hat) - ref["xi_hat"]).max(),
        "H_chol": np.abs(np.asarray(L) - ref["H_chol"]).max(),
        "counts": np.abs(counts - ref["counts"]).max(),
        "f_p": np.abs(f_p - ref["completeness"]).max(),
        "W": np.abs(np.asarray(W_ref) - ref["shell_response"]).max(),
    }
    print(f"[G1] vs {ref_path.name} (sha {ref_sha[:16]}): "
          + " ".join(f"{k}={v:.3e}" for k, v in g1.items()), flush=True)
    if max(g1.values()) != 0.0:
        raise SystemExit(
            "G1 FAILED: the M=256 build's count-channel solve is not "
            "bit-identical to the M=8 anchor's, so the two artifacts do not "
            f"describe the same posterior: {g1}")

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

    k = np.arange(args.n_b_nodes)
    b_nodes = 0.5 * args.b_max * (1 - np.cos(np.pi * k / (args.n_b_nodes - 1)))
    P_F, F_F = sky_constant_coeffs(f_p)

    # ------------------------------------------------------------ the members
    def _members(draws, tag):
        """``row_fac`` + chunked ``(A, B)`` for a stack of fields."""
        t = time.time()
        row_fac = np.stack([
            np.asarray(row_factor(basis, d)).astype(np.float32) for d in draws])
        print(f"[{tag}] row_fac {row_fac.shape} ({time.time() - t:.1f} s)",
              flush=True)
        t = time.time()
        As, Bs = [], []
        for lo in range(0, row_fac.shape[0], args.moment_chunk):
            rf = row_fac[lo:lo + args.moment_chunk]
            A_c, B_c = sky_moments(basis, np.asarray(draws)[lo:lo + rf.shape[0]],
                                   b_nodes, f_p, row_fac=rf)
            As.append(np.asarray(A_c)); Bs.append(np.asarray(B_c))
            print(f"[{tag}] moments {lo + rf.shape[0]}/{row_fac.shape[0]} "
                  f"({time.time() - t:.1f} s)", flush=True)
        return row_fac, np.concatenate(As), np.concatenate(Bs)

    draws = laplace_draws(xi_hat, L, args.m_draw, jax.random.PRNGKey(args.seed))
    draws = np.asarray(draws)

    # ------------------------------------------------------------- G3/G4
    half = args.m_draw // 2
    g_all = np.asarray(jnp.concatenate([
        jax.random.normal(jax.random.PRNGKey(args.seed), (half, op.rank)),
        -jax.random.normal(jax.random.PRNGKey(args.seed), (half, op.rank))]))
    anti_g = np.abs(g_all[:half] + g_all[half:]).max()
    anti_xi = np.abs(draws[:half] + draws[half:]
                     - 2.0 * np.asarray(xi_hat)[None, :]).max()
    print(f"[G3] antithetic: max|g_k + g_(k+M/2)| = {anti_g:.3e}, "
          f"max|xi_k + xi_(k+M/2) - 2 xi_hat| = {anti_xi:.3e}", flush=True)
    if anti_g != 0.0 or anti_xi > 1e-9:
        raise SystemExit("G3 FAILED: laplace_draws is not antithetic at "
                         f"M_draw={args.m_draw}")
    with h5py.File(ref_path) as f:
        ref_draws = f["latent_field"]["Xi_members"][...].reshape(
            -1, args.m_sph * args.m_z)
    nest = np.abs(draws[:ref_draws.shape[0]] - ref_draws).max()
    print(f"[G4] max|xi_m(256)[:8] - xi_m(8)| = {nest:.3e}  "
          "(EXPECTED nonzero: jax.random.normal(key, (M/2, rank)) is a "
          "different draw for a different shape, so the 256-member set is "
          "NOT a superset of the 8-member one)", flush=True)
    if nest == 0.0:
        raise SystemExit("G4: the two draw sets are identical -- re-derive "
                         "the CRN claim before quoting anything")

    row_fac, A, B = _members(draws, "m256")

    # ------------------------------------------- dA/dB: measure, then skip
    dadb_note = "skipped (zero-width theta axis); PR-6b object, unread here"
    if args.time_dadb:
        phi_z_out = np.asarray(basis.phi_z_out)
        proj = np.asarray(basis.proj_sph)
        fprime = np.stack([
            (proj @ S[:, j].reshape(args.m_sph, args.m_z)) @ phi_z_out.T
            for j in range(S.shape[1])]).astype(np.float32)
        f_m = row_fac[0].astype(np.float64) @ phi_z_out.T
        t = time.time()
        for i in range(3):
            e = np.exp(b_nodes[i] * f_m)
            np.einsum("pz,jpz->zj", e, fprime)
            np.einsum("p,pz,jpz->zj", f_p, e, fprime)
        per = (time.time() - t) / 3
        dadb_note = (f"skipped; measured {per * 1e3:.0f} ms per (member, b) "
                     f"pass -> {per * 8 * args.n_b_nodes:.0f} s at M_draw=8, "
                     f"{per * args.m_draw * args.n_b_nodes / 60:.1f} min at "
                     f"M_draw={args.m_draw}")
        print(f"[dA/dB] {dadb_note}", flush=True)

    dA = np.zeros(A.shape + (0,), dtype=np.float32)
    dB = np.zeros_like(dA)

    # ---------------------------------------------------------------- write
    def _write(out, *, m_draw, g_members, Xi_members, row_fac, A, B, dA, dB,
               variant):
        cfg = dict(vars(args), theta_ref=theta_ref, labels=labels,
                   jitter=dict(basis.meta), variant=variant, m_draw=m_draw,
                   survey=survey_path, selection_fit=str(C.FIT_JSON),
                   n0_calibration=str(C.DATA_DIR / "n0_calibration.json"),
                   per_pixel_completeness=depth_map)
        sha = hashlib.sha256()
        for arr in (np.asarray(xi_hat), np.asarray(L), S, counts, f_p,
                    np.asarray(W_ref), A, B, b_nodes, z_sub, edges):
            sha.update(np.ascontiguousarray(arr).tobytes())
        sha.update(json.dumps(cfg, sort_keys=True, default=str).encode())
        digest = sha.hexdigest()
        with h5py.File(out, "w") as f:
            g = f.create_group("latent_field")
            g.create_dataset("xi_hat", data=np.asarray(xi_hat))
            g.create_dataset("H_chol", data=np.asarray(L))
            g.create_dataset("sensitivity_S", data=S)
            g.attrs["sensitivity_labels"] = json.dumps(labels)
            # NOTE the PR-5b finding recorded in PREDICTION.md: the CLI writes
            # ``create_dataset("g_members", data=draws)``, i.e. the dataset
            # named g_members holds the MEMBERS xi_m, not the standard normals
            # its header advertises.  This build writes the actual g.
            g.create_dataset("g_members", data=g_members)
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
            g.attrs["seed"] = args.seed
            g.attrs["grad_inf"] = grad_inf
            g.attrs["sha256"] = digest
            g.attrs["format_version"] = "darksirens-latent-field-1.0"
            g.attrs["pr5b_variant"] = variant
            g.attrs["pr5b_dA_dB_skipped"] = True
            g.attrs["pr5b_dA_dB_note"] = dadb_note
            g.attrs["pr5b_reference_anchor"] = str(ref_path)
            g.attrs["pr5b_reference_sha256"] = ref_sha
            g.attrs["pr5b_g_members_are_normals"] = True
        print(f"[artifact] {out}  sha256={digest}  "
              f"{out.stat().st_size / 1e6:.1f} MB", flush=True)
        return digest

    sha256 = _write(
        H.ANCHOR_M256, m_draw=args.m_draw, g_members=g_all,
        Xi_members=draws.reshape(args.m_draw, args.m_sph, args.m_z),
        row_fac=row_fac, A=A, B=B, dA=dA, dB=dB, variant="pr5b-m256")

    # ------------------------------------------------- the MAP twin (P17 b)
    map_field = np.asarray(xi_hat)[None, :]
    rf_map, A_map, B_map = _members(map_field, "map")
    sha_map = _write(
        H.ANCHOR_MAP, m_draw=1, g_members=np.zeros((1, op.rank)),
        Xi_members=map_field.reshape(1, args.m_sph, args.m_z),
        row_fac=rf_map, A=A_map, B=B_map,
        dA=np.zeros(A_map.shape + (0,), np.float32),
        dB=np.zeros(A_map.shape + (0,), np.float32), variant="pr5b-map-m1")

    json.dump({
        "m_draw": args.m_draw, "sha256_m256": sha256, "sha256_map": sha_map,
        "reference_anchor": str(ref_path), "reference_sha256": ref_sha,
        "G1_max_dev": {k: float(v) for k, v in g1.items()},
        "G2_grad_inf": grad_inf,
        "G3_antithetic_g": float(anti_g), "G3_antithetic_xi": float(anti_xi),
        "G4_nesting_dev": float(nest),
        "n_fit": int(n_fit), "n_z_sub": int(n_sub), "rank": int(op.rank),
        "P_F": float(P_F), "F_F": float(F_F),
        "dA_dB": dadb_note,
    }, open(H.PR5B_DIR / "anchor_m256_build.json", "w"), indent=1)
    print("[build] ALL GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
