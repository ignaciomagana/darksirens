"""PR-3 promotion gates P7 / P7b / P7c / P7d / P7e on the real catalog (K9).

Builds the guard-compliant count channel on the DESI union (occupied
nside-64 pixels, shells over [0, z_depth], OD3 kernel 315 x 8), solves the
anchor, then re-solves at 20 theta drawn from the prior with the
theta-dependent within-shell base ``base(z; theta) =
C_sel(z; theta) (1+z)^delta dVc/dz(Om0)`` inside ``W(theta)``, and reports:

  tau  (P7)   max_theta ||xi_theta - xi_ref||_H / sqrt(M)     (diagnostic)
  P7b         max_theta ||xi_theta - xi_ref - S dtheta||_H / sqrt(M)
  P7c         osc_theta [ a . (xi_theta - xi_ref) ]  nats     (GW-side GATE)
  P7d         osc_theta 0.5 |logdet H(theta) - logdet H_ref|  nats
  P7e         osc_theta [ Laplace evidence(theta) ]           (galaxy side)
              + its restriction to the (delta, theta_sel) directions
              (guard 5's rung-1 overlap bound)

K9: P7c >= 0.1 nat AND P7b irreducible above 0.1 -> refuse promotion (ship
PR-6a); P7c < 0.1 -> the GW-side field shift is negligible and PR-6b is a
no-op on that channel.  ``a`` (eq. 6) comes from PR-0's computation
(pr0/a_vector.npz — the same kernel, so basis-compatible by construction).

Occupancy guard 7: every shell must carry >= 1e4 galaxies and >= 500
occupied pixels; shells failing are dropped LOUDLY.

Run from experiments/desi_full259 (ZMAX pin):
    python ../field_level_plan/pr3/run_promotion_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PR3_DIR = Path(__file__).resolve().parent
FULL259 = PR3_DIR.parent.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.latent_counts import (  # noqa: E402
    TracerCounts,
    count_map_solve,
    counts_from_catalog,
    laplace_evidence,
    make_count_operator,
    sensitivity,
    shell_multinomial_logl,
)
from darksirens.redshift.latent_field import (  # noqa: E402
    build_latent_basis,
    shell_response,
)
from darksirens.redshift.selection import (  # noqa: E402
    c_sel_gaussian,
    load_selection_fit_json,
)

ANCHOR_H0 = 67.74
LS_SPH, LS_Z, M_SPH, M_Z = 0.2, 0.039, 315, 8
G_S = 12
N_THETA = 20
SIGMA_Z_POP = 0.023          # population-median photo-z scatter (PLAN §1.4)
CLIGHT = 299792.458
SEED = 22


def _dvdz_shape(z, Om0):
    E = np.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))])
    return dc ** 2 / E                      # H0-free shape


def main():
    sel = load_selection_fit_json(C.FIT_JSON)
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    theta_ref = dict(M0hat=float(sel["M0hat"]), sigma_M=float(sel["sigma_M"]),
                     delta=float(cal["delta"]), Om0=C.OM0)
    m_lim = float(sel["m_lim"])
    kcorr = tuple(sel["k_corr_coeffs"])

    # ---------------------------------------------------------------- data
    with h5py.File(C.SURVEY_N64) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
    occ = np.where(ng > 0)[0]
    edges = np.linspace(0.0, C.Z_DEPTH, G_S + 1)
    counts = counts_from_catalog(zg, ng, occ, edges)
    # occupancy guard 7
    gal_ok = counts.sum(0) >= 1e4
    pix_ok = (counts > 0).sum(0) >= 500
    keep = gal_ok & pix_ok
    if not keep.all():
        print(f"[guard7] dropping shells {np.where(~keep)[0].tolist()} "
              f"(galaxies {counts.sum(0)[~keep].astype(int).tolist()}, "
              f"occupied {(counts > 0).sum(0)[~keep].tolist()})", flush=True)
    counts = counts[:, keep]
    edges_kept = np.append(edges[:-1][keep], edges[1:][keep][-1])
    print(f"[data] {occ.size} occupied pixels, "
          f"{int(counts.sum())} galaxies in {keep.sum()} shells", flush=True)

    # f_p from PR-2's loader (real depth map), aligned with occ rows
    from darksirens.catalogs.depth_map import load_selection_fraction
    sfm = load_selection_fraction(
        PR3_DIR.parent.parent / "desi_ingest" / "data" / "mth_map_nside128.h5",
        64)
    f_p = np.maximum(sfm.f_p[occ], 1e-3)   # floor: occupied rows with f=0

    # ---------------------------------------------------------------- basis
    z_fine = np.linspace(1e-4, C.Z_DEPTH, 400)
    vec = np.column_stack(hp.pix2vec(64, occ))
    basis = build_latent_basis(
        vec, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=C.Z_DEPTH, ls_sph=LS_SPH, ls_z=LS_Z,
        zeta_fine=np.log1p(z_fine))

    def base_fn(theta):
        def _f(z):
            Cz = np.asarray(c_sel_gaussian(
                jnp.asarray(z), m_lim, theta["M0hat"], theta["sigma_M"],
                ANCHOR_H0, theta["Om0"], k_corr_coeffs=kcorr))
            return (Cz * (1 + z) ** theta["delta"]
                    * _dvdz_shape(z, theta["Om0"]) + 1e-300)
        return _f

    # basis rows were built AT the occupied pixels, so proj rows = all rows
    def op_at(theta):
        W = shell_response(edges_kept, z_fine,
                           lambda z: SIGMA_Z_POP * np.ones_like(z),
                           base_fn(theta))
        tracer = TracerCounts(pix=occ, counts=counts, completeness=f_p,
                              bias=1.0)
        return make_count_operator(basis.phi_sph, basis.phi_z_fine, W,
                                   tracer)

    print("[solve] anchor ...", flush=True)
    op_ref = op_at(theta_ref)
    sol = count_map_solve(op_ref)
    xi_ref = sol["xi_hat"]
    L_ref = sol["H_chol"]
    logdet_ref = 2.0 * float(jnp.sum(jnp.log(jnp.diagonal(L_ref))))
    ev_ref = float(laplace_evidence(op_ref, xi_ref, L_ref))
    print(f"[solve] anchor grad_inf={float(sol['grad_inf']):.2e} "
          f"||xi_hat||/sqrt(M)={float(jnp.linalg.norm(xi_ref)) / np.sqrt(op_ref.rank):.4f}",
          flush=True)

    # ------------------------------------------------- S columns (jacfwd FD)
    # d grad / d theta at (xi_ref, theta_ref) by central differences of the
    # analytic gradient through W(theta) — 2 x n_theta cheap rebuilds.
    from darksirens.redshift.latent_counts import gradient
    names = ["M0hat", "sigma_M", "delta", "Om0"]
    steps = dict(M0hat=1e-3, sigma_M=1e-3, delta=1e-2, Om0=1e-3)
    dgrad = np.zeros((op_ref.rank, len(names)))
    for j, nme in enumerate(names):
        tp = dict(theta_ref); tp[nme] += steps[nme]
        tm = dict(theta_ref); tm[nme] -= steps[nme]
        gp = np.asarray(gradient(xi_ref, op_at(tp)))
        gm = np.asarray(gradient(xi_ref, op_at(tm)))
        dgrad[:, j] = (gp - gm) / (2 * steps[nme])
    S = np.asarray(sensitivity(xi_ref, L_ref, jnp.asarray(dgrad)))

    # ---------------------------------------------------------------- a
    a_npz = PR3_DIR.parent / "pr0" / "a_vector.npz"
    a_vec = np.load(a_npz)["a_mat"].reshape(-1)
    print(f"[a] loaded eq.(6) a from {a_npz}: ||a|| = "
          f"{np.linalg.norm(a_vec):.4e}", flush=True)

    # ---------------------------------------------------------------- 20 theta
    rng = np.random.default_rng(SEED)
    prior_sd = dict(M0hat=0.05, sigma_M=0.05, delta=0.10, Om0=0.01)
    Hn = np.asarray(L_ref @ L_ref.T)
    Ln = np.asarray(L_ref)
    rows = []
    for t in range(N_THETA):
        th = {k: theta_ref[k] + prior_sd[k] * rng.standard_normal()
              for k in names}
        op_t = op_at(th)
        sol_t = count_map_solve(op_t, xi0=xi_ref)
        xi_t = np.asarray(sol_t["xi_hat"])
        d = xi_t - np.asarray(xi_ref)
        dth = np.array([th[k] - theta_ref[k] for k in names])
        lin = S @ dth
        r = d - lin
        Hnorm = lambda v: float(np.linalg.norm(Ln.T @ v))
        logdet_t = 2.0 * float(np.sum(np.log(np.diag(
            np.asarray(sol_t["H_chol"])))))
        rows.append(dict(
            theta=th,
            grad_inf=float(sol_t["grad_inf"]),
            tau=Hnorm(d) / np.sqrt(op_ref.rank),
            p7b=Hnorm(r) / np.sqrt(op_ref.rank),
            p7c=float(a_vec @ d),
            p7d=0.5 * abs(logdet_t - logdet_ref),
            evidence=float(laplace_evidence(
                op_t, sol_t["xi_hat"], sol_t["H_chol"])) - ev_ref,
        ))
        print(f"  theta {t + 1}/{N_THETA}: tau={rows[-1]['tau']:.3e} "
              f"p7b={rows[-1]['p7b']:.3e} p7c={rows[-1]['p7c']:+.4e} "
              f"p7d={rows[-1]['p7d']:.3e} dEv={rows[-1]['evidence']:+.3f}",
              flush=True)

    p7c_vals = np.array([r["p7c"] for r in rows])
    ev_vals = np.array([r["evidence"] for r in rows])
    out = dict(
        theta_ref=theta_ref, n_theta=N_THETA, kernel=dict(
            ls_sph=LS_SPH, ls_z=LS_Z, M_sph=M_SPH, M_z=M_Z, G_s=int(keep.sum())),
        anchor=dict(grad_inf=float(sol["grad_inf"]),
                    xi_norm_per_mode=float(
                        jnp.linalg.norm(xi_ref)) / np.sqrt(op_ref.rank),
                    logdet=logdet_ref, evidence=ev_ref),
        sensitivity_labels=names,
        tau_max=max(r["tau"] for r in rows),
        p7b_max=max(r["p7b"] for r in rows),
        p7c_osc=float(p7c_vals.max() - p7c_vals.min()),
        p7d_max=max(r["p7d"] for r in rows),
        p7e_osc=float(ev_vals.max() - ev_vals.min()),
        rows=rows, git_sha=C.git_sha(),
    )
    with open(PR3_DIR / "promotion_gates.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[P7 ] tau_max        = {out['tau_max']:.4e}  (diagnostic)")
    print(f"[P7b] max residual   = {out['p7b_max']:.4e}  (< 0.1 per-mode sd)")
    print(f"[P7c] osc a.dxi      = {out['p7c_osc']:.4e} nat  (GATE: < 0.1 -> rung 0 == rung 1 GW-side)")
    print(f"[P7d] max logdet     = {out['p7d_max']:.4e} nat  (< 0.1)")
    print(f"[P7e] osc evidence   = {out['p7e_osc']:.4e} nat  (galaxy side)")
    print(f"[pr3] wrote {PR3_DIR / 'promotion_gates.json'}")


if __name__ == "__main__":
    main()
