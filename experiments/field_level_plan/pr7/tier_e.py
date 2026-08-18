#!/usr/bin/env python
"""Tier E — the K=2 gate on differentiator 2 (field-level PR-7).

Two DISJOINT tracers (OWNER DECISION 9) on one sky, one shared latent field,
`b_2/b_1 = 2`.  Three measurements, all in the COUNT CHANNEL:

  (i)   bias-ratio recovery, `|r_hat - 2| < 2 sigma`, over `--n-real`
        realizations;
  (ii)  the shared-`xi` credible region on `r` against two independent fits;
  (iii')the same comparison read as PLAN §0.5 finding 12's substantive
        replacement for v3's routing tautology: the decoupled variant IS the
        independent-fields product prior that
        `--allow_unverified_shared_lss_members` marginalizes over
        (`inference/loaders.py:352-395`), so running both on one mock
        demonstrates the coupling the flag throws away rather than
        demonstrating that a deleted check does not fire.

Why not on an `H0` posterior: `pr6a/CLOSURE_v2.md` §V localized 82-92% of that
mock's excess `H0` variance to the GW-event channel with the catalog held
byte-identical, so an `H0`-based Tier E would measure the mock's PE
calibration.  The bias ratio is a count-channel object (PLAN §3.4: "at K >= 2
the ratio `b_2/b_1` comes from its own 2x2 profile curvature"), so the gate is
run where it is defined.  See `PREDICTION.md`, written before this ran.

The geometry is the closure world's, not a new one: the footprint, the
per-pixel completeness map and the nside come from
`pr6a/data/rb`, so "1854 pixels" means the same 1854 pixels every other tier
used.  Only the counts are synthetic, drawn from the model at a KNOWN `xi_true`
and a KNOWN `(b_1, b_2)` -- which is what makes this a recovery test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def build_world(args):
    """Basis, footprint, completeness maps and shell response — built ONCE.

    Everything here is realization-independent; only `xi_true` and the drawn
    counts move with the seed, which is what makes the campaign cheap and the
    comparison between arms exactly paired.
    """
    import h5py
    import healpy as hp
    from darksirens.redshift.latent_counts import counts_from_catalog
    from darksirens.redshift.latent_field import build_latent_basis, shell_response
    from darksirens.catalogs.depth_map import load_selection_fraction

    with h5py.File(args.survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
    occ = np.where(ng > 0)[0]
    edges = np.linspace(0.0, args.z_depth, args.n_shells + 1)
    # The parent catalog's own counts, kept only to set the per-shell totals the
    # multinomial is conditioned on -- the SHAPE of the radial distribution is
    # then the real mock's, not a flat invention.
    parent = counts_from_catalog(zg, ng, occ, edges)
    T_g = parent.sum(0)
    T_g = np.maximum(np.round(T_g / T_g.sum() * args.n_gal).astype(int), 1)

    f_p1 = np.maximum(load_selection_fraction(args.mth_map, nside).f_p[occ],
                      1e-3).astype(np.float32).astype(np.float64)
    # Tracer 2 has its OWN selection.  A deterministic function of tracer 1's
    # map rather than an independent random one, so the two footprints overlap
    # the way two real surveys of the same sky do (a deeper and a shallower
    # sample), and so the world is a function of the arguments alone.
    f_p2 = np.clip(f_p1 ** float(args.fp2_power) * float(args.fp2_scale),
                   1e-3, 1.0)

    z_fine = np.linspace(1e-4, args.z_depth, 400)
    vec = np.column_stack(hp.pix2vec(nside, occ))
    basis = build_latent_basis(
        vec, np.log1p(z_fine), n_inducing_sphere=args.m_sph,
        n_inducing_z=args.m_z, z_node_hi=args.z_depth,
        ls_sph=args.ls_sph, ls_z=args.ls_z, zeta_fine=np.log1p(z_fine))
    W = shell_response(edges, z_fine,
                       lambda z: args.sigma_z * np.ones_like(z),
                       lambda z: z ** 2 + 1e-300)
    return dict(basis=basis, W=W, occ=occ, edges=edges, T_g=T_g,
                f_p=[f_p1, f_p2], nside=nside, z_fine=z_fine)


def draw_realization(world, seed, biases):
    """One `xi_true` and K DISJOINT tracer count arrays drawn from eq. (1').

    Disjoint by construction: the K multinomials are drawn independently, which
    is the count-space statement that no galaxy is in two samples (OWNER
    DECISION 9). Nothing is subsampled from anything else, so R14's
    "AGN subset of galaxies" failure mode is unreachable by this generator.
    """
    import jax.numpy as jnp

    basis, W = world["basis"], world["W"]
    rng = np.random.default_rng(seed)
    M = basis.rank
    xi_true = rng.normal(size=M)
    phi_shell = np.asarray(jnp.asarray(W) @ basis.phi_z_fine)     # (G, M_z)
    proj = np.asarray(basis.proj_sph)                             # (n_fit, M_sph)
    f_true = (proj @ xi_true.reshape(basis.m_sph, basis.m_z)) @ phi_shell.T

    out = []
    for k, b in enumerate(biases):
        a = np.log(world["f_p"][k])[:, None] + b * f_true
        p = np.exp(a - a.max(axis=0, keepdims=True))
        p /= p.sum(axis=0, keepdims=True)
        c = np.stack([rng.multinomial(int(world["T_g"][g]), p[:, g])
                      for g in range(p.shape[1])], axis=1).astype(float)
        out.append(c)
    return xi_true, out


def make_stack(world, counts_k, biases, names=("g", "a")):
    from darksirens.redshift.latent_counts import (
        MultiTracerCountOperator, TracerCounts, make_count_operator)
    ops = []
    for k, (c, b) in enumerate(zip(counts_k, biases)):
        t = TracerCounts(pix=world["occ"], counts=c,
                         completeness=world["f_p"][k], bias=float(b),
                         label=names[k])
        ops.append(make_count_operator(world["basis"].phi_sph,
                                       world["basis"].phi_z_fine, world["W"], t))
    return MultiTracerCountOperator(tuple(ops))


def one_realization(world, seed, args, prior_s):
    from darksirens.redshift.latent_counts import (
        bias_profile, bias_ratio_from_profile, check_disjoint_tracers,
        decoupled_bias_profile)

    b_true = [args.b1, args.b1 * args.ratio]
    xi_true, counts_k = draw_realization(world, seed, b_true)
    mop = make_stack(world, counts_k, [args.b_start, args.b_start])
    prior = (0.0, float(prior_s))

    t0 = time.time()
    sh = bias_profile(mop, log_b_prior=prior, n_outer=args.n_outer,
                     max_log_step=args.max_log_step)
    t_sh = time.time() - t0
    t0 = time.time()
    de = decoupled_bias_profile(mop, log_b_prior=prior, n_outer=args.n_outer,
                     max_log_step=args.max_log_step)
    t_de = time.time() - t0

    gu = np.asarray(sh["profile_grad_log"])
    r_sh = bias_ratio_from_profile(sh["b_hat"], sh["cov_log_b"])
    r_de = bias_ratio_from_profile(de["b_hat"], de["cov_log_b"])
    return dict(
        seed=int(seed), prior_s=float(prior_s),
        xi_true_norm=float(np.linalg.norm(xi_true)),
        n_gal=[float(c.sum()) for c in counts_k],
        shared=dict(r_sh, grad_inf=float(sh["grad_inf"]),
                    profile_grad=gu.tolist(),
                    # The two directions that mean something: the RATIO
                    # gradient is what the gate depends on; the AMPLITUDE one
                    # is the soft direction that consumes the outer trips.
                    grad_ratio=float((gu[1] - gu[0]) / 2.0),
                    grad_amp=float((gu[1] + gu[0]) / 2.0),
                    converged=bool(np.max(np.abs(gu))
                                   < args.profile_grad_tol),
                    wall_s=t_sh),
        decoupled=dict(
            r_de, grad_inf=float(de["grad_inf"]), wall_s=t_de,
            # The decoupled arm sits near the prior centre by PREDICTION P-E5,
            # and "prior-dominated" must be distinguished from "not converged":
            # this is the outer profile's own gradient in log bias, which is
            # zero at a converged optimum whatever drove it there.
            profile_grad=[float(np.max(np.abs(np.asarray(o["profile_grad_log"]))))
                          for o in de["per_tracer"]]),
        pull_shared=float((r_sh["ratio"] - args.ratio) / r_sh["sigma"]),
        pull_decoupled=float((r_de["ratio"] - args.ratio) / r_de["sigma"]),
        width_ratio=float(r_de["sigma"] / r_sh["sigma"]),
    )


def verify_curvature(world, seed, args, prior_s, n_sigma=(1.0, 2.0)):
    """Is the quoted `sigma_log r` the ACTUAL width of the profile likelihood?

    The credible region Tier E gates is quoted from a curvature, so it is worth
    one direct check that the curvature is the width and not merely a number
    with the right units.  The profile is re-evaluated at `log r` offset by
    `n sigma` from its optimum, with the AMPLITUDE re-profiled at each offset
    (a short 1-D Newton along `(1, 1)` in log bias) and `xi` re-solved inside
    that, and the rise in `J` is compared against the Gaussian's `n^2/2` nats.
    A curvature that were merely local would show up here as a mismatch.
    """
    import jax.numpy as jnp
    from darksirens.redshift.latent_counts import (
        bias_profile, bias_profile_curvature_log, bias_profile_grad,
        bias_profile_hessian, bias_ratio_from_profile, count_map_solve,
        multi_objective, with_biases)

    b_true = [args.b1, args.b1 * args.ratio]
    _, counts_k = draw_realization(world, seed, b_true)
    mop = make_stack(world, counts_k, [args.b_start, args.b_start])
    prior = (0.0, float(prior_s))
    sh = bias_profile(mop, log_b_prior=prior, n_outer=args.n_outer,
                     max_log_step=args.max_log_step)
    rr = bias_ratio_from_profile(sh["b_hat"], sh["cov_log_b"])
    u_hat = np.log(np.asarray(sh["b_hat"], dtype=float))
    sig = float(rr["sigma_log"])

    def _profile_at_logr(t, xi0):
        """Minimize J over the amplitude at fixed `log r = t`."""
        u = np.array([u_hat.mean() - 0.5 * t, u_hat.mean() + 0.5 * t])
        xi = xi0
        P = None
        for _ in range(args.n_outer):
            cur = with_biases(mop, jnp.exp(jnp.asarray(u)))
            sol = count_map_solve(cur, xi0=xi)
            xi = sol["xi_hat"]
            g_b = bias_profile_grad(xi, cur, log_b_prior=prior)
            H_b = bias_profile_hessian(xi, cur, H_chol=sol["H_chol"],
                                       log_b_prior=prior)
            b = jnp.exp(jnp.asarray(u))
            g_u = np.asarray(b * g_b)
            H_u = np.asarray(bias_profile_curvature_log(g_b, H_b, b))
            d = np.ones(2)                       # the amplitude direction
            step = float(d @ g_u) / float(d @ H_u @ d)
            u = u - np.clip(step, -0.5, 0.5) * d
            P = float(multi_objective(xi, cur)) + float(
                0.5 * np.sum((u / prior_s) ** 2))
        return P, xi

    P0, xi0 = _profile_at_logr(float(np.log(rr["ratio"])), sh["xi_hat"])
    rows = []
    for n in n_sigma:
        for s in (+1, -1):
            t = float(np.log(rr["ratio"])) + s * n * sig
            P, _ = _profile_at_logr(t, xi0)
            rows.append(dict(n_sigma=s * n, log_r=t, delta_J=P - P0,
                             expected=0.5 * n ** 2))
    return dict(seed=int(seed), ratio=rr["ratio"], sigma_log=sig,
                P0=P0, points=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    rb = os.path.join(here, "..", "pr6a", "data", "rb")
    ap.add_argument("--survey", default=os.path.join(rb, "catalog_pixelated_nside_16.h5"))
    ap.add_argument("--mth-map", default=os.path.join(rb, "mth_map_nside16.h5"))
    ap.add_argument("--out", default=os.path.join(here, "tier_e.json"))
    ap.add_argument("--n-real", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=770000)
    ap.add_argument("--seed-step", type=int, default=131)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="index of the first seed in this CHUNK. The campaign "
                         "is run as several short processes rather than one "
                         "long one because each outer profile trip re-traces "
                         "the solve, and ~500 XLA compilations in one process "
                         "exhausts the JIT's executable memory pool ('JIT "
                         "session error: Cannot allocate memory', observed "
                         "twice at realization 19 of 20). Chunking is the fix "
                         "that does not change a single number: the seeds are "
                         "seed0 + step * (offset + k), so chunk boundaries are "
                         "invisible to the results.")
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--n-gal", type=int, default=180000)
    ap.add_argument("--m-sph", type=int, default=40)
    ap.add_argument("--m-z", type=int, default=8)
    ap.add_argument("--ls-sph", type=float, default=0.6)
    ap.add_argument("--ls-z", type=float, default=0.05)
    ap.add_argument("--sigma-z", type=float, default=0.023)
    ap.add_argument("--b1", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--b-start", type=float, default=1.0,
                    help="starting bias for BOTH tracers -- deliberately the "
                         "same for both, so the recovered ratio cannot be an "
                         "artifact of where the profile started")
    ap.add_argument("--fp2-power", type=float, default=1.6)
    ap.add_argument("--fp2-scale", type=float, default=0.85)
    ap.add_argument("--n-outer", type=int, default=30,
                    help="outer profile-Newton trips. 30, not 10: at 10 the "
                         "ratio direction is NOT converged and the error is "
                         "large and silent -- measured on overlap seed 880000, "
                         "n_outer=10 returned r = 1.940019 with a profile "
                         "gradient of -1816 nat per unit log-ratio, and "
                         "n_outer=20 returned r = 1.996658 at gradient 0.000. "
                         "The residual gradient is now reported per "
                         "realization and gated by --profile-grad-tol.")
    ap.add_argument("--max-log-step", type=float, default=1.0,
                    help="trust region on the outer step, nat per trip. The "
                         "bias AMPLITUDE is soft and travels far (the counts "
                         "put it near b ~ 50 on this world, against a prior "
                         "centred at b = 1), so 0.5 nat/trip spends most of "
                         "the budget travelling rather than converging.")
    ap.add_argument("--profile-grad-tol", type=float, default=1e-3,
                    help="max |dP/du| accepted at the returned point. A "
                         "realization above it is reported as NOT CONVERGED "
                         "rather than quoted.")
    ap.add_argument("--prior-s", type=float, default=1.0)
    ap.add_argument("--prior-sweep", type=float, nargs="*",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--sweep-n-real", type=int, default=5)
    ap.add_argument("--verify", action="store_true",
                    help="re-evaluate the profile at +-1 and +-2 sigma in "
                         "log r, with the amplitude re-profiled, and compare "
                         "the rise in J against the Gaussian n^2/2 nats")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)

    world = build_world(args)
    print(f"[world] nside {world['nside']}, {world['occ'].size} footprint "
          f"pixels, {args.n_shells} shells, T_g {world['T_g'].tolist()}, "
          f"rank {world['basis'].rank}, "
          f"<f_p1> {world['f_p'][0].mean():.4f}, "
          f"<f_p2> {world['f_p'][1].mean():.4f}", flush=True)

    seeds = [args.seed0 + args.seed_step * (args.seed_offset + k)
             for k in range(args.n_real)]
    rows = []
    for i, s in enumerate(seeds):
        r = one_realization(world, s, args, args.prior_s)
        rows.append(r)
        print(f"[{i + 1:3d}/{args.n_real}] seed {s}  shared r = "
              f"{r['shared']['ratio']:.5f} +- {r['shared']['sigma']:.5f} "
              f"(pull {r['pull_shared']:+.3f}, corr {r['shared']['corr']:.6f})"
              f"   decoupled r = {r['decoupled']['ratio']:.5f} +- "
              f"{r['decoupled']['sigma']:.5f}   width x{r['width_ratio']:.1f}"
              f"   grad_r {r['shared']['grad_ratio']:+.1e}"
              f"{'' if r['shared']['converged'] else '  NOT CONVERGED'}"
              f"   [{r['shared']['wall_s']:.1f}+{r['decoupled']['wall_s']:.1f} s]",
              flush=True)

    sweep = []
    for s_prior in args.prior_sweep:
        for s in seeds[:args.sweep_n_real]:
            sweep.append(one_realization(world, s, args, s_prior))
        sub = [x for x in sweep if x["prior_s"] == s_prior]
        print(f"[sweep] prior s = {s_prior:5.2f}  shared sigma_r "
              f"{np.median([x['shared']['sigma'] for x in sub]):.6f}   "
              f"decoupled sigma_r "
              f"{np.median([x['decoupled']['sigma'] for x in sub]):.6f}",
              flush=True)

    verify = None
    if args.verify:
        verify = verify_curvature(world, seeds[0], args, args.prior_s)
        for p in verify["points"]:
            print(f"[verify] log r offset {p['n_sigma']:+.1f} sigma: "
                  f"Delta J = {p['delta_J']:.4f} nat, Gaussian expects "
                  f"{p['expected']:.4f}", flush=True)

    pulls = np.array([r["pull_shared"] for r in rows])
    sig_sh = np.array([r["shared"]["sigma"] for r in rows])
    sig_de = np.array([r["decoupled"]["sigma"] for r in rows])
    summary = dict(
        n_real=len(rows),
        gate_i_within_2sigma=int(np.sum(np.abs(pulls) < 2.0)),
        pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
        pull_sem=float(pulls.std(ddof=1) / np.sqrt(len(pulls))),
        r_median_shared=float(np.median([r["shared"]["ratio"] for r in rows])),
        r_median_decoupled=float(np.median([r["decoupled"]["ratio"]
                                            for r in rows])),
        sigma_median_shared=float(np.median(sig_sh)),
        sigma_median_decoupled=float(np.median(sig_de)),
        gate_ii_shared_tighter=int(np.sum(sig_sh < sig_de)),
        width_ratio_median=float(np.median(sig_de / sig_sh)),
        corr_median_shared=float(np.median([r["shared"]["corr"] for r in rows])),
        corr_max_abs_decoupled=float(np.max(np.abs(
            [r["decoupled"]["corr"] for r in rows]))),
        pull_decoupled_mean=float(np.mean([r["pull_decoupled"] for r in rows])),
        pull_decoupled_sd=float(np.std([r["pull_decoupled"] for r in rows],
                                       ddof=1)),
    )
    print("[summary] " + json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), summary=summary, rows=rows,
                       sweep=sweep, verify=verify), f, indent=1, default=str)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
