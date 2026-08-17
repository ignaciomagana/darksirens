"""PR-5b item 1: MEASURE the member spread on the production 259-event line.

PLAN §6.5 item 1, verbatim: *"with latent leaves live but the likelihood
otherwise unchanged, emit the ``ll_m`` vector at ``M_draw = 256`` at 33 ``H0``
nodes across [20, 140]. Report ``sigma(H0)``, ``ESS(H0)``, and
``log Zhat_M - log Zhat_256`` for ``M in {4, 8, 16, 32, 64, 128}``."*

This is the measurement the plan's largest open number is decided by.  The
closed-form prediction phase (``predict_sigma.py`` / ``PREDICTION.md``, same
directory) already published, on the same anchor, what this run must find:
``sigma = 0.10278`` nats at the anchor node, monotone in ``H0`` from 1.3388
at 20 to 5.31e-3 near 121, ``E[ESS]/M = 0.98949`` at the anchor, P14 met at
``M_draw = 32`` over the full prior and at 4 over the posterior bulk, and
``0.5 sigma^2 = 5.2821e-3`` nats for P17 arm (b).  Six refutation criteria
R1-R6 were stated in advance (PREDICTION.md §6); this script measures every
input they need and writes the comparison into the output JSON so the report
phase compares rather than re-derives.

Three arms, because they answer different questions
---------------------------------------------------
``m256``  the deliverable: 256 members from ``latent_anchor_m256.h5``.
          ``sigma(H0)``, ``ESS(H0)``, the ``M``-prefix convergence table, P14.

``m8``    the SHIPPED 8-member anchor -- **the sharpest test in the whole
          rung and the only one with zero Monte-Carlo error** (PREDICTION.md
          §6, R4).  The prediction published a per-member vector
          ``dll_m = ll_m - ll(xi_hat)`` for these EXACT eight draws at every
          node.  The 256-member set is a different draw set (a build at a
          different ``M_draw`` keys ``jax.random.normal`` on a different
          shape, gated as G4 in ``build_anchor_m256.py``), so only this arm
          can be compared member-by-member instead of in distribution.

``map``   ONE member that IS ``xi_hat`` (``latent_anchor_map_m1.h5``).
          ``ll(xi_hat)`` is not a diagnostic here: PLAN §1.6 Limit III / §6.5
          item 5 state P17 arm (b) as
          ``LSE_m ll_m - log M - ll(xi_hat) -> 0.5 sigma^2``, so without it
          there is no P17 measurement and no ``dll_m`` to correlate.

Conventions this run takes, all of which change how the numbers read
--------------------------------------------------------------------
**Guard.** PR-0's ``*_nogv`` clean arm: soft guard OFF, variance cap lifted
to 1e6, Vitale ``5 N_obs`` floor retained.  Reason in
``latent_harness.clean_arm_opts``: the hard GWTC-4/5 criterion fails at every
node on this line, and the soft guard's ~-1e6 nat wall is a function of
``Neff`` and ``log mu``, both MEMBER-dependent -- under the soft guard the
"member spread" would be the spread of the guard.  A soft-guard arm is
measured at a few nodes anyway so the size of that contamination is on the
record rather than asserted.

**Prefixes are ANTITHETICALLY BALANCED.**  ``laplace_draws`` builds
``g = [g_half, -g_half]``, so the antithetic partner of member ``k`` is
``k + M/2`` and the naive prefix ``ll[:4]`` contains FOUR UNPAIRED draws --
it is not the ``M_draw = 4`` configuration anyone would ship, it is a
half-ensemble with the odd part of the response left in.  The primary
``log Zhat_M`` series therefore takes members in the order
``[0, M/2, 1, M/2+1, ...]``, so every even prefix is balanced exactly as a
dedicated ``M_draw = M`` build would be (PLAN §6.5 item 3).  The naive
ordering is computed too and reported alongside: the gap between them IS the
value of antithetic pairing, measured.

**Common random numbers.**  PLAN §6.5 item 2: P14 is a statement about a
DETERMINISTIC function of theta, which requires the same ``g_m`` at every
node.  Here that is structural -- the draws are baked into the artifact,
``load_latent_plan`` runs ONCE per member chunk at likelihood-build time, and
the resulting ``row_fac``/``(A, B)`` are jit-argument device buffers that the
H0 loop never touches.  It is also VERIFIED: every chunk re-evaluates its
first node after finishing the sweep and asserts bit-identity with the value
it produced before the sweep, which can only hold if the member leaves were
unchanged throughout.

Run from ``experiments/desi_full259`` (its ``common.py`` pins ZMAX = 6.0).
"""
from __future__ import annotations

import argparse
import json
import time

import latent_harness as H
from latent_harness import jax, jnp, np

#: PLAN §6.5 item 1's 33 nodes across [20, 140], plus the anchor node itself
#: (67.74, where the closed-form prediction is quoted and where the artifact
#: was built).  Identical to the node set of ``sigma_prediction.json``, so the
#: predicted-vs-measured comparison is node-for-node with no interpolation.
H0_NODES = np.concatenate([np.linspace(20.0, 140.0, 33), [67.74]])

M_SERIES = (4, 8, 16, 32, 64, 128, 256)
BULK = (75.0, 105.0)


def _lse(x, axis=None):
    from scipy.special import logsumexp
    return logsumexp(x, axis=axis)


def balanced_order(m: int) -> np.ndarray:
    """``[0, m/2, 1, m/2+1, ...]`` -- every even prefix antithetically paired."""
    half = m // 2
    order = np.empty(m, dtype=int)
    order[0::2] = np.arange(half)
    order[1::2] = np.arange(half, m)
    return order


# ------------------------------------------------------------------- one arm

def run_arm(anchor, m_draw, chunk, data, nodes, *, tag, soft_guard=False,
            gate_unpatched=False):
    """The ``(n_nodes, m_draw)`` ``ll_m`` matrix for one artifact."""
    ll = np.full((nodes.size, m_draw), np.nan)
    timings = []
    gate = None
    # The reference node for the two per-chunk gates is the LAST one, which is
    # the anchor 67.74 -- deliberately not ``nodes[0] = 20``: PR-0 measured
    # that the low-H0 end of this line is where the Vitale floor bites, and a
    # gate that compares -inf against -inf would pass (or, subtracting, throw
    # NaN) for reasons having nothing to do with what it is gating.
    ref = int(nodes.size - 1)
    for lo in range(0, m_draw, chunk):
        hi = min(lo + chunk, m_draw)
        n = hi - lo
        t_build = time.time()
        with H.member_ll_patch(n) as seen, H.member_slice_patch(lo, hi):
            opts = H.clean_arm_opts(anchor, soft_guard=soft_guard)
            logl = H.build_likelihood(opts, data)
            t_build = time.time() - t_build
            t0 = time.time()
            first = np.asarray(logl(jnp.asarray([nodes[ref]]))) + np.log(n)
            t_compile = time.time() - t0
            if not seen["n"] or first.shape != (n,):
                raise SystemExit(
                    f"[{tag}] the member-ll patch never fired (intercepts="
                    f"{seen['n']}, shape={first.shape}, expected ({n},)): the "
                    "build re-used a cached executable instead of re-tracing, "
                    "so this would be a 1-member 'ensemble' with sigma = 0. "
                    "See latent_harness.build_likelihood(fresh_trace).")
            ll[ref, lo:hi] = first
            t0 = time.time()
            for i in range(nodes.size):
                if i == ref:
                    continue
                ll[i, lo:hi] = (np.asarray(logl(jnp.asarray([nodes[i]])))
                                + np.log(n))
            t_sweep = time.time() - t0
            # --- CRN / determinism: the member leaves did not move ---------
            again = np.asarray(logl(jnp.asarray([nodes[ref]]))) + np.log(n)
            if not np.array_equal(again, first):
                raise SystemExit(
                    f"[{tag}] CRN GATE FAILED on chunk [{lo},{hi}): node "
                    f"{nodes[ref]} did not reproduce after the sweep; the "
                    "member draws are not common across H0 nodes and P14 is "
                    "not defined. max dev "
                    f"{np.abs(again - first).max():.3e}")
        # --- the patch gate, at the production artifact -------------------
        if gate_unpatched and lo == 0:
            with H.member_slice_patch(lo, hi):
                logl_u = H.build_likelihood(
                    H.clean_arm_opts(anchor, soft_guard=soft_guard), data)
                scalar = float(logl_u(jnp.asarray([nodes[ref]])))
            recon = float(_lse(first) - np.log(n))
            gate = {"unpatched": scalar, "reduced_patched": recon,
                    "delta": recon - scalar, "n_members": n,
                    "H0": float(nodes[ref])}
            print(f"[{tag}] PATCH GATE: reduced {recon!r} vs unpatched "
                  f"{scalar!r}  delta = {recon - scalar:.3e}", flush=True)
            if not abs(recon - scalar) < 1e-6:
                raise SystemExit(
                    f"[{tag}] PATCH GATE FAILED: member_ll_patch changed the "
                    "likelihood; no PR-5b number may be quoted.")
        timings.append(dict(lo=lo, hi=hi, build_s=t_build,
                            compile_s=t_compile, sweep_s=t_sweep,
                            per_eval_ms=1e3 * t_sweep / max(nodes.size - 1, 1)))
        print(f"[{tag}] members [{lo},{hi}) done: build {t_build:.1f} s, "
              f"compile {t_compile:.1f} s, {nodes.size - 1} evals in "
              f"{t_sweep:.1f} s ({1e3 * t_sweep / max(nodes.size - 1, 1):.0f} "
              f"ms/eval)", flush=True)
    if not np.isfinite(ll).all():
        n_bad = int((~np.isfinite(ll)).sum())
        print(f"[{tag}] WARNING: {n_bad} of {ll.size} ll_m are non-finite",
              flush=True)
    return ll, timings, gate


# ------------------------------------------------------------------ analysis

def derive(ll, nodes, ll_map=None):
    """Everything PLAN §6.5 item 1 asks to be reported, from one ll matrix."""
    m = ll.shape[1]
    out = {"M_draw": m}
    # A node where ANY member is non-finite carries no member spread and no
    # log Zhat: PR-0 measured that the Vitale 5*N_obs floor can still return
    # -inf at the low-H0 end of this line even with the variance cap lifted.
    # Such nodes are recorded and EXCLUDED from the P14 maxima rather than
    # silently turned into NaN -- a NaN would propagate into the max and make
    # the gate unreadable, and a zero would make it pass.
    ok = np.isfinite(ll).all(axis=1)
    out["node_finite"] = ok.tolist()
    out["n_nodes_finite"] = int(ok.sum())
    sigma = ll.std(axis=1, ddof=0)
    sigma1 = ll.std(axis=1, ddof=1) if m > 1 else np.zeros(nodes.size)

    # ESS of the self-normalized member average -- the quantity K5 gates.
    w = np.exp(ll - ll.max(axis=1, keepdims=True))
    ess = w.sum(axis=1) ** 2 / (w ** 2).sum(axis=1)

    orders = {"balanced": balanced_order(m), "naive": np.arange(m)}
    series = {}
    for name, order in orders.items():
        logZ = {}
        for M in M_SERIES:
            if M > m:
                continue
            sub = ll[:, order[:M]]
            logZ[M] = _lse(sub, axis=1) - np.log(M)
        ref = logZ[max(logZ)]
        delta = {M: (v - ref) for M, v in logZ.items()}
        # PLAN §6.5 item 2: P14 gates the theta-VARIATION, not the level --
        # with CRN a constant offset is absorbed into the evidence.
        p14 = {M: float(np.max(np.abs(d - d.mean()))) for M, d in delta.items()}
        in_bulk = (nodes >= BULK[0]) & (nodes <= BULK[1])
        p14_bulk = {
            M: float(np.max(np.abs(d[in_bulk] - d[in_bulk].mean())))
            for M, d in delta.items()}
        series[name] = {
            "log_Zhat": {str(M): v.tolist() for M, v in logZ.items()},
            "log_Zhat_minus_ref": {str(M): v.tolist()
                                   for M, v in delta.items()},
            "p14_theta_variation": {str(M): v for M, v in p14.items()},
            "p14_theta_variation_bulk_75_105": {str(M): v
                                                for M, v in p14_bulk.items()},
            "abs_bias_mean": {str(M): float(d.mean())
                              for M, d in delta.items()},
        }
    out.update({
        "sigma": sigma.tolist(), "sigma_ddof1": sigma1.tolist(),
        "ess": ess.tolist(), "ess_over_M": (ess / m).tolist(),
        "log_Zhat_full": (_lse(ll, axis=1) - np.log(m)).tolist(),
        "mean_ll": ll.mean(axis=1).tolist(),
        "series": series,
    })
    if ll_map is not None:
        dll = ll - ll_map[:, None]
        # P17 arm (b): LSE_m ll_m - log M - ll(xi_hat) -> 0.5 sigma^2.
        p17b = _lse(ll, axis=1) - np.log(m) - ll_map
        out["dll_members"] = dll.tolist()
        out["p17b_measured"] = p17b.tolist()
        out["p17b_target_half_sigma2"] = (0.5 * sigma ** 2).tolist()
        # antithetic residual: ll_k + ll_{k+M/2} - 2 ll(xi_hat) is the pure
        # SECOND-order part; PLAN §6.5 item 3's "cancels the odd part exactly".
        half = m // 2
        if half:
            anti = dll[:, :half] + dll[:, half:]
            out["antithetic_residual_max"] = np.abs(anti).max(axis=1).tolist()
            out["antithetic_residual_rms"] = np.sqrt(
                (anti ** 2).mean(axis=1)).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=32,
                    help="members per likelihood build (exact; see "
                         "latent_harness.member_slice_patch)")
    ap.add_argument("--soft-guard-nodes", type=int, default=5)
    ap.add_argument("--skip-m256", action="store_true")
    args = ap.parse_args(argv)

    t_all = time.time()
    anchor8 = H.resolve_anchor_m8()
    print(f"[campaign] devices {jax.devices()}", flush=True)
    print(f"[campaign] M=8 anchor  {anchor8}", flush=True)
    print(f"[campaign] M=256 anchor {H.ANCHOR_M256}  "
          f"exists={H.ANCHOR_M256.exists()}", flush=True)
    print(f"[campaign] MAP anchor   {H.ANCHOR_MAP}  "
          f"exists={H.ANCHOR_MAP.exists()}", flush=True)

    import h5py

    def _sha(p):
        with h5py.File(p) as f:
            return str(f["latent_field"].attrs["sha256"])

    t0 = time.time()
    data = H.load_data(H.clean_arm_opts(anchor8))
    print(f"[campaign] data loaded in {time.time() - t0:.1f} s", flush=True)

    nodes = H0_NODES
    out = {
        "script": "experiments/field_level_plan/pr5b/run_member_spread.py",
        "plan_section": "6.5 item 1",
        "h0_nodes": nodes.tolist(),
        "n_nodes": int(nodes.size),
        "anchor_H0": 67.74,
        "anchor_node_index": int(nodes.size - 1),
        "guard_convention": {
            "selection_neff_soft_guard": False,
            "max_likelihood_variance": 1e6,
            "name": "PR-0 clean arm (*_nogv)",
            "why": ("the hard GWTC-4/5 variance criterion fails at every H0 "
                    "node on this line (PR-0 REPORT item 3: needs Neff ~92k, "
                    "has 31-36k), and the soft guard's ~-1e6 nat wall is a "
                    "function of the MEMBER-dependent Neff and log mu"),
        },
        "b_GW": H.B_GW,
        "fixed": H.fixed_values(),
        "git_sha": H.C.git_sha(),
        "DARKSIRENS_ZMAX": H.C.ZMAX,
        "devices": [str(d) for d in jax.devices()],
        "arms": {},
    }
    outfile = H.PR5B_DIR / "member_spread.json"

    def _save():
        out["total_seconds"] = time.time() - t_all
        with open(outfile, "w") as f:
            json.dump(out, f, indent=1)

    # ---------------------------------------------------------------- MAP
    ll_map = None
    if H.ANCHOR_MAP.exists():
        ll_map_mat, tim, _ = run_arm(H.ANCHOR_MAP, 1, 1, data, nodes,
                                     tag="map")
        ll_map = ll_map_mat[:, 0]
        out["arms"]["map"] = {
            "artifact": str(H.ANCHOR_MAP), "sha256": _sha(H.ANCHOR_MAP),
            "ll": ll_map.tolist(), "timings": tim,
            "what": "ll(xi_hat) -- the P17 arm (b) reference point",
        }
        _save()
    else:
        print("[campaign] no MAP artifact: P17 arm (b) and dll_m unavailable",
              flush=True)

    # ------------------------------------------------------------- M = 8
    ll8, tim8, gate8 = run_arm(anchor8, 8, 8, data, nodes, tag="m8",
                               gate_unpatched=True)
    out["arms"]["m8"] = {
        "artifact": str(anchor8), "sha256": _sha(anchor8),
        "ll": ll8.tolist(), "timings": tim8, "patch_gate": gate8,
        "derived": derive(ll8, nodes, ll_map),
        "what": ("the SHIPPED anchor -- the draws PREDICTION.md published a "
                 "per-member dll vector for (R4, zero MC error)"),
    }
    _save()

    # ------------------------------------------------------------- M = 256
    if H.ANCHOR_M256.exists() and not args.skip_m256:
        with h5py.File(H.ANCHOR_M256) as f:
            m256 = int(f["latent_field"]["row_fac"].shape[0])
        ll256, tim256, gate256 = run_arm(H.ANCHOR_M256, m256, args.chunk,
                                         data, nodes, tag="m256",
                                         gate_unpatched=True)
        out["arms"]["m256"] = {
            "artifact": str(H.ANCHOR_M256), "sha256": _sha(H.ANCHOR_M256),
            "M_draw": m256, "chunk": args.chunk,
            "ll": ll256.tolist(), "timings": tim256, "patch_gate": gate256,
            "derived": derive(ll256, nodes, ll_map),
            "what": "PLAN 6.5 item 1's deliverable",
        }
        _save()
    else:
        print("[campaign] no M=256 artifact -- run build_anchor_m256.py first",
              flush=True)

    # -------------------------------------------------- the guard's own size
    if args.soft_guard_nodes:
        sub = nodes[np.linspace(0, nodes.size - 2, args.soft_guard_nodes,
                                dtype=int)]
        ll_soft, tim_soft, _ = run_arm(anchor8, 8, 8, data, sub,
                                       tag="m8-softguard", soft_guard=True)
        idx = [int(np.argmin(np.abs(nodes - h))) for h in sub]
        out["arms"]["m8_soft_guard"] = {
            "h0_nodes": sub.tolist(), "ll": ll_soft.tolist(),
            "sigma": ll_soft.std(axis=1, ddof=0).tolist(),
            "sigma_clean_same_nodes": ll8[idx].std(axis=1, ddof=0).tolist(),
            "timings": tim_soft,
            "what": ("selection_neff_soft_guard=True, default variance cap -- "
                     "the arm PR-5b does NOT quote, measured so the size of "
                     "the guard's member-dependent contamination is on the "
                     "record"),
        }
        _save()

    # ------------------------------------- predicted vs measured, node-wise
    pred_path = H.PR5B_DIR / "sigma_prediction.json"
    if pred_path.exists():
        pred = json.load(open(pred_path))
        pn = {round(float(n["H0"]), 6): n for n in pred["nodes"]}
        cmp_rows = []
        d8 = out["arms"]["m8"]["derived"]
        d256 = out["arms"].get("m256", {}).get("derived")
        for i, h in enumerate(nodes):
            p = pn.get(round(float(h), 6))
            if p is None:
                continue
            row = {
                "H0": float(h),
                "sigma_predicted_Hnorm": p["sigma_anchor"],
                "sigma_predicted_euclid": p["sigma_euclid"],
                "sigma_predicted_pr0_factored": p["sigma_pr0_factored"],
                "sigma_predicted_pe_reweighted": p["sigma_anchor_pe_reweighted"],
                "sigma_measured_m8": d8["sigma"][i],
            }
            if d256:
                row["sigma_measured_m256"] = d256["sigma"][i]
                row["ess_over_M_measured_m256"] = d256["ess_over_M"][i]
                row["ess_over_M_predicted"] = float(
                    np.exp(-p["sigma_anchor"] ** 2))
            if ll_map is not None:
                dm = np.asarray(d8["dll_members"][i])
                dp = np.asarray(p["dll_members"])
                denom = np.linalg.norm(dm) * np.linalg.norm(dp)
                row["dll_measured_m8"] = dm.tolist()
                row["dll_predicted_m8"] = dp.tolist()
                row["dll_corr"] = float(dm @ dp / denom) if denom > 0 else None
                row["dll_sd_measured"] = float(dm.std(ddof=0))
                row["dll_sd_predicted"] = p["dll_sd"]
                row["p17b_measured_m8"] = d8["p17b_measured"][i]
                row["p17b_predicted"] = p["p17b_realized_nats"]
                if d256:
                    row["p17b_measured_m256"] = d256["p17b_measured"][i]
            cmp_rows.append(row)
        out["predicted_vs_measured"] = cmp_rows
        _save()

    _save()
    print(f"[campaign] wrote {outfile}  "
          f"({out['total_seconds'] / 60:.1f} min)", flush=True)

    # ------------------------------------------------------------- console
    d = out["arms"].get("m256", out["arms"]["m8"])["derived"]
    sig = np.asarray(d["sigma"])
    ess = np.asarray(d["ess_over_M"])
    print(f"[campaign] sigma(H0): {sig.min():.6e} .. {sig.max():.6e} nats "
          f"(anchor node {sig[-1]:.6e})", flush=True)
    print(f"[campaign] ESS/M(H0): {ess.min():.6f} .. {ess.max():.6f} "
          f"(anchor node {ess[-1]:.6f})", flush=True)
    for name in ("balanced", "naive"):
        p14 = d["series"][name]["p14_theta_variation"]
        print(f"[campaign] P14 ({name} prefixes): "
              + "  ".join(f"M={M}:{p14[str(M)]:.4e}"
                          for M in M_SERIES if str(M) in p14), flush=True)


if __name__ == "__main__":
    main()
