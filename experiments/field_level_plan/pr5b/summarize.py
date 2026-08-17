"""Turn ``member_spread.json`` into the tables PLAN §6.5 item 1 asks for.

Pure post-processing: it reads the campaign's raw ``ll_m`` matrix and the
closed-form ``sigma_prediction.json`` and prints (a) ``sigma(H0)`` and
``ESS(H0)``, (b) the ``log Zhat_M - log Zhat_256`` convergence table, (c) the
P14 verdict, (d) P17 arm (b), and (e) the six pre-registered refutation
criteria R1-R6 of ``PREDICTION.md`` §6, each with PASS / REFUTED.  Nothing is
recomputed from the likelihood, so this can be re-run at any time without a
GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    d = json.load(open(HERE / "member_spread.json"))
    nodes = np.asarray(d["h0_nodes"])
    anchor_i = int(d["anchor_node_index"])
    pred = json.load(open(HERE / "sigma_prediction.json"))

    print(f"# PR-5b measurement summary")
    print(f"guard: {d['guard_convention']['name']}  b_GW={d['b_GW']}  "
          f"git {d['git_sha'][:12]}  {d['total_seconds'] / 60:.1f} min")
    print(f"devices: {d['devices']}")

    for arm in ("m256", "m8"):
        if arm not in d["arms"]:
            continue
        a = d["arms"][arm]
        der = a["derived"]
        ll = np.asarray(a["ll"])
        sig = np.asarray(der["sigma"])
        ess = np.asarray(der["ess_over_M"])
        ok = np.asarray(der["node_finite"])
        print(f"\n## arm {arm}  M_draw={der['M_draw']}  "
              f"sha {a['sha256'][:16]}  finite nodes {int(ok.sum())}/{ok.size}")
        print(f"   sigma(H0): min {np.nanmin(sig[ok]):.6e}  "
              f"max {np.nanmax(sig[ok]):.6e}  anchor {sig[anchor_i]:.6e}")
        print(f"   ESS/M(H0): min {np.nanmin(ess[ok]):.6f}  "
              f"max {np.nanmax(ess[ok]):.6f}  anchor {ess[anchor_i]:.6f}")
        print(f"   {'H0':>7} {'sigma':>12} {'ESS/M':>9} {'sigma_pred':>12} "
              f"{'ratio':>8}")
        pn = {round(float(x["H0"]), 6): x for x in pred["nodes"]}
        for i, h in enumerate(nodes):
            p = pn.get(round(float(h), 6))
            sp = p["sigma_anchor"] if p else float("nan")
            print(f"   {h:7.2f} {sig[i]:12.6e} {ess[i]:9.6f} {sp:12.6e} "
                  f"{sig[i] / sp if sp else float('nan'):8.4f}")

        for order in ("balanced", "naive"):
            s = der["series"][order]
            print(f"\n   log Zhat_M - log Zhat_256 ({order} prefixes)")
            Ms = sorted(int(k) for k in s["p14_theta_variation"])
            print(f"   {'M':>5} {'mean offset':>14} {'P14 full':>12} "
                  f"{'P14 bulk':>12}")
            for M in Ms:
                print(f"   {M:5d} {s['abs_bias_mean'][str(M)]:14.6e} "
                      f"{s['p14_theta_variation'][str(M)]:12.6e} "
                      f"{s['p14_theta_variation_bulk_75_105'][str(M)]:12.6e}")

        if "p17b_measured" in der:
            p17 = np.asarray(der["p17b_measured"])
            tgt = np.asarray(der["p17b_target_half_sigma2"])
            print(f"\n   P17 arm (b) at the anchor node: measured "
                  f"{p17[anchor_i]:.6e} vs 0.5 sigma^2 {tgt[anchor_i]:.6e} "
                  f"(ratio {p17[anchor_i] / tgt[anchor_i]:.4f})")

    if "predicted_vs_measured" in d:
        rows = d["predicted_vs_measured"]
        print("\n## R4 (zero-MC-error member-by-member test, M=8 arm)")
        print(f"   {'H0':>7} {'corr':>8} {'sd_meas':>12} {'sd_pred':>12} "
              f"{'ratio':>8}")
        for r in rows:
            if "dll_corr" not in r:
                continue
            print(f"   {r['H0']:7.2f} {r['dll_corr']:8.4f} "
                  f"{r['dll_sd_measured']:12.6e} {r['dll_sd_predicted']:12.6e} "
                  f"{r['dll_sd_measured'] / r['dll_sd_predicted']:8.4f}")

    _refutation(d, nodes, anchor_i)


def _refutation(d, nodes, anchor_i):
    """Score PREDICTION.md §6's six PRE-REGISTERED criteria, verbatim bands.

    Stated before the measurement existed, so they are scored mechanically
    here rather than argued: each one is a band, and the only judgement is
    which arm supplies the input (§6 names ``M_draw = 256`` for R1-R3/R6, the
    shipped eight members for R4-R5).
    """
    arm = d["arms"].get("m256") or d["arms"]["m8"]
    der = arm["derived"]
    sig = np.asarray(der["sigma"])
    d8 = d["arms"]["m8"]["derived"]
    print("\n## PREDICTION.md §6 refutation criteria (pre-registered)")
    v = []

    s_anchor = sig[anchor_i]
    v.append(("R1 level at anchor", 0.0514 <= s_anchor <= 0.2056,
              f"sigma = {s_anchor:.6e} vs band [0.0514, 0.2056] "
              f"(predicted 0.102782)"))

    i20 = int(np.argmin(np.abs(nodes - 20.0)))
    ratio = sig[i20] / s_anchor
    # The node list is the 33-point linspace with the anchor 67.74 APPENDED,
    # so it is not sorted; sort before testing monotonicity (the first version
    # of this check did not, and reported a spurious non-monotonicity created
    # entirely by the trailing anchor node).
    grid = np.arange(nodes.size)[:-1]                    # drop the anchor copy
    order = grid[np.argsort(nodes[grid])]
    body = sig[order][(np.sort(nodes[grid]) >= 20) & (np.sort(nodes[grid]) <= 120)]
    mono = bool(np.all(np.diff(body[np.isfinite(body)]) <= 0))
    v.append(("R2 shape", mono and 6.5 <= ratio <= 26.0,
              f"monotone 20->120: {mono}; sigma(20)/sigma(anchor) = "
              f"{ratio:.4f} vs band [6.5, 26] (predicted 13.0)"))

    v.append(("R3 mechanism", not (abs(s_anchor / 0.15577 - 1) < 0.10
                                   and abs(s_anchor / 0.102782 - 1) > 0.25),
              f"Euclidean ||a||_2 = 0.15577 would need |ratio-1| < 0.10; "
              f"measured ratio {s_anchor / 0.15577:.4f}"))

    if "dll_members" in d8:
        dm = np.asarray(d8["dll_members"][anchor_i])
        pred = json.load(open(HERE / "sigma_prediction.json"))
        dp = np.asarray(pred["dll_members_M8"])
        corr = float(dm @ dp / np.linalg.norm(dm) / np.linalg.norm(dp))
        sd = float(dm.std(ddof=0))
        half = dm.size // 2
        anti = np.abs(dm[:half] + dm[half:]).max() / np.sqrt((dm ** 2).mean())
        v.append(("R4 member-level", (anti <= 0.01) and corr >= 0.9
                  and 0.5 <= sd / 0.10330 <= 2.0,
                  f"corr = {corr:.4f} (needs >= 0.9); sd = {sd:.6e} vs "
                  f"0.10330 (factor {sd / 0.10330:.4f}); antithetic residual "
                  f"{anti:.4f} of rms (needs <= 0.01)"))
        p17 = d8["p17b_measured"][anchor_i]
        v.append(("R5 P17 arm (b)", 2.7e-3 <= p17 <= 1.1e-2,
                  f"measured {p17:.6e} vs band [2.7e-3, 1.1e-2]; sign "
                  f"{'positive (Jensen holds)' if p17 > 0 else 'NEGATIVE'}"))

    s = der["series"]["balanced"]["p14_theta_variation"]
    if "32" in s and "8" in s:
        v.append(("R6 decision", s["32"] <= 0.1 and s["8"] > 0.1,
                  f"P14(M=32) = {s['32']:.4e} (needs <= 0.1); "
                  f"P14(M=8) = {s['8']:.4e} (predicted > 0.1)"))

    for name, ok, why in v:
        print(f"   {'PASS    ' if ok else 'REFUTED '} {name:18s} {why}")


if __name__ == "__main__":
    main()
