#!/usr/bin/env python
"""P_det emulator validation gates for the --pdet_flow_path selection input.

Drives the estimator kernels directly (no CLI), mirroring
scripts/flows_diagnostics.py.  Four gates:

(a) golden  — pins ~12 theta vectors: our vendored log p_inj cross-checked
              against the gw-nf reference implementation (pdet.gwtc4_model,
              via --gwnf_src), and p_det = xi * q_flow / p_inj from the
              wrapped reconstruction.  Prints paste-ready constants for
              tests/test_pdet_selection.py::test_golden_parity_with_gwnf.
(b) xi      — E_{p_inj}[p_det] == xi*(1 - leak) within Monte-Carlo error
              (draws from the ANALYTIC injection prior via inverse-CDF
              sampling; ``leak`` is the flow mass outside the injection
              prior's support, ~1%, measured with the production drop mask —
              without it the gate's verdict is a function of --n_pinj).
(c) marginals — pseudo-injection detector-frame marginals vs the found
              injections in the selection HDF5 (quantile errors + overlay
              PNG).  Expected close but not identical: the h5 mixes ~8% O3
              rows and used the 67.74/0.3089 dL override.
(d) logmu   — log mu(Lambda) emulator-vs-h5 through the production spectral
              log_weight kernel across a grid of powerlaw+peak draws.
              Success: Delta log mu approximately Lambda-independent
              (std <~ 0.1 nats); Neff_emulator within ~2x of h5.

Usage:
    python scripts/pdet_emulator_validation.py \
        --pdet_flow_path PDetO4NF.npz \
        --gwselection_path selection_o3o4ab_allsky.h5 \
        --outdir runs/pdet_validation [--gwnf_src /path/to/gw-nf-main/src] \
        [--gates abcd] [--n_pinj 2000000] [--n_pop 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.core.constants import H0_FID, OM0_FID
from darksirens.gw import selection as sel
from darksirens.gw.utils import load_selection_samples
from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.inference.utils import log_sample_weight
from darksirens.likelihood.selection import compute_selection_term
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.utils.cosmology import dL_grid_bounds


def _toy_catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _gw_event_from(m1det, m2det, dL, chieff, prior_wt):
    n = m1det.shape[0]
    return GWEvent(
        m1det=jnp.asarray(m1det),
        m2det=jnp.asarray(m2det),
        dL=jnp.asarray(dL),
        chieff=jnp.asarray(chieff),
        prior_wt=jnp.asarray(prior_wt),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        q=jnp.asarray(m2det / m1det),
        valid=jnp.ones(n, dtype=bool),
    )


# ── the full 13-dim analytic injection prior (angles included) ──────────────


def log_pinj_full(theta, z_grid, z_pdf):
    """log p_inj over all 13 dims — comparable to gw-nf's eval_log_pdf."""
    c = {n: theta[:, i] for i, n in enumerate(sel.PDET_COLUMNS)}
    return (
        sel._gwtc4_log_p_m1(c["m1_source"])
        + sel._gwtc4_log_p_m2_given_m1(c["m2_source"], c["m1_source"])
        + sel._gwtc4_log_p_z(c["redshift"], z_grid, z_pdf)
        + sel._gwtc4_log_p_spin_mag(c["a1"])
        + sel._gwtc4_log_p_spin_mag(c["a2"])
        + sel._gwtc4_log_p_cos_tilt(c["cos_tilt1"])
        + sel._gwtc4_log_p_cos_tilt(c["cos_tilt2"])
        - np.log(2 * np.pi)                       # ra
        - np.log(2.0)                             # sin dec
        - np.log(np.pi)                           # psi
        - np.log(2.0)                             # cos iota
        - 2.0 * np.log(2 * np.pi)                 # phi1, phi2
    )


def flow_log_prob_batched(flow, theta, batch=131072):
    out = []
    for i in range(0, len(theta), batch):
        out.append(np.asarray(flow.log_prob(jnp.asarray(theta[i:i + batch]))))
    return np.concatenate(out)


# ── analytic p_inj sampler (inverse-CDF; validation-only) ───────────────────


def _grid_inverse_cdf(rng, n, grid, pdf):
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, grid)


def sample_pinj(n, z_grid, z_pdf, seed=123):
    rng = np.random.default_rng(seed)
    mg = np.logspace(0.0, 3.0, 200001)
    m1 = _grid_inverse_cdf(rng, n, mg, np.exp(sel._gwtc4_log_p_m1(mg)))
    m2 = np.sqrt(rng.uniform(size=n) * (m1 * m1 - 1.0) + 1.0)   # p(m2|m1) ~ m2
    z = _grid_inverse_cdf(rng, n, z_grid, z_pdf)
    ag = np.linspace(0.0, 1.0, 200001)
    a_pdf = np.exp(sel._gwtc4_log_p_spin_mag(ag))
    a1 = _grid_inverse_cdf(rng, n, ag, a_pdf)
    a2 = _grid_inverse_cdf(rng, n, ag, a_pdf)
    cg = np.linspace(-1.0, 1.0, 200001)
    c_pdf = np.exp(sel._gwtc4_log_p_cos_tilt(cg))
    ct1 = _grid_inverse_cdf(rng, n, cg, c_pdf)
    ct2 = _grid_inverse_cdf(rng, n, cg, c_pdf)
    ra = rng.uniform(0.0, 2 * np.pi, n)
    sindec = rng.uniform(-1.0, 1.0, n)
    psi = rng.uniform(0.0, np.pi, n)
    cosinc = rng.uniform(-1.0, 1.0, n)
    phi1 = rng.uniform(0.0, 2 * np.pi, n)
    phi2 = rng.uniform(0.0, 2 * np.pi, n)
    return np.column_stack(
        [m1, m2, z, a1, a2, ra, sindec, psi, cosinc, ct1, ct2, phi1, phi2]
    )


# ── gates ───────────────────────────────────────────────────────────────────


def gate_a_golden(flow, xi, z_grid, z_pdf, gwnf_src):
    print("\n=== gate (a): golden pins ===")
    s = np.asarray(flow.sample(jax.random.key(5), (64,)))
    # keep 9 in-support draws + 3 crafted edge cases
    keep = (
        (s[:, 0] >= 1) & (s[:, 0] <= 1000) & (s[:, 1] >= 1) & (s[:, 1] <= s[:, 0])
        & (s[:, 2] >= 0) & (s[:, 2] <= 3) & (s[:, 3] >= 0) & (s[:, 3] <= 1)
        & (s[:, 4] >= 0) & (s[:, 4] <= 1)
        & (np.abs(s[:, 9]) <= 1) & (np.abs(s[:, 10]) <= 1)
        & (np.abs(s[:, 6]) <= 1) & (np.abs(s[:, 8]) <= 1)
        & (s[:, 7] >= 0) & (s[:, 7] <= np.pi)
    )
    theta = s[keep][:9]
    edges = np.array([
        # z ~ 2.0 (flow training edge), m1 ~ 80, a1 ~ 0.95
        [45.0, 30.0, 2.0, 0.3, 0.2, 1.0, 0.1, 1.0, 0.2, 0.5, -0.3, 1.0, 2.0],
        [80.0, 60.0, 0.5, 0.2, 0.1, 4.0, -0.5, 2.0, -0.4, -0.2, 0.6, 3.0, 5.0],
        [35.0, 25.0, 0.3, 0.95, 0.9, 2.0, 0.6, 0.5, 0.8, 0.9, 0.8, 5.0, 1.0],
    ])
    theta = np.vstack([theta, edges])

    log_pinj = log_pinj_full(theta, z_grid, z_pdf)
    log_q = np.asarray(flow.log_prob(jnp.asarray(theta)))
    pdet = xi * np.exp(log_q - log_pinj)
    print("p_det on pins:", np.array2string(pdet, precision=4))

    if gwnf_src:
        sys.path.insert(0, str(gwnf_src))
        from pdet.gwtc4_model import GWTC4Model
        from core.gw_utils import get_redshift_grid

        zg, dvdz = get_redshift_grid()
        ref = GWTC4Model(z_grid=zg, dVdz_grid=dvdz)
        log_pinj_ref = np.asarray(ref.eval_log_pdf(jnp.asarray(theta)))
        err = np.max(np.abs(log_pinj - log_pinj_ref))
        print(f"log p_inj vs gw-nf reference: max |diff| = {err:.2e}")
        assert err < 1e-5, "gate (a) FAILED: p_inj mismatch vs gw-nf"
        print("gate (a) p_inj cross-check PASSED")
    else:
        print("(--gwnf_src not given: skipping cross-implementation p_inj check)")

    fmt = lambda arr: np.array2string(
        np.asarray(arr), separator=", ", precision=None,
        floatmode="unique", max_line_width=88,
    )
    print("\n# --- paste into tests/test_pdet_selection.py ---")
    print(f"GOLDEN_THETA = {fmt(theta)}".replace("[", "[\n ", 1))
    print(f"GOLDEN_LOG_PINJ = {fmt(log_pinj)}")
    print(f"GOLDEN_PDET = {fmt(pdet)}")
    return theta, log_pinj, pdet


def flow_support_leakage(flow_path, nsamp, seed=17):
    """Flow probability mass OUTSIDE the injection prior's support.

    Estimated with the PRODUCTION support definition -- the drop fraction of
    ``pseudo_injections_from_pdet_flow``, whose draws are exactly flow draws --
    so this number cannot drift away from the mask the selection path applies.
    Returns ``None`` when the pseudo-injection path is unavailable (it needs
    gwcat), in which case gate (b) reports only its raw n_sigma.
    """
    try:
        kept = sel.pseudo_injections_from_pdet_flow(
            flow_path, nsamp=int(nsamp), seed=int(seed))
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        print(f"(leakage estimate unavailable: {exc})")
        return None
    return 1.0 - len(np.asarray(kept[0])) / float(nsamp)


def gate_b_xi(flow, xi, z_grid, z_pdf, n_pinj, flow_path=None, leak_nsamp=200_000):
    """E_{p_inj}[xi q/p_inj] == xi -- corrected for the flow's out-of-support mass.

    The identity holds only if ``q`` integrates to 1 over the injection prior's
    SUPPORT:  E_{p_inj}[xi q/p_inj] = xi * int_supp q dtheta = xi (1 - leak).
    The spline flow has full R^13 support and the selection path budgets ~1%
    leakage, and that is a FIXED RELATIVE bias while ``mc_sigma`` shrinks as
    1/sqrt(n_pinj) -- so a raw ``|E - xi| < 4 mc_sigma`` gate passes or fails
    according to the sample count rather than the emulator's quality, and
    raising --n_pinj to sharpen the test makes a CORRECT emulator fail.  The
    target is therefore ``xi (1 - leak)`` with the leakage's own Monte-Carlo
    error folded in; both the raw and the corrected n_sigma are reported.
    """
    print("\n=== gate (b): E_pinj[p_det] vs xi ===")
    theta = sample_pinj(n_pinj, z_grid, z_pdf)
    log_pinj = log_pinj_full(theta, z_grid, z_pdf)
    log_q = flow_log_prob_batched(flow, theta)
    pdet = xi * np.exp(log_q - log_pinj)
    est = pdet.mean()
    mc_sigma = pdet.std(ddof=1) / np.sqrt(len(pdet))
    nsig_raw = abs(est - xi) / mc_sigma

    leak = None if flow_path is None else flow_support_leakage(flow_path, leak_nsamp)
    if leak is None:
        target, sigma, nsig = xi, mc_sigma, nsig_raw
    else:
        target = xi * (1.0 - leak)
        leak_sigma = xi * np.sqrt(max(leak * (1.0 - leak), 0.0) / leak_nsamp)
        sigma = float(np.hypot(mc_sigma, leak_sigma))
        nsig = abs(est - target) / sigma
        print(f"flow out-of-support mass: leak = {leak:.4%} "
              f"-> target xi(1-leak) = {target:.6e}")
    print(f"E[p_det] = {est:.6e}  (xi = {xi:.6e})  MC sigma = {mc_sigma:.2e}  "
          f"-> raw {nsig_raw:.2f} sigma, leakage-corrected {nsig:.2f} sigma;  "
          f"max p_det = {pdet.max():.3f}  P(p_det > 1) = {(pdet > 1).mean():.2e}")
    ok = nsig < 4.0
    print(f"gate (b) {'PASSED' if ok else 'FAILED'} "
          "(|E - xi(1-leak)| < 4 sigma, MC + leakage)")
    return {"E_pdet": float(est), "xi": float(xi), "target": float(target),
            "leak": None if leak is None else float(leak),
            "mc_sigma": float(mc_sigma), "sigma": float(sigma),
            "n_sigma_raw": float(nsig_raw), "n_sigma": float(nsig), "ok": bool(ok)}


def gate_c_marginals(pseudo, h5_tuple, outdir):
    print("\n=== gate (c): detector-frame marginals vs h5 ===")
    m1p, m2p, dLp, chip = (np.asarray(x) for x in pseudo[:4])
    m1h, m2h, dLh, chih = (np.asarray(x) for x in h5_tuple[:4])
    pairs = {
        "m1det": (m1p, m1h), "q": (m2p / m1p, m2h / m1h),
        "dL": (dLp, dLh), "chieff": (chip, chih),
        "sindec": (np.sin(np.asarray(pseudo[5])), np.sin(np.asarray(h5_tuple[5]))),
    }
    qs = np.linspace(0.02, 0.98, 25)
    stats, ok = {}, True
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.2))
    for ax, (name, (xp, xh)) in zip(axes, pairs.items()):
        err = float(np.max(np.abs(np.quantile(xp, qs) - np.quantile(xh, qs)))
                    / np.std(xh))
        stats[name] = err
        ok &= err < 0.25
        print(f"  {name:8s} max quantile err = {err:.3f} std units")
        lo, hi = np.percentile(xh, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(xh, bins=bins, density=True, alpha=0.45, label="h5 found inj")
        ax.hist(xp, bins=bins, density=True, histtype="step", lw=1.6,
                label="pseudo-inj", color="C3")
        ax.set_xlabel(name)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "marginals.png", dpi=140)
    plt.close(fig)
    print(f"gate (c) {'PASSED' if ok else 'FAILED'} "
          f"(all < 0.25 std; expected offsets: O3 mix, dL-cosmology override)")
    return {"quantile_err_std_units": stats, "ok": bool(ok)}


def gate_d_logmu(pseudo, ndraw_em, h5_tuple, ndraw_h5, n_pop, sel_batch, outdir):
    print("\n=== gate (d): log mu parity through the production kernel ===")
    cosmo = CosmoParams(H0=H0_FID, Om0=OM0_FID)
    survey = SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0,
                          b_miss=1.0, alpha_miss=0.5)
    catalog = _toy_catalog()
    model = get_model("powerlaw+peak")
    theta_fid = np.asarray(get_fixed_population_params("powerlaw+peak"))
    state = prepare_redshift_prior_state("spectral_sirens", cosmo, survey, catalog)

    def log_prior_z(z, pix, cat):
        return eval_redshift_prior_with_state(
            "spectral_sirens", state, z, pix, cosmo, survey, cat
        )

    def make_fn(gw_sel, Ndraw):
        H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

        @jax.jit
        def fn(pop_params):
            def log_weight(m1det, q, dL, chieff, pix, prior_wt, cat):
                dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
                supported = (dL >= dL_lo) & (dL <= dL_hi)
                dL_c = jnp.clip(dL, dL_lo, dL_hi)
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey,
                    pop_params, cat, model.log_p_pop, log_prior_z,
                )
                return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

            log_mu, Neff, _ = compute_selection_term(
                gw_sel, catalog, log_weight, Ndraw, 1,
                sel_batch_size=sel_batch,
            )
            return log_mu, Neff

        return fn

    fn_em = make_fn(_gw_event_from(*pseudo[:4], pseudo[6]), ndraw_em)
    fn_h5 = make_fn(_gw_event_from(*h5_tuple[:4], h5_tuple[6]), ndraw_h5)

    # Latin-hypercube over the end-to-end free parameters (others fiducial).
    # powerlaw+peak vector: (v1, alpha, m_min, m_max, dm_min, dm_max, mu_G,
    # sigma_G, beta, mu_chi, sigma_chi, gamma).
    rng = np.random.default_rng(7)
    box = {1: (1.0, 4.0), 6: (25.0, 45.0), 11: (-4.0, 6.0)}
    U = (rng.permuted(np.tile(np.arange(n_pop), (3, 1)), axis=1).T
         + rng.uniform(size=(n_pop, 3))) / n_pop
    thetas = np.tile(theta_fid, (n_pop, 1))
    for j, (idx, (lo, hi)) in enumerate(box.items()):
        thetas[:, idx] = lo + (hi - lo) * U[:, j]

    rows = []
    for t in thetas:
        lm_e, ne_e = fn_em(jnp.asarray(t))
        lm_h, ne_h = fn_h5(jnp.asarray(t))
        rows.append((float(lm_e), float(ne_e), float(lm_h), float(ne_h)))
    rows = np.array(rows)
    delta = rows[:, 0] - rows[:, 2]
    finite = np.isfinite(delta)
    d = delta[finite]
    neff_ratio = rows[finite, 1] / rows[finite, 3]
    print(f"  {finite.sum()}/{n_pop} draws finite;  "
          f"Delta log mu: mean = {d.mean():.3f}  std = {d.std(ddof=1):.4f}")
    print(f"  Neff emulator/h5: median = {np.median(neff_ratio):.2f}  "
          f"range = [{neff_ratio.min():.2f}, {neff_ratio.max():.2f}]")
    ok = (d.std(ddof=1) < 0.1) and (finite.sum() == n_pop)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (idx, name) in zip(axes, [(1, "alpha_PL"), (6, "mu_G"), (11, "gamma")]):
        ax.scatter(thetas[finite, idx], d, s=12)
        ax.set_xlabel(name)
        ax.set_ylabel(r"$\Delta \log \mu$ (emulator $-$ h5)")
    fig.tight_layout()
    fig.savefig(outdir / "logmu_parity.png", dpi=140)
    plt.close(fig)
    print(f"gate (d) {'PASSED' if ok else 'FAILED'} "
          f"(std < 0.1 nats; constant offset = T_obs/campaign-mix, expected)")
    return {"delta_mean": float(d.mean()), "delta_std": float(d.std(ddof=1)),
            "neff_ratio_median": float(np.median(neff_ratio)),
            "n_finite": int(finite.sum()), "ok": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdet_flow_path", default="PDetO4NF.npz")
    ap.add_argument("--gwselection_path", default="selection_o3o4ab_allsky.h5")
    ap.add_argument("--outdir", default="runs/pdet_validation")
    ap.add_argument("--gwnf_src", default=None,
                    help="Path to gw-nf src/ for the cross-implementation "
                         "p_inj check in gate (a).")
    ap.add_argument("--gates", default="abcd")
    ap.add_argument("--n_pinj", type=int, default=2_000_000)
    ap.add_argument("--leak_nsamp", type=int, default=200_000,
                    help="Flow draws used to estimate the out-of-support mass "
                         "that gate (b) corrects for (default 200k).")
    ap.add_argument("--pdet_nsamp", type=int, default=1_000_000)
    ap.add_argument("--pdet_seed", type=int, default=42)
    ap.add_argument("--n_pop", type=int, default=50)
    ap.add_argument("--sel_batch_size", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}

    flow, config = sel.load_pdet_flow(args.pdet_flow_path)
    xi = float(config["xi"])
    z_grid, z_pdf = sel.build_injection_z_prior()

    if "a" in args.gates:
        gate_a_golden(flow, xi, z_grid, z_pdf, args.gwnf_src)
    if "b" in args.gates:
        results["b"] = gate_b_xi(flow, xi, z_grid, z_pdf, args.n_pinj,
                                 flow_path=args.pdet_flow_path,
                                 leak_nsamp=args.leak_nsamp)

    if "c" in args.gates or "d" in args.gates:
        pseudo = sel.pseudo_injections_from_pdet_flow(
            args.pdet_flow_path, nsamp=args.pdet_nsamp, seed=args.pdet_seed,
        )
        h5_tuple = load_selection_samples(args.gwselection_path)
    if "c" in args.gates:
        results["c"] = gate_c_marginals(pseudo, h5_tuple, outdir)
    if "d" in args.gates:
        results["d"] = gate_d_logmu(
            pseudo[:7], float(pseudo[7]), h5_tuple[:7], float(h5_tuple[7]),
            args.n_pop, args.sel_batch_size, outdir,
        )

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nresults -> {outdir / 'results.json'}")
    if not all(v.get("ok", True) for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
