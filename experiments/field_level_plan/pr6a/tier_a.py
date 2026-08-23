"""Tier A -- field recovery, no GW (PLAN §6.2, CPU, nside 16).

    Draw xi_true, Poisson-sample counts, apply the completeness map, run
    count_map_solve, compare logQ_fit vs logQ_truth on voxels with N >= 10.
    Accept: slope 1.00 +- 0.05 (versus the 0.04 prior-collapse signature),
    Pearson r > 0.90, ||xi_hat - xi_true||/sqrt(M) consistent with
    sqrt(tr(H^-1)/M) within 20%.

Three things this script does that the one-line spec does not say, each of
which is required for the numbers to mean what the gate says they mean.

**1. The gauge.**  The count channel is shell-total-CONDITIONED (PLAN §1.1):
``pi_pg`` is invariant under ``eta_pg -> eta_pg + c_g``, so the per-shell
monopole of the field carries ZERO information and the MAP shrinks it to the
prior mean while ``xi_true`` keeps its drawn value.  Regressing the raw
``logQ`` therefore measures the monopole's shrinkage, not the estimator.  The
primary slope is computed on the gauge-fixed field -- each shell's footprint
mean subtracted from BOTH sides, which is exactly the quotient the likelihood
sees -- and the raw slope is reported next to it so the size of the deleted
mode is on the record rather than hidden by the subtraction.

**2. The prior-collapse control.**  PLAN quotes "the 0.04 prior-collapse
signature" as the contrast the slope is read against.  A number with no
contrast is not a test, so the same solve is run on a catalog thinned by
1000x; that arm is the signature, measured here rather than quoted.

**3. The two draws.**  ``representable=False`` (primary) draws Poisson on the
fine grid and thins into shells, so the within-shell linearization of PLAN
§0.5 D1 is live.  ``representable=True`` draws from the fit operator's own
``phi_shell``.  Reporting both means a slope failure can be attributed to the
model class rather than to the solver, instead of being an undifferentiated
"Tier A failed".

The calibration check ``||xi_hat - xi_true||/sqrt(M)`` vs ``sqrt(tr H^-1/M)``
is run over several independent ``xi_true`` draws: the statistic is chi-like
with ``M = 320`` degrees of freedom, so ONE realization carries an intrinsic
~4% spread on the ratio against a 20% gate, and quoting a single draw would
be quoting noise.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import world16 as W16
from world16 import np as _np  # noqa: F401

import jax.numpy as jnp
import jax.scipy.linalg as jsl

from darksirens.redshift.latent_counts import count_map_solve


def _gauge(eta):
    """Subtract each shell's footprint mean -- the quotient ``pi`` sees."""
    return eta - eta.mean(axis=0, keepdims=True)


def _slope_r(fit, truth, mask):
    x = truth[mask]
    y = fit[mask]
    xc = x - x.mean()
    yc = y - y.mean()
    slope = float(np.dot(xc, yc) / np.dot(xc, xc))
    r = float(np.dot(xc, yc) / np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    return slope, r


def one_solve(world, xi_true, rng, *, n_target, representable=False,
              n_iter=13, **draw_kw):
    counts, mu = W16.draw_counts(world, xi_true, rng, n_target=n_target,
                                 representable=representable, **draw_kw)
    op = W16.fit_operator(world, counts)
    # The SHIPPED solve first, always, so its convergence rate at this scale
    # is measured rather than assumed; then the damped solve, whose docstring
    # in world16 records why it has to exist.  P6's pin is grad_inf < 1e-8.
    t0 = time.time()
    shipped = count_map_solve(op, n_iter=n_iter)
    dt_shipped = time.time() - t0
    grad_shipped = float(shipped["grad_inf"])
    t0 = time.time()
    sol = W16.solve_damped(op)
    dt = time.time() - t0
    xi_hat = np.asarray(sol["xi_hat"])
    L = np.asarray(sol["H_chol"])

    # logQ on the SHELL voxels: eta_pg at the fitted xi and at the truth,
    # through the same phi_shell, so the comparison is of the same object.
    phi_shell = np.asarray(world.W) @ np.asarray(world.basis.phi_z_fine)
    ms, mz = world.meta["m_sph"], world.meta["m_z"]
    P = np.asarray(world.basis.proj_sph)
    eta_fit = world.b_gal * (P @ xi_hat.reshape(ms, mz) @ phi_shell.T)
    eta_tru = world.b_gal * (P @ np.asarray(xi_true).reshape(ms, mz)
                             @ phi_shell.T)

    mask = counts >= 10
    mask_all = np.ones_like(counts, dtype=bool)
    out = dict(
        n_gal=float(counts.sum()), n_vox=int(counts.size),
        n_vox_ge10=int(mask.sum()), grad_inf=float(sol["grad_inf"]),
        grad_inf_shipped=grad_shipped,
        shipped_p6_pass=float(grad_shipped < 1e-8),
        n_backtrack=float(sol["n_backtrack"]),
        solve_s=dt, solve_s_shipped=dt_shipped)
    # ``slope_all`` uses EVERY voxel: it is the only column the prior-collapse
    # control can report (a thinned catalog has no voxel with N >= 10), so the
    # contrast PLAN quotes has to be read on a column both arms populate.
    out["slope_all"], out["r_all"] = _slope_r(
        _gauge(eta_fit), _gauge(eta_tru), mask_all)
    if mask.sum() >= 10:
        out["slope_gauge"], out["r_gauge"] = _slope_r(
            _gauge(eta_fit), _gauge(eta_tru), mask)
        out["slope_raw"], out["r_raw"] = _slope_r(eta_fit, eta_tru, mask)
    # calibration: ||xi_hat - xi_true|| / sqrt(M)  vs  sqrt(tr(H^-1)/M)
    M = xi_hat.size
    Linv = np.asarray(jsl.solve_triangular(jnp.asarray(L), jnp.eye(M),
                                           lower=True))
    tr_hinv = float((Linv ** 2).sum())
    out["rms_err"] = float(np.linalg.norm(xi_hat - np.asarray(xi_true))
                           / np.sqrt(M))
    out["rms_pred"] = float(np.sqrt(tr_hinv / M))
    out["ratio"] = out["rms_err"] / out["rms_pred"]
    out["tr_Hinv_over_M"] = tr_hinv / M
    return out, dict(counts=counts, xi_hat=xi_hat, L=L, mask=mask)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=6100)
    p.add_argument("--n-real", type=int, default=5)
    p.add_argument("--n-target", type=float, default=1.1e6)
    p.add_argument("--thin", type=float, default=1000.0,
                   help="prior-collapse control: catalog thinned by this")
    p.add_argument("--out", default="tier_a.json")
    args = p.parse_args(argv)

    world = W16.build_world()
    print(f"[world] {world.meta}", flush=True)
    print(f"[world] footprint {world.f_p.size} pixels, f_p mean "
          f"{world.f_p.mean():.4f} sd {world.f_p.std():.4f} "
          f"min {world.f_p.min():.4f}", flush=True)

    res = {"world": world.meta, "arms": {}}
    res["f_p"] = dict(n=int(world.f_p.size), mean=float(world.f_p.mean()),
                      sd=float(world.f_p.std()), min=float(world.f_p.min()))

    arms = [("fine", dict(representable=False, n_target=args.n_target)),
            ("representable", dict(representable=True,
                                   n_target=args.n_target))]
    for name, kw in arms:
        rows = []
        for k in range(args.n_real):
            rng = np.random.default_rng(args.seed + 977 * k)
            xi_true = W16.draw_xi_true(world, rng)
            row, _ = one_solve(world, xi_true, rng, **kw)
            rows.append(row)
            print(f"[{name}] real {k}: N={row['n_gal']:.3g} "
                  f"vox>=10 {row['n_vox_ge10']}/{row['n_vox']} "
                  f"slope_g={row.get('slope_gauge', float('nan')):.4f} "
                  f"r_g={row.get('r_gauge', float('nan')):.4f} "
                  f"slope_raw={row.get('slope_raw', float('nan')):.4f} "
                  f"ratio={row['ratio']:.4f} grad={row['grad_inf']:.1e} "
                  f"grad_shipped={row['grad_inf_shipped']:.1e} "
                  f"({row['solve_s']:.1f}s)", flush=True)
        agg = {}
        for key in rows[0]:
            v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            agg[key] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                            values=[float(x) for x in v])
        res["arms"][name] = agg

    # ------------------------------------------------ prior-collapse control
    # PLAN quotes "the 0.04 prior-collapse signature" from §11: "any argument
    # that reaches for a smaller M is reaching for the prior-collapse regime
    # the hard radial guard exists to refuse (measured fitted-vs-truth logQ
    # slope 0.04)".  It is therefore a RANK/SCALE signature, not a
    # sample-size one, so the control is the rank-starved fit, not a thinned
    # catalog: the same 1.1e6-galaxy data fitted with M = 12 x 2 = 24 modes
    # (ls_sph = 1.5, ls_z = 0.30 -- still guard-compliant, just far too
    # coarse) against a truth living in the M = 320 basis.  The count ladder
    # is reported next to it so both axes of collapse are on the record.
    res["controls"] = {}
    coarse = W16.build_world(m_sph=12, m_z=2, ls_sph=1.5, ls_z=0.30)
    ps_c = np.asarray(coarse.W) @ np.asarray(coarse.basis.phi_z_fine)
    ps_t = np.asarray(world.W) @ np.asarray(world.basis.phi_z_fine)
    rows = []
    for k in range(args.n_real):
        rng = np.random.default_rng(args.seed + 977 * k)
        xi_true = W16.draw_xi_true(world, rng)
        counts, _ = W16.draw_counts(world, xi_true, rng,
                                    n_target=args.n_target)
        op = W16.fit_operator(coarse, counts)
        sol = W16.solve_damped(op)
        xh = np.asarray(sol["xi_hat"]).reshape(12, 2)
        ef = np.asarray(coarse.basis.proj_sph) @ xh @ ps_c.T
        et = np.asarray(world.basis.proj_sph) @ np.asarray(
            xi_true).reshape(64, 5) @ ps_t.T
        s, r = _slope_r(_gauge(ef), _gauge(et), np.ones_like(counts, bool))
        s10, r10 = _slope_r(_gauge(ef), _gauge(et), counts >= 10)
        rows.append(dict(slope_all=s, r_all=r, slope_ge10=s10, r_ge10=r10))
        print(f"[collapse M=24] real {k}: slope_all={s:.4f} r={r:.4f} "
              f"slope_ge10={s10:.4f}", flush=True)
    res["controls"]["rank_starved_M24"] = {
        k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    ladder = []
    for nt in (1.1e6, 1.1e5, 1.1e4, 1.1e3, 1.1e2, 1.1e1):
        ss = []
        for k in range(min(args.n_real, 3)):
            rng = np.random.default_rng(args.seed + 977 * k)
            xi_true = W16.draw_xi_true(world, rng)
            counts, _ = W16.draw_counts(world, xi_true, rng, n_target=nt)
            op = W16.fit_operator(world, counts)
            sol = W16.solve_damped(op)
            ef = np.asarray(world.basis.proj_sph) @ np.asarray(
                sol["xi_hat"]).reshape(64, 5) @ ps_t.T
            et = np.asarray(world.basis.proj_sph) @ np.asarray(
                xi_true).reshape(64, 5) @ ps_t.T
            ss.append(_slope_r(_gauge(ef), _gauge(et),
                               np.ones_like(counts, bool))[0])
        ladder.append(dict(n_target=float(nt), slope_all=float(np.mean(ss))))
        print(f"[ladder] N~{nt:.2g}: slope_all={np.mean(ss):.4f}", flush=True)
    res["controls"]["count_ladder"] = ladder

    # ---------------------------------------------------------- the verdict
    a = res["arms"]["fine"]
    verdict = dict(
        slope=dict(value=a["slope_gauge"]["mean"], sd=a["slope_gauge"]["sd"],
                   gate="1.00 +- 0.05",
                   pass_=bool(abs(a["slope_gauge"]["mean"] - 1.0) <= 0.05)),
        pearson=dict(value=a["r_gauge"]["mean"], gate="> 0.90",
                     pass_=bool(a["r_gauge"]["mean"] > 0.90)),
        calibration=dict(value=a["ratio"]["mean"], sd=a["ratio"]["sd"],
                         gate="within 20% of 1",
                         pass_=bool(abs(a["ratio"]["mean"] - 1.0) <= 0.20)),
        prior_collapse_control=dict(
            rank_starved_M24=res["controls"]["rank_starved_M24"],
            count_ladder=res["controls"]["count_ladder"],
            reference="PLAN §6.2/§11 quote 0.04 as the collapse signature"),
        slope_all_fine=a["slope_all"]["mean"],
        representable_slope=res["arms"]["representable"]["slope_gauge"]["mean"],
        shipped_solver_p6_rate=a["shipped_p6_pass"]["mean"],
    )
    verdict["TIER_A"] = bool(verdict["slope"]["pass_"]
                             and verdict["pearson"]["pass_"]
                             and verdict["calibration"]["pass_"])
    res["verdict"] = verdict
    print(json.dumps(verdict, indent=2))
    with open(W16.PR6A_DIR / args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / args.out}")


if __name__ == "__main__":
    main()
