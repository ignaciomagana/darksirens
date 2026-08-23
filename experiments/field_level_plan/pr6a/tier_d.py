"""Tier D -- misspecification at MEASURED amplitudes (PLAN §6.2).

    (i)   completeness map perturbed by the measured masked_frac sd 0.104;
    (ii)  a z- and density-dependent unmodelled incompleteness at 5%
          amplitude (the fibre-assignment proxy);
    (iii) ls_ang wrong by 2x;
    (iv)  a lognormal truth with a non-Gaussian tail.
    Accept: H0 bias < 0.5 sigma; report each as a systematic.

**Each stress is applied on one side only, and which side is the whole
content of the test.**

* (i) the SURVEY masks with ``f_p + N(0, 0.104)`` while the inference keeps
  reading the unperturbed ``mth_map`` -- so ``C_p = f_p C(z)`` and the count
  channel's ``log f_p`` are both wrong by the measured scatter.  0.104 is the
  DESI ``masked_frac`` sd of PLAN §1.2's table (this repo's own map gives
  0.1087 at nside 64); at nside 16 the map's OWN sd is only 0.045, so this
  perturbation is 2.3x the structure it perturbs, which is what PLAN means by
  "50-100% of the signal".
* (ii) the survey drops galaxies with probability ``0.05 (z/z_depth) tanh
  (logQ)`` -- lower completeness in dense regions at high ``z``, the shape a
  fibre-assignment residual has.  It acts at the SURVEY step, so the galaxy
  universe and therefore the GW hosts are untouched: it is an incompleteness,
  not a different universe.
* (iii) only the ANCHOR's basis changes (``ls_sph = 1.0`` against the truth's
  0.5).  The mock is bit-identical to the matched arm at the same seed, which
  makes this the one perfectly paired stress of the four.
* (iv) only ``xi_true`` changes, to a variance-preserving skewed mixture
  (``world16.draw_xi_true``'s docstring gives the normalization and why it
  matters).  The realization is completely different from the matched arm at
  the same seed, so this one is unpaired by construction.

**The resolution of the test is quoted with the test.**  A single realization's
``H0`` bias has ~1 sigma of scatter by construction -- that is what Tier C's
coverage means -- so a 0.5 sigma gate on a 1-realization median would be
noise.  Each stress runs ``--n-real`` seeds and the reported statistic is the
MEDIAN bias with its bootstrap standard error; at the default 20 seeds the SE
is ~0.28 sigma, i.e. the test resolves a 0.5 sigma systematic at about 1.8x
its own error and no better.  Both numbers are in the output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import world16 as W16
import build_anchor16
import make_mock
import tier_b

MASKED_FRAC_SD = 0.104          # PLAN §1.2's measured DESI value
FIBRE_AMPLITUDE = 0.05
LS_ANG_FACTOR = 2.0
TAIL_AMPLITUDE = 0.5


def _fibre(z, logq):
    """Tier D-ii: 5% z- and density-dependent unmodelled incompleteness."""
    return 1.0 - FIBRE_AMPLITUDE * np.clip(z / W16.Z_DEPTH, 0.0, 1.0) \
        * np.tanh(logq)


def stress_kwargs(name, world, seed):
    """(mock kwargs, anchor world) for one stress."""
    if name == "matched":
        return {}, world
    if name == "fp_perturbed":
        rng = np.random.default_rng(1_000_000 + seed)
        fp = np.clip(np.asarray(world.f_p)
                     + rng.normal(0.0, MASKED_FRAC_SD, world.f_p.size),
                     0.02, 1.0)
        return dict(f_p_survey=fp), world
    if name == "fibre_5pc":
        return dict(extra_selection=_fibre), world
    if name == "ls_ang_2x":
        return {}, W16.build_world(ls_sph_fit=W16.LS_SPH * LS_ANG_FACTOR)
    if name == "lognormal_tail":
        return dict(lognormal_tail=TAIL_AMPLITUDE), world
    raise ValueError(name)


def one(seed, workdir, world, anchor_world, *, grid, n0, injections,
        arm_names, **mock_kw):
    d = Path(workdir)
    make_mock.build(seed, d, world=world, n0=n0, verbose=False,
                    reuse_injections=injections, **mock_kw)
    build_anchor16.build(survey=d / "catalog_pixelated_nside_16.h5",
                         mth_map=d / "mth_map_nside16.h5",
                         out=d / "latent_anchor.h5", world=anchor_world,
                         verbose=False)
    res = tier_b.run(d, grid=grid, arm_names=arm_names, quiet=True)
    for f in d.glob("*"):
        f.unlink()
    d.rmdir()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-real", type=int, default=20)
    p.add_argument("--seed0", type=int, default=50000)
    p.add_argument("--h0-step", type=float, default=2.5)
    p.add_argument("--n0", type=float, default=5e-5)
    p.add_argument("--injections", default="data/injections.h5")
    p.add_argument("--stresses", nargs="*", default=[
        "matched", "fp_perturbed", "fibre_5pc", "ls_ang_2x",
        "lognormal_tail"])
    p.add_argument("--arms", nargs="*", default=["latent"])
    p.add_argument("--out", default="tier_d.json")
    a = p.parse_args(argv)

    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    world = W16.build_world()
    out = {"stresses": {}, "config": dict(
        n_real=a.n_real, seed0=a.seed0, h0_step=a.h0_step, n0=a.n0,
        masked_frac_sd=MASKED_FRAC_SD, fibre_amplitude=FIBRE_AMPLITUDE,
        ls_ang_factor=LS_ANG_FACTOR, tail_amplitude=TAIL_AMPLITUDE,
        arms=a.arms)}
    t0 = time.time()
    for name in a.stresses:
        rows = []
        for k in range(a.n_real):
            seed = a.seed0 + 37 * k
            mk, aw = stress_kwargs(name, world, seed)
            r = one(seed, W16.PR6A_DIR / "data" / f"d_{name}_{k:03d}",
                    world, aw, grid=grid, n0=a.n0,
                    injections=a.injections, arm_names=tuple(a.arms), **mk)
            rows.append({arm: {kk: r[arm][kk] for kk in
                               ("median", "sigma", "cdf_at_truth", "width90")}
                         for arm in a.arms} | {"seed": seed})
            print(f"[tier D] {name} {k + 1}/{a.n_real}: "
                  + "  ".join(f"{arm} H0={r[arm]['median']:.1f} "
                              f"bias={(r[arm]['median'] - W16.H0_TRUE) / r[arm]['sigma']:+.2f}s"
                              for arm in a.arms)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        out["stresses"][name] = {"rows": rows}
        with open(W16.PR6A_DIR / a.out, "w") as f:
            json.dump(out, f, indent=1)

    rng = np.random.default_rng(0)
    for name, blk in out["stresses"].items():
        for arm in a.arms:
            b = np.array([(r[arm]["median"] - W16.H0_TRUE) / r[arm]["sigma"]
                          for r in blk["rows"]])
            boot = np.median(rng.choice(b, (2000, b.size)), axis=1)
            blk.setdefault("summary", {})[arm] = dict(
                n=int(b.size), median_bias_sigma=float(np.median(b)),
                se_bootstrap=float(boot.std()),
                mean_bias_sigma=float(b.mean()),
                sd_bias_sigma=float(b.std()),
                pass_0p5sigma=bool(abs(np.median(b)) < 0.5))
    # Paired shift against the matched arm at the same seeds.
    if "matched" in out["stresses"]:
        base = {r["seed"]: r for r in out["stresses"]["matched"]["rows"]}
        for name, blk in out["stresses"].items():
            if name == "matched":
                continue
            for arm in a.arms:
                d = np.array([r[arm]["median"] - base[r["seed"]][arm]["median"]
                              for r in blk["rows"] if r["seed"] in base])
                s = np.array([r[arm]["sigma"] for r in blk["rows"]
                              if r["seed"] in base])
                blk["summary"][arm]["paired_shift_sigma_median"] = float(
                    np.median(d / s))
                blk["summary"][arm]["paired_shift_sigma_se"] = float(
                    (d / s).std() / np.sqrt(d.size))
    out["verdict"] = {
        name: blk["summary"] for name, blk in out["stresses"].items()}
    out["verdict"]["TIER_D"] = bool(all(
        blk["summary"][arm]["pass_0p5sigma"]
        for name, blk in out["stresses"].items() if name != "matched"
        for arm in a.arms))
    print(json.dumps(out["verdict"], indent=2))
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
