"""How large is the assumption? -- the numbers that make the PR-8 table readable.

Three measurements, all on the anchors the scan consumes, none of them a gate:

1. **Where the missing budget is.**  The fraction of ``(1 - C(z)) dN_exp(z)``
   above the fitted depth, on the mock's own grid and completeness curve.  PLAN
   §4.2 quotes 99.994% at DESI scale (R1); this is the same number for the
   nside-16 world, and it is why an assumption about ``z > z_depth`` can move
   ``H0`` at all.

2. **How much modulation each ``amp_hi`` injects.**  The per-member spread of
   ``logQ`` over footprint pixels at a few redshifts above the depth, at
   ``b_GW = 1``.  ``amp = 1`` is PLAN §4.2's "factor of e" statement; the scan's
   values are that, scaled.

3. **Where the events live.**  The PE samples' redshift distribution at the two
   ends of the ``H0`` scan, since a modulation above the depth can only move a
   posterior through events whose redshift prior reaches there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PR8 = Path(__file__).resolve().parent
sys.path.insert(0, str(PR8.parent / "pr6a"))

import world16 as W16                                          # noqa: E402


def main():
    import h5py
    from darksirens.likelihood.latent_q import load_latent_plan
    from darksirens.redshift.grid import zgrid
    from darksirens.redshift.selection import c_sel_gaussian
    import jax.numpy as jnp

    z = np.asarray(zgrid)
    z_depth = W16.Z_DEPTH
    above = z > z_depth
    out = {}

    # ---- 1. where the missing budget is -------------------------------------
    # dN_exp ~ (1+z)^delta dV_c/dz on this grid, and the missing fraction is
    # (1 - C) with C the survey's own aggregate completeness curve, relaxed to
    # 0 above the depth exactly as the likelihood does.
    Om0, H0 = W16.OM0, W16.H0_TRUE
    E = np.sqrt(Om0 * (1 + z) ** 3 + 1 - Om0)
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))]) * (299792.458 / H0)
    dV = dc ** 2 / E * (299792.458 / H0)
    C = np.asarray(c_sel_gaussian(jnp.asarray(z), 20.0, W16.M0_HAT,
                                  W16.SIGMA_M, H0, Om0))
    C = np.where(above, 0.0, np.clip(C, 0.0, 1.0))
    miss = (1.0 - C) * dV
    tot = np.trapz(miss, z)
    out["missing_budget_above_depth_frac"] = float(
        np.trapz(np.where(above, miss, 0.0), z) / tot)
    out["missing_budget_above_z0.6_frac"] = float(
        np.trapz(np.where(z > 0.6, miss, 0.0), z) / tot)

    # ---- 2. the injected modulation ----------------------------------------
    rows = []
    for p in sorted((PR8 / "anchors").glob("anchor_*.h5")):
        plan = load_latent_plan(str(p), z_depth=z_depth)
        with h5py.File(p) as f:
            meta = json.loads(f["latent_field"].attrs["basis_meta"])
        rf = np.asarray(plan.row_fac)[:, :-1, :]        # drop the pad row
        phi = np.asarray(plan.phi_z)
        rec = dict(anchor=p.name, amp_hi=meta.get("amp_hi"),
                   amp_kind=meta.get("amp_kind"),
                   support_nodes=int(np.asarray(plan.below_depth).sum()))
        for z_at in (0.2, 0.4, 0.6, 1.0):
            j = int(np.argmin(np.abs(z - z_at)))
            f_pz = rf @ phi[j]                          # (M_draw, n_fit)
            rec[f"sd_field_z{z_at}"] = float(np.std(f_pz))
            # The member-to-member spread at fixed pixel: the part of the
            # modulation that the marginalization actually carries.
            rec[f"member_sd_z{z_at}"] = float(np.mean(np.std(f_pz, axis=0)))
        rows.append(rec)
    out["anchors"] = rows

    # ---- 3. where the events live ------------------------------------------
    from darksirens.utils.cosmology import r_of_z
    d = PR8.parent / "pr6a" / "data" / "rb"
    with h5py.File(d / "gw_events.h5") as f:
        dl = np.asarray(f["dL"][...], dtype=float).reshape(-1)
    ev = {}
    zfine = np.linspace(0.0, 6.0, 60001)
    for h0 in (20.0, 67.74, 140.0):
        # r(z) is monotone, so one interpolation inverts it for every sample.
        rr = np.asarray(r_of_z(jnp.asarray(zfine), h0, Om0))
        zz = np.interp(dl, rr, zfine)
        ev[f"H0_{h0:g}"] = dict(
            median_z=float(np.median(zz)), max_z=float(np.max(zz)),
            frac_above_depth=float(np.mean(zz > z_depth)))
    out["pe_samples"] = ev

    (PR8 / "amp_diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
