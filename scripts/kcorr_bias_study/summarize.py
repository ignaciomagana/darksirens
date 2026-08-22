#!/usr/bin/env python
"""Compact tables from measure_kcorr_anchor_residual.py JSON output."""
from __future__ import annotations

import json
import sys


def main(path):
    r = json.load(open(path))
    cfg = r["config"]
    print(f"# config: ngal={cfg['ngal']} m_lim={cfg['m_lim']} "
          f"z in [{cfg['zlo']},{cfg['zhi']}] kcorr={cfg['kcorr']}")
    if "dm" in r:
        print("\n## Delta DM(z; Theta) - DM(z; fid)  [mag, H0=100]")
        zg = r["dm"]["zgrid"]
        hdr = "  ".join(f"{z:>7g}" for z in zg)
        print(f"{'corner':<22} {hdr}")
        for row in r["dm"]["rows"]:
            vals = "  ".join(f"{row['dDM'][f'{z:g}']:>7.3f}" for z in zg)
            print(f"{row['name']:<22} {vals}")
        print("\n### docstring cross-check")
        for k, v in r["dm"]["docstring_check"].items():
            print(f"  {k:<32} {v:+.4f} mag")
        print(f"\n### spread of Delta DM over z in {r['dm']['zrange']}")
        print(f"{'corner':<22} {'min':>8} {'max':>8} {'ptp':>8}")
        for k, v in r["dm"]["spread_over_zrange"].items():
            print(f"{k:<22} {v['min']:>8.4f} {v['max']:>8.4f} {v['ptp']:>8.4f}")
    if "refit" in r:
        f = r["refit"]["fiducial"]
        print(f"\n## refit: n_gal={r['refit']['n_gal']} "
              f"M0hat_fid={f['M0hat']:.5f} sigma_fid={f['sigma_M']:.5f}")
        print(f"   Laplace sd: M0hat {f['laplace_sd_M0hat']:.5f} mag, "
              f"sigma_M {f['laplace_sd_sigma_M']:.5f} mag")
        zs = r["refit"]["zfit_summary"]
        print(f"   fit-sample z: mean {zs['mean']:.4f} median {zs['median']:.4f} "
              f"p10 {zs['p10']:.4f} p90 {zs['p90']:.4f}")
        print(f"\n{'corner':<22} {'dM0hat':>9} {'dsigma':>9} "
              f"{'<dDM>fit':>9} {'dM0+<dDM>':>10} {'dM0/sd':>9}")
        for row in r["refit"]["rows"]:
            print(f"{row['name']:<22} {row['delta_M0hat']:>+9.4f} "
                  f"{row['delta_sigma_M']:>+9.4f} "
                  f"{row['mean_dDM_over_fit_sample']:>+9.4f} "
                  f"{row['residual_vs_meanshift']:>+10.4f} "
                  f"{row['delta_over_laplace_sd']:>+9.1f}")
    if "csel" in r:
        print("\n## C_sel error from the stale anchor (model - re-anchored)")
        print(f"{'corner':<22} {'dM0hat':>9} {'max|dC|':>9} {'z@max':>7} "
              f"{'C_mod':>7} {'C_cor':>7} {'miss_frac_err':>13}")
        for row in r["csel"]["rows"]:
            fe = row["missing_budget_frac_error"]
            print(f"{row['name']:<22} {row['delta_M0hat']:>+9.4f} "
                  f"{row['max_abs_dC']:>9.4f} {row['z_at_max_dC']:>7.3f} "
                  f"{row['C_model_at_max']:>7.4f} {row['C_correct_at_max']:>7.4f} "
                  f"{fe:>+13.5f}")


if __name__ == "__main__":
    main(sys.argv[1])
