"""Is the mock's own PE over-sharp?  A suspect ruled out, with a number.

Tier C's failure is dispersion: the ``H0`` median scatters ~2.6x more than the
posterior's quoted ``sigma``.  The most economical explanation for a CENTRED
estimator with a too-narrow interval is that the per-event likelihood is
over-sharp, and the most economical way for the per-event likelihood to be
over-sharp in a MOCK is for the synthetic PE posteriors to be narrower than the
scatter of the observation around the truth -- ``mock-data-dag``'s standing
pitfall, and one that would make every downstream interval too tight by the
same factor.

The test is the PE's own PP: for each event, the quantile of the TRUE ``dL``
inside that event's ``dL`` posterior samples.  Under a calibrated PE those
quantiles are uniform and the standardized residual ``(dL_true - mean)/sd`` has
unit variance.  A PE too narrow by a factor ``k`` shows up as a residual sd of
``k``, directly comparable with the 2.6 that has to be explained.

**Read the mean and the sd separately.**  The events are DETECTED events, so a
non-zero mean is expected and is not a defect: detection selects upward SNR
fluctuations, which under-estimate ``dL``, so the truth sits systematically
above the PE mean.  That offset is what the hierarchical likelihood's selection
term exists to absorb.  It is the SD that speaks to dispersion.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def measure(path, nobs=None, nsamp=None):
    import h5py
    from scipy import stats

    with h5py.File(path) as f:
        n = int(nobs if nobs is not None else f.attrs["nobs"])
        ns = int(nsamp if nsamp is not None else f.attrs["nsamp"])
        dL = f["dL"][...].reshape(n, ns)
        dl_true = f["truth"]["dl"][...]
        z_true = f["truth"]["z"][...]
        snr = f["truth"]["snr"][...]

    u = np.array([(dL[i] < dl_true[i]).mean() for i in range(n)])
    zs = np.array([(dl_true[i] - dL[i].mean()) / dL[i].std() for i in range(n)])
    ks = stats.kstest(u, "uniform")
    return dict(
        path=str(path), n_events=n, n_samples=ns,
        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
        frac_in_90=float(((u >= 0.05) & (u <= 0.95)).mean()),
        frac_in_68=float(((u >= 0.16) & (u <= 0.84)).mean()),
        n_outside_99=int(((u < 0.005) | (u > 0.995)).sum()),
        resid_mean=float(zs.mean()), resid_sd=float(zs.std(ddof=1)),
        pe_width_misscaling=float(zs.std(ddof=1)),
        median_frac_dl_error=float(np.median(dL.std(axis=1) / dl_true)),
        z_true_median=float(np.median(z_true)), snr_median=float(np.median(snr)),
        quantiles=u.tolist())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gw", default="data/rb/gw_events.h5")
    p.add_argument("--out", default="pe_calibration.json")
    a = p.parse_args(argv)
    r = measure(a.gw)
    print(json.dumps({k: v for k, v in r.items() if k != "quantiles"},
                     indent=2))
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
