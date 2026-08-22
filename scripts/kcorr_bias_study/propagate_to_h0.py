#!/usr/bin/env python
"""Propagate the stale selection-fit anchor into an H0 (and w0) bias.

Review task #18 measurement, part 2.  READ-ONLY with respect to
``darksirens``: the real :func:`c_sel_gaussian`, :func:`fit_selection_from_mags`
and :func:`darksirens.utils.cosmology.dL_of_z` are used unmodified.

Model (a line-of-sight dark-siren likelihood; approximations stated in the
report and in --help)
---------------------------------------------------------------------------
Host-redshift prior, exactly the completion mixture that
``darksirens.redshift.completion`` forms under ``c_mode="selection"``:

    p_host(z | theta_sel, Theta) ~ n_obs(z) + (1 - C_sel(z; theta_sel, Theta))
                                              * dN_exp(z; Theta)

with ``n_obs`` the OBSERVED catalog's redshift histogram (data, fixed) and
``dN_exp ~ dV_c/dz`` the homogeneous expectation at the proposal cosmology.
Per event, ``p(dL_obs | z) = N(dL_obs; dL(z; H0, Theta), sigma_dL)``, and the
detection normalization is ``beta = int dz p_host(z) P_det(dL(z))`` with a
hard ``dL <= dL_max`` threshold (so beta is exact, not estimated).

The two models compared
-----------------------
* ``stamped``     : ``C_sel`` uses the SAMPLED ``M0hat``, whose Gaussian prior
                    is centred at the fiducial fit value for every Theta --
                    the shipped behaviour.
* ``reanchored``  : identical, except the curve uses
                    ``M0hat + delta(Theta)``, with ``delta(Theta)`` the
                    measured re-fit shift of the zero point at Theta (option
                    (a), the exact per-sample re-anchor).
* ``firstorder_meandm`` : the curve uses ``M0hat - <dDM(Theta)>_fit`` --
                    option (b) spelled the obvious way, a shift by the
                    fit-sample mean distance-modulus change.
* ``firstorder_mle``    : the curve uses ``M0hat + sum_j s_j (Theta_j -
                    Theta_j^fid)`` with ``s_j = dM0hat_MLE/dTheta_j`` measured
                    by central differences of the ACTUAL offline fit at the
                    fiducial -- option (b) done right (6 extra offline fits,
                    zero per-likelihood-call cost).

Everything else (data, priors, grids) is identical, so the difference in the
recovered H0/w0 posteriors IS the induced bias.

Usage
-----
    JAX_PLATFORMS=cpu python scripts/kcorr_bias_study/propagate_to_h0.py \
        --out /path/to/h0_bias.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from darksirens.redshift.selection import (  # noqa: E402
    H0_REF,
    c_sel_gaussian,
    fit_selection_from_mags,
    k_of_z,
)
from darksirens.utils.cosmology import (  # noqa: E402
    Om0Planck,
    Om0PriorLower,
    Om0PriorUpper,
    dL_of_z,
    distance_modulus,
    w0Fiducial,
    w0PriorLower,
    w0PriorUpper,
    waFiducial,
    waPriorLower,
    waPriorUpper,
)

FID = dict(Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)


def dm100(z, Om0, w0, wa):
    return np.asarray(distance_modulus(np.asarray(z, float), H0_REF,
                                       Om0, w0, wa), float)


def dl100(z, Om0, w0, wa):
    """dL(z) at H0 = 100; dL at any H0 is this times 100/H0 (exact)."""
    return np.asarray(dL_of_z(np.asarray(z, float), H0_REF, Om0, w0, wa),
                      float)


def _dN_exp(z, Om0, w0, wa):
    """Homogeneous comoving-volume expectation dV_c/dz (arbitrary scale)."""
    chi = dl100(z, Om0, w0, wa) / (1.0 + z)
    return np.maximum(chi ** 2 * np.gradient(chi, z), 0.0)


def _smooth(y, nbins):
    """Gaussian smoothing over ``nbins`` grid cells (shot-noise control)."""
    if nbins <= 0:
        return np.asarray(y, float)
    k = np.arange(-4 * nbins, 4 * nbins + 1)
    g = np.exp(-0.5 * (k / float(nbins)) ** 2)
    g /= g.sum()
    return np.convolve(np.asarray(y, float), g, mode="same")


# ------------------------------------------------------------------ the mock
def build_mock(a, rng):
    """Complete galaxy catalog + magnitude-limited observed catalog + events.

    Galaxies follow the comoving volume element of the TRUE cosmology times a
    lognormal line-of-sight density contrast (so the catalog has genuine
    radial structure -- a perfectly smooth n(z) carries no H0 information at
    all and the comparison would be vacuous).
    """
    zg = np.linspace(a.zlo, a.zhi, 4000)
    # dV_c/dz ~ chi^2 / E(z); chi = dL/(1+z) at H0=100 in the true cosmology.
    chi = dl100(zg, **FID) / (1.0 + zg)
    dchi = np.gradient(chi, zg)
    dV = np.maximum(chi ** 2 * dchi, 0.0)
    # Lognormal radial contrast, correlation length ~ a few percent in z.
    nk = 24
    ph = rng.uniform(0, 2 * np.pi, nk)
    amp = rng.standard_normal(nk) / np.arange(1, nk + 1) ** 0.7
    u = (zg - a.zlo) / (a.zhi - a.zlo)
    g = sum(amp[k] * np.cos(2 * np.pi * (k + 1) * u + ph[k]) for k in range(nk))
    g = g / g.std()
    contrast = np.exp(a.sigma_lss * g - 0.5 * a.sigma_lss ** 2)
    pz = dV * contrast
    pz = pz / np.trapz(pz, zg)

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pz[1:] + pz[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    z_all = np.interp(rng.uniform(0, 1, a.ngal), cdf, zg)

    kc = tuple(a.kcorr)
    M_all = a.M0hat_true + a.sigma_true * rng.standard_normal(a.ngal)
    kz = k_of_z(z_all, kc, xp=np)
    m_all = M_all + dm100(z_all, **FID) + (kz if kz is not None else 0.0)
    obs = m_all <= a.m_lim
    z_obs, m_obs = z_all[obs], m_all[obs]

    # GW hosts: uniform over ALL galaxies (catalogued or not), detected iff
    # dL(z; H0_true) <= dL_max.  A hard cut makes beta exact.
    dl_true_all = dl100(z_all, **FID) * (H0_REF / a.H0_true)
    det = dl_true_all <= a.dl_max
    idx = np.nonzero(det)[0]
    if idx.size < a.nevents:
        raise RuntimeError(f"only {idx.size} detectable hosts; lower --nevents")
    pick = rng.choice(idx, size=a.nevents, replace=False)
    dl_ev = dl_true_all[pick]
    dl_obs = dl_ev * np.exp(a.dl_frac_err * rng.standard_normal(a.nevents))
    return dict(zgrid=zg, pz_true=pz, z_all=z_all, z_obs=z_obs, m_obs=m_obs,
                z_host=z_all[pick], dl_obs=dl_obs,
                n_obs=int(z_obs.size), n_all=int(z_all.size))


# --------------------------------------------------------------- the anchors
def anchor_table(mock, a, thetas):
    """delta(Theta) by ACTUAL re-fit, plus the two cheap approximations.

    Returns ``(base, laplace_sd, delta, meandd, lin)`` where

    * ``delta``  : exact per-proposal re-anchor -- option (a).
    * ``meandd`` : ``<Delta DM(Theta)>`` over the fit sample.  The NAIVE
      option (b): shift the zero point by minus the mean distance-modulus
      change.
    * ``lin``    : the SMART option (b): a linear model in
      ``(dOm0, dw0, dwa)`` whose three slopes are central finite differences
      of the ACTUAL MLE at the fiducial (6 extra offline fits, zero
      per-likelihood-call cost).
    """
    kc = tuple(a.kcorr) or None

    def _fit(Om0, w0, wa):
        return fit_selection_from_mags(mock["m_obs"], mock["z_obs"], a.m_lim,
                                       family="gaussian", k_corr_coeffs=kc,
                                       Om0=Om0, w0=w0, wa=wa)

    base = _fit(**FID)
    zfit = mock["z_obs"][mock["z_obs"] >= 0.01]
    base_dm = dm100(zfit, **FID)

    # Central-difference MLE slopes at the fiducial, for BOTH sampled
    # selection parameters: sigma_M's prior centre is stale for exactly the
    # same reason M0hat's is.
    steps = dict(Om0=0.05, w0=0.5, wa=1.0)
    slope, slope_s = {}, {}
    for key, h in steps.items():
        hi, lo = dict(FID), dict(FID)
        hi[key] += h
        lo[key] -= h
        fh, fl = _fit(**hi), _fit(**lo)
        slope[key] = (fh.M0hat - fl.M0hat) / (2.0 * h)
        slope_s[key] = (fh.sigma_M - fl.sigma_M) / (2.0 * h)

    delta = np.zeros(len(thetas))
    dsig = np.zeros(len(thetas))
    meandd = np.zeros(len(thetas))
    lin = np.zeros(len(thetas))
    lin_s = np.zeros(len(thetas))
    for i, (Om0, w0, wa) in enumerate(thetas):
        f = _fit(Om0, w0, wa)
        d = (Om0 - FID["Om0"], w0 - FID["w0"], wa - FID["wa"])
        delta[i] = float(f.M0hat - base.M0hat)
        dsig[i] = float(f.sigma_M - base.sigma_M)
        meandd[i] = float((dm100(zfit, Om0, w0, wa) - base_dm).mean())
        lin[i] = sum(slope[k] * v for k, v in zip(("Om0", "w0", "wa"), d))
        lin_s[i] = sum(slope_s[k] * v for k, v in zip(("Om0", "w0", "wa"), d))
    cov = np.asarray(base.cov, float)
    return (base, float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])),
            delta, dsig, meandd, lin, lin_s,
            {k: float(v) for k, v in slope.items()},
            {k: float(v) for k, v in slope_s.items()})


# ---------------------------------------------------------------- likelihood
def run_grid(mock, a, base, sd_M0hat, thetas, delta, dsig, meandd, lin,
             lin_s, mode):
    """log-posterior on (H0, Om0, w0, wa, M0hat) for one anchoring mode."""
    # Observed-catalog redshift histogram on the same grid (data; fixed).
    hist, edges = np.histogram(mock["z_obs"], bins=a.nz, range=(a.zlo, a.zhi))
    n_obs_z = hist / np.diff(edges)
    zc = 0.5 * (edges[1:] + edges[:-1])

    H0 = np.linspace(a.h0_lo, a.h0_hi, a.nh0)
    # nm0 = 1 PINS M0hat at the fit value: the large-catalog limit in
    # which the Laplace sd -> 0 and the sampler has no freedom to
    # absorb any part of the stale anchor.
    M0 = (np.array([base.M0hat]) if a.nm0 == 1 else
          base.M0hat + sd_M0hat * np.linspace(-4, 4, a.nm0))
    logprior_M0 = -0.5 * ((M0 - base.M0hat) / sd_M0hat) ** 2

    dl_obs = mock["dl_obs"][:, None]                 # (E, 1)
    sig = a.dl_frac_err                              # lognormal width

    # LSS-conditioned missing-budget shape, the toy's stand-in for the
    # lognormal Q table.  Built ONCE, at the FIDUCIAL anchor, exactly as
    # build_lognormal_completion.py stamps it: Q does not move with the
    # proposal, only C_sel does.  Without it the out-of-catalog budget is
    # smooth while the true missing galaxies are clustered, and the H0
    # posterior is dominated by that (unrelated) mismatch.
    dN_exp_fid = _dN_exp(zc, **FID)
    C_fid = np.asarray(c_sel_gaussian(zc, a.m_lim, base.M0hat, base.sigma_M,
                                      70.0, k_corr_coeffs=tuple(a.kcorr) or None,
                                      **FID), float)
    Qz = _smooth(n_obs_z, a.q_smooth_bins) / np.maximum(C_fid * dN_exp_fid, 1e-30)
    Qz = Qz / np.maximum(np.average(Qz, weights=dN_exp_fid), 1e-30)

    dz = np.gradient(zc)                             # trapz weights (uniform)
    nev = mock["dl_obs"].size
    logpost = np.full((a.nh0, len(thetas), a.nm0), -np.inf)
    for it, (Om0, w0, wa) in enumerate(thetas):
        dN_exp = _dN_exp(zc, Om0, w0, wa) * Qz
        dl100_z = dl100(zc, Om0, w0, wa)
        # (M, Z) host priors -- one row per sampled M0hat, built ONCE per
        # Theta because C_sel carries no H0 (the firewall).
        p_host = np.empty((a.nm0, zc.size))
        ok = np.ones(a.nm0, dtype=bool)
        for im, m0 in enumerate(M0):
            sig_eff = base.sigma_M
            if mode == "stamped":
                m0_eff = m0
            elif mode == "reanchored":
                m0_eff = m0 + delta[it]
                sig_eff = base.sigma_M + dsig[it]
            elif mode == "firstorder_meandm":
                m0_eff = m0 - meandd[it]
            elif mode == "firstorder_mle":
                m0_eff = m0 + lin[it]
                sig_eff = base.sigma_M + lin_s[it]
            else:
                raise ValueError(mode)
            C = np.asarray(c_sel_gaussian(zc, a.m_lim, m0_eff, sig_eff,
                                          70.0, Om0, w0, wa,
                                          k_corr_coeffs=tuple(a.kcorr) or None),
                           float)
            # Calibrate the homogeneous expectation against the counts the
            # catalog actually has: the completion's own normalization is
            # ``int C(z) dN_exp dz = N_obs`` (this is what the sampled
            # log10n0 does).  It depends on C, hence on the anchor -- which
            # is precisely the channel the stale zero point acts through.
            denom = float((C * dN_exp * dz).sum())
            row = n_obs_z + a.miss_scale * (1.0 - C) * dN_exp * (
                float(mock["n_obs"]) / denom if denom > 0 else 0.0)
            s = float((row * dz).sum())
            if not (denom > 0 and s > 0):
                ok[im] = False
                row = np.ones_like(row)
                s = float((row * dz).sum())
            p_host[im] = row / s
        for ih, h0 in enumerate(H0):
            dl = dl100_z * (H0_REF / h0)                    # (Z,)
            # Detection is deterministic in the TRUE dL, so the
            # detectability indicator belongs in the per-event numerator as
            # well as in beta: only hosts inside the horizon can produce a
            # detected event.  (Omitting it makes -N log beta an
            # uncompensated reward for small horizons and biases H0 low.)
            det = (dl <= a.dl_max).astype(float)
            w = (p_host * det[None, :]) * dz[None, :]        # (M, Z)
            beta = w.sum(axis=1)                             # (M,)
            ll = np.exp(-0.5 * (np.log(dl_obs / dl[None, :]) / sig) ** 2)
            num = ll @ w.T                                   # (E, M)
            good = ok & (beta > 0) & np.all(num > 0, axis=0)
            val = (np.log(np.where(num > 0, num, 1.0)).sum(axis=0)
                   - nev * np.log(np.where(beta > 0, beta, 1.0))
                   + logprior_M0)
            logpost[ih, it] = np.where(good, val, -np.inf)
    return H0, M0, logpost


def marginals(H0, logpost):
    p = np.exp(logpost - np.nanmax(logpost))
    p = np.where(np.isfinite(p), p, 0.0)
    pH = p.sum(axis=(1, 2))
    pH /= np.trapz(pH, H0)
    mean = float(np.trapz(H0 * pH, H0))
    var = float(np.trapz((H0 - mean) ** 2 * pH, H0))
    return dict(mean=mean, sd=float(np.sqrt(var)),
                mode=float(H0[int(np.argmax(pH))]),
                pdf=[float(x) for x in pH])


def theta_marginal(thetas, axis_vals, logpost, which):
    """Posterior mean/sd of one Theta component."""
    p = np.exp(logpost - np.nanmax(logpost))
    p = np.where(np.isfinite(p), p, 0.0)
    w = p.sum(axis=(0, 2))                      # weight per theta point
    v = np.array([t[which] for t in thetas])
    tot = w.sum()
    mean = float((w * v).sum() / tot)
    sd = float(np.sqrt((w * (v - mean) ** 2).sum() / tot))
    return dict(mean=mean, sd=sd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ngal", type=int, default=120000)
    p.add_argument("--nevents", type=int, default=40)
    p.add_argument("--m-lim", dest="m_lim", type=float, default=19.0)
    p.add_argument("--M0hat-true", dest="M0hat_true", type=float, default=-20.5)
    p.add_argument("--sigma-true", dest="sigma_true", type=float, default=0.9)
    p.add_argument("--kcorr", type=float, nargs="*", default=[2.0, -1.0])
    p.add_argument("--zlo", type=float, default=0.02)
    p.add_argument("--zhi", type=float, default=0.5)
    p.add_argument("--sigma-lss", type=float, default=0.7)
    p.add_argument("--H0-true", dest="H0_true", type=float, default=70.0)
    p.add_argument("--dl-max", dest="dl_max", type=float, default=1500.0)
    p.add_argument("--dl-frac-err", dest="dl_frac_err", type=float, default=0.15)
    p.add_argument("--miss-scale", dest="miss_scale", type=float, default=1.0,
                   help="Relative weight of the out-of-catalog budget "
                        "(1.0 = the completion's own normalization).")
    p.add_argument("--q-smooth-bins", dest="q_smooth_bins", type=int, default=3,
                   help="Gaussian smoothing (in z-grid cells) of the frozen "
                        "LSS Q shape; the toy stand-in for the lognormal "
                        "completion table's finite resolution.")
    p.add_argument("--nz", type=int, default=240)
    p.add_argument("--nh0", type=int, default=71)
    p.add_argument("--nm0", type=int, default=9)
    p.add_argument("--nom0", type=int, default=5)
    p.add_argument("--nw0", type=int, default=5)
    p.add_argument("--nwa", type=int, default=5)
    p.add_argument("--h0-lo", dest="h0_lo", type=float, default=45.0)
    p.add_argument("--h0-hi", dest="h0_hi", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    mock = build_mock(a, rng)

    om = np.linspace(Om0PriorLower, Om0PriorUpper, a.nom0)
    w0 = np.linspace(w0PriorLower, w0PriorUpper, a.nw0)
    wa = np.linspace(waPriorLower, waPriorUpper, a.nwa)
    thetas = [(float(o), float(x), float(y)) for o in om for x in w0 for y in wa]

    (base, sd_M0hat, sd_sigma, delta, dsig, meandd, lin, lin_s,
     slope, slope_s) = anchor_table(mock, a, thetas)
    print(f"[mock] n_all={mock['n_all']} n_obs={mock['n_obs']} "
          f"n_events={a.nevents}", file=sys.stderr)
    print(f"[fit ] M0hat={base.M0hat:.5f} sigma_M={base.sigma_M:.5f} "
          f"laplace_sd={sd_M0hat:.5f}", file=sys.stderr)
    print(f"[delta] range {delta.min():+.4f} .. {delta.max():+.4f} mag "
          f"({delta.min()/sd_M0hat:+.1f} .. {delta.max()/sd_M0hat:+.1f} sd)",
          file=sys.stderr)

    res = {"config": vars(a),
           "mock": {k: mock[k] for k in ("n_all", "n_obs")},
           "fit": dict(M0hat=float(base.M0hat), sigma_M=float(base.sigma_M),
                       laplace_sd_M0hat=sd_M0hat),
           "delta_range": [float(delta.min()), float(delta.max())],
           "mle_slopes": slope,
           "mle_slopes_sigma": slope_s,
           "laplace_sd_sigma_M": sd_sigma,
           "dsigma_range": [float(dsig.min()), float(dsig.max())],
           "dsigma_over_laplace_sd": [float(dsig.min() / sd_sigma),
                                      float(dsig.max() / sd_sigma)],
           "approx_residual_mag": {
               "firstorder_meandm": {
                   "max_abs": float(np.abs(delta + meandd).max()),
                   "max_abs_over_laplace_sd":
                       float(np.abs(delta + meandd).max() / sd_M0hat)},
               "firstorder_mle": {
                   "max_abs": float(np.abs(delta - lin).max()),
                   "max_abs_over_laplace_sd":
                       float(np.abs(delta - lin).max() / sd_M0hat)},
               "stamped": {
                   "max_abs": float(np.abs(delta).max()),
                   "max_abs_over_laplace_sd":
                       float(np.abs(delta).max() / sd_M0hat)}},
           "models": {}}
    ref = None
    for mode in ("reanchored", "stamped", "firstorder_meandm",
                 "firstorder_mle"):
        H0, M0, lp = run_grid(mock, a, base, sd_M0hat, thetas, delta, dsig,
                              meandd, lin, lin_s, mode)
        m = marginals(H0, lp)
        entry = dict(H0=m, Om0=theta_marginal(thetas, om, lp, 0),
                     w0=theta_marginal(thetas, w0, lp, 1),
                     wa=theta_marginal(thetas, wa, lp, 2))
        if ref is None:
            ref = entry
        else:
            for k in ("H0", "Om0", "w0", "wa"):
                d = entry[k]["mean"] - ref[k]["mean"]
                entry[k]["bias_vs_reanchored"] = d
                entry[k]["bias_over_sigma"] = d / ref[k]["sd"] if ref[k]["sd"] else None
        res["models"][mode] = entry
        print(f"[{mode:>11}] H0 = {m['mean']:.3f} +/- {m['sd']:.3f} "
              f"(mode {m['mode']:.3f})", file=sys.stderr)
    res["H0grid"] = [float(x) for x in H0]

    txt = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
