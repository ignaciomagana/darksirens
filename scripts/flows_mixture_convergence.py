#!/usr/bin/env python
"""Convergence study for the flow-surrogate mixture proposal (issue #260).

The flow event term integrates ``t(theta) * flow(X(theta)) / pi_PE(X(theta))``
over the source-frame parameters.  Before #260 the proposal was confined to
per-event support boxes measured from a few thousand flow samples, so the
learned flow's tails were dropped — by an amount that MOVES WITH THE
HYPERPARAMETERS, hence a posterior bias rather than a per-event constant.
``build_flow_loglike`` now draws a fraction ``w_full`` of each event's points
from the full population support and weights EVERY draw by the exact mixture
density.

This script measures that, against an independent reference.

Reference estimator (no windows, no grid samplers, no mixture)
--------------------------------------------------------------
Change variables from theta = (m1src, q, chi, z) to X = (m1det, m2det, dL,
chi).  The Jacobian is ``|dX/dtheta| = m1src (1+z)^2 ddL/dz``, so

    Z_i = E_{X ~ flow_i}[ t(theta(X)) / (pi_PE(X) * m1src (1+z)^2 ddL/dz) ],

i.e. importance sampling FROM THE FLOW ITSELF.  The flow is the sharply peaked
factor of the integrand, so this converges at ESS ~ N (measured: 40-70% of N)
and shares no code with the estimator under test.

Two synthetic event families, both with a computable truth:
  * ``spline``   — untrained triangular spline flows (the production
                   architecture), light tails.
  * ``heavytail``— an analytic stand-in: Student-t(nu) in log m1det and log
                   dL, logit-normal in q, truncated normal in chi_eff.  A
                   deliberately fat-tailed "flow" of the kind a spline can
                   learn from a poorly converged PE run.

Usage:
    python scripts/flows_mixture_convergence.py --outdir /tmp/mix260
    python scripts/flows_mixture_convergence.py --family heavytail --nu 3

Result (2 events per family, J = 32768, reference at N = 2e6, CPU, 2026-07-27).
Per-event ln Z minus the reference, summed over the two events:

    family     theta    n_supp  margin   w=0     w=0.02   w=0.05   w=0.2   MC sig
    spline     fid        4096    0.25   +0.040  +0.042   +0.047   +0.045   0.031
    spline     tail       4096    0.25   +0.049  +0.048   +0.057   +0.079   0.037
    heavytail  fid         512    0.00   -0.076  -0.070   -0.059   -0.026   0.023
    heavytail  fid        4096    0.00   -0.050  -0.042   -0.034   -0.013   0.022
    heavytail  fid       32768    0.00   -0.043  -0.035   -0.029   -0.005   0.023
    heavytail  tail        512    0.00   -0.056  -0.057   -0.051   -0.040   0.030
    heavytail  tail       4096    0.00   -0.030  -0.031   -0.028   -0.016   0.026

At the SHIPPED window settings (margin 0.25) every estimator sits within
1-2 sigma of the reference: the truncation deficit is real but below the
Monte-Carlo error, and raising w_full does not move the answer.  Where the
window under-resolves the flow (zero margin) the windowed-only estimator is
systematically LOW at every support-sample count -- it does not converge away
with more support samples, which is the signature of a truncation bias rather
than a box-estimation error -- and w_full removes it monotonically.  The
severe case (a window that under-COVERS the flow: -5 to -300 nats per event,
rescued to <0.2 nats by w_full = 0.3) is in
``tests/test_flow_mixture_proposal.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.gw.populations.sampling import (
    make_mass_q_edges,
    resolve_mass_grid_bounds,
)
from darksirens.gw.flows import support_boxes_from_samples
from darksirens.likelihood.flow_events import (
    PePriorTables,
    build_chi_eff_prior_table,
    build_flow_loglike,
    build_pe_dl_prior_table,
    chi_eff_prior_prob,
)
from darksirens.redshift.grid import zgrid
from darksirens.redshift.prior import prepare_redshift_prior_state
from darksirens.utils.cosmology import H0Planck, Om0Planck, dL_of_z, z_of_dL

PE_H0, PE_OM0 = 67.74, 0.3089
C_KM_S = 299792.458

# powerlaw+peak parameter order:
# v1 alpha m_min m_max dm_min dm_max mu sigma beta mu_chi sigma_chi gamma
IDX = dict(v1=0, alpha=1, m_min=2, m_max=3, dm_min=4, dm_max=5, mu=6,
           sigma=7, beta=8, mu_chi=9, sigma_chi=10, gamma=11)


# ── synthetic event families ────────────────────────────────────────────────


def _t_logpdf(x, loc, scale, nu):
    from scipy.special import gammaln

    z = (x - loc) / scale
    return (
        gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi) - np.log(scale)
        - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    )


def _t_logpdf_jnp(x, loc, scale, nu):
    from jax.scipy.special import gammaln

    z = (x - loc) / scale
    return (
        gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(scale)
        - 0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)
    )


class HeavyTailFamily:
    """Analytic stand-in "flows": independent heavy-tailed marginals.

    Density in (m1det, m2det, dL, chi_eff), built from
    u = log m1det ~ t(nu), v = logit(q) ~ N, w = log dL ~ t(nu),
    chi ~ N truncated to [-1, 1], with q = m2det/m1det:

        log p = log p_u + log p_v + log p_w + log p_chi
                - 2 log m1det - log q - log(1 - q) - log dL
    """

    name = "heavytail"

    def __init__(self, n_events=2, nu=3.0, seed=5):
        rng = np.random.default_rng(seed)
        self.nu = float(nu)
        self.n_events = n_events
        self.loc = np.stack([
            np.log(rng.uniform(30.0, 45.0, n_events)),        # log m1det
            rng.uniform(1.0, 2.0, n_events),                  # logit q
            np.log(rng.uniform(900.0, 1400.0, n_events)),     # log dL
            rng.uniform(-0.05, 0.05, n_events),               # chi_eff
        ], axis=1)
        self.scale = np.tile(np.array([0.10, 0.35, 0.12, 0.08]), (n_events, 1))

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        nu = self.nu
        out = np.empty((self.n_events, n, 4))
        for i in range(self.n_events):
            u = self.loc[i, 0] + self.scale[i, 0] * rng.standard_t(nu, n)
            v = self.loc[i, 1] + self.scale[i, 1] * rng.standard_normal(n)
            w = self.loc[i, 2] + self.scale[i, 2] * rng.standard_t(nu, n)
            chi = self.loc[i, 3] + self.scale[i, 3] * rng.standard_normal(n)
            chi = np.clip(chi, -0.999, 0.999)
            m1det = np.exp(u)
            q = 1.0 / (1.0 + np.exp(-v))
            out[i] = np.stack([m1det, q * m1det, np.exp(w), chi], axis=1)
        return out

    def log_prob_np(self, X):
        """(n_events, n) log density of ``X`` (n_events, n, 4), numpy."""
        m1det, m2det, dL, chi = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
        q = m2det / m1det
        ok = (m1det > 0) & (dL > 0) & (q > 1e-12) & (q < 1.0 - 1e-12) \
            & (np.abs(chi) <= 1.0)
        qs = np.clip(q, 1e-12, 1.0 - 1e-12)
        u, w = np.log(np.maximum(m1det, 1e-300)), np.log(np.maximum(dL, 1e-300))
        v = np.log(qs) - np.log1p(-qs)
        lo, sc = self.loc[:, None, :], self.scale[:, None, :]
        lp = (
            _t_logpdf(u, lo[..., 0], sc[..., 0], self.nu)
            + (-0.5 * ((v - lo[..., 1]) / sc[..., 1]) ** 2
               - np.log(sc[..., 1]) - 0.5 * np.log(2 * np.pi))
            + _t_logpdf(w, lo[..., 2], sc[..., 2], self.nu)
            + (-0.5 * ((chi - lo[..., 3]) / sc[..., 3]) ** 2
               - np.log(sc[..., 3]) - 0.5 * np.log(2 * np.pi))
            - 2.0 * u - np.log(qs) - np.log1p(-qs) - w
        )
        return np.where(ok, lp, -np.inf)

    def eval_logflows(self):
        """``(group_params, X) -> (n_events, J)`` for ``build_flow_loglike``."""
        loc = jnp.asarray(self.loc)[:, None, :]
        sc = jnp.asarray(self.scale)[:, None, :]
        nu = self.nu

        def f(group_params, X):
            m1det, m2det, dL, chi = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
            q = m2det / m1det
            ok = (m1det > 0) & (dL > 0) & (q > 1e-12) & (q < 1.0 - 1e-12) \
                & (jnp.abs(chi) <= 1.0)
            qs = jnp.clip(q, 1e-12, 1.0 - 1e-12)
            u = jnp.log(jnp.maximum(m1det, 1e-300))
            w = jnp.log(jnp.maximum(dL, 1e-300))
            v = jnp.log(qs) - jnp.log1p(-qs)
            lp = (
                _t_logpdf_jnp(u, loc[..., 0], sc[..., 0], nu)
                + (-0.5 * ((v - loc[..., 1]) / sc[..., 1]) ** 2
                   - jnp.log(sc[..., 1]) - 0.5 * jnp.log(2 * jnp.pi))
                + _t_logpdf_jnp(w, loc[..., 2], sc[..., 2], nu)
                + (-0.5 * ((chi - loc[..., 3]) / sc[..., 3]) ** 2
                   - jnp.log(sc[..., 3]) - 0.5 * jnp.log(2 * jnp.pi))
                - 2.0 * u - jnp.log(qs) - jnp.log1p(-qs) - w
            )
            return jnp.where(ok, lp, -jnp.inf)

        return f

    def group_params(self):
        return (jnp.zeros(1),)


class SplineFamily:
    """Untrained triangular spline flows — the production architecture."""

    name = "spline"

    def __init__(self, root, n_events=2, seed=3):
        import json as _json
        import equinox as eqx
        import jax.tree_util as jtu
        import darksirens.gw.flows as flows_mod

        root = Path(root)
        for k in range(n_events):
            d = root / f"GW_TOY_{k}"
            d.mkdir(parents=True, exist_ok=True)
            cfg = {
                "base_dist": "Normal", "data_dim": 4, "type": "spline",
                "flow_layers": 2, "knots": 4, "key": seed + 7 * k,
                "columns": list(flows_mod.SPECTRAL_COLUMNS),
                "Z_mean": [2.0, 3.4, 7.0, 0.0],
                "Z_std": [0.3, 0.1, 0.15, 0.15],
                "constraints": {"0": {"type": "ordered_positive", "dims": [0, 1]},
                                "1": None, "2": {"type": "positive"},
                                "3": {"type": "interval", "low": -1, "high": 1}},
            }
            flow = flows_mod.create_flow_from_config(cfg)
            arrays, _ = eqx.partition(flow, eqx.is_array)
            leaves, _ = jtu.tree_flatten(arrays)
            np.savez(d / f"GW_TOY_{k}_flow.npz",
                     *[np.asarray(l) for l in leaves],
                     config_json=_json.dumps(cfg))
        self._mod = flows_mod
        self.ens = flows_mod.load_flow_ensemble(root)
        self.n_events = self.ens.n_flows

    def sample(self, n, seed=0):
        return np.asarray(
            self._mod.ensemble_sample(self.ens, jax.random.key(seed), n)
        )

    def log_prob_np(self, X):
        f = self._mod.make_ensemble_log_prob_per_event(self.ens)
        return np.asarray(f(self.ens.group_params(), jnp.asarray(X)))

    def eval_logflows(self):
        return self._mod.make_ensemble_log_prob_per_event(self.ens)

    def group_params(self):
        return self.ens.group_params()


# ── shared plumbing ─────────────────────────────────────────────────────────


def toy_catalog():
    return EMCatalog(apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
                     wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
                     delta_g_pix_z=jnp.zeros((1, 1)), dN_obs_kde=None,
                     pixel_to_cache_idx=None)


def toy_selection(n_sel=8000, seed=7):
    rng = np.random.default_rng(seed)
    m1det = np.exp(rng.uniform(np.log(5.0), np.log(150.0), n_sel))
    q = rng.uniform(0.2, 1.0, n_sel)
    dL = rng.uniform(100.0, 8000.0, n_sel)
    chieff = rng.uniform(-0.5, 0.5, n_sel)
    p_draw = (1.0 / (m1det * np.log(30.0))) * (1.0 / 0.8) * (1.0 / 7900.0)
    return GWEvent(m1det=jnp.asarray(m1det), m2det=jnp.asarray(q * m1det),
                   dL=jnp.asarray(dL), chieff=jnp.asarray(chieff),
                   prior_wt=jnp.asarray(p_draw),
                   pixels=jnp.zeros(n_sel, dtype=jnp.int32), q=jnp.asarray(q),
                   valid=jnp.ones(n_sel, dtype=bool))


def make_pe_tables(amax=0.99):
    log_dl, log_p_dl = build_pe_dl_prior_table(PE_H0, PE_OM0)
    qg, cg, tab = build_chi_eff_prior_table(amax)
    return PePriorTables(log_dl_grid=jnp.asarray(log_dl),
                         log_p_dl=jnp.asarray(log_p_dl),
                         chi_q_grid=jnp.asarray(qg), chi_grid=jnp.asarray(cg),
                         chi_table=jnp.asarray(tab))


def reference_lnZ(family, model, theta, cosmo, survey, pe_tab, m1_edges,
                  q_edges, n=2_000_000, seed=1234):
    """Flow-space importance sampling — see the module docstring."""
    state = prepare_redshift_prior_state("spectral_sirens", cosmo, survey,
                                         toy_catalog(), materialize_state=True)
    log_pvol = np.asarray(state.log_pvol)
    zg = np.asarray(zgrid)
    X = family.sample(n, seed=seed)
    lnZ, ess = [], []
    for i in range(family.n_events):
        m1det, m2det, dL, chi = X[i, :, 0], X[i, :, 1], X[i, :, 2], X[i, :, 3]
        z = np.asarray(z_of_dL(jnp.asarray(dL), cosmo.H0, cosmo.Om0))
        ok = np.isfinite(z)
        z = np.where(ok, z, 0.0)
        m1src, q = m1det / (1.0 + z), m2det / m1det
        ok &= (m1src >= float(m1_edges[0])) & (m1src <= float(m1_edges[-1]))
        ok &= (q >= float(q_edges[0])) & (q <= float(q_edges[-1]))
        ok &= np.abs(chi) <= 1.0
        log_t = np.asarray(
            model.log_p_massspin(jnp.asarray(m1src), jnp.asarray(q),
                                 jnp.asarray(chi), theta)
            + model.log_rate_z(jnp.asarray(z), theta)
        ) + np.interp(z, zg, log_pvol)
        p_chi = np.asarray(chi_eff_prior_prob(
            jnp.asarray(1.0 / (1.0 + q)), jnp.asarray(chi),
            pe_tab.chi_q_grid, pe_tab.chi_grid, pe_tab.chi_table))
        ok &= p_chi > 0.0
        log_pi = np.interp(np.log(np.maximum(dL, 1e-300)),
                           np.asarray(pe_tab.log_dl_grid),
                           np.asarray(pe_tab.log_p_dl)) \
            + np.log(np.maximum(p_chi, 1e-300))
        DC = np.asarray(dL_of_z(jnp.asarray(z), cosmo.H0, cosmo.Om0)) / (1.0 + z)
        E = np.sqrt(cosmo.Om0 * (1.0 + z) ** 3 + (1.0 - cosmo.Om0))
        ddL_dz = DC + (1.0 + z) * (C_KM_S / cosmo.H0) / E
        log_jac = np.log(m1src) + 2.0 * np.log1p(z) + np.log(ddL_dz)
        lw = np.where(ok, log_t - log_pi - log_jac, -np.inf)
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        mx = lw.max()
        w = np.exp(lw - mx)
        lnZ.append(mx + np.log(w.mean()))
        ess.append(float(w.sum() ** 2 / np.sum(w ** 2)))
    return np.asarray(lnZ), np.asarray(ess)


def estimator_lnZ(family, model, cosmo, survey, theta, pe_tab, m1_edges,
                  q_edges, boxes, J, w_full, seed=0):
    ll = build_flow_loglike(
        model=model,
        eval_logflows=family.eval_logflows(),
        group_params=family.group_params(),
        u_base=jax.random.uniform(jax.random.PRNGKey(seed),
                                  (family.n_events, J, 4), dtype=jnp.float64),
        m1_edges=m1_edges, q_edges=q_edges, pe_tables=pe_tab,
        support_boxes=boxes, gw_sel=toy_selection(),
        em_catalog_sel=toy_catalog(), Ndraw=16000.0,
        nEvents=family.n_events, w_full=w_full,
    )
    lnz, var, ess = ll.event_diagnostics(cosmo, survey, theta)
    return np.asarray(lnz), np.asarray(var), np.asarray(ess)


def tail_favoring(base):
    """A population that upweights exactly the clipped high-mass / high-z tails."""
    t = np.asarray(base, dtype=np.float64).copy()
    t[IDX["alpha"]] = -6.0      # rising power law: t ~ m1^{+6}
    t[IDX["m_max"]] = 180.0
    t[IDX["gamma"]] = 6.0       # strong redshift evolution
    return t


# ── study ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", default="both",
                   choices=["spline", "heavytail", "both"])
    p.add_argument("--nu", type=float, default=3.0)
    p.add_argument("--n_events", type=int, default=2)
    p.add_argument("--J", type=int, default=65536)
    p.add_argument("--ref_n", type=int, default=2_000_000)
    p.add_argument("--outdir", default=None)
    p.add_argument("--workdir", default="/tmp/flows_mix260")
    args = p.parse_args()

    model = get_model("powerlaw+peak")
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges, q_edges = make_mass_q_edges(m1_lo, m1_hi, n_m1=512, n_q=256)
    pe_tab = make_pe_tables()
    cosmo = CosmoParams(H0=H0Planck, Om0=Om0Planck)
    survey = SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.5)
    base = np.asarray(get_fixed_population_params("powerlaw+peak"),
                      dtype=np.float64)
    thetas = {"fiducial": base.copy(), "tail-favoring": tail_favoring(base)}

    families = []
    if args.family in ("spline", "both"):
        families.append(SplineFamily(Path(args.workdir) / "spline",
                                     n_events=args.n_events))
    if args.family in ("heavytail", "both"):
        families.append(HeavyTailFamily(n_events=args.n_events, nu=args.nu))

    records = []
    for fam in families:
        for tname, th in thetas.items():
            theta = jnp.asarray(th)
            ref, ref_ess = reference_lnZ(fam, model, theta, cosmo, survey,
                                         pe_tab, m1_edges, q_edges,
                                         n=args.ref_n)
            ref_err = 1.0 / np.sqrt(ref_ess)
            print(f"\n=== {fam.name} / {tname} ===")
            print(f"reference lnZ  = {np.round(ref, 4)}  "
                  f"(ESS {ref_ess.astype(int)}, +/-{np.round(ref_err, 4)})")
            print(f"total          = {ref.sum():.4f}")

            # (a) support-sample count, (b) margin, (c) w_full
            for n_supp in (512, 4096, 32768):
                for margin in (0.0, 0.25, 1.0):
                    S = fam.sample(n_supp, seed=99)
                    boxes = support_boxes_from_samples(S, margin=margin)
                    for w_full in (0.0, 0.02, 0.05, 0.2):
                        lnz, var, ess = estimator_lnZ(
                            fam, model, cosmo, survey, theta, pe_tab,
                            m1_edges, q_edges, boxes, args.J, w_full)
                        d = lnz - ref
                        rec = dict(family=fam.name, theta=tname,
                                   n_support=n_supp, margin=margin,
                                   w_full=w_full, J=args.J,
                                   lnZ=lnz.tolist(), dlnZ=d.tolist(),
                                   total=float(lnz.sum()),
                                   dtotal=float(lnz.sum() - ref.sum()),
                                   ess=ess.tolist(),
                                   mc_sigma=np.sqrt(var).tolist())
                        records.append(rec)
                        print(f"  n_supp={n_supp:6d} margin={margin:4.2f} "
                              f"w_full={w_full:4.2f}  "
                              f"dlnZ={np.round(d, 4)}  "
                              f"dtotal={rec['dtotal']:+.4f}  "
                              f"ESS={ess.astype(int)}  "
                              f"MCsig={np.round(np.sqrt(var), 4)}")

            # (d) J convergence at the recommended setting
            for J in (8192, 32768, 131072):
                S = fam.sample(4096, seed=99)
                boxes = support_boxes_from_samples(S, margin=0.25)
                for w_full in (0.0, 0.05):
                    lnz, var, ess = estimator_lnZ(
                        fam, model, cosmo, survey, theta, pe_tab, m1_edges,
                        q_edges, boxes, J, w_full)
                    print(f"  J={J:7d} w_full={w_full:4.2f}  "
                          f"dlnZ={np.round(lnz - ref, 4)}  "
                          f"dtotal={lnz.sum() - ref.sum():+.4f}  "
                          f"ESS={ess.astype(int)}")
                    records.append(dict(family=fam.name, theta=tname,
                                        n_support=4096, margin=0.25,
                                        w_full=w_full, J=J,
                                        lnZ=lnz.tolist(),
                                        dlnZ=(lnz - ref).tolist(),
                                        total=float(lnz.sum()),
                                        dtotal=float(lnz.sum() - ref.sum()),
                                        ess=ess.tolist(),
                                        mc_sigma=np.sqrt(var).tolist()))

            records.append(dict(family=fam.name, theta=tname, reference=True,
                                lnZ=ref.tolist(), ess=ref_ess.tolist(),
                                total=float(ref.sum())))

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "mixture_convergence.json").write_text(json.dumps(records, indent=2))
        print(f"\nwrote {out / 'mixture_convergence.json'}")


if __name__ == "__main__":
    main()
