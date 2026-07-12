"""
test_flow_pe_prior.py
---------------------
The analytic PE prior used by the flow-surrogate likelihood must reproduce
gwcat's exported ``p_pe`` convention exactly:

    p_pe = m1det * p_dL_UniformSourceFrame(dL) * p_chi(chi_eff | q, amax)

1. The JAX bilinear chi_eff-prior evaluator must match
   ``gwcat.spin.chi_eff_prior_logprob`` (including the -50 clip and the
   q_frac = m1/(m1+m2) convention).
2. Pin-down: recomputing p_pe for the samples of a REAL gwcat-1.0 store from
   the analytic form must agree with the stored p_pe up to one constant per
   event (the prior's [dmin, dmax] normalisation, which is hyperparameter-
   independent).  Skipped when the store is not on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

pytest.importorskip("darksirens.gw.flows")
gwcat_spin = pytest.importorskip("gwcat.spin")

from darksirens.likelihood.flow_events import (
    build_chi_eff_prior_table,
    build_pe_dl_prior_table,
    chi_eff_prior_logpdf,
)

REAL_STORE = Path(
    "/hildafs/home/magana/tmp_ondemand_hildafs_phy220048p_symlink/share/"
    "GWTC-PESamples/latest/gwsamples_bbh_whitelist_all_events.h5"
)


def test_chi_prior_port_matches_gwcat():
    amax = 0.99
    qg, cg, tab = build_chi_eff_prior_table(amax)

    rng = np.random.default_rng(3)
    m1 = rng.uniform(5.0, 90.0, 4000)
    m2 = rng.uniform(0.1, 1.0, 4000) * m1  # m2 <= m1
    chi = rng.uniform(-1.05, 1.05, 4000)  # include out-of-support points

    ref = gwcat_spin.chi_eff_prior_logprob(chi, m1, m2, amax=amax)
    q_frac = m1 / (m1 + m2)
    got = np.asarray(
        chi_eff_prior_logpdf(
            jnp.asarray(q_frac), jnp.asarray(chi),
            jnp.asarray(qg), jnp.asarray(cg), jnp.asarray(tab),
        )
    )
    np.testing.assert_allclose(got, np.asarray(ref), atol=1e-10)


@pytest.mark.skipif(not REAL_STORE.exists(), reason="real gwcat PE store not on disk")
def test_pe_prior_pins_down_stored_p_pe():
    import h5py

    with h5py.File(REAL_STORE, "r") as f:
        assert f.attrs["format_version"].startswith("gwcat-1.0") or \
            f.attrs["format_version"] == "gwcat-1.0"
        nsamp = int(f.attrs["nsamp"])
        pe_H0 = float(f.attrs["pe_cosmology_H0"])
        pe_Om0 = float(f.attrs["pe_cosmology_Om0"])
        amax = float(f.attrs["chi_eff_amax"])
        assert bool(f.attrs["chi_eff_in_p_pe"])
        n_check = 12  # events
        n = n_check * nsamp
        m1det = f["m1det"][:n]
        m2det = f["m2det"][:n]
        dL = f["dL"][:n]
        chieff = f["chieff"][:n]
        p_pe = f["p_pe"][:n]

    log_dl_grid, log_p_dl = build_pe_dl_prior_table(pe_H0, pe_Om0)
    qg, cg, tab = build_chi_eff_prior_table(amax)

    q = m2det / m1det
    log_pe = (
        np.log(m1det)
        + np.interp(np.log(dL), log_dl_grid, log_p_dl)
        + np.asarray(
            chi_eff_prior_logpdf(
                jnp.asarray(1.0 / (1.0 + q)), jnp.asarray(chieff),
                jnp.asarray(qg), jnp.asarray(cg), jnp.asarray(tab),
            )
        )
    )

    log_ratio = (np.log(p_pe) - log_pe).reshape(n_check, nsamp)
    # One constant per event: the [dmin, dmax] normalisation of the stored
    # prior (ours is deliberately unnormalised).  Within an event the ratio
    # must be flat to interpolation error.
    scatter = np.nanstd(log_ratio, axis=1)
    assert np.all(scatter < 5e-3), f"per-event log-ratio scatter: {scatter}"
