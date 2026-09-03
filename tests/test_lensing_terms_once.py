"""The marginalize_exact paths evaluate the cluster master likelihood ONCE per
proposal -- every event as a singleton row, every candidate edge as a pair row
-- and assemble each partition from those rows
(``cli.inference_lensing._assemble_partition``).  This pins the assembly to the
master likelihood's own evaluation of the same partition: the rows are
partition-independent, so the two must agree to floating-point reassociation.

Run with ``JAX_PLATFORMS=cpu``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.lensing.lensed_injections import make_lensed_injection_set
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.likelihood.likelihood_with_clusters import (
    CLUSTER_MODE_J2,
    WL_BACKEND_DISABLED,
    darksiren_likelihood_diagnostics_with_clusters,
)
from darksirens.likelihood.pair_kde import make_pair_kde, stack_pair_kdes

N_EVENTS, N_SAMP, N_SEL = 8, 60, 300
EDGES = ((0, 3), (6, 0), (2, 5))            # candidate edges, in edge order
PARTITIONS = (                              # (singletons, edge indices)
    (tuple(range(N_EVENTS)), ()),
    ((1, 2, 4, 5, 6, 7), (0,)),
    ((1, 2, 3, 4, 5, 7), (1,)),
    ((1, 4, 6, 7), (0, 2)),
)


def _fixture():
    rng = np.random.default_rng(3)
    total = N_EVENTS * N_SAMP
    gw_pe = GWEvent(
        m1det=jnp.asarray(rng.uniform(20.0, 60.0, total)),
        m2det=jnp.asarray(rng.uniform(10.0, 30.0, total)),
        dL=jnp.asarray(rng.uniform(400.0, 3000.0, total)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, total)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, total)),
        pixels=jnp.zeros(total, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, total)),
        valid=jnp.ones(total, dtype=jnp.bool_),
    )
    gw_sel = GWEvent(
        m1det=jnp.asarray(rng.uniform(15.0, 70.0, N_SEL)),
        m2det=jnp.asarray(rng.uniform(8.0, 35.0, N_SEL)),
        dL=jnp.asarray(rng.uniform(200.0, 3000.0, N_SEL)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, N_SEL)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, N_SEL)),
        pixels=jnp.zeros(N_SEL, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, N_SEL)),
        valid=jnp.ones(N_SEL, dtype=jnp.bool_),
    )
    n = 300
    y = rng.uniform(0.05, 0.95, n)
    lensed = make_lensed_injection_set(
        source_id=np.repeat(np.arange(n, dtype=np.int32), 2),
        image_id=np.tile(np.array([0, 1], dtype=np.int32), n),
        m1_src=np.repeat(rng.uniform(10.0, 70.0, n), 2),
        q_src=np.repeat(rng.uniform(0.3, 1.0, n), 2),
        z_src=np.repeat(rng.uniform(0.05, 1.5, n), 2),
        chieff=np.repeat(rng.uniform(-0.4, 0.4, n), 2),
        y_source=np.repeat(y, 2),
        mu=np.stack([(1.0 + y) / y, (1.0 - y) / y], axis=1).reshape(-1),
        detected=np.ones(2 * n, dtype=bool),
        p_prop_src=np.full(2 * n, 1.0 / (60 * 0.7 * 1.45 * 0.8)),
        p_prop_y=np.full(2 * n, 1.0 / 0.9),
        n_draw_sources=3000,
    )
    kdes = []
    for e in range(N_EVENTS):
        sl = slice(e * N_SAMP, (e + 1) * N_SAMP)
        kdes.append(make_pair_kde(
            np.asarray(gw_pe.m1det[sl]), np.asarray(gw_pe.q[sl]),
            np.asarray(gw_pe.dL[sl]), np.asarray(gw_pe.chieff[sl]),
            np.asarray(gw_pe.prior_wt[sl]),
        ))
    catalog = EMCatalog(
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    return dict(
        cosmo=CosmoParams(H0=67.74, Om0=0.3075),
        survey=SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5),
        pop_params=get_fixed_population_params("powerlaw+peak"),
        gw_pe=gw_pe, gw_sel=gw_sel, catalog=catalog, lensed=lensed,
        pair_kdes=stack_pair_kdes(kdes),
        sis=make_sis_lens_params(A_tau=0.3, n_tau=0.0, T0_seconds=1.0),
    )


def _master(fx, singletons, pairs, pair_batch_size=0):
    singletons = np.asarray(singletons, dtype=np.int32)
    pairs = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
    return darksiren_likelihood_diagnostics_with_clusters(
        fx["cosmo"], fx["survey"], fx["pop_params"],
        fx["gw_pe"], fx["catalog"], fx["gw_sel"], fx["catalog"],
        N_EVENTS, N_SAMP, 1000.0,
        singleton_indices=jnp.asarray(singletons),
        pair_indices=jnp.asarray(pairs),
        n_singletons=int(singletons.size), n_pairs=int(pairs.shape[0]),
        lensed_injections=fx["lensed"], pair_kdes=fx["pair_kdes"],
        sis_params=fx["sis"],
        log_p_tag_per_source=jnp.zeros(fx["lensed"].n_kept),
        pop_model="powerlaw+peak", universe_model="spectral_sirens",
        sel_batch_size=None, cluster_mode=CLUSTER_MODE_J2,
        wl_backend=WL_BACKEND_DISABLED, pe_event_block=None,
        pair_batch_size=pair_batch_size, y_nodes_pair=8,
        # a toy selection set: disable the total-variance guard so every
        # partition's selection correction is a finite number to compare
        max_likelihood_variance=1e6,
    )


@pytest.fixture(scope="module")
def terms_and_fixture():
    fx = _fixture()
    edges = np.asarray(EDGES, dtype=np.int32)
    # the once-per-proposal layout: every event a singleton row, every edge a
    # pair row, pairs through the master's scan (pair_batch_size >= 1)
    terms = _master(fx, np.arange(N_EVENTS), edges, pair_batch_size=2)
    return fx, terms


def test_terms_call_exposes_per_row_terms(terms_and_fixture):
    fx, terms = terms_and_fixture
    assert terms["per_event_logL"].shape == (N_EVENTS,)
    assert terms["per_event_var"].shape == (N_EVENTS,)
    assert terms["per_pair_logL"].shape == (len(EDGES),)
    assert terms["per_pair_var"].shape == (len(EDGES),)
    assert bool(jnp.all(jnp.isfinite(terms["per_event_logL"])))
    assert bool(jnp.all(jnp.isfinite(terms["per_pair_logL"])))
    # the master's own sums are these rows summed
    np.testing.assert_allclose(
        float(jnp.sum(terms["per_event_logL"])), float(terms["singleton_logL_sum"]), rtol=1e-13)
    np.testing.assert_allclose(
        float(jnp.sum(terms["per_pair_logL"])), float(terms["pair_logL_sum"]), rtol=1e-13)


@pytest.mark.parametrize("partition", PARTITIONS, ids=[f"p{k}" for k in range(len(PARTITIONS))])
def test_assembled_partition_matches_the_master_evaluated_on_it(terms_and_fixture, partition):
    from types import SimpleNamespace

    import darksirens.cli.inference_lensing as cli

    fx, terms = terms_and_fixture
    singletons, edge_idx = partition
    pairs = np.asarray(EDGES, dtype=np.int32)[list(edge_idx)].reshape((-1, 2))
    direct = _master(fx, singletons, pairs)
    opts = SimpleNamespace(selection_neff_soft_guard=False, max_likelihood_variance=1e6)
    assembled = cli._assemble_partition(terms, np.asarray(singletons), np.asarray(edge_idx), opts)
    for key in ("singleton_logL_sum", "pair_logL_sum", "singleton_variance_sum",
                "pair_variance_sum", "pe_variance_sum"):
        a, b = float(np.asarray(assembled[key])), float(np.asarray(direct[key]))
        assert np.isfinite(b), (key, b)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=key)
    # The toy injection set can trip the sparse-N_eff guard (-inf on both
    # sides); where the master is finite the assembly must match it.
    for key in ("selection_correction_total", "logL_total"):
        a, b = float(np.asarray(assembled[key])), float(np.asarray(direct[key]))
        if np.isfinite(b):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=key)
        else:
            assert a == b == -np.inf, (key, a, b)
    if edge_idx:
        np.testing.assert_allclose(
            np.asarray(assembled["per_pair_logL"]), np.asarray(direct["per_pair_logL"]),
            rtol=1e-12, atol=0.0,
        )


def test_edge_row_order_follows_the_edge_list(terms_and_fixture):
    """Rows are indexed by candidate-edge position, so a partition's edges may
    be gathered in any order and the pair row must not depend on where the
    edge sat in the terms call."""
    fx, terms = terms_and_fixture
    alone = _master(fx, (1, 2, 4, 5, 6, 7), np.asarray([EDGES[0]]))
    np.testing.assert_allclose(
        float(terms["per_pair_logL"][0]), float(alone["per_pair_logL"][0]), rtol=1e-12)
    np.testing.assert_allclose(
        float(terms["per_pair_var"][0]), float(alone["pair_variance_sum"]), rtol=1e-12)
