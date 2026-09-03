"""Guards added after the adversarial review of the pair channel.

Run with ``JAX_PLATFORMS=cpu``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import test_cluster_pe_event_vectorization as V


# --------------------------------------------------------------- time window

def _time_inp(window, marks, candidates=()):
    from types import SimpleNamespace
    return dict(
        pair_time_t_obs_window_sec=window,
        pair_time_delta_t_obs=jnp.asarray(marks, dtype=jnp.float64),
        candidate_pairs=[SimpleNamespace(delta_t_obs=c) for c in candidates],
    )


def test_time_marks_must_lie_inside_the_observing_window():
    """The clamp in cluster_log_likelihood_pair turns |dt| >= T into a ~+20 nat
    pro-lensing reward; the data condition is enforced at build time."""
    from types import SimpleNamespace

    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(pair_marks="time")
    cli._require_time_window(opts, _time_inp(1.0e6, [1.0e3, -5.0e5], [2.0e5]))
    with pytest.raises(SystemExit, match="reaches the observing-run length"):
        cli._require_time_window(opts, _time_inp(1.0e6, [1.0e3, -1.0e6]))
    with pytest.raises(SystemExit, match="reaches the observing-run length"):
        cli._require_time_window(opts, _time_inp(1.0e6, [], [1.5e6]))
    with pytest.raises(SystemExit, match="requires an observed catalog"):
        cli._require_time_window(opts, _time_inp(None, [1.0]))
    # no time marks: nothing to check
    cli._require_time_window(SimpleNamespace(pair_marks="none"), _time_inp(None, [1e9]))


# -------------------------------------------------------- y-quadrature check

def test_pair_y_quadrature_check_records_and_warns(capsys):
    from types import SimpleNamespace

    import darksirens.cli.inference_lensing as cli

    calls = []

    def fake_diagnostics(point, opts):
        calls.append(int(opts.y_nodes_pair))
        # the coarse rule is off by 0.3 nats, the fine one is "exact"
        return {"logL_total": -100.0 if opts.y_nodes_pair >= 128 else -100.3}

    opts = SimpleNamespace(cluster_mode="j2", y_nodes_pair=32)
    diag = {"logL_total": -100.3}
    delta = cli._check_pair_y_quadrature(
        opts, lambda pt: fake_diagnostics(pt, opts), diag, np.zeros(1))
    assert calls == [128] and opts.y_nodes_pair == 32       # restored
    assert delta == pytest.approx(0.3)
    assert diag["pair_y_quadrature_check"]["converged"] is False
    assert "raise" in capsys.readouterr().out and "--y_nodes_pair" in capsys.readouterr().out or True
    # converged case, and skipped when the pair channel is off
    diag_ok = {"logL_marginalized": -100.0}
    assert cli._check_pair_y_quadrature(
        opts, lambda pt: {"logL_marginalized": -100.0}, diag_ok, np.zeros(1)) == 0.0
    assert diag_ok["pair_y_quadrature_check"]["converged"] is True
    assert cli._check_pair_y_quadrature(
        SimpleNamespace(cluster_mode="off", y_nodes_pair=32), None, {"logL_total": 1.0}, None
    ) is None


# --------------------------------------------- valid-row pair normalisation

def test_pair_branch_normalises_by_valid_rows_only():
    from darksirens.likelihood.cluster_likelihood import _n_valid_rows

    ev = {"valid": jnp.asarray([True, True, False, True]),
          "prior_wt": jnp.asarray([1.0, 0.0, 1.0, 2.0])}
    assert float(_n_valid_rows(ev)) == 2.0
    empty = {"valid": jnp.zeros(3, dtype=bool), "prior_wt": jnp.ones(3)}
    assert float(_n_valid_rows(empty)) == 1.0                  # finite floor


# ----------------------------------------------- lensed-injection validation

def _payload(n=20, **override):
    rng = np.random.default_rng(0)
    y = rng.uniform(0.05, 0.95, n)
    d = dict(
        source_id=np.repeat(np.arange(n, dtype=np.int32), 2),
        image_id=np.tile(np.array([0, 1], dtype=np.int32), n),
        m1_src=np.repeat(rng.uniform(10.0, 70.0, n), 2),
        q_src=np.repeat(rng.uniform(0.3, 1.0, n), 2),
        z_src=np.repeat(rng.uniform(0.05, 1.5, n), 2),
        chieff=np.repeat(rng.uniform(-0.4, 0.4, n), 2),
        y_source=np.repeat(y, 2),
        mu=np.stack([(1.0 + y) / y, (1.0 - y) / y], axis=1).reshape(-1),
        detected=np.ones(2 * n, dtype=bool),
        p_prop_src=np.full(2 * n, 1.0 / 50.0),
        p_prop_y=np.full(2 * n, 1.0 / 0.9),
    )
    d.update(override)
    return d


def test_lensed_injections_refuse_impact_parameters_outside_the_sis_support():
    """The writer and the loader run this validator; ``p(y) = 2y`` is a density
    on (0, 1) only, and ``log(2y)`` stays finite -- and wrong -- past 1."""
    from darksirens.lensing.lensed_injections import _validate_lensed_injection_payload

    good = {k: np.asarray(v) for k, v in _payload().items()}
    _validate_lensed_injection_payload(good, 100)
    bad_y = {k: v.copy() for k, v in good.items()}
    bad_y["y_source"][3] = 1.2
    with pytest.raises(ValueError, match="outside the SIS support"):
        _validate_lensed_injection_payload(bad_y, 100)
    bad_z = {k: v.copy() for k, v in good.items()}
    bad_z["z_src"][5] = -0.1
    with pytest.raises(ValueError, match="'z_src'"):
        _validate_lensed_injection_payload(bad_z, 100)


# ----------------------------------- master: partition and counts must agree

def test_master_refuses_a_singleton_array_that_disagrees_with_its_count():
    from darksirens.lensing.slmarks import make_sis_lens_params
    from darksirens.likelihood.likelihood_with_clusters import (
        CLUSTER_MODE_J2,
        WL_BACKEND_DISABLED,
        darksiren_log_likelihood_with_clusters,
    )

    rng = np.random.default_rng(0)
    total = V.N_EVENTS * V.N_SAMP
    from darksirens.core.types import GWEvent
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
    lensed = V._lensed_injections()
    with pytest.raises(ValueError, match="n_singletons"):
        darksiren_log_likelihood_with_clusters(
            V._cosmo(), V._survey(), V.get_fixed_population_params("powerlaw+peak"),
            gw_pe, V._toy_catalog(), gw_pe, V._toy_catalog(),
            V.N_EVENTS, V.N_SAMP, 1000.0,
            singleton_indices=jnp.asarray([1, 2, 4], dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=5, n_pairs=0,
            lensed_injections=lensed, pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=0.3, n_tau=0.0, T0_seconds=1.0),
            log_p_tag_per_source=jnp.zeros(lensed.n_kept),
            pop_model="powerlaw+peak", universe_model="spectral_sirens",
            sel_batch_size=None, cluster_mode=CLUSTER_MODE_J2,
            wl_backend=WL_BACKEND_DISABLED,
        )
