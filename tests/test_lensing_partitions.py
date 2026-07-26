import math

import numpy as np
import pytest
from scipy.special import logsumexp

from darksirens.lensing.marginal_diagnostics import (
    component_partition_diagnostic_rows,
    compute_componentwise_factorized_partition_diagnostics,
    compute_marginalized_partition_diagnostics,
)
from darksirens.lensing.partitions import (
    CandidatePair,
    connected_components_from_candidate_pairs,
    enumerate_compatible_partitions,
    exact_partition_components,
    exact_partitions_componentwise,
    exact_partitions_from_json,
)

def test_exact_partitions_track_candidate_edge_indices_and_prior_only_posteriors():
    candidates = [
        CandidatePair(0, 1, math.log(2.0), "a"),
        CandidatePair(1, 2, math.log(3.0), "b"),
    ]
    states = enumerate_compatible_partitions(3, candidates)
    assert [s.pair_indices.tolist() for s in states] == [[], [[1, 2]], [[0, 1]]]
    assert [s.candidate_edge_indices.tolist() for s in states] == [[], [1], [0]]

    diag = compute_marginalized_partition_diagnostics(
        states, candidates, lambda _s: 5.0
    )
    weights = np.array([1.0, 3.0, 2.0]) / 6.0
    np.testing.assert_allclose(diag["partition_posterior_probability"], weights)
    assert diag["posterior_pair_probabilities"][0]["label"] == "a"
    assert diag["posterior_pair_probabilities"][0]["p_pair"] == pytest.approx(2.0 / 6.0)
    assert diag["posterior_pair_probabilities"][1]["p_pair"] == pytest.approx(3.0 / 6.0)


def test_connected_components_and_componentwise_equal_global():
    candidates = [
        CandidatePair(0, 1, math.log(2.0)),
        CandidatePair(1, 2, math.log(3.0)),
        CandidatePair(3, 4, math.log(5.0)),
    ]
    components = connected_components_from_candidate_pairs(6, candidates)
    assert [c["event_indices"] for c in components] == [(0, 1, 2), (3, 4), (5,)]
    assert [c["candidate_edge_indices"] for c in components] == [(0, 1), (2,), ()]

    global_states = enumerate_compatible_partitions(6, candidates)
    component_states, summaries, _ = exact_partitions_componentwise(
        6, candidates, max_total_partitions=100
    )
    assert [s["n_partitions"] for s in summaries] == [3, 2, 1]

    def key(state):
        return (
            tuple(map(tuple, np.asarray(state.pair_indices).reshape((-1, 2)).tolist())),
            tuple(np.asarray(state.candidate_edge_indices).tolist()),
            pytest.approx(state.log_prior_weight),
        )

    assert len(component_states) == len(global_states)
    assert sorted(
        (
            tuple(map(tuple, s.pair_indices.tolist())),
            tuple(s.candidate_edge_indices.tolist()),
        )
        for s in component_states
    ) == sorted(
        (
            tuple(map(tuple, s.pair_indices.tolist())),
            tuple(s.candidate_edge_indices.tolist()),
        )
        for s in global_states
    )
    np.testing.assert_allclose(
        sorted(s.log_prior_weight for s in component_states),
        sorted(s.log_prior_weight for s in global_states),
    )


def test_component_caps_fail_helpfully():
    candidates = [CandidatePair(0, 1, 0.0), CandidatePair(1, 2, 0.0)]
    with pytest.raises(ValueError, match="max_component_events=2.*prune"):
        exact_partitions_componentwise(3, candidates, max_component_events=2)
    with pytest.raises(ValueError, match="max_component_edges=1.*prune"):
        exact_partitions_componentwise(3, candidates, max_component_edges=1)
    with pytest.raises(ValueError, match="max_total_partitions=2.*prune"):
        exact_partitions_componentwise(3, candidates, max_total_partitions=2)


def test_diagnostics_include_component_summary():
    candidates = [CandidatePair(0, 1, 0.0), CandidatePair(2, 3, 0.0)]
    states, _, _ = exact_partitions_componentwise(
        4, candidates, max_total_partitions=10
    )
    diag = compute_marginalized_partition_diagnostics(states, candidates, lambda s: 0.0)
    assert diag["n_components"] == 2
    assert diag["component_event_indices"] == [[0, 1], [2, 3]]
    assert diag["component_candidate_edge_indices"] == [[0], [1]]
    assert diag["component_n_partitions"] == [2, 2]
    assert diag["component_expected_n_pairs"] == pytest.approx([0.5, 0.5])
    assert diag["component_max_p_pair"] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("component_mode", ["global", "componentwise"])
def test_diagnostics_handle_events_above_the_largest_candidate_endpoint(component_mode):
    """Regression: n_events was derived ONLY from the candidate endpoints, so
    the singleton-probability accumulator was allocated too short whenever an
    observed event had no candidate partner at all (real campaign candidate
    files have n_events=280 with max endpoint 277) -> IndexError in the
    midpoint-diagnostics phase, after the full data load.  The componentwise
    twin already widened with the states' own indices."""
    candidates = [CandidatePair(0, 1, 0.5), CandidatePair(2, 3, -0.3)]
    n_events = 6  # events 4, 5 are candidate-free singletons in every partition
    if component_mode == "global":
        states = enumerate_compatible_partitions(n_events, candidates)
    else:
        states, _, _ = exact_partitions_componentwise(
            n_events, candidates, max_total_partitions=100
        )
    diag = compute_marginalized_partition_diagnostics(
        states, candidates, lambda s: float(s.n_pairs)
    )
    singleton_probs = diag["posterior_singleton_probability"]
    assert len(singleton_probs) == n_events
    # events 4 and 5 are singletons in every compatible partition
    assert singleton_probs[4] == pytest.approx(1.0)
    assert singleton_probs[5] == pytest.approx(1.0)
    assert sum(singleton_probs) == pytest.approx(diag["expected_n_singletons"])



def test_factorized_component_diagnostics_match_global_with_count_coupling():
    candidates = [CandidatePair(0, 1, math.log(2.0)), CandidatePair(2, 3, math.log(3.0))]
    global_states = enumerate_compatible_partitions(4, candidates)
    summaries, component_states, approximate_total = exact_partition_components(4, candidates)
    assert approximate_total == len(global_states) == 4

    baseline = 7.0
    edge_delta = {0: 0.25, 1: -0.4}
    count_delta = [0.0, -0.3, -1.1]

    def loglike(state):
        edges = np.asarray(state.candidate_edge_indices, dtype=int).tolist()
        return baseline + count_delta[state.n_pairs] + sum(edge_delta[e] for e in edges)

    global_diag = compute_marginalized_partition_diagnostics(
        global_states, candidates, loglike
    )
    component_deltas = []
    for states in component_states:
        component_deltas.append(
            [sum(edge_delta[e] for e in np.asarray(state.candidate_edge_indices, dtype=int).tolist()) for state in states]
        )
    fact_diag = compute_componentwise_factorized_partition_diagnostics(
        component_states,
        candidates,
        component_deltas,
        count_delta,
        baseline,
        component_summaries=summaries,
    )

    assert fact_diag["factorized_exact"] is True
    assert fact_diag["global_partitions_enumerated"] is False
    assert fact_diag["logL_marginalized"] == pytest.approx(global_diag["logL_marginalized"], abs=1e-10)
    assert fact_diag["expected_n_pairs"] == pytest.approx(global_diag["expected_n_pairs"], abs=1e-10)
    np.testing.assert_allclose(
        [x["p_pair"] for x in fact_diag["posterior_pair_probabilities"]],
        [x["p_pair"] for x in global_diag["posterior_pair_probabilities"]],
        atol=1e-10,
    )
    assert {tuple(x) for x in fact_diag["map_partition"]["pair_indices"]} == {
        tuple(x) for x in global_diag["map_partition"]["pair_indices"]
    }
    rows = component_partition_diagnostic_rows(fact_diag, case="case")
    assert len(rows) == sum(len(states) for states in component_states)
def test_single_candidate_graph_has_empty_and_pair_partitions():
    data = {
        "n_events": 2,
        "candidate_pairs": [{"i": 1, "j": 0, "log_prior_odds": 0.0, "label": "x"}],
    }
    n_events, states, _ = exact_partitions_from_json(data)
    assert n_events == 2
    assert len(states) == 2
    assert states[0].n_pairs == 0
    assert states[1].pair_indices.tolist() == [[0, 1]]


def test_map_partition_uses_highest_posterior_log_weight():
    candidates = [CandidatePair(0, 1, 0.0), CandidatePair(2, 3, 0.0)]
    states = enumerate_compatible_partitions(4, candidates)
    target = next(i for i, s in enumerate(states) if s.n_pairs == 2)
    diag = compute_marginalized_partition_diagnostics(
        states, candidates, lambda s: 10.0 if s.n_pairs == 2 else 0.0
    )
    assert diag["map_partition_index"] == target
    assert diag["map_partition"]["pair_indices"] == [[0, 1], [2, 3]]


def test_expected_n_pairs_equals_sum_of_candidate_pair_probabilities():
    candidates = [CandidatePair(0, 1, 0.2), CandidatePair(1, 2, -0.1)]
    states = enumerate_compatible_partitions(3, candidates)
    diag = compute_marginalized_partition_diagnostics(
        states, candidates, lambda s: float(s.n_pairs)
    )
    pair_sum = sum(item["p_pair"] for item in diag["posterior_pair_probabilities"])
    assert diag["expected_n_pairs"] == pytest.approx(pair_sum)
    assert diag["logL_marginalized"] == pytest.approx(
        logsumexp(
            np.array(diag["partition_log_prior_weight"])
            + np.array(diag["partition_logL"])
        )
        - diag["log_z_partition_prior"]
    )


def test_validate_candidate_pairs_accepts_edge_time_marks():
    from darksirens.lensing.partitions import validate_candidate_pairs

    n_events, pairs = validate_candidate_pairs(
        {
            "n_events": 2,
            "candidate_pairs": [
                {
                    "i": 1,
                    "j": 0,
                    "log_prior_odds": 0.0,
                    "marks": {"delta_t_obs": 12.5, "sigma_delta_t": 2.0},
                }
            ],
        }
    )
    assert n_events == 2
    assert pairs[0].i == 0
    assert pairs[0].j == 1
    assert pairs[0].delta_t_obs == pytest.approx(12.5)
    assert pairs[0].sigma_delta_t == pytest.approx(2.0)


def test_validate_candidate_pairs_rejects_incomplete_or_invalid_time_marks():
    from darksirens.lensing.partitions import validate_candidate_pairs

    base = {"n_events": 2, "candidate_pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0}]}
    for marks in (
        {"delta_t_obs": 1.0},
        {"sigma_delta_t": 1.0},
        {"delta_t_obs": float("nan"), "sigma_delta_t": 1.0},
        {"delta_t_obs": 1.0, "sigma_delta_t": 0.0},
    ):
        data = {
            **base,
            "candidate_pairs": [{**base["candidate_pairs"][0], "marks": marks}],
        }
        with pytest.raises(ValueError):
            validate_candidate_pairs(data)


def test_enumerate_compatible_partitions_preserves_candidate_edge_indices():
    candidates = [CandidatePair(0, 1, 0.0), CandidatePair(2, 3, 0.0)]
    states = enumerate_compatible_partitions(4, candidates)
    assert any(s.candidate_edge_indices.tolist() == [0, 1] for s in states)


def _minimal_gw(path, n_events=2, nsamp=2):
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["nobs"] = n_events
        f.attrs["nsamp"] = nsamp
        for name in ("m1det", "m2det", "dL", "chieff", "p_pe"):
            f.create_dataset(name, data=np.ones(n_events * nsamp))


def _minimal_selection(path):
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["ok"] = True


def _minimal_lensed(path):
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["Ndraw_sources"] = 1
        f.create_dataset("p_tag_per_source", data=np.array([1.0]))


def _partition(path):
    import json

    path.write_text(
        json.dumps(
            {
                "n_singletons": 0,
                "n_pairs": 1,
                "singleton_indices": [],
                "pair_indices": [[0, 1]],
            }
        )
    )


def _metadata_h5(path, *, image_groups=False):
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "lensing-pair-metadata-1.0"
        f.attrs["npairs"] = 1
        g = f.create_group("pair_0")
        g.attrs["event_index_image0"] = 0
        g.attrs["event_index_image1"] = 1
        g.attrs["delta_t_obs"] = 1.0
        g.attrs["sigma_delta_t"] = 0.1
        if image_groups:
            for im in ("image0", "image1"):
                gi = g.create_group(im)
                for d in ("m1det", "q", "dL_app", "chieff", "prior_wt"):
                    gi.create_dataset(d, data=np.ones(2))


def test_metadata_only_pair_file_preflight_unified_and_fails_legacy(tmp_path):
    from types import SimpleNamespace
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    part = tmp_path / "partition.json"
    meta = tmp_path / "pair_meta.h5"
    _minimal_gw(gw)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    _partition(part)
    _metadata_h5(meta)
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=str(part),
        candidate_pairs_path=None,
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=str(meta),
        cluster_mode="j2",
        partition_mode="fixed",
        pair_marks="time",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
    )
    report = run_lensing_preflight(opts)
    assert report["ok"], report
    opts.observed_catalog_path = None
    _minimal_gw(gw, n_events=3)  # partition no longer implies unified catalog
    opts.pair_metadata_path = None
    opts.pair_pe_path = str(meta)
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("missing image0" in e for e in report["errors"])


def test_candidate_pairs_marks_suffice_for_marginalized_preflight(tmp_path):
    from types import SimpleNamespace
    import json
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    cand = tmp_path / "candidate_pairs.json"
    _minimal_gw(gw)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    cand.write_text(
        json.dumps(
            {
                "n_events": 2,
                "candidate_pairs": [
                    {
                        "i": 0,
                        "j": 1,
                        "log_prior_odds": 0.0,
                        "marks": {"delta_t_obs": 1.0, "sigma_delta_t": 0.1},
                    }
                ],
            }
        )
    )
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=None,
        candidate_pairs_path=str(cand),
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=None,
        cluster_mode="j2",
        partition_mode="marginalize_exact",
        pair_marks="time",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
    )
    report = run_lensing_preflight(opts)
    assert report["ok"], report



def test_componentwise_preflight_warns_on_total_cap_but_global_fails(tmp_path):
    from types import SimpleNamespace
    import json
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    cand = tmp_path / "candidate_pairs.json"
    _minimal_gw(gw, n_events=8)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    cand.write_text(
        json.dumps(
            {
                "n_events": 8,
                "candidate_pairs": [
                    {"i": 0, "j": 1, "log_prior_odds": 0.0},
                    {"i": 2, "j": 3, "log_prior_odds": 0.0},
                    {"i": 4, "j": 5, "log_prior_odds": 0.0},
                    {"i": 6, "j": 7, "log_prior_odds": 0.0},
                ],
            }
        )
    )
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=None,
        candidate_pairs_path=str(cand),
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=None,
        cluster_mode="j2",
        partition_mode="marginalize_exact",
        partition_component_mode="componentwise",
        pair_marks="none",
        pair_time_sigma_sec=None,
        edge_mark_prior_keys=None,
        edge_mark_likelihood_keys=None,
        max_exact_partitions=10000,
        max_component_events=None,
        max_component_edges=None,
        max_component_partitions=4,
        max_total_partitions=4,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
    )
    report = run_lensing_preflight(opts)
    assert report["ok"], report
    summary = report["summary"]
    assert summary["factorized_exact_enabled"] is True
    assert summary["approximate_total_partitions"] == 16
    assert summary["component_n_partitions"] == [2, 2, 2, 2]
    assert any("exceeds max_total_partitions=4" in w for w in report["warnings"])

    opts.partition_component_mode = "global"
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("exact partition enumeration exceeded max_partitions=4" in e for e in report["errors"])

    opts.partition_component_mode = "componentwise"
    opts.max_component_partitions = 1
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("max_partitions=1" in e for e in report["errors"])
def test_unified_legacy_pair_pe_warns_image_groups_ignored(tmp_path):
    from types import SimpleNamespace
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    part = tmp_path / "partition.json"
    pe = tmp_path / "pair_pe.h5"
    _minimal_gw(gw)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    _partition(part)
    _metadata_h5(pe, image_groups=True)
    obs = tmp_path / "observed_catalog.json"
    obs.write_text(
        '{"format_version":"observed-lensing-catalog-1.0","event_indexing":"global","n_events":2,"events":[{"event_index":0,"event_id":"e0"},{"event_index":1,"event_id":"e1"}]}'
    )
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=str(part),
        candidate_pairs_path=None,
        observed_catalog_path=str(obs),
        pair_pe_path=str(pe),
        pair_metadata_path=None,
        cluster_mode="j2",
        partition_mode="fixed",
        pair_marks="none",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
    )
    report = run_lensing_preflight(opts)
    assert report["ok"], report
    assert any("ignores them" in w for w in report["warnings"])


def _observed_catalog_with_truth(path, n_events=4):
    import json

    events = []
    for i in range(n_events):
        events.append(
            {
                "event_index": i,
                "event_id": f"e{i}",
                "gps_time": 1000.0 + i,
                "truth_source_id": 10 if i in (0, 1) else 20 + i,
                "truth_image_index": i if i in (0, 1) else None,
                "truth_is_lensed_image": i in (0, 1),
                "ra_mean": 1.0 + (0.01 * i if i in (0, 1) else 1.0 + i),
                "dec_mean": 0.5 + (0.01 * i if i in (0, 1) else 0.5 + 0.1 * i),
                "sky_sigma_rad": 0.05,
                "ra_true": 1.0 if i in (0, 1) else 2.0 + i,
                "dec_true": 0.5 if i in (0, 1) else 1.0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "format_version": "observed-lensing-catalog-1.0",
                "event_indexing": "global",
                "n_events": n_events,
                "events": events,
            }
        )
    )
    return path


def _observed_gw(path, n_events=4, nsamp=8):
    import h5py

    m1 = []
    m2 = []
    dL = []
    chi = []
    for i in range(n_events):
        center = 30.0 if i in (0, 1) else 50.0 + i
        m1.extend(center + 0.01 * np.arange(nsamp))
        m2.extend(0.8 * (center + 0.01 * np.arange(nsamp)))
        dL.extend((1000.0 + 10 * i) + np.arange(nsamp))
        chi.extend(np.zeros(nsamp) + 0.01 * i)
    with h5py.File(path, "w") as f:
        f.attrs["nsamp"] = nsamp
        f.attrs["nobs"] = n_events
        f.create_dataset("m1det", data=np.asarray(m1))
        f.create_dataset("m2det", data=np.asarray(m2))
        f.create_dataset("dL", data=np.asarray(dL))
        f.create_dataset("chieff", data=np.asarray(chi))
    return path


def test_observed_builder_labels_and_degree_limit(tmp_path):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )
    from darksirens.lensing.partitions import validate_candidate_pairs

    gw = _observed_gw(tmp_path / "gw.h5")
    cat = _observed_catalog_with_truth(tmp_path / "observed_catalog.json")
    data = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=1,
        max_total_edges=10,
        time_window_sec=100.0,
        include_time_marks=True,
        include_truth_labels=True,
        seed=3,
    )
    assert data["format_version"] == "candidate-pairs-1.0"
    assert data["n_events"] == 4
    degree = [0] * 4
    for edge in data["pairs"]:
        degree[edge["i"]] += 1
        degree[edge["j"]] += 1
    assert max(degree) <= 1
    assert any(
        edge["i"] == 0 and edge["j"] == 1 and edge["label"] == "true"
        for edge in data["pairs"]
    )
    validate_candidate_pairs(data)
    assert all(
        "delta_t_obs" not in edge["marks"] and "sigma_delta_t" not in edge["marks"]
        for edge in data["pairs"]
    )


def test_observed_builder_without_truth_labels(tmp_path):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )

    gw = _observed_gw(tmp_path / "gw.h5", n_events=3)
    cat = _observed_catalog_with_truth(tmp_path / "observed_catalog.json", n_events=3)
    data = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=2,
        max_total_edges=2,
        include_time_marks=False,
        include_truth_labels=False,
        seed=4,
    )
    assert len(data["pairs"]) <= 2
    assert all("label" not in edge for edge in data["pairs"])
    assert all("log_mass_distance_score" in edge["marks"] for edge in data["pairs"])


def test_observed_builder_defaults_to_no_time_marks(tmp_path):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )

    gw = _observed_gw(tmp_path / "gw.h5", n_events=3)
    cat = _observed_catalog_with_truth(tmp_path / "observed_catalog.json", n_events=3)
    data = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=2,
        max_total_edges=2,
        seed=5,
    )
    assert all("delta_t_obs" not in edge["marks"] for edge in data["pairs"])
    assert all("sigma_delta_t" not in edge["marks"] for edge in data["pairs"])


def test_validate_candidate_pairs_accepts_generic_log_marks():
    from darksirens.lensing.partitions import validate_candidate_pairs

    n_events, pairs = validate_candidate_pairs(
        {
            "n_events": 2,
            "candidate_pairs": [
                {
                    "i": 0,
                    "j": 1,
                    "log_prior_odds": -1.0,
                    "marks": {
                        "delta_t_obs": 12.5,
                        "sigma_delta_t": 2.0,
                        "log_sky_overlap": -0.2,
                        "log_mass_distance_score": -0.3,
                        "log_custom_score": -0.4,
                    },
                }
            ],
        }
    )
    assert n_events == 2
    assert pairs[0].marks.log_sky_overlap == pytest.approx(-0.2)
    assert pairs[0].marks.extra["log_custom_score"] == pytest.approx(-0.4)


def test_validate_candidate_pairs_rejects_bad_generic_marks():
    from darksirens.lensing.partitions import validate_candidate_pairs

    base = {"n_events": 2, "candidate_pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0}]}
    for marks in (
        {"sky_overlap": 1.0},
        {"log_sky_overlap": float("inf")},
        {"log_custom": float("nan")},
    ):
        data = {
            **base,
            "candidate_pairs": [{**base["candidate_pairs"][0], "marks": marks}],
        }
        with pytest.raises(ValueError):
            validate_candidate_pairs(data)


def test_requested_prior_key_adds_to_effective_edge_log_prior_once():
    from darksirens.lensing.partitions import (
        exact_partitions_from_pairs,
        exact_partitions_from_json,
        prepare_candidate_pairs_for_partitioning,
    )

    data = {
        "n_events": 2,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": -3.0,
                "marks": {"log_sky_overlap": -2.0},
            }
        ],
    }
    _, states, _ = exact_partitions_from_json(
        data, edge_mark_prior_keys="log_sky_overlap"
    )
    pair_state = next(s for s in states if s.n_pairs == 1)
    assert pair_state.log_prior_weight == pytest.approx(-5.0)

    n_events, raw_pairs, effective_pairs, contributions = (
        prepare_candidate_pairs_for_partitioning(data, "log_sky_overlap")
    )
    assert raw_pairs[0].log_prior_odds == pytest.approx(-3.0)
    assert effective_pairs[0].log_prior_odds == pytest.approx(-5.0)
    assert contributions == pytest.approx((-2.0,))
    _, states, _ = exact_partitions_from_pairs(n_events, effective_pairs)
    pair_state = next(s for s in states if s.n_pairs == 1)
    assert pair_state.log_prior_weight == pytest.approx(-5.0)


def test_posterior_pair_probabilities_include_prior_semantics_and_marks():
    from darksirens.lensing.partitions import EdgeMarks

    raw = [CandidatePair(0, 1, -3.0, marks=EdgeMarks(log_sky_overlap=-2.0))]
    candidates = [CandidatePair(0, 1, -5.0, marks=EdgeMarks(log_sky_overlap=-2.0))]
    states = enumerate_compatible_partitions(2, candidates)
    diag = compute_marginalized_partition_diagnostics(
        states,
        candidates,
        lambda _s: 0.0,
        raw_candidate_pairs=raw,
        edge_mark_prior_contributions=(-2.0,),
    )
    item = diag["posterior_pair_probabilities"][0]
    assert item["marks"] == {"log_sky_overlap": -2.0}
    assert item["log_prior_odds"] == pytest.approx(-5.0)
    assert item["log_prior_odds_raw"] == pytest.approx(-3.0)
    assert item["log_prior_odds_effective"] == pytest.approx(-5.0)
    assert item["edge_mark_prior_contribution"] == pytest.approx(-2.0)


def test_preflight_fails_missing_requested_prior_mark(tmp_path):
    from types import SimpleNamespace
    import json
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    cand = tmp_path / "candidate_pairs.json"
    _minimal_gw(gw)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    cand.write_text(
        json.dumps(
            {
                "n_events": 2,
                "candidate_pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0}],
            }
        )
    )
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=None,
        candidate_pairs_path=str(cand),
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=None,
        cluster_mode="j2",
        partition_mode="marginalize_exact",
        pair_marks="none",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
        edge_mark_prior_keys="log_sky_overlap",
        edge_mark_likelihood_keys="",
    )
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("missing requested edge prior mark" in e for e in report["errors"])


def test_observed_builder_computes_finite_sky_overlap_and_true_pairs_are_higher(
    tmp_path,
):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )

    gw = _observed_gw(tmp_path / "gw.h5", n_events=4)
    cat = _observed_catalog_with_truth(tmp_path / "observed_catalog.json", n_events=4)
    data = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=3,
        max_total_edges=6,
        include_time_marks=False,
        include_truth_labels=True,
        include_sky_marks=True,
        seed=9,
    )
    overlaps = [edge["marks"]["log_sky_overlap"] for edge in data["pairs"]]
    assert all(np.isfinite(overlaps))
    true_overlap = next(
        edge["marks"]["log_sky_overlap"]
        for edge in data["pairs"]
        if edge.get("label") == "true"
    )
    wrong = [
        edge["marks"]["log_sky_overlap"]
        for edge in data["pairs"]
        if edge.get("label") == "wrong"
    ]
    assert wrong
    assert true_overlap > max(wrong)


def test_preflight_rejects_invalid_requested_sky_overlap_mark(tmp_path):
    from types import SimpleNamespace
    import json
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"
    sel = tmp_path / "sel.h5"
    linj = tmp_path / "linj.h5"
    cand = tmp_path / "candidate_pairs.json"
    _minimal_gw(gw)
    _minimal_selection(sel)
    _minimal_lensed(linj)
    cand.write_text(
        json.dumps(
            {
                "n_events": 2,
                "candidate_pairs": [
                    {
                        "i": 0,
                        "j": 1,
                        "log_prior_odds": 0.0,
                        "marks": {"log_sky_overlap": float("nan")},
                    }
                ],
            }
        )
    )
    opts = SimpleNamespace(
        gw_path=str(gw),
        gwselection_path=str(sel),
        lensed_injections_path=str(linj),
        partition_path=None,
        candidate_pairs_path=str(cand),
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=None,
        cluster_mode="j2",
        partition_mode="marginalize_exact",
        pair_marks="none",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
        edge_mark_prior_keys="log_sky_overlap",
        edge_mark_likelihood_keys="",
    )
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("log_sky_overlap" in e for e in report["errors"])


def test_suspicious_small_integer_time_marks_are_flagged():
    from darksirens.lensing.partitions import EdgeMarks

    candidates = [
        CandidatePair(0, 1, 0.0, marks=EdgeMarks(delta_t_obs=1.0, sigma_delta_t=1.0)),
        CandidatePair(2, 3, 0.0, marks=EdgeMarks(delta_t_obs=2.0, sigma_delta_t=1.0)),
    ]
    states = enumerate_compatible_partitions(4, candidates)
    diag = compute_marginalized_partition_diagnostics(states, candidates, lambda _s: 0.0)
    assert diag["candidate_time_marks_placeholder"] is True
    assert diag["candidate_time_marks_suspicious"] is True
    assert "small integer" in diag["candidate_time_marks_warning"]
    assert "pair_time_placeholder_warning" in diag["posterior_pair_probabilities"][0]


def test_partition_diagnostic_rows_handles_fake_exact_payload():
    from darksirens.lensing.marginal_diagnostics import partition_diagnostic_rows

    diag = {
        "n_partitions": 2,
        "map_partition_index": 1,
        "partition_logL": [-2.0, -1.0],
        "partition_log_prior_weight": [0.0, -0.5],
        "partition_log_posterior_weight": [-1.0, -0.2],
        "partition_posterior_probability": [0.3, 0.7],
        "partitions": [
            {"n_pairs": 0, "pair_edges": []},
            {"n_pairs": 1, "pair_edges": [[0, 1]]},
        ],
    }
    rows = partition_diagnostic_rows(diag, case="X", truth_edges={(0, 1)})
    assert rows[1]["case"] == "X"
    assert rows[1]["pair_edges"] == "[[0, 1]]"
    assert rows[1]["is_map_partition"] is True
    assert rows[1]["is_truth_partition"] is True
    assert rows[1]["n_true_edges"] == 1
    assert rows[1]["n_false_edges"] == 0


def test_folded_mark_keys_guard_rejects_double_count():
    from darksirens.lensing.partitions import prepare_candidate_pairs_for_partitioning

    data = {
        "n_events": 2,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": -3.0,
                "marks": {
                    "log_mass_distance_score": -1.0,
                    "log_sky_overlap": -2.0,
                },
            }
        ],
        "folded_mark_keys": ["log_mass_distance_score"],
    }
    # Requesting a key the builder already folded into log_prior_odds must fail.
    with pytest.raises(ValueError, match="double-count"):
        prepare_candidate_pairs_for_partitioning(data, "log_mass_distance_score")
    # Non-folded keys still apply exactly once.
    _, raw_pairs, effective_pairs, contributions = (
        prepare_candidate_pairs_for_partitioning(data, "log_sky_overlap")
    )
    assert raw_pairs[0].log_prior_odds == pytest.approx(-3.0)
    assert effective_pairs[0].log_prior_odds == pytest.approx(-5.0)
    assert contributions == pytest.approx((-2.0,))
    # Legacy files without the metadata keep the old permissive behavior.
    legacy = {k: v for k, v in data.items() if k != "folded_mark_keys"}
    _, _, effective_legacy, _ = prepare_candidate_pairs_for_partitioning(
        legacy, "log_mass_distance_score"
    )
    assert effective_legacy[0].log_prior_odds == pytest.approx(-4.0)


def test_parse_folded_mark_keys_validates_schema():
    from darksirens.lensing.partitions import parse_folded_mark_keys

    assert parse_folded_mark_keys({}) == ()
    assert parse_folded_mark_keys({"folded_mark_keys": []}) == ()
    assert parse_folded_mark_keys({"folded_mark_keys": ["log_a", "log_b"]}) == (
        "log_a",
        "log_b",
    )
    for bad in ("log_a", [1], [""], 3, {"log_a": True}):
        with pytest.raises(ValueError, match="folded_mark_keys"):
            parse_folded_mark_keys({"folded_mark_keys": bad})


def test_observed_builder_records_folded_mark_keys(tmp_path):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )
    from darksirens.lensing.partitions import prepare_candidate_pairs_for_partitioning

    gw = _observed_gw(tmp_path / "gw.h5", n_events=3)
    cat = _observed_catalog_with_truth(tmp_path / "observed_catalog.json", n_events=3)
    data = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=2,
        max_total_edges=2,
        seed=6,
    )
    # Mass-distance score is always folded into log_prior_odds by the builder.
    assert data["folded_mark_keys"] == ["log_mass_distance_score"]
    with pytest.raises(ValueError, match="double-count"):
        prepare_candidate_pairs_for_partitioning(data, "log_mass_distance_score")
    # Sky overlap is folded only when it carries nonzero weight in the score.
    data_weighted = build_candidate_pairs(
        gw_path=gw,
        observed_catalog_path=cat,
        max_edges_per_event=2,
        max_total_edges=2,
        sky_overlap_weight=1.0,
        seed=6,
    )
    assert set(data_weighted["folded_mark_keys"]) == {
        "log_mass_distance_score",
        "log_sky_overlap",
    }
    with pytest.raises(ValueError, match="double-count"):
        prepare_candidate_pairs_for_partitioning(data_weighted, "log_sky_overlap")
    # The default study configuration (log_sky_overlap requested, weight 0)
    # must keep working.
    prepare_candidate_pairs_for_partitioning(data, "log_sky_overlap")


def test_preflight_flags_folded_edge_mark_double_count(tmp_path):
    import json
    from types import SimpleNamespace

    from darksirens.lensing.preflight import _check_candidates

    data = {
        "format_version": "candidate-pairs-1.0",
        "n_events": 2,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": -3.0,
                "marks": {"log_mass_distance_score": -1.0},
            }
        ],
        "folded_mark_keys": ["log_mass_distance_score"],
    }
    path = tmp_path / "candidate_pairs.json"
    path.write_text(json.dumps(data))
    errors: list = []
    warns: list = []
    summary: dict = {}
    opts = SimpleNamespace(
        edge_mark_prior_keys="log_mass_distance_score",
        edge_mark_likelihood_keys="",
        pair_marks="none",
    )
    _check_candidates(str(path), 2, opts, errors, warns, summary)
    assert any("double-count" in e for e in errors)
    # Without the folded key requested there is no error.
    errors2: list = []
    opts_ok = SimpleNamespace(
        edge_mark_prior_keys="",
        edge_mark_likelihood_keys="",
        pair_marks="none",
    )
    _check_candidates(str(path), 2, opts_ok, errors2, [], {})
    assert not errors2


def test_cli_gate_refuses_suspicious_time_marks():
    from types import SimpleNamespace

    from darksirens.cli.inference_lensing import _gate_suspicious_time_marks
    from darksirens.lensing.partitions import validate_candidate_pairs

    placeholder = {
        "n_events": 4,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": 0.0,
                "marks": {"delta_t_obs": 1.0, "sigma_delta_t": 1.0},
            },
            {
                "i": 2,
                "j": 3,
                "log_prior_odds": 0.0,
                "marks": {"delta_t_obs": 2.0, "sigma_delta_t": 1.0},
            },
        ],
    }
    _, pairs = validate_candidate_pairs(placeholder)
    strict = SimpleNamespace(
        pair_marks="time",
        edge_mark_likelihood_keys="",
        candidate_pairs_path="candidate_pairs.json",
        allow_suspicious_time_marks=False,
    )
    with pytest.raises(SystemExit, match="suspicious candidate time marks"):
        _gate_suspicious_time_marks(strict, pairs)
    # The escape hatch downgrades the error to a warning.
    permissive = SimpleNamespace(
        pair_marks="time",
        edge_mark_likelihood_keys="",
        candidate_pairs_path="candidate_pairs.json",
        allow_suspicious_time_marks="true",
    )
    with pytest.warns(RuntimeWarning, match="placeholder"):
        _gate_suspicious_time_marks(permissive, pairs)
    # Inert when time marks are not used in the likelihood.
    unused = SimpleNamespace(
        pair_marks="none",
        edge_mark_likelihood_keys="",
        allow_suspicious_time_marks=False,
    )
    _gate_suspicious_time_marks(unused, pairs)
    # Physical time marks (hour-scale sigma) pass the strict gate.
    physical = {
        "n_events": 4,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": 0.0,
                "marks": {"delta_t_obs": 43210.5, "sigma_delta_t": 3600.0},
            }
        ],
    }
    _, physical_pairs = validate_candidate_pairs(physical)
    _gate_suspicious_time_marks(strict, physical_pairs)


def test_builder_uniform_observation_times_marks_every_edge(tmp_path):
    """Catalogs with observation_times='uniform' carry physical arrivals, so
    EVERY candidate edge gets |Delta t_gps| with the catalog sigma — truth
    pairs reproduce their generator delta_t_obs exactly, false edges get
    their real separations (the case-G requirement at scale)."""
    import json

    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )

    gw = _observed_gw(tmp_path / "gw.h5", n_events=4)
    base = 1234567890.0
    dt_true_pair = 43210.5
    times = [base + 0.0, base + dt_true_pair, base + 9.9e6, base + 2.02e7]
    events = []
    for k, t in enumerate(times):
        events.append(dict(
            event_index=k, event_id=f"ev{k:03d}", kind="singleton_or_image",
            gps_time=t,
            truth_source_id=(0 if k < 2 else k), truth_image_index=(k if k < 2 else None),
            truth_is_lensed_image=(k < 2),
            ra_mean=0.1 * k, dec_mean=0.05 * k, sky_sigma_rad=0.01,
            ra_true=0.1 * k, dec_true=0.05 * k,
        ))
    catalog = dict(
        format_version="observed-lensing-catalog-1.0",
        event_indexing="global",
        observation_times="uniform",
        t_obs_days=365.25,
        time_delay_sigma_sec=3600.0,
        n_events=4,
        events=events,
    )
    path = tmp_path / "observed_catalog.json"
    path.write_text(json.dumps(catalog))

    data = build_candidate_pairs(
        gw_path=gw, observed_catalog_path=path,
        max_edges_per_event=3, max_total_edges=10,
        include_time_marks=True, include_truth_labels=True, seed=1,
    )
    assert data["pairs"], "no candidate edges built"
    for edge in data["pairs"]:
        marks = edge["marks"]
        assert "delta_t_obs" in marks and "sigma_delta_t" in marks, edge
        expected = abs(times[edge["i"]] - times[edge["j"]])
        assert marks["delta_t_obs"] == pytest.approx(expected)
        assert marks["sigma_delta_t"] == pytest.approx(3600.0)
    true_edges = [e for e in data["pairs"] if e.get("label") == "true"]
    assert true_edges and true_edges[0]["marks"]["delta_t_obs"] == pytest.approx(dt_true_pair)


def test_builder_uniform_mode_requires_catalog_sigma(tmp_path):
    import json

    from scripts.mock_lensing.build_candidate_pairs_from_observed import (
        build_candidate_pairs,
    )

    gw = _observed_gw(tmp_path / "gw.h5", n_events=3)
    catalog = dict(
        format_version="observed-lensing-catalog-1.0",
        event_indexing="global",
        observation_times="uniform",
        n_events=3,
        events=[dict(event_index=k, event_id=f"e{k}", kind="singleton_or_image",
                     gps_time=1e9 + 1e5 * k, truth_source_id=k,
                     truth_image_index=None, truth_is_lensed_image=False,
                     ra_mean=0.0, dec_mean=0.0, sky_sigma_rad=0.01,
                     ra_true=0.0, dec_true=0.0) for k in range(3)],
    )
    path = tmp_path / "observed_catalog.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="time_delay_sigma_sec"):
        build_candidate_pairs(
            gw_path=gw, observed_catalog_path=path,
            max_edges_per_event=2, max_total_edges=4,
            include_time_marks=True, seed=1,
        )


# ---------------------------------------------------------------------------
# fixed-partition coverage
# ---------------------------------------------------------------------------

def _preflight_opts(tmp_path, partition_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        gw_path=str(tmp_path / "gw.h5"),
        gwselection_path=str(tmp_path / "sel.h5"),
        lensed_injections_path=str(tmp_path / "linj.h5"),
        partition_path=str(partition_path),
        candidate_pairs_path=None,
        observed_catalog_path=None,
        pair_pe_path=None,
        pair_metadata_path=str(tmp_path / "pair_meta.h5"),
        cluster_mode="j2",
        partition_mode="fixed",
        pair_marks="time",
        pair_time_sigma_sec=None,
        max_exact_partitions=10000,
        wl_selection="standard",
        wl_backend="lognormal",
        fix_lens_rate=True,
        sl_tau_A=5e-4,
        sl_tau_n=3.0,
        lens_prior_overrides=None,
    )


def test_preflight_rejects_a_fixed_partition_that_drops_events(tmp_path):
    """A fixed partition must partition the WHOLE observed catalog.  Shape,
    count self-consistency, index range and duplicates were all checked, but not
    coverage: the cluster likelihood only touches the listed indices, so an
    uncovered event was silently excluded from the product while results.hdf5
    still reported the PE file's n_events."""
    import json

    from darksirens.lensing.preflight import run_lensing_preflight

    _minimal_gw(tmp_path / "gw.h5", n_events=4)
    _minimal_selection(tmp_path / "sel.h5")
    _minimal_lensed(tmp_path / "linj.h5")
    _metadata_h5(tmp_path / "pair_meta.h5")
    part = tmp_path / "partition.json"
    part.write_text(
        json.dumps(
            {
                "n_singletons": 1,
                "n_pairs": 1,
                "singleton_indices": [2],   # event 3 covered by nothing
                "pair_indices": [[0, 1]],
            }
        )
    )
    report = run_lensing_preflight(_preflight_opts(tmp_path, part))
    assert not report["ok"]
    coverage = [e for e in report["errors"] if "does not cover every observed event" in e]
    assert coverage, report["errors"]
    assert "Missing: 3" in coverage[0]


def test_preflight_accepts_a_covering_fixed_partition(tmp_path):
    """The complementary cell: the same files with event 3 as a singleton pass."""
    import json

    from darksirens.lensing.preflight import run_lensing_preflight

    _minimal_gw(tmp_path / "gw.h5", n_events=4)
    _minimal_selection(tmp_path / "sel.h5")
    _minimal_lensed(tmp_path / "linj.h5")
    _metadata_h5(tmp_path / "pair_meta.h5")
    part = tmp_path / "partition.json"
    part.write_text(
        json.dumps(
            {
                "n_singletons": 2,
                "n_pairs": 1,
                "singleton_indices": [2, 3],
                "pair_indices": [[0, 1]],
            }
        )
    )
    report = run_lensing_preflight(_preflight_opts(tmp_path, part))
    assert report["ok"], report


def test_fixed_partition_coverage_error_lists_the_dropped_events():
    from darksirens.lensing.preflight import fixed_partition_coverage_error

    assert fixed_partition_coverage_error([0, 1, 2], 3) is None
    assert fixed_partition_coverage_error([0, 1, 2], None) is None
    msg = fixed_partition_coverage_error(list(range(90)), 100, path="p.json")
    assert "10 of 100 events" in msg
    assert "p.json" in msg
    assert "Missing: 90, 91" in msg
    # long lists are truncated rather than dumping thousands of indices
    long_msg = fixed_partition_coverage_error([0], 100)
    assert "(+79 more)" in long_msg
