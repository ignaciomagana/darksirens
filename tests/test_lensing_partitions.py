import math

import numpy as np
import pytest
from scipy.special import logsumexp

from darksirens.lensing.marginal_diagnostics import (
    compute_marginalized_partition_diagnostics,
)
from darksirens.lensing.partitions import (
    CandidatePair,
    enumerate_compatible_partitions,
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


def test_requested_prior_key_adds_to_effective_edge_log_prior():
    from darksirens.lensing.partitions import exact_partitions_from_json

    data = {
        "n_events": 2,
        "candidate_pairs": [
            {
                "i": 0,
                "j": 1,
                "log_prior_odds": -2.0,
                "marks": {"log_sky_overlap": -0.5},
            }
        ],
    }
    _, states, _ = exact_partitions_from_json(
        data, edge_mark_prior_keys="log_sky_overlap"
    )
    pair_state = next(s for s in states if s.n_pairs == 1)
    assert pair_state.log_prior_weight == pytest.approx(-2.5)


def test_posterior_pair_probabilities_include_marks():
    from darksirens.lensing.partitions import EdgeMarks

    candidates = [CandidatePair(0, 1, 0.0, marks=EdgeMarks(log_sky_overlap=-0.2))]
    states = enumerate_compatible_partitions(2, candidates)
    diag = compute_marginalized_partition_diagnostics(
        states, candidates, lambda _s: 0.0
    )
    assert diag["posterior_pair_probabilities"][0]["marks"] == {"log_sky_overlap": -0.2}


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


def test_observed_builder_computes_finite_sky_overlap_and_true_pairs_are_higher(tmp_path):
    from scripts.mock_lensing.build_candidate_pairs_from_observed import build_candidate_pairs

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
    true_overlap = next(edge["marks"]["log_sky_overlap"] for edge in data["pairs"] if edge.get("label") == "true")
    wrong = [edge["marks"]["log_sky_overlap"] for edge in data["pairs"] if edge.get("label") == "wrong"]
    assert wrong
    assert true_overlap > max(wrong)


def test_preflight_rejects_invalid_requested_sky_overlap_mark(tmp_path):
    from types import SimpleNamespace
    import json
    from darksirens.lensing.preflight import run_lensing_preflight

    gw = tmp_path / "gw.h5"; sel = tmp_path / "sel.h5"; linj = tmp_path / "linj.h5"; cand = tmp_path / "candidate_pairs.json"
    _minimal_gw(gw); _minimal_selection(sel); _minimal_lensed(linj)
    cand.write_text(json.dumps({"n_events": 2, "candidate_pairs": [{"i": 0, "j": 1, "log_prior_odds": 0.0, "marks": {"log_sky_overlap": float("nan")}}]}))
    opts = SimpleNamespace(
        gw_path=str(gw), gwselection_path=str(sel), lensed_injections_path=str(linj),
        partition_path=None, candidate_pairs_path=str(cand), observed_catalog_path=None,
        pair_pe_path=None, pair_metadata_path=None, cluster_mode="j2", partition_mode="marginalize_exact",
        pair_marks="none", pair_time_sigma_sec=None, max_exact_partitions=10000,
        wl_selection="standard", wl_backend="lognormal", fix_lens_rate=True, sl_tau_A=5e-4, sl_tau_n=3.0,
        lens_prior_overrides=None, edge_mark_prior_keys="log_sky_overlap", edge_mark_likelihood_keys="",
    )
    report = run_lensing_preflight(opts)
    assert not report["ok"]
    assert any("log_sky_overlap" in e for e in report["errors"])
