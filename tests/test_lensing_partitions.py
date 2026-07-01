import math

import numpy as np
import pytest
from scipy.special import logsumexp

from darksirens.lensing.marginal_diagnostics import compute_marginalized_partition_diagnostics
from darksirens.lensing.partitions import (
    CandidatePair,
    enumerate_compatible_partitions,
    exact_partitions_from_json,
)


def test_exact_partitions_track_candidate_edge_indices_and_prior_only_posteriors():
    candidates = [CandidatePair(0, 1, math.log(2.0), "a"), CandidatePair(1, 2, math.log(3.0), "b")]
    states = enumerate_compatible_partitions(3, candidates)
    assert [s.pair_indices.tolist() for s in states] == [[], [[1, 2]], [[0, 1]]]
    assert [s.candidate_edge_indices.tolist() for s in states] == [[], [1], [0]]

    diag = compute_marginalized_partition_diagnostics(states, candidates, lambda _s: 5.0)
    weights = np.array([1.0, 3.0, 2.0]) / 6.0
    np.testing.assert_allclose(diag["partition_posterior_probability"], weights)
    assert diag["posterior_pair_probabilities"][0]["label"] == "a"
    assert diag["posterior_pair_probabilities"][0]["p_pair"] == pytest.approx(2.0 / 6.0)
    assert diag["posterior_pair_probabilities"][1]["p_pair"] == pytest.approx(3.0 / 6.0)


def test_single_candidate_graph_has_empty_and_pair_partitions():
    data = {"n_events": 2, "candidate_pairs": [{"i": 1, "j": 0, "log_prior_odds": 0.0, "label": "x"}]}
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
        logsumexp(np.array(diag["partition_log_prior_weight"]) + np.array(diag["partition_logL"]))
        - diag["log_z_partition_prior"]
    )
