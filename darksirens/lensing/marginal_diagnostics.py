"""Posterior diagnostics for exact partition-marginalized lensing runs."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.special import logsumexp

from darksirens.lensing.partitions import CandidatePair, PartitionState


def _partition_object(state: PartitionState) -> dict:
    return {
        "singleton_indices": np.asarray(state.singleton_indices, dtype=int).tolist(),
        "pair_indices": np.asarray(state.pair_indices, dtype=int)
        .reshape((-1, 2))
        .tolist(),
        "n_singletons": int(state.n_singletons),
        "n_pairs": int(state.n_pairs),
    }


def compute_marginalized_partition_diagnostics(
    partition_states: Iterable[PartitionState],
    candidate_pairs: Iterable[CandidatePair],
    partition_loglike: Callable[[PartitionState], float],
    *,
    log_z_partition_prior: float | None = None,
) -> dict:
    """Compute exact posterior diagnostics over a finite set of partitions."""
    states = tuple(partition_states)
    candidates = tuple(candidate_pairs)
    if not states:
        raise ValueError("at least one partition is required")

    log_prior = np.asarray([s.log_prior_weight for s in states], dtype=float)
    logL = np.asarray([float(partition_loglike(s)) for s in states], dtype=float)
    if log_z_partition_prior is None:
        log_z_partition_prior = float(logsumexp(log_prior))
    log_norm = float(logsumexp(log_prior + logL))
    log_post = log_prior + logL - log_norm
    post = np.exp(log_post)
    map_idx = int(np.argmax(log_post))

    expected_n_pairs = float(
        np.sum(post * np.asarray([s.n_pairs for s in states], dtype=float))
    )
    expected_n_singletons = float(
        np.sum(post * np.asarray([s.n_singletons for s in states], dtype=float))
    )

    pair_probs = np.zeros(len(candidates), dtype=float)
    singleton_probs = None
    n_events = 0
    for c in candidates:
        n_events = max(n_events, c.i + 1, c.j + 1)
    if n_events:
        singleton_probs = np.zeros(n_events, dtype=float)
    for w, state in zip(post, states):
        edge_idx = state.candidate_edge_indices
        if edge_idx is not None:
            pair_probs[np.asarray(edge_idx, dtype=int)] += w
        else:  # compatibility fallback by endpoint lookup
            lookup = {(c.i, c.j): i for i, c in enumerate(candidates)}
            for i, j in np.asarray(state.pair_indices, dtype=int).reshape((-1, 2)):
                pair_probs[lookup[(int(i), int(j))]] += w
        if singleton_probs is not None:
            singleton_probs[np.asarray(state.singleton_indices, dtype=int)] += w

    posterior_pair_probabilities = []
    for c, p_pair in zip(candidates, pair_probs):
        item = {
            "i": int(c.i),
            "j": int(c.j),
            "p_pair": float(p_pair),
            "log_prior_odds": float(c.log_prior_odds),
            "marks": c.marks.to_dict(),
        }
        if c.label is not None:
            item["label"] = c.label
        posterior_pair_probabilities.append(item)

    out = {
        "partition_mode": "marginalize_exact",
        "n_partitions": int(len(states)),
        "log_z_partition_prior": float(log_z_partition_prior),
        "logL_marginalized": float(log_norm - log_z_partition_prior),
        "logL_total": float(log_norm - log_z_partition_prior),
        "map_partition_index": map_idx,
        "expected_n_pairs": expected_n_pairs,
        "expected_n_singletons": expected_n_singletons,
        "partition_log_prior_weight": log_prior.tolist(),
        "partition_logL": logL.tolist(),
        "partition_log_posterior_weight": log_post.tolist(),
        "partition_posterior_probability": post.tolist(),
        "posterior_pair_probabilities": posterior_pair_probabilities,
        "map_partition": _partition_object(states[map_idx]),
    }
    if singleton_probs is not None:
        out["posterior_singleton_probability"] = singleton_probs.tolist()
    return out
