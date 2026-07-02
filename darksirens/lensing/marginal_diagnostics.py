"""Posterior diagnostics for exact partition-marginalized lensing runs."""

from __future__ import annotations

from typing import Callable, Iterable
import json

import numpy as np
from scipy.special import logsumexp

from darksirens.lensing.partitions import (
    CandidatePair,
    PartitionState,
    connected_components_from_candidate_pairs,
    exact_component_partitions,
)


def _state_pair_edges(state: PartitionState) -> list[list[int]]:
    return np.asarray(state.pair_indices, dtype=int).reshape((-1, 2)).tolist()

def _partition_object(state: PartitionState) -> dict:
    return {
        "singleton_indices": np.asarray(state.singleton_indices, dtype=int).tolist(),
        "pair_indices": _state_pair_edges(state),
        "n_singletons": int(state.n_singletons),
        "n_pairs": int(state.n_pairs),
    }

def _looks_like_suspicious_time_mark(delta_t: float | None, sigma: float | None) -> bool:
    if delta_t is None or sigma is None:
        return False
    dt = float(delta_t)
    sig = float(sigma)
    return (
        np.isfinite(dt)
        and np.isfinite(sig)
        and np.isclose(sig, 1.0, rtol=0.0, atol=1e-12)
        and 0.0 < abs(dt) <= 10.0
        and np.isclose(abs(dt), round(abs(dt)), rtol=0.0, atol=1e-12)
    )

def candidate_time_mark_suspicion(candidate_pairs: Iterable[CandidatePair]) -> dict:
    pairs = tuple(candidate_pairs)
    marked = [p for p in pairs if p.delta_t_obs is not None and p.sigma_delta_t is not None]
    if not marked:
        return {"candidate_time_marks_placeholder": False, "candidate_time_marks_suspicious": False}
    suspicious = [p for p in marked if _looks_like_suspicious_time_mark(p.delta_t_obs, p.sigma_delta_t)]
    n_one_one = sum(
        np.isclose(float(p.delta_t_obs), 1.0, rtol=0.0, atol=1e-12)
        and np.isclose(float(p.sigma_delta_t), 1.0, rtol=0.0, atol=1e-12)
        for p in marked
    )
    placeholder = len(marked) == len(pairs) and len(suspicious) == len(marked)
    many = len(suspicious) >= max(2, int(np.ceil(0.5 * len(marked)))) or n_one_one >= 2
    warning = None
    if placeholder or many:
        warning = (
            "candidate time marks look like placeholders/synthetic values; small integer second-level time marks "
            "with sigma_delta_t=1 may dominate the time-mark likelihood"
        )
    return {
        "candidate_time_marks_placeholder": bool(placeholder),
        "candidate_time_marks_suspicious": bool(placeholder or many),
        "candidate_time_marks_warning": warning,
    }

def partition_diagnostic_rows(diagnostics: dict, *, case: str = "", truth_edges: set[tuple[int, int]] | None = None) -> list[dict]:
    truth_edges = truth_edges or set()
    n = int(diagnostics.get("n_partitions", len(diagnostics.get("partition_logL", []))))
    map_idx = diagnostics.get("map_partition_index")
    partitions = diagnostics.get("partitions", [])
    rows = []
    for idx in range(n):
        pobj = partitions[idx] if idx < len(partitions) else {}
        pair_edges = [tuple(map(int, e)) for e in pobj.get("pair_edges", pobj.get("pair_indices", []))]
        n_true = sum((min(i, j), max(i, j)) in truth_edges for i, j in pair_edges)
        rows.append({
            "case": case,
            "partition_index": idx,
            "n_pairs": pobj.get("n_pairs", len(pair_edges)),
            "pair_edges": json.dumps([list(e) for e in pair_edges]),
            "log_likelihood": _list_get(diagnostics.get("partition_logL"), idx),
            "log_prior_weight": _list_get(diagnostics.get("partition_log_prior_weight"), idx),
            "log_posterior_weight": _list_get(diagnostics.get("partition_log_posterior_weight"), idx),
            "posterior_probability": _list_get(diagnostics.get("partition_posterior_probability"), idx),
            "is_map_partition": idx == map_idx,
            "is_truth_partition": (len(pair_edges) == len(truth_edges) and n_true == len(truth_edges)) if truth_edges else None,
            "n_true_edges": n_true if truth_edges else None,
            "n_false_edges": (len(pair_edges) - n_true) if truth_edges else None,
        })
    return rows

def _list_get(values, idx):
    return values[idx] if isinstance(values, list) and idx < len(values) else None


def compute_marginalized_partition_diagnostics(
    partition_states: Iterable[PartitionState],
    candidate_pairs: Iterable[CandidatePair],
    partition_loglike: Callable[[PartitionState], float],
    *,
    log_z_partition_prior: float | None = None,
    raw_candidate_pairs: Iterable[CandidatePair] | None = None,
    edge_mark_prior_contributions: Iterable[float] | None = None,
) -> dict:
    """Compute exact posterior diagnostics over a finite set of partitions."""
    states = tuple(partition_states)
    candidates = tuple(candidate_pairs)
    raw_candidates = tuple(raw_candidate_pairs) if raw_candidate_pairs is not None else ()
    contributions = (
        tuple(float(x) for x in edge_mark_prior_contributions)
        if edge_mark_prior_contributions is not None
        else ()
    )
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
    for edge_idx, (c, p_pair) in enumerate(zip(candidates, pair_probs)):
        raw_c = raw_candidates[edge_idx] if edge_idx < len(raw_candidates) else None
        contribution = (
            contributions[edge_idx]
            if edge_idx < len(contributions)
            else (
                float(c.log_prior_odds) - float(raw_c.log_prior_odds)
                if raw_c is not None
                else 0.0
            )
        )
        raw_log_prior = (
            float(raw_c.log_prior_odds) if raw_c is not None else float(c.log_prior_odds)
        )
        effective_log_prior = float(c.log_prior_odds)
        item = {
            "i": int(c.i),
            "j": int(c.j),
            "p_pair": float(p_pair),
            "log_prior_odds": effective_log_prior,
            "log_prior_odds_raw": raw_log_prior,
            "log_prior_odds_effective": effective_log_prior,
            "edge_mark_prior_contribution": float(contribution),
            "marks": c.marks.to_dict(),
        }
        if c.delta_t_obs is not None:
            item["pair_time_delta_t_obs"] = float(c.delta_t_obs)
        if c.sigma_delta_t is not None:
            item["pair_time_sigma"] = float(c.sigma_delta_t)
        if _looks_like_suspicious_time_mark(c.delta_t_obs, c.sigma_delta_t):
            item["pair_time_placeholder_warning"] = (
                "candidate time marks look synthetic/placeholder-like: small integer second-level time marks with sigma_delta_t=1"
            )
        if c.label is not None:
            item["label"] = c.label
        posterior_pair_probabilities.append(item)

    components = (
        connected_components_from_candidate_pairs(n_events, candidates)
        if n_events
        else ()
    )
    component_n_partitions = []
    component_log_z = []
    component_expected = []
    component_max_p = []
    for comp in components:
        try:
            cstates = exact_component_partitions(comp, candidates)
            component_n_partitions.append(int(len(cstates)))
            component_log_z.append(
                float(logsumexp([s.log_prior_weight for s in cstates]))
            )
        except Exception:
            component_n_partitions.append(None)
            component_log_z.append(None)
        edges = list(comp["candidate_edge_indices"])
        probs = (
            pair_probs[np.asarray(edges, dtype=int)]
            if edges
            else np.asarray([], dtype=float)
        )
        component_expected.append(float(np.sum(probs)))
        component_max_p.append(float(np.max(probs)) if probs.size else 0.0)

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
        "partitions": [
            {**_partition_object(state), "pair_edges": _state_pair_edges(state)}
            for state in states
        ],
        "posterior_pair_probabilities": posterior_pair_probabilities,
        "n_components": int(len(components)),
        "component_event_indices": [list(c["event_indices"]) for c in components],
        "component_candidate_edge_indices": [
            list(c["candidate_edge_indices"]) for c in components
        ],
        "component_n_partitions": component_n_partitions,
        "component_log_z_partition_prior": component_log_z,
        "component_expected_n_pairs": component_expected,
        "component_max_p_pair": component_max_p,
        "map_partition": _partition_object(states[map_idx]),
    }
    out.update(candidate_time_mark_suspicion(candidates))
    if singleton_probs is not None:
        out["posterior_singleton_probability"] = singleton_probs.tolist()
    return out
