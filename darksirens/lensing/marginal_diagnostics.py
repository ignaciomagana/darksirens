"""Posterior diagnostics for exact partition-marginalized lensing runs."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence
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

def sis_time_mark_support(delta_t_seconds: Iterable[float], T0_seconds: float) -> dict:
    """Where the observed delays sit relative to the SIS support y ∈ (0, 1).

    The time mark converts an observed delay to an impact parameter as
    ``y* = |Δt| / T0``, and the SIS support is ``y ∈ (0, 1)``. A pair with
    ``y* >= 1`` has EVERY collapse node masked, so its pair log-likelihood is
    exactly -inf: in fixed-partition mode the master log-likelihood is -inf
    everywhere (the sampler cannot initialise), and in ``marginalize_exact``
    the pairing silently receives zero posterior weight at every parameter
    value, i.e. the analysis reports "no lensing" by construction.

    Returns
    -------
    dict with ``n_marked``, ``n_out_of_support``, ``all_out_of_support``,
    ``max_y_star``, ``min_y_star`` and ``T0_seconds``.
    """
    dt = np.asarray([abs(float(x)) for x in delta_t_seconds if x is not None], dtype=float)
    dt = dt[np.isfinite(dt)]
    T0 = float(T0_seconds)
    if dt.size == 0 or not np.isfinite(T0) or T0 <= 0.0:
        return {
            "n_marked": int(dt.size),
            "n_out_of_support": 0,
            "all_out_of_support": False,
            "max_y_star": None,
            "min_y_star": None,
            "T0_seconds": T0,
        }
    y_star = dt / T0
    n_out = int(np.sum(y_star >= 1.0))
    return {
        "n_marked": int(dt.size),
        "n_out_of_support": n_out,
        "all_out_of_support": bool(n_out == dt.size),
        "max_y_star": float(np.max(y_star)),
        "min_y_star": float(np.min(y_star)),
        "T0_seconds": T0,
    }


def sis_time_mark_support_message(support: dict) -> str:
    """Human-readable diagnosis of an out-of-support time-mark set."""
    return (
        "time-marked candidate pairs fall outside the SIS support y in (0, 1): "
        f"{support['n_out_of_support']}/{support['n_marked']} pairs have "
        f"y* = |dt|/T0 >= 1 (max y* = {support['max_y_star']:.4g}, "
        f"min y* = {support['min_y_star']:.4g}, T0 = {support['T0_seconds']:.4g} s "
        f"= {support['T0_seconds'] / 86400.0:.4g} d). Those pairs get an exactly "
        "-inf time-marked pair likelihood, so the pairing is excluded by "
        "construction. Raise --sl_T0_sec to the time-delay scale of the lens "
        "population you are modelling (T0 = 2(1+z_L) theta_E^2 D_L D_S/(c D_LS); "
        "5.36e6 s at z_L=0.5, z_s=1, sigma_v=200 km/s), or drop --pair_marks time."
    )


def partition_diagnostic_rows(diagnostics: dict, *, case: str = "", truth_edges: set[tuple[int, int]] | None = None) -> list[dict]:
    truth_edges = truth_edges or set()
    if diagnostics.get("global_partitions_enumerated") is False:
        return []
    partitions = diagnostics.get("partitions", [])
    if not partitions and not diagnostics.get("partition_logL"):
        return []
    n = int(diagnostics.get("n_partitions", len(diagnostics.get("partition_logL", []))))
    map_idx = diagnostics.get("map_partition_index")
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



def _combine_component_states(states: Sequence[PartitionState]) -> PartitionState:
    singletons = (
        np.concatenate([np.asarray(s.singleton_indices, dtype=np.int32) for s in states])
        if states
        else np.asarray([], dtype=np.int32)
    )
    pair_arrays = [
        np.asarray(s.pair_indices, dtype=np.int32).reshape((-1, 2))
        for s in states
        if s.n_pairs
    ]
    pair_indices = (
        np.concatenate(pair_arrays).astype(np.int32)
        if pair_arrays
        else np.asarray([], dtype=np.int32).reshape((0, 2))
    )
    edge_arrays = [
        np.asarray(s.candidate_edge_indices, dtype=np.int32)
        for s in states
        if s.candidate_edge_indices is not None and len(s.candidate_edge_indices)
    ]
    edge_indices = (
        np.concatenate(edge_arrays).astype(np.int32)
        if edge_arrays
        else np.asarray([], dtype=np.int32)
    )
    if edge_indices.size:
        order = np.argsort(edge_indices)
        edge_indices = edge_indices[order]
        pair_indices = pair_indices[order]
    return PartitionState(
        np.sort(singletons).astype(np.int32),
        pair_indices,
        int(singletons.size),
        int(pair_indices.shape[0]),
        float(sum(s.log_prior_weight for s in states)),
        edge_indices,
    )


def _component_event_indices(states: Sequence[PartitionState]) -> list[int]:
    events: set[int] = set()
    for state in states:
        events.update(int(x) for x in np.asarray(state.singleton_indices, dtype=int))
        for i, j in np.asarray(state.pair_indices, dtype=int).reshape((-1, 2)):
            events.add(int(i))
            events.add(int(j))
    return sorted(events)


def _prefix_suffix_count_dp(
    component_counts: Sequence[np.ndarray],
    component_log_weights: Sequence[np.ndarray],
    max_pairs: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    prefix = [np.full(max_pairs + 1, -np.inf, dtype=float)]
    prefix[0][0] = 0.0
    for counts, weights in zip(component_counts, component_log_weights):
        new = np.full(max_pairs + 1, -np.inf, dtype=float)
        for n_pairs, logw in zip(counts, weights):
            n_pairs = int(n_pairs)
            if n_pairs > max_pairs:
                continue
            valid = max_pairs + 1 - n_pairs
            new[n_pairs:] = np.logaddexp(new[n_pairs:], prefix[-1][:valid] + float(logw))
        prefix.append(new)

    suffix = [np.full(max_pairs + 1, -np.inf, dtype=float) for _ in range(len(component_counts) + 1)]
    suffix[-1][0] = 0.0
    for idx in range(len(component_counts) - 1, -1, -1):
        new = np.full(max_pairs + 1, -np.inf, dtype=float)
        for n_pairs, logw in zip(component_counts[idx], component_log_weights[idx]):
            n_pairs = int(n_pairs)
            if n_pairs > max_pairs:
                continue
            valid = max_pairs + 1 - n_pairs
            new[n_pairs:] = np.logaddexp(new[n_pairs:], suffix[idx + 1][:valid] + float(logw))
        suffix[idx] = new
    return prefix, suffix


def _component_state_log_marginal(
    component_index: int,
    state_index: int,
    component_counts: Sequence[np.ndarray],
    component_log_weights: Sequence[np.ndarray],
    count_loglike_delta: np.ndarray,
    prefix: Sequence[np.ndarray],
    suffix: Sequence[np.ndarray],
    log_total_inner: float,
) -> float:
    n_pairs = int(component_counts[component_index][state_index])
    local_logw = float(component_log_weights[component_index][state_index])
    terms = []
    left = prefix[component_index]
    right = suffix[component_index + 1]
    for left_count, left_logw in enumerate(left):
        if not np.isfinite(left_logw):
            continue
        max_right = len(count_loglike_delta) - left_count - n_pairs
        if max_right <= 0:
            continue
        vals = right[:max_right]
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        right_counts = np.nonzero(finite)[0]
        total_counts = left_count + n_pairs + right_counts
        terms.extend((left_logw + local_logw + vals[finite] + count_loglike_delta[total_counts]).tolist())
    return float(logsumexp(terms) - log_total_inner) if terms else -np.inf


def _componentwise_map_state_indices(
    component_counts: Sequence[np.ndarray],
    component_log_weights: Sequence[np.ndarray],
    count_loglike_delta: np.ndarray,
) -> list[int]:
    max_pairs = len(count_loglike_delta) - 1
    dp = np.full(max_pairs + 1, -np.inf, dtype=float)
    dp[0] = 0.0
    back_counts: list[np.ndarray] = []
    back_states: list[np.ndarray] = []
    for counts, weights in zip(component_counts, component_log_weights):
        new = np.full(max_pairs + 1, -np.inf, dtype=float)
        prev_count = np.full(max_pairs + 1, -1, dtype=int)
        prev_state = np.full(max_pairs + 1, -1, dtype=int)
        for old_count, old_logw in enumerate(dp):
            if not np.isfinite(old_logw):
                continue
            for state_idx, (n_pairs, logw) in enumerate(zip(counts, weights)):
                total = old_count + int(n_pairs)
                if total > max_pairs:
                    continue
                val = old_logw + float(logw)
                if val > new[total]:
                    new[total] = val
                    prev_count[total] = old_count
                    prev_state[total] = state_idx
        dp = new
        back_counts.append(prev_count)
        back_states.append(prev_state)
    final = dp + count_loglike_delta
    if not np.any(np.isfinite(final)):
        raise ValueError("factorized partition posterior has no finite states")
    count = int(np.nanargmax(final))
    chosen = [0] * len(component_counts)
    for idx in range(len(component_counts) - 1, -1, -1):
        state_idx = int(back_states[idx][count])
        if state_idx < 0:
            raise ValueError("factorized partition MAP backtrack failed")
        chosen[idx] = state_idx
        count = int(back_counts[idx][count])
    return chosen


def component_partition_diagnostic_rows(
    diagnostics: dict,
    *,
    case: str = "",
    truth_edges: set[tuple[int, int]] | None = None,
) -> list[dict]:
    truth_edges = truth_edges or set()
    rows = []
    for comp in diagnostics.get("component_partition_diagnostics", []):
        pair_edges = [tuple(map(int, e)) for e in comp.get("pair_edges", [])]
        n_true = sum((min(i, j), max(i, j)) in truth_edges for i, j in pair_edges)
        rows.append({
            "case": case,
            "component_index": comp.get("component_index"),
            "local_partition_index": comp.get("local_partition_index"),
            "n_pairs": comp.get("n_pairs"),
            "pair_edges": json.dumps([list(e) for e in pair_edges]),
            "log_likelihood_delta_or_local": comp.get("log_likelihood_delta_or_local"),
            "log_prior_weight": comp.get("log_prior_weight"),
            "log_posterior_weight_local": comp.get("log_posterior_weight_local"),
            "posterior_probability_local": comp.get("posterior_probability_local"),
            "is_component_map_partition": comp.get("is_component_map_partition"),
            "n_true_edges": n_true if truth_edges else None,
            "n_false_edges": (len(pair_edges) - n_true) if truth_edges else None,
        })
    return rows


def compute_componentwise_factorized_partition_diagnostics(
    component_partition_states: Sequence[Sequence[PartitionState]],
    candidate_pairs: Iterable[CandidatePair],
    component_loglike_delta: Sequence[Sequence[float]],
    count_loglike_delta: Sequence[float],
    baseline_loglike: float,
    *,
    log_z_partition_prior: float | None = None,
    component_summaries: Sequence[dict] | None = None,
    raw_candidate_pairs: Iterable[CandidatePair] | None = None,
    edge_mark_prior_contributions: Iterable[float] | None = None,
) -> dict:
    """Compute exact factorized diagnostics without global partition rows.

    Component-local likelihood deltas must exclude any global count-only term.
    ``count_loglike_delta[k]`` supplies that global term relative to the
    all-singleton baseline for total pair count ``k``.
    """
    component_states = tuple(tuple(states) for states in component_partition_states)
    if not component_states:
        raise ValueError("at least one component is required")
    candidates = tuple(candidate_pairs)
    raw_candidates = tuple(raw_candidate_pairs) if raw_candidate_pairs is not None else ()
    contributions = (
        tuple(float(x) for x in edge_mark_prior_contributions)
        if edge_mark_prior_contributions is not None
        else ()
    )
    count_delta = np.asarray(tuple(float(x) for x in count_loglike_delta), dtype=float)
    if count_delta.ndim != 1 or count_delta.size == 0:
        raise ValueError("count_loglike_delta must be a non-empty vector")
    max_pairs = int(count_delta.size - 1)

    component_counts = [
        np.asarray([int(state.n_pairs) for state in states], dtype=int)
        for states in component_states
    ]
    like_delta = [np.asarray(tuple(float(x) for x in vals), dtype=float) for vals in component_loglike_delta]
    if len(like_delta) != len(component_states):
        raise ValueError("component_loglike_delta length does not match component states")
    component_log_weights = []
    for states, deltas in zip(component_states, like_delta):
        if len(states) != len(deltas):
            raise ValueError("component loglike-delta length does not match states")
        priors = np.asarray([float(state.log_prior_weight) for state in states], dtype=float)
        component_log_weights.append(priors + deltas)

    approximate_total = 1
    for states in component_states:
        approximate_total *= len(states)
    if log_z_partition_prior is None:
        log_z_partition_prior = float(
            sum(logsumexp([state.log_prior_weight for state in states]) for states in component_states)
        )

    prefix, suffix = _prefix_suffix_count_dp(component_counts, component_log_weights, max_pairs)
    final_terms = prefix[-1] + count_delta
    log_total_inner = float(logsumexp(final_terms))
    log_norm = float(baseline_loglike + log_total_inner)

    state_log_probs: list[list[float]] = []
    state_probs: list[list[float]] = []
    for comp_idx, states in enumerate(component_states):
        logs = []
        probs = []
        for state_idx, _state in enumerate(states):
            lp = _component_state_log_marginal(
                comp_idx,
                state_idx,
                component_counts,
                component_log_weights,
                count_delta,
                prefix,
                suffix,
                log_total_inner,
            )
            logs.append(lp)
            probs.append(float(np.exp(lp)) if np.isfinite(lp) else 0.0)
        state_log_probs.append(logs)
        state_probs.append(probs)

    pair_probs = np.zeros(len(candidates), dtype=float)
    n_events = 0
    for c in candidates:
        n_events = max(n_events, c.i + 1, c.j + 1)
    for states in component_states:
        for state in states:
            if state.singleton_indices.size:
                n_events = max(n_events, int(np.max(state.singleton_indices)) + 1)
            if state.pair_indices.size:
                n_events = max(n_events, int(np.max(state.pair_indices)) + 1)
    singleton_probs = np.zeros(n_events, dtype=float) if n_events else None

    expected_n_pairs = 0.0
    expected_n_singletons = 0.0
    component_expected = []
    component_max_p = []
    component_rows = []
    component_maps = []
    for comp_idx, states in enumerate(component_states):
        probs = np.asarray(state_probs[comp_idx], dtype=float)
        counts = component_counts[comp_idx].astype(float)
        component_expected.append(float(np.sum(probs * counts)))
        expected_n_pairs += component_expected[-1]
        expected_n_singletons += float(
            np.sum(probs * np.asarray([state.n_singletons for state in states], dtype=float))
        )
        local_map_idx = int(np.argmax(probs)) if len(probs) else 0
        component_maps.append(local_map_idx)
        for state_idx, (state, prob, log_prob) in enumerate(zip(states, probs, state_log_probs[comp_idx])):
            edge_idx = np.asarray(state.candidate_edge_indices, dtype=int) if state.candidate_edge_indices is not None else np.asarray([], dtype=int)
            if edge_idx.size:
                pair_probs[edge_idx] += float(prob)
            if singleton_probs is not None:
                singleton_probs[np.asarray(state.singleton_indices, dtype=int)] += float(prob)
            component_rows.append({
                "component_index": comp_idx,
                "local_partition_index": state_idx,
                "n_pairs": int(state.n_pairs),
                "pair_edges": _state_pair_edges(state),
                "log_likelihood_delta_or_local": float(like_delta[comp_idx][state_idx]),
                "log_prior_weight": float(state.log_prior_weight),
                "log_posterior_weight_local": float(log_prob),
                "posterior_probability_local": float(prob),
                "is_component_map_partition": state_idx == local_map_idx,
            })
        edges = []
        if component_summaries is not None and comp_idx < len(component_summaries):
            edges = list(component_summaries[comp_idx].get("candidate_edge_indices", []))
        else:
            edges = sorted({int(e) for state in states for e in np.asarray(state.candidate_edge_indices if state.candidate_edge_indices is not None else [], dtype=int)})
        edge_probs = pair_probs[np.asarray(edges, dtype=int)] if edges else np.asarray([], dtype=float)
        component_max_p.append(float(np.max(edge_probs)) if edge_probs.size else 0.0)

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
        raw_log_prior = float(raw_c.log_prior_odds) if raw_c is not None else float(c.log_prior_odds)
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

    map_indices = _componentwise_map_state_indices(component_counts, component_log_weights, count_delta)
    map_state = _combine_component_states(
        [component_states[comp_idx][state_idx] for comp_idx, state_idx in enumerate(map_indices)]
    )

    summaries = tuple(component_summaries or ())
    component_diagnostics = []
    for comp_idx, states in enumerate(component_states):
        summary = summaries[comp_idx] if comp_idx < len(summaries) else {}
        component_diagnostics.append({
            "component_index": comp_idx,
            "event_indices": list(summary.get("event_indices", _component_event_indices(states))),
            "candidate_edge_indices": list(summary.get("candidate_edge_indices", [])),
            "n_partitions": int(len(states)),
            "logZ_component_delta": float(logsumexp(component_log_weights[comp_idx])),
            "expected_n_pairs_component": float(component_expected[comp_idx]),
            "max_p_pair_component": float(component_max_p[comp_idx]),
            "map_local_partition": _partition_object(states[component_maps[comp_idx]]),
        })

    out = {
        "partition_mode": "marginalize_exact",
        "partition_component_mode": "componentwise",
        "factorized_exact": True,
        "global_partitions_enumerated": False,
        "n_partitions": int(approximate_total),
        "approximate_total_partitions": int(approximate_total),
        "log_z_partition_prior": float(log_z_partition_prior),
        "baseline_logL_all_singletons": float(baseline_loglike),
        "count_loglike_delta": count_delta.tolist(),
        "logL_marginalized": float(log_norm - log_z_partition_prior),
        "logL_total": float(log_norm - log_z_partition_prior),
        "map_partition_index": -1,
        "map_component_partition_indices": [int(x) for x in map_indices],
        "expected_n_pairs": float(expected_n_pairs),
        "expected_n_singletons": float(expected_n_singletons),
        "map_partition": _partition_object(map_state),
        "map_n_pairs": int(map_state.n_pairs),
        "posterior_pair_probabilities": posterior_pair_probabilities,
        "n_components": int(len(component_states)),
        "component_event_indices": [d["event_indices"] for d in component_diagnostics],
        "component_candidate_edge_indices": [d["candidate_edge_indices"] for d in component_diagnostics],
        "component_n_partitions": [int(len(states)) for states in component_states],
        "largest_component_n_partitions": int(max(len(states) for states in component_states)),
        "component_expected_n_pairs": component_expected,
        "component_max_p_pair": component_max_p,
        "component_diagnostics": component_diagnostics,
        "component_partition_diagnostics": component_rows,
    }
    out.update(candidate_time_mark_suspicion(candidates))
    if singleton_probs is not None:
        out["posterior_singleton_probability"] = singleton_probs.tolist()
    return out

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
    # The candidate endpoints alone do NOT bound the event count: a partition
    # state's singleton_indices span every observed event (0..N_obs-1), so any
    # event above the largest candidate endpoint would fancy-index past the end
    # of singleton_probs (IndexError).  Widen with the states' own indices, the
    # way compute_componentwise_factorized_partition_diagnostics already does --
    # real campaign candidate files have n_events > max endpoint + 1 (events
    # with no candidate partner at all).
    n_events = 0
    for c in candidates:
        n_events = max(n_events, c.i + 1, c.j + 1)
    for state in states:
        if state.singleton_indices.size:
            n_events = max(n_events, int(np.max(state.singleton_indices)) + 1)
        if state.pair_indices.size:
            n_events = max(n_events, int(np.max(state.pair_indices)) + 1)
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
