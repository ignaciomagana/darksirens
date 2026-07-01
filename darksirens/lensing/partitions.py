"""Candidate-pair partitions for spectral-siren lensing inference.

This module validates candidate-pair JSON files and enumerates all compatible
pair matchings for exact marginalization on small graphs.  Candidate edges carry
``log_prior_odds`` relative to leaving their two endpoints unpaired, so a
matching's unnormalized log prior is the sum of included edge log-odds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class CandidatePair:
    """One unordered candidate pair edge from the JSON schema."""

    i: int
    j: int
    log_prior_odds: float
    label: str | None = None
    delta_t_obs: float | None = None
    sigma_delta_t: float | None = None


@dataclass(frozen=True)
class PartitionState:
    """One compatible matching interpreted as a lensing partition."""

    singleton_indices: np.ndarray
    pair_indices: np.ndarray
    n_singletons: int
    n_pairs: int
    log_prior_weight: float
    candidate_edge_indices: np.ndarray | None = None


def load_candidate_pairs_json(path: str | Path) -> dict:
    """Load a candidate-pair JSON document from ``path``."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_candidate_pairs(data: dict) -> tuple[int, tuple[CandidatePair, ...]]:
    """Validate and normalize the candidate-pair JSON schema.

    Raises
    ------
    ValueError
        If the schema is malformed, contains self-pairs, duplicate unordered
        pairs, non-finite log odds, or indices outside ``[0, n_events)``.
    """

    if not isinstance(data, dict):
        raise ValueError("candidate-pair file must contain a JSON object")
    fmt = data.get("format_version")
    if fmt is not None and fmt != "candidate-pairs-1.0":
        raise ValueError("candidate-pair format_version must be 'candidate-pairs-1.0'")
    if "n_events" not in data:
        raise ValueError("candidate-pair file requires 'n_events'")
    n_events = int(data["n_events"])
    if n_events < 0:
        raise ValueError(f"n_events must be non-negative, got {n_events}")
    if "pairs" in data:
        raw_pairs = data["pairs"]
    elif "candidate_pairs" in data:
        raw_pairs = data["candidate_pairs"]
    else:
        raise ValueError("candidate-pair file requires 'pairs' (or legacy 'candidate_pairs')")
    if not isinstance(raw_pairs, list):
        raise ValueError("candidate_pairs must be a list")

    seen: set[tuple[int, int]] = set()
    pairs: list[CandidatePair] = []
    for k, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"candidate_pairs[{k}] must be an object")
        missing = {"i", "j", "log_prior_odds"} - set(item)
        if missing:
            raise ValueError(f"candidate_pairs[{k}] missing required keys: {sorted(missing)}")
        i = int(item["i"])
        j = int(item["j"])
        if i == j:
            raise ValueError(f"candidate_pairs[{k}] is a self-pair ({i}, {j})")
        if not (0 <= i < n_events and 0 <= j < n_events):
            raise ValueError(
                f"candidate_pairs[{k}] index out of range for n_events={n_events}: ({i}, {j})"
            )
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            raise ValueError(f"duplicate unordered candidate pair ({a}, {b})")
        seen.add((a, b))
        log_prior_odds = float(item["log_prior_odds"])
        if not np.isfinite(log_prior_odds):
            raise ValueError(f"candidate_pairs[{k}].log_prior_odds must be finite")
        delta_t_obs = None
        sigma_delta_t = None
        if "marks" in item:
            marks = item["marks"]
            if not isinstance(marks, dict):
                raise ValueError(f"candidate_pairs[{k}].marks must be an object")
            has_dt = "delta_t_obs" in marks
            has_sigma = "sigma_delta_t" in marks
            if has_dt != has_sigma:
                raise ValueError(
                    f"candidate_pairs[{k}].marks must define both delta_t_obs and sigma_delta_t, or neither"
                )
            if has_dt:
                delta_t_obs = float(marks["delta_t_obs"])
                sigma_delta_t = float(marks["sigma_delta_t"])
                if not np.isfinite(delta_t_obs):
                    raise ValueError(f"candidate_pairs[{k}].marks.delta_t_obs must be finite")
                if not np.isfinite(sigma_delta_t) or sigma_delta_t <= 0:
                    raise ValueError(f"candidate_pairs[{k}].marks.sigma_delta_t must be finite and positive")
            for mark_name, mark_value in marks.items():
                try:
                    arr = np.asarray(mark_value, dtype=float)
                except Exception:
                    continue
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"candidate_pairs[{k}].marks.{mark_name} must be finite")
        label = item.get("label")
        pairs.append(
            CandidatePair(
                a, b, log_prior_odds, None if label is None else str(label),
                delta_t_obs, sigma_delta_t,
            )
        )
    return n_events, tuple(pairs)


def enumerate_compatible_partitions(
    n_events: int,
    candidate_pairs: Iterable[CandidatePair],
    *,
    max_partitions: int = 10_000,
) -> tuple[PartitionState, ...]:
    """Enumerate every matching compatible with ``candidate_pairs``.

    The empty matching is always included.  ``max_partitions`` protects exact
    marginalization from exponential candidate graphs.
    """

    pairs = tuple(candidate_pairs)
    if max_partitions < 1:
        raise ValueError("max_partitions must be at least 1")
    states: list[PartitionState] = []

    def emit(chosen: list[tuple[int, CandidatePair]], used: set[int], logw: float) -> None:
        if len(states) >= max_partitions:
            raise ValueError(
                f"exact partition enumeration exceeded max_partitions={max_partitions}"
            )
        pair_indices = np.asarray([[p.i, p.j] for _, p in chosen], dtype=np.int32).reshape((-1, 2))
        candidate_edge_indices = np.asarray([idx for idx, _ in chosen], dtype=np.int32)
        singleton_indices = np.asarray(
            [idx for idx in range(n_events) if idx not in used], dtype=np.int32
        )
        states.append(
            PartitionState(
                singleton_indices=singleton_indices,
                pair_indices=pair_indices,
                n_singletons=int(singleton_indices.size),
                n_pairs=int(pair_indices.shape[0]),
                log_prior_weight=float(logw),
                candidate_edge_indices=candidate_edge_indices,
            )
        )

    def rec(pos: int, chosen: list[tuple[int, CandidatePair]], used: set[int], logw: float) -> None:
        if pos == len(pairs):
            emit(chosen, used, logw)
            return
        rec(pos + 1, chosen, used, logw)
        p = pairs[pos]
        if p.i not in used and p.j not in used:
            used.update((p.i, p.j))
            chosen.append((pos, p))
            rec(pos + 1, chosen, used, logw + p.log_prior_odds)
            chosen.pop()
            used.remove(p.i)
            used.remove(p.j)

    rec(0, [], set(), 0.0)
    return tuple(states)


def exact_partitions_from_json(data: dict, *, max_partitions: int = 10_000):
    """Validate a candidate-pair JSON object and enumerate exact partitions."""

    n_events, pairs = validate_candidate_pairs(data)
    states = enumerate_compatible_partitions(n_events, pairs, max_partitions=max_partitions)
    log_z_prior = float(logsumexp([s.log_prior_weight for s in states])) if states else -np.inf
    return n_events, states, log_z_prior
