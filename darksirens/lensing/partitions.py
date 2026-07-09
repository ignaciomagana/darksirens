"""Candidate-pair partitions for spectral-siren lensing inference.

This module validates candidate-pair JSON files and enumerates all compatible
pair matchings for exact marginalization on small graphs.  Candidate edges carry
``log_prior_odds`` relative to leaving their two endpoints unpaired, so a
matching's unnormalized log prior is the sum of included edge log-odds.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

KNOWN_EDGE_MARKS = {
    "delta_t_obs",
    "sigma_delta_t",
    "log_sky_overlap",
    "log_mass_distance_score",
    "log_spin_score",
}


@dataclass(frozen=True)
class EdgeMarks:
    """Structured candidate-edge marks used by simulations and diagnostics."""

    delta_t_obs: float | None = None
    sigma_delta_t: float | None = None
    log_sky_overlap: float | None = None
    log_mass_distance_score: float | None = None
    log_spin_score: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in (
            "delta_t_obs",
            "sigma_delta_t",
            "log_sky_overlap",
            "log_mass_distance_score",
            "log_spin_score",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = float(value)
        out.update({str(k): float(v) for k, v in sorted(self.extra.items())})
        return out

    def get(self, key: str) -> float | None:
        if key in KNOWN_EDGE_MARKS:
            return getattr(self, key)
        return self.extra.get(key)

    @property
    def keys(self) -> set[str]:
        return set(self.to_dict())


@dataclass(frozen=True, init=False)
class CandidatePair:
    """One unordered candidate pair edge from the JSON schema.

    The constructor accepts both the new ``marks=EdgeMarks(...)`` form and the
    legacy positional ``delta_t_obs, sigma_delta_t`` fields.
    """

    i: int
    j: int
    log_prior_odds: float
    label: str | None = None
    marks: EdgeMarks = field(default_factory=EdgeMarks)

    def __init__(
        self,
        i: int,
        j: int,
        log_prior_odds: float,
        label: str | None = None,
        marks: EdgeMarks | None = None,
        sigma_delta_t: float | None = None,
        *,
        delta_t_obs: float | None = None,
    ):
        if marks is not None and not isinstance(marks, EdgeMarks):
            # Legacy positional form: CandidatePair(i, j, logp, label, dt, sigma).
            if delta_t_obs is not None:
                raise TypeError("delta_t_obs specified twice")
            delta_t_obs = float(marks)
            marks = None
        if marks is None:
            marks = EdgeMarks(delta_t_obs=delta_t_obs, sigma_delta_t=sigma_delta_t)
        elif sigma_delta_t is not None or delta_t_obs is not None:
            raise TypeError(
                "pass either marks=EdgeMarks(...) or legacy delta_t_obs/sigma_delta_t, not both"
            )
        object.__setattr__(self, "i", int(i))
        object.__setattr__(self, "j", int(j))
        object.__setattr__(self, "log_prior_odds", float(log_prior_odds))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "marks", marks)

    @property
    def delta_t_obs(self) -> float | None:
        """Backward-compatible access to ``marks.delta_t_obs``."""
        return self.marks.delta_t_obs

    @property
    def sigma_delta_t(self) -> float | None:
        """Backward-compatible access to ``marks.sigma_delta_t``."""
        return self.marks.sigma_delta_t


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


def _finite_float(value, *, context: str, positive: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out) or (positive and out <= 0):
        suffix = "finite and positive" if positive else "finite"
        raise ValueError(f"{context} must be {suffix}")
    return out


def validate_edge_marks(raw_marks: Mapping | None, *, context: str) -> EdgeMarks:
    """Validate and normalize one candidate-edge ``marks`` object."""
    if raw_marks is None:
        return EdgeMarks()
    if not isinstance(raw_marks, Mapping):
        raise ValueError(f"{context}.marks must be an object")

    has_dt = "delta_t_obs" in raw_marks
    has_sigma = "sigma_delta_t" in raw_marks
    if has_dt != has_sigma:
        raise ValueError(
            f"{context}.marks must define both delta_t_obs and sigma_delta_t, or neither"
        )

    values: dict[str, float | None] = {k: None for k in KNOWN_EDGE_MARKS}
    extra: dict[str, float] = {}
    for key, value in raw_marks.items():
        if key == "delta_t_obs":
            values[key] = _finite_float(value, context=f"{context}.marks.{key}")
        elif key == "sigma_delta_t":
            values[key] = _finite_float(
                value, context=f"{context}.marks.{key}", positive=True
            )
        elif key in ("log_sky_overlap", "log_mass_distance_score", "log_spin_score"):
            values[key] = _finite_float(value, context=f"{context}.marks.{key}")
        elif isinstance(key, str) and key.startswith("log_"):
            extra[key] = _finite_float(value, context=f"{context}.marks.{key}")
        else:
            raise ValueError(
                f"{context}.marks has unsupported key {key!r}; custom marks must start with 'log_'"
            )
    return EdgeMarks(
        delta_t_obs=values["delta_t_obs"],
        sigma_delta_t=values["sigma_delta_t"],
        log_sky_overlap=values["log_sky_overlap"],
        log_mass_distance_score=values["log_mass_distance_score"],
        log_spin_score=values["log_spin_score"],
        extra=extra,
    )


def parse_edge_mark_keys(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated edge-mark key list."""
    if value is None:
        return ()
    if isinstance(value, str):
        pieces = [p.strip() for p in value.split(",")]
    else:
        pieces = [str(p).strip() for p in value]
    return tuple(p for p in pieces if p)


def edge_mark_prior_contributions(
    candidate_pairs: Iterable[CandidatePair], keys: Sequence[str]
) -> tuple[float, ...]:
    """Return selected log-mark prior contributions for each raw candidate pair."""
    keys = parse_edge_mark_keys(keys)
    out: list[float] = []
    for pair in candidate_pairs:
        total = 0.0
        for key in keys:
            value = pair.marks.get(key)
            if value is None:
                raise ValueError(
                    f"candidate pair ({pair.i},{pair.j}) missing requested edge prior mark {key!r}"
                )
            if not key.startswith("log_"):
                raise ValueError(f"edge prior mark {key!r} is not a log_* mark")
            total += float(value)
        out.append(total)
    return tuple(out)


def apply_edge_mark_prior_keys(
    candidate_pairs: Iterable[CandidatePair], keys: Sequence[str]
) -> tuple[CandidatePair, ...]:
    """Return pairs with selected log-mark values added to effective log prior odds."""
    pairs = tuple(candidate_pairs)
    contributions = edge_mark_prior_contributions(pairs, keys)
    return tuple(
        CandidatePair(
            pair.i,
            pair.j,
            float(pair.log_prior_odds) + float(contribution),
            pair.label,
            pair.marks,
        )
        for pair, contribution in zip(pairs, contributions)
    )


def parse_folded_mark_keys(data: dict) -> tuple[str, ...]:
    """Return mark keys the builder already folded into log_prior_odds.

    Producers that bake mark values into the edge scores (e.g.
    build_candidate_pairs_from_observed folds log_mass_distance_score, and
    log_sky_overlap when sky_overlap_weight != 0) record them under the
    optional top-level 'folded_mark_keys' field. Absent field = unknown
    provenance, no guard (backward compatible with candidate-pairs-1.0
    files written before this field existed).
    """
    folded = data.get("folded_mark_keys")
    if folded is None:
        return ()
    if isinstance(folded, str) or not isinstance(folded, Sequence):
        raise ValueError("folded_mark_keys must be a list of mark-key strings")
    out = []
    for k in folded:
        if not isinstance(k, str) or not k:
            raise ValueError("folded_mark_keys must be a list of mark-key strings")
        out.append(k)
    return tuple(out)


def prepare_candidate_pairs_for_partitioning(
    data: dict, edge_mark_prior_keys: Sequence[str] | None = None
) -> tuple[int, tuple[CandidatePair, ...], tuple[CandidatePair, ...], tuple[float, ...]]:
    """Validate candidate-pair JSON and apply requested edge priors exactly once."""
    n_events, raw_pairs = validate_candidate_pairs(data)
    prior_keys = parse_edge_mark_keys(edge_mark_prior_keys)
    folded_keys = set(parse_folded_mark_keys(data))
    double_counted = [k for k in prior_keys if k in folded_keys]
    if double_counted:
        raise ValueError(
            f"edge mark prior keys {double_counted} are already folded into "
            f"log_prior_odds by the candidate-pair builder "
            f"(folded_mark_keys={sorted(folded_keys)}); applying them again via "
            "--edge_mark_prior_keys would double-count the pairing prior"
        )
    contributions = edge_mark_prior_contributions(raw_pairs, prior_keys)
    effective_pairs = tuple(
        CandidatePair(
            pair.i,
            pair.j,
            float(pair.log_prior_odds) + float(contribution),
            pair.label,
            pair.marks,
        )
        for pair, contribution in zip(raw_pairs, contributions)
    )
    return n_events, raw_pairs, effective_pairs, contributions


def validate_candidate_pairs(data: dict) -> tuple[int, tuple[CandidatePair, ...]]:
    """Validate and normalize the candidate-pair JSON schema."""
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
    raw_pairs = data.get("pairs", data.get("candidate_pairs"))
    if raw_pairs is None:
        raise ValueError(
            "candidate-pair file requires 'pairs' (or legacy 'candidate_pairs')"
        )
    if not isinstance(raw_pairs, list):
        raise ValueError("candidate_pairs must be a list")

    seen: set[tuple[int, int]] = set()
    pairs: list[CandidatePair] = []
    for k, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"candidate_pairs[{k}] must be an object")
        missing = {"i", "j", "log_prior_odds"} - set(item)
        if missing:
            raise ValueError(
                f"candidate_pairs[{k}] missing required keys: {sorted(missing)}"
            )
        i, j = int(item["i"]), int(item["j"])
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
        log_prior_odds = _finite_float(
            item["log_prior_odds"], context=f"candidate_pairs[{k}].log_prior_odds"
        )
        marks = validate_edge_marks(item.get("marks"), context=f"candidate_pairs[{k}]")
        label = item.get("label")
        pairs.append(
            CandidatePair(
                a, b, log_prior_odds, None if label is None else str(label), marks
            )
        )
    return n_events, tuple(pairs)


def connected_components_from_candidate_pairs(
    n_events: int, candidate_pairs: Iterable[CandidatePair]
) -> tuple[dict, ...]:
    """Return connected components of the candidate-pair graph.

    Isolated events are included as singleton components with no candidate edges.
    Edge indices refer to the input candidate-pair order.
    """
    pairs = tuple(candidate_pairs)
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    parent = list(range(n_events))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        if not (0 <= p.i < n_events and 0 <= p.j < n_events):
            raise ValueError(
                f"candidate pair ({p.i},{p.j}) index out of range for n_events={n_events}"
            )
        union(p.i, p.j)

    comps: dict[int, dict[str, list[int]]] = {}
    for event in range(n_events):
        comps.setdefault(
            find(event), {"event_indices": [], "candidate_edge_indices": []}
        )["event_indices"].append(event)
    for edge_idx, p in enumerate(pairs):
        comps[find(p.i)]["candidate_edge_indices"].append(edge_idx)

    return tuple(
        {
            "event_indices": tuple(v["event_indices"]),
            "candidate_edge_indices": tuple(v["candidate_edge_indices"]),
        }
        for _, v in sorted(
            comps.items(), key=lambda item: min(item[1]["event_indices"])
        )
    )


def exact_component_partitions(
    component: Mapping[str, Sequence[int]],
    candidate_pairs: Sequence[CandidatePair],
    *,
    max_partitions: int = 10_000,
) -> tuple[PartitionState, ...]:
    """Enumerate exact matching partitions for one connected component."""
    events = tuple(int(i) for i in component["event_indices"])
    edge_indices = tuple(int(i) for i in component["candidate_edge_indices"])
    local_pairs = tuple(candidate_pairs[i] for i in edge_indices)
    local_states = enumerate_compatible_partitions(
        max(events) + 1 if events else 0, local_pairs, max_partitions=max_partitions
    )
    out = []
    event_set = set(events)
    for state in local_states:
        singletons = np.asarray(
            [
                i
                for i in np.asarray(state.singleton_indices, dtype=int)
                if i in event_set
            ],
            dtype=np.int32,
        )
        global_edges = np.asarray(
            [
                edge_indices[int(i)]
                for i in np.asarray(state.candidate_edge_indices, dtype=int)
            ],
            dtype=np.int32,
        )
        out.append(
            PartitionState(
                singletons,
                np.asarray(state.pair_indices, dtype=np.int32).reshape((-1, 2)),
                int(singletons.size),
                int(state.n_pairs),
                float(state.log_prior_weight),
                global_edges,
            )
        )
    return tuple(out)


def exact_partitions_componentwise(
    n_events: int,
    candidate_pairs: Iterable[CandidatePair],
    *,
    max_component_events: int | None = None,
    max_component_edges: int | None = None,
    max_component_partitions: int = 10_000,
    max_total_partitions: int | None = 10_000,
) -> tuple[tuple[PartitionState, ...], tuple[dict, ...], tuple[tuple[PartitionState, ...], ...]]:
    """Enumerate matchings per component and combine by Cartesian product."""
    summaries, component_states, total = exact_partition_components(
        n_events,
        candidate_pairs,
        max_component_events=max_component_events,
        max_component_edges=max_component_edges,
        max_component_partitions=max_component_partitions,
    )
    if max_total_partitions is not None and total > max_total_partitions:
        raise ValueError(
            f"component-wise exact enumeration would produce {total} total partitions, exceeding max_total_partitions={max_total_partitions}; prune the candidate graph before inference"
        )
    combined = combine_component_partitions(component_states)
    return combined, summaries, component_states


def exact_partition_components(
    n_events: int,
    candidate_pairs: Iterable[CandidatePair],
    *,
    max_component_events: int | None = None,
    max_component_edges: int | None = None,
    max_component_partitions: int = 10_000,
) -> tuple[tuple[dict, ...], tuple[tuple[PartitionState, ...], ...], int]:
    """Enumerate exact matching partitions independently inside each component.

    This deliberately does not enumerate the Cartesian product across
    components.  The returned ``total`` is the size of that product if a caller
    chooses to materialize it for small compatibility tests or diagnostics.
    """
    pairs = tuple(candidate_pairs)
    components = connected_components_from_candidate_pairs(n_events, pairs)
    component_states = []
    summaries = []
    total = 1
    for idx, comp in enumerate(components):
        n_comp_events = len(comp["event_indices"])
        n_comp_edges = len(comp["candidate_edge_indices"])
        if max_component_events is not None and n_comp_events > max_component_events:
            raise ValueError(
                f"candidate component {idx} has {n_comp_events} events, exceeding max_component_events={max_component_events}; prune the candidate graph before inference"
            )
        if max_component_edges is not None and n_comp_edges > max_component_edges:
            raise ValueError(
                f"candidate component {idx} has {n_comp_edges} edges, exceeding max_component_edges={max_component_edges}; prune the candidate graph before inference"
            )
        states = exact_component_partitions(
            comp, pairs, max_partitions=max_component_partitions
        )
        total *= len(states)
        component_states.append(states)
        summaries.append({**comp, "n_partitions": len(states)})
    return tuple(summaries), tuple(component_states), int(total)


def combine_component_partitions(
    component_states: Sequence[Sequence[PartitionState]],
) -> tuple[PartitionState, ...]:
    """Materialize the global Cartesian product of component-local states."""
    combined = []
    for product in itertools.product(*component_states):
        singletons = (
            np.concatenate([s.singleton_indices for s in product]).astype(np.int32)
            if product
            else np.asarray([], dtype=np.int32)
        )
        pair_arrays = [
            np.asarray(s.pair_indices, dtype=np.int32).reshape((-1, 2))
            for s in product
            if s.n_pairs
        ]
        pair_indices = (
            np.concatenate(pair_arrays).astype(np.int32)
            if pair_arrays
            else np.asarray([], dtype=np.int32).reshape((0, 2))
        )
        edge_arrays = [
            np.asarray(s.candidate_edge_indices, dtype=np.int32)
            for s in product
            if s.candidate_edge_indices is not None and len(s.candidate_edge_indices)
        ]
        edge_indices = (
            np.concatenate(edge_arrays).astype(np.int32)
            if edge_arrays
            else np.asarray([], dtype=np.int32)
        )
        order = (
            np.argsort(edge_indices) if edge_indices.size else np.asarray([], dtype=int)
        )
        if edge_indices.size:
            edge_indices = edge_indices[order]
            pair_indices = pair_indices[order]
        combined.append(
            PartitionState(
                np.sort(singletons).astype(np.int32),
                pair_indices,
                int(singletons.size),
                int(pair_indices.shape[0]),
                float(sum(s.log_prior_weight for s in product)),
                edge_indices,
            )
        )
    return tuple(combined)

def enumerate_compatible_partitions(
    n_events: int,
    candidate_pairs: Iterable[CandidatePair],
    *,
    max_partitions: int = 10_000,
) -> tuple[PartitionState, ...]:
    """Enumerate every matching compatible with ``candidate_pairs``."""
    pairs = tuple(candidate_pairs)
    if max_partitions < 1:
        raise ValueError("max_partitions must be at least 1")
    states: list[PartitionState] = []

    def emit(
        chosen: list[tuple[int, CandidatePair]], used: set[int], logw: float
    ) -> None:
        if len(states) >= max_partitions:
            raise ValueError(
                f"exact partition enumeration exceeded max_partitions={max_partitions}"
            )
        pair_indices = np.asarray(
            [[p.i, p.j] for _, p in chosen], dtype=np.int32
        ).reshape((-1, 2))
        candidate_edge_indices = np.asarray([idx for idx, _ in chosen], dtype=np.int32)
        singleton_indices = np.asarray(
            [idx for idx in range(n_events) if idx not in used], dtype=np.int32
        )
        states.append(
            PartitionState(
                singleton_indices,
                pair_indices,
                int(singleton_indices.size),
                int(pair_indices.shape[0]),
                float(logw),
                candidate_edge_indices,
            )
        )

    def rec(
        pos: int, chosen: list[tuple[int, CandidatePair]], used: set[int], logw: float
    ) -> None:
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


def exact_partitions_from_pairs(
    n_events: int,
    candidate_pairs: Iterable[CandidatePair],
    *,
    max_partitions: int = 10_000,
    component_mode: str = "global",
    max_component_events: int | None = None,
    max_component_edges: int | None = None,
    max_component_partitions: int | None = None,
    max_total_partitions: int | None = None,
):
    """Enumerate exact partitions from already-effective candidate pairs."""
    pairs = tuple(candidate_pairs)
    if component_mode == "global":
        states = enumerate_compatible_partitions(
            n_events, pairs, max_partitions=max_partitions
        )
    elif component_mode == "componentwise":
        states, _summaries, _component_states = exact_partitions_componentwise(
            n_events,
            pairs,
            max_component_events=max_component_events,
            max_component_edges=max_component_edges,
            max_component_partitions=max_component_partitions or max_partitions,
            max_total_partitions=max_total_partitions or max_partitions,
        )
    else:
        raise ValueError("component_mode must be 'global' or 'componentwise'")
    log_z_prior = (
        float(logsumexp([s.log_prior_weight for s in states])) if states else -np.inf
    )
    return n_events, states, log_z_prior


def exact_partitions_from_json(
    data: dict,
    *,
    max_partitions: int = 10_000,
    edge_mark_prior_keys: Sequence[str] | None = None,
    component_mode: str = "global",
    max_component_events: int | None = None,
    max_component_edges: int | None = None,
    max_component_partitions: int | None = None,
    max_total_partitions: int | None = None,
):
    """Validate a candidate-pair JSON object and enumerate exact partitions."""
    n_events, _raw_pairs, effective_pairs, _contributions = (
        prepare_candidate_pairs_for_partitioning(data, edge_mark_prior_keys)
    )
    return exact_partitions_from_pairs(
        n_events,
        effective_pairs,
        max_partitions=max_partitions,
        component_mode=component_mode,
        max_component_events=max_component_events,
        max_component_edges=max_component_edges,
        max_component_partitions=max_component_partitions,
        max_total_partitions=max_total_partitions,
    )
