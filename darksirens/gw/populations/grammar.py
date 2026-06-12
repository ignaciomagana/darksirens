r"""Model-name grammar and generic mixture assembly.

The population model name IS the model definition::

    name        := composition suffix?
    composition := term ("+" term)*
    term        := TOKEN            # "powerlaw", "brokenpowerlaw", "peak"
                 | DIGITS PLURAL    # "2peaks", "3powerlaws"
    suffix      := "_shared_beta" | "_shared_spin" | "_shared_beta_spin"

so ``"brokenpowerlaw+2peaks+powerlaw"`` parses to the mass composition
``[BrokenPowerLaw, Gaussian, Gaussian, PowerLaw]``.  Components self-register
grammar tokens through :mod:`darksirens.gw.populations.components`; any unseen
composition (``"powerlaw+3peaks"``) therefore builds with blueprint-default
priors and fiducials, no registration required.

Label rule (uniform, no per-model overrides)
--------------------------------------------
* weights: ``$v_1$ ... $v_{k-1}$`` (stick-breaking inputs, from MixtureModel);
* mass parameters: base label if the mixture has a single mass component,
  otherwise the base label tagged with the slot tag, e.g. ``$\alpha_{\rm PL}$``,
  ``$\mu_{\rm G1}$``, ``$m_{\min,\rm BPL}$``;
* pairing/spin: bare (``$\beta$``, ``$\mu_\chi$``) when shared or k == 1,
  tagged with the mass slot tag (``$\beta_{\rm G2}$``) when per-component;
* ``$\gamma$`` always last (appended by PopulationModel).

Slot tags are the component short name (``PL``, ``BPL``, ``G``) plus a 1-based
index iff that token occurs more than once: ``brokenpowerlaw+2peaks+powerlaw``
tags ``[BPL, G1, G2, PL]``.

Parameter ordering (the sampler contract) is unchanged from the old registry:
``v_weights -> mass slots (composition order) -> pairing -> spin -> gamma``.
The fiducial assembler walks exactly the same slot lists as the model builder,
so bounds, labels, and fiducials can never disagree on order or length.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from .base import MixtureModel, ParamSpec, PopulationModel
from .components import COMPONENTS, MASS_TOKENS, PLURALS, _B, tag_latex

_LOG = logging.getLogger(__name__)

DEFAULT_PAIRING = "pl_pairing"
DEFAULT_SPIN = "spin"
GAMMA_SPEC = ("gamma", r"\gamma", -10.0, 10.0)


class ModelNameError(ValueError):
    """Raised when a population model name cannot be parsed."""


# ============================================================
# Parsing
# ============================================================

_SUFFIXES = (
    ("_shared_beta_spin", True, True),
    ("_shared_beta", True, False),
    ("_shared_spin", False, True),
)
_COUNT_RE = re.compile(r"^([1-9]\d*)([a-z_]+)$")


@dataclass(frozen=True)
class SlotSpec:
    """One mass-component slot in a parsed composition."""

    token: str        # component key, e.g. "powerlaw"
    occurrence: int   # 1-based among slots with the same token
    tag: str          # "PL", "G1", ...


@dataclass(frozen=True)
class ModelIR:
    """Parsed model name: composition slots plus sharing flags."""

    slots: Tuple[SlotSpec, ...]
    shared_beta: bool
    shared_spin: bool
    base_name: str    # name as typed, suffix stripped
    canonical: str    # run-collapsed canonical composition key


def _split_suffix(name: str):
    for suffix, sb, ss in _SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], sb, ss
    return name, False, False


def _unknown_term_error(term: str, name: str) -> ModelNameError:
    vocabulary = sorted({*MASS_TOKENS, *PLURALS})
    hints = difflib.get_close_matches(term, vocabulary, n=3)
    suggestion = f" Did you mean {' or '.join(map(repr, hints))}?" if hints else ""
    return ModelNameError(
        f"Unknown component {term!r} in population model {name!r}."
        f"{suggestion} Known tokens: {sorted(MASS_TOKENS)} "
        f"(count prefixes use plurals, e.g. '2peaks', '3powerlaws')."
    )


def _parse_term(term: str, name: str) -> Tuple[str, int]:
    """Parse one '+'-separated term into (token, count)."""
    if term in MASS_TOKENS:
        return term, 1
    if term in PLURALS:
        raise ModelNameError(
            f"Ambiguous bare plural {term!r} in population model {name!r}: "
            f"write an explicit count ('2{term}') or the singular "
            f"({PLURALS[term]!r})."
        )
    m = _COUNT_RE.match(term)
    if m:
        count, stem = int(m.group(1)), m.group(2)
        token = PLURALS.get(stem, stem if stem in MASS_TOKENS else None)
        if token is not None:
            return token, count
        raise _unknown_term_error(stem, name)
    raise _unknown_term_error(term, name)


def make_slots(tokens: Sequence[str]) -> Tuple[SlotSpec, ...]:
    """Assign tags to a token sequence: short name + index iff repeated."""
    totals = Counter(tokens)
    seen: Counter = Counter()
    slots = []
    for token in tokens:
        blueprint = COMPONENTS[token]
        if blueprint.kind != "mass":
            raise ValueError(f"Component {token!r} is not a mass component")
        seen[token] += 1
        tag = blueprint.short + (str(seen[token]) if totals[token] > 1 else "")
        slots.append(SlotSpec(token, seen[token], tag))
    return tuple(slots)


def canonical_composition(tokens: Sequence[str]) -> str:
    """Collapse consecutive runs: [powerlaw, powerlaw, peak] -> '2powerlaws+peak'."""
    runs: list[list] = []
    for token in tokens:
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    parts = []
    for token, n in runs:
        parts.append(token if n == 1 else f"{n}{MASS_TOKENS[token].plural}")
    return "+".join(parts)


@lru_cache(maxsize=None)
def parse_model_name(name: str) -> ModelIR:
    """Parse a population model name into its intermediate representation."""
    base, shared_beta, shared_spin = _split_suffix(name)
    if not base:
        raise ModelNameError(f"Empty composition in population model {name!r}")
    tokens: list[str] = []
    for term in base.split("+"):
        token, count = _parse_term(term, name)
        tokens.extend([token] * count)
    return ModelIR(
        slots=make_slots(tokens),
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        base_name=base,
        canonical=canonical_composition(tokens),
    )


def derive_latex(tokens: Sequence[str]) -> str:
    """Display name from token runs: [bpl, peak, peak, pl] -> 'BPL+2G+PL'."""
    runs: list[list] = []
    for token in tokens:
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    parts = []
    for token, n in runs:
        short = COMPONENTS[token].short
        parts.append(short if n == 1 else f"{n}{short}")
    return "+".join(parts)


# ============================================================
# Stick-breaking weight conversion (moved verbatim from registry.py)
# ============================================================

def _w_to_v(w_desired_full) -> list:
    """
    Convert a k-vector of desired mixture weights to k-1 stick-breaking inputs.

    Formula
    -------
        v_1 = w_1
        v_i = w_i / (1 - w_1 - ... - w_{i-1})   for i >= 2

    Inverse of _stick_breaking_weights in base.py.
    """
    w = np.asarray(w_desired_full, dtype=float)
    k = len(w)
    if k == 1:
        return []
    v = []
    remaining = 1.0
    for i in range(k - 1):
        vi = float(w[i] / remaining) if remaining > 1e-12 else 0.0
        v.append(vi)
        remaining -= float(w[i])
    return v


def _stick_breaking_weights_np(v) -> np.ndarray:
    """
    Pure-numpy stick-breaking for use at import time before JAX x64 is configured.
    Mathematically identical to _stick_breaking_weights in base.py.
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return np.array([1.0])
    cumprod = np.cumprod(1.0 - v)
    remaining = np.concatenate([[1.0], cumprod[:-1]])
    return np.concatenate([v * remaining, [cumprod[-1]]])


def _verify_w_to_v_roundtrip():
    """Sanity check at import time: _w_to_v o _stick_breaking_weights = id."""
    cases = [
        [1.0],
        [0.1, 0.9],
        [0.10, 0.05, 0.85],
        [0.10, 0.05, 0.03, 0.82],
        [0.10, 0.05, 0.05, 0.80],
        [0.20, 0.10, 0.70],
        [0.15, 0.10, 0.05, 0.70],
        [0.15, 0.10, 0.05, 0.03, 0.67],
    ]
    for w in cases:
        v = _w_to_v(w)
        w_r = list(_stick_breaking_weights_np(v))
        err = max(abs(a - b) for a, b in zip(w, w_r))
        assert err < 1e-10, (
            f"_w_to_v round-trip failed: desired={w}, recovered={w_r}, err={err:.2e}"
        )


_verify_w_to_v_roundtrip()


# ============================================================
# Generic builder
# ============================================================

def _validate_override_keys(overrides: Mapping, blueprint, slot_index: int):
    known = {p.name for p in blueprint.resolve_params()}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(
            f"Unknown parameter override(s) {sorted(unknown)} for slot "
            f"{slot_index} ({blueprint.key!r}); known parameters: {sorted(known)}"
        )


def _make_specs(blueprint, tag: Optional[str], bounds_override: Mapping[str, _B]):
    """ParamSpec list for one component slot; ``tag=None`` means untagged."""
    specs = []
    for p in blueprint.resolve_params():
        bounds = bounds_override.get(p.name, p.bounds)
        body = tag_latex(p.latex, tag) if tag else p.latex
        ascii_name = f"{tag}.{p.name}" if tag else p.name
        specs.append(ParamSpec(f"${body}$", bounds.lo, bounds.hi, name=ascii_name))
    return specs


def _resolve_sub_slots(kind: str, default_key: str, explicit, slots, shared: bool):
    """Pairing/spin slot list as (component_key, tag_or_None) pairs.

    Explicit slot lists (GP models) are never tagged and ignore the shared
    flag, matching the old registry's flag-ignoring GP factories.  Default
    slots are one untagged component when shared (or k == 1), else one
    tagged component per mass slot.
    """
    if explicit is not None:
        return [(key, None) for key in explicit]
    if shared or len(slots) == 1:
        return [(default_key, None)]
    return [(default_key, slot.tag) for slot in slots]


def build_population_model(
    mass_tokens: Sequence[str],
    *,
    shared_beta: bool = False,
    shared_spin: bool = False,
    pairing: Optional[Sequence[str]] = None,
    spin: Optional[Sequence[str]] = None,
    bounds: Optional[Mapping[int, Mapping[str, _B]]] = None,
) -> PopulationModel:
    """Assemble a PopulationModel from component keys plus optional overrides.

    ``bounds`` maps mass-slot index -> {param name -> _B} for curated
    per-slot prior tuning; everything else comes from the blueprints.
    """
    slots = make_slots(mass_tokens)
    tagged = len(slots) >= 2

    mass_components = []
    for i, slot in enumerate(slots):
        blueprint = COMPONENTS[slot.token]
        slot_bounds = dict((bounds or {}).get(i, {}))
        _validate_override_keys(slot_bounds, blueprint, i)
        specs = _make_specs(blueprint, slot.tag if tagged else None, slot_bounds)
        mass_components.append(blueprint.instantiate(specs))

    def _build_sub(kind, default_key, explicit, shared):
        comps = []
        for key, tag in _resolve_sub_slots(kind, default_key, explicit, slots, shared):
            blueprint = COMPONENTS[key]
            if blueprint.kind != kind:
                raise ValueError(f"Component {key!r} is not a {kind} component")
            comps.append(blueprint.instantiate(_make_specs(blueprint, tag, {})))
        return comps

    model = PopulationModel(mixture=MixtureModel(
        mass_components,
        _build_sub("pairing", DEFAULT_PAIRING, pairing, shared_beta),
        _build_sub("spin", DEFAULT_SPIN, spin, shared_spin),
    ))

    # Backstop: downstream prior code keys on labels, so they must be unique.
    # The always-tag rule makes collisions impossible for any composition the
    # grammar can produce; this guards future blueprint mistakes.
    labels = [s.label for s in model.param_specs]
    duplicates = sorted({l for l, n in Counter(labels).items() if n > 1})
    if duplicates:
        raise ValueError(f"Duplicate parameter labels in mixture: {duplicates}")
    return model


def build_fiducial_vector(
    mass_tokens: Sequence[str],
    *,
    shared_beta: bool = False,
    shared_spin: bool = False,
    pairing: Optional[Sequence[str]] = None,
    spin: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    fids: Optional[Mapping[int, Mapping[str, float]]] = None,
    bounds: Optional[Mapping[int, Mapping[str, _B]]] = None,
    gamma: float = 0.0,
    on_violation: str = "ignore",
) -> jnp.ndarray:
    """Fiducial parameter vector in exactly the model's parameter order.

    ``weights`` holds the k-1 leading DESIRED FINAL FRACTIONS (human-readable;
    the last fraction is implied) and is converted to stick-breaking inputs
    via :func:`_w_to_v`.  ``None`` means uniform weights.  ``fids`` patches
    individual mass-slot fiducials by parameter name.

    ``bounds`` (the same per-slot prior overrides handed to
    :func:`build_population_model`) enables a fiducial-within-prior check:
    ``on_violation`` is ``"ignore"``, ``"warn"`` (curated legacy models carry
    known out-of-bounds fiducials that are preserved verbatim), or ``"error"``
    (novel compositions, where a violation means a blueprint bug).
    """
    slots = make_slots(mass_tokens)
    k = len(slots)

    if weights is None:
        w_full = [1.0 / k] * k
    else:
        if len(weights) != k - 1:
            raise ValueError(
                f"weights must have k-1={k - 1} entries, got {len(weights)}"
            )
        w_full = list(weights) + [1.0 - sum(weights)]
    v_weights = _w_to_v(w_full)

    values = list(v_weights)
    violations = []

    def _collect(blueprint, slot_fids, slot_bounds, where):
        for p in blueprint.resolve_params():
            value = slot_fids.get(p.name, p.fiducial)
            lo, hi = slot_bounds.get(p.name, p.bounds)
            if not (lo <= value <= hi):
                violations.append(
                    f"{where}.{p.name}: fiducial {value} outside prior [{lo}, {hi}]"
                )
            values.append(value)

    for i, slot in enumerate(slots):
        blueprint = COMPONENTS[slot.token]
        slot_fids = dict((fids or {}).get(i, {}))
        slot_bounds = dict((bounds or {}).get(i, {}))
        _validate_override_keys(slot_fids, blueprint, i)
        _collect(blueprint, slot_fids, slot_bounds, slot.tag or slot.token)

    for key, tag in _resolve_sub_slots("pairing", DEFAULT_PAIRING, pairing, slots, shared_beta):
        _collect(COMPONENTS[key], {}, {}, tag or key)
    for key, tag in _resolve_sub_slots("spin", DEFAULT_SPIN, spin, slots, shared_spin):
        _collect(COMPONENTS[key], {}, {}, tag or key)

    values.append(gamma)

    if violations and on_violation != "ignore":
        message = "; ".join(violations)
        if on_violation == "error":
            raise ValueError(f"Fiducial parameters outside prior bounds: {message}")
        _LOG.warning("Fiducial parameters outside prior bounds: %s", message)

    return jnp.array(values, dtype=float)
