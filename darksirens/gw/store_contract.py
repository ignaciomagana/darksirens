"""One requirement table for gwcat PE/selection stores, shared by every gate.

The loaders (``darksirens.gw.utils``) and the preflight validators
(``darksirens.lensing.file_contract``) used to carry independent copies of
"what a gwcat file must contain", and the copies drifted: the preflight did
not require ``m1src``/``m2src`` (which the selection loader does), and neither
side checked ``ra``/``dec`` at all.  A file could therefore pass preflight and
fail at load -- or worse, load and feed NaN sky positions to healpy.

This module is the single declarative source of truth.  Each accepted
``format_version`` maps to a :class:`StoreContract` naming the required
datasets and attrs, plus the quality sets: which datasets must be finite,
strictly positive, non-negative, and range-bound.  Consumers run
:func:`quality_problems` (report-oriented; returns strings) and raise in
whatever exception style their layer uses.

Quality semantics worth recording:

* ``p_pe`` is required to be non-negative but NOT strictly positive: a zero
  PE density is legal (the likelihood masks ``prior_wt > 0`` at every
  consumption site, and the shipped whitelist product carries exactly one
  zero from a distance-prior truncation).
* ``pdraw`` IS required to be strictly positive: gwcat refuses to write a
  zero draw density (GW-03), and a detected injection with zero ``pdraw``
  would be silently excluded from the selection sum by the same mask that
  drops padding rows -- biasing mu low with no signature anywhere.
* Sky ranges are radians, ``ra`` in [0, 2*pi) and ``dec`` in [-pi/2, pi/2];
  a degrees-valued file fails by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RangeSpec:
    lo: float
    hi: float
    hi_open: bool = False

    def describe(self) -> str:
        close = ")" if self.hi_open else "]"
        return f"[{self.lo:g}, {self.hi:g}{close}"


#: Shared sky-angle ranges (radians).  Referenced by both loaders and the
#: lensing preflight so the two cannot drift again.
SKY_RANGES: dict[str, RangeSpec] = {
    "ra": RangeSpec(0.0, 2.0 * np.pi, hi_open=True),
    "dec": RangeSpec(-np.pi / 2.0, np.pi / 2.0),
}


@dataclass(frozen=True)
class StoreContract:
    """Requirements for one accepted ``format_version`` family."""

    kind: str
    datasets: tuple[str, ...]
    attrs: tuple[str, ...]
    finite: tuple[str, ...]
    positive: tuple[str, ...] = ()
    nonnegative: tuple[str, ...] = ()
    ranges: Mapping[str, RangeSpec] = field(default_factory=dict)


_PE_CONTRACT = StoreContract(
    kind="pe",
    datasets=(
        "ra",
        "dec",
        "m1det",
        "m2det",
        "dL",
        "chieff",
        "p_pe",
        "m1src",
        "m2src",
    ),
    attrs=(
        "nsamp",
        "nobs",
        "pe_cosmology_H0",
        "pe_cosmology_Om0",
        "chi_eff_in_p_pe",
        "chi_eff_amax",
    ),
    finite=(
        "ra",
        "dec",
        "m1det",
        "m2det",
        "dL",
        "chieff",
        "p_pe",
        "m1src",
        "m2src",
    ),
    positive=("m1det", "m2det", "dL", "m1src", "m2src"),
    nonnegative=("p_pe",),
    ranges={**SKY_RANGES, "chieff": RangeSpec(-1.0, 1.0)},
)

_SELECTION_CONTRACT = StoreContract(
    kind="selection",
    datasets=(
        "m1det",
        "m2det",
        "dL",
        "chieff",
        "ra",
        "dec",
        "pdraw",
        "m1src",
        "m2src",
    ),
    attrs=("ndraw", "chi_eff_swap_applied"),
    finite=(
        "m1det",
        "m2det",
        "dL",
        "chieff",
        "ra",
        "dec",
        "pdraw",
        "m1src",
        "m2src",
    ),
    positive=("m1det", "m2det", "dL", "m1src", "m2src", "pdraw"),
    ranges={**SKY_RANGES, "chieff": RangeSpec(-1.0, 1.0)},
)

#: format_version -> contract.  Adding an accepted format is a deliberate,
#: reviewed act: extend this mapping AND the loader's accepted tuple.
STORE_CONTRACTS: dict[str, StoreContract] = {
    "gwcat-1.0": _PE_CONTRACT,
    "observed-lensing-pe-1.0": _PE_CONTRACT,
    "gwcat-pe-2.0": _PE_CONTRACT,
    "gwcat-pe-2.1": _PE_CONTRACT,
    "gwcat-selection-1.0": _SELECTION_CONTRACT,
    "gwcat-selection-2.0": _SELECTION_CONTRACT,
    "gwcat-selection-2.1": _SELECTION_CONTRACT,
}

#: Formats that carry the gwcat-2.x ``spin_basis`` attr (and, at 2.1, the
#: declarative pairing contract: ``fit_columns``, ``contract``,
#: ``contract_hash``).  The loaders gate these on the chi_eff basis until the
#: capability negotiation of DS-09 lands.
SPIN_BASIS_FORMATS: tuple[str, ...] = (
    "gwcat-pe-2.0",
    "gwcat-pe-2.1",
    "gwcat-selection-2.0",
    "gwcat-selection-2.1",
)

PE_FORMATS: tuple[str, ...] = tuple(
    fmt for fmt, c in STORE_CONTRACTS.items() if c.kind == "pe"
)
SELECTION_FORMATS: tuple[str, ...] = tuple(
    fmt for fmt, c in STORE_CONTRACTS.items() if c.kind == "selection"
)


#: Component-spin datasets (the GWEvent.spin block, in column order).
COMPONENT_SPIN_DATASETS: tuple[str, ...] = ("a1", "a2", "cost1", "cost2")

#: fit_columns implied by each gwcat spin_basis for files that predate the
#: explicit 2.1 attr.
IMPLIED_FIT_COLUMNS: dict[str, tuple[str, ...]] = {
    "chieff": ("m1det", "q", "dL", "chieff"),
    "component": ("m1det", "q", "dL") + COMPONENT_SPIN_DATASETS,
    "chieff_chip": ("m1det", "q", "dL", "chieff", "chip"),
}

#: Advisory columns implied per basis for pre-2.1 files: datasets gwcat ships
#: for convenience whose draw/prior density is NOT in p_pe/pdraw, so fitting
#: on them is invalid (e.g. the component export's chieff and chip columns).
IMPLIED_ADVISORY_COLUMNS: dict[str, tuple[str, ...]] = {
    "chieff": (),
    "component": ("chieff", "chip"),
    "chieff_chip": (),
}

_COMPONENT_RANGES = {
    "a1": RangeSpec(0.0, 1.0),
    "a2": RangeSpec(0.0, 1.0),
    "cost1": RangeSpec(-1.0, 1.0),
    "cost2": RangeSpec(-1.0, 1.0),
}


def _component_variant(base: StoreContract) -> StoreContract:
    """The component-basis variant of a PE/selection contract.

    Adds the four component-spin datasets (finite + range-checked) and, on
    the PE side, DROPS the chi_eff-compat attrs: a component export's p_pe is
    exact in its own basis and legitimately omits chi_eff_in_p_pe /
    chi_eff_amax.  The selection side keeps chi_eff_swap_applied -- it states
    which spin measure pdraw is on, which a component file must still declare.
    """
    attrs = tuple(
        a for a in base.attrs if a not in ("chi_eff_in_p_pe", "chi_eff_amax")
    )
    return StoreContract(
        kind=base.kind,
        datasets=base.datasets + COMPONENT_SPIN_DATASETS,
        attrs=attrs,
        finite=base.finite + COMPONENT_SPIN_DATASETS,
        positive=base.positive,
        nonnegative=base.nonnegative,
        ranges={**base.ranges, **_COMPONENT_RANGES},
    )


def contract_for(fmt: str, basis: str = "chieff") -> StoreContract:
    try:
        base = STORE_CONTRACTS[fmt]
    except KeyError:
        raise KeyError(
            f"no store contract for format_version {fmt!r}; accepted: "
            f"{sorted(STORE_CONTRACTS)}"
        ) from None
    if basis == "component":
        return _component_variant(base)
    return base


def missing_members(f: Any, contract: StoreContract) -> tuple[list[str], list[str]]:
    """Return (missing datasets, missing attrs) against the contract."""
    missing_datasets = [name for name in contract.datasets if name not in f]
    missing_attrs = [name for name in contract.attrs if name not in f.attrs]
    return missing_datasets, missing_attrs


def sky_availability_problem(attrs: Mapping[str, Any]) -> str | None:
    """Refuse a file that declares sky-less campaigns.

    gwcat stamps a per-campaign ``sky_position_available`` bool array; a False
    entry means that campaign's ``ra``/``dec`` rows are NaN placeholders (the
    semianalytic O1/O2 half of a cumulative-mixture file).  darksirens assigns
    every sample to a HEALPix pixel, so a partially-skyless product cannot be
    consumed correctly; this names the campaign structure instead of the
    (equally fatal) NaN row count.
    """
    if "sky_position_available" not in attrs:
        return None
    available = np.atleast_1d(np.asarray(attrs["sky_position_available"])).astype(bool)
    if available.all():
        return None
    return (
        f"sky_position_available={available.tolist()}: at least one campaign "
        "carries no sky positions (NaN ra/dec placeholders); re-export "
        "without the skyless campaigns or use a fully sky-resolved "
        "injection set"
    )


def quality_problems(f: Any, contract: StoreContract) -> list[str]:
    """Finiteness / positivity / range problems for the contract's datasets.

    ``f`` is an open ``h5py.File`` (or any mapping of 1-D arrays with an
    ``attrs`` mapping).  Presence is NOT checked here -- run
    :func:`missing_members` first so absence gets its own actionable error.
    Returns human-readable problem strings; empty means clean.
    """
    problems: list[str] = []
    for name in contract.finite:
        if name not in f:
            continue
        arr = np.asarray(f[name])
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            problems.append(f"dataset {name!r} contains {n_bad} non-finite values")
    for name in contract.positive:
        if name not in f:
            continue
        arr = np.asarray(f[name])
        n_bad = int((np.isfinite(arr) & (arr <= 0.0)).sum())
        if n_bad:
            problems.append(
                f"dataset {name!r} contains {n_bad} non-positive values"
            )
    for name in contract.nonnegative:
        if name not in f:
            continue
        arr = np.asarray(f[name])
        n_bad = int((np.isfinite(arr) & (arr < 0.0)).sum())
        if n_bad:
            problems.append(f"dataset {name!r} contains {n_bad} negative values")
    for name, spec in contract.ranges.items():
        if name not in f:
            continue
        arr = np.asarray(f[name])
        finite = arr[np.isfinite(arr)]
        below = int((finite < spec.lo).sum())
        above = int(
            ((finite >= spec.hi) if spec.hi_open else (finite > spec.hi)).sum()
        )
        if below or above:
            hint = ""
            if name == "ra":
                hint = " (radians; ra must be in [0, 2*pi))"
            elif name == "dec":
                hint = " (radians; dec must be in [-pi/2, pi/2])"
            problems.append(
                f"dataset {name!r} contains {below + above} values outside "
                f"{spec.describe()}{hint}"
            )
    sky_problem = sky_availability_problem(f.attrs)
    if sky_problem is not None:
        problems.append(sky_problem)
    return problems
