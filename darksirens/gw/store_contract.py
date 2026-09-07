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

Layout semantics (:func:`layout_problems`, :func:`count_problems`) are as
load-bearing as the value checks.  The contract used to say nothing about
shape, so a PE file whose ``ra`` had length ONE loaded cleanly and then
BROADCAST that single right ascension across every posterior sample: measured
on a nobs=2 x nsamp=3 fixture, the six samples were assigned HEALPix pixels
[561, 497, 401, 337, 241, 177] from one shared ra=0.25 and six distinct decs,
i.e. six wrong sky rows with no error anywhere (review DATA-01).  A length-5
``ra`` against six samples merely loaded and failed later, deep in a
broadcast; the singleton case never failed at all.  Every required dataset is
therefore pinned 1-D at one common length -- ``nobs * nsamp`` for PE, one
shared detected-injection length for selection -- and the campaign-size attrs
(``nobs``, ``nsamp``, ``ndraw``) are required to be positive, with ``ndraw``
at least the number of detected rows it is the denominator for.
"""
from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from math import prod
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
    #: (primary, secondary) dataset pairs that must satisfy secondary <=
    #: primary.  ``q = m2/m1 <= 1`` is a labelling convention, not a physical
    #: bound, but every consumer here relies on it: the pairing models
    #: normalise p(q|m1) over (m_min/m1, 1], so a row with m2 > m1 is handed a
    #: finite density from outside the domain the normaliser covers (PHY-09).
    ordered_masses: tuple[tuple[str, str], ...] = ()


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
    ordered_masses=(("m1det", "m2det"), ("m1src", "m2src")),
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
    ordered_masses=(("m1det", "m2det"), ("m1src", "m2src")),
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
        ordered_masses=base.ordered_masses,
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


def _length_checked_names(f: Any, contract: StoreContract) -> list[str]:
    """Contract datasets plus any OPTIONAL column the loaders still read.

    A chi_eff-basis export may carry the four component-spin datasets even
    though its contract does not require them; both loaders copy every one
    that is present into the store's columns, so a short ``a1`` would ride
    into the likelihood exactly like a short ``ra``.  ``chip`` is never read
    and is deliberately not length-checked.
    """
    names = list(contract.datasets)
    names += [
        name
        for name in COMPONENT_SPIN_DATASETS
        if name not in contract.datasets and name in f
    ]
    return names


def _shape_of(f: Any, name: str) -> tuple[int, ...]:
    """The shape of ``f[name]`` WITHOUT materialising its data.

    A shape test needs no bytes: an ``h5py.Dataset`` carries ``.shape`` in the
    file's metadata, and reading the column instead costs a full gzip
    decompression (measured 42.5 ms per 1.07e6-row selection column, and this
    module walks every column twice).  Plain mappings of ndarrays -- which
    both validators also accept -- answer ``.shape`` for free; anything else
    (a list, say) still falls back to ``np.asarray``.
    """
    value = f[name]
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    return tuple(shape)


class ColumnView(MappingABC):
    """The store's columns read ONCE, still answering ``.attrs``.

    :func:`quality_problems` accepts "any mapping of 1-D arrays with an
    ``attrs`` mapping"; handing it one of these instead of the open
    ``h5py.File`` makes the validators and the loader that follows them share
    a single read per dataset rather than each decompressing the file anew
    (measured 49 full-dataset reads for the 9 selection columns).
    """

    def __init__(self, columns: Mapping[str, np.ndarray], attrs: Mapping[str, Any]):
        self._columns = dict(columns)
        self.attrs = attrs

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


def read_columns(f: Any, contract: StoreContract) -> ColumnView:
    """Materialise every column this module's gates and the loaders read.

    Only names PRESENT in ``f`` are read, so a missing dataset still reaches
    the gate that owns its error message, and ``np.asarray`` is used verbatim
    (no reshape/astype) so a malformed 2-D or scalar column still arrives at
    :func:`layout_problems` exactly as it does when read straight from the
    file.
    """
    names: list[str] = list(_length_checked_names(f, contract))
    for extra in (
        contract.finite,
        contract.positive,
        contract.nonnegative,
        tuple(contract.ranges),
        tuple(n for pair in contract.ordered_masses for n in pair),
    ):
        names += [name for name in extra if name not in names]
    return ColumnView(
        {name: np.asarray(f[name]) for name in names if name in f},
        f.attrs,
    )


def layout_problems(
    f: Any,
    contract: StoreContract,
    *,
    expected_size: int | None = None,
) -> list[str]:
    """One-dimensionality and common-length problems for the store's columns.

    Presence is NOT checked here -- run :func:`missing_members` first.  With
    ``expected_size`` every present column must have exactly that length (the
    PE case, ``nobs * nsamp``); without it they must merely agree with one
    another (the selection case, where the detected count is whatever the
    campaign produced).  Shape problems short-circuit the length report: a
    2-D column has no meaningful length to compare.
    """
    problems: list[str] = []
    sizes: dict[str, int] = {}
    for name in _length_checked_names(f, contract):
        if name not in f:
            continue
        shape = _shape_of(f, name)
        if len(shape) != 1:
            problems.append(
                f"dataset {name!r} has shape {shape}; store "
                "columns must be one-dimensional"
            )
            continue
        sizes[name] = int(shape[0])
    if problems:
        return problems
    if expected_size is not None:
        wrong = {n: s for n, s in sizes.items() if s != expected_size}
        if wrong:
            problems.append(
                "dataset length(s) "
                + ", ".join(f"{n}={s}" for n, s in sorted(wrong.items()))
                + f" != expected {expected_size} (nobs * nsamp); a shorter "
                "column BROADCASTS silently over the samples and assigns the "
                "wrong sky/mass rows"
            )
    elif len(set(sizes.values())) > 1:
        problems.append(
            "store columns have inconsistent lengths ("
            + ", ".join(f"{n}={s}" for n, s in sorted(sizes.items()))
            + "); every detected-injection column must share one length, or "
            "a shorter column broadcasts silently over the rest"
        )
    return problems


def common_length(f: Any, contract: StoreContract) -> int:
    """The shared 1-D length of the store's columns.

    Only meaningful once :func:`layout_problems` is clean; returns 0 for a
    store with no length-checked column present.
    """
    sizes = {
        int(prod(_shape_of(f, name)))
        for name in _length_checked_names(f, contract)
        if name in f
    }
    return max(sizes) if sizes else 0


def _positive_count(attrs: Mapping[str, Any], name: str) -> tuple[int | None, str | None]:
    """(value, problem) for a scalar positive-integer attr."""
    if name not in attrs:
        return None, None
    raw = np.atleast_1d(np.asarray(attrs[name]))
    if raw.size != 1:
        return None, f"attr {name!r} must be a scalar count, got {raw.size} values"
    try:
        value = int(raw.reshape(())[()])
    except (TypeError, ValueError):
        return None, f"attr {name!r} is not an integer count"
    if value < 1:
        return None, f"attr {name!r}={value} must be a positive count"
    return value, None


def count_problems(
    attrs: Mapping[str, Any],
    contract: StoreContract,
    *,
    n_rows: int | None = None,
) -> list[str]:
    """Problems with the campaign-size attrs (``nobs``/``nsamp``/``ndraw``).

    A zero or negative count is not a harmless oddity: ``nobs * nsamp`` is the
    PE reshape target (a zero divides by zero in the per-event normalisation),
    and ``ndraw`` is the DENOMINATOR of the selection integral, so a
    non-positive value silently produces an infinite or sign-flipped mu.
    ``n_rows``, when given, additionally requires ``ndraw`` to be at least the
    number of detected rows it normalises -- more detections than draws is
    arithmetically impossible and means the two were paired from different
    campaigns.
    """
    problems: list[str] = []
    names = ("nobs", "nsamp") if contract.kind == "pe" else ("ndraw",)
    values: dict[str, int] = {}
    for name in names:
        value, problem = _positive_count(attrs, name)
        if problem is not None:
            problems.append(problem)
        elif value is not None:
            values[name] = value
    if (
        contract.kind == "selection"
        and n_rows is not None
        and "ndraw" in values
        and values["ndraw"] < n_rows
    ):
        problems.append(
            f"attr 'ndraw'={values['ndraw']} is smaller than the "
            f"{n_rows} detected injection(s) it normalises; the detected set "
            "and the draw count come from different campaigns"
        )
    return problems


def expected_pe_size(attrs: Mapping[str, Any]) -> int | None:
    """``nobs * nsamp`` for a PE store, or None if either attr is unusable.

    Only meaningful once :func:`count_problems` is clean.
    """
    nobs, nobs_problem = _positive_count(attrs, "nobs")
    nsamp, nsamp_problem = _positive_count(attrs, "nsamp")
    if nobs is None or nsamp is None or nobs_problem or nsamp_problem:
        return None
    return nobs * nsamp


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
    for primary, secondary in contract.ordered_masses:
        if primary not in f or secondary not in f:
            continue
        m1 = np.asarray(f[primary])
        m2 = np.asarray(f[secondary])
        if m1.shape != m2.shape:
            continue  # layout_problems owns that failure
        ok = np.isfinite(m1) & np.isfinite(m2)
        n_bad = int((m2[ok] > m1[ok]).sum())
        if n_bad:
            worst = float(np.max(m2[ok] / m1[ok]))
            problems.append(
                f"dataset {secondary!r} exceeds {primary!r} in {n_bad} row(s) "
                f"(largest q = {worst:.6g} > 1); the pairing models normalise "
                "p(q|m1) over q <= 1, so such a row is scored with a density "
                "from outside the normalisation domain"
            )
    sky_problem = sky_availability_problem(f.attrs)
    if sky_problem is not None:
        problems.append(sky_problem)
    return problems
