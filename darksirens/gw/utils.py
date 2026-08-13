import importlib
import importlib.util
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping

# Same two knobs, same values, as core.jax_config.configure_jax_runtime (which
# also enables x64; this module must not, so it only sets the env).  setdefault
# on both sides means the effective configuration is order-independent as long
# as the two stay in sync.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Respect the start method if the embedding application already selected one.
    pass

import jax

from jax import numpy as jnp

import numpy as np
try:
    import healpy as hp
except ModuleNotFoundError:
    class _HealpyFallback:
        __version__ = "unavailable"

    hp = _HealpyFallback()

import h5py

from darksirens.gw import store_contract

_TQDM_MODULE = sys.modules.get("tqdm")
if _TQDM_MODULE is not None and hasattr(_TQDM_MODULE, "tqdm"):
    tqdm = _TQDM_MODULE.tqdm
elif importlib.util.find_spec("tqdm") is not None:
    tqdm = importlib.import_module("tqdm").tqdm
else:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _NoOpProgress:
                def update(self, *_args, **_kwargs):
                    return None

                def close(self):
                    return None

                def set_postfix(self, *_args, **_kwargs):
                    return None

            return _NoOpProgress()
        return iterable

try:
    from gwcat.spin import chi_eff_prior_logprob
except ModuleNotFoundError:
    def chi_eff_prior_logprob(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "gwcat is required to load gwcat PE/selection files; install the gwcat package or use precomputed prior weights"
        )

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in arctanh")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

def _decode_hdf5_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _require_hdf5_format(f, expected, conversion_hint):
    observed = _decode_hdf5_attr(f.attrs.get("format_version", ""))
    expected_values = (expected,) if isinstance(expected, str) else tuple(expected)
    if observed not in expected_values:
        raise RuntimeError(
            f"Unsupported GW catalog format {observed!r}. Expected {expected!r}. "
            f"{conversion_hint}"
        )
    return observed


def _require_chieff_spin_basis(f, path, reexport_hint):
    """Reject a gwcat-2.x export that is not written in the chi_eff spin basis.

    gwcat-2.x files carry a ``spin_basis`` attr in {"chieff", "component",
    "chieff_chip"}.  Only the "chieff" basis is array-compatible with the
    legacy 1.0 contract: the "component" and "chieff_chip" bases fold a
    different spin-prior/draw factor into ``p_pe``/``pdraw`` and omit the
    chi_eff-compat attrs.  darksirens' likelihood is chi_eff-based (its
    population model uses a 1-D ``SpinModel``), so consuming those bases would
    be silently wrong; reject them with an actionable error.
    """
    basis = _decode_hdf5_attr(f.attrs.get("spin_basis", ""))
    if basis != "chieff":
        raise RuntimeError(
            f"gwcat-2.x file {path!r} was exported with spin_basis={basis!r}, but "
            "darksirens' likelihood is chi_eff-based and can only consume the "
            f"'chieff' basis. {reexport_hint}"
        )


#: Fraction of ``max_likelihood_variance`` the PE reweighting may consume before
#: the loader calls it out.  Not a guard -- the guard lives in
#: ``likelihood/selection.py`` and is enforced there.  This is visibility: a
#: PE product that spends most of the budget makes the run's selection-N_eff
#: requirement much harder to clear, and there was previously no way to see that
#: short of the run failing.
PE_VARIANCE_NOTICE_FRACTION = 0.20


def _report_pe_weight_health(p_pe_2d, fmt, nEvents, nsamp, f_attrs=None):
    """Print the PE reweighting's ESS and its share of the variance budget.

    The reliability guard bounds the variance of the TOTAL log-likelihood
    estimator, so every nat the PE weights spend RAISES the selection N_eff the
    run must clear::

        budget    = max_likelihood_variance - pe_variance_sum
        threshold = max(5 N_obs, N_obs^2 / budget)

    ``pe_variance_sum`` was threaded into that guard and reported only by the
    LENSING CLI, so a standard dark-siren run could sit at three quarters of its
    budget -- or fail the guard -- with no indication of why.  This is the
    load-time half of ``scripts/pe_weight_diagnostics.py``.

    What is reported is the PE-PRIOR share: the ESS of ``1/p_pe`` within each
    event.  The run's true ``pe_variance_sum`` uses the full per-sample weight
    (population, redshift prior, Jacobian) and is therefore LARGER.  The
    reported number is cosmology-independent -- it is a property of the file --
    which is exactly why it is worth printing before any sampling starts.

    Scale-invariant in ``p_pe``: the estimator is a ratio of weight moments, so
    it does not matter that this runs before the per-event normalisation.
    Non-positive samples are zero-weighted, matching the likelihood's own
    ``prior_wt > 0`` masking, and still count in ``n``.
    """
    good = p_pe_2d > 0.0
    w = np.where(good, 1.0 / np.where(good, p_pe_2d, 1.0), 0.0)
    sw = w.sum(axis=1)
    sw2 = (w ** 2).sum(axis=1)
    ess = np.where(sw2 > 0.0, sw ** 2 / np.where(sw2 > 0.0, sw2, 1.0), 0.0)
    frac = ess / nsamp
    var = np.maximum(
        np.where(sw > 0.0, sw2 / np.where(sw > 0.0, sw ** 2, 1.0), 0.0)
        - 1.0 / nsamp,
        0.0,
    )
    total = float(var.sum())
    attrs = f_attrs or {}
    n_masked = int((~good).sum())
    print(f"    [gwcat PE] format={fmt}  {nEvents:,} events x {nsamp:,} samples  "
          f"H0={attrs.get('H0', '?')}  Om0={attrs.get('Om0', '?')}"
          + (f"  ({n_masked} non-positive p_pe zero-weighted)" if n_masked else ""))
    print(f"    PE reweighting ESS/nsamp: min={frac.min():.4f}  "
          f"median={np.median(frac):.4f}  max={frac.max():.4f}  "
          f"({int((frac < 0.1).sum())} events < 0.1)")
    print(f"    pe_variance_sum = {total:.4f}  "
          f"(PE-prior share of the total-variance budget; the run's value is "
          "larger)")
    if total > PE_VARIANCE_NOTICE_FRACTION:
        worst = int(np.argmax(var))
        print(f"    [!] that is {100.0 * total:.0f}% of a "
              f"max_likelihood_variance of 1.0, so the selection N_eff this run "
              f"must clear is inflated by ~{1.0 / max(1.0 - total, 1e-12):.2f}x. "
              f"Event index {worst} alone contributes {var[worst]:.4f}. "
              "See scripts/pe_weight_diagnostics.py; re-analysing or dropping "
              "the worst events is usually cheaper than raising the budget.")
    return total


def _require_store_quality(f, contract, path, conversion_hint=""):
    """Raise on finiteness / positivity / range violations against the contract.

    The checks themselves (and the requirement tables) live in
    ``darksirens.gw.store_contract`` -- one declarative source shared with the
    lensing preflight validator, so the loader and the preflight cannot drift
    apart again.  Notable consequences: NaN or degrees-valued ``ra``/``dec``
    is rejected before it reaches ``hp.ang2pix`` (gwcat legitimately writes
    NaN sky for semianalytic O1/O2 campaigns), a partially-skyless
    ``sky_position_available`` is refused by name, and a zero ``pdraw`` --
    which the likelihood would silently exclude from the selection sum with
    the same mask that drops padding rows -- fails loudly here instead.
    """
    problems = store_contract.quality_problems(f, contract)
    if problems:
        raise RuntimeError(
            f"Invalid data in {path!r}: "
            + "; ".join(problems)
            + "."
            + (f" {conversion_hint}" if conversion_hint else "")
        )


def _require_hdf5_members(f, datasets=(), attrs=(), conversion_hint=""):
    missing_datasets = [name for name in datasets if name not in f]
    missing_attrs = [name for name in attrs if name not in f.attrs]
    if missing_datasets or missing_attrs:
        details = []
        if missing_datasets:
            details.append("datasets: " + ", ".join(missing_datasets))
        if missing_attrs:
            details.append("attributes: " + ", ".join(missing_attrs))
        raise RuntimeError(
            "Incomplete gwcat export; missing "
            + "; ".join(details)
            + (f". {conversion_hint}" if conversion_hint else ".")
        )


#: Columns a chi_eff-basis gwcat file's density covers.  1.0 / 2.0 files do
#: not declare ``fit_columns``; this is their implied value.  gwcat 2.1 files
#: will carry the attr explicitly (review DS-05/DS-09).
_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")


def _decoded_attrs(f):
    """All file attrs as a plain dict with bytes decoded (arrays passed through)."""
    return {key: _decode_hdf5_attr(f.attrs[key]) for key in f.attrs}


def _decoded_event_names(f):
    """gwcat's ``event_names`` attr (array of strings) as a tuple, or None."""
    if "event_names" not in f.attrs:
        return None
    return tuple(
        v.decode() if isinstance(v, bytes) else str(v)
        for v in np.atleast_1d(f.attrs["event_names"])
    )


@dataclass(frozen=True)
class GWStore:
    """A loaded gwcat PE store: named columns, provenance, and the prior weight.

    The positional 9-tuple returned by :func:`load_gw_samples` is unpacked
    through ``loaders.py`` -> ``data.py`` -> ``factory.py`` by position, which
    is why adding one coordinate means a cascade across every unpack site.
    This record is the extensible surface: consumers pick columns by NAME, new
    columns / attrs are additive, and the tuple loader is a thin wrapper kept
    byte-identical for existing call sites.

    ``columns`` holds the raw file datasets (numpy, flattened, length
    ``n_events * nsamp``).  ``prior_wt`` is the PROCESSED PE proposal density:
    chi_eff prior folded in when the file declares it absent, then normalised
    per event -- i.e. exactly the ``p_pe`` the tuple loader returns, not the
    raw dataset.  ``event_names`` surfaces gwcat's per-event identity attr
    (previously read by nothing, so bright-siren counterpart triplets aligned
    by position only); None when the file predates it.
    """

    format_version: str
    path: str
    fit_columns: tuple[str, ...]
    columns: Mapping[str, np.ndarray]
    attrs: Mapping[str, Any]
    n_events: int
    nsamp: int
    prior_wt: np.ndarray
    event_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SelectionStore:
    """A loaded gwcat selection store (detected injections + draw density).

    ``prior_wt`` is the PROCESSED physical draw density ``pdraw``: the 1-D
    chi_eff swap is folded in when the file declares it absent.  Its absolute
    scale is load-bearing (it enters the expected-detection integral), so it
    is never normalised.  ``ndraw`` is the campaign's total generated count.
    """

    format_version: str
    path: str
    fit_columns: tuple[str, ...]
    columns: Mapping[str, np.ndarray]
    attrs: Mapping[str, Any]
    n_injections: int
    ndraw: int
    prior_wt: np.ndarray


def load_gw_store(gw_path) -> GWStore:
    """
    Load a gwcat PE export as a :class:`GWStore` record.

    ``gw_path`` must point to a file with ``format_version`` in
    {"gwcat-1.0", "observed-lensing-pe-1.0", "gwcat-pe-2.0", "gwcat-pe-2.1"}
    produced by
    ``gwcat.GWCatalog.to_darksirens(...)`` / the versioned export layer.  A
    ``gwcat-pe-2.x`` file is only accepted when its ``spin_basis`` attr is
    ``"chieff"`` (the sole basis array-compatible with darksirens'
    chi_eff-based likelihood); the ``"component"`` and ``"chieff_chip"`` bases
    are rejected with an actionable error.  The loader intentionally rejects
    raw PE files and partially-populated HDF5 files so that all catalog
    ingestion and coordinate conversions remain owned by gwcat.

    ``GWStore.columns`` holds the raw file datasets (numpy, flattened,
    length ``n_events * nsamp``); ``GWStore.prior_wt`` is the processed
    per-event-normalised PE proposal density in the likelihood's canonical
    sample basis ``(m1det, q, dL)`` with ``q = m2det / m1det``, chi_eff
    prior included.  :func:`load_gw_samples` is the positional-tuple
    wrapper over this record.
    """

    # Name the FROZEN v1 entry point, not the versioned export.  gwcat's
    # ``export()`` resolves its parameter space from a registry at call time,
    # and that registry's default is now ``component`` -- which this loader
    # rejects -- so ``export(path)`` with no basis produces a file that cannot
    # be loaded here.  ``to_darksirens`` takes no spin-basis argument at all and
    # is therefore immune to that default moving again.  (The old
    # ``_to_darksirens_format`` spelling still works but is a deprecated alias
    # scheduled for removal, so pointing users at it in an error message would
    # hand them a DeprecationWarning today and a broken instruction later.)
    conversion_hint = (
        "Create the PE file with gwcat.GWCatalog.to_darksirens(...), or with "
        "the versioned export pinned to the chi_eff basis: "
        "GWCatalog.export(path, spin_basis=\"chieff\") -- export() defaults to "
        "the component basis, which this loader rejects."
    )
    reexport_hint = (
        "Re-export in the chi_eff basis, e.g. "
        "gwcat ... --spin-basis chieff or GWCatalog.export(spin_basis=\"chieff\")."
    )

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "load_gw_samples requires x64; call configure_jax_runtime() (or "
            "jax.config.update('jax_enable_x64', True)) before loading. Under "
            "float32 the PE weights lose precision in the selection-integral sum."
        )

    with h5py.File(gw_path, "r") as f:
        fmt = _require_hdf5_format(
            f,
            ("gwcat-1.0", "observed-lensing-pe-1.0", "gwcat-pe-2.0",
             "gwcat-pe-2.1"),
            conversion_hint,
        )
        # gwcat-2.x gate: reject non-chi_eff spin bases BEFORE the member
        # checks so the failure names the incompatible basis rather than the
        # (legitimately absent) chi_eff-compat attrs.
        if fmt in store_contract.SPIN_BASIS_FORMATS:
            _require_chieff_spin_basis(f, gw_path, reexport_hint)
        contract = store_contract.contract_for(fmt)
        _require_hdf5_members(
            f,
            datasets=contract.datasets,
            attrs=contract.attrs,
            conversion_hint=conversion_hint,
        )
        _require_store_quality(f, contract, gw_path, conversion_hint=conversion_hint)

        nsamp = int(f.attrs["nsamp"])
        nEvents = int(f.attrs["nobs"])

        # Load arrays as NumPy first (safer for reshaping). gwcat stores dL
        # in Mpc and p_pe in the canonical (m1det, q, dL) basis.
        ra = np.array(f["ra"])
        dec = np.array(f["dec"])
        m1det = np.array(f["m1det"])
        m2det = np.array(f["m2det"])
        dL = np.array(f["dL"])
        chieff = np.array(f["chieff"])
        p_pe = np.array(f["p_pe"])
        m1source = np.array(f["m1src"])
        m2source = np.array(f["m2src"])

        _chi_in_ppe = bool(f.attrs["chi_eff_in_p_pe"])
        _chi_amax = float(f.attrs["chi_eff_amax"])
        is_mock = bool(f.attrs.get("mock_data", False))
        _pe_attrs = {
            "H0": f.attrs.get("pe_cosmology_H0", "?"),
            "Om0": f.attrs.get("pe_cosmology_Om0", "?"),
        }
        attrs = _decoded_attrs(f)
        event_names = _decoded_event_names(f)
        fit_columns = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        ) if "fit_columns" in f.attrs else _CHIEFF_FIT_COLUMNS

    # ------------------------------------------------------------
    # p_pe handling
    # ------------------------------------------------------------
    if is_mock:
        print("This is using mock data.")
    # ``chi_eff_in_p_pe`` is a required attr precisely so this decision is the
    # file's, not the loader's: mock_data used to short-circuit it, so a mock
    # declaring chi_eff_in_p_pe=False got p_pe WITHOUT the chi_eff factor while
    # the selection loader below folded the chi_eff draw density into pdraw --
    # numerator and denominator on different spin measures, which biases the
    # spin population and, through the shared normalisation, H0.  Every in-repo
    # generator writes chi_eff_in_p_pe=True, so honouring the attr is inert for
    # them and fail-closed for anything else.
    raw_columns = {
        "ra": ra,
        "dec": dec,
        "m1det": m1det,
        "m2det": m2det,
        "dL": dL,
        "chieff": chieff,
        "p_pe": np.array(p_pe),
        "m1src": m1source,
        "m2src": m2source,
    }
    if not _chi_in_ppe:
        # chi_eff not yet in p_pe — apply it now
        logp_chi = chi_eff_prior_logprob(chieff, m1source, m2source, amax=_chi_amax)
        p_pe = p_pe * np.exp(np.clip(logp_chi, -50.0, None))

    # Normalise per event so that each event's importance weights are
    # independent.  The per-event marginal likelihood is
    #   (1/nsamp) Σ_j  p_pop(θ_j) / p_pe(θ_j)
    # and dividing by the per-event sum makes the effective weights
    # dimensionless while preserving the correct relative scale within
    # each event.  Global normalisation (over nEvents*nsamp) would
    # introduce a factor of nEvents into every per-event sum, biasing
    # log μ and therefore the posterior on H0.
    p_pe = p_pe.reshape(nEvents, nsamp)
    _report_pe_weight_health(p_pe, fmt, nEvents, nsamp, f_attrs=_pe_attrs)
    p_pe = p_pe / p_pe.sum(axis=1, keepdims=True)
    p_pe = p_pe.flatten()

    return GWStore(
        format_version=fmt,
        path=str(gw_path),
        fit_columns=fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_events=nEvents,
        nsamp=nsamp,
        prior_wt=p_pe,
        event_names=event_names,
    )


def load_gw_samples(gw_path):
    """
    Load GW posterior samples from a gwcat HDF5 export (positional tuple).

    Thin wrapper over :func:`load_gw_store`; see there for the accepted
    formats and gates.  Kept byte-identical for the existing positional
    unpack sites; new consumers should prefer the record API.

    Returns
    -------
    m1det, m2det, dL, chieff, ra, dec, p_pe : jnp.ndarray
        Flattened arrays of length ``nEvents * nsamp``. ``m1det`` and
        ``m2det`` are detector-frame masses in solar masses, ``dL`` is in
        Mpc, sky angles are radians, and ``chieff`` is dimensionless.
        ``p_pe`` is the per-event-normalised PE proposal density in the
        likelihood's canonical sample basis ``(m1det, q, dL)`` with
        ``q = m2det / m1det``.
    nEvents : int
        Number of GW events.
    nsamp : int
        Number of posterior samples per event.
    """
    store = load_gw_store(gw_path)
    # Convert to jnp in requested order.  m2det is retained so
    # make_gw_event can form q, but prior_wt is already in the (m1det, q, dL)
    # proposal-density basis used by the likelihood.
    return (
        jnp.array(store.columns["m1det"]),
        jnp.array(store.columns["m2det"]),
        jnp.array(store.columns["dL"]),
        jnp.array(store.columns["chieff"]),
        jnp.array(store.columns["ra"]),
        jnp.array(store.columns["dec"]),
        jnp.array(store.prior_wt),
        store.n_events,
        store.nsamp,
    )


def load_selection_store(file) -> SelectionStore:
    """
    Load a gwcat selection export as a :class:`SelectionStore` record.

    ``file`` must point to a file with ``format_version`` in
    {"gwcat-selection-1.0", "gwcat-selection-2.0", "gwcat-selection-2.1"}
    produced by
    ``gwcat.SelectionSet.to_darksirens(...)`` /
    ``gwcat.CombinedSelectionSet.to_darksirens(...)`` or the versioned export
    layer.  A ``gwcat-selection-2.x`` file is only accepted when its
    ``spin_basis`` attr is ``"chieff"`` (the sole basis compatible with
    darksirens' chi_eff-based likelihood); the ``"component"`` and
    ``"chieff_chip"`` bases are rejected with an actionable error, and any
    extra spin datasets (``a1, a2, cost1, cost2, chip``) are ignored.  Raw LVK
    injection files and darksirens mock-selection files are intentionally
    rejected so that all catalog ingestion, FAR cuts, coordinate transforms,
    and draw density conversions happen in gwcat.

    Integration convention after gwcat preprocessing:

    - ``m1det`` and ``m2det`` are detector-frame masses in solar masses.
    - ``dL`` is luminosity distance in Mpc.
    - ``ra`` and ``dec`` are radians, and ``chieff`` is dimensionless.
    - ``SelectionStore.prior_wt`` is the physical injection proposal density
      in the likelihood's canonical basis ``(m1det, q, dL)`` with
      ``q = m2det / m1det``, chi_eff draw density included. Its absolute
      scale is retained because it enters the expected-detection integral.
    - ``ndraw`` is the total number of generated injections represented by
      the gwcat export.

    :func:`load_selection_samples` is the positional-tuple wrapper over this
    record.
    """
    conversion_hint = (
        "Use gwcat.SelectionSet.to_darksirens(...) or "
        "gwcat.CombinedSelectionSet.to_darksirens(...) to create a "
        "gwcat-selection-1.0 / 2.0 / 2.1 file."
    )
    reexport_hint = (
        "Re-export in the chi_eff basis, e.g. "
        "gwcat ... --spin-basis chieff or SelectionSet.export(spin_basis=\"chieff\")."
    )
    # ``chi_eff_swap_applied`` (required by the selection contract, not
    # defaulted to True) states whether pdraw already carries the 1-D chi_eff
    # draw density, for the same reason ``chi_eff_in_p_pe`` is required of a
    # PE file: the numerator and the denominator of the hierarchical
    # likelihood must be on the same spin measure, and a file that does not
    # say cannot be paired with one that does.  Every gwcat chieff export and
    # every in-repo mock generator stamps it.
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "load_selection_samples requires x64; call configure_jax_runtime() "
            "(or jax.config.update('jax_enable_x64', True)) before loading. Under "
            "float32 the draw weights lose precision in the selection-integral sum."
        )

    with h5py.File(file, "r") as f:
        fmt = _require_hdf5_format(
            f,
            ("gwcat-selection-1.0", "gwcat-selection-2.0",
             "gwcat-selection-2.1"),
            conversion_hint,
        )
        # gwcat-2.x gate: reject non-chi_eff spin bases before the member
        # checks (extra spin datasets a1/a2/cost1/cost2/chip are ignored).
        if fmt in store_contract.SPIN_BASIS_FORMATS:
            _require_chieff_spin_basis(f, file, reexport_hint)
        contract = store_contract.contract_for(fmt)
        _require_hdf5_members(
            f,
            datasets=contract.datasets,
            attrs=contract.attrs,
            conversion_hint=conversion_hint,
        )
        _require_store_quality(f, contract, file, conversion_hint=conversion_hint)

        m1detsels = np.array(f["m1det"])
        m2detsels = np.array(f["m2det"])
        dLsels = np.array(f["dL"])
        chieffsels = np.array(f["chieff"])
        rasels = np.array(f["ra"])
        decsels = np.array(f["dec"])
        pdraw_sel = np.array(f["pdraw"])
        m1src_sel = np.array(f["m1src"])
        m2src_sel = np.array(f["m2src"])
        ndraw = int(f.attrs["ndraw"])
        attrs = _decoded_attrs(f)
        fit_columns = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        ) if "fit_columns" in f.attrs else _CHIEFF_FIT_COLUMNS
        raw_columns = {
            "m1det": m1detsels,
            "m2det": m2detsels,
            "dL": dLsels,
            "chieff": chieffsels,
            "ra": rasels,
            "dec": decsels,
            "pdraw": np.array(pdraw_sel),
            "m1src": m1src_sel,
            "m2src": m2src_sel,
        }

        # Apply 1-D chi_eff spin-prior swap if not already done.
        if not bool(f.attrs["chi_eff_swap_applied"]):
            if "chi_eff_amax" not in f.attrs:
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied=False "
                    "but carries no chi_eff_amax, so the chi_eff draw density "
                    "this loader has to fold into pdraw is undefined. "
                    f"{conversion_hint}"
                )
            _amax = float(f.attrs["chi_eff_amax"])
            log_p_chi = chi_eff_prior_logprob(chieffsels, m1src_sel, m2src_sel, amax=_amax)
            safe_log_p_chi = np.clip(log_p_chi, a_min=-50.0, a_max=None)
            pdraw_sel = pdraw_sel * np.exp(safe_log_p_chi)

        n_det = len(pdraw_sel)
        _basis = _decode_hdf5_attr(f.attrs.get("spin_basis", "chieff"))
        print(f"    [gwcat selection] format={fmt} spin_basis={_basis}  "
              f"{n_det:,} detected injections  "
              f"Ndraw={ndraw:,}  "
              f"H0={f.attrs.get('cosmology_H0', '?')}  "
              f"Om0={f.attrs.get('cosmology_Om0', '?')}")
        print(f"    p_draw: min={pdraw_sel.min():.3e}  "
              f"max={pdraw_sel.max():.3e}  "
              f"mean={pdraw_sel.mean():.3e}")

    # The selection integral requires
    #
    #   μ = (1/N_draw) Σ_det  p_pop(d_i|λ) / p_draw(d_i)
    #
    # so p_draw must retain its physical scale (per unit volume per unit
    # mass per unit time).
    return SelectionStore(
        format_version=fmt,
        path=str(file),
        fit_columns=fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_injections=n_det,
        ndraw=ndraw,
        prior_wt=pdraw_sel,
    )


def load_selection_samples(file):
    """
    Return detected GW selection samples from a gwcat HDF5 export (tuple).

    Thin wrapper over :func:`load_selection_store`; see there for the
    accepted formats, gates, and conventions.  Kept byte-identical for the
    existing positional unpack sites; new consumers should prefer the record
    API.

    Returns
    -------
    m1detsels : jnp.ndarray
    m2detsels : jnp.ndarray
    dLsels    : jnp.ndarray
    chieffsels: jnp.ndarray
    rasels    : jnp.ndarray
    decsels   : jnp.ndarray
    pdraw_sel : jnp.ndarray
    ndraw     : int
    """
    store = load_selection_store(file)
    # Convert to jnp in requested order.  m2det is retained for q
    # construction, but prior_wt is already in the (m1det, q, dL) basis.
    return (
        jnp.array(store.columns["m1det"]),
        jnp.array(store.columns["m2det"]),
        jnp.array(store.columns["dL"]),
        jnp.array(store.columns["chieff"]),
        jnp.array(store.columns["ra"]),
        jnp.array(store.columns["dec"]),
        jnp.array(store.prior_wt),
        store.ndraw,
    )
