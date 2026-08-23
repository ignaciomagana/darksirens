import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping

# Same two knobs, same values, as core.jax_config.configure_jax_runtime (which
# also enables x64; this module must not, so it only sets the env).  The values
# are IMPORTED from there rather than repeated, so the two cannot drift; that
# module is import-side-effect free and does not import JAX at module level, so
# this stays safe to do before the JAX import below.  setdefault on both sides
# means the effective configuration is order-independent, and an explicit
# environment override still wins.
from darksirens.core.jax_config import (  # noqa: E402
    DEFAULT_XLA_ALLOCATOR,
    DEFAULT_XLA_PREALLOCATE,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_XLA_ALLOCATOR)

# NOTE (OPS-02): this module used to call multiprocessing.set_start_method(
# "spawn") at import time.  That is a PROCESS-WIDE mutation performed by a
# plain `import darksirens.gw.utils`, and it reached every executor in the host
# program -- imposing picklability and __main__-guard requirements on pools
# this package never created.  The one parallel path darksirens owns
# (redshift.lognormal_completion) already asks for its own context explicitly
# with multiprocessing.get_context("spawn"), which is the correct scope and
# needs no global default.  Anything added later that forks workers must do the
# same rather than reinstating the global.  tests/test_import_side_effects.py
# pins this.

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
    from gwcat.spin import ChiEffPrior as _ChiEffPrior
    from gwcat.spin import chi_eff_prior_logprob

    # Convention guard (GW-03/DS-06): this module treats an out-of-support
    # chi_eff as ZERO density (-inf log), matching gwcat's refusals-not-floors
    # convention.  A pre-GW-03 gwcat instead floors at -50, i.e. density
    # ~2e-22 -- in a DENOMINATOR, a weight ~1e21 above the median -- so the
    # two conventions must never be silently mixed.  ``support`` was added by
    # the same gwcat change that removed the floor.
    if not hasattr(_ChiEffPrior, "support"):
        raise ImportError(
            "the installed gwcat predates the -inf out-of-support convention "
            "(gwcat GW-03: ChiEffPrior.support + logprob returning -inf); "
            "darksirens no longer floors log-densities at -50, so mixing the "
            "two conventions would disagree about what an out-of-support "
            "sample means. Upgrade gwcat."
        )
except ModuleNotFoundError:
    def chi_eff_prior_logprob(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "gwcat is required to load gwcat PE/selection files; install the gwcat package or use precomputed prior weights"
        )

def _chi_eff_errstate():
    """numpy error scope for the gwcat chi_eff prior evaluation.

    gwcat's ``chi_eff_prior_logprob`` evaluates ``log``/``arctanh`` over the
    whole sample array including out-of-support points, where numpy reports
    ``invalid value encountered in log`` / ``in arctanh`` / ``divide by zero``.
    Those are EXPECTED here: GW-03's convention is that an out-of-support
    sample has zero density (-inf log), which is what the NaN/-inf then
    becomes, and ``_report_pe_weight_health`` counts them explicitly.

    This used to be three process-wide ``warnings.filterwarnings("ignore", ...)``
    calls executed at import time (OPS-02).  Message-matched global filters
    silence the same numpy warnings everywhere in the host program, including
    in code that has nothing to do with darksirens -- and a suppressed
    ``invalid value in log`` elsewhere is exactly the kind of thing a user
    needs to see.  ``np.errstate`` is thread-local and scoped to the block, so
    it hides only these evaluations.
    """
    return np.errstate(invalid="ignore", divide="ignore")


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


def _file_spin_basis(f):
    """The file's declared spin basis; 1.0-era files are implicitly chieff."""
    return _decode_hdf5_attr(f.attrs.get("spin_basis", "chieff")) or "chieff"


def _negotiate_spin_basis(f, path, required_fit_columns, reexport_hint):
    """Capability negotiation replacing the old hard chieff gate (DS-09).

    A pairing is valid only when the model FITS exactly the columns the
    file's density COVERS:

    * a required column the file's density does not cover (a component-spin
      model against a chieff export) cannot be fitted at all;
    * a covered column the model does not fit (a chieff model against a
      chieff_chip export) leaves that column's prior divided out of
      ``p_pe``/``pdraw`` with no population term replacing it -- silently
      wrong weights, not merely inefficiency;
    * an ADVISORY column (shipped for convenience, its density NOT in
      ``p_pe``/``pdraw`` -- the component export's ``chieff``/``chip``) may
      never be fitted.

    2.1 files declare ``fit_columns``/``advisory_columns`` explicitly; for
    older files both are implied by ``spin_basis``.  Returns the file's
    basis so callers can pick the matching store contract.
    """
    basis = _file_spin_basis(f)
    if "fit_columns" in f.attrs:
        file_fit = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        )
    else:
        file_fit = store_contract.IMPLIED_FIT_COLUMNS.get(basis)
        if file_fit is None:
            raise RuntimeError(
                f"gwcat file {path!r} declares spin_basis={basis!r}, which "
                "this darksirens does not know how to consume. "
                f"{reexport_hint}"
            )
    if "advisory_columns" in f.attrs:
        advisory = tuple(
            _decode_hdf5_attr(v)
            for v in np.atleast_1d(f.attrs["advisory_columns"])
        )
    else:
        advisory = store_contract.IMPLIED_ADVISORY_COLUMNS.get(basis, ())

    # Compare the SPIN portion only.  The mass/distance/sky block is spelled
    # differently by the two packages for the same parameterisation (gwcat
    # exports m1det/m2det/ra/dec; darksirens consumes m1det/q=m2det/m1det and
    # the sky block via its own convention) and is fixed on both sides -- the
    # store contract's presence checks already guarantee it.  The negotiated
    # degree of freedom is which SPIN columns the density covers.
    spin_universe = {"chieff", "chip"} | set(store_contract.COMPONENT_SPIN_DATASETS)
    required = tuple(c for c in required_fit_columns if c in spin_universe)
    file_fit = tuple(c for c in file_fit if c in spin_universe)
    advisory = tuple(c for c in advisory if c in spin_universe)
    fitted_advisory = sorted(set(required) & set(advisory))
    if fitted_advisory:
        raise RuntimeError(
            f"gwcat file {path!r} carries {fitted_advisory} only as ADVISORY "
            "columns: the datasets exist but their density is not in "
            "p_pe/pdraw, so they cannot be fitted. Use a model that does not "
            f"fit them, or a store whose basis covers them. {reexport_hint}"
        )
    missing = sorted(set(required) - set(file_fit))
    unmodelled = sorted(set(file_fit) - set(required))
    if missing or unmodelled:
        parts = []
        if missing:
            parts.append(
                f"the model fits {missing}, which the file's density does "
                "not cover"
            )
        if unmodelled:
            parts.append(
                f"the file's density covers {unmodelled}, which the model "
                "does not fit (their prior would be divided out with no "
                "population term replacing it)"
            )
        raise RuntimeError(
            f"gwcat file {path!r} (spin_basis={basis!r}, spin fit columns "
            f"{list(file_fit)}) cannot be paired with a model fitting spin "
            f"columns {list(required)}: " + "; ".join(parts)
            + f". {reexport_hint}"
        )
    return basis


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


def _require_valid_spin_swap(f, path, allow_invalid=False):
    """Refuse a chi_eff selection product whose spin swap is invalid.

    The chi_eff basis replaces each campaign's real spin-draw density with
    the ANALYTIC uniform-magnitude/isotropic chi_eff marginal (the "swap",
    stamped as ``chi_eff_swap_applied``).  That replacement is only exact when
    the campaign actually drew spins uniform-in-magnitude and isotropic; for a
    campaign that did not (O4ab measures isotropy_dev = 0.64), the exported
    ``pdraw`` is the wrong density by an O(1), chi_eff-dependent factor -- the
    textbook Essick & Fishbach p_draw mismatch, biasing the spin population
    and leaking into masses, rate, and H0.

    gwcat stamps the evidence per campaign as ``injected_spin_uniform_isotropic``
    (GW-24 re-exports) and, when a non-strict export applied the swap anyway,
    records ``spin_basis_assumption_violations``.  Any False campaign, or any
    recorded violation, refuses the file; ``allow_invalid=True``
    (``--allow_invalid_spin_swap``) downgrades to one loud warning for
    deliberate legacy comparisons.  Files predating the attr (the shipped
    gwcat-selection-1.0 product carries neither attr) load unchanged -- this
    gate cannot read what is not there, which is why the re-export is the
    other half of retiring the stale product.
    """
    problems = []
    if "injected_spin_uniform_isotropic" in f.attrs:
        uniform = np.atleast_1d(
            np.asarray(f.attrs["injected_spin_uniform_isotropic"])
        ).astype(bool)
        if not uniform.all():
            problems.append(
                f"injected_spin_uniform_isotropic={uniform.tolist()}: at least "
                "one campaign did not draw spins uniform-in-magnitude/"
                "isotropic, so the analytic chi_eff spin swap baked into "
                "pdraw is the wrong draw density for that campaign"
            )
    violations = _decode_hdf5_attr(
        f.attrs.get("spin_basis_assumption_violations", "")
    )
    if violations and violations not in ("[]", "{}"):
        problems.append(
            "the exporter recorded spin_basis_assumption_violations="
            f"{violations}"
        )
    if not problems:
        return
    message = (
        f"Selection file {path!r} is not a valid chi_eff-basis product: "
        + "; ".join(problems)
        + ". Re-export the campaign in the component spin basis (exact for "
        "any campaign) once basis negotiation supports it, or drop the "
        "non-uniform campaign."
    )
    if allow_invalid:
        print(f"    [!] --allow_invalid_spin_swap: {message}")
        return
    raise RuntimeError(
        message + " Pass --allow_invalid_spin_swap to load it anyway "
        "(deliberate legacy comparisons only)."
    )


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


def _require_store_layout(f, contract, path, conversion_hint=""):
    """Raise unless every store column is 1-D at one common length.

    The value gates above run per column, so a file could pass all of them
    while its columns disagreed on LENGTH -- and numpy broadcasting turns a
    length-1 column into a constant repeated over every sample rather than an
    error (review DATA-01: a singleton ``ra`` gave six PE samples six
    different HEALPix pixels off one shared right ascension, with nothing
    raised anywhere).  The campaign counts are validated FIRST so
    ``nobs * nsamp`` is a trustworthy expected length before it is used as
    one, and, on the selection side, the detected row count is known before
    ``ndraw`` is compared against it.
    """
    problems = store_contract.count_problems(f.attrs, contract)
    if not problems:
        expected = (
            store_contract.expected_pe_size(f.attrs)
            if contract.kind == "pe"
            else None
        )
        problems = store_contract.layout_problems(
            f, contract, expected_size=expected
        )
    if not problems and contract.kind == "selection":
        problems = store_contract.count_problems(
            f.attrs, contract, n_rows=store_contract.common_length(f, contract)
        )
    if problems:
        raise RuntimeError(
            f"Malformed store layout in {path!r}: "
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


def load_gw_store(gw_path, fit_columns=None) -> GWStore:
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
        # Basis negotiation (DS-09) BEFORE the member checks, so a basis
        # mismatch names the incompatible fit columns rather than the
        # (legitimately absent) chi_eff-compat attrs.  1.0-era files carry no
        # spin_basis attr and are implicitly chieff.
        required = tuple(fit_columns) if fit_columns is not None else _CHIEFF_FIT_COLUMNS
        file_basis = _negotiate_spin_basis(f, gw_path, required, reexport_hint)
        contract = store_contract.contract_for(fmt, file_basis)
        _require_hdf5_members(
            f,
            datasets=contract.datasets,
            attrs=contract.attrs,
            conversion_hint=conversion_hint,
        )
        _require_store_layout(f, contract, gw_path, conversion_hint=conversion_hint)
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
        spin_cols = {
            name: np.array(f[name])
            for name in store_contract.COMPONENT_SPIN_DATASETS
            if name in f
        }

        # chi_eff-compat attrs exist only on chieff-basis files; a component
        # export's p_pe is exact in its own basis with no chi_eff factor to
        # fold in or remove.
        if file_basis == "chieff":
            _chi_in_ppe = bool(f.attrs["chi_eff_in_p_pe"])
            _chi_amax = float(f.attrs["chi_eff_amax"])
        else:
            _chi_in_ppe = True
            _chi_amax = float("nan")
        is_mock = bool(f.attrs.get("mock_data", False))
        _pe_attrs = {
            "H0": f.attrs.get("pe_cosmology_H0", "?"),
            "Om0": f.attrs.get("pe_cosmology_Om0", "?"),
        }
        attrs = _decoded_attrs(f)
        event_names = _decoded_event_names(f)
        record_fit_columns = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        ) if "fit_columns" in f.attrs else store_contract.IMPLIED_FIT_COLUMNS[file_basis]

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
        **spin_cols,
    }
    if not _chi_in_ppe:
        # chi_eff not yet in p_pe — apply it now.  gwcat's convention (GW-03)
        # is -inf outside the prior support: such a sample gets p_pe = 0,
        # which the likelihood masks (prior_wt > 0) while it still counts in
        # n.  The old -50 floor instead handed it density 2e-22 in a
        # denominator -- a weight ~1e21 above the median.  The masked count
        # is reported by _report_pe_weight_health below.
        with _chi_eff_errstate():
            logp_chi = chi_eff_prior_logprob(chieff, m1source, m2source, amax=_chi_amax)
            p_pe = p_pe * np.exp(logp_chi)

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
        fit_columns=record_fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_events=nEvents,
        nsamp=nsamp,
        prior_wt=p_pe,
        event_names=event_names,
    )


def load_gw_samples(gw_path, fit_columns=None):
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
    store = load_gw_store(gw_path, fit_columns=fit_columns)
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


def load_selection_store(file, allow_invalid_spin_swap=False,
                         fit_columns=None) -> SelectionStore:
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
        # Basis negotiation (DS-09) before the member checks.
        required = tuple(fit_columns) if fit_columns is not None else _CHIEFF_FIT_COLUMNS
        file_basis = _negotiate_spin_basis(f, file, required, reexport_hint)
        # The spin-swap validity gate applies to PROJECTION bases only: the
        # component basis needs no swap (its flat draw factor is exact for
        # any campaign), which is precisely why it is the remedy the gate
        # names.
        if file_basis == "chieff":
            _require_valid_spin_swap(f, file, allow_invalid=allow_invalid_spin_swap)
        contract = store_contract.contract_for(fmt, file_basis)
        _require_hdf5_members(
            f,
            datasets=contract.datasets,
            attrs=contract.attrs,
            conversion_hint=conversion_hint,
        )
        _require_store_layout(f, contract, file, conversion_hint=conversion_hint)
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
        spin_cols = {
            name: np.array(f[name])
            for name in store_contract.COMPONENT_SPIN_DATASETS
            if name in f
        }
        attrs = _decoded_attrs(f)
        record_fit_columns = tuple(
            _decode_hdf5_attr(v) for v in np.atleast_1d(f.attrs["fit_columns"])
        ) if "fit_columns" in f.attrs else store_contract.IMPLIED_FIT_COLUMNS[file_basis]
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
            **spin_cols,
        }

        # Apply 1-D chi_eff spin-prior swap if not already done.  The attr
        # states TRUTHFULLY whether pdraw carries the 1-D chi_eff marginal
        # (gwcat GW-27): a chieff export stamps True, a component export
        # stamps False -- its pdraw carries the campaign's own 4-D spin
        # density, no swap exists to apply, and nothing must be folded in
        # here.  A component file claiming True is contradictory (a 1-D
        # marginal folded into a 4-D-basis density) and refused.
        if file_basis != "chieff":
            if bool(f.attrs["chi_eff_swap_applied"]):
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied="
                    f"True in the {file_basis!r} basis, where the 1-D chi_eff "
                    "swap is undefined; the export is malformed."
                )
        elif not bool(f.attrs["chi_eff_swap_applied"]):
            if "chi_eff_amax" not in f.attrs:
                raise RuntimeError(
                    f"Selection file {file!r} declares chi_eff_swap_applied=False "
                    "but carries no chi_eff_amax, so the chi_eff draw density "
                    "this loader has to fold into pdraw is undefined. "
                    f"{conversion_hint}"
                )
            _amax = float(f.attrs["chi_eff_amax"])
            with _chi_eff_errstate():
                log_p_chi = chi_eff_prior_logprob(
                    chieffsels, m1src_sel, m2src_sel, amax=_amax)
            # GW-03 convention: -inf means zero density, and a DETECTED
            # injection with zero draw density must not be floored or
            # silently dropped -- Ndraw is the campaign's fixed total, so
            # excluding one biases mu low while flooring hands it a weight
            # ~1e21 above the median.  Refuse instead: the injection lies
            # outside the analytic chi_eff prior this swap assumes, so the
            # swap itself is invalid for this file.
            n_out = int((~np.isfinite(log_p_chi)).sum())
            if n_out:
                raise RuntimeError(
                    f"Selection file {file!r}: {n_out} detected injection(s) "
                    f"fall outside the chi_eff prior support (amax={_amax}) "
                    "that chi_eff_swap_applied=False asks this loader to fold "
                    "into pdraw. The swap is invalid for this file; re-export "
                    "with the swap applied by gwcat, or with a basis that "
                    "does not assume it."
                )
            pdraw_sel = pdraw_sel * np.exp(log_p_chi)

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
        fit_columns=record_fit_columns,
        columns=raw_columns,
        attrs=attrs,
        n_injections=n_det,
        ndraw=ndraw,
        prior_wt=pdraw_sel,
    )


def load_selection_samples(file, allow_invalid_spin_swap=False,
                           fit_columns=None):
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
    store = load_selection_store(
        file, allow_invalid_spin_swap=allow_invalid_spin_swap,
        fit_columns=fit_columns,
    )
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
