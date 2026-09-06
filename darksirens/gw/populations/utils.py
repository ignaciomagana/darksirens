from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import ensure_compile_time_eval, jit

# ======================================================================
# Configurable normalisation grids
# ======================================================================

M_LO: float = 1.0
M_HI: float = 200.0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 2:
        raise ValueError(f"{name} must be >= 2, got {parsed}")
    return parsed


def _env_int_opt(name: str, default: int | None) -> int | None:
    """Optional int env var: unset -> ``default``; ``"0"`` or ``"none"`` -> None."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in ("0", "none", "off", ""):
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 2:
        raise ValueError(f"{name} must be >= 2, got {parsed}")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if not parsed > 0.0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _min_pairing_m1_grid(m_lo: float, pairing_m_hi: float, n_q: int) -> int:
    """Smallest pairing m1-grid size whose cells are narrower than one q-interval.

    The opt-in grid's support-edge cell is guarded by a SINGLE q-interval
    trapezoid term (``PairingModel.__call__``).  That guard is only close to the
    true normaliser while the q-support inside the edge cell,
    ``1 - m_min/m1 < 1 - exp(-dlog m1)``, fits in one q-interval, i.e.

        log(pairing_m_hi / m_lo) / (N_grid - 1)  <=  1 / (n_q - 1).

    HISTORICAL: this floor existed only to keep the old fixed-q-grid edge clamp
    (removed 2026-08-28) inside its factor-of-two bound.  The per-sample
    Gauss-Legendre edge rule that replaced it is accurate for ANY relative sizing
    of the two grids, so the floor is now only a mild over-resolution of the m1
    grid; it is kept so that an existing ``--pairing_norm_grid`` request keeps
    resolving the same nodes.

    Nothing enforced that coupling, so a small ``--pairing_norm_grid`` (or a
    FINER ``--norm_nq``, which naively should be more accurate) silently
    reintroduced the blow-up the guard exists to prevent: measured at N_grid = 64
    with the default n_q = 200, ``PowerLawPairing`` densities inside the first
    supported cell were 6.1e8 x (+20 nats) above a 2e6-node reference.  Settings
    therefore RAISE the node count to this floor -- the same auto-sizing policy
    as :func:`size_pairing_grid_to_support`, and cheap (1024 -> 1056 at the
    defaults, O(N_grid * n_q) work per proposal).
    """
    return 1 + int(math.ceil(math.log(pairing_m_hi / m_lo) * (int(n_q) - 1)))


@dataclass(frozen=True)
class NormalizationGridSettings:
    """Settings controlling GW-population normalisation quadrature grids.

    The mass, mass-ratio, and spin normalisations are evaluated on cached
    trapezoid grids.  Values can be configured with ``configure_normalization_grids``
    or the environment variables ``DARKSIRENS_GW_N_MASS``,
    ``DARKSIRENS_GW_N_Q``, and ``DARKSIRENS_GW_N_CHI``.

    ``pairing_m1_grid`` is an OPT-IN accuracy knob (default None = exact; env
    ``DARKSIRENS_GW_PAIRING_M1_GRID``, an int to enable, ``0`` or ``none`` to
    explicitly disable): when set to an int N the pairing model's
    per-sample q-normalisation is interpolated from an N-node static m1 grid
    instead of integrated exactly per sample (see ``get_pairing_m1_grid`` and
    ``PairingModel.__call__``).  Samples the interpolant cannot resolve -- those
    near the normaliser's SUPPORT EDGE, where I(m1) ~ eps exp(-dm/eps) has an
    essential singularity -- are not interpolated at all but integrated on their
    OWN support with the Gauss-Legendre rule below.

    ``pairing_edge_nq`` (env ``DARKSIRENS_GW_PAIRING_EDGE_NQ``) is the node count
    PER PANEL of that per-sample edge rule and ``pairing_edge_tol`` (env
    ``DARKSIRENS_GW_PAIRING_EDGE_TOL``) the interpolation-error budget, in nats,
    that decides which samples take it: a grid node is TRUSTED when the second
    difference of log I over 8 -- the linear-interpolation error bound on a
    uniform-in-log-m1 grid -- is within ``pairing_edge_tol`` and no zero-support
    node lies within one cell.  Since 2026-09-05 the edge rule is the SAME
    two-panel Gauss-Legendre split as the exact branch (see
    ``get_pairing_panel_quadrature`` and ``PairingModel._panel_norm``), just at
    ``pairing_edge_nq`` nodes per panel instead of ``PAIRING_PANEL_NQ``.  Measured
    over 50 (m_min, dm_min, beta) prior draws, worst |Delta log density| against a
    composite-Gauss-Legendre reference (256+32 sub-panels, self-converged to
    2e-16) in the 24 grid cells above the support edge: 22.6 nats with the
    pre-2026-08-28 fixed-q-grid clamp, 2.6e-3 with ``pairing_edge_nq = 24`` --
    against 7.6e-3 for the EXACT branch itself at its 16 nodes per panel (3.1e-2
    for the 200-node uniform q-trapezoid the panel split replaced, which did not
    resolve the taper boundary layer at all).  KEEP ``pairing_edge_nq`` ABOVE
    ``PAIRING_PANEL_NQ``: it is what makes "the grid branch is never worse than
    the exact branch it approximates"
    (tests/test_pairing_edge_fix.py) true by construction -- more nodes on an
    identical panel split -- rather than by calibration; at 8 nodes per panel the
    rule degrades to 6.2e-1 nats.  Raising it costs ``2 * pairing_edge_nq``
    ``_eval_unnorm`` evaluations per sample (against ``2 * PAIRING_PANEL_NQ`` for
    the exact branch).  Lowering ``pairing_edge_tol`` routes more samples onto
    that rule, which since the panel split is no longer a trade: measured at
    beta = -2 with a narrow taper over the bulk of the support, the rule alone is
    1.8e-6 nats against 3.2e-4 for the interpolant (whose residual there is not
    interpolation at all but the 16-node panel rule its grid nodes are built on --
    the two agree to every printed digit).  It used to be the other way round
    (0.19 nats for the single-panel rule against 2e-5 for the interpolant); that
    warning belonged to the pre-2026-09-05 rule and no longer applies.

    ``pairing_m_hi`` is the UPPER bound of that opt-in pairing m1 grid.  It is a
    SEPARATE knob from the mass-grid ceiling ``m_hi`` precisely so that sizing
    the pairing grid to a mass model whose primary-mass support extends past
    ``m_hi`` (e.g. ``gwtc5_fiducial_bpl2peaks`` with a fixed high-mass edge at
    300 Msun) never perturbs the mass-normalisation grid of any other model.
    ``jnp.interp`` clamps queries outside the grid ends, so a ceiling BELOW a
    model's support would silently bias the pairing normaliser inside the
    support; :func:`size_pairing_grid_to_support` /
    :func:`assert_pairing_grid_covers_support` keep it truthful.
    """

    n_mass: int = _env_int("DARKSIRENS_GW_N_MASS", 500)
    n_q: int = _env_int("DARKSIRENS_GW_N_Q", 200)
    n_chi: int = _env_int("DARKSIRENS_GW_N_CHI", 200)
    pairing_m1_grid: int | None = _env_int_opt("DARKSIRENS_GW_PAIRING_M1_GRID", None)
    pairing_edge_nq: int = _env_int("DARKSIRENS_GW_PAIRING_EDGE_NQ", 24)
    pairing_edge_tol: float = _env_float("DARKSIRENS_GW_PAIRING_EDGE_TOL", 1.0e-4)
    m_lo: float = M_LO
    m_hi: float = M_HI
    pairing_m_hi: float = M_HI
    q_lo: float = 0.0
    q_hi: float = 1.0
    chi_lo: float = -1.0
    chi_hi: float = 1.0

    def __post_init__(self):
        for name in ("n_mass", "n_q", "n_chi"):
            value = int(getattr(self, name))
            if value < 2:
                raise ValueError(f"{name} must be >= 2, got {value}")
            object.__setattr__(self, name, value)
        # None = exact per-sample q-integration; an int is the m1-grid node count.
        if self.pairing_m1_grid is not None:
            pg = int(self.pairing_m1_grid)
            if pg < 2:
                raise ValueError(f"pairing_m1_grid must be >= 2 or None, got {pg}")
            object.__setattr__(self, "pairing_m1_grid", pg)
        pe = int(self.pairing_edge_nq)
        if pe < 2:
            raise ValueError(f"pairing_edge_nq must be >= 2, got {pe}")
        object.__setattr__(self, "pairing_edge_nq", pe)
        pt = float(self.pairing_edge_tol)
        if not pt > 0.0:
            raise ValueError(f"pairing_edge_tol must be > 0, got {pt}")
        object.__setattr__(self, "pairing_edge_tol", pt)
        for lo_name, hi_name in (("m_lo", "m_hi"), ("q_lo", "q_hi"), ("chi_lo", "chi_hi")):
            lo = float(getattr(self, lo_name))
            hi = float(getattr(self, hi_name))
            if not hi > lo:
                raise ValueError(f"{hi_name} must be greater than {lo_name}: {lo} >= {hi}")
            object.__setattr__(self, lo_name, lo)
            object.__setattr__(self, hi_name, hi)
        pairing_m_hi = float(self.pairing_m_hi)
        if not pairing_m_hi > self.m_lo:
            raise ValueError(
                f"pairing_m_hi must be greater than m_lo: {self.m_lo} >= {pairing_m_hi}"
            )
        object.__setattr__(self, "pairing_m_hi", pairing_m_hi)
        if self.pairing_m1_grid is not None:
            object.__setattr__(self, "pairing_m1_grid",
                               max(int(self.pairing_m1_grid),
                                   _min_pairing_m1_grid(self.m_lo, pairing_m_hi,
                                                        self.n_q)))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


_NORMALIZATION_GRID_SETTINGS = NormalizationGridSettings()


def normalization_grid_settings() -> NormalizationGridSettings:
    """Return the active GW-population normalisation-grid settings."""

    return _NORMALIZATION_GRID_SETTINGS


_SENTINEL = object()


def configure_normalization_grids(
    *,
    n_mass: int | None = None,
    n_q: int | None = None,
    n_chi: int | None = None,
    pairing_m1_grid: int | None | object = _SENTINEL,
    pairing_m_hi: float | None = None,
    pairing_edge_nq: int | None = None,
    pairing_edge_tol: float | None = None,
) -> NormalizationGridSettings:
    """Update cached normalisation-grid sizes and clear derived grids.

    Every argument follows the same convention: omitting it (or the default
    sentinel) leaves the current setting untouched; a value overrides it.
    To ENABLE the pairing m1-grid pass an int for ``pairing_m1_grid``.  To
    explicitly DISABLE it (revert to exact per-sample q-normalisation), pass
    ``pairing_m1_grid=None``.  ``pairing_m_hi`` raises the opt-in pairing
    grid's upper bound (see :func:`size_pairing_grid_to_support`); callers
    should normally use that helper rather than setting the bound directly.
    """

    global _NORMALIZATION_GRID_SETTINGS, N_MASS, N_Q, N_CHI

    updates = {}
    for key, value in {"n_mass": n_mass, "n_q": n_q, "n_chi": n_chi,
                        "pairing_m_hi": pairing_m_hi,
                        "pairing_edge_nq": pairing_edge_nq,
                        "pairing_edge_tol": pairing_edge_tol}.items():
        if value is not None:
            updates[key] = value
    if pairing_m1_grid is not _SENTINEL:
        updates["pairing_m1_grid"] = pairing_m1_grid
    if updates:
        _NORMALIZATION_GRID_SETTINGS = replace(_NORMALIZATION_GRID_SETTINGS, **updates)
        _clear_grid_caches()

    # Backward-compatible scalar aliases for callers that inspect the values
    # after configuration.  Population code should use get_*_grid() directly.
    N_MASS = _NORMALIZATION_GRID_SETTINGS.n_mass
    N_Q = _NORMALIZATION_GRID_SETTINGS.n_q
    N_CHI = _NORMALIZATION_GRID_SETTINGS.n_chi
    return _NORMALIZATION_GRID_SETTINGS


def size_pairing_grid_to_support(m1_support_max: float) -> NormalizationGridSettings:
    """Grow the opt-in pairing m1 grid so it covers a mass model's m1 support.

    No-op when the pairing grid is disabled (``pairing_m1_grid is None``) or the
    current ``pairing_m_hi`` already covers ``m1_support_max``.  Otherwise the
    upper bound is raised to ``m1_support_max`` and the node COUNT is scaled up
    in proportion to the added log-range, so log spacing is preserved and the
    per-node density (hence the documented interpolation accuracy near the
    ``m_min`` log-kink) does NOT drop as the range grows.  The default ceiling
    is never lowered, so models whose support ends at or below ``M_HI`` keep the
    historical ``[m_lo, M_HI]`` grid bit-for-bit.
    """
    s = normalization_grid_settings()
    if s.pairing_m1_grid is None:
        return s
    new_hi = max(s.pairing_m_hi, float(m1_support_max))
    if new_hi <= s.pairing_m_hi:
        return s
    old_span = math.log(s.pairing_m_hi / s.m_lo)
    new_span = math.log(new_hi / s.m_lo)
    new_n = int(math.ceil(s.pairing_m1_grid * new_span / old_span))
    return configure_normalization_grids(pairing_m1_grid=new_n, pairing_m_hi=new_hi)


def assert_pairing_grid_covers_support(
    m1_support_max: float, *, model_name: str = ""
) -> None:
    """Fail loudly if the opt-in pairing grid does not cover an m1 support.

    Defense in depth against a silent ``jnp.interp`` clamp INSIDE the mass
    model's support: with the grid enabled, any sample at
    ``pairing_m_hi < m1 <= m1_support_max`` would reuse the endpoint normaliser
    (measured up to ~0.4 in log density for ``gwtc5_fiducial_bpl2peaks`` at
    m1=299 with a 200-node ceiling).  Raises ``ValueError`` so CLI callers can
    surface it through their ``_fatal`` path.  No-op when the grid is disabled.
    """
    s = normalization_grid_settings()
    if s.pairing_m1_grid is None:
        return
    support = float(m1_support_max)
    # Tiny tolerance: a bound sized exactly to the support is fine.
    if s.pairing_m_hi + 1.0e-6 < support:
        who = f" of population model {model_name!r}" if model_name else ""
        raise ValueError(
            f"pairing-normalisation grid upper bound pairing_m_hi={s.pairing_m_hi} "
            f"does not cover the primary-mass support{who} "
            f"(m1_support_max={support}): samples with "
            f"{s.pairing_m_hi} < m1 <= {support} would be clamped to the grid "
            "endpoint by jnp.interp, silently biasing the pairing normaliser "
            "inside the model support. Let the CLI size the grid "
            "(size_pairing_grid_to_support) or disable --pairing_norm_grid "
            "(exact per-sample q-normalisation)."
        )


# ── Trace-safe cached grids ──────────────────────────────────────────────────
# The grid builders below are ``lru_cache``d and compute with ``jnp`` ops.  If
# the FIRST call after a cache clear (``configure_normalization_grids`` clears
# them) happens INSIDE a jit trace -- as it does for the lazily-normalising
# ``gwtc5_fiducial_bpl2peaks`` model, whose ``_norm`` first touches
# ``get_mass_grid`` inside the selection scan -- a naive ``jnp.linspace`` would
# cache a ``DynamicJaxprTracer`` that then leaks into the next trace
# (``jax.errors.UnexpectedTracerError``).  ``ensure_compile_time_eval`` forces
# each builder to evaluate to a CONCRETE array (bit-identical to the plain
# ``jnp`` result, x64 preserved), so the cached value is a trace-independent
# constant that is safe to reuse across traces.
@lru_cache(maxsize=8)
def _linspace(lo: float, hi: float, n: int):
    with ensure_compile_time_eval():
        return jnp.linspace(lo, hi, int(n))


def _clear_grid_caches() -> None:
    _linspace.cache_clear()
    get_mass_grid.cache_clear()
    get_q_grid.cache_clear()
    get_chi_grid.cache_clear()
    get_m1_q_mesh.cache_clear()
    get_pairing_m1_grid.cache_clear()
    get_pairing_edge_quadrature.cache_clear()


@lru_cache(maxsize=1)
def get_mass_grid():
    s = normalization_grid_settings()
    return _linspace(s.m_lo, s.m_hi, s.n_mass)


@lru_cache(maxsize=1)
def get_q_grid():
    s = normalization_grid_settings()
    return _linspace(s.q_lo, s.q_hi, s.n_q)


@lru_cache(maxsize=1)
def get_chi_grid():
    s = normalization_grid_settings()
    return _linspace(s.chi_lo, s.chi_hi, s.n_chi)


@lru_cache(maxsize=1)
def get_m1_q_mesh():
    # ensure_compile_time_eval so the meshgrid is concrete even when this cache
    # is first filled inside a jit trace (see _linspace note).
    with ensure_compile_time_eval():
        return jnp.meshgrid(get_mass_grid(), get_q_grid(), indexing="ij")


@lru_cache(maxsize=1)
def get_pairing_m1_grid():
    """Static LOG-spaced m1 grid for the opt-in pairing q-normalisation.

    Spans ``[m_lo, pairing_m_hi]``; log spacing clusters nodes at low m1 where
    the pairing normaliser I(m1) turns on fastest.  Callers evaluate I on these
    nodes once per proposal and interpolate per sample; ``jnp.interp`` CLAMPS m1
    outside the bounds to the grid ends.  Clamping is harmless ONLY where the
    mass model has no support (p(m1)=0 makes the normaliser there irrelevant),
    so ``pairing_m_hi`` MUST cover the selected mass model's primary-mass
    support -- otherwise a valid sample inside the support silently uses the
    endpoint normaliser.  The CLI sizes ``pairing_m_hi`` to the model via
    :func:`size_pairing_grid_to_support`; :func:`assert_pairing_grid_covers_support`
    guards against an undersized grid.  Only meaningful when ``pairing_m1_grid``
    is configured (not None).
    """
    s = normalization_grid_settings()
    n = s.pairing_m1_grid
    if n is None:
        raise ValueError(
            "get_pairing_m1_grid requires pairing_m1_grid to be configured; "
            "it is None (exact per-sample q-normalisation)."
        )
    # ensure_compile_time_eval so a cold-cache first eval inside a jit trace
    # yields a concrete constant, not a leaking tracer (see _linspace note).
    with ensure_compile_time_eval():
        return jnp.exp(jnp.linspace(jnp.log(s.m_lo), jnp.log(s.pairing_m_hi), int(n)))


@lru_cache(maxsize=4)
def _gauss_legendre_01(n: int):
    x, w = np.polynomial.legendre.leggauss(int(n))
    # ensure_compile_time_eval for the same reason as _linspace: a cold-cache
    # first evaluation inside a jit trace must yield CONCRETE constants, not
    # tracers that leak into the next trace.
    with ensure_compile_time_eval():
        return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


# Node count of EACH of the two Gauss-Legendre panels the pairing normaliser is
# integrated on (see ``PairingModel._panel_norm``).  Deliberately a module
# constant and NOT a setting: the rule is calibrated as a pair (16 + 16) against
# a converged reference, it is exercised on every likelihood call, and a knob
# here would let a run silently pick a rule nobody measured.  ``pairing_edge_nq``
# remains configurable because it only serves the opt-in grid branch's rare
# near-edge samples.
PAIRING_PANEL_NQ: int = 16


def get_pairing_panel_quadrature():
    """Gauss-Legendre nodes/weights on (0, 1) for ONE pairing normaliser panel.

    ``PairingModel.__call__`` integrates ``N(m1) = int p(q|m1) dq`` over the
    support ``(q_cut, 1]`` as TWO panels split at the taper shoulder
    ``q_a = m_shoulder/m1``: the integrand is a Planck-taper boundary layer below
    ``q_a`` and the bare (analytic) pairing kernel above it, so a single smooth
    rule has to resolve a corner that neither side actually has.  Splitting there
    gives Gauss-Legendre two analytic pieces and lets 2 x 16 nodes beat the
    200-node uniform trapezoid this replaced by 4x on the worst case and by
    100-170x on the H0-CORRELATED part of the residual, which is the statistic
    the cosmology actually sees (measured; see ``PairingModel._panel_nodes`` and
    the module note on ``pairing_edge_tol``).

    16 is a calibrated pair, not a default to tune: 32 per panel buys another 4x
    on the worst case (1.9e-3 nats) for twice the per-sample work, while the
    residual at 16 already sits ~50x under the 0.05-nat H0-tilt budget.

    The node set is static -- it depends on no setting -- so the quadrature is a
    compile-time constant and a proposal never retraces on it.
    """
    return _gauss_legendre_01(PAIRING_PANEL_NQ)


@lru_cache(maxsize=1)
def get_pairing_edge_quadrature():
    """Gauss-Legendre nodes/weights on (0, 1) for ONE panel of the edge rule.

    ``PairingModel.__call__``'s grid branch integrates I(m1) on the SAMPLE'S OWN
    support wherever the m1 interpolant is not trustworthy.  Since 2026-09-05 it
    does so with the SAME two-panel split as the exact branch
    (``PairingModel._panel_norm``), only at ``pairing_edge_nq`` nodes per panel
    instead of ``PAIRING_PANEL_NQ`` -- which is what makes the grid branch
    provably never worse than the branch it approximates, the invariant
    tests/test_pairing_edge_fix.py asserts.

    The integrand below the shoulder is the taper boundary layer
    ``S(m_min + t (m1 - m_min))``, of width ``A^-1 = (m1 - m_min)/dm_min`` in
    ``t``; Gauss-Legendre resolves a layer of width ``1/A`` with ~``sqrt(A)``
    nodes where a uniform rule needs ~``A`` of them, which is what makes a
    per-sample quadrature affordable at all (measured, worst |Delta log I| over
    the near-edge region across corner hyperparameters: 1.3e-3 for 16 GL nodes
    against 1.4 for 16 uniform-trapezoid nodes).  The nodes are open, so the
    exact support edge -- where the Planck taper is identically zero and its
    derivatives are singular -- is never evaluated.
    """
    s = normalization_grid_settings()
    return _gauss_legendre_01(s.pairing_edge_nq)


# Backward-compatible aliases.  They reflect import-time/default settings;
# configure_normalization_grids updates scalar aliases and code should prefer
# get_mass_grid(), get_q_grid(), and get_chi_grid() for live grids.
N_MASS: int = _NORMALIZATION_GRID_SETTINGS.n_mass
N_Q: int = _NORMALIZATION_GRID_SETTINGS.n_q
N_CHI: int = _NORMALIZATION_GRID_SETTINGS.n_chi
MASS_GRID = get_mass_grid()
Q_GRID = get_q_grid()
CHI_GRID = get_chi_grid()
# Indexing instead of tuple-unpacking keeps this importable when jax is
# replaced by an autodoc mock during documentation builds.
_M1_Q_MESH = get_m1_q_mesh()
M1_MESH = _M1_Q_MESH[0]
Q_MESH = _M1_Q_MESH[1]

# ======================================================================
# Smooth edge filters
# ======================================================================
def _sfilter_expo_cap(x):
    # Bound the taper's dynamic range so that NO downstream product, square,
    # or reciprocal of S (or of q-integrals of S) can overflow in forward or
    # reverse mode, under ANY XLA fusion order. The historical cap of 500
    # left S ~ exp(-500) ~ 7e-218 in the taper toe: well-conditioned RATIOS
    # of such values are fine op-by-op, but compiled backward passes fuse
    # them into transient 0 * inf = NaN (measured: one bright-mock injection
    # at m1src = m_min + 0.01 poisoned the full hyperparameter gradient, and
    # only under jit). exp(-130) is physically indistinguishable from
    # exp(-500) — both vanish against any O(1) density in every logsumexp —
    # while exp(+-130)^2 stays ~180 orders of magnitude inside float64.
    # The dtype-aware minimum keeps float32 safe too (exp caps near 88).
    return jnp.minimum(130.0, 0.9 * jnp.log(jnp.finfo(jnp.result_type(x)).max))


# Floor for the taper's division denominators: a sample or q-grid node can
# land within ~1e-300 of the window edge (measured: 1 in 105,170 bright-mock
# injections), making dm/delta OVERFLOW to inf before the clip — and although
# the clip's select zeroes the cotangent, mul's VJP multiplies by the stored
# inf, so the whole gradient goes NaN. At/above the floor the filter is
# bit-identical; below it the filter is saturated anyway (expo >> cap).
# 1e-140 (not 1e-290): at exactly delta == 0 the floored value must keep
# dm/delta^2 FINITE in the clip's backward pass (0 * inf = NaN at the exact
# window edge, e.g. round fiducial edges meeting round mock masses); both
# regimes clip to the cap, so the forward value is unchanged.
_SFILTER_DELTA_FLOOR = 1e-140


def _floor_signed(x):
    """Push |x| out of the overflow-producing range, preserving sign."""
    mag = jnp.maximum(jnp.abs(x), _SFILTER_DELTA_FLOOR)
    return jnp.where(x < 0, -mag, mag)


@jit
def sfilter_low(m, m_min, dm):
    delta = m - m_min
    safe_d = jnp.where(delta > 0, jnp.maximum(delta, _SFILTER_DELTA_FLOOR), 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    cap = _sfilter_expo_cap(safe_d)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / _floor_signed(safe_d - safe_dm),
        -cap, cap
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m <= m_min, 0.0, S)
    S = jnp.where(m >= m_min + dm, 1.0, S)
    return S

@jit
def sfilter_high(m, m_max, dm):
    delta = m_max - m
    safe_d = jnp.where(delta > 0, jnp.maximum(delta, _SFILTER_DELTA_FLOOR), 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    cap = _sfilter_expo_cap(safe_d)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / _floor_signed(safe_d - safe_dm),
        -cap, cap
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m >= m_max, 0.0, S)
    S = jnp.where(m <= m_max - dm, 1.0, S)
    return S
