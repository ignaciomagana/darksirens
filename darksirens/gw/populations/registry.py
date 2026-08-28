r"""
registry.py
-----------
Population model registry for gravitational-wave source inference.

The model NAME is the model definition.  ``--pop_model`` strings are parsed by
:mod:`darksirens.gw.populations.grammar`::

    powerlaw+peak                       PowerLaw + Gaussian peak mixture
    brokenpowerlaw+2peaks+powerlaw      BPL + 2 Gaussians + PL tail
    3powerlaws+peak                     novel composition - works with NO code

Sharing of beta/pairing, spin, and gamma is controlled by separate
``shared_beta``, ``shared_spin``, and ``shared_gamma`` flags, not by suffixes
in the model name.

Components self-register grammar tokens with default labels, prior bounds, and
fiducials (see :mod:`.components`).  Curated legacy compositions keep their
physics-tuned per-slot priors and fiducial weights through the pure-data
``CURATED`` table below.  Anything else builds from blueprint defaults (an
``logging.info`` notes this).

Adding a model
--------------
* New combination of existing components: just use the name - nothing to add.
* Tuned priors/fiducials for a composition: add ONE ``Curated`` entry.
* New mass component type: decorate the class with ``@mass_token`` in
  parametric.py (or register a blueprint in components.py for lazy classes).
* Non-mixture model: decorate a factory with ``@register_model`` below.

Stick-breaking weight parameterisation
--------------------------------------
Sampled weight parameters are stick-breaking inputs ``$v_i$`` in [0, 1].
``Curated.weights`` stores DESIRED FINAL FRACTIONS (human-readable; the last
fraction is implied); ``build_fiducial_vector`` converts them via ``_w_to_v``.

Legacy names (pre-grammar registry)
-----------------------------------
``twopowerlaws+*`` spellings and the long GWTC-5 names resolve through
``LEGACY_BASE_ALIASES`` / ``LEGACY_NAME_ALIASES`` with a DeprecationWarning,
so old settings.json files and HDF5 ``pop_model`` attrs keep working.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Tuple

import jax.numpy as jnp

from .base import ParamSpec, PopulationModel
from .components import (
    _B,
    CHI_MU,
    CHI_SIGMA,
    GOL_A,
    GOL_B,
    GOL_BETA_M,
    GOL_C,
    GOL_DELTA,
    GOL_DM_TAIL,
    GOL_LOG10_FPL,
    GOL_MTR,
    GOL_SIGMA,
)
from .grammar import (
    ModelNameError,
    _stick_breaking_weights_np,  # noqa: F401  (re-exported for callers/tests)
    build_fiducial_vector,
    build_population_model,
    derive_latex,
    parse_model_name,
)
from .parametric import (
    GolombRemnantMass1G,
    GolombRemnantMass1GPlusTail,
    GolombSymmetricMassPopulationModel,
    GWTC3PowerLawPeakPopulationModel,
    GWTC5FiducialBPL2PeaksPopulationModel,
    TruncatedGaussianSpin,
)
from .utils import M_HI

_LOG = logging.getLogger(__name__)


# ============================================================
# 1. Curated mixture models (pure data - deltas from blueprint defaults)
# ============================================================

@dataclass(frozen=True)
class Curated:
    """Physics tuning for one composition, keyed by canonical name.

    ``mass``/``pairing``/``spin`` are explicit component-key lists for models
    whose name is not grammar-parseable (the GP models); grammar-parseable
    entries leave them ``None`` and only patch priors/fiducials.

    ``bounds`` / ``fids`` map mass-slot index -> {param name -> override}.
    They are written verbatim as the pre-refactor factories passed them, even
    where a value equals the blueprint default, for auditability.
    """

    latex: str
    weights: Optional[Tuple[float, ...]] = None
    bounds: Mapping[int, Mapping[str, _B]] = field(default_factory=dict)
    fids: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    mass: Optional[Tuple[str, ...]] = None
    pairing: Optional[Tuple[str, ...]] = None
    spin: Optional[Tuple[str, ...]] = None


_PL1_BOUNDS = {"alpha": _B(0, 6), "m_min": _B(2, 10), "m_max": _B(15, 50)}
_PL2_BOUNDS = {"alpha": _B(0, 6), "m_min": _B(20, 40), "m_max": _B(50, 100)}

# Weight convention, stated once: ``weights`` are the leading FINAL fractions and
# the LAST component takes the remainder, so ``weights=(0.10,)`` puts w_G = 0.90 in
# the Gaussian peak -- 90% of the fiducial population inside one 35 Msun peak,
# against a measured peak fraction of ~0.03-0.1 with the power-law continuum
# dominating.  These are round tuning values inherited from the pre-grammar
# factories, NOT release medians (only ``gwtc5_fiducial_bpl2peaks`` and
# ``gwtc3_fiducial_plpeak`` carry published values), and they set the truth of
# every ``--fix_population`` run and mock built from
# ``get_fixed_population_params``.  Left verbatim so existing fixed-population runs
# stay comparable; flip them (and the mock generator's ``peak_fraction``) together
# if the fiducial population is meant to look like the measured one.
CURATED: dict[str, Curated] = {
    # NOT a GWTC-like population, and not an approximation of one.  This entry
    # is a CURATED TEST COMPOSITION whose numbers were tuned before the grammar
    # existed; it is kept exactly as-is because it defines the truth of every
    # archived fixed-population run and every mock built from it, so changing
    # it would silently invalidate those comparisons.  Two specific ways it
    # departs from the published GWTC-3 Power Law + Peak model, both
    # deliberate and both load-bearing for anyone reading a plot made with it:
    #
    #   * w_G = 0.90.  Ninety per cent of this population sits inside the
    #     Gaussian peak.  GWTC-3 measures lambda_peak = 0.038 (+0.058, -0.026)
    #     (arXiv:2111.03634 Sec. V B) -- the real BBH mass function is a
    #     power-law continuum with a ~4% excess at ~34 Msun, i.e. very nearly
    #     the mirror image of this one.
    #   * The Gaussian peak is UNTAPERED on primary mass.  The grammar's
    #     mixture applies each component's own smoothing, so the peak here has
    #     no low-mass taper at all, where GWTC-3 Eq. B4 multiplies the WHOLE
    #     mixture (peak included) by S(m1 | m_min, delta_m).  That is a
    #     structural difference, not a parameter choice: no setting of the
    #     weights turns this composition into that model.
    #
    # Use ``gwtc3_fiducial_plpeak`` (registered below) when the population is
    # meant to be the published GWTC-3 one; use this when you want the legacy
    # curated composition and its archived comparability.
    "powerlaw+peak": Curated(
        # The display name stays "PL+G" verbatim: it is compared by the
        # grammar tests and stored in the registry golden, and the honest
        # description belongs in the comment above and in the docs table, not
        # in a label that would silently reflow every archived corner plot.
        latex="PL+G",
        weights=(0.10,),                       # w_PL=0.10, w_G=0.90
        bounds={1: {"mu": _B(20, 50)}},
    ),
    "brokenpowerlaw+2peaks": Curated(
        latex="BPL+2G",
        weights=(0.10, 0.05),                  # w_BPL=0.10, w_G1=0.05, w_G2=0.85
        bounds={1: {"mu": _B(5, 20)}, 2: {"mu": _B(25, 40)}},
        fids={1: {"mu": 10.0, "sigma": 3.0}},
    ),
    "brokenpowerlaw+3peaks": Curated(
        latex="BPL+3G",
        weights=(0.10, 0.05, 0.03),
        bounds={
            1: {"mu": _B(5, 20)},
            2: {"mu": _B(25, 40)},
            3: {"mu": _B(50, 100), "sigma": _B(1, 20)},
        },
        fids={1: {"mu": 10.0, "sigma": 3.0}, 3: {"mu": 70.0, "sigma": 10.0}},
    ),
    "brokenpowerlaw+2peaks+powerlaw": Curated(
        latex="BPL+2G+PL",
        weights=(0.10, 0.05, 0.05),
        bounds={
            1: {"mu": _B(5, 20)},
            2: {"mu": _B(25, 40)},
            3: {"alpha": _B(0, 6), "m_min": _B(40, 60), "m_max": _B(80, 120)},
        },
        fids={
            1: {"mu": 10.0, "sigma": 3.0},
            3: {"alpha": 3.0, "m_min": 50.0, "m_max": 100.0},
        },
    ),
    "2powerlaws+peak": Curated(
        latex="2PL+G",
        weights=(0.20, 0.10),
        bounds={
            0: _PL1_BOUNDS,
            1: _PL2_BOUNDS,
            2: {"mu": _B(50, 100), "sigma": _B(1, 20)},
        },
        # NOTE: inherited verbatim from the old registry - the PL m_max and
        # Gaussian mu fiducials sit OUTSIDE these priors.  Preserved so
        # fixed-population runs stay bit-identical; the corrected in-prior set
        # lives in ``_IN_PRIOR_FIDUCIAL_PATCHES`` and is selected by
        # ``fiducials=FIDUCIAL_SET_IN_PRIOR``.
    ),
    "2powerlaws+2peaks": Curated(
        latex="2PL+2G",
        weights=(0.15, 0.10, 0.05),
        bounds={
            0: _PL1_BOUNDS,
            1: _PL2_BOUNDS,
            2: {"mu": _B(5, 20), "sigma": _B(1, 10)},
            3: {"mu": _B(20, 50), "sigma": _B(1, 15)},
        },
        fids={2: {"mu": 10.0, "sigma": 3.0}},
    ),
    "2powerlaws+3peaks": Curated(
        latex="2PL+3G",
        weights=(0.15, 0.10, 0.05, 0.03),
        bounds={
            0: _PL1_BOUNDS,
            1: _PL2_BOUNDS,
            2: {"mu": _B(5, 20), "sigma": _B(1, 10)},
            3: {"mu": _B(20, 50), "sigma": _B(1, 10)},
            4: {"mu": _B(50, 100), "sigma": _B(1, 20)},
        },
        fids={2: {"mu": 10.0, "sigma": 3.0}, 4: {"mu": 70.0, "sigma": 10.0}},
    ),
}


# ── the curated fiducials that violate their own overridden priors ────────────
#
# Three curated entries inherit the component blueprints' DEFAULT fiducials
# (powerlaw m_max = 80, m_min = 5; peak mu = 35) while overriding the priors to
# split a low-mass and a high-mass power law and to push the peak above 50.
# MEASURED violations (``get_fixed_population_params``, warn path):
#
#   2powerlaws+peak    PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40];
#                      G.mu 35 vs [50, 100]
#   2powerlaws+2peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]
#   2powerlaws+3peaks  PL1.m_max 80 vs [15, 50]; PL2.m_min 5 vs [20, 40]
#
# A ``--fix_population`` run does not apply the sampling prior, so an
# out-of-prior fiducial is not by itself an invalid density.  It becomes a
# defect the moment the fixed arm is read as NESTED IN, or directly comparable
# to, the sampled model: the sampled model cannot represent its own fiducial,
# so no amount of data moves the posterior onto it and every arm-to-arm shift
# is contaminated by that gap.
#
# The corrected set below is chosen by ONE mechanical rule -- the MIDPOINT of
# the overridden prior for each violating parameter -- so it is reproducible
# and carries no new tuning.  It leaves the composition's intent intact: PL1
# becomes the low-mass continuum [5, 32.5], PL2 the high-mass one [30, 80], and
# the peak sits at 75 where its own prior puts it.
#
#: Version tag of the corrected set; stamped into settings by the CLI so a run
#: records WHICH fiducial vector it fixed the population at.  The names live in
#: the dependency-free :mod:`darksirens.core.constants` so the CLIs can build
#: their ``--population_fiducials`` option without importing this registry (and
#: the seconds of model/JAX imports behind it); they are re-exported here and
#: from ``darksirens.gw.populations``, which stay the canonical spellings.
from darksirens.core.constants import (  # noqa: E402,F401
    FIDUCIAL_SET_IN_PRIOR,
    FIDUCIAL_SET_LEGACY,
    FIDUCIAL_SETS,
)

#: Per-slot fiducial patches applied on top of ``Curated.fids`` under
#: ``fiducials=FIDUCIAL_SET_IN_PRIOR``.  Every value is the midpoint of that
#: slot's overridden prior; the in-prior build additionally runs with
#: ``on_violation="error"`` so this table can never drift back out of bounds.
_IN_PRIOR_FIDUCIAL_PATCHES: dict[str, dict[int, dict[str, float]]] = {
    "2powerlaws+peak":   {0: {"m_max": 32.5}, 1: {"m_min": 30.0},
                          2: {"mu": 75.0}},
    "2powerlaws+2peaks": {0: {"m_max": 32.5}, 1: {"m_min": 30.0}},
    "2powerlaws+3peaks": {0: {"m_max": 32.5}, 1: {"m_min": 30.0}},
}

#: Appended to the out-of-prior warning so the escape hatch is named where the
#: problem is reported.
_FIDUCIAL_HINT = (
    "This is the LEGACY curated set, kept reachable for reproducing archived "
    "fixed-population runs. It is not nested in the sampled model: the sampler "
    "can never reach this point, so a fixed-vs-sampled comparison against it is "
    "not a comparison of the same model family. Pass "
    "--population_fiducials in_prior_v2 (or fiducials=FIDUCIAL_SET_IN_PRIOR) "
    "for the corrected, in-prior set."
)


def _merge_fiducial_patch(base_fids, canonical, fiducials):
    """``Curated.fids`` with the in-prior patch layered on, if requested."""
    merged = {k: dict(v) for k, v in (base_fids or {}).items()}
    if fiducials == FIDUCIAL_SET_LEGACY:
        return merged or None
    patch = _IN_PRIOR_FIDUCIAL_PATCHES.get(canonical, {})
    for slot, values in patch.items():
        merged.setdefault(slot, {}).update(values)
    return merged or None


def _validate_fiducial_set(fiducials: str) -> str:
    if fiducials not in FIDUCIAL_SETS:
        raise ValueError(
            f"unknown fiducial set {fiducials!r}; expected one of "
            f"{list(FIDUCIAL_SETS)}"
        )
    return fiducials


# ============================================================
# 2. Legacy name aliases (clean-break migration aids)
# ============================================================

# Base-name aliases.
LEGACY_BASE_ALIASES = {
    "twopowerlaws+peak": "2powerlaws+peak",
    "twopowerlaws+2peaks": "2powerlaws+2peaks",
    "twopowerlaws+3peaks": "2powerlaws+3peaks",
}

# Full-name aliases (custom models; never suffixed).
LEGACY_NAME_ALIASES = {
    "gwtc5_fiducial_brokenpowerlaw+2peaks": "gwtc5_fiducial_bpl2peaks",
    "gwtc5_brokenpowerlaw+2peaks": "gwtc5_fiducial_bpl2peaks",
    # Same two spellings for the GWTC-3 preset: the grammar-shaped name a user
    # naturally writes, and the release-prefixed short form.  Both resolve (with
    # a DeprecationWarning) so nobody lands on the CURATED ``powerlaw+peak``
    # composition by typing what looks like the published model's name -- the
    # two are structurally different densities, not two tunings of one.
    "gwtc3_fiducial_powerlaw+peak": "gwtc3_fiducial_plpeak",
    "gwtc3_powerlaw+peak": "gwtc3_fiducial_plpeak",
}


# ============================================================
# Rate-evolution decorations: "<base>@md" selects the Madau-Dickinson-like
# peaked merger rate (gamma, kappa, z_peak) instead of the (1+z)^(gamma-1)
# power law. The decoration travels inside the pop_model STRING so every
# existing static-argument path (CLI -> factory -> jitted core) carries it
# with no signature changes.
# ============================================================

RATE_DECORATIONS = ("md",)

MD_RATE_FIDUCIALS = (2.7, 2.9, 1.9)
"""Fiducial (gamma, kappa, z_peak) for '@md' models — Madau-Dickinson-like
star-formation history values (rise ~(1+z)^2.7, fall ~(1+z)^-2.9, peak z~1.9)."""


def split_rate_decoration(pop_model: str) -> tuple[str, str]:
    """Split ``'<base>@<rate>'`` into (base model name, rate_evolution)."""
    if "@" not in pop_model:
        return pop_model, "powerlaw"
    base, _, rate = pop_model.partition("@")
    if rate not in RATE_DECORATIONS or not base:
        raise ValueError(
            f"Unknown rate-evolution decoration in pop_model {pop_model!r}; "
            f"supported: {[f'<base>@{r}' for r in RATE_DECORATIONS]}"
        )
    return base, rate


def _resolve_legacy(pop_model: str) -> str:
    if pop_model in LEGACY_NAME_ALIASES:
        target = LEGACY_NAME_ALIASES[pop_model]
    elif pop_model in LEGACY_BASE_ALIASES:
        target = LEGACY_BASE_ALIASES[pop_model]
    else:
        return pop_model
    warnings.warn(
        f"Population model name {pop_model!r} is deprecated; use {target!r}.",
        DeprecationWarning,
        stacklevel=3,
    )
    return target


# ============================================================
# 3. Custom (non-mixture) models
# ============================================================

_CUSTOM_MODEL_FACTORIES: dict[str, Callable[[], object]] = {}
_CUSTOM_FIDUCIALS: dict[str, Tuple[float, ...]] = {}
MODEL_NAME_LATEX: dict[str, str] = {"mock_data": r"\text{Mock}"}


def register_model(*names: str, latex: str, fiducial: Optional[Tuple[float, ...]] = None):
    """Register a bespoke population model factory under one or more names.

    The factory must return an object with ``param_specs``, ``prior_bounds()``,
    and ``log_p_pop(m1, q, z, chieff, theta)`` - it bypasses the mixture
    grammar entirely.
    """

    def deco(factory):
        for name in names:
            _CUSTOM_MODEL_FACTORIES[name] = factory
            MODEL_NAME_LATEX[name] = latex
            if fiducial is not None:
                _CUSTOM_FIDUCIALS[name] = fiducial
        return factory

    return deco


def _default_spin() -> TruncatedGaussianSpin:
    return TruncatedGaussianSpin(
        ParamSpec(r"$\mu_\chi$", CHI_MU.lo, CHI_MU.hi, name="mu_chi"),
        ParamSpec(r"$\sigma_\chi$", CHI_SIGMA.lo, CHI_SIGMA.hi, name="sigma_chi"),
    )


@register_model(
    "golomb_1g",
    latex=r"\text{Golomb 1G}",
    # a, b, M_tr, Delta_PPI, sigma_map, beta_M, mu_chi, sigma_chi, gamma
    fiducial=(2.35, 1.90, 35.0, 3.0, 0.08, 0.0, 0.0, 0.10, 0.0),
)
def _model_golomb_1g() -> GolombSymmetricMassPopulationModel:
    return GolombSymmetricMassPopulationModel(
        mass_component=GolombRemnantMass1G(
            ParamSpec(r"$a$", GOL_A.lo, GOL_A.hi),
            ParamSpec(r"$b$", GOL_B.lo, GOL_B.hi),
            ParamSpec(r"$M_{\rm tr}$", GOL_MTR.lo, GOL_MTR.hi),
            ParamSpec(r"$\Delta_{\rm PPI}$", GOL_DELTA.lo, GOL_DELTA.hi),
            ParamSpec(r"$\sigma_{\rm map}$", GOL_SIGMA.lo, GOL_SIGMA.hi),
        ),
        spin_component=_default_spin(),
        beta_spec=ParamSpec(r"$\beta_M$", GOL_BETA_M.lo, GOL_BETA_M.hi),
        gamma_spec=ParamSpec(r"$\gamma$", -10.0, 10.0),
    )


@register_model(
    "golomb_1g+tail",
    latex=r"\text{Golomb 1G+PL}",
    # a, b, M_tr, Delta_PPI, sigma_map, c_PL, log10_f_PL, delta_m_PL,
    # beta_M, mu_chi, sigma_chi, gamma
    fiducial=(2.35, 1.90, 35.0, 3.0, 0.08, 4.0, -1.4, 5.0, 0.0, 0.0, 0.10, 0.0),
)
def _model_golomb_tail() -> GolombSymmetricMassPopulationModel:
    return GolombSymmetricMassPopulationModel(
        mass_component=GolombRemnantMass1GPlusTail(
            ParamSpec(r"$a$", GOL_A.lo, GOL_A.hi),
            ParamSpec(r"$b$", GOL_B.lo, GOL_B.hi),
            ParamSpec(r"$M_{\rm tr}$", GOL_MTR.lo, GOL_MTR.hi),
            ParamSpec(r"$\Delta_{\rm PPI}$", GOL_DELTA.lo, GOL_DELTA.hi),
            ParamSpec(r"$\sigma_{\rm map}$", GOL_SIGMA.lo, GOL_SIGMA.hi),
            ParamSpec(r"$c_{\rm PL}$", GOL_C.lo, GOL_C.hi),
            ParamSpec(r"$\log_{10} f_{\rm PL}$", GOL_LOG10_FPL.lo, GOL_LOG10_FPL.hi),
            ParamSpec(r"$\delta m_{\rm PL}$", GOL_DM_TAIL.lo, GOL_DM_TAIL.hi),
        ),
        spin_component=_default_spin(),
        beta_spec=ParamSpec(r"$\beta_M$", GOL_BETA_M.lo, GOL_BETA_M.hi),
        gamma_spec=ParamSpec(r"$\gamma$", -10.0, 10.0),
    )


register_model(
    "gwtc5_fiducial_bpl2peaks",
    latex=r"\text{GWTC-5 Fiducial BPL+2G}",
    # Ordering follows GWTC5FiducialBPL2PeaksPopulationModel.param_specs:
    # alpha1, alpha2, mbreak, mu1, sigma1, mu2, sigma2,
    # m1_low, delta_m_1, lambda0, lambda1, beta_q, m2_low,
    # delta_m_2, mu_chi, sigma_chi, gamma.
    #
    # Marginal posterior MEDIANS from the LVK GWTC-5.0 data release, run
    # ``gwtc5_updated_default_mmax`` (mass TwoPeakBrokenPowerLawSmoothed x
    # redshift PowerLawRedshift x iid_spin_magnitude_gaussian), 8200 samples.
    # The PowerLawRedshift arm is the matching one: this model's rate law is
    # (1+z)^(gamma-1), i.e. R(z) ~ (1+z)^gamma, so ``gamma`` maps to the
    # release's ``lamb`` with no offset (base.py, "Rate evolution").
    #
    # Cross-checked against the quoted values in arXiv:2605.27226 (GWTC-5.0
    # population properties): alpha_1 1.48 [0.11, 2.52] vs 1.5 +1.0 -1.4;
    # alpha_2 5.42 vs 5.4 +1.4 -1.6; break_mass 37.45 vs 37.5 +10.9 -7.8;
    # beta_q 1.04 vs 1.0 +0.8 -0.7; gamma 2.54 vs kappa_z 2.5 +/- 0.7.
    #
    # These replace round placeholder values, five of which sat OUTSIDE the
    # release's 90% credible interval -- most consequentially gamma = 0.0
    # (a non-evolving comoving rate, 3.6 sigma from the measurement), which
    # biases any dark-siren H0 through the source redshift distribution.
    # lambda0/lambda1 are 96% anti-correlated, so the median pair implies a
    # second-peak weight 1 - l0 - l1 = 0.054 against its own marginal median
    # of 0.044 -- both well inside the simplex, and far from the old 1/3.
    fiducial=(
        1.4816, 5.4187, 37.451, 9.9109, 0.7841, 32.3273, 5.7263,
        4.4856, 3.5302, 0.4004, 0.5457, 1.0438, 3.4633,
        4.8128, 0.0633, 0.3654, 2.5439,
    ),
)(GWTC5FiducialBPL2PeaksPopulationModel)


register_model(
    "gwtc3_fiducial_plpeak",
    latex=r"\text{GWTC-3 Fiducial PL+G}",
    # Ordering follows GWTC3PowerLawPeakPopulationModel.param_specs:
    # alpha, m_min, m_max, lambda_peak, mu_m, sigma_m, delta_m,
    # beta_q, mu_chi, sigma_chi, gamma.
    #
    # The MODEL and the PRIOR BOUNDS are arXiv:2111.03634 (GWTC-3 population
    # properties) Table VI and Eqs. B4-B7 verbatim: alpha U(-4, 12),
    # beta_q U(-2, 7), m_min U(2, 10), m_max U(30, 100), lambda_peak U(0, 1),
    # mu_m U(20, 50), sigma_m U(1, 10), delta_m U(0, 10).  Table VI is a table
    # of PRIORS, so it fixes the model and its box, not a parameter point.
    #
    # The fiducial vector below is the paper's own reported posterior MEDIANS
    # wherever the paper reports one, each traceable to a sentence:
    #   alpha      = 3.5   ("alpha = 3.5 +0.6 -0.56",     Sec. V B)
    #   m_min      = 5.0   ("mmin = 5.0 +0.86 -1.7 Msun", Sec. V B)
    #   lambda_pk  = 0.038 ("lambda = 0.038 +0.058 -0.026", Sec. V B)
    #   mu_m       = 34.0  ("a Gaussian peak at 34 +2.6 -4.0 Msun", Sec. V B)
    #   delta_m    = 4.9   ("delta_m = 4.9 +3.4 -3.2 Msun", Sec. V B)
    #   beta_q     = 1.1   ("beta_q = 1.1 +1.7 -1.3",     Sec. V B)
    #   gamma      = 2.9   (the rate index kappa = 2.9 +1.7 -1.8, Fig. 13;
    #                       this model's rate law is (1+z)^(gamma-1), i.e.
    #                       R(z) ~ (1+z)^gamma, so gamma maps to kappa with no
    #                       offset -- same convention as the GWTC-5 entry.)
    #
    # TWO ENTRIES ARE NOT MEASUREMENTS, and are marked here rather than left
    # to look like the rest: the paper quotes NO posterior for the power law's
    # cut-off m_max (it reports the derived m_99% = 44 +9.2 -5.1 Msun instead)
    # and none for sigma_m ("both the mean and the standard deviation of the
    # Gaussian component are consistent with previous inferences").  Both are
    # set to the MIDPOINT of their own Table VI prior -- m_max = 65.0,
    # sigma_m = 5.5 -- as a neutral placeholder.  Do not quote either as a
    # GWTC-3 value; source them from the LVK data release if a fully measured
    # fiducial is needed.
    #
    # mu_chi/sigma_chi are the repo's default chi_eff spin fiducials.  GWTC-3's
    # Default spin model (Table XII) is a Beta magnitude distribution plus a
    # cos-tilt mixture, which this chi_eff likelihood cannot represent, so no
    # GWTC-3 spin number is claimed here (see the model's __init__).
    fiducial=(
        3.5, 5.0, 65.0, 0.038, 34.0, 5.5, 4.9,
        1.1, 0.0, 0.1, 2.9,
    ),
)(GWTC3PowerLawPeakPopulationModel)


def _component_spin_default():
    from .component_spin import default_component_spin

    return default_component_spin()


register_model(
    "gwtc3_plpeak_component_spin",
    latex=r"\text{GWTC-3 PL+G, component spin}",
    # Ordering: the 7 GWTC-3 mass parameters and beta_q as in
    # gwtc3_fiducial_plpeak above, then the 4-D component-spin block
    # (alpha_chi, beta_chi, zeta_spin, sigma_t), then gamma.
    #
    # Mass/pairing/gamma fiducials are gwtc3_fiducial_plpeak's, with the same
    # caveats (m_max = 65.0 and sigma_m = 5.5 are prior midpoints, NOT
    # measurements).  The spin entries are GWTC-3 Default-model-LIKE
    # placeholders (a Beta near mean 0.26 and a mostly-aligned tilt mixture),
    # NOT quoted release values: the paper reports the magnitude posterior in
    # (mu_chi, sigma_chi^2) space and this preset parameterises the Beta
    # directly, so source exact numbers from the LVK data release before
    # building mocks on this preset.
    #
    # This preset requires a COMPONENT-basis gwcat pair: its spin model
    # consumes GWEvent.spin = (a1, a2, cost1, cost2) and raises against a
    # chieff-basis store (basis negotiation, DS-09, refuses the pairing
    # before the likelihood is ever built).
    fiducial=(
        3.5, 5.0, 65.0, 0.038, 34.0, 5.5, 4.9,
        1.1, 1.6, 4.5, 0.75, 0.9, 2.9,
    ),
)(lambda: GWTC3PowerLawPeakPopulationModel(
    spin_component=_component_spin_default()))


# ============================================================
# 3b. Gaussian-process model family (true-GP prior, mixture-decoupled)
# ============================================================
# Registered lazily: importing ``.gp`` pulls in neither ``tinygp`` (imported
# only inside the field evaluation) nor any heavy state, so registry import
# stays light.  Each factory builds a configured GP population on demand; the
# fiducial (xi = 0, central hyperparameters) is computed from the spec list
# without constructing a kernel.
from .gp import GP_MODEL_NAMES, GP_LATEX, build_gp_model, gp_fiducial  # noqa: E402

for _gp_name in GP_MODEL_NAMES:
    register_model(
        _gp_name,
        latex=GP_LATEX[_gp_name],
        fiducial=gp_fiducial(_gp_name),
    )((lambda _n: (lambda: build_gp_model(_n)))(_gp_name))
del _gp_name


# ============================================================
# 4. Display names
# ============================================================

_SHARING_LABELS = {
    "shared_beta": r"Shared $\beta$",
    "shared_spin": "Shared Spin",
    "shared_gamma": r"Shared $\gamma$",
    "per_component_beta": r"Per-component $\beta$",
    "per_component_spin": "Per-component Spin",
    "per_component_gamma": r"Per-component $\gamma$",
}

for _base, _entry in CURATED.items():
    MODEL_NAME_LATEX[_base] = _entry.latex
for _legacy, _canon in LEGACY_BASE_ALIASES.items():
    MODEL_NAME_LATEX[_legacy] = CURATED[_canon].latex
for _legacy, _canon in LEGACY_NAME_ALIASES.items():
    MODEL_NAME_LATEX[_legacy] = MODEL_NAME_LATEX[_canon]


def _sharing_key(
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> tuple[bool, bool, bool]:
    return bool(shared_beta), bool(shared_spin), bool(shared_gamma)


def _sharing_decor(
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> str:
    if shared_beta and shared_spin and shared_gamma:
        return ""
    labels = []
    labels.append(_SHARING_LABELS["shared_beta" if shared_beta else "per_component_beta"])
    labels.append(_SHARING_LABELS["shared_spin" if shared_spin else "per_component_spin"])
    labels.append(_SHARING_LABELS["shared_gamma" if shared_gamma else "per_component_gamma"])
    return " (" + ", ".join(labels) + ")"


def _latex_name(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> str:
    try:
        base_name, rate = split_rate_decoration(pop_model)
    except ValueError:
        return pop_model
    rate_decor = r"\;[\mathrm{MD}(z)]" if rate == "md" else ""
    if base_name in MODEL_NAME_LATEX:
        base = MODEL_NAME_LATEX[base_name]
    else:
        try:
            ir = parse_model_name(base_name)
        except ModelNameError:
            return pop_model
        base = derive_latex([s.token for s in ir.slots])
    return base + rate_decor + _sharing_decor(
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )


# ============================================================
# 5. Public API
# ============================================================

_MODEL_REGISTRY: dict[tuple[str, bool, bool, bool], object] = {}


def available_models() -> list[str]:
    """Curated and custom model names (any grammar composition also works)."""
    names = {*MODEL_NAME_LATEX} - {"mock_data"}
    return sorted(names)


def _format_available_model_error(pop_model: str, cause: Exception) -> str:
    """Return a CLI-friendly message for an unrecognised population model.

    The grammar accepts infinitely many mass-component mixtures, so an exhaustive
    list can only cover explicitly registered curated/custom names.  Include
    both: the parser's token-level reason for the failure and the complete
    registry vocabulary for non-grammar models such as GP, GWTC-5, and Golomb.
    """
    registered = ", ".join(available_models())
    return (
        f"Unknown population model {pop_model!r}. {cause} "
        f"Registered population models: {registered}. "
        "Mixture grammar: combine mass tokens 'powerlaw', 'brokenpowerlaw', "
        "and 'peak' with '+', using count prefixes with plurals such as "
        "'2peaks' or '3powerlaws'."
    )


def _raise_unknown_pop_model(pop_model: str, cause: Exception):
    raise ModelNameError(_format_available_model_error(pop_model, cause)) from cause


def _build_mixture_model(
    name: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> PopulationModel:
    entry = CURATED.get(name)
    if entry is not None and entry.mass is not None:
        # Explicit composition (GP models).
        return build_population_model(
            entry.mass,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            pairing=entry.pairing,
            spin=entry.spin,
            bounds=entry.bounds,
        )

    try:
        ir = parse_model_name(name)
    except ModelNameError as exc:
        _raise_unknown_pop_model(name, exc)
    entry = CURATED.get(ir.canonical)
    tokens = [s.token for s in ir.slots]
    if entry is None:
        _LOG.info(
            "Population model %r is not curated: building ad hoc from the "
            "grammar with blueprint-default priors and fiducials.", name,
        )
        return build_population_model(
            tokens,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
        )
    return build_population_model(
        tokens,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
        bounds=entry.bounds,
    )


def get_model(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> PopulationModel:
    flags = _sharing_key(shared_beta, shared_spin, shared_gamma)
    cache_key = (pop_model, *flags)
    try:
        return _MODEL_REGISTRY[cache_key]
    except KeyError:
        pass

    base, rate = split_rate_decoration(pop_model)
    name = _resolve_legacy(base)
    resolved_name = name if rate == "powerlaw" else f"{name}@{rate}"
    resolved_key = (resolved_name, *flags)
    if name in _CUSTOM_MODEL_FACTORIES:
        model = _CUSTOM_MODEL_FACTORIES[name]()
        if rate != "powerlaw":
            if not isinstance(model, PopulationModel):
                raise NotImplementedError(
                    f"Rate decoration '@{rate}' is only supported for grammar "
                    f"mixture models; {name!r} is a bespoke registered model."
                )
            model = replace(model, rate_evolution=rate)
    else:
        model = _build_mixture_model(
            name,
            shared_beta=flags[0],
            shared_spin=flags[1],
            shared_gamma=flags[2],
        )
        if rate != "powerlaw":
            model = replace(model, rate_evolution=rate)

    _MODEL_REGISTRY[cache_key] = model
    _MODEL_REGISTRY[resolved_key] = model
    return model


def _component_m1_support_max(component) -> float:
    """Truthful m1 support of one mass component, defaulting to ``M_HI``."""
    return float(getattr(component, "m1_support_max", M_HI))


def population_m1_support_max(model) -> float:
    """Largest primary mass at which ``model`` can have non-zero support.

    Sizes the opt-in pairing normalisation grid (see
    :func:`~darksirens.gw.populations.utils.size_pairing_grid_to_support`) so it
    never clamps inside the model's support.  Handles the three model layouts in
    the registry: grammar mixtures (``model.mixture.mass_components``), the
    bespoke single-mass models (``model.mass_component`` -- e.g. the GWTC-5
    fiducial BPL+2peaks, which reaches 300 Msun, and the Golomb models), and any
    other object (GP populations), which fall back to the conventional ``M_HI``
    ceiling that over-covers their sub-100 support.
    """
    mixture = getattr(model, "mixture", None)
    mass_components = getattr(mixture, "mass_components", None)
    if mass_components:
        return max(_component_m1_support_max(c) for c in mass_components)
    mass_component = getattr(model, "mass_component", None)
    if mass_component is not None:
        return _component_m1_support_max(mass_component)
    return float(M_HI)


def get_fixed_population_params(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    fiducials: str = FIDUCIAL_SET_LEGACY,
) -> jnp.ndarray:
    """
    Return the fiducial population parameter vector for --fix_population=True.

    ``fiducials`` selects WHICH curated fiducial vector: ``"legacy"`` (the
    default, bit-identical to every archived fixed-population run and to every
    mock built from it) or ``"in_prior_v2"``, the corrected set that lies
    inside each model's own declared priors -- required whenever the fixed arm
    is to be read as nested in, or comparable to, the sampled model.  See
    ``_IN_PRIOR_FIDUCIAL_PATCHES``.

    Ordering matches PopulationModel.param_specs exactly:
        v_weights -> mass params -> pairing params -> spin params -> rate params

    Rate params are the shared gamma (power law) or, for '@md'-decorated
    models, the (gamma, kappa, z_peak) triple with MD_RATE_FIDUCIALS.

    Weight parameters are stick-breaking inputs (v_i), not direct fractions;
    curated weights are stored as human-readable fractions and converted here
    via _w_to_v.
    """
    _validate_fiducial_set(fiducials)
    base_name, rate = split_rate_decoration(pop_model)
    if rate == "md":
        if not shared_gamma:
            raise ValueError(
                "rate_evolution='md' requires shared_gamma=True."
            )
        if _resolve_legacy(base_name) in _CUSTOM_FIDUCIALS:
            raise NotImplementedError(
                f"Rate decoration '@{rate}' is only supported for grammar "
                f"mixture models; {base_name!r} is a bespoke registered model."
            )
        base_vec = _fixed_population_params_base(
            base_name,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            fiducials=fiducials,
        )
        # Replace the trailing shared gamma with the MD triple.
        return jnp.concatenate(
            [base_vec[:-1], jnp.asarray(MD_RATE_FIDUCIALS, dtype=float)]
        )
    return _fixed_population_params_base(
        base_name,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
        fiducials=fiducials,
    )


def _fixed_population_params_base(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    fiducials: str = FIDUCIAL_SET_LEGACY,
) -> jnp.ndarray:
    _validate_fiducial_set(fiducials)
    name = _resolve_legacy(pop_model)

    if name in _CUSTOM_FIDUCIALS:
        # Bespoke registered models carry a published parameter vector, not a
        # grammar composition with overridable per-slot priors, so there is
        # nothing to patch and both fiducial sets are the same numbers.
        return jnp.array(_CUSTOM_FIDUCIALS[name], dtype=float)

    # The in-prior set is validated as an ERROR, not a warning: it exists
    # precisely to be inside the priors, so a violation there is a bug in the
    # patch table rather than an inherited wart to be tolerated.
    _violation = ("warn" if fiducials == FIDUCIAL_SET_LEGACY else "error")
    _hint = _FIDUCIAL_HINT if fiducials == FIDUCIAL_SET_LEGACY else ""

    entry = CURATED.get(name)
    if entry is not None and entry.mass is not None:
        return build_fiducial_vector(
            entry.mass,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            pairing=entry.pairing,
            spin=entry.spin,
            weights=entry.weights,
            fids=_merge_fiducial_patch(entry.fids, name, fiducials),
            bounds=entry.bounds,
            on_violation=_violation,
            violation_hint=_hint,
        )

    try:
        ir = parse_model_name(name)
    except ModelNameError as exc:
        _raise_unknown_pop_model(pop_model, exc)
    entry = CURATED.get(ir.canonical)
    tokens = [s.token for s in ir.slots]
    if entry is None:
        return build_fiducial_vector(
            tokens,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            on_violation="error",
        )
    return build_fiducial_vector(
        tokens,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
        weights=entry.weights,
        fids=_merge_fiducial_patch(entry.fids, ir.canonical, fiducials),
        bounds=entry.bounds,
        on_violation=_violation,
        violation_hint=_hint,
    )


def pop_model_parser(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
):
    return get_model(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    ).log_p_pop


def pop_model_prior_parser(
    pop_model: str,
    *,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
) -> tuple[list, list, list, list, str]:
    model = get_model(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    lows, highs, labels = model.prior_bounds()
    # Per-parameter prior family (read straight off the specs; defaults to
    # "uniform").  Aligned to ``labels`` because both derive from ``param_specs``
    # in the same order.  This is the Option-A channel that lets whitened GP
    # latents declare a standard-normal prior to the sampler.
    kinds = [(s.prior_kind, s.prior_loc, s.prior_scale) for s in model.param_specs]
    return lows, highs, labels, kinds, _latex_name(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
