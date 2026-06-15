"""Declarative component blueprints for population models.

This module is the single lookup table of the population registry: every
component class declares ONCE — via a blueprint — its parameters' ASCII names,
base LaTeX labels, default prior bounds, and default fiducial values.  The
grammar (:mod:`darksirens.gw.populations.grammar`) composes these blueprints
into mixtures, so no per-model factory functions are needed.

Import constraints
------------------
This module must import neither JAX-heavy code nor :mod:`.gp` at module import
time.  GP component classes are referenced by dotted string
(``"package.module:ClassName"``) and resolved lazily when a GP model is built,
so importing the registry never imports ``tinygp`` (some tests stub it out).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, NamedTuple, Optional, Tuple, Union


# ============================================================
# Prior bound constants (moved verbatim from the old registry.py)
# ============================================================

class _B(NamedTuple):
    lo: float
    hi: float

ALPHA        = _B(-4.0,   6.0)
ALPHA_BPL    = _B( 0.0,   6.0)
M_BREAK      = _B(20.0,  50.0)
M_MIN        = _B( 2.0,  10.0)
M_MAX        = _B(50.0, 100.0)
DM_MIN       = _B( 0.01, 10.0)
DM_MAX       = _B( 0.01, 20.0)
GAUSS_MU     = _B( 5.0,  50.0)
GAUSS_SIGMA  = _B( 1.0,  10.0)
BETA         = _B(-2.0,   7.0)
CHI_MU       = _B(-1.0,   1.0)
CHI_SIGMA    = _B( 0.01,  1.0)
GP_AMP       = _B( 0.01,  5.0)
GP_LS        = _B( 0.05,  1.0)
GP_LS_M_PAIR = _B( 0.5,   3.0)
GP_LS_Q_PAIR = _B( 0.2,   1.0)
GP_ALPHA     = _B(-4.0,  12.0)
GP_BETA_Q    = _B(-4.0,  12.0)
GP_Y         = _B(-7.0,   7.0)
GP_Y_PAIR    = _B(-5.0,   5.0)

# Golomb--Isi--Farr physical remnant-map model.
GOL_A = _B(-1.65, 6.35)
GOL_B = _B(-2.10, 5.90)
GOL_MTR = _B(20.0, 50.0)
GOL_DELTA = _B(0.5, 7.0)      # M_BH,max - M_tr
GOL_SIGMA = _B(0.05, 0.50)
GOL_C = _B(0.0, 8.0)
GOL_LOG10_FPL = _B(-3.0, 0.0) # f_PL in [1e-3, 1]
GOL_DM_TAIL = _B(0.1, 20.0)
GOL_BETA_M = _B(-6.0, 6.0)


# ============================================================
# Blueprints
# ============================================================

@dataclass(frozen=True)
class ParamBlueprint:
    """One parameter of a component type, declared once per component class.

    Attributes
    ----------
    name:
        ASCII identifier used in machine-readable parameter names and in the
        curated override tables (e.g. ``"alpha"``, ``"mu"``).
    latex:
        Base LaTeX body WITHOUT the surrounding ``$...$`` (e.g. ``r"\\alpha"``).
        Slot tagging composes subscripts onto this via :func:`tag_latex`.
    bounds:
        Default prior bounds.
    fiducial:
        Default fiducial value used by ``get_fixed_population_params``.
    """

    name: str
    latex: str
    bounds: _B
    fiducial: float


ParamTuple = Tuple[ParamBlueprint, ...]


@dataclass(frozen=True)
class ComponentBlueprint:
    """Registry entry for one component class.

    Attributes
    ----------
    key:
        Registry key; for grammar-visible mass components this is also the
        grammar token (``"powerlaw"``, ``"peak"``).
    kind:
        ``"mass"``, ``"pairing"``, or ``"spin"``.
    short:
        Tag stem used in slot tags and label subscripts (``"PL"``, ``"G"``).
    cls:
        The component class, or a lazy dotted reference
        ``"package.module:ClassName"`` (used for GP classes so that importing
        the registry never imports tinygp).
    params:
        Parameter blueprints in the exact order consumed by the component, or
        a zero-argument callable returning them (used when the parameter count
        itself requires a lazy import, e.g. the 2D GP node grid).
    plural:
        Grammar plural for count prefixes (``"powerlaws"``); mass tokens only.
    build:
        Optional custom constructor ``build(cls, specs) -> instance`` for
        classes whose __init__ is not ``cls(*specs)``.
    """

    key: str
    kind: str
    short: str
    cls: Union[type, str]
    params: Union[ParamTuple, Callable[[], ParamTuple]]
    plural: Optional[str] = None
    build: Optional[Callable] = None

    def resolve_params(self) -> ParamTuple:
        return self.params() if callable(self.params) else self.params

    def resolve_cls(self) -> type:
        if isinstance(self.cls, str):
            mod_name, _, cls_name = self.cls.partition(":")
            return getattr(import_module(mod_name), cls_name)
        return self.cls

    def instantiate(self, specs):
        """Build a component instance from ordered ParamSpec objects."""
        cls = self.resolve_cls()
        if self.build is not None:
            return self.build(cls, specs)
        return cls(*specs)


# ============================================================
# Registries + registration decorators
# ============================================================

COMPONENTS: dict[str, ComponentBlueprint] = {}
MASS_TOKENS: dict[str, ComponentBlueprint] = {}
PLURALS: dict[str, str] = {}


def _register(bp: ComponentBlueprint):
    if bp.key in COMPONENTS:
        raise ValueError(f"Component {bp.key!r} registered twice")
    COMPONENTS[bp.key] = bp
    if bp.kind == "mass" and bp.plural is not None:
        MASS_TOKENS[bp.key] = bp
        PLURALS[bp.plural] = bp.key
    return bp


def mass_token(token: str, *, plural: str, short: str, params: ParamTuple):
    """Class decorator registering a grammar-visible mass component."""

    def deco(cls):
        _register(ComponentBlueprint(
            key=token, kind="mass", short=short, cls=cls,
            params=params, plural=plural,
        ))
        return cls

    return deco


def component(key: str, *, kind: str, params: ParamTuple, short: str = ""):
    """Class decorator registering a pairing/spin (or non-grammar) component."""

    def deco(cls):
        _register(ComponentBlueprint(
            key=key, kind=kind, short=short, cls=cls, params=params,
        ))
        return cls

    return deco


# ============================================================
# Label composition
# ============================================================

_SUB_BRACED = re.compile(r"^(.*)_\{(.+)\}$")
_SUB_SIMPLE = re.compile(r"^(.*)_(\\?[A-Za-z0-9]+)$")


def tag_latex(base: str, tag: str) -> str:
    r"""Append a roman slot tag to a base LaTeX body's subscript.

    ``\alpha`` -> ``\alpha_{\rm PL}``;  ``m_{\min}`` -> ``m_{\min,\rm PL}``;
    ``\mu_\chi`` -> ``\mu_{\chi,\rm G1}``;  ``\alpha_1`` -> ``\alpha_{1,\rm BPL}``.
    """
    m = _SUB_BRACED.match(base)
    if m:
        return rf"{m.group(1)}_{{{m.group(2)},\rm {tag}}}"
    m = _SUB_SIMPLE.match(base)
    if m:
        return rf"{m.group(1)}_{{{m.group(2)},\rm {tag}}}"
    return rf"{base}_{{\rm {tag}}}"


# ============================================================
# GP component blueprints
# ============================================================
# The legacy mixture-coupled GP components (``gp_mass``, ``gp_pairing``,
# ``gp_pairing_2d``) have been removed.  GP population models are now standalone,
# true-GP-prior models registered directly via ``register_model`` in
# :mod:`darksirens.gw.populations.registry` (see :mod:`darksirens.gw.populations.gp`)
# and no longer participate in the stick-breaking mixture grammar.
