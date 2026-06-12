from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import jax.numpy as jnp
from jax import jit

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


@dataclass(frozen=True)
class NormalizationGridSettings:
    """Settings controlling GW-population normalisation quadrature grids.

    The mass, mass-ratio, and spin normalisations are evaluated on cached
    trapezoid grids.  Values can be configured with ``configure_normalization_grids``
    or the environment variables ``DARKSIRENS_GW_N_MASS``,
    ``DARKSIRENS_GW_N_Q``, and ``DARKSIRENS_GW_N_CHI``.
    """

    n_mass: int = _env_int("DARKSIRENS_GW_N_MASS", 500)
    n_q: int = _env_int("DARKSIRENS_GW_N_Q", 200)
    n_chi: int = _env_int("DARKSIRENS_GW_N_CHI", 200)
    m_lo: float = M_LO
    m_hi: float = M_HI
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
        for lo_name, hi_name in (("m_lo", "m_hi"), ("q_lo", "q_hi"), ("chi_lo", "chi_hi")):
            lo = float(getattr(self, lo_name))
            hi = float(getattr(self, hi_name))
            if not hi > lo:
                raise ValueError(f"{hi_name} must be greater than {lo_name}: {lo} >= {hi}")
            object.__setattr__(self, lo_name, lo)
            object.__setattr__(self, hi_name, hi)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


_NORMALIZATION_GRID_SETTINGS = NormalizationGridSettings()


def normalization_grid_settings() -> NormalizationGridSettings:
    """Return the active GW-population normalisation-grid settings."""

    return _NORMALIZATION_GRID_SETTINGS


def configure_normalization_grids(
    *,
    n_mass: int | None = None,
    n_q: int | None = None,
    n_chi: int | None = None,
) -> NormalizationGridSettings:
    """Update cached normalisation-grid sizes and clear derived grids."""

    global _NORMALIZATION_GRID_SETTINGS, N_MASS, N_Q, N_CHI

    updates = {
        key: value
        for key, value in {"n_mass": n_mass, "n_q": n_q, "n_chi": n_chi}.items()
        if value is not None
    }
    if updates:
        _NORMALIZATION_GRID_SETTINGS = replace(_NORMALIZATION_GRID_SETTINGS, **updates)
        _clear_grid_caches()

    # Backward-compatible scalar aliases for callers that inspect the values
    # after configuration.  Population code should use get_*_grid() directly.
    N_MASS = _NORMALIZATION_GRID_SETTINGS.n_mass
    N_Q = _NORMALIZATION_GRID_SETTINGS.n_q
    N_CHI = _NORMALIZATION_GRID_SETTINGS.n_chi
    return _NORMALIZATION_GRID_SETTINGS


@lru_cache(maxsize=8)
def _linspace(lo: float, hi: float, n: int):
    return jnp.linspace(lo, hi, int(n))


def _clear_grid_caches() -> None:
    _linspace.cache_clear()
    get_mass_grid.cache_clear()
    get_q_grid.cache_clear()
    get_chi_grid.cache_clear()
    get_m1_q_mesh.cache_clear()


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
    return jnp.meshgrid(get_mass_grid(), get_q_grid(), indexing="ij")


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
@jit
def sfilter_low(m, m_min, dm):
    delta = m - m_min
    safe_d = jnp.where(delta > 0, delta, 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / (safe_d - safe_dm),
        -500.0, 500.0
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m <= m_min, 0.0, S)
    S = jnp.where(m >= m_min + dm, 1.0, S)
    return S

@jit
def sfilter_high(m, m_max, dm):
    delta = m_max - m
    safe_d = jnp.where(delta > 0, delta, 1.0)
    safe_dm = jnp.where(dm > 0, dm, 1.0)
    expo = jnp.clip(
        safe_dm / safe_d + safe_dm / (safe_d - safe_dm),
        -500.0, 500.0
    )
    S = 1.0 / (jnp.exp(expo) + 1.0)
    S = jnp.where(m >= m_max, 0.0, S)
    S = jnp.where(m <= m_max - dm, 1.0, S)
    return S
