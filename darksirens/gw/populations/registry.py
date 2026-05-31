"""
registry.py
-----------
Population model registry for gravitational-wave source inference.

Structure
---------
1. Prior bounds constants      — all default prior ranges in one place
2. Stick-breaking conversion   — _w_to_v converts desired weights → v inputs
3. Component factory functions — each builds one model component
4. Mixture factories           — assemble components into MixtureModel objects
5. Model registry              — maps string keys to PopulationModel instances
6. Fiducial parameters         — default values for --fix_population=True
7. Public API                  — get_model, pop_model_parser, pop_model_prior_parser

Adding a new model
------------------
(a) Write a mixture factory _mixture_<name>().
(b) Add it to _RAW_MODELS with a LaTeX label.
(c) Add a fiducial parameter entry to _FIDUCIAL_PARAMS.

Stick-breaking weight parameterisation
---------------------------------------
Fiducial weights in _FIDUCIAL_PARAMS are stored as DESIRED FINAL FRACTIONS
(e.g. [0.10, 0.05] means "10 % in component 1, 5 % in component 2, 85 % rest").
get_fixed_population_params converts these to stick-breaking inputs v_i via
_w_to_v before assembling the parameter vector.  This keeps _FIDUCIAL_PARAMS
human-readable while ensuring the assembled vector is correct.

Weight parameter labels changed from $f_i$ to $v_i$
------------------------------------------------------
The sampled parameters are stick-breaking inputs, not direct fractions.
If fixed_parameter_values in a settings JSON previously used "$f_1$" as a key,
rename it to "$v_1$" and convert the value:
    v_1 = w_1                           (k=2, unchanged)
    v_i = w_i / (1 - w_1 - ... - w_{i-1})   (k≥3)
"""

from __future__ import annotations
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from .base import ParamSpec, MixtureModel, PopulationModel, _stick_breaking_weights
from .parametric import (
    PowerLaw, BrokenPowerLaw, Gaussian,
    PowerLawPairing, TruncatedGaussianSpin,
)


# ============================================================
# 1. Prior bounds
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


# ============================================================
# 2. Stick-breaking weight conversion
# ============================================================

def _w_to_v(w_desired_full: list | np.ndarray) -> list:
    """
    Convert a k-vector of desired mixture weights to k-1 stick-breaking inputs.

    Parameters
    ----------
    w_desired_full : sequence of length k, all ≥ 0, summing to 1.
        Desired final mixture weights.

    Returns
    -------
    list of length k-1  (empty list for k=1).

    Formula
    -------
        v_1 = w_1
        v_i = w_i / (1 - w_1 - ... - w_{i-1})   for i ≥ 2

    Inverse of _stick_breaking_weights in base.py.
    """
    w = np.asarray(w_desired_full, dtype=float)
    k = len(w)
    if k == 1:
        return []
    v         = []
    remaining = 1.0
    for i in range(k - 1):
        vi = float(w[i] / remaining) if remaining > 1e-12 else 0.0
        v.append(vi)
        remaining -= float(w[i])
    return v


def _stick_breaking_weights_np(v: list) -> np.ndarray:
    """
    Pure-numpy stick-breaking for use at import time before JAX x64 is configured.
    Mathematically identical to _stick_breaking_weights in base.py.
    """
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return np.array([1.0])
    cumprod   = np.cumprod(1.0 - v)
    remaining = np.concatenate([[1.0], cumprod[:-1]])
    return np.concatenate([v * remaining, [cumprod[-1]]])


def _verify_w_to_v_roundtrip():
    """Sanity check at import time: _w_to_v ∘ _stick_breaking_weights = id.

    Uses the pure-numpy implementation so this runs correctly regardless of
    whether jax_enable_x64 has been set (float32 only has ~1e-7 precision,
    which would fail a 1e-10 tolerance check).
    """
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
        v   = _w_to_v(w)
        w_r = list(_stick_breaking_weights_np(v))
        err = max(abs(a - b) for a, b in zip(w, w_r))
        assert err < 1e-10, (
            f"_w_to_v round-trip failed: desired={w}, recovered={w_r}, err={err:.2e}"
        )

_verify_w_to_v_roundtrip()


# ============================================================
# 3. Component factory functions
# ============================================================

def _pl(
    alpha_label = r"$\alpha$",
    mmin_label  = r"$m_{\min}$",
    mmax_label  = r"$m_{\max}$",
    dmmin_label = r"$\delta m_{\min}$",
    dmmax_label = r"$\delta m_{\max}$",
    *,
    alpha = ALPHA,
    mmin  = M_MIN,
    mmax  = M_MAX,
    dmmin = DM_MIN,
    dmmax = DM_MAX,
) -> PowerLaw:
    return PowerLaw(
        ParamSpec(alpha_label, alpha.lo, alpha.hi),
        ParamSpec(mmin_label,  mmin.lo,  mmin.hi),
        ParamSpec(mmax_label,  mmax.lo,  mmax.hi),
        ParamSpec(dmmin_label, dmmin.lo, dmmin.hi),
        ParamSpec(dmmax_label, dmmax.lo, dmmax.hi),
    )


def _bpl(
    alpha1_label = r"$\alpha_1$",
    alpha2_label = r"$\alpha_2$",
    mbreak_label = r"$m_{\rm break}$",
    mmin_label   = r"$m_{\min}$",
    mmax_label   = r"$m_{\max}$",
    dmmin_label  = r"$\delta m_{\min}$",
    dmmax_label  = r"$\delta m_{\max}$",
    *,
    alpha1 = ALPHA_BPL,
    alpha2 = ALPHA_BPL,
    mbreak = M_BREAK,
    mmin   = M_MIN,
    mmax   = M_MAX,
    dmmin  = DM_MIN,
    dmmax  = DM_MAX,
) -> BrokenPowerLaw:
    return BrokenPowerLaw(
        ParamSpec(alpha1_label, alpha1.lo, alpha1.hi),
        ParamSpec(alpha2_label, alpha2.lo, alpha2.hi),
        ParamSpec(mbreak_label, mbreak.lo, mbreak.hi),
        ParamSpec(mmin_label,   mmin.lo,   mmin.hi),
        ParamSpec(mmax_label,   mmax.lo,   mmax.hi),
        ParamSpec(dmmin_label,  dmmin.lo,  dmmin.hi),
        ParamSpec(dmmax_label,  dmmax.lo,  dmmax.hi),
    )


def _gauss(
    mu_label  = r"$\mu_G$",
    sig_label = r"$\sigma_G$",
    *,
    mu  = GAUSS_MU,
    sig = GAUSS_SIGMA,
) -> Gaussian:
    return Gaussian(
        ParamSpec(mu_label,  mu.lo,  mu.hi),
        ParamSpec(sig_label, sig.lo, sig.hi),
    )


def _plpairing(
    beta_label = r"$\beta$",
    *,
    beta = BETA,
) -> PowerLawPairing:
    return PowerLawPairing(ParamSpec(beta_label, beta.lo, beta.hi))


def _spin(
    mu_label  = r"$\mu_\chi$",
    sig_label = r"$\sigma_\chi$",
    *,
    mu  = CHI_MU,
    sig = CHI_SIGMA,
) -> TruncatedGaussianSpin:
    return TruncatedGaussianSpin(
        ParamSpec(mu_label,  mu.lo,  mu.hi),
        ParamSpec(sig_label, sig.lo, sig.hi),
    )


def _gp_mass(
    mmin_label  = r"$m_{\min}$",
    mmax_label  = r"$m_{\max}$",
    dmmin_label = r"$\delta m_{\min}$",
    dmmax_label = r"$\delta m_{\max}$",
    alpha_label = r"$\alpha$",
    amp_label   = r"$A$",
    ls_label    = r"$\ell$",
    y_labels    = None,
    *,
    mmin  = M_MIN,
    mmax  = M_MAX,
    dmmin = DM_MIN,
    dmmax = DM_MAX,
    alpha = GP_ALPHA,
    amp   = GP_AMP,
    ls    = GP_LS,
    y     = GP_Y,
) -> GaussianProcessMass1D:
    from .gp import GaussianProcessMass1D

    if y_labels is None:
        y_labels = [rf"$y_{{{i}}}$" for i in range(11)]
    return GaussianProcessMass1D(
        ParamSpec(mmin_label,  mmin.lo,  mmin.hi),
        ParamSpec(mmax_label,  mmax.lo,  mmax.hi),
        ParamSpec(dmmin_label, dmmin.lo, dmmin.hi),
        ParamSpec(dmmax_label, dmmax.lo, dmmax.hi),
        ParamSpec(alpha_label, alpha.lo, alpha.hi),
        ParamSpec(amp_label,   amp.lo,   amp.hi),
        ParamSpec(ls_label,    ls.lo,    ls.hi),
        *[ParamSpec(label, y.lo, y.hi) for label in y_labels],
    )


def _gp_pairing(
    beta_label = r"$\beta_q$",
    amp_label  = r"$A_q$",
    ls_label   = r"$\ell_q$",
    y_labels   = None,
    *,
    beta = GP_BETA_Q,
    amp  = GP_AMP,
    ls   = GP_LS,
    y    = GP_Y_PAIR,
) -> GaussianProcessMassRatio1D:
    from .gp import GaussianProcessMassRatio1D

    if y_labels is None:
        y_labels = [rf"$y_{{q,{i}}}$" for i in range(5)]
    return GaussianProcessMassRatio1D(
        ParamSpec(beta_label, beta.lo, beta.hi),
        ParamSpec(amp_label,  amp.lo,  amp.hi),
        ParamSpec(ls_label,   ls.lo,   ls.hi),
        *[ParamSpec(label, y.lo, y.hi) for label in y_labels],
    )


def _gp_mass_pairing_2d(
    beta_label  = r"$\beta$",
    amp_label   = r"$A_q$",
    ls_m_label  = r"$\ell_{m,q}$",
    ls_q_label  = r"$\ell_{q,q}$",
    y_labels    = None,
    *,
    beta  = GP_BETA_Q,
    amp   = _B(0.01, 3.0),
    ls_m  = GP_LS_M_PAIR,
    ls_q  = GP_LS_Q_PAIR,
    y     = GP_Y_PAIR,
) -> GaussianProcessPairing2D:
    from .gp import GaussianProcessPairing2D, _PAIR2D_N_NODES

    if y_labels is None:
        y_labels = [rf"$y_{{q,{i}}}$" for i in range(_PAIR2D_N_NODES)]
    y_specs = tuple(ParamSpec(label, y.lo, y.hi) for label in y_labels)
    return GaussianProcessPairing2D(
        ParamSpec(beta_label,  beta.lo,  beta.hi),
        ParamSpec(amp_label,   amp.lo,   amp.hi),
        ParamSpec(ls_m_label,  ls_m.lo,  ls_m.hi),
        ParamSpec(ls_q_label,  ls_q.lo,  ls_q.hi),
        y_specs,
    )


# ============================================================
# 4. Mixture factories
# ============================================================

def _make_spins(labels, shared):
    if shared:
        return [_spin()]
    return [_spin(mu_label=rf"$\mu_{{\chi,\rm {s}}}$",
                  sig_label=rf"$\sigma_{{\chi,\rm {s}}}$") for s in labels]


def _make_pairings(labels, shared):
    if shared:
        return [_plpairing()]
    return [_plpairing(beta_label=rf"$\beta_{{\rm {s}}}$") for s in labels]


def _mixture_plpeak(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Power-law + Gaussian peak  (LVK POWER LAW + PEAK)."""
    labels = ["PL", "G"]
    return MixtureModel(
        [_pl(), _gauss(mu_label=r"$\mu_G$", sig_label=r"$\sigma_G$",
                       mu=_B(20, 50), sig=GAUSS_SIGMA)],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_bpl2peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + two Gaussian peaks."""
    labels = ["BPL", "G1", "G2"]
    return MixtureModel(
        [_bpl(),
         _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5, 20)),
         _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(25, 40))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_bpl3peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + three Gaussian peaks."""
    labels = ["BPL", "G1", "G2", "G3"]
    return MixtureModel(
        [_bpl(),
         _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5, 20)),
         _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(25, 40)),
         _gauss(mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", mu=_B(50,100), sig=_B(1, 20))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_bpl2peaks1pl(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Broken power-law + two Gaussian peaks + high-mass power-law tail."""
    labels = ["BPL", "G1", "G2", "PL"]
    return MixtureModel(
        [_bpl(),
         _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5, 20)),
         _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(25, 40)),
         _pl(alpha_label=r"$\alpha_{\rm PL}$", mmin_label=r"$m_{\min,\rm PL}$",
             mmax_label=r"$m_{\max,\rm PL}$", dmmin_label=r"$\delta m_{\min,\rm PL}$",
             dmmax_label=r"$\delta m_{\max,\rm PL}$",
             alpha=_B(0,6), mmin=_B(40,60), mmax=_B(80,120))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_2pl1peak(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + one Gaussian peak."""
    labels = ["PL1", "PL2", "G"]
    return MixtureModel(
        [_pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
             mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
             dmmax_label=r"$\delta m_{\max,1}$", alpha=_B(0,6), mmin=_B(2,10), mmax=_B(15,50)),
         _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
             mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
             dmmax_label=r"$\delta m_{\max,2}$", alpha=_B(0,6), mmin=_B(20,40), mmax=_B(50,100)),
         _gauss(mu_label=r"$\mu_G$", sig_label=r"$\sigma_G$", mu=_B(50,100), sig=_B(1,20))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_2pl2peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + two Gaussian peaks."""
    labels = ["PL1", "PL2", "G1", "G2"]
    return MixtureModel(
        [_pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
             mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
             dmmax_label=r"$\delta m_{\max,1}$", alpha=_B(0,6), mmin=_B(2,10), mmax=_B(15,50)),
         _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
             mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
             dmmax_label=r"$\delta m_{\max,2}$", alpha=_B(0,6), mmin=_B(20,40), mmax=_B(50,100)),
         _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5,20), sig=_B(1,10)),
         _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(20,50), sig=_B(1,15))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_2pl3peaks(shared_beta=False, shared_spin=False) -> MixtureModel:
    """Two power-law components + three Gaussian peaks."""
    labels = ["PL1", "PL2", "G1", "G2", "G3"]
    return MixtureModel(
        [_pl(alpha_label=r"$\alpha_1$", mmin_label=r"$m_{\min,1}$",
             mmax_label=r"$m_{\max,1}$", dmmin_label=r"$\delta m_{\min,1}$",
             dmmax_label=r"$\delta m_{\max,1}$", alpha=_B(0,6), mmin=_B(2,10), mmax=_B(15,50)),
         _pl(alpha_label=r"$\alpha_2$", mmin_label=r"$m_{\min,2}$",
             mmax_label=r"$m_{\max,2}$", dmmin_label=r"$\delta m_{\min,2}$",
             dmmax_label=r"$\delta m_{\max,2}$", alpha=_B(0,6), mmin=_B(20,40), mmax=_B(50,100)),
         _gauss(mu_label=r"$\mu_1$", sig_label=r"$\sigma_1$", mu=_B( 5,20), sig=_B(1,10)),
         _gauss(mu_label=r"$\mu_2$", sig_label=r"$\sigma_2$", mu=_B(20,50), sig=_B(1,10)),
         _gauss(mu_label=r"$\mu_3$", sig_label=r"$\sigma_3$", mu=_B(50,100), sig=_B(1,20))],
        _make_pairings(labels, shared_beta),
        _make_spins(labels, shared_spin),
    )


def _mixture_gp_mass(shared_beta=True, shared_spin=True) -> MixtureModel:
    return MixtureModel([_gp_mass()], [_plpairing()], [_spin()])


def _mixture_gp_mass_pairing(shared_beta=True, shared_spin=True) -> MixtureModel:
    return MixtureModel([_gp_mass()], [_gp_pairing()], [_spin()])


def _mixture_gp_mass_pairing_joint(shared_beta=True, shared_spin=True) -> MixtureModel:
    return MixtureModel([_gp_mass()], [_gp_mass_pairing_2d()], [_spin()])


# ============================================================
# 5. Model registry
# ============================================================

_RAW_MODELS: dict[str, tuple] = {
    "powerlaw+peak":                  (_mixture_plpeak,               "PL+G"),
    "brokenpowerlaw+2peaks":          (_mixture_bpl2peaks,            "BPL+2G"),
    "brokenpowerlaw+3peaks":          (_mixture_bpl3peaks,            "BPL+3G"),
    "brokenpowerlaw+2peaks+powerlaw": (_mixture_bpl2peaks1pl,         "BPL+2G+PL"),
    "twopowerlaws+peak":              (_mixture_2pl1peak,             "2PL+G"),
    "twopowerlaws+2peaks":            (_mixture_2pl2peaks,            "2PL+2G"),
    "twopowerlaws+3peaks":            (_mixture_2pl3peaks,            "2PL+3G"),
    "gp_mass":                        (_mixture_gp_mass,              "GP"),
    "gp_mass_pairing":                (_mixture_gp_mass_pairing,      "GPxGP"),
    "gp_mass_pairing_joint":          (_mixture_gp_mass_pairing_joint,"GP 2D"),
}

_MODEL_REGISTRY:  dict[str, PopulationModel] = {}
_MODEL_FACTORIES: dict[str, tuple] = {}
MODEL_NAME_LATEX: dict[str, str]             = {"mock_data": r"\text{Mock}"}

for _name, (_mix_fn, _latex) in _RAW_MODELS.items():
    for _sb, _ss, _suffix, _label_suffix in [
        (False, False, "",                  ""),
        (True,  False, "_shared_beta",      r" (Shared $\beta$)"),
        (False, True,  "_shared_spin",      r" (Shared Spin)"),
        (True,  True,  "_shared_beta_spin", r" (Shared $\beta$, Spin)"),
    ]:
        _key = _name + _suffix
        _MODEL_FACTORIES[_key] = (_mix_fn, _sb, _ss)
        MODEL_NAME_LATEX[_key] = _latex + _label_suffix


# ============================================================
# 6. Fiducial parameters
# ============================================================
# weights  — DESIRED FINAL FRACTIONS (human-readable).
#            get_fixed_population_params converts these to stick-breaking
#            inputs v_i via _w_to_v before assembling the vector.
# masses   — mass / pairing / spin params; ordering matches param_specs.
# Ordering: weights → masses → pairing → spins → gamma

def _fiducial_pl():
    return [2.3, 5.0, 80.0, 3.0, 10.0]       # alpha mmin mmax dmmin dmmax

def _fiducial_bpl():
    return [2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0]  # α1 α2 mbreak mmin mmax dmmin dmmax

def _fiducial_gauss_low():  return [10.0, 3.0]   # mu sigma
def _fiducial_gauss_mid():  return [35.0, 5.0]
def _fiducial_gauss_high(): return [70.0, 10.0]

def _fiducial_pl_pairing(n=1): return [1.0] * n         # beta
def _fiducial_spin(n=1):       return [0.0, 0.1] * n    # mu_chi sigma_chi

def _fiducial_gp_mass():
    return [5.0, 80.0, 3.0, 10.0,   # mmin mmax dmmin dmmax
            2.3,  1.0,  1.0,        # alpha amp ls
            *([0.0] * 11)]           # y0..y10

def _fiducial_gp_pairing():
    return [1.5, 1.0, 0.5,           # beta amp ls
            *([0.0] * 5)]            # y0..y4

def _fiducial_gp_pairing_2d():
    return [0.0, 1.0, 1.0, 0.5,      # beta amp ls_m ls_q
            *([0.0] * 30)]            # y nodes on the 6x5 GP grid


_FIDUCIAL_PARAMS: dict[str, dict] = {
    # ---- powerlaw+peak (k=2) ----
    "powerlaw+peak": dict(
        n_comp  = 2,
        weights = [0.10],                    # w_PL=0.10, w_G=0.90
        masses  = [*_fiducial_pl(), *_fiducial_gauss_mid()],
    ),

    # ---- brokenpowerlaw+2peaks (k=3) ----
    "brokenpowerlaw+2peaks": dict(
        n_comp  = 3,
        weights = [0.10, 0.05],              # w_BPL=0.10, w_G1=0.05, w_G2=0.85
        masses  = [*_fiducial_bpl(), *_fiducial_gauss_low(), *_fiducial_gauss_mid()],
    ),

    # ---- brokenpowerlaw+3peaks (k=4) ----
    "brokenpowerlaw+3peaks": dict(
        n_comp  = 4,
        weights = [0.10, 0.05, 0.03],        # w_BPL=0.10, w_G1=0.05, w_G2=0.03, w_G3=0.82
        masses  = [*_fiducial_bpl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid(), *_fiducial_gauss_high()],
    ),

    # ---- brokenpowerlaw+2peaks+powerlaw (k=4) ----
    "brokenpowerlaw+2peaks+powerlaw": dict(
        n_comp  = 4,
        weights = [0.10, 0.05, 0.05],        # w_BPL=0.10, w_G1=0.05, w_G2=0.05, w_PL=0.80
        masses  = [*_fiducial_bpl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid(),
                   3.0, 50.0, 100.0, 3.0, 10.0],   # high-mass PL tail
    ),

    # ---- twopowerlaws+peak (k=3) ----
    "twopowerlaws+peak": dict(
        n_comp  = 3,
        weights = [0.20, 0.10],              # w_PL1=0.20, w_PL2=0.10, w_G=0.70
        masses  = [*_fiducial_pl(), *_fiducial_pl(), *_fiducial_gauss_mid()],
    ),

    # ---- twopowerlaws+2peaks (k=4) ----
    "twopowerlaws+2peaks": dict(
        n_comp  = 4,
        weights = [0.15, 0.10, 0.05],        # w_PL1=0.15, w_PL2=0.10, w_G1=0.05, w_G2=0.70
        masses  = [*_fiducial_pl(), *_fiducial_pl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid()],
    ),

    # ---- twopowerlaws+3peaks (k=5) ----
    "twopowerlaws+3peaks": dict(
        n_comp  = 5,
        weights = [0.15, 0.10, 0.05, 0.03],  # w_PL1=0.15,..,w_G3=0.03, w_G4=0.67
        masses  = [*_fiducial_pl(), *_fiducial_pl(),
                   *_fiducial_gauss_low(), *_fiducial_gauss_mid(), *_fiducial_gauss_high()],
    ),

    # ---- GP models (k=1, no weight params) ----
    "gp_mass": dict(
        n_comp  = 1,
        weights = [],
        masses  = _fiducial_gp_mass(),
    ),
    "gp_mass_pairing": dict(
        n_comp            = 1,
        weights           = [],
        masses            = _fiducial_gp_mass(),
        _pairing_override = _fiducial_gp_pairing(),
    ),
    "gp_mass_pairing_joint": dict(
        n_comp            = 1,
        weights           = [],
        masses            = _fiducial_gp_mass(),
        _pairing_override = _fiducial_gp_pairing_2d(),
    ),
}


# ============================================================
# 7. Public API
# ============================================================

def get_fixed_population_params(pop_model: str) -> jnp.ndarray:
    """
    Return the fiducial population parameter vector for --fix_population=True.

    Ordering matches PopulationModel.param_specs exactly:
        v_weights → mass params → pairing params → spin params → gamma

    Weight parameters are stick-breaking inputs (v_i), not direct fractions.
    The conversion from the human-readable fractions in _FIDUCIAL_PARAMS is
    done here via _w_to_v so that _FIDUCIAL_PARAMS remains easy to audit.
    """
    base_model  = pop_model
    shared_beta = False
    shared_spin = False

    if "_shared_beta_spin" in base_model:
        base_model  = base_model.replace("_shared_beta_spin", "")
        shared_beta = shared_spin = True
    else:
        if "_shared_beta" in base_model:
            base_model  = base_model.replace("_shared_beta", "")
            shared_beta = True
        if "_shared_spin" in base_model:
            base_model  = base_model.replace("_shared_spin", "")
            shared_spin = True

    if base_model not in _FIDUCIAL_PARAMS:
        raise ValueError(
            f"No fiducial parameters for '{base_model}'. "
            f"Available: {sorted(_FIDUCIAL_PARAMS.keys())}."
        )

    spec   = _FIDUCIAL_PARAMS[base_model]
    n_comp = spec["n_comp"]

    # --- Weight params: convert desired fractions → stick-breaking inputs ---
    w_head = spec["weights"]                           # e.g. [0.10, 0.05]
    w_last = 1.0 - sum(w_head)
    w_full = w_head + [w_last]                         # e.g. [0.10, 0.05, 0.85]
    v_weights = _w_to_v(w_full)                        # e.g. [0.10, 0.0556]

    # --- Pairing fiducials ---
    if "_pairing_override" in spec:
        betas = spec["_pairing_override"]
    elif shared_beta:
        betas = _fiducial_pl_pairing(n=1)
    else:
        betas = _fiducial_pl_pairing(n=n_comp)

    # --- Spin fiducials ---
    spins = _fiducial_spin(n=1 if shared_spin else n_comp)

    # --- Assemble in param_specs order ---
    full = v_weights + spec["masses"] + betas + spins + [0.0]  # 0.0 = gamma
    return jnp.array(full, dtype=float)


def get_model(pop_model: str) -> PopulationModel:
    try:
        return _MODEL_REGISTRY[pop_model]
    except KeyError:
        pass

    try:
        mix_fn, shared_beta, shared_spin = _MODEL_FACTORIES[pop_model]
    except KeyError:
        raise ValueError(
            f"Unknown model {pop_model!r}. "
            f"Available: {sorted(_MODEL_FACTORIES.keys())}"
        )

    model = PopulationModel(mixture=mix_fn(shared_beta=shared_beta, shared_spin=shared_spin))
    _MODEL_REGISTRY[pop_model] = model
    return model

def pop_model_parser(pop_model: str):
    return get_model(pop_model).log_p_pop


def pop_model_prior_parser(pop_model: str) -> tuple[list, list, list, str]:
    model = get_model(pop_model)
    lows, highs, labels = model.prior_bounds()
    return lows, highs, labels, MODEL_NAME_LATEX.get(pop_model, pop_model)