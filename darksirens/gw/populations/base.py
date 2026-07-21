"""
base.py
-------
Abstract base classes and the MixtureModel / PopulationModel assemblers.

Mixture weight parameterisation — stick-breaking
-------------------------------------------------
A k-component mixture has k-1 free weight parameters v_1 ... v_{k-1},
each sampled from U[0, 1].  The final k weights are produced by
stick-breaking:

    w_1     = v_1
    w_i     = v_i * prod_{j < i}(1 − v_j)    for i = 2 … k−1
    w_k     = prod_{j=1}^{k-1}(1 − v_j)

All v_i ∈ [0, 1]  →  all w_i ∈ [0, 1]  and  Σ w_i = 1, by construction.
Negative weights are impossible regardless of the sampled values.

Why this replaces the old "last weight = 1 − Σ" approach
---------------------------------------------------------
Under the old parameterisation every v_i was sampled independently from
U[0, 1] and the last weight was computed as w_k = 1 − Σ_{i<k} v_i.
The fraction of the U[0,1]^{k-1} prior volume where w_k ≥ 0 is 1/(k−1)!:

    k=2 → 100 %   k=3 → 50 %   k=4 → 17 %   k=5 → 4 %

For five-component models the sampler spent 96 % of proposals in a region
that is silently mapped to −∞, collapsing effective sample sizes and
biasing evidence estimates.

Prior bounds on the weight parameters stay [0, 1] — no change to prior
transforms, CLI flags, or existing sampler configuration.

Labels: weight parameters are now named $v_i$ (stick-breaking inputs),
not $f_i$ (direct fractions).  If you use ``fixed_parameter_values`` in
a settings JSON and previously wrote ``"$f_1$": 0.3``, rename the key to
``"$v_1$"`` and convert the value: for k=2 the number is unchanged; for
k≥3 use v_i = w_i / (1 − w_1 − … − w_{i−1}).

Sentinel convention
-------------------
p ≤ 0  →  log p = −jnp.inf.

The old code used −1e10.  That value is finite, so it propagated through
logsumexp / jnp.sum and appeared as a valid (very negative) likelihood to
the sampler.  Using −∞ everywhere is correct: logsumexp and jnp.sum
handle −∞ entries properly, and the final jnp.isfinite guard in the
likelihood rejects any proposal that produces −∞.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp

from .utils import (
    get_mass_grid,
    get_q_grid,
    get_chi_grid,
    get_pairing_m1_grid,
    normalization_grid_settings,
    M_LO,
    M_HI,
)


# ── Parameter bookkeeping ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """Metadata for one sampled population-model parameter.

    Attributes
    ----------
    label:
        Human-readable label used in tables, diagnostics, and sampler outputs.
    low, high:
        Inclusive prior-transform bounds for the unit-cube samplers.
    name:
        Optional machine-readable ASCII identifier (e.g. ``"G1.mu"``).  Unlike
        ``label`` it contains no LaTeX and is stable under display changes.
    prior_kind:
        Prior family the sampler should place on this parameter.  One of
        ``"uniform"`` (default; affine unit-cube transform / ``dist.Uniform``),
        ``"normal"`` (truncated standard normal — used for whitened GP latents
        ``xi``), or ``"lognormal"`` (exp of a truncated normal in log-space).
        ``low``/``high`` always act as the truncation bounds, so existing
        bounds-based machinery (overrides, fixed-value validation, sampler
        bound checks) keeps working unchanged.
    prior_loc, prior_scale:
        Location/scale of the underlying (log-)normal for ``"normal"`` and
        ``"lognormal"`` kinds.  ``None`` defaults to ``(0.0, 1.0)``, i.e. the
        standard normal appropriate for whitened ``xi``.  Ignored for
        ``"uniform"``.
    """

    label: str
    low: float
    high: float
    name: str = ""
    prior_kind: str = "uniform"
    prior_loc: float | None = None
    prior_scale: float | None = None


def pack_specs(*specs: ParamSpec):
    """Split ``ParamSpec`` objects into lower bounds, upper bounds, and labels.

    Registry constructors use this helper to build the arrays expected by the
    sampler-facing prior transform while preserving a single source of truth for
    parameter ordering.
    """
    return (
        [s.low  for s in specs],
        [s.high for s in specs],
        [s.label for s in specs],
    )


# ── Stick-breaking ───────────────────────────────────────────────────────────

def _stick_breaking_weights(v_raw: jnp.ndarray) -> jnp.ndarray:
    """
    Map k−1 stick-breaking inputs v ∈ [0,1]^{k-1} to k mixture weights
    that are guaranteed positive and sum to 1.

    Algorithm
    ---------
    remaining[0] = 1
    remaining[i] = prod_{j<i}(1 − v_j)

    w[i] = v[i] * remaining[i]      for i = 0 … k−2
    w[k-1] = remaining[k-1]         (the last piece of stick)

    Proof that Σ w_i = 1: by induction, each break uses fraction v_i of
    the remaining stick; the final term consumes the remainder.
    """
    cumprod   = jnp.cumprod(1.0 - v_raw)               # (k-1,)
    remaining = jnp.concatenate([jnp.ones(1), cumprod[:-1]])   # (k-1,)
    return jnp.concatenate([v_raw * remaining, cumprod[-1:]])   # (k,)


# ── Abstract base classes ────────────────────────────────────────────────────

class MassComponent(ABC):
    """Interface for normalised primary-mass distributions.

    Subclasses implement ``_eval_unnorm(m, theta)`` only.  The base class
    computes the normalisation integral over the configured mass grid and then
    returns a probability density for arbitrary scalar or array inputs.
    """

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        """Parameter metadata in the order consumed by this component."""
        ...

    @property
    def n_params(self) -> int:
        """Number of free parameters consumed by this component."""
        return len(self.param_specs)

    @property
    def m1_support_max(self) -> float:
        """Largest primary mass at which this component can have non-zero density.

        Used to size the opt-in pairing normalisation grid so it never clamps
        inside the model's support (see
        :func:`~darksirens.gw.populations.utils.size_pairing_grid_to_support`).
        Defaults to the conventional ``M_HI`` ceiling; components whose support
        extends higher (a fixed high-mass edge, or a sampled ``m_max`` prior
        reaching past ``M_HI``) override this with a truthful value.
        """
        return float(M_HI)

    @abstractmethod
    def _eval_unnorm(self, m, theta):
        """Evaluate the component's unnormalised primary-mass density."""
        ...

    def _norm(self, theta) -> jnp.ndarray:
        """
        Normalisation integral over MASS_GRID.
        Depends only on theta — call once per proposal, reuse across samples.
        """
        mass_grid = get_mass_grid()
        return jnp.trapezoid(self._eval_unnorm(mass_grid, theta), mass_grid)

    def __call__(self, m, theta, norm=None):
        """Evaluate the normalised density at mass ``m``.

        ``norm`` may be supplied by callers that already computed the component
        normalisation for the same ``theta``; this avoids repeated grid
        integrations inside vectorised population evaluations.
        """
        p = self._eval_unnorm(m, theta)
        n = norm if norm is not None else self._norm(theta)
        return p / jnp.where(n > 0, n, 1.0)


class PairingModel(ABC):
    """Interface for conditional mass-ratio distributions ``p(q | m1)``.

    Pairing normalisation depends on the primary mass because the secondary-mass
    cut is imposed through ``m2 = q * m1``.  The base implementation therefore
    integrates over the mass-ratio grid for each supplied ``m1``.
    """

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        """Parameter metadata in the order consumed by this pairing model."""
        ...

    @property
    def n_params(self):
        """Number of free parameters consumed by this pairing model."""
        return len(self.param_specs)

    @abstractmethod
    def _eval_unnorm(self, m1, q, m_min, dm_min, theta):
        """Evaluate the unnormalised conditional mass-ratio density."""
        ...

    def __call__(self, m1, q, m_min, dm_min, theta):
        p = self._eval_unnorm(m1, q, m_min, dm_min, theta)
        # OPT-IN accuracy knob (STATIC branch on the module-global setting, read
        # at trace time): default None keeps the EXACT per-sample q-integration
        # below (bit-identical to the historical code path); an int precomputes
        # the normaliser once on a static m1 grid and interpolates it per sample.
        n_grid = normalization_grid_settings().pairing_m1_grid
        if n_grid is None:
            # PairingModel norm integrates over q for each m1 — sample-dependent,
            # cannot be lifted out of the per-sample loop.
            m1_exp  = jnp.expand_dims(jnp.atleast_1d(m1), axis=-1)
            q_grid  = get_q_grid()
            p_grid  = self._eval_unnorm(m1_exp, q_grid, m_min, dm_min, theta)
            # Scale-invariant normalisation: for an m1 in the low-mass taper toe
            # both p and its q-integral are ~exp(-500)-tiny; the RATIO is well
            # conditioned, but a direct p / n has divide's VJP square n, which
            # UNDERFLOWS to zero and turns the cotangent into inf -> NaN
            # (measured: one bright-mock injection at m1src = m_min + 0.01
            # poisoned the whole d logL/d(m_min, dm_min, beta) gradient).
            # Factoring the row maximum out of the quadrature keeps every
            # backward division at O(1) scale; the forward value is the same
            # ratio up to association order (ULP-level).
            scale   = jnp.max(p_grid, axis=-1, keepdims=True)
            scale_s = jnp.where(scale > 0, scale, 1.0)
            n_sc    = jnp.trapezoid(p_grid / scale_s, q_grid, axis=-1)
            n_sc    = n_sc.reshape(jnp.shape(m1))
            scale_m = scale_s[..., 0].reshape(jnp.shape(m1))
            return jnp.where(
                n_sc > 0, (p / scale_m) / jnp.where(n_sc > 0, n_sc, 1.0), 0.0
            )
        # GRID path: the q-support depends on m1 only through the m2 = q*m1 cut,
        # so N(m1) = ∫ p_unnorm(m1, q) dq is a smooth 1-D function.  Precompute it
        # ONCE per proposal on the static log-spaced m1 grid (N_grid x N_Q work,
        # theta-traced) and interpolate log N in log m1 per sample.
        m1_grid = get_pairing_m1_grid()                     # (N_grid,) static nodes
        log_m1_grid = jnp.log(m1_grid)
        q_grid  = get_q_grid()
        # Same scale-factored quadrature as the exact branch, per grid node, so
        # the grid normaliser never underflows while forming I = scale * n_sc.
        p_grid  = self._eval_unnorm(m1_grid[:, None], q_grid[None, :],
                                    m_min, dm_min, theta)    # (N_grid, N_Q)
        scale   = jnp.max(p_grid, axis=-1, keepdims=True)
        scale_s = jnp.where(scale > 0, scale, 1.0)
        n_sc_g  = jnp.trapezoid(p_grid / scale_s, q_grid, axis=-1)
        I_grid  = scale_s[..., 0] * n_sc_g                   # (N_grid,) normaliser
        # Log-space with a tiny floor (the eval_dark_member_completion idiom): a
        # zero-support node (I==0, i.e. m1<=m_min so p_unnorm==0 for every q)
        # floors to log(1e-300) rather than -inf, keeping jnp.interp finite when
        # a sample straddles the support edge.  A true -inf would poison interp
        # (-inf + inf*t -> NaN); the floor is safe because a sample in that
        # region has p==0 too, so the density below is exactly 0 either way.
        log_I_grid = jnp.log(jnp.maximum(I_grid, 1e-300))
        # Linear interp of log N in log m1 (jnp.interp clamps out-of-range x to
        # the grid ends).  N = exp(log_I): density = p / N = p * exp(-log_I),
        # a SINGLE reciprocal so the VJP never squares N (same reason the exact
        # branch factors out scale_m).  Zero-support samples have p==0 exactly,
        # so p * exp(-log_I) == 0, matching the exact branch's guarded 0.
        log_I   = jnp.interp(jnp.log(jnp.atleast_1d(m1)), log_m1_grid, log_I_grid)
        log_I   = log_I.reshape(jnp.shape(m1))
        return p * jnp.exp(-log_I)


class SpinModel(ABC):
    """Interface for normalised effective-spin distributions.

    Subclasses define an unnormalised density in ``chieff``.  The base class
    integrates over the configured spin grid and returns a density suitable for
    multiplication with mass and pairing factors.
    """

    @property
    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        """Parameter metadata in the order consumed by this spin model."""
        ...

    @property
    def n_params(self):
        """Number of free parameters consumed by this spin model."""
        return len(self.param_specs)

    @abstractmethod
    def _eval_unnorm(self, chieff, theta):
        """Evaluate the component's unnormalised effective-spin density."""
        ...

    def _norm(self, theta) -> jnp.ndarray:
        """
        Normalisation integral over CHI_GRID.
        Depends only on theta — call once per proposal, reuse across samples.
        """
        chi_grid = get_chi_grid()
        return jnp.trapezoid(self._eval_unnorm(chi_grid, theta), chi_grid)

    def __call__(self, chieff, theta, norm=None):
        """Evaluate the normalised effective-spin density at ``chieff``."""
        p = self._eval_unnorm(chieff, theta)
        n = norm if norm is not None else self._norm(theta)
        return p / jnp.where(n > 0, n, 1.0)


# ── Mixture model ────────────────────────────────────────────────────────────

@dataclass
class MixtureModel:
    """Stick-breaking mixture of mass, pairing, and spin components.

    A mixture with ``k`` mass components consumes all component parameters plus
    ``k - 1`` stick-breaking variables.  Pairing and spin components can either
    be shared across all mass components or supplied one-per-component.  The
    final component receives the remaining stick by construction, so mixture
    weights are always non-negative and sum to one.
    """

    mass_components:    list[MassComponent]
    pairing_components: list[PairingModel]
    spin_components:    list[SpinModel]

    def __post_init__(self):
        self.k             = len(self.mass_components)
        self.shared_pairing = len(self.pairing_components) == 1
        self.shared_spin    = len(self.spin_components)    == 1

        if not self.shared_pairing and len(self.pairing_components) != self.k:
            raise ValueError(
                f"Expected {self.k} or 1 pairing components, got {len(self.pairing_components)}"
            )
        if not self.shared_spin and len(self.spin_components) != self.k:
            raise ValueError(
                f"Expected {self.k} or 1 spin components, got {len(self.spin_components)}"
            )

    @property
    def n_weight_params(self):
        """Number of stick-breaking variables required for this mixture."""
        return max(self.k - 1, 0)

    @property
    def param_specs(self):
        """Return weight, mass, pairing, and spin parameter specs in order."""
        # v_i are stick-breaking inputs, bounded [0, 1].
        specs = [
            ParamSpec(rf"$v_{i+1}$", 0.0, 1.0, name=f"v{i+1}")
            for i in range(self.n_weight_params)
        ]
        for c in self.mass_components:    specs.extend(c.param_specs)
        for c in self.pairing_components: specs.extend(c.param_specs)
        for c in self.spin_components:    specs.extend(c.param_specs)
        return specs

    @property
    def n_params(self):
        """Total number of free mixture parameters."""
        return len(self.param_specs)

    def _split_theta(self, theta):
        """Split the flat mixture vector into weights and per-component slices.

        Returns ``(w, tm_list, tp_list, ts_list)`` where ``w`` are the
        stick-breaking mixture weights and the lists hold the mass, pairing,
        and spin sub-vectors in component order.
        """
        n_w = self.n_weight_params

        # Stick-breaking: all weights guaranteed ≥ 0, Σ = 1.
        w = _stick_breaking_weights(theta[:n_w]) if n_w > 0 else jnp.array([1.0])

        # Slice per-component sub-vectors.
        idx = n_w
        tm_list, tp_list, ts_list = [], [], []

        for c in self.mass_components:
            tm_list.append(theta[idx : idx + c.n_params]); idx += c.n_params
        for c in self.pairing_components:
            tp_list.append(theta[idx : idx + c.n_params]); idx += c.n_params
        for c in self.spin_components:
            ts_list.append(theta[idx : idx + c.n_params]); idx += c.n_params

        return w, tm_list, tp_list, ts_list

    def spin_theta(self, theta):
        """Return the parameter slice of the shared spin component.

        Only meaningful for shared-spin mixtures, where the population
        factorises as ``p(m1, q) * p(chieff)``; callers that sample the
        spin dimension separately (the flow-surrogate path) rely on this.
        """
        if not self.shared_spin:
            raise NotImplementedError(
                "spin_theta requires a shared spin component; per-component "
                "spin models do not factorise out of the mass mixture."
            )
        _, _, _, ts_list = self._split_theta(theta)
        return ts_list[0]

    def component_densities(self, m1, q, chieff, theta):
        """Return weighted source-density contributions for each component.

        The leading axis indexes mass-mixture components; summing over that
        axis reproduces :meth:`__call__`.  Exposing contributions before the
        mixture sum lets callers apply component-specific factors, such as a
        per-component redshift evolution, without changing the source-density
        parameter ordering.
        """
        w, tm_list, tp_list, ts_list = self._split_theta(theta)

        # Normalisation integrals — depend only on theta, not on samples.
        # Lifted out of the per-sample loop; pairing norm stays inside (m1-dependent).
        mass_norms = [c._norm(tm_list[i]) for i, c in enumerate(self.mass_components)]
        spin_norms = [
            c._norm(ts_list[0] if self.shared_spin else ts_list[i])
            for i, c in enumerate(self.spin_components)
        ]

        contributions = []
        for i in range(self.k):
            c_m  = self.mass_components[i];    tm = tm_list[i]
            c_p  = self.pairing_components[0 if self.shared_pairing else i]
            tp   = tp_list[0 if self.shared_pairing else i]
            c_s  = self.spin_components[0 if self.shared_spin else i]
            ts   = ts_list[0 if self.shared_spin else i]
            s_idx = 0 if self.shared_spin else i

            mmin, dmmin = M_LO, 0.01
            if hasattr(c_m, "m_min_spec"):
                mmin  = tm[c_m.param_specs.index(c_m.m_min_spec)]
            if hasattr(c_m, "dm_min_spec"):
                dmmin = tm[c_m.param_specs.index(c_m.dm_min_spec)]

            contributions.append(w[i] * (
                c_m(m1, tm, norm=mass_norms[i])
                * c_p(m1, q, mmin, dmmin, tp)
                * c_s(chieff, ts, norm=spin_norms[s_idx])
            ))

        return jnp.stack(contributions, axis=0)

    def mass_q_density(self, m1, q, theta):
        """Mixture density with the shared spin factor removed.

        Returns ``Σ_i w_i c_m,i(m1) c_p,i(q | m1)`` so that, for shared-spin
        mixtures, ``mass_q_density * spin_density == __call__``.  The
        flow-surrogate sampler draws (m1, q) from this 2-D density and chieff
        from the shared spin component separately.
        """
        if not self.shared_spin:
            raise NotImplementedError(
                "mass_q_density requires a shared spin component; with "
                "per-component spins the mass-q marginal is spin-coupled."
            )
        w, tm_list, tp_list, _ = self._split_theta(theta)

        mass_norms = [c._norm(tm_list[i]) for i, c in enumerate(self.mass_components)]

        total = 0.0
        for i in range(self.k):
            c_m = self.mass_components[i];    tm = tm_list[i]
            c_p = self.pairing_components[0 if self.shared_pairing else i]
            tp  = tp_list[0 if self.shared_pairing else i]

            mmin, dmmin = M_LO, 0.01
            if hasattr(c_m, "m_min_spec"):
                mmin  = tm[c_m.param_specs.index(c_m.m_min_spec)]
            if hasattr(c_m, "dm_min_spec"):
                dmmin = tm[c_m.param_specs.index(c_m.dm_min_spec)]

            total = total + w[i] * (
                c_m(m1, tm, norm=mass_norms[i])
                * c_p(m1, q, mmin, dmmin, tp)
            )
        return total

    def __call__(self, m1, q, chieff, theta):
        """Evaluate the normalised mixture density for source parameters."""
        return jnp.sum(self.component_densities(m1, q, chieff, theta), axis=0)


# ── Population model ─────────────────────────────────────────────────────────

@dataclass
class PopulationModel:
    """Full compact-binary population model used by the likelihood.

    The model combines a normalised source-parameter mixture with a redshift
    evolution term.  ``rate_evolution`` selects it:

    - ``"powerlaw"`` (default): ``(1 + z)**(gamma - 1)`` — the merger rate
      density R(z) ∝ (1+z)^gamma with the 1/(1+z) source-time dilation.
    - ``"md"``: Madau–Dickinson-like peaked rate (select via the ``@md``
      pop-model name decoration, e.g. ``powerlaw+peak@md``),

          psi(z) = (1+z)^gamma / (1 + ((1+z)/(1+z_peak))^(gamma+kappa)),

      applied as ``psi(z)/(1+z)``: rises like (1+z)^gamma below z_peak and
      falls like (1+z)^(-kappa) above it.  The conventional normalisation
      constant [1 + (1+z_peak)^-(gamma+kappa)] is a hyperparameter-dependent
      overall factor and is deliberately omitted: it multiplies the per-event
      weights and the selection integral mu(Lambda) identically, so it
      cancels exactly in the scale-free (rate-marginalised) likelihood.
      Appends (gamma, kappa, z_peak) to the parameter vector instead of
      gamma, and requires shared (global) redshift evolution.

    Registry entries create instances of this class and expose the
    corresponding prior bounds to samplers.
    """

    mixture: MixtureModel
    shared_gamma: bool = True
    rate_evolution: str = "powerlaw"

    def __post_init__(self):
        if self.rate_evolution not in ("powerlaw", "md"):
            raise ValueError(
                f"Unknown rate_evolution {self.rate_evolution!r}; "
                "expected 'powerlaw' or 'md'."
            )
        if self.rate_evolution == "md" and not (self.shared_gamma or self.mixture.k == 1):
            raise ValueError(
                "rate_evolution='md' requires shared redshift evolution "
                "(shared_gamma=True): per-component Madau-Dickinson rates are "
                "not supported."
            )

    def _component_gamma_spec(self, i: int) -> ParamSpec:
        """Return a redshift-slope spec tagged to mass component ``i``."""
        for spec in self.mixture.mass_components[i].param_specs:
            if spec.name and "." in spec.name:
                tag = spec.name.split(".", 1)[0]
                return ParamSpec(
                    rf"$\gamma_{{\rm {tag}}}$",
                    -10.0,
                    10.0,
                    name=f"{tag}.gamma",
                )

        # Fallback for custom components without tagged ASCII parameter names.
        j = i + 1
        return ParamSpec(
            rf"$\gamma_{{{j}}}$",
            -10.0,
            10.0,
            name=f"component{j}.gamma",
        )

    @property
    def gamma_param_specs(self):
        """Return the redshift-evolution parameter specs.

        Power law: shared or per-component gamma.  Madau-Dickinson: the
        shared triple (gamma, kappa, z_peak) — low-z slope, high-z falling
        slope, and turnover redshift.
        """
        if self.rate_evolution == "md":
            return [
                ParamSpec(r"$\gamma$", -10.0, 10.0, name="gamma"),
                ParamSpec(r"$\kappa$", 0.0, 10.0, name="kappa"),
                ParamSpec(r"$z_{\rm peak}$", 0.2, 4.0, name="z_peak"),
            ]
        if self.shared_gamma or self.mixture.k == 1:
            return [ParamSpec(r"$\gamma$", -10.0, 10.0, name="gamma")]
        return [self._component_gamma_spec(i) for i in range(self.mixture.k)]

    @property
    def param_specs(self):
        """Return mixture parameter specs followed by rate-evolution params."""
        return [*self.mixture.param_specs, *self.gamma_param_specs]

    def prior_bounds(self):
        """Return lower bounds, upper bounds, and labels for all parameters."""
        return pack_specs(*self.param_specs)

    @property
    def has_additive_rate_split(self):
        """Whether ``log_p_pop == log_p_massspin + log_rate_z`` holds.

        True for shared redshift evolution (powerlaw with shared gamma, or
        Madau-Dickinson); False for per-component gamma, where the z factor
        couples to the mixture sum and no split exists.
        """
        return self.rate_evolution == "md" or self.shared_gamma or self.mixture.k == 1

    def mixture_theta(self, theta):
        """Return the mixture slice of the full parameter vector."""
        if self.rate_evolution == "md":
            return theta[:-3]
        if self.shared_gamma or self.mixture.k == 1:
            return theta[:-1]
        return theta[: self.mixture.n_params]

    def log_rate_z(self, z, theta):
        """Log redshift-evolution factor of the population density.

        The additive counterpart to :meth:`log_p_massspin`:
        ``log_p_pop = log_p_massspin + log_rate_z`` whenever
        :attr:`has_additive_rate_split` is True.
        """
        if self.rate_evolution == "md":
            gamma = theta[-3]
            kappa = theta[-2]
            z_pk  = theta[-1]
            # log[psi(z)/(1+z)] with psi the (unnormalised) Madau-Dickinson
            # form; softplus keeps the turnover term stable at any slope.
            log_turnover = jnp.logaddexp(
                0.0, (gamma + kappa) * (jnp.log1p(z) - jnp.log1p(z_pk))
            )
            return (gamma - 1.0) * jnp.log1p(z) - log_turnover

        if self.shared_gamma or self.mixture.k == 1:
            gamma = theta[-1]
            return (gamma - 1.0) * jnp.log1p(z)

        raise NotImplementedError(
            "log_rate_z is undefined for per-component redshift evolution: "
            "the z factor does not separate from the mixture sum."
        )

    def log_p_massspin(self, m1, q, chieff, theta):
        """Log source-parameter (mass, mass-ratio, spin) mixture density.

        Sentinel: p = 0  →  log p = −jnp.inf, matching :meth:`log_p_pop`.
        """
        if not self.has_additive_rate_split:
            raise NotImplementedError(
                "log_p_massspin is undefined for per-component redshift "
                "evolution: use log_p_pop."
            )
        p = self.mixture(m1, q, chieff, self.mixture_theta(theta))
        return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)

    def log_p_pop(self, m1, q, z, chieff, theta):
        """
        Log population probability at (m1, q, z, chieff) under parameters theta.

        Sentinel: p = 0  →  log p = −jnp.inf  (not −1e10).
        −∞ propagates correctly through logsumexp / jnp.sum; the final
        jnp.isfinite guard in the likelihood rejects the proposal cleanly.
        """
        if self.has_additive_rate_split:
            return (
                self.log_p_massspin(m1, q, chieff, theta)
                + self.log_rate_z(z, theta)
            )

        n_mix  = self.mixture.n_params
        tm     = theta[:n_mix]
        gamma  = theta[n_mix : n_mix + self.mixture.k]
        p_comp = self.mixture.component_densities(m1, q, chieff, tm)

        gamma_shape = (self.mixture.k,) + (1,) * (jnp.ndim(p_comp) - 1)
        z_factor = jnp.power(1.0 + z, gamma.reshape(gamma_shape) - 1.0)
        p = jnp.sum(p_comp * z_factor, axis=0)
        return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
