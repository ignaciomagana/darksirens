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

Prior bounds on the weight parameters stay [0, 1], as do the CLI flags and
sampler configuration.  The PRIOR FAMILY on them is Beta(1, k-i) for v_i, which
is what makes the induced weight prior the uniform Dirichlet(1, ..., 1) the old
"last weight = 1 - Σ" parameterisation had (uniform on the box conditioned on the
simplex IS uniform on the simplex).  Sampling every v_i from U[0, 1] instead —
as this file did until the stick prior was declared — gives
E[w] = (1/2, 1/4, 1/8, ...): informative, and dependent on the ORDER the
components appear in the model name.  k = 2 is unaffected (Beta(1, 1) = U[0, 1]).

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
from jax import lax

from .utils import (
    get_mass_grid,
    get_chi_grid,
    get_pairing_m1_grid,
    get_pairing_edge_quadrature,
    get_pairing_panel_quadrature,
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

    @property
    def support(self) -> tuple[float | None, float | None]:
        """Explicit support edges the common call path masks the density to.

        A ``None`` edge means "this component owns that edge itself" -- either
        because ``_eval_unnorm`` already returns zero there (the tapered power
        laws), or because the model DELIBERATELY has no finite edge on that
        side.  Both cases exist and neither may be blindly masked, which is
        why the metadata is explicit rather than inferred.

        The default is the NORMALISATION grid's own span: the base
        :meth:`_norm` integrates exactly ``[m_lo, m_hi]``, so density outside
        it is density the normaliser never counted.  Nothing masked it, and
        the grammar Gaussian is positive for every finite mass: at the
        fixed/custom corner ``mu = 100``, ``sigma = 20`` it returned
        ``p_G(201) = 5.78e-8`` per solar mass with ``2.87e-7`` of nominal
        normalisation leaked above 200 (and ``3.71e-7`` below M_LO = 1),
        scored against a normaliser that integrated neither (review PHY-08).
        PE weights carry no compensating mass mask, so the leak is a real
        (if, under the shipped priors, tiny) density error.
        """
        settings = normalization_grid_settings()
        return float(settings.m_lo), float(settings.m_hi)

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

    def _mask_to_support(self, m, dens):
        """Zero ``dens`` outside :attr:`support`.

        ``support`` is Python floats/None, so both bounds are STATIC: the
        branches below are taken at trace time and a component with no finite
        edges costs nothing.
        """
        lo, hi = self.support
        if lo is None and hi is None:
            return dens
        in_support = None
        if lo is not None:
            in_support = m >= lo
        if hi is not None:
            above = m <= hi
            in_support = above if in_support is None else (in_support & above)
        return jnp.where(in_support, dens, jnp.zeros_like(dens))

    def __call__(self, m, theta, norm=None):
        """Evaluate the normalised density at mass ``m``.

        ``norm`` may be supplied by callers that already computed the component
        normalisation for the same ``theta``; this avoids repeated grid
        integrations inside vectorised population evaluations.
        """
        p = self._eval_unnorm(m, theta)
        n = norm if norm is not None else self._norm(theta)
        return self._mask_to_support(m, p / jnp.where(n > 0, n, 1.0))


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

    def _taper_shoulder(self, m_min, dm_min, theta):
        r"""Lower support edge and taper shoulder of THIS pairing's own filter.

        Returns ``(m_edge, m_shoulder)`` in SECONDARY-mass units: ``_eval_unnorm``
        is identically zero below ``m_edge`` and its taper is identically one at
        and above ``m_shoulder``, so the two mark the ends of the boundary layer
        the q-quadrature has to resolve.

        The default reads the ``(m_min, dm_min)`` the caller passes, which is what
        :class:`PowerLawPairing` and :class:`GaussianPairing` taper on.  A
        subclass whose ``_eval_unnorm`` tapers on parameters of its OWN ``theta``
        -- :class:`GWTC5FiducialBPL2PeaksPairing` does exactly that: it deletes
        ``m_min``/``dm_min`` and uses ``(m2_low, delta_m2)`` -- MUST override this,
        because the panel split below is only exact when it lands on the shoulder
        the integrand actually has.  Nothing forces the two to agree: the
        production caller happens to pass ``m2_low`` as ``m_min``, but
        ``MixtureModel.component_densities`` hands a shared pairing the MASS
        component's ``(mmin, dmmin)``.
        """
        del theta
        return m_min, m_min + dm_min

    def _panel_edges(self, m1, m_min, dm_min, theta):
        r"""EXTRA interior split points of the q-quadrature (default: none).

        THE CONTRACT THIS HOOK EXISTS FOR.  The default split (see
        :meth:`_panel_nodes`) puts ``PAIRING_PANEL_NQ`` Gauss-Legendre nodes on
        the taper boundary layer and the same number on everything above the
        shoulder, and Gauss-Legendre is only near-exact on a panel the integrand
        is SMOOTH over on the scale of that panel's width.  Both production
        pairings clear that easily: above the shoulder they are a bare
        ``q**beta``, which 16 nodes integrate to ~1e-16.  A kernel with a
        FEATURE NARROWER THAN ITS PANEL does not, and it fails silently -- the
        rule simply misses the feature.  Such a subclass MUST declare the
        feature's edges here.  Measured, ``GaussianPairing`` before it did
        (:meth:`GaussianPairing._panel_edges`): worst 2.7e+01 nats off a
        Gauss-Kronrod reference over nine corners spanning
        ``sigma_q`` in [0.001, 0.5], against 6.7e-7 with the edges declared;
        at ``mu_q = 0.5``, ``sigma_q = 0.02``, m_min = 5, dm_min = 3, m1 = 60
        it was 2.7e-1 against 9.8e-8.

        Returns a tuple of q values (scalars or arrays broadcastable against
        ``m1``); they are clipped into the support and sorted, so neither their
        order nor whether they land inside the taper matters.  A tuple of length
        k costs ``k`` extra panels of ``PAIRING_PANEL_NQ`` nodes per sample, so
        declare only edges the integrand actually needs.
        """
        del m1, m_min, dm_min, theta
        return ()

    def _panel_nodes(self, m1, m_min, dm_min, theta, t):
        r"""Support-relative q panels, split at the taper shoulder.

        The q-support is ``(q_cut, 1]`` with ``q_cut = m_edge/m1``: it depends on
        a SAMPLED parameter and on the query m1.  Nodes are therefore placed
        RELATIVE to the support, ``q = q_cut + t w``, so they follow the edge
        instead of being crossed by it -- the same trick
        ``GWTC5FiducialBPL2PeaksMass._taper_window_grid`` uses for the mass taper.

        With the historical FIXED grid the normaliser
        ``N(m1) = int p(q|m1) dq`` was a staircase in both ``m1`` and ``m_min``
        (it only changed when ``q_cut`` crossed a node): measured on
        ``PowerLawPairing`` at m_min = 5, dm_min = 0.01, q = 0.9,
        ``p(q|m1)`` was bit-identical over m1 in [60, 62] while the true N grows
        3%, and ``d log p/d m_min`` was 0 almost everywhere with spurious +1.37
        spikes wherever a node happened to land inside the taper window (against a
        true value of ~0.03).  Support-relative nodes also spend every node inside
        the support, so the edge itself is resolved for any m1.

        The support is then SPLIT at ``q_a = m_shoulder/m1`` (clipped into
        ``[q_cut, 1]``).  ``p(q|m1)`` is a Planck taper rising from an exactly-zero
        edge on ``[q_cut, q_a]``, and the bare pairing kernel -- ``q**beta`` for
        both production models, the taper being identically 1.0 there
        (``utils.sfilter_low`` returns a literal one at and above the shoulder) --
        on ``[q_a, 1]``.  The joint integrand has a corner at ``q_a`` that neither
        piece has, and a corner is exactly what a single smooth rule cannot
        resolve.

        What the split buys is NOT mainly a smaller worst case -- 16 nodes per
        panel are 7.6e-3 nats off near the support edge against the 200-node
        uniform trapezoid's 3.1e-2 (50 prior draws, composite-GL reference), only
        4x -- but a far smaller COHERENT error.  The trapezoid's residual is a
        one-sided endpoint deficit that does not self-average over the m1
        population, so it tilts with H0 (m1src = m1det/(1+z(H0)) rescales the
        whole population): measured over 259 events, the H0-correlated part of
        the residual is 7.4e-3 .. 1.6e-1 nats for the trapezoid and
        1.4e-4 .. 9.6e-4 nats for this rule, and on the real 259-event dark-siren
        likelihood the trapezoid carries a -0.10-nat monotone slope across the
        H0 prior [20, 140] where this rule carries +1.2e-3.  Amplitude is the
        wrong statistic here; coherence is.

        Either panel may collapse to zero width -- ``dm_min = 0`` kills panel A,
        ``m_shoulder >= m1`` kills panel B, ``m_edge >= m1`` kills both -- and a
        zero-width panel contributes exactly 0.0 to the sum (a finite integrand
        times a zero width), so no branch is needed for them.

        A subclass whose kernel carries a feature narrower than a panel declares
        that feature's edges through :meth:`_panel_edges` and gets one panel per
        extra edge; the two-panel split above is what every model without such a
        feature -- both production pairings -- takes, unchanged.

        A subclass that knows the CLOSED FORM of the topmost panel's integral
        declares it through :meth:`_plateau_integral`, and that panel then costs
        no nodes at all.  Both production pairings do; it is where half the
        nodes and two thirds of the transcendentals of this rule went.

        Returns a tuple of ``(nodes, width)`` panels, ``nodes`` of shape
        ``m1.shape + (len(t),)``.  The panel BOUNDARIES themselves are built by
        :meth:`_panel_boundaries`, which :meth:`_panel_norm` also needs on its
        own so it can hand the last panel's lower edge to
        :meth:`_plateau_integral`.
        """
        edges = self._panel_boundaries(m1, m_min, dm_min, theta)
        return tuple(self._panel_from_edges(lo, hi, t)
                     for lo, hi in zip(edges[:-1], edges[1:]))

    @staticmethod
    def _panel_from_edges(lo, hi, t):
        """One ``(nodes, width)`` panel: ``len(t)`` nodes mapped onto ``[lo, hi]``."""
        width = hi - lo
        return lo[..., None] + t * width[..., None], width

    def _panel_boundaries(self, m1, m_min, dm_min, theta):
        """Ordered q panel edges; ``len(edges) - 1`` panels tile ``(q_cut, 1]``.

        See :meth:`_panel_nodes` for what the split is and why.  The last entry
        is the PYTHON float ``1.0``, not an array of ones, so the last width
        stays the literal ``1.0 - q_a`` the two-panel rule formed.
        """
        m_edge, m_shoulder = self._taper_shoulder(m_min, dm_min, theta)
        # m1 == 0 would make m_edge/m1 an inf whose clip VJP is 0 * inf = NaN,
        # poisoning d/d(m_min) AND d/d(dm_min) for the whole row (measured:
        # [nan, 0.00815, -0.282] before the panel split, [nan, nan, -0.282]
        # with it, [0.0159, 0.00815, -0.282] here).  No shipped path feeds a
        # zero primary mass and the returned density there is exactly 0.0 either
        # way (m2 = q*m1 = 0 < m_min kills every node), but the double-where
        # keeps the gradient finite for a padded store.  It is the ONE piece of
        # this file that is not bit-identical to the pre-2026-09-06 arithmetic:
        # the extra select changes how XLA fuses the division on GPU, which
        # moves the 259-event production logL by 2.3e-10 nats (1.4e-16 relative,
        # one ulp; measured A/B on an H100 NVL with and without this line, the
        # rest of the panel-list restructure bit-identical in both arms).  CPU
        # is bit-identical, so no golden moves.
        safe_m1 = jnp.where(m1 > 0.0, m1, 1.0)
        q_cut = jnp.clip(m_edge / safe_m1, 0.0, 1.0)
        q_a   = jnp.clip(m_shoulder / safe_m1, q_cut, 1.0)
        extra = self._panel_edges(m1, m_min, dm_min, theta)
        if extra:
            # A subclass declared extra interior split points.  Clip them into
            # the support and SORT, so the panels stay ordered whatever the
            # relative position of the declared feature and the taper shoulder
            # is; a split point that falls outside (q_cut, q_a] collapses onto
            # an edge and its panel gets zero width.  The default path takes
            # neither the sort nor the stack: the edges are ordered by
            # construction there, so the hot path is arithmetically unchanged.
            cuts = jnp.sort(
                jnp.stack(
                    [q_cut, q_a]
                    + [jnp.clip(e + jnp.zeros_like(q_cut), q_cut, 1.0)
                       for e in extra],
                    axis=-1),
                axis=-1)
            return [cuts[..., i] for i in range(cuts.shape[-1])] + [1.0]
        return [q_cut, q_a, 1.0]

    def _plateau_integral(self, m1, q_lo, m_min, dm_min, theta):
        r"""CLOSED FORM for the topmost panel ``[q_lo, 1]``, or ``None``.

        Above the taper shoulder the low-mass filter is identically 1.0
        (``utils.sfilter_low`` returns a literal one there), so on the last panel
        the integrand is the BARE pairing kernel -- for both production models a
        plain ``q**beta``, whose integral is one line of algebra.  A subclass
        that knows that closed form returns ``(integral, sup)``:

        * ``integral`` = ``int_{q_lo}^{1} p_unnorm(q | m1) dq`` exactly, and
        * ``sup`` = an upper bound on ``p_unnorm`` over ``[q_lo, 1]``, which
          replaces the panel's node maximum in the scale factoring of
          :meth:`_panel_norm` (there are no nodes left to take a maximum over).
          It MUST be 0.0 wherever the panel has zero width, or a row whose whole
          integrand is ~exp(-500)-tiny in the taper toe would be rescaled by an
          O(1) bound and lose exactly the underflow protection the factoring
          exists for.

        Returning ``None`` -- the default -- keeps the Gauss-Legendre panel, so a
        pairing without an analytic plateau (``GaussianPairing``) and any
        out-of-tree subclass are unaffected.  The hook is handed the last panel's
        lower edge rather than the shoulder itself, so it stays correct for a
        subclass that declares extra edges through :meth:`_panel_edges` above the
        shoulder: those are sorted into the edge list, so ``q_lo >= q_a`` always
        and the kernel is bare over the whole panel either way.

        Worth doing because the plateau panel is half the per-sample quadrature
        and the more expensive half: a node there costs a ``log`` and two ``exp``
        (``q**beta`` is ``exp(beta log q)``, the Planck taper another ``exp``)
        against ONE ``exp`` for the whole closed form.  Measured on an H100 NVL,
        float64, three interleaved launches per arm, 20 timed calls each,
        replacing this panel removes 0.69 ms of a 3.57 ms 259-event spectral
        likelihood call (1.24x, and 0.776 -> 0.640 GiB of peak device memory) and
        0.90 ms of a 47.49 ms 259-event dark-siren call (1.02x).

        It is not only cheaper.  At steep negative ``beta`` the plateau integrand
        is itself a boundary layer at ``q_lo = m_shoulder/m1``, which shrinks like
        1/m1, so Gauss-Legendre's residual there GROWS with m1 and is coherent
        across the mass population -- exactly the shape that tilts with H0, since
        ``m1src = m1det/(1+z(H0))`` rescales the whole population.  Measured on
        the 259-event dark-siren likelihood at
        ``(beta_q, m2_low, delta_m2) = (-1.9, 3.05, 1.15)``, inside the shipped
        prior: 0.93 nats peak-to-peak across H0 in [20, 140] with a -1.04-nat
        slope for the GL panel, against 2.0e-4 nats and -2.8e-5 for this.
        """
        del m1, q_lo, m_min, dm_min, theta
        return None

    def _panel_norm(self, m1, m_min, dm_min, theta, t, w):
        r"""Scale-factored panel-sum normaliser ``(n_sc, scale)``; ``N = scale n_sc``.

        Scale-invariant normalisation: for an m1 in the low-mass taper toe both
        ``p`` and its q-integral are ~exp(-500)-tiny; the RATIO is well
        conditioned, but a direct ``p / N`` has divide's VJP square ``N``, which
        UNDERFLOWS to zero and turns the cotangent into inf -> NaN (measured: one
        bright-mock injection at m1src = m_min + 0.01 poisoned the whole
        ``d logL/d(m_min, dm_min, beta)`` gradient).  Factoring the row maximum
        over EVERY panel's nodes out of the quadrature keeps every backward
        division at O(1) scale; the forward value is the same integral up to
        association order (ULP-level).

        When the class supplies a :meth:`_plateau_integral` the topmost panel is
        taken in closed form instead of by Gauss-Legendre: its exact integral is
        added to the scaled sum and its analytic supremum joins the maximum, so
        the scale still bounds the whole integrand and still cancels out of
        ``(p / scale) / n_sc``.  Every other panel's arithmetic is untouched.
        """
        edges = self._panel_boundaries(m1, m_min, dm_min, theta)
        # Closed form for the LAST panel, if this class has one; edges[-2] is
        # its lower edge and edges[-1] is the literal 1.0.
        closed = self._plateau_integral(m1, edges[-2], m_min, dm_min, theta)
        n_quad = len(edges) - 1 - (1 if closed is not None else 0)
        vals = [(self._eval_unnorm(m1[..., None], nodes, m_min, dm_min, theta),
                 width)
                for nodes, width in (self._panel_from_edges(lo, hi, t)
                                     for lo, hi in zip(edges[:n_quad],
                                                       edges[1:n_quad + 1]))]
        scale = jnp.max(vals[0][0], axis=-1, keepdims=True)
        for p_i, _ in vals[1:]:
            scale = jnp.maximum(scale, jnp.max(p_i, axis=-1, keepdims=True))
        if closed is not None:
            scale = jnp.maximum(scale, closed[1][..., None])
        scale_s = jnp.where(scale > 0, scale, 1.0)
        n_sc = jnp.sum(w * (vals[0][0] / scale_s), axis=-1) * vals[0][1]
        for p_i, width in vals[1:]:
            n_sc = n_sc + jnp.sum(w * (p_i / scale_s), axis=-1) * width
        if closed is not None:
            n_sc = n_sc + closed[0] / scale_s[..., 0]
        return n_sc, scale_s[..., 0]

    def __call__(self, m1, q, m_min, dm_min, theta):
        p = self._eval_unnorm(m1, q, m_min, dm_min, theta)
        # SUPPORT MASK.  Every normaliser below integrates over (q_cut, 1] --
        # the q-support implied by the m2 = q*m1 >= m_min cut and the m2 <= m1
        # labelling convention -- but the concrete _eval_unnorm implementations
        # only require q > 0 and m2 >= m_min.  A row with m2 > m1 therefore
        # received a finite density from OUTSIDE the domain the division
        # normalises over: measured on PowerLawPairing at beta = 1, m_min = 5,
        # dm_min = 3, m1 = 30, the returned density integrated to 1.0000 over
        # (q_cut, 1] yet still handed p = 2.52 at q = 1.2 and p = 4.20 at
        # q = 2, i.e. total mass well above one (review PHY-09).  q = 1 is IN
        # support (equal masses are physical); q > 1 and q <= 0 are not.  The
        # densities are finite there, so a plain where is NaN-safe in the VJP.
        in_support = (q > 0.0) & (q <= 1.0)
        # OPT-IN accuracy knob (STATIC branch on the module-global setting, read
        # at trace time): default None keeps the EXACT per-sample q-integration
        # below; an int precomputes the normaliser once on a static m1 grid and
        # interpolates it per sample.
        settings = normalization_grid_settings()
        n_grid = settings.pairing_m1_grid
        if n_grid is None:
            # PairingModel norm integrates over q for each m1 — sample-dependent,
            # cannot be lifted out of the per-sample loop.
            m1_a    = jnp.atleast_1d(m1)
            # TWO Gauss-Legendre panels split at the taper shoulder (see
            # _panel_nodes / _panel_norm): 2 x 16 nodes per sample instead of the
            # historical n_q = 200 uniform trapezoid, because the split removes
            # the one corner the integrand has and leaves GL two analytic
            # pieces.  That is 4x on the worst-case discretisation error and
            # 100-170x on its H0-COHERENT part, which is the statistic the
            # cosmology sees -- see _panel_nodes for both measurements and their
            # configuration.
            t_p, w_p = get_pairing_panel_quadrature()    # (16,) static nodes
            n_sc, scale_p = self._panel_norm(m1_a, m_min, dm_min, theta, t_p, w_p)
            n_sc    = n_sc.reshape(jnp.shape(m1))
            scale_m = scale_p.reshape(jnp.shape(m1))
            return jnp.where(
                in_support & (n_sc > 0),
                (p / scale_m) / jnp.where(n_sc > 0, n_sc, 1.0),
                0.0,
            )
        # GRID path: the q-support depends on m1 only through the m2 = q*m1 cut,
        # so N(m1) = ∫ p_unnorm(m1, q) dq is a smooth 1-D function.  Precompute it
        # ONCE per proposal on the static log-spaced m1 grid (N_grid x N_Q work,
        # theta-traced) and interpolate log N in log m1 per sample.
        m1_grid = get_pairing_m1_grid()                     # (N_grid,) static nodes
        log_m1_grid = jnp.log(m1_grid)
        # Same scale-factored, SUPPORT-RELATIVE 2-panel quadrature as the exact
        # branch, per grid node, so the grid normaliser never underflows while
        # forming I = scale * n_sc and agrees with the exact branch node-for-node.
        t_p, w_p = get_pairing_panel_quadrature()
        n_sc_g, scale_g = self._panel_norm(m1_grid, m_min, dm_min, theta,
                                           t_p, w_p)
        I_grid  = scale_g * n_sc_g                           # (N_grid,) normaliser
        # SUPPORT-EDGE HANDLING.  Nodes with no support have I == 0 exactly (for
        # every q, p_unnorm == 0 -- e.g. m1 <= m_min, so m2 = q*m1 <= m_min).  The
        # historical code floored those to log(1e-300) and interpolated THROUGH
        # them, which is only sound AT a zero-support node, never INSIDE the cell
        # that straddles the edge: there p is small-but-nonzero while the
        # interpolated log_I is hundreds of nats below the truth, so
        # p * exp(-log_I) EXPLODES (measured on powerlaw+peak at the prior
        # midpoint: log density +547.96 nats too large at m1 = m_lo * (1+2.6e-5),
        # which took one injection's log_mu from -4.485 to +360.4 and logL from
        # -1027.3 to -114483.4).
        #
        # Fix: never interpolate through the floor.  A floored node inherits the
        # LARGER of its two nearest supported neighbours' normalisers, so a cell
        # touching the edge interpolates between real (finite) normalisers.  I(m1)
        # is monotone toward a support edge (both the q-domain [m_min/m1, 1] and
        # the taper S(q*m1) grow with m1), hence the inherited I is an UPPER bound
        # on the true I inside that cell and the returned density is a strict
        # UNDER-estimate: the catastrophic direction is impossible by
        # construction.  The residual is a one-sided truncation of a
        # sub-grid-cell sliver of the support whose exact value the exact branch
        # cannot resolve either (its q-quadrature sees a support narrower than one
        # q-grid interval there).  Away from the edge every node keeps its own
        # normaliser, so interior cells are bit-for-bit unchanged.
        n_nodes = I_grid.shape[0]
        has_sup = I_grid > 0
        node_ix = jnp.arange(n_nodes)
        # Nearest supported node at or below / at or above each node (sentinels
        # -1 / n_nodes clip onto a zero node, which the maximum then discards).
        prev_ix = jnp.clip(lax.cummax(jnp.where(has_sup, node_ix, -1), axis=0),
                           0, n_nodes - 1)
        next_ix = jnp.clip(lax.cummin(jnp.where(has_sup, node_ix, n_nodes), axis=0,
                                      reverse=True), 0, n_nodes - 1)
        I_fill  = jnp.maximum(I_grid[prev_ix], I_grid[next_ix])
        I_grid  = jnp.where(has_sup, I_grid, I_fill)
        # No node has support at all: the exact branch returns 0 (its n_sc == 0
        # guard), so return 0 here too instead of dividing by the floor.
        any_sup = jnp.any(has_sup)
        log_I_grid = jnp.log(jnp.where(any_sup, jnp.maximum(I_grid, 1e-300), 1.0))
        # Linear interp of log N in log m1 (jnp.interp clamps out-of-range x to
        # the grid ends).  N = exp(log_I): density = p / N = p * exp(-log_I),
        # a SINGLE reciprocal so the VJP never squares N (same reason the exact
        # branch factors out scale_m).  Zero-support samples have p==0 exactly,
        # so p * exp(-log_I) == 0, matching the exact branch's guarded 0.
        log_m1_q = jnp.log(jnp.atleast_1d(m1))
        log_I   = jnp.interp(log_m1_q, log_m1_grid, log_I_grid).reshape(jnp.shape(m1))
        # TRUST MAP for the interpolant.  Linear interpolation of log I on the
        # cell [i, i+1] is in error by at most h^2 |(log I)''| / 8, and this grid
        # is uniform in log m1, so that bound IS the node second difference over
        # 8.  Near the support edge the bound explodes -- I(m1) ~ eps exp(-dm/eps)
        # with eps = m1 - m_min carries an essential singularity no interpolant
        # can follow -- and it does so over MANY cells, not only the one that
        # straddles the edge: measured on PowerLawPairing at the powerlaw+peak
        # prior midpoint, N_grid = 2048, the interpolated density was still
        # 5.3 nats high THREE cells above m_min.  A node is TRUSTED when its own
        # second difference is inside the ``pairing_edge_tol`` budget and no
        # zero-support node lies within one cell of it; a SAMPLE is trusted when
        # both bracketing nodes are, which interpolating the indicator gives
        # exactly (jnp.interp's clamp makes out-of-grid m1 inherit the end node's
        # flag).  The criterion is SCALE INVARIANT -- it flags the same m1 range
        # at every N_grid, where a fixed cell window does not (measured: a
        # 16-cell window leaves 0.031 nats at N_grid = 2048 but 0.072 at 8192).
        d2      = jnp.abs(log_I_grid[2:] - 2.0 * log_I_grid[1:-1]
                          + log_I_grid[:-2]) / 8.0
        d2      = jnp.concatenate([d2[:1], d2, d2[-1:]])
        true1   = jnp.ones((1,), dtype=bool)
        near_sup = (has_sup
                    & jnp.concatenate([true1, has_sup[:-1]])
                    & jnp.concatenate([has_sup[1:], true1]))
        trusted = near_sup & (d2 <= settings.pairing_edge_tol)
        resolved = jnp.interp(log_m1_q, log_m1_grid,
                              trusted.astype(log_I_grid.dtype)
                              ).reshape(jnp.shape(m1)) >= 1.0
        # UNTRUSTED samples get the normaliser the exact branch DEFINES -- the
        # same support-relative panel split -- but evaluated on the sample's OWN
        # support with ``pairing_edge_nq`` Gauss-Legendre nodes per panel instead
        # of the exact branch's PAIRING_PANEL_NQ.  GL is what makes a per-sample
        # quadrature affordable here: the integrand is a taper boundary layer of
        # width (m1 - m_min)/dm_min in the support-relative variable, which GL
        # resolves with ~sqrt of the nodes a uniform rule needs.  Measured over
        # 50 (m_min, dm_min, beta) prior draws, worst |Delta log density| in the
        # 24 grid cells above the support edge at N_grid = 1056, against a
        # composite-Gauss-Legendre reference on the same split (256+32
        # sub-panels, self-converged to 2e-16 -- it replaced a 200001-node
        # uniform trapezoid whose own 2.5e-6 endpoint deficit had grown larger
        # than the errors being measured):
        #     old fixed-q-grid clamp   22.6 nats
        #     this rule, 24 GL nodes   2.6e-3 nats
        #     the EXACT branch itself  7.6e-3 nats
        #     (the 200-node q-trapezoid the split replaced   3.1e-2 nats)
        # -- i.e. the grid path stays closer to the truth at the edge than the
        # default it approximates, and the old clamp's 251x-and-unbounded density
        # deficit (it pinned the normaliser to dq p(m1,1)/2, killing the
        # m1 / m_min / beta dependence: d log p / d m_min came out -7.7e3 against
        # a true +1.67e7) is gone.
        t_e, w_e = get_pairing_edge_quadrature()             # (K,) static nodes
        m1_a     = jnp.atleast_1d(m1)
        # The SAME panel split as the exact branch, with pairing_edge_nq nodes
        # per panel instead of PAIRING_PANEL_NQ.  Sharing the split is what makes
        # "the grid branch is never worse than the exact branch it approximates"
        # (tests/test_pairing_edge_fix.py) hold BY CONSTRUCTION for the samples
        # that take this rule -- a strictly finer rule on an identical split --
        # and by MEASUREMENT for the trusted ones, which take jnp.interp of
        # log I and are constrained by no node count at all: the smallest margin
        # measured over that test's corners is 1.45x (m_min = 2, dm_min = 10,
        # beta = 0 at N_grid = 2048), against 199x at the other end.
        n_sc_e, scale_e = self._panel_norm(m1_a, m_min, dm_min, theta, t_e, w_e)
        I_edge   = (n_sc_e * scale_e).reshape(jnp.shape(m1))
        edge_ok  = I_edge > 0
        # Safe log: an empty support gives I_edge == 0, and there p == 0 too, so
        # keep the interpolated value rather than build a -inf whose VJP is
        # 0 * inf = NaN.
        log_I_edge = jnp.log(jnp.where(edge_ok, I_edge, 1.0))
        log_I   = jnp.where(resolved | (~edge_ok), log_I, log_I_edge)
        dens    = p * jnp.exp(-log_I)
        return jnp.where(any_sup & in_support, dens, jnp.zeros_like(dens))


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
        # v_i are stick-breaking inputs, bounded [0, 1], with the Beta(1, k-i)
        # prior that induces the UNIFORM Dirichlet(1, ..., 1) on the weights
        # (mirrors the multitracer sticks, darksirens/inference/prior.py).  Under
        # the plain U[0, 1] they carried before, the induced weight prior was
        # E[w] = (1/2, 1/4, 1/8, ...) -- strongly informative and dependent on the
        # ORDER the components appear in the model name, so a headline mixture
        # fraction (e.g. the peak fraction) was prior-dominated.  k = 2 is
        # unaffected: Beta(1, 1) IS U[0, 1].
        specs = [
            ParamSpec(rf"$v_{i+1}$", 0.0, 1.0, name=f"v{i+1}",
                      prior_kind="beta", prior_loc=1.0,
                      prior_scale=float(self.k - 1 - i))
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

    def _low_mass_edge(self, tm_list):
        """Mixture-level secondary-mass cut ``(m_min, dm_min)``.

        The first mass component that declares a low-mass edge sets it.  A
        component that declares none (the Gaussian ``peak``, which has no taper
        parameters) inherits THIS edge rather than the quadrature floor
        ``M_LO``: the secondary of a peak-component binary is drawn from the
        same black-hole population as every other component, so its floor is a
        sampled model parameter, not the normalisation grid's lower bound.
        Falling back to ``M_LO`` there made the low-mass edge of most of the
        fiducial population a quadrature knob and admitted 1 Msun secondaries
        (measured on ``powerlaw+peak`` at the registered fiducial:
        P(m2 < m_min = 5) = 0.019 at beta = 1 and 0.41 at beta = -1).
        ``(M_LO, 0.01)`` survives only for a mixture in which NO component
        declares an edge.
        """
        for j, c in enumerate(self.mass_components):
            if hasattr(c, "m_min_spec"):
                tm = tm_list[j]
                dmmin = (tm[c.param_specs.index(c.dm_min_spec)]
                         if hasattr(c, "dm_min_spec") else 0.01)
                return tm[c.param_specs.index(c.m_min_spec)], dmmin
        return M_LO, 0.01

    def component_densities(self, m1, q, chieff, theta, spin=None):
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

        edge = self._low_mass_edge(tm_list)

        contributions = []
        for i in range(self.k):
            c_m  = self.mass_components[i];    tm = tm_list[i]
            c_p  = self.pairing_components[0 if self.shared_pairing else i]
            tp   = tp_list[0 if self.shared_pairing else i]
            c_s  = self.spin_components[0 if self.shared_spin else i]
            ts   = ts_list[0 if self.shared_spin else i]
            s_idx = 0 if self.shared_spin else i

            mmin, dmmin = edge
            if hasattr(c_m, "m_min_spec"):
                mmin  = tm[c_m.param_specs.index(c_m.m_min_spec)]
            if hasattr(c_m, "dm_min_spec"):
                dmmin = tm[c_m.param_specs.index(c_m.dm_min_spec)]

            if getattr(c_s, "consumes_spin_block", False):
                # 4-D component-spin model: consumes the event spin block
                # (a1, a2, cost1, cost2) and ignores chieff.  Raises inside
                # the component if spin is None, i.e. if it was paired with a
                # chieff-basis store (basis negotiation, DS-09, prevents
                # reaching that state from the CLI).
                spin_term = c_s(chieff, ts, norm=spin_norms[s_idx], spin=spin)
            else:
                spin_term = c_s(chieff, ts, norm=spin_norms[s_idx])
            contributions.append(w[i] * (
                c_m(m1, tm, norm=mass_norms[i])
                * c_p(m1, q, mmin, dmmin, tp)
                * spin_term
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
        edge = self._low_mass_edge(tm_list)

        total = 0.0
        for i in range(self.k):
            c_m = self.mass_components[i];    tm = tm_list[i]
            c_p = self.pairing_components[0 if self.shared_pairing else i]
            tp  = tp_list[0 if self.shared_pairing else i]

            mmin, dmmin = edge
            if hasattr(c_m, "m_min_spec"):
                mmin  = tm[c_m.param_specs.index(c_m.m_min_spec)]
            if hasattr(c_m, "dm_min_spec"):
                dmmin = tm[c_m.param_specs.index(c_m.dm_min_spec)]

            total = total + w[i] * (
                c_m(m1, tm, norm=mass_norms[i])
                * c_p(m1, q, mmin, dmmin, tp)
            )
        return total

    def __call__(self, m1, q, chieff, theta, spin=None):
        """Evaluate the normalised mixture density for source parameters."""
        return jnp.sum(
            self.component_densities(m1, q, chieff, theta, spin=spin), axis=0
        )


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

    def log_p_massspin(self, m1, q, chieff, theta, spin=None):
        """Log source-parameter (mass, mass-ratio, spin) mixture density.

        Sentinel: p = 0  →  log p = −jnp.inf, matching :meth:`log_p_pop`.
        """
        if not self.has_additive_rate_split:
            raise NotImplementedError(
                "log_p_massspin is undefined for per-component redshift "
                "evolution: use log_p_pop."
            )
        p = self.mixture(m1, q, chieff, self.mixture_theta(theta), spin=spin)
        return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)

    def log_p_pop(self, m1, q, z, chieff, theta, spin=None):
        """
        Log population probability at (m1, q, z, chieff) under parameters theta.

        Sentinel: p = 0  →  log p = −jnp.inf  (not −1e10).
        −∞ propagates correctly through logsumexp / jnp.sum; the final
        jnp.isfinite guard in the likelihood rejects the proposal cleanly.

        ``spin`` is the optional (N, d) component-spin block; forwarded to the
        mixture, where only a spin component with ``consumes_spin_block`` ever
        reads it.
        """
        if self.has_additive_rate_split:
            return (
                self.log_p_massspin(m1, q, chieff, theta, spin=spin)
                + self.log_rate_z(z, theta)
            )

        n_mix  = self.mixture.n_params
        tm     = theta[:n_mix]
        gamma  = theta[n_mix : n_mix + self.mixture.k]
        p_comp = self.mixture.component_densities(m1, q, chieff, tm, spin=spin)

        gamma_shape = (self.mixture.k,) + (1,) * (jnp.ndim(p_comp) - 1)
        z_factor = jnp.power(1.0 + z, gamma.reshape(gamma_shape) - 1.0)
        p = jnp.sum(p_comp * z_factor, axis=0)
        return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
