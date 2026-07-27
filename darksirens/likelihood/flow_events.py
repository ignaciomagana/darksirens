"""
flow_events.py
--------------
Flow-surrogate hierarchical likelihood (spectral sirens).

Replaces the stored-PE-sample per-event term with per-event normalizing
flows: at every likelihood call, source-frame points are drawn from the
CURRENT population target (common random numbers -> deterministic,
continuous in the hyperparameters), mapped to detector frame, and scored by
every event's flow.  Per event i,

    ln Z_i = -ln J + logsumexp_j [ ln t(theta_j) - ln s(theta_j)
                                   + ln flow_i(theta_det,j)
                                   - ln pi_PE(theta_det,j) ]

with t = p_pop(m1, q, chieff) * rate_z(z) * p_vol(z) the SAME unnormalised
target the selection integral uses (mandatory: the population normalisation
must cancel between the event terms and N_obs * log mu), s the exact
density of the grid/analytic samplers (darksirens/gw/populations/sampling.py),
and pi_PE the analytic PE prior in the flow's (m1det, m2det, dL, chieff)
basis.  Both flow_i and pi_PE are densities in that same basis, so no extra
Jacobian appears in their ratio; the m2det <-> q basis factor m1det cancels.

s is a two-component MIXTURE (issue #260),

    s(theta) = (1 - w) q_win(theta) + w q_full(theta),

with q_win the event's empirical-support window and q_full the FULL
population support at the current hyperparameters; every draw is scored under
BOTH components.  A windowed-only proposal truncates the event integral to
boxes measured from finite flow samples, and the omitted learned-flow tail
mass depends on the hyperparameters, so it biases the posterior rather than
shifting each ln Z_i by a constant.  See :func:`build_flow_loglike`.

pi_PE convention (must match gwcat's exported p_pe, verified against
gwcat@3a61a24): uniform detector-frame component masses (constant),
UniformSourceFrame luminosity-distance prior (uniform in comoving volume and
source time under the PE cosmology -- NOT dL^2), and the 1-D isotropic
chi_eff prior p(chi_eff | q, amax) from gwcat.spin.ChiEffPrior.  Per-event
normalisation constants of pi_PE are hyperparameter-independent: they shift
each event's ln Z_i by a constant (and total logZ), never the posterior.

Selection term, per-event MC-variance accounting, and the total-variance
guard are the standard ones from likelihood/selection.py, evaluated with the
same population/redshift-prior closures as the event term.

Architecture: unlike core.darksiren_log_likelihood (a module-level jit with
hashable statics), the flow ensemble's equinox static halves are unhashable
pytrees, so the jitted body is FACTORY-LOCAL: statics (flow structure, model
closures, Python ints) live in the closure, arrays are traced operands,
barrier-wrapped at build time per the events.py doctrine.

Scope: universe_model == "spectral_sirens" only.  Dark sirens need 6-D flows
(ra/dec) plus the per-pixel redshift-prior sampler -- the loader recognises
that layout and fails with an explicit message (gw/flows.py).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from darksirens.core.constants import H0_FID, OM0_FID
from darksirens.core.types import EMCatalog
from darksirens.gw.flows import (
    SPECTRAL_COLUMNS,
    FlowEnsemble,
    compute_support_boxes,
    make_ensemble_log_prob_per_event,
)
from darksirens.gw.populations.registry import get_model
from darksirens.gw.populations.parametric import TruncatedGaussianSpin
from darksirens.gw.populations.sampling import (
    build_column_cdf_tables,
    cell_centers,
    histogram_trunc_logpdf,
    m1_given_q_trunc_logpdf,
    make_mass_q_edges,
    q_marginal_log_dens,
    resolve_mass_grid_bounds,
    sample_histogram_trunc,
    sample_m1_given_q_trunc,
    truncnorm_logpdf,
    truncnorm_sample,
    _floored,
)
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.likelihood.catalog_views import barrier, prepare_catalog_views
from darksirens.likelihood.events import pad_gw_event_to_multiple
from darksirens.likelihood.selection import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    compute_selection_term,
    log_evidence_and_mc_variance,
    selection_log_correction,
)
from darksirens.inference.utils import log_sample_weight
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.redshift.grid import zgrid
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import GWEvent
from darksirens.utils.cosmology import dL_grid_bounds, dL_of_z, z_of_dL


# ── analytic PE prior pieces (host-side builders + in-jit evaluators) ───────


class PePriorTables(NamedTuple):
    """Precomputed PE-prior interpolation tables (all barrier-wrapped)."""

    log_dl_grid: jnp.ndarray    # (Nd,) log dL [Mpc], geometric grid
    log_p_dl: jnp.ndarray       # (Nd,) log UniformSourceFrame p(dL) (unnormalised)
    chi_q_grid: jnp.ndarray     # (nq,) gwcat q_frac = m1/(m1+m2) grid
    chi_grid: jnp.ndarray       # (nchi,) chi_eff grid
    chi_table: jnp.ndarray      # (nq, nchi) p(chi_eff | q_frac) probabilities


def build_pe_dl_prior_table(
    pe_H0: float, pe_Om0: float, n: int = 4096, z_max: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """UniformSourceFrame log p(dL) shape on a wide log-dL grid (host-side).

    Same physics as gwcat.cosmology.uniform_source_frame_prob / bilby's
    UniformSourceFrame: p(dL) ∝ [dV_c/dz / (1+z)] / (d dL/dz) under the PE
    cosmology.  Deliberately UNNORMALISED and untruncated: the PE prior's
    [dmin, dmax] window only contributes per-event constants, and a smooth
    tail avoids infinite weights where a flow leaks density beyond the
    window.  ``z_max=100`` keeps the table covering any population draw even
    at the H0 prior floor.
    """
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cos = FlatLambdaCDM(H0=pe_H0, Om0=pe_Om0)
    z = np.expm1(np.linspace(np.log(1.0 + 1e-6), np.log(1.0 + z_max), n))
    DC = cos.comoving_distance(z).to_value(u.Mpc)
    E = np.sqrt(pe_Om0 * (1.0 + z) ** 3 + (1.0 - pe_Om0))
    dH = 299792.458 / pe_H0
    dL = (1.0 + z) * DC
    ddL_dz = DC + (1.0 + z) * dH / E
    p = (DC**2 / E) / (1.0 + z) / ddL_dz
    logp = np.log(np.maximum(p, 1e-300))
    return np.log(dL), logp


def build_chi_eff_prior_table(amax: float):
    """gwcat's 1-D isotropic chi_eff prior table (host-side, numpy)."""
    from gwcat.spin import ChiEffPrior

    prior = ChiEffPrior(amax=float(amax))
    return (
        np.asarray(prior.q_grid),
        np.asarray(prior.chi_grid),
        np.asarray(prior.table),
    )


def chi_eff_prior_prob(
    q_frac: jnp.ndarray,
    chieff: jnp.ndarray,
    q_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    table: jnp.ndarray,
) -> jnp.ndarray:
    """JAX port of gwcat.spin.ChiEffPrior._interp2d — the PROBABILITY itself.

    Bilinear interpolation in probability (matching gwcat exactly).  Returned
    un-logged so callers can distinguish "prior density is exactly zero" (the
    PE prior has no support there, so the posterior — and the event integrand
    — is zero) from "small but positive".  ``q_frac = m1/(m1+m2) = 1/(1+q)``
    with q = m2/m1.
    """
    nq = q_grid.shape[0]
    nchi = chi_grid.shape[0]

    q_idx = jnp.interp(q_frac, q_grid, jnp.arange(nq, dtype=q_grid.dtype))
    q_lo = jnp.clip(jnp.floor(q_idx).astype(jnp.int32), 0, nq - 2)
    q_f = q_idx - q_lo

    chi_idx = jnp.interp(chieff, chi_grid, jnp.arange(nchi, dtype=chi_grid.dtype))
    c_lo = jnp.clip(jnp.floor(chi_idx).astype(jnp.int32), 0, nchi - 2)
    c_f = chi_idx - c_lo

    v00 = table[q_lo, c_lo]
    v01 = table[q_lo, c_lo + 1]
    v10 = table[q_lo + 1, c_lo]
    v11 = table[q_lo + 1, c_lo + 1]
    v0 = v00 * (1.0 - c_f) + v01 * c_f
    v1 = v10 * (1.0 - c_f) + v11 * c_f
    return v0 * (1.0 - q_f) + v1 * q_f


def chi_eff_prior_logpdf(
    q_frac: jnp.ndarray,
    chieff: jnp.ndarray,
    q_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    table: jnp.ndarray,
) -> jnp.ndarray:
    """gwcat's ``chi_eff_prior_logprob``: log of :func:`chi_eff_prior_prob`.

    Zero density is reported as gwcat reports it, clipped at -50, so this
    stays a bit-for-bit port of the reference implementation.  The likelihood
    does NOT use that clip as a density: see ``_event_ldw``, which masks
    zero-prior draws out of the integral entirely (a -50 floor would hand
    e^50 of importance weight to flow density outside the PE support).
    """
    p = chi_eff_prior_prob(q_frac, chieff, q_grid, chi_grid, table)
    return jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -50.0)


def _parse_pe_cosmology(opts) -> tuple[float, float]:
    raw = getattr(opts, "flows_pe_cosmology", None)
    if raw is None or raw == "":
        return float(H0_FID), float(OM0_FID)
    if isinstance(raw, (tuple, list)):
        h0, om0 = raw
    else:
        parts = str(raw).split(",")
        if len(parts) != 2:
            raise ValueError(
                "--flows_pe_cosmology must be 'H0,Om0' (e.g. '67.74,0.3089'); "
                f"got {raw!r}."
            )
        h0, om0 = parts
    return float(h0), float(om0)


# ── mixture proposal (issue #260) ───────────────────────────────────────────

#: Default fraction of each event's draws taken from the FULL population
#: support rather than from the event's empirical support window.
#:
#: Chosen from ``scripts/flows_mixture_convergence.py``.  MEASURED cost at the
#: shipped settings (margin 0.25, 4096 support samples, J = 16384-32768): 4-6%
#: of the windowed component's effective sample size, i.e. +0.0006 nat on the
#: per-event Monte-Carlo error, and no measurable wall-clock cost.  MEASURED
#: benefit: where the window under-resolves the flow (512 support samples,
#: zero margin) it recovers a third of a systematic ln Z deficit that
#: ``w_full = 0`` cannot see at any J; where the window under-COVERS the flow
#: it recovers 5-300 nats per event.  At the shipped settings on synthetic
#: flows the deficit is already below the Monte-Carlo error -- 5% is
#: insurance, and it is what turns an invisible bias into a variance the
#: total-lnL guard can act on.
DEFAULT_W_FULL = 0.05

#: Finite stand-in for "no upper bound" in the full-support window.  Divided
#: by (1+z) and multiplied by g(q) <~ 20 it must stay far above the mass grid
#: and far below overflow.
_UNBOUNDED = 1e30


class _Win(NamedTuple):
    """One proposal component's windows, as traced scalars (per event).

    Both mixture components are the SAME sampler with different windows, so a
    single ``_draw``/``_log_q`` pair serves both and the two can never drift
    apart.  ``m1det``/``mc`` bound the DETECTOR-frame chirp-mass band; the m1
    and chi_eff windows are per-draw functions of (q, z) — see ``_m1_window``
    and ``_chi_window`` in :func:`build_flow_loglike`.
    """

    z_lo: jnp.ndarray
    z_hi: jnp.ndarray
    q_lo: jnp.ndarray
    q_hi: jnp.ndarray
    m1det_lo: jnp.ndarray
    m1det_hi: jnp.ndarray
    mc_lo: jnp.ndarray
    mc_hi: jnp.ndarray
    chi_lo: jnp.ndarray
    chi_hi: jnp.ndarray
    chi_a: jnp.ndarray
    chi_b: jnp.ndarray
    chi_r_lo: jnp.ndarray
    chi_r_hi: jnp.ndarray


class SupportBoxes(NamedTuple):
    """:func:`darksirens.gw.flows.compute_support_boxes` as a stable pytree."""

    m1det: jnp.ndarray      # (nEvents, 2)
    q: jnp.ndarray
    dL: jnp.ndarray
    chieff: jnp.ndarray
    mc_det: jnp.ndarray
    chi_ab: jnp.ndarray
    chi_resid: jnp.ndarray
    pe_box: jnp.ndarray     # (nEvents, 4, 2) original PE-prior support

    @classmethod
    def from_dict(cls, boxes: dict, nEvents: int) -> "SupportBoxes":
        pe_box = boxes.get("pe_box")
        if pe_box is None:
            pe_box = jnp.stack(
                [
                    jnp.full((nEvents, 4), -jnp.inf),
                    jnp.full((nEvents, 4), jnp.inf),
                ],
                axis=-1,
            )
        return cls(
            m1det=boxes["m1det"], q=boxes["q"], dL=boxes["dL"],
            chieff=boxes["chieff"], mc_det=boxes["mc_det"],
            chi_ab=boxes["chi_ab"], chi_resid=boxes["chi_resid"],
            pe_box=jnp.asarray(pe_box),
        )


class FlowOperands(NamedTuple):
    """Device arrays threaded through ``jax.jit`` as ARGUMENTS, not constants.

    ``jax.jit`` lowers a closed-over concrete ``jax.Array`` to a ``dense<>``
    HLO constant (~8 bytes of module text per element), so closing over the
    selection set, the CRN block and the stacked flow parameters embedded tens
    of MB of literals in the flow likelihood's HLO — the same defect
    ``likelihood/factory._jit_likelihood_body`` fixes for the stored-PE path
    (PR #296).  Everything here is already barrier-wrapped and device-resident
    when the factory builds it; passing it as one pytree argument keeps the
    traced signature constant, so the body still compiles exactly once.
    """

    u_win: jnp.ndarray          # (nEvents, J_win, 4) windowed-component CRN
    u_full: jnp.ndarray         # (nEvents, J_full, 4) full-support CRN
    m1_edges: jnp.ndarray
    q_edges: jnp.ndarray
    pe_tables: PePriorTables
    boxes: SupportBoxes
    gw_sel: GWEvent
    em_catalog_sel: EMCatalog
    group_params: tuple


def resolve_full_draws(J: int, w_full: float) -> int:
    """Number of full-support draws for ``J`` total draws at fraction ``w_full``.

    Deterministic allocation (not a per-draw coin flip): the estimator below
    uses the REALISED fraction ``J_full / J`` in the mixture density, which
    keeps it exactly unbiased and strictly lower-variance than randomising the
    component label, and keeps the likelihood a deterministic function of the
    hyperparameters under common random numbers.  A requested ``w_full > 0``
    never rounds away to zero.
    """
    if not (0.0 <= w_full < 1.0):
        raise ValueError(f"w_full must be in [0, 1); got {w_full}.")
    if w_full == 0.0:
        return 0
    return int(min(max(1, round(w_full * J)), J - 1))


# ── jitted-body builder (separable from opts/data plumbing for testing) ─────


def build_flow_loglike(
    *,
    model,
    eval_logflows,
    group_params: tuple,
    u_base: jnp.ndarray,
    m1_edges: jnp.ndarray,
    q_edges: jnp.ndarray,
    pe_tables: PePriorTables,
    support_boxes: dict,
    gw_sel: GWEvent,
    em_catalog_sel: EMCatalog,
    Ndraw: float,
    nEvents: int,
    w_full: float = DEFAULT_W_FULL,
    sel_batch_size: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    materialize_redshift_prior_state: bool = True,
):
    """Return the jitted ``(cosmo, survey, pop_params) -> logL`` flow body.

    Non-hashable statics (population-model closures, flow ensemble structure,
    Python scalars) are closure-captured; every device array travels as a
    :class:`FlowOperands` jit ARGUMENT and must already be barrier-wrapped by
    the caller.  See the module docstring for the estimator.

    Mixture proposal (issue #260)
    -----------------------------
    Each event's J draws are split into ``J_win`` from the event's empirical
    support window and ``J_full = round(w_full * J)`` from the FULL population
    support, and EVERY draw — whichever component produced it — is weighted by
    the exact mixture density

        q_mix(θ) = (1 - w) q_win(θ) + w q_full(θ),   w = J_full / J,

    so ``ln Z_i = -ln J + logsumexp_j [ln t_j - ln q_mix,j + ln flow_j
    - ln pi_PE,j]``.  Scoring a draw only under the component that produced it
    is the classic multiple-importance-sampling error; it is structurally
    impossible here because ``_log_q`` is a standalone density function of the
    point, evaluated twice per draw.

    The windowed component alone (``w_full = 0``, the pre-#260 estimator)
    truncates the integral to boxes measured from 4096 flow samples, and the
    omitted learned-flow tail mass moves with the hyperparameters — a bias,
    not an ignorable per-event constant.  The full component is derived from
    the population model's OWN support at the proposal's hyperparameters (the
    mass grid box from ``resolve_mass_grid_bounds``, q over the grid, chi_eff
    over [-1, 1], z over ``zgrid``), never from the empirical boxes, so the
    proposal covers every point where the target ``t`` is non-zero.

    ``w_full`` trades a little effective sample size (the windowed component
    is far better matched to a narrow event posterior) for tail coverage.

    Shapes: ``u_base`` is (nEvents, J, 4); ``eval_logflows`` must be the
    per-event variant (:func:`make_ensemble_log_prob_per_event`).
    """
    mixture = model.mixture
    if not model.has_additive_rate_split:
        raise NotImplementedError(
            "The flow likelihood requires a shared redshift evolution "
            "(log_p_pop = log_p_massspin + log_rate_z); per-component gamma "
            "is not supported."
        )
    if not mixture.shared_spin:
        raise NotImplementedError(
            "The flow likelihood samples chi_eff separately and requires a "
            "shared spin component; per-component spins are not supported."
        )
    spin_component = mixture.spin_components[0]
    spin_is_truncnorm = isinstance(spin_component, TruncatedGaussianSpin)
    log_p_pop = model.log_p_pop

    if u_base.ndim != 3 or u_base.shape[0] != nEvents or u_base.shape[2] != 4:
        raise ValueError(
            f"u_base must be (nEvents, J, 4); got {tuple(u_base.shape)} for "
            f"nEvents={nEvents}."
        )
    J = int(u_base.shape[1])
    J_full = resolve_full_draws(J, float(w_full))
    J_win = J - J_full
    w_eff = J_full / J
    log_w_full = np.log(w_eff) if J_full else -np.inf
    log_w_win = np.log1p(-w_eff)

    chi_edges = jnp.linspace(-1.0, 1.0, 513)

    operands = FlowOperands(
        u_win=u_base[:, :J_win, :],
        u_full=u_base[:, J_win:, :],
        m1_edges=m1_edges,
        q_edges=q_edges,
        pe_tables=pe_tables,
        boxes=SupportBoxes.from_dict(support_boxes, nEvents),
        gw_sel=gw_sel,
        em_catalog_sel=em_catalog_sel,
        group_params=group_params,
    )

    def _prepare_state(cosmo, survey, em_catalog):
        return prepare_redshift_prior_state(
            "spectral_sirens",
            cosmo,
            survey,
            em_catalog,
            materialize_state=materialize_redshift_prior_state,
        )

    def _event_ldw(cosmo, pop_params, prior_state, ops):
        """Per-event log importance weights ldw (nEvents, J) — the flow term."""
        H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        m1_edges_, q_edges_ = ops.m1_edges, ops.q_edges
        boxes, pe_tab = ops.boxes, ops.pe_tables

        # -- population draws (source frame; common random numbers) --------
        # Shared per-call ingredients: the mass-q target on the grid, the
        # spin parameters, and the redshift target on the module zgrid.
        tm = model.mixture_theta(pop_params)

        m1_centers = cell_centers(m1_edges_)
        q_centers = cell_centers(q_edges_)
        chi_centers = cell_centers(chi_edges)
        M1, Q = jnp.meshgrid(m1_centers, q_centers, indexing="ij")
        p_mq = mixture.mass_q_density(M1.reshape(-1), Q.reshape(-1), tm)
        log_t_cells = jnp.where(
            p_mq > 0.0,
            jnp.log(jnp.maximum(p_mq, jnp.finfo(p_mq.dtype).tiny)),
            -jnp.inf,
        ).reshape(M1.shape)

        ts = mixture.spin_theta(tm)
        if not spin_is_truncnorm:
            spin_norm = spin_component._norm(ts)
            p_chi_cells = spin_component(chi_centers, ts, norm=spin_norm)
            log_chi_cells = _floored(jnp.where(
                p_chi_cells > 0.0,
                jnp.log(jnp.maximum(p_chi_cells, jnp.finfo(p_chi_cells.dtype).tiny)),
                -jnp.inf,
            ))

        log_pvol_nodes = prior_state.log_pvol
        log_t_z_nodes = log_pvol_nodes + model.log_rate_z(zgrid, pop_params)
        log_z_cell_dens = _floored(
            jnp.logaddexp(log_t_z_nodes[:-1], log_t_z_nodes[1:]) - jnp.log(2.0)
        )
        dL_lo_g, dL_hi_g = dL_grid_bounds(H0, Om0, w0, wa)

        log_t_cells_fl = _floored(log_t_cells)
        # Event-independent proposal ingredients, built ONCE per call instead
        # of once per (event, draw) inside the vmap: the m1-marginalised q
        # density and the per-q-column m1 CDF tables.  The mixture evaluates
        # the m1 conditional three times per draw (sample + two component
        # densities), which only stays cheap because of this hoist.
        log_q_col_dens = q_marginal_log_dens(m1_edges_, q_edges_, log_t_cells_fl)
        m1_tab = build_column_cdf_tables(m1_edges_, log_t_cells_fl)
        n_qcell = q_edges_.shape[0] - 1

        def _q_cell(q):
            """Grid column of q — the SAME index for sampling and for density."""
            return jnp.clip(
                jnp.searchsorted(q_edges_, q, side="right") - 1, 0, n_qcell - 1
            )

        def _m1_window(win, q, z):
            """Source-frame m1 window at the drawn (q, z): the Mc band ∩ m1det box."""
            g_q = (1.0 + q) ** 0.2 / q**0.6  # m1 = Mc * g(q)
            lo = jnp.maximum(win.mc_lo / (1.0 + z) * g_q, win.m1det_lo / (1.0 + z))
            hi = jnp.minimum(win.mc_hi / (1.0 + z) * g_q, win.m1det_hi / (1.0 + z))
            return lo, jnp.maximum(hi, lo)

        def _chi_window(win, q):
            """chi_eff window at the drawn q: the (q, chi) band ∩ box ∩ [-1, 1]."""
            mid = win.chi_a + win.chi_b * q
            lo = jnp.clip(jnp.maximum(mid + win.chi_r_lo, win.chi_lo), -1.0, 1.0)
            hi = jnp.clip(jnp.minimum(mid + win.chi_r_hi, win.chi_hi), -1.0, 1.0)
            return lo, jnp.maximum(hi, lo)

        def _draw(u, win):
            """Draw (m1src, q, chieff, z) from the target truncated to ``win``."""
            z = sample_histogram_trunc(
                u[:, 3], zgrid, log_z_cell_dens, win.z_lo, win.z_hi
            ).x
            q = sample_histogram_trunc(
                u[:, 1], q_edges_, log_q_col_dens, win.q_lo, win.q_hi
            ).x
            m1_lo, m1_hi = _m1_window(win, q, z)
            m1src, _ = sample_m1_given_q_trunc(
                u[:, 0], m1_edges_, log_t_cells_fl, _q_cell(q), m1_lo, m1_hi,
                tables=m1_tab,
            )
            chi_lo, chi_hi = _chi_window(win, q)
            if spin_is_truncnorm:
                chieff = truncnorm_sample(u[:, 2], ts[0], ts[1], chi_lo, chi_hi).x
            else:
                chieff = sample_histogram_trunc(
                    u[:, 2], chi_edges, log_chi_cells, chi_lo, chi_hi
                ).x
            return m1src, q, chieff, z

        def _log_q(m1src, q, chieff, z, win):
            """Exact log density of ``_draw(., win)`` at an ARBITRARY point.

            Standalone in the point, so a draw from one mixture component is
            scored correctly under the other; -inf outside ``win``.
            """
            lz = histogram_trunc_logpdf(
                z, zgrid, log_z_cell_dens, win.z_lo, win.z_hi
            )
            lq = histogram_trunc_logpdf(
                q, q_edges_, log_q_col_dens, win.q_lo, win.q_hi
            )
            m1_lo, m1_hi = _m1_window(win, q, z)
            lm1 = m1_given_q_trunc_logpdf(
                m1src, m1_edges_, log_t_cells_fl, _q_cell(q), m1_lo, m1_hi,
                tables=m1_tab,
            )
            chi_lo, chi_hi = _chi_window(win, q)
            if spin_is_truncnorm:
                lchi = truncnorm_logpdf(chieff, ts[0], ts[1], chi_lo, chi_hi)
            else:
                lchi = histogram_trunc_logpdf(
                    chieff, chi_edges, log_chi_cells, chi_lo, chi_hi
                )
            return lz + lq + lm1 + lchi

        # Full population-support component: the model's OWN support at these
        # hyperparameters (mass grid box, q grid, chi_eff on [-1, 1], the whole
        # zgrid) — never the empirical boxes.  ``t`` vanishes outside it, so
        # this component covers the entire integrand.
        win_full = _Win(
            z_lo=zgrid[0], z_hi=zgrid[-1],
            q_lo=q_edges_[0], q_hi=q_edges_[-1],
            m1det_lo=jnp.asarray(0.0), m1det_hi=jnp.asarray(_UNBOUNDED),
            mc_lo=jnp.asarray(0.0), mc_hi=jnp.asarray(_UNBOUNDED),
            chi_lo=jnp.asarray(-1.0), chi_hi=jnp.asarray(1.0),
            chi_a=jnp.asarray(0.0), chi_b=jnp.asarray(0.0),
            chi_r_lo=jnp.asarray(-2.0), chi_r_hi=jnp.asarray(2.0),
        )

        def _event_draws(u_w, u_f, m1det_win, q_win, dL_win, chi_win, mc_win,
                         chi_ab, chi_resid, pe_box):
            # z window from the event's dL window under the CURRENT cosmology.
            win_e = _Win(
                z_lo=z_of_dL(jnp.clip(dL_win[0], dL_lo_g, dL_hi_g), H0, Om0, w0, wa),
                z_hi=z_of_dL(jnp.clip(dL_win[1], dL_lo_g, dL_hi_g), H0, Om0, w0, wa),
                q_lo=q_win[0], q_hi=q_win[1],
                m1det_lo=m1det_win[0], m1det_hi=m1det_win[1],
                mc_lo=mc_win[0], mc_hi=mc_win[1],
                chi_lo=chi_win[0], chi_hi=chi_win[1],
                chi_a=chi_ab[0], chi_b=chi_ab[1],
                chi_r_lo=chi_resid[0], chi_r_hi=chi_resid[1],
            )

            th = _draw(u_w, win_e)
            if J_full:
                th_f = _draw(u_f, win_full)
                th = tuple(jnp.concatenate([a, b]) for a, b in zip(th, th_f))
            m1src, q, chieff, z = th

            # EXACT mixture density at every draw: both components evaluated
            # at both kinds of point (deterministic-mixture / balance-heuristic
            # MIS).  Recomputed from the point rather than reusing a sampler's
            # own log_s, which would only be right for its own component.
            log_s = _log_q(m1src, q, chieff, z, win_e) + log_w_win
            if J_full:
                log_s = jnp.logaddexp(
                    log_s, _log_q(m1src, q, chieff, z, win_full) + log_w_full
                )

            m1det = m1src * (1.0 + z)
            m2det = q * m1det
            dL = dL_of_z(z, H0, Om0, w0, wa)

            log_t = (
                model.log_p_massspin(m1src, q, chieff, pop_params)
                + model.log_rate_z(z, pop_params)
                + jnp.interp(z, zgrid, log_pvol_nodes)
            )

            # PE prior.  The dL table is deliberately untruncated and its
            # clamp is conservative: p_dL rises from ~0 at z -> 0, so below
            # the table the clamped value EXCEEDS the true prior and the
            # weight is understated, never inflated.
            p_chi = chi_eff_prior_prob(
                1.0 / (1.0 + q),
                chieff,
                pe_tab.chi_q_grid,
                pe_tab.chi_grid,
                pe_tab.chi_table,
            )
            log_pi_pe = jnp.interp(
                jnp.log(dL), pe_tab.log_dl_grid, pe_tab.log_p_dl
            ) + jnp.log(jnp.maximum(p_chi, jnp.finfo(p_chi.dtype).tiny))

            # PE-support mask.  The chi_eff prior is EXACTLY zero beyond amax,
            # so the true posterior is too; gwcat's -50 log floor would instead
            # hand e^50 of importance weight to whatever the flow extrapolates
            # there.  Unreachable for the windowed component (its chi window
            # hugs the posterior), reachable for the full-support component —
            # this is the tightening issue #260 calls for.  ``pe_box`` adds any
            # PE window the checkpoint recorded (+/-inf when it did not).
            X = jnp.stack([m1det, m2det, dL, chieff], axis=-1)
            pe_ok = (p_chi > 0.0) & jnp.all(
                (X >= pe_box[:, 0]) & (X <= pe_box[:, 1]), axis=-1
            )

            base = log_t - log_s - log_pi_pe
            base = jnp.where(pe_ok & jnp.isfinite(base), base, -jnp.inf)
            return X, base

        X, base = jax.vmap(_event_draws)(
            ops.u_win, ops.u_full, boxes.m1det, boxes.q, boxes.dL, boxes.chieff,
            boxes.mc_det, boxes.chi_ab, boxes.chi_resid, boxes.pe_box,
        )  # (nEvents, J, 4), (nEvents, J)

        logflows = eval_logflows(ops.group_params, X)  # (nEvents, J)
        ldw = logflows + base
        return jnp.where(jnp.isfinite(ldw), ldw, -jnp.inf)

    def _ll_flows_impl(cosmo, survey, pop_params, ops):
        H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        gw_sel_, em_catalog_sel_ = ops.gw_sel, ops.em_catalog_sel
        prior_state = _prepare_state(cosmo, survey, em_catalog_sel_)

        def log_prior_z(z, pix, catalog):
            return eval_redshift_prior_with_state(
                "spectral_sirens", prior_state, z, pix, cosmo, survey, catalog
            )

        # -- selection term: verbatim spectral log_weight from core.py -----
        def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
            dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            ldw = log_sample_weight(
                m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey,
                pop_params, catalog, log_p_pop, log_prior_z,
            )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        log_mu, Neff, _log_sigma2 = compute_selection_term(
            gw_sel_,
            em_catalog_sel_,
            log_weight,
            Ndraw,
            nEvents,
            sel_batch_size=sel_batch_size,
        )

        ldw = _event_ldw(cosmo, pop_params, prior_state, ops)
        event_lls, event_vars = jax.vmap(
            lambda row: log_evidence_and_mc_variance(row, J)
        )(ldw)

        ll = selection_log_correction(
            log_mu,
            Neff,
            nEvents,
            soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            pe_variance_sum=jnp.sum(event_vars),
        ) + jnp.sum(event_lls)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    def _event_diagnostics_impl(cosmo, survey, pop_params, ops):
        """Per-event lnZ_i, delta-method variance, and ESS (no selection)."""
        prior_state = _prepare_state(cosmo, survey, ops.em_catalog_sel)
        ldw = _event_ldw(cosmo, pop_params, prior_state, ops)
        event_lls, event_vars = jax.vmap(
            lambda row: log_evidence_and_mc_variance(row, J)
        )(ldw)
        w = jnp.exp(ldw - jnp.max(ldw, axis=1, keepdims=True))
        ess = jnp.sum(w, axis=1) ** 2 / jnp.maximum(
            jnp.sum(w**2, axis=1), jnp.finfo(w.dtype).tiny
        )
        return event_lls, event_vars, ess

    # Operands travel as jit ARGUMENTS (see :class:`FlowOperands`); the public
    # callable keeps its 3-argument signature by binding them in a wrapper.
    _jitted = jax.jit(_ll_flows_impl)
    _jitted_diag = jax.jit(_event_diagnostics_impl)

    def ll(cosmo, survey, pop_params):
        return _jitted(cosmo, survey, pop_params, operands)

    def event_diagnostics(cosmo, survey, pop_params):
        return _jitted_diag(cosmo, survey, pop_params, operands)

    ll.event_diagnostics = event_diagnostics
    ll.jitted_body = _jitted
    ll.jitted_event_diagnostics = _jitted_diag
    ll.operands = operands
    ll.n_full_draws = J_full
    ll.w_full = w_eff
    return ll


# ── likelihood factory ───────────────────────────────────────────────────────


def make_flow_likelihood(
    opts, data: dict, pop_params_fid, fixed_parameter_values: dict | None = None
):
    """Build the flow-surrogate likelihood callable for the sampler.

    Mirrors :func:`darksirens.likelihood.factory.make_likelihood` for the
    selection side and the parameter decoding; the per-event PE term is
    replaced by the flow estimator described in the module docstring.
    """
    ensemble: FlowEnsemble = data["flow_ensemble"]
    if ensemble is None:
        raise ValueError("make_flow_likelihood requires data['flow_ensemble'].")
    if ensemble.columns != SPECTRAL_COLUMNS:
        raise NotImplementedError(
            f"Flow column layout {list(ensemble.columns)} is not supported by "
            "the spectral-sirens flow likelihood."
        )

    universe_model = opts.universe_model
    if universe_model != "spectral_sirens":
        raise NotImplementedError(
            "--gw_flows_path currently supports --universe_model "
            "spectral_sirens only. Dark sirens need 6-D flows over "
            "(m1det, m2det, dL, chieff, ra, dec) plus the per-pixel "
            "catalog redshift-prior sampler; the scaffold hooks are the "
            "columns registry in darksirens/gw/flows.py and the z-target "
            f"construction in this factory. Got {universe_model!r}."
        )

    nEvents = ensemble.n_flows
    J = int(getattr(opts, "flows_nsamp", 4096))
    if J <= 0:
        raise ValueError(f"--flows_nsamp must be positive; got {J}.")
    Ndraw = data["Ndraw"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    sel_batch_size = getattr(opts, "sel_batch_size", None)
    w_full = float(getattr(opts, "flows_wfull", DEFAULT_W_FULL))
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    max_likelihood_variance = float(
        getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    )
    # Same policy as the stored-PE factory: the redshift-prior barrier must
    # come off for gradient-based sampling (no differentiation rule).
    from darksirens.likelihood.factory import _resolve_redshift_prior_materialization

    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)

    # Population model object: the SAME log_p_pop feeds the event and selection
    # terms; the additive split + shared-spin factorisation drive the samplers.
    model = get_model(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )

    # ── static proposal machinery (host side, once) ─────────────────────
    seed = int(getattr(opts, "flows_seed", 42))
    u_base = barrier(
        jax.random.uniform(
            jax.random.PRNGKey(seed),
            (nEvents, J, 4),
            dtype=jnp.float64,
        )
    )

    # Per-event support boxes for the event-windowed population proposal.
    boxes = compute_support_boxes(
        ensemble,
        key=jax.random.key(seed + 1),  # flowjax sampling needs typed keys
        n=int(getattr(opts, "flows_support_nsamples", 4096)),
        margin=float(getattr(opts, "flows_support_margin", 0.25)),
    )
    support_boxes = {k: barrier(v) for k, v in boxes.items()}

    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges_h, q_edges_h = make_mass_q_edges(
        m1_lo,
        m1_hi,
        n_m1=int(getattr(opts, "flows_grid_nm", 512)),
        n_q=int(getattr(opts, "flows_grid_nq", 256)),
    )
    m1_edges = barrier(m1_edges_h)
    q_edges = barrier(q_edges_h)

    pe_H0, pe_Om0 = _parse_pe_cosmology(opts)
    log_dl_h, log_p_dl_h = build_pe_dl_prior_table(pe_H0, pe_Om0)
    chi_qg_h, chi_cg_h, chi_tab_h = build_chi_eff_prior_table(
        float(getattr(opts, "flows_chieff_amax", 0.99))
    )
    pe_tables = PePriorTables(
        log_dl_grid=barrier(jnp.asarray(log_dl_h)),
        log_p_dl=barrier(jnp.asarray(log_p_dl_h)),
        chi_q_grid=barrier(jnp.asarray(chi_qg_h)),
        chi_grid=barrier(jnp.asarray(chi_cg_h)),
        chi_table=barrier(jnp.asarray(chi_tab_h)),
    )

    # Flow ensemble: stacked params traced + barriered; statics in closure.
    eval_logflows = make_ensemble_log_prob_per_event(ensemble)
    group_params = tuple(
        jtu.tree_map(lambda a: barrier(a), g) for g in ensemble.group_params()
    )

    # ── selection side (identical prep to make_likelihood) ──────────────
    apix = data["apix"]
    catalogs = prepare_catalog_views(
        opts, data, universe_model, None, cache_builder=build_pixel_kde_cache
    )
    m1det_sel = barrier(jnp.asarray(data["m1detsels"]))
    m2det_sel = barrier(jnp.asarray(data["m2detsels"]))
    dL_sel = barrier(jnp.asarray(data["dLsels"]))
    chieff_sel = barrier(jnp.asarray(data["chieffsels"]))
    p_draw = barrier(jnp.asarray(data["p_draw"]))
    pixels_sel = catalogs.sample_to_unique_sel
    q_sel = barrier(m2det_sel / m1det_sel)

    em_catalog_sel = EMCatalog(
        apix=apix,
        zgals=catalogs.zgals_sel_catalog,
        dzgals=catalogs.dzgals_sel_catalog,
        wgals=catalogs.wgals_sel_catalog,
        ngals=catalogs.ngals_sel_catalog,
        delta_g_pix_z=catalogs.delta_g_pix_z,
        dN_obs_kde=catalogs.dN_obs_kde_sel,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_sel,
        unique_pixels=catalogs.unique_pixels_sel,
        sample_to_unique_idx=catalogs.sample_to_unique_sel,
    )
    gw_sel = GWEvent(
        m1det=m1det_sel,
        m2det=m2det_sel,
        dL=dL_sel,
        chieff=chieff_sel,
        prior_wt=p_draw,
        pixels=pixels_sel,
        q=q_sel,
        valid=jnp.ones_like(dL_sel, dtype=bool),
        nx=barrier(jnp.zeros_like(dL_sel)),
        ny=barrier(jnp.zeros_like(dL_sel)),
        nz=barrier(jnp.zeros_like(dL_sel)),
    )
    if sel_batch_size is not None:
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=None,
    )

    # ── jitted body (factory-local: closes over statics/closures) ───────
    _ll_flows = build_flow_loglike(
        model=model,
        eval_logflows=eval_logflows,
        group_params=group_params,
        u_base=u_base,
        m1_edges=m1_edges,
        q_edges=q_edges,
        pe_tables=pe_tables,
        support_boxes=support_boxes,
        gw_sel=gw_sel,
        em_catalog_sel=em_catalog_sel,
        Ndraw=Ndraw,
        nEvents=nEvents,
        w_full=w_full,
        sel_batch_size=sel_batch_size,
        selection_neff_soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        materialize_redshift_prior_state=materialize_redshift_prior_state,
    )

    # Jitted call body: ``parameter_decoder.decode(coord)`` used to run EAGERLY on
    # every sampler evaluation (~30 individual device ops, ~6 ms measured), even
    # though ``_ll_flows`` itself is jitted.  Folding the decode into the same trace
    # removes that per-call dispatch.  ``operands`` is threaded through as an
    # ARGUMENT of this outer jit too: calling the bound wrapper here would close
    # the arrays over the outer trace and re-embed them as HLO constants, which is
    # exactly the leak :class:`FlowOperands` exists to close.
    def _body(coord: jnp.ndarray, operands) -> jnp.ndarray:
        cosmo, survey, pop_params, _sky_params, _mark_params = parameter_decoder.decode(
            coord
        )
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}."
            )
        return _ll_flows.jitted_body(cosmo, survey, pop_params, operands)

    _jitted = jax.jit(_body)
    operands = _ll_flows.operands

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        return _jitted(coord, operands)

    # Diagnostics hooks (host-side introspection; not used by samplers).
    likelihood.jitted_body = _jitted
    likelihood.operands = operands
    likelihood.event_loglike = _ll_flows
    likelihood.flow_event_names = list(ensemble.names)
    likelihood.flows_nsamp = J
    likelihood.flows_wfull = _ll_flows.w_full
    return likelihood
