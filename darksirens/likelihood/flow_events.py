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
    cell_centers,
    make_mass_q_edges,
    resolve_mass_grid_bounds,
    sample_histogram_trunc,
    sample_m1_given_q_trunc,
    sample_q_marginal_trunc,
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


def chi_eff_prior_logpdf(
    q_frac: jnp.ndarray,
    chieff: jnp.ndarray,
    q_grid: jnp.ndarray,
    chi_grid: jnp.ndarray,
    table: jnp.ndarray,
) -> jnp.ndarray:
    """JAX port of gwcat.spin.ChiEffPrior._interp2d + logprob.

    Bilinear interpolation in PROBABILITY (matching gwcat exactly), then
    log clipped at -50.  ``q_frac = m1/(m1+m2) = 1/(1+q)`` with q = m2/m1.
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
    p = v0 * (1.0 - q_f) + v1 * q_f
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
    sel_batch_size: int | None = None,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    materialize_redshift_prior_state: bool = True,
):
    """Return the jitted ``(cosmo, survey, pop_params) -> logL`` flow body.

    All non-hashable statics (population-model closures, flow ensemble
    structure) are closure-captured; array operands must already be
    barrier-wrapped by the caller.  See the module docstring for the
    estimator.

    Event-windowed proposal: every event draws its own J points from the
    population target truncated to its support box (``support_boxes`` from
    :func:`darksirens.gw.flows.compute_support_boxes`; the dL window maps to
    a z window under the CURRENT cosmology each call).  The truncation
    normalisers enter the exact proposal density, so this is plain
    importance sampling — unbiased for any box covering the flow's support —
    and it rescues the effective sample size of narrow (low-mass / low-z)
    events that a single population-wide proposal starves.

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
    m1_centers = cell_centers(m1_edges)
    q_centers = cell_centers(q_edges)
    chi_edges = jnp.linspace(-1.0, 1.0, 513)
    chi_centers = cell_centers(chi_edges)
    box_m1det = support_boxes["m1det"]  # (nEvents, 2)
    box_q = support_boxes["q"]
    box_dL = support_boxes["dL"]
    box_chi = support_boxes["chieff"]
    box_mc = support_boxes["mc_det"]
    box_chi_ab = support_boxes["chi_ab"]      # chi ~ a + b q linear band
    box_chi_resid = support_boxes["chi_resid"]

    def _prepare_state(cosmo, survey):
        return prepare_redshift_prior_state(
            "spectral_sirens",
            cosmo,
            survey,
            em_catalog_sel,
            materialize_state=materialize_redshift_prior_state,
        )

    def _event_ldw(cosmo, pop_params, prior_state):
        """Per-event log importance weights ldw (nEvents, J) — the flow term."""
        H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

        # -- population draws (source frame; common random numbers) --------
        # Shared per-call ingredients: the mass-q target on the grid, the
        # spin parameters, and the redshift target on the module zgrid.
        tm = model.mixture_theta(pop_params)

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

        def _event_draws(u_e, m1det_win, q_win, dL_win, chi_win, mc_win,
                         chi_ab, chi_resid):
            # z window from the event's dL window under the CURRENT cosmology.
            z_lo = z_of_dL(jnp.clip(dL_win[0], dL_lo_g, dL_hi_g), H0, Om0, w0, wa)
            z_hi = z_of_dL(jnp.clip(dL_win[1], dL_lo_g, dL_hi_g), H0, Om0, w0, wa)
            zs = sample_histogram_trunc(u_e[:, 3], zgrid, log_z_cell_dens, z_lo, z_hi)
            z = zs.x

            # q from the m1-marginalised target, then m1 within the event's
            # detector-frame CHIRP-MASS band at the drawn (q, z) — the thin
            # constant-Mc ridge of well-measured events — intersected with
            # the m1det box.  All windows are exactly corrected in log_s.
            qs = sample_q_marginal_trunc(
                u_e[:, 1], m1_edges, q_edges, log_t_cells_fl, q_win
            )
            q = qs.x
            g_q = (1.0 + q) ** 0.2 / q**0.6  # m1 = Mc * g(q)
            m1_lo_j = jnp.maximum(
                mc_win[0] / (1.0 + z) * g_q, m1det_win[0] / (1.0 + z)
            )
            m1_hi_j = jnp.minimum(
                mc_win[1] / (1.0 + z) * g_q, m1det_win[1] / (1.0 + z)
            )
            m1_hi_j = jnp.maximum(m1_hi_j, m1_lo_j)
            m1src, log_s_m1 = sample_m1_given_q_trunc(
                u_e[:, 0], m1_edges, log_t_cells_fl, qs.cell, m1_lo_j, m1_hi_j
            )
            log_s_mq = qs.log_s + log_s_m1

            # chi_eff window: the (q, chi_eff) degeneracy band at the drawn q,
            # intersected with the event's chi_eff box and [-1, 1].
            chi_mid = chi_ab[0] + chi_ab[1] * q
            chi_lo_j = jnp.clip(
                jnp.maximum(chi_mid + chi_resid[0], chi_win[0]), -1.0, 1.0
            )
            chi_hi_j = jnp.clip(
                jnp.minimum(chi_mid + chi_resid[1], chi_win[1]), -1.0, 1.0
            )
            chi_hi_j = jnp.maximum(chi_hi_j, chi_lo_j)
            if spin_is_truncnorm:
                chi = truncnorm_sample(u_e[:, 2], ts[0], ts[1], chi_lo_j, chi_hi_j)
            else:
                chi = sample_histogram_trunc(
                    u_e[:, 2], chi_edges, log_chi_cells, chi_lo_j, chi_hi_j
                )
            chieff, log_s_chi = chi.x, chi.log_s

            m1det = m1src * (1.0 + z)
            m2det = q * m1det
            dL = dL_of_z(z, H0, Om0, w0, wa)

            log_t = (
                model.log_p_massspin(m1src, q, chieff, pop_params)
                + model.log_rate_z(z, pop_params)
                + jnp.interp(z, zgrid, log_pvol_nodes)
            )
            log_s = log_s_mq + log_s_chi + zs.log_s

            log_pi_pe = jnp.interp(
                jnp.log(dL), pe_tables.log_dl_grid, pe_tables.log_p_dl
            ) + chi_eff_prior_logpdf(
                1.0 / (1.0 + q),
                chieff,
                pe_tables.chi_q_grid,
                pe_tables.chi_grid,
                pe_tables.chi_table,
            )

            base = log_t - log_s - log_pi_pe
            base = jnp.where(jnp.isfinite(base), base, -jnp.inf)
            X = jnp.stack([m1det, m2det, dL, chieff], axis=-1)
            return X, base

        X, base = jax.vmap(_event_draws)(
            u_base, box_m1det, box_q, box_dL, box_chi, box_mc,
            box_chi_ab, box_chi_resid,
        )  # (nEvents, J, 4), (nEvents, J)

        logflows = eval_logflows(group_params, X)  # (nEvents, J)
        ldw = logflows + base
        return jnp.where(jnp.isfinite(ldw), ldw, -jnp.inf)

    def _ll_flows_impl(cosmo, survey, pop_params):
        H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        prior_state = _prepare_state(cosmo, survey)

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
            gw_sel,
            em_catalog_sel,
            log_weight,
            Ndraw,
            nEvents,
            sel_batch_size=sel_batch_size,
        )

        ldw = _event_ldw(cosmo, pop_params, prior_state)
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

    def _event_diagnostics_impl(cosmo, survey, pop_params):
        """Per-event lnZ_i, delta-method variance, and ESS (no selection)."""
        prior_state = _prepare_state(cosmo, survey)
        ldw = _event_ldw(cosmo, pop_params, prior_state)
        event_lls, event_vars = jax.vmap(
            lambda row: log_evidence_and_mc_variance(row, J)
        )(ldw)
        w = jnp.exp(ldw - jnp.max(ldw, axis=1, keepdims=True))
        ess = jnp.sum(w, axis=1) ** 2 / jnp.maximum(
            jnp.sum(w**2, axis=1), jnp.finfo(w.dtype).tiny
        )
        return event_lls, event_vars, ess

    ll = jax.jit(_ll_flows_impl)
    ll.event_diagnostics = jax.jit(_event_diagnostics_impl)
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
        sel_batch_size=sel_batch_size,
        selection_neff_soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        materialize_redshift_prior_state=materialize_redshift_prior_state,
    )

    # Jitted call body: ``parameter_decoder.decode(coord)`` used to run EAGERLY on
    # every sampler evaluation (~30 individual device ops, ~6 ms measured), even
    # though ``_ll_flows`` itself is jitted.  Folding the decode into the same trace
    # removes that per-call dispatch; ``_ll_flows``' own operands are unchanged
    # (they were, and remain, captured by that inner jit).
    def _body(coord: jnp.ndarray) -> jnp.ndarray:
        cosmo, survey, pop_params, _sky_params, _mark_params = parameter_decoder.decode(
            coord
        )
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}."
            )
        return _ll_flows(cosmo, survey, pop_params)

    _jitted = jax.jit(_body)

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        return _jitted(coord)

    # Diagnostics hooks (host-side introspection; not used by samplers).
    likelihood.jitted_body = _jitted
    likelihood.flow_event_names = list(ensemble.names)
    likelihood.flows_nsamp = J
    return likelihood
