"""
likelihood.py
-------------
Hierarchical dark-siren likelihood factory.

Sentinel convention
-------------------
All log-probability floors are -jnp.inf, not finite magic numbers.

RAM note
--------
optimization_barrier MUST be applied before arrays enter any JIT closure
(i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from darksirens.redshift.catalog import attest_rows_sorted_for_windowing
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.likelihood.catalog_views import barrier, prepare_catalog_views
from darksirens.likelihood.events import pad_gw_event_to_multiple
from darksirens.likelihood.block_sizing import require_resolved_block_size
from darksirens.likelihood.core import (
    darksiren_log_likelihood,
    redshift_prior_state_sharing,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
)
from darksirens.core.constants import H0_FID, OM0_FID
from darksirens.inference.parameters import (
    build_parameter_decoder,
    complete_empty_pixel_policy_code,
)
from darksirens.core.types import EMCatalog, GWEvent
from darksirens.utils import cosmology

# Backward-compatible aliases for callers/tests that imported private helpers.
_barrier = barrier
_complete_empty_pixel_policy_code = complete_empty_pixel_policy_code


def _resolve_redshift_prior_materialization(opts) -> bool:
    """Resolve whether to keep the likelihood-internal redshift-prior barrier."""
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError(
            "redshift_prior_barrier must be one of {'auto', 'on', 'off'}, "
            f"got {mode!r}."
        )
    tinyns_cfg = getattr(opts, "tinyns_resolved_config", None) or {}
    is_tinyns_jax_rwalk = (
        getattr(opts, "sampler", None) == "tinyns"
        and tinyns_cfg.get("kernel") == "jax"
        and tinyns_cfg.get("sample") == "rwalk"
    )
    # NumPyro NUTS differentiates the likelihood w.r.t. theta; the redshift
    # prior STATE depends on theta and ``lax.optimization_barrier`` has no
    # differentiation rule, so the barrier must come off for gradient-based
    # sampling (otherwise dark/bright-siren NUTS dies with
    # NotImplementedError at the first gradient).
    is_numpyro = getattr(opts, "sampler", None) == "numpyro"
    return not (is_tinyns_jax_rwalk or is_numpyro)


def _redshift_prior_materialization_reason(opts, materialize: bool) -> str:
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode in {"on", "off"}:
        return f"forced {mode}"
    if materialize:
        return "auto -> on"
    if getattr(opts, "sampler", None) == "numpyro":
        return "auto -> off for NumPyro NUTS (optimization_barrier is not differentiable)"
    return "auto -> off for TinyNS JAX rwalk"


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


def _jit_likelihood_body(body, operands):
    """Wrap a ``(coord, operands) -> logL`` body in ``jax.jit`` and bind ``operands``.

    Why jit the factory's own closure at all
    ----------------------------------------
    Only the inner :func:`darksiren_log_likelihood` was jitted, so everything the
    returned closure did around it ran EAGERLY — and ``dynesty`` calls that closure
    once per live point (``inference/sampling.py``: ``float(np.asarray(
    likelihood(jnp.asarray(theta))))``), i.e. ~1e5-1e6 times per run.  The eager
    work is ``parameter_decoder.decode(coord)`` (~30 individual device ops: one
    gather per sampled label, then ``jnp.array`` over the parameter sub-vectors)
    plus, before the hoists above, the GW container rebuild and selection padding.
    MEASURED on the H100 NVL: 7.7 ms of 30.1 ms for the spectral single pass, 13.8
    of 51.3 ms at the production auto plan, 7.8 of 17.9 ms for a dark-siren mock.
    ``tinyns`` and ``numpyro`` escape it because they trace the closure themselves;
    this is a dynesty tax, and dynesty is the sampler of most production runs.

    Why ``operands`` is an ARGUMENT and not a closure capture
    --------------------------------------------------------
    ``jax.jit`` lowers a closed-over concrete ``jax.Array`` to a ``dense<>`` HLO
    **constant**, not to a parameter — verified on jax 0.4.34: the lowered module
    text grows ~8 bytes per array element, so the ~1.07e6-row GW arrays alone would
    add tens of MB of HLO (and a dark-siren catalog's multi-GB tables far more),
    ballooning compile time and duplicating buffers that are already device
    resident.  Threading them through as an argument keeps them exactly what they
    are today for the inner jit: already-barriered, already-resident parameters.
    The barrier contract in this module's docstring is preserved — the barriers are
    applied eagerly at build time, before the arrays enter any jit.

    The cosmology table is an operand too
    -------------------------------------
    The same argument applies to the 106.8 MB comoving-distance table in
    ``utils.cosmology``, which every likelihood reaches through ``z_of_dL`` /
    ``dL_grid_bounds`` far below this frame.  It used to be a module global, i.e.
    a closure capture of THIS jit, and cost 427.5 MB of lowered module text for
    the production spectral likelihood (two ``dense<21x41x31x500xf64>`` literals
    at ~16 bytes of text per element).  It now travels as the third jit argument
    and is installed as the active table for the trace, so the ~45 cosmology call
    sites resolve it without carrying it in their signatures.

    Recompilation: ``coord`` is a 1-D float array of fixed length (the sampled
    dimension), ``operands`` is the same pytree of the same shapes/dtypes on
    every call, and ``distance_table`` is the same buffer, so the traced
    signature is constant and the body compiles once.
    """
    distance_table = cosmology.distance_table()

    def _body_with_tables(coord: jnp.ndarray, operands, distance_table):
        with cosmology.bound_distance_table(distance_table):
            return body(coord, operands)

    jitted = jax.jit(_body_with_tables)

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        return jitted(coord, operands, distance_table)

    # Host-side hook for tests/diagnostics that want to count compilations or reach
    # the un-bound body.
    likelihood.jitted_body = jitted
    likelihood.operands = operands
    likelihood.distance_table = distance_table
    return likelihood


def _make_mixture_likelihood(
    opts,
    data: dict,
    pop_params_fid,
    fixed_parameter_values: dict | None,
    n_catalogs: int,
):
    """Build the bundle-source likelihood callable (galaxy-aware models, K >= 1).

    Each catalog in ``data["catalogs"]`` carries its OWN nside/apix, compact
    PE/selection views, KDE cache, and Q_LSS table; the GW posterior and
    selection *physics* arrays (masses, distances, spins, sky vectors) are shared
    across catalogs.  The per-catalog compact pixel maps are stacked into ``(N,
    K)`` matrices; for K >= 2 the mixture weights come from ``decode_mixture``,
    while a single-bundle K = 1 run uses the plain decoder (same parameters as a
    flat-data K = 1 run).
    """
    bundles = data.get("catalogs")
    if bundles is None or len(bundles) != n_catalogs:
        raise ValueError(
            f"n_catalogs={n_catalogs} requires data['catalogs'] with "
            f"{n_catalogs} bundles; got "
            f"{None if bundles is None else len(bundles)}."
        )

    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    universe_model = opts.universe_model
    sel_batch_size = require_resolved_block_size(
        "sel_batch_size", getattr(opts, "sel_batch_size", None))
    pe_event_block = require_resolved_block_size(
        "pe_event_block", getattr(opts, "pe_event_block", None))
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
    _mnbc = getattr(opts, "mark_names_by_catalog", None)
    mark_names_all = (
        tuple(tuple(names or ()) for names in _mnbc) if _mnbc
        else (mark_names,) + ((),) * (n_catalogs - 1)
    )
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    max_likelihood_variance = float(getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

    def _bundle_field_inputs(bundle):
        """Per-bundle FIELD-convention normalization inputs (survey-global),
        precomputed by the loader (loaders.py) or supplied directly in tests.
        Shared by the bundle's PE and selection EMCatalogs so their global Z is
        the SAME value for the same theta (constants cancel structurally).
        Includes the budget-modulation rows (Q_LSS / delta_g) when present."""
        fobs = bundle.get("field_dN_obs_s")
        if fobs is None:
            return dict(
                field_dN_obs_s=None, field_n_empty=None, field_N_obs_total=None,
                field_occupied_pixels=None, field_lss_q=None,
                field_lss_q_empty_sum=None, field_delta_g=None,
                field_lss_q_members=None, field_lss_q_empty_sum_members=None,
                field_mark_z=None, field_mark_w=None, field_mark_values=None,
                field_depth_z=None, field_depth_dz=None, field_depth_c=None,
            )

        def _maybe(key, dtype=None):
            val = bundle.get(key)
            if val is None:
                return None
            arr = jnp.asarray(val) if dtype is None else jnp.asarray(val, dtype=dtype)
            return barrier(arr)

        return dict(
            field_dN_obs_s=barrier(jnp.asarray(fobs)),
            field_n_empty=jnp.asarray(bundle["field_n_empty"], dtype=jnp.float64),
            field_N_obs_total=jnp.asarray(
                bundle["field_N_obs_total"], dtype=jnp.float64
            ),
            field_occupied_pixels=_maybe("field_occupied_pixels", jnp.int32),
            field_lss_q=_maybe("field_lss_q"),
            field_lss_q_empty_sum=_maybe("field_lss_q_empty_sum"),
            field_delta_g=_maybe("field_delta_g"),
            field_lss_q_members=_maybe("field_lss_q_members"),
            field_lss_q_empty_sum_members=_maybe("field_lss_q_empty_sum_members"),
            field_mark_z=_maybe("field_mark_z"),
            field_mark_w=_maybe("field_mark_w"),
            field_mark_values=_maybe("field_mark_values"),
            # DEPTH-consistent global observed term (loaders build these only when
            # a survey depth is active; see build_field_depth_inputs).
            field_depth_z=_maybe("field_depth_z", jnp.float64),
            field_depth_dz=_maybe("field_depth_dz", jnp.float64),
            field_depth_c=_maybe("field_depth_c", jnp.float64),
        )

    # Shared (catalog-independent) GW / selection physics arrays.
    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    def _compact_lss_q_for(views, unique_pixels):
        # Per-catalog analogue of make_likelihood._compact_lss_q: slice each
        # catalog's own Q_LSS table to its union pixels so only the compact block
        # reaches the device.
        full = views.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(views.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            return barrier(full_j), (1 if idx == 1 else idx)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    def _compact_lss_members_for(views, unique_pixels):
        # Per-catalog analogue of make_likelihood._compact_lss_members: slice
        # this catalog's (M, n_pix, n_grid) Q ensemble to the view's pixels.
        full = views.lss_completion_logq_members
        if full is None:
            return None
        full_j = jnp.asarray(full)
        idx = int(views.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            return barrier(full_j)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[1]:
            raise ValueError(
                f"LSS completion ensemble has {full_j.shape[1]} pixels but a "
                f"catalog pixel index reaches {int(jnp.max(up))} (rebuild Q "
                "over the full nside)."
            )
        return barrier(full_j[:, up])

    em_catalogs_pe = []
    em_catalogs_sel = []
    pe_pixel_cols = []
    sel_pixel_cols = []
    for bundle in bundles:
        views = prepare_catalog_views(
            opts,
            bundle,
            universe_model,
            counterpart_pixel=None,
            cache_builder=build_pixel_kde_cache,
        )
        # Each catalog uses ITS OWN pixel area (its own nside): sharing a single
        # apix across catalogs of different resolutions would silently bias the
        # per-pixel galaxy densities that enter the completion.
        apix_k = bundle["apix"]
        # Bundle union (loaders.py compacts PE-union-selection once): the two
        # views share one galaxy table / cache / unique-pixel array, so slice the
        # Q table, Q ensemble and mark rows ONCE and alias them to both
        # EMCatalogs (IS-identical leaves).  Detected by identity on the aliased
        # view arrays; a non-union bundle (e.g. a hand-built test fixture with
        # distinct PE/sel rows) keeps the separate per-view compaction below.
        bundle_union = (
            views.zgals_pe_catalog is views.zgals_sel_catalog
            and views.unique_pixels_pe is views.unique_pixels_sel
        )
        lss_q_pe_k, lss_idx_pe_k = _compact_lss_q_for(views, views.unique_pixels_pe)
        lss_qm_pe_k = _compact_lss_members_for(views, views.unique_pixels_pe)
        field_k = _bundle_field_inputs(bundle)

        def _bundle_marks(unique_pixels):
            # Gather this bundle's full-sky z-centred mark tables (loaded by
            # load_multitracer_catalog_bundles) to the view's compact rows,
            # mirroring make_likelihood._compact_marks.
            from darksirens.marks import MARK_FIELDS as _MARK_FIELDS

            out = {}
            for field in _MARK_FIELDS.values():
                full = bundle.get(field)
                if full is None:
                    out[field] = None
                else:
                    full_j = jnp.asarray(full)
                    arr = (
                        full_j if unique_pixels is None
                        else full_j[jnp.asarray(unique_pixels)]
                    )
                    out[field] = barrier(arr)
            return out

        marks_pe_k = _bundle_marks(views.unique_pixels_pe)
        if bundle_union:
            lss_q_sel_k, lss_idx_sel_k = lss_q_pe_k, lss_idx_pe_k
            lss_qm_sel_k = lss_qm_pe_k
            marks_sel_k = marks_pe_k
        else:
            lss_q_sel_k, lss_idx_sel_k = _compact_lss_q_for(
                views, views.unique_pixels_sel
            )
            lss_qm_sel_k = _compact_lss_members_for(views, views.unique_pixels_sel)
            marks_sel_k = _bundle_marks(views.unique_pixels_sel)

        em_catalogs_pe.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_pe_catalog,
            dzgals=views.dzgals_pe_catalog,
            wgals=views.wgals_pe_catalog,
            ngals=views.ngals_pe_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_pe,
            pixel_to_cache_idx=views.pixel_to_cache_idx_pe,
            unique_pixels=views.unique_pixels_pe,
            sample_to_unique_idx=views.sample_to_unique_pe,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_pe_k,
            lss_completion_logq_members=lss_qm_pe_k,
            lss_completion_indexing=lss_idx_pe_k,
            mark_logmstar=marks_pe_k["mark_logmstar"],
            mark_logssfr=marks_pe_k["mark_logssfr"],
            mark_metallicity=marks_pe_k["mark_metallicity"],
            mark_color=marks_pe_k["mark_color"],
            **field_k,
        ))
        em_catalogs_sel.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_sel_catalog,
            dzgals=views.dzgals_sel_catalog,
            wgals=views.wgals_sel_catalog,
            ngals=views.ngals_sel_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_sel,
            pixel_to_cache_idx=views.pixel_to_cache_idx_sel,
            unique_pixels=views.unique_pixels_sel,
            sample_to_unique_idx=views.sample_to_unique_sel,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_sel_k,
            lss_completion_logq_members=lss_qm_sel_k,
            lss_completion_indexing=lss_idx_sel_k,
            mark_logmstar=marks_sel_k["mark_logmstar"],
            mark_logssfr=marks_sel_k["mark_logssfr"],
            mark_metallicity=marks_sel_k["mark_metallicity"],
            mark_color=marks_sel_k["mark_color"],
            **field_k,
        ))
        pe_pixel_cols.append(jnp.asarray(views.sample_to_unique_pe, dtype=jnp.int32))
        sel_pixel_cols.append(jnp.asarray(views.sample_to_unique_sel, dtype=jnp.int32))

    # Stack the per-catalog compact pixel maps into (N, K) matrices.
    pixels_pe = barrier(jnp.stack(pe_pixel_cols, axis=1))
    pixels_sel = barrier(jnp.stack(sel_pixel_cols, axis=1))

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=None,
    )

    gw_pe = GWEvent(
        m1det=m1det_pe,
        m2det=m2det_pe,
        dL=dL_pe,
        chieff=chieff_pe,
        prior_wt=p_pe,
        pixels=pixels_pe,
        q=q_pe,
        valid=jnp.ones_like(dL_pe, dtype=bool),
        nx=nx_pe,
        ny=ny_pe,
        nz=nz_pe,
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
        nx=nx_sel,
        ny=ny_sel,
        nz=nz_sel,
    )
    if sel_batch_size is not None:
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

    em_catalog_pe_0 = em_catalogs_pe[0]
    em_catalog_sel_0 = em_catalogs_sel[0]
    mixture_em_catalogs_pe = tuple(em_catalogs_pe[1:])
    mixture_em_catalogs_sel = tuple(em_catalogs_sel[1:])

    # Decide PE/selection state sharing EAGERLY (concrete EMCatalogs, before jit
    # erases object identity): a union bundle's two views share every consumed
    # leaf, so darksiren_log_likelihood builds each catalog's redshift-prior
    # state once instead of twice.  Static (a tuple of bools).
    share_prior_state_by_catalog = redshift_prior_state_sharing(
        universe_model, em_catalogs_pe, em_catalogs_sel
    )

    # Device operands travel as jit ARGUMENTS (see :func:`_jit_likelihood_body` for
    # why closing over them would embed them as HLO constants instead).
    operands = (
        gw_pe, em_catalog_pe_0, gw_sel, em_catalog_sel_0,
        mixture_em_catalogs_pe, mixture_em_catalogs_sel,
    )

    # Arm the windowed catalog-KDE evaluator for the traced path: the catalogs
    # cross the jit boundary as ARGUMENTS, so the evaluator sees tracers and
    # cannot verify the z-sort invariant itself — verify every bound view HERE,
    # while the arrays are concrete (see catalog.attest_rows_sorted_for_windowing).
    attest_rows_sorted_for_windowing(
        em_catalog_pe_0, em_catalog_sel_0,
        *mixture_em_catalogs_pe, *mixture_em_catalogs_sel,
    )

    def _body(coord: jnp.ndarray, operands) -> jnp.ndarray:
        (
            gw_pe_, em_catalog_pe_0_, gw_sel_, em_catalog_sel_0_,
            mixture_em_catalogs_pe_, mixture_em_catalogs_sel_,
        ) = operands
        if n_catalogs >= 2:
            (
                cosmo,
                surveys,
                pop_params,
                sky_params,
                mark_params_all,
                log_w,
            ) = parameter_decoder.decode_mixture(coord)
            survey_0 = surveys[0]
            mixture_surveys = tuple(surveys[1:])
            mark_params = mark_params_all[0]
        else:
            # K = 1 bundle source: the plain decoder (no sticks, no per-catalog
            # blocks) -- the same parameters a flat-data K=1 run samples.
            cosmo, survey_0, pop_params, sky_params, mark_params = (
                parameter_decoder.decode(coord)
            )
            mixture_surveys = ()
            log_w = None
            mark_params_all = (mark_params,)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}."
            )
        return darksiren_log_likelihood(
            cosmo,
            survey_0,
            pop_params,
            gw_pe_,
            em_catalog_pe_0_,
            gw_sel_,
            em_catalog_sel_0_,
            nEvents,
            nsamp,
            Ndraw,
            pop_model,
            universe_model,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            sky_model=sky_model,
            sky_params=sky_params,
            mark_model=mark_model,
            mark_params=mark_params,
            mark_names=mark_names,
            mark_params_all=tuple(mark_params_all),
            mark_names_all=mark_names_all,
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            lss_marginalize=bool(getattr(opts, "lss_marginalize", False)),
            n_catalogs=n_catalogs,
            mixture_surveys=mixture_surveys,
            mixture_em_catalogs_pe=mixture_em_catalogs_pe_,
            mixture_em_catalogs_sel=mixture_em_catalogs_sel_,
            mixture_log_weights=log_w,
            catalog_sky_weighting=catalog_sky_weighting,
            share_prior_state_by_catalog=share_prior_state_by_catalog,
        )

    return _jit_likelihood_body(_body, operands)


def make_likelihood(opts, data: dict, pop_params_fid, fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    This wrapper prepares static catalog/GW views, decodes sampler coordinates,
    and delegates the pure JIT likelihood evaluation to
    :func:`darksirens.likelihood.core.darksiren_log_likelihood`.
    """
    # Flow-surrogate path: per-event normalizing flows replace the stored PE
    # samples (mutually exclusive with --gw_path; validated by the CLI).
    if data.get("flow_ensemble") is not None:
        from darksirens.likelihood.flow_events import make_flow_likelihood

        return make_flow_likelihood(
            opts, data, pop_params_fid, fixed_parameter_values
        )

    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    apix = data["apix"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    universe_model = opts.universe_model
    sel_batch_size = require_resolved_block_size(
        "sel_batch_size", getattr(opts, "sel_batch_size", None))
    pe_event_block = require_resolved_block_size(
        "pe_event_block", getattr(opts, "pe_event_block", None))
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    max_likelihood_variance = float(getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

    # Weak-lensing magnification backend (resolved up front, before the heavy
    # catalog-view prep, so a missing WL config fails fast).  All values are
    # inert when wl_backend == WL_BACKEND_DISABLED, preserving behaviour for
    # every non-WL universe model.
    wl_backend = WL_BACKEND_DISABLED
    wl_a = jnp.asarray(0.0)
    wl_b = jnp.asarray(0.0)
    wl_z_grid = jnp.asarray([0.0, 1.0])
    wl_log_mu_grid = jnp.asarray([0.0, 1.0])
    wl_log_p_table = jnp.asarray([[0.0, 0.0], [0.0, 0.0]])
    wl_params = data.get("wl_params")
    if universe_model == "spectral_sirens_wl":
        if wl_params is None:
            raise ValueError(
                "universe_model='spectral_sirens_wl' requires data['wl_params'] "
                "to be present."
            )
        backend = int(wl_params.backend)
        if backend == WL_BACKEND_LOGNORMAL:
            wl_backend = WL_BACKEND_LOGNORMAL
            wl_a = jnp.asarray(wl_params.a)
            wl_b = jnp.asarray(wl_params.b)
        elif backend == WL_BACKEND_TABULATED:
            wl_backend = WL_BACKEND_TABULATED
            wl_z_grid = jnp.asarray(wl_params.z_grid)
            wl_log_mu_grid = jnp.asarray(wl_params.log_mu_grid)
            wl_log_p_table = jnp.asarray(wl_params.log_p_table)
        else:
            raise ValueError(
                "Unsupported weak-lensing backend in data['wl_params']: "
                f"{backend}. Expected {WL_BACKEND_LOGNORMAL} (LOGNORMAL) or "
                f"{WL_BACKEND_TABULATED} (TABULATED)."
            )
    # Selection-side WL marginalization (opt-in; lognormal backend only —
    # the core silently keeps the legacy selection path otherwise, mirroring
    # the cluster wrapper's semantics).
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if getattr(opts, "wl_selection", "standard") == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )

    counterpart_pixel = data.get("counterpart_pixel")
    counterpart_pixels = (
        barrier(jnp.asarray(data["counterpart_pixels"], dtype=jnp.int32))
        if data.get("counterpart_pixels") is not None else None
    )
    counterpart_zs = (
        barrier(jnp.asarray(data["counterpart_zs"], dtype=float))
        if data.get("counterpart_zs") is not None else None
    )
    counterpart_dzs = (
        barrier(jnp.asarray(data["counterpart_dzs"], dtype=float))
        if data.get("counterpart_dzs") is not None else None
    )
    bright_siren_sky_marginalized = bool(
        data.get(
            "bright_siren_sky_marginalized",
            getattr(opts, "bright_siren_sky_marginalized", False),
        )
    )

    # Bundle operand source: per-catalog bundles in data["catalogs"] (built by
    # loaders.load_multitracer_catalog_bundles) for ANY K >= 1; required for
    # K >= 2, where the catalog-completed redshift prior becomes a per-catalog
    # mixture.  The flat-data path below stays for K = 1 callers without
    # bundles and is bit-identical to the pre-unification behaviour.
    n_catalogs = int(getattr(opts, "n_catalogs", 1))
    if data.get("catalogs") is not None or n_catalogs >= 2:
        return _make_mixture_likelihood(
            opts, data, pop_params_fid, fixed_parameter_values, n_catalogs
        )

    catalogs = prepare_catalog_views(
        opts,
        data,
        universe_model,
        counterpart_pixel,
        cache_builder=build_pixel_kde_cache,
    )

    # Slice the (global, host-side) Q_LSS table to each view's union pixels, so
    # only the compact (n_union, n_grid) block becomes a device/jit operand
    # rather than the full (n_pix, n_grid) table.  Returns (compact_logq, indexing).
    def _compact_lss_q(unique_pixels):
        full = catalogs.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            # already compact, or a legacy full catalog (rows are global pixels)
            return barrier(full_j), (1 if idx == 1 else idx)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    lss_q_pe, lss_idx_pe = _compact_lss_q(catalogs.unique_pixels_pe)
    lss_q_sel, lss_idx_sel = _compact_lss_q(catalogs.unique_pixels_sel)

    # Slice the (optional) Q_LSS ENSEMBLE (M, n_pix, n_grid) to each view's union
    # pixels the same way, for the fully-Bayesian marginalisation (--lss_marginalize).
    def _compact_lss_members(unique_pixels):
        full = catalogs.lss_completion_logq_members
        if full is None:
            return None
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1 or unique_pixels is None:
            return barrier(full_j)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[1]:
            raise ValueError(
                f"LSS completion ensemble has {full_j.shape[1]} pixels but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[:, up])

    lss_qm_pe = _compact_lss_members(catalogs.unique_pixels_pe)
    lss_qm_sel = _compact_lss_members(catalogs.unique_pixels_sel)
    lss_marginalize = bool(getattr(opts, "lss_marginalize", False))

    # Per-galaxy marks: gathered to the compact catalog rows using the SAME
    # unique-pixel map that compacts zgals, so they align row-for-row.  None
    # (mark absent) flows through to the legacy galaxy-count host model.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS

    def _compact_marks(unique_pixels):
        out = {}
        for field in _MARK_FIELDS.values():
            full = data.get(field)
            if full is None:
                out[field] = None
            else:
                full = jnp.asarray(full)
                arr = full if unique_pixels is None else full[jnp.asarray(unique_pixels)]
                out[field] = barrier(arr)
        return out

    marks_pe = _compact_marks(catalogs.unique_pixels_pe)
    marks_sel = _compact_marks(catalogs.unique_pixels_sel)

    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    pixels_pe = catalogs.sample_to_unique_pe
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    pixels_sel = catalogs.sample_to_unique_sel
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=data.get("wl_params"),
    )

    # The PE / selection EMCatalogs are coord-INDEPENDENT (every field is a
    # static, already-barriered array), so build them ONCE here rather than on
    # every likelihood call.  Building them eagerly also resolves the
    # redshift-prior-state sharing verdict NOW, before jit erases the leaf object
    # identity it depends on: on the flat union path PE and selection share every
    # consumed leaf, so the state is built once instead of twice.
    em_catalog_pe = EMCatalog(
        apix=apix,
        zgals=catalogs.zgals_pe_catalog,
        dzgals=catalogs.dzgals_pe_catalog,
        wgals=catalogs.wgals_pe_catalog,
        ngals=catalogs.ngals_pe_catalog,
        delta_g_pix_z=catalogs.delta_g_pix_z,
        dN_obs_kde=catalogs.dN_obs_kde_pe,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_pe,
        unique_pixels=catalogs.unique_pixels_pe,
        sample_to_unique_idx=catalogs.sample_to_unique_pe,
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        active_counterpart_index=0,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        lss_completion_logq=lss_q_pe,
        lss_completion_logq_members=lss_qm_pe,
        lss_completion_indexing=lss_idx_pe,
        mark_logmstar=marks_pe["mark_logmstar"],
        mark_logssfr=marks_pe["mark_logssfr"],
        mark_metallicity=marks_pe["mark_metallicity"],
        mark_color=marks_pe["mark_color"],
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=getattr(catalogs, "field_lss_q", None),
        field_lss_q_empty_sum=getattr(catalogs, "field_lss_q_empty_sum", None),
        # PER-MEMBER survey-global Q rows: required by the field-convention
        # scope guard (redshift/prior.py) whenever the catalog carries a Q
        # ENSEMBLE.  prepare_catalog_views builds them; omitting them here made
        # every K=1 `--lss_marginalize` run under the DEFAULT field weighting
        # abort with "requires the per-member survey-global Q rows" before the
        # first likelihood evaluation.
        field_lss_q_members=getattr(catalogs, "field_lss_q_members", None),
        field_lss_q_empty_sum_members=getattr(
            catalogs, "field_lss_q_empty_sum_members", None
        ),
        field_delta_g=getattr(catalogs, "field_delta_g", None),
        field_mark_z=getattr(catalogs, "field_mark_z", None),
        field_mark_w=getattr(catalogs, "field_mark_w", None),
        field_mark_values=getattr(catalogs, "field_mark_values", None),
        field_depth_z=getattr(catalogs, "field_depth_z", None),
        field_depth_dz=getattr(catalogs, "field_depth_dz", None),
        field_depth_c=getattr(catalogs, "field_depth_c", None),
    )
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
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        active_counterpart_index=0,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        lss_completion_logq=lss_q_sel,
        lss_completion_logq_members=lss_qm_sel,
        lss_completion_indexing=lss_idx_sel,
        mark_logmstar=marks_sel["mark_logmstar"],
        mark_logssfr=marks_sel["mark_logssfr"],
        mark_metallicity=marks_sel["mark_metallicity"],
        mark_color=marks_sel["mark_color"],
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=getattr(catalogs, "field_lss_q", None),
        field_lss_q_empty_sum=getattr(catalogs, "field_lss_q_empty_sum", None),
        # PER-MEMBER survey-global Q rows: required by the field-convention
        # scope guard (redshift/prior.py) whenever the catalog carries a Q
        # ENSEMBLE.  prepare_catalog_views builds them; omitting them here made
        # every K=1 `--lss_marginalize` run under the DEFAULT field weighting
        # abort with "requires the per-member survey-global Q rows" before the
        # first likelihood evaluation.
        field_lss_q_members=getattr(catalogs, "field_lss_q_members", None),
        field_lss_q_empty_sum_members=getattr(
            catalogs, "field_lss_q_empty_sum_members", None
        ),
        field_delta_g=getattr(catalogs, "field_delta_g", None),
        field_mark_z=getattr(catalogs, "field_mark_z", None),
        field_mark_w=getattr(catalogs, "field_mark_w", None),
        field_mark_values=getattr(catalogs, "field_mark_values", None),
        field_depth_z=getattr(catalogs, "field_depth_z", None),
        field_depth_dz=getattr(catalogs, "field_depth_dz", None),
        field_depth_c=getattr(catalogs, "field_depth_c", None),
    )
    share_prior_state_by_catalog = redshift_prior_state_sharing(
        universe_model, (em_catalog_pe,), (em_catalog_sel,)
    )

    # The GW containers are coord-INDEPENDENT too — every field is an
    # already-barriered data constant — so build them, and apply the selection
    # padding, ONCE here rather than on every likelihood call.  Building them
    # inside the (eager) closure re-ran two ``jnp.ones_like`` allocations plus, via
    # ``make_gw_event`` inside ``pad_gw_event_to_multiple``, 11 concatenates and 11
    # optimization barriers over the ~1.07e6-injection arrays on EVERY sampler
    # evaluation: 13.9 ms and 83 MB of identical fresh device buffers per call at
    # the production ``sel_batch_size=32768`` (1,067,946 % 32768 != 0, so the
    # padding never takes its early return).  ``_make_mixture_likelihood`` has
    # always hoisted them; this is the unpropagated twin.
    gw_pe = GWEvent(
        m1det=m1det_pe,
        m2det=m2det_pe,
        dL=dL_pe,
        chieff=chieff_pe,
        prior_wt=p_pe,
        pixels=pixels_pe,
        q=q_pe,
        valid=jnp.ones_like(dL_pe, dtype=bool),
        nx=nx_pe,
        ny=ny_pe,
        nz=nz_pe,
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
        nx=nx_sel,
        ny=ny_sel,
        nz=nz_sel,
    )
    if sel_batch_size is not None:
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

    # Device operands are passed as jit ARGUMENTS, never closed over — see
    # :func:`_jit_likelihood_body`.
    operands = (
        gw_pe, em_catalog_pe, gw_sel, em_catalog_sel,
        wl_a, wl_b, wl_z_grid, wl_log_mu_grid, wl_log_p_table,
    )

    # Same attestation as the mixture factory: window the traced catalogs only
    # after verifying the concrete arrays here at build time.
    attest_rows_sorted_for_windowing(em_catalog_pe, em_catalog_sel)

    def _body(coord: jnp.ndarray, operands) -> jnp.ndarray:
        (
            gw_pe_, em_catalog_pe_, gw_sel_, em_catalog_sel_,
            wl_a_, wl_b_, wl_z_grid_, wl_log_mu_grid_, wl_log_p_table_,
        ) = operands
        cosmo, survey, pop_params, sky_params, mark_params = parameter_decoder.decode(coord)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}. "
                "Verify parameter-space construction for this population model."
            )

        if shared_beta and shared_spin and shared_gamma:
            return darksiren_log_likelihood(
                cosmo,
                survey,
                pop_params,
                gw_pe_,
                em_catalog_pe_,
                gw_sel_,
                em_catalog_sel_,
                nEvents,
                nsamp,
                Ndraw,
                pop_model,
                universe_model,
                sel_batch_size=sel_batch_size,
                pe_event_block=pe_event_block,
                sky_model=sky_model,
                sky_params=sky_params,
                mark_model=mark_model,
                mark_params=mark_params,
                mark_names=mark_names,
                wl_backend=wl_backend,
                wl_a=wl_a_,
                wl_b=wl_b_,
                wl_z_grid=wl_z_grid_,
                wl_log_mu_grid=wl_log_mu_grid_,
                wl_log_p_table=wl_log_p_table_,
                wl_selection=wl_selection,
                lss_marginalize=lss_marginalize,
                materialize_redshift_prior_state=materialize_redshift_prior_state,
                selection_neff_soft_guard=selection_neff_soft_guard,
                max_likelihood_variance=max_likelihood_variance,
                catalog_sky_weighting=catalog_sky_weighting,
                share_prior_state_by_catalog=share_prior_state_by_catalog,
            )
        return darksiren_log_likelihood(
            cosmo,
            survey,
            pop_params,
            gw_pe_,
            em_catalog_pe_,
            gw_sel_,
            em_catalog_sel_,
            nEvents,
            nsamp,
            Ndraw,
            pop_model,
            universe_model,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            sky_model=sky_model,
            sky_params=sky_params,
            mark_model=mark_model,
            mark_params=mark_params,
            mark_names=mark_names,
            wl_backend=wl_backend,
            wl_a=wl_a_,
            wl_b=wl_b_,
            wl_z_grid=wl_z_grid_,
            wl_log_mu_grid=wl_log_mu_grid_,
            wl_log_p_table=wl_log_p_table_,
            wl_selection=wl_selection,
            lss_marginalize=lss_marginalize,
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            catalog_sky_weighting=catalog_sky_weighting,
            share_prior_state_by_catalog=share_prior_state_by_catalog,
        )

    return _jit_likelihood_body(_body, operands)
