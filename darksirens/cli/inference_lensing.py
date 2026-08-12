#!/usr/bin/env python3
"""
darksirens_inference_lensing.py
===============================
Full hierarchical inference CLI for the darksirens **strong-lensing branch**,
covering the joint singleton + J=2 cluster (pair) likelihood.

This CLI is the **sole owner of the weak-lensing (``spectral_sirens_wl``)
universe model**: the WL singleton channel selected via ``--wl_backend`` (the
stock ``darksirens_inference`` no longer exposes ``spectral_sirens_wl``). It
adds, on top of that WL singleton fit:

  * loading of the lensed-injection set + per-pair PE,
  * the cluster master likelihood (``darksiren_log_likelihood_with_clusters``),
  * ``--cluster_mode {off,j2}`` so it subsumes the pure-singleton fit,
  * memory-safe PE down-sampling for the O(N_pe^2 N_y) pair likelihood.

It REUSES the branch's machinery wherever possible: prior construction,
``run_sampler`` (tinyns/dynesty/numpyro), result saving, and the singleton data
loaders. Only the cluster-specific assembly is added here.

Design notes
------------
* Singleton-only run: ``--cluster_mode off`` (no ``--lensed_*`` paths needed) is
  numerically the commit-2 / branch singleton likelihood.
* Joint run: ``--cluster_mode j2`` requires ``--lensed_injections_path`` and
  ``--pair_pe_path`` and a ``--partition_path`` (the candidate-pair list).
* WL: ``--wl_backend {lognormal,tabulated,disabled}``; lognormal uses
  ``--lensing_wl_a/b``, tabulated reads ``--lensing_wl_table_path`` (HDF5 with
  ``z_grid``/``log_mu_grid``/``log_p_table``), with optional
  singleton-selection matching via ``--wl_selection``.

Examples
--------
# Joint singleton + J=2, WL on, NUTS on GPU (recommended for 12-D):
darksirens_inference_lensing \
    --gw_path              mock_gw_pe.h5 \
    --gwselection_path     mock_gw_selection.h5 \
    --lensed_injections_path mock_lensed_injections.h5 \
    --pair_pe_path         mock_pair_pe.h5 \
    --partition_path       partition.json \
    --pop_model            powerlaw+peak \
    --cluster_mode         j2 \
    --wl_backend           lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 \
    --fix_cosmology true --fix_survey true \
    --sampler numpyro --nuts_warmup 500 --nuts_samples 2000 --nuts_chains 4 \
    --pe_max_per_pair 400 \
    --save_path ./run_joint/

# Pure singleton (no pairs), tinyns (rwalk + jax kernel):
darksirens_inference_lensing \
    --gw_path mock_gw_pe.h5 --gwselection_path mock_gw_selection.h5 \
    --pop_model powerlaw+peak --cluster_mode off \
    --wl_backend lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 \
    --fix_cosmology true --fix_survey true \
    --sampler tinyns --nlive 2000 --tinyns_sample rwalk --tinyns_kernel jax \
    --save_path ./run_singleton/
"""

from __future__ import annotations

# JAX memory configuration (before any JAX import)
from darksirens.core.jax_config import configure_jax_runtime

configure_jax_runtime()

import os
import sys
import json
import time
import argparse
import dataclasses
import datetime
import traceback
import warnings

import numpy as np
import h5py
try:
    import healpy as hp
except ModuleNotFoundError:
    class _HealpyFallback:
        __version__ = "unavailable"

        @staticmethod
        def nside2npix(nside):
            return 12 * int(nside) * int(nside)

        @staticmethod
        def nside2pixarea(nside):
            return 4.0 * np.pi / _HealpyFallback.nside2npix(nside)

    hp = _HealpyFallback()
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from scipy.special import logsumexp

# ── branch machinery we reuse ────────────────────────────────────────────────
from darksirens.redshift import zgrid
from darksirens.core.types import (
    EMCatalog,
    GWEvent,
)
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.gw.populations.utils import normalization_grid_settings

from darksirens.inference.checkpointing import (
    add_checkpoint_arguments,
    find_resume_target,
    resolve_checkpoint_plan,
)
from darksirens.inference.run_fingerprint import (
    ResumeFingerprintError,
    build_run_fingerprint,
    check_resume_fingerprint,
    save_run_fingerprint,
)
from darksirens.inference.prior import (
    build_parameter_space,
    make_prior_transform,
    resolve_joint_prior_constraints,
)
from darksirens.inference.sampling import run_sampler
from darksirens.io.results import (
    save_tinyns_diagnostics_json,
    write_dead_point_datasets,
    write_tinyns_metadata,
)
from darksirens.io.settings import environment_block
from darksirens.inference.tinyns_config import (
    add_tinyns_arguments,
    build_tinyns_config,
    TINYNS_RESOLVED_DISPLAY_KEYS,
)
from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.likelihood.block_sizing import (
    block_size_arg,
    resolve_block_sizes,
    sampler_block_sizing_profile,
)
from darksirens.cli.common import (
    _banner,
    _section,
    _row,
    _end,
    _ok,
    _warn,
    _err,
    add_normalization_grid_arguments,
    configure_normalization_grids_for_model,
    str_to_bool,
    parse_json_arg,
    _print_all_cli_options,
    resolve_redshift_prior_barrier,
    resolve_selection_neff_guard,
    run_cli,
)
from darksirens.inference.parameters import (
    build_parameter_decoder,
)
from darksirens.gw.samples import load_gw_samples, load_selection_samples

# ── lensing / cluster pieces ─────────────────────────────────────────────────
from darksirens.lensing.slmarks import make_sis_lens_params, DEFAULT_T0_SECONDS
from darksirens.lensing.wlmagnification import (
    make_lognormal_wl_params,
    make_tabulated_wl_params,
    validate_wl_mu_quadrature,
)
from darksirens.lensing.grids import make_wl_mu_quadrature
from darksirens.lensing.lensed_injections import load_lensed_injections
from darksirens.lensing.pair_tag_selection import (
    PAIR_TAG_SELECTION_MODEL_KINDS,
    make_pair_tag_selection_model,
    load_pair_tag_selection_file,
)
from darksirens.lensing.observed_catalog import (
    observed_catalog_metadata_from_hdf5,
    validate_observed_catalog_file,
)
from darksirens.lensing.partitions import (
    CandidatePair,
    PartitionState,
    exact_partitions_from_pairs,
    exact_partition_components,
    combine_component_partitions,
    prepare_candidate_pairs_for_partitioning,
    parse_edge_mark_keys,
)
from darksirens.lensing.preflight import (
    fixed_partition_coverage_error,
    run_lensing_preflight,
)
from darksirens.lensing.marginal_diagnostics import (
    compute_marginalized_partition_diagnostics,
    compute_componentwise_factorized_partition_diagnostics,
    candidate_time_mark_suspicion,
    resolve_sis_time_mark_impl,
    SIS_TIME_MARK_SHARPNESS,
    sis_time_mark_support,
    sis_time_mark_support_message,
)
from darksirens.likelihood.pair_kde import (
    make_pair_kde,
    stack_pair_kdes,
    validate_pair_prior_wt,
)
from darksirens.likelihood.cluster_selection import combined_selection_log_correction
from darksirens.likelihood.likelihood_with_clusters import (
    darksiren_log_likelihood_with_clusters,
    darksiren_likelihood_diagnostics_with_clusters,
    CLUSTER_MODE_J2,
    CLUSTER_MODE_OFF,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
    WL_BACKEND_DISABLED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
    PAIR_MARKS_NONE,
    PAIR_MARKS_TIME,
    PAIR_MARKS_TIME_DELTA,
    SINGLETON_LENSING_OFF,
    SINGLETON_LENSING_MIXTURE,
)
from darksirens.lensing.lensed_injections import (
    DEFAULT_PAIR_ORIENTATION_MODE,
    load_lensed_single_image_set,
    pair_orientation_mismatch_message,
    read_fc_pdet_attrs,
    read_pair_orientation_mode,
)
from darksirens.lensing.fcpdet import make_fc_pdet_params


def _pair_tag_log_probs_from_options(opts, lensed):
    if lensed is None:
        return jnp.zeros(0)
    kind = getattr(opts, "pair_tag_model", "constant")
    perturb = float(getattr(opts, "pair_tag_perturb_logit", 0.0) or 0.0)
    if kind == "file":
        model = load_pair_tag_selection_file(opts.pair_tag_selection_path, perturb_logit=perturb)
    elif kind == "constant" and perturb == 0.0 and float(getattr(opts, "pair_tag_constant", 1.0)) == 1.0:
        # Preserve legacy behavior: consume the p_tag dataset written by older mocks.
        fallback = jnp.zeros(int(np.asarray(lensed.m1_src).shape[0]))
        return jnp.asarray(getattr(lensed, "log_p_tag_per_source", fallback))
    else:
        model = make_pair_tag_selection_model(
            kind,
            constant=float(getattr(opts, "pair_tag_constant", 1.0)),
            perturb_logit=perturb,
        )
    missing = []
    fields = {}
    for name in model.required_fields:
        # required_fields is the single source of truth for the field name; the
        # true_delta_t alias is resolved by the loader, not here.
        arr = np.asarray(getattr(lensed, name))
        if arr.size == 0 or np.all(~np.isfinite(arr)):
            missing.append(name)
        fields[name] = arr
    if missing:
        raise SystemExit(f"pair_tag_model={model.kind} requires lensed injection fields: {missing}")
    return jnp.asarray(model.log_probability(**fields))

# =============================================================================
# Local lensing-parameter space
# =============================================================================
LENS_PARAMETER_PRIORS = {
    "log10_tau_A": (-7.0, -2.0),
    "tau_n": (0.0, 6.0),
}


def _split_lensing_fixed_parameters(fixed):
    """Separate shared/base fixed values from lensing-only fixed values."""
    if fixed is None:
        fixed = {}
    if not isinstance(fixed, dict):
        raise TypeError("fixed_parameter_values must be a JSON object")

    base_fixed, lens_fixed = {}, {}
    for label, value in fixed.items():
        try:
            value_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fixed value for {label!r} must be a finite number"
            ) from exc
        if not np.isfinite(value_float):
            raise ValueError(f"fixed value for {label!r} must be finite")
        if label in LENS_PARAMETER_PRIORS:
            lens_fixed[label] = value_float
        else:
            base_fixed[label] = value_float
    return base_fixed, lens_fixed


def _build_lens_parameter_space(opts, fixed_parameter_values, lens_prior_overrides):
    """Return sampled SIS optical-depth labels and bounds for this lensing CLI.

    The main darksirens parameter machinery intentionally remains unchanged:
    these parameters are spectral-siren lensing-only hyperparameters and are
    appended locally when ``--fix_lens_rate false``.  A label present in
    ``fixed_parameter_values`` is treated as fixed, which allows tiny recovery
    runs such as sampling ``log10_tau_A`` while fixing ``tau_n``.
    """
    if getattr(opts, "fix_lens_rate", True):
        return [], np.asarray([], dtype=float), np.asarray([], dtype=float)

    labels, lower, upper = [], [], []
    for label, default_bounds in LENS_PARAMETER_PRIORS.items():
        if label in fixed_parameter_values:
            continue
        bounds = lens_prior_overrides.get(label, default_bounds)
        if len(bounds) != 2:
            raise ValueError(f"lens prior override for {label!r} must be [lo, hi]")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(f"invalid lens prior bounds for {label!r}: {bounds!r}")
        labels.append(label)
        lower.append(lo)
        upper.append(hi)
    return labels, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _sl_T0_seconds(opts) -> float:
    """The configured SIS time-delay scale T0, in seconds."""
    return float(getattr(opts, "sl_T0_sec", None) or DEFAULT_T0_SECONDS)


def _decode_lens_params(coord, sampled_labels, fixed_parameter_values, opts):
    """Construct ``SISLensParams`` from fixed CLI values or sampled lens coords."""
    T0 = _sl_T0_seconds(opts)
    if getattr(opts, "fix_lens_rate", True):
        return make_sis_lens_params(
            A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n, T0_seconds=T0,
        )

    values = {label: jnp.asarray(coord)[i] for i, label in enumerate(sampled_labels)}
    values.update(
        {label: float(value) for label, value in fixed_parameter_values.items()}
    )
    log10_tau_A = values.get("log10_tau_A", np.log10(float(opts.sl_tau_A)))
    tau_n = values.get("tau_n", float(opts.sl_tau_n))
    return make_sis_lens_params(
        A_tau=10.0**log10_tau_A, n_tau=tau_n, T0_seconds=T0,
    )


#: Which lens label each archived SIS optical-depth value is decoded from, so a
#: SAMPLED parameter's archived value can be named for the point it was
#: evaluated at instead of masquerading as the run's value.
_LENS_VALUE_SOURCE_LABEL = {"lens_A_tau": "log10_tau_A", "lens_n_tau": "tau_n"}


def _lens_settings_dict(
    coord, sampled_labels, fixed_parameter_values, opts, eval_point="prior_midpoint"
):
    """Archive the SIS optical-depth block for one evaluation point.

    ``coord`` is a DIAGNOSTICS point (the prior midpoint, or a guard-clear prior
    draw), never a posterior summary, so a SAMPLED lens parameter has no
    run-level value here.  Writing it under the bare name ``lens_A_tau`` let a
    downstream table read a pre-posterior number -- in practice
    ``10**((lo+hi)/2)`` from the prior box, which tracks the prior bounds rather
    than the data -- as "the SIS optical-depth normalisation of this run".
    Mirror the partition-diagnostics convention: bare names ONLY when the value
    genuinely equals the run's fixed value (``--fix_lens_rate true``, or the
    label pinned via ``--fixed_parameter_values``), otherwise prefix with
    ``{eval_point}_`` and stamp ``lens_parameters_eval_point``.
    """
    sis = _decode_lens_params(coord, sampled_labels, fixed_parameter_values, opts)
    lens_labels = [
        label for label in sampled_labels if label in LENS_PARAMETER_PRIORS
    ]
    out = dict(
        lens_labels=lens_labels,
        fix_lens_rate=bool(getattr(opts, "fix_lens_rate", True)),
        sl_tau_A=float(opts.sl_tau_A),
        sl_tau_n=float(opts.sl_tau_n),
        # T0 is a run-configuration constant (never sampled), so a bare name is
        # correct here; the sampled A_tau/n_tau go through the per-parameter
        # bare-vs-eval-point gate below.
        sl_T0_sec=float(np.asarray(sis.T0)),
    )
    any_sampled = False
    for key, value in (
        ("lens_A_tau", float(np.asarray(sis.A_tau))),
        ("lens_n_tau", float(np.asarray(sis.n_tau))),
    ):
        if _LENS_VALUE_SOURCE_LABEL[key] in lens_labels:
            out[f"{eval_point}_{key}"] = value
            any_sampled = True
        else:
            out[key] = value
    if any_sampled:
        out["lens_parameters_eval_point"] = str(eval_point)
    return out


def _write_lens_settings_attrs(attrs, lens_settings):
    """Splat a ``_lens_settings_dict`` payload into an HDF5 attrs mapping.

    A loop rather than fixed keys because whether ``lens_A_tau`` is bare or
    ``{eval_point}_``-prefixed depends on which lens parameters are sampled.
    """
    for key, value in lens_settings.items():
        if key == "lens_labels":
            attrs["lens_labels"] = json.dumps(value)
        elif isinstance(value, bool):
            attrs[key] = bool(value)
        elif isinstance(value, str):
            attrs[key] = value
        else:
            attrs[key] = float(value)


def _decode_base_parameters(decoder, coord):
    """Decode only the base-parameter prefix from a combined sampler vector."""
    coord = jnp.asarray(coord)
    sampled_labels = getattr(decoder, "sampled_labels", None)
    if sampled_labels is None:
        return decoder.decode(coord)
    return decoder.decode(coord[: len(sampled_labels)])

# =============================================================================
# Container builders (mirror darksirens.likelihood.factory)
# =============================================================================
def _empty_em_catalog(nside=1):
    npix = hp.nside2npix(nside)
    return EMCatalog(
        apix=hp.nside2pixarea(nside),
        zgals=jnp.full((npix, 1), 0.1),
        dzgals=jnp.full((npix, 1), 0.02),
        wgals=jnp.ones((npix, 1)),
        ngals=jnp.ones(npix, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((npix, len(zgrid))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=None,
        sample_to_unique_idx=None,
        counterpart_pixel=None,
        counterpart_pixels=None,
        counterpart_zs=None,
        counterpart_dzs=None,
        active_counterpart_index=0,
        bright_siren_sky_marginalized=False,
    )


def _gw_event(m1det, m2det, dL, chieff, prior_wt):
    m1det = jnp.asarray(m1det)
    m2det = jnp.asarray(m2det)
    dL = jnp.asarray(dL)
    chieff = jnp.asarray(chieff)
    return GWEvent(
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        prior_wt=jnp.asarray(prior_wt),
        pixels=jnp.zeros_like(dL, dtype=jnp.int32),
        q=m2det / m1det,
        valid=jnp.ones_like(dL, dtype=bool),
    )


def _event_gps_times_from_catalog(raw_catalog):
    """Per-event arrival times (s) from an observed catalog, or None.

    Only returned when EVERY event carries a finite ``gps_time``: the signed
    arrival order is either known for the whole catalog or not used at all.
    """
    if not isinstance(raw_catalog, dict):
        return None
    events = raw_catalog.get("events")
    if not isinstance(events, list) or not events:
        return None
    times = []
    for ev in events:
        t = (ev or {}).get("gps_time")
        if t is None:
            return None
        try:
            t = float(t)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(t):
            return None
        times.append(t)
    return np.asarray(times, dtype=float)


def _orient_time_marks(candidate_pairs, gps_times, opts=None):
    """Give each time mark the sign of ``t_j - t_i`` for its stored order.

    For SIS the type-I minimum arrives BEFORE the type-II saddle
    (Delta t = t_- - t_+ = T0 y > 0), so the two image-assignment branches of
    the pair likelihood predict opposite signs of the observed delay and at
    most one of them is compatible with the data. The recorded mark is a
    MAGNITUDE (the candidate builder writes abs(t_i - t_j)), so we restore the
    sign from the catalog's arrival times, keeping the mark's magnitude and
    sigma untouched.

    Two inconsistencies are fatal here rather than warned about, because both
    put a mark the run cannot justify into the time-marked pair likelihood:

    * a non-finite ``delta_t_obs``/``sigma_delta_t`` (NaN/inf propagates
      straight through the y-integral), and
    * a mark magnitude that does not reproduce the catalog's arrival
      separation -- the mark and the catalog then describe different events,
      and the old behaviour silently kept the MARK's magnitude with the
      CATALOG's sign, i.e. a delay no input actually claims.

    ``--allow_time_mark_mismatch true`` downgrades both to loud warnings and
    restores exactly that legacy behaviour, so deliberate ablations remain
    possible.  A sign-only disagreement (magnitude agrees, stored order
    reversed) is not an inconsistency: resolving it is this function's job.

    Returns ``(pairs, signed)``. ``signed`` is True only when every marked pair
    could be oriented; otherwise the pairs are returned unchanged and the
    likelihood keeps the legacy |dt|-in-both-branches behaviour.
    """
    if candidate_pairs is None:
        return candidate_pairs, False
    marked = [p for p in candidate_pairs if p.delta_t_obs is not None]
    if not marked:
        return candidate_pairs, False
    allow_mismatch = str_to_bool(getattr(opts, "allow_time_mark_mismatch", False))
    # Ahead of the gps_times gate below: a broken mark is broken whether or not
    # the catalog can orient it.
    nonfinite = [
        (int(p.i), int(p.j), float(p.delta_t_obs),
         float(p.sigma_delta_t) if p.sigma_delta_t is not None else float("nan"))
        for p in marked
        if not np.isfinite(float(p.delta_t_obs))
        or (p.sigma_delta_t is not None and not np.isfinite(float(p.sigma_delta_t)))
    ]
    if nonfinite:
        shown = ", ".join(
            f"({i},{j}): delta_t_obs={dt:.6g}, sigma_delta_t={sig:.6g}"
            for i, j, dt, sig in nonfinite[:5]
        )
        if len(nonfinite) > 5:
            shown += f", ... (+{len(nonfinite) - 5} more)"
        message = (
            f"{len(nonfinite)}/{len(marked)} time-marked candidate edges carry a "
            f"non-finite time mark: {shown}. A NaN/inf delay or width propagates "
            "through the marked pair likelihood instead of being rejected. Fix "
            "the candidate-pair file, or pass --allow_time_mark_mismatch true to "
            "proceed anyway."
        )
        if not allow_mismatch:
            raise SystemExit(f"non-finite candidate time marks: {message}")
        _warn(message)
    if gps_times is None:
        return candidate_pairs, False
    n = int(gps_times.shape[0])
    out = []
    inconsistent = []
    for p in candidate_pairs:
        if p.delta_t_obs is None or not (0 <= int(p.i) < n and 0 <= int(p.j) < n):
            out.append(p)
            continue
        dt_catalog = float(gps_times[int(p.j)] - gps_times[int(p.i)])
        dt_mark = abs(float(p.delta_t_obs))
        # Validate the convention instead of trusting it: the recorded mark
        # magnitude must reproduce the catalog's arrival separation.
        sig = abs(float(p.sigma_delta_t)) if p.sigma_delta_t is not None else 0.0
        tol = max(5.0 * sig, 1e-6 * max(dt_mark, 1.0))
        if abs(abs(dt_catalog) - dt_mark) > tol:
            inconsistent.append((int(p.i), int(p.j), dt_mark, dt_catalog))
        sign = 1.0 if dt_catalog >= 0.0 else -1.0
        out.append(
            CandidatePair(
                p.i, p.j, p.log_prior_odds, p.label,
                dataclasses.replace(p.marks, delta_t_obs=sign * dt_mark),
            )
        )
    if inconsistent:
        # Report the WORST edge, not the first: with many marked edges the first
        # disagreement is rarely the informative one.
        i, j, dt_mark, dt_catalog = max(
            inconsistent, key=lambda e: abs(abs(e[3]) - e[2])
        )
        message = (
            f"{len(inconsistent)}/{len(marked)} time-marked candidate edges "
            "disagree with the observed catalog's arrival times, worst edge "
            f"({i},{j}): |marks.delta_t_obs| = {dt_mark:.6g} s vs "
            f"|t_j - t_i| = {abs(dt_catalog):.6g} s. The mark and the catalog "
            "therefore describe different events; --allow_time_mark_mismatch "
            "true proceeds with the mark's magnitude and the catalog's arrival "
            "ORDER, but verify first that the two agree on which event came "
            "first."
        )
        if not allow_mismatch:
            raise SystemExit(f"time-mark magnitude disagreement: {message}")
        _warn(message)
    # Every marked pair oriented iff every marked pair had in-range endpoints.
    ok = all(0 <= int(p.i) < n and 0 <= int(p.j) < n for p in marked)
    return (out, True) if ok else (candidate_pairs, False)


def _time_mark_arrays_for_partition_state(state, candidate_pairs):
    """Return edge-level time marks ordered like ``state.pair_indices``."""
    dt_obs = []
    dt_sigma = []
    for edge_idx in np.asarray(state.candidate_edge_indices, dtype=int):
        pair = candidate_pairs[int(edge_idx)]
        if pair.delta_t_obs is None or pair.sigma_delta_t is None:
            raise SystemExit(
                "candidate pair "
                f"({pair.i},{pair.j}) missing marks.delta_t_obs/sigma_delta_t "
                "required by pair_marks=time"
            )
        # The mark is passed through with whatever sign it carries. When
        # _orient_time_marks resolved the arrival order from the catalog, that
        # sign selects the image assignment (see cluster_log_likelihood_pair's
        # pair_time_signed); otherwise it is already a magnitude.
        dt = float(pair.delta_t_obs)
        sig = float(pair.sigma_delta_t)
        if not np.isfinite(dt) or not np.isfinite(sig) or sig <= 0:
            raise SystemExit(
                "candidate pair "
                f"({pair.i},{pair.j}) has invalid marks.delta_t_obs/sigma_delta_t "
                "required by pair_marks=time"
            )
        dt_obs.append(dt)
        dt_sigma.append(sig)
    return (
        jnp.asarray(dt_obs, dtype=jnp.float64),
        jnp.asarray(dt_sigma, dtype=jnp.float64),
    )


# =============================================================================
# Data loading
# =============================================================================

def _all_singleton_partition_state(n_events: int) -> PartitionState:
    return PartitionState(
        np.arange(int(n_events), dtype=np.int32),
        np.asarray([], dtype=np.int32).reshape((0, 2)),
        int(n_events),
        0,
        0.0,
        np.asarray([], dtype=np.int32),
    )


def _full_state_for_component(
    n_events: int, component: dict, local_state: PartitionState
) -> PartitionState:
    component_events = {int(x) for x in component["event_indices"]}
    outside_singletons = [idx for idx in range(int(n_events)) if idx not in component_events]
    singletons = np.asarray(
        outside_singletons + np.asarray(local_state.singleton_indices, dtype=int).tolist(),
        dtype=np.int32,
    )
    pair_indices = np.asarray(local_state.pair_indices, dtype=np.int32).reshape((-1, 2))
    edge_indices = np.asarray(
        local_state.candidate_edge_indices if local_state.candidate_edge_indices is not None else [],
        dtype=np.int32,
    )
    if edge_indices.size:
        order = np.argsort(edge_indices)
        edge_indices = edge_indices[order]
        pair_indices = pair_indices[order]
    return PartitionState(
        np.sort(singletons).astype(np.int32),
        pair_indices,
        int(singletons.size),
        int(pair_indices.shape[0]),
        float(local_state.log_prior_weight),
        edge_indices,
    )


def _runtime_part_from_state(
    state: PartitionState,
    candidate_pairs,
    *,
    pair_marks: str = "none",
    pair_time_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> dict:
    part = dict(
        singleton_indices=jnp.asarray(state.singleton_indices, dtype=jnp.int32),
        pair_indices=jnp.asarray(state.pair_indices, dtype=jnp.int32),
        n_singletons=int(state.n_singletons),
        n_pairs=int(state.n_pairs),
        log_prior_weight=float(state.log_prior_weight),
        candidate_edge_indices=jnp.asarray(
            state.candidate_edge_indices if state.candidate_edge_indices is not None else [],
            dtype=jnp.int32,
        ),
    )
    if pair_marks == "time":
        if pair_time_override is not None:
            dt_obs, dt_sigma = pair_time_override
        else:
            dt_obs, dt_sigma = _time_mark_arrays_for_partition_state(state, candidate_pairs)
        part["pair_time_delta_t_obs"] = dt_obs
        part["pair_time_sigma"] = dt_sigma
    return part


def _count_probe_partition_state(n_events: int, n_pairs: int) -> PartitionState:
    pairs = np.asarray([[2 * k, 2 * k + 1] for k in range(int(n_pairs))], dtype=np.int32)
    used = {int(x) for x in pairs.reshape(-1)} if pairs.size else set()
    singletons = np.asarray([idx for idx in range(int(n_events)) if idx not in used], dtype=np.int32)
    return PartitionState(
        singletons,
        pairs.reshape((-1, 2)),
        int(singletons.size),
        int(n_pairs),
        0.0,
        np.asarray([], dtype=np.int32),
    )


def _count_correction_closed_form(baseline_raw, opts):
    """Closed-form count-only selection correction, as a function of the
    partition's ``(n_singletons, n_pairs)``.

    The selection integrals (mu, sigma^2 per channel) do not depend on the
    partition — only the marked-Poisson correction's counts do — so the whole
    componentwise factorization needs just this scalar function of the counts.
    Evaluating full-likelihood probes per count instead cost ~n_pairs extra
    likelihood evaluations PER SAMPLER CALL and one multi-GB XLA specialization
    per distinct ``(n_singletons, n_pairs)`` — the host-RAM OOM that killed
    paper-scale j2 on 58 GB nodes.

    ``pe_variance_sum`` matters here even though it does not depend on the
    counts: ``selection0`` CANCELS EXACTLY out of
    ``baseline + LSE_k(dp_k + count_delta_k)``, so this closed form IS the
    sampler-facing selection correction and the master likelihood's value never
    reaches the sampler on the factorized path.  Without it the guard reverted
    to the selection-only bound ``N_obs^2/max_likelihood_variance`` and the band
    ``N^2/max_var < Neff < N^2/(max_var - pe_var)`` went unguarded — the exact
    defect P1-09 / ``tests/test_cluster_pe_variance_guard.py`` threaded it for
    (review F-001).  The BASELINE's ``pe_variance_sum`` is the right scalar: it
    keeps the invariant ``count_delta[0] == 0`` the factorization assumes, and
    the residual partition dependence (``pair_variance_sum`` depends on WHICH
    pairs are formed) is what ``build_cluster_diagnostics``' count-only
    cross-check verifies is negligible at the evaluation point.

    Shared by the sampler path and the diagnostics path so the two cannot drift
    onto different variance budgets (review F-005).
    """
    def _count_correction(n_sing, n_prs):
        return combined_selection_log_correction(
            baseline_raw["log_mu_singleton"],
            baseline_raw["log_sigma2_singleton"],
            baseline_raw["log_mu_cluster"],
            baseline_raw["log_sigma2_cluster"],
            n_singletons_observed=n_sing,
            n_clusters_observed=n_prs,
            soft_guard=bool(getattr(opts, "selection_neff_soft_guard", False)),
            max_likelihood_variance=float(
                getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE)
            ),
            pe_variance_sum=baseline_raw["pe_variance_sum"],
        )

    return _count_correction


def _factorized_logsumexp_jax(component_terms, count_loglike_delta):
    max_pairs = int(count_loglike_delta.shape[0]) - 1
    dp = jnp.full((max_pairs + 1,), -jnp.inf, dtype=jnp.float64)
    dp = dp.at[0].set(0.0)
    for terms in component_terms:
        new = jnp.full((max_pairs + 1,), -jnp.inf, dtype=jnp.float64)
        for n_pairs, logw in terms:
            n_pairs = int(n_pairs)
            valid = max_pairs + 1 - n_pairs
            if valid <= 0:
                continue
            proposed = dp[:valid] + logw
            current = new[n_pairs : n_pairs + valid]
            new = new.at[n_pairs : n_pairs + valid].set(jnp.logaddexp(current, proposed))
        dp = new
    return jax_logsumexp(dp + count_loglike_delta)
def _downsample(arrs_dict, n_keep, rng):
    """Down-sample a dict of per-sample arrays to n_keep (memory control for the
    O(N_pe^2) pair KDE). Returns the same dict with shorter arrays."""
    n = len(next(iter(arrs_dict.values())))
    if n_keep >= n:
        return arrs_dict
    idx = rng.choice(n, size=n_keep, replace=False)
    return {k: np.asarray(v)[idx] for k, v in arrs_dict.items()}


def _normalize_pair_image_prior_wt(prior_wt, *, context):
    """Validate and normalize one pair-image PE ``prior_wt`` array.

    Singleton PE loading normalizes ``p_pe`` independently for each event.
    Pair PE is read directly rather than through ``load_gw_samples``, so we
    apply the matching convention here: every image's finite positive
    proposal-density weights are divided by their own sum.  This keeps pair
    likelihood constants independent of arbitrary PE prior-density scale.
    """
    prior_wt = validate_pair_prior_wt(prior_wt, context=context)
    norm = float(np.sum(prior_wt))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(
            f"{context}: finite positive prior_wt values must have a "
            f"finite positive sum, got {norm}."
        )
    return prior_wt / norm


def extract_event_samples_from_gw_pe(
    gw_pe_arrays, event_index, nsamp, pe_max=0, rng=None
):
    """Extract one observed-event PE sample block from event-major GW PE arrays."""
    m1det, m2det, dL, chieff, p_pe = gw_pe_arrays
    event_index = int(event_index)
    sl = slice(event_index * nsamp, (event_index + 1) * nsamp)
    d = dict(
        m1det=np.asarray(m1det[sl]),
        q=np.asarray(m2det[sl]) / np.asarray(m1det[sl]),
        dL_app=np.asarray(dL[sl]),
        chieff=np.asarray(chieff[sl]),
        prior_wt=np.asarray(p_pe[sl]),
    )
    if pe_max and pe_max > 0:
        if rng is None:
            rng = np.random.default_rng()
        d = _downsample(d, int(pe_max), rng)
        # Do NOT renormalise after downsampling: load_gw_samples already
        # normalised prior_wt over the FULL nsamp per event, and the
        # singleton/driven-sample roles keep that convention. Renormalising
        # the subset multiplied the KDE estimand by ~ nsamp/pe_max, silently
        # suppressing every pairing by log(nsamp/pe_max) (~1.6 nats at the
        # 2000->400 defaults) in marginalize_exact partition posteriors and
        # j2-vs-off evidence comparisons (library review; found independently
        # by both lensing reviewers with matching measured offsets). The
        # unbiased subset convention is weights summing to ~ N_sub/N_full.
        d["prior_wt"] = validate_pair_prior_wt(
            d["prior_wt"], context=f"gw_pe event {event_index}/prior_wt"
        )
        return d
    d["prior_wt"] = _normalize_pair_image_prior_wt(
        d["prior_wt"], context=f"gw_pe event {event_index}/prior_wt"
    )
    return d


def make_pair_kdes_from_gw_pe(
    gw_pe_arrays, event_indices, nsamp, pe_max_per_pair=0, rng=None
):
    """Build per-observed-event KDEs directly from unified GW PE samples."""
    kdes = []
    for event_index in event_indices:
        d = extract_event_samples_from_gw_pe(
            gw_pe_arrays, int(event_index), nsamp, pe_max_per_pair, rng
        )
        kdes.append(
            make_pair_kde(d["m1det"], d["q"], d["dL_app"], d["chieff"], d["prior_wt"])
        )
    return kdes


def _gate_suspicious_time_marks(opts, candidate_pairs_raw) -> None:
    """Refuse placeholder/synthetic time marks that would enter the likelihood.

    Placeholder marks (integer-second delta_t with sigma_delta_t = 1 s) are
    astronomically sharp on the ~T0 SIS time-delay scale and silently dominate
    the y-integral. Preflight has always warned about them; the runtime path
    previously ran anyway. Raise unless --allow_suspicious_time_marks true,
    and only when time marks are actually in use (pair_marks=time or a
    time/delta_t_obs edge-mark likelihood key).
    """
    time_like_keys = set(
        parse_edge_mark_keys(getattr(opts, "edge_mark_likelihood_keys", ""))
    )
    time_marks_in_use = (
        getattr(opts, "pair_marks", "none") == "time"
        or bool(time_like_keys & {"time", "delta_t_obs"})
    )
    if not time_marks_in_use or not candidate_pairs_raw:
        return
    suspicion = candidate_time_mark_suspicion(candidate_pairs_raw)
    if not suspicion.get("candidate_time_marks_suspicious"):
        return
    message = (
        f"{suspicion.get('candidate_time_marks_warning')} "
        f"(candidate_pairs_path={getattr(opts, 'candidate_pairs_path', None)}). "
        "Pass --allow_suspicious_time_marks true to proceed anyway."
    )
    if str_to_bool(getattr(opts, "allow_suspicious_time_marks", False)):
        warnings.warn(message, RuntimeWarning)
        return
    raise SystemExit(f"suspicious candidate time marks: {message}")


def _validate_partition_mode_against_cluster_mode(opts):
    """Reject ``--cluster_mode off --partition_mode marginalize_exact``.

    ``off`` is the singleton-only control: every observed event is its own
    cluster, so there are no candidate pairs to marginalise over and the only
    compatible partition is the all-singleton one ``--partition_mode fixed``
    already evaluates.  The combination was not merely unimplemented, it was
    silently accepted: load_inputs returns early for off mode BEFORE the
    partition-mode requirement checks (so not even --candidate_pairs_path was
    demanded) and run_lensing_preflight skips every partition check unless
    cluster_mode == 'j2', so preflight reported ok, the run directory and
    settings.json were written, the PE/selection load ran for minutes, and only
    then did the likelihood closure -- which branches on the flag alone -- raise
    KeyError: 'marginal_partitions' at the midpoint smoke test.  The natural way
    in is copying the campaign's joint command and flipping only
    --cluster_mode.
    """
    if (
        getattr(opts, "cluster_mode", "j2") == "off"
        and getattr(opts, "partition_mode", "fixed") == "marginalize_exact"
    ):
        raise SystemExit(
            "--partition_mode marginalize_exact requires --cluster_mode j2 "
            "(the off control is singleton-only: there are no candidate pairs "
            "to marginalise over). Drop --partition_mode (or set it to fixed) "
            "for the off control run."
        )


def _validate_fixed_cosmology_for_lensed_channels(opts):
    """Reject ``--fix_cosmology false`` while a lensed-injection channel is on.

    Both lensed selection channels (``compute_lensed_pair_selection_term`` for
    ``--cluster_mode j2`` and ``compute_lensed_single_selection_term`` for
    ``--singleton_lensing sl_mixture``) reweight SOURCE-FRAME campaign columns,
    and the detection efficiency enters ONLY through subset membership — the
    per-image flags rendered at generation time.  Detection is a function of the
    observables (m1_det, dL_app = dL(z)/sqrt(mu)) and hence of cosmology, so at
    fixed (m1_src, q, z, y) a change in H0/w0/wa changes P_det while the stored
    flags cannot follow: the estimators are valid only AT THE CAMPAIGN'S
    FIDUCIAL COSMOLOGY (cluster_selection module docstring, "Frozen detection
    realization").  The unlensed singleton campaign is stored in the detector
    frame and IS valid for any cosmology, so sampling cosmology puts the two
    channels of mu_tot on different selection conventions and gives the ratio
    that sets A_tau a spurious cosmology dependence.

    This cannot be caught at runtime: the campaign file keeps only the selected
    subsets and records no fiducial cosmology, and H0 arrives traced.  Gate it
    on the flags instead (review F-031; every in-repo lensing invocation already
    passes ``--fix_cosmology true``, the default).
    """
    if getattr(opts, "fix_cosmology", True):
        return
    channels = []
    if getattr(opts, "cluster_mode", "off") == "j2":
        channels.append("--cluster_mode j2")
    if getattr(opts, "singleton_lensing", "off") == "sl_mixture":
        channels.append("--singleton_lensing sl_mixture")
    if not channels:
        return
    raise SystemExit(
        f"--fix_cosmology false is not valid with {' and '.join(channels)}: the "
        "lensed-injection selection terms reweight source-frame campaign columns "
        "whose detection flags were rendered at the campaign's fiducial "
        "cosmology, so they are only valid there (the file records no fiducial "
        "cosmology and H0 arrives traced, so this cannot be checked at "
        "runtime). Sampling cosmology would give mu_sel^(2)/mu_sel^(1) — the "
        "ratio that sets A_tau — a spurious cosmology dependence. Run with "
        "--fix_cosmology true (the default), or use --cluster_mode off "
        "--singleton_lensing off for a cosmology-sampling control."
    )


def _derive_universe_model(wl_backend):
    """Map --wl_backend to the library universe-model kind.

    Both WL backends (lognormal and tabulated) drive the ``spectral_sirens_wl``
    universe model; ``disabled`` falls back to plain ``spectral_sirens``.
    """
    return (
        "spectral_sirens_wl"
        if wl_backend in ("lognormal", "tabulated")
        else "spectral_sirens"
    )


def _wl_backend_code(opts):
    """Map the --wl_backend string to the library integer WL backend code."""
    return {
        "lognormal": WL_BACKEND_LOGNORMAL,
        "tabulated": WL_BACKEND_TABULATED,
        "disabled": WL_BACKEND_DISABLED,
    }[opts.wl_backend]


def _load_wl_table_arrays(opts):
    """Load the tabulated-WL grids as jnp arrays for the cluster likelihood.

    Returns an empty dict for every backend except ``tabulated`` (so it splats
    inertly into the load_inputs bundles). For ``tabulated`` it reads the HDF5
    table (datasets ``z_grid`` (Nz,), ``log_mu_grid`` (Nmu,),
    ``log_p_table`` (Nz, Nmu)) and returns ``wl_z_grid``/``wl_log_mu_grid``/
    ``wl_log_p_table``. Missing/unspecified --lensing_wl_table_path is a clean
    SystemExit.

    The loaded table is validated against the mu-quadrature the likelihood
    actually integrates on (``make_wl_mu_quadrature``): a table narrower than
    the quadrature's node spacing makes int p_WL(mu|z) dmu collapse to ~0 and
    silently DELETES those events from the likelihood. That is a hard startup
    error, never a silent rescale.

    The grids themselves (finite, >= 2 points, strictly increasing, matching
    table shape) are validated here too, via ``make_tabulated_wl_params``:
    this loader is the ONLY eager chokepoint for the arrays the likelihood
    interpolates on, because the likelihood builds its closure from tracers
    (``make_tabulated_log_p_wl(..., validate=False)``).
    """
    if getattr(opts, "wl_backend", None) != "tabulated":
        return {}
    path = getattr(opts, "lensing_wl_table_path", None)
    if not path:
        raise SystemExit(
            "--wl_backend tabulated requires --lensing_wl_table_path"
        )
    if not os.path.exists(path):
        raise SystemExit(f"--lensing_wl_table_path does not exist: {path}")
    with h5py.File(path, "r") as f:
        wl_z_grid = jnp.asarray(f["z_grid"][()], dtype=jnp.float64)
        wl_log_mu_grid = jnp.asarray(f["log_mu_grid"][()], dtype=jnp.float64)
        wl_log_p_table = jnp.asarray(f["log_p_table"][()], dtype=jnp.float64)
    try:
        make_tabulated_wl_params(wl_z_grid, wl_log_mu_grid, wl_log_p_table)
    except ValueError as exc:
        raise SystemExit(
            f"--lensing_wl_table_path {path}: malformed WL table. {exc}"
        ) from exc
    mu_nodes, log_w_nodes = make_wl_mu_quadrature()
    try:
        validate_wl_mu_quadrature(
            mu_nodes, log_w_nodes, wl_z_grid, wl_log_mu_grid, wl_log_p_table,
            context=f"--wl_backend tabulated ({path})",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return dict(
        wl_z_grid=wl_z_grid,
        wl_log_mu_grid=wl_log_mu_grid,
        wl_log_p_table=wl_log_p_table,
    )


def _gate_pair_orientation_mismatch(opts):
    """Refuse an inference-side --pair_orientation_mode the campaign was not
    rendered with, while the affected channel is enabled.

    The generator records ``pair_orientation_mode`` as a file attr (files from
    older generators lack it and were always rendered 'independent').  The
    runtime flag selects the lensed-singleton censoring factor, so a mismatch
    mis-normalises the J=2 / lensed-singleton channel ratio that sets A_tau by
    up to ~2.6x (see darksirens.lensing.fcpdet).

    That ratio only enters the likelihood through the lensed-singleton channel,
    so the verdict is gated on it:

    * ``--singleton_lensing sl_mixture`` -- fatal, because the run's headline
      lens-rate normalisation would be wrong by a factor no output records;
    * ``off`` -- warning, the censoring factor is not evaluated at all.

    ``--allow_pair_orientation_mismatch true`` downgrades the fatal case to a
    loud warning so deliberate convention ablations remain possible.
    """
    path = getattr(opts, "lensed_injections_path", None)
    if not path:
        return
    try:
        campaign_mode = read_pair_orientation_mode(path)
    except OSError:
        return
    _require_pair_orientation_match(opts, campaign_mode, path)


def _require_pair_orientation_match(opts, campaign_mode, path):
    """The verdict half of ``_gate_pair_orientation_mismatch``.

    Split out so the lensed-singleton loader can re-check against the campaign
    attrs IT read, at the point where the censoring factor is actually built,
    instead of trusting a startup check on a path that may since have changed.
    """
    run_mode = getattr(
        opts, "pair_orientation_mode", DEFAULT_PAIR_ORIENTATION_MODE
    )
    if campaign_mode == run_mode:
        return
    message = pair_orientation_mismatch_message(path, campaign_mode, run_mode)
    channel_enabled = getattr(opts, "singleton_lensing", "off") == "sl_mixture"
    if not channel_enabled:
        warnings.warn(
            f"{message} (--singleton_lensing off, so the censoring factor is "
            "not evaluated in this run.)",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if str_to_bool(getattr(opts, "allow_pair_orientation_mismatch", False)):
        warnings.warn(
            f"{message} Proceeding under --allow_pair_orientation_mismatch: "
            "the lens-rate normalisation of this run is NOT trustworthy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise SystemExit(
        f"pair_orientation_mode mismatch: {message} Pass "
        "--allow_pair_orientation_mismatch true to proceed anyway."
    )


def _load_singleton_lensing_inputs(opts):
    """Load the lensed-singleton channel inputs (sl_mixture), or inert Nones.

    Requires --lensed_injections_path in EITHER cluster mode (the Mould-style
    per-event-only ablation is cluster_mode=off + sl_mixture). Finn-Chernoff
    detection constants come from the injection file's fc_* attrs, overridable
    via --fc_rho_thr/--fc_r0/--fc_mc_bar.
    """
    if getattr(opts, "singleton_lensing", "off") != "sl_mixture":
        return dict(lensed_singles=None, fc_pdet_params=None)
    if not opts.lensed_injections_path:
        raise SystemExit(
            "--singleton_lensing sl_mixture requires --lensed_injections_path"
        )
    lensed_singles, campaign = load_lensed_single_image_set(
        opts.lensed_injections_path, return_campaign_attrs=True
    )
    # Last net, at the point where the censoring factor enters the likelihood.
    _require_pair_orientation_match(
        opts, campaign["pair_orientation_mode"], opts.lensed_injections_path
    )
    if lensed_singles.n_kept == 0:
        raise SystemExit(
            "--singleton_lensing sl_mixture: the lensed-injection file has no "
            "exactly-one-detected sources; regenerate the campaign or use off."
        )
    attrs = read_fc_pdet_attrs(opts.lensed_injections_path)
    rho_thr = getattr(opts, "fc_rho_thr", None) or attrs.get("fc_rho_thr")
    r0 = getattr(opts, "fc_r0", None) or attrs.get("fc_r0")
    mc_bar = getattr(opts, "fc_mc_bar", None) or attrs.get("fc_mc_bar")
    if rho_thr is None or r0 is None or mc_bar is None:
        raise SystemExit(
            "--singleton_lensing sl_mixture: missing Finn-Chernoff detection "
            "constants. The injection file lacks fc_rho_thr/fc_r0/fc_mc_bar "
            "attrs (older generator); pass --fc_rho_thr/--fc_r0/--fc_mc_bar."
        )
    fc = make_fc_pdet_params(rho_thr=rho_thr, mc_bar=mc_bar, r0=r0)
    return dict(lensed_singles=lensed_singles, fc_pdet_params=fc)


def load_inputs(opts):
    """Load singleton PE + selection, and (for j2) the lensed injections + pair
    PE + partition. Returns the assembled inputs for the cluster likelihood."""
    # BEFORE the off-mode early return below (whose bundle has no
    # marginal_partitions / log_z_prior keys), so a direct library caller gets
    # the same actionable error the CLI does instead of a KeyError from the
    # likelihood closure.  main() already rejects this via
    # _resolve_lensing_run_config; this is the second net.
    _validate_partition_mode_against_cluster_mode(opts)
    _validate_fixed_cosmology_for_lensed_channels(opts)
    _gate_pair_orientation_mismatch(opts)
    rng = np.random.default_rng(opts.seed)

    # --- singleton PE (event-major flatten) ---
    out = load_gw_samples(opts.gw_path)
    m1det, m2det, dL, chieff, ra, dec, p_pe, n_sing, nsamp = out
    m1det = np.asarray(m1det)
    m2det = np.asarray(m2det)
    dL = np.asarray(dL)
    chieff = np.asarray(chieff)
    p_pe = np.asarray(p_pe)
    observed_catalog_meta = None
    pair_time_t_obs_window_sec = None
    event_gps_times = None
    if getattr(opts, "observed_catalog_path", None):
        observed_catalog_meta = validate_observed_catalog_file(
            opts.observed_catalog_path
        )
        with open(opts.observed_catalog_path, "r", encoding="utf-8") as _f:
            _raw_catalog = json.load(_f)
        if _raw_catalog.get("observation_times") == "uniform":
            pair_time_t_obs_window_sec = float(_raw_catalog["t_obs_days"]) * 86400.0
            # Physical arrival times: the SIS arrival ORDER of a candidate pair
            # is then known, which is what lets the time mark select the image
            # assignment instead of rewarding both (see _orient_time_marks).
            event_gps_times = _event_gps_times_from_catalog(_raw_catalog)
        if int(observed_catalog_meta["n_events"]) != int(n_sing):
            raise SystemExit(
                f"observed_catalog n_events={observed_catalog_meta['n_events']} "
                f"does not match GW PE n_events={n_sing}"
            )
    else:
        observed_catalog_meta = observed_catalog_metadata_from_hdf5(opts.gw_path)
        if observed_catalog_meta is not None and int(
            observed_catalog_meta["n_events"]
        ) != int(n_sing):
            raise SystemExit(
                f"observed PE n_events={observed_catalog_meta['n_events']} "
                f"does not match GW PE n_events={n_sing}"
            )

    # --- selection ---
    sel = load_selection_samples(opts.gwselection_path)
    m1s, m2s, dLs, chis, ras, decs, pdraw, Ndraw = [np.asarray(x) for x in sel[:7]] + [
        sel[7]
    ]
    gw_sel = _gw_event(m1s, m2s, dLs, chis, pdraw)

    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF

    if cluster_mode == CLUSTER_MODE_OFF:
        gw_pe = _gw_event(m1det, m2det, dL, chieff, p_pe)
        # trivial KDE per singleton (never indexed when no pairs)
        kdes = []
        for i in range(n_sing):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            kdes.append(
                make_pair_kde(
                    m1det[sl], m2det[sl] / m1det[sl], dL[sl], chieff[sl], p_pe[sl]
                )
            )
        pair_kdes = stack_pair_kdes(kdes) if kdes else None
        return dict(
            gw_pe=gw_pe,
            gw_sel=gw_sel,
            nEvents=n_sing,
            nsamp=nsamp,
            Ndraw=float(Ndraw),
            singleton_indices=jnp.arange(n_sing, dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=n_sing,
            n_pairs=0,
            pair_kdes=pair_kdes,
            lensed=None,
            pair_time_delta_t_obs=jnp.zeros((0,), dtype=jnp.float64),
            pair_time_sigma=jnp.zeros((0,), dtype=jnp.float64),
            pair_time_t_obs_window_sec=pair_time_t_obs_window_sec,
            pair_time_signed=False,
            observed_catalog=observed_catalog_meta,
            **_load_singleton_lensing_inputs(opts),
            **_load_wl_table_arrays(opts),
        )

    # --- j2: lensed injections + pair PE/metadata + partition(s) ---
    if not opts.lensed_injections_path:
        raise SystemExit("--cluster_mode j2 requires --lensed_injections_path")
    partition_mode = getattr(opts, "partition_mode", "fixed")
    if partition_mode == "fixed" and not opts.partition_path:
        raise SystemExit("--partition_mode fixed requires --partition_path")
    if partition_mode == "marginalize_exact" and not opts.candidate_pairs_path:
        raise SystemExit(
            "--partition_mode marginalize_exact requires --candidate_pairs_path"
        )
    lensed = load_lensed_injections(opts.lensed_injections_path)
    partition = (
        json.load(open(opts.partition_path, "r", encoding="utf-8"))
        if opts.partition_path
        else None
    )
    candidate_data = None
    candidate_n_events = None
    candidate_pairs = None
    candidate_pairs_raw = None
    edge_prior_contributions = None
    pair_time_signed = False
    if opts.candidate_pairs_path:
        candidate_data = json.load(
            open(opts.candidate_pairs_path, "r", encoding="utf-8")
        )
        (
            candidate_n_events,
            candidate_pairs_raw,
            candidate_pairs,
            edge_prior_contributions,
        ) = prepare_candidate_pairs_for_partitioning(
            candidate_data, getattr(opts, "edge_mark_prior_keys", None)
        )
        _gate_suspicious_time_marks(opts, candidate_pairs_raw)
        candidate_pairs, pair_time_signed = _orient_time_marks(
            candidate_pairs, event_gps_times, opts
        )
        candidate_pairs_raw, _ = _orient_time_marks(
            candidate_pairs_raw, event_gps_times, opts
        )
    partition_pair_indices = []
    if partition is not None:
        partition_pair_indices = [
            tuple(map(int, pair)) for pair in partition.get("pair_indices", [])
        ]

    max_partition_event = -1
    if partition is not None:
        all_part_idx = list(map(int, partition.get("singleton_indices", [])))
        for a, b in partition.get("pair_indices", []):
            all_part_idx.extend([int(a), int(b)])
        max_partition_event = max(all_part_idx, default=-1)
    if max_partition_event < 0 and candidate_n_events is not None:
        max_partition_event = int(candidate_n_events) - 1
    explicit_unified_observed_catalog = observed_catalog_meta is not None
    heuristic_unified_observed_catalog = (
        max_partition_event >= 0 and n_sing == max_partition_event + 1
    )
    unified_observed_catalog = (
        explicit_unified_observed_catalog or heuristic_unified_observed_catalog
    )

    pair_metadata_path = getattr(opts, "pair_metadata_path", None)
    if (
        pair_metadata_path
        and opts.pair_pe_path
        and os.path.abspath(pair_metadata_path) != os.path.abspath(opts.pair_pe_path)
    ):
        raise SystemExit(
            "--pair_metadata_path and --pair_pe_path both provided but point to different paths"
        )
    pair_file_path = pair_metadata_path or opts.pair_pe_path

    if not unified_observed_catalog and not opts.pair_pe_path:
        raise SystemExit(
            "legacy split-pair --cluster_mode j2 requires --pair_pe_path with image0/image1 groups"
        )

    pairs = []
    pair_metadata_indices = []
    pair_time_delta_t_obs = []
    pair_time_sigma = []
    # Whether every mark taken from the pair FILE could be given an arrival
    # order. Starts True and is cleared by the first mark we cannot orient.
    pair_file_time_signed = True
    if pair_file_path:
        with h5py.File(pair_file_path) as f:
            npairs = int(f.attrs.get("npairs", 0))
            for k in range(npairs):
                g = f[f"pair_{k}"]
                imgs = []
                meta_pair = None
                if "event_index_image0" in g.attrs or "event_index_image1" in g.attrs:
                    if not (
                        "event_index_image0" in g.attrs
                        and "event_index_image1" in g.attrs
                    ):
                        raise SystemExit(
                            f"pair_{k} must define both event_index_image0 and event_index_image1"
                        )
                    meta_pair = (
                        int(g.attrs["event_index_image0"]),
                        int(g.attrs["event_index_image1"]),
                    )
                    if (
                        meta_pair[0] < 0
                        or meta_pair[1] < 0
                        or meta_pair[0] == meta_pair[1]
                    ):
                        raise SystemExit(
                            f"pair_{k} has invalid event-index metadata: {meta_pair}"
                        )
                    if (
                        partition_pair_indices
                        and k < len(partition_pair_indices)
                        and meta_pair != partition_pair_indices[k]
                    ):
                        raise SystemExit(
                            f"pair_{k} event-index metadata {meta_pair} does not match partition pair "
                            f"{partition_pair_indices[k]}"
                        )
                has_dt = "delta_t_obs" in g.attrs or "delta_t_obs" in g
                use_pair_pe_time = has_dt or (
                    getattr(opts, "pair_marks", "none") == "time"
                    and partition_mode == "fixed"
                )
                if use_pair_pe_time:
                    if not has_dt:
                        raise SystemExit(
                            f"pair_marks=time requires delta_t_obs metadata for pair_{k}"
                        )
                    # The marks below are appended in pair-FILE order and the
                    # master likelihood reads them POSITIONALLY against the
                    # partition's pair rows, so without the event indices
                    # nothing checks that pair_k describes partition pair k:
                    # a differently-ordered metadata file hands every pair
                    # another pair's |dt| (wrong y* = |dt|/T0, wrong
                    # coincidence-odds denominator) and only the length is
                    # verified (review F-014). Every in-repo writer emits the
                    # attrs; require them.
                    if (
                        meta_pair is None
                        and getattr(opts, "pair_marks", "none") == "time"
                        and partition_mode == "fixed"
                        and partition_pair_indices
                    ):
                        raise SystemExit(
                            f"pair_{k} carries a time mark but no "
                            "event_index_image0/event_index_image1: with "
                            "--pair_marks time --partition_mode fixed the marks "
                            "are consumed in partition pair order, so they "
                            "cannot be aligned to the partition's pairs"
                        )
                    _dt_raw = float(
                        g.attrs["delta_t_obs"]
                        if "delta_t_obs" in g.attrs
                        else np.asarray(g["delta_t_obs"])[()]
                    )
                    # The pair file's delta_t_obs is written as
                    # t(image1) - t(image0) with (image0, image1) =
                    # (event_index_image0, event_index_image1), i.e. already
                    # signed in the stored pair order. Only orient it when the
                    # arrival times back that up; otherwise keep the magnitude
                    # so the legacy both-branch behaviour is preserved.
                    _dt_signed = None
                    if meta_pair is not None and event_gps_times is not None:
                        _n_ev = int(event_gps_times.shape[0])
                        if 0 <= meta_pair[0] < _n_ev and 0 <= meta_pair[1] < _n_ev:
                            _sign = (
                                1.0
                                if (event_gps_times[meta_pair[1]]
                                    - event_gps_times[meta_pair[0]]) >= 0.0
                                else -1.0
                            )
                            _dt_signed = _sign * abs(_dt_raw)
                    if _dt_signed is None:
                        pair_file_time_signed = False
                        pair_time_delta_t_obs.append(abs(_dt_raw))
                    else:
                        pair_time_delta_t_obs.append(_dt_signed)
                    if "sigma_delta_t" in g.attrs:
                        pair_time_sigma.append(float(g.attrs["sigma_delta_t"]))
                    elif "sigma_delta_t" in g:
                        pair_time_sigma.append(
                            float(np.asarray(g["sigma_delta_t"])[()])
                        )
                    elif opts.pair_time_sigma_sec is not None:
                        pair_time_sigma.append(float(opts.pair_time_sigma_sec))
                    elif getattr(opts, "pair_marks", "none") == "time":
                        raise SystemExit(
                            f"pair_marks=time requires sigma_delta_t in pair_{k} or --pair_time_sigma_sec"
                        )
                    else:
                        # delta_t_obs present but no sigma anywhere: the run does
                        # not use time marks, so drop this pair's mark instead of
                        # aborting over metadata the likelihood never reads.
                        pair_time_delta_t_obs.pop()
                if not unified_observed_catalog:
                    for name in ("image0", "image1"):
                        if name not in g:
                            raise SystemExit(
                                f"legacy pair_pe_path missing pair_{k}/{name} group"
                            )
                        gi = g[name]
                        d = dict(
                            m1det=np.array(gi["m1det"]),
                            q=np.array(gi["q"]),
                            dL_app=np.array(gi["dL_app"]),
                            chieff=np.array(gi["chieff"]),
                            prior_wt=np.array(gi["prior_wt"]),
                        )
                        # Normalize over the FULL sample set, THEN downsample
                        # (deprecated legacy split-pair path; unreachable via the
                        # unified catalog). Renormalizing AFTER downsampling would
                        # divide by the subset sum and suppress each pairing by
                        # ~log(nsamp/pe_max), the P1.5 bias the unified path avoids
                        # (extract_event_samples_from_gw_pe: "do NOT renormalise
                        # after downsampling").
                        d["prior_wt"] = _normalize_pair_image_prior_wt(
                            d["prior_wt"],
                            context=f"{pair_file_path}: pair_{k}/{name}/prior_wt",
                        )
                        if opts.pe_max_per_pair > 0:
                            d = _downsample(d, opts.pe_max_per_pair, rng)
                        imgs.append(d)
                    pairs.append(imgs)
                pair_metadata_indices.append(meta_pair)
    if pair_time_delta_t_obs:
        # Marks came from the pair FILE; its orientation flag wins over the
        # candidate-pairs one (which describes a source we did not use).
        pair_time_signed = bool(pair_file_time_signed)
    P = (
        len(pairs)
        if not unified_observed_catalog
        else (
            int(partition.get("n_pairs", len(partition_pair_indices)))
            if partition is not None
            else 0
        )
    )

    if (
        unified_observed_catalog
        and partition_mode == "fixed"
        and getattr(opts, "pair_marks", "none") == "time"
        and not pair_time_delta_t_obs
        and candidate_pairs is not None
        and partition_pair_indices
    ):
        mark_by_edge = {(p.i, p.j): p for p in candidate_pairs}
        for pair in partition_pair_indices:
            edge = tuple(sorted(pair))
            meta = mark_by_edge.get(edge)
            if meta is None or meta.delta_t_obs is None or meta.sigma_delta_t is None:
                raise SystemExit(
                    f"pair_marks=time requires candidate_pairs.json marks or pair metadata for fixed pair {pair}"
                )
            # The mark's sign convention is "t_j - t_i for the mark's own
            # (i, j)". Re-orient it to the order this partition stores the pair
            # in, so the arrival-order branch selection in
            # cluster_log_likelihood_pair refers to the right event.
            dt = float(meta.delta_t_obs)
            if pair_time_signed and (int(pair[0]), int(pair[1])) != (meta.i, meta.j):
                dt = -dt
            sig = float(meta.sigma_delta_t)
            if not np.isfinite(dt) or not np.isfinite(sig) or sig <= 0:
                raise SystemExit(
                    f"fixed pair {pair} has invalid candidate_pairs.json "
                    "marks.delta_t_obs/sigma_delta_t required by pair_marks=time"
                )
            pair_time_delta_t_obs.append(dt)
            pair_time_sigma.append(sig)

    if (
        unified_observed_catalog
        and partition is not None
        and getattr(opts, "pair_marks", "none") == "time"
        and len(pair_time_delta_t_obs) not in (0, int(partition.get("n_pairs", 0)))
    ):
        raise SystemExit(
            "pair time metadata count does not match fixed partition n_pairs"
        )

    # Build the full event catalog: [singletons 0..S-1, then pair images].
    # NOTE the singleton PE arrays are already nsamp-per-event; pair images use
    # (possibly) a different per-image sample count after down-sampling. The
    # branch's GWEvent stores ragged events via the flat (nEvents*nsamp) layout
    # ONLY when all events share nsamp. To keep shapes uniform we down-sample the
    # singleton PE to the SAME pe count as the pairs when they differ.
    pe_per_pair = (
        len(pairs[0][0]["m1det"]) if (P and not unified_observed_catalog) else nsamp
    )
    if (not unified_observed_catalog) and pe_per_pair != nsamp:
        # downsample singleton PE to pe_per_pair for a uniform array
        new_m1, new_m2, new_dL, new_chi, new_pw = [], [], [], [], []
        for i in range(n_sing):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            d = _downsample(
                dict(
                    m1=m1det[sl], m2=m2det[sl], dL=dL[sl], chi=chieff[sl], pw=p_pe[sl]
                ),
                pe_per_pair,
                rng,
            )
            new_m1.append(d["m1"])
            new_m2.append(d["m2"])
            new_dL.append(d["dL"])
            new_chi.append(d["chi"])
            new_pw.append(d["pw"])
        m1det = np.concatenate(new_m1)
        m2det = np.concatenate(new_m2)
        dL = np.concatenate(new_dL)
        chieff = np.concatenate(new_chi)
        p_pe = np.concatenate(new_pw)
        nsamp = pe_per_pair

    if unified_observed_catalog:
        gw_pe = _gw_event(m1det, m2det, dL, chieff, p_pe)
        n_events_total = n_sing
        kdes = make_pair_kdes_from_gw_pe(
            (m1det, m2det, dL, chieff, p_pe),
            range(n_events_total),
            nsamp,
            opts.pe_max_per_pair,
            rng,
        )
    else:
        m1_all = [m1det]
        m2_all = [m2det]
        dL_all = [dL]
        chi_all = [chieff]
        pw_all = [p_pe]
        for imgs in pairs:
            for img in imgs:
                m1_all.append(img["m1det"])
                m2_all.append(img["q"] * img["m1det"])
                dL_all.append(img["dL_app"])
                chi_all.append(img["chieff"])
                pw_all.append(img["prior_wt"])
        gw_pe = _gw_event(
            np.concatenate(m1_all),
            np.concatenate(m2_all),
            np.concatenate(dL_all),
            np.concatenate(chi_all),
            np.concatenate(pw_all),
        )
        kdes = []
        for i in range(n_sing):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            kdes.append(
                make_pair_kde(
                    m1det[sl], m2det[sl] / m1det[sl], dL[sl], chieff[sl], p_pe[sl]
                )
            )
        for imgs in pairs:
            for img in imgs:
                kdes.append(
                    make_pair_kde(
                        img["m1det"],
                        img["q"],
                        img["dL_app"],
                        img["chieff"],
                        img["prior_wt"],
                    )
                )
        n_events_total = n_sing + 2 * P
    pair_kdes = stack_pair_kdes(kdes)
    if partition_mode == "marginalize_exact":
        if candidate_n_events != n_events_total:
            raise SystemExit(
                f"candidate_pairs n_events={candidate_n_events} does not match loaded event count "
                f"{n_events_total}"
            )
        partition_component_mode = getattr(opts, "partition_component_mode", "componentwise")
        pair_marks_mode = getattr(opts, "pair_marks", "none")
        max_exact_partitions = int(getattr(opts, "max_exact_partitions", 10000))
        max_component_partitions = (
            getattr(opts, "max_component_partitions", None) or max_exact_partitions
        )
        max_total_partitions = getattr(opts, "max_total_partitions", None)
        global_row_cap = int(max_total_partitions or max_exact_partitions)
        factorized_exact = partition_component_mode == "componentwise"
        component_partition_summaries = None
        component_partition_states = None
        component_marginal_partitions = None
        component_full_partition_states = None
        component_full_partitions = None
        baseline_partition = None
        selection_probe_partitions = None
        approximate_total_partitions = None

        if partition_component_mode == "global":
            _candidate_n_events2, partition_states, log_z_prior = exact_partitions_from_pairs(
                candidate_n_events,
                candidate_pairs,
                max_partitions=max_exact_partitions,
                component_mode="global",
            )
            marginal_partitions = tuple(
                _runtime_part_from_state(state, candidate_pairs, pair_marks=pair_marks_mode)
                for state in partition_states
            )
            fixed_singletons = marginal_partitions[0]["singleton_indices"]
            fixed_pairs = marginal_partitions[0]["pair_indices"]
            fixed_n_singletons = marginal_partitions[0]["n_singletons"]
            fixed_n_pairs = marginal_partitions[0]["n_pairs"]
            approximate_total_partitions = len(partition_states)
        else:
            component_partition_summaries, component_partition_states, approximate_total_partitions = exact_partition_components(
                candidate_n_events,
                candidate_pairs,
                max_component_events=getattr(opts, "max_component_events", None),
                max_component_edges=getattr(opts, "max_component_edges", None),
                max_component_partitions=int(max_component_partitions),
            )
            log_z_prior = float(
                sum(
                    logsumexp([state.log_prior_weight for state in states])
                    for states in component_partition_states
                )
            )
            component_full_partition_states = tuple(
                tuple(
                    _full_state_for_component(candidate_n_events, summary, state)
                    for state in states
                )
                for summary, states in zip(component_partition_summaries, component_partition_states)
            )
            component_full_partitions = tuple(
                tuple(
                    _runtime_part_from_state(state, candidate_pairs, pair_marks=pair_marks_mode)
                    for state in states
                )
                for states in component_full_partition_states
            )
            component_marginal_partitions = tuple(
                tuple(
                    _runtime_part_from_state(state, candidate_pairs, pair_marks=pair_marks_mode)
                    for state in states
                )
                for states in component_partition_states
            )
            baseline_state = _all_singleton_partition_state(candidate_n_events)
            baseline_partition = _runtime_part_from_state(
                baseline_state, candidate_pairs, pair_marks=pair_marks_mode
            )
            max_factorized_pairs = sum(
                max(int(state.n_pairs) for state in states)
                for states in component_partition_states
            )
            # Count carriers for the closed-form selection deltas: ONLY
            # ``(n_singletons, n_pairs)`` is read off these, by both the sampler
            # and the diagnostics path (_count_correction_closed_form). The
            # fabricated (2k, 2k+1) pairing and its placeholder time marks are
            # never evaluated by any likelihood -- they were, before review
            # F-005, and their fictitious pairs' pair_variance_sum leaked into
            # the guard threshold.
            selection_probe_partitions = []
            for n_pairs_probe in range(max_factorized_pairs + 1):
                probe_state = _count_probe_partition_state(candidate_n_events, n_pairs_probe)
                time_override = None
                if pair_marks_mode == "time":
                    time_override = (
                        jnp.zeros((n_pairs_probe,), dtype=jnp.float64),
                        jnp.ones((n_pairs_probe,), dtype=jnp.float64),
                    )
                selection_probe_partitions.append(
                    _runtime_part_from_state(
                        probe_state,
                        candidate_pairs,
                        pair_marks=pair_marks_mode,
                        pair_time_override=time_override,
                    )
                )
            selection_probe_partitions = tuple(selection_probe_partitions)
            if approximate_total_partitions <= global_row_cap:
                partition_states = combine_component_partitions(component_partition_states)
                marginal_partitions = tuple(
                    _runtime_part_from_state(state, candidate_pairs, pair_marks=pair_marks_mode)
                    for state in partition_states
                )
            else:
                partition_states = None
                marginal_partitions = None
            fixed_singletons = baseline_partition["singleton_indices"]
            fixed_pairs = baseline_partition["pair_indices"]
            fixed_n_singletons = baseline_partition["n_singletons"]
            fixed_n_pairs = baseline_partition["n_pairs"]
    else:
        partition_component_mode = None
        factorized_exact = False
        component_partition_summaries = None
        component_partition_states = None
        component_marginal_partitions = None
        component_full_partition_states = None
        component_full_partitions = None
        baseline_partition = None
        selection_probe_partitions = None
        approximate_total_partitions = None
        marginal_partitions = None
        partition_states = None
        log_z_prior = 0.0
        # A fixed partition must partition the WHOLE observed catalog: the
        # likelihood only touches the listed indices, so an uncovered event is
        # silently dropped from the product while the run still reports the PE
        # file's event count. Preflight checks this too; repeat it here so a
        # --preflight_only false run (or a direct library caller) cannot slip past.
        _coverage_error = fixed_partition_coverage_error(
            list(map(int, partition["singleton_indices"]))
            + [int(i) for i in np.asarray(partition["pair_indices"], dtype=int).reshape(-1)],
            n_events_total,
            path=opts.partition_path,
        )
        if _coverage_error is not None:
            raise SystemExit(_coverage_error)
        fixed_singletons = jnp.asarray(partition["singleton_indices"], dtype=jnp.int32)
        fixed_pairs = jnp.asarray(partition["pair_indices"], dtype=jnp.int32)
        fixed_n_singletons = int(partition["n_singletons"])
        fixed_n_pairs = int(partition["n_pairs"])
    return dict(
        gw_pe=gw_pe,
        gw_sel=gw_sel,
        nEvents=n_events_total,
        nsamp=nsamp,
        Ndraw=float(Ndraw),
        singleton_indices=fixed_singletons,
        pair_indices=fixed_pairs,
        n_singletons=fixed_n_singletons,
        n_pairs=fixed_n_pairs,
        pair_kdes=pair_kdes,
        lensed=lensed,
        marginal_partitions=marginal_partitions,
        log_z_prior=float(log_z_prior),
        pair_time_delta_t_obs=jnp.asarray(pair_time_delta_t_obs, dtype=jnp.float64),
        pair_time_sigma=jnp.asarray(pair_time_sigma, dtype=jnp.float64),
        partition_states=(
            partition_states if partition_mode == "marginalize_exact" else None
        ),
        partition_component_mode=partition_component_mode,
        factorized_exact=bool(factorized_exact),
        global_partitions_enumerated=bool(partition_states is not None),
        approximate_total_partitions=approximate_total_partitions,
        component_partition_summaries=component_partition_summaries,
        component_partition_states=component_partition_states,
        component_marginal_partitions=component_marginal_partitions,
        component_full_partition_states=component_full_partition_states,
        component_full_partitions=component_full_partitions,
        baseline_partition=baseline_partition,
        selection_probe_partitions=selection_probe_partitions,
        candidate_pairs=(
            candidate_pairs if partition_mode == "marginalize_exact" else None
        ),
        candidate_pairs_raw=(
            candidate_pairs_raw if partition_mode == "marginalize_exact" else None
        ),
        edge_mark_prior_contributions=(
            edge_prior_contributions if partition_mode == "marginalize_exact" else None
        ),
        edge_mark_prior_keys=list(parse_edge_mark_keys(getattr(opts, "edge_mark_prior_keys", None))),
        pair_time_t_obs_window_sec=pair_time_t_obs_window_sec,
        pair_time_signed=bool(
            pair_time_signed and getattr(opts, "pair_marks", "none") == "time"
        ),
        observed_catalog=observed_catalog_meta,
        observed_catalog_heuristic=bool(
            heuristic_unified_observed_catalog and not explicit_unified_observed_catalog
        ),
        **_load_singleton_lensing_inputs(opts),
        **_load_wl_table_arrays(opts),
    )


# =============================================================================
# Likelihood closure
# =============================================================================
def _diagnostics_to_python(diag):
    out = {}
    for key, value in diag.items():
        arr = np.asarray(value)
        if arr.shape == ():
            if np.issubdtype(arr.dtype, np.integer):
                out[key] = int(arr)
            else:
                out[key] = float(arr)
        else:
            out[key] = arr.astype(float).tolist()
    return out


def _is_finite_number(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _add_off_control_nonfinite_diagnostics(diagnostics, *, opts, inp):
    """Attach source-oriented diagnostics when singleton/off control is nonfinite."""
    if getattr(opts, "cluster_mode", None) != "off":
        return diagnostics
    keys = (
        "logL_total",
        "singleton_logL_sum",
        "selection_correction_total",
        "Neff_singleton",
        "log_mu_singleton",
        "log_sigma2_singleton",
    )
    flags = {key: (key in diagnostics and not _is_finite_number(diagnostics.get(key))) for key in keys}
    diagnostics["off_control_nonfinite_component_flags"] = flags
    diagnostics["off_control_has_nonfinite_component"] = any(flags.values())
    diagnostics["n_events"] = int(inp["nEvents"])
    diagnostics["nsamp"] = int(inp["nsamp"])
    diagnostics["selection_file_summary"] = {
        "path": str(opts.gwselection_path),
        "n_draw": float(inp["Ndraw"]),
        "n_samples": int(np.asarray(inp["gw_sel"].m1det).size),
        "sel_batch_size": opts.sel_batch_size,
        "wl_selection": opts.wl_selection,
    }
    return diagnostics


def _write_diagnostics(run_dir, diagnostics):
    with open(os.path.join(run_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, allow_nan=True)
    numeric_array_keys = {
        "partition_log_prior_weight",
        "partition_logL",
        "partition_log_posterior_weight",
        "partition_posterior_probability",
        "posterior_singleton_probability",
    }
    with h5py.File(os.path.join(run_dir, "diagnostics.hdf5"), "w") as f:
        for key, value in diagnostics.items():
            if key in numeric_array_keys:
                f.create_dataset(key, data=np.asarray(value, dtype=float))
            elif key == "posterior_pair_probabilities":
                f.create_dataset(
                    "posterior_pair_probability_i",
                    data=np.asarray([x["i"] for x in value], dtype=np.int32),
                )
                f.create_dataset(
                    "posterior_pair_probability_j",
                    data=np.asarray([x["j"] for x in value], dtype=np.int32),
                )
                f.create_dataset(
                    "posterior_pair_probability",
                    data=np.asarray([x["p_pair"] for x in value], dtype=float),
                )
                f.attrs[key] = json.dumps(value)
            elif isinstance(value, (dict, list, tuple)):
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.dtype.kind in "fiu" and arr.ndim > 0:
                        f.create_dataset(key, data=arr)
                    else:
                        f.attrs[key] = json.dumps(value)
                except (TypeError, ValueError):
                    f.attrs[key] = json.dumps(value)
            elif isinstance(value, str):
                f.attrs[key] = value
            elif value is None:
                # h5py cannot store None (object dtype); keep the key visible.
                f.attrs[key] = "null"
            else:
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = json.dumps(value, default=str)


def _run_name_prefix(opts):
    """Run-directory name minus the timestamp; also the --resume auto filter."""
    return (
        f"{opts.pop_model}__{opts.cluster_mode}__{opts.sampler}"
        f"__seed{getattr(opts, 'seed', 'NA')}__"
    )


def _make_run_dir(opts):
    """Create opts.save_path/<run_name>, disambiguating name collisions.

    The name embeds pop_model/cluster_mode/sampler/seed/timestamp (second
    resolution).  Without the seed, and under exist_ok=True, two jobs of the
    same configuration launched in the same second -- routine in a SLURM array
    that varies only --seed -- shared a directory and silently overwrote each
    other's results.hdf5 / samples.npy / settings.json.  Retry with a
    monotonically increasing numeric suffix instead, matching
    cli/inference.py::_make_run_dir, which was already fixed for exactly this.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{_run_name_prefix(opts)}{ts}"
    run_dir = os.path.join(opts.save_path, run_name)
    suffix = 0
    while True:
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = os.path.join(opts.save_path, f"{run_name}-{suffix:02d}")


def _jsonable_settings(opts):
    settings = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in vars(opts).items()
    }
    # Same code-identity + numerics block the main CLI writes (io/settings.py),
    # so a lensing run is as attributable as a dark-siren one.  This file used
    # to persist no `environment` key at all.
    settings["environment"] = environment_block()
    return settings


def _write_json(path, payload, *, allow_nan=True):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=allow_nan)
        f.write("\n")


def _write_failure(run_dir, stage, exc, *, labels=None, settings=None):
    payload = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "labels": list(labels or []),
        "settings": settings or {},
        "command": [sys.executable, "-m", "darksirens.cli.inference_lensing", *sys.argv[1:]],
    }
    _write_json(os.path.join(run_dir, "failure.json"), payload, allow_nan=True)


def _partition_diagnostics_key_prefix(eval_point_label):
    """Attr/settings prefix for the partition diagnostics' eval point.

    ``prior_midpoint_*`` is reserved for numbers actually evaluated at the
    midpoint; anything the guard-clear fallback picked instead gets the generic
    ``diagnostics_point_*`` prefix already used for the lens block
    (``_lens_settings_dict(..., eval_point="diagnostics_point")``).
    """
    return (
        "prior_midpoint"
        if str(eval_point_label or "prior_midpoint") == "prior_midpoint"
        else "diagnostics_point"
    )


def _write_result_partition_metadata(
    attrs, *, opts, inp, diagnostics,
    eval_point_label="prior_midpoint", eval_point=None,
):
    """Write unambiguous partition-count metadata to ``results.hdf5`` attrs."""
    partition_mode = str(getattr(opts, "partition_mode", "fixed"))
    attrs["partition_mode"] = partition_mode
    attrs["cluster_mode"] = opts.cluster_mode
    attrs["wl_backend"] = opts.wl_backend
    attrs["wl_table_path"] = str(getattr(opts, "lensing_wl_table_path", None) or "")
    attrs["wl_selection"] = opts.wl_selection
    attrs["singleton_lensing"] = getattr(opts, "singleton_lensing", "off")
    attrs["pair_orientation_mode"] = getattr(
        opts, "pair_orientation_mode", "independent"
    )
    attrs["pair_time_mark_impl"] = getattr(opts, "pair_time_mark_impl", "auto")
    attrs["n_events"] = int(inp["nEvents"])
    attrs["reference_partition_n_singletons"] = int(inp["n_singletons"])
    attrs["reference_partition_n_pairs"] = int(inp["n_pairs"])
    if partition_mode == "marginalize_exact":
        # Backward-compatible legacy fields.  They intentionally remain the
        # enumerator/reference partition counts rather than posterior means.
        attrs["n_singletons"] = int(inp["n_singletons"])
        attrs["n_pairs"] = int(inp["n_pairs"])
        attrs["n_singletons_meaning"] = "reference_partition_n_singletons"
        attrs["n_pairs_meaning"] = "reference_partition_n_pairs"
        attrs["n_partitions"] = int(diagnostics["n_partitions"])
        attrs["partition_component_mode"] = str(diagnostics.get("partition_component_mode", getattr(opts, "partition_component_mode", "global")))
        attrs["factorized_exact"] = bool(diagnostics.get("factorized_exact", False))
        attrs["global_partitions_enumerated"] = bool(diagnostics.get("global_partitions_enumerated", True))
        if diagnostics.get("approximate_total_partitions") is not None:
            attrs["approximate_total_partitions"] = int(diagnostics["approximate_total_partitions"])
        # These are evaluated at ONE parameter point, not at the posterior (the
        # stdout print says so; the file attrs previously did not — library
        # review, lensing CLI finding 6): label them explicitly so nobody traces
        # pre-posterior numbers into a paper as run results.  The point is the
        # prior midpoint only when the reliability guard cleared there; the
        # guard-clear fallback (registry fiducial, then seeded prior draws) is
        # the documented common case on paper-scale joint runs, and the attrs
        # used to claim prior_midpoint regardless (review F-006).
        prefix = _partition_diagnostics_key_prefix(eval_point_label)
        attrs["partition_diagnostics_eval_point"] = str(
            eval_point_label or "prior_midpoint"
        )
        if eval_point is not None:
            attrs["partition_diagnostics_eval_point_values"] = json.dumps(
                np.asarray(eval_point, dtype=float).tolist()
            )
        attrs[f"{prefix}_expected_n_singletons"] = float(diagnostics["expected_n_singletons"])
        attrs[f"{prefix}_expected_n_pairs"] = float(diagnostics["expected_n_pairs"])
        attrs[f"{prefix}_map_partition_index"] = int(diagnostics["map_partition_index"])
        map_partition = diagnostics["map_partition"]
        attrs[f"{prefix}_map_partition_n_singletons"] = int(map_partition["n_singletons"])
        attrs[f"{prefix}_map_partition_n_pairs"] = int(map_partition["n_pairs"])
        attrs[f"{prefix}_logL_marginalized"] = float(diagnostics["logL_marginalized"])
        attrs["log_z_partition_prior"] = float(diagnostics["log_z_partition_prior"])
        attrs["edge_mark_prior_keys"] = json.dumps(inp.get("edge_mark_prior_keys", []))
        attrs["edge_prior_semantics"] = "effective_log_prior_odds = raw_log_prior_odds + sum(requested log_* marks)"
        attrs["edge_prior_applied_once"] = True
    else:
        attrs["n_singletons"] = int(inp["n_singletons"])
        attrs["n_pairs"] = int(inp["n_pairs"])
    observed_catalog = inp.get("observed_catalog")
    if observed_catalog:
        if observed_catalog.get("path"):
            attrs["observed_catalog_path"] = str(observed_catalog["path"])
        attrs["observed_catalog_format_version"] = str(
            observed_catalog.get("format_version")
            or observed_catalog.get("pe_format_version", "")
        )


def _fiducial_candidate_point(labels, mid, lower, upper, opts, pop_params_fid):
    """Sampled-space point at the registry fiducial (mock-generator truth).

    The selection injection campaigns are sized at the fiducial population,
    so the reliability guard is clear there by construction while random
    prior draws concentrate the injection weights and collapse Neff. Labels
    without a fiducial (cosmology/survey blocks) fall back to the midpoint;
    the result is clipped strictly inside the prior box.
    """
    from darksirens.inference.prior import pop_model_prior_parser

    _, _, pop_labels_all, _, _ = pop_model_prior_parser(opts.pop_model)
    fid = {
        lab: float(v)
        for lab, v in zip(pop_labels_all, np.asarray(pop_params_fid, dtype=float))
    }
    try:
        fid["log10_tau_A"] = float(np.log10(float(opts.sl_tau_A)))
    except (TypeError, ValueError):
        pass
    try:
        fid["tau_n"] = float(opts.sl_tau_n)
    except (TypeError, ValueError):
        pass
    mid = np.asarray(mid, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    point = np.array([fid.get(lab, m) for lab, m in zip(labels, mid)], dtype=float)
    eps = 1e-6 * (upper - lower)
    return np.clip(point, lower + eps, upper - eps)


def _diagnostics_at_guard_clear_point(
    diagnostics_fn, mid, lower, upper, seed, n_draws=32, fiducial=None
):
    """Evaluate the factorization diagnostics at a point the selection
    reliability guard does not clip to -inf.

    The prior midpoint is tried first; on paper-scale joint runs the guard
    often fires there (extreme hyperparameters -> tiny selection Neff), which
    makes the exactness cross-check impossible at that point even though the
    sampler itself is fine (it concentrates where the guard is clear). Fall
    back to seeded prior draws; if none of them clears the guard the sampler
    would not find live points either, so re-raise the last error.
    Returns (diagnostics, evaluation_point, evaluation_point_label) -- the label
    is what the archived attrs/settings are keyed on, so numbers computed at the
    fallback point cannot be filed as prior_midpoint_* (review F-006)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    points = [("prior midpoint", np.asarray(mid, dtype=float))]
    if fiducial is not None:
        points.append(("registry fiducial", np.asarray(fiducial, dtype=float)))
    points += [
        (f"prior draw {k + 1}", lower + (upper - lower) * rng.random(lower.shape))
        for k in range(n_draws)
    ]
    last_exc = None
    for k, (name, point) in enumerate(points):
        try:
            diagnostics = diagnostics_fn(jnp.asarray(point))
            if k > 0:
                _warn(
                    f"diagnostics point: {name} "
                    "(reliability guard fired at the prior midpoint)"
                )
            return diagnostics, point, name.replace(" ", "_")
        except RuntimeError as exc:
            if "NON-FINITE" not in str(exc):
                raise
            last_exc = exc
    raise RuntimeError(
        f"factorization diagnostics: no guard-clear evaluation point found in "
        f"the prior midpoint + {n_draws} prior draws; the selection reliability "
        "guard rejects essentially all of the prior. The sampler would not "
        "initialize either — increase the injection campaign or relax "
        "--max_likelihood_variance."
    ) from last_exc


def _print_diagnostics_summary(diagnostics):
    _section("Likelihood diagnostics")
    for key in (
        "logL_total",
        "selection_correction_total",
        "singleton_logL_sum",
        "pair_logL_sum",
        "log_mu_singleton",
        "Neff_singleton",
        "log_mu_cluster",
        "Neff_cluster",
        "Neff_combined",
    ):
        if key in diagnostics:
            _row(key, diagnostics[key])
    print("  │")
    if diagnostics.get("partition_mode") == "marginalize_exact":
        _row("expected_n_singletons", diagnostics.get("expected_n_singletons"))
        _row("expected_n_pairs",      diagnostics.get("expected_n_pairs"))
        _row("map_n_pairs",           diagnostics.get("map_partition", {}).get("n_pairs"))
        _row("n_partitions",          diagnostics.get("n_partitions"))
    else:
        _row("n_singletons", diagnostics.get("n_singletons"))
        _row("n_pairs",      diagnostics.get("n_pairs"))
    _row("pair_batch_size", diagnostics.get("pair_batch_size"))
    _row("y_nodes_pair",    diagnostics.get("y_nodes_pair"))
    _row("pe_max_per_pair", diagnostics.get("pe_max_per_pair"))
    _row("pair_eval_shape", diagnostics.get("approximate_pair_evaluation_shape"))
    _row("cluster_mode",    diagnostics["cluster_mode"])
    _row("wl_backend",      diagnostics["wl_backend"])
    _end()


# max(sigma_dt)/T0 below this -> delta collapse (shared with the SIS-support
# diagnostic, which resolves the same rule).
_TIME_DELTA_SHARPNESS = SIS_TIME_MARK_SHARPNESS


def _resolve_pair_marks(opts, inp):
    """Map the user-facing --pair_marks to the static implementation code.

    For pair_marks=time, sharp marks (max sigma_dt / T0 below the sharpness
    threshold) pin y more tightly than any practical quadrature resolves —
    the G-pilot failure mode — so the y-integral is delta-collapsed
    analytically (PAIR_MARKS_TIME_DELTA). --pair_time_mark_impl overrides.
    """
    if getattr(opts, "pair_marks", "none") != "time":
        return PAIR_MARKS_NONE
    impl = getattr(opts, "pair_time_mark_impl", "auto")
    if impl == "quadrature":
        return PAIR_MARKS_TIME
    if impl == "delta":
        return PAIR_MARKS_TIME_DELTA
    sigmas = np.asarray(inp.get("pair_time_sigma", []), dtype=float)
    if sigmas.size == 0:
        # --partition_mode marginalize_exact keeps the per-edge widths in
        # candidate_pairs and only materialises them PER PARTITION inside
        # _runtime_part_from_state, so inp["pair_time_sigma"] is empty and the
        # auto rule silently fell through to the quadrature implementation --
        # the one this function exists to avoid for sharp marks.  Read the
        # authoritative widths straight off the candidate pairs instead.
        sigmas = np.asarray(
            [
                float(p.sigma_delta_t)
                for p in (inp.get("candidate_pairs") or [])
                if getattr(p, "sigma_delta_t", None) is not None
            ],
            dtype=float,
        )
    if sigmas.size == 0:
        return PAIR_MARKS_TIME
    # Same auto rule the SIS-support diagnostic resolves the mark with.
    return (
        PAIR_MARKS_TIME_DELTA
        if resolve_sis_time_mark_impl(impl, sigmas, _sl_T0_seconds(opts)) == "delta"
        else PAIR_MARKS_TIME
    )


def _require_time_window(opts, inp):
    """Time marks need the observing-run length for the coincidence odds."""
    if (
        getattr(opts, "pair_marks", "none") == "time"
        and inp.get("pair_time_t_obs_window_sec") is None
    ):
        raise SystemExit(
            "--pair_marks time requires an observed catalog with "
            "observation_times='uniform' (t_obs_days): the time-mark "
            "coincidence odds need the observing-run length. Regenerate the "
            "mock with --observation-times uniform."
        )


def _require_time_marks_in_sis_support(opts, inp):
    """Refuse to build a likelihood whose every time-marked pair is -inf.

    ``y* = |dt| / T0`` must land inside the SIS support (0, 1), but the mark's
    own width decides whether an out-of-support delay is actually annihilated:
    the delta-collapse mark masks every node only once
    ``y* - u_max sigma_dt/T0 >= 1``, and the quadrature mark never masks at all.
    Only the annihilated case is fatal (fixed-partition runs cannot initialise
    live points, and ``marginalize_exact`` runs report "no lensing" with
    certainty because the true pairing carries zero weight at every parameter
    value); the rest is a warning. Preflight raises the same error; this guard
    also covers runs that skip preflight.
    """
    if getattr(opts, "pair_marks", "none") != "time":
        return
    dt_values = list(np.asarray(inp.get("pair_time_delta_t_obs", []), dtype=float).ravel())
    sigma_values = list(np.asarray(inp.get("pair_time_sigma", []), dtype=float).ravel())
    if not dt_values:
        marked = [
            p
            for p in (inp.get("candidate_pairs") or [])
            if getattr(p, "delta_t_obs", None) is not None
        ]
        dt_values = [p.delta_t_obs for p in marked]
        sigma_values = [getattr(p, "sigma_delta_t", None) for p in marked]
    T0 = _sl_T0_seconds(opts)
    # Per-delay widths when they line up, otherwise the widest known one (the
    # lenient direction: a false fatal is worse than a missed one here).
    sigmas = (
        sigma_values
        if len(sigma_values) == len(dt_values)
        else (max((s for s in sigma_values if s is not None), default=None))
    )
    support = sis_time_mark_support(
        dt_values,
        T0,
        sigma_delta_t_seconds=sigmas,
        mark_impl=resolve_sis_time_mark_impl(
            getattr(opts, "pair_time_mark_impl", "auto"), sigma_values, T0
        ),
    )
    if support["n_marked"] and support["all_annihilated"]:
        raise SystemExit(sis_time_mark_support_message(support))
    if support["n_out_of_support"]:
        _warn(sis_time_mark_support_message(support))


def build_cluster_likelihood(
    opts, inp, decoder, lens_sampled_labels=None, fixed_parameter_values=None
):
    """Return a closure logL(sampler_coord) using the branch ParameterDecoder.

    ``decoder.decode(coord) -> (cosmo, survey, pop_params)`` and the decoder was
    built with our ``wl_params``, so ``survey`` already carries the lognormal WL
    parameters and the integer empty-pixel policy. We pass them straight through.
    """
    em = _empty_em_catalog()
    lens_sampled_labels = list(lens_sampled_labels or [])
    fixed_parameter_values = dict(fixed_parameter_values or {})
    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF
    wl_backend = _wl_backend_code(opts)
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if opts.wl_selection == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )
    pair_marks = _resolve_pair_marks(opts, inp)
    _require_time_window(opts, inp)
    _require_time_marks_in_sis_support(opts, inp)
    universe_model = opts.universe_model

    log_p_tag = _pair_tag_log_probs_from_options(opts, inp["lensed"])
    singleton_lensing = (
        SINGLETON_LENSING_MIXTURE
        if getattr(opts, "singleton_lensing", "off") == "sl_mixture"
        else SINGLETON_LENSING_OFF
    )
    if (
        singleton_lensing == SINGLETON_LENSING_MIXTURE
        and cluster_mode == CLUSTER_MODE_J2
        and log_p_tag is not None
        and np.asarray(log_p_tag).size > 0
        and not bool(np.all(np.asarray(log_p_tag) == 0.0))
    ):
        raise SystemExit(
            "--singleton_lensing sl_mixture requires a certain pair tag "
            "(pair_tag probability 1): untagged both-detected pairs would leak "
            "into the singleton stream, which this channel does not model. "
            "Use --pair_tag_model constant --pair_tag_constant 1.0 (study "
            "runner: selection.pair_tag_model/selection.pair_tag_constant)."
        )

    def loglike(coord):
        # decode() returns a 5-tuple (cosmo, survey, pop, sky, mark) on current
        # master; the cluster likelihood is WL-only (sky/mark unused here).
        cosmo, survey, pop_params, _sky_params, _mark_params = _decode_base_parameters(
            decoder, coord
        )
        # Decode the lens block ONCE per likelihood call.  It depends only on
        # ``coord``, so calling it inside ``_call_partition`` re-ran the eager decode
        # for every partition in the Python loop below — up to
        # ``--max_exact_partitions`` times per evaluation under
        # ``--partition_mode marginalize_exact``.
        lens_params = _decode_lens_params(
            coord, lens_sampled_labels, fixed_parameter_values, opts
        )

        def _call_partition(part, *, diagnostics=False):
            fn = (
                darksiren_likelihood_diagnostics_with_clusters
                if diagnostics
                else darksiren_log_likelihood_with_clusters
            )
            return fn(
                cosmo,
                survey,
                pop_params,
                inp["gw_pe"],
                em,
                inp["gw_sel"],
                em,
                inp["nEvents"],
                inp["nsamp"],
                inp["Ndraw"],
                part["singleton_indices"],
                part["pair_indices"],
                part["n_singletons"],
                part["n_pairs"],
                inp["lensed"],
                inp["pair_kdes"],
                lens_params,
                log_p_tag,
                opts.pop_model,
                universe_model,
                sel_batch_size=opts.sel_batch_size,
                cluster_mode=cluster_mode,
                wl_backend=wl_backend,
                wl_a=opts.lensing_wl_a,
                wl_b=opts.lensing_wl_b,
                wl_z_grid=inp.get("wl_z_grid"),
                wl_log_mu_grid=inp.get("wl_log_mu_grid"),
                wl_log_p_table=inp.get("wl_log_p_table"),
                wl_selection=wl_selection,
                pair_marks=pair_marks,
                pair_time_delta_t_obs=part.get(
                    "pair_time_delta_t_obs",
                    inp.get(
                        "pair_time_delta_t_obs", jnp.zeros((0,), dtype=jnp.float64)
                    ),
                ),
                pair_time_sigma=part.get(
                    "pair_time_sigma",
                    inp.get("pair_time_sigma", jnp.zeros((0,), dtype=jnp.float64)),
                ),
                pair_time_t_obs_window_sec=inp.get("pair_time_t_obs_window_sec"),
                pair_time_signed=bool(inp.get("pair_time_signed", False)),
                pair_batch_size=getattr(opts, "pair_batch_size", 0),
                y_nodes_pair=getattr(opts, "y_nodes_pair", 32),
                singleton_lensing=singleton_lensing,
                lensed_singles=inp.get("lensed_singles"),
                fc_pdet_params=inp.get("fc_pdet_params"),
                y_nodes_single=int(getattr(opts, "y_nodes_single", 32)),
                selection_neff_soft_guard=bool(
                    getattr(opts, "selection_neff_soft_guard", False)
                ),
                max_likelihood_variance=float(
                    getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE)
                ),
                pair_orientation_mode=getattr(
                    opts, "pair_orientation_mode", "independent"
                ),
            )

        def _eval_partition(part):
            return _call_partition(part, diagnostics=False)

        def _selection_correction(part):
            return _call_partition(part, diagnostics=True)["selection_correction_total"]

        def _content_loglike(raw):
            return raw["singleton_logL_sum"] + raw["pair_logL_sum"]

        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
            if inp.get("factorized_exact"):
                baseline_raw = _call_partition(inp["baseline_partition"], diagnostics=True)
                baseline = baseline_raw["logL_total"]
                baseline_content = _content_loglike(baseline_raw)
                selection0 = baseline_raw["selection_correction_total"]
                # Count-only selection deltas, CLOSED FORM (see
                # _count_correction_closed_form for why the baseline's
                # pe_variance_sum has to be in there).
                _count_correction = _count_correction_closed_form(baseline_raw, opts)
                count_delta = jnp.stack(
                    [
                        _count_correction(
                            int(part["n_singletons"]), int(part["n_pairs"])
                        )
                        - selection0
                        for part in inp["selection_probe_partitions"]
                    ]
                )
                component_terms = []
                for states, parts in zip(
                    inp["component_partition_states"], inp["component_full_partitions"]
                ):
                    terms = []
                    for state, part in zip(states, parts):
                        n_pairs = int(state.n_pairs)
                        if n_pairs == 0:
                            content_delta = jnp.asarray(0.0, dtype=jnp.float64)
                        else:
                            raw = _call_partition(part, diagnostics=True)
                            content_delta = _content_loglike(raw) - baseline_content
                        terms.append(
                            (
                                n_pairs,
                                jnp.asarray(float(state.log_prior_weight), dtype=jnp.float64)
                                + content_delta,
                            )
                        )
                    component_terms.append(tuple(terms))
                total = (
                    baseline
                    + _factorized_logsumexp_jax(component_terms, count_delta)
                    - inp["log_z_prior"]
                )
                # Prior-corner draws (e.g. a population box that empties the
                # selection integral) give -inf corrections, whose deltas are
                # NaN (-inf minus -inf) and would abort dynesty at live-point
                # init. -inf is the correct sampler-facing value there.
                return jnp.where(jnp.isfinite(total), total, -jnp.inf)
            terms = [
                part["log_prior_weight"] + _eval_partition(part)
                for part in inp["marginal_partitions"]
            ]
            total = jax_logsumexp(jnp.stack(terms)) - inp["log_z_prior"]
            return jnp.where(jnp.isfinite(total), total, -jnp.inf)

        return _eval_partition(inp)
    return loglike


def build_cluster_diagnostics(
    opts, inp, decoder, lens_sampled_labels=None, fixed_parameter_values=None
):
    em = _empty_em_catalog()
    lens_sampled_labels = list(lens_sampled_labels or [])
    fixed_parameter_values = dict(fixed_parameter_values or {})
    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF
    wl_backend = _wl_backend_code(opts)
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if opts.wl_selection == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )
    pair_marks = _resolve_pair_marks(opts, inp)
    _require_time_window(opts, inp)
    _require_time_marks_in_sis_support(opts, inp)
    universe_model = opts.universe_model
    log_p_tag = _pair_tag_log_probs_from_options(opts, inp["lensed"])
    singleton_lensing = (
        SINGLETON_LENSING_MIXTURE
        if getattr(opts, "singleton_lensing", "off") == "sl_mixture"
        else SINGLETON_LENSING_OFF
    )

    def diagnostics(coord):
        coord = jnp.asarray(coord)
        cosmo, survey, pop_params, _sky_params, _mark_params = _decode_base_parameters(
            decoder, coord
        )
        # Decode the lens block once per call, not once per partition probe
        # (mirrors the same hoist in build_cluster_loglike).
        lens_params = _decode_lens_params(
            coord, lens_sampled_labels, fixed_parameter_values, opts
        )

        def _raw_for(singletons, pairs, n_singletons, n_pairs, part=None):
            return darksiren_likelihood_diagnostics_with_clusters(
                cosmo,
                survey,
                pop_params,
                inp["gw_pe"],
                em,
                inp["gw_sel"],
                em,
                inp["nEvents"],
                inp["nsamp"],
                inp["Ndraw"],
                singletons,
                pairs,
                n_singletons,
                n_pairs,
                inp["lensed"],
                inp["pair_kdes"],
                lens_params,
                log_p_tag,
                opts.pop_model,
                universe_model,
                sel_batch_size=opts.sel_batch_size,
                cluster_mode=cluster_mode,
                wl_backend=wl_backend,
                wl_a=opts.lensing_wl_a,
                wl_b=opts.lensing_wl_b,
                wl_z_grid=inp.get("wl_z_grid"),
                wl_log_mu_grid=inp.get("wl_log_mu_grid"),
                wl_log_p_table=inp.get("wl_log_p_table"),
                wl_selection=wl_selection,
                pair_marks=pair_marks,
                pair_time_delta_t_obs=(part or {}).get(
                    "pair_time_delta_t_obs",
                    inp.get(
                        "pair_time_delta_t_obs", jnp.zeros((0,), dtype=jnp.float64)
                    ),
                ),
                pair_time_sigma=(part or {}).get(
                    "pair_time_sigma",
                    inp.get("pair_time_sigma", jnp.zeros((0,), dtype=jnp.float64)),
                ),
                pair_time_t_obs_window_sec=inp.get("pair_time_t_obs_window_sec"),
                pair_time_signed=bool(inp.get("pair_time_signed", False)),
                pair_batch_size=getattr(opts, "pair_batch_size", 0),
                y_nodes_pair=getattr(opts, "y_nodes_pair", 32),
                singleton_lensing=singleton_lensing,
                lensed_singles=inp.get("lensed_singles"),
                fc_pdet_params=inp.get("fc_pdet_params"),
                y_nodes_single=int(getattr(opts, "y_nodes_single", 32)),
                selection_neff_soft_guard=bool(
                    getattr(opts, "selection_neff_soft_guard", False)
                ),
                max_likelihood_variance=float(
                    getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE)
                ),
                pair_orientation_mode=getattr(
                    opts, "pair_orientation_mode", "independent"
                ),
            )

        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
            def _global_diagnostics():
                state_to_part = {
                    id(state): part
                    for state, part in zip(
                        inp["partition_states"], inp["marginal_partitions"]
                    )
                }

                def _part_loglike(state):
                    part = state_to_part[id(state)]
                    raw = _raw_for(
                        jnp.asarray(state.singleton_indices, dtype=jnp.int32),
                        jnp.asarray(state.pair_indices, dtype=jnp.int32),
                        state.n_singletons,
                        state.n_pairs,
                        part,
                    )
                    return float(np.asarray(raw["logL_total"]))

                return compute_marginalized_partition_diagnostics(
                    inp["partition_states"],
                    inp["candidate_pairs"],
                    _part_loglike,
                    log_z_partition_prior=inp["log_z_prior"],
                    raw_candidate_pairs=inp.get("candidate_pairs_raw"),
                    edge_mark_prior_contributions=inp.get("edge_mark_prior_contributions"),
                )

            if inp.get("factorized_exact"):
                baseline_part = inp["baseline_partition"]
                baseline_raw = _raw_for(
                    baseline_part["singleton_indices"],
                    baseline_part["pair_indices"],
                    baseline_part["n_singletons"],
                    baseline_part["n_pairs"],
                    baseline_part,
                )
                baseline_loglike = float(np.asarray(baseline_raw["logL_total"]))
                baseline_content = float(
                    np.asarray(
                        baseline_raw["singleton_logL_sum"] + baseline_raw["pair_logL_sum"]
                    )
                )
                # Count-only selection deltas from the SAME closed form the
                # sampler uses, so the two paths cannot drift apart. Each count
                # used to be read off a FULL master-likelihood evaluation of a
                # fabricated partition pairing events (0,1),(2,3),... with
                # delta_t_obs = 0 and sigma = 1 s: those fictitious pairs carry
                # their own pair_variance_sum into the guard threshold (so under
                # the soft guard the count-only check below compared two
                # different variance budgets), and the probes run out to
                # n_pairs = sum of per-component maxima — far beyond any real
                # component partition — re-incurring the per-(n_singletons,
                # n_pairs) XLA specialization the sampler path was refactored
                # away from (review F-005). The check below now compares the
                # closed form against REAL component partitions, which the
                # diagnostics evaluate anyway: a stronger invariant at no cost.
                _count_correction = _count_correction_closed_form(baseline_raw, opts)
                selection0 = float(
                    np.asarray(baseline_raw["selection_correction_total"])
                )
                count_delta = [
                    float(
                        np.asarray(
                            _count_correction(
                                int(probe_part["n_singletons"]),
                                int(probe_part["n_pairs"]),
                            )
                        )
                    )
                    - selection0
                    for probe_part in inp["selection_probe_partitions"]
                ]
                soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
                count_only_warned = False
                component_deltas = []
                for states, parts in zip(
                    inp["component_partition_states"], inp["component_full_partitions"]
                ):
                    deltas = []
                    for state, part in zip(states, parts):
                        n_pairs = int(state.n_pairs)
                        if n_pairs == 0:
                            deltas.append(0.0)
                            continue
                        raw = _raw_for(
                            part["singleton_indices"],
                            part["pair_indices"],
                            part["n_singletons"],
                            part["n_pairs"],
                            part,
                        )
                        local_content = float(
                            np.asarray(raw["singleton_logL_sum"] + raw["pair_logL_sum"])
                        )
                        local_selection_delta = float(
                            np.asarray(raw["selection_correction_total"])
                        ) - float(selection0)
                        if not (
                            np.isfinite(local_selection_delta)
                            and np.isfinite(count_delta[n_pairs])
                        ):
                            raise RuntimeError(
                                "componentwise exact factorization: selection "
                                "correction is NON-FINITE at the evaluation point "
                                f"(delta {local_selection_delta!r} vs "
                                f"{count_delta[n_pairs]!r}) — typically the "
                                "reliability guard firing at the prior midpoint "
                                "(the Neff <= 5 N_obs floor or the variance "
                                "criterion Neff <= N_obs^2/max_likelihood_"
                                "variance). Increase the injection campaign "
                                "(or use a profile whose Neff clears the "
                                "threshold, or relax --max_likelihood_variance "
                                "for exploratory runs); the factorization "
                                "check cannot run on -inf corrections."
                            )
                        if not np.isclose(
                            local_selection_delta,
                            count_delta[n_pairs],
                            rtol=0.0,
                            atol=1e-8,
                        ):
                            detail = (
                                f"(n_pairs={n_pairs}: component delta "
                                f"{local_selection_delta!r} vs closed-form delta "
                                f"{count_delta[n_pairs] if n_pairs < len(count_delta) else 'MISSING'!r}, "
                                f"n_count_values={len(count_delta)}; component part "
                                f"n_singletons={part['n_singletons']}, n_pairs={part['n_pairs']})"
                            )
                            if soft_guard:
                                # Under the soft guard the correction is count-only
                                # ONLY where the smooth wall is inactive: inside it
                                # the wall tracks the threshold, which the
                                # partition's own pair_variance_sum moves. The
                                # sampler's factorization deliberately uses the
                                # baseline budget there, so this is an accuracy
                                # notice about a region the guard is repelling the
                                # sampler out of -- not a reason to kill the run at
                                # build time after the full PE/injection load
                                # (review F-005).
                                if not count_only_warned:
                                    count_only_warned = True
                                    _warn(
                                        "componentwise factorization is inexact at "
                                        "the diagnostics point: the soft guard's "
                                        "wall is active, so the selection "
                                        f"correction is not count-only {detail}. "
                                        "The sampler uses the baseline variance "
                                        "budget; check that no posterior mass "
                                        "sits in the walled region."
                                    )
                            else:
                                raise RuntimeError(
                                    "componentwise exact factorization failed: "
                                    f"selection correction is not count-only {detail}"
                                )
                        deltas.append(local_content - baseline_content)
                    component_deltas.append(deltas)
                out = compute_componentwise_factorized_partition_diagnostics(
                    inp["component_partition_states"],
                    inp["candidate_pairs"],
                    component_deltas,
                    count_delta,
                    baseline_loglike,
                    log_z_partition_prior=inp["log_z_prior"],
                    component_summaries=inp.get("component_partition_summaries"),
                    raw_candidate_pairs=inp.get("candidate_pairs_raw"),
                    edge_mark_prior_contributions=inp.get("edge_mark_prior_contributions"),
                )
                if inp.get("partition_states") is not None:
                    global_out = _global_diagnostics()
                    pair_diff = max(
                        [
                            abs(float(a["p_pair"]) - float(b["p_pair"]))
                            for a, b in zip(
                                out["posterior_pair_probabilities"],
                                global_out["posterior_pair_probabilities"],
                            )
                        ]
                        or [0.0]
                    )
                    log_diff = abs(float(out["logL_marginalized"]) - float(global_out["logL_marginalized"]))
                    exp_diff = abs(float(out["expected_n_pairs"]) - float(global_out["expected_n_pairs"]))
                    map_pairs = {tuple(pair) for pair in out["map_partition"].get("pair_indices", [])}
                    global_map_pairs = {tuple(pair) for pair in global_out["map_partition"].get("pair_indices", [])}
                    if log_diff > 1e-6 or exp_diff > 1e-6 or pair_diff > 1e-6 or map_pairs != global_map_pairs:
                        raise RuntimeError(
                            "componentwise factorized exact diagnostics do not match global exact enumeration"
                        )
                    for key in (
                        "partition_log_prior_weight",
                        "partition_logL",
                        "partition_log_posterior_weight",
                        "partition_posterior_probability",
                        "partitions",
                        "map_partition_index",
                    ):
                        out[key] = global_out[key]
                    out["global_partitions_enumerated"] = True
                    out["factorized_global_check"] = {
                        "logL_abs_diff": log_diff,
                        "expected_n_pairs_abs_diff": exp_diff,
                        "posterior_pair_probability_max_abs_diff": pair_diff,
                    }
                selected_states = [
                    inp["component_partition_states"][idx][state_idx]
                    for idx, state_idx in enumerate(out["map_component_partition_indices"])
                ]
                map_state = combine_component_partitions(tuple((state,) for state in selected_states))[0]
                map_part = _runtime_part_from_state(
                    map_state,
                    inp["candidate_pairs"],
                    pair_marks=getattr(opts, "pair_marks", "none"),
                )
                raw = _raw_for(
                    map_part["singleton_indices"],
                    map_part["pair_indices"],
                    map_part["n_singletons"],
                    map_part["n_pairs"],
                    map_part,
                )
                out.update({f"map_{k}": v for k, v in _diagnostics_to_python(raw).items()})
            else:
                out = _global_diagnostics()
                out["partition_component_mode"] = "global"
                out["factorized_exact"] = False
                out["global_partitions_enumerated"] = True
                map_part = inp["marginal_partitions"][int(out["map_partition_index"])]
                raw = _raw_for(
                    jnp.asarray(out["map_partition"]["singleton_indices"], dtype=jnp.int32),
                    jnp.asarray(out["map_partition"]["pair_indices"], dtype=jnp.int32),
                    out["map_partition"]["n_singletons"],
                    out["map_partition"]["n_pairs"],
                    map_part,
                )
                out.update({f"map_{k}": v for k, v in _diagnostics_to_python(raw).items()})
        else:
            raw = _raw_for(
                inp["singleton_indices"],
                inp["pair_indices"],
                inp["n_singletons"],
                inp["n_pairs"],
            )
            out = _diagnostics_to_python(raw)
            out.update(partition_mode="fixed", n_partitions=1)
        out.update(
            cluster_mode=opts.cluster_mode,
            wl_backend=opts.wl_backend,
            wl_selection=opts.wl_selection,
            singleton_lensing=getattr(opts, "singleton_lensing", "off"),
            pair_orientation_mode=getattr(
                opts, "pair_orientation_mode", "independent"
            ),
            pair_batch_size=getattr(opts, "pair_batch_size", 0),
            y_nodes_pair=getattr(opts, "y_nodes_pair", 32),
            pe_max_per_pair=opts.pe_max_per_pair,
            edge_mark_prior_keys=inp.get("edge_mark_prior_keys", []),
            edge_prior_semantics="effective_log_prior_odds = raw_log_prior_odds + sum(requested log_* marks)",
            edge_prior_applied_once=True,
            approximate_pair_evaluation_shape=[
                int(out.get("expected_n_pairs", inp["n_pairs"])),
                int(inp["nsamp"]),
                int(opts.y_nodes_pair),
            ],
            # `coord` here is whatever point cleared the reliability guard --
            # the prior midpoint, the registry fiducial, or a seeded prior draw
            # -- so sampled lens values are named for that generic eval point.
            **_lens_settings_dict(
                coord,
                lens_sampled_labels,
                fixed_parameter_values,
                opts,
                eval_point="diagnostics_point",
            ),
        )
        _add_off_control_nonfinite_diagnostics(out, opts=opts, inp=inp)
        return out

    return diagnostics


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="darksirens lensing inference (singleton + J=2 cluster)"
    )
    data = p.add_argument_group("Data")
    data.add_argument("--gw_path", required=True)
    data.add_argument("--gwselection_path", required=True)
    data.add_argument("--lensed_injections_path", default=None)
    data.add_argument(
        "--pair_pe_path", default=None,
        help=(
            "DEPRECATED legacy split-pair layout: the mock generator stopped "
            "writing mock_pair_pe.h5 by default (2026-07-01 unified-catalog "
            "migration) and preflight's event-index range check has never "
            "accepted the split layout. Use the unified observed catalog "
            "(--observed_catalog_path + optional --pair_metadata_path)."
        ),
    )
    data.add_argument(
        "--pair_metadata_path",
        default=None,
        help="Optional pair/candidate-edge metadata file. Preferred over --pair_pe_path in unified observed mode.",
    )
    data.add_argument("--partition_path", default=None)
    data.add_argument("--candidate_pairs_path", default=None)
    data.add_argument(
        "--observed_catalog_path",
        default=None,
        help="Explicit observed_catalog.json for unified observed lensing mode.",
    )
    data.add_argument(
        "--partition_mode", choices=["fixed", "marginalize_exact"], default="fixed"
    )
    data.add_argument("--max_exact_partitions", type=int, default=10000)
    data.add_argument(
        "--partition_component_mode",
        choices=["global", "componentwise"],
        default="componentwise",
    )
    data.add_argument("--max_component_events", type=int, default=None)
    data.add_argument("--max_component_edges", type=int, default=None)
    data.add_argument("--max_component_partitions", type=int, default=None)
    data.add_argument("--max_total_partitions", type=int, default=None)
    model = p.add_argument_group("Model")
    model.add_argument("--pop_model", default="powerlaw+peak")
    model.add_argument("--cluster_mode", choices=["off", "j2"], default="j2")
    model.add_argument(
        "--wl_backend",
        choices=["lognormal", "tabulated", "disabled"],
        default="lognormal",
    )
    model.add_argument(
        "--lensing_wl_table_path",
        type=str,
        default=None,
        help="Path to HDF5 table of log p_WL(mu|z) for --wl_backend tabulated. "
        "Datasets: z_grid (Nz,), log_mu_grid (Nmu,), log_p_table (Nz, Nmu).",
    )
    model.add_argument(
        "--wl_selection",
        choices=["auto", "standard", "wl_lognormal"],
        default="auto",
        help="Singleton selection treatment. 'auto' (default) matches the "
             "event-side --wl_backend (lognormal -> wl_lognormal, disabled -> "
             "standard) so the hierarchical numerator and denominator share "
             "one observation model; an explicit mismatched pair is fatal "
             "unless --allow_mismatched_wl_selection. wl_lognormal uses "
             "lognormal/Hermite WL marginalization for singleton injections "
             "(wl_a=0 reduces to standard).",
    )
    model.add_argument(
        "--allow_mismatched_wl_selection",
        action="store_true",
        default=False,
        help="UNSAFE ablation switch: accept an event/selection WL pairing "
             "that normalizes the hierarchy under a different observation "
             "model than the per-event numerator (biases the posterior). The "
             "override is recorded in settings.json.",
    )
    model.add_argument("--lensing_wl_a", type=float, default=4e-3)
    model.add_argument("--lensing_wl_b", type=float, default=1.5)
    model.add_argument("--sl_tau_A", type=float, default=5e-4)
    model.add_argument("--sl_tau_n", type=float, default=3.0)
    model.add_argument(
        "--sl_T0_sec",
        type=float,
        default=DEFAULT_T0_SECONDS,
        help="SIS time-delay scale T0 in seconds (Delta t = T0 * y). Default "
        f"{DEFAULT_T0_SECONDS:.3g} s (~62 d) is the SIS scale at z_L=0.5, "
        "z_s=1, sigma_v=200 km/s under this repo's cosmology. Candidate pairs "
        "with |dt| >= T0 fall outside the SIS support y in (0,1) and get an "
        "exactly -inf time-marked pair likelihood.",
    )
    model.add_argument(
        "--fix_lens_rate",
        type=str_to_bool, default=True, metavar="BOOL",
        help="true fixes SIS optical-depth parameters to --sl_tau_A/--sl_tau_n; false samples lensing hyperparameters",
    )
    model.add_argument(
        "--lens_prior_overrides",
        default=None,
        help='JSON dict of SIS lens prior overrides, e.g. {"log10_tau_A": [-6, -3]}',
    )
    model.add_argument(
        "--pair_marks",
        choices=["none", "time"],
        default="none",
        help="Optional J=2 pair marks. 'time' uses candidate_pairs.json marks in marginalized mode or pair metadata in fixed mode.",
    )
    model.add_argument(
        "--pair_time_sigma_sec",
        type=float,
        default=None,
        help="Fallback sigma_delta_t in seconds when pair time metadata omits sigma.",
    )
    model.add_argument("--pair_tag_model", choices=PAIR_TAG_SELECTION_MODEL_KINDS, default="constant")
    model.add_argument("--pair_tag_constant", type=float, default=1.0)
    model.add_argument("--pair_tag_perturb_logit", type=float, default=0.0)
    model.add_argument("--pair_tag_selection_path", default=None)
    model.add_argument(
        "--edge_mark_prior_keys",
        default="",
        help="Comma-separated log_* candidate edge marks to add to edge log_prior_odds in exact marginalization.",
    )
    model.add_argument(
        "--edge_mark_likelihood_keys",
        default="",
        help="Comma-separated edge mark likelihood keys. Only time/delta_t_obs "
             "is implemented in this PR, and it is not a route of its own: the "
             "marked pair likelihood is switched on by --pair_marks time, so "
             "requesting a time-like key without it is a fatal error rather "
             "than a silently inert flag.",
    )
    model.add_argument(
        "--allow_suspicious_time_marks",
        type=str_to_bool, default=False, metavar="BOOL",
        help="true downgrades the placeholder/synthetic time-mark hard error to a warning",
    )
    model.add_argument(
        "--allow_time_mark_mismatch",
        type=str_to_bool, nargs="?", const=True, default=False, metavar="BOOL",
        help="true downgrades the non-finite / catalog-disagreeing time-mark "
             "hard errors to warnings, keeping the legacy behaviour of using "
             "the MARK's magnitude with the catalog's arrival order",
    )
    model.add_argument(
        "--pair_time_mark_impl",
        choices=["auto", "quadrature", "delta"],
        default="auto",
        help="time-mark y-integral implementation: auto delta-collapses when "
             "max(sigma_dt)/T0 < 0.02 (sharp marks unresolvable by quadrature), "
             "quadrature/delta force the respective path",
    )
    model.add_argument(
        "--singleton_lensing",
        choices=["off", "sl_mixture"],
        default="off",
        help="off keeps the legacy singleton channel (drop-single-image protocol). "
             "sl_mixture models observed singletons as a mixture of unlensed sources "
             "and strongly lensed sources with exactly one detected image "
             "(evidence mixture + exactly-one-detected selection subset + analytic "
             "Finn-Chernoff partner censoring); requires --lensed_injections_path "
             "and a mock generated with --include-lensed-singletons true.",
    )
    model.add_argument("--y_nodes_single", type=int, default=32,
                   help="Gauss-Legendre y nodes for the lensed-singleton evidence")
    model.add_argument("--fc_rho_thr", type=float, default=None,
                   help="override the injection file's fc_rho_thr attr")
    model.add_argument("--fc_r0", type=float, default=None,
                   help="override the injection file's fc_r0 attr")
    model.add_argument("--fc_mc_bar", type=float, default=None,
                   help="override the injection file's fc_mc_bar attr")
    model.add_argument(
        "--pair_orientation_mode",
        choices=["independent", "shared_iota"],
        default="independent",
        help="two-image orientation convention of the lensed-singleton "
             "censoring factor. 'independent' (default) marginalises the "
             "partner image's Finn-Chernoff Theta independently — matching "
             "mocks rendered with independent per-image draws. 'shared_iota' "
             "uses the joint-orientation model (shared inclination, antenna "
             "decorrelated over the SIS time delay) — REQUIRES a campaign "
             "generated with --pair-orientation-mode shared_iota; mixing "
             "conventions mis-normalises the J=2/lensed-singleton ratio by "
             "up to ~2.6x (see darksirens.lensing.fcpdet).",
    )
    model.add_argument(
        "--allow_pair_orientation_mismatch",
        type=str_to_bool, nargs="?", const=True, default=False, metavar="BOOL",
        help="true downgrades the campaign-vs-runtime --pair_orientation_mode "
             "hard error to a warning. Fatal by default only when "
             "--singleton_lensing sl_mixture puts the censoring factor in the "
             "likelihood; use for deliberate convention ablations only.",
    )
    fixing = p.add_argument_group("Fixing")
    fixing.add_argument(
        "--fix_cosmology", type=str_to_bool, default=True, metavar="BOOL",
        help="false samples H0/Om0/w0/wa. REFUSED with --cluster_mode j2 or "
             "--singleton_lensing sl_mixture: the lensed-injection selection "
             "terms are valid only at the campaign's fiducial cosmology.",
    )
    fixing.add_argument("--fix_survey", type=str_to_bool, default=True, metavar="BOOL")
    fixing.add_argument("--fix_population", type=str_to_bool, default=False, metavar="BOOL")
    fixing.add_argument(
        "--fixed_parameter_values", default=None, help="JSON dict of {label: value}"
    )
    fixing.add_argument(
        "--prior_overrides", default=None, help="JSON dict of {label: [lo, hi]}"
    )
    fixing.add_argument(
        "--redshift_prior_barrier",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Controls the internal redshift-prior state optimization barrier. 'auto' keeps "
            "the barrier for ordinary JIT paths but disables the likelihood-internal barrier "
            "for TinyNS JAX rwalk because JAX cannot batch optimization_barrier under vmap. "
            "Use 'on' only for non-vmapped samplers; use 'off' to force vmappable behavior."
        ),
    )
    fixing.add_argument(
        "--selection_neff_guard",
        choices=["auto", "hard", "soft"],
        default="auto",
        help=(
            "Sparse-selection (Neff <= 5 N_obs) validity guard for the combined "
            "singleton+cluster correction. 'hard' returns -inf (nested samplers; "
            "historical behavior). 'soft' replaces the wall with a steep smooth "
            "penalty so gradient-based samplers are not divergence-flagged on "
            "every trajectory that brushes it. 'auto' picks soft for "
            "--sampler numpyro and hard otherwise."
        ),
    )
    fixing.add_argument(
        "--max_likelihood_variance", type=float, default=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        help=("Cap on the Monte-Carlo variance of the log-likelihood estimator "
              "(Essick & Farr 2022; Talbot & Golomb 2023, arXiv:2304.06138; the "
              "GWTC-4.0/5.0 criterion is sigma^2_lnL <= 1, the default). The cap "
              "bounds the TOTAL sigma^2_lnL = pe_variance_sum + "
              "N_obs^2/Neff_sel: the summed per-event, lensed-singleton and "
              "per-pair importance variances spend the budget before the "
              "selection term, on every evaluation path (fixed partition, both "
              "--partition_component_mode settings, and the diagnostics). "
              "Proposals exceeding it are guarded (hard -inf or the soft wall "
              "per --selection_neff_guard). The Vitale 5 N_obs mean floor "
              "always applies."))
    sampler = p.add_argument_group("Sampler")
    sampler.add_argument("--sampler", required=True, choices=["tinyns", "dynesty", "numpyro"])
    sampler.add_argument("--nlive", type=int, default=2000)
    sampler.add_argument("--dlogz", type=float, default=0.1)
    sampler.add_argument("--max_samples", type=int, default=2_000_000)
    add_tinyns_arguments(sampler)
    sampler.add_argument("--nuts_warmup", type=int, default=500)
    sampler.add_argument("--nuts_samples", type=int, default=2000)
    sampler.add_argument("--nuts_chains", type=int, default=4)
    sampler.add_argument("--nuts_target_accept", type=float, default=0.8)
    sampler.add_argument("--nuts_max_tree_depth", type=int, default=10)
    sampler.add_argument("--nuts_chain_method", default="sequential",
                   choices=["sequential", "parallel", "vectorized"])
    sampler.add_argument("--nuts_init_tries", type=int, default=32)
    # Shared with the main CLI (darksirens/inference/checkpointing.py): sampler
    # checkpoints land in the run directory and --resume auto continues there.
    add_checkpoint_arguments(sampler)
    performance = p.add_argument_group("Performance")
    performance.add_argument(
        "--pe_max_per_pair",
        type=int,
        default=400,
        help="down-sample PE per pair image (0=keep all). Controls "
        "the O(N_pe^2 N_y) pair-KDE memory.",
    )
    performance.add_argument(
        "--pair_batch_size",
        type=int,
        default=0,
        help="candidate-pair batch size for J=2 likelihood scans (0 keeps legacy unbatched path)",
    )
    performance.add_argument(
        "--y_nodes_pair",
        type=int,
        default=32,
        help="Gauss-Legendre y nodes for each J=2 pair likelihood",
    )
    performance.add_argument(
        "--sel_batch_size", type=block_size_arg, default="auto",
        metavar="N|auto|off",
        help="Injections per selection-integral chunk. 'auto' (default) sizes "
             "the block from probed free device memory after the data load; "
             "'off'/'none'/'0' forces a single pass; a positive integer pins it.")
    # The lensing likelihood evaluates the SAME PairingModel grid branch as the
    # main CLI, and the pairing grid is a process-global setting read from the
    # environment, so this stack inherited DARKSIRENS_GW_PAIRING_M1_GRID with no
    # way to opt in or out and no coverage guard.
    add_normalization_grid_arguments(performance)
    output = p.add_argument_group("Output")
    output.add_argument("--seed", type=int, default=42)
    output.add_argument("--show_progress", type=str_to_bool, nargs="?", const=True,
                        default=False, metavar="BOOL")
    output.add_argument("--save_path", default="./")
    output.add_argument(
        "--preflight_only",
        type=str_to_bool, default=False, metavar="BOOL",
        help="true runs lensing input preflight checks, writes JSON, and exits before compilation/sampling",
    )
    output.add_argument(
        "--preflight_json",
        default=None,
        help="optional output path for preflight JSON (defaults to save_path/preflight.json when --preflight_only true)",
    )
    return p


def _resolve_lensing_block_sizes(opts, inp, settings):
    """Resolve the lensing selection block size from a probed memory budget.

    The lensing CLI exposes only ``--sel_batch_size`` (no per-event PE knob).
    The per-injection memory constants are those calibrated on the main
    spectral/dark-siren selection integral (see docs/source/performance.md); the
    lensing selection is the same order of magnitude, so they serve as a
    reasonable default here — pin an integer to override.  Mutates ``opts`` and
    the already-built ``settings`` dict so the final settings.json persists the
    resolved value (the early settings write predates the data load).
    """
    import numpy as _np
    import jax as _jax

    from darksirens.gw.populations.utils import normalization_grid_settings

    gw_sel = inp.get("gw_sel")
    n_sel = int(_np.asarray(gw_sel.m1det).shape[0]) if gw_sel is not None else 0
    # The default constants are the value+GRAD peak; the lensing campaign runs
    # ``--sampler tinyns``, which never differentiates the likelihood, so it must
    # get the value-only constant set (14x smaller peak on the calibration config)
    # instead of being blocked into a 1.7-2.2x slower plan for memory it has spare
    # of.  ``concurrent_evals`` covers tinyns' vmapped replacement chains.
    needs_grad, concurrent_evals = sampler_block_sizing_profile(opts)
    n_q = int(normalization_grid_settings().n_q)
    plan = resolve_block_sizes(
        n_events=int(inp.get("nEvents", 0) or 0),
        n_samp=int(inp.get("nsamp", 0) or 0),
        n_sel=n_sel,
        sel_requested=getattr(opts, "sel_batch_size", None),
        pe_requested=None,          # lensing has no PE-event block knob
        has_catalog=False,          # lensing selection carries no galaxy catalog
        flow_path=False,
        n_q=n_q,
        needs_grad=needs_grad,
        concurrent_evals=concurrent_evals,
        backend=_jax.default_backend(),
    )
    opts.sel_batch_size = plan.sel_batch_size
    opts.block_size_resolution = plan.source
    settings["sel_batch_size"] = plan.sel_batch_size
    settings["block_size_resolution"] = plan.source
    _sel = plan.sel_batch_size if plan.sel_batch_size is not None else "single pass"
    _ok(f"Block size [{plan.source}]: sel_batch_size={_sel}")
    _ok(
        "Peak model: "
        f"{'value+grad' if needs_grad else 'value-only'} "
        f"({getattr(opts, 'sampler', '?')}"
        + (f", {concurrent_evals} concurrent evals" if concurrent_evals > 1 else "")
        + f"); n_q={n_q}"
    )


def _validate_json_options(opts):
    """Validate the JSON flags before the run directory is created so malformed
    JSON exits cleanly (code 1) rather than deep inside the run."""
    parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    parse_json_arg(opts.prior_overrides, "prior_overrides")
    parse_json_arg(opts.lens_prior_overrides, "lens_prior_overrides")


def _resolve_lensing_run_config(opts):
    """Resolve sampler/barrier/guard config, validate the edge-mark and WL
    flags, and derive the universe model from the WL backend."""
    if opts.sampler == "tinyns":
        build_tinyns_config(opts)
    resolve_redshift_prior_barrier(opts, flush=True)

    guard_mode, max_likelihood_variance = resolve_selection_neff_guard(opts)
    if opts.selection_neff_soft_guard:
        print(
            "  [i] Sparse-selection guard: SOFT (smooth wall for "
            f"gradient-based sampling; mode={guard_mode}). Verify the "
            "posterior clears the total-variance criterion "
            "(pe_variance_sum + N_obs^2/Neff <= max_likelihood_variance, and "
            "the Neff <= 5 N_obs floor) post hoc. Per-event and per-pair "
            "importance variances are included in this stack's guard.",
            flush=True,
        )
    if max_likelihood_variance != DEFAULT_MAX_LIKELIHOOD_VARIANCE:
        print(
            "  [i] Total-log-likelihood variance cap (per-event and per-pair "
            "importance variances included): "
            f"max_likelihood_variance={max_likelihood_variance} "
            "(default 1.0 mirrors the GWTC-4.0/5.0 total criterion).",
            flush=True,
        )

    edge_like_keys = parse_edge_mark_keys(opts.edge_mark_likelihood_keys)
    unsupported_edge_like = [
        k for k in edge_like_keys if k not in ("time", "delta_t_obs")
    ]
    if unsupported_edge_like:
        raise NotImplementedError(
            "edge_mark_likelihood_keys only supports time/delta_t_obs in this PR; "
            f"unsupported keys: {unsupported_edge_like}"
        )
    # The ONLY switch that puts arrival-time information into the pair
    # likelihood is --pair_marks time (_resolve_pair_marks branches on it
    # alone); the likelihood keys reach only the placeholder-mark gate and the
    # preflight summary. So `--edge_mark_likelihood_keys time --pair_marks none`
    # used to run with no time term at all while settings.json and the preflight
    # summary recorded the key as honoured -- and could even abort on
    # "suspicious candidate time marks" for marks nothing would have read
    # (review F-007). Resolve the two through one predicate instead.
    if set(edge_like_keys) & {"time", "delta_t_obs"} and (
        getattr(opts, "pair_marks", "none") != "time"
    ):
        raise SystemExit(
            "--edge_mark_likelihood_keys "
            f"{','.join(edge_like_keys)} requires --pair_marks time: the marked "
            "pair likelihood is enabled by --pair_marks alone, so this "
            "combination would discard the arrival-time term while recording it "
            "as used. Add --pair_marks time (or drop the likelihood key)."
        )
    if any(
        not k.startswith("log_")
        for k in parse_edge_mark_keys(opts.edge_mark_prior_keys)
    ):
        raise SystemExit("--edge_mark_prior_keys entries must be log_* marks")

    if opts.wl_backend == "tabulated" and not opts.lensing_wl_table_path:
        raise SystemExit(
            "--wl_backend tabulated requires --lensing_wl_table_path"
        )

    _resolve_wl_selection(opts)
    _validate_partition_mode_against_cluster_mode(opts)
    _validate_fixed_cosmology_for_lensed_channels(opts)

    opts.universe_model = _derive_universe_model(opts.wl_backend)


def _resolve_wl_selection(opts):
    """Resolve ``--wl_selection auto`` and reject event/selection mismatches.

    The per-event (numerator) WL treatment comes from ``--wl_backend``; the
    hierarchical selection normalization (denominator) from
    ``--wl_selection``.  Computing the two under DIFFERENT observation models
    normalizes the hierarchy against a likelihood it is not using — the old
    default did exactly that (lognormal events over standard selection) —
    so ``auto`` picks the matched pair and an explicit mismatch is fatal
    unless ``--allow_mismatched_wl_selection`` stamps a deliberate ablation.

    ``--wl_backend tabulated`` has NO matched selection integral in this
    stack: 'auto' refuses it outright, and running it mismatched requires
    both explicit flags.  The one exact exception: ``lensing_wl_a == 0``
    collapses the lognormal magnification kernel to a delta function, so
    'standard' selection is then identical, not mismatched.
    """
    requested = str(opts.wl_selection)
    # The pre-resolution request is provenance: settings.json serializes
    # vars(opts), so both the ask and the answer are recorded.
    opts.wl_selection_requested = requested
    matched = {"lognormal": "wl_lognormal", "disabled": "standard"}

    if requested == "auto":
        resolved = matched.get(opts.wl_backend)
        if resolved is None:
            raise SystemExit(
                "--wl_backend tabulated has no matched selection integral: "
                "the event side would marginalize the tabulated p_WL(mu|z) "
                "while selection stayed 'standard', normalizing the "
                "hierarchy under a different observation model than the "
                "numerator. Pass an explicit '--wl_selection standard "
                "--allow_mismatched_wl_selection' only for a deliberately "
                "mismatched ablation."
            )
        opts.wl_selection = resolved
        return

    mismatch = None
    if requested == "wl_lognormal" and opts.wl_backend != "lognormal":
        mismatch = (
            f"--wl_selection wl_lognormal needs --wl_backend lognormal (got "
            f"{opts.wl_backend!r}): selection would marginalize a WL kernel "
            "the event side never applies."
        )
    elif requested == "standard" and opts.wl_backend == "lognormal" and (
            float(opts.lensing_wl_a) != 0.0):
        mismatch = (
            "--wl_selection standard under --wl_backend lognormal (wl_a = "
            f"{opts.lensing_wl_a}): events marginalize lognormal WL while "
            "selection does not — the hierarchy is normalized under a "
            "different observation model than the numerator."
        )
    elif requested == "standard" and opts.wl_backend == "tabulated":
        mismatch = (
            "--wl_selection standard under --wl_backend tabulated: events "
            "marginalize the tabulated p_WL(mu|z) while selection does not."
        )
    if mismatch is None:
        return
    if getattr(opts, "allow_mismatched_wl_selection", False):
        warnings.warn(
            "MISMATCHED WL event/selection pairing accepted by "
            "--allow_mismatched_wl_selection (ablation only, the posterior "
            "is biased): " + mismatch,
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise SystemExit(
        "FATAL: " + mismatch + " Use --wl_selection auto (the default), or "
        "pass --allow_mismatched_wl_selection for a deliberately mismatched "
        "ablation."
    )


def _print_run_configuration(opts):
    _section("Run Configuration")
    _row("Cluster mode",      opts.cluster_mode)
    _row("Universe model",    opts.universe_model)
    _row("Partition mode",    opts.partition_mode)
    _row("Population model",  opts.pop_model)
    print("  │")
    _row("WL backend",        opts.wl_backend)
    _row("WL selection",      opts.wl_selection)
    _row("WL a, b",           f"{opts.lensing_wl_a}, {opts.lensing_wl_b}")
    if opts.lensing_wl_table_path:
        _row("WL table",      opts.lensing_wl_table_path)
    print("  │")
    _row("Pair marks",        opts.pair_marks)
    _row("Pair tag model",    f"{opts.pair_tag_model} "
                              f"(constant={opts.pair_tag_constant}, "
                              f"perturb_logit={opts.pair_tag_perturb_logit})")
    if opts.edge_mark_prior_keys:
        _row("Edge prior marks", opts.edge_mark_prior_keys)
    _row("Singleton lensing", opts.singleton_lensing)
    print("  │")
    _row("Fix lens rate",     "yes" if opts.fix_lens_rate else "no")
    _row("SIS tau A, n",      f"{opts.sl_tau_A}, {opts.sl_tau_n}")
    _row("SIS T0", f"{_sl_T0_seconds(opts):.4g} s ({_sl_T0_seconds(opts) / 86400.0:.3g} d)")
    _row("Sampler",           opts.sampler)
    _row("Seed",              opts.seed)
    _row("Output root",       opts.save_path)
    _row("JAX backend",       jax.default_backend())
    _end()


def _run_and_report_preflight(opts):
    """Run preflight, report it in a frame, and exit on failure (or after
    writing the JSON when ``--preflight_only``)."""
    _section("Preflight")
    preflight = run_lensing_preflight(opts)
    # One machine-greppable JSON summary line, kept verbatim inside the frame.
    print("  │  " + json.dumps(preflight["summary"], sort_keys=True))
    print("  │")
    for warning in preflight["warnings"]:
        _warn(warning)
    for error in preflight["errors"]:
        _err(error)
    if preflight["ok"]:
        _ok("Preflight passed.")
    if opts.preflight_only:
        out_path = opts.preflight_json or os.path.join(opts.save_path, "preflight.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preflight, f, indent=2, allow_nan=False)
            f.write("\n")
        _ok(f"preflight JSON  →  {out_path}")
        _end()
        raise SystemExit(0 if preflight["ok"] else 2)
    _end()
    if not preflight["ok"]:
        raise SystemExit(
            "preflight failed; fix input errors or run --preflight_only true for JSON details"
        )


def _prepare_run_dir(opts):
    """Create the run directory, split the fixed-parameter JSON, and write the
    pre-load settings.json.  Returns (run_dir, settings, fixed, base_fixed,
    lens_fixed, resume_dir) -- resume_dir is None for a fresh run."""
    # --resume continues inside the ORIGINAL run directory rather than opening a
    # new one per SLURM attempt; resolve_checkpoint_plan then points the sampler
    # at <run_dir>/checkpoint.<sampler>.* (see inference/checkpointing.py).
    prefix = _run_name_prefix(opts)
    resume_ckpt, resume_dir = find_resume_target(opts, opts.sampler, name_prefix=prefix)
    run_dir = resume_dir or _make_run_dir(opts)
    opts.run_dir = run_dir
    plan = resolve_checkpoint_plan(
        opts, run_dir, name_prefix=prefix, resume_from=resume_ckpt
    )
    if resume_ckpt:
        print(f"  [i] resuming run directory {run_dir}", flush=True)
    print(f"  [i] checkpointing: {plan.summary()}", flush=True)
    # A resumed run must NOT overwrite the original settings.json -- it is the
    # record of the configuration that created the checkpoint being restored.
    # Each resume attempt leaves its own timestamped settings file instead.
    # The fingerprint gate itself runs in main() once the sampled labels and
    # bounds exist (see _gate_or_stamp_resume_fingerprint).
    settings_basename = (
        "settings.json" if not resume_dir else
        "settings.resume-"
        + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        + ".json"
    )
    # Keep settings serialization equivalent to: for k, v in vars(opts).items()
    settings = _jsonable_settings(opts)
    fixed = {}
    base_fixed = {}
    lens_fixed = {}
    try:
        fixed = (
            json.loads(opts.fixed_parameter_values) if opts.fixed_parameter_values else {}
        )
        base_fixed, lens_fixed = _split_lensing_fixed_parameters(fixed)
        settings.update(
            fixed_parameter_values_raw=fixed,
            fixed_parameter_values_base=base_fixed,
            fixed_parameter_values_lens=lens_fixed,
        )
    except Exception as exc:
        settings.update(
            fixed_parameter_values_raw=opts.fixed_parameter_values,
            fixed_parameter_values_base={},
            fixed_parameter_values_lens={},
        )
        _write_json(os.path.join(run_dir, settings_basename), settings)
        _write_failure(
            run_dir, "build_parameter_space", exc, labels=[], settings=settings
        )
        raise
    _write_json(os.path.join(run_dir, settings_basename), settings)
    return run_dir, settings, fixed, base_fixed, lens_fixed, resume_dir


def _gate_or_stamp_resume_fingerprint(
    opts, run_dir, resume_dir, labels, lower, upper, prior_kinds, fixed
):
    """Fingerprint gate: sampler state must never be restored under a
    different statistical target (the P0-01 failure mode).

    Runs here rather than in ``_prepare_run_dir`` because the sampled labels
    and bounds only exist once ``_build_space_and_closures`` has returned.
    That ordering is safe: a checkpoint file cannot exist before sampling
    starts, so no restorable state ever predates the fresh run's stamp.
    """
    fingerprint = build_run_fingerprint(
        opts,
        labels=labels,
        lower_bound=lower,
        upper_bound=upper,
        prior_kinds=prior_kinds,
        prior_overrides=getattr(opts, "prior_overrides", None),
        fixed_parameter_values=fixed,
    )
    if resume_dir:
        try:
            stored = check_resume_fingerprint(
                run_dir, fingerprint,
                force=bool(getattr(opts, "resume_force", False)),
            )
        except ResumeFingerprintError as exc:
            raise SystemExit(str(exc)) from None
        if stored is None:
            # --resume_force accepted a fingerprint-less legacy run dir;
            # stamp it now so later requeues are validated, not forced.
            save_run_fingerprint(run_dir, fingerprint)
    else:
        save_run_fingerprint(run_dir, fingerprint)


def _load_and_report_inputs(opts, run_dir, settings):
    """Load the singleton/pair inputs inside a framed section and resolve the
    selection block size (post-load).  Returns the ``inp`` bundle."""
    _section("Loading data")
    print("  │", flush=True)
    try:
        inp = load_inputs(opts)
    except Exception as exc:
        _write_failure(run_dir, "load_inputs", exc, labels=[], settings=settings)
        raise

    if inp.get("observed_catalog_heuristic"):
        _warn("unified observed mode inferred by deprecated event-count heuristic")
    _ok(
        f"GW events:              {inp['nEvents']} "
        f"({inp['n_singletons']} singletons + {inp['n_pairs']} pairs)"
    )
    _ok(f"PE samples/event:       {inp['nsamp']}")
    _ok(f"Selection injections:   {int(inp['Ndraw']):,} total generated")
    _resolve_lensing_block_sizes(opts, inp, settings)
    _end()
    return inp


def _build_space_and_closures(opts, inp, run_dir, settings, base_fixed, lens_fixed):
    """Build the base+lens parameter space, the decoder, and the likelihood /
    diagnostics closures.  Returns (labels, lower, upper, prior_transform,
    loglike, diagnostics_fn, pop_params_fid, lens_overrides)."""
    labels = []
    try:
        overrides = json.loads(opts.prior_overrides) if opts.prior_overrides else {}
        lens_overrides = (
            json.loads(opts.lens_prior_overrides) if opts.lens_prior_overrides else {}
        )
        pop_params_fid = get_fixed_population_params(opts.pop_model)

        # the branch parses "true"/"false" strings into bools internally via opts;
        # build_parameter_space takes the raw opts.fix_* values it was given.
        #
        # universe_model GATES THE SAMPLED SURVEY BLOCK (the survey registry in
        # inference/prior.py).  Omitting it left the model unrecognised, which
        # samples the whole block, so --fix_survey false sampled SEVEN flat
        # survey nuisance dimensions (12 -> 19) that the decoder --
        # which DOES read opts.universe_model, i.e. spectral_sirens_wl -- then
        # discarded: _decode_base_parameters slices coord[:len(sampled_labels)]
        # and the survey block sits past that prefix, so those coordinates never
        # reached the likelihood.  Thread every flag build_parameter_decoder
        # re-derives the space from, using its own getattr defaults, so the two
        # derivations cannot diverge.
        space = build_parameter_space(
            opts.pop_model,
            opts.fix_population,
            opts.fix_cosmology,
            opts.fix_survey,
            prior_overrides=overrides,
            fixed_parameter_values=base_fixed,
            fix_de=getattr(opts, "fix_de", getattr(opts, "fixed_de", False)),
            universe_model=getattr(opts, "universe_model", None),
            use_lss=bool(getattr(opts, "use_LSS", True)),
        )
        base_labels = list(space[0])
        base_lower = np.asarray(space[1])
        base_upper = np.asarray(space[2])
        lens_labels, lens_lower, lens_upper = _build_lens_parameter_space(
            opts, lens_fixed, lens_overrides
        )
        labels = base_labels + lens_labels
        lower = np.concatenate([base_lower, lens_lower])
        upper = np.concatenate([base_upper, lens_upper])
        # PRIOR FAMILIES: build_parameter_space returns per-parameter
        # (kind, loc, scale) triples, and dropping them silently downgrades every
        # non-uniform prior to uniform -- make_prior_transform's documented
        # behaviour for prior_kinds=None is "the legacy all-uniform affine map".
        # That matters because the whitened GP latents (population gp_* models
        # and the GP sky models) declare prior_kind="normal", xi ~ N(0, 1): that
        # unit-normal prior IS the GP regularisation, so sampling it flat changes
        # the model, the posterior and the evidence. --pop_model is unrestricted
        # here, so a GP model is reachable. The lensing-only SIS optical-depth
        # parameters appended below are all uniform.
        base_prior_kinds = list(space[11])
        prior_kinds = base_prior_kinds + [("uniform", None, None)] * len(lens_labels)
        assert len(prior_kinds) == len(labels), (
            "prior_kinds must stay aligned with labels/lower/upper"
        )
        joint_constraints = resolve_joint_prior_constraints(
            opts.pop_model, labels, lower, upper, prior_kinds,
        )
        prior_transform = make_prior_transform(
            lower, upper, prior_kinds, joint_constraints=joint_constraints,
        )
        _section("Parameter Space")
        print(f"  │    {'Parameter':<24} {'Lower':>12}  {'Upper':>12}")
        print(f"  │    {'─' * 24} {'─' * 12}  {'─' * 12}")
        for _lbl, _lo, _hi in zip(labels, lower, upper):
            print(f"  │    {_lbl:<24} {_lo:>12.4g}  {_hi:>12.4g}")
        print(f"  │    {'─' * 24} {'─' * 12}  {'─' * 12}")
        _ok(f"Parameter space built:  {len(labels)} free dimensions")
        _end()

        # decoder carries the WL params so decode() returns a survey with wl_params set
        opts.prior_overrides = overrides  # decoder reads getattr(opts,'prior_overrides')
        # Arm the P0.1 fail-fast net for this CLI too: record the BASE sampler
        # labels (the lens-only optical-depth block is appended afterwards and is
        # not part of the decoder's space) so build_parameter_decoder raises if it
        # re-derives a different set.  Previously set only in cli/inference.py, so
        # every lensing run sampled with the net disarmed.
        opts.expected_sampled_labels = tuple(base_labels)
        if opts.wl_backend == "tabulated":
            _wl_arrays = _load_wl_table_arrays(opts)
            wl_params = make_tabulated_wl_params(
                _wl_arrays["wl_z_grid"],
                _wl_arrays["wl_log_mu_grid"],
                _wl_arrays["wl_log_p_table"],
            )
        else:
            wl_params = make_lognormal_wl_params(a=opts.lensing_wl_a, b=opts.lensing_wl_b)
        decoder = build_parameter_decoder(
            opts,
            pop_params_fid,
            fixed_parameter_values=base_fixed,
            wl_params=wl_params,
        )

        loglike = build_cluster_likelihood(opts, inp, decoder, labels, lens_fixed)
        diagnostics_fn = build_cluster_diagnostics(opts, inp, decoder, labels, lens_fixed)
    except Exception as exc:
        _write_failure(run_dir, "build_parameter_space", exc, labels=labels, settings=settings)
        raise
    return (labels, lower, upper, prior_transform, loglike, diagnostics_fn,
            pop_params_fid, lens_overrides, prior_kinds)


def _cross_check_loglike_against_diagnostics(loglike, diagnostics, point, *, opts):
    """Pin the sampler-facing log-likelihood to the diagnostics total AT THE
    SAME POINT, and return the checked value.

    Under ``--partition_mode marginalize_exact --partition_component_mode
    componentwise`` (the default) the sampler path does NOT read the master
    likelihood's selection correction: it rebuilds the count-only term in
    closed form (``_count_correction_closed_form``) and the master's value
    cancels out of the factorized total.  Nothing tied the two
    together, so a closed form missing the total-variance budget produced a
    sampler likelihood ~1e4 nats away from every other evaluation path with no
    symptom at all (review F-001).  One extra likelihood call at startup —
    already-compiled shapes, so it is essentially free — makes that class of
    divergence loud.
    """
    reference = diagnostics.get("logL_marginalized", diagnostics.get("logL_total"))
    if reference is None:
        return None
    reference = float(reference)
    value = float(loglike(jnp.asarray(point)))
    # Both guarded to -inf is agreement; ``==`` (not isclose) so that case,
    # and an exact match, short-circuit before the finite comparison.
    if value == reference:
        _ok(f"logL(diagnostics point) = {value:.6g}  ==  diagnostics total")
        return value
    tol = 1e-6 + 1e-9 * abs(reference)
    if (
        not np.isfinite(value)
        or not np.isfinite(reference)
        or abs(value - reference) > tol
    ):
        raise RuntimeError(
            "sampler log-likelihood disagrees with the partition diagnostics at "
            f"the SAME evaluation point: loglike={value!r} vs diagnostics "
            f"{'logL_marginalized' if 'logL_marginalized' in diagnostics else 'logL_total'}"
            f"={reference!r} (tol {tol:.3g}; --partition_mode "
            f"{getattr(opts, 'partition_mode', 'fixed')} "
            "--partition_component_mode "
            f"{getattr(opts, 'partition_component_mode', 'componentwise')}). "
            "Under componentwise factorization the sampler rebuilds the "
            "count-only selection correction in closed form while the "
            "diagnostics take it from the master likelihood; a mismatch means "
            "the two are no longer the same statistical target, so posterior "
            "and logZ would not correspond to the reported diagnostics."
        )
    _ok(
        f"logL(diagnostics point) = {value:.6g}  ==  diagnostics total "
        f"(|Δ| = {abs(value - reference):.3g})"
    )
    return value


def _smoke_test_likelihood(opts, run_dir, settings, labels, lower, upper,
                           loglike, diagnostics_fn, pop_params_fid):
    """Smoke-eval at the prior midpoint so JIT compile errors surface early,
    then evaluate and persist the factorization diagnostics.  Returns
    (mid, diagnostics, diagnostics_point, diagnostics_point_label) -- the point
    and its label travel with the numbers so the archives cannot mislabel
    them."""
    _section("Building likelihood")
    print("  │  JIT compiling at the prior midpoint...", flush=True)
    mid = 0.5 * (lower + upper)
    t = time.time()
    try:
        v = float(loglike(jnp.asarray(mid)))
    except Exception as exc:
        _write_failure(run_dir, "midpoint_loglike", exc, labels=labels, settings=settings)
        raise
    _write_json(os.path.join(run_dir, "midpoint.json"), {"labels": labels, "values": np.asarray(mid, dtype=float).tolist(), "loglike": v})
    _ok(f"logL(prior midpoint) = {v:.3f}  [compile {time.time() - t:.1f}s]")
    _end()
    try:
        fid_point = _fiducial_candidate_point(
            labels, mid, lower, upper, opts, pop_params_fid
        )
        diagnostics, diag_point, diag_label = _diagnostics_at_guard_clear_point(
            diagnostics_fn, mid, lower, upper, int(opts.seed), fiducial=fid_point
        )
        _write_json(
            os.path.join(run_dir, "midpoint_diagnostics.json"),
            {
                **diagnostics,
                "evaluation_point": np.asarray(diag_point, dtype=float).tolist(),
                "evaluation_point_label": diag_label,
            },
        )
    except Exception as exc:
        _write_failure(run_dir, "midpoint_diagnostics", exc, labels=labels, settings=settings)
        raise
    try:
        _cross_check_loglike_against_diagnostics(
            loglike, diagnostics, diag_point, opts=opts
        )
    except Exception as exc:
        _write_failure(
            run_dir, "loglike_diagnostics_cross_check", exc,
            labels=labels, settings=settings,
        )
        raise
    _print_diagnostics_summary(diagnostics)
    return mid, diagnostics, diag_point, diag_label


def _run_lensing_sampling(opts, run_dir, settings, labels, lower, upper,
                          prior_transform, loglike, prior_kinds=None):
    """Run the sampler inside a framed section and return its results."""
    _section(f"Sampling  [{opts.sampler.upper()}]")
    _row("ndim", len(labels))
    if opts.sampler in ("tinyns", "dynesty"):
        _row("live points", opts.nlive)
        _row("ΔlogZ stop",  opts.dlogz)
        _row("max samples", opts.max_samples)
    if opts.sampler == "numpyro":
        _row("warmup",  opts.nuts_warmup)
        _row("samples", opts.nuts_samples)
        _row("chains",  opts.nuts_chains)
    _row("seed", opts.seed)
    if opts.sampler == "tinyns":
        cfg = opts.tinyns_resolved_config
        for key in TINYNS_RESOLVED_DISPLAY_KEYS:
            _row(f"  {key}", cfg[key])
        _row("  redshift_prior_barrier", opts.redshift_prior_barrier_resolved)
    print("  │", flush=True)
    t_sample_start = datetime.datetime.now()
    try:
        results = run_sampler(
            method=opts.sampler,
            likelihood=loglike,
            prior_transform=prior_transform,
            labels=labels,
            lower_bound=lower,
            upper_bound=upper,
            opts=opts,
            # numpyro builds its own prior from these rather than using
            # prior_transform, so they must be threaded here too or the NUTS
            # path keeps sampling GP latents uniform.
            prior_kinds=prior_kinds,
        )
    except Exception as exc:
        _write_failure(run_dir, "sampler", exc, labels=labels, settings=settings)
        raise
    wall_sampling = datetime.datetime.now() - t_sample_start
    print("  │")
    _ok(f"Sampling complete.  Wall time: {wall_sampling}")
    if results.get("logZ") is not None:
        _zerr = results.get("logZerr")
        _ok(
            f"log Z = {float(results['logZ']):.3f}"
            + (f" ± {float(_zerr):.3f}" if _zerr is not None else "")
        )
    _end()
    return results


def _save_lensing_outputs(opts, run_dir, settings, inp, results, diagnostics,
                          labels, mid, fixed, base_fixed, lens_fixed,
                          lens_overrides, *, diagnostics_point=None,
                          diagnostics_point_label="prior_midpoint"):
    """Persist samples/results/settings/diagnostics and the best-effort corner
    plot inside a framed section."""
    _section("Saving outputs")
    _row("Run directory", run_dir)
    print("  │")
    try:
        samples = np.asarray(results["samples"])
        np.save(os.path.join(run_dir, "samples.npy"), samples)
        with h5py.File(os.path.join(run_dir, "results.hdf5"), "w") as f:
            f.create_dataset("samples", data=samples)
            # Nested-sampler dead-point record (logl_dead/logwt_dead + n_dead /
            # n_live attrs), additive and indexed by dead point rather than by
            # posterior sample.  The lensing paper's central quantity is a
            # logZ difference, so the raw evidence record is worth archiving:
            # it is what an evidence bootstrap or a runplot has to be rebuilt
            # from.  Mirrors io.results.save_results_hdf5.
            write_dead_point_datasets(f, results)
            f.attrs["labels"] = json.dumps(labels)
            f.attrs["fixed_parameter_values_raw"] = json.dumps(fixed, default=str)
            f.attrs["fixed_parameter_values_base"] = json.dumps(base_fixed, default=str)
            f.attrs["fixed_parameter_values_lens"] = json.dumps(lens_fixed, default=str)
            f.attrs["wl_selection"] = opts.wl_selection
            f.attrs["pair_tag_model"] = opts.pair_tag_model
            f.attrs["pair_tag_constant"] = float(opts.pair_tag_constant)
            f.attrs["pair_tag_perturb_logit"] = float(opts.pair_tag_perturb_logit)
            _write_result_partition_metadata(
                f.attrs, opts=opts, inp=inp, diagnostics=diagnostics,
                eval_point_label=diagnostics_point_label,
                eval_point=diagnostics_point,
            )
            f.attrs["wl_a"] = float(opts.lensing_wl_a)
            f.attrs["wl_b"] = float(opts.lensing_wl_b)
            lens_settings = _lens_settings_dict(mid, labels, lens_fixed, opts)
            _write_lens_settings_attrs(f.attrs, lens_settings)
            if results.get("logZ") is not None:
                f.attrs["logZ"] = float(results["logZ"])
                # logZerr existed only in the run.log, yet the lensing paper's
                # central quantity is logZ(j2) - logZ(off): without the error the
                # archived run directory cannot say whether a 2-3 nat difference
                # is significant.  Mirrors io.results.save_results_hdf5 -- NaN
                # rather than absent when the sampler reports no error, so the
                # attr is always readable.
                _zerr = results.get("logZerr")
                f.attrs["logZerr"] = (
                    float(_zerr) if _zerr is not None else float("nan")
                )
            write_tinyns_metadata(f.attrs, results, opts)
            # NUTS health metadata, mirroring io.results.save_results_hdf5 —
            # divergence counts previously existed only in stdout for lensing
            # runs (library review, lensing CLI finding 7).
            if results.get("numpyro_diagnostics") is not None:
                try:
                    f.attrs["numpyro_diagnostics"] = json.dumps(
                        results["numpyro_diagnostics"], default=str)
                except (TypeError, ValueError):
                    pass
            f.attrs["selection_neff_soft_guard"] = bool(
                getattr(opts, "selection_neff_soft_guard", False))
            f.attrs["max_likelihood_variance"] = float(
                getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
        settings.update(
            wl_a=float(opts.lensing_wl_a),
            wl_b=float(opts.lensing_wl_b),
            partition_mode=getattr(opts, "partition_mode", "fixed"),
            lens_prior_overrides=lens_overrides,
            fixed_parameter_values_raw=fixed,
            fixed_parameter_values_base=base_fixed,
            fixed_parameter_values_lens=lens_fixed,
            **_lens_settings_dict(mid, labels, lens_fixed, opts),
        )
        if results.get("logZ") is not None:
            # Evidence AND its error in settings.json too, so a Bayes factor can
            # be assembled from the run directories alone.  Both nested samplers
            # report an error; numpyro reports no logZ at all and skips this.
            settings["logZ"] = float(results["logZ"])
            settings["logZerr"] = (
                float(results["logZerr"])
                if results.get("logZerr") is not None
                else None
            )
        if inp.get("observed_catalog"):
            settings["observed_catalog_path"] = inp["observed_catalog"].get("path")
            settings["observed_catalog_format_version"] = inp["observed_catalog"].get(
                "format_version"
            ) or inp["observed_catalog"].get("pe_format_version")
        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
            # The partition diagnostics are evaluated ONCE, at a single
            # parameter point, so the parameter-dependent quantities
            # (expected/MAP pair counts, marginalized logL) are NOT posterior
            # expectations. Prefix them with that point and stamp its name,
            # mirroring results.hdf5 -- bare names let a downstream tool read a
            # pre-posterior number as the run's E[n_pairs], and a hard-coded
            # prior_midpoint_ prefix mislabels everything the guard-clear
            # fallback evaluated elsewhere (review F-006). Structural counts
            # (n_partitions, ...) are eval-point-independent and stay bare.
            _diag_prefix = _partition_diagnostics_key_prefix(diagnostics_point_label)
            settings.update({
                "partition_diagnostics_eval_point": str(
                    diagnostics_point_label or "prior_midpoint"
                ),
                "partition_diagnostics_eval_point_values": (
                    np.asarray(diagnostics_point, dtype=float).tolist()
                    if diagnostics_point is not None
                    else None
                ),
                f"{_diag_prefix}_expected_n_pairs": float(diagnostics["expected_n_pairs"]),
                f"{_diag_prefix}_map_n_pairs": int(diagnostics["map_partition"]["n_pairs"]),
                f"{_diag_prefix}_logL_marginalized": float(diagnostics["logL_marginalized"]),
            })
            settings.update(
                n_partitions=int(diagnostics["n_partitions"]),
                approximate_total_partitions=int(diagnostics.get("approximate_total_partitions", diagnostics["n_partitions"])),
                partition_component_mode=diagnostics.get("partition_component_mode"),
                factorized_exact=bool(diagnostics.get("factorized_exact", False)),
                global_partitions_enumerated=bool(diagnostics.get("global_partitions_enumerated", True)),
            )
        _write_json(os.path.join(run_dir, "settings.json"), settings)
        save_tinyns_diagnostics_json(results, run_dir, opts)
        _write_diagnostics(run_dir, diagnostics)
    except Exception as exc:
        _write_failure(run_dir, "save", exc, labels=labels, settings=settings)
        raise
    _ok(f"samples.npy    →  {os.path.join(run_dir, 'samples.npy')}")
    _ok(f"results.hdf5   →  {os.path.join(run_dir, 'results.hdf5')}")
    _ok(f"settings.json  →  {os.path.join(run_dir, 'settings.json')}")
    _ok(f"diagnostics    →  {os.path.join(run_dir, 'diagnostics.json')} (+ .hdf5)")
    _ok(f"Posterior samples:  {samples.shape[0]:,}")

    # corner (best-effort)
    try:
        from darksirens.utils.plotting import make_production_corner

        fig = make_production_corner(samples, labels)
        fig.savefig(os.path.join(run_dir, "corner.pdf"), bbox_inches="tight", dpi=200)
        _ok(f"corner.pdf     →  {os.path.join(run_dir, 'corner.pdf')}")
    except Exception as e:
        _warn(f"Corner plot skipped: {e}")
    _end()


def main(argv=None):
    t_start = datetime.datetime.now()
    parser = build_parser()
    opts = parser.parse_args(argv)
    _validate_json_options(opts)
    os.makedirs(opts.save_path, exist_ok=True)

    print()
    _banner(f"DARK SIRENS LENSING  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    _resolve_lensing_run_config(opts)
    # Resolve the GW-population normalisation grids and size the OPT-IN pairing
    # m1 grid to --pop_model's primary-mass support.  This stack does evaluate
    # the interpolated PairingModel branch -- the setting is process-global and
    # read from the environment -- so skipping the guard let a lensing run with
    # DARKSIRENS_GW_PAIRING_M1_GRID set silently clamp the q-normaliser above
    # m1 = 200 for models whose support reaches 300 (PHY-5).
    configure_normalization_grids_for_model(opts)
    # Log the fully-resolved CLI options in argparse group order, including the
    # resolved normalization grid (part of the run's provenance).
    _print_all_cli_options(
        parser, opts, normalization_grid=normalization_grid_settings().to_dict()
    )
    _print_run_configuration(opts)
    _run_and_report_preflight(opts)
    run_dir, settings, fixed, base_fixed, lens_fixed, resume_dir = (
        _prepare_run_dir(opts)
    )
    inp = _load_and_report_inputs(opts, run_dir, settings)
    (labels, lower, upper, prior_transform, loglike, diagnostics_fn,
     pop_params_fid, lens_overrides, prior_kinds) = _build_space_and_closures(
        opts, inp, run_dir, settings, base_fixed, lens_fixed)
    _gate_or_stamp_resume_fingerprint(
        opts, run_dir, resume_dir, labels, lower, upper, prior_kinds, fixed
    )
    mid, diagnostics, diag_point, diag_label = _smoke_test_likelihood(
        opts, run_dir, settings, labels, lower, upper, loglike,
        diagnostics_fn, pop_params_fid)
    results = _run_lensing_sampling(
        opts, run_dir, settings, labels, lower, upper, prior_transform, loglike,
        prior_kinds=prior_kinds)
    _save_lensing_outputs(
        opts, run_dir, settings, inp, results, diagnostics, labels, mid,
        fixed, base_fixed, lens_fixed, lens_overrides,
        diagnostics_point=diag_point, diagnostics_point_label=diag_label)

    print()
    _banner(f"DONE  │  total wall time {datetime.datetime.now() - t_start}")
    print()


def console_main():
    """``darksirens_inference_lensing`` console-script target.

    Routes the installed script through ``run_cli`` so the console-script path
    gets the same GPU teardown guard as
    ``python -m darksirens.cli.inference_lensing``.
    """
    run_cli(main)


if __name__ == "__main__":
    run_cli(main)
