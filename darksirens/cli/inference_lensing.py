#!/usr/bin/env python3
"""
darksirens_inference_lensing.py
===============================
Full hierarchical inference CLI for the darksirens **strong-lensing branch**,
covering the joint singleton + J=2 cluster (pair) likelihood.

The stock ``darksirens_inference`` runs the WL singleton channel
(``--universe_model spectral_sirens_wl``) but has no flags for the cluster
channel. This tool adds:

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
* WL: ``--wl_backend {lognormal,disabled}`` with ``--lensing_wl_a/b`` and optional singleton-selection matching via ``--wl_selection``.

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

configure_jax_runtime(mem_fraction="0.90")

import os
import sys
import json
import time
import argparse
import datetime
import traceback

import numpy as np
import h5py
import healpy as hp
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp

# ── branch machinery we reuse ────────────────────────────────────────────────
from darksirens.redshift import zgrid
from darksirens.core.types import (
    CosmoParams,
    SurveyParams,
    EMCatalog,
    GWEvent,
)
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.gw.populations.registry import get_fixed_population_params, get_model

from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler
from darksirens.inference.tinyns_config import add_tinyns_arguments, build_tinyns_config
from darksirens.likelihood.factory import (
    _redshift_prior_materialization_reason,
    _resolve_redshift_prior_materialization,
)
from darksirens.inference.parameters import (
    build_parameter_decoder,
    complete_empty_pixel_policy_code,
)
from darksirens.gw.samples import load_gw_samples, load_selection_samples

# ── lensing / cluster pieces ─────────────────────────────────────────────────
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.lensing.wlmagnification import make_lognormal_wl_params
from darksirens.lensing.lensed_injections import load_lensed_injections
from darksirens.lensing.pair_tag_selection import make_pair_tag_selection_model, load_pair_tag_selection_file
from darksirens.lensing.observed_catalog import (
    observed_catalog_metadata_from_hdf5,
    validate_observed_catalog_file,
)
from darksirens.lensing.partitions import (
    exact_partitions_from_pairs,
    prepare_candidate_pairs_for_partitioning,
    parse_edge_mark_keys,
    validate_candidate_pairs,
)
from darksirens.lensing.preflight import run_lensing_preflight
from darksirens.lensing.marginal_diagnostics import (
    compute_marginalized_partition_diagnostics,
)
from darksirens.likelihood.pair_kde import (
    make_pair_kde,
    stack_pair_kdes,
    validate_pair_prior_wt,
)
from darksirens.likelihood.likelihood_with_clusters import (
    darksiren_log_likelihood_with_clusters,
    darksiren_likelihood_diagnostics_with_clusters,
    CLUSTER_MODE_J2,
    CLUSTER_MODE_OFF,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_DISABLED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
    PAIR_MARKS_NONE,
    PAIR_MARKS_TIME,
)


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
        attr = "delta_t_obs" if name in ("delta_t_obs", "true_delta_t") else name
        arr = np.asarray(getattr(lensed, attr))
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


def _decode_lens_params(coord, sampled_labels, fixed_parameter_values, opts):
    """Construct ``SISLensParams`` from fixed CLI values or sampled lens coords."""
    if getattr(opts, "fix_lens_rate", True):
        return make_sis_lens_params(A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n)

    values = {label: jnp.asarray(coord)[i] for i, label in enumerate(sampled_labels)}
    values.update(
        {label: float(value) for label, value in fixed_parameter_values.items()}
    )
    log10_tau_A = values.get("log10_tau_A", np.log10(float(opts.sl_tau_A)))
    tau_n = values.get("tau_n", float(opts.sl_tau_n))
    return make_sis_lens_params(A_tau=10.0**log10_tau_A, n_tau=tau_n)


def _lens_settings_dict(coord, sampled_labels, fixed_parameter_values, opts):
    sis = _decode_lens_params(coord, sampled_labels, fixed_parameter_values, opts)
    return dict(
        lens_labels=[
            label for label in sampled_labels if label in LENS_PARAMETER_PRIORS
        ],
        fix_lens_rate=bool(getattr(opts, "fix_lens_rate", True)),
        sl_tau_A=float(opts.sl_tau_A),
        sl_tau_n=float(opts.sl_tau_n),
        lens_A_tau=float(np.asarray(sis.A_tau)),
        lens_n_tau=float(np.asarray(sis.n_tau)),
    )


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


def load_inputs(opts):
    """Load singleton PE + selection, and (for j2) the lensed injections + pair
    PE + partition. Returns the assembled inputs for the cluster likelihood."""
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
    if getattr(opts, "observed_catalog_path", None):
        observed_catalog_meta = validate_observed_catalog_file(
            opts.observed_catalog_path
        )
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
            observed_catalog=observed_catalog_meta,
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
                    pair_time_delta_t_obs.append(
                        float(
                            g.attrs["delta_t_obs"]
                            if "delta_t_obs" in g.attrs
                            else np.asarray(g["delta_t_obs"])[()]
                        )
                    )
                    if "sigma_delta_t" in g.attrs:
                        pair_time_sigma.append(float(g.attrs["sigma_delta_t"]))
                    elif "sigma_delta_t" in g:
                        pair_time_sigma.append(
                            float(np.asarray(g["sigma_delta_t"])[()])
                        )
                    elif opts.pair_time_sigma_sec is not None:
                        pair_time_sigma.append(float(opts.pair_time_sigma_sec))
                    else:
                        raise SystemExit(
                            f"pair_marks=time requires sigma_delta_t in pair_{k} or --pair_time_sigma_sec"
                        )
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
                        if opts.pe_max_per_pair > 0:
                            d = _downsample(d, opts.pe_max_per_pair, rng)
                        d["prior_wt"] = _normalize_pair_image_prior_wt(
                            d["prior_wt"],
                            context=f"{pair_file_path}: pair_{k}/{name}/prior_wt",
                        )
                        imgs.append(d)
                    pairs.append(imgs)
                pair_metadata_indices.append(meta_pair)
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
            pair_time_delta_t_obs.append(float(meta.delta_t_obs))
            pair_time_sigma.append(float(meta.sigma_delta_t))

    if (
        unified_observed_catalog
        and partition is not None
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
        _candidate_n_events2, partition_states, log_z_prior = (
            exact_partitions_from_pairs(
                candidate_n_events,
                candidate_pairs,
                max_partitions=getattr(opts, "max_exact_partitions", 10000),
                component_mode=getattr(
                    opts, "partition_component_mode", "componentwise"
                ),
                max_component_events=getattr(opts, "max_component_events", None),
                max_component_edges=getattr(opts, "max_component_edges", None),
                max_component_partitions=getattr(
                    opts, "max_component_partitions", None
                ),
                max_total_partitions=getattr(opts, "max_total_partitions", None),
            )
        )
        if candidate_n_events != n_events_total:
            raise SystemExit(
                f"candidate_pairs n_events={candidate_n_events} does not match loaded event count "
                f"{n_events_total}"
            )
        marginal_parts = []
        for state in partition_states:
            part = dict(
                singleton_indices=jnp.asarray(state.singleton_indices, dtype=jnp.int32),
                pair_indices=jnp.asarray(state.pair_indices, dtype=jnp.int32),
                n_singletons=state.n_singletons,
                n_pairs=state.n_pairs,
                log_prior_weight=state.log_prior_weight,
                candidate_edge_indices=jnp.asarray(
                    state.candidate_edge_indices, dtype=jnp.int32
                ),
            )
            if getattr(opts, "pair_marks", "none") == "time":
                dt_obs, dt_sigma = _time_mark_arrays_for_partition_state(
                    state, candidate_pairs
                )
                part["pair_time_delta_t_obs"] = dt_obs
                part["pair_time_sigma"] = dt_sigma
            marginal_parts.append(part)
        marginal_partitions = tuple(marginal_parts)
        fixed_singletons = marginal_partitions[0]["singleton_indices"]
        fixed_pairs = marginal_partitions[0]["pair_indices"]
        fixed_n_singletons = marginal_partitions[0]["n_singletons"]
        fixed_n_pairs = marginal_partitions[0]["n_pairs"]
    else:
        marginal_partitions = None
        log_z_prior = 0.0
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
        observed_catalog=observed_catalog_meta,
        observed_catalog_heuristic=bool(
            heuristic_unified_observed_catalog and not explicit_unified_observed_catalog
        ),
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
            else:
                f.attrs[key] = value


def _make_run_dir(opts):
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(
        opts.save_path, f"{opts.pop_model}__{opts.cluster_mode}__{opts.sampler}__{ts}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _jsonable_settings(opts):
    return {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in vars(opts).items()
    }


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


def _write_result_partition_metadata(attrs, *, opts, inp, diagnostics):
    """Write unambiguous partition-count metadata to ``results.hdf5`` attrs."""
    partition_mode = str(getattr(opts, "partition_mode", "fixed"))
    attrs["partition_mode"] = partition_mode
    attrs["cluster_mode"] = opts.cluster_mode
    attrs["wl_backend"] = opts.wl_backend
    attrs["wl_selection"] = opts.wl_selection
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
        attrs["expected_n_singletons"] = float(diagnostics["expected_n_singletons"])
        attrs["expected_n_pairs"] = float(diagnostics["expected_n_pairs"])
        attrs["map_partition_index"] = int(diagnostics["map_partition_index"])
        map_partition = diagnostics["map_partition"]
        attrs["map_partition_n_singletons"] = int(map_partition["n_singletons"])
        attrs["map_partition_n_pairs"] = int(map_partition["n_pairs"])
        attrs["logL_marginalized"] = float(diagnostics["logL_marginalized"])
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


def _print_diagnostics_summary(diagnostics):
    print("  likelihood diagnostics (prior midpoint):", flush=True)
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
            print(f"    {key}: {diagnostics[key]}", flush=True)
    if diagnostics.get("partition_mode") == "marginalize_exact":
        count_summary = (
            f"    expected_n_singletons: {diagnostics.get('expected_n_singletons')}  "
            f"expected_n_pairs: {diagnostics.get('expected_n_pairs')}  "
            f"map_n_pairs: {diagnostics.get('map_partition', {}).get('n_pairs')}  "
            f"n_partitions: {diagnostics.get('n_partitions')}  "
        )
    else:
        count_summary = (
            f"    n_singletons: {diagnostics.get('n_singletons')}  "
            f"n_pairs: {diagnostics.get('n_pairs')}  "
        )
    print(
        count_summary + f"pair_batch_size: {diagnostics.get('pair_batch_size')}  "
        f"y_nodes_pair: {diagnostics.get('y_nodes_pair')}  "
        f"pe_max_per_pair: {diagnostics.get('pe_max_per_pair')}  "
        f"pair_eval_shape: {diagnostics.get('approximate_pair_evaluation_shape')}  "
        f"cluster_mode: {diagnostics['cluster_mode']}  "
        f"wl_backend: {diagnostics['wl_backend']}",
        flush=True,
    )


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
    wl_backend = (
        WL_BACKEND_LOGNORMAL if opts.wl_backend == "lognormal" else WL_BACKEND_DISABLED
    )
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if opts.wl_selection == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )
    pair_marks = (
        PAIR_MARKS_TIME
        if getattr(opts, "pair_marks", "none") == "time"
        else PAIR_MARKS_NONE
    )
    universe_model = opts.universe_model

    log_p_tag = _pair_tag_log_probs_from_options(opts, inp["lensed"])

    def loglike(coord):
        # decode() returns a 5-tuple (cosmo, survey, pop, sky, mark) on current
        # master; the cluster likelihood is WL-only (sky/mark unused here).
        cosmo, survey, pop_params, _sky_params, _mark_params = _decode_base_parameters(
            decoder, coord
        )

        def _eval_partition(part):
            return darksiren_log_likelihood_with_clusters(
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
                _decode_lens_params(
                    coord, lens_sampled_labels, fixed_parameter_values, opts
                ),
                log_p_tag,
                opts.pop_model,
                universe_model,
                sel_batch_size=opts.sel_batch_size,
                cluster_mode=cluster_mode,
                wl_backend=wl_backend,
                wl_a=opts.lensing_wl_a,
                wl_b=opts.lensing_wl_b,
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
                pair_batch_size=getattr(opts, "pair_batch_size", 0),
                y_nodes_pair=getattr(opts, "y_nodes_pair", 32),
            )

        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
            terms = [
                part["log_prior_weight"] + _eval_partition(part)
                for part in inp["marginal_partitions"]
            ]
            return jax_logsumexp(jnp.stack(terms)) - inp["log_z_prior"]

        return _eval_partition(inp)

    return loglike


def build_cluster_diagnostics(
    opts, inp, decoder, lens_sampled_labels=None, fixed_parameter_values=None
):
    em = _empty_em_catalog()
    lens_sampled_labels = list(lens_sampled_labels or [])
    fixed_parameter_values = dict(fixed_parameter_values or {})
    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF
    wl_backend = (
        WL_BACKEND_LOGNORMAL if opts.wl_backend == "lognormal" else WL_BACKEND_DISABLED
    )
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if opts.wl_selection == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )
    pair_marks = (
        PAIR_MARKS_TIME
        if getattr(opts, "pair_marks", "none") == "time"
        else PAIR_MARKS_NONE
    )
    universe_model = opts.universe_model
    log_p_tag = _pair_tag_log_probs_from_options(opts, inp["lensed"])

    def diagnostics(coord):
        coord = jnp.asarray(coord)
        cosmo, survey, pop_params, _sky_params, _mark_params = _decode_base_parameters(
            decoder, coord
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
                _decode_lens_params(
                    coord, lens_sampled_labels, fixed_parameter_values, opts
                ),
                log_p_tag,
                opts.pop_model,
                universe_model,
                sel_batch_size=opts.sel_batch_size,
                cluster_mode=cluster_mode,
                wl_backend=wl_backend,
                wl_a=opts.lensing_wl_a,
                wl_b=opts.lensing_wl_b,
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
                pair_batch_size=getattr(opts, "pair_batch_size", 0),
                y_nodes_pair=getattr(opts, "y_nodes_pair", 32),
            )

        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
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

            out = compute_marginalized_partition_diagnostics(
                inp["partition_states"],
                inp["candidate_pairs"],
                _part_loglike,
                log_z_partition_prior=inp["log_z_prior"],
                raw_candidate_pairs=inp.get("candidate_pairs_raw"),
                edge_mark_prior_contributions=inp.get("edge_mark_prior_contributions"),
            )
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
            **_lens_settings_dict(
                coord, lens_sampled_labels, fixed_parameter_values, opts
            ),
        )
        return out

    return diagnostics


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="darksirens lensing inference (singleton + J=2 cluster)"
    )
    # data
    p.add_argument("--gw_path", required=True)
    p.add_argument("--gwselection_path", required=True)
    p.add_argument("--lensed_injections_path", default=None)
    p.add_argument("--pair_pe_path", default=None)
    p.add_argument(
        "--pair_metadata_path",
        default=None,
        help="Optional pair/candidate-edge metadata file. Preferred over --pair_pe_path in unified observed mode.",
    )
    p.add_argument("--partition_path", default=None)
    p.add_argument("--candidate_pairs_path", default=None)
    p.add_argument(
        "--observed_catalog_path",
        default=None,
        help="Explicit observed_catalog.json for unified observed lensing mode.",
    )
    p.add_argument(
        "--partition_mode", choices=["fixed", "marginalize_exact"], default="fixed"
    )
    p.add_argument("--max_exact_partitions", type=int, default=10000)
    p.add_argument(
        "--partition_component_mode",
        choices=["global", "componentwise"],
        default="componentwise",
    )
    p.add_argument("--max_component_events", type=int, default=None)
    p.add_argument("--max_component_edges", type=int, default=None)
    p.add_argument("--max_component_partitions", type=int, default=None)
    p.add_argument("--max_total_partitions", type=int, default=None)
    # model
    p.add_argument("--pop_model", default="powerlaw+peak")
    p.add_argument("--cluster_mode", choices=["off", "j2"], default="j2")
    p.add_argument(
        "--wl_backend", choices=["lognormal", "disabled"], default="lognormal"
    )
    p.add_argument(
        "--wl_selection",
        choices=["standard", "wl_lognormal"],
        default="standard",
        help="Singleton selection treatment. standard preserves legacy selection; wl_lognormal uses lognormal/Hermite WL marginalization for singleton injections when wl_backend=lognormal (wl_a=0 reduces to standard).",
    )
    p.add_argument("--lensing_wl_a", type=float, default=4e-3)
    p.add_argument("--lensing_wl_b", type=float, default=1.5)
    p.add_argument("--sl_tau_A", type=float, default=5e-4)
    p.add_argument("--sl_tau_n", type=float, default=3.0)
    p.add_argument(
        "--fix_lens_rate",
        default="true",
        help="true fixes SIS optical-depth parameters to --sl_tau_A/--sl_tau_n; false samples lensing hyperparameters",
    )
    p.add_argument(
        "--lens_prior_overrides",
        default=None,
        help='JSON dict of SIS lens prior overrides, e.g. {"log10_tau_A": [-6, -3]}',
    )
    p.add_argument(
        "--pair_marks",
        choices=["none", "time"],
        default="none",
        help="Optional J=2 pair marks. 'time' uses candidate_pairs.json marks in marginalized mode or pair metadata in fixed mode.",
    )
    p.add_argument(
        "--pair_time_sigma_sec",
        type=float,
        default=None,
        help="Fallback sigma_delta_t in seconds when pair time metadata omits sigma.",
    )
    p.add_argument("--pair_tag_model", choices=["constant", "snr_time", "snr_time_sky", "file"], default="constant")
    p.add_argument("--pair_tag_constant", type=float, default=1.0)
    p.add_argument("--pair_tag_perturb_logit", type=float, default=0.0)
    p.add_argument("--pair_tag_selection_path", default=None)
    p.add_argument(
        "--edge_mark_prior_keys",
        default="",
        help="Comma-separated log_* candidate edge marks to add to edge log_prior_odds in exact marginalization.",
    )
    p.add_argument(
        "--edge_mark_likelihood_keys",
        default="",
        help="Comma-separated edge mark likelihood keys. Only time/delta_t_obs is implemented in this PR.",
    )
    # fixing
    p.add_argument("--fix_cosmology", default="true")
    p.add_argument("--fix_survey", default="true")
    p.add_argument("--fix_population", default="false")
    p.add_argument(
        "--fixed_parameter_values", default=None, help="JSON dict of {label: value}"
    )
    p.add_argument(
        "--prior_overrides", default=None, help="JSON dict of {label: [lo, hi]}"
    )
    p.add_argument(
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
    # sampler
    p.add_argument("--sampler", required=True, choices=["tinyns", "dynesty", "numpyro"])
    p.add_argument("--nlive", type=int, default=2000)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=2_000_000)
    add_tinyns_arguments(p)
    p.add_argument("--nuts_warmup", type=int, default=500)
    p.add_argument("--nuts_samples", type=int, default=2000)
    p.add_argument("--nuts_chains", type=int, default=4)
    # perf / memory
    p.add_argument(
        "--pe_max_per_pair",
        type=int,
        default=400,
        help="down-sample PE per pair image (0=keep all). Controls "
        "the O(N_pe^2 N_y) pair-KDE memory.",
    )
    p.add_argument(
        "--pair_batch_size",
        type=int,
        default=0,
        help="candidate-pair batch size for J=2 likelihood scans (0 keeps legacy unbatched path)",
    )
    p.add_argument(
        "--y_nodes_pair",
        type=int,
        default=32,
        help="Gauss-Legendre y nodes for each J=2 pair likelihood",
    )
    p.add_argument("--sel_batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show_progress", action="store_true")
    p.add_argument("--save_path", default="./")
    p.add_argument(
        "--preflight_only",
        default="false",
        help="true runs lensing input preflight checks, writes JSON, and exits before compilation/sampling",
    )
    p.add_argument(
        "--preflight_json",
        default=None,
        help="optional output path for preflight JSON (defaults to save_path/preflight.json when --preflight_only true)",
    )
    return p


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def main():
    opts = build_parser().parse_args()
    # build_parameter_space expects real bools (it does `if not fix_population:`)
    opts.fix_cosmology = _str2bool(opts.fix_cosmology)
    opts.fix_survey = _str2bool(opts.fix_survey)
    opts.fix_population = _str2bool(opts.fix_population)
    opts.fix_lens_rate = _str2bool(opts.fix_lens_rate)
    opts.preflight_only = _str2bool(opts.preflight_only)
    os.makedirs(opts.save_path, exist_ok=True)
    if opts.sampler == "tinyns":
        build_tinyns_config(opts)
    opts.materialize_redshift_prior_state = _resolve_redshift_prior_materialization(
        opts
    )
    opts.redshift_prior_barrier_resolved = _redshift_prior_materialization_reason(
        opts, opts.materialize_redshift_prior_state
    )
    if (
        opts.redshift_prior_barrier == "auto"
        and not opts.materialize_redshift_prior_state
    ):
        print(
            "  [i] Disabling likelihood-internal redshift-prior optimization_barrier "
            "for TinyNS JAX rwalk vmappability.",
            flush=True,
        )

    unsupported_edge_like = [
        k
        for k in parse_edge_mark_keys(opts.edge_mark_likelihood_keys)
        if k not in ("time", "delta_t_obs")
    ]
    if unsupported_edge_like:
        raise NotImplementedError(
            "edge_mark_likelihood_keys only supports time/delta_t_obs in this PR; "
            f"unsupported keys: {unsupported_edge_like}"
        )
    if any(
        not k.startswith("log_")
        for k in parse_edge_mark_keys(opts.edge_mark_prior_keys)
    ):
        raise SystemExit("--edge_mark_prior_keys entries must be log_* marks")

    opts.universe_model = (
        "spectral_sirens_wl" if opts.wl_backend == "lognormal" else "spectral_sirens"
    )

    print(
        f"=== darksirens_inference_lensing  [{opts.cluster_mode} | wl={opts.wl_backend} | wl_selection={opts.wl_selection}] ==="
    )
    print(
        "  lensing hyperparameters: "
        f"cluster_mode={opts.cluster_mode}, wl_backend={opts.wl_backend}, wl_selection={opts.wl_selection}, pair_marks={opts.pair_marks}, pair_tag_model={opts.pair_tag_model}, pair_tag_perturb_logit={opts.pair_tag_perturb_logit}, edge_mark_prior_keys={opts.edge_mark_prior_keys}, "
        f"wl_a={opts.lensing_wl_a}, wl_b={opts.lensing_wl_b}, "
        f"fix_lens_rate={opts.fix_lens_rate}, sl_tau_A={opts.sl_tau_A}, sl_tau_n={opts.sl_tau_n}",
        flush=True,
    )

    preflight = run_lensing_preflight(opts)
    print(
        "preflight summary:",
        json.dumps(preflight["summary"], sort_keys=True),
        flush=True,
    )
    for warning in preflight["warnings"]:
        print(f"  [preflight warning] {warning}", flush=True)
    for error in preflight["errors"]:
        print(f"  [preflight error] {error}", flush=True)
    if opts.preflight_only:
        out_path = opts.preflight_json or os.path.join(opts.save_path, "preflight.json")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preflight, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"wrote preflight JSON: {out_path}", flush=True)
        raise SystemExit(0 if preflight["ok"] else 2)
    if not preflight["ok"]:
        raise SystemExit(
            "preflight failed; fix input errors or run --preflight_only true for JSON details"
        )

    run_dir = _make_run_dir(opts)
    settings = _jsonable_settings(opts)
    labels = []
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
        _write_json(os.path.join(run_dir, "settings.json"), settings)
        _write_failure(
            run_dir, "build_parameter_space", exc, labels=labels, settings=settings
        )
        raise
    _write_json(os.path.join(run_dir, "settings.json"), settings)

    print("loading data ...", flush=True)
    try:
        inp = load_inputs(opts)
    except Exception as exc:
        _write_failure(run_dir, "load_inputs", exc, labels=labels, settings=settings)
        raise
    if inp.get("observed_catalog_heuristic"):
        print(
            "  [warning] unified observed mode inferred by deprecated event-count heuristic",
            flush=True,
        )
    print(
        f"  events: {inp['nEvents']}  ({inp['n_singletons']} singletons "
        f"+ {inp['n_pairs']} pairs)  nsamp/event={inp['nsamp']}",
        flush=True,
    )

    # --- build parameter space + prior + decoder using branch machinery ---
    try:
        overrides = json.loads(opts.prior_overrides) if opts.prior_overrides else {}
        lens_overrides = (
            json.loads(opts.lens_prior_overrides) if opts.lens_prior_overrides else {}
        )
        pop_params_fid = get_fixed_population_params(opts.pop_model)

        # the branch parses "true"/"false" strings into bools internally via opts;
        # build_parameter_space takes the raw opts.fix_* values it was given.
        space = build_parameter_space(
            opts.pop_model,
            opts.fix_population,
            opts.fix_cosmology,
            opts.fix_survey,
            prior_overrides=overrides,
            fixed_parameter_values=base_fixed,
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
        prior_transform = make_prior_transform(lower, upper)
        print(f"  free parameters ({len(labels)}): {labels}", flush=True)

        # decoder carries the WL params so decode() returns a survey with wl_params set
        opts.prior_overrides = overrides  # decoder reads getattr(opts,'prior_overrides')
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

    # smoke eval at the prior midpoint so JIT compile errors surface early
    mid = 0.5 * (lower + upper)
    t = time.time()
    try:
        v = float(loglike(jnp.asarray(mid)))
    except Exception as exc:
        _write_failure(run_dir, "midpoint_loglike", exc, labels=labels, settings=settings)
        raise
    _write_json(os.path.join(run_dir, "midpoint.json"), {"labels": labels, "values": np.asarray(mid, dtype=float).tolist(), "loglike": v})
    print(
        f"  logL(prior midpoint) = {v:.3f}  [compile {time.time()-t:.1f}s]", flush=True
    )
    try:
        diagnostics = diagnostics_fn(jnp.asarray(mid))
        _write_json(os.path.join(run_dir, "midpoint_diagnostics.json"), diagnostics)
    except Exception as exc:
        _write_failure(run_dir, "midpoint_diagnostics", exc, labels=labels, settings=settings)
        raise
    _print_diagnostics_summary(diagnostics)

    # --- sample ---
    if opts.sampler == "tinyns":
        cfg = opts.tinyns_resolved_config
        print("  TinyNS resolved config:", flush=True)
        for key in [
            "preset",
            "sample",
            "kernel",
            "rwalk_proposal",
            "walks",
            "step_scale",
            "min_accepts",
            "replacement_chains",
            "max_attempts",
            "jax_block_size",
        ]:
            print(f"    {key}: {cfg[key]}", flush=True)
        print(f"    dlogz: {opts.dlogz}", flush=True)
        print(f"    nlive: {opts.nlive}", flush=True)
        print(f"    max_samples: {opts.max_samples}", flush=True)
        print(
            f"    redshift_prior_barrier: {opts.redshift_prior_barrier_resolved}",
            flush=True,
        )
    print(f"sampling with {opts.sampler} ...", flush=True)
    try:
        results = run_sampler(
            method=opts.sampler,
            likelihood=loglike,
            prior_transform=prior_transform,
            labels=labels,
            lower_bound=lower,
            upper_bound=upper,
            opts=opts,
        )
    except Exception as exc:
        _write_failure(run_dir, "sampler", exc, labels=labels, settings=settings)
        raise

    # --- save ---
    try:
        samples = np.asarray(results["samples"])
        np.save(os.path.join(run_dir, "samples.npy"), samples)
        with h5py.File(os.path.join(run_dir, "results.hdf5"), "w") as f:
            f.create_dataset("samples", data=samples)
            f.attrs["labels"] = json.dumps(labels)
            f.attrs["fixed_parameter_values_raw"] = json.dumps(fixed, default=str)
            f.attrs["fixed_parameter_values_base"] = json.dumps(base_fixed, default=str)
            f.attrs["fixed_parameter_values_lens"] = json.dumps(lens_fixed, default=str)
            f.attrs["wl_selection"] = opts.wl_selection
            f.attrs["pair_tag_model"] = opts.pair_tag_model
            f.attrs["pair_tag_constant"] = float(opts.pair_tag_constant)
            f.attrs["pair_tag_perturb_logit"] = float(opts.pair_tag_perturb_logit)
            _write_result_partition_metadata(
                f.attrs, opts=opts, inp=inp, diagnostics=diagnostics
            )
            f.attrs["wl_a"] = float(opts.lensing_wl_a)
            f.attrs["wl_b"] = float(opts.lensing_wl_b)
            lens_settings = _lens_settings_dict(mid, labels, lens_fixed, opts)
            f.attrs["lens_labels"] = json.dumps(lens_settings["lens_labels"])
            f.attrs["fix_lens_rate"] = bool(opts.fix_lens_rate)
            f.attrs["sl_tau_A"] = float(opts.sl_tau_A)
            f.attrs["sl_tau_n"] = float(opts.sl_tau_n)
            f.attrs["lens_A_tau"] = float(lens_settings["lens_A_tau"])
            f.attrs["lens_n_tau"] = float(lens_settings["lens_n_tau"])
            if results.get("logZ") is not None:
                f.attrs["logZ"] = float(results["logZ"])
            if getattr(opts, "tinyns_resolved_config", None) is not None:
                f.attrs["tinyns_resolved_config"] = json.dumps(
                    opts.tinyns_resolved_config, default=str
                )
            if results.get("tinyns_summary") is not None:
                f.attrs["tinyns_summary"] = json.dumps(
                    results["tinyns_summary"], default=str
                )
            if results.get("tinyns_diagnostics") is not None:
                f.attrs["tinyns_diagnostics"] = json.dumps(
                    results["tinyns_diagnostics"], default=str
                )
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
        if inp.get("observed_catalog"):
            settings["observed_catalog_path"] = inp["observed_catalog"].get("path")
            settings["observed_catalog_format_version"] = inp["observed_catalog"].get(
                "format_version"
            ) or inp["observed_catalog"].get("pe_format_version")
        if getattr(opts, "partition_mode", "fixed") == "marginalize_exact":
            settings.update(
                expected_n_pairs=float(diagnostics["expected_n_pairs"]),
                map_n_pairs=int(diagnostics["map_partition"]["n_pairs"]),
                n_partitions=int(diagnostics["n_partitions"]),
                logL_marginalized=float(diagnostics["logL_marginalized"]),
            )
        _write_json(os.path.join(run_dir, "settings.json"), settings)
        _write_diagnostics(run_dir, diagnostics)
    except Exception as exc:
        _write_failure(run_dir, "save", exc, labels=labels, settings=settings)
        raise
    print(f"saved {samples.shape[0]} samples -> {run_dir}", flush=True)

    # corner (best-effort)
    try:
        from darksirens.utils.plotting import make_production_corner

        fig = make_production_corner(samples, labels)
        fig.savefig(os.path.join(run_dir, "corner.pdf"), bbox_inches="tight", dpi=200)
        print("  corner.pdf written", flush=True)
    except Exception as e:
        print(f"  corner skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
