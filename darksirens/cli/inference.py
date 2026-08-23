#!/usr/bin/env python3
"""
darksirens_inference.py
-----------------------
Entry point for the dark-siren / spectral-siren hierarchical inference pipeline.

Usage examples
--------------
# Spectral sirens, dynesty, free cosmology + population:
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --sampler           dynesty \
    --pop_model         powerlaw+peak \
    --universe_model    spectral_sirens \
    --nlive             2000

# Dark sirens with galaxy catalog, fixed cosmology, tinyns (rwalk + jax kernel):
python darksirens_inference.py \
    --gw_path           gw_events.h5 \
    --gwselection_path  injections.h5 \
    --survey_path       catalog_nside64.h5 \
    --sampler           tinyns \
    --nlive             1000 \
    --tinyns_sample     rwalk \
    --tinyns_kernel     jax \
    --pop_model         brokenpowerlaw+2peaks \
    --universe_model    dark_sirens \
    --fix_cosmology     true \

# Fix individual parameters via JSON:
    --fixed_parameter_values '{"$v_1$": 0.1}'
    --prior_overrides        '{"H0": [60.0, 80.0]}'

Long runs / SLURM
-----------------
The run directory and its settings.json are created BEFORE sampling, and the
sampler checkpoints into it every --checkpoint_interval seconds (default 1800;
'off' disables).  Requeue the identical command with --resume auto to continue
inside the same run directory:

    --sampler dynesty --nlive 2000 --checkpoint_interval 1800 --resume auto

Run-directory layout:  settings.json, checkpoint.<sampler>.{pkl,npz},
samples.npy (written before results.hdf5), results.hdf5, corner.pdf,
latents.pdf, dynesty_diagnostics/, and failure.json if the run dies.
"""

import os

# ── JAX memory configuration (before any JAX import) ──────────────────────────
from darksirens.core.jax_config import configure_jax_runtime

configure_jax_runtime()

import sys
import json
import datetime
import math
import traceback
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import h5py

from argparse import ArgumentParser, ArgumentTypeError, RawDescriptionHelpFormatter

from darksirens.gw.populations import (
    FIDUCIAL_SETS,
    FIDUCIAL_SET_LEGACY,
    get_fixed_population_params,
    pop_model_prior_parser,
)
from darksirens.gw.populations.utils import normalization_grid_settings
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.inference.validation import validate_multitracer_run
from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.likelihood.factory import make_likelihood
from darksirens.likelihood.block_sizing import (
    block_size_arg,
    resolve_block_sizes,
    measure_static_state_bytes,
    estimate_pending_static_bytes,
    probe_device_memory_bytes,
    format_block_size_request,
    sampler_block_sizing_profile,
)
from darksirens.redshift.completion import build_pixel_kde_cache, completion_clip_diagnostics
from darksirens.redshift.grid import zMax as _REDSHIFT_GRID_ZMAX
from darksirens.core.types import (
    C_MODE_AGGREGATE_STRUCT,
    C_MODE_SELECTION_STRUCT,
    SELECTION_FAMILY_SCHECHTER_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.inference.checkpointing import (
    add_checkpoint_arguments,
    find_resume_target,
    parse_checkpoint_interval,
    resolve_checkpoint_plan,
)
from darksirens.inference.run_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    ResumeFingerprintError,
    build_run_fingerprint,
    check_resume_fingerprint,
    save_run_fingerprint,
)
from darksirens.inference.sampling import run_sampler
from darksirens.inference.tinyns_config import add_tinyns_arguments, build_tinyns_config
from darksirens.inference.prior import (
    build_parameter_space,
    make_prior_transform,
    resolve_joint_prior_constraints,
)
from darksirens.inference.q_provenance import check_lss_completion_provenance
from darksirens.core.constants import (
    H0_FID,
    OM0_FID,
    SURVEY_PARAMS_FID_BY_NAME,
    W0_FID,
    WA_FID,
)

warnings.simplefilter("ignore", FutureWarning)


# ── Imports from refactored CLI/IO modules ─────────────────────────────────────

from darksirens.cli.common import (
    _banner,
    _section,
    _row,
    _end,
    _ok,
    _warn,
    _fatal,
    _fixed_dark_energy_metadata,
    _format_fixed_dark_energy_summary,
    _print_all_cli_options,
    add_normalization_grid_arguments,
    configure_normalization_grids_for_model,
    str_to_bool,
    parse_json_arg,
    parse_counterpart_arg,
    DeprecatedSpellingAction,
    resolve_redshift_prior_barrier,
    resolve_selection_neff_guard,
    run_cli,
)
from darksirens.io.results import (
    _sky_log_prior_volume_correction,
    save_results_hdf5,
)
from darksirens.io.settings import save_settings_json


# ── Parameter table ────────────────────────────────────────────────────────────

def _print_parameter_table(
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    fixed_parameter_statuses: dict,
    fix_cosmology:          bool,
    fix_de:                 bool,
    fix_population:         bool,
    fix_survey:             bool,
    pop_params_fid:         np.ndarray,
    pop_labels_all:         list,
):
    """
    Print sampled parameters with bounds, individually-fixed params with their
    values, and block-fixed parameters with their fiducial values.
    """
    COSMO_FID  = {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID}
    DE_FID = {"w0": W0_FID, "wa": WA_FID}
    # The survey fiducials the decoder pins non-sampled SurveyParams fields at
    # (core.constants is their single source; this table used to restate them).
    SURVEY_FID = dict(SURVEY_PARAMS_FID_BY_NAME)
    pop_fid_map = {lbl: float(pop_params_fid[i])
                   for i, lbl in enumerate(pop_labels_all)}

    block_fixed_sections: list[tuple[str, dict[str, float]]] = []
    if fix_cosmology:
        block_fixed_sections.append(("cosmology", COSMO_FID))
    elif fix_de:
        block_fixed_sections.append(("dark energy", DE_FID))
    if fix_population:
        block_fixed_sections.append(("population", pop_fid_map))
    if fix_survey:
        block_fixed_sections.append(("survey", SURVEY_FID))

    _section("Parameter Space")
    rule = f"  │    {'─' * 24} {'─' * 12}  {'─' * 12}  {'─' * 20}"

    _row("Sampled parameters", "")
    print(f"  │    {'Parameter':<24} {'Lower':>12}  {'Upper':>12}  Status")
    print(rule)
    for label, lo, hi in zip(labels, lower_bound, upper_bound):
        note = "← overridden" if label in prior_overrides else ""
        print(f"  │    {label:<24} {lo:>12.4g}  {hi:>12.4g}  {note}")

    print(rule)
    _row("Individually fixed parameters", "")
    if fixed_parameter_values:
        for label, value in fixed_parameter_values.items():
            note = fixed_parameter_statuses.get(label, f"fixed = {value:.6g}")
            if label in prior_overrides:
                lo, hi = prior_overrides[label]
                print(
                    f"  │    {label:<24} {float(lo):>12.4g}  "
                    f"{float(hi):>12.4g}  {note}"
                )
            else:
                print(f"  │    {label:<24} {'—':>12}  {'—':>12}  {note}")
    else:
        print(f"  │    {'(none)':<24}")

    print(rule)
    _row("Block-fixed parameters", "")
    if block_fixed_sections:
        for block_name, fiducials in block_fixed_sections:
            print(f"  │    [{block_name}]")
            for label, value in fiducials.items():
                print(
                    f"  │    {label:<24} {'—':>12}  {'—':>12}  "
                    f"fixed = {value:.6g}  (block)"
                )
    else:
        print(f"  │    {'(none)':<24}")

    print(rule)

    n_free = sum(1 for lo, hi in zip(lower_bound, upper_bound) if lo != hi)
    n_fix_ind   = len(fixed_parameter_values)
    n_fix_block = ((len(COSMO_FID) if fix_cosmology else len(DE_FID) if fix_de else 0)
                   + (len(pop_labels_all) if fix_population else 0)
                   + (len(SURVEY_FID) if fix_survey else 0))
    _row("Free (sampled)",      n_free)
    if n_fix_ind:   _row("Fixed individually", n_fix_ind)
    if n_fix_block: _row("Fixed (block)",      n_fix_block)
    _row("Total in coord vec",  len(labels))
    _end()


def resolve_survey_z_depth(cli_override, file_attr, zmax=None):
    """Resolve one catalog's ``z_depth``: CLI override > per-catalog file attr > None.

    Precedence: ``cli_override`` (the ``--survey_z_depth`` value, applied to
    every catalog when set) wins over ``file_attr`` (the per-catalog
    ``f.attrs['z_depth']`` written by ``darksirens_pixelate --z_depth``);
    ``None`` means no depth prior -- completeness is estimated over the full
    grid (:mod:`darksirens.redshift.completion`).  A resolved value at/above the
    redshift grid ``zMax`` (:data:`darksirens.redshift.grid.zMax`) asserts full
    coverage of the grid, i.e. no depth prior, so it is clamped to ``zMax`` with
    a warning rather than silently accepted or rejected.
    """
    if zmax is None:
        zmax = _REDSHIFT_GRID_ZMAX
    z = cli_override if cli_override is not None else file_attr
    if z is None:
        return None
    z = float(z)
    if not math.isfinite(z) or z <= 0.0:
        # NaN slips through every comparison guard (NaN > zmax is False) and
        # would silently zero the whole missing-galaxy budget downstream
        # (depth_mask = zgrid <= NaN is all-False); 0/negative likewise.
        raise ValueError(
            f"survey z_depth must be a finite positive redshift; got {z!r}."
        )
    if z > zmax:
        msg = (
            f"z_depth={z:g} exceeds the redshift grid zMax={zmax:g}; clamping "
            "to zMax (equivalent to the legacy full-grid missing-galaxy budget)."
        )
        warnings.warn(msg)
        _warn(msg)
        z = float(zmax)
    return z


def _report_survey_z_depth(label: str, resolved) -> None:
    """Print one loud config line for a catalog's resolved ``z_depth``."""
    _ok(
        f"{label}: "
        + (
            f"{resolved:g} (completeness zero beyond z = {resolved:g}; "
            "hosts there are missing, not nonexistent)"
            if resolved is not None
            else "None -> completeness over the full grid (legacy)"
        )
    )


def _completion_validation_point(
    fiducials: dict,
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> dict[str, float]:
    """Representative dry-run point: fixed > prior-override midpoint > fiducial.

    Shared by the survey and the cosmology blocks so the diagnostic cannot sit at
    a fiducial the sampler never visits: the clip fractions are ratios of
    ``dN_obs`` to ``dN_exp = n0 apix exp(log_galaxy_measure_grid(cosmo, survey))``,
    i.e. cosmology-dependent through the comoving volume element, so a run with
    e.g. ``--prior_overrides '{"H0": [40, 55]}'`` carries ~2x the volume per unit
    z of Planck15 and can rail C where the fiducial point reports 0 clipping.
    """
    values = dict(fiducials)
    for label in values:
        if label in (prior_overrides or {}):
            lo, hi = prior_overrides[label]
            values[label] = 0.5 * (float(lo) + float(hi))
        if label in (fixed_parameter_values or {}):
            values[label] = float(fixed_parameter_values[label])
    return values


def _completion_validation_survey_values(
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> dict[str, float]:
    """Choose representative survey values for dry-run clipping diagnostics."""
    return _completion_validation_point(
        {
            "log10n0": -2.0,
            "z50": 1.0,
            "w": 0.5,
            "delta": 0.0,
            "b_miss": 1.0,
            "alpha_miss": 1.0,
            "sigma_kde": 0.0,
        },
        prior_overrides,
        fixed_parameter_values,
    )


#: Row axis of each Q_LSS table leaf carried on ``EMCatalog``.
_LSS_TABLE_ROW_AXIS = {
    "lss_completion_logq": 0,
    "lss_completion_q": 0,
    "lss_completion_logq_members": 1,
    "lss_completion_q_members": 1,
}


def _completion_validation_lss_tables(data: dict, unique_pixels, n_pix_catalog: int):
    """Q_LSS table leaves for the dry-run validation catalog, sliced to its rows.

    ``completion_clip_diagnostics`` resolves the missing-galaxy rate factor FROM
    THE CATALOG, so a validation catalog built without the loaded Q tables silently
    takes the legacy ``delta_g`` branch and reports the delta_g-negativity fraction
    (0 for the production dummy overdensity) instead of the ``+-7`` logQ clip
    fraction the diagnostic exists to surface -- the run then looks clean while the
    Q the likelihood will use is railed over much of the grid.

    The tables are sliced HOST-side to the validation rows (mirroring the factory's
    eager global->compact gather) so a full ``(M, n_pix, N_grid)`` ensemble never
    reaches the device.  Returns ``(kwargs, provenance, provenance_by_key)``;
    the provenance records why nothing was attached so the JSON never implies a
    clean Q when the Q was simply not readable.  Per leaf:

    * ``"global_table_sliced"`` -- attached, rows gathered by global pixel;
    * ``"compact_table_skipped"`` -- the table is a COMPACT per-view block
      (``lss_completion_indexing == 1``, or its row count is not the full nside),
      which carries no global pixel key, so the validation pixels cannot be
      aligned to it.

    The back-compatible scalar summarises those: ``"none"`` when the run carries
    no Q table at all (the legacy delta_g diagnostic is the right one), the
    common per-leaf value when every present leaf agrees, and ``"partial"`` for a
    MIXED outcome -- which a single last-write-wins scalar reported as whichever
    leaf happened to come last in ``_LSS_TABLE_ROW_AXIS`` order, so a reader
    could not tell which leaf the clip fraction came from.
    """
    pix = np.asarray(unique_pixels, dtype=np.int64).reshape(-1)
    indexing = int(data.get("lss_completion_indexing", 0) or 0)
    kwargs: dict[str, object] = {}
    by_key: dict[str, str] = {}
    for key, axis in _LSS_TABLE_ROW_AXIS.items():
        tab = data.get(key)
        if tab is None:
            continue
        arr = np.asarray(tab)
        if indexing == 1 or arr.shape[axis] != n_pix_catalog:
            by_key[key] = "compact_table_skipped"
            continue
        if n_pix_catalog != pix.size:
            arr = np.take(arr, pix, axis=axis)
        kwargs[key] = jnp.asarray(arr)
        by_key[key] = "global_table_sliced"
    outcomes = set(by_key.values())
    if not outcomes:
        provenance = "none"
    elif len(outcomes) == 1:
        provenance = outcomes.pop()
    else:
        provenance = "partial"
    return kwargs, provenance, by_key


def run_completion_validation(
    opts,
    data: dict,
    prior_overrides: dict,
    fixed_parameter_values: dict,
    suffix: str = "",
) -> str:
    """Save a dry-run completion clipping diagnostic and return its path.

    ``suffix`` distinguishes per-catalog outputs for a K>=2 mixture run
    (e.g. ``"_c2"`` -> ``completion_validation_c2__<ts>.json``).
    """
    required = ["zgals_catalog", "dzgals_catalog", "wgals_catalog", "ngals_catalog"]
    if any(data.get(key) is None for key in required):
        # Backward-compatible tests/callers may use unsuffixed catalog keys.
        fallback = {
            "zgals_catalog": "zgals",
            "dzgals_catalog": "dzgals",
            "wgals_catalog": "wgals",
        }
        for dst, src in fallback.items():
            if data.get(dst) is None and data.get(src) is not None:
                data[dst] = data[src]
    full_z = data.get("zgals_catalog")
    full_dz = data.get("dzgals_catalog")
    full_w = data.get("wgals_catalog")
    full_n = data.get("ngals_catalog")
    if any(value is None for value in (full_z, full_dz, full_w, full_n)):
        _fatal(
            "--validate_completion requires a loaded galaxy catalog with "
            "zgals/dzgals/wgals/ngals arrays."
        )

    pixels_pe = np.asarray(data["pixels_pe"], dtype=np.int32)
    pixels_sel = np.asarray(data["pixels_sel"], dtype=np.int32)
    unique_pixels = np.unique(np.concatenate([pixels_pe, pixels_sel])).astype(
        np.int32, copy=False
    )
    max_pixels = max(1, int(opts.completion_validation_pixels))
    unique_pixels = unique_pixels[:max_pixels]

    # The completion code indexes the KDE cache directly by row (cache row k ==
    # unique_pixels[k]); the dense pixel_to_cache_idx lookup is no longer used.
    dN_obs_kde, _ = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=full_z,
        n_pix_catalog=int(data.get("n_pix_catalog", np.asarray(full_z).shape[0])),
        wgals=full_w,
        ngals=full_n,
    )

    survey_values = _completion_validation_survey_values(
        prior_overrides, fixed_parameter_values
    )
    # Completion validation is a dry run, so the cosmology is a REPRESENTATIVE
    # point on the same fixed > prior-override midpoint > fiducial precedence as
    # the survey block above (the clip fractions are cosmology-dependent through
    # the comoving volume element, so pinning Planck15 under an H0 override
    # certified a completeness budget the run never forms).  W0_FID/WA_FID rather
    # than literals so the dry run cannot drift from the decoder.
    cosmology_values = _completion_validation_point(
        {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID},
        prior_overrides, fixed_parameter_values,
    )
    cosmo = CosmoParams(
        H0=float(cosmology_values["H0"]),
        Om0=float(cosmology_values["Om0"]),
        w0=float(cosmology_values["w0"]),
        wa=float(cosmology_values["wa"]),
    )
    # NOTE: this dry-run deliberately leaves z_depth at its default (None):
    # completion_clip_diagnostics reports clip fractions of the completeness
    # ratio only, which are z_depth-independent, so threading the resolved
    # depth here would change nothing. Do not "fix" this without a reason.
    # (With a Q_LSS table attached below, the reported logQ-clip fraction is the
    # WHOLE-grid one; a depth-bounded run relaxes Q_eff -> 1 beyond the depth, so
    # the number is an upper bound on the railing the likelihood will see.)
    #
    # c_mode-aware (deferred review MINOR): the dry run must diagnose the
    # SAME completeness estimator the likelihood will form.  Without the mode
    # (and, for c_mode=selection, the theta/family/K(z)/strata block) the
    # validation silently reported the legacy per-pixel ratio's clipping for
    # every run -- an aggregate/selection run then looked clean while its
    # actual budget was never inspected.  _precompute_grids is mode-generic;
    # only this SurveyParams needs to carry the structure.
    _c_mode = str(getattr(opts, "c_mode", None) or "per_pixel")
    _k_catalog = int(suffix[2:]) if suffix.startswith("_c") else 1
    _mode_kwargs: dict = {}
    if _c_mode == "aggregate":
        _mode_kwargs["c_mode"] = C_MODE_AGGREGATE_STRUCT
    elif _c_mode == "selection":
        # Resolve the per-catalog fit state through the SAME resolver the
        # run uses (idempotent: pins are setdefault, stamps overwrite), so
        # theta / family / K(z) / strata cannot drift from the likelihood's.
        if getattr(opts, "selection_fits", None) is None:
            _resolve_selection_fits(opts, data, fixed_parameter_values)
        _fits = opts.selection_fits or []
        _fit = _fits[_k_catalog - 1] if len(_fits) >= _k_catalog else None
        _family = str(getattr(opts, "selection_family", None) or "gaussian")
        _theta = dict(_fit["theta"]) if _fit else {}

        def _tval(name):
            # fixed value > fit's effective theta > registry fiducial: the
            # dry run sits where the likelihood's theta prior is centered
            # (theta is SAMPLED in the real run; this is a representative
            # point, exactly like the cosmology above).
            _sfx_key = f"{name}{suffix}" if suffix else name
            if _sfx_key in fixed_parameter_values:
                return float(fixed_parameter_values[_sfx_key])
            if name in _theta:
                return float(_theta[name])
            if name == "M_faint_offset" and \
                    "M_faint_offset" in fixed_parameter_values:
                return float(fixed_parameter_values["M_faint_offset"])
            return float(SURVEY_PARAMS_FID_BY_NAME[name])

        _mode_kwargs = dict(
            c_mode=C_MODE_SELECTION_STRUCT,
            selection_family=(SELECTION_FAMILY_SCHECHTER_STRUCT
                              if _family == "schechter" else None),
            m_lim=_tval("m_lim"),
            M0hat=_tval("M0hat"),
            sigma_M=_tval("sigma_M"),
            Mstar_hat=_tval("Mstar_hat"),
            alpha=_tval("alpha"),
            M_faint_offset=_tval("M_faint_offset"),
            k_corr_coeffs=((tuple(_fit["k_corr_coeffs"]) or None)
                           if _fit else None),
            selection_strata=(
                tuple(tuple(float(x) for x in s)
                      for s in _fit["strata_struct"])
                if _fit and _fit.get("strata_struct") else None),
        )
    survey = SurveyParams(
        n0=10.0 ** survey_values["log10n0"],
        z50=survey_values["z50"],
        w=survey_values["w"],
        delta=survey_values["delta"],
        b_miss=survey_values["b_miss"],
        alpha_miss=survey_values["alpha_miss"],
        sigma_kde=survey_values["sigma_kde"],
        **_mode_kwargs,
    )
    lss_kwargs, lss_attached, lss_attached_by_key = _completion_validation_lss_tables(
        data, unique_pixels, int(data.get("n_pix_catalog", np.asarray(full_z).shape[0]))
    )
    if _c_mode == "aggregate":
        # The aggregate Cbar sums the OBSERVED rows over the whole sky, which
        # a compact catalog carries as the field-convention inputs; build them
        # from the full-sky rows exactly as the loader does.
        from darksirens.redshift.completion import (
            build_field_normalization_inputs,
        )

        _field = build_field_normalization_inputs(
            jnp.asarray(full_z), jnp.asarray(full_w),
            jnp.asarray(full_n, dtype=jnp.int32))
        lss_kwargs = dict(
            lss_kwargs,
            field_dN_obs_s=_field.dN_obs_s,
            field_n_empty=jnp.asarray(float(_field.n_empty)),
            field_N_obs_total=jnp.asarray(float(_field.N_obs_total)),
            field_occupied_pixels=jnp.asarray(
                np.asarray(_field.occupied_pixels), dtype=jnp.int32),
        )
    if data.get("pixel_stratum_map") is not None:
        # Stratified selection routes each pixel to its own C_sel curve by
        # GLOBAL pixel, so the map stays full-sky (unsliced) like the run's.
        # Without it the dry run cannot form the stratified estimator at all.
        lss_kwargs = dict(
            lss_kwargs,
            pixel_stratum_map=jnp.asarray(
                np.asarray(data["pixel_stratum_map"], dtype=np.int32)),
        )
    em_catalog = EMCatalog(
        apix=data["apix"],
        zgals=jnp.asarray(full_z[unique_pixels]),
        dzgals=jnp.asarray(full_dz[unique_pixels]),
        wgals=jnp.asarray(full_w[unique_pixels]),
        ngals=jnp.asarray(full_n[unique_pixels], dtype=jnp.int32),
        delta_g_pix_z=jnp.asarray(
            data.get("delta_g_pix_z", jnp.zeros((1, dN_obs_kde.shape[1])))
        ),
        dN_obs_kde=dN_obs_kde,
        pixel_to_cache_idx=None,
        unique_pixels=jnp.asarray(unique_pixels, dtype=jnp.int32),
        **lss_kwargs,
    )
    diagnostics = completion_clip_diagnostics(
        cosmo=cosmo,
        survey=survey,
        em_catalog=em_catalog,
        max_pixels=max_pixels,
    )
    diagnostics["lss_completion_attached"] = lss_attached
    # Per-leaf provenance: the scalar above cannot express a MIXED outcome, so a
    # reader could not tell which Q leaf the reported clip fraction came from.
    diagnostics["lss_completion_attached_by_key"] = lss_attached_by_key or None
    diagnostics["c_mode"] = _c_mode
    if _c_mode == "selection":
        diagnostics["selection_theta_used"] = {
            k: float(getattr(survey, k))
            for k in ("m_lim", "M0hat", "sigma_M",
                      "Mstar_hat", "alpha", "M_faint_offset")}
        diagnostics["selection_family"] = str(
            getattr(opts, "selection_family", None) or "gaussian")
        # The selection completeness never sees a galaxy count, so the missing
        # budget amplitude (proportional to n0/H0^3) is identified only by its
        # prior. Record the model-vs-observed counts so a prior-driven budget is
        # auditable, and say so loudly when they disagree grossly.
        from darksirens.redshift.completion import selection_budget_audit

        _n_full = np.asarray(full_n).reshape(-1)
        _audit = selection_budget_audit(
            cosmo, survey, em_catalog,
            N_obs_total=float(_n_full.sum()),
            n_occupied=int((_n_full > 0).sum()),
            n_pix_total=int(data.get("n_pix_catalog", _n_full.size)),
        )
        diagnostics["selection_budget_audit"] = _audit
        _ratio = float(_audit["selection_model_over_observed_footprint"])
        if not (1.0 / 3.0 < _ratio < 3.0):
            _warn(
                f"c_mode=selection budget audit: the model predicts "
                f"{_audit['selection_model_N_obs_footprint']:.3g} catalogued "
                f"galaxies over the footprint but the catalog holds "
                f"{_audit['N_obs_total']:.3g} (ratio {_ratio:.3g}). The "
                "selection completeness carries no counts, so the missing "
                "budget is proportional to n0/H0^3 with nothing in the "
                "likelihood calibrating it: at this (log10n0, delta, theta) the "
                "in-vs-out-of-catalog odds -- and therefore the H0 information "
                "-- are set by the log10n0 prior, not by the data. Recalibrate "
                "log10n0 to N/(f_sky V_c) or narrow its prior."
            )
    diagnostics["survey_values"] = survey_values
    diagnostics["cosmology_values"] = {
        "H0": float(cosmo.H0),
        "Om0": float(cosmo.Om0),
        "w0": float(cosmo.w0),
        "wa": float(cosmo.wa),
    }
    diagnostics["dark_energy_fixed"] = _fixed_dark_energy_metadata(
        opts, fixed_parameter_values
    )
    diagnostics["prior_overrides"] = prior_overrides or None
    diagnostics["fixed_parameter_values"] = fixed_parameter_values or None

    os.makedirs(opts.save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = os.path.join(
        opts.save_path, f"completion_validation{suffix}__{timestamp}.json"
    )
    with open(path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    return path


def run_completion_validation_multitracer(
    opts,
    data: dict,
    prior_overrides: dict,
    fixed_parameter_values: dict,
) -> list:
    """Per-catalog completion validation for a K>=2 mixture run.

    Each catalog gets the SAME dry-run diagnostic as a single-catalog run:
    the full survey rows are re-read host-side (bundles keep only compact
    views) and paired with that catalog's own per-sample pixel indices,
    stashed on the bundle by ``load_multitracer_catalog_bundles`` when
    ``--validate_completion`` is set.  Returns one JSON path per catalog,
    suffixed ``_c{k}``.
    """
    from darksirens.inference.loaders import load_survey

    bundles = data.get("catalogs") or []
    if len(bundles) != len(opts.survey_paths):
        _fatal(
            "Per-catalog completion validation needs one loaded bundle per "
            f"--survey_path (got {len(bundles)} bundles for "
            f"{len(opts.survey_paths)} paths)."
        )
    paths = []
    for k, (survey_path, bundle) in enumerate(
        zip(opts.survey_paths, bundles), start=1
    ):
        if bundle.get("pixels_pe_full") is None or bundle.get("pixels_sel_full") is None:
            _fatal(
                "Per-catalog completion validation requires the per-sample "
                f"pixel indices for catalog {k}; the bundle loader stashes "
                "them only when --validate_completion is set."
            )
        _, ngals, zgals, dzgals, wgals, _ = load_survey(survey_path, to_device=False)
        data_k = dict(
            zgals_catalog=zgals,
            dzgals_catalog=dzgals,
            wgals_catalog=wgals,
            ngals_catalog=ngals,
            pixels_pe=bundle["pixels_pe_full"],
            pixels_sel=bundle["pixels_sel_full"],
            apix=bundle["apix"],
            n_pix_catalog=bundle["n_pix_catalog"],
        )
        # This catalog's OWN missing-galaxy rate factor: its Q_LSS table (or, with
        # --use_lss and no Q, its per-pixel delta_g field).  Omitting them made
        # every per-catalog dry run report the (1, N_grid) dummy overdensity, i.e.
        # a 0.0 clip fraction, whatever the likelihood was actually going to use.
        for key in ("delta_g_pix_z", "lss_completion_indexing",
                    *_LSS_TABLE_ROW_AXIS):
            if bundle.get(key) is not None:
                data_k[key] = bundle[key]
        paths.append(
            run_completion_validation(
                opts, data_k, prior_overrides, fixed_parameter_values,
                suffix=f"_c{k}",
            )
        )
    return paths


def _resolve_selection_fit_pins(sel, family, fixed_parameter_values):
    """Fold a --selection_fit's PINNED constants into ``fixed_parameter_values``
    and return the EFFECTIVE theta the likelihood will run at.

    The two pinned fields are not the same kind of thing.  ``m_lim`` is the
    per-run truncation DATUM -- the fit only supplies a default, and a run may
    legitimately declare a shallower limit -- so it is a ``setdefault``.
    ``M_faint_offset`` is a PROTOCOL constant of the Schechter model: it fixes
    the completeness DENOMINATOR (what counts as a galaxy), so the fit, the
    Q-table base and the likelihood must integrate the same faint end, and an
    explicit ``--fixed_parameter_values`` entry may only restate it.  Under
    ``setdefault`` such an entry silently won over the fit while every
    provenance stamp still agreed.

    The returned theta is the fit's, with every ``--fixed_parameter_values``
    pin folded in, so :func:`_check_selection_qtable_theta` compares the Q
    table's stamps against what the likelihood ACTUALLY uses (an overridden
    m_lim the table was not built with is the same stale-base mismatch).
    """
    from darksirens.redshift.selection import SELECTION_THETA_FIELDS

    fixed_parameter_values.setdefault("m_lim", float(sel["m_lim"]))
    if family == "schechter":
        offset_fit = float(sel["M_faint_offset"])
        offset_cli = fixed_parameter_values.get("M_faint_offset")
        if offset_cli is not None and abs(float(offset_cli) - offset_fit) > 1e-9:
            _fatal(
                f"--fixed_parameter_values M_faint_offset={float(offset_cli)} "
                f"contradicts the --selection_fit's "
                f"M_faint_offset={offset_fit}: the faint cutoff "
                "M_faint = Mstar_hat + M_faint_offset is a PROTOCOL constant "
                "fixing the completeness denominator, not a per-run knob, so "
                "the fit, any Q-table base and the likelihood must all "
                "integrate the same luminosity function. Drop the override, "
                "or refit with darksirens_fit_selection --m_faint_offset "
                f"{float(offset_cli)}.")
        fixed_parameter_values["M_faint_offset"] = offset_fit
    return {name: float(fixed_parameter_values.get(name, sel[name]))
            for name in SELECTION_THETA_FIELDS[family]}


def _check_aggregate_requires_q(opts, lss_active_by_catalog):
    """``--c_mode aggregate`` is only meaningful with a full-sky Q table.

    Aggregate mode replaces the per-pixel completeness with ONE sky curve
    ``Cbar(z) = Sum_p dN_obs_s / (N_pix_total dN_exp_s)`` -- normalised over the
    WHOLE SPHERE, so for a footprint survey ``Cbar ~ f_sky C_footprint`` --
    broadcast to every pixel.  The design intends the mean-one ``Q`` field to
    put that budget back where the galaxies are ("C says HOW MUCH is missing, Q
    says WHERE it goes"), and the aggregate Q builder fits the full sky for
    exactly that reason.  With no Q table the missing density falls back to the
    legacy ``delta_g`` factor (or 1), both of which are mean-one over the sky
    and therefore CANNOT encode a footprint: inside the footprint the prior then
    claims ``1 - f_sky C_foot`` of hosts are uncatalogued (75% at f_sky = 0.25
    with a locally complete catalog), diluting the very pixels that carry the
    localization information.
    """
    if str(getattr(opts, "c_mode", None) or "per_pixel") != "aggregate":
        return
    if latent_field_mode(opts):
        # Latent mode HAS a Q -- it is generated per member from the anchor
        # artifact instead of loaded -- so the premise of this refusal (no Q,
        # therefore a mean-one factor that cannot encode a footprint) does not
        # hold.  The latent Q is explicitly footprint-shaped: bit-zero logQ
        # off-footprint (pin P13b), and the f_p that guard 6 forces alongside
        # it carries the footprint into both sides of the budget.
        return
    missing = [k for k, active in enumerate(lss_active_by_catalog, start=1)
               if not active]
    if missing:
        _fatal(
            "--c_mode aggregate requires an --lss_completion table for every "
            f"catalog (catalog(s) {missing} have none). Aggregate mode "
            "normalises the completeness over the WHOLE SKY and delegates all "
            "angular structure to the mean-one Q field, so without a full-sky Q "
            "a footprint survey's missing budget is diluted by f_sky inside its "
            "own footprint -- the legacy delta_g factor is mean-one over the sky "
            "and cannot encode a footprint. Build Q with "
            "darksirens_build_lognormal_completion --c-mode aggregate, or run "
            "--c_mode per_pixel."
        )


def _check_q_table_z_depth(q_fiducials, opts):
    """Fail-closed depth provenance of Q tables, PER CATALOG.

    ``z_depth`` sets the completeness DENOMINATOR: with a depth,
    ``_precompute_grids`` forms ``S @ (dN_exp 1[z <= z_depth])``, and without it
    the denominator keeps mass above the depth the numerator can never match, so
    C is biased low (and ``(1 - C) dN_exp`` high) over roughly the last ~0.1 in
    z below the edge.  Q is the fit RESIDUAL to that base, so a table built at a
    different depth than the run resolves places the missing galaxies against
    the wrong completeness near the survey edge.  The builder stamps
    ``fiducial_z_depth`` (absent = no depth prior at build time, or a table
    predating the stamp), and this run's per-catalog depth is
    ``opts.resolved_survey_z_depths``.
    """
    depths = list(getattr(opts, "resolved_survey_z_depths", None) or [])
    for _k, _fid in enumerate(q_fiducials, start=1):
        if not _fid:
            continue
        run_depth = depths[_k - 1] if len(depths) >= _k else None
        tab_depth = _fid.get("fiducial_z_depth")
        if tab_depth is not None and run_depth is not None:
            if abs(float(tab_depth) - float(run_depth)) > 1e-9:
                _fatal(
                    f"catalog {_k}: Q table {_fid.get('path')} was built with "
                    f"z_depth={float(tab_depth):g} but this run resolves "
                    f"z_depth={float(run_depth):g}. The depth truncates the "
                    "completeness denominator the field was fit residual to, "
                    "so the table's Q would be consumed against a different "
                    "C near the survey edge. Rebuild with "
                    "darksirens_build_lognormal_completion --z-depth "
                    f"{float(run_depth):g}, or run with --survey_z_depth "
                    f"{float(tab_depth):g}."
                )
        elif tab_depth is None and run_depth is not None:
            _warn(
                f"catalog {_k}: Q table {_fid.get('path')} carries no "
                f"fiducial_z_depth stamp but this run truncates completeness "
                f"at z_depth={float(run_depth):g}. The table was almost "
                "certainly fit against an UNTRUNCATED denominator, which "
                "biases C low (and the missing budget high) just below the "
                "depth. Rebuild it with the current builder, which reads the "
                "catalog's z_depth attr."
            )
        elif tab_depth is not None and run_depth is None:
            _fatal(
                f"catalog {_k}: Q table {_fid.get('path')} was built with "
                f"z_depth={float(tab_depth):g} but this run applies NO depth "
                "prior: the table's completeness base is truncated and the "
                "run's is not. Pass --survey_z_depth "
                f"{float(tab_depth):g} (or rebuild the table without a depth)."
            )


def _check_selection_qtable_theta(q_fiducials, opts):
    """Fail-closed theta provenance of c_mode=selection Q tables, PER CATALOG.

    ``q_fiducials`` is one entry per catalog in catalog order; catalog k's
    table is checked against catalog k's OWN magnitude fit
    (``opts.selection_fits[k-1]``).  Checking every table against one fit was
    the single-catalog assumption this replaces: with per-catalog fits, a
    catalog-2 table built on catalog 1's theta must be caught, and a legal
    "Q on catalog 1 only" config must still pass.

    theta is SAMPLED by design (exempt from the pinned-to-build rule, like
    H0), but a table's FIXED C_sel base must come from the SAME offline fit
    that centers that catalog's theta prior: first the luminosity-function
    FAMILY (the two bases differ by the whole LF shape, not a parameter),
    then that family's theta compared to 1e-6, then the K(z) template
    compared elementwise INCLUDING the coefficient count (a pre-K table --
    no ``selection_kcorr_c*`` stamps -- is only consistent with a K = 0 fit).

    The FAMILY check runs before the no-fit escape hatch, because that hatch
    only widens theta WITHIN the run's family: the run's family is itself
    set by --selection_fit alone, so a schechter-stamped table with no fit
    means the likelihood forms a GAUSSIAN C_sel against a schechter fixed
    budget -- a mismatch no configuration can make consistent, and therefore
    an error rather than the sanctioned wide-open ablation.  Only the
    same-family, no-fit case warns (deliberate-ablation escape hatch).

    Each fit record's ``theta`` is the EFFECTIVE theta (the fit's, with any
    --fixed_parameter_values pin folded in), so an overridden m_lim is
    compared against the stamp the likelihood will actually contradict.
    """
    from darksirens.redshift.selection import SELECTION_THETA_FIELDS

    if latent_field_mode(opts):
        # BYPASS, latent mode only (PLAN 4.4: this check is named in the list
        # of firewall pieces retired for the latent path).  Every statement
        # below is about a Q TABLE's build-time theta stamps -- the fixed C_sel
        # base frozen into a resident logq_map.  Latent mode has no Q table
        # (guard 6 refuses both --lss_completion and the in-catalog group), so
        # there is nothing to compare theta against: the check has no referent
        # rather than a weaker one.  Note this is NOT a loss of provenance --
        # the seam's C_sel is formed in-likelihood at the SAMPLED theta, so the
        # stale-base failure mode this firewall exists to catch cannot occur,
        # and the artifact's own sha256 over (basis, footprint, completeness
        # map, anchor theta_ref, b_gal) is the successor stamp.  Table mode
        # reaches this line and falls through, unchanged.
        return

    fits = getattr(opts, "selection_fits", None)
    if fits is None:
        fits = [None] * len(q_fiducials)
    fits = list(fits)
    if len(fits) != len(q_fiducials):
        _fatal(
            f"selection-fit provenance: {len(fits)} per-catalog fits against "
            f"{len(q_fiducials)} Q-table fiducial records; both are indexed "
            "by catalog order, so a length mismatch means the theta firewall "
            "would compare a table against another catalog's fit."
        )
    for _k, (_fid, _fit) in enumerate(zip(q_fiducials, fits), start=1):
        # Activation on EITHER family's magnitude center: keying on
        # selection_M0hat alone would silently skip every schechter table.
        if not _fid or not ({"selection_M0hat", "selection_Mstar_hat"}
                            & _fid.keys()):
            continue
        # The stamp when present; inferred from the theta keys for tables
        # built before the family tag existed.
        _tab_family = str(_fid.get("selection_family") or (
            "schechter" if "selection_Mstar_hat" in _fid else "gaussian"))
        _run_family = (str(_fit.get("family", "gaussian")) if _fit
                       else getattr(opts, "selection_family", None)
                       or "gaussian")
        if _tab_family != _run_family:
            _fatal(
                f"catalog {_k}: Q table {_fid.get('path')}'s fixed C_sel "
                f"base is a '{_tab_family}' curve but this run forms the "
                f"'{_run_family}' family in-likelihood; the two completeness "
                "bases differ by the whole LF shape, not by a parameter, so "
                "the table's fixed (1 - C_sel) dN_exp budget and the sampled "
                "completeness would carry different luminosity functions."
                + (" The run's family comes from --selection_fit alone, so "
                   f"there is no run in which a '{_tab_family}' table is "
                   "consistent without one: pass the --selection_fit this "
                   "table was built with."
                   if _fit is None else
                   " Rebuild the table from this fit (or point "
                   "--selection_fit at the fit the table was built with).")
            )
        if _fit is None:
            _warn(
                f"catalog {_k}: Q table was built on a c_mode=selection "
                f"'{_tab_family}' base at theta_hat=("
                + ", ".join(
                    f"{_n}={float(_fid[f'selection_{_n}']):.4f}"
                    for _n in SELECTION_THETA_FIELDS[_tab_family]
                    if f"selection_{_n}" in _fid)
                + f") but no --selection_fit entry anchors catalog {_k}'s "
                "theta prior: "
                "the table's fixed base and the wide-open sampled theta are "
                "only consistent as a deliberate ablation."
            )
            continue
        _theta = _fit["theta"]
        _run_k = tuple(_fit["k_corr_coeffs"] or ())
        _run_strata = [tuple(s) for s in (_fit["strata_fit"] or ())] or None
        _run_sha = _fit["stratum_map_sha256"] or ""
        # Family-driven key set: a schechter base is verified on
        # (m_lim, Mstar_hat, alpha, M_faint_offset), a gaussian one on
        # (m_lim, M0hat, sigma_M).
        for _name in SELECTION_THETA_FIELDS[_run_family]:
            _key = f"selection_{_name}"
            if _key not in _fid:
                _fatal(
                    f"catalog {_k}: Q table {_fid.get('path')} carries no "
                    f"{_key} stamp, so its fixed '{_run_family}' C_sel base "
                    "cannot be verified against --selection_fit. Rebuild the "
                    "table with the current "
                    "darksirens_build_lognormal_completion."
                )
            if abs(_theta[_name] - float(_fid[_key])) > 1e-6:
                _fatal(
                    f"catalog {_k}: Q table {_fid.get('path')} was built at "
                    f"{_name}={float(_fid[_key]):.6f} but --selection_fit "
                    f"gives {_name}={_theta[_name]:.6f}: the table's fixed "
                    "C_sel base and this catalog's theta prior must come "
                    "from the SAME darksirens_fit_selection output. Rebuild "
                    "the table with the current fit (or point --selection_fit "
                    "at the fit the table was built with)."
                )
        _tab_k = []
        _j = 1
        while f"selection_kcorr_c{_j}" in _fid:
            _tab_k.append(float(_fid[f"selection_kcorr_c{_j}"]))
            _j += 1
        _tab_k = tuple(_tab_k)
        if len(_tab_k) != len(_run_k) or any(
                abs(a - b) > 1e-9 for a, b in zip(_tab_k, _run_k)):
            _fatal(
                f"catalog {_k}: Q table {_fid.get('path')} was built with "
                f"K(z) coefficients {list(_tab_k)} but --selection_fit "
                f"carries {list(_run_k)}: the fixed C_sel base and the run's "
                "selection model must share the SAME K-correction "
                "template. Rebuild the table from the current fit."
            )
        # Stratified base: the run's fit strata and stratum map must match
        # what the table was built with, per stratum and by map hash -- a
        # mismatch means the table's per-pixel base rows disagree with the
        # numerator's routing.  A stratified RUN against an unstamped
        # (single-stratum-base) table, or vice versa, is equally fatal.
        _tab_n = int(_fid.get("selection_n_strata", 0) or 0)
        if _run_strata and not _tab_n:
            _fatal(
                f"catalog {_k}: Q table {_fid.get('path')} was built on a "
                "SINGLE-stratum C_sel base but this run's --selection_fit is "
                "stratified: rebuild the table with --stratum-map (the "
                "stratified builder base) or run with the single-stratum fit."
            )
        if _tab_n and not _run_strata:
            _fatal(
                f"catalog {_k}: Q table {_fid.get('path')} was built on a "
                f"stratified C_sel base ({_tab_n} strata) but this run "
                "carries a single-stratum --selection_fit: point "
                "--selection_fit/--stratum_map at the fit and map the table "
                "was built with."
            )
        if _run_strata and _tab_n:
            if _tab_n != len(_run_strata):
                _fatal(
                    f"catalog {_k}: Q table {_fid.get('path')}: {_tab_n} "
                    f"strata stamped but the run's fit carries "
                    f"{len(_run_strata)}."
                )
            for _j, (_ml, _m0, _sg) in enumerate(_run_strata):
                for _nm, _val in (("m_lim", _ml), ("M0hat", _m0),
                                  ("sigma_M", _sg)):
                    _tv = float(_fid.get(f"selection_s{_j}_{_nm}", np.nan))
                    if not np.isfinite(_tv) or abs(_tv - _val) > 1e-6:
                        _fatal(
                            f"catalog {_k}: Q table {_fid.get('path')}: "
                            f"stratum {_j} {_nm}={_tv:.6f} but the run's fit "
                            f"gives {_val:.6f}; rebuild from the current fit."
                        )
            _tab_sha = str(_fid.get("selection_stratum_map_sha256", ""))
            if _tab_sha != str(_run_sha):
                _fatal(
                    f"catalog {_k}: Q table {_fid.get('path')} was built with "
                    "a different stratum map (sha256 mismatch); the per-pixel "
                    "base routing must be identical between build and run."
                )


def format_selection_guard_summary(
    guard_mode, sampler, soft_guard, max_likelihood_variance
):
    """One-line, always-printed summary of the resolved sparse-selection guard.

    Pure (string in, string out) so it is unit-testable without a full CLI
    run.  ``guard_mode`` is the raw ``--selection_neff_guard`` value
    ("auto"/"hard"/"soft"); ``soft_guard`` is the resolved boolean (which the
    CLI derives from ``guard_mode`` and ``sampler``).  The returned line is
    printed with the surrounding block's ``  [i] `` prefix.
    """
    resolved = "SOFT" if soft_guard else "HARD"
    why = f"auto: {sampler}" if guard_mode == "auto" else "explicit"
    criterion = (
        "sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel > "
        f"{max_likelihood_variance} (--max_likelihood_variance)"
    )
    if soft_guard:
        # SOFT never returns -inf; it replaces the wall with a smooth penalty
        # and the posterior must be checked to clear the boundary post hoc.
        action = f"smooth penalty wall as {criterion}; verify the posterior clears it post hoc"
    else:
        action = f"-inf when {criterion}"
    return (
        f"selection guard: {resolved} ({why}) — {action}; "
        "Vitale floor Neff > 5*N_obs always applies"
    )


def _universe_model_arg(value):
    """argparse ``type`` for --universe_model.

    The weak-lensing (``spectral_sirens_wl``) universe model moved to the
    ``darksirens_inference_lensing`` CLI. argparse runs ``type`` before
    ``choices``, so intercept the retired value here and raise a migration
    hint instead of the generic "invalid choice" message.
    """
    if value == "spectral_sirens_wl":
        raise ArgumentTypeError(
            "universe_model 'spectral_sirens_wl' moved to the lensing CLI; run "
            "`darksirens_inference_lensing --cluster_mode off "
            "--wl_backend lognormal|tabulated ...` instead."
        )
    return value


# ── Main ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ParameterSpace:
    """What the likelihood/sampling/save phases need from the built space."""
    labels: object
    lower_bound: object
    upper_bound: object
    prior_kinds: object
    prior_transform: object
    pop_params_fid: object
    n_pop_eff: object
    n_cosmo_eff: object
    n_survey_eff: object
    model_name: object
    # Index-resolved joint prior constraints applied by make_prior_transform's
    # cube maps (empty when the model declares none); carried so the samplers
    # can see which constraints the transform -- and only the transform --
    # enforces.
    joint_constraints: object = ()


def build_parser():
    """Construct the darksirens_inference ArgumentParser.

    Extracted from main() so tooling can build the parser without running a
    full inference; PR 3 depends on this contract (returns the parser).
    """
    optp = ArgumentParser(description=__doc__,
                          formatter_class=RawDescriptionHelpFormatter)

    g = optp.add_argument_group("Data")
    g.add_argument("--gw_path",          default=None,
                   help=("gwcat PE posterior-samples file (format_version "
                         "gwcat-1.0, gwcat-pe-2.0 or gwcat-pe-2.1; a 2.x "
                         "file must use spin_basis=chieff). Exactly one of "
                         "--gw_path / --gw_flows_path is required."))
    g.add_argument("--gw_flows_path",    default=None,
                   help=("Directory of per-event normalizing-flow checkpoints "
                         "(<EVENT>/<EVENT>_flow.npz) replacing stored PE "
                         "samples. Mutually exclusive with --gw_path; "
                         "spectral_sirens only."))
    g.add_argument("--gwselection_path", default=None,
                   help=("gwcat selection/injection file (format_version "
                         "gwcat-selection-1.0, -2.0 or -2.1; a 2.x "
                         "file must use spin_basis=chieff). Exactly one of "
                         "--gwselection_path / --pdet_flow_path is required."))
    g.add_argument("--allow_invalid_spin_swap", action="store_true",
                   help=("Load a chi_eff selection file whose per-campaign "
                         "injected_spin_uniform_isotropic evidence says the "
                         "analytic chi_eff spin swap was invalid (warn instead "
                         "of refuse). Deliberate legacy comparisons only: the "
                         "resulting pdraw is the wrong draw density for the "
                         "non-uniform campaign(s)."))
    g.add_argument("--survey_path",      default=None, nargs="+", metavar="PATH",
                   help=("Galaxy survey catalog(s). One path = current "
                         "single-catalog behaviour. Multiple paths define a "
                         "K-catalog dark-siren mixture: the redshift prior "
                         "becomes sum_k w_k p_k(z) with per-catalog survey "
                         "blocks (_c{k} labels) and sampled stick weights "
                         "fcat_2..fcat_K (dark_sirens only)."))
    g.add_argument("--save_path",        default="./")

    g = optp.add_argument_group("Flow surrogates (with --gw_flows_path)")
    g.add_argument("--flows_nsamp", type=int, default=16384,
                   help=("Population draws J per event per likelihood call "
                         "(common random numbers, event-windowed proposal). "
                         "Per-event ESS is typically 1-5%% of J on real "
                         "events; the total-lnL variance guard "
                         "(--max_likelihood_variance) rejects undersampled "
                         "settings."))
    g.add_argument("--flows_seed", type=int, default=42,
                   help="Seed of the fixed base-uniform array (CRN).")
    g.add_argument("--flows_pattern", default="*/*_flow.npz",
                   help="Checkpoint glob relative to --gw_flows_path.")
    g.add_argument("--flows_on_mismatch", default="error",
                   choices=["error", "skip", "load"],
                   help=("Structural-check policy for checkpoints that do not "
                         "match the installed flowjax (version drift)."))
    g.add_argument("--flows_chieff_amax", type=float, default=0.99,
                   help="amax of the 1-D isotropic chi_eff PE prior.")
    g.add_argument("--flows_pe_cosmology", default="67.74,0.3089",
                   help=("'H0,Om0' of the PE prior cosmology (UniformSourceFrame "
                         "distance prior), matching the PE release."))
    g.add_argument("--flows_grid_nm", type=int, default=512,
                   help="m1 cells of the (m1, q) population sampling grid.")
    g.add_argument("--flows_grid_nq", type=int, default=256,
                   help="q cells of the (m1, q) population sampling grid.")
    g.add_argument("--flows_support_margin", type=float, default=0.25,
                   help=("Per-event proposal window: expand each flow's "
                         "sampled parameter range by this fraction per side "
                         "(exactly corrected in the proposal density)."))
    g.add_argument("--flows_support_nsamples", type=int, default=4096,
                   help="Flow draws used to measure each event's support box.")
    # Literal, not an import: darksirens.likelihood.flow_events pulls in
    # flowjax, and the parser must build without the flow extras installed.
    # tests/test_flow_mixture_proposal.py pins this to DEFAULT_W_FULL.
    g.add_argument("--flows_wfull", type=float, default=0.05,
                   help=("Fraction of each event's draws taken from the FULL "
                         "population support instead of its empirical support "
                         "window (mixture proposal, issue #260). The exact "
                         "mixture density enters every weight, so any value in "
                         "[0, 1) is unbiased; 0 restores the windowed-only "
                         "estimator, which truncates the event integral by a "
                         "hyperparameter-dependent amount."))

    g = optp.add_argument_group(
        "Selection-function emulator (alternative to --gwselection_path)")
    g.add_argument("--pdet_flow_path", default=None,
                   help=("NF P_det emulator checkpoint (npz: found-injection "
                         "flow + xi). Drives the selection integral through "
                         "pseudo-injections drawn once at load time. Exactly "
                         "one of --gwselection_path / --pdet_flow_path is "
                         "required."))
    g.add_argument("--pdet_nsamp", type=int, default=1_000_000,
                   help=("Pseudo-injections drawn from the emulator flow at "
                         "load (Ndraw = nsamp / xi)."))
    g.add_argument("--pdet_seed", type=int, default=42,
                   help="Seed of the emulator pseudo-injection draw.")
    g.add_argument("--pdet_cosmology", default="67.9,0.3065",
                   help=("'H0,Om0' of the injection-campaign cosmology used "
                         "for z->dL and the Jacobian; must match the campaign "
                         "the emulator was trained on."))
    g.add_argument("--pdet_chieff_amax", type=float, default=0.99,
                   help=("Reference amax of the chi_eff spin-prior swap; keep "
                         "equal to --flows_chieff_amax and the injection-file "
                         "convention."))

    g = optp.add_argument_group("Physical model")
    g.add_argument("--universe_model", default="spectral_sirens",
                   type=_universe_model_arg,
                   choices=["spectral_sirens", "dark_sirens",
                            "dark_sirens_complete", "bright_sirens"])
    g.add_argument(
        "--sky_model", default="isotropic",
        choices=["isotropic", "dipole", "sphere_gp", "sphere_gp_z", "overdensity_gp",
                 "multipole", "multipole_l3"],
        help=(
            "Sky distribution of the source rate. 'isotropic' (default) is the "
            "null; 'dipole' (Isi, Farr & Varma 2023) and 'sphere_gp' "
            "(log-Gaussian random field, Essick et al. 2023) are angular g(n). "
            "'sphere_gp_z' is a (sphere x z) GP normalised per z-shell "
            "(directional anisotropy evolving with distance); 'overdensity_gp' "
            "is the same field normalised over the comoving volume (full 3-D "
            "clustering, use with gamma fixed). 'multipole'/'multipole_l3' are "
            "low-order spherical-harmonic expansions g=1+sum a_lm Y_lm (l<=2/3) "
            "-- the perturbative, sharply-constrained choice for a small "
            "deviation, giving the angular power spectrum C_l. All compared to "
            "isotropy by evidence; forced to 'isotropic' for bright_sirens."
        ),
    )
    g.add_argument(
        "--mark_model", default="none", choices=["none", "loglinear"],
        help=(
            "Marked-host model (dark_sirens): reweight catalog galaxies by a "
            "BBH-host efficiency h(m|eta) over per-galaxy marks. 'none' (default) "
            "is the legacy galaxy-count host model. 'loglinear' fits "
            "h=exp(sum_k eta_k m_tilde_k) over the (z-centred) marks selected by "
            "--marks. Measures whether GW hosts prefer high-M*/sSFR/low-Z "
            "galaxies at fixed redshift."
        ),
    )
    g.add_argument(
        "--marks", default=None, metavar="LIST",
        help=(
            "Comma-separated marks for --mark_model loglinear "
            "(subset of: logmstar,logssfr,metallicity,color). Default: all marks "
            "present in the catalog."
        ),
    )
    g.add_argument(
        "--pop_model",
        default="powerlaw+peak",
        help=(
            "Population model composition. Grammar mixtures combine mass "
            "tokens (powerlaw, brokenpowerlaw, peak) with '+' and optional "
            "count prefixes (for example, brokenpowerlaw+2peaks). Sharing is "
            "controlled separately by --shared_beta, --shared_spin, and "
            "--shared_gamma."
        ),
    )
    g.add_argument(
        "--shared_beta",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared beta/pairing distribution; false gives one beta per mass component.",
    )
    g.add_argument(
        "--shared_spin",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared spin distribution; false gives one spin distribution per mass component.",
    )
    g.add_argument(
        "--shared_gamma",
        type=str_to_bool,
        default=True,
        metavar="BOOL",
        help="Use one shared redshift-evolution gamma; false gives one gamma per mass component.",
    )
    g.add_argument("--fix_population",  type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--population_fiducials", default=FIDUCIAL_SET_LEGACY,
                   choices=list(FIDUCIAL_SETS), metavar="SET",
                   help=("WHICH curated fiducial vector --fix_population fixes "
                         "the population at. 'legacy' (DEFAULT) is the "
                         "inherited curated set, bit-identical to every "
                         "archived fixed-population run and to every mock built "
                         "from it -- but for 2powerlaws+peak / +2peaks / "
                         "+3peaks it sits OUTSIDE the model's own declared "
                         "priors (PL1.m_max 80 vs [15,50], PL2.m_min 5 vs "
                         "[20,40], G.mu 35 vs [50,100]), so the sampled model "
                         "cannot represent it and a fixed-vs-sampled comparison "
                         "against it is not a comparison of one model family. "
                         "'in_prior_v2' is the corrected set (each violating "
                         "parameter moved to the MIDPOINT of its own prior); "
                         "use it whenever the fixed arm is read as nested in "
                         "the sampled one."))
    g.add_argument("--allow_skymap_population",
                   type=str_to_bool, nargs="?", const=True, default=False,
                   metavar="BOOL",
                   help=("Bypass the refusal to run a FREE population block on a "
                         "PE file written by darksirens_skymaps_to_samples "
                         "(requires_fixed_population). Those files carry "
                         "surrogate mass/spin draws in place of the coordinates "
                         "a 3D skymap has marginalised away, so the population "
                         "posterior measures the surrogate proposal and the "
                         "selection integral stops cancelling it. Emits a loud "
                         "warning and proceeds; for methodological studies only."))
    g.add_argument("--fix_cosmology", "--fixed_cosmology", dest="fix_cosmology",
                   action=DeprecatedSpellingAction, deprecated=["--fixed_cosmology"],
                   type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fix the full cosmology block (H0, Om0, w0, wa). "
                         "--fixed_cosmology is a deprecated alias."))
    g.add_argument("--fix_de",          "--fixed_de", dest="fix_de",
                   action=DeprecatedSpellingAction, deprecated=["--fixed_de"],
                   type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fix only dark-energy cosmology parameters (w0, wa); "
                         "ignored when --fix_cosmology is true. "
                         "--fixed_de is a deprecated alias."))
    g.add_argument("--fix_survey",      type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument(
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
    g.add_argument(
        "--selection_neff_guard",
        choices=["auto", "hard", "soft"],
        default="auto",
        help=(
            "Sparse-selection (Neff <= 5 N_obs) validity guard. 'hard' returns -inf "
            "(nested samplers; the historical behavior). 'soft' replaces the wall with a "
            "steep smooth penalty so gradient-based samplers are not divergence-flagged "
            "on every trajectory that brushes it. 'auto' picks soft for --sampler numpyro "
            "and hard otherwise. Check the posterior clears the Neff boundary post hoc "
            "whenever the soft guard ran."
        ),
    )
    g.add_argument(
        "--max_likelihood_variance", type=float, default=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        help=("Cap on the Monte-Carlo variance of the TOTAL log-likelihood "
              "estimator, sigma^2_lnL = sum_i sigma^2_i + N_obs^2/Neff_sel "
              "(Essick & Farr 2022; Talbot & Golomb 2023, arXiv:2304.06138; the "
              "GWTC-4.0/5.0 population-analysis criterion, default 1.0). Proposals "
              "exceeding it are guarded (hard -inf or the soft wall per "
              "--selection_neff_guard). The Vitale 5 N_obs mean floor always applies."))
    g.add_argument("--prior_overrides", default=None, metavar="JSON")
    g.add_argument("--fixed_parameter_values", default=None, metavar="JSON")
    g.add_argument("--counterpart", nargs="+", metavar="RA_DEC_Z",
                   help=("Bright-siren counterpart RA DEC Z triplet(s), ordered by event; "
                         "angles are radians."))
    g.add_argument("--counterpart_dz", type=float, default=1.0e-4,
                   help="Gaussian redshift uncertainty for --counterpart.")
    g.add_argument("--counterpart_nside", type=int, default=1,
                   help="HEALPix NSIDE for the synthetic bright-siren counterpart catalog.")
    g.add_argument("--bright_siren_sky_marginalized", type=str_to_bool, default=False, metavar="BOOL",
                   help=("For bright_sirens, ignore the counterpart sky-pixel gate and "
                         "apply only the counterpart redshift prior."))
    g.add_argument("--complete_empty_pixel_policy", default="zero",
                   choices=["zero", "volume"],
                   help=("Policy for genuinely empty pixels in complete-catalog models: "
                         "'zero' is the formal complete-catalog default; "
                         "'volume' preserves the historical volume-prior robustness approximation."))

    g = optp.add_argument_group("Catalog")
    g.add_argument("--use_lss", "--use_LSS", dest="use_LSS",
                   action=DeprecatedSpellingAction, deprecated=["--use_LSS"],
                   type=str_to_bool, default=False, metavar="BOOL")
    g.add_argument("--lss_floor", default="conserve",
                   choices=["conserve", "legacy"],
                   help=("Number-conservation policy of the LEGACY "
                         "local-overdensity missing-galaxy factor "
                         "max(1 + alpha_miss*b_miss*delta_g, 0) (--use_lss "
                         "without an --lss_completion table). 'conserve' "
                         "(DEFAULT) renormalizes the FLOORED factor to "
                         "full-sky mean one at every redshift, restoring "
                         "Sum_p lss_p(z) == N_pix exactly for every parameter "
                         "draw. delta_g is built mean-zero over the full sky, "
                         "so the unfloored factor is a pure REDISTRIBUTION of "
                         "the budget (1-C) and n0 set; the floor breaks that "
                         "wherever b_eff*delta_g < -1 (reachable for "
                         "b_miss > 1, since delta_g >= -1) and INFLATES the "
                         "total missing count -- two equal pixels with "
                         "delta_g = (-1,+1) at b_eff = 2 floor to (0,3), mean "
                         "1.5, a 50%% budget inflation, 100%% at b_eff = 3. "
                         "That changes the catalog-versus-missing odds that "
                         "carry the dark-siren H0 information, not just where "
                         "the missing galaxies sit. 'conserve' is BIT-"
                         "IDENTICAL wherever the floor never engages (the "
                         "normalizer is then exactly 1.0). 'legacy' restores "
                         "the unrenormalized floor, for reproducing a "
                         "measurement made with it. Inert with a Q_LSS table "
                         "or --lss_field_mode latent (Q carries its own "
                         "mean-one renormalization)."))
    g.add_argument("--catalog_sky_weighting", default=None,
                   choices=["conditional", "field"],
                   help=("Catalog redshift-prior normalization convention (dark_sirens). "
                         "'field' (DEFAULT) is the JOINT catalog host-density estimand: it "
                         "normalizes by the survey-GLOBAL Z(theta)=Sum_all-pixels[N_obs+N_miss], "
                         "so RELATIVE angular host density is preserved (a pixel with 100 "
                         "candidate hosts carries ~100x the angular weight of a pixel with 1), "
                         "and for a K>=2 mixture the weight fcat_k measures the host FRACTION "
                         "(number-density / sky-clustering contrast). 'conditional' is the "
                         "radial-only LEGACY estimand: it normalizes each pixel by its own "
                         "Z[pix]=N_obs+N_miss so every pixel integrates to unit mass and "
                         "relative angular host density is DISCARDED (kept for reproducing "
                         "older single-catalog runs; bit-identical to the pre-existing "
                         "behaviour). Unset auto-resolves to field for every K. At K=1 field's "
                         "log10n0 (number-density) channel cancels between the PE and selection "
                         "terms so log10n0 is only weakly identified and marginalizes against "
                         "its prior -- what field restores at K=1 is the relative angular host "
                         "weighting conditional discards. For dark_sirens the degenerate "
                         "explicit conditional at K>=2 is fatal (fcat rails on clustered "
                         "catalogs). "
                         "Field requires the full-sky catalog (incompatible with "
                         "--drop_full_catalog); composes with --lss_completion, --use_lss "
                         "and --mark_model (the global normalizer carries the same "
                         "modulated missing budget as the numerator, per member under "
                         "--lss_marginalize); incompatible with universe models other than "
                         "dark_sirens/dark_sirens_complete."))
    g.add_argument("--survey_z_depth", type=float, default=None, metavar="Z",
                   help=("Survey redshift depth as a COMPLETENESS PRIOR: completeness is zero "
                         "beyond z_depth, so every modeled host there is uncatalogued (missing, "
                         "NOT nonexistent) and the source prior above the depth is the plain "
                         "volumetric x population shape. The observed catalog kernel is likewise "
                         "zeroed beyond the depth. Applies to ALL catalogs and overrides any "
                         "per-catalog f.attrs['z_depth'] written by darksirens_pixelate "
                         "--z_depth. Default None: per-catalog file attr if present, else "
                         "completeness is estimated over the full [0, DARKSIRENS_ZMAX] grid "
                         "(bit-identical to pre-existing behaviour)."))
    g.add_argument("--c_mode", default="per_pixel",
                   choices=["per_pixel", "aggregate", "selection"],
                   help=("Completeness estimator mode (dark_sirens). 'per_pixel' "
                         "(default): the legacy per-pixel matched-kernel ratio, "
                         "bit-identical to pre-existing behaviour -- its numerator "
                         "absorbs the observed angular clustering into C, so "
                         "(1 - C) anti-tracks the true missing density. "
                         "'aggregate': ONE sky-aggregate curve Cbar(z) per "
                         "proposal, broadcast to every pixel (occupied and "
                         "empty), so C carries only the radial budget and all "
                         "angular structure is Q's job; the field-convention "
                         "global normalizer uses the same curve. Requires "
                         "--catalog_sky_weighting field (the default) for "
                         "compact catalogs, and any --lss_completion table must "
                         "be built with the matching darksirens_build_"
                         "lognormal_completion --c-mode (hard-checked at load: "
                         "the two bases differ by the entire clustering "
                         "signal). 'selection': the PARAMETRIC magnitude-"
                         "limited completeness C_sel(z; theta) -- no counts in "
                         "the budget at all; the LUMINOSITY-FUNCTION FAMILY "
                         "comes from --selection_fit (gaussian: sampled "
                         "M0hat/sigma_M; schechter: sampled Mstar_hat/alpha), "
                         "and without a fit the run is the wide-open gaussian "
                         "ablation. theta is SAMPLED (under the "
                         "--selection_fit Gaussian prior when given, flat "
                         "within bounds otherwise) and m_lim is a fixed "
                         "truncation datum settable via "
                         "--fixed_parameter_values."))
    g.add_argument("--selection_fit", default=None, metavar="JSON[,JSON...]",
                   help=("selection_fit.json from darksirens_fit_selection "
                         "(c_mode=selection only), ONE PER CATALOG as a "
                         "comma-separated list in --survey_path order (a "
                         "single path for a single catalog). Each fit "
                         "DECLARES the luminosity-function family: gaussian "
                         "(sampled M0hat/sigma_M) or schechter (sampled "
                         "Mstar_hat/alpha, which additionally pins the "
                         "M_faint_offset protocol constant -- a "
                         "--fixed_parameter_values entry may restate it but "
                         "never change it); every fit in a run must declare "
                         "the SAME family. Catalog k's fit anchors ITS OWN "
                         "labels: theta_hat becomes the center and the "
                         "marginal Laplace sds the widths of truncated-normal "
                         "priors on that family's sampled labels (e.g. "
                         "M0hat_c{k}/sigma_M_c{k} for gaussian catalogs "
                         "k>=2), and m_lim (m_lim_c{k}) is pinned "
                         "to that fit's datum unless --fixed_parameter_values "
                         "overrides it. A single path is NEVER broadcast "
                         "across catalogs. Use \"\" as a placeholder for a "
                         "catalog with no fit -- fatal while that catalog "
                         "runs c_mode=selection, since its theta would sample "
                         "flat while the others are anchored. Catalog k's "
                         "--lss_completion table must have been built with "
                         "catalog k's fit (family + theta_hat stamped and "
                         "hard-checked at load). Omit the flag entirely for "
                         "flat-within-bounds selection priors on every "
                         "catalog (the wide-open ablation)."))
    g.add_argument("--stratum_map", default=None,
                   help=("HDF5 file with a full-sky 'stratum_map' dataset "
                         "(RING, at the SURVEY's nside) assigning every pixel "
                         "a selection stratum in [0, S). Required by, and "
                         "only legal with, a MULTI-stratum --selection_fit "
                         "(darksirens-selection-fit-1.1): stratum s uses the "
                         "fit's per-stratum (m_lim_s, theta_s) offsets around "
                         "the SAMPLED common-mode (M0hat, sigma_M), and the "
                         "global normalizer decomposes the empty-pixel "
                         "budget per stratum. Off-footprint pixels must "
                         "carry the map builder's documented policy label. "
                         "Single-catalog only (a K>=2 mixture cannot route "
                         "per-catalog stratum maps)."))
    g.add_argument("--per_pixel_completeness", default=None, metavar="H5",
                   help=("Depth-map HDF5 (build_mth_map output: masked_frac "
                         "+ counts, RING) supplying the per-pixel selection "
                         "fraction f_p = 1 - masked_frac, degraded to the "
                         "catalog nside by area weighting. Puts "
                         "C_p(z) = f_p C(z) into BOTH sides of the missing "
                         "budget (field-level PR-2, OWNER DECISION 4a). "
                         "Requires c_mode aggregate|selection with the "
                         "gaussian family; admits a deterministic Q table only "
                         "if its artifact stamps f_p_aware (otherwise the mask "
                         "is applied twice -- see --allow_double_counted_mask), "
                         "and then the f_p-weighted empty-pixel Q budget is "
                         "built alongside the plain one; refuses Q ENSEMBLES, "
                         "stratified selection, and K>=2 (their empty-pixel "
                         "budgets need per-member/per-stratum f_p-weighted "
                         "twins not yet derived)."))
    g.add_argument("--allow_double_counted_mask", action="store_true",
                   help=("Permit --per_pixel_completeness together with a Q "
                         "table whose artifact does not stamp f_p_aware. "
                         "Refused by default because it applies the survey mask "
                         "TWICE: a Q table is fit to observed counts and so "
                         "already carries the footprint (measured on the closure "
                         "mock: mean Q 1.624 on-footprint vs 0.050 off), and "
                         "C_p = f_p C then applies it again. The paired arm put "
                         "H0 at 41.24 [36.1, 46.3] against a truth of 67.74 -- "
                         "confidently wrong, and the tightest arm in that run. "
                         "Use only to reproduce that measurement."))
    g.add_argument("--allow_selection_fit_free_background", action="store_true",
                   help=("Permit a magnitude-selection --selection_fit while "
                         "the background cosmology beyond H0 (Om0/w0/wa) is "
                         "SAMPLED. Refused by default: the h-scaled zero point "
                         "absorbs H0 exactly but not Om0/w0/wa, which change "
                         "the shape of DM(z); the fit's zero point is anchored "
                         "at the package fiducial and is never re-fitted per "
                         "proposal, leaving ~0.1 mag of non-absorbable shape "
                         "across a z = 0.05-0.5 catalog against a ~0.02 mag "
                         "Laplace sd, so the selection model can inject "
                         "artificial information about the dark-energy "
                         "parameters. Use only for methodological studies."))
    g.add_argument("--allow_unmasked_footprint", action="store_true",
                   help=("Run c_mode aggregate|selection on a "
                         "footprint-limited catalog WITHOUT a per-pixel "
                         "selection fraction. Refused by default (S-3): the "
                         "survey completeness curve Cbar(z) is applied to "
                         "empty pixels too, so sky the survey never observed "
                         "is modelled as Cbar-COMPLETE and its hosts leave "
                         "the missing budget while the numerator keeps them. "
                         "16 of 16 footprint runs of the PR-6a mock railed to "
                         "H0 = 125-138 without a mask, and on the production "
                         "line the mask moves the median by 4.3 sigma. Sparse "
                         "all-sky catalogs are not affected: the guard fires "
                         "only when the empty fraction exceeds what the mean "
                         "count predicts under Poisson. Pass this flag for "
                         "the no-mask CONTROL arms, which want the exposed "
                         "configuration on purpose."))
    g.add_argument("--validate_completion", type=str_to_bool, default=False, metavar="BOOL",
                   help=("Run a dry-run completion clipping diagnostic, save JSON under "
                         "--save_path, and exit before building the likelihood."))
    g.add_argument("--completion_validation_pixels", type=int, default=64, metavar="N",
                   help="Maximum number of unique catalog pixels to inspect in --validate_completion.")
    g.add_argument("--lss_completion", default=None, nargs="+", metavar="PATH",
                   help=("Path to a precomputed LSS-conditioned lognormal completion file "
                         "(/lss_completion/logq_map), built offline by "
                         "darksirens-build-lognormal-completion. Replaces the legacy "
                         "max(1+b*delta_g,0) missing-galaxy factor for dark_sirens. If unset, "
                         "an in-catalog /lss_completion group (if present) is used automatically. "
                         "For a multi-catalog mixture pass 0, 1, or n_catalogs paths, "
                         "positionally aligned with --survey_path (use \"\" as a placeholder "
                         "for a catalog with no external completion)."))
    g.add_argument("--lss_field_mode", default="table",
                   choices=["table", "latent"],
                   help=("Where the missing-galaxy modulation Q comes from "
                         "(dark_sirens). 'table' (DEFAULT): the resident "
                         "(M_draw, N_rows, N_grid) log-Q table read from "
                         "--lss_completion or an in-catalog /lss_completion "
                         "group -- bit-identical to pre-existing behaviour. "
                         "'latent': Q is GENERATED in-likelihood from the "
                         "compact anchor artifact (--lss_field_artifact, built "
                         "by darksirens_build_latent_field) through the "
                         "factored (sphere x z) basis, "
                         "logQ_m(p,z) = 1[p in F] (b_GW row_fac_m[p].phi_z[z] "
                         "- rho_m(z)), with rho the closed-form budget "
                         "normalizer that conserves "
                         "Sum_p (1 - f_p C(z)) Q_p(z) exactly at every z, "
                         "member and theta (PLAN eq. 4). Off-footprint pixels "
                         "return bit-zero logQ. Latent mode is a DIFFERENT "
                         "PARAMETER SPACE, not a storage optimization: b_miss "
                         "becomes b_GW (see --lss_field_artifact), the Q-table "
                         "provenance firewall has no referent, and the run "
                         "requires --per_pixel_completeness with an "
                         "aggregate|selection --c_mode."))
    g.add_argument("--lss_field_artifact", default=None, metavar="H5",
                   help=("Latent-field anchor artifact (HDF5 /latent_field "
                         "group from darksirens_build_latent_field): the "
                         "solved field draws (row_fac), the z factor basis, "
                         "the theta-free sky moments (A, B) over the fitted "
                         "footprint and the (P_F, F_F) footprint constants. "
                         "Required by, and only legal with, "
                         "--lss_field_mode latent. The artifact carries the "
                         "footprint, the completeness map f_p and the anchor "
                         "(theta_ref, b_gal) it was solved at; its sha256 is "
                         "the provenance successor to the retired Q-table "
                         "stamps (PLAN 4.4). NOTE the parameter-space change "
                         "it brings: in table mode b_miss reaches the "
                         "likelihood only through max(1 + alpha_miss b_miss "
                         "delta_g, 0) and is inert without --use_lss, but in "
                         "latent mode b_miss IS b_GW -- the bias with which GW "
                         "hosts trace this artifact's field -- and is sampled "
                         "(weakly identified by the events, prior-dominated). "
                         "The run fingerprint's schema_version is bumped so a "
                         "latent run can never resume a table run's "
                         "checkpoint."))
    g.add_argument("--lss_field_sha256", default=None, metavar="HEX",
                   help=("Pin the identity of --lss_field_artifact (PLAN 4.4 "
                         "guard 1). Latent mode retires the five Q-table "
                         "provenance checks -- they describe a table's "
                         "build-time stamps and latent mode has no table -- so "
                         "the artifact fingerprint is the ENTIRE provenance "
                         "surface. Give either digest: the sha256 the builder "
                         "printed (stamped in /latent_field.attrs) or the "
                         "recomputed guard-1 CONTENT digest over (basis "
                         "geometry, nside, completeness map, shell response, "
                         "z_count_edges, counts, theta_ref, b_gal), which the "
                         "run prints and writes to settings.json and which "
                         "survives a copy that dropped the attrs. A mismatch "
                         "is fatal: the run would otherwise consume a "
                         "different field than the one whose budget, footprint "
                         "and anchor theta were verified. Latent mode only."))
    g.add_argument("--allow_unanchored_budget", action="store_true",
                   help=("Escape hatch for PLAN 4.4 guard 5 (the budget-anchor "
                         "guard) in --lss_field_mode latent. At rung 0 -- what "
                         "PR-6a ships -- the count channel is conditioned on "
                         "the observed shell totals and therefore carries ZERO "
                         "information about (log10n0, delta, theta_sel) BY "
                         "CONSTRUCTION, so a FLAT prior on log10n0 or delta is "
                         "not marginalisation over an uncertain budget: the "
                         "posterior on those directions is the prior, and the "
                         "missing-galaxy budget -- hence H0 -- is set by "
                         "whatever prior volume was declared. Latent mode "
                         "therefore requires them either pinned by "
                         "--fixed_parameter_values to the calibration fit "
                         "(experiments/desi_ingest/calibrate_n0.py; OWNER "
                         "DECISION 6) or carrying a normal calibration prior. "
                         "This flag runs anyway and is for deliberate "
                         "prior-sensitivity ablations ONLY: with it, any H0 "
                         "shift is an artifact of the declared prior volume, "
                         "the same class of defect as K10."))
    g.add_argument("--lss_marginalize", type=str_to_bool, default=False, metavar="BOOL",
                   help=("Fully-Bayesian marginalisation of the GW likelihood over the "
                         "Q_LSS ENSEMBLE: logL = logsumexp_m logL(Q_m) - log M, instead of "
                         "the deterministic posterior-mean Q. Requires --lss_completion to "
                         "point at a file built with members "
                         "(darksirens-build-lognormal-completion --n-members M>0); "
                         "dark_sirens only. Off (default) = deterministic Q, unchanged. "
                         "For a K>=2 mixture the marginalisation uses ONE SHARED member "
                         "index (member m of every catalog must sample the SAME LSS "
                         "realization): build the per-catalog ensembles jointly with "
                         "darksirens_build_joint_lognormal_completion (equal --n-members), "
                         "or the run aborts. "
                         "The matched-realizations requirement is verified from each "
                         "file's /lss_completion realization_set_id (they must be equal "
                         "and non-null); darksirens_build_joint_lognormal_completion "
                         "stamps one shared id across the K files, or pass "
                         "--allow_unverified_shared_lss_members to accept the "
                         "independent-fields approximation."))
    g.add_argument("--allow_unverified_shared_lss_members",
                   type=str_to_bool, nargs="?", const=True, default=False, metavar="BOOL",
                   help=("Bypass the K>=2 --lss_marginalize provenance check that the "
                         "per-catalog Q_LSS ensembles share a matched realization set "
                         "(equal, non-null realization_set_id). With this flag, catalogs "
                         "whose ensembles were built independently are marginalized "
                         "anyway, treating member m of each catalog as an INDEPENDENT LSS "
                         "draw -- an independent-fields product prior, NOT the matched "
                         "shared-field prior the estimator assumes. Emits a loud warning "
                         "and proceeds; use only if you understand the approximation."))

    g = optp.add_argument_group("Sampler")
    g.add_argument("--sampler",      required=True, choices=["tinyns", "dynesty", "numpyro"])
    g.add_argument("--nlive",        type=int,   default=1000)
    g.add_argument("--dlogz",        type=float, default=0.1)
    g.add_argument("--max_samples",  type=int,   default=1_000_000,
                   help="Max call/iteration budget for nested samplers "
                        "(dynesty call cap, tinyns iteration cap); 0 = unlimited. "
                        "The UNITS differ: tinyns has no call cap and one rwalk "
                        "iteration costs walks x max_active_chains likelihood "
                        "evaluations (5 for --tinyns_preset recommended, 1280 for "
                        "heavy_darksirens), so the same number buys far more "
                        "compute there -- the resolved cap and its call "
                        "equivalent are printed at sampler start.")
    add_tinyns_arguments(g, bool_type=str_to_bool)
    g.add_argument("--nuts_warmup",  type=int,   default=500)
    g.add_argument("--nuts_samples", type=int,   default=1000)
    g.add_argument("--nuts_chains",  type=int,   default=1)
    g.add_argument("--nuts_target_accept", type=float, default=0.8)
    g.add_argument("--nuts_max_tree_depth", type=int, default=10)
    g.add_argument("--nuts_chain_method", default="sequential", choices=["sequential", "parallel", "vectorized"])
    g.add_argument("--nuts_init_tries", type=int, default=32)
    g.add_argument("--nuts_init_seed_offset", type=int, default=100_000)
    g.add_argument("--seed",         type=int,   default=22)
    g.add_argument("--show_progress",type=str_to_bool, default=True, metavar="BOOL")
    g.add_argument("--dynesty_diagnostics", type=str_to_bool, default=False, metavar="BOOL",
                   help="Write dynesty runplot/traceplot PDFs every 10 minutes to "
                        "<run_dir>/dynesty_diagnostics/. Only used with --sampler dynesty.")
    g.add_argument("--sampler_preflight", choices=["on", "off"], default="on",
                   help="Probe 32 prior draws before nested sampling (dynesty/tinyns) and "
                        "fail fast if the likelihood is -inf on all of them (usually the "
                        "selection variance guard); 'off' skips the probe.")
    add_checkpoint_arguments(g)

    g = optp.add_argument_group("Performance")
    g.add_argument("--sel_batch_size", type=block_size_arg, default="auto",
                   metavar="N|auto|off",
                   help="Injections per selection-integral chunk. 'auto' (default) "
                        "sizes the block from probed free device memory after the "
                        "data load; 'off'/'none'/'0' forces a single pass (the "
                        "historical default); a positive integer pins the block.")
    g.add_argument("--pe_event_block", type=block_size_arg, default="auto",
                   metavar="N|auto|off",
                   help="Events per PE-reduction chunk. 'auto' (default) sizes from "
                        "free memory; 'off'/'none'/'0' keeps all events in one "
                        "vectorized block (fastest); a positive integer (e.g. 32) "
                        "bounds memory for dense dark-siren catalogs.")
    g.add_argument("--drop_full_catalog", type=str_to_bool, default=False, metavar="BOOL",
                   help="Discard the dense full-sky (npix, n_max_gals) galaxy arrays after "
                        "compacting to inference pixels, keeping only the compact PE/selection "
                        "views on device. Cuts startup GPU memory drastically for large nside. "
                        "Incompatible with --use_lss and bright-siren models, which need the "
                        "full-sky rows.")
    add_normalization_grid_arguments(g)
    g.add_argument("--kernel_gl_nodes", type=int, default=None, metavar="N",
                   help="Gauss-Legendre nodes for the per-galaxy kernel normalisation Z_i "
                        "(default 24). Spectroscopic catalogs (sigma_eff <~ 5e-3) are exact "
                        "to likelihood precision at 4-8 nodes and the quadrature dominates "
                        "wide-sky dark-siren runs; do NOT reduce for broad photo-z kernels.")
    g.add_argument("--kde_window", type=int, default=None, metavar="W",
                   help="Static window size for the per-sample catalog KDE: only the W "
                        "galaxies nearest each sample's redshift are evaluated (rows are "
                        "z-sorted at load). Default 1024 holds |delta log p_cat| < 1e-6 "
                        "against the full row across the ENTIRE sampled sigma_kde prior "
                        "[0, 0.05] on a review-scale 2113-galaxy row (~2x wall-clock, "
                        "~2x peak transient on that shape); denser rows need a larger W — "
                        "size it with darksirens.redshift.catalog.recommended_kde_window. "
                        "0 disables windowing entirely (full-row escape hatch for A/B "
                        "validation).")
    g.add_argument("--kde_window_nsigma", type=float, default=None, metavar="X",
                   help="Half-width multiplier for the KDE window: contributing galaxies "
                        "are located within +/- X * max_row(sigma_eff) of the sample "
                        "(default 8). The half-width is traced (sigma_kde is sampled); "
                        "only the window SIZE W is static.")
    g.add_argument("--row_chunk", default=None, metavar="auto|off|N",
                   help="Row-chunking for catalog kernel-state builds "
                        "(lax.map over N-row chunks instead of one full vmap). "
                        "'auto' (the default) chunks only above an internal size "
                        "threshold; 'off' disables; an integer forces that chunk "
                        "size. Measured on a DESI-scale slice on CPU, a forced "
                        "512 was 5.1x faster and 1.7x lighter in peak RSS than "
                        "the auto default -- for CPU-heavy catalog work, try "
                        "--row_chunk 512. Trace-time config (applied before the "
                        "first likelihood evaluation).")
    return optp


def _normalize_multitracer_paths(opts):
    # ── Multitracer survey-path normalization ──────────────────────
    # --survey_path / --lss_completion accept multiple values (nargs="+"); a
    # single path reproduces the legacy single-catalog behaviour exactly.  We
    # keep the aligned lists (opts.survey_paths / opts.lss_completions) for the
    # mixture loader and re-scalarize opts.survey_path / opts.lss_completion so
    # every single-catalog reader (loaders, catalogs/lss, io/results) is
    # untouched.
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    survey_paths = _as_list(opts.survey_path)
    opts.survey_paths = survey_paths
    opts.n_catalogs = max(1, len(survey_paths))

    lss_values = _as_list(opts.lss_completion)
    if opts.n_catalogs == 1:
        valid_counts = (0, 1)
    else:
        # With K catalogs the completion tables MUST be spelled out per catalog:
        # broadcasting one table across catalogs would silently apply e.g. a
        # galaxy Q table to the AGN catalog whenever the nsides happen to match.
        valid_counts = (0, opts.n_catalogs)
    if len(lss_values) not in valid_counts:
        _fatal(
            "--lss_completion must have "
            + ("0 or 1 entries" if opts.n_catalogs == 1
               else f"0 or exactly n_catalogs={opts.n_catalogs} entries")
            + f" (got {len(lss_values)}); use \"\" as a placeholder for a "
            "catalog with no external completion."
        )
    opts.lss_completions = [v if v not in (None, "") else None for v in lss_values]
    opts.survey_path = survey_paths[0] if survey_paths else None
    opts.lss_completion = opts.lss_completions[0] if opts.lss_completions else None

    _normalize_selection_fit_paths(opts)


def _normalize_selection_fit_paths(opts):
    """--selection_fit -> opts.selection_fit_paths, one entry per catalog.

    Comma-separated, positionally aligned with --survey_path; "" is a
    placeholder for a catalog with no fit.  A single path is NEVER broadcast
    across K catalogs: each catalog's magnitude fit is a property of ITS
    survey's depth and luminosity function, and sharing one would pin every
    catalog's completeness base to the first catalog's selection -- the same
    reasoning that makes --lss_completion per catalog.
    """
    raw = getattr(opts, "selection_fit", None)
    if raw in (None, ""):
        opts.selection_fit_paths = [None] * opts.n_catalogs
        # Still assert the map pairing: --stratum_map with NO fit anywhere
        # must be refused pre-load too, or the docstring's "before any
        # catalog is read" promise silently narrows to fit-carrying runs.
        _check_stratum_map_pairing(opts)
        return
    entries = [s.strip() for s in str(raw).split(",")]
    if len(entries) != opts.n_catalogs:
        _fatal(
            f"--selection_fit has {len(entries)} comma-separated entries but "
            f"n_catalogs={opts.n_catalogs}; pass exactly one magnitude fit "
            "per --survey_path, in the same order (use \"\" as a placeholder "
            "for a catalog with no fit). A single fit is not broadcast: it "
            "would pin every catalog's completeness base to the first "
            "catalog's selection."
        )
    opts.selection_fit_paths = [e or None for e in entries]
    _check_stratum_map_pairing(opts)


def _check_stratum_map_pairing(opts):
    """--stratum_map is legal only alongside a SINGLE-catalog magnitude fit.

    Both checks are pure opts arithmetic, so they run here (pre-load, first
    thing in ``main``) rather than at fit-resolution time: a mixture must be
    refused before any catalog is read, not after.  Which strata a fit
    actually carries is only knowable once the JSON is opened, so the
    single-stratum-fit-with-a-map direction stays in ``_resolve_selection_fits``.
    """
    if not getattr(opts, "stratum_map", None):
        return
    if not any(p is not None for p in getattr(opts, "selection_fit_paths", ())):
        _fatal("--stratum_map without --selection_fit: the stratum map only "
               "routes a multi-stratum selection fit's per-stratum curves.")
    if opts.n_catalogs >= 2:
        # The full-sky map is attached to the SHARED data bundle while a K>=2
        # mixture builds each catalog's views from its OWN bundle
        # (likelihood/factory.py -> prepare_catalog_views), so the map would
        # never reach catalog k's EMCatalog and its per-stratum empty-pixel
        # budget would be missing.
        _fatal("--stratum_map is single-catalog: a K>=2 mixture builds each "
               "catalog's views from its own bundle, so the shared full-sky "
               "stratum map never reaches catalog k and its per-stratum "
               "empty-pixel budget would be missing. Use single-stratum fits "
               "for a mixture, or run the stratified fit single-catalog.")


# ── Latent-field mode (field-level PR-5, the seam) ─────────────────────────────

def latent_field_mode(opts) -> bool:
    """True when this run generates Q from the anchor artifact.

    ONE reader for the mode string, so every latent branch in this module is a
    static Python-level test on the same predicate and table mode stays
    textually untouched.  ``getattr`` default "table" keeps every caller that
    builds an opts namespace without the flag (tests, the lensing CLI's
    borrowed helpers) on the shipped path.
    """
    return str(getattr(opts, "lss_field_mode", "table") or "table") == "latent"


def _check_latent_field_mode(opts):
    """Guard 6 (PLAN 4.4): the c_mode / exclusivity gate for latent mode.

    Pure opts arithmetic, so it runs pre-load in ``main`` alongside
    ``_check_stratum_map_pairing``: a run whose completion model is
    over-specified must be refused before any catalog, artifact or depth map is
    opened.  Every branch below is inside ``if latent``, so a table run reaches
    exactly one ``getattr`` and returns.

    The five refusals, and why each is a refusal rather than a preference:

    * ``latent + --c_mode per_pixel`` -- CIRCULAR.  The per-pixel matched-kernel
      ratio absorbs the observed angular clustering INTO C, so (1 - C) already
      contains the missing-galaxy structure the latent field exists to model;
      running both makes the run's estimand unreadable, and no choice of b_GW
      recovers the intended one.  (``redshift/completion.py`` refuses the same
      pairing at likelihood-build time; this is the same gate said early enough
      that the operator sees it before a 20-minute load.)
    * ``latent + aggregate|selection`` WITHOUT ``--per_pixel_completeness`` --
      the sky moments the artifact ships, B(z; b) = Sum_p f_p e^{b f_p-weighted
      field}, CARRY f_p, so rho conserves an f_p-weighted budget.  A run whose
      completion side has no f_p would consume a DIFFERENT budget than the one
      rho normalizes, and the exact conservation of PLAN eq. (4) -- the single
      property that makes the seam safe -- would silently stop holding.
    * ``latent + --lss_completion`` -- two completion models.  A resident log-Q
      table and a generated one are different objects built from different
      conditioning; silently preferring either makes the estimand unreadable.
    * ``latent + --use_lss`` -- inherits the existing ``field_lss_q`` vs
      ``field_delta_g`` exclusion (``redshift/completion.py:1698``): Q and the
      local-overdensity factor are two spellings of the same modulation, and
      latent mode supplies Q.  This one also protects the b_miss inversion
      below: with --use_lss off, b_miss reaching the likelihood can only mean
      b_GW.
    * ``latent + stratified selection`` -- each stratum has its own C_sel(z),
      hence its own (A, B) moment pair and its own rho; the artifact carries
      ONE pair.  ``--stratum_map`` is the pre-load proxy (a fit's stratum count
      is only knowable once its JSON is opened), and
      ``_check_latent_selection_strata`` closes the multi-stratum-fit-without-
      a-map direction after the fits are read.
    """
    if not latent_field_mode(opts):
        # Table mode: --lss_field_artifact would be silently ignored, which is
        # the failure mode where an operator believes they ran the seam and did
        # not.  Refuse the orphan flag; that is the only latent-aware statement
        # a table run ever executes.
        if getattr(opts, "lss_field_artifact", None):
            _fatal(
                "--lss_field_artifact given without --lss_field_mode latent: "
                "in table mode Q comes from the resident log-Q table and the "
                "artifact is never read, so this run would silently be a plain "
                "table run. Pass --lss_field_mode latent, or drop the artifact."
            )
        # Same treatment for the two latent-only flags this PR adds: an option
        # that is parsed and then ignored is the failure mode where an operator
        # believes a guard is armed and it is not.  --lss_field_sha256 pins an
        # artifact table mode never reads; --allow_unanchored_budget waives a
        # guard that only exists in latent mode (table mode's Q table carries
        # its own n0/delta firewall, check_lss_completion_provenance, which
        # this flag does NOT and must not touch).
        if getattr(opts, "lss_field_sha256", None):
            _fatal(
                "--lss_field_sha256 given without --lss_field_mode latent: it "
                "pins the identity of --lss_field_artifact, which table mode "
                "never reads, so the pin would be checked against nothing. "
                "Pass --lss_field_mode latent, or drop the pin."
            )
        if getattr(opts, "allow_unanchored_budget", False):
            _fatal(
                "--allow_unanchored_budget given without --lss_field_mode "
                "latent: it waives PLAN 4.4 guard 5, which exists only in "
                "latent mode (the shell-total conditioning that makes the "
                "count channel carry zero information about log10n0/delta is a "
                "latent-mode construction). In table mode the corresponding "
                "firewall is check_lss_completion_provenance, which this flag "
                "does not and must not bypass. Drop the flag."
            )
        return

    artifact = getattr(opts, "lss_field_artifact", None)
    if not artifact:
        _fatal(
            "--lss_field_mode latent requires --lss_field_artifact: latent "
            "mode GENERATES Q from the anchor artifact's factored basis "
            "(row_fac, phi_z) and its theta-free sky moments (A, B); there is "
            "no default and no in-catalog fallback. Build one with "
            "darksirens_build_latent_field."
        )
    if not os.path.isfile(artifact):
        _fatal(f"--lss_field_artifact is not a file: {artifact}")

    # FORMAT of the guard-1 pin only; the COMPARISON needs the artifact open
    # and lives in _stamp_latent_artifact_fingerprint (and again at
    # likelihood-build time in likelihood/factory._latent_guard_fingerprint).
    # A typo'd pin should cost a second, not a load.
    _pin = getattr(opts, "lss_field_sha256", None)
    if _pin is not None:
        _p = str(_pin).strip().lower()
        if len(_p) != 64 or any(c not in "0123456789abcdef" for c in _p):
            _fatal(
                f"--lss_field_sha256 {_pin!r} is not a 64-character hex "
                "sha256 digest. Pass either the digest "
                "darksirens_build_latent_field printed ('[artifact] sha256 = "
                "...') or the guard-1 content digest this run prints under "
                "'LSS field artifact' (PLAN 4.4 guard 1)."
            )

    if opts.universe_model not in ("dark_sirens", "dark_sirens_complete"):
        _fatal(
            f"--lss_field_mode latent needs a galaxy-aware universe model "
            f"(dark_sirens / dark_sirens_complete); got "
            f"'{opts.universe_model}', which never evaluates the catalog "
            "missing-galaxy budget the latent Q modulates."
        )

    c_mode = str(getattr(opts, "c_mode", None) or "per_pixel")
    if c_mode == "per_pixel":
        _fatal(
            "--lss_field_mode latent is incompatible with --c_mode per_pixel: "
            "the per-pixel matched-kernel ratio absorbs the observed angular "
            "clustering INTO C, so (1 - C) already carries the missing-galaxy "
            "structure the latent field is there to model -- the two "
            "completion models would double-count each other and the run's "
            "estimand would be unreadable. Use --c_mode aggregate or "
            "--c_mode selection with --per_pixel_completeness (PLAN 4.4 "
            "guard 6)."
        )
    if not getattr(opts, "per_pixel_completeness", None):
        _fatal(
            f"--lss_field_mode latent with --c_mode {c_mode} requires "
            "--per_pixel_completeness: the artifact's sky moments "
            "B(z; b) = Sum_p f_p e^{b f} carry the per-pixel selection "
            "fraction f_p, so the budget normalizer rho conserves an "
            "f_p-weighted missing budget. Without f_p on the completion side "
            "the two sides would conserve different budgets and PLAN eq. (4) "
            "-- Sum_p (1 - f_p C(z)) Q_p(z) == Sum_p (1 - f_p C(z)) at every "
            "z, member and theta -- would silently stop holding. Pass the "
            "SAME depth map the artifact was built with (PLAN 4.4 guard 6)."
        )

    _lss = [p for p in (getattr(opts, "lss_completions", None) or []) if p]
    if _lss:
        _fatal(
            f"--lss_field_mode latent is incompatible with --lss_completion "
            f"({len(_lss)} table(s) given): a resident log-Q table and a "
            "latent-generated Q are two different completion models built "
            "from different conditioning, and silently preferring one would "
            "make the run's estimand unreadable. Drop --lss_completion "
            "(PLAN 4.4 guard 6)."
        )
    if bool(getattr(opts, "use_LSS", False)):
        _fatal(
            "--lss_field_mode latent is incompatible with --use_lss: the "
            "local-overdensity factor max(1 + alpha_miss b_miss delta_g, 0) "
            "and Q are two spellings of the SAME missing-galaxy modulation "
            "(the field_lss_q vs field_delta_g exclusion at "
            "redshift/completion.py:1698), and latent mode supplies Q. It is "
            "also what keeps the b_miss -> b_GW inversion unambiguous: with "
            "--use_lss off, a sampled b_miss in latent mode can only mean "
            "b_GW (PLAN 4.3, 4.4 guard 6)."
        )
    if getattr(opts, "stratum_map", None):
        _fatal(
            "--lss_field_mode latent is incompatible with --stratum_map "
            "(stratified selection): each stratum carries its own C_sel(z), "
            "hence its own theta-free moment pair (A_s, B_s) and its own "
            "budget normalizer rho_s, while the anchor artifact ships ONE "
            "(A, B) pair over the fitted footprint. Run the single-stratum "
            "fit, or stay in table mode (PLAN 4.4 guard 6)."
        )


def _check_latent_selection_strata(opts):
    """The post-load half of guard 6's stratified refusal.

    ``--stratum_map`` is refused pre-load, but a ``--selection_fit`` may carry
    several strata without a map ever being passed (the map only ROUTES them);
    that is only knowable once ``_resolve_selection_fits`` has opened the JSON.
    Same reason the single-stratum-fit-with-a-map direction lives there.
    """
    if not latent_field_mode(opts):
        return
    for _k, _strata in enumerate(
            getattr(opts, "selection_strata_by_catalog", None) or (), start=1):
        if _strata and len(_strata) > 1:
            _fatal(
                f"catalog {_k}: --selection_fit declares {len(_strata)} "
                "selection strata, but --lss_field_mode latent ships ONE "
                "theta-free moment pair (A, B) over the fitted footprint and "
                "therefore ONE budget normalizer rho. A stratified run needs "
                "one (A_s, B_s) pair per stratum, which the anchor artifact "
                "does not carry. Use the single-stratum fit, or stay in table "
                "mode (PLAN 4.4 guard 6)."
            )


def _check_latent_selection_family(opts):
    """PR-6a gate list: latent mode refuses ``selection_family='schechter'``.

    Not a capability gap -- the seam is family-agnostic, it consumes whatever
    ``C_sel(z)`` the completion side forms -- but a DERIVATION gap, and the plan
    is explicit about which (PLAN §3, "(F2) caveat", and §7's PR-6a entry:
    "refuses ... schechter (until PR-2 clears it)").

    The latent likelihood multiplies a count factor whose base is
    ``f_p C(z; theta_sel) Nbar`` by a GW factor that also reads ``C``, and the
    two may be used together only because the chain-rule factorization
    ``p({z},{pix},{m}) = p({z}) p({pix}|{z}) p({m}|{z},{pix})`` makes them
    disjoint.  That argument was re-derived for ``_fit_gaussian_truncated``
    (``redshift/selection.py:675``), a clean conditional density
    ``p(Mhat_i | z_i, T_i)`` with ``sigma(M0hat) = 1.6e-4`` mag.  It has NOT
    been re-derived for ``_fit_schechter_truncated`` (``:725``), which master
    added along with per-catalog fits, K>=2 homogeneous-Schechter mixtures and
    ``M_faint_offset`` -- and ``M_faint_offset`` is the awkward one, because it
    never enters the fit likelihood while ``m_faint_cut`` is the fit-side
    truncation, so the schechter base and the run's completeness truncation are
    not the same object by construction.  Until PR-2 does that work, a schechter
    latent run would be a number nobody can defend, so it is refused rather than
    warned about.

    Post-load, because the family is a property of the ``--selection_fit``
    JSON: ``_resolve_selection_fits`` stamps ``opts.selection_family`` from the
    fits (scalar; a mixture must be homogeneous), and it is ``"gaussian"``
    whenever no fit is given -- which is why the wide-open ablation and every
    ``c_mode=aggregate`` run pass this guard untouched.
    """
    if not latent_field_mode(opts):
        return
    fam = str(getattr(opts, "selection_family", None) or "gaussian")
    if fam != "gaussian":
        _fatal(
            f"--lss_field_mode latent is incompatible with the '{fam}' "
            "luminosity-function family: the disjointness argument that lets "
            "the latent count channel coexist with an anchored theta_sel prior "
            "was re-derived for the truncated-Gaussian fit only "
            "(redshift/selection.py:675). The Schechter branch (:725) adds "
            "M_faint_offset, which never enters the fit likelihood while "
            "m_faint_cut is the fit-side truncation, so the fixed base and the "
            "run's completeness truncation are not the same object by "
            "construction -- and PLAN PR-2 owes that derivation before latent "
            "mode may admit it. Refit the catalog under the gaussian family, "
            "or stay in table mode (PLAN §3 '(F2) caveat', §7 PR-6a)."
        )


#: The two budget parameters PLAN §4.4 guard 5 names, plus their per-catalog
#: suffixed forms (``log10n0_c2`` ...).  ``theta_sel`` is the third name in the
#: guard, and it is already anchored by construction: ``inference/prior.py``
#: flips M0hat/sigma_M to ``("normal", loc, scale)`` from the offline fit's
#: covariance whenever ``--selection_fit`` is given, and a latent
#: ``c_mode=selection`` run without a fit is the deliberate wide-open ablation
#: the codebase already warns about.  So the guard below is written for the two
#: that have NO anchoring machinery of their own.
_LATENT_BUDGET_PARAMS = ("log10n0", "delta")


def _check_latent_budget_anchor(opts, labels, prior_kinds,
                                fixed_parameter_values):
    """PLAN §4.4 guard 5, the budget-anchor guard, restated per rung (v4).

    **The statement.**  At RUNG 0 -- which is exactly what PR-6a ships -- the
    count channel is conditioned on the OBSERVED shell totals, so by
    construction (``prop:cancel``) it carries ZERO information about
    ``(n0, delta, theta_sel)``: the shell multinomial's normalizer cancels the
    monopole.  A FLAT prior on ``log10n0`` or ``delta`` in latent mode is
    therefore not "marginalising over an uncertain budget"; the posterior in
    those directions IS the prior, and the missing-galaxy budget -- which is
    the whole coupling between the field and H0 -- is set by whatever prior
    volume happened to be declared.  Any H0 shift read off such a run is an
    artifact of the declared volume, the same class of defect as K10.  The plan
    says it in one line: "A **flat** prior on ``log10n0``/``delta`` in latent
    mode is refused at every rung; the calibration prior (OWNER DECISION 6) or
    ``--allow_unanchored_budget`` is required."

    **Per rung, so the guard survives the ladder.**  Rung 1 (PR-6b) makes the
    count channel carry the within-shell RESIDUAL information about
    ``(delta, theta_sel)`` -- the same object P7e measures -- so the refusal
    stays but its justification becomes an overlap bound rather than an
    identity; rung 2 restores the shell totals and K10 applies outright.  The
    refusal is unchanged at all three, which is why it is written once here.

    **What counts as anchored.**  Three admissible configurations:

    * the label is PINNED (absent from ``labels`` because
      ``--fixed_parameter_values`` fixed it) -- how the production line runs it,
      at the ``experiments/desi_ingest/calibrate_n0.py`` fit;
    * the label is SAMPLED under a NORMAL prior -- the calibration prior of
      OWNER DECISION 6 proper, the same machinery ``--selection_fit`` uses for
      ``theta_sel`` (``inference/prior.py``: ``kind_map[lbl] = ("normal", ...)``);
    * the label is INERT for this universe model and never sampled at all.

    Everything else -- sampled with ``("uniform", ...)`` -- is refused, unless
    ``--allow_unanchored_budget`` is passed, and then the run says out loud what
    the resulting posterior means.

    A pin at the DECODER DEFAULT rather than at the calibration fit is
    permitted but WARNED: it is still a plug-in, and an arbitrary one.  Table
    mode never reaches any of this; its Q table has its own n0/delta firewall
    (``inference/q_provenance.py``), which latent mode retires for want of a
    referent (§4.4).
    """
    if not latent_field_mode(opts):
        return
    import re as _re

    kinds = {lbl: knd for lbl, knd in zip(labels, prior_kinds or ())}
    fixed = dict(fixed_parameter_values or {})
    # ``log10n0`` and ``log10n0_c2`` alike; the suffix is the per-catalog form
    # (inference/prior.py appends ``_c{k}`` for catalogs 2..K).
    _budget = _re.compile(r"^(%s)(_c\d+)?$" % "|".join(_LATENT_BUDGET_PARAMS))
    unanchored, plugged_at_default = [], []
    for lbl in labels:
        if not _budget.match(lbl):
            continue
        if str(kinds.get(lbl, ("uniform",))[0]) == "uniform":
            unanchored.append(lbl)
    for name in _LATENT_BUDGET_PARAMS:
        for lbl in [name] + [f"{name}_c{k}" for k in
                             range(2, int(getattr(opts, "n_catalogs", 1)) + 1)]:
            if lbl not in labels and lbl not in fixed:
                plugged_at_default.append(lbl)

    if unanchored:
        msg = (
            f"--lss_field_mode latent samples {', '.join(unanchored)} under a "
            "FLAT prior. PLAN 4.4 guard 5 refuses that at every rung. At rung "
            "0 -- what PR-6a ships -- the count channel is conditioned on the "
            "observed shell totals, so it carries ZERO information about "
            "(log10n0, delta, theta_sel) BY CONSTRUCTION: the posterior in "
            "these directions is exactly the prior you declared, and the "
            "missing-galaxy budget they set -- the only channel through which "
            "the latent field reaches H0 -- is then a statement about the "
            "prior volume rather than about the data. An H0 shift measured "
            "this way is an artifact of the fiducial calibration, the same "
            "class of defect as kill criterion K10. Fix them with "
            "--fixed_parameter_values at the calibration fit "
            "(experiments/desi_ingest/calibrate_n0.py; OWNER DECISION 6), give "
            "them a normal calibration prior, or pass "
            "--allow_unanchored_budget for a deliberate prior-sensitivity "
            "ablation."
        )
        if not getattr(opts, "allow_unanchored_budget", False):
            _fatal(msg)
        _warn(
            "--allow_unanchored_budget: " + msg + " RUNNING ANYWAY -- treat "
            "every H0 number from this run as a prior-volume ablation, not a "
            "measurement."
        )
    for lbl in plugged_at_default:
        _warn(
            f"latent mode: {lbl} is neither sampled nor pinned by "
            "--fixed_parameter_values, so the likelihood decoder plugs it at "
            "its package fiducial. That is a legal budget anchor for PLAN 4.4 "
            "guard 5 (it is not a flat prior), but it is an ARBITRARY one: the "
            "shell-total conditioning gives the data no say in it, so the "
            "missing-galaxy budget is whatever the fiducial says. The "
            "production line pins both to "
            "experiments/desi_ingest/calibrate_n0.py's fit instead."
        )


def _stamp_latent_artifact_fingerprint(opts):
    """PLAN §4.4 successor 1 at the CLI, so the digests reach settings.json.

    The authoritative check is at likelihood-build time
    (``likelihood/factory._latent_guard_fingerprint``, where §4.4 places every
    successor).  This call runs the SAME guard pre-load for two reasons that
    are not redundancy: a bad artifact should cost the operator a second rather
    than a catalog load, and ``settings.json`` is written BEFORE the likelihood
    is built, so the only way the run's provenance record can carry the
    fingerprint is to compute it here.

    In latent mode that record is the whole provenance surface.  Table mode
    writes a ``Q_LSS build fiducials`` block into the loaded-data report and
    enforces it through ``check_lss_completion_provenance``; latent mode has no
    table, so what stands in its place is exactly these two digests plus the
    anchor ``theta_ref`` and ``b_gal`` they cover.
    """
    if not latent_field_mode(opts):
        return
    from darksirens.likelihood.factory import _latent_guard_fingerprint

    try:
        fp = _latent_guard_fingerprint(opts.lss_field_artifact, opts)
    except (ValueError, OSError, KeyError) as exc:
        _fatal(str(exc))
        return
    # Onto opts so save_settings_json picks them up with every other option:
    # a latent run's settings.json then names the exact field it consumed.
    opts.lss_field_stored_sha256 = fp["stored"]
    opts.lss_field_content_sha256 = fp["content"]
    opts.lss_field_theta_ref = fp["theta_ref"]
    opts.lss_field_b_gal = fp["b_gal"]
    _ok(f"latent artifact sha256   →  {fp['stored']}")
    _ok(f"latent guard-1 content   →  {fp['content']}")
    _ok(f"latent anchor theta_ref  →  {json.dumps(fp['theta_ref'], sort_keys=True)}"
        f"  b_gal={fp['b_gal']}")


def _resolve_catalog_sky_weighting(opts):
    # ── Catalog sky-weighting resolution (estimand choice) ─────────
    # The two dark-siren normalization conventions are DIFFERENT estimands:
    #   field       = the JOINT catalog host-density estimand (DEFAULT, all K).
    #                 The per-pixel numerator N_obs*p_cat + dN_miss is normalized
    #                 by the survey-GLOBAL Z(theta), so RELATIVE angular host
    #                 density is preserved -- a pixel with 100 candidate hosts
    #                 carries ~100x the angular weight of a pixel with 1.  For
    #                 K>=2 the per-catalog Z_k additionally turns the mixture
    #                 weight fcat_k into the host FRACTION (number-density /
    #                 sky-clustering contrast).
    #   conditional = the radial-only LEGACY estimand.  Each pixel is normalized
    #                 by its OWN Z[pix]=N_obs+N_miss, so every pixel integrates to
    #                 unit mass and RELATIVE angular host density is DISCARDED
    #                 (kept for reproducing older single-catalog runs, and as a
    #                 pure radial-shape estimand).  At K>=2 it strips the
    #                 number-density channel from the mixture weight and fcat
    #                 rails on clustered catalogs (measured:
    #                 tests/test_multitracer_field_recovery.py), so it stays fatal
    #                 there for dark_sirens.
    # Unset resolves to field for EVERY K of the galaxy-aware models.  At K=1
    # field's dedicated number-density channel -- the survey-global constant
    # log_Z_global -- cancels between the PE and selection terms, so log10n0 is
    # only weakly identified there (through the completeness ratio C(z;n0) and
    # the missing-branch shape in the per-pixel numerator) and otherwise
    # marginalizes against its prior; that is acceptable (NOTE (K=1) in
    # darksirens/redshift/prior.py).  What field DOES restore at K=1 is the
    # relative angular host weighting the conditional per-pixel normalizer
    # discards.  dark_sirens_complete keeps its own pre-existing rules (K>=2
    # requires field, checked below; K=1 allows both).
    # --drop_full_catalog is INERT for a K>=2 mixture: inference/data.py stubs the
    # top-level catalog and returns before maybe_drop_full_catalog, while
    # load_multitracer_catalog_bundles already loads every catalog HOST-side
    # (to_device=False) and keeps only the compact per-bundle views.  Left to the
    # resolution below, the flag would buy the legacy 'conditional' estimand --
    # not a host-fraction estimand at K>=2 (fcat_k rails; see the refusal below)
    # -- in exchange for no memory saving at all, and on the AUTO path that
    # refusal never fired.  Refuse the combination instead.
    if opts.n_catalogs >= 2 and getattr(opts, "drop_full_catalog", False):
        _fatal(
            "--drop_full_catalog has no effect on a K>=2 mixture: every catalog "
            "is already loaded host-side and kept as compact per-bundle views "
            "(inference/loaders.py: load_multitracer_catalog_bundles), so the "
            "flag would only resolve an unset --catalog_sky_weighting to the "
            "legacy 'conditional' estimand, whose fcat_k carries no "
            "number-density / sky-clustering information and rails to 1 on "
            "clustered sparse-contrast catalogs. Drop the flag."
        )
    opts.catalog_sky_weighting_source = (
        "explicit" if opts.catalog_sky_weighting is not None else "auto"
    )
    if opts.catalog_sky_weighting is None:
        if opts.universe_model not in ("dark_sirens", "dark_sirens_complete"):
            # Non-catalog models never evaluate the catalog prior; keep the
            # inert legacy value so the field-scope validation stays quiet.
            opts.catalog_sky_weighting = "conditional"
        elif getattr(opts, "drop_full_catalog", False):
            # field needs the full-sky rows to count empty pixels; honor the
            # explicit memory request and keep the legacy estimand (an
            # EXPLICIT field + drop_full_catalog request stays fatal below).
            _warn(
                "--drop_full_catalog requested: resolving unset "
                "--catalog_sky_weighting to the legacy 'conditional' estimand "
                "(the 'field' default needs the full-sky catalog rows). Drop "
                "the flag to use the joint host-density estimand."
            )
            opts.catalog_sky_weighting = "conditional"
        else:
            opts.catalog_sky_weighting = "field"
    if (opts.universe_model == "dark_sirens"
            and opts.n_catalogs == 1
            and opts.catalog_sky_weighting == "field"):
        # Informational (never fatal): field is the default single-catalog
        # estimand, but its log10n0 channel is only weakly identified at K=1.
        _warn(
            "K=1 field sky-weighting: log10n0 is only weakly identified "
            "(the survey-global normalizer cancels between the PE and selection "
            "terms); it marginalizes against its prior. Field still restores the "
            "relative angular host weighting the legacy 'conditional' estimand "
            "discards. Pass --catalog_sky_weighting conditional for the "
            "radial-only legacy estimand."
        )
    # Provenance-INDEPENDENT: the estimand is wrong at K>=2 however it was
    # chosen, so gating this on source == "explicit" only hid the auto paths
    # (--drop_full_catalog above, and any future one) behind no diagnostic.
    if opts.universe_model == "dark_sirens":
        if opts.n_catalogs >= 2 and opts.catalog_sky_weighting == "conditional":
            _fatal(
                "--catalog_sky_weighting conditional is not a host-fraction "
                "estimand for a K>=2 mixture: each catalog's prior is "
                "normalized per pixel, so fcat_k carries no number-density / "
                "sky-clustering information and rails to 1 on clustered "
                "sparse-contrast catalogs (measured: "
                "tests/test_multitracer_field_recovery.py). Omit the flag "
                "(K>=2 resolves to field), which makes fcat_k the catalog-k "
                "GW host fraction."
            )

    if opts.n_catalogs >= 2:
        _sky_w = getattr(opts, "catalog_sky_weighting", "conditional")
        if opts.universe_model == "dark_sirens_complete":
            # Complete-model mixture is coherent only under the field
            # (survey-global) normalizer; the conditional (per-pixel) normalizer
            # would mix per-pixel-normalized complete priors (incoherent estimand).
            if _sky_w != "field":
                _fatal("A K>=2 mixture with --universe_model dark_sirens_complete "
                       "requires --catalog_sky_weighting field; the conditional "
                       "(per-pixel) normalizer makes the complete-model mixture an "
                       "incoherent estimand (mixing per-pixel-normalized complete "
                       "priors).")
        elif opts.universe_model != "dark_sirens":
            _fatal("Multiple --survey_path catalogs (a K>=2 mixture) require "
                   "--universe_model dark_sirens, or dark_sirens_complete with "
                   "--catalog_sky_weighting field. spectral_sirens / "
                   "bright_sirens are inherently "
                   "single-catalog (catalog-free or counterpart-pinned) and "
                   "have no K>=2 mixture.")
        # Marked-host models compose with the mixture via PER-CATALOG eta
        # blocks (eta_<mark> for catalog 1, eta_<mark>_c{k} for catalogs
        # 2..K).  Resolve each catalog's marks from its FILE's datasets here
        # (pre-load), so the bundle loader and the parameter space see the
        # same ordered per-catalog lists.  mark_model == none still needs the
        # empty defaults (the K=1 data-based resolution below is skipped for
        # a mixture).
        opts.mark_names = ()
        opts.mark_names_by_catalog = ((),) * opts.n_catalogs
        if getattr(opts, "mark_model", "none") not in (None, "none"):
            import h5py as _h5py
            from darksirens.marks import MARK_FIELDS as _MF_K2

            _req = tuple(
                s.strip() for s in (opts.marks or "").split(",") if s.strip()
            ) or None
            _by_catalog = []
            for _p in opts.survey_paths:
                try:
                    with _h5py.File(_p, "r") as _f:
                        _avail = tuple(
                            n for n, ds in _MF_K2.items() if ds in _f
                        )
                except OSError as _exc:
                    _fatal(f"Cannot open --survey_path '{_p}' to resolve "
                           f"galaxy marks: {_exc}")
                _by_catalog.append(
                    tuple(n for n in (_req or _avail) if n in _avail)
                )
            if _req:
                _missing_everywhere = [
                    m for m in _req
                    if all(m not in names for names in _by_catalog)
                ]
                if _missing_everywhere:
                    _fatal(f"--marks requested {_missing_everywhere} but no "
                           "catalog provides them.")
            if all(not names for names in _by_catalog):
                _fatal("--mark_model loglinear requires per-galaxy marks, but "
                       "none of the catalogs provide any (expected datasets "
                       "like MARK_LOGMSTAR/MARK_LOGSSFR).")
            opts.mark_names_by_catalog = tuple(_by_catalog)
            opts.mark_names = _by_catalog[0]
            for _i, _names in enumerate(_by_catalog, start=1):
                if _names:
                    print(f"  [multitracer] catalog {_i} marks: "
                          f"{', '.join(_names)}")
                else:
                    print(f"  [multitracer] catalog {_i} has NO selected "
                          "marks -> plain galaxy-count host model (h = 1) "
                          "inside the mixture.")
        if opts.counterpart is not None:
            _fatal("--counterpart is not supported with a multi-catalog mixture.")


def _validate_multitracer_config(opts):
    # FIELD-convention sky weighting scope (host-fraction estimand).  The
    # missing-galaxy budget modulations (deterministic Q_LSS, use_LSS delta_g)
    # and the marked-host model ARE supported: the survey-global normalizer
    # carries the SAME per-pixel budget as the numerator (field_global_log_Z /
    # field_global_log_Z_marked; per-member normalizers under
    # --lss_marginalize).
    if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
        if opts.universe_model not in ("dark_sirens", "dark_sirens_complete"):
            _fatal("--catalog_sky_weighting field supports --universe_model "
                   "dark_sirens or dark_sirens_complete only (got "
                   f"'{opts.universe_model}').")
        if getattr(opts, "drop_full_catalog", False):
            _fatal("--catalog_sky_weighting field needs the full-sky catalog rows to "
                   "count empty pixels; it is incompatible with --drop_full_catalog.")

    # Marked-host model scope: the eta reweighting enters the dark_sirens
    # catalog prior only (prepare_redshift_prior_state ignores marks for every
    # other model), so any other universe_model would silently sample phantom
    # flat eta dimensions with zero likelihood effect.
    if (getattr(opts, "mark_model", "none") not in (None, "none")
            and opts.universe_model != "dark_sirens"):
        _fatal(f"--mark_model {opts.mark_model} requires --universe_model "
               f"dark_sirens (got '{opts.universe_model}'): marks reweight the "
               "dark-siren catalog prior and are inert for every other model, "
               "leaving phantom flat eta dimensions in the sampled space.")


def _canonicalize_fixed_flags(opts):
    # Persist the canonical names in settings while keeping opts.fix_cosmology
    # for backward-compatible internal callers and saved metadata.
    opts.fixed_cosmology = bool(opts.fix_cosmology)
    opts.fixed_de = bool(opts.fix_de)
    opts._fixed_de_superseded = bool(opts.fix_cosmology and opts.fix_de)
    if opts._fixed_de_superseded:
        opts.fix_de = False
        opts.fixed_de = False


def _configure_performance_grids(opts):
    # Normalisation grids + the PHY-5 pairing-grid coverage guard live in
    # cli.common so the lensing CLI runs exactly the same resolution: the pairing
    # grid is a process-global env-configurable setting that BOTH CLIs inherit,
    # and it was guarded in this twin only.
    configure_normalization_grids_for_model(opts)

    if opts.kernel_gl_nodes is not None:
        from darksirens.redshift.catalog import configure_kernel_quadrature
        if opts.kernel_gl_nodes < 2:
            _fatal("--kernel_gl_nodes must be >= 2")
        configure_kernel_quadrature(opts.kernel_gl_nodes)

    kde_window = getattr(opts, "kde_window", None)
    kde_nsigma = getattr(opts, "kde_window_nsigma", None)
    if kde_window is not None or kde_nsigma is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        kw = {}
        if kde_window is not None:
            if kde_window < 0:
                _fatal("--kde_window must be >= 0 (0 disables windowing)")
            kw["size"] = None if kde_window == 0 else kde_window
        if kde_nsigma is not None:
            kw["n_sigma"] = kde_nsigma
        try:
            configure_catalog_kde_window(**kw)
        except ValueError as e:
            _fatal(str(e))

    row_chunk = getattr(opts, "row_chunk", None)
    if row_chunk is not None:
        from darksirens.redshift.catalog import configure_catalog_row_chunk
        text = str(row_chunk).strip().lower()
        try:
            if text == "auto":
                configure_catalog_row_chunk("auto")
            elif text in ("off", "none", "0"):
                configure_catalog_row_chunk(None)
            else:
                configure_catalog_row_chunk(int(text))
        except ValueError as e:
            _fatal(f"--row_chunk {row_chunk!r}: {e}")


def _parse_structured_options(opts):
    prior_overrides        = parse_json_arg(opts.prior_overrides,        "prior_overrides")
    fixed_parameter_values = parse_json_arg(opts.fixed_parameter_values, "fixed_parameter_values")
    opts.prior_overrides        = prior_overrides if prior_overrides else None
    opts.fixed_parameter_values = (
        fixed_parameter_values if fixed_parameter_values else None
    )
    opts.counterpart            = parse_counterpart_arg(opts.counterpart)
    return prior_overrides, fixed_parameter_values


def _resolve_sampler_config(opts):
    if opts.sampler == "tinyns":
        try:
            build_tinyns_config(opts)
        except ValueError as e:
            _fatal(str(e))

    # Fail fast on a malformed --checkpoint_interval / --resume here, before the
    # data load: a checkpoint flag that only errors out (or silently does
    # nothing) once sampling starts is exactly the class of bug this feature
    # exists to remove.
    try:
        find_resume_target(opts, opts.sampler, name_prefix=_run_name_prefix(opts))
        parse_checkpoint_interval(opts.checkpoint_interval)
    except ValueError as e:
        _fatal(str(e))

    resolve_redshift_prior_barrier(opts)

    guard_mode, max_likelihood_variance = resolve_selection_neff_guard(opts)
    # Always print a compact, resolved guard summary (mode + why, cap, and the
    # criterion) so the log records why the likelihood may return -inf; the
    # cap and the soft-mode post-hoc caveat are folded into the one line.
    print(
        "  [i] "
        + format_selection_guard_summary(
            guard_mode,
            opts.sampler,
            opts.selection_neff_soft_guard,
            max_likelihood_variance,
        )
    )


def _apply_bright_siren_overrides(opts):
    if opts.universe_model == "bright_sirens":
        # Bright sirens use a synthetic one-object catalog fixed by the
        # counterpart rather than survey-completion hyperparameters.
        opts.fix_survey = True
        # The sky direction is pinned to the counterpart, so an anisotropic
        # source-rate model is not identifiable; force the isotropic null.
        if getattr(opts, "sky_model", "isotropic") != "isotropic":
            print(
                f"  [!] --sky_model '{opts.sky_model}' is not supported with "
                "universe_model 'bright_sirens'; forcing 'isotropic'."
            )
            opts.sky_model = "isotropic"


def _gw_file_requires_fixed_population(gw_path):
    """Whether a PE file declares that its population block must stay FIXED.

    True for the products of ``darksirens_skymaps_to_samples``: a 3D skymap is
    marginalised over masses and spins, so that converter fills those
    coordinates with surrogate draws from a broad proposal and sets ``p_pe``
    equal to the same proposal. They carry no information — the likelihood's
    mass/spin factors only cancel if the population is held fixed AND the
    injection set is reweighted to that same fixed model.

    Recognised from either marker, so files written before the explicit attr
    existed are still caught:

      * ``requires_fixed_population = True`` (written since this guard landed);
      * ``source == "darksirens_skymaps_to_samples"`` (always written).

    Any file that cannot be opened or read as HDF5 returns False: this is a
    marker probe, not a format check — ``load_gw_samples`` reports a missing or
    malformed PE file with its own, better message.
    """
    from darksirens.cli.skymaps_to_samples import SOURCE_TAG

    try:
        with h5py.File(gw_path, "r") as f:
            attrs = dict(f.attrs)
    except Exception:
        return False

    if bool(np.asarray(attrs.get("requires_fixed_population", False)).item()):
        return True
    source = attrs.get("source", "")
    if isinstance(source, bytes):
        source = source.decode("utf-8", "replace")
    return str(source) == SOURCE_TAG


def _validate_skymap_surrogate_population(opts):
    """Refuse to infer a population from skymap-surrogate mass/spin draws.

    Runs at configuration time, before the data load and long before sampling,
    so the failure costs seconds rather than a wasted allocation.
    """
    if not getattr(opts, "gw_path", None):
        return
    if not _gw_file_requires_fixed_population(opts.gw_path):
        return

    if getattr(opts, "fix_population", False):
        _ok(
            "GW file is a 3D-skymap surrogate product; population block is "
            "fixed, as it requires."
        )
        return

    message = (
        f"{opts.gw_path} was written by darksirens_skymaps_to_samples and "
        "declares requires_fixed_population. A 3D skymap is marginalised over "
        "masses and spins, so that file's m1det/m2det/chieff samples are "
        "ARTIFICIAL draws from a broad surrogate proposal, not posterior "
        "samples. With --fix_population false those surrogates drive both the "
        "population fit (which then measures the proposal) and the selection "
        "integral (whose mass/spin factors no longer cancel), biasing the "
        "cosmology that rides on them. Re-run with --fix_population true."
    )
    if str_to_bool(getattr(opts, "allow_skymap_population", False)):
        _warn(
            f"{message} Proceeding anyway under --allow_skymap_population: the "
            "population posterior from this run is NOT a measurement."
        )
        return
    _fatal(
        f"{message} Pass --allow_skymap_population to override for a "
        "deliberate methodological study."
    )


def _validate_run_config(opts):
    # ── Validation ─────────────────────────────────────────────────

    _section("Validating configuration")
    GALAXY_AWARE = {"dark_sirens", "dark_sirens_complete"}

    if bool(opts.gw_path) == bool(opts.gw_flows_path):
        _fatal(
            "Exactly one of --gw_path (stored PE samples) or --gw_flows_path "
            "(per-event flow surrogates) is required."
        )
    if bool(opts.gwselection_path) == bool(opts.pdet_flow_path):
        _fatal(
            "Exactly one of --gwselection_path (injection file) or "
            "--pdet_flow_path (P_det emulator) is required."
        )
    if opts.pdet_flow_path:
        # Orthogonal to the event source: valid with --gw_path or
        # --gw_flows_path alike.
        if not os.path.isfile(opts.pdet_flow_path):
            _fatal(f"--pdet_flow_path is not a file: {opts.pdet_flow_path}")
        if opts.pdet_nsamp <= 0:
            _fatal("--pdet_nsamp must be positive.")
        try:
            _pdet_h0, _pdet_om0 = (float(v) for v in
                                   str(opts.pdet_cosmology).split(","))
        except ValueError:
            _fatal("--pdet_cosmology must be 'H0,Om0' (e.g. '67.9,0.3065'); "
                   f"got {opts.pdet_cosmology!r}.")
        else:
            if _pdet_h0 <= 0.0 or not (0.0 < _pdet_om0 < 1.0):
                _fatal("--pdet_cosmology needs H0 > 0 and 0 < Om0 < 1; "
                       f"got {opts.pdet_cosmology!r}.")
        if not (0.0 < opts.pdet_chieff_amax <= 1.0):
            _fatal("--pdet_chieff_amax must be in (0, 1].")
    if opts.gw_flows_path:
        if not os.path.isdir(opts.gw_flows_path):
            _fatal(f"--gw_flows_path is not a directory: {opts.gw_flows_path}")
        if opts.universe_model != "spectral_sirens":
            _fatal(
                "--gw_flows_path currently supports --universe_model "
                "spectral_sirens only. Dark sirens need 6-D flows with "
                "(ra, dec) plus the per-pixel catalog redshift-prior sampler "
                "(scaffolded: columns registry in darksirens/gw/flows.py, "
                "z-target hook in likelihood/flow_events.py). "
                f"Got '{opts.universe_model}'."
            )
        if not opts.shared_gamma:
            _fatal("--gw_flows_path requires shared redshift evolution "
                   "(--shared_gamma true).")
        if not opts.shared_spin:
            _fatal("--gw_flows_path requires a shared spin component "
                   "(--shared_spin true).")
        if opts.sky_model != "isotropic":
            _fatal("--gw_flows_path supports --sky_model isotropic only.")
        if getattr(opts, "mark_model", "none") not in (None, "none"):
            _fatal("--gw_flows_path does not support a marked-host model.")
        if opts.survey_path:
            _fatal("--gw_flows_path (spectral sirens) does not use --survey_path.")
        if opts.flows_nsamp <= 0:
            _fatal("--flows_nsamp must be positive.")
        if not (0.0 <= opts.flows_wfull < 1.0):
            _fatal("--flows_wfull must be in [0, 1).")
        if opts.flows_wfull == 0.0:
            _warn(
                "--flows_wfull 0 disables the full-population-support mixture "
                "component: the event integral is then truncated to each "
                "event's empirical support box by a hyperparameter-dependent "
                "amount (issue #260). Evidence differences below that "
                "systematic are unresolved."
            )
        if not (0.0 < opts.flows_chieff_amax <= 1.0):
            _fatal("--flows_chieff_amax must be in (0, 1].")
        if opts.pdet_flow_path and abs(
                float(opts.flows_chieff_amax)
                - float(opts.pdet_chieff_amax)) > 1e-12:
            # The chi_eff PE prior is DIVIDED OUT of every flow density
            # (numerator) with --flows_chieff_amax and BAKED INTO the
            # pseudo-injection p_draw (denominator) with --pdet_chieff_amax.
            # Both sides marginalize the 4-D spin nuisance under the SAME
            # amax-truncated reference conditional, so two different amax
            # values leave a chi_eff- and q-dependent residual that does not
            # cancel between the events and the injections.
            _fatal(
                f"--flows_chieff_amax {opts.flows_chieff_amax} != "
                f"--pdet_chieff_amax {opts.pdet_chieff_amax}: the chi_eff PE "
                "prior divided out of each event flow (numerator) and the one "
                "baked into the emulator pseudo-injections' p_draw "
                "(denominator) must be the SAME reference convention, or the "
                "hierarchical ratio carries a chi_eff/mass-ratio-dependent "
                "reweighting of the selection function. Set both to the "
                "injection-file convention."
            )
        if opts.sampler == "numpyro":
            _warn(
                "flows + NumPyro NUTS: the grid inverse-CDF samplers give "
                "piecewise gradients; nested sampling is the validated path."
            )

    _validate_skymap_surrogate_population(opts)

    if opts.universe_model == "bright_sirens" and opts.counterpart is None:
        _fatal("'bright_sirens' requires --counterpart RA DEC Z triplet(s) (angles in radians).")
    if opts.universe_model != "bright_sirens" and opts.counterpart is not None:
        _warn("--counterpart is ignored unless --universe_model bright_sirens.")
    if not np.isfinite(opts.counterpart_dz) or opts.counterpart_dz <= 0.0:
        # argparse's ``type=float`` happily parses 'nan'/'inf', and a bare
        # ``<= 0`` lets both through to norm.logpdf() in redshift/prior.py.
        _fatal("--counterpart_dz must be a finite positive number.")
    if opts.counterpart_nside < 1 or not hp.isnsideok(opts.counterpart_nside):
        _fatal("--counterpart_nside must be a valid positive HEALPix NSIDE.")

    if opts.universe_model in GALAXY_AWARE and not opts.survey_path:
        _fatal(f"'{opts.universe_model}' requires --survey_path.")
    if opts.universe_model not in GALAXY_AWARE and opts.survey_path:
        _warn(f"--survey_path provided but '{opts.universe_model}' does not use it.")
    if getattr(opts, "_fixed_de_superseded", False):
        _warn(
            "--fix_cosmology supersedes --fix_de; "
            "dark energy is included in the fixed cosmology block."
        )
    if opts.fix_population and opts.fix_cosmology and opts.fix_survey:
        _warn("All blocks fixed — nothing will be inferred.")

    _ok("Configuration is valid.")
    _end()


def _print_run_configuration(opts, prior_overrides, fixed_parameter_values):
    # ── Run configuration printout ─────────────────────────────────

    _section("Run Configuration")
    _row("Universe model",   opts.universe_model)
    if opts.counterpart is not None:
        first_cp = opts.counterpart[0]
        _row("Counterparts", len(opts.counterpart))
        _row("First counterpart", f"ra={first_cp[0]:.8g}, dec={first_cp[1]:.8g}, z={first_cp[2]:.8g}")
        _row("Counterpart dz", opts.counterpart_dz)
        _row("Counterpart nside", opts.counterpart_nside)
        _row("Bright-siren sky marginalized", opts.bright_siren_sky_marginalized)
    _row("Population model", opts.pop_model)
    _row("Shared beta", "yes" if opts.shared_beta else "no")
    _row("Shared spin", "yes" if opts.shared_spin else "no")
    _row("Shared gamma", "yes" if opts.shared_gamma else "no")
    if opts.universe_model in {"dark_sirens_complete", "bright_sirens"}:
        _row("Empty-pixel policy", opts.complete_empty_pixel_policy)
    if opts.survey_path or opts.survey_paths:
        _row(
            "Survey z_depth override",
            opts.survey_z_depth if opts.survey_z_depth is not None
            else "none (per-catalog file attr, else legacy full-grid)",
        )
    if latent_field_mode(opts):
        # Printed only in latent mode: the table-mode configuration block is
        # byte-for-byte what it was.  Both rows are loud on purpose -- this is
        # a different parameter space, not a different Q backend.
        _row("LSS field mode", "latent (Q generated from the anchor artifact)")
        _row("  anchor artifact", opts.lss_field_artifact)
        # The provenance record.  In table mode this slot is the "Q_LSS build
        # fiducials" block of the loaded-data report, enforced by
        # check_lss_completion_provenance; latent mode retires that (no table,
        # no referent -- PLAN 4.4) and these two digests are what replaces it,
        # so they are printed with the same prominence.
        _row("  sha256 (stamped)",
             getattr(opts, "lss_field_stored_sha256", None) or "unknown")
        _row("  sha256 (guard-1 content)",
             getattr(opts, "lss_field_content_sha256", None) or "unknown")
        _row("  pinned by --lss_field_sha256",
             getattr(opts, "lss_field_sha256", None) or "no (identity unpinned)")
        _row("  anchor theta_ref",
             json.dumps(getattr(opts, "lss_field_theta_ref", None),
                        sort_keys=True))
        _row("  anchor b_gal", getattr(opts, "lss_field_b_gal", None))
        _row("  b_miss", "= b_GW, SAMPLED (PLAN 4.3 rule inversion)")
    print("  │")
    _row("Fix cosmology",    "yes" if opts.fix_cosmology  else "no")
    _row(
        "Fix dark energy",
        _format_fixed_dark_energy_summary(opts, fixed_parameter_values),
    )
    _row("Fix population",   "yes" if opts.fix_population else "no")
    _row("Fix survey",       "yes" if opts.fix_survey     else "no")
    _row("Prior overrides",  json.dumps(prior_overrides) if prior_overrides else "none")
    _row("Validate completion", "yes" if opts.validate_completion else "no")
    _row("Redshift prior barrier", opts.redshift_prior_barrier_resolved)
    if fixed_parameter_values:
        for lbl, val in fixed_parameter_values.items():
            _row(f"  fixed: {lbl}", val)
    else:
        _row("Fixed param values", "none")
    print("  │")
    _row("Sampler", opts.sampler)
    if opts.sampler in ("tinyns", "dynesty"):
        _row("  live points", opts.nlive)
        _row("  ΔlogZ stop",  opts.dlogz)
    if opts.sampler == "tinyns":
        cfg = opts.tinyns_resolved_config
        _row("  preset", cfg["preset"])
        _row("  sample", cfg["sample"])
        _row("  kernel", cfg["kernel"])
        _row("  bound", cfg["bound"])
        _row("  rwalk seed", cfg["rwalk_seed"])
        _row("  rwalk proposal", cfg["rwalk_proposal"])
        _row("  walks", cfg["walks"])
        _row("  step scale", cfg["step_scale"])
        _row("  min accepts", cfg["min_accepts"])
        _row("  repl. chains", cfg["replacement_chains"])
        _row("  chain sched.", cfg["replacement_chain_schedule"] or "none")
        _row("  max attempts", cfg["max_attempts"])
        _row("  jax block size", cfg["jax_block_size"])
        _row("  jax vectorized", cfg.get("jax_vectorized"))
        _row("  progress interval", cfg.get("progress_interval"))
        if cfg.get("checkpoint_path") or cfg.get("resume_from") or cfg.get("checkpoint_path_out"):
            _row("  checkpoint", cfg.get("checkpoint_path") or "none")
            _row("  resume from", cfg.get("resume_from") or "none")
            _row("  checkpoint out", cfg.get("checkpoint_path_out") or "none")
    if opts.sampler == "numpyro":
        _row("  warmup", opts.nuts_warmup)
        _row("  samples", opts.nuts_samples)
        _row("  chains", opts.nuts_chains)
        _row("  target accept", opts.nuts_target_accept)
        _row("  init tries", opts.nuts_init_tries)
        _row("  init seed offset", opts.nuts_init_seed_offset)
    _row("  seed", opts.seed)
    print("  │")
    norm_grid = normalization_grid_settings()
    _pairing_grid = (
        "exact" if norm_grid.pairing_m1_grid is None
        else f"grid({norm_grid.pairing_m1_grid}, m1<={norm_grid.pairing_m_hi:g})"
    )
    _row("Norm grids", (
        f"mass={norm_grid.n_mass}, q={norm_grid.n_q}, chi={norm_grid.n_chi}, "
        f"pairing={_pairing_grid}"
    ))
    _row("JAX backend", jax.default_backend())
    _row("JAX devices",  ", ".join(str(d) for d in jax.devices()))
    print("  │")
    if opts.gw_flows_path:
        _row("GW flows path", opts.gw_flows_path)
        _row("Flow draws J",  f"{opts.flows_nsamp:,} (seed {opts.flows_seed})")
        _row("Flow proposal", (
            f"window + {opts.flows_wfull:.3g} full support "
            f"(margin {opts.flows_support_margin:g})"
        ))
        _row("Flow PE cosmo", opts.flows_pe_cosmology)
    else:
        _row("GW events path",  opts.gw_path)
    if opts.pdet_flow_path:
        _row("Pdet emulator",   opts.pdet_flow_path)
        _row("Pdet draws M",    f"{opts.pdet_nsamp:,} (seed {opts.pdet_seed})")
        _row("Pdet inj. cosmo", opts.pdet_cosmology)
        _row("Pdet chieff amax", opts.pdet_chieff_amax)
    else:
        _row("Selection path",  opts.gwselection_path)
    if opts.survey_path:
        if opts.n_catalogs == 1:
            _row("Survey path",  opts.survey_path)
        else:
            _row("Catalog mixture", f"{opts.n_catalogs} catalogs")
            for _i, _p in enumerate(opts.survey_paths):
                _row(f"  catalog {_i + 1}", _p)
        _row("Use LSS",      "yes" if opts.use_LSS else "no")
        if opts.use_LSS:
            _row("  legacy-floor budget",
                 "conserved (renormalized to full-sky mean one)"
                 if str(getattr(opts, "lss_floor", "conserve")) == "conserve"
                 else "LEGACY unrenormalized floor (budget can inflate)")
        _row("Catalog sky weighting",
             f"{getattr(opts, 'catalog_sky_weighting', 'conditional')} "
             f"({getattr(opts, 'catalog_sky_weighting_source', 'explicit')})")
    _row("Output root",     opts.save_path)
    _row("Sel. batch",     format_block_size_request(opts.sel_batch_size))
    _row("PE event block", format_block_size_request(getattr(opts, "pe_event_block", None)))
    if opts.drop_full_catalog:
        _row("Drop full catalog", "yes (compact views only)")
    _end()


def _bundle_max_gals_per_row(data):
    """Widest compact galaxy row across the per-catalog bundles (0 if none).

    A K >= 2 multitracer run keeps its catalogs in ``data['catalogs']``, one bundle
    each, and leaves the top-level ``catalog_memory`` / ``zgals_*`` slots empty on
    purpose.  Each bundle's ``zgals_pe`` is the compact ``(n_union_rows, max_gals)``
    table, so its second axis IS the per-injection redshift-prior row width the
    block-size model needs.  Taking the max over bundles is the right aggregate:
    the mixture evaluates every catalog on the same injection axis, so the widest
    row bounds the per-injection working set (``n_catalogs`` carries the count).
    """
    bundles = data.get("catalogs") if isinstance(data, dict) else None
    if not bundles:
        return 0
    widest = 0
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        for key in ("zgals_pe", "zgals_sel"):
            shape = getattr(bundle.get(key), "shape", None)
            if shape and len(shape) >= 2:
                widest = max(widest, int(shape[1]))
    return widest


def _block_sizing_inputs(opts, data):
    """Derive the ``resolve_block_sizes`` inputs from loaded ``opts``/``data``.

    Split out (and kept free of the device probe / mutation) so the CLI plumbing —
    the dominant per-block dimensions and the MEASURED static-state bytes — can be
    unit tested with a fake ``opts``/``data`` pair.  Returns a plain kwargs dict.

    ``resolve_block_sizes`` runs at the end of the data load, BEFORE the likelihood
    factory: the loaded arrays are already device-resident (reflected in the probed
    free memory), so the resolver reserves only the PENDING factory-built state
    (:func:`estimate_pending_static_bytes` — KDE caches / ``base_miss``).  The FULL
    measured static (:func:`measure_static_state_bytes`) is returned alongside for
    the run-config report only.
    """
    _msel = data.get("m1detsels")
    n_sel = int(np.asarray(_msel).shape[0]) if _msel is not None else 0
    has_catalog = bool(getattr(opts, "survey_path", None))
    n_catalogs = int(getattr(opts, "n_catalogs", 1) or 1)

    # n_grid: the shared redshift grid actually used everywhere downstream.
    from darksirens.redshift.grid import zgrid
    n_grid = int(np.asarray(zgrid).shape[0])

    # n_q: the population q-quadrature node count (--norm_nq / DARKSIRENS_GW_N_Q).
    # The default pairing normaliser integrates over q PER SAMPLE, so this is the
    # axis that multiplies the selection / PE working sets — the memory model was
    # blind to it before, which made --norm_nq defeat the OOM guard entirely.
    n_q = int(normalization_grid_settings().n_q)

    # Max galaxies per unique pixel — the row width that multiplies the catalog
    # per-injection / per-sample redshift-prior work (1 for the catalog-free path).
    # For a K >= 2 multitracer run the TOP-LEVEL catalog_memory is absent by
    # construction (inference/data.py stubs catalog_inputs with zgals=None, so
    # loaders.compute_sky_pixels_and_vectors never builds it) and the real catalogs
    # live per-bundle in data['catalogs'] — so aggregate the bundles' own compact
    # row widths instead of silently falling back to 1.
    # ``None`` means "could not determine": the resolver then falls back to the
    # calibration reference with a loud warning, instead of the silent
    # ``max_gals_per_row = 1`` that scaled the _CAT slopes ~2000x too light.
    catalog_memory = data.get("catalog_memory") or {}
    max_gals = catalog_memory.get("max_galaxies_per_unique_pixel")
    max_gals = int(max_gals) if max_gals else None
    if max_gals is None:
        max_gals = _bundle_max_gals_per_row(data) or None

    # PENDING static (factory KDE caches / base_miss) is what the resolver reserves:
    # the loaded arrays are already device-resident, hence already in the probed
    # free_bytes (see block_sizing.estimate_pending_static_bytes).  The FULL measured
    # static is carried alongside for the run-config report only.
    pending_static_bytes = estimate_pending_static_bytes(
        data, n_grid=n_grid, has_catalog=has_catalog,
        catalog_memory=catalog_memory or None,
        catalog_sky_weighting=getattr(opts, "catalog_sky_weighting", "conditional"),
    )
    full_static_bytes = measure_static_state_bytes(
        data,
        n_grid=n_grid,
        has_catalog=has_catalog,
        n_catalogs=n_catalogs,
        catalog_memory=catalog_memory or None,
        drop_full_catalog=bool(getattr(opts, "drop_full_catalog", False)),
        catalog_sky_weighting=getattr(opts, "catalog_sky_weighting", "conditional"),
    )
    needs_grad, concurrent_evals = sampler_block_sizing_profile(opts)
    return dict(
        n_events=int(data.get("nEvents", 0)),
        n_samp=int(data.get("nsamp", 0)),
        n_sel=n_sel,
        sel_requested=getattr(opts, "sel_batch_size", None),
        pe_requested=getattr(opts, "pe_event_block", None),
        has_catalog=has_catalog,
        flow_path=bool(getattr(opts, "gw_flows_path", None)),
        n_grid=n_grid,
        n_q=n_q,
        max_gals_per_row=max_gals,
        n_catalogs=n_catalogs,
        static_state_bytes=pending_static_bytes,
        needs_grad=needs_grad,
        concurrent_evals=concurrent_evals,
        # Report-only extras (stripped before resolve_block_sizes):
        static_state_full_bytes=full_static_bytes,
    )


def _resolve_and_report_block_sizes(opts, data):
    """Resolve the 'auto' block-size knobs from a probed memory budget (post-load).

    Mutates ``opts.sel_batch_size`` / ``opts.pe_event_block`` to concrete
    ``int``/``None`` values and records the provenance on
    ``opts.block_size_resolution`` (persisted via settings.json), so the factory,
    flow builder and settings dump downstream all see plain ints/None.
    """
    kwargs = _block_sizing_inputs(opts, data)
    full_static_bytes = kwargs.pop("static_state_full_bytes")
    pending_static_bytes = kwargs["static_state_bytes"]
    opts.block_size_static_state_bytes = int(full_static_bytes)
    free_bytes, free_src = probe_device_memory_bytes()
    plan = resolve_block_sizes(
        backend=jax.default_backend(), free_bytes=free_bytes,
        free_bytes_reliable=(free_src != "fallback-default"), **kwargs)
    opts.sel_batch_size = plan.sel_batch_size
    opts.pe_event_block = plan.pe_event_block
    opts.block_size_resolution = plan.source
    _sel = plan.sel_batch_size if plan.sel_batch_size is not None else "single pass"
    _pe = plan.pe_event_block if plan.pe_event_block is not None else "single pass"
    _ok(f"Block sizes [{plan.source}]: sel_batch_size={_sel}, pe_event_block={_pe}")
    _ok(
        "Static state: "
        f"{full_static_bytes / 1024**3:.3f} GiB total "
        f"({pending_static_bytes / 1024**3:.3f} GiB pending/factory reserved); "
        f"free {free_bytes / 1024**3:.1f} GiB ({free_src}); "
        f"n_grid={kwargs['n_grid']}, n_q={kwargs['n_q']}, "
        f"max_gals/row="
        + ("unknown" if kwargs["max_gals_per_row"] is None
           else f"{kwargs['max_gals_per_row']:,}")
        + f", K={kwargs['n_catalogs']}"
    )
    _ok(
        "Peak model: "
        f"{'value+grad' if kwargs['needs_grad'] else 'value-only'} "
        f"({getattr(opts, 'sampler', '?')}"
        + (f", {kwargs['concurrent_evals']} concurrent evals"
           if kwargs["concurrent_evals"] > 1 else "")
        + ")"
    )


def _load_and_report_data(opts):
    # ── Load data ──────────────────────────────────────────────────

    _section("Loading data")
    print("  │")
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)

    nEvents = data["nEvents"]
    nsamp   = data["nsamp"]
    Ndraw   = data["Ndraw"]
    nside   = data.get("nside", "N/A")

    if data.get("flow_event_names"):
        # Persist the ensemble's event identity in settings.json (which
        # serialises opts attributes) alongside the results.hdf5 meta.
        opts.flow_event_names = list(data["flow_event_names"])
        _ok(
            f"Flow surrogates:        {nEvents} events × J={nsamp:,} population "
            "draws/likelihood call"
        )
    else:
        _ok(f"GW posterior samples:   {nEvents} events × {nsamp} samples/event = {nEvents*nsamp:,} total")
    _ok(f"Selection injections:   {int(Ndraw):,} total generated")

    if opts.survey_path:
        ngals_pe  = data.get("ngals_pe",  None)
        ngals_sel = data.get("ngals_sel", None)
        _ok(f"HEALPix nside:          {nside}")
        if ngals_pe  is not None:
            _ok(f"Catalog galaxies (PE pixels):  {int(np.asarray(ngals_pe).sum()):,}")
        if ngals_sel is not None:
            _ok(f"Catalog galaxies (sel pixels): {int(np.asarray(ngals_sel).sum()):,}")
        catalog_memory = data.get("catalog_memory")
        if catalog_memory is not None:
            _ok(
                "Unique catalog pixels:   "
                f"PE {catalog_memory['unique_pe_pixels']:,}, "
                f"selection {catalog_memory['unique_sel_pixels']:,}"
            )
            _ok(
                "Duplicated catalog bytes avoided: "
                f"{catalog_memory['duplicated_catalog_bytes_avoided'] / 1e9:.3f} GB"
            )
            _ok(
                "Max galaxies/unique pixel: "
                f"{catalog_memory['max_galaxies_per_unique_pixel']:,}"
            )

    dg = data.get("delta_g_pix_z")
    if dg is not None:
        gb = np.asarray(dg).nbytes / 1e9
        _ok(f"δ_g field shape:        {np.asarray(dg).shape}  ({gb:.3f} GB)")

    # ── Survey z_depth resolution (bounds the completion missing-galaxy
    #    budget) ─────────────────────────────────────────────────────────
    # Precedence: --survey_z_depth (CLI, applies to every catalog) >
    # per-catalog f.attrs['z_depth'] (written by darksirens_pixelate
    # --z_depth) > None (legacy full-grid budget, bit-identical). Resolved
    # host-side here (files are already loaded) and threaded to SurveyParams
    # via ParameterDecoder.z_depths (inference/parameters.py).
    if opts.survey_path:
        if opts.n_catalogs == 1:
            resolved = resolve_survey_z_depth(opts.survey_z_depth, data.get("z_depth"))
            opts.resolved_survey_z_depths = [resolved]
            _report_survey_z_depth("Survey z_depth", resolved)
        else:
            bundles = data.get("catalogs") or []
            resolved_list = [
                resolve_survey_z_depth(opts.survey_z_depth, b.get("z_depth"))
                for b in bundles
            ]
            opts.resolved_survey_z_depths = resolved_list
            for _i, _resolved in enumerate(resolved_list):
                _report_survey_z_depth(f"Catalog {_i + 1} z_depth", _resolved)
    else:
        opts.resolved_survey_z_depths = []

    # Canonical post-load validation: assert the per-catalog config sequences
    # resolved above agree with n_catalogs before anything downstream consumes
    # them (single place; no behaviour change for a valid config).
    validate_multitracer_run(opts, data)

    _resolve_and_report_block_sizes(opts, data)

    _end()
    return data


def _maybe_run_completion_validation(opts, data, prior_overrides, fixed_parameter_values):
    if opts.validate_completion:
        _section("Completion validation dry run")
        if opts.n_catalogs >= 2:
            for validation_path in run_completion_validation_multitracer(
                opts, data, prior_overrides, fixed_parameter_values
            ):
                _ok(f"completion_validation JSON → {validation_path}")
        else:
            validation_path = run_completion_validation(
                opts, data, prior_overrides, fixed_parameter_values
            )
            _ok(f"completion_validation JSON → {validation_path}")
        _row("Action", "exiting before likelihood/sampling")
        _end()
        return True
    return False


def _selection_c_mode_by_catalog(opts):
    """Per-catalog completeness mode.

    --c_mode is a single choice today (one estimator for the run), so this
    broadcasts; it accepts a sequence so that a future per-catalog flag needs
    no change in the pairing rules below, which are all stated per catalog.
    """
    mode = getattr(opts, "c_mode", "per_pixel") or "per_pixel"
    if isinstance(mode, str):
        return (mode,) * int(getattr(opts, "n_catalogs", 1))
    return tuple(str(m) for m in mode)


#: Background parameters the h-scaled magnitude zero point does NOT absorb.
_SELECTION_FIT_BACKGROUND_LABELS = ("Om0", "w0", "wa")


def _check_selection_fit_background(opts, labels):
    """Refuse a magnitude-selection FIT while the background beyond H0 is sampled.

    ``C_sel(z; theta)`` is built from ``m_lim - M0hat - DM(z; H0=100)``.  The
    h-scaled zero point absorbs ``H0`` EXACTLY -- the tabulated ``dL`` is exactly
    proportional to ``1/H0``, so ``M0 + DM(z)`` is H0-free (the firewall in
    ``darksirens/redshift/selection.py``).  It does NOT absorb ``Om0``/``w0``/
    ``wa``: those change the SHAPE of ``DM(z)``, and only the z-INDEPENDENT part
    of that change is absorbable by a fitted zero point.  The module measures
    the residual at ~0.1 mag of non-absorbable shape across a z = 0.05-0.5
    catalog (0.012 -> 0.110 mag from z = 0.05 to 0.5 between Om0 = 0.25 and
    0.40), against a Laplace fit sd of ~0.02 mag: five times the fit's own
    uncertainty.

    ``_validate_fit_background`` already refuses a fit MEASURED at a background
    other than the package fiducial.  What it cannot see is the run: with the
    background SAMPLED, the completeness curve is evaluated at each proposal's
    cosmology while its zero point stays pinned where it was measured, so the
    missing-galaxy budget acquires a spurious dependence on Om0/w0/wa and the
    selection model injects artificial information about them.  Fixing only
    w0/wa is not enough while Om0 is free.

    The escape hatch is deliberately explicit and stamped in settings.json;
    the real fix (persist the fit sample's redshift distribution and re-apply
    the mean DM offset inside the curves per proposal) is not implemented.
    """
    if getattr(opts, "allow_selection_fit_free_background", False):
        _warn(
            "--allow_selection_fit_free_background: running a magnitude-"
            "selection FIT with a sampled background cosmology. The fitted "
            "zero point is anchored at the package fiducial, so the "
            "completeness budget carries ~0.1 mag of non-absorbable DM(z) "
            "shape (~5x the fit's own Laplace sd) as Om0/w0/wa move. "
            "Methodological studies only -- this arm can inject artificial "
            "information about the dark-energy parameters."
        )
        return
    modes = _selection_c_mode_by_catalog(opts)
    fits = getattr(opts, "selection_fits", None) or []
    anchored = [
        k for k, mode in enumerate(modes, start=1)
        if mode == "selection" and len(fits) >= k and fits[k - 1]
    ]
    if not anchored:
        return
    sampled = [lab for lab in _SELECTION_FIT_BACKGROUND_LABELS if lab in labels]
    if not sampled:
        return
    _fatal(
        f"catalog(s) {anchored} run c_mode='selection' anchored by a "
        f"--selection_fit while {sampled} is SAMPLED. The h-scaled magnitude "
        "zero point absorbs H0 exactly, but NOT Om0/w0/wa: they change the "
        "shape of DM(z), the fit measured its zero point at the package "
        "fiducial background, and nothing re-fits it per proposal -- so the "
        "completeness curve carries ~0.1 mag of non-absorbable shape across a "
        "z = 0.05-0.5 catalog against a Laplace sd of ~0.02 mag, and the "
        "selection model can inject artificial information about the "
        "background. Pass --fix_cosmology true (fixing only w0/wa via --fix_de "
        "is NOT enough while Om0 is free), or pin Om0/w0/wa individually with "
        "--fixed_parameters, or drop --selection_fit for the wide-open "
        "ablation. --allow_selection_fit_free_background runs it anyway, "
        "loudly and stamped in settings.json."
    )


def _resolve_selection_fits(opts, data, fixed_parameter_values):
    """Load each catalog's --selection_fit and stamp the per-catalog state.

    Stamps on ``opts`` (all length-n_catalogs, all JSON-serialisable so
    settings.json records exactly what anchored the run):

    * ``selection_fits``            -- per catalog record dict or None (the
                                       SINGLE source of truth; the Q-table
                                       theta firewall reads only this)
    * ``selection_prior``           -- FLAT {suffixed label: (loc, sd)} for
                                       build_parameter_space, or None
    * ``selection_kcorr_by_catalog``  -- structural K(z) template per catalog
    * ``selection_strata_by_catalog`` -- structural stratum offsets per catalog

    and pins ``m_lim{sfx}`` per catalog in ``fixed_parameter_values``.

    Also stamps ``opts.selection_family`` (SCALAR: the fits' shared
    luminosity-function family, ``"gaussian"`` when no fit is given).  A
    K >= 2 mixture must be HOMOGENEOUS in the LF: every fit declares the
    SAME family (mixed families are refused -- the completeness bases would
    differ by the whole LF shape), and a schechter mixture must additionally
    share ONE ``M_faint_offset`` (the protocol constant is pinned unsuffixed:
    one completeness denominator per run; per-catalog offsets are refused
    both here and in the prior validation).
    """
    from darksirens.redshift.selection import (
        SELECTION_SAMPLED_FIELDS,
        SELECTION_THETA_FIELDS,
        load_selection_fit_strata,
    )

    modes = _selection_c_mode_by_catalog(opts)
    # Always recorded in settings.json so post-processing rebuilds the SAME
    # label ordering (the family gates which theta the survey block carries).
    opts.selection_family = "gaussian"
    paths = (getattr(opts, "selection_fit_paths", None)
             or [None] * opts.n_catalogs)
    _any_fit = any(p is not None for p in paths)
    # The --stratum_map x (no fit) and --stratum_map x (K>=2) pairings fire
    # pre-load in main(); re-asserted here so a direct call to the resolver is
    # guarded identically.
    _check_stratum_map_pairing(opts)

    fits = []
    _run_family = None      # family of the first catalog with a fit
    _run_offset = None      # the run's ONE schechter M_faint_offset
    for k, (path, mode) in enumerate(zip(paths, modes), start=1):
        sfx = "" if k == 1 else f"_c{k}"
        if path is None:
            if mode == "selection" and _any_fit:
                _others = [j for j, p in enumerate(paths, start=1)
                           if p is not None]
                _fatal(
                    f"catalog {k} runs c_mode='selection' but --selection_fit "
                    f"gives it no magnitude fit while catalog(s) {_others} "
                    f"are anchored: its M0hat{sfx} / sigma_M{sfx} would "
                    "sample flat across the truncation bounds while the "
                    "anchored catalogs sit inside ~1e-2 mag of their fits, "
                    "so this mixture's missing-galaxy budget would be set by "
                    "an unconstrained nuisance. Pass one fit per catalog, or "
                    "drop --selection_fit for the wide-open ablation on "
                    "every catalog."
                )
            fits.append(None)
            continue
        if mode != "selection":
            _fatal(f"--selection_fit entry {k} given but catalog {k} runs "
                   f"--c_mode '{mode}': the fit parameterizes the "
                   "c_mode=selection completeness only.")
        _strata = load_selection_fit_strata(path)
        _sel = _strata[0]     # reference stratum: prior center + m_lim datum
        # The fit declares the luminosity-function family; the stratified
        # sub-branch below is reachable for gaussian only (loader-enforced).
        # A mixture must be HOMOGENEOUS: the run's family is a scalar (one
        # C_sel shape for every catalog), so the first fit sets it and every
        # later fit must agree.
        _family = str(_sel.get("family", "gaussian"))
        if _run_family is None:
            _run_family = _family
        elif _family != _run_family:
            _fatal(
                f"catalog {k}: --selection_fit {path} declares the "
                f"'{_family}' family but catalog(s) before it fitted "
                f"'{_run_family}': a mixture's catalogs share one "
                "luminosity-function family (the completeness bases would "
                "otherwise differ by the whole LF shape, not a parameter). "
                "Refit the catalogs under one family.")
        opts.selection_family = _run_family
        _map_sha = None
        if len(_strata) > 1:
            if not getattr(opts, "stratum_map", None):
                _fatal(f"--selection_fit carries {len(_strata)} strata but no "
                       "--stratum_map assigns pixels to them; pass the map "
                       "the strata were defined on.")
            # A prebuilt Q table is allowed with a stratified fit ONLY when
            # it was built on the SAME stratified base (per-stratum thetas +
            # stratum-map hash stamped); verified below in
            # _check_selection_qtable_theta once the table fiducials load.
            import hashlib as _hashlib
            _h = _hashlib.sha256()
            with open(opts.stratum_map, "rb") as _fmap:
                for _chunk in iter(lambda: _fmap.read(1 << 20), b""):
                    _h.update(_chunk)
            _map_sha = _h.hexdigest()
            # Common-mode + fixed offsets: stratum 0 is the reference (the
            # sampled M0hat/sigma_M ARE its theta); stratum s carries the
            # fixed, fit-measured offsets. The strata must share one K
            # template -- the common mode is defined against a single curve.
            for _s in _strata[1:]:
                if tuple(_s["k_corr_coeffs"]) != tuple(_sel["k_corr_coeffs"]):
                    _fatal("per-stratum K(z) templates differ inside "
                           f"{path}; strata must share one template (refit "
                           "with a common --k_corr_coeffs).")
            _note = (f"stratified selection: {len(_strata)} strata; sampled "
                     "(M0hat, sigma_M) = stratum "
                     f"'{_sel.get('stratum', '0')}' common mode, offsets "
                     "fixed at the fit values (measured at ~1e-3 mag from "
                     "the full catalog)")
            print(f"    - {_note}")
        elif getattr(opts, "stratum_map", None):
            _fatal("--stratum_map given but the --selection_fit carries a "
                   "single stratum; drop the map or refit with --strata.")
        _sd = np.sqrt(np.diag(np.asarray(_sel["cov"], dtype=float)))
        # Prior centers, sds and the provenance theta are all FAMILY-DRIVEN,
        # off the one table in redshift/selection.py, so the sd order can
        # never drift from the covariance's row/column order.
        _fields = SELECTION_SAMPLED_FIELDS[_family]
        if k == 1:
            # Folds every --fixed_parameter_values pin (m_lim; the schechter
            # M_faint_offset protocol constant) into the EFFECTIVE theta the
            # Q-table firewall compares.
            _theta_eff = _resolve_selection_fit_pins(
                _sel, _family, fixed_parameter_values)
        else:
            # k >= 2: the suffixed m_lim datum is this catalog's own pin;
            # everything else in the family's theta is the fit's.
            fixed_parameter_values.setdefault(
                f"m_lim{sfx}", float(_sel["m_lim"]))
            _theta_eff = {
                "m_lim": float(fixed_parameter_values[f"m_lim{sfx}"]),
                **{n: float(_sel[n])
                   for n in SELECTION_THETA_FIELDS[_family]
                   if n != "m_lim"},
            }
        if _family == "schechter":
            # ONE completeness denominator per run: the shared protocol
            # constant is pinned UNSUFFIXED by catalog 1's pins fold above
            # (the decoder broadcasts it to every suffixed catalog), so
            # every fit must carry the same offset -- a per-catalog offset
            # has no defined semantics.
            if _run_offset is None:
                _run_offset = float(_theta_eff["M_faint_offset"])
            elif abs(float(_sel["M_faint_offset"]) - _run_offset) > 1e-9:
                _fatal(
                    f"catalog {k}: --selection_fit {path} carries "
                    f"M_faint_offset={float(_sel['M_faint_offset'])} but the "
                    f"run's shared protocol constant is {_run_offset}: a "
                    "schechter mixture integrates ONE completeness "
                    "denominator (M_faint = Mstar_hat + M_faint_offset), so "
                    "every catalog's fit must be run with the same "
                    "--m_faint_offset. Refit with a common value.")
        fits.append({
            "catalog": k,
            "path": str(path),
            "family": _family,
            "theta": _theta_eff,
            "k_corr_coeffs": [float(c)
                              for c in (_sel.get("k_corr_coeffs") or ())],
            "strata_fit": ([[float(s["m_lim"]), float(s["M0hat"]),
                             float(s["sigma_M"])] for s in _strata]
                           if len(_strata) > 1 else None),
            "strata_struct": ([[float(s["m_lim"]),
                                float(s["M0hat"]) - float(_sel["M0hat"]),
                                float(s["sigma_M"]) / float(_sel["sigma_M"])]
                               for s in _strata] if len(_strata) > 1 else None),
            "stratum_map": (str(opts.stratum_map)
                            if len(_strata) > 1 else None),
            "stratum_map_sha256": (_map_sha if len(_strata) > 1 else None),
            "prior": {f"{f}{sfx}": [float(_sel[f]), float(_sd[i])]
                      for i, f in enumerate(_fields)},
        })
        print(f"    - catalog {k} selection fit [{_family}]: {path} -- "
              + ", ".join(f"{f}{sfx}={_sel[f]:.3f} ± {_sd[i]:.3f}"
                          for i, f in enumerate(_fields))
              + f", m_lim{sfx} pinned at "
              f"{fixed_parameter_values[f'm_lim{sfx}']:.3f}")

    # The three consumer views are DERIVED here, in one place, so they cannot
    # drift from the per-catalog records they are projections of.
    opts.selection_fits = fits
    opts.selection_prior = {
        lbl: tuple(v)
        for f in fits if f for lbl, v in f["prior"].items()
    } or None
    opts.selection_kcorr_by_catalog = [
        (tuple(f["k_corr_coeffs"]) or None) if f else None for f in fits
    ]
    opts.selection_strata_by_catalog = [
        ([tuple(s) for s in f["strata_struct"]] if f and f["strata_struct"]
         else None) for f in fits
    ]

    if not any(opts.selection_strata_by_catalog):
        return
    # Attach the full-sky stratum map to the data bundle so
    # prepare_catalog_views can build the per-stratum empty-pixel budgets
    # alongside the field-normalization inputs.  Single-catalog only (the
    # K>=2 refusal above), so catalog 1's strata are the run's strata.
    import h5py as _h5py
    _struct = opts.selection_strata_by_catalog[0]
    with _h5py.File(opts.stratum_map, "r") as _f:
        if "stratum_map" not in _f:
            _fatal(f"{opts.stratum_map}: no 'stratum_map' dataset.")
        _smap = np.asarray(_f["stratum_map"], dtype=np.int32)
    _npix_survey = 12 * int(data["nside"]) ** 2
    if _smap.shape[0] != _npix_survey:
        _fatal(
            f"stratum map covers {_smap.shape[0]} pixels but the "
            f"survey nside={data['nside']} sky has {_npix_survey}; "
            "regenerate the map at the survey nside (RING).")
    _n_strata_map = int(_smap.max()) + 1
    if _smap.min() < 0 or _n_strata_map != len(_struct):
        _fatal(
            f"stratum map labels span [{int(_smap.min())}, "
            f"{int(_smap.max())}] but the fit carries "
            f"{len(_struct)} strata; every "
            "pixel needs a label in [0, S) matching the fit.")
    data["pixel_stratum_map"] = _smap


def _resolve_single_catalog_marks(opts, data):
    # ── Parameter space ────────────────────────────────────────────

    # Resolve the galaxy marks for the marked-host model (dark_sirens): those
    # present in the catalog, optionally narrowed by --marks.  Stored on opts so
    # the decoder/likelihood see the same ordered list as the parameter space.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS
    if opts.n_catalogs == 1:
        _present_marks = tuple(n for n in _MARK_FIELDS if data.get(_MARK_FIELDS[n]) is not None)
        if opts.marks:
            _req = tuple(s.strip() for s in opts.marks.split(",") if s.strip())
            _missing = [m for m in _req if m not in _present_marks]
            if _missing:
                _fatal(f"--marks requested {_missing} but the catalog provides "
                       f"{list(_present_marks)}.")
            opts.mark_names = _req
        else:
            opts.mark_names = _present_marks
        if opts.mark_model != "none" and not opts.mark_names:
            _fatal("--mark_model loglinear requires per-galaxy marks, but the catalog "
                   "provides none (expected datasets like LOGMSTAR/LOGSSFR).")
        opts.mark_names_by_catalog = (tuple(opts.mark_names),)
    # K >= 2: opts.mark_names / opts.mark_names_by_catalog were resolved
    # PRE-LOAD from each catalog file's datasets (see the multitracer guard
    # block); the flat data dict carries no top-level marks for a mixture.


def _build_and_report_parameter_space(opts, data, prior_overrides, fixed_parameter_values):
    _section("Building parameter space")
    # A Q_LSS completion table (explicit --lss_completion path or an in-catalog
    # /lss_completion group) REPLACES the b_miss local-overdensity factor, so
    # b_miss must be dropped from the sampled dark_sirens block or it is a
    # phantom flat nuisance dimension. Mirror the likelihood's Q-active test:
    # an explicit path, a table on this catalog's loaded bundle (K>=2), or (K=1
    # flat path, no bundle) the in-catalog table already merged into flat data.
    # PER-CATALOG Q activity: the likelihood chooses Q-vs-b_miss per catalog
    # (completion.field_lss_q is not None), so b_miss must be dropped ONLY for
    # the catalogs that actually carry a Q table.  A global OR would drop
    # b_miss_c{k} for a Q-free catalog too (a silently frozen nuisance).
    _lss_bundles = data.get("catalogs") or []
    _lss_explicit = list(getattr(opts, "lss_completions", None) or [])
    _LSS_Q_KEYS = ("lss_completion_logq", "lss_completion_q",
                   "lss_completion_logq_members", "lss_completion_q_members")

    def _catalog_lss_active(i):
        # Explicit --lss_completion path for catalog i (positionally aligned).
        if i < len(_lss_explicit) and _lss_explicit[i] is not None:
            return True
        # K>=2: this catalog's own bundle carrying any Q table/ensemble.
        if i < len(_lss_bundles):
            b = _lss_bundles[i]
            return any(b.get(k) is not None for k in _LSS_Q_KEYS)
        # K=1 flat path: no per-catalog bundle, so an in-catalog /lss_completion
        # group (or explicit table) landed in the flat data dict via
        # attach_lss_inputs -> maybe_load_lss_completion; mirror the same key
        # probe there so an in-catalog K=1 Q table also drops b_miss.
        if not _lss_bundles:
            return any(data.get(k) is not None for k in _LSS_Q_KEYS)
        return False

    lss_completion_active_by_catalog = tuple(
        bool(_catalog_lss_active(i)) for i in range(opts.n_catalogs)
    )
    lss_completion_active = any(lss_completion_active_by_catalog)
    # Persist BOTH on opts (-> settings.json) so post-processing rebuilds the
    # SAME parameter space (b_miss dropped per catalog when Q is active) as this
    # run sampled: the per-catalog tuple is authoritative, the scalar is the
    # legacy back-compat fallback.
    opts.lss_completion_active_by_catalog = lss_completion_active_by_catalog
    opts.lss_completion_active = bool(lss_completion_active)
    # Guard 6's --lss_completion refusal is pre-load and can only see the FLAG.
    # An in-catalog /lss_completion group is discovered here, after the load, so
    # close the same exclusion on the loaded data: latent mode generates Q, and
    # a catalog that also ships a resident table is two completion models.
    if latent_field_mode(opts) and lss_completion_active:
        _bad = [k for k, a in enumerate(lss_completion_active_by_catalog, start=1)
                if a]
        _fatal(
            f"--lss_field_mode latent, but catalog(s) {_bad} carry an "
            "in-catalog /lss_completion Q table (loaded automatically when no "
            "--lss_completion path is given). Latent mode GENERATES Q from the "
            "anchor artifact; a resident table alongside it is a second, "
            "differently-conditioned completion model, and silently preferring "
            "one would make the run's estimand unreadable. Re-pixelate without "
            "the group, or stay in table mode (PLAN 4.4 guard 6)."
        )
    _check_aggregate_requires_q(opts, lss_completion_active_by_catalog)

    _resolve_selection_fits(opts, data, fixed_parameter_values)
    # The two post-load halves of the latent gate list: guard 6's stratified
    # refusal, and the schechter refusal (both are properties of the
    # --selection_fit JSON, so neither is knowable pre-load).
    _check_latent_selection_strata(opts)
    _check_latent_selection_family(opts)

    # ── b_miss -> b_GW: the PLAN 4.3 rule INVERSION ────────────────────────
    # TABLE MODE (unchanged).  b_miss reaches the likelihood ONLY through the
    # local-overdensity factor max(1 + alpha_miss*b_miss*delta_g, 0), so
    # inference/prior.py:_b_miss_rule declares it INERT whenever that factor
    # cannot depend on it, and the second of its two gates reads, verbatim:
    #
    #     "b_miss is inert with --use_lss off: delta_g is the all-zero
    #      (1, N_grid) dummy built by inference/loaders.py, so the
    #      (1 + alpha_miss*b_miss*delta_g) factor is identically 1 for any
    #      b_miss"                       (remedy: "or pass --use_lss true ...")
    #
    # which this call site arms by passing use_lss = bool(opts.use_LSS): with
    # --use_lss off, b_miss is dropped from the sampled dark_sirens block so it
    # is not a phantom flat nuisance dimension.  It is not identified because
    # it multiplies a zero.
    #
    # LATENT MODE (the inversion).  The rule flips because the parameter does.
    # PLAN 4.3: (b, xi) and (amp, xi) enter only through b*xi and b*amp, so amp
    # is fixed at 1 and the bias is the sampled degree of freedom; b_gal is
    # pinned at the anchor inside the artifact, and what remains free is b_GW,
    # the bias with which GW HOSTS trace the latent field.  It enters the
    # likelihood as the b_GW multiplying (row_fac_m[p] . phi_z[z]) in the seam's
    # logQ, and again inside rho(z; C, b_GW) through the moments (A, B) --
    # completion.latent_b_gw(survey) reads survey.b_miss for exactly this
    # reason.  So in latent mode b_miss is NOT inert with --use_lss off: it is
    # the field's GW bias and it IS genuinely (if weakly -- 259 events, expected
    # prior-dominated) identified.  The guard becomes, in latent mode:
    #
    #     b_miss IS b_GW and is SAMPLED with --use_lss off; --use_lss on is
    #     refused outright (guard 6 above), because delta_g and Q are the same
    #     modulation twice.
    #
    # Mechanically the inversion is this one argument: prior.py's ``use_lss``
    # parameter means "the missing-galaxy modulation is a function of b_miss",
    # and in latent mode it is -- through b_GW, not through delta_g.  Passing
    # True is therefore the honest answer to the question that rule asks, and
    # guard 6 has already guaranteed opts.use_LSS is False here, so the two
    # never disagree about which field b_miss multiplies.  q_active is likewise
    # all-False by the refusals above, so the Q-table gate cannot fire either.
    # Old and new runs do not share a parameter space: the run fingerprint's
    # schema_version is bumped (inference/run_fingerprint.py) so a latent run
    # can never silently resume a table run's checkpoint, and lss_field_mode
    # itself is fingerprinted, so the two digests differ even at equal labels.
    _latent = latent_field_mode(opts)
    _b_miss_identified = bool(getattr(opts, "use_LSS", False)) or _latent

    res = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        fix_de                 = opts.fix_de,
        prior_overrides        = prior_overrides,
        fixed_parameter_values = fixed_parameter_values,
        universe_model         = opts.universe_model,
        shared_beta            = opts.shared_beta,
        shared_spin            = opts.shared_spin,
        shared_gamma           = opts.shared_gamma,
        sky_model              = opts.sky_model,
        mark_model             = opts.mark_model,
        mark_names             = opts.mark_names,
        n_catalogs             = opts.n_catalogs,
        lss_completion_active  = lss_completion_active_by_catalog,
        # "Is the missing-galaxy modulation a function of b_miss?"  In table
        # mode that is --use_lss (b_miss is inert when delta_g is the all-zero
        # dummy, so it must not be sampled as a phantom flat dimension); in
        # latent mode it is unconditionally yes, because b_miss IS b_GW.  See
        # the inversion note above.
        use_lss                = _b_miss_identified,
        mark_names_by_catalog  = getattr(opts, "mark_names_by_catalog", None),
        c_mode                 = opts.c_mode,
        selection_prior        = opts.selection_prior,
        selection_family       = opts.selection_family,
    )
    labels, lower_bound, upper_bound = res[0], res[1], res[2]
    n_pop_eff, n_cosmo_eff, n_survey_eff, model_name = res[3], res[7], res[8], res[9]
    fixed_parameter_statuses = res[10]
    prior_kinds = res[11]

    # PLAN 4.4 guard 5, the budget-anchor guard.  Here and not earlier: it is a
    # statement about which budget labels are SAMPLED and under WHICH prior
    # family, and this is the first line at which both are known.  No-op in
    # table mode, where the corresponding firewall is the Q-table one below.
    _check_latent_budget_anchor(
        opts, labels, prior_kinds, fixed_parameter_values)

    # A prebuilt Q_LSS table is conditioned on build-time n0/delta/bias/cosmology;
    # sampling or re-fixing those does not propagate into Q, it is absorbed as
    # spurious redshift structure and biases H0. This is the first point where
    # both the table's fiducials and the final sampled set are known.
    # ONE ENTRY PER CATALOG, in catalog order. Survey parameters are per
    # catalog (log10n0_c2 belongs to catalog 2), so passing a single table for a
    # K>=2 mixture would check catalog 2..K's parameters against catalog 1's
    # build values -- rejecting the supported "Q on one catalog only" setup and
    # leaving the other catalogs' tables unverified.
    if _lss_bundles:
        _q_fiducials = [
            _b.get("lss_completion_fiducials") for _b in _lss_bundles
        ]
    else:
        _q_fiducials = [data.get("lss_completion_fiducials")]
    # ── PLAN 4.4: the provenance rewiring, mode-routed BOTH WAYS ───────────
    # Table mode falls through to the three checks below exactly as it always
    # has.  Latent mode retires all three -- plus the catalogs/lss.py float
    # whitelist, the c_mode table-vs-run check and realization_set_id /
    # member_content_sha256 matching -- and the reason is uniform: every one of
    # them is a statement about a Q TABLE's build-time stamps, and latent mode
    # has no table to stamp.  They are retired for want of a REFERENT, not
    # because they are inconvenient; their successor is the artifact
    # fingerprint (guard 1), checked twice: pre-load in
    # _stamp_latent_artifact_fingerprint (which is also what puts the two
    # digests in settings.json) and again at likelihood-build time in
    # likelihood/factory._latent_guard_fingerprint.
    #
    # The bypass is written as an ASSERTION and not as an early return, because
    # a bypass that could hide a live table would be exactly the defect it is
    # meant to avoid.  Guard 6 refuses --lss_completion pre-load and the
    # in-catalog /lss_completion group above, so in latent mode _q_fiducials is
    # necessarily all-None and the three checks below are already no-ops on
    # their own terms; this makes that a checked invariant rather than a
    # reading of the control flow.  The other three retirements are structural
    # in the same way: the lss.py whitelist and the c_mode table-vs-run check
    # run only inside maybe_load_lss_completion (never entered), and
    # realization_set_id / member_content_sha256 matching runs only for a K>=2
    # --lss_marginalize mixture, which likelihood/factory refuses outright in
    # latent mode ("not wired on the K >= 2 mixture path").
    if latent_field_mode(opts):
        _live = [k for k, _f in enumerate(_q_fiducials, start=1) if _f]
        if _live:
            _fatal(
                f"--lss_field_mode latent, but catalog(s) {_live} produced "
                "Q-table build fiducials. The latent path RETIRES the Q-table "
                "provenance checks (check_lss_completion_provenance, the "
                "c_mode table-vs-run check, _check_selection_qtable_theta, the "
                "z_depth check, the catalogs/lss.py float whitelist and "
                "realization_set_id matching) on the grounds that there is no "
                "table to stamp -- so a table that reached this line would be "
                "consumed with NO provenance enforcement at all. This is a "
                "bug: guard 6 should have refused the table before the load "
                "(PLAN 4.4)."
            )
    else:
        check_lss_completion_provenance(
            _q_fiducials, labels, fixed_parameter_values)

        # Selection-base Q tables: theta is SAMPLED by design (exempt from the
        # pinned-to-build rule, like H0), but the table's FIXED base must have
        # been built from the SAME offline fit that centers this run's theta
        # prior -- a stale table silently carries the wrong completeness base.
        _check_selection_qtable_theta(_q_fiducials, opts)
        # The completeness depth is Q-conditioning too (it truncates the base
        # the field is residual to), but it is not a sampled parameter, so it
        # cannot ride q_provenance's label machinery.
        _check_q_table_z_depth(_q_fiducials, opts)
    # The magnitude-selection fit is anchored at ONE background cosmology.
    # First point where both the fits and the final sampled label set are known.
    _check_selection_fit_background(opts, labels)

    # Prior wider than the base is first-order-consistent only near theta_hat:
    # warn when the sampled bounds reach beyond +-5 prior sds of the center.
    if getattr(opts, "selection_prior", None):
        for _lbl, (_loc, _sd) in opts.selection_prior.items():
            if _lbl in labels:
                _i = labels.index(_lbl)
                _reach = max(abs(upper_bound[_i] - _loc),
                             abs(_loc - lower_bound[_i]))
                if _reach > 5.0 * _sd:
                    _warn(
                        f"{_lbl}: sampling bounds reach {_reach:.3f} from the "
                        f"fit center ({_reach/_sd:.0f} prior sds); the Q "
                        "table's fixed C_sel base is only first-order "
                        "consistent near theta_hat."
                    )
    # Record the exact sampler labels so build_parameter_decoder (inside
    # make_likelihood) can fail fast if it re-derives a different set -- the
    # P0.1 fail-fast net against any future CLI/decoder flag drift.
    opts.expected_sampled_labels = tuple(labels)

    _, _, pop_labels_all, _, _ = pop_model_prior_parser(
        opts.pop_model,
        shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    pop_params_fid  = get_fixed_population_params(
        opts.pop_model,
        shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
        fiducials=getattr(opts, "population_fiducials",
                          FIDUCIAL_SET_LEGACY),
    )
    joint_constraints = resolve_joint_prior_constraints(
        opts.pop_model, labels, lower_bound, upper_bound, prior_kinds,
        shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
        sky_model=getattr(opts, "sky_model", None),
    )
    prior_transform = make_prior_transform(
        lower_bound, upper_bound, prior_kinds,
        joint_constraints=joint_constraints,
    )

    _ok(f"Parameter space built:  {len(labels)} free dimensions")
    _end()

    _print_parameter_table(
        labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, fixed_parameter_statuses,
        opts.fix_cosmology, opts.fix_de, opts.fix_population, opts.fix_survey,
        pop_params_fid, pop_labels_all,
    )
    return _ParameterSpace(
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prior_kinds=prior_kinds,
        prior_transform=prior_transform,
        pop_params_fid=pop_params_fid,
        n_pop_eff=n_pop_eff,
        n_cosmo_eff=n_cosmo_eff,
        n_survey_eff=n_survey_eff,
        model_name=model_name,
        joint_constraints=tuple(joint_constraints or ()),
    )


def _build_likelihood(opts, data, pspace, fixed_parameter_values):
    pop_params_fid = pspace.pop_params_fid
    # ── Build likelihood ───────────────────────────────────────────

    _section("Building likelihood")
    print("  │  Applying optimization barriers...")
    print("  │  JIT compilation deferred to first call.")
    print("  │")
    likelihood = make_likelihood(
        opts                   = opts,
        data                   = data,
        pop_params_fid         = pop_params_fid,
        fixed_parameter_values = fixed_parameter_values,
    )
    _ok("Likelihood closure ready.")
    _end()
    return likelihood


def _run_sampling(opts, likelihood, pspace):
    labels = pspace.labels
    lower_bound = pspace.lower_bound
    upper_bound = pspace.upper_bound
    prior_kinds = pspace.prior_kinds
    prior_transform = pspace.prior_transform
    # ── Sampling ───────────────────────────────────────────────────

    _section(f"Sampling  [{opts.sampler.upper()}]")
    tinyns_cfg = getattr(opts, "tinyns_resolved_config", {})
    sampler_info = {
        "tinyns": (
            f"nlive={opts.nlive}  dlogz={opts.dlogz}  "
            f"sample={tinyns_cfg.get('sample')}  kernel={tinyns_cfg.get('kernel')}  "
            f"seed={opts.seed}"
        ),
        "dynesty": f"nlive={opts.nlive}  dlogz={opts.dlogz}  seed={opts.seed}",
        "numpyro": f"warmup={opts.nuts_warmup}  samples={opts.nuts_samples}  chains={opts.nuts_chains}  seed={opts.seed}",
    }
    _row("Configuration", sampler_info[opts.sampler])
    _row("ndim", len(labels))
    print("  │")

    t_sample_start = datetime.datetime.now()
    results = run_sampler(
        method=opts.sampler, likelihood=likelihood,
        prior_transform=prior_transform, labels=labels,
        lower_bound=lower_bound, upper_bound=upper_bound, opts=opts,
        prior_kinds=prior_kinds,
        joint_constraints=getattr(pspace, "joint_constraints", ()),
    )
    t_sample_end  = datetime.datetime.now()
    wall_sampling = t_sample_end - t_sample_start

    if results is None or results.get("samples") is None:
        _fatal("Sampler returned no results.")

    n_samples = np.asarray(results["samples"]).shape[0]
    print("  │")
    _ok(f"Sampling complete.  Wall time: {wall_sampling}")
    _ok(f"Posterior samples:  {n_samples:,}")

    logZ    = results.get("logZ")
    logZerr = results.get("logZerr")
    if logZ is not None:
        zerr = float(logZerr) if logZerr is not None else float("nan")
        _ok(f"log Z = {float(logZ):.3f} ± {zerr:.3f}")
        # A sky model that REJECTS part of its prior box leaves the raw
        # evidence multiplied by the valid fraction, so two such models cannot
        # be compared on the raw number.  Print the corrected value next to it
        # whenever it differs; models with no rejection report exactly 0.0 and
        # this stays silent (bit-identical output).
        _log_fvalid = _sky_log_prior_volume_correction(opts)
        if _log_fvalid != 0.0:
            _ok(f"log Z (prior-volume corrected) = "
                f"{float(logZ) - _log_fvalid:.3f} ± {zerr:.3f}   "
                f"[log f_valid = {_log_fvalid:.3f}; use THIS for model "
                f"comparison]")
    _end()
    return results, wall_sampling


def _run_name_prefix(opts):
    """Everything in the run-directory name except the timestamp.

    Doubles as the ``--resume auto`` filter: one --save_path normally holds many
    runs, so auto-resume only considers directories carrying THIS
    configuration's prefix (model / universe / sampler / seed).
    """
    return (
        f"{opts.pop_model}__{opts.universe_model}__{opts.sampler}"
        f"__seed{opts.seed}__"
    )


def _make_run_dir(opts, timestamp):
    """Create opts.save_path/<run_name>, disambiguating name collisions.

    run_name embeds pop_model/universe_model/sampler/seed/timestamp
    (second resolution); two same-config jobs (same seed too) launched
    within the same second would otherwise collide under exist_ok=True
    and later overwrite each other's results.hdf5. Retry with a
    monotonically increasing numeric suffix instead.
    """
    run_name = f"{_run_name_prefix(opts)}{timestamp}"
    run_dir = os.path.join(opts.save_path, run_name)
    suffix = 0
    while True:
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1
            run_dir = os.path.join(opts.save_path, f"{run_name}-{suffix:02d}")


def _write_failure(run_dir, stage, exc, *, labels=None, settings=None):
    """Best-effort ``failure.json`` next to whatever the run already wrote.

    A run that dies at hour 47 used to leave NOTHING under --save_path: no run
    directory, no settings, no traceback, so the only record of the
    configuration was a SLURM .out log.  Mirrors
    cli/inference_lensing.py::_write_failure, which already did this.
    """
    if not run_dir:
        return
    payload = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "labels": list(labels or []),
        "settings": settings or {},
        "command": [sys.executable, "-m", "darksirens.cli.inference", *sys.argv[1:]],
    }
    try:
        with open(os.path.join(run_dir, "failure.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str, allow_nan=True)
            fh.write("\n")
    except Exception as write_exc:      # never mask the real failure
        _warn(f"Could not write failure.json: {write_exc}")


class _failure_guard:
    """Context manager recording ``failure.json`` for one pipeline stage."""

    def __init__(self, run_dir, stage, *, labels=None, settings=None):
        self.run_dir = run_dir
        self.stage = stage
        self.labels = labels
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and isinstance(exc, (Exception, KeyboardInterrupt)):
            _write_failure(
                self.run_dir, self.stage, exc,
                labels=self.labels, settings=self.settings,
            )
        return False


def _base_meta(opts, data, pspace, timestamp, wall_sampling=None, t_start=None,
               t_end=None):
    """The ``meta`` block shared by the pre-sampling and final settings writes."""
    meta = {
        "n_events":         data["nEvents"],
        "n_samp_per_event": data["nsamp"],
        "n_draw":           int(data["Ndraw"]),
        "n_pop_eff":        pspace.n_pop_eff,
        "n_cosmo_eff":      pspace.n_cosmo_eff,
        "n_survey_eff":     pspace.n_survey_eff,
        "model_name":       pspace.model_name,
        "total_runtime":    None if t_end is None else str(t_end - t_start),
        "sampling_runtime": None if wall_sampling is None else str(wall_sampling),
        "timestamp":        timestamp,
        # The run's parameter-space CONVENTION, not just its digest.  Bumped to
        # 3 by --lss_field_mode latent, whose b_miss -> b_GW inversion changes
        # what the label means while leaving the label, its bounds and its
        # prior family byte-identical (PLAN 4.3).  settings.json is the file a
        # human reads when deciding whether two archived runs are comparable,
        # so the version belongs here and not only inside run_fingerprint.json.
        "settings_schema_version": FINGERPRINT_SCHEMA_VERSION,
    }
    if data.get("flow_event_names"):
        # Flow runs have no event names in a PE file: the ensemble's sorted
        # checkpoint order IS the event identity; persist it for diagnostics.
        # (JSON-encoded, and keyed *_json so it does not shadow the plain list
        # that settings.json serialises from opts.flow_event_names.)
        meta["flow_event_names_json"] = json.dumps(list(data["flow_event_names"]))
    return meta


def _prepare_run_dir(opts, data, pspace, fixed_parameter_values, prior_overrides):
    """Create the run directory and persist settings.json BEFORE sampling.

    The run directory used to be created inside ``_save_outputs``, i.e. only
    once the sampler had already returned.  Anything that killed the job during
    sampling -- SLURM TIMEOUT, host-RAM OOM, a NaN raised inside the likelihood
    -- therefore left --save_path completely empty, with no settings.json to
    re-derive the configuration from and nowhere for a checkpoint or a
    dynesty diagnostic plot to land.  The lensing CLI has created its run
    directory pre-load since it was written; this brings the main CLI in line.

    Also resolves the checkpoint/resume plan: ``--resume`` continues INSIDE the
    original run directory instead of starting a new one, so a requeued SLURM
    job accumulates one directory, not one per attempt.  A resume is gated on
    the run directory's ``run_fingerprint.json`` matching this run's semantic
    configuration (see darksirens/inference/run_fingerprint.py), and preserves
    the original ``settings.json``, writing a timestamped
    ``settings.resume-*.json`` per attempt instead.

    Returns ``(run_dir, run_timestamp, settings_snapshot)``.
    """
    _section("Preparing run directory")
    prefix = _run_name_prefix(opts)
    resume_ckpt, resume_dir = find_resume_target(
        opts, opts.sampler, name_prefix=prefix
    )
    # The full semantic identity of this run (inputs by content, priors,
    # fixed values, model flags, sampler settings).  Restoring sampler state
    # under a different target mixes two posteriors/logZ into one output, so
    # a resume is gated on an exact fingerprint match (--resume_force to
    # override deliberately).
    fingerprint = build_run_fingerprint(
        opts,
        labels=pspace.labels,
        lower_bound=pspace.lower_bound,
        upper_bound=pspace.upper_bound,
        prior_kinds=pspace.prior_kinds,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_parameter_values,
        joint_constraints=getattr(pspace, "joint_constraints", ()),
    )
    # Recorded on opts (hence in settings.json and results.hdf5) so an archived
    # artifact carries the identity of the configuration that produced it.  Set
    # AFTER the build above, so it never feeds back into the digest.
    opts.run_fingerprint_digest = fingerprint["digest"]
    opts.resume_forced_mismatch = False
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if resume_dir:
        try:
            stored = check_resume_fingerprint(
                resume_dir, fingerprint,
                force=bool(getattr(opts, "resume_force", False)),
            )
        except ResumeFingerprintError as exc:
            _fatal(str(exc))
        run_dir = resume_dir
        _row("Resuming run", run_dir)
        _row("From checkpoint", resume_ckpt)
        if stored is None:
            # --resume_force accepted a fingerprint-less legacy run dir;
            # stamp it now so later requeues are validated, not forced.
            save_run_fingerprint(run_dir, fingerprint)
        elif stored.get("digest") != fingerprint["digest"]:
            # --resume_force accepted a MISMATCH: the stored fingerprint is the
            # record of the configuration that created the checkpoint, so keep
            # it, but stamp this run's configuration beside it -- otherwise the
            # directory advertises a configuration that did not produce its
            # results.hdf5, and every later requeue must be forced too.
            opts.resume_forced_mismatch = True
            save_run_fingerprint(
                run_dir, fingerprint,
                basename=f"run_fingerprint.forced-{run_timestamp}.json",
            )
    else:
        run_dir = _make_run_dir(opts, run_timestamp)
    opts.run_dir = run_dir
    plan = resolve_checkpoint_plan(
        opts, run_dir, name_prefix=prefix, resume_from=resume_ckpt
    )
    _row("Run directory", run_dir)
    _row("Checkpointing", plan.summary())

    meta = _base_meta(opts, data, pspace, run_timestamp)
    meta["run_status"] = "sampling"
    # A resumed run must NOT overwrite the original settings.json -- it is
    # the record of the configuration that created the checkpoint.  Each
    # resume attempt leaves its own timestamped settings file instead.
    settings_basename = (
        f"settings.resume-{run_timestamp}.json" if resume_dir else "settings.json"
    )
    json_path = save_settings_json(
        opts, run_dir, pspace.labels, pspace.lower_bound, pspace.upper_bound,
        fixed_parameter_values, prior_overrides, meta,
        basename=settings_basename,
    )
    if not resume_dir:
        save_run_fingerprint(run_dir, fingerprint)
    _ok(f"{settings_basename}  →  {json_path}")
    _end()
    settings_snapshot = {
        "run_dir": run_dir,
        "settings_json": json_path,
        "sampler": opts.sampler,
        "checkpoint_file": plan.path,
        "resume_from": plan.resume_from,
    }
    return run_dir, run_timestamp, settings_snapshot


def _save_outputs(opts, data, results, pspace, fixed_parameter_values, prior_overrides, wall_sampling, t_start, run_dir, run_timestamp):
    labels = pspace.labels
    lower_bound = pspace.lower_bound
    upper_bound = pspace.upper_bound
    prior_kinds = pspace.prior_kinds
    # ── Save outputs ───────────────────────────────────────────────

    t_end     = datetime.datetime.now()
    meta = _base_meta(
        opts, data, pspace, run_timestamp,
        wall_sampling=wall_sampling, t_start=t_start, t_end=t_end,
    )
    meta["run_status"]     = "complete"
    meta["end_timestamp"]  = t_end.strftime("%Y-%m-%dT%H-%M-%S")

    _section("Saving outputs")
    _row("Run directory", run_dir)
    print("  │")

    # CHAIN FIRST.  Until this file existed the posterior lived only in the
    # results dict in RAM, and the first thing written was the HDF5 -- so any
    # failure in the save path (h5py attribute error, full filesystem, quota)
    # destroyed a multi-day chain.  Mirrors the lensing CLI, which has always
    # written samples.npy before results.hdf5 for this reason.
    samples_path = os.path.join(run_dir, "samples.npy")
    np.save(samples_path, np.asarray(results["samples"]))
    _ok(f"samples.npy    →  {samples_path}")

    hdf5_path = save_results_hdf5(
        results, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, opts, meta,
        prior_kinds=prior_kinds,
    )
    _ok(f"results.hdf5   →  {hdf5_path}")

    json_path = save_settings_json(
        opts, run_dir, labels, lower_bound, upper_bound,
        fixed_parameter_values, prior_overrides, meta,
    )
    _ok(f"settings.json  →  {json_path}")

    if not labels:
        # 0 free parameters (all blocks fixed) — there is nothing to plot.
        _ok("0 free parameters (all fixed) - corner plot skipped.")
    else:
        print("  │  Generating corner plot...")
        try:
            import matplotlib.pyplot as _plt
            from darksirens.utils.plotting import make_production_corner, make_latent_summary

            # Headline corner: cosmology + physical hyperparameters + survey.
            # For high-dimensional models (GP / gppop) the dozens of xi_* latents
            # are excluded for legibility and summarized separately below.
            fig = make_production_corner(results["samples"], labels)
            corner_path = os.path.join(run_dir, "corner.pdf")
            fig.savefig(corner_path, bbox_inches="tight", dpi=200)
            _plt.close(fig)
            _ok(f"corner.pdf     →  {corner_path}")

            fig_lat = make_latent_summary(results["samples"], labels)
            if fig_lat is not None:
                latents_path = os.path.join(run_dir, "latents.pdf")
                fig_lat.savefig(latents_path, bbox_inches="tight", dpi=200)
                _plt.close(fig_lat)
                _ok(f"latents.pdf    →  {latents_path}")
        except ModuleNotFoundError as e:
            _warn(f"Corner plot skipped; optional plotting dependency is missing: {e.name}")
        except Exception as e:
            _warn(f"Corner plot failed: {e}")

    _end()

    print()
    _banner(f"DONE  │  total wall time {t_end - t_start}")
    print()


def main(argv=None):
    t_start = datetime.datetime.now()
    print()
    _banner(f"DARK SIRENS  │  {t_start.strftime('%Y-%m-%d  %H:%M:%S')}")
    print()

    optp = build_parser()
    opts = optp.parse_args(argv)

    _normalize_multitracer_paths(opts)
    # Guard 6 (PLAN 4.4) is pure opts arithmetic, so it fires here -- before any
    # catalog, artifact or depth map is opened -- rather than at
    # likelihood-build time where redshift/completion.py re-asserts the same
    # exclusions on the loaded pytree.  An over-specified completion model
    # should cost the operator a second, not a load.
    _check_latent_field_mode(opts)
    # Guard 1 (PLAN 4.4 successor 1) immediately after: the artifact is now
    # known to exist, the fingerprint reads only its small datasets, and
    # settings.json -- written further down, BEFORE the likelihood build -- is
    # the run's provenance record, which in latent mode is these two digests.
    _stamp_latent_artifact_fingerprint(opts)
    _resolve_catalog_sky_weighting(opts)
    _validate_multitracer_config(opts)
    _canonicalize_fixed_flags(opts)
    _configure_performance_grids(opts)
    prior_overrides, fixed_parameter_values = _parse_structured_options(opts)
    _resolve_sampler_config(opts)
    _apply_bright_siren_overrides(opts)
    _print_all_cli_options(
        optp,
        opts,
        normalization_grid=normalization_grid_settings().to_dict(),
    )
    _validate_run_config(opts)
    _print_run_configuration(opts, prior_overrides, fixed_parameter_values)
    data = _load_and_report_data(opts)
    if _maybe_run_completion_validation(
        opts, data, prior_overrides, fixed_parameter_values
    ):
        return
    _resolve_single_catalog_marks(opts, data)
    pspace = _build_and_report_parameter_space(
        opts, data, prior_overrides, fixed_parameter_values
    )
    # Run directory + settings.json BEFORE the likelihood build and sampling:
    # from here on every failure mode leaves a record on disk, the sampler has
    # somewhere to checkpoint, and --resume has a directory to continue in.
    run_dir, run_timestamp, settings = _prepare_run_dir(
        opts, data, pspace, fixed_parameter_values, prior_overrides
    )

    def guard(stage):
        return _failure_guard(
            run_dir, stage, labels=pspace.labels, settings=settings
        )

    with guard("build_likelihood"):
        likelihood = _build_likelihood(opts, data, pspace, fixed_parameter_values)
    with guard("sampler"):
        results, wall_sampling = _run_sampling(opts, likelihood, pspace)
    with guard("save"):
        _save_outputs(
            opts, data, results, pspace,
            fixed_parameter_values, prior_overrides, wall_sampling, t_start,
            run_dir, run_timestamp,
        )


def console_main():
    """``darksirens_inference`` console-script target.

    Routes the installed script through ``run_cli`` so the console-script path
    gets the same GPU teardown guard as ``python -m darksirens.cli.inference``.
    """
    run_cli(main)


if __name__ == "__main__":
    run_cli(main)
