"""Staged data loading helpers for inference."""

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.gw.samples import load_gw_samples, load_selection_samples
from darksirens.catalogs.io import load_survey

from darksirens.redshift.grid import zgrid
from darksirens.redshift.completion import (
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
    build_field_normalization_inputs,
    compute_lss_overdensity,
)
from darksirens.core.model_kinds import BRIGHT_SIREN_MODELS, GALAXY_AWARE_MODELS

from darksirens.catalogs.compact import (
    _catalog_memory_diagnostics,
    _compact_catalog_for_pixels,
    validate_loaded_survey_shapes,
)
from darksirens.catalogs.counterparts import build_counterpart_catalog
from darksirens.catalogs.lss import maybe_load_lss_completion
from darksirens.catalogs.marks import load_and_center_survey_marks



def load_or_build_catalog_inputs(opts) -> dict:
    """Load survey catalog inputs or build the bright-siren counterpart catalog."""
    nside = None
    npix = None
    zgals = dzgals = wgals = None
    ngals = None
    apix = 0.0
    z_depth = None
    counterpart_pixel = None
    counterpart_pixels = None
    counterpart_zs = None
    counterpart_dzs = None
    bright_siren_sky_marginalized = bool(
        getattr(opts, "bright_siren_sky_marginalized", False)
    )

    if getattr(opts, "drop_full_catalog", False) and opts.universe_model in BRIGHT_SIREN_MODELS:
        raise ValueError(
            "--drop_full_catalog is incompatible with bright-siren models: "
            "the counterpart prior needs the full-sky galaxy rows."
        )

    # Load survey data, or build the synthetic counterpart catalog used by
    # bright sirens.  The counterpart is not a survey hyperparameter: it is
    # fixed event metadata supplied through the inference CLI.
    if opts.universe_model in BRIGHT_SIREN_MODELS:
        if opts.counterpart is None:
            raise ValueError("bright_sirens requires opts.counterpart=RA DEC Z triplets.")
        counterpart_catalog = build_counterpart_catalog(
            opts.counterpart,
            opts.counterpart_dz,
            opts.counterpart_nside,
            sky_marginalized=bright_siren_sky_marginalized,
        )
        nside = counterpart_catalog["nside"]
        npix = counterpart_catalog["npix"]
        zgals = counterpart_catalog["zgals"]
        dzgals = counterpart_catalog["dzgals"]
        wgals = counterpart_catalog["wgals"]
        ngals = counterpart_catalog["ngals"]
        apix = counterpart_catalog["apix"]
        counterpart_pixel = counterpart_catalog["counterpart_pixel"]
        counterpart_pixels = counterpart_catalog["counterpart_pixels"]
        counterpart_zs = counterpart_catalog["counterpart_zs"]
        counterpart_dzs = counterpart_catalog["counterpart_dzs"]
        bright_siren_sky_marginalized = counterpart_catalog[
            "bright_siren_sky_marginalized"
        ]
    elif opts.survey_path is not None:
        # When dropping the full-sky catalog, load it on the host so compaction
        # happens before any device transfer; only the compact views go to GPU.
        drop_full_catalog = getattr(opts, "drop_full_catalog", False)
        if drop_full_catalog and opts.use_LSS:
            raise ValueError(
                "--drop_full_catalog is incompatible with --use_LSS: the LSS "
                "overdensity field needs the full-sky galaxy rows."
            )
        nside, ngals, zgals, dzgals, wgals, z_depth = load_survey(
            opts.survey_path, to_device=not drop_full_catalog
        )
        npix = hp.nside2npix(nside)
        apix = hp.nside2pixarea(nside)
    else:
        # If no survey, we might still need a default nside for 
        # pixelization logic in other parts of the code
        nside = 1
        npix = hp.nside2npix(nside)

    return dict(
        nside=nside,
        npix=npix,
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
        ngals=ngals,
        apix=apix,
        z_depth=z_depth,
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized,
    )


def load_multitracer_catalog_bundles(opts, gw_inputs) -> list:
    """Load one compact catalog bundle per survey path for the K-catalog mixture.

    Each bundle is self-contained input to
    :func:`darksirens.likelihood.catalog_views.prepare_catalog_views`: its own
    ``nside``/``apix``, compact PE and selection views (built from the SAME GW
    posterior / injection sky directions via a per-catalog ``hp.ang2pix``), a
    dummy ``(1, N_grid)`` overdensity field (LSS is unsupported for the mixture),
    and its deterministic Q_LSS table.

    ``opts.survey_paths`` is the aligned list of catalog paths;
    ``opts.lss_completions`` (if present) is positionally aligned, with ``""`` or
    a missing entry meaning "no external completion for that catalog".
    """
    from types import SimpleNamespace

    survey_paths = list(getattr(opts, "survey_paths", None) or [])
    lss_list = list(getattr(opts, "lss_completions", None) or [])

    def _lss_for(i):
        if i < len(lss_list):
            entry = lss_list[i]
            return entry if entry not in (None, "") else None
        return None

    ra = np.asarray(gw_inputs["ra"])
    dec = np.asarray(gw_inputs["dec"])
    rasels = np.asarray(gw_inputs["rasels"])
    decsels = np.asarray(gw_inputs["decsels"])

    bundles = []
    for i, path in enumerate(survey_paths):
        # Host-side load + compaction so only the compact per-catalog views reach
        # the device (mirrors the single-catalog drop_full_catalog memory path).
        nside, ngals, zgals, dzgals, wgals, z_depth = load_survey(path, to_device=False)
        npix = hp.nside2npix(nside)
        apix = hp.nside2pixarea(nside)

        pixels_pe = np.asarray(
            hp.ang2pix(nside, np.pi / 2 - dec, ra), dtype=np.int32
        )
        pixels_sel = np.asarray(
            hp.ang2pix(nside, np.pi / 2 - decsels, rasels), dtype=np.int32
        )
        (
            up_pe, s2u_pe, zpe, dzpe, wpe, npe,
        ) = _compact_catalog_for_pixels(pixels_pe, zgals, dzgals, wgals, ngals)
        (
            up_se, s2u_se, zse, dzse, wse, nse,
        ) = _compact_catalog_for_pixels(pixels_sel, zgals, dzgals, wgals, ngals)

        # Per-catalog Q_LSS (deterministic table + optional ensemble when
        # --lss_marginalize); a private namespace keeps the single-catalog
        # loader contract.
        lss_ns = SimpleNamespace(
            survey_path=path,
            lss_completion=_lss_for(i),
            universe_model=opts.universe_model,
            lss_marginalize=bool(getattr(opts, "lss_marginalize", False)),
        )
        lss = maybe_load_lss_completion(lss_ns, zgrid=zgrid)

        bundle = dict(
            nside=nside,
            apix=apix,
            z_depth=z_depth,
            n_pix_catalog=npix,
            delta_g_pix_z=jnp.zeros((1, len(zgrid))),
            zgals_pe=zpe, dzgals_pe=dzpe, wgals_pe=wpe, ngals_pe=npe,
            unique_pixels_pe=up_pe, sample_to_unique_pe=s2u_pe,
            zgals_sel=zse, dzgals_sel=dzse, wgals_sel=wse, ngals_sel=nse,
            unique_pixels_sel=up_se, sample_to_unique_sel=s2u_se,
        )
        bundle.update(lss)  # lss_completion_logq / _logq_members / _indexing

        # FIELD-convention sky weighting: precompute this catalog's survey-global
        # normalization inputs from the FULL-sky rows (before they are dropped),
        # so the mixture weight measures the host FRACTION for each tracer.
        if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
            field = build_field_normalization_inputs(zgals, wgals, ngals)
            bundle["field_dN_obs_s"] = field.dN_obs_s
            bundle["field_n_empty"] = field.n_empty
            bundle["field_N_obs_total"] = field.N_obs_total
            bundle["field_occupied_pixels"] = field.occupied_pixels
            # Budget-modulation rows: mirror this catalog's deterministic Q
            # table into its survey-global normalizer so numerator and Z carry
            # the SAME missing-galaxy budget (field_global_log_Z).
            logq_full = bundle.get("lss_completion_logq")
            q_full = bundle.get("lss_completion_q")
            if logq_full is not None or q_full is not None:
                logq_np = (
                    np.asarray(logq_full)
                    if logq_full is not None
                    else np.log(np.maximum(np.asarray(q_full), 1e-300))
                )
                q_occ, q_empty_sum = build_field_lss_q_inputs(
                    logq_np, field.occupied_pixels, npix
                )
                bundle["field_lss_q"] = q_occ
                bundle["field_lss_q_empty_sum"] = q_empty_sum
            logq_members = bundle.get("lss_completion_logq_members")
            if logq_members is not None:
                qm_occ, qm_empty_sum = build_field_lss_q_member_inputs(
                    np.asarray(logq_members), field.occupied_pixels, npix
                )
                bundle["field_lss_q_members"] = qm_occ
                bundle["field_lss_q_empty_sum_members"] = qm_empty_sum

        # Per-catalog completion validation (dry run) needs each catalog's own
        # per-sample GLOBAL pixel indices, which compaction otherwise discards.
        if getattr(opts, "validate_completion", False):
            bundle["pixels_pe_full"] = pixels_pe
            bundle["pixels_sel_full"] = pixels_sel

        bundles.append(bundle)

    # SHARED-member-index contract: the K-catalog mixture marginalizes over ONE
    # member index (member m of every catalog samples the same LSS
    # realization), so the per-catalog ensembles must have equal M.
    if bool(getattr(opts, "lss_marginalize", False)) and len(bundles) >= 2:
        member_counts = {}
        for path, bundle in zip(survey_paths, bundles):
            members = bundle.get("lss_completion_logq_members")
            if members is not None:
                member_counts[path] = int(np.asarray(members).shape[0])
        if len(set(member_counts.values())) > 1:
            detail = ", ".join(f"{p}: M={m}" for p, m in member_counts.items())
            raise ValueError(
                "--lss_marginalize with a K-catalog mixture requires EQUAL "
                "ensemble sizes across catalogs (a shared member index over "
                f"matched LSS realizations); got {detail}. Rebuild the Q "
                "ensembles with the same --n-members from matched realizations."
            )

    return bundles


def _parse_pdet_cosmology(opts) -> tuple[float, float]:
    """'H0,Om0' -> floats for the P_det emulator's injection-campaign cosmology."""
    raw = getattr(opts, "pdet_cosmology", None)
    if raw is None or raw == "":
        from darksirens.gw.selection import PDET_INJ_H0, PDET_INJ_OM0

        return float(PDET_INJ_H0), float(PDET_INJ_OM0)
    if isinstance(raw, (tuple, list)):
        h0, om0 = raw
    else:
        parts = str(raw).split(",")
        if len(parts) != 2:
            raise ValueError(
                "--pdet_cosmology must be 'H0,Om0' (e.g. '67.9,0.3065'); "
                f"got {raw!r}."
            )
        h0, om0 = parts
    return float(h0), float(om0)


def resolve_selection_inputs(opts):
    """Selection inputs from exactly one source: injection HDF5 or P_det emulator.

    Returns the 8-tuple ``(m1detsels, m2detsels, dLsels, chieffsels, rasels,
    decsels, p_draw, Ndraw)`` shared by both sources.  ``--pdet_flow_path``
    generates pseudo-injections from the emulator flow once at load time;
    everything downstream (GWEvent packing, compute_selection_term, Neff
    guards, batching) is source-agnostic.
    """
    if getattr(opts, "pdet_flow_path", None):
        from darksirens.gw.selection import pseudo_injections_from_pdet_flow

        h0, om0 = _parse_pdet_cosmology(opts)
        return pseudo_injections_from_pdet_flow(
            opts.pdet_flow_path,
            nsamp=int(getattr(opts, "pdet_nsamp", 1_000_000)),
            seed=int(getattr(opts, "pdet_seed", 42)),
            H0=h0,
            Om0=om0,
            chieff_amax=float(getattr(opts, "pdet_chieff_amax", 0.99)),
        )
    return load_selection_samples(opts.gwselection_path)


def load_gw_and_selection_inputs(opts) -> dict:
    """Load GW posterior and selection samples."""
    # Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
        opts.gw_path
    )

    # Load Selection samples (Always required)
    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = resolve_selection_inputs(opts)

    return dict(
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        ra=ra,
        dec=dec,
        p_pe=p_pe,
        nEvents=nEvents,
        nsamp=nsamp,
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        rasels=rasels,
        decsels=decsels,
        p_draw=p_draw,
        Ndraw=Ndraw,
    )


def load_flow_and_selection_inputs(opts) -> dict:
    """Load per-event flow surrogates plus selection samples (no PE samples).

    The flow-surrogate path replaces the gwcat PE store: single-event
    posteriors come from ``--gw_flows_path`` (a directory of
    ``<EVENT>/<EVENT>_flow.npz`` checkpoints) and the per-event term draws
    from the population model at every likelihood call.  All PE-sample keys
    are returned as None so downstream sky/catalog plumbing can skip them.
    Event order/identity is the ensemble's sorted checkpoint order,
    surfaced as ``flow_event_names``.
    """
    from darksirens.gw.flows import load_flow_ensemble

    ensemble = load_flow_ensemble(
        opts.gw_flows_path,
        pattern=getattr(opts, "flows_pattern", "*/*_flow.npz"),
        on_mismatch=getattr(opts, "flows_on_mismatch", "error"),
    )
    print(f"    - {ensemble.summary().splitlines()[0]}")

    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = resolve_selection_inputs(opts)

    return dict(
        m1det=None,
        m2det=None,
        dL=None,
        chieff=None,
        ra=None,
        dec=None,
        p_pe=None,
        nEvents=ensemble.n_flows,
        nsamp=int(getattr(opts, "flows_nsamp", 4096)),
        flow_ensemble=ensemble,
        flow_event_names=list(ensemble.names),
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        rasels=rasels,
        decsels=decsels,
        p_draw=p_draw,
        Ndraw=Ndraw,
    )


def compute_sky_pixels_and_vectors(opts, catalog_inputs, gw_inputs) -> dict:
    """Compute HEALPix indices, sky vectors, and compact catalog views."""
    nside = catalog_inputs["nside"]
    zgals = catalog_inputs["zgals"]
    dzgals = catalog_inputs["dzgals"]
    wgals = catalog_inputs["wgals"]
    ngals = catalog_inputs["ngals"]
    counterpart_pixels = catalog_inputs["counterpart_pixels"]

    # Flow-surrogate runs carry no PE samples (ra is None): the PE-side sky
    # products stay None and only the selection side is computed.
    has_pe = gw_inputs.get("ra") is not None

    pixels_pe = (
        hp.ang2pix(nside, jnp.pi/2 - gw_inputs["dec"], gw_inputs["ra"])
        if has_pe else None
    )
    pixels_sel = hp.ang2pix(nside, jnp.pi/2 - gw_inputs["decsels"], gw_inputs["rasels"])

    # Sky-direction unit vectors n̂ = (cos δ cos α, cos δ sin α, sin δ), retained
    # per sample for the angular/sky model.  Unlike ``pixels`` (whose resolution
    # collapses to nside=1 with no survey), these give the sky model full
    # angular resolution in every universe model, including GW-only runs.
    if has_pe:
        nx_pe, ny_pe, nz_pe = (
            jnp.cos(gw_inputs["dec"]) * jnp.cos(gw_inputs["ra"]),
            jnp.cos(gw_inputs["dec"]) * jnp.sin(gw_inputs["ra"]),
            jnp.sin(gw_inputs["dec"]),
        )
    else:
        nx_pe = ny_pe = nz_pe = None
    nx_sel, ny_sel, nz_sel = (
        jnp.cos(gw_inputs["decsels"]) * jnp.cos(gw_inputs["rasels"]),
        jnp.cos(gw_inputs["decsels"]) * jnp.sin(gw_inputs["rasels"]),
        jnp.sin(gw_inputs["decsels"]),
    )

    zgals_pe = dzgals_pe = wgals_pe = None
    zgals_sel = dzgals_sel = wgals_sel = None
    unique_pixels_pe = unique_pixels_sel = None
    sample_to_unique_pe = sample_to_unique_sel = None
    ngals_pe = ngals_sel = None
    catalog_memory = None

    if zgals is not None:
        if not has_pe:
            raise NotImplementedError(
                "Galaxy-catalog runs require PE-side sky positions; the "
                "flow-surrogate path (no stored PE samples) supports "
                "universe_model='spectral_sirens' only."
            )
        required_pixels = (
            counterpart_pixels
            if opts.universe_model in BRIGHT_SIREN_MODELS and counterpart_pixels is not None
            else None
        )
        (
            unique_pixels_pe, sample_to_unique_pe,
            zgals_pe, dzgals_pe, wgals_pe, ngals_pe,
        ) = _compact_catalog_for_pixels(
            pixels_pe, zgals, dzgals, wgals, ngals, required_pixels=required_pixels
        )
        (
            unique_pixels_sel, sample_to_unique_sel,
            zgals_sel, dzgals_sel, wgals_sel, ngals_sel,
        ) = _compact_catalog_for_pixels(
            pixels_sel, zgals, dzgals, wgals, ngals, required_pixels=required_pixels
        )

        catalog_memory = _catalog_memory_diagnostics(
            zgals, dzgals, wgals, pixels_pe, pixels_sel, ngals_pe, ngals_sel
        )
        print(
            "    - Compact catalog rows: "
            f"PE {catalog_memory['unique_pe_pixels']:,}, "
            f"selection {catalog_memory['unique_sel_pixels']:,}"
        )
        print(
            "    - Duplicated catalog bytes avoided: "
            f"{catalog_memory['duplicated_catalog_bytes_avoided'] / 1e9:.4f} GB"
        )
        print(
            "    - Max galaxies per unique inference pixel: "
            f"{catalog_memory['max_galaxies_per_unique_pixel']:,}"
        )

    return dict(
        pixels_pe=pixels_pe,
        pixels_sel=pixels_sel,
        nx_pe=nx_pe,
        ny_pe=ny_pe,
        nz_pe=nz_pe,
        nx_sel=nx_sel,
        ny_sel=ny_sel,
        nz_sel=nz_sel,
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,
        ngals_pe=ngals_pe,
        unique_pixels_pe=unique_pixels_pe,
        sample_to_unique_pe=sample_to_unique_pe,
        zgals_sel=zgals_sel,
        dzgals_sel=dzgals_sel,
        wgals_sel=wgals_sel,
        ngals_sel=ngals_sel,
        unique_pixels_sel=unique_pixels_sel,
        sample_to_unique_sel=sample_to_unique_sel,
        catalog_memory=catalog_memory,
    )


def attach_lss_inputs(opts, data) -> dict:
    """Attach LSS overdensity and optional LSS-conditioned completion inputs."""
    nside_check = data.get("nside", "N/A")

    # --------------------------------------------------------
    # LSS overdensity field (Handle memory carefully)
    # --------------------------------------------------------
    print(f"[*] Preparing LSS/Overdensity Field...")
    if opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS:
        print(f"    - Calculating high-resolution overdensity grid...")
        delta_g_pix_z = compute_lss_overdensity(
            data["zgals"],
            nside_check,
            wgals=data.get("wgals"),
            ngals=data.get("ngals_catalog"),
        )
    else:
        print(f"    - Non-LSS run. Creating memory-efficient dummy (1, {len(zgrid)}) grid.")
        # We use shape (1, nz) to satisfy JAX broadcasting without 93GB allocations
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))

    mem_usage = delta_g_pix_z.nbytes / 1e9
    print(f"    - Overdensity array shape: {delta_g_pix_z.shape} ({mem_usage:.4f} GB)")

    # Append the LSS overdensity field to the returned dictionary
    data["delta_g_pix_z"] = delta_g_pix_z

    # --------------------------------------------------------
    # LSS-conditioned lognormal completion table Q_LSS (optional)
    # --------------------------------------------------------
    data.update(maybe_load_lss_completion(opts, zgrid=zgrid))
    return data


def attach_mark_inputs(opts, data) -> dict:
    """Attach per-galaxy marks for the marked-host model when available."""
    # --------------------------------------------------------
    # Per-galaxy marks for the marked-host model (optional)
    # --------------------------------------------------------
    for _ds in ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color"):
        data[_ds] = None
    if opts.survey_path is not None and opts.universe_model in GALAXY_AWARE_MODELS:
        data.update(
            load_and_center_survey_marks(
                opts.survey_path, data["zgals"], data.get("ngals_catalog")
            )
        )
    return data


def attach_wl_inputs(opts, data) -> dict:
    """Attach weak-lensing magnification inputs."""
    # --------------------------------------------------------
    # Weak-lensing magnification model (optional; built only for the WL
    # universe model, inert/None otherwise).
    # --------------------------------------------------------
    if getattr(opts, "universe_model", None) == "spectral_sirens_wl":
        import h5py
        from darksirens.lensing.wlmagnification import (
            make_lognormal_wl_params,
            make_tabulated_wl_params,
        )

        if opts.lensing_wl_model == "lognormal":
            wl_params = make_lognormal_wl_params(
                a=opts.lensing_wl_a,
                b=opts.lensing_wl_b,
            )
        elif opts.lensing_wl_model == "tabulated":
            with h5py.File(opts.lensing_wl_table_path, "r") as f:
                z_grid = jnp.asarray(f["z_grid"])
                log_mu_grid = jnp.asarray(f["log_mu_grid"])
                log_p_table = jnp.asarray(f["log_p_table"])
            wl_params = make_tabulated_wl_params(
                z_grid, log_mu_grid, log_p_table,
            )
        else:
            raise ValueError(
                f"Unknown --lensing_wl_model '{opts.lensing_wl_model}'. "
                "Expected 'lognormal' or 'tabulated'."
            )
        data["wl_params"] = wl_params
    else:
        data["wl_params"] = None
    return data


def maybe_drop_full_catalog(opts, data) -> dict:
    """Optionally drop the dense full-sky catalog arrays."""
    # Optionally drop the dense full-sky catalog arrays
    if getattr(opts, "drop_full_catalog", False):
        if data.get("zgals_pe") is None and data.get("zgals_sel") is None:
            raise ValueError(
                "--drop_full_catalog requires a compacted survey catalog, but no "
                "compact PE/selection views were built (no survey loaded?)."
            )
        dropped_bytes = sum(
            int(np.asarray(data[k]).nbytes)
            for k in ("zgals_catalog", "dzgals_catalog", "wgals_catalog", "ngals_catalog")
            if data.get(k) is not None
        )
        for key in (
            "zgals", "dzgals", "wgals",
            "zgals_catalog", "dzgals_catalog", "wgals_catalog", "ngals_catalog",
        ):
            data[key] = None
        print(
            "    - Dropped dense full-sky catalog arrays "
            f"({dropped_bytes / 1e9:.2f} GB freed); using compact views only."
        )
    return data
