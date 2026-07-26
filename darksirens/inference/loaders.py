"""Staged data loading helpers for inference."""

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.gw.samples import load_gw_samples, load_selection_samples
from darksirens.catalogs.io import load_survey

from darksirens.redshift.grid import zgrid
from darksirens.redshift.completion import (
    build_field_delta_g_inputs,
    build_field_depth_inputs,
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
    build_field_mark_inputs,
    build_field_normalization_inputs,
    compute_lss_overdensity,
)
from darksirens.core.model_kinds import BRIGHT_SIREN_MODELS, GALAXY_AWARE_MODELS

from darksirens.catalogs.compact import (
    _catalog_memory_diagnostics,
    _compact_catalog_for_pixels,
    validate_loaded_survey_shapes,
)
from darksirens.likelihood.catalog_views import (
    field_depth_inputs_required,
    unique_inference_pixels,
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
                "--drop_full_catalog is incompatible with --use_lss: the LSS "
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
    posterior / injection sky directions via a per-catalog ``hp.ang2pix``), its
    own LSS overdensity field (computed per catalog under --use_lss; the
    memory-efficient dummy otherwise), its Q_LSS table/ensemble, its galaxy
    marks, and -- under the field convention -- its survey-global
    normalization inputs including the matching budget-modulation rows.

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
        # Compact ONCE over the PE-union-selection pixel set so this bundle's PE
        # and selection views SHARE a single galaxy table (and, downstream, a
        # single KDE cache and singly-compacted Q/mark rows), mirroring the flat
        # single-catalog union path in
        # ``catalog_views.prepare_catalog_views``.  Only the per-view
        # sample->row maps differ; ``zgals_pe``/``unique_pixels_pe`` are the SAME
        # objects as their ``_sel`` counterparts, which ``prepare_catalog_views``
        # and ``factory`` detect (``is``) to alias the device arrays and build one
        # cache per bundle instead of duplicating both across the two views.
        union_pixels = unique_inference_pixels(pixels_pe, pixels_sel)
        z_u = zgals[union_pixels]
        dz_u = dzgals[union_pixels]
        w_u = wgals[union_pixels]
        n_u = ngals[union_pixels]
        s2u_pe = np.searchsorted(union_pixels, pixels_pe).astype(np.int32, copy=False)
        s2u_se = np.searchsorted(union_pixels, pixels_sel).astype(np.int32, copy=False)

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

        # Per-catalog galaxy marks (marked-host model): load + z-centre THIS
        # catalog's selected marks from its own file, keyed by dataset name
        # (mark_logmstar, ...).  The factory gathers them to the compact view
        # rows; a catalog with an empty mark list runs the plain galaxy-count
        # host model (h == 1) inside the mixture.
        _mnbc = getattr(opts, "mark_names_by_catalog", None)
        mark_names_k = (
            tuple(_mnbc[i] or ()) if _mnbc and i < len(_mnbc) else ()
        )
        centred_marks_k = {}
        if (getattr(opts, "mark_model", "none") not in (None, "none")
                and mark_names_k):
            from darksirens.catalogs.marks import load_and_center_survey_marks
            from darksirens.marks import MARK_FIELDS

            # Dataset-level filter: this catalog's file may carry marks beyond
            # the ones selected for it (mark_names_k), so read/centre only the
            # requested datasets instead of every mark table present.
            centred = load_and_center_survey_marks(
                path, zgals, ngals,
                datasets=tuple(MARK_FIELDS[name] for name in mark_names_k),
            )
            for name in mark_names_k:
                key = MARK_FIELDS[name]
                if key not in centred:
                    raise ValueError(
                        f"Catalog {i + 1} ({path}) does not provide the "
                        f"selected mark '{name}' (dataset {key})."
                    )
                centred_marks_k[name] = centred[key]

        # Per-catalog LSS overdensity: computed from THIS catalog's full-sky
        # rows when --use_lss (each tracer has its own clustering field and
        # its own sampled b_miss_c{k}); the memory-efficient (1, N_grid) dummy
        # otherwise.  A catalog carrying a Q_LSS table keeps the dummy: Q
        # REPLACES the local-overdensity factor in the numerator.
        if bool(getattr(opts, "use_LSS", False)) and _lss_for(i) is None:
            delta_g_k = compute_lss_overdensity(
                zgals, nside, wgals=wgals, ngals=ngals
            )
        else:
            delta_g_k = jnp.zeros((1, len(zgrid)))

        bundle = dict(
            nside=nside,
            apix=apix,
            z_depth=z_depth,
            n_pix_catalog=npix,
            delta_g_pix_z=delta_g_k,
            zgals_pe=z_u, dzgals_pe=dz_u, wgals_pe=w_u, ngals_pe=n_u,
            unique_pixels_pe=union_pixels, sample_to_unique_pe=s2u_pe,
            zgals_sel=z_u, dzgals_sel=dz_u, wgals_sel=w_u, ngals_sel=n_u,
            unique_pixels_sel=union_pixels, sample_to_unique_sel=s2u_se,
        )
        bundle.update(lss)  # lss_completion_logq / _logq_members / _indexing
        from darksirens.marks import MARK_FIELDS as _MF
        for _name, _tbl in centred_marks_k.items():
            bundle[_MF[_name]] = _tbl

        # FIELD-convention sky weighting: precompute this catalog's survey-global
        # normalization inputs from the FULL-sky rows (before they are dropped),
        # so the mixture weight measures the host FRACTION for each tracer.
        if getattr(opts, "catalog_sky_weighting", "conditional") == "field":
            field = build_field_normalization_inputs(zgals, wgals, ngals)
            bundle["field_dN_obs_s"] = field.dN_obs_s
            bundle["field_n_empty"] = field.n_empty
            bundle["field_N_obs_total"] = field.N_obs_total
            bundle["field_occupied_pixels"] = field.occupied_pixels
            # DEPTH-consistent global observed term: with a survey depth the raw
            # field_N_obs_total counts every above-depth catalogued galaxy twice
            # (once observed, again in the relaxed missing budget), so the
            # normalizer needs the per-galaxy below-depth kernel geometry.  Built
            # from the FULL-sky rows here -- the bundle is compacted below and the
            # full rows are dropped.
            if field_depth_inputs_required(opts, z_depth):
                depth_inputs = build_field_depth_inputs(
                    zgals, dzgals, wgals, ngals
                )
                bundle["field_depth_z"] = depth_inputs.z
                bundle["field_depth_dz"] = depth_inputs.dz
                bundle["field_depth_c"] = depth_inputs.c
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
            if centred_marks_k:
                fz, fw, fvals = build_field_mark_inputs(
                    zgals, wgals, ngals, centred_marks_k, mark_names_k
                )
                bundle["field_mark_z"] = fz
                bundle["field_mark_w"] = fw
                bundle["field_mark_values"] = fvals
            if np.asarray(delta_g_k).shape[0] > 1:
                bundle["field_delta_g"] = build_field_delta_g_inputs(
                    delta_g_k, field.occupied_pixels
                )

        # Per-catalog completion validation (dry run) needs each catalog's own
        # per-sample GLOBAL pixel indices, which compaction otherwise discards.
        if getattr(opts, "validate_completion", False):
            bundle["pixels_pe_full"] = pixels_pe
            bundle["pixels_sel_full"] = pixels_sel

        bundles.append(bundle)

    # SHARED-member-index contract: the K-catalog mixture marginalizes over ONE
    # member index (member m of every catalog samples the SAME LSS realization),
    # so the per-catalog ensembles must (a) come from ONE matched realization
    # set and (b) have equal M.  Provenance (a) is verified from the
    # realization_set_id stamped by save_lss_completion_hdf5 -- two surveys'
    # ensembles built in separate builder runs carry DISTINCT ids, so pairing
    # member m across them marginalizes over an INDEPENDENT-fields product prior
    # rather than the matched shared-field prior the estimator assumes.
    if bool(getattr(opts, "lss_marginalize", False)) and len(bundles) >= 2:
        allow_unverified = bool(
            getattr(opts, "allow_unverified_shared_lss_members", False)
        )
        prov_by_catalog = []
        for path, bundle in zip(survey_paths, bundles):
            prov = bundle.get("lss_completion_provenance") or {}
            prov_by_catalog.append(
                (prov.get("path") or path, prov.get("realization_set_id"))
            )
        ids = [rid for _p, rid in prov_by_catalog]
        matched = all(rid is not None for rid in ids) and len(set(ids)) == 1
        if not matched:
            listing = "\n".join(
                f"  - {p}: "
                + (f"realization_set_id={rid}" if rid is not None
                   else "LEGACY (no provenance)")
                for p, rid in prov_by_catalog
            )
            if allow_unverified:
                print(
                    "\n"
                    "  ##########################################################\n"
                    "  ## [!!] --allow_unverified_shared_lss_members\n"
                    "  ## The per-catalog Q_LSS ensembles do NOT share a verified\n"
                    "  ## realization set. Treating the member pairing as\n"
                    "  ## unverified: this marginalizes over an INDEPENDENT-fields\n"
                    "  ## product prior across catalogs, NOT the matched\n"
                    "  ## shared-field prior the estimator assumes. member m of\n"
                    "  ## each catalog is an INDEPENDENT LSS draw, so\n"
                    "  ## logsumexp_m logL(Q_m) - log M does not integrate the\n"
                    "  ## single shared missing-galaxy field. Proceeding anyway.\n"
                    "  ## per-catalog Q provenance:\n"
                    + "\n".join(
                        "  ## " + ln.lstrip() for ln in listing.splitlines()
                    )
                    + "\n"
                    "  ##########################################################\n"
                )
            else:
                raise ValueError(
                    "--lss_marginalize with a K>=2 mixture marginalizes over ONE "
                    "SHARED member index: member m of every catalog must sample "
                    "the SAME LSS realization. The per-catalog Q_LSS ensembles do "
                    "not share a verified realization set (equal, non-null "
                    "realization_set_id):\n" + listing + "\n"
                    "Pairing member m across independently built ensembles "
                    "marginalizes over an INDEPENDENT-fields product prior, not "
                    "the matched shared-field prior. Remedies: (1) rebuild the "
                    "ensembles JOINTLY with "
                    "darksirens_build_joint_lognormal_completion (ONE builder run "
                    "over all K catalogs infers a single shared LSS field and "
                    "stamps the same realization_set_id across the K files), or "
                    "(2) pass --allow_unverified_shared_lss_members to explicitly "
                    "accept the independent-fields approximation."
                )

        # Equal-M is the in-jit vmap requirement; still enforced even under the
        # allow_unverified path (mismatched but same-length ensembles proceed).
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
    # LSS-conditioned lognormal completion table Q_LSS (optional)
    # --------------------------------------------------------
    # Loaded FIRST because a Q table REPLACES the local-overdensity factor:
    # the numerator's rule is `max(1 + b_eff delta_g, 0)` OR Q, never both
    # (redshift/completion.py:_completion_curves_row_q), and the field-convention
    # normalizer enforces that as a hard invariant -- field_global_log_Z raises
    # NotImplementedError if both field_lss_q and field_delta_g are populated.
    # Building delta_g anyway made `--use_lss --lss_completion Q.h5` abort during
    # the jit trace, AFTER the full catalog load and KDE-cache build, with a
    # message about an internal field-normalizer invariant rather than about the
    # offending flag combination.
    data.update(maybe_load_lss_completion(opts, zgrid=zgrid))
    _q_active = data.get("lss_completion_logq") is not None

    # --------------------------------------------------------
    # LSS overdensity field (Handle memory carefully)
    # --------------------------------------------------------
    print(f"[*] Preparing LSS/Overdensity Field...")
    if opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS and _q_active:
        print(
            "    - Q_LSS completion table is active: it REPLACES the "
            "(1 + b_eff*delta_g) local-overdensity factor, so --use_lss is "
            "redundant here and the overdensity grid is NOT built."
        )
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))
    elif opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS:
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
    return data


def attach_mark_inputs(opts, data) -> dict:
    """Attach per-galaxy marks for the marked-host model when available."""
    # --------------------------------------------------------
    # Per-galaxy marks for the marked-host model (optional)
    # --------------------------------------------------------
    for _ds in ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color"):
        data[_ds] = None
    # mark_model="none" (the default) never reads the marked-host efficiency
    # h(m|eta), so skip the mark I/O entirely: load_and_center_survey_marks
    # loads and z-centers every full-size padded mark table present in the
    # survey file (~0.83 GB per float64 mark at DESI-wide shape), which is
    # pure waste when no mark model is selected.
    mark_model = getattr(opts, "mark_model", "none")
    if (mark_model not in (None, "none")
            and opts.survey_path is not None
            and opts.universe_model in GALAXY_AWARE_MODELS):
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
