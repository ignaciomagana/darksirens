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
            # Thread the run's completeness mode so the per-catalog loader can
            # hard-check each table's c_mode stamp against it.
            c_mode=getattr(opts, "c_mode", None) or "per_pixel",
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
        # REPLACES the local-overdensity factor in the numerator.  The test is
        # the LOADED table state, not the CLI path helper: an EMBEDDED
        # /lss_completion group (auto-discovered by maybe_load_lss_completion
        # from the survey file itself) arrives with _lss_for(i) None, and
        # building delta_g anyway would populate both field inputs — the
        # field-convention normalizer's hard mutual-exclusion invariant then
        # aborts inside the jit trace (the K=1 flat path was fixed the same
        # way; this is its mixture twin).
        if bool(getattr(opts, "use_LSS", False)) and (
                lss.get("lss_completion_logq") is None):
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
        # A catalog with NO ensemble at all is not a provenance failure: the
        # estimator needs one on EVERY catalog (likelihood/core.py refuses a
        # missing base_miss on either seam, since Z_m cancels only against
        # mu(Q_m)), so diagnose that here instead of reporting the Q-less
        # catalog as "LEGACY (no provenance)" and telling the operator to
        # rebuild the ensembles jointly -- impossible for a catalog that has no
        # LSS field.  Not bypassable by --allow_unverified_shared_lss_members:
        # that flag accepts an unmatched realization set, not a missing one.
        no_ensemble = [
            path for path, bundle in zip(survey_paths, bundles)
            if bundle.get("lss_completion_logq_members") is None
            and bundle.get("lss_completion_q_members") is None
        ]
        if no_ensemble:
            raise ValueError(
                "--lss_marginalize with a K>=2 mixture marginalizes over ONE "
                "SHARED member index, so EVERY catalog needs its own Q_LSS "
                "ENSEMBLE; these carry none:\n"
                + "\n".join(f"  - {p}" for p in no_ensemble) + "\n"
                "Build each catalog's completion with members "
                "(darksirens_build_joint_lognormal_completion, or "
                "darksirens_build_lognormal_completion --n-members M > 0) and "
                "pass them positionally via --lss_completion, or drop "
                "--lss_marginalize for the deterministic posterior-mean Q."
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


#: The two fit-column sets darksirens' population models can request.
_CHIEFF_FIT_COLUMNS = ("m1det", "q", "dL", "chieff")
_COMPONENT_FIT_COLUMNS = ("m1det", "q", "dL", "a1", "a2", "cost1", "cost2")


def _model_requests_component_spin(opts) -> bool:
    """Whether the configured population model consumes the spin block.

    Basis negotiation (DS-09) starts from the MODEL: a model whose spin
    component declares ``consumes_spin_block`` fits the component columns and
    needs a component-basis store; every other model fits chi_eff.  Errors in
    the model name are deliberately not raised here -- the real model build
    reports them with full context.
    """
    pop_model = getattr(opts, "pop_model", None)
    if not pop_model:
        return False
    try:
        from darksirens.gw.populations.registry import get_model

        model = get_model(
            pop_model,
            shared_beta=bool(getattr(opts, "shared_beta", True)),
            shared_spin=bool(getattr(opts, "shared_spin", True)),
            shared_gamma=bool(getattr(opts, "shared_gamma", True)),
        )
    except Exception:
        return False
    components = []
    if hasattr(model, "spin_component"):
        components.append(model.spin_component)
    mixture = getattr(model, "mixture", None)
    if mixture is not None:
        components.extend(getattr(mixture, "spin_components", ()))
    return any(
        getattr(c, "consumes_spin_block", False) for c in components
    )


def required_fit_columns_for(opts):
    """The fit columns the configured run consumes (see DS-09)."""
    return (_COMPONENT_FIT_COLUMNS if _model_requests_component_spin(opts)
            else _CHIEFF_FIT_COLUMNS)


def _read_spin_block(path):
    """(N, 4) component-spin block (a1, a2, cost1, cost2) from a gwcat file.

    Called only after the loader has validated the file against the
    component contract, so the datasets exist.
    """
    import h5py

    with h5py.File(path, "r") as f:
        return np.column_stack([
            np.asarray(f[name]) for name in ("a1", "a2", "cost1", "cost2")
        ])


def resolve_selection_inputs(opts, fit_columns=None):
    """Selection inputs from exactly one source: injection HDF5 or P_det emulator.

    Returns the 8-tuple ``(m1detsels, m2detsels, dLsels, chieffsels, rasels,
    decsels, p_draw, Ndraw)`` shared by both sources.  ``--pdet_flow_path``
    generates pseudo-injections from the emulator flow once at load time;
    everything downstream (GWEvent packing, compute_selection_term, Neff
    guards, batching) is source-agnostic.  ``fit_columns`` (DS-09) is passed
    through to the file loader only when it departs from the chi_eff
    default, so test doubles with the legacy signature keep working.
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
    kwargs = {}
    if fit_columns is not None and tuple(fit_columns) != _CHIEFF_FIT_COLUMNS:
        kwargs["fit_columns"] = tuple(fit_columns)
    return load_selection_samples(
        opts.gwselection_path,
        allow_invalid_spin_swap=bool(
            getattr(opts, "allow_invalid_spin_swap", False)
        ),
        **kwargs,
    )


def _read_store_attrs(path) -> dict:
    """Decoded attrs of a gwcat HDF5 file, or ``{}`` when unreadable.

    Attrs-only open: cheap even for a GB-scale product.  Tolerating an
    unreadable path is deliberate -- the tuple loaders are the validation
    (and, in tests, the monkeypatch) seam, so by the time this runs a real
    file has already been opened and gated; only a test double's placeholder
    path lands in the except arm.
    """
    import h5py

    def _decode(value):
        return value.decode() if isinstance(value, bytes) else value

    try:
        with h5py.File(path, "r") as f:
            return {key: _decode(f.attrs[key]) for key in f.attrs}
    except OSError:
        return {}


def _require_matching_contract(gw_attrs, sel_attrs, gw_path, sel_path) -> None:
    """Compare the gwcat-2.1 pairing contract across a PE/selection pair.

    2.1 files carry a ``contract_hash`` (a digest over the pairing-critical
    declarations: parameter space, fit/advisory columns, spin basis kind,
    sky-measure convention, source-class filter) plus the full ``contract``
    JSON.  A mismatched pair -- e.g. a component-basis PE file against a
    chieff selection file, or a class-filtered PE store against an unfiltered
    injection set -- is silently wrong, so it is refused here with the
    field-by-field difference rather than a bare hash.  Files that predate
    the contract (1.0 / 2.0) are exempt; the per-attr gates still apply.
    """
    pe_hash = gw_attrs.get("contract_hash")
    sel_hash = sel_attrs.get("contract_hash")
    if pe_hash is None or sel_hash is None:
        return
    if pe_hash == sel_hash:
        return
    import json

    diff = ""
    try:
        pe_contract = json.loads(gw_attrs.get("contract", "{}"))
        sel_contract = json.loads(sel_attrs.get("contract", "{}"))
        fields = sorted(set(pe_contract) | set(sel_contract))
        parts = [
            f"{k}: PE={pe_contract.get(k)!r} vs selection={sel_contract.get(k)!r}"
            for k in fields
            if pe_contract.get(k) != sel_contract.get(k)
        ]
        diff = " Differing fields: " + "; ".join(parts) if parts else ""
    except Exception:
        pass
    raise RuntimeError(
        f"PE file {gw_path!r} and selection file {sel_path!r} "
        f"declare different pairing contracts (contract_hash {pe_hash} != "
        f"{sel_hash}): the pair does not describe the same fit and cannot be "
        f"combined in one likelihood.{diff}"
    )


def _require_chieff_pe_for_emulator(gw_attrs, gw_path) -> None:
    """The P_det emulator path is chi_eff-basis by construction.

    ``pseudo_injections_from_pdet_flow`` bakes the 1-D chi_eff draw density
    into its pseudo-injections' ``p_draw``, so pairing it with a PE store in
    any other spin basis puts numerator and denominator on different spin
    measures -- a wrong mu that looks entirely healthy.  Today the PE loader
    also refuses non-chieff bases outright; this check is the emulator-path
    half of that gate, and stays load-bearing once basis negotiation (DS-09)
    widens the loader.
    """
    basis = gw_attrs.get("spin_basis", "chieff")
    space = gw_attrs.get("parameter_space", basis)
    if basis != "chieff" or space != "chieff":
        raise RuntimeError(
            f"--pdet_flow_path generates chi_eff-basis pseudo-injections, but "
            f"the PE file {gw_path!r} declares "
            f"spin_basis={basis!r} / parameter_space={space!r}. Pair the "
            "emulator with a chieff-basis PE store, or use a gwcat selection "
            "file exported in the PE file's basis."
        )


#: Physical tolerances for "these two declared cosmologies are the same one",
#: matching gwcat's pair validator: differences below these move nothing at
#: the precision of the products.
_COSMO_H0_TOL = 1.0
_COSMO_OM0_TOL = 0.05


def _require_matching_pdet_cosmology(gw_attrs, gw_path, opts) -> None:
    """--pdet_cosmology must be the PE store's declared cosmology.

    ``pe_cosmology_H0``/``pe_cosmology_Om0`` are REQUIRED attrs of every PE
    file and were previously read by no consumer anywhere in the package,
    while ``--pdet_cosmology`` was validated for format and range only -- so
    an emulator campaign generated under one cosmology could be paired with a
    PE store built under another and nothing noticed.  Both fiducials enter
    the fixed densities (p_pe's source-frame construction, the emulator's
    injection z-prior and detector-frame conversion), so a mismatched pair
    mixes two cosmologies in one likelihood.
    """
    pe_h0 = gw_attrs.get("pe_cosmology_H0")
    pe_om0 = gw_attrs.get("pe_cosmology_Om0")
    if pe_h0 is None or pe_om0 is None:
        return
    h0, om0 = _parse_pdet_cosmology(opts)
    if (abs(float(pe_h0) - h0) >= _COSMO_H0_TOL
            or abs(float(pe_om0) - om0) >= _COSMO_OM0_TOL):
        raise RuntimeError(
            f"--pdet_cosmology ({h0:g}, {om0:g}) does not match the PE "
            f"store's declared cosmology ({float(pe_h0):g}, {float(pe_om0):g}) "
            f"from {gw_path!r}: the emulator's pseudo-injections and the PE "
            "densities would be built under two different fiducials in one "
            "likelihood. Set --pdet_cosmology to the PE store's values (or "
            "rebuild the PE store)."
        )


def _warn_pair_cosmology(gw_attrs, sel_attrs, gw_path, sel_path) -> None:
    """Surface a PE/selection fiducial-cosmology disagreement.

    A warning, not a refusal: campaigns legitimately carry their own
    generation cosmology (gwcat GW-09 records ``cosmology_source`` per
    campaign, with 'file' meaning no cosmology entered pdraw at all), so
    difference alone is not an error -- but it is exactly the kind of quiet
    configuration drift an operator should see once, at load.
    """
    pe_h0 = gw_attrs.get("pe_cosmology_H0")
    pe_om0 = gw_attrs.get("pe_cosmology_Om0")
    sel_h0 = sel_attrs.get("cosmology_H0")
    sel_om0 = sel_attrs.get("cosmology_Om0")
    if None in (pe_h0, pe_om0, sel_h0, sel_om0):
        return
    try:
        mismatch = (abs(float(pe_h0) - float(sel_h0)) >= _COSMO_H0_TOL
                    or abs(float(pe_om0) - float(sel_om0)) >= _COSMO_OM0_TOL)
    except (TypeError, ValueError):
        return
    if mismatch:
        print(f"    [!] PE store {gw_path!r} declares cosmology "
              f"({pe_h0}, {pe_om0}) but selection file {sel_path!r} declares "
              f"({sel_h0}, {sel_om0}); campaigns may legitimately differ "
              "(per-campaign cosmology_source), but verify this pair was "
              "built together.")


def _warn_per_event_cosmology(gw_attrs, gw_path) -> None:
    """darksirens consumes ONE scalar PE cosmology; say so if the file varies.

    gwcat writes per-event ``cosmology_H0_per_event``/``Om0_per_event``
    arrays plus ``cosmology_per_event_varies``; darksirens requires the
    scalar attrs and never reads the arrays, so if the flag ever flips True
    the scalar is a fiction.  All shipped products are False today -- this is
    the signal for when that stops being true.
    """
    if gw_attrs.get("cosmology_per_event_varies"):
        print(f"    [!] PE store {gw_path!r} declares cosmology_per_event_"
              "varies=True (mode="
              f"{gw_attrs.get('cosmology_mode', '?')!r}): darksirens uses the "
              "single scalar pe_cosmology_H0/Om0 for every event, which does "
              "not describe this file. Downstream cosmology-sensitive terms "
              "will be inconsistent across events.")


def _warn_writer_commit(attrs, path) -> None:
    """Warn when the gwcat that WROTE a file is not the gwcat imported now.

    ``code_identity()`` records the installed gwcat commit, but nothing
    compared it to the commit that produced the input file -- and gwcat is an
    editable install here, so the functions this loader imports change
    whenever that worktree switches branches.  gwcat files that carry a
    ``writer_commit`` (or legacy ``gwcat_commit``) attr get the comparison;
    files that predate the attr are silently exempt.
    """
    file_commit = attrs.get("writer_commit") or attrs.get("gwcat_commit")
    if not file_commit:
        return
    from darksirens.io.settings import code_identity

    installed = str(code_identity().get("gwcat_commit") or "")
    file_commit = str(file_commit)
    if not installed or installed == "unknown":
        return
    # Tolerate short-vs-full SHAs and "-dirty" suffixes.
    a, b = installed.split("-")[0], file_commit.split("-")[0]
    if a and b and not (a.startswith(b) or b.startswith(a)):
        print(f"    [!] {path!r} was written by gwcat commit {file_commit} "
              f"but the imported gwcat is {installed}; prior/draw-density "
              "conventions may not match the file. Pin gwcat for any run "
              "whose numbers will be quoted.")


def load_gw_and_selection_inputs(opts) -> dict:
    """Load GW posterior and selection samples."""
    # Basis negotiation (DS-09): the configured population model decides
    # which fit columns the pair must cover; the file loaders enforce it.
    required_columns = required_fit_columns_for(opts)
    component_basis = "a1" in required_columns

    # Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    if component_basis:
        m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
            opts.gw_path, fit_columns=required_columns
        )
    else:
        m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
            opts.gw_path
        )
    gw_attrs = _read_store_attrs(opts.gw_path)
    spin_pe = _read_spin_block(opts.gw_path) if component_basis else None

    # Load Selection samples (Always required).  When both sides are gwcat
    # files, cross-check the 2.1 pairing contract; the emulator path instead
    # checks the PE file's basis against the emulator's fixed chieff basis.
    selection_attrs = None
    _warn_per_event_cosmology(gw_attrs, opts.gw_path)
    _warn_writer_commit(gw_attrs, opts.gw_path)
    if getattr(opts, "pdet_flow_path", None):
        if component_basis:
            raise RuntimeError(
                "--pdet_flow_path is unsupported in the component spin "
                "basis: the P_det emulator generates chi_eff-basis "
                "pseudo-injections. Use a component-basis gwcat selection "
                "file (--gwselection_path)."
            )
        _require_chieff_pe_for_emulator(gw_attrs, opts.gw_path)
        _require_matching_pdet_cosmology(gw_attrs, opts.gw_path, opts)
    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = resolve_selection_inputs(opts, fit_columns=required_columns)
    spin_sel = (
        _read_spin_block(opts.gwselection_path) if component_basis else None
    )
    if not getattr(opts, "pdet_flow_path", None):
        selection_attrs = _read_store_attrs(opts.gwselection_path)
        _require_matching_contract(
            gw_attrs, selection_attrs, opts.gw_path, opts.gwselection_path
        )
        _warn_pair_cosmology(
            gw_attrs, selection_attrs, opts.gw_path, opts.gwselection_path
        )
        _warn_writer_commit(selection_attrs, opts.gwselection_path)

    return dict(
        gw_attrs=gw_attrs,
        selection_attrs=selection_attrs,
        spin_pe=spin_pe,
        spin_sel=spin_sel,
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

    if _model_requests_component_spin(opts):
        raise RuntimeError(
            "--gw_flows_path is unsupported in the component spin basis: "
            "the flow surrogates are trained in (m1det, q, dL, chieff). "
            "Use stored PE samples from a component-basis gwcat store."
        )

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


def attach_selection_fraction_inputs(opts, data) -> dict:
    """Attach the per-pixel selection fraction ``f_p`` (field-level PR-2).

    ``--per_pixel_completeness <mth_map.h5>`` puts ``C_p(z) = f_p C(z)`` into
    both sides of the missing budget (``f_p = 1 - masked_frac``, degraded to
    the catalog nside by area weighting).  Admissible only where the
    combination is derived:

    * ``c_mode`` in {aggregate, selection} — a per-pixel count-derived ``C``
      already contains the mask loss (multiplying would double-count it);
    * gaussian selection family — the truncated-Schechter disjointness
      argument (F2) has not been re-derived under ``f_p`` (PLAN §7 PR-2);
    * no Q table, no stratified selection — their empty-pixel budgets would
      need ``f_p``-weighted twins (NotImplemented until a rung needs them);
    * K = 1 — the multitracer bundle loader does not thread ``f_p`` yet.
    """
    path = getattr(opts, "per_pixel_completeness", None)
    if not path:
        return data
    c_mode = getattr(opts, "c_mode", None) or "per_pixel"
    if c_mode not in ("aggregate", "selection"):
        raise ValueError(
            f"--per_pixel_completeness requires c_mode aggregate|selection "
            f"(got {c_mode!r}): a per-pixel count-derived C already contains "
            f"the mask loss.")
    if int(getattr(opts, "n_catalogs", 1) or 1) > 1:
        raise NotImplementedError(
            "--per_pixel_completeness is K=1 only for now (the multitracer "
            "bundle loader does not thread f_p).")
    if getattr(opts, "lss_completion", None):
        raise NotImplementedError(
            "--per_pixel_completeness with a Q table needs an f_p-weighted "
            "empty-pixel Q budget that is not implemented; drop one of them.")
    if getattr(opts, "selection_strata_by_catalog", None):
        raise NotImplementedError(
            "--per_pixel_completeness with stratified selection needs "
            "per-stratum f_p empty sums that are not implemented.")
    fit_path = getattr(opts, "selection_fit", None)
    if fit_path:
        from darksirens.redshift.selection import load_selection_fit_json
        family = str(load_selection_fit_json(fit_path).get(
            "family", "gaussian"))
        if family != "gaussian":
            raise NotImplementedError(
                f"--per_pixel_completeness admits only the gaussian selection "
                f"family until the truncated-{family} disjointness argument "
                f"is re-derived under f_p (PLAN §7 PR-2).")

    from darksirens.catalogs.depth_map import load_selection_fraction

    nside = int(data["nside"])
    sfm = load_selection_fraction(path, nside)
    ngals_full = np.asarray(data["ngals"])
    if ngals_full.shape[0] != sfm.f_p.shape[0]:
        raise ValueError(
            f"--per_pixel_completeness: catalog has {ngals_full.shape[0]} "
            f"full-sky rows but the degraded f_p map has {sfm.f_p.shape[0]} "
            f"pixels (nside mismatch?).")
    report = sfm.coverage_report(ngals_full)
    print(f"    [f_p] per-pixel selection fraction from {path}")
    print(f"    [f_p] covered area sum_p f_p Omega_pix = "
          f"{report['area_deg2']:.1f} deg^2; occupied mean f_p = "
          f"{report['f_p_occupied_mean']:.4f} (min "
          f"{report['f_p_occupied_min']:.4f})")
    print(f"    [f_p] coverage: occupied {report['n_occupied']}, "
          f"occupied-partial {report['n_occupied_partial']}, empty-covered "
          f"{report['n_empty_covered']}, off-footprint "
          f"{report['n_off_footprint']}, occupied-but-uncovered "
          f"{report['n_occupied_uncovered']}")
    data["f_p_map"] = sfm.f_p.astype(np.float32)
    data["f_p_coverage_report"] = report
    return data


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
    print("[*] Preparing LSS/Overdensity Field...")
    if opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS and _q_active:
        print(
            "    - Q_LSS completion table is active: it REPLACES the "
            "(1 + b_eff*delta_g) local-overdensity factor, so --use_lss is "
            "redundant here and the overdensity grid is NOT built."
        )
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))
    elif opts.universe_model in GALAXY_AWARE_MODELS and opts.use_LSS:
        print("    - Calculating high-resolution overdensity grid...")
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
