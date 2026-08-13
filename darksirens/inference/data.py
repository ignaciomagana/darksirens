# data.py

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.catalogs.compact import validate_loaded_survey_shapes
from darksirens.inference import loaders
from darksirens.core.model_kinds import BRIGHT_SIREN_MODELS

#: Store attrs worth carrying into the run record: they exist in the input
#: files and previously vanished from every artifact, so an archived run
#: could not say what detection cut, observing time, or campaign structure
#: produced its numbers.
_PROVENANCE_ATTRS = (
    "format_version",
    "spin_basis",
    "parameter_space",
    "contract_hash",
    "far_threshold",
    "far_max",
    "T_obs_yr",
    "campaign_ndraws",
    "n_campaigns",
    "ndraw",
    "nobs",
    "source_class_filter",
    "detection_statistic",
    "detection_threshold",
    "allow_missing_far",
    "chi_eff_amax",
    "pe_cosmology_H0",
    "pe_cosmology_Om0",
    "cosmology_H0",
    "cosmology_Om0",
    "writer_commit",
    "gwcat_commit",
    "event_names",
)


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _provenance_block(attrs):
    if not attrs:
        return None
    return {k: _jsonable(attrs[k]) for k in _PROVENANCE_ATTRS if k in attrs}


def _attr_event_names(attrs):
    names = (attrs or {}).get("event_names")
    if names is None:
        return None
    return [v.decode() if isinstance(v, bytes) else str(v)
            for v in np.atleast_1d(names)]


def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data.
    Handles cases where survey_path might be None (non-dark sirens models).
    """
    n_catalogs = int(getattr(opts, "n_catalogs", 1))

    if n_catalogs >= 2:
        # Multitracer mixture (K >= 2): every catalog is loaded exactly ONCE,
        # per bundle, by load_multitracer_catalog_bundles further below.
        # opts.survey_path is set to survey_paths[0] for a K>=2 run (see
        # cli/inference.py), so calling loaders.load_or_build_catalog_inputs(opts)
        # here would load catalog 1's full-sky arrays a SECOND time into a
        # top-level slot the mixture never reads (and one that
        # maybe_drop_full_catalog -- a K==1-only step, see below -- never gets a
        # chance to drop, since the old code returned before reaching it).
        # Stub catalog_inputs the same way the no-survey case does (nside=1,
        # no galaxy arrays) so the shared sky-vector / dict-building code below
        # is identical for K=1 and K>=2, without ever touching a survey file.
        catalog_inputs = dict(
            nside=1,
            npix=hp.nside2npix(1),
            zgals=None, dzgals=None, wgals=None, ngals=None,
            apix=0.0, z_depth=None,
            counterpart_pixel=None, counterpart_pixels=None,
            counterpart_zs=None, counterpart_dzs=None,
            bright_siren_sky_marginalized=False,
        )
    else:
        catalog_inputs = loaders.load_or_build_catalog_inputs(opts)

    if getattr(opts, "gw_flows_path", None):
        gw_inputs = loaders.load_flow_and_selection_inputs(opts)
    else:
        gw_inputs = loaders.load_gw_and_selection_inputs(opts)

    gw_event_names = _attr_event_names(gw_inputs.get("gw_attrs"))

    # Persist store provenance in settings.json (opts attributes are
    # serialised there), mirroring how the flow path records
    # flow_event_names: the run record can then say which detection cut,
    # observing time, campaign structure, and event identity produced it.
    opts.gw_store_provenance = _provenance_block(gw_inputs.get("gw_attrs"))
    opts.selection_store_provenance = _provenance_block(
        gw_inputs.get("selection_attrs")
    )
    if gw_event_names is not None:
        opts.gw_event_names = gw_event_names

    if opts.universe_model in BRIGHT_SIREN_MODELS and catalog_inputs["counterpart_zs"] is not None:
        if len(catalog_inputs["counterpart_zs"]) != int(gw_inputs["nEvents"]):
            raise ValueError(
                "bright_sirens requires one counterpart RA DEC Z triplet per GW event: "
                f"got {len(catalog_inputs['counterpart_zs'])} counterpart(s) "
                f"for {int(gw_inputs['nEvents'])} event(s)."
            )
        # Counterpart triplets are aligned to events BY POSITION -- nothing
        # else identifies them -- so a store rebuilt with a different
        # whitelist silently re-pairs every counterpart.  Print the identity
        # this run assumed, so the pairing is at least verifiable.
        if gw_event_names is not None:
            print("    Bright-siren counterpart <-> event pairing (by position):")
            for k, name in enumerate(gw_event_names):
                z_k = float(np.asarray(catalog_inputs["counterpart_zs"])[k])
                print(f"      counterpart[{k}] (z={z_k:.4f}) -> {name}")

    sky_inputs = loaders.compute_sky_pixels_and_vectors(opts, catalog_inputs, gw_inputs)

    def _maybe_jnp(value):
        return jnp.asarray(value) if value is not None else None

    # Pack into dictionary
    data = dict(
        # GW PE samples (all None on the flow-surrogate path)
        m1det=gw_inputs["m1det"],
        m2det=gw_inputs["m2det"],
        dL=gw_inputs["dL"],
        chieff=gw_inputs["chieff"],
        p_pe=gw_inputs["p_pe"],
        flow_ensemble=gw_inputs.get("flow_ensemble"),
        flow_event_names=gw_inputs.get("flow_event_names"),
        gw_attrs=gw_inputs.get("gw_attrs"),
        selection_attrs=gw_inputs.get("selection_attrs"),
        gw_event_names=gw_event_names,
        pixels_pe=_maybe_jnp(sky_inputs["pixels_pe"]),
        nx_pe=_maybe_jnp(sky_inputs["nx_pe"]),
        ny_pe=_maybe_jnp(sky_inputs["ny_pe"]),
        nz_pe=_maybe_jnp(sky_inputs["nz_pe"]),
        zgals_pe=sky_inputs["zgals_pe"],
        dzgals_pe=sky_inputs["dzgals_pe"],
        wgals_pe=sky_inputs["wgals_pe"],
        ngals_pe=sky_inputs["ngals_pe"],
        unique_pixels_pe=sky_inputs["unique_pixels_pe"],
        sample_to_unique_pe=sky_inputs["sample_to_unique_pe"],

        # Selection samples
        m1detsels=gw_inputs["m1detsels"],
        m2detsels=gw_inputs["m2detsels"],
        dLsels=gw_inputs["dLsels"],
        chieffsels=gw_inputs["chieffsels"],
        p_draw=gw_inputs["p_draw"],
        pixels_sel=jnp.asarray(sky_inputs["pixels_sel"]),
        nx_sel=jnp.asarray(sky_inputs["nx_sel"]),
        ny_sel=jnp.asarray(sky_inputs["ny_sel"]),
        nz_sel=jnp.asarray(sky_inputs["nz_sel"]),
        zgals_sel=sky_inputs["zgals_sel"],
        dzgals_sel=sky_inputs["dzgals_sel"],
        wgals_sel=sky_inputs["wgals_sel"],
        ngals_sel=sky_inputs["ngals_sel"],
        unique_pixels_sel=sky_inputs["unique_pixels_sel"],
        sample_to_unique_sel=sky_inputs["sample_to_unique_sel"],

        # Survey metadata and full catalog arrays.  Full pixel indexing is kept
        # only for operations that need global HEALPix rows, such as LSS
        # overdensity construction and startup cache generation.
        nEvents=gw_inputs["nEvents"],
        Ndraw=gw_inputs["Ndraw"],
        nsamp=gw_inputs["nsamp"],
        apix=catalog_inputs["apix"],
        z_depth=catalog_inputs.get("z_depth"),
        nside=catalog_inputs["nside"],
        n_pix_catalog=catalog_inputs["npix"],
        zgals=catalog_inputs["zgals"],
        dzgals=catalog_inputs["dzgals"],
        wgals=catalog_inputs["wgals"],
        ngals_catalog=catalog_inputs["ngals"],
        zgals_catalog=catalog_inputs["zgals"],
        dzgals_catalog=catalog_inputs["dzgals"],
        wgals_catalog=catalog_inputs["wgals"],
        catalog_memory=sky_inputs["catalog_memory"],
        counterpart_pixel=catalog_inputs["counterpart_pixel"],
        counterpart_pixels=catalog_inputs["counterpart_pixels"],
        counterpart_zs=catalog_inputs["counterpart_zs"],
        counterpart_dzs=catalog_inputs["counterpart_dzs"],
        bright_siren_sky_marginalized=catalog_inputs["bright_siren_sky_marginalized"]
    )

    validate_loaded_survey_shapes(data)

    nEvents_check = data.get("nEvents", "Unknown")
    print(f"    - Data loaded. Found {nEvents_check} GW events.")
    if n_catalogs >= 2:
        print(f"    - K={n_catalogs} multitracer mixture: HEALPix nside is per-catalog "
              "(see catalog bundle loads below).")
    else:
        nside_check = data.get("nside", "N/A")
        print(f"    - HEALPix nside detected: {nside_check}")

    # Multitracer: for K >= 2 attach the per-catalog compact bundles and skip the
    # top-level LSS / mark / weak-lensing inputs.  Q tables are loaded PER
    # BUNDLE inside load_multitracer_catalog_bundles; a top-level attach would
    # redundantly load catalog 1's table into an unused slot and print a
    # misleading "LSS completion loaded" line.
    if n_catalogs >= 2:
        data["catalogs"] = loaders.load_multitracer_catalog_bundles(opts, gw_inputs)
        for _ds in ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color"):
            data.setdefault(_ds, None)
        data.setdefault("wl_params", None)
        return data

    data = loaders.attach_lss_inputs(opts, data)
    data = loaders.attach_mark_inputs(opts, data)
    data = loaders.attach_wl_inputs(opts, data)
    data = loaders.maybe_drop_full_catalog(opts, data)

    return data
