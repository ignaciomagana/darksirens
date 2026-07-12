# data.py

import jax.numpy as jnp

from darksirens.catalogs.compact import validate_loaded_survey_shapes
from darksirens.inference import loaders
from darksirens.core.model_kinds import BRIGHT_SIREN_MODELS, GALAXY_AWARE_MODELS


def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data.
    Handles cases where survey_path might be None (non-dark sirens models).
    """
    catalog_inputs = loaders.load_or_build_catalog_inputs(opts)
    if getattr(opts, "gw_flows_path", None):
        gw_inputs = loaders.load_flow_and_selection_inputs(opts)
    else:
        gw_inputs = loaders.load_gw_and_selection_inputs(opts)

    if opts.universe_model in BRIGHT_SIREN_MODELS and catalog_inputs["counterpart_zs"] is not None:
        if len(catalog_inputs["counterpart_zs"]) != int(gw_inputs["nEvents"]):
            raise ValueError(
                "bright_sirens requires one counterpart RA DEC Z triplet per GW event: "
                f"got {len(catalog_inputs['counterpart_zs'])} counterpart(s) "
                f"for {int(gw_inputs['nEvents'])} event(s)."
            )

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
    nside_check = data.get("nside", "N/A")
    print(f"    - Data loaded. Found {nEvents_check} GW events.")
    print(f"    - HEALPix nside detected: {nside_check}")

    # Multitracer: for K >= 2 attach the per-catalog compact bundles and skip the
    # top-level LSS / mark / weak-lensing inputs.  Q tables are loaded PER
    # BUNDLE inside load_multitracer_catalog_bundles; a top-level attach would
    # redundantly load catalog 1's table into an unused slot and print a
    # misleading "LSS completion loaded" line.
    n_catalogs = int(getattr(opts, "n_catalogs", 1))
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
