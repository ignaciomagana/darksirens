# data.py

import jax.numpy as jnp
import healpy as hp
import numpy as np

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

from darksirens.em import zgrid, compute_lss_overdensity

GALAXY_AWARE_MODELS = ["dark_sirens", "dark_sirens_complete"]
BRIGHT_SIREN_MODELS = ["bright_sirens"]

from darksirens.catalogs.compact import (
    _catalog_memory_diagnostics,
    _compact_catalog_for_pixels,
    validate_loaded_survey_shapes,
)
from darksirens.catalogs.counterparts import (
    as_counterpart_array as _as_counterpart_array,
    build_counterpart_catalog,
)
from darksirens.catalogs.lss import maybe_load_lss_completion
from darksirens.catalogs.marks import (
    _center_marks,
    load_and_center_survey_marks,
)

def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data. 
    Handles cases where survey_path might be None (non-dark sirens models).
    """

    # 1. Initialize survey variables as None/defaults
    nside = None
    npix = None
    zgals = dzgals = wgals = None
    zgals_pe = dzgals_pe = wgals_pe = None
    zgals_sel = dzgals_sel = wgals_sel = None
    unique_pixels_pe = unique_pixels_sel = None
    sample_to_unique_pe = sample_to_unique_sel = None
    ngals = ngals_pe = ngals_sel = None
    catalog_memory = None
    apix = 0.0
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

    # 2. Load survey data, or build the synthetic counterpart catalog used by
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
        nside, ngals, zgals, dzgals, wgals = load_survey(
            opts.survey_path, to_device=not drop_full_catalog
        )
        npix = hp.nside2npix(nside)
        apix = hp.nside2pixarea(nside)
    else:
        # If no survey, we might still need a default nside for 
        # pixelization logic in other parts of the code
        nside = 1
        npix = hp.nside2npix(nside)

    # 3. Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
        opts.gw_path
    )
    if opts.universe_model in BRIGHT_SIREN_MODELS and counterpart_zs is not None:
        if len(counterpart_zs) != int(nEvents):
            raise ValueError(
                "bright_sirens requires one counterpart RA DEC Z triplet per GW event: "
                f"got {len(counterpart_zs)} counterpart(s) for {int(nEvents)} event(s)."
            )

    # 4. Load Selection samples (Always required)
    (
        m1detsels, m2detsels, dLsels, chieffsels,
        rasels, decsels, p_draw, Ndraw,
    ) = load_selection_samples(opts.gwselection_path)

    # 5. Pixel indexing and Galaxy lookups
    # Only perform these if a survey was actually loaded
    pixels_pe = hp.ang2pix(nside, jnp.pi/2 - dec, ra)
    pixels_sel = hp.ang2pix(nside, jnp.pi/2 - decsels, rasels)

    # Sky-direction unit vectors n̂ = (cos δ cos α, cos δ sin α, sin δ), retained
    # per sample for the angular/sky model.  Unlike ``pixels`` (whose resolution
    # collapses to nside=1 with no survey), these give the sky model full
    # angular resolution in every universe model, including GW-only runs.
    nx_pe, ny_pe, nz_pe = (
        jnp.cos(dec) * jnp.cos(ra),
        jnp.cos(dec) * jnp.sin(ra),
        jnp.sin(dec),
    )
    nx_sel, ny_sel, nz_sel = (
        jnp.cos(decsels) * jnp.cos(rasels),
        jnp.cos(decsels) * jnp.sin(rasels),
        jnp.sin(decsels),
    )

    if zgals is not None:
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
        print("samples" + str(ngals_pe[sample_to_unique_pe].sum()))
        print("selection" + str(ngals_sel[sample_to_unique_sel].sum()))
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

    # 6. Pack into dictionary
    data = dict(
        # GW PE samples
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        p_pe=p_pe,
        pixels_pe=jnp.asarray(pixels_pe),
        nx_pe=jnp.asarray(nx_pe),
        ny_pe=jnp.asarray(ny_pe),
        nz_pe=jnp.asarray(nz_pe),
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,
        ngals_pe=ngals_pe,
        unique_pixels_pe=unique_pixels_pe,
        sample_to_unique_pe=sample_to_unique_pe,

        # Selection samples
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        p_draw=p_draw,
        pixels_sel=jnp.asarray(pixels_sel),
        nx_sel=jnp.asarray(nx_sel),
        ny_sel=jnp.asarray(ny_sel),
        nz_sel=jnp.asarray(nz_sel),
        zgals_sel=zgals_sel,
        dzgals_sel=dzgals_sel,
        wgals_sel=wgals_sel,
        ngals_sel=ngals_sel,
        unique_pixels_sel=unique_pixels_sel,
        sample_to_unique_sel=sample_to_unique_sel,

        # Survey metadata and full catalog arrays.  Full pixel indexing is kept
        # only for operations that need global HEALPix rows, such as LSS
        # overdensity construction and startup cache generation.
        nEvents=nEvents,
        Ndraw=Ndraw,
        nsamp=nsamp,
        apix=apix,
        nside=nside,
        n_pix_catalog=npix,
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
        ngals_catalog=ngals,
        zgals_catalog=zgals,
        dzgals_catalog=dzgals,
        wgals_catalog=wgals,
        catalog_memory=catalog_memory,
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized
    )

    validate_loaded_survey_shapes(data)

    nEvents_check = data.get("nEvents", "Unknown")
    nside_check = data.get("nside", "N/A")
    print(f"    - Data loaded. Found {nEvents_check} GW events.")
    print(f"    - HEALPix nside detected: {nside_check}")

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