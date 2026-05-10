# data.py

import jax.numpy as jnp
import healpy as hp
import numpy as np

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

from darksirens.em import zgrid, compute_lss_overdensity

GALAXY_AWARE_MODELS = ["dark_sirens", "dark_sirens_complete"]
BRIGHT_SIREN_MODELS = ["bright_sirens"]

def load_all_data(opts):
    """
    Loads survey, GW posterior, and selection data. 
    Handles cases where survey_path might be None (non-dark sirens models).
    """

    # 1. Initialize survey variables as None/defaults
    nside = None
    zgals = dzgals = wgals = None
    zgals_pe = dzgals_pe = wgals_pe = None
    zgals_sel = dzgals_sel = wgals_sel = None
    ngals = ngals_pe = ngals_sel = None
    apix = 0.0
    sigma_kernel = 0.0

    # 2. Load survey data, or build the synthetic one-object catalog used by
    # bright sirens.  The counterpart is not a survey hyperparameter: it is
    # fixed event metadata supplied through the inference CLI.
    if opts.universe_model in BRIGHT_SIREN_MODELS:
        if opts.counterpart is None:
            raise ValueError("bright_sirens requires opts.counterpart=(ra, dec, z).")
        ra_cp, dec_cp, z_cp = opts.counterpart
        nside = int(opts.counterpart_nside)
        npix = hp.nside2npix(nside)
        cp_pix = int(hp.ang2pix(nside, np.pi / 2.0 - dec_cp, ra_cp))

        zgals = np.zeros((npix, 1), dtype=float)
        dzgals = np.ones((npix, 1), dtype=float) * float(opts.counterpart_dz)
        wgals = np.zeros((npix, 1), dtype=float)
        ngals = np.zeros(npix, dtype=np.int32)

        zgals[cp_pix, 0] = z_cp
        wgals[cp_pix, 0] = 1.0
        ngals[cp_pix] = 1

        apix = hp.nside2pixarea(nside)
        sigma_kernel = opts.sigma_kernel
        print(
            "Using bright-siren counterpart catalog: "
            f"ra={ra_cp}, dec={dec_cp}, z={z_cp}, pixel={cp_pix}, nside={nside}"
        )
    elif opts.survey_path is not None:
        nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)
        apix = hp.nside2pixarea(nside)
        sigma_kernel = opts.sigma_kernel
        print("Using a smoothing kernel of sigma: " + str(sigma_kernel))
    else:
        # If no survey, we might still need a default nside for 
        # pixelization logic in other parts of the code
        nside = 1 

    # 3. Load GW posterior samples (Always required)
    # Following the new convention: m1det, m2det, dL, chieff, ra, ...
    m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp = load_gw_samples(
        opts.gw_path
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

    if zgals is not None:
        zgals_pe = zgals[pixels_pe]
        dzgals_pe = dzgals[pixels_pe]
        wgals_pe = wgals[pixels_pe]
        ngals_pe = ngals[pixels_pe]
        
        zgals_sel = zgals[pixels_sel]
        dzgals_sel = dzgals[pixels_sel]
        wgals_sel = wgals[pixels_sel]
        ngals_sel = ngals[pixels_sel]
        
        print("samples" + str(ngals_pe.sum()))
        print("selection" + str(ngals_sel.sum()))

    # 6. Pack into dictionary
    data = dict(
        # GW PE samples
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        p_pe=p_pe,
        pixels_pe=jnp.asarray(pixels_pe),
        zgals_pe=zgals_pe,
        dzgals_pe=dzgals_pe,
        wgals_pe=wgals_pe,
        ngals=ngals_pe,
        ngals_pe=ngals_pe,

        # Selection samples
        m1detsels=m1detsels,
        m2detsels=m2detsels,
        dLsels=dLsels,
        chieffsels=chieffsels,
        p_draw=p_draw,
        pixels_sel=jnp.asarray(pixels_sel),
        zgals_sel=zgals_sel,
        dzgals_sel=dzgals_sel,
        wgals_sel=wgals_sel,
        ngals_sel=ngals_sel,

        # Survey metadata
        nEvents=nEvents,
        Ndraw=Ndraw,
        nsamp=nsamp,
        apix=apix,
        nside=nside,
        n_pix_catalog=int(zgals.shape[0]) if zgals is not None else hp.nside2npix(nside),
        zgals=zgals,
        dzgals=dzgals,
        wgals=wgals,
        ngals_catalog=ngals,
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        sigma_kernel=sigma_kernel
    )

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
        delta_g_pix_z = compute_lss_overdensity(data["zgals"], nside_check)
    else:
        print(f"    - Non-LSS run. Creating memory-efficient dummy (1, {len(zgrid)}) grid.")
        # We use shape (1, nz) to satisfy JAX broadcasting without 93GB allocations
        delta_g_pix_z = jnp.zeros((1, len(zgrid)))

    mem_usage = delta_g_pix_z.nbytes / 1e9
    print(f"    - Overdensity array shape: {delta_g_pix_z.shape} ({mem_usage:.4f} GB)")

    # Append the LSS overdensity field to the returned dictionary
    data["delta_g_pix_z"] = delta_g_pix_z

    return data