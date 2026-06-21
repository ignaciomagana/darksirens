# data.py

import jax.numpy as jnp
import healpy as hp
import numpy as np

from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.em.utils import load_survey

from darksirens.em import zgrid, compute_lss_overdensity

GALAXY_AWARE_MODELS = ["dark_sirens", "dark_sirens_complete"]
BRIGHT_SIREN_MODELS = ["bright_sirens"]

#: z-bins for centering per-galaxy marks (subtract the running mean E[m|z]).
_MARK_CENTER_NBINS = 40


def _center_marks(raw_marks: dict, zgals, ngals) -> dict:
    """Return z-centred marks ``m - E[m|z]`` (the global running mean over real
    galaxies), so the sampled ``eta`` measure host preference at fixed redshift.
    Padded slots are set to 0 (masked downstream)."""
    zgals = np.asarray(zgals, dtype=float)
    maxg = zgals.shape[1]
    real = np.arange(maxg)[None, :] < np.asarray(ngals)[:, None]
    z_hi = float(np.asarray(zgrid)[-1])
    edges = np.linspace(0.0, z_hi, _MARK_CENTER_NBINS + 1)
    binc = np.clip(np.searchsorted(edges, zgals, side="right") - 1, 0, _MARK_CENTER_NBINS - 1)
    out = {}
    for name, M in raw_marks.items():
        M = np.asarray(M, dtype=float)
        sums = np.zeros(_MARK_CENTER_NBINS)
        cnts = np.zeros(_MARK_CENTER_NBINS)
        np.add.at(sums, binc[real], M[real])
        np.add.at(cnts, binc[real], 1.0)
        binmean = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1.0), 0.0)
        Mc = np.where(real, M - binmean[binc], 0.0)
        out[name] = Mc
    return out


def _as_counterpart_array(counterpart) -> np.ndarray:
    """Return counterpart metadata as an ``(N, 3)`` float array.

    The public CLI accepts either one ``RA DEC Z`` triplet or a flattened list
    of triplets for multi-event bright-siren analyses.
    """
    arr = np.asarray(counterpart, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError("bright_sirens counterpart metadata must be RA DEC Z triplets.")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("bright_sirens counterpart metadata must have shape (N, 3).")
    return arr


def _compact_catalog_for_pixels(pixels, zgals, dzgals, wgals, ngals, required_pixels=None):
    """Return compact catalog rows and sample→row lookup for pixels.

    ``required_pixels`` are included in the compact catalog even if no sample
    falls in them.  The sample-to-row lookup still covers only ``pixels``.
    """
    pixels = np.asarray(pixels, dtype=np.int32)
    if required_pixels is None:
        unique_pixels, sample_to_unique_idx = np.unique(pixels, return_inverse=True)
    else:
        required_pixels = np.asarray(required_pixels, dtype=np.int32).reshape(-1)
        unique_pixels = np.unique(np.concatenate([pixels, required_pixels]))
        sample_to_unique_idx = np.searchsorted(unique_pixels, pixels)
    unique_pixels = unique_pixels.astype(np.int32, copy=False)
    sample_to_unique_idx = sample_to_unique_idx.astype(np.int32, copy=False)
    return (
        unique_pixels,
        sample_to_unique_idx,
        zgals[unique_pixels],
        dzgals[unique_pixels],
        wgals[unique_pixels],
        ngals[unique_pixels],
    )


def validate_loaded_survey_shapes(data):
    """Validate compact per-pixel galaxy counts returned by ``load_all_data``.

    ``ngals_pe`` and ``ngals_sel`` are compact catalog-row arrays, so they
    should match the corresponding compact pixel arrays whenever a survey (or
    bright-siren counterpart catalog) is available.
    """
    if data.get("zgals_catalog") is None:
        return

    compact_shape_checks = (
        ("ngals_pe", "unique_pixels_pe", "PE"),
        ("ngals_sel", "unique_pixels_sel", "selection"),
    )
    for ngals_key, pixels_key, label in compact_shape_checks:
        ngals_value = data.get(ngals_key)
        pixels_value = data.get(pixels_key)
        if ngals_value is None or pixels_value is None:
            continue
        ngals_n = int(np.asarray(ngals_value).shape[0])
        pixels_n = int(np.asarray(pixels_value).shape[0])
        if ngals_n != pixels_n:
            raise ValueError(
                f"Survey {label} count shape mismatch: {ngals_key}.shape[0] "
                f"({ngals_n}) must equal {pixels_key}.shape[0] ({pixels_n})."
            )

    sample_shape_checks = (
        ("sample_to_unique_pe", "pixels_pe", "PE"),
        ("sample_to_unique_sel", "pixels_sel", "selection"),
    )
    for sample_key, pixels_key, label in sample_shape_checks:
        sample_value = data.get(sample_key)
        pixels_value = data.get(pixels_key)
        if sample_value is None or pixels_value is None:
            continue
        sample_n = int(np.asarray(sample_value).shape[0])
        pixels_n = int(np.asarray(pixels_value).shape[0])
        if sample_n != pixels_n:
            raise ValueError(
                f"Survey {label} sample map shape mismatch: "
                f"{sample_key}.shape[0] ({sample_n}) must equal "
                f"{pixels_key}.shape[0] ({pixels_n})."
            )


def _catalog_memory_diagnostics(zgals, dzgals, wgals, pixels_pe, pixels_sel, ngals_pe, ngals_sel):
    """Summarise memory saved by compact unique-pixel catalog views."""
    unique_pe = np.unique(np.asarray(pixels_pe, dtype=np.int32))
    unique_sel = np.unique(np.asarray(pixels_sel, dtype=np.int32))
    row_bytes = sum(arr.dtype.itemsize * arr.shape[1] for arr in (zgals, dzgals, wgals))
    duplicated_pe = max(0, np.asarray(pixels_pe).size - unique_pe.size) * row_bytes
    duplicated_sel = max(0, np.asarray(pixels_sel).size - unique_sel.size) * row_bytes
    max_gals = 0
    if ngals_pe is not None and ngals_pe.size:
        max_gals = max(max_gals, int(np.max(ngals_pe)))
    if ngals_sel is not None and ngals_sel.size:
        max_gals = max(max_gals, int(np.max(ngals_sel)))
    return {
        "unique_pe_pixels": int(unique_pe.size),
        "unique_sel_pixels": int(unique_sel.size),
        "duplicated_catalog_bytes_avoided": int(duplicated_pe + duplicated_sel),
        "max_galaxies_per_unique_pixel": max_gals,
    }


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

    # 2. Load survey data, or build the synthetic counterpart catalog used by
    # bright sirens.  The counterpart is not a survey hyperparameter: it is
    # fixed event metadata supplied through the inference CLI.
    if opts.universe_model in BRIGHT_SIREN_MODELS:
        if opts.counterpart is None:
            raise ValueError("bright_sirens requires opts.counterpart=RA DEC Z triplets.")
        counterparts = _as_counterpart_array(opts.counterpart)
        ra_cp = counterparts[:, 0]
        dec_cp = counterparts[:, 1]
        z_cp = counterparts[:, 2]
        nside = int(opts.counterpart_nside)
        npix = hp.nside2npix(nside)
        counterpart_pixels = hp.ang2pix(nside, np.pi / 2.0 - dec_cp, ra_cp).astype(np.int32)
        counterpart_pixel = int(counterpart_pixels[0])
        counterpart_zs = z_cp.astype(float, copy=False)
        counterpart_dzs = np.ones_like(counterpart_zs, dtype=float) * float(opts.counterpart_dz)

        counts = np.bincount(counterpart_pixels, minlength=npix).astype(np.int32)
        max_counterparts = max(1, int(counts.max()))
        zgals = np.zeros((npix, max_counterparts), dtype=float)
        dzgals = np.ones((npix, max_counterparts), dtype=float) * float(opts.counterpart_dz)
        wgals = np.zeros((npix, max_counterparts), dtype=float)
        ngals = counts
        offsets = np.zeros(npix, dtype=np.int32)
        for pix_i, z_i in zip(counterpart_pixels, counterpart_zs):
            j = offsets[pix_i]
            zgals[pix_i, j] = z_i
            wgals[pix_i, j] = 1.0
            offsets[pix_i] += 1

        apix = hp.nside2pixarea(nside)
        print(
            "Using bright-siren counterpart catalog: "
            f"{len(counterpart_zs)} counterpart(s), nside={nside}, "
            f"pixels={counterpart_pixels.tolist()}"
        )
    elif opts.survey_path is not None:
        nside, ngals, zgals, dzgals, wgals = load_survey(opts.survey_path)
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
    # Deterministic log Q_LSS only — the ensemble members are a diagnostic and
    # are intentionally NOT threaded into the (jit'd) likelihood.  An explicit
    # --lss_completion path overrides an in-catalog /lss_completion group.
    lss_completion_logq = None
    lss_completion_logq_members = None  # (M, n_pix|n_rows, n_grid) ensemble | None
    lss_completion_indexing = 0  # int enum: 0=auto, 1=compact, 2=global
    lss_path = getattr(opts, "lss_completion", None)
    if lss_path is None and opts.survey_path is not None:
        try:
            import h5py
            with h5py.File(opts.survey_path, "r") as _f:
                if "lss_completion" in _f:
                    lss_path = opts.survey_path
        except Exception:
            lss_path = None
    if lss_path is not None and opts.universe_model in GALAXY_AWARE_MODELS:
        from darksirens.em.lognormal_completion import load_lss_completion_hdf5
        loaded = load_lss_completion_hdf5(lss_path)
        logq = loaded.get("logq_map")
        if logq is None:
            raise ValueError(
                f"LSS completion file '{lss_path}' has no /lss_completion/logq_map "
                "(deterministic table required for inference)."
            )
        logq = np.asarray(logq, dtype=float)
        if logq.shape[-1] != len(zgrid):
            raise ValueError(
                f"LSS completion N_grid={logq.shape[-1]} but the package zgrid has "
                f"size {len(zgrid)}; rebuild the completion on the package grid."
            )
        zg_file = loaded.get("zgrid")
        if zg_file is not None and not np.allclose(
            np.asarray(zg_file, dtype=float), np.asarray(zgrid, dtype=float),
            rtol=1e-5, atol=1e-8,
        ):
            raise ValueError(
                "LSS completion zgrid does not match the package zgrid (no silent interpolation)."
            )
        # Keep the full global table HOST-side (numpy); the likelihood slices it
        # to the union(PE,selection) pixels so only that compact block reaches
        # the device / jit (avoids capturing the full (n_pix, n_grid) table).
        lss_completion_logq = np.asarray(logq)
        lss_completion_indexing = {"compact": 1, "global": 2}.get(
            str(loaded.get("indexing", "compact")), 0
        )
        # Optional Q ENSEMBLE for the fully-Bayesian marginalisation
        # (logL = logsumexp_m logL(Q_m) − log M).  Loaded only when requested, to
        # avoid carrying the (M, n_pix, n_grid) members table otherwise.  Members
        # share the deterministic table's indexing/grid, so the likelihood slices
        # them to the union pixels the same way.
        if getattr(opts, "lss_marginalize", False):
            logq_m = loaded.get("logq_members")
            if logq_m is None:
                raise ValueError(
                    f"--lss_marginalize requires an LSS-completion ENSEMBLE, but "
                    f"'{lss_path}' has no /lss_completion/logq_members. Rebuild Q with "
                    "darksirens_build_lognormal_completion --n-members M (M > 0)."
                )
            logq_m = np.asarray(logq_m, dtype=float)
            if logq_m.shape[-1] != len(zgrid):
                raise ValueError(
                    f"LSS completion members N_grid={logq_m.shape[-1]} but the package "
                    f"zgrid has size {len(zgrid)}; rebuild on the package grid."
                )
            lss_completion_logq_members = logq_m
            print(
                f"    - LSS completion ENSEMBLE loaded: logq_members "
                f"{tuple(logq_m.shape)} (M={logq_m.shape[0]}) for fully-Bayesian "
                "marginalisation over the missing-galaxy field"
            )
        print(
            f"    - LSS completion loaded from {lss_path}: logq_map {tuple(logq.shape)}, "
            f"indexing={loaded.get('indexing')}"
        )
        _diag = loaded.get("diagnostics") or {}
        _fid = {k: _diag[k] for k in (
            "fiducial_H0", "fiducial_Om0", "fiducial_n0", "fiducial_delta",
            "bias_b_miss", "lss_corr_length_mpc", "lss_sigma",
        ) if k in _diag}
        if _fid:
            print(f"    - Q_LSS build fiducials: {_fid}")
        print(
            "    [!] Q_LSS is FIXED at its build-time fiducials (cosmology, n0, "
            "delta, bias); the inference will vary some of these. Q is a "
            "radial completion field on the SAME zgrid (validated), interpreted "
            "as a dimensionless density-ratio. Rebuild Q if your fiducials differ "
            "substantially."
        )
    data["lss_completion_logq"] = lss_completion_logq
    data["lss_completion_logq_members"] = lss_completion_logq_members
    data["lss_completion_indexing"] = lss_completion_indexing

    # --------------------------------------------------------
    # Per-galaxy marks for the marked-host model (optional)
    # --------------------------------------------------------
    for _ds in ("mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color"):
        data[_ds] = None
    if opts.survey_path is not None and opts.universe_model in GALAXY_AWARE_MODELS:
        from darksirens.em.utils import load_survey_marks
        raw_marks = load_survey_marks(opts.survey_path)
        if raw_marks:
            centered = _center_marks(
                raw_marks, data["zgals"], data.get("ngals_catalog")
            )
            for _name, _arr in centered.items():
                data[_name] = jnp.asarray(_arr)
            print(f"    - Loaded galaxy marks {sorted(raw_marks)} (z-centred)")

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