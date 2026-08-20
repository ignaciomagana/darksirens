"""One realization of the nside-16 closure mock (PLAN §6.1's three extensions).

`experiments/completeness_viz/generate_clustered_mock.py` writes a clustered
galaxy catalog but no GW side; `scripts/mock_dark_sirens/generate_mock_data.py`
writes a full GW side but an unclustered, unmasked catalog.  PLAN §6.1 asks for
three extensions to the testbed, and this module is exactly those three, with
the GW machinery of the shipped generator reused verbatim rather than
retranscribed:

  (i)   **a completeness map** ``f_p`` -- the real DESI depth map degraded to
        nside 16 (:func:`world16.footprint`), applied as a per-galaxy areal
        mask, and written back out as an ``mth_map`` the inference reads
        through the SHIPPED ``catalogs.depth_map.load_selection_fraction``;
  (ii)  **GW hosts placed by the TRUE field including missing hosts** -- the
        complete catalog is drawn from ``exp(b_gal f)`` over the whole sky and
        the whole universe, and ``_draw_events_until_detected`` picks hosts
        uniformly FROM THAT catalog, so a host in a masked pixel, a host too
        faint to be catalogued and a host beyond the survey depth are all
        reachable and all carry the field's clustering;
  (iii) **a matched injection set** -- ``_selection_injections`` with the SAME
        ``grids``, ``pop`` and ``MeasurementConfig`` object the detection rule
        and the posteriors used, per `mock-data-dag`.

Four modelling decisions, each of which the inference's own model dictates:

**The universe extends past the survey depth; the catalog does not.**
``completion.py:1588`` states the contract: "``z_depth`` encodes prior
knowledge that the EM survey does not catalog galaxies past its own depth:
beyond ``z_depth`` completeness is 0, ``dN_miss = dN_exp``".  So the mock
draws galaxies (and therefore hosts) out to ``Z_UNIVERSE = 0.6`` and
catalogues only those with TRUE ``z <= Z_DEPTH = 0.30``.  A mock truncated at
the depth would make the model's above-depth missing budget spurious; a mock
with no truncation would make it absent.  The field itself is defined only
below the depth (``z_node_hi = 0.30``) and the mock sets ``Q = 1`` above it --
which is bit-for-bit what the seam does (``latent_q.load_latent_plan`` zeroes
``phi_z`` above the depth).

**The photo-z width is 0.023, constant.**  Not the generator's default
``0.0005 + 0.0015 (1+z)``: ``cli/build_latent_field.py:156`` HARD-CODES
``sigma_z = 0.023`` in the shell response ``W``, so a mock with a 0.002 width
would be fitting a ``W`` that is wrong by 10x and Tier B would be measuring
that, not the field.  Matching the two is the difference between a closure
test and a misspecification test; the misspecification tests are Tier D.

**The detection horizon is tuned to sit inside the catalog.**  ``snr_ref``
is lowered so that the ``rho = 8`` horizon is ~700 Mpc: with the shipped
default the horizon is ~2.9 Gpc, 60 events would place almost none inside a
``z <= 0.3`` catalog, and every arm's ``H0`` posterior would be prior-wide --
"``H0_true`` inside the 90% CI" would then pass for all three arms while
measuring nothing.  The universe's ``z`` ceiling is set so the horizon is
still inside it at ``H0 = 140`` (where ``dL(z)`` is roughly halved), which is
what keeps the injection set a valid cover of the population support.

**The selection fit and ``n0`` are set at their injected truth.**  They are
survey-level calibrations, not per-event constants, so this does not violate
`mock-data-dag`'s rule; it is a stated idealization that isolates the field,
and it is recorded in the realization's ``truth.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import world16 as W16

# --------------------------------------------------------------- constants

Z_UNIVERSE = 0.60          # galaxies (and hosts) exist out to here
Z_DEPTH = W16.Z_DEPTH      # 0.30 -- the catalog's declared depth
#: The catalog's redshift-error floor -- the scatter applied to every galaxy AND
#: the ``dz`` the catalog stores, so it is also the width of the analysis kernel.
#:
#: **0.003, SPECTROSCOPIC, changed from W16.SIGMA_Z = 0.023 (2026-08-19).**  The
#: photometric value was measured to be the sole remaining cause of the closure
#: tiers' residual bias and overconfidence: at a median host ``z`` of 0.11 a
#: 0.023 scatter is a 21% FRACTIONAL kernel width, the kernel is not truncated at
#: ``z >= 0`` (129 galaxies per realization had ``z_obs < 0``, and a Gaussian
#: centred below zero can only put mass at higher ``z``), and at that width it
#: interacts with the rising volumetric prior at exactly the few-percent level the
#: bias sat at.  Setting it to zero took median ``u`` from 0.277 to 0.449, the
#: bias from +3.00 to +0.42 km/s, the overconfidence from 1.714 to 1.224 and the
#: 90% coverage from 0.62 to 0.88 (``gate_specz.py``).
#:
#: **1e-4, and NOT 0.003 -- a first attempt used 0.003 and it was wrong.**  The
#: analysis kernel is ``sig_eff = sqrt(dz^2 + sigma_kde^2)``
#: (``redshift/catalog.py:540``) with the production ``sigma_kde = 0.003``, so
#: setting the catalog's own scatter to 0.003 makes the kernel
#: ``sqrt(2) x 0.003``: **41% WIDER than the truth**, a brand-new mismatch where
#: the photometric 0.023 had made ``sigma_kde`` negligible (0.9%).  Measured: at
#: n = 17 that configuration left Tier C at overconfidence 1.603 and 90% coverage
#: 0.53 -- no better than photo-`z`.
#:
#: What production actually looks like is a catalog redshift that is essentially
#: EXACT (DESI spectroscopy, ~30 km/s) with ``sigma_kde = 0.003`` as the
#: analysis's own deliberate smoothing.  1e-4 reproduces that: it sits at
#: ``catalog.SIGMA_EFF_FLOOR``, so ``sig_eff -> 0.003002`` and the kernel is
#: `sigma_kde` alone, exactly as on the production line.
#:
#: ``W16.SIGMA_Z`` is UNCHANGED at 0.023: it is the width
#: ``build_latent_field``'s frozen W was constructed with, and the latent basis
#: must not move underneath the anchors.
SIGMA_Z_CAT = 1.0e-4
N0_TRUE = 2.0e-4           # Mpc^-3, comoving, delta = 0
SNR_REF = 5.0
SNR_THRESHOLD = 8.0
NOBS = 60
NSAMP = 512
NDRAW = 4_000_000
TARGET_DET = 60_000


def _dark():
    path = W16.REPO_DIR / "scripts" / "mock_dark_sirens" / "generate_mock_data.py"
    spec = importlib.util.spec_from_file_location("pr6a_mock_gen", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sample_sky_in_pixels(rng, counts, nside):
    """Uniform sky positions grouped by HEALPix RING pixel (verbatim from
    ``experiments/completeness_viz/generate_clustered_mock.py``)."""
    import healpy as hp

    counts = np.asarray(counts, dtype=int)
    npix = counts.size
    total = int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)])
    ra_out = np.empty(total)
    dec_out = np.empty(total)
    filled = np.zeros(npix, dtype=int)
    n_done = 0
    while n_done < total:
        batch = max(4 * (total - n_done), 100_000)
        ra = rng.uniform(0.0, 2.0 * np.pi, batch)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, batch))
        pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
        order = np.argsort(pix, kind="stable")
        pix_s, ra_s, dec_s = pix[order], ra[order], dec[order]
        uniq, first, navail = np.unique(pix_s, return_index=True,
                                        return_counts=True)
        for u, f0, k in zip(uniq, first, navail):
            need = counts[u] - filled[u]
            if need <= 0:
                continue
            t = min(int(need), int(k))
            dst = starts[u] + filled[u]
            ra_out[dst:dst + t] = ra_s[f0:f0 + t]
            dec_out[dst:dst + t] = dec_s[f0:f0 + t]
            filled[u] += t
            n_done += t
    return ra_out, dec_out


def clustered_catalog(world, xi_true, rng, grids, *, n0=N0_TRUE,
                      z_universe=Z_UNIVERSE, n_zbins=240, field_scale=1.0):
    """The COMPLETE catalog: ra, dec, z, drawn from ``n0 dV_c exp(b f)``.

    ``f`` is the truth field below ``Z_DEPTH`` and identically zero above it
    (see the module docstring), with the mean-one lognormal shift applied so
    ``n0`` is the true comoving density.
    """
    import healpy as hp

    npix = hp.nside2npix(world.nside)
    zg, dvc = grids["z"], grids["dvc_dz"]
    Vcum = np.concatenate([[0.0], np.cumsum(
        0.5 * (dvc[1:] + dvc[:-1]) * np.diff(zg))])
    z_edges = np.linspace(0.0, z_universe, n_zbins + 1)
    z_cen = 0.5 * (z_edges[:-1] + z_edges[1:])
    dV_bin = np.diff(np.interp(z_edges, zg, Vcum))            # (nbin,)

    # logQ on the bin centers: interpolate the fine-grid field in z, zero above
    # the depth (the seam's own above-depth convention).
    # The mean-one shift must scale with the field it normalizes: at
    # ``field_scale = 0`` the density has to be exactly ``n0``, and a shift
    # frozen at ``-b^2 v/2 = -0.5`` would instead make it ``0.607 n0`` -- a 65%
    # error in the very calibration the likelihood is handed, which presents as
    # a runaway H0 rather than as anything recognizable.
    fs = float(field_scale)
    f_fine = W16.field_fine(world, xi_true, all_sky=True)      # (npix, N_fine)
    shift = (fs ** 2) * W16.mean_one_shift(world, all_sky=True)
    logq_fine = fs * world.b_gal * f_fine + shift
    logq = np.zeros((npix, n_zbins))
    below = z_cen <= Z_DEPTH
    for p in range(npix):
        logq[p, below] = np.interp(z_cen[below], world.z_fine, logq_fine[p])
    Q = np.exp(logq)
    mu = n0 * (dV_bin[None, :] / npix) * Q
    N = rng.poisson(mu)
    total = int(N.sum())
    pix_flat = np.repeat(np.arange(npix), N.sum(axis=1))
    bin_flat = np.concatenate([np.repeat(np.arange(n_zbins), N[p])
                               for p in range(npix)])
    cdf = Vcum / Vcum[-1]
    cdf_e = np.interp(z_edges, zg, cdf)
    u = rng.uniform(cdf_e[bin_flat], cdf_e[bin_flat + 1])
    z = np.interp(u, cdf, zg)
    ra, dec = _sample_sky_in_pixels(rng, N.sum(axis=1), world.nside)
    # The per-galaxy log-overdensity, carried out so a SURVEY-side stress can
    # depend on the local density (Tier D-ii's fibre-assignment proxy) without
    # touching the universe -- an incompleteness must not move the hosts.
    logq_gal = logq[pix_flat, bin_flat]
    return dict(ra=ra, dec=dec, z=z), pix_flat, total, logq_gal


def build(seed, outdir, *, world=None, nobs=NOBS, nsamp=NSAMP, ndraw=NDRAW,
          n0=N0_TRUE, f_p_survey=None, extra_selection=None,
          lognormal_tail=0.0, field_scale=1.0, z_universe=Z_UNIVERSE,
          target_det=TARGET_DET, verbose=True,
          reuse_injections=None, event_seed=None):
    """Write one realization's data products under ``outdir``.

    ``f_p_survey`` is the map the SURVEY applies (Tier D-i perturbs it while
    the inference keeps reading the unperturbed one).  ``extra_selection(z,
    logq)`` is Tier D-ii's unmodelled ``z``- and density-dependent survey
    incompleteness: a multiplicative acceptance applied at the SURVEY step, so
    the galaxy universe -- and therefore the GW hosts -- is untouched, which is
    what makes it an incompleteness rather than a different universe.
    ``reuse_injections`` points at an already-written selection file: the
    injection set is a property of the detector and the DAG, not of the field
    realization, so Tier C reuses one set across realizations (stated in
    CLOSURE.md) instead of redrawing 4e6 proposals fifty times.
    """
    import h5py
    import healpy as hp

    dark = _dark()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    world = world or W16.build_world()
    rng = np.random.default_rng(seed)

    pop = dark.PopulationConfig()
    survey_cfg = dark.SurveyConfig(
        footprint_dec_min_deg=-90.0, footprint_dec_max_deg=90.0,
        z_hard_max=Z_DEPTH, magnitude_limit=W16.M_LIM,
        z50=1.0e6, width=1.0,          # the logistic completeness made INERT:
        absolute_mag_mean=W16.M0_PHYS,  # the only z-selection is the magnitude
        absolute_mag_sigma=W16.SIGMA_M,  # limit, which c_sel_gaussian models
        redshift_error_floor=SIGMA_Z_CAT, redshift_error_slope=0.0,
    )
    meas = dark.MeasurementConfig(
        snr_ref=SNR_REF, snr_threshold=SNR_THRESHOLD,
        sigma_rho=1.0, a_mc=0.08, a_q=0.60, a_chi=0.20,
        sky_uncertainty_deg=0.5)
    cosmo = dark._build_cosmology(W16.H0_TRUE, W16.OM0, -1.0, 0.0)
    grids = dark._cosmology_grids(cosmo, float(z_universe))

    # ------------------------------------------------------- galaxy universe
    xi_true = W16.draw_xi_true(world, rng, lognormal_tail=lognormal_tail)
    # ``field_scale = 0`` is the HOMOGENEOUS control: same galaxies, same
    # events, same injections, no clustering.  It is the only way to separate
    # "the mock and the likelihood disagree" from "the clustering is doing
    # something", and it consumes the same rng draws so the two are paired.

    complete, pix_flat, total, logq_gal = clustered_catalog(
        world, xi_true, rng, grids, n0=n0, field_scale=field_scale,
        z_universe=float(z_universe))
    dl_pc = dark._interp_dl(complete["z"], grids) * 1.0e6
    complete["abs_mag"] = rng.normal(survey_cfg.absolute_mag_mean,
                                     survey_cfg.absolute_mag_sigma, total)
    complete["app_mag"] = complete["abs_mag"] + 5.0 * np.log10(
        np.maximum(dl_pc, 10.0) / 10.0)
    zerr = dark._catalog_zerr(complete["z"], survey_cfg)
    complete["z_obs"] = complete["z"] + zerr * rng.standard_normal(total)
    if verbose:
        print(f"[mock] complete catalog {total} galaxies "
              f"(n0={n0:g} Mpc^-3, z<={z_universe})", flush=True)

    # ------------------------------------------------------------ the survey
    # areal mask (per-pixel f_p, Bernoulli per galaxy -- same expected pi_pg as
    # a true areal cut) x magnitude limit x the hard depth cut.
    f_map = np.zeros(hp.nside2npix(world.nside))
    f_map[world.pix] = world.f_p if f_p_survey is None else np.asarray(f_p_survey)
    p_keep = f_map[pix_flat]
    if extra_selection is not None:
        p_keep = p_keep * np.clip(
            extra_selection(complete["z"], logq_gal), 0.0, 1.0)
    observed = ((complete["z"] <= Z_DEPTH)
                & (complete["app_mag"] <= survey_cfg.magnitude_limit)
                & (rng.uniform(size=total) < p_keep))
    n_obs = int(observed.sum())
    z_obs_s = complete["z_obs"][observed]
    zerr_s = dark._catalog_zerr(z_obs_s, survey_cfg)
    pixelated = dark._pixelate_catalog(
        complete["ra"][observed], complete["dec"][observed], z_obs_s, zerr_s,
        np.ones(n_obs), world.nside,
        marks={"gal_app_mag": complete["app_mag"][observed]})
    if verbose:
        print(f"[mock] survey {n_obs} galaxies ({n_obs / total:.3f} of the "
              f"universe)", flush=True)

    # ---------------------------------------------------------------- the GW
    # ``event_seed`` splits the realization in two.  Everything above this line
    # -- the field, the complete catalog, the mask, the survey -- is a function
    # of ``seed`` alone; everything below it (which 60 hosts are detected, and
    # their PE noise) becomes a function of ``event_seed`` when one is given.
    # That is what turns Tier C's variance into a DECOMPOSITION: holding
    # ``seed`` fixed and varying ``event_seed`` gives the event-and-PE half of
    # the realization-to-realization spread, and comparing it against the full
    # spread says whether the overconfidence is per-event likelihood
    # sharpness or a catalog-level common mode.  With ``event_seed=None`` the
    # stream is untouched and every existing seed reproduces bit-for-bit.
    if event_seed is not None:
        rng = np.random.default_rng(int(event_seed))
    truth, _ = dark._draw_events_until_detected(
        rng, nobs, complete, grids, pop, SNR_THRESHOLD, meas=meas)
    post, pe_obs = dark._posterior_samples(rng, truth, nsamp, meas)
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)
    if verbose:
        print(f"[mock] {nobs} events: z median {np.median(truth['z']):.4f}, "
              f"max {truth['z'].max():.4f}, frac below depth "
              f"{(truth['z'] <= Z_DEPTH).mean():.3f}; PE dL max "
              f"{post['dL'].max():.1f} Mpc", flush=True)

    sel_path = out / "gw_selection.h5"
    if reuse_injections is not None:
        import shutil
        shutil.copyfile(reuse_injections, sel_path)
        with h5py.File(sel_path) as f:
            n_det = int(f["dL"].shape[0])
        sel_neff_fid = None
    else:
        sel = dark._selection_injections(
            rng, ndraw, grids, pop, SNR_THRESHOLD, min(ndraw, 500_000),
            target_detections=target_det, verbose=False,
            proposal="population+uniform", meas=meas)
        inv = 1.0 / np.asarray(sel["pdraw"])
        neff = float(inv.sum() ** 2 / np.square(inv).sum())
        sel_neff_fid = dark._selection_neff_at_fiducial(sel, grids, pop)
        n_det = int(sel["n_detected"])
        with h5py.File(sel_path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["mock_data"] = True
            f.attrs["ndraw"] = int(sel["Ndraw"])
            f.attrs["Neff"] = neff
            f.attrs["Neff_flat"] = neff
            f.attrs["Neff_fiducial"] = sel_neff_fid
            f.attrs["selection_proposal"] = "population+uniform"
            f.attrs["measurement_family"] = dark.MEASUREMENT_FAMILY
            f.attrs["snr_ref"] = float(meas.snr_ref)
            f.attrs["snr_threshold"] = float(meas.snr_threshold)
            f.attrs["sigma_rho"] = float(meas.sigma_rho)
            f.attrs["chi_eff_swap_applied"] = True
            f.attrs["chi_eff_amax"] = 0.99
            f.attrs["cosmology_H0"] = float(W16.H0_TRUE)
            f.attrs["cosmology_Om0"] = float(W16.OM0)
            f.attrs["pop_model"] = "powerlaw+peak"
            f.attrs["shared_beta"] = True
            f.attrs["shared_spin"] = True
            f.attrs["shared_gamma"] = True
            for k in ("m1det", "m2det", "m1src", "m2src", "dL", "chieff",
                      "ra", "dec", "pdraw"):
                f.create_dataset(k, data=sel[k], compression="gzip")
        if verbose:
            print(f"[mock] injections {n_det}/{sel['Ndraw']} detected, "
                  f"Neff_fid={sel_neff_fid:.0f}", flush=True)

    # ------------------------------------------------------------- write out
    cat_path = out / f"catalog_pixelated_nside_{world.nside}.h5"
    with h5py.File(cat_path, "w") as f:
        f.attrs["nside"] = int(world.nside)
        f.attrs["z_depth"] = float(Z_DEPTH)
        f.attrs["mock_data"] = True
        for k, v in pixelated.items():
            f.create_dataset(k, data=v, compression="gzip")

    gw_path = out / "gw_events.h5"
    with h5py.File(gw_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(nsamp)
        f.attrs["pe_cosmology_H0"] = float(W16.H0_TRUE)
        f.attrs["pe_cosmology_Om0"] = float(W16.OM0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["measurement_family"] = dark.MEASUREMENT_FAMILY
        f.attrs["p_pe_basis"] = dark.P_PE_BASIS
        f.attrs["snr_ref"] = float(meas.snr_ref)
        f.attrs["snr_threshold"] = float(meas.snr_threshold)
        f.attrs["sigma_rho"] = float(meas.sigma_rho)
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        for k, v in post.items():
            f.create_dataset(k, data=v, compression="gzip")
        tg = f.create_group("truth")
        for k, v in truth.items():
            tg.create_dataset(k, data=v)
        for k, v in pe_obs.items():
            if k not in truth:
                tg.create_dataset(k, data=v)

    # The depth map the inference reads: written in the SHIPPED mth_map schema
    # so ``catalogs.depth_map.load_selection_fraction`` is the only reader.
    # It always carries the UNPERTURBED map: Tier D-i's perturbation lives in
    # the survey, which is what "the completeness map is wrong" means.
    mth_path = out / "mth_map_nside16.h5"
    f_true = np.zeros(hp.nside2npix(world.nside))
    f_true[world.pix] = world.f_p
    with h5py.File(mth_path, "w") as f:
        f.attrs["nside"] = int(world.nside)
        f.attrs["ordering"] = "RING"
        f.create_dataset("masked_frac", data=1.0 - f_true)
        f.create_dataset("counts", data=(f_true > 0).astype(float))

    # The selection fit, written in the SHIPPED ``darksirens-selection-fit-1.0``
    # schema so every consumer (``load_selection_fit_json``, the Q-table
    # builder, the CLI guards) reads it through its own validator, but with
    # theta at the INJECTED TRUTH rather than from ``darksirens_fit_selection``.
    #
    # That is a deliberate idealization and it is measured, not assumed.  Run
    # on this mock, the shipped fitter returns M0hat = -19.9768 +- 0.0023 and
    # sigma_M = 1.1269 +- 0.0014 against the injected (-20.1542, 1.0) -- an
    # 8-sigma pull that moves C(z_depth) from 0.4909 to 0.4208, a 14% error in
    # the missing budget at the survey edge.  The cause is the mock's own
    # photo-z: sigma_z = 0.023 at z ~ 0.15 is a 15% distance error, i.e. 0.33
    # mag of scatter in the distance modulus, which the truncated-LF likelihood
    # absorbs into sigma_M (1.0 -> sqrt(1 + 0.33^2) = 1.05, and more once the
    # truncation couples the two).  Carrying that into Tier B would put a
    # selection systematic into all three arms and PR-6a would be measuring the
    # LF fitter.  The bias is reported in CLOSURE.md as an observation about
    # ``fit_selection_from_mags`` under photo-z, and the tiers run matched.
    fit_path = out / "selection_fit.json"
    with open(fit_path, "w") as f:
        json.dump({
            "format_version": "darksirens-selection-fit-1.0",
            "strata": [{
                "family": "gaussian", "m_lim": float(W16.M_LIM),
                "M0hat": float(W16.M0_HAT), "sigma_M": float(W16.SIGMA_M),
                # The Laplace covariance the shipped fitter returned on this
                # catalog; theta is FIXED in every scan below, so it is
                # provenance only (it would be the sampled-theta prior).
                "cov": [[5.2678458674121555e-06, 2.0751686935660284e-06],
                        [2.0751686935660284e-06, 1.8149002329439392e-06]],
                "n_gal": int(n_obs), "stratum": "all", "k_corr_coeffs": [],
                "meta": {"Om0": float(W16.OM0), "w0": -1.0, "wa": 0.0,
                         "H0_ref": 100.0, "nll": 0.0},
            }],
            "survey_path": str(cat_path),
            "pr6a_note": "theta at the injected truth; see make_mock.py",
        }, f, indent=1)
    cal_path = out / "n0_calibration.json"
    with open(cal_path, "w") as f:
        json.dump({"log10n0": float(np.log10(n0)), "delta": 0.0,
                   "n0_true_Mpc3_no_evo": float(n0),
                   "note": "injected truth"}, f, indent=1)

    truth_path = out / "truth.json"
    meta = dict(seed=int(seed), n0=float(n0), log10n0=float(np.log10(n0)),
                H0_true=float(W16.H0_TRUE), Om0=float(W16.OM0),
                z_depth=float(Z_DEPTH), z_universe=float(z_universe),
                n_complete=int(total), n_survey=int(n_obs), nobs=int(nobs),
                nsamp=int(nsamp), n_injections_detected=int(n_det),
                sel_neff_fiducial=sel_neff_fid,
                event_z_median=float(np.median(truth["z"])),
                event_z_max=float(truth["z"].max()),
                event_frac_below_depth=float((truth["z"] <= Z_DEPTH).mean()),
                pe_dl_max=float(post["dL"].max()),
                snr_ref=SNR_REF, snr_threshold=SNR_THRESHOLD,
                lognormal_tail=float(lognormal_tail),
                field_scale=float(field_scale),
                world=world.meta)
    with open(truth_path, "w") as f:
        json.dump(meta, f, indent=1)
    np.save(out / "xi_true.npy", np.asarray(xi_true))
    if verbose:
        print(f"[mock] wrote {out}", flush=True)
    return dict(outdir=str(out), catalog=str(cat_path), gw=str(gw_path),
                selection=str(sel_path), mth=str(mth_path),
                selection_fit=str(fit_path), n0_calibration=str(cal_path),
                truth=meta, xi_true=np.asarray(xi_true))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7001)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nobs", type=int, default=NOBS)
    p.add_argument("--ndraw", type=int, default=NDRAW)
    p.add_argument("--reuse-injections", default=None)
    a = p.parse_args(argv)
    build(a.seed, a.outdir, nobs=a.nobs, ndraw=a.ndraw,
          reuse_injections=a.reuse_injections)


if __name__ == "__main__":
    main()
