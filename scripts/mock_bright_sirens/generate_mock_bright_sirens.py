#!/usr/bin/env python3
"""Generate end-to-end mock data for multi-event bright-siren inference.

This workflow mirrors ``scripts/mock_dark_sirens`` without modifying it.  It builds a
complete galaxy population, applies an EM survey selection, draws GW events only
from galaxies with detectable counterparts, writes bright-siren PE samples with
fixed event sky coordinates, and generates joint GW+EM selection injections.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import expit as jexpit
from scipy.special import expit


def _load_dark_mock_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "mock_dark_sirens" / "generate_mock_data.py"
    spec = importlib.util.spec_from_file_location("dark_mock_data_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import dark-siren mock helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_dark = _load_dark_mock_module()

SNR_REF_DEFAULT = 11.5


def _network_snr(m1, m2, z, dl, rng):
    """Semi-analytic network SNR with a ``Beta(2,5)**0.5`` projection latent.

    Moved here from the dark-siren generator, which no longer has a
    true-parameter detection rule: its threshold now acts on a recorded
    ``rho_obs`` and the projection latent is gone (a detection decision that
    depends on a variable absent from the data leaves an extra
    ``P(det|theta)`` inside each event's integral).  The bright-siren mock
    still uses the historical rule, so the historical implementation lives
    with its only remaining consumer rather than in a module that no longer
    contains it.
    """
    mchirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mchirp_det = mchirp * (1.0 + z)
    projection = rng.beta(2.0, 5.0, size=len(np.atleast_1d(m1))) ** 0.5
    return SNR_REF_DEFAULT * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl) * projection


def _legacy_posterior_samples(rng, truth, nsamp, dL_fractional_uncertainty=None,
                              m1det_fractional_uncertainty=0.08,
                              m2det_fractional_uncertainty=0.10,
                              chieff_uncertainty=0.08,
                              sky_uncertainty_deg=None):
    """Flat-prior posterior samples of one noisy measurement per event.

    Moved verbatim (draw order included) from the dark-siren generator's
    ``_posterior_samples`` with ``pe_centering="observed"``, which the
    all-observable rewrite of that generator removed.  The measurement is
    independent per channel: ``ln d_obs ~ N(ln dL, s)`` and additive Gaussians
    on the component masses / spin / sky, so the flat-prior posteriors are
    ``ln dL ~ N(ln d_obs + s^2, s)`` (the flat-in-dL volume factor) and normals
    centred on the observed value elsewhere.

    ``p_pe`` is the PE PRIOR in darksirens' canonical integration basis, which
    is ``(m1det, q, dL, chieff)`` and NOT ``(m1det, m2det, dL, chieff)`` --
    see ``darksirens/inference/utils.py``, "Integration variables".  A prior
    flat in the drawn variables ``(m1det, m2det, dL, chieff)`` therefore
    carries the Jacobian ``|dm2det/dq| = m1det``, so ``p_pe ∝ m1det``; the
    historical all-ones column mis-weighted the mass channel of every mock
    this generator has ever produced.  darksirens renormalises ``p_pe`` per
    event, so the column is stored normalised to mean 1 per event and only its
    shape matters.
    """
    nobs = len(truth["z"])
    arrays = {"ra": [], "dec": [], "dL": [], "m1det": [], "m2det": [],
              "chieff": [], "p_pe": []}
    for i in range(nobs):
        rho = truth["snr"][i]
        frac_dl = (dL_fractional_uncertainty if dL_fractional_uncertainty is not None
                   else np.clip(1.8 / rho, 0.08, 0.35))
        sigma_ang = np.deg2rad(sky_uncertainty_deg if sky_uncertainty_deg is not None
                               else np.clip(35.0 / rho, 1.0, 12.0))
        m1det = truth["m1"][i] * (1.0 + truth["z"][i])
        m2det = truth["m2"][i] * (1.0 + truth["z"][i])
        sig_m1 = m1det_fractional_uncertainty * m1det
        sig_m2 = m2det_fractional_uncertainty * m2det

        # One measurement per event ...
        dl_obs = float(truth["dl"][i] * np.exp(frac_dl * rng.normal()))
        ra_obs = float((truth["ra"][i]
                        + rng.normal(0.0, sigma_ang / max(np.cos(truth["dec"][i]), 0.1)))
                       % (2.0 * np.pi))
        dec_obs = float(np.clip(truth["dec"][i] + rng.normal(0.0, sigma_ang),
                                -0.5 * np.pi, 0.5 * np.pi))
        m1_obs = float(np.clip(rng.normal(m1det, sig_m1), 2.0, None))
        m2_obs = float(np.clip(rng.normal(m2det, sig_m2), 1.0, None))
        chi_obs = float(np.clip(rng.normal(truth["chi"][i], chieff_uncertainty),
                                -1.0, 1.0))
        # ... then the flat-prior posterior GIVEN that measurement.
        dl = rng.lognormal(np.log(dl_obs) + frac_dl**2, frac_dl, nsamp)
        dra = rng.normal(0.0, sigma_ang / max(np.cos(dec_obs), 0.1), nsamp)
        ddec = rng.normal(0.0, sigma_ang, nsamp)
        m1_draws = rng.normal(m1_obs, sig_m1, nsamp)
        m2_draws = rng.normal(m2_obs, sig_m2, nsamp)
        chi_draws = rng.normal(chi_obs, chieff_uncertainty, nsamp)

        m1_samples = np.clip(m1_draws, 2.0, None)
        arrays["ra"].append((ra_obs + dra) % (2.0 * np.pi))
        arrays["dec"].append(np.clip(dec_obs + ddec, -0.5 * np.pi, 0.5 * np.pi))
        arrays["dL"].append(dl)
        arrays["m1det"].append(m1_samples)
        arrays["m2det"].append(np.clip(m2_draws, 1.0, None))
        arrays["chieff"].append(np.clip(chi_draws, -1.0, 1.0))
        arrays["p_pe"].append(m1_samples / m1_samples.mean())
    return {k: np.concatenate(v) for k, v in arrays.items()}


def _joint_em_detected(rng, ra, dec, z, dl, survey):
    abs_mag = rng.normal(survey.absolute_mag_mean, survey.absolute_mag_sigma, len(z))
    dl_pc = dl * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    dec_deg = np.rad2deg(dec)
    footprint = (dec_deg >= survey.footprint_dec_min_deg) & (dec_deg <= survey.footprint_dec_max_deg)
    depth = (z <= survey.z_hard_max) & (app_mag <= survey.magnitude_limit)
    completeness = expit((survey.z50 - z) / survey.width)
    return footprint & depth & (rng.uniform(size=len(z)) < completeness)


def _draw_bright_events_until_detected(rng, nobs, observed_catalog, grids, pop, survey, snr_threshold):
    kept = []
    n_available = len(observed_catalog["z"])
    if n_available == 0:
        raise RuntimeError("EM selection retained no galaxies; increase n0/zmax or loosen survey settings.")
    while sum(len(x["z"]) for x in kept) < nobs:
        ntry = max(4 * nobs, 256)
        host_idx = rng.integers(0, n_available, ntry)
        z = observed_catalog["z"][host_idx]
        ra = observed_catalog["ra"][host_idx]
        dec = observed_catalog["dec"][host_idx]
        dl = _dark._interp_dl(z, grids)
        m1 = _dark._sample_powerlaw_peak_m1(rng, ntry, pop)
        q = _dark._sample_q(rng, m1, pop)
        m2 = q * m1
        chi = _dark._sample_chieff(rng, ntry, pop)
        snr = _network_snr(m1, m2, z, dl, rng)
        det = snr >= snr_threshold
        # Match the dark-siren generator's source-frame rate evolution.  The
        # host catalog is uniform in comoving volume, so accepting hosts with
        # probability proportional to (1+z)**(gamma-1) makes the bright-siren
        # population use the same redshift convention as the inference model.
        # With the registry-fixed powerlaw+peak fiducial, gamma=0; if the shared
        # PopulationConfig is ever changed deliberately, bright mocks follow it
        # automatically instead of silently using a different redshift truth.
        zmax_grid = float(grids["z"][-1])
        rate_gmax = max(1.0, (1.0 + zmax_grid) ** (pop.gamma - 1.0))
        rate_acc = (1.0 + z) ** (pop.gamma - 1.0) / rate_gmax
        det = det & (rng.uniform(size=len(z)) < rate_acc)
        if np.any(det):
            kept.append({k: v[det] for k, v in dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr).items()})
    return {k: np.concatenate([x[k] for x in kept])[:nobs] for k in kept[0]}


def _bright_posterior_samples(rng, truth, nsamp, **kwargs):
    post = _legacy_posterior_samples(rng, truth, nsamp, **kwargs)
    # Bright sirens have localized counterparts: every PE sample for an event
    # uses the same sky coordinates as the associated EM counterpart.
    for i, (ra, dec) in enumerate(zip(truth["ra"], truth["dec"])):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        post["ra"][sl] = ra
        post["dec"][sl] = dec
    return post


def _draw_joint_selection_batch(rng, ndraw, grids, pop, survey, snr_threshold,
                                proposal="population", m1det_range=_dark._M1DET_RANGE):
    """Numpy reference joint GW+EM selection draw (the JAX path mirrors it).  See
    generate_mock_data._draw_selection_batch for the ``proposal`` semantics; the EM
    footprint/depth/completeness cut multiplies the GW-SNR detection mask, and pdraw
    is unchanged (the EM selection is part of the selection function, not the proposal)."""
    z = _dark._sample_uniform_comoving_z(rng, grids, ndraw)
    ra, dec = _dark._sample_sky(rng, ndraw)
    dl = _dark._interp_dl(z, grids)
    if proposal == "uniform":
        m1lo, m1hi = m1det_range
        m1det = rng.uniform(m1lo, m1hi, ndraw)
        q = rng.uniform(0.0, 1.0, ndraw)
        chi = rng.uniform(-1.0, 1.0, ndraw)
        m1src = m1det / (1.0 + z)
    else:
        # Per-component draw + optional defensive-uniform lanes, mirroring
        # generate_mock_data._draw_selection_batch so the stored pdraw
        # (population or 0.9*pop + 0.1*unif mixture) matches the draws.
        m1src = _dark._sample_powerlaw_peak_m1(rng, ndraw, pop)
        q = _dark._sample_q(rng, m1src, pop)
        chi = _dark._sample_chieff(rng, ndraw, pop)
        if proposal == "population+uniform":
            m1lo, m1hi = m1det_range
            use_unif = rng.uniform(size=ndraw) < 0.1
            m1src = np.where(
                use_unif, rng.uniform(m1lo, m1hi, ndraw) / (1.0 + z), m1src
            )
            q = np.where(use_unif, rng.uniform(0.0, 1.0, ndraw), q)
            chi = np.where(use_unif, rng.uniform(-1.0, 1.0, ndraw), chi)
    m2src = q * m1src
    gw_det = _network_snr(m1src, m2src, z, dl, rng) >= snr_threshold
    em_det = _joint_em_detected(rng, ra, dec, z, dl, survey)
    det = gw_det & em_det

    p_draw = _dark._selection_pdraw(proposal, m1src, q, chi, z, grids, pop, m1det_range)

    return {
        "m1det": (m1src * (1.0 + z))[det],
        "m2det": (q * m1src * (1.0 + z))[det],
        "m1src": m1src[det],
        "m2src": m2src[det],
        "dL": dl[det],
        "chieff": chi[det],
        "ra": ra[det],
        "dec": dec[det],
        "pdraw": p_draw[det],
        "Ndraw": ndraw,
        "n_detected": int(det.sum()),
    }


def _em_extra_detect(survey):
    """Per-injection JAX EM-selection cut (footprint, depth, completeness) for the
    joint GW+EM selection kernel.  Uses a key derived from the injection key via
    fold_in so its randoms are independent of the GW draws in the dark kernel."""
    dec_min = float(survey.footprint_dec_min_deg)
    dec_max = float(survey.footprint_dec_max_deg)
    z_hard = float(survey.z_hard_max)
    mag_lim = float(survey.magnitude_limit)
    abs_mu = float(survey.absolute_mag_mean)
    abs_sig = float(survey.absolute_mag_sigma)
    z50 = float(survey.z50)
    width = float(survey.width)

    def extra(key, state):
        z, dl, dec = state["z"], state["dL"], state["dec"]
        k1, k2 = jax.random.split(jax.random.fold_in(key, 101), 2)
        abs_mag = abs_mu + abs_sig * jax.random.normal(k1)
        app_mag = abs_mag + 5.0 * jnp.log10(jnp.maximum(dl * 1.0e6, 10.0) / 10.0)
        dec_deg = jnp.rad2deg(dec)
        footprint = jnp.logical_and(dec_deg >= dec_min, dec_deg <= dec_max)
        depth = jnp.logical_and(z <= z_hard, app_mag <= mag_lim)
        completeness = jexpit((z50 - z) / width)
        return jnp.logical_and(jnp.logical_and(footprint, depth),
                               jax.random.uniform(k2) < completeness)

    return extra


def _projection_snr_jax(key, state):
    """The historical projection-latent network SNR, in JAX.

    The dark-siren selection kernel's own statistic is the all-observable
    ``rho_obs = rho_opt(theta) + N(0, sigma_rho)``.  The bright-siren events are
    still drawn against :func:`_network_snr`, and mu(theta) has to be the
    probability of passing the SAME rule the events passed, so the kernel's
    statistic is overridden here rather than composed with it."""
    m1src, q, z, dl = state["m1src"], state["q"], state["z"], state["dL"]
    m2src = q * m1src
    mchirp_det = (m1src * m2src) ** 0.6 / (m1src + m2src) ** 0.2 * (1.0 + z)
    proj = jnp.sqrt(jax.random.beta(key, 2.0, 5.0))
    return SNR_REF_DEFAULT * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl) * proj


def _joint_selection_injections(rng, ndraw, grids, pop, survey, snr_threshold, batch_size,
                                target_detections=None, verbose=False,
                                proposal="population", m1det_range=_dark._M1DET_RANGE):
    kernel = _dark._make_selection_kernel(
        grids, pop, float(snr_threshold), proposal, m1det_range,
        extra_detect=_em_extra_detect(survey), snr_fn=_projection_snr_jax,
    )
    return _dark._run_selection_chunks(
        rng, ndraw, grids, pop, proposal, batch_size, kernel,
        target_detections=target_detections, verbose=verbose,
        label="joint selection", m1det_range=m1det_range,
    )


def write_mock_data(args):
    rng = np.random.default_rng(args.seed)
    pop = _dark.PopulationConfig()
    survey = _dark.SurveyConfig(z50=args.survey_z50, width=args.survey_width, delta=args.galaxy_density_delta)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cosmo = _dark._build_cosmology(args.H0, args.Om0, args.w0, args.wa)
    grids = _dark._cosmology_grids(cosmo, float(args.zmax))
    n_galaxies = _dark._galaxy_count_from_density(args.n0, args.galaxy_density_delta, grids)
    complete = _dark._generate_complete_catalog(rng, n_galaxies, grids, survey)
    observed_mask = _dark._apply_survey_selection(rng, complete, survey)
    observed_catalog = {k: v[observed_mask] for k, v in complete.items()}

    truth = _draw_bright_events_until_detected(rng, args.nobs, observed_catalog, grids, pop, survey, args.snr_threshold)
    post = _bright_posterior_samples(
        rng,
        truth,
        args.nsamp,
        dL_fractional_uncertainty=args.dL_fractional_uncertainty,
        m1det_fractional_uncertainty=args.m1det_fractional_uncertainty,
        m2det_fractional_uncertainty=args.m2det_fractional_uncertainty,
        chieff_uncertainty=args.chieff_uncertainty,
        sky_uncertainty_deg=0.0,
    )
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)

    target = args.selection_target_detections
    if args.selection_per_observation_factor is not None:
        target = int(np.ceil(args.selection_per_observation_factor * args.nobs))
    # Chunk the nselection draws: explicit --selection-batch-size wins (legacy),
    # otherwise split into --nbatches equal chunks (default 1 = one kernel call).
    selection_batch_size = (
        args.selection_batch_size
        if args.selection_batch_size is not None
        else int(np.ceil(args.ndraw / args.nbatches))
    )
    sel = _joint_selection_injections(
        rng, args.ndraw, grids, pop, survey, args.snr_threshold, selection_batch_size,
        target_detections=target, verbose=args.verbose, proposal=args.proposal,
    )

    inv_pdraw = 1.0 / np.asarray(sel["pdraw"])
    selection_neff = float(inv_pdraw.sum() ** 2 / np.square(inv_pdraw).sum()) if len(inv_pdraw) else 0.0
    metadata = {
        "seed": args.seed,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": args.w0, "wa": args.wa},
        "population": asdict(pop),
        "survey": asdict(survey),
        "snr_threshold": args.snr_threshold,
        "selection_proposal": args.proposal,
        "universe_model_for_inference": "bright_sirens",
        "pop_model_for_inference": "powerlaw+peak",
        "shared_beta_for_inference": True,
        "shared_spin_for_inference": True,
        "shared_gamma_for_inference": True,
    }

    complete_path = out / "mock_galaxy_catalog_complete.h5"
    with h5py.File(complete_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Complete bright-siren mock galaxy catalog before EM incompleteness."
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in complete.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    raw_path = out / "mock_survey_raw.h5"
    zerr = survey.redshift_error_floor + survey.redshift_error_slope * (1.0 + complete["z"])
    with h5py.File(raw_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Observed EM survey used to select possible bright-siren counterparts."
        f.attrs["metadata_json"] = json.dumps(metadata)
        f.create_dataset("TARGET_RA", data=np.rad2deg(complete["ra"][observed_mask]), compression="gzip", shuffle=True)
        f.create_dataset("TARGET_DEC", data=np.rad2deg(complete["dec"][observed_mask]), compression="gzip", shuffle=True)
        f.create_dataset("Z", data=complete["z"][observed_mask], compression="gzip", shuffle=True)
        f.create_dataset("ZERR", data=zerr[observed_mask], compression="gzip", shuffle=True)
        f.create_dataset("WEIGHT", data=np.ones(int(observed_mask.sum())), compression="gzip", shuffle=True)

    gw_path = out / "mock_bright_gw_events.h5"
    with h5py.File(gw_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(args.nobs)
        f.attrs["nsamp"] = int(args.nsamp)
        f.attrs["pe_cosmology_H0"] = float(args.H0)
        f.attrs["pe_cosmology_Om0"] = float(args.Om0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key, val in post.items():
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)
        truth_group = f.create_group("truth")
        for key, val in truth.items():
            truth_group.create_dataset(key, data=val)

    sel_path = out / "mock_bright_gw_selection.h5"
    with h5py.File(sel_path, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])
        f.attrs["Neff"] = selection_neff
        f.attrs["selection_proposal"] = args.proposal
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["cosmology_H0"] = float(args.H0)
        f.attrs["cosmology_Om0"] = float(args.Om0)
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["metadata_json"] = json.dumps(metadata)
        for key in ["m1det", "m2det", "m1src", "m2src", "dL", "chieff", "ra", "dec", "pdraw"]:
            f.create_dataset(key, data=sel[key], compression="gzip", shuffle=True)

    counterparts = [
        {"ra_rad": float(ra), "dec_rad": float(dec), "z": float(z), "counterpart_dz": args.counterpart_dz}
        for ra, dec, z in zip(truth["ra"], truth["dec"], truth["z"])
    ]
    counterpart_path = out / "bright_counterparts.json"
    counterpart_path.write_text(json.dumps({"counterparts": counterparts, "metadata": metadata}, indent=2))

    print("Mock bright-siren data written:")
    print(f"  complete catalog : {complete_path} ({n_galaxies:,} galaxies)")
    print(f"  observed survey  : {raw_path} ({observed_mask.sum():,} galaxies retained)")
    print(f"  GW posteriors    : {gw_path} ({args.nobs} events x {args.nsamp} samples)")
    print(f"  GW+EM selection  : {sel_path} ({sel['n_detected']:,}/{sel['Ndraw']:,} detected injections, Neff={selection_neff:.1f})")
    print(f"  counterparts     : {counterpart_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data/mock_bright_sirens")
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--n0", type=_dark._positive_float, default=1.0e-3)
    parser.add_argument("--nobs", type=_dark._positive_int, default=3)
    parser.add_argument("--nsamp", type=_dark._positive_int, default=512)
    parser.add_argument("--nselection", "--ndraw", dest="ndraw", type=_dark._positive_int, default=100_000,
                        help="Total number of joint GW+EM selection injections to PROPOSE (the "
                             "detected subset is stored). Without an early-stop target, exactly "
                             "this many are drawn. --ndraw is a legacy alias.")
    parser.add_argument("--nbatches", type=_dark._positive_int, default=1,
                        help="Split the --nselection draws into this many equal chunks for the "
                             "jit/vmap kernel; chunking bounds device memory. Default 1 draws all "
                             "injections in a single kernel call (best on GPU).")
    parser.add_argument("--proposal", choices=_dark.SELECTION_PROPOSALS, default="population",
                        help="Selection-injection proposal. 'population' (default) draws "
                             "masses/spins from the fiducial population; 'uniform' draws a broad, "
                             "population-independent proposal that keeps the importance-sampling "
                             "Neff high across the prior.")
    parser.add_argument("--zmax", type=_dark._positive_float, default=0.08)
    parser.add_argument("--H0", type=_dark._positive_float, default=67.74)
    parser.add_argument("--Om0", type=_dark._positive_float, default=0.3075)
    parser.add_argument("--w0", type=float, default=-1.0, help="CPL dark-energy equation-of-state value today.")
    parser.add_argument("--wa", type=float, default=0.0, help="CPL dark-energy evolution parameter.")
    parser.add_argument("--snr-threshold", type=_dark._positive_float, default=8.0)
    parser.add_argument("--survey-z50", type=float, default=_dark.SurveyConfig.z50)
    parser.add_argument("--survey-width", type=_dark._positive_float, default=_dark.SurveyConfig.width)
    parser.add_argument("--galaxy-density-delta", type=float, default=_dark.SurveyConfig.delta)
    parser.add_argument("--selection-batch-size", type=_dark._positive_int, default=None,
                        help="Explicit chunk size (legacy). If set, it takes precedence over "
                             "--nbatches; otherwise the chunk size is --nselection / --nbatches.")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--selection-target-detections", type=_dark._positive_int, default=None)
    targets.add_argument("--selection-per-observation-factor", type=_dark._positive_float, default=None)
    parser.add_argument("--dL-fractional-uncertainty", type=_dark._positive_float, default=0.10)
    parser.add_argument("--m1det-fractional-uncertainty", type=_dark._positive_float, default=0.08)
    parser.add_argument("--m2det-fractional-uncertainty", type=_dark._positive_float, default=0.10)
    parser.add_argument("--chieff-uncertainty", type=_dark._positive_float, default=0.08)
    parser.add_argument("--counterpart-dz", type=_dark._positive_float, default=1.0e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    write_mock_data(parse_args())
