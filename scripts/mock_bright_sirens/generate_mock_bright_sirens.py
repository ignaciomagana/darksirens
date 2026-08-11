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

SNR_REF_DEFAULT = _dark.SNR_REF_DEFAULT

# MEASUREMENT FAMILY.  The bright-siren PE and detection rule are the dark
# generator's all-observable family (``_dark.MEASUREMENT_FAMILY``), used through
# ``_dark._detect_on_observation`` / ``_dark._posterior_samples``: the threshold
# acts on a RECORDED ``rho_obs = rho_opt(theta) + N(0, sigma_rho)`` and every
# measurement width is a function of that recorded number alone.
#
# The historical bright rule -- widths scaled by the LATENT truth
# (``0.08 m1det_true``) and a threshold on a latent Beta(2,5)^0.5 projection
# amplitude -- is not kept, not even behind a flag: the sibling generator
# measured both defects on a matched campaign (the detected-set score identity
# E[C] = E[A] violated at 11.3 sigma, and -0.49 +- 0.08 km/s/Mpc of recovered H0
# from the sky channel alone), so any closure run on it would validate against a
# biased truth.  See the measurement-family header in
# scripts/mock_dark_sirens/generate_mock_data.py for the derivations.
#
# Only the sky channel differs: a bright siren's EM counterpart localises it
# exactly, so every PE sample of an event carries the counterpart's direction
# (:func:`_bright_posterior_samples`) instead of a width-``sigma_ang`` cloud.
# ``p_pe`` carries no sky factor either way (the prior is flat in the sky), so
# the stored column stays the exact PE prior of the recorded measurement.


def _joint_em_detected(rng, ra, dec, z, dl, survey):
    abs_mag = rng.normal(survey.absolute_mag_mean, survey.absolute_mag_sigma, len(z))
    dl_pc = dl * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    dec_deg = np.rad2deg(dec)
    footprint = (dec_deg >= survey.footprint_dec_min_deg) & (dec_deg <= survey.footprint_dec_max_deg)
    depth = (z <= survey.z_hard_max) & (app_mag <= survey.magnitude_limit)
    completeness = expit((survey.z50 - z) / survey.width)
    return footprint & depth & (rng.uniform(size=len(z)) < completeness)


def _draw_bright_events_until_detected(rng, nobs, observed_catalog, grids, pop,
                                      meas):
    """Detected bright-siren events from the EM-observed host catalog.

    Delegates to the dark generator's ``_draw_events_until_detected``, i.e. the
    threshold acts on the RECORDED ``rho_obs`` and the measurement it thresholded
    is carried forward under ``obs_*`` so the PE conditions on the very data the
    selection saw.  The host-acceptance rate factor ``(1+z)**(gamma-1)`` (which
    makes the detected events follow the inference population's redshift model)
    lives there too, so the two generators cannot drift apart.
    """
    if len(observed_catalog["z"]) == 0:
        raise RuntimeError("EM selection retained no galaxies; increase n0/zmax "
                           "or loosen survey settings.")
    truth, _rejected = _dark._draw_events_until_detected(
        rng, nobs, observed_catalog, grids, pop, meas.snr_threshold, meas=meas,
    )
    return truth


def _bright_posterior_samples(rng, truth, nsamp, meas):
    """Exact flat-prior posteriors of the recorded measurement, with EM sky.

    ``_dark._posterior_samples`` conditions on the ``obs_*`` columns recorded by
    the event loop; the sky samples are then replaced by the associated EM
    counterpart's direction, which is what a bright siren's localisation is.
    """
    post, _observations = _dark._posterior_samples(rng, truth, nsamp, meas)
    for i, (ra, dec) in enumerate(zip(truth["ra"], truth["dec"])):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        post["ra"][sl] = ra
        post["dec"][sl] = dec
    return post


def _draw_joint_selection_batch(rng, ndraw, grids, pop, survey, meas,
                                proposal="population", m1det_range=_dark._M1DET_RANGE):
    """Numpy reference joint GW+EM selection draw (the JAX path mirrors it).  See
    generate_mock_data._draw_selection_batch for the ``proposal`` semantics; the EM
    footprint/depth/completeness cut multiplies the GW detection mask, and pdraw
    is unchanged (the EM selection is part of the selection function, not the proposal).

    The GW cut is ``_dark._detect_on_observation``, i.e. the SAME recorded-rho_obs
    rule the events passed -- mu(theta) must be the probability of passing the
    event rule, so the two can never be allowed to differ."""
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
    gw_det, _obs = _dark._detect_on_observation(
        rng, m1src, m2src, z, dl, ra, dec, chi, meas, need_sky=False)
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


def _joint_selection_injections(rng, ndraw, grids, pop, survey, meas, batch_size,
                                target_detections=None, verbose=False,
                                proposal="population", m1det_range=_dark._M1DET_RANGE):
    # No snr_fn override: the kernel's own statistic is the recorded
    # ``rho_obs = rho_opt(theta) + N(0, sigma_rho)``, which is exactly the rule
    # the bright events are now drawn against.
    kernel = _dark._make_selection_kernel(
        grids, pop, float(meas.snr_threshold), proposal, m1det_range,
        extra_detect=_em_extra_detect(survey), meas=meas,
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

    meas = _dark.MeasurementConfig(
        snr_ref=args.snr_ref, snr_threshold=args.snr_threshold,
        sigma_rho=args.snr_uncertainty, a_mc=args.lnmc_uncertainty,
        a_q=args.lnq_uncertainty, a_chi=args.chieff_uncertainty,
    )
    cosmo = _dark._build_cosmology(args.H0, args.Om0, args.w0, args.wa)
    grids = _dark._cosmology_grids(cosmo, float(args.zmax))
    n_galaxies = _dark._galaxy_count_from_density(args.n0, args.galaxy_density_delta, grids)
    complete = _dark._generate_complete_catalog(rng, n_galaxies, grids, survey)
    observed_mask = _dark._apply_survey_selection(rng, complete, survey)
    observed_catalog = {k: v[observed_mask] for k, v in complete.items()}

    truth = _draw_bright_events_until_detected(
        rng, args.nobs, observed_catalog, grids, pop, meas)
    post = _bright_posterior_samples(rng, truth, args.nsamp, meas)
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
        rng, args.ndraw, grids, pop, survey, meas, selection_batch_size,
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
        "measurement_family": _dark.MEASUREMENT_FAMILY,
        "measurement": asdict(meas),
        "p_pe_basis": _dark.P_PE_BASIS,
        "selection_proposal": args.proposal,
        "redshifts": {
            "survey": "Z = z_obs = z + N(0, redshift_error_floor + "
                      "redshift_error_slope (1+z)); ZERR = zerr(z_obs)",
            "counterpart": "z = z_true + N(0, counterpart_dz)",
        },
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
    # The survey records the REALISED photo-z and declares the width of the model
    # at the value it recorded, dz = zerr(z_obs) -- exactly as the dark generator
    # does (_generate_complete_catalog / _catalog_zerr).  Writing the TRUE
    # redshift while declaring a width makes the likelihood smooth a comb that
    # carries no error, so the per-galaxy kernel g(z) N(z; z_g, sigma_g)/Z(z_g)
    # is not the Bayesian posterior for the host's redshift; that inconsistency
    # measured 7.6 sigma on a matched dark-siren mock.  z_obs is deliberately not
    # clipped (clipping re-introduces a censored observation); the realised
    # negative count is reported instead.
    z_obs_survey = complete["z_obs"][observed_mask]
    zerr_survey = _dark._catalog_zerr(z_obs_survey, survey)
    n_negative_z_obs = int((z_obs_survey < 0.0).sum())
    with h5py.File(raw_path, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = "Observed EM survey used to select possible bright-siren counterparts."
        f.attrs["metadata_json"] = json.dumps(metadata)
        f.create_dataset("TARGET_RA", data=np.rad2deg(complete["ra"][observed_mask]), compression="gzip", shuffle=True)
        f.create_dataset("TARGET_DEC", data=np.rad2deg(complete["dec"][observed_mask]), compression="gzip", shuffle=True)
        f.create_dataset("Z", data=z_obs_survey, compression="gzip", shuffle=True)
        f.create_dataset("ZERR", data=zerr_survey, compression="gzip", shuffle=True)
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

    # The counterpart redshift is a MEASUREMENT of width --counterpart-dz, and for
    # a bright-siren analysis it is the channel that sets H0: handing the exact
    # true redshift to a likelihood that convolves it with counterpart_dz is the
    # same zero-error-comb inconsistency as the survey block above, in the one
    # place it matters most.  Realise the declared error.
    z_counterpart = np.asarray(truth["z"], dtype=float) + args.counterpart_dz * rng.normal(
        size=len(truth["z"]))
    counterparts = [
        {"ra_rad": float(ra), "dec_rad": float(dec), "z": float(z), "counterpart_dz": args.counterpart_dz}
        for ra, dec, z in zip(truth["ra"], truth["dec"], z_counterpart)
    ]
    counterpart_path = out / "bright_counterparts.json"
    counterpart_path.write_text(json.dumps({"counterparts": counterparts, "metadata": metadata}, indent=2))

    print("Mock bright-siren data written:")
    print(f"  complete catalog : {complete_path} ({n_galaxies:,} galaxies)")
    print(f"  observed survey  : {raw_path} ({observed_mask.sum():,} galaxies retained, "
          f"photo-z realised, {n_negative_z_obs} with z_obs < 0 left unclipped)")
    print(f"  GW posteriors    : {gw_path} ({args.nobs} events x {args.nsamp} samples)")
    print(f"  GW+EM selection  : {sel_path} ({sel['n_detected']:,}/{sel['Ndraw']:,} detected injections, Neff={selection_neff:.1f})")
    print(f"  counterparts     : {counterpart_path}")


_REMOVED_FLAGS = {
    "--dL-fractional-uncertainty":
        "dL is no longer measured on its own -- it is DERIVED from (Mc_det, rho), "
        "so the distance precision follows from --snr-uncertainty and "
        "--lnmc-uncertainty.",
    "--m1det-fractional-uncertainty":
        "component masses are no longer measured independently (and a "
        "truth-scaled width is not a measurement at all); the mass channel is "
        "(ln Mc_det, ln q). Use --lnmc-uncertainty / --lnq-uncertainty.",
    "--m2det-fractional-uncertainty":
        "component masses are no longer measured independently (and a "
        "truth-scaled width is not a measurement at all); the mass channel is "
        "(ln Mc_det, ln q). Use --lnmc-uncertainty / --lnq-uncertainty.",
}


def parse_args():
    for arg in sys.argv[1:]:
        name = arg.split("=", 1)[0]
        if name in _REMOVED_FLAGS:
            raise SystemExit(
                f"{name} was removed with the all-observable measurement family: "
                f"{_REMOVED_FLAGS[name]}")
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
    # Measurement family (identical to the dark generator's; see its --snr-ref /
    # --snr-uncertainty / --lnmc-uncertainty / --lnq-uncertainty help).
    parser.add_argument("--snr-ref", type=_dark._positive_float,
                        default=_dark.SNR_REF_DEFAULT,
                        help="Amplitude scale of the detection statistic "
                             "rho_opt = snr_ref (Mc_det/30)^(5/6) (1000 Mpc/dL).")
    parser.add_argument("--snr-uncertainty", type=_dark._positive_float,
                        default=_dark.SIGMA_RHO_DEFAULT,
                        help="sigma_rho, the additive SNR noise: "
                             "rho_obs = rho_opt + N(0, sigma_rho).")
    parser.add_argument("--lnmc-uncertainty", type=_dark._positive_float,
                        default=_dark.A_MC_DEFAULT,
                        help="Width of ln Mc_det at rho_obs = --snr-threshold.")
    parser.add_argument("--lnq-uncertainty", type=_dark._positive_float,
                        default=_dark.A_Q_DEFAULT,
                        help="Width of ln q at rho_obs = --snr-threshold.")
    parser.add_argument("--chieff-uncertainty", type=_dark._positive_float,
                        default=_dark.A_CHI_DEFAULT,
                        help="Width of chi_eff at rho_obs = --snr-threshold.")
    parser.add_argument("--counterpart-dz", type=_dark._positive_float, default=1.0e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    write_mock_data(parse_args())
