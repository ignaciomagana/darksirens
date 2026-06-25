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
        snr = _dark._network_snr(m1, m2, z, dl, rng)
        det = snr >= snr_threshold
        if np.any(det):
            kept.append({k: v[det] for k, v in dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr).items()})
    return {k: np.concatenate([x[k] for x in kept])[:nobs] for k in kept[0]}


def _bright_posterior_samples(rng, truth, nsamp, **kwargs):
    post = _dark._posterior_samples(rng, truth, nsamp, **kwargs)
    # Bright sirens have localized counterparts: every PE sample for an event
    # uses the same sky coordinates as the associated EM counterpart.
    for i, (ra, dec) in enumerate(zip(truth["ra"], truth["dec"])):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        post["ra"][sl] = ra
        post["dec"][sl] = dec
    return post


def _draw_joint_selection_batch(rng, ndraw, grids, pop, survey, snr_threshold):
    z = _dark._sample_uniform_comoving_z(rng, grids, ndraw)
    ra, dec = _dark._sample_sky(rng, ndraw)
    dl = _dark._interp_dl(z, grids)
    m1 = _dark._sample_powerlaw_peak_m1(rng, ndraw, pop)
    q = _dark._sample_q(rng, m1, pop)
    m2 = q * m1
    chi = _dark._sample_chieff(rng, ndraw, pop)
    gw_det = _dark._network_snr(m1, m2, z, dl, rng) >= snr_threshold
    em_det = _joint_em_detected(rng, ra, dec, z, dl, survey)
    det = gw_det & em_det

    # np.trapz was removed in NumPy 2.0; reuse the dark generator's version-safe shim.
    pz = np.interp(z, grids["z"], grids["dvc_dz"]) / _dark._trapz(grids["dvc_dz"], grids["z"])
    ddldz = np.gradient(grids["dl"], grids["z"])
    jac = np.interp(z, grids["z"], ddldz) * (1.0 + z)
    p_draw = _dark._mass_spin_pdf(m1, q, chi, pop) * pz / np.maximum(jac, 1.0e-300) / (4.0 * np.pi)
    p_draw = np.maximum(p_draw, 1.0e-300)

    return {
        "m1det": m1[det] * (1.0 + z[det]),
        "m2det": m2[det] * (1.0 + z[det]),
        "m1src": m1[det],
        "m2src": m2[det],
        "dL": dl[det],
        "chieff": chi[det],
        "ra": ra[det],
        "dec": dec[det],
        "pdraw": p_draw[det],
        "Ndraw": ndraw,
        "n_detected": int(det.sum()),
    }


def _joint_selection_injections(rng, ndraw, grids, pop, survey, snr_threshold, batch_size, target_detections=None, verbose=False):
    chunks = []
    n_proposed = 0
    n_detected = 0
    keys = ["m1det", "m2det", "m1src", "m2src", "dL", "chieff", "ra", "dec", "pdraw"]
    while n_proposed < ndraw:
        n_batch = min(batch_size, ndraw - n_proposed)
        chunk = _draw_joint_selection_batch(rng, n_batch, grids, pop, survey, snr_threshold)
        chunks.append(chunk)
        n_proposed += int(chunk["Ndraw"])
        n_detected += int(chunk["n_detected"])
        if verbose:
            print(f"  joint selection batch: proposed={n_proposed:,}/{ndraw:,}, detected={n_detected:,}")
        if target_detections is not None and n_detected >= target_detections:
            break
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected}


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
    sel = _joint_selection_injections(
        rng, args.ndraw, grids, pop, survey, args.snr_threshold, args.selection_batch_size,
        target_detections=target, verbose=args.verbose,
    )

    inv_pdraw = 1.0 / np.asarray(sel["pdraw"])
    selection_neff = float(inv_pdraw.sum() ** 2 / np.square(inv_pdraw).sum()) if len(inv_pdraw) else 0.0
    metadata = {
        "seed": args.seed,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": args.w0, "wa": args.wa},
        "population": asdict(pop),
        "survey": asdict(survey),
        "snr_threshold": args.snr_threshold,
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
    parser.add_argument("--ndraw", type=_dark._positive_int, default=100_000)
    parser.add_argument("--zmax", type=_dark._positive_float, default=0.08)
    parser.add_argument("--H0", type=_dark._positive_float, default=67.74)
    parser.add_argument("--Om0", type=_dark._positive_float, default=0.3075)
    parser.add_argument("--w0", type=float, default=-1.0, help="CPL dark-energy equation-of-state value today.")
    parser.add_argument("--wa", type=float, default=0.0, help="CPL dark-energy evolution parameter.")
    parser.add_argument("--snr-threshold", type=_dark._positive_float, default=8.0)
    parser.add_argument("--survey-z50", type=float, default=_dark.SurveyConfig.z50)
    parser.add_argument("--survey-width", type=_dark._positive_float, default=_dark.SurveyConfig.width)
    parser.add_argument("--galaxy-density-delta", type=float, default=_dark.SurveyConfig.delta)
    parser.add_argument("--selection-batch-size", type=_dark._positive_int, default=50_000)
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
