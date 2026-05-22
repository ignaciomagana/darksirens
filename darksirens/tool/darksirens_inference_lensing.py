#!/usr/bin/env python3
from __future__ import annotations
import argparse
import jax.numpy as jnp
import numpy as np

from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.data import load_all_data
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler
from darksirens.inference.likelihood_with_clusters import (
    darksiren_log_likelihood_with_clusters,
    CLUSTER_MODE_J2,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
)
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.inference.catalog_views import prepare_catalog_views, barrier
from darksirens.em.completion import build_pixel_kde_cache
from darksirens.utils.containers import EMCatalog, GWEvent
from darksirens.inference.pair_kde import make_pair_kde
from darksirens.lensing.lensed_injections import load_lensed_injections
from darksirens.lensing.clusters import load_clusters
from darksirens.lensing.slmarks import SISLensParams


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run singleton+pair (J=2) darksirens inference.")
    p.add_argument("--gw_path", required=True)
    p.add_argument("--gwselection_path", required=True)
    p.add_argument("--survey_path", default=None)
    p.add_argument("--lensed_injections_path", required=True)
    p.add_argument("--clusters_path", required=True)
    p.add_argument("--save_path", default="./")
    p.add_argument("--sampler", required=True, choices=["dynesty", "jaxns", "emcee", "numpyro"])
    p.add_argument("--pop_model", default="powerlaw+peak")
    p.add_argument("--universe_model", default="spectral_sirens", choices=["spectral_sirens", "spectral_sirens_wl", "dark_sirens", "dark_sirens_complete"])
    p.add_argument("--fix_population", action="store_true")
    p.add_argument("--fix_cosmology", action="store_true")
    p.add_argument("--fix_survey", action="store_true")
    p.add_argument("--nlive", type=int, default=1000)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--nsteps", type=int, default=1000)
    p.add_argument("--nuts_warmup", type=int, default=500)
    p.add_argument("--nuts_samples", type=int, default=1000)
    p.add_argument("--nuts_chains", type=int, default=1)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--show_progress", default=True)
    p.add_argument("--sigma_kernel", type=float, default=0.0)
    p.add_argument("--use_LSS", default=False)
    p.add_argument("--sel_batch_size", type=int, default=None)
    p.add_argument("--lensing_wl_model", choices=["lognormal", "tabulated"], default="lognormal")
    p.add_argument("--lensing_wl_a", type=float, default=4e-3)
    p.add_argument("--lensing_wl_b", type=float, default=1.5)
    p.add_argument("--lensing_wl_table_path", default=None)
    return p


def main() -> None:
    opts = _build_parser().parse_args()
    data = load_all_data(opts)
    nEvents = int(data["nEvents"])

    pop_params_fid = get_fixed_population_params(opts.pop_model)
    labels, lower_bound, upper_bound, *_ = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        prior_overrides={},
        fixed_parameter_values={},
    )
    prior_transform = make_prior_transform(lower_bound, upper_bound)
    decoder = build_parameter_decoder(opts, pop_params_fid, fixed_parameter_values={}, wl_params=data.get("wl_params"))

    catalogs = prepare_catalog_views(opts, data, opts.universe_model, data.get("counterpart_pixel"), cache_builder=build_pixel_kde_cache)

    gw_pe = GWEvent(
        m1det=barrier(jnp.asarray(data["m1det"])),
        m2det=barrier(jnp.asarray(data["m2det"])),
        dL=barrier(jnp.asarray(data["dL"])),
        chieff=barrier(jnp.asarray(data["chieff"])),
        prior_wt=barrier(jnp.asarray(data["p_pe"])),
        pixels=catalogs.sample_to_unique_pe,
        q=barrier(jnp.asarray(data["m2det"]) / jnp.asarray(data["m1det"])),
        valid=jnp.ones_like(jnp.asarray(data["dL"]), dtype=bool),
    )
    gw_sel = GWEvent(
        m1det=barrier(jnp.asarray(data["m1detsels"])),
        m2det=barrier(jnp.asarray(data["m2detsels"])),
        dL=barrier(jnp.asarray(data["dLsels"])),
        chieff=barrier(jnp.asarray(data["chieffsels"])),
        prior_wt=barrier(jnp.asarray(data["p_draw"])),
        pixels=catalogs.sample_to_unique_sel,
        q=barrier(jnp.asarray(data["m2detsels"]) / jnp.asarray(data["m1detsels"])),
        valid=jnp.ones_like(jnp.asarray(data["dLsels"]), dtype=bool),
    )

    em_catalog_pe = EMCatalog(catalogs.apix, catalogs.zgals_pe_catalog, catalogs.dzgals_pe_catalog, catalogs.wgals_pe_catalog, catalogs.ngals_pe_catalog, catalogs.delta_g_pix_z, catalogs.sigma_kernel, catalogs.dN_obs_kde_pe, catalogs.pixel_to_cache_idx_pe, catalogs.unique_pixels_pe, catalogs.sample_to_unique_pe)
    em_catalog_sel = EMCatalog(catalogs.apix, catalogs.zgals_sel_catalog, catalogs.dzgals_sel_catalog, catalogs.wgals_sel_catalog, catalogs.ngals_sel_catalog, catalogs.delta_g_pix_z, catalogs.sigma_kernel, catalogs.dN_obs_kde_sel, catalogs.pixel_to_cache_idx_sel, catalogs.unique_pixels_sel, catalogs.sample_to_unique_sel)

    kdes = [make_pair_kde(np.asarray(data["m1det"])[i], np.asarray(data["m2det"])[i] / np.asarray(data["m1det"])[i], np.asarray(data["dL"])[i], np.asarray(data["chieff"])[i], np.asarray(data["p_pe"])[i]) for i in range(nEvents)]

    lensed_injections = load_lensed_injections(opts.lensed_injections_path)
    clusters = load_clusters(opts.clusters_path, n_events=nEvents)
    paired = np.asarray(clusters.pair_indices)
    used = np.unique(paired.reshape(-1)) if paired.size else np.array([], dtype=np.int32)
    singletons = np.array([i for i in range(nEvents) if i not in set(used.tolist())], dtype=np.int32)

    wl_backend = WL_BACKEND_DISABLED
    wl_a = wl_b = jnp.asarray(0.0)
    wl_z_grid = jnp.asarray([0.0, 1.0]); wl_log_mu_grid = jnp.asarray([0.0, 1.0]); wl_log_p_table = jnp.asarray([[0.0, 0.0], [0.0, 0.0]])
    wl_params = data.get("wl_params")
    if opts.universe_model == "spectral_sirens_wl" and wl_params is not None:
        if int(wl_params.backend) == WL_BACKEND_LOGNORMAL:
            wl_backend = WL_BACKEND_LOGNORMAL; wl_a = jnp.asarray(wl_params.a); wl_b = jnp.asarray(wl_params.b)
        else:
            wl_backend = WL_BACKEND_TABULATED; wl_z_grid = jnp.asarray(wl_params.z_grid); wl_log_mu_grid = jnp.asarray(wl_params.log_mu_grid); wl_log_p_table = jnp.asarray(wl_params.log_p_table)

    def likelihood(coord):
        cosmo, survey, pop_params = decoder.decode(coord)
        return darksiren_log_likelihood_with_clusters(
            cosmo, survey, pop_params, gw_pe, em_catalog_pe, gw_sel, em_catalog_sel,
            nEvents, int(data["nsamp"]), float(data["Ndraw"]),
            jnp.asarray(singletons, dtype=jnp.int32), jnp.asarray(paired, dtype=jnp.int32), int(singletons.shape[0]), int(paired.shape[0]),
            lensed_injections, tuple(kdes), SISLensParams(), jnp.zeros((int(lensed_injections.n_kept),), dtype=jnp.float64),
            opts.pop_model, opts.universe_model, sel_batch_size=opts.sel_batch_size, cluster_mode=CLUSTER_MODE_J2,
            wl_backend=wl_backend, wl_a=wl_a, wl_b=wl_b, wl_z_grid=wl_z_grid, wl_log_mu_grid=wl_log_mu_grid, wl_log_p_table=wl_log_p_table,
        )

    run_sampler(method=opts.sampler, likelihood=likelihood, prior_transform=prior_transform, labels=labels, lower_bound=lower_bound, upper_bound=upper_bound, opts=opts)


if __name__ == "__main__":
    main()
