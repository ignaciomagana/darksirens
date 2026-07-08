"""The selection integral must use the catalog-completed prior for the dark
models (library review P0.2: commit e779816 silently hard-wired EVERY model's
selection to the pure volume prior, so the sampled survey block, Q_LSS, and
the catalog never entered mu — contradicting the methods paper's same-weight,
self-calibrating selection estimator)."""
from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.likelihood.core import selection_prior_model
from darksirens.likelihood.factory import make_likelihood
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params


def test_selection_prior_model_mapping():
    assert selection_prior_model("dark_sirens") == "dark_sirens"
    assert selection_prior_model("dark_sirens_complete") == "dark_sirens_complete"
    assert selection_prior_model("spectral_sirens") == "spectral_sirens"
    assert selection_prior_model("bright_sirens") == "spectral_sirens"
    assert selection_prior_model("spectral_sirens_wl") == "spectral_sirens"


def _dark_likelihood_value(z_pixel2):
    """Tiny dark-siren likelihood where PE samples live ONLY in pixel 7 while
    the injection set also covers pixel 2: moving pixel 2's galaxy redshift
    can change logL ONLY through the selection term's catalog prior."""
    nside = 1
    n_pix = hp.nside2npix(nside)
    nsamp, n_sel = 2, 8

    zgals = np.full((n_pix, 1), 0.10, dtype=float)
    zgals[2, 0] = z_pixel2
    dzgals = np.full((n_pix, 1), 0.02, dtype=float)
    wgals = np.ones((n_pix, 1), dtype=float)
    ngals = np.ones(n_pix, dtype=np.int32)

    data = {
        "nEvents": 1, "nsamp": nsamp, "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside), "nside": nside, "n_pix_catalog": n_pix,
        "zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals_catalog": ngals,
        "zgals_catalog": zgals, "dzgals_catalog": dzgals, "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((n_pix, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0]), "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]), "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp), "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel), "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=jnp.int32),
    }

    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled}
    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    mid = jnp.asarray([0.5 * (float(pop_lower[0]) + float(pop_upper[0]))])
    return float(ll(mid))


def test_dark_selection_responds_to_selection_catalog():
    """Fails under the e779816 hard-wire: with a volume-prior selection the
    pixel-2 galaxy redshift is invisible and the two values are identical."""
    v_a = _dark_likelihood_value(0.10)
    v_b = _dark_likelihood_value(0.28)
    assert np.isfinite(v_a) and np.isfinite(v_b)
    # The tiny fixture's response is small (~7e-7: the missing branch dominates
    # at the fiducial survey) but structural; under the volume-prior hard-wire
    # the two values are bit-identical, so any threshold above float noise
    # discriminates.
    assert abs(v_a - v_b) > 1e-8, (v_a, v_b)
