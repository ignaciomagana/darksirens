from types import SimpleNamespace

import pytest

pytest.importorskip("h5py")
pytest.importorskip("tinygp")

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from darksirens.em import zgrid
from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.likelihood import make_likelihood


def test_dark_sirens_likelihood_evaluates_before_sampling():
    nside = 1
    npix = hp.nside2npix(nside)
    n_events = 1
    nsamp = 2
    n_sel = 16

    zgals = np.full((npix, 2), 0.12, dtype=float)
    zgals[:, 1] = 0.22
    dzgals = np.full_like(zgals, 0.03)
    wgals = np.ones_like(zgals)
    ngals = np.full(npix, 2, dtype=np.int32)

    data = {
        "m1det": jnp.array([35.0, 36.0]),
        "m2det": jnp.array([25.0, 24.0]),
        "dL": jnp.array([500.0, 700.0]),
        "chieff": jnp.array([0.0, 0.05]),
        "p_pe": jnp.ones(nsamp),
        "pixels_pe": jnp.array([0, 1], dtype=jnp.int32),
        "m1detsels": jnp.full(n_sel, 35.0),
        "m2detsels": jnp.full(n_sel, 25.0),
        "dLsels": jnp.linspace(300.0, 900.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.arange(n_sel, dtype=jnp.int32) % npix,
        "nEvents": n_events,
        "Ndraw": float(n_sel),
        "nsamp": nsamp,
        "apix": hp.nside2pixarea(nside),
        "nside": nside,
        "n_pix_catalog": npix,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "delta_g_pix_z": jnp.zeros((npix, len(zgrid))),
        "dN_obs_kde": None,
        "pixel_to_cache_idx": None,
        "sigma_kernel": 0.03,
    }
    opts = SimpleNamespace(
        pop_model="powerlaw+peak_shared_beta_spin",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    likelihood = make_likelihood(
        opts,
        data,
        get_fixed_population_params(opts.pop_model),
    )

    value = likelihood(jnp.array([]))
    jax.block_until_ready(value)

    assert data["dN_obs_kde"] is not None
    assert data["pixel_to_cache_idx"] is not None
    assert value.shape == ()
