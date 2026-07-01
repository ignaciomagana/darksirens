from argparse import Namespace

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from darksirens.cli.inference_lensing import build_parser as build_lensing_parser
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.likelihood.factory import _resolve_redshift_prior_materialization
from darksirens.redshift import zgrid
from darksirens.redshift.prior import prepare_redshift_prior_state


def _opts(**kwargs):
    base = {
        "sampler": "dynesty",
        "tinyns_resolved_config": {},
        "redshift_prior_barrier": "auto",
    }
    base.update(kwargs)
    return Namespace(**base)


def test_redshift_prior_barrier_resolver_modes():
    assert not _resolve_redshift_prior_materialization(
        _opts(
            sampler="tinyns",
            tinyns_resolved_config={"sample": "rwalk", "kernel": "jax"},
        )
    )
    assert _resolve_redshift_prior_materialization(
        _opts(
            sampler="tinyns",
            tinyns_resolved_config={"sample": "rwalk", "kernel": "python"},
        )
    )
    assert _resolve_redshift_prior_materialization(_opts(sampler="dynesty"))
    assert not _resolve_redshift_prior_materialization(_opts(redshift_prior_barrier="off"))
    assert _resolve_redshift_prior_materialization(_opts(redshift_prior_barrier="on"))


def test_prepare_redshift_prior_state_without_materialization_is_vmappable():
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)
    cat = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, zgrid.size)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )

    def build_log_pvol(h0):
        state = prepare_redshift_prior_state(
            "spectral_sirens",
            CosmoParams(H0=h0, Om0=0.3075),
            survey,
            cat,
            materialize_state=False,
        )
        return state.log_pvol[0]

    out = jax.vmap(build_log_pvol)(jnp.asarray([60.0, 70.0]))
    assert out.shape == (2,)


def test_prepare_redshift_prior_state_default_materializes(monkeypatch):
    calls = {"n": 0}

    def fake_materialize(state):
        calls["n"] += 1
        return state

    monkeypatch.setattr("darksirens.redshift.prior._materialize", fake_materialize)
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)
    cat = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, zgrid.size)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    prepare_redshift_prior_state("spectral_sirens", CosmoParams(H0=67.74, Om0=0.3075), survey, cat)
    assert calls["n"] == 1


def test_lensing_parser_accepts_redshift_prior_barrier_default():
    parser = build_lensing_parser()
    opts = parser.parse_args([
        "--gw_path", "gw.h5",
        "--gwselection_path", "sel.h5",
        "--sampler", "tinyns",
    ])
    assert opts.redshift_prior_barrier == "auto"
