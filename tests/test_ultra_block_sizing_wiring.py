"""Block-sizing wiring regressions (codex review 2026-08-23, findings 9/10).

Two under-reservation defects of the same class as the remediated 46x
sparse-catalog one:

* the CLI never threaded ``lss_field_mode``/``latent_dims`` into the static
  estimators or the resolver, so a ``--lss_field_mode latent`` run was memory-
  modelled as a table run with NO Q state at all (the mode loads no log-Q
  table, so the members branch never fired): ~854 MB of ``base_miss``, the
  latent leaves and the rho transient were all un-reserved, and the entire
  latent accounting subsystem in ``block_sizing`` was dead in production;
* ``sampler_block_sizing_profile`` returned ``concurrent_evals=1`` for numpyro
  even under ``--nuts_chain_method vectorized``, which ``vmap``s the NUTS
  kernel over ``--nuts_chains`` — N chains' value+grad transients live at
  once, so the predicted peak was understated by ~N x.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.likelihood.block_sizing import (
    BLOCK_AUTO,
    LatentDims,
    latent_pending_bytes,
    resolve_block_sizes,
    sampler_block_sizing_profile,
)

GB = 1024**3


# ── finding 10: numpyro vectorized-chain concurrency ─────────────────────────

def test_numpyro_vectorized_chains_scale_concurrent_evals():
    NS = SimpleNamespace
    assert sampler_block_sizing_profile(
        NS(sampler="numpyro", nuts_chain_method="vectorized", nuts_chains=4)
    ) == (True, 4)
    # sequential laxmaps one chain at a time; parallel pmaps one per device —
    # both are concurrency 1 PER DEVICE, whatever nuts_chains says.
    assert sampler_block_sizing_profile(
        NS(sampler="numpyro", nuts_chain_method="sequential", nuts_chains=4)
    ) == (True, 1)
    assert sampler_block_sizing_profile(
        NS(sampler="numpyro", nuts_chain_method="parallel", nuts_chains=4)
    ) == (True, 1)
    # A single vectorized chain is still one evaluation.
    assert sampler_block_sizing_profile(
        NS(sampler="numpyro", nuts_chain_method="vectorized", nuts_chains=1)
    ) == (True, 1)
    # Missing attrs (older opts namespaces) keep the shipped default.
    assert sampler_block_sizing_profile(NS(sampler="numpyro")) == (True, 1)
    # Garbage degrades to 1, never raises.
    assert sampler_block_sizing_profile(
        NS(sampler="numpyro", nuts_chain_method="vectorized",
           nuts_chains="not-a-number")
    ) == (True, 1)
    # nuts_* attrs on a non-numpyro sampler stay inert.
    assert sampler_block_sizing_profile(
        NS(sampler="dynesty", nuts_chain_method="vectorized", nuts_chains=8)
    ) == (False, 1)


def test_vectorized_chains_shrink_the_resolved_blocks():
    """The multiplier must reach the resolver: 8 vectorized chains cannot be
    promised the single pass 1 chain gets on the same card."""
    base = dict(
        n_events=250, n_samp=4000, n_sel=1_000_000,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        has_catalog=False, flow_path=False, needs_grad=True,
        free_bytes=120 * GB, backend="gpu",
    )
    one = resolve_block_sizes(concurrent_evals=1, **base)
    eight = resolve_block_sizes(concurrent_evals=8, **base)
    assert one.source == "auto-single-pass"
    assert eight.source != "auto-single-pass"
    assert eight.sel_batch_size is not None
    assert eight.sel_batch_size < base["n_sel"]


# ── finding 9: CLI threading of latent dims into the sizing inputs ───────────

M_DRAW, N_FIT, M_Z, N_B, M_SPH, N_THETA = 3, 40, 6, 5, 4, 2


def _write_anchor(path, *, with_sensitivity=True):
    """Minimal /latent_field group with load_latent_plan's dataset layout:
    row_fac WITHOUT the pad row, A_moments on the sub grid, sensitivity_S
    resident unconditionally."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset(
            "row_fac", data=np.zeros((M_DRAW, N_FIT, M_Z), np.float32))
        g.create_dataset(
            "A_moments", data=np.zeros((M_DRAW, N_B, 10), np.float64))
        if with_sensitivity:
            g.create_dataset(
                "sensitivity_S",
                data=np.zeros((M_SPH * M_Z, N_THETA), np.float64))
        g.attrs["basis_meta"] = json.dumps({"M_sph": M_SPH, "M_z": M_Z})


def _opts_and_data(artifact, mode="latent"):
    data = {
        "m1detsels": np.zeros(1000, np.float64),
        "nEvents": 5, "nsamp": 100,
        "zgals_pe": np.zeros((10, 50), np.float64),
        "zgals_sel": np.zeros((7, 50), np.float64),
        "catalog_memory": {
            "max_galaxies_per_unique_pixel": 50,
            "unique_pe_pixels": 10, "unique_sel_pixels": 7,
        },
    }
    opts = SimpleNamespace(
        survey_path="cat.h5", n_catalogs=1, gw_flows_path=None,
        drop_full_catalog=False, sampler="numpyro",
        sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
        catalog_sky_weighting="conditional",
        lss_field_mode=mode, lss_field_artifact=artifact,
    )
    return opts, data


def test_cli_block_sizing_inputs_carry_latent_dims(tmp_path):
    pytest.importorskip("jax")
    from darksirens.cli.inference import _block_sizing_inputs
    from darksirens.redshift.grid import zgrid

    artifact = str(tmp_path / "anchor.h5")
    _write_anchor(artifact)
    opts, data = _opts_and_data(artifact)
    kw = _block_sizing_inputs(opts, data)

    ng = int(np.asarray(zgrid).shape[0])
    dims = kw["latent_dims"]
    assert isinstance(dims, LatentDims)
    # Artifact shapes verbatim (row_fac is stored WITHOUT the pad row) plus the
    # RUN's grid (the loader pads the artifact's z axis to it).
    assert (dims.m_draw, dims.n_fit, dims.m_z) == (M_DRAW, N_FIT, M_Z)
    assert (dims.n_b, dims.m_sph, dims.n_theta) == (N_B, M_SPH, N_THETA)
    assert dims.n_grid == ng
    # Per-VIEW rows: base_miss_bytes carries the PE+sel factor 2 itself, so the
    # widest view is what never under-reserves the pair.
    assert dims.n_rows == 10
    assert dims.n_field_rows == 0
    assert kw["latent_rung"] == 0

    # The pending reserve is the KDE cache PLUS the latent substitution — the
    # exact quantity the un-wired CLI silently dropped ("a catalog run with no
    # Q state at all").
    kde = (10 + 7) * ng * 8
    assert kw["static_state_bytes"] == kde + latent_pending_bytes(dims)
    assert kw["static_state_bytes"] > kde

    # And the dims survive to the resolver without tripping its mode guard.
    kw.pop("static_state_full_bytes")
    plan = resolve_block_sizes(free_bytes=200 * GB, backend="gpu", **kw)
    assert plan.source in ("auto", "auto-single-pass", "auto-floor-reduced")


def test_cli_latent_static_strictly_exceeds_the_table_accounting(tmp_path):
    pytest.importorskip("jax")
    from darksirens.cli.inference import _block_sizing_inputs

    artifact = str(tmp_path / "anchor.h5")
    _write_anchor(artifact)
    opts_lat, data = _opts_and_data(artifact)
    opts_tab, _ = _opts_and_data(None, mode="table")

    lat = _block_sizing_inputs(opts_lat, data)
    tab = _block_sizing_inputs(opts_tab, data)
    assert tab["latent_dims"] is None
    dims = lat["latent_dims"]
    assert (lat["static_state_bytes"] - tab["static_state_bytes"]
            == latent_pending_bytes(dims))
    # base_miss alone (the 854 MB term at production rows) must be inside it.
    assert (lat["static_state_bytes"] - tab["static_state_bytes"]
            >= dims.base_miss_bytes)
    # The report-only full figure carries the same substitution.
    assert (lat["static_state_full_bytes"] - tab["static_state_full_bytes"]
            == latent_pending_bytes(dims))


def test_cli_latent_dims_without_sensitivity_block(tmp_path):
    pytest.importorskip("jax")
    from darksirens.cli.inference import _latent_block_sizing_dims

    artifact = str(tmp_path / "anchor_nos.h5")
    _write_anchor(artifact, with_sensitivity=False)
    opts, data = _opts_and_data(artifact)
    dims = _latent_block_sizing_dims(opts, data, 64)
    assert dims.n_theta == 0 and dims.sensitivity_bytes == 0

    # Table mode never opens the artifact and returns None.
    opts.lss_field_mode = "table"
    assert _latent_block_sizing_dims(opts, data, 64) is None
