"""Four independent defects in the lensing CLI, each already fixed in its twin.

* prior_kinds was never passed to make_prior_transform / run_sampler, so every
  non-uniform prior family (the whitened GP latents, xi ~ N(0,1)) was silently
  sampled FLAT. The main CLI threads them.
* _make_run_dir omitted the seed and used exist_ok=True, so same-config jobs
  started in the same second clobbered each other. The main CLI embeds the seed
  and retries with a numeric suffix.
* _resolve_pair_marks read inp["pair_time_sigma"], which is empty under
  --partition_mode marginalize_exact (the widths live per-partition), so the
  auto rule fell through to the quadrature implementation it exists to avoid.
* The lensed-injection loaders read attrs["n_draw_sources"] unconditionally
  while the module docstring documents Ndraw_sources and preflight accepts
  either, so a file matching the documented schema passed preflight then
  raised KeyError.
"""
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# prior_kinds
# ---------------------------------------------------------------------------

def test_lensing_cli_threads_prior_kinds_to_transform_and_sampler():
    import darksirens.cli.inference_lensing as cli

    build_src = inspect.getsource(cli._build_space_and_closures)
    assert "make_prior_transform(lower, upper, prior_kinds)" in build_src, (
        "make_prior_transform called without prior_kinds -> every non-uniform "
        "prior silently becomes uniform"
    )
    run_src = inspect.getsource(cli._run_lensing_sampling)
    assert "prior_kinds=prior_kinds" in run_src, (
        "run_sampler called without prior_kinds -> the numpyro path builds its "
        "own prior and keeps sampling GP latents uniform"
    )
    # and the value must actually reach the sampler function
    assert "prior_kinds" in inspect.signature(cli._run_lensing_sampling).parameters


def test_gp_models_declare_a_non_uniform_prior():
    """Establishes the stake: there really are non-uniform priors to lose."""
    from darksirens.sky.registry import get_sky_model

    specs = get_sky_model("sphere_gp").param_specs
    kinds = {s.prior_kind for s in specs}
    assert "normal" in kinds, "no normal-prior latents -> this defect is moot"


def test_uniform_only_prior_kinds_reproduce_the_affine_map():
    """Threading prior_kinds must not perturb an all-uniform lensing run."""
    import jax.numpy as jnp

    from darksirens.inference.prior import make_prior_transform

    lo = np.array([0.0, -1.0, 10.0])
    hi = np.array([1.0, 2.0, 20.0])
    kinds = [("uniform", None, None)] * 3
    u = jnp.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        np.asarray(make_prior_transform(lo, hi)(u)),
        np.asarray(make_prior_transform(lo, hi, kinds)(u)),
        rtol=0, atol=0,
    )


# ---------------------------------------------------------------------------
# run directory collisions
# ---------------------------------------------------------------------------

def test_lensing_run_dir_embeds_seed_and_never_collides(tmp_path):
    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(save_path=str(tmp_path), pop_model="powerlaw+peak",
                           cluster_mode="off", sampler="tinyns", seed=7)
    first = cli._make_run_dir(opts)
    second = cli._make_run_dir(opts)          # same config, same second
    assert first != second, "two same-config runs shared a directory"
    assert "seed7" in os.path.basename(first)
    assert os.path.isdir(first) and os.path.isdir(second)

    other = SimpleNamespace(**{**vars(opts), "seed": 8})
    assert "seed8" in os.path.basename(cli._make_run_dir(other))


# ---------------------------------------------------------------------------
# pair_time_sigma under marginalize_exact
# ---------------------------------------------------------------------------

def _pair(sigma):
    return SimpleNamespace(i=0, j=1, delta_t_obs=1.0, sigma_delta_t=sigma)


def test_resolve_pair_marks_falls_back_to_candidate_pairs():
    """marginalize_exact keeps the widths on candidate_pairs, not on inp."""
    import darksirens.cli.inference_lensing as cli

    opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl="auto",
                           sl_tau_A=1.0, sl_tau_n=1.0)
    from darksirens.lensing.slmarks import make_sis_lens_params

    T0 = float(make_sis_lens_params(A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n).T0)
    sharp = 0.5 * cli._TIME_DELTA_SHARPNESS * T0     # well inside the delta regime

    # No top-level pair_time_sigma (the marginalize_exact case), sharp marks.
    inp = {"candidate_pairs": [_pair(sharp), _pair(sharp)]}
    assert cli._resolve_pair_marks(opts, inp) == cli.PAIR_MARKS_TIME_DELTA

    # Broad marks still select the quadrature implementation.
    broad = 5.0 * cli._TIME_DELTA_SHARPNESS * T0
    inp_broad = {"candidate_pairs": [_pair(broad)]}
    assert cli._resolve_pair_marks(opts, inp_broad) == cli.PAIR_MARKS_TIME


def test_resolve_pair_marks_prefers_explicit_inp_sigmas():
    """The pre-existing path (sigmas on inp) is unchanged."""
    import darksirens.cli.inference_lensing as cli
    from darksirens.lensing.slmarks import make_sis_lens_params

    opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl="auto",
                           sl_tau_A=1.0, sl_tau_n=1.0)
    T0 = float(make_sis_lens_params(A_tau=opts.sl_tau_A, n_tau=opts.sl_tau_n).T0)
    inp = {"pair_time_sigma": np.array([0.5 * cli._TIME_DELTA_SHARPNESS * T0])}
    assert cli._resolve_pair_marks(opts, inp) == cli.PAIR_MARKS_TIME_DELTA


def test_resolve_pair_marks_explicit_impl_still_wins():
    import darksirens.cli.inference_lensing as cli

    for impl, expected in (("quadrature", cli.PAIR_MARKS_TIME),
                           ("delta", cli.PAIR_MARKS_TIME_DELTA)):
        opts = SimpleNamespace(pair_marks="time", pair_time_mark_impl=impl,
                               sl_tau_A=1.0, sl_tau_n=1.0)
        assert cli._resolve_pair_marks(opts, {}) == expected


# ---------------------------------------------------------------------------
# Ndraw_sources / n_draw_sources
# ---------------------------------------------------------------------------

def test_both_ndraw_source_spellings_are_accepted():
    from darksirens.lensing.lensed_injections import _read_n_draw_sources

    assert _read_n_draw_sources({"n_draw_sources": 11}) == 11.0
    assert _read_n_draw_sources({"Ndraw_sources": 13}) == 13.0
    # the documented spelling is what preflight validates under
    assert _read_n_draw_sources({"Ndraw_sources": 5, "n_draw_sources": 5}) == 5.0


def test_missing_ndraw_sources_names_both_spellings():
    from darksirens.lensing.lensed_injections import _read_n_draw_sources

    with pytest.raises(KeyError, match="Ndraw_sources"):
        _read_n_draw_sources({})
