"""dynesty must honour --seed: without an explicit rstate it draws fresh
global NumPy entropy, so identical-seed runs differed while results.hdf5
recorded a seed that had no effect (library review, CLI finding 2). tinyns
and numpyro already used opts.seed."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dynesty")

from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler

from test_numpyro_sampler import _small_likelihood_inputs


def _run(seed):
    opts, data, prior_overrides, fixed, sampled = _small_likelihood_inputs(
        "spectral_sirens"
    )
    pop_fid = get_fixed_population_params(opts.pop_model)
    likelihood = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    labels, lower, upper, *_ = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        prior_overrides=prior_overrides, fixed_parameter_values=fixed,
    )
    sampler_opts = SimpleNamespace(
        seed=seed, show_progress=False, nlive=40, dlogz=5.0,
        max_samples=0, dynesty_diagnostics=False, save_path=".",
    )
    out = run_sampler(
        "dynesty", likelihood, make_prior_transform(lower, upper),
        labels, lower, upper, sampler_opts,
    )
    return np.asarray(out["samples"]), float(out["logZ"])


def test_dynesty_same_seed_reproduces():
    s1, z1 = _run(17)
    s2, z2 = _run(17)
    assert z1 == z2
    assert s1.shape == s2.shape
    np.testing.assert_array_equal(s1, s2)


def test_dynesty_different_seed_differs():
    s1, _ = _run(17)
    s3, _ = _run(18)
    assert s1.shape != s3.shape or not np.array_equal(s1, s3)
