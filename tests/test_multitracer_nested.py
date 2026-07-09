"""K-catalog multitracer mixture (n_catalogs=2) through the nested samplers
(dynesty, tinyns), mirroring test_numpyro_sampler.py's
``_small_multitracer_likelihood_inputs`` two-catalog fixture.  Both samplers
route the free ``fcat_2`` coordinate through ``make_prior_transform``'s
closed-form Beta(1, b) PPF branch (darksirens/inference/prior.py); NUTS uses
the equivalent ``dist.Beta`` branch of sampling.py's ``_site`` (covered in
test_numpyro_sampler.py) so all three samplers infer the same posterior
measure over fcat_2."""
from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood
from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler

from test_numpyro_sampler import _small_multitracer_likelihood_inputs


def _build(opts, data, prior_overrides, fixed_parameter_values):
    pop_params_fid = get_fixed_population_params(opts.pop_model)
    likelihood = make_likelihood(
        opts, data, pop_params_fid, fixed_parameter_values=fixed_parameter_values,
    )
    labels, lower, upper, *rest = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        prior_overrides=prior_overrides, fixed_parameter_values=fixed_parameter_values,
        universe_model=opts.universe_model, n_catalogs=opts.n_catalogs,
    )
    prior_kinds = rest[8]
    prior_transform = make_prior_transform(lower, upper, prior_kinds)
    return likelihood, labels, lower, upper, prior_kinds, prior_transform


def test_dynesty_multitracer_k2_fcat2_finite_logz_and_bounded_samples():
    pytest.importorskip("dynesty")
    opts, data, prior_overrides, fixed, sampled = _small_multitracer_likelihood_inputs()
    likelihood, labels, lower, upper, prior_kinds, prior_transform = _build(
        opts, data, prior_overrides, fixed
    )
    assert labels == [sampled, "fcat_2"]

    sampler_opts = SimpleNamespace(
        seed=11, show_progress=False, nlive=25, dlogz=5.0, max_samples=400,
        dynesty_diagnostics=False, save_path=".",
    )
    out = run_sampler(
        "dynesty", likelihood, prior_transform, labels, lower, upper, sampler_opts,
    )
    samples = np.asarray(out["samples"])
    assert np.isfinite(float(out["logZ"]))
    assert samples.shape[1] == 2
    fcat_2 = samples[:, labels.index("fcat_2")]
    assert np.all(np.isfinite(fcat_2))
    assert np.all(fcat_2 >= 0.0) and np.all(fcat_2 <= 1.0)


def test_tinyns_multitracer_k2_fcat2_finite_logz_and_bounded_samples():
    pytest.importorskip("tinyns")
    opts, data, prior_overrides, fixed, sampled = _small_multitracer_likelihood_inputs()
    likelihood, labels, lower, upper, prior_kinds, prior_transform = _build(
        opts, data, prior_overrides, fixed
    )
    assert labels == [sampled, "fcat_2"]

    sampler_opts = SimpleNamespace(
        seed=11, show_progress=False, nlive=25, dlogz=5.0, max_samples=400,
        tinyns_preset="recommended",
    )
    out = run_sampler(
        "tinyns", likelihood, prior_transform, labels, lower, upper, sampler_opts,
    )
    samples = np.asarray(out["samples"])
    assert np.isfinite(float(out["logZ"]))
    assert samples.shape[1] == 2
    fcat_2 = samples[:, labels.index("fcat_2")]
    assert np.all(np.isfinite(fcat_2))
    assert np.all(fcat_2 >= 0.0) and np.all(fcat_2 <= 1.0)
