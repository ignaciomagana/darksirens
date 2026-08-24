"""Bundle-path f_p refusal (darksirens/likelihood/factory.py).

The bundle-source likelihood (any run with ``data["catalogs"]``, including
K=1 bundle sources) threads NONE of the per-pixel selection-fraction (f_p)
leaves into its EMCatalogs, while ``prepare_catalog_views`` happily consumes
a bundle's ``f_p_map`` into views that were then thrown away: a caller
supplying f_p on this path used to get the UNMASKED-footprint estimand
(aggregate Cbar applied to unobserved sky, the S-3 bias) with no error.
The factory's design contract is that unsupported operand combinations fail
at BUILD time (the dropped-operand guard), so f_p must fail there too --
both as a top-level ``data["f_p_map"]`` and as a bundle-carried key the
top-level guard cannot see.
"""
import numpy as np
import pytest

from darksirens.likelihood.factory import make_likelihood

from test_multitracer_likelihood import (
    APIX1,
    Z_A,
    Z_B,
    _base_opts,
    _bundle,
    _pop_bits,
    _shared_physics,
)


def _fp_map(nside=1, value=0.5):
    """A full-sky fractional-coverage map (nside 1 -> 12 pixels)."""
    return np.full(12 * nside * nside, value, dtype=np.float32)


def test_top_level_f_p_map_refused_on_bundle_path():
    """``data["f_p_map"]`` alongside ``data["catalogs"]`` must hit the
    dropped-operand guard by name, like counterparts / WL / strata."""
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A)]
    data["f_p_map"] = _fp_map()
    opts = _base_opts(n_catalogs=1)
    with pytest.raises(NotImplementedError, match="f_p_map"):
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


@pytest.mark.parametrize("bad_idx, n_catalogs", [(0, 1), (1, 2)])
def test_bundle_carried_f_p_map_refused(bad_idx, n_catalogs):
    """A bundle-carried ``f_p_map`` is invisible to the top-level guard but
    IS consumed by ``prepare_catalog_views`` into f_p views the bundle
    EMCatalogs never thread -- the per-bundle check must refuse it, naming
    the offending catalog."""
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    bundles = [_bundle(APIX1, z) for z in (Z_A, Z_B)[:n_catalogs]]
    bundles[bad_idx]["f_p_map"] = _fp_map()
    data["catalogs"] = bundles
    opts = _base_opts(n_catalogs=n_catalogs)
    with pytest.raises(
            NotImplementedError,
            match=rf"catalog {bad_idx + 1}: per-pixel selection fraction"):
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_bundle_path_without_f_p_still_builds():
    """Positive control: the guard must not over-fire -- an f_p-free bundle
    build succeeds exactly as before."""
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A)]
    opts = _base_opts(n_catalogs=1)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    assert callable(ll)
