"""The catalog-prior jits must take the comoving-distance table as a PARAMETER.

``log_catalog_prior`` reaches the 106.8 MB table through
``log_galaxy_measure_grid`` -> ``dV_of_z`` -> ``r_of_z``.  It was declared with a
plain ``@jit`` (and ``log_catalog_prior_vmap`` as ``jit(vmap(...))`` on top of
it), which the repository's own rule in ``utils.cosmology`` forbids.  Two
MEASURED consequences on jax 0.4.34, both reproduced before this test existed:

* lowering the scalar function serialised 229.7 MB of module text -- the table
  as a ``dense<>`` HLO constant instead of a parameter, rebuilt, re-serialised
  and re-parsed by XLA on every compilation;
* in a cold process the public legacy bright-siren path succeeded for ``z``
  shape ``(1,)`` and then raised ``UnexpectedTracerError`` for shape ``(2,)``,
  because the plain jit closed over the enclosing ``threads_distance_table``
  boundary's tracer and JAX replayed the cached jaxpr from that dead trace.

The respecialization test runs in a COLD SUBPROCESS: once any of these jits has
been traced in the pytest process the caches are warm and the failure hides.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens.utils.cosmology as _cosmo
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift.catalog import log_catalog_prior, log_catalog_prior_vmap


REPO_ROOT = Path(__file__).resolve().parents[1]

#: A ``dense<>`` literal of the table costs at LEAST this much module text
#: (~8 bytes per float64 element as a hex blob) -- see
#: tests/test_cosmology_interpolation.py, which owns this bound.
_ONE_EMBEDDING_CHARS = 8 * int(np.prod(_cosmo.rs.shape))


def _dense_literal_lengths(module_text):
    return [len(m) for m in re.findall(r"dense<[^>]*>", module_text)]


def _cosmo_params():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(
        n0=1.0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
        complete_empty_pixel_policy=0,
    )


def _catalog():
    return EMCatalog(
        apix=hp.nside2pixarea(2),
        zgals=jnp.array([[0.2], [0.0]]),
        dzgals=jnp.array([[0.01], [1.0]]),
        wgals=jnp.array([[1.0], [0.0]]),
        ngals=jnp.array([1, 0], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=jnp.array([7, 8], dtype=jnp.int32),
        counterpart_pixel=7,
        bright_siren_sky_marginalized=False,
    )


@pytest.mark.parametrize("vectorised", [False, True])
def test_catalog_prior_takes_the_distance_table_as_a_parameter(vectorised):
    """Neither boundary may embed the table as a ``dense<>`` constant."""
    if vectorised:
        fn = log_catalog_prior_vmap
        z = jnp.array([0.2, 0.2])
        pix = jnp.array([0, 1], dtype=jnp.int32)
    else:
        fn = log_catalog_prior
        z = jnp.float64(0.2)
        pix = jnp.int32(0)

    text = fn.jitted.lower(
        z, pix, _cosmo_params(), _survey(), _catalog(), distance_table=_cosmo.rs
    ).as_text()

    assert max(_dense_literal_lengths(text), default=0) < _ONE_EMBEDDING_CHARS
    assert len(text) < _ONE_EMBEDDING_CHARS // 100, len(text)
    # ... and it is declared on @main, i.e. genuinely a parameter.
    assert "21x41x31x500xf64" in text.split("\n")[1]


def test_scalar_and_vector_boundaries_agree():
    """Splitting the shared body out must not have changed any number."""
    z = jnp.array([0.2, 0.35])
    pix = jnp.array([0, 1], dtype=jnp.int32)
    vector = log_catalog_prior_vmap(z, pix, _cosmo_params(), _survey(), _catalog())
    scalar = jnp.stack([
        log_catalog_prior(z[i], pix[i], _cosmo_params(), _survey(), _catalog())
        for i in range(2)
    ])
    np.testing.assert_array_equal(np.asarray(vector), np.asarray(scalar))


_COLD_RESPECIALIZATION = textwrap.dedent(
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import healpy as hp
    import jax.numpy as jnp
    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.redshift.prior import _log_prior_bright_sirens

    cosmo = CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    survey = SurveyParams(n0=1.0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.5, complete_empty_pixel_policy=0)
    catalog = EMCatalog(
        apix=hp.nside2pixarea(2),
        zgals=jnp.array([[0.2], [0.0]]),
        dzgals=jnp.array([[0.01], [1.0]]),
        wgals=jnp.array([[1.0], [0.0]]),
        ngals=jnp.array([1, 0], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=jnp.array([7, 8], dtype=jnp.int32),
        counterpart_pixel=7,
        bright_siren_sky_marginalized=False,
    )

    # Shape (1,) first, then (2,) in the SAME process: the second call is the
    # one that used to raise UnexpectedTracerError.
    for n in (1, 2):
        z = jnp.full((n,), 0.2)
        pix = jnp.arange(n, dtype=jnp.int32)
        out = _log_prior_bright_sirens(z, pix, cosmo, survey, catalog)
        assert out.shape == (n,), out.shape
    print("OK")
    """
)


def test_repeated_shape_specialization_in_a_cold_process():
    """The public legacy bright-siren path, two shapes, one fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", _COLD_RESPECIALIZATION],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "PYTHONPATH": str(REPO_ROOT),
            "JAX_PLATFORMS": "cpu",
        },
    )
    assert result.returncode == 0, (
        "second shape specialization failed:\n"
        + result.stdout[-4000:]
        + "\n"
        + result.stderr[-4000:]
    )
    assert "OK" in result.stdout
