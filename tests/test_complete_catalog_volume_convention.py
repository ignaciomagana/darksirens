"""Complete-catalog host prior must NOT double-count the comoving volume.

The catalog's galaxy counts already track candidate hosts per redshift
shell, so the complete-model prior on a catalog whose counts follow
dV_c/dz (a uniform-in-comoving-volume null, or any volume-limited catalog)
must reproduce the volume prior itself: the binned ratio
p_cat(z | pix) / (dV_c/dz) is flat in z.

Weighting each galaxy by g(z_i) = dV_c/dz(z_i) (1+z_i)^delta on top of the
counts (volume_weighted=True kernels) squares the volume element: the same
ratio rises as dV_c/dz (measured log-log slope ~ +0.94). This regression
test pins the unit-mass convention for ``dark_sirens_complete``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.redshift.volume import dV_of_z

N_ROWS = 40
N_PER_ROW = 400
Z_MAX = 0.30


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(
        n0=1.0,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
        sigma_kde=0.0,
        complete_empty_pixel_policy=0,
    )


def _uniform_in_volume_catalog(rng):
    """Rows of galaxies with z drawn from dV_c/dz truncated to [0, Z_MAX]."""
    zg = np.linspace(1e-4, Z_MAX, 2000)
    dv = np.asarray(dV_of_z(jnp.asarray(zg), 67.74, 0.3075, -1.0, 0.0))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    z = np.interp(rng.uniform(0, 1, (N_ROWS, N_PER_ROW)), cdf, zg)
    return EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(z),
        dzgals=jnp.full((N_ROWS, N_PER_ROW), 1e-4),
        wgals=jnp.ones((N_ROWS, N_PER_ROW)),
        ngals=jnp.full((N_ROWS,), N_PER_ROW, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 8)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _ratio_slope(em_catalog):
    """Log-log slope of [row-averaged p_cat / (dV_c/dz)] against dV_c/dz."""
    cosmo, survey = _cosmo(), _survey()
    state = prepare_redshift_prior_state("dark_sirens_complete", cosmo, survey, em_catalog)

    zq = jnp.linspace(0.02, 0.29, 720)
    p_mean = np.zeros(len(zq))
    for row in range(N_ROWS):
        lp = eval_redshift_prior_with_state(
            "dark_sirens_complete", state, zq,
            jnp.full(len(zq), row, dtype=jnp.int32), cosmo, survey, em_catalog)
        p_mean += np.exp(np.nan_to_num(np.asarray(lp), neginf=-745.0)) / N_ROWS

    bins = np.linspace(0.02, 0.29, 10)
    zc = 0.5 * (bins[1:] + bins[:-1])
    zq_np = np.asarray(zq)
    binned = np.array([p_mean[(zq_np >= bins[j]) & (zq_np < bins[j + 1])].mean()
                       for j in range(len(zc))])
    dv = np.asarray(dV_of_z(jnp.asarray(zc), 67.74, 0.3075, -1.0, 0.0))
    ratio = binned / dv
    x, y = np.log(dv), np.log(ratio / ratio[4])
    a = np.vstack([x - x.mean(), np.ones_like(x)]).T
    return float(np.linalg.lstsq(a, y, rcond=None)[0][0])


def test_complete_prior_tracks_volume_prior_on_uniform_catalog():
    rng = np.random.default_rng(20260710)
    slope = _ratio_slope(_uniform_in_volume_catalog(rng))
    # unit-mass kernels: flat ratio (|slope| ~ 0.01 measured); the volume-
    # weighted double-count fails this hard (slope ~ +0.94).
    assert abs(slope) < 0.1, (
        f"complete-catalog prior does not track dV_c/dz on a uniform-in-volume "
        f"catalog: log-log ratio slope {slope:+.3f} (double-counted volume?)"
    )


def test_complete_prior_depends_on_delta_with_photometric_redshifts():
    """``delta`` is an ACTIVE parameter of the complete-catalog likelihood.

    g(z) = dV_c/dz (1+z)^delta is the interim prior on each galaxy's true
    redshift: the kernels are divided by Z_i = integral N(z; z_i, sig_eff) g(z) dz
    and the evaluator reapplies g(z) as a front factor.  The two cancel only as
    sig_eff -> 0, so with photometric dzgals the prior tilts with delta -- which
    is why inference/prior.py must NOT declare delta inert for this model.
    """
    zg = np.array([[0.10, 0.30, 100.0], [0.20, 100.0, 100.0]])
    ng = np.array([2, 1], dtype=np.int32)
    wg = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cosmo = _cosmo()
    zq = jnp.asarray([0.10, 0.30])

    def _p(dz_value, delta):
        survey = _survey()._replace(delta=delta)
        cat = EMCatalog(
            apix=1.0,
            zgals=jnp.asarray(zg),
            dzgals=jnp.full((2, 3), dz_value),
            wgals=jnp.asarray(wg),
            ngals=jnp.asarray(ng),
            delta_g_pix_z=jnp.zeros((1, 8)),
            dN_obs_kde=None,
            pixel_to_cache_idx=None,
        )
        state = prepare_redshift_prior_state(
            "dark_sirens_complete", cosmo, survey, cat
        )
        return np.exp(np.asarray(eval_redshift_prior_with_state(
            "dark_sirens_complete", state, zq,
            jnp.zeros(2, dtype=jnp.int32), cosmo, survey, cat)))

    # Spectroscopic: g(z) cancels to <0.5% (effectively inert).
    spec = _p(1e-3, 3.0) / _p(1e-3, 0.0)
    assert np.all(np.abs(spec - 1.0) < 5e-3)
    # Photometric: a several-per-cent tilt that also VARIES across z, so it is
    # not absorbed by any normalisation.
    photo = _p(0.05, 3.0) / _p(0.05, 0.0)
    assert np.max(np.abs(photo - 1.0)) > 0.02
    assert abs(photo[0] - photo[1]) > 0.02
