"""Deterministic clustered sparse-contrast K=2 mixture mock (in-memory).

A galaxy catalog (GAL, dense over all pixels) and an AGN catalog (sparse: one AGN
per occupied pixel, clustered into a contiguous ~20% of the sky) built from the
same redshift range, plus N GW events whose true hosts are drawn from the AGN
members with probability ``fcat_true`` (else from the galaxy field), PE dL clouds
at the true cosmology, and a detectability injection set.  Everything is returned
in the SAME compact-view / bundle contract as
``load_multitracer_catalog_bundles`` (see ``darksirens/likelihood/catalog_views.py``),
so ``make_likelihood`` consumes it identically to a real multitracer run -- the
tiny-fixture style of ``tests/test_multitracer_likelihood.py`` /
``tests/test_catalog_sky_weighting.py`` scaled up to a discriminating mock.

Design (why each knob is what it is)
------------------------------------
* GAL galaxy redshifts ~ (dV/dz)^``gal_alpha`` over ``[~0, z_depth]``.  The
  catalog kernel already carries a dV/dz volume factor, so ``gal_alpha`` ~ 0.35
  tunes the GAL per-pixel *conditional* prior to coincide with the empty-pixel
  (volume-measure) missing-galaxy prior: a GAL-hosted event is then INDIFFERENT
  between the two catalogs in conditional mode -- the campaign's ~-0.086
  logL/event -- which is exactly what lets the conditional scan rail to f=1.
* GAL-hosted events live in NON-AGN pixels (clean host-fraction geometry); AGN
  hosts sit at their pixel's sharp AGN redshift (a razor ``dz_agn`` kernel), so
  AGN-hosted events gain a large own-host KDE spike as f grows.
* ``log10n0`` is tiny so the AGN missing-galaxy budget is negligible: in the FIELD
  convention the empty-sky AGN probability is ~ 0 (the number-density channel that
  identifies the host fraction), while the conditional empty-pixel prior is the
  n0-independent volume prior (the shape n0 cancels).
* Injections are a detectable population: a GAL component (volume z over all-sky
  pixels) PLUS an AGN-host-like component (at AGN pixels/redshifts) with the exact
  mixture ``p_draw`` so the field selection integral is well sampled at every f --
  otherwise its effective sample size Neff crashes at high f (the campaign's latent
  Neff->1 issue) and the 5*N_obs guard carves a spurious cliff into the scan.

All randomness uses fixed, separated seeds (catalog / events / injections) so
catalog-shape knobs never reshuffle the host assignments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_field_normalization_inputs
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID
from darksirens.utils.cosmology import dL_of_z, dV_of_z, ddL_of_z

POP_MODEL = "powerlaw+peak"
POP_FID = get_fixed_population_params(POP_MODEL)
FCAT_TRUE = 0.30
H0_TRUE = H0_FID

# Nuisance scan point (shared by field and conditional): tiny log10n0 (AGN treated
# as complete -> clean field host fraction; conditional shape is n0-invariant) and
# a missing-galaxy depth matched to the galaxy support.
LOG10N0 = -11.0
Z_DEPTH = 0.35

_ZG = np.linspace(1e-3, 1.5, 4000)
_DV = np.asarray(dV_of_z(jnp.asarray(_ZG), H0_TRUE, OM0_FID, W0_FID, WA_FID))


def _power_sample(rng, n, z_lo, z_hi, alpha):
    """Sample z with density proportional to (dV/dz)^alpha over [z_lo, z_hi]."""
    m = (_ZG >= z_lo) & (_ZG <= z_hi)
    zz, pdf = _ZG[m], _DV[m] ** alpha
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, size=n), cdf, zz)


def _volume_sample(rng, n, z_lo, z_hi):
    return _power_sample(rng, n, z_lo, z_hi, 1.0)


@dataclass(frozen=True)
class ClusteredMock:
    """In-memory K=2 mock: shared physics + per-catalog compact bundles."""

    n_events: int
    nsamp: int
    n_sel: float
    apix: float
    _shared: dict
    _gal: tuple           # (full_z, full_n, full_w)
    _agn: tuple
    _compact: dict        # per-catalog compact PE/sel views + field inputs
    is_agn: np.ndarray    # (n_events,) host-type mask
    n_agn_host: int
    n_agn_pix: int
    npix: int
    sparsity: float

    def make_data(self, field: bool) -> dict:
        """Assemble the ``make_likelihood`` data dict for the given mode.

        ``field=True`` attaches the per-catalog survey-global field-normalization
        inputs (``field_dN_obs_s`` / ``field_n_empty`` / ``field_N_obs_total``); the
        conditional path omits them (bit-identical to the legacy per-pixel path).
        """
        data = dict(self._shared)
        data["catalogs"] = [
            self._bundle("gal", field),
            self._bundle("agn", field),
        ]
        return data

    def _bundle(self, which: str, field: bool) -> dict:
        b = dict(self._compact[which])
        if not field:
            for k in ("field_dN_obs_s", "field_n_empty", "field_N_obs_total"):
                b.pop(k, None)
        return b


def build_clustered_mock(
    nside: int = 8,
    g_gal: int = 45,
    g_agn: int = 1,
    agn_pix_frac: float = 0.20,
    n_events: int = 50,
    nsamp: int = 48,
    n_sel: int = 1500,
    fcat_true: float = FCAT_TRUE,
    z_depth: float = Z_DEPTH,
    ev_lo: float = 0.05,
    ev_hi: float = 0.30,
    sigma_dL_frac: float = 0.04,
    dz_gal: float = 0.08,
    dz_agn: float = 0.01,
    gal_alpha: float = 0.35,
    inj_agn_frac: float = 0.30,
    seed_cat: int = 1,
    seed_ev: int = 2,
    seed_sel: int = 3,
) -> ClusteredMock:
    """Build the deterministic clustered sparse-contrast K=2 mock (see module docstring)."""
    rc = np.random.default_rng(seed_cat)
    re = np.random.default_rng(seed_ev)
    rs = np.random.default_rng(seed_sel)
    npix = hp.nside2npix(nside)
    apix = hp.nside2pixarea(nside)
    n_agn_pix = max(1, int(round(agn_pix_frac * npix)))
    agn_pixels = np.arange(n_agn_pix, dtype=np.int32)          # contiguous block
    nonagn_pixels = np.arange(n_agn_pix, npix, dtype=np.int32)

    # ---- full-sky catalogs -------------------------------------------------
    gal_z = _power_sample(rc, npix * g_gal, 1e-3, z_depth, gal_alpha).reshape(npix, g_gal)
    gal_n = np.full(npix, g_gal, dtype=np.int32)
    gal_w = np.ones((npix, g_gal))
    agn_z = np.zeros((npix, g_agn))
    agn_n = np.zeros(npix, dtype=np.int32)
    agn_w = np.zeros((npix, g_agn))
    agn_z[agn_pixels] = _volume_sample(rc, n_agn_pix * g_agn, ev_lo, ev_hi).reshape(n_agn_pix, g_agn)
    agn_n[agn_pixels] = g_agn
    agn_w[agn_pixels] = 1.0

    # ---- events: hosts fcat_true AGN / rest galaxy -------------------------
    n_agn_host = int(round(fcat_true * n_events))
    is_agn = np.zeros(n_events, dtype=bool)
    is_agn[:n_agn_host] = True
    re.shuffle(is_agn)
    host_pix = np.empty(n_events, dtype=np.int32)
    host_z = np.empty(n_events, dtype=float)
    gal_host_z = _volume_sample(re, n_events - n_agn_host, ev_lo, ev_hi)
    gi = 0
    for i in range(n_events):
        if is_agn[i]:
            p = int(re.choice(agn_pixels))
            host_pix[i] = p
            host_z[i] = agn_z[p, int(re.integers(g_agn))]
        else:
            host_pix[i] = int(re.choice(nonagn_pixels))
            host_z[i] = gal_host_z[gi]
            gi += 1

    dL_true = np.asarray(dL_of_z(jnp.asarray(host_z), H0_TRUE, OM0_FID, W0_FID, WA_FID))
    dL_pe = np.clip(
        re.normal(dL_true[:, None], sigma_dL_frac * dL_true[:, None], size=(n_events, nsamp)),
        10.0, None,
    ).reshape(-1)
    pix_pe = np.repeat(host_pix, nsamp)
    m1_pe = np.full(n_events * nsamp, 35.0)
    m2_pe = np.full(n_events * nsamp, 28.0)
    chi_pe = np.zeros(n_events * nsamp)

    # ---- injection set: GAL-like + AGN-host-like, with exact mixture p_draw ---
    sig_inj = 0.5 * dz_agn
    n_inj_agn = int(round(inj_agn_frac * n_sel))
    n_inj_gal = n_sel - n_inj_agn
    z_gal_inj = _volume_sample(rs, n_inj_gal, ev_lo, ev_hi)
    pix_gal_inj = rs.integers(0, npix, size=n_inj_gal).astype(np.int32)
    ap = rs.choice(agn_pixels, size=n_inj_agn)
    z_agn_inj = np.clip(agn_z[ap, 0] + rs.normal(0.0, sig_inj, size=n_inj_agn), 1e-3, None)
    z_sel = np.concatenate([z_gal_inj, z_agn_inj])
    pix_sel = np.concatenate([pix_gal_inj, ap.astype(np.int32)])
    dL_sel = np.asarray(dL_of_z(jnp.asarray(z_sel), H0_TRUE, OM0_FID, W0_FID, WA_FID))

    zgn = np.linspace(ev_lo, ev_hi, 2000)
    vol_norm = float(np.trapezoid(
        np.asarray(dV_of_z(jnp.asarray(zgn), H0_TRUE, OM0_FID, W0_FID, WA_FID)), zgn))
    p_vol = np.asarray(dV_of_z(jnp.asarray(z_sel), H0_TRUE, OM0_FID, W0_FID, WA_FID)) / vol_norm
    p_vol = np.where((z_sel >= ev_lo) & (z_sel <= ev_hi), p_vol, 0.0)
    is_agn_pix = pix_sel < n_agn_pix
    z_at_pix = np.where(is_agn_pix, agn_z[np.clip(pix_sel, 0, n_agn_pix - 1), 0], 0.0)
    gauss = np.exp(-0.5 * ((z_sel - z_at_pix) / sig_inj) ** 2) / (np.sqrt(2 * np.pi) * sig_inj)
    q_z = (1.0 - inj_agn_frac) * p_vol / npix + inj_agn_frac * np.where(is_agn_pix, gauss / n_agn_pix, 0.0)
    ddL = np.asarray(ddL_of_z(jnp.asarray(z_sel), H0_TRUE, OM0_FID, W0_FID, WA_FID))
    p_draw = np.maximum(q_z / ddL, 1e-300)
    m1_sel = np.full(n_sel, 35.0)
    m2_sel = np.full(n_sel, 28.0)
    chi_sel = np.zeros(n_sel)

    shared = dict(
        nEvents=n_events, nsamp=nsamp, Ndraw=float(n_sel), apix=apix,
        m1det=jnp.asarray(m1_pe), m2det=jnp.asarray(m2_pe), dL=jnp.asarray(dL_pe),
        chieff=jnp.asarray(chi_pe), p_pe=jnp.ones(n_events * nsamp),
        m1detsels=jnp.asarray(m1_sel), m2detsels=jnp.asarray(m2_sel),
        dLsels=jnp.asarray(dL_sel), chieffsels=jnp.asarray(chi_sel), p_draw=jnp.asarray(p_draw),
    )

    def _compact(pixels_global, full_z, full_n, full_w, dz):
        uniq = np.unique(pixels_global).astype(np.int32)
        s2u = np.searchsorted(uniq, pixels_global).astype(np.int32)
        z_rows = full_z[uniq]
        return dict(
            unique_pixels=uniq, sample_to_unique=s2u,
            zgals=z_rows, dzgals=np.full_like(z_rows, dz),
            wgals=full_w[uniq], ngals=full_n[uniq].astype(np.int32),
        )

    def _compact_bundle(full_z, full_n, full_w, dz):
        pe = _compact(pix_pe, full_z, full_n, full_w, dz)
        sel = _compact(pix_sel, full_z, full_n, full_w, dz)
        fobs, ne, nobs, _occ = build_field_normalization_inputs(
            jnp.asarray(full_z), jnp.asarray(full_w), jnp.asarray(full_n))
        return dict(
            apix=apix, delta_g_pix_z=jnp.zeros((1, len(zgrid))),
            zgals_pe=pe["zgals"], dzgals_pe=pe["dzgals"], wgals_pe=pe["wgals"], ngals_pe=pe["ngals"],
            unique_pixels_pe=pe["unique_pixels"], sample_to_unique_pe=pe["sample_to_unique"],
            zgals_sel=sel["zgals"], dzgals_sel=sel["dzgals"], wgals_sel=sel["wgals"], ngals_sel=sel["ngals"],
            unique_pixels_sel=sel["unique_pixels"], sample_to_unique_sel=sel["sample_to_unique"],
            field_dN_obs_s=fobs, field_n_empty=float(ne), field_N_obs_total=float(nobs),
        )

    compact = {
        "gal": _compact_bundle(gal_z, gal_n, gal_w, dz_gal),
        "agn": _compact_bundle(agn_z, agn_n, agn_w, dz_agn),
    }
    return ClusteredMock(
        n_events=n_events, nsamp=nsamp, n_sel=float(n_sel), apix=apix,
        _shared=shared, _gal=(gal_z, gal_n, gal_w), _agn=(agn_z, agn_n, agn_w),
        _compact=compact, is_agn=is_agn, n_agn_host=n_agn_host, n_agn_pix=n_agn_pix,
        npix=npix, sparsity=float(gal_n.sum()) / max(int(agn_n.sum()), 1),
    )


# ---------------------------------------------------------------------------
# opts builders (the K=2 dark-siren likelihood configuration for the scans)
# ---------------------------------------------------------------------------

def scan_opts(catalog_sky_weighting: str, z_depth: float = Z_DEPTH) -> SimpleNamespace:
    """Opts for the fcat_2 scan: fixed cosmology/pop/survey, only ``fcat_2`` free."""
    return SimpleNamespace(
        pop_model=POP_MODEL, universe_model="dark_sirens", sel_batch_size=None,
        fix_cosmology=True, fix_population=True, fix_survey=True,
        complete_empty_pixel_policy="volume", bright_siren_sky_marginalized=False,
        n_catalogs=2, catalog_sky_weighting=catalog_sky_weighting,
        selection_neff_soft_guard=True, resolved_survey_z_depths=(z_depth, z_depth),
    )


def joint_opts(z_depth: float = Z_DEPTH) -> SimpleNamespace:
    """Opts for the joint (H0, fcat_2) scan: H0 free; Om0/w0/wa/pop/survey fixed."""
    return SimpleNamespace(
        pop_model=POP_MODEL, universe_model="dark_sirens", sel_batch_size=None,
        fix_cosmology=False, fix_population=True, fix_survey=True,
        fixed_parameter_values={"Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID},
        prior_overrides={"H0": [40.0, 100.0]},
        complete_empty_pixel_policy="volume", bright_siren_sky_marginalized=False,
        n_catalogs=2, catalog_sky_weighting="field",
        selection_neff_soft_guard=True, resolved_survey_z_depths=(z_depth, z_depth),
    )


def scan_fixed_values() -> dict:
    """``fixed_parameter_values`` for the fcat scan (both catalogs' log10n0)."""
    return {"log10n0": LOG10N0, "log10n0_c2": LOG10N0}


def joint_fixed_values() -> dict:
    """``fixed_parameter_values`` for the joint scan (cosmology nuisances + log10n0)."""
    return {"Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID,
            "log10n0": LOG10N0, "log10n0_c2": LOG10N0}
