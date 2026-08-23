"""PR-5 end-to-end pins: the latent seam inside the real GW likelihood.

``tests/test_latent_seam.py`` pins the seam kernel.  This module pins the
STACK: that swapping the Q source changes the numbers the likelihood produces
in exactly one way -- not at all, once the table is the one the seam would have
generated.  Three pins:

E1  **The decisive equivalence.**  ``lss_field_mode='latent'`` with
    ``--lss_marginalize`` equals the shipped table path fed the log-Q tables
    the seam generates, to floating point.  This exercises the whole latent
    stack at once -- the per-sample numerator gather, the per-member missing
    integrals ``N_miss_members``, and the normalizers built from them -- so a
    divergence anywhere in it fails here.  It is the latent analogue of
    ``test_marginalize_equals_logmeanexp_of_members``.

E2  **P12, inertness.**  A run with no latent leaves is bit-identical to the
    shipped one, and the mode string cannot disagree with the data: a "latent"
    run without leaves and a "table" run with them are both hard errors rather
    than silent fallbacks.

E3  **P16, the complete-catalog limit.**  At ``C == 1`` the missing branch
    vanishes and every member returns the same likelihood, so the member
    average collapses onto the single-member value and latent-on equals
    latent-off.  A physics identity (PLAN §1.6 Limit I), which is why it holds
    with a large field and a nonzero ``b_GW`` rather than only in a small-field
    limit.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import (
    C_MODE_AGGREGATE_STRUCT,
    CosmoParams,
    EMCatalog,
    GWEvent,
    SurveyParams,
)
from darksirens.gw.populations import get_fixed_population_params
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.likelihood.latent_q import footprint_row_map, on_footprint_mask
from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.redshift.latent_field import build_latent_basis, sky_moments
from darksirens.utils.cosmology import H0Planck, Om0Planck

NG = int(zgrid.size)
_Z = np.asarray(zgrid)
Z_DEPTH = 0.35
N_ROWS = 6          # catalog rows (pixels)
N_FIT = 4           # of which these are the fitted footprint
M_SPH, M_Z = 12, 5
M_DRAW = 3
N_B, B_MAX = 33, 4.0
B_GW = 1.4

COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))

#: ``b_miss`` IS ``b_GW`` in latent mode (PLAN §4.3 inverts the guard).  The
#: table-mode comparison runs consume the generated table, in which ``b_GW``
#: is already baked, so they set it to 0 -- if they did not, the legacy
#: local-overdensity factor would multiply the modulation a second time.
SURVEY_LATENT = SurveyParams(
    n0=1.0, z50=0.15, w=0.08, delta=0.0, b_miss=B_GW, alpha_miss=1.0,
    z_depth=Z_DEPTH, c_mode=C_MODE_AGGREGATE_STRUCT)
SURVEY_TABLE = SURVEY_LATENT._replace(b_miss=0.0)


# ------------------------------------------------------------------ fixtures

def _basis_and_field(seed=4):
    """The latent pieces: basis on the footprint rows, draws, moments."""
    below = _Z <= Z_DEPTH
    n_sub = int(below.sum())
    z_sub = _Z[:n_sub]

    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(N_FIT, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_DEPTH, ls_sph=0.7, ls_z=0.15)

    f_p = rng.uniform(0.6, 1.0, size=N_FIT)
    xi = rng.normal(size=(M_DRAW, M_SPH * M_Z)) * 0.5
    row_fac_fit = np.stack([
        np.asarray(basis.phi_sph @ x.reshape(M_SPH, M_Z)) for x in xi
    ]).astype(np.float32)

    k = np.arange(N_B)
    b_nodes = 0.5 * B_MAX * (1.0 - np.cos(np.pi * k / (N_B - 1)))
    A_sub, B_sub = sky_moments(basis, xi, b_nodes, f_p, row_fac=row_fac_fit)

    A = np.zeros((M_DRAW, N_B, NG)); A[:, :, :n_sub] = np.asarray(A_sub)
    B = np.zeros((M_DRAW, N_B, NG)); B[:, :, :n_sub] = np.asarray(B_sub)
    phi_z = np.zeros((NG, M_Z)); phi_z[:n_sub] = np.asarray(basis.phi_z_out)
    row_fac = np.concatenate(
        [row_fac_fit, np.zeros((M_DRAW, 1, M_Z), np.float32)], axis=1)
    return dict(row_fac=row_fac, phi_z=phi_z, A=A, B=B, b_nodes=b_nodes,
                f_p=f_p, P_F=float(N_FIT), F_F=float(f_p.sum()))


_FIELD = _basis_and_field()
#: Catalog rows 0..N_FIT-1 are the footprint; rows N_FIT.. are outside it, so
#: every run here exercises the off-footprint branch (pin P13b) as the
#: production run does for ~38% of its gathers.
_ROW_MAP = footprint_row_map(np.arange(N_ROWS), np.arange(N_FIT), N_FIT)
_ON_FP = np.asarray(on_footprint_mask(_ROW_MAP, N_FIT))


def _latent_leaves():
    return dict(
        latent_row_fac=jnp.asarray(_FIELD["row_fac"]),
        latent_phi_z=jnp.asarray(_FIELD["phi_z"]),
        latent_row_map=jnp.asarray(_ROW_MAP),
        latent_on_fp=jnp.asarray(_ON_FP),
        latent_A=jnp.asarray(_FIELD["A"]),
        latent_B=jnp.asarray(_FIELD["B"]),
        latent_b_nodes=jnp.asarray(_FIELD["b_nodes"]),
        latent_P_F=_FIELD["P_F"],
        latent_F_F=_FIELD["F_F"],
    )


def _f_p_rows():
    """``f_p`` on CATALOG rows: the footprint's own values, 0 outside it.

    Off-footprint rows have no selection fraction (no depth measured there), so
    ``C_p = f_p C = 0`` and the whole pixel is missing -- consistent with the
    seam's ``Q == 1`` convention there.
    """
    out = np.zeros(N_ROWS, dtype=np.float32)
    out[:N_FIT] = _FIELD["f_p"]
    return jnp.asarray(out)


def _catalog(*, latent=False, logq_members=None, dz=0.003):
    rng = np.random.default_rng(1)
    dz_w = dz
    nmax = 4
    zg = np.full((N_ROWS, nmax), 100.0)
    dz = np.full((N_ROWS, nmax), 1.0)
    w = np.zeros((N_ROWS, nmax))
    ng = np.zeros(N_ROWS, dtype=np.int32)
    for i in range(N_ROWS):
        n = 2 + (i % 3)
        zs = rng.uniform(0.05, 0.30, size=n)
        zg[i, :n] = zs
        dz[i, :n] = dz_w
        w[i, :n] = 1.0
        ng[i] = n
    zg, dz, w, ng = (jnp.asarray(a) for a in (zg, dz, w, ng))
    kde, idx = build_pixel_kde_cache(
        np.arange(N_ROWS, dtype=np.int32), zg, N_ROWS, ngals=ng)
    extra = _latent_leaves() if latent else {}
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((N_ROWS, NG)), dN_obs_kde=kde,
        pixel_to_cache_idx=idx, unique_pixels=None,
        f_p_rows=_f_p_rows(),
        lss_completion_logq_members=(
            None if logq_members is None else jnp.asarray(logq_members)),
        **extra,
    )


def _gw(n_events, n_samp, seed):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(420.0, 1400.0, total))
    chieff = jnp.asarray(rng.uniform(-0.2, 0.2, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.asarray(rng.integers(0, N_ROWS, total), dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=m2det / m1det,
                   valid=valid)


_N_EV, _N_SAMP, _N_SEL = 3, 48, 240
_GW_PE = _gw(_N_EV, _N_SAMP, seed=0)
_GW_SEL = _gw(_N_SEL, 1, seed=10)


def _ll(cat, survey, *, marginalize, field_mode="table", max_var=None):
    kw = {} if max_var is None else dict(max_likelihood_variance=max_var)
    return float(darksiren_log_likelihood(
        COSMO, survey, POP, _GW_PE, cat, _GW_SEL, cat,
        _N_EV, _N_SAMP, float(_N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=marginalize,
        lss_field_mode=field_mode, **kw,
    ))


# ------------------------------------------------ the generated table (ref)

def _generated_logq_members():
    """The ``(M, N_rows, N_grid)`` table the seam generates, built the SLOW way.

    Deliberately independent of ``completion.latent_member_logq_rows``: the
    completeness curve is rebuilt here from the same aggregate grids the
    likelihood uses, and ``rho`` from the moment definition rather than the
    shipped helper.  A pin that reused the implementation would only be
    testing that the code calls itself.
    """
    from darksirens.likelihood.latent_q import interp_b
    from darksirens.redshift.completion import _precompute_grids

    grids = _precompute_grids(COSMO, SURVEY_LATENT, _catalog(latent=True))
    C = np.asarray(jnp.clip(grids.C_bar_raw, 0.0, 1.0))
    below = _Z <= Z_DEPTH

    out = np.zeros((M_DRAW, N_ROWS, NG))
    for m in range(M_DRAW):
        A = np.asarray(interp_b(jnp.asarray(_FIELD["A"][m]),
                                jnp.asarray(_FIELD["b_nodes"]), B_GW))
        B = np.asarray(interp_b(jnp.asarray(_FIELD["B"][m]),
                                jnp.asarray(_FIELD["b_nodes"]), B_GW))
        rho = np.where(
            below,
            np.log(np.maximum(A - C * B, 1e-300))
            - np.log(np.maximum(_FIELD["P_F"] - C * _FIELD["F_F"], 1e-300)),
            0.0)
        f = _FIELD["row_fac"][m].astype(np.float64) @ _FIELD["phi_z"].T
        lq = B_GW * f - rho[None, :]
        lq = np.where(_ON_FP[:, None], lq[_ROW_MAP], 0.0)
        out[m] = lq
    return out


# ---------------------------------------------------------------------- E1

def test_latent_equals_the_table_it_generates():
    """THE pin: the latent stack == the table stack fed the seam's own table.

    Both sides marginalize over the same 3 members; the only difference is
    whether Q was read or generated.  Agreement to floating point means the
    substitution is confined to the seam -- numerator, per-member missing
    integrals and normalizers all follow.
    """
    ll_latent = _ll(_catalog(latent=True), SURVEY_LATENT,
                    marginalize=True, field_mode="latent")
    ll_table = _ll(_catalog(logq_members=_generated_logq_members()),
                   SURVEY_TABLE, marginalize=True, field_mode="table")
    assert np.isfinite(ll_latent)
    assert ll_latent == pytest.approx(ll_table, rel=0, abs=1e-8), (
        f"latent {ll_latent} != table-of-the-same-Q {ll_table}")


def test_the_field_actually_moves_the_likelihood():
    """Guards E1 against vacuity: a zero field must give a DIFFERENT answer.

    Without this, an implementation that silently returned ``Q == 1`` would
    pass E1 (both sides would be the same wrong number).
    """
    zeroed = _latent_leaves()
    zeroed["latent_row_fac"] = jnp.zeros_like(zeroed["latent_row_fac"])
    cat_zero = _catalog(latent=True)._replace(**zeroed)
    ll_field = _ll(_catalog(latent=True), SURVEY_LATENT,
                   marginalize=True, field_mode="latent")
    ll_zero = _ll(cat_zero, SURVEY_LATENT, marginalize=True,
                  field_mode="latent")
    assert abs(ll_field - ll_zero) > 1e-3, (
        "the latent field left the likelihood unchanged -- the seam is inert "
        f"when it should not be ({ll_field} vs {ll_zero})")


# ---------------------------------------------------------------------- E2

def test_p12_no_latent_leaves_is_the_shipped_path():
    """A catalog without latent leaves runs the shipped table path unchanged."""
    tab = _generated_logq_members()
    a = _ll(_catalog(logq_members=tab), SURVEY_TABLE, marginalize=True)
    b = _ll(_catalog(logq_members=tab), SURVEY_TABLE, marginalize=True,
            field_mode="table")
    assert a == b


def test_mode_string_cannot_disagree_with_the_data():
    """Neither half of the mismatch is allowed to pass silently."""
    with pytest.raises(ValueError, match="requires the latent-field leaves"):
        _ll(_catalog(logq_members=_generated_logq_members()), SURVEY_TABLE,
            marginalize=True, field_mode="latent")
    with pytest.raises(ValueError, match="lss_field_mode is"):
        _ll(_catalog(latent=True), SURVEY_LATENT, marginalize=True,
            field_mode="table")


def test_latent_refuses_a_loaded_q_table():
    """Two completion models at once is refused, not silently resolved."""
    cat = _catalog(latent=True)._replace(
        lss_completion_logq_members=jnp.asarray(_generated_logq_members()))
    with pytest.raises(NotImplementedError, match="refused"):
        _ll(cat, SURVEY_LATENT, marginalize=True, field_mode="latent")


def test_latent_refuses_per_pixel_c_mode():
    """PLAN guard 6 at the point of use: a per-pixel C does not factor through
    the sky moments, so the budget normalizer would be wrong."""
    survey = SURVEY_LATENT._replace(c_mode=None)
    with pytest.raises(NotImplementedError, match="aggregate/selection c_mode"):
        _ll(_catalog(latent=True), survey, marginalize=True,
            field_mode="latent")


# ---------------------------------------------------------------------- E3

def test_p16_complete_catalog_limit_at_the_likelihood_level():
    """PLAN §1.6 Limit I, in the likelihood: at ``C == 1`` the field drops out.

    ``C == 1`` with ``f_p == 1`` makes ``base_miss = (1 - C) dN_exp == 0``, the
    missing branch vanishes, and the prior collapses to the observed host
    spikes -- for EVERY member and every field, so the member average equals
    any single member's value exactly.  The field is not small (``b_GW = 1.4``
    on a real draw): this is the physics identity, not a small-field
    expansion.

    The WIDER host kernels (``dz = 0.02`` against the module default 0.003)
    are load-bearing for the same reason and are likewise a property of the
    limit, not of the seam: at ``C == 1`` the prior is pure host spikes, and at
    the default width 240 injections all miss them, so the selection integral
    ``mu`` underflows and the likelihood is ``-inf`` before the identity can be
    observed at all.  Measured: ``dz = 0.003`` gives ``-inf``; 0.02 gives
    0.861; 0.06 gives 0.493.  (The selection VARIANCE guard is not involved --
    raising ``max_likelihood_variance`` to 1e12 changes none of those numbers.)

    ``z_depth=None`` is load-bearing and was found the hard way.  With a finite
    depth, ``_assemble_curves`` relaxes ``dN_miss`` to the FULL ``dN_exp``
    above it regardless of ``C`` -- that is the "hosts there are missing, not
    nonexistent" convention -- so with the grid running to ``zMax`` the
    above-depth budget swamps ``Z`` by ~10 orders of magnitude and every
    event's prior underflows.  The first version of this test did exactly that
    and compared ``-inf`` to ``-inf`` on CPU while GPU float noise made one
    side finite.  Hence the finiteness assertions below: a degenerate
    configuration must fail this test loudly, never pass it vacuously.
    """
    from darksirens.redshift import completion as _c

    survey = SURVEY_LATENT._replace(z_depth=None)
    cat = _catalog(latent=True, dz=0.02)._replace(
        f_p_rows=jnp.ones(N_ROWS, dtype=jnp.float32))

    # Force Cbar == 1 at the single formation site, so the numerator and the
    # normalizer both see it (patching one would prove nothing -- PLAN §4.2's
    # "one budget from one formation site" is the property under test).
    real = _c._precompute_grids

    def _unit_C(cosmo, survey_, em_catalog):
        g = real(cosmo, survey_, em_catalog)
        return g._replace(C_bar_raw=jnp.ones_like(jnp.asarray(_Z)))

    _c._precompute_grids = _unit_C
    try:
        vals = {}
        for M in (M_DRAW, 1):
            sliced = {
                k: (v[:M] if k in ("latent_row_fac", "latent_A", "latent_B")
                    else v)
                for k, v in _latent_leaves().items()}
            vals[M] = _ll(cat._replace(**sliced), survey, marginalize=True,
                          field_mode="latent")
    finally:
        _c._precompute_grids = real

    # Non-vacuity FIRST: -inf == -inf would otherwise "prove" the identity.
    assert np.isfinite(vals[M_DRAW]) and np.isfinite(vals[1]), (
        "the C == 1 configuration went degenerate (-inf), so the identity "
        f"below would be vacuous: {vals}")
    assert vals[M_DRAW] == pytest.approx(vals[1], rel=0, abs=1e-9), (
        "at C == 1 the members must be indistinguishable; got "
        f"{vals[M_DRAW]} for M={M_DRAW} vs {vals[1]} for M=1")
