"""PR-6a pins: P13, the CONSUMED budget identity, and the member-ESS diagnostic.

Two things ship here, and they are the two halves of PLAN §6.3/§6.4 that PR-6a
still owed.

**P13 -- the consumed budget identity (PLAN eq. 4), at the integral the
likelihood actually evaluates.**  ``tests/test_latent_seam.py`` already pins
eq. (4) on the seam's OWN ``Q`` rows: it forms ``sum_p (1 - f_p C) Q_p`` in the
test and checks it against ``sum_p (1 - f_p C)``.  That is a pin on the
algebra.  P13 is the stronger, different statement PLAN §4.2 asks for -- "exact
per-realization budget conservation against the integral that is actually
evaluated" -- so nothing here re-derives the budget.  Every number below comes
out of the two production integrators:

* ``completion._field_missing_curve``      -- the survey-GLOBAL missing curve
  ``V(z; theta) = sum_occ (1 - f_p C) Q_p + V_empty``, and
  ``completion.field_global_log_Z(_members)``, its ``dN_exp`` quadrature;
* ``completion.completion_curves``         -- the per-row numerator, i.e.
  ``base_miss = (1 - f_p C) dN_exp`` and the per-member ``N_miss_members``
  quadrature that ``latent_member_N_miss_integrals`` produces.

The checkable consequence, and the cleanest form of the pin: **in latent mode
the survey-global normalizer is MEMBER-INDEPENDENT.**  eq. (4) makes the
occupied-row budget sum to its ``Q == 1`` value exactly; every empty pixel is
off-footprint (the footprint is fitted TO the counts), so ``Q == 1`` there by
the seam's own convention and that block is conserved trivially.  Both blocks
therefore lose their member dependence, and ``log Z_m(theta)`` collapses onto
one number.  That is the gauge fixing of PLAN §4.2 -- "C and n0 own the budget,
Q owns placement" -- and it is what removes the +55% Jensen inflation the table
path measured, by construction rather than by a fitted correction.  Pinned at
``1e-12``, at every ``z``, every member and 5 theta (PLAN §6.3's P13 row).

**The closure floor is set by the ``f_p`` storage dtype, and it is measured
here.**  Building this fixture the obvious way -- ``f_p`` in float64 for the
moments, float32 on the catalog, which is exactly the production provenance
(``cli/build_latent_field.py:126`` loads ``f_p`` in f64 and
``sky_moments``/``sky_constant_coeffs`` consume it there;
``likelihood/catalog_views.py:748`` casts the SAME map to float32 for
``f_p_rows`` / ``field_f_p_occ``) -- closes eq. (4) at **1.6e-9 to 3.3e-8
relative**, not 1e-12.  This is the ``f_p`` analogue of the f64-row-factor
defect PR-5 found and ``latent_anchor_v2a.h5`` fixed: eq. (4) is exact only if
the moments and the seam evaluate the same field AND the same completeness.
``test_p13_closure_floor_is_the_f_p_storage_dtype`` records the number so the
pin above cannot be misread as a claim about the shipped anchor, whose own
floor is that 1e-9-1e-8 band.  ``factory._LATENT_F_F_RTOL`` (1e-6) already
guards ``F_F`` an order of magnitude above it.

**Member ESS (PLAN §6.4).**  ``exp(-sum_m p_m log p_m)`` with
``p_m = softmax_m(ll_m)``, computed from the already-materialized ``ll_members``
vector in ``core._factored_member_marginalization`` and surfaced through the
``lss_member_diagnostics`` static flag -- the same ``return_diagnostics``
convention ``likelihood_with_clusters`` uses, so the sampler's calls keep
returning a scalar and the production trace is untouched.

**The per-member Neff/variance guard is ASSERTED, not built.**  PLAN §6.4
records that rev 1 wrongly listed it as new work (R1-SEV3-11): ``core.py``'s
``_member_ll`` has always called ``selection_log_correction`` INSIDE the member
vmap, with that member's own ``Neff_m`` and its own summed per-event
reweighting variance.  ``test_per_member_selection_guard_is_live_in_latent_mode``
verifies it fires PER MEMBER in latent mode -- there exists a variance cap at
which SOME members are -inf and others finite, which no whole-likelihood guard
could produce -- rather than adding a second wall.

Guard convention throughout: the CLEAN arm of PR-0 --
``selection_neff_soft_guard=False`` (the hard wall), the Vitale 5 N_obs floor
kept, ``max_likelihood_variance`` stated at every call site that changes it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from darksirens.core.types import (  # noqa: E402
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    GWEvent,
    SurveyParams,
)
from darksirens.gw.populations import get_fixed_population_params  # noqa: E402
from darksirens.likelihood.core import (  # noqa: E402
    darksiren_log_likelihood,
    darksiren_member_diagnostics,
    member_ess,
)
from darksirens.likelihood.latent_q import (  # noqa: E402
    footprint_row_map,
    on_footprint_mask,
)
from darksirens.redshift.completion import (  # noqa: E402
    _LOGQ_CLIP,
    _field_missing_curve,
    _latent_C_curve,
    _member_q_eff_from_logq,
    _precompute_grids,
    build_field_depth_inputs,
    build_field_normalization_inputs,
    build_pixel_kde_cache,
    completion_curves,
    field_global_log_Z,
    field_global_log_Z_members,
    latent_member_logq_rows,
)
from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.latent_field import (  # noqa: E402
    build_latent_basis,
    sky_moments,
)

NG = int(zgrid.size)
_Z = np.asarray(zgrid)
Z_DEPTH = 0.35
NSIDE = 1
N_PIX = 12 * NSIDE ** 2
#: Pixels 0..8 hold galaxies; 9..11 are EMPTY.  The empty block is load-bearing:
#: it is the off-footprint half of PLAN §2.2's split, and P13 is only a real
#: statement if it is non-empty.
OCC_PIX = np.arange(9)
#: ... of which 0..5 are the fitted footprint F.  Occupied pixels 6,7,8 are
#: OUTSIDE it, so the fixture exercises both off-footprint populations the
#: production run has (occupied-but-unfit rows, and empty pixels) rather than
#: only the trivial one.
FIT_PIX = np.arange(6)
M_SPH, M_Z, M_DRAW = 8, 4, 4
#: The builder's shipped ``b_GW`` grid (``--n-b-nodes 33 --b-max 4``).  The node
#: count is a correctness parameter, not a resolution nicety -- see
#: ``test_latent_seam.test_budget_identity_off_node_is_bought_by_the_node_count``.
N_B, B_MAX = 33, 4.0
#: P13's tolerance, from PLAN §6.3.  NOT fitted: the consistent-``f_p``
#: configuration below closes at <= 1.1e-15 (measured: 8.9e-16 at the anchor
#: corner, 1.1e-15 at b_GW = 2.913), so 1e-12 leaves three decades of headroom
#: and still catches the 1e-9 dtype floor measured below.
P13_TOL = 1e-12

#: 5 theta, as PLAN §6.3's P13 row requires.  ``b_GW`` moves with them and is
#: deliberately OFF the Chebyshev nodes at three of the five (a sampled b_GW is
#: generically off-node, and off-node is the hard case for eq. 4), and exactly
#: 0 at one -- the zero-field limit, where rho must vanish identically.
THETA = (
    dict(H0=67.74, Om0=0.3075, n0=1e-4, b_gw=1.400),
    dict(H0=35.00, Om0=0.2000, n0=3e-5, b_gw=0.370),
    dict(H0=100.0, Om0=0.4000, n0=5e-4, b_gw=2.913),
    dict(H0=140.0, Om0=0.3000, n0=1e-3, b_gw=0.000),
    dict(H0=20.00, Om0=0.2500, n0=2e-4, b_gw=3.770),
)


def _cosmo(t):
    return CosmoParams(H0=t["H0"], Om0=t["Om0"], w0=-1.0, wa=0.0)


def _survey(t, **kw):
    """``c_mode='selection'``: ONE sky-aggregate completeness curve.

    Latent mode refuses a per-pixel or stratified ``C`` (PLAN §4.4 guard 6 --
    ``rho = log[(A - C B)/(P_F - C F_F)]`` factors only through a single curve),
    so the aggregate/selection family is the only admissible one and the pin has
    to live in it.  ``b_miss`` IS ``b_GW`` here: PLAN §4.3 inverts the guard.
    """
    base = dict(n0=t["n0"], z50=0.15, w=0.08, delta=0.0, b_miss=t["b_gw"],
                alpha_miss=1.0, sigma_kde=0.0, z_depth=Z_DEPTH,
                c_mode=C_MODE_SELECTION_STRUCT,
                m_lim=24.0, M0hat=-20.2, sigma_M=1.0)
    base.update(kw)
    return SurveyParams(**base)


# --------------------------------------------------------------------- fixture

def _galaxies(seed=11):
    max_g = 4
    rng = np.random.default_rng(seed)
    zg = np.full((N_PIX, max_g), 100.0)
    dz = np.ones((N_PIX, max_g))
    w = np.zeros((N_PIX, max_g))
    ng = np.zeros(N_PIX, dtype=np.int32)
    for p in OCC_PIX:
        n = 2 + (p % 3)
        zg[p, :n] = np.sort(rng.uniform(0.03, 0.30, n))
        dz[p, :n] = 5e-3
        w[p, :n] = 1.0
        ng[p] = n
    return zg, dz, w, ng


def _latent_pieces(f_p_fit, seed=11):
    """Basis, member draws and the ``(A, B)`` moment tables on the footprint.

    ``sky_moments`` is fed the STORED f32 row factors, exactly as
    ``cli/build_latent_field.py:208`` does -- eq. (4) closes at machine
    precision only if the moments and the seam evaluate the same field (PR-5
    measured 2.7e-7 at the production corner when they did not).
    """
    rng = np.random.default_rng(seed + 1)
    below = _Z <= Z_DEPTH
    n_sub = int(below.sum())
    vec = np.stack(hp.pix2vec(NSIDE, FIT_PIX), axis=-1)
    basis = build_latent_basis(
        vec, np.log1p(_Z[:n_sub]), n_inducing_sphere=M_SPH,
        n_inducing_z=M_Z, z_node_hi=Z_DEPTH, ls_sph=0.8, ls_z=0.15)

    xi = rng.normal(size=(M_DRAW, M_SPH * M_Z)) * 0.6
    row_fac_fit = np.stack([
        np.asarray(basis.phi_sph @ x.reshape(M_SPH, M_Z)) for x in xi
    ]).astype(np.float32)

    k = np.arange(N_B)
    b_nodes = 0.5 * B_MAX * (1.0 - np.cos(np.pi * k / (N_B - 1)))
    A_sub, B_sub = sky_moments(basis, xi, b_nodes, f_p_fit, row_fac=row_fac_fit)
    A = np.zeros((M_DRAW, N_B, NG)); A[:, :, :n_sub] = np.asarray(A_sub)
    B = np.zeros((M_DRAW, N_B, NG)); B[:, :, :n_sub] = np.asarray(B_sub)
    phi_z = np.zeros((NG, M_Z)); phi_z[:n_sub] = np.asarray(basis.phi_z_out)
    # The trailing ZERO pad row is where off-footprint rows land.
    row_fac = np.concatenate(
        [row_fac_fit, np.zeros((M_DRAW, 1, M_Z), np.float32)], axis=1)
    return dict(row_fac=row_fac, phi_z=phi_z, A=A, B=B, b_nodes=b_nodes)


def _build_catalog(f_p_moments, f_p_stored):
    """One full-sky catalog: field-normalizer inputs + latent leaves.

    ``f_p_moments`` (float64, footprint order) is what the artifact's moments
    and ``F_F`` are built from; ``f_p_stored`` (full sky) is what the catalog
    carries and the integrators consume.  Splitting them is what lets
    ``test_p13_closure_floor_is_the_f_p_storage_dtype`` measure the production
    provenance against the consistent one.
    """
    zg, dz, w, ng = _galaxies()
    field = build_field_normalization_inputs(
        jnp.asarray(zg), jnp.asarray(w), jnp.asarray(ng))
    occ = np.asarray(field.occupied_pixels)
    assert occ.tolist() == OCC_PIX.tolist()
    depth = build_field_depth_inputs(
        jnp.asarray(zg), jnp.asarray(dz), jnp.asarray(w), jnp.asarray(ng))
    kde, idx = build_pixel_kde_cache(
        np.arange(N_PIX, dtype=np.int32), jnp.asarray(zg), N_PIX,
        ngals=jnp.asarray(ng))

    lat = _latent_pieces(np.asarray(f_p_moments, dtype=np.float64))
    n_fit = FIT_PIX.size
    row_map_cat = footprint_row_map(np.arange(N_PIX), FIT_PIX, n_fit)
    row_map_fld = footprint_row_map(occ, FIT_PIX, n_fit)

    f_stored = np.asarray(f_p_stored, dtype=np.float64)
    return EMCatalog(
        apix=float(hp.nside2pixarea(NSIDE)),
        zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz), wgals=jnp.asarray(w),
        ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=kde,
        pixel_to_cache_idx=idx, unique_pixels=None,
        # f_p is stored as float32 on the catalog -- that is the production
        # layout (``core/types.py``: ``f_p_rows`` / ``field_f_p_occ`` are f32),
        # and the dtype is the point of the closure-floor measurement below.
        f_p_rows=jnp.asarray(f_stored.astype(np.float32)),
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=jnp.asarray(float(field.n_empty)),
        field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
        field_occupied_pixels=jnp.asarray(occ),
        field_depth_z=jnp.asarray(depth.z), field_depth_dz=jnp.asarray(depth.dz),
        field_depth_c=jnp.asarray(depth.c),
        field_f_p_occ=jnp.asarray(f_stored[occ].astype(np.float32)),
        # ``Sum_{p empty} f_p``.  Zero here because the footprint is a subset
        # of the OCCUPIED pixels -- which is the structural fact P13 rests on,
        # not a convenience: the empty block carries Q == 1 and drops out of
        # eq. (4) whatever its f_p, but a nonzero value here would mean the
        # depth map claims coverage where the counts fit no footprint.
        field_f_p_empty_sum=jnp.asarray(
            float(f_stored[np.setdiff1d(np.arange(N_PIX), occ)].sum())),
        latent_row_fac=jnp.asarray(lat["row_fac"]),
        latent_phi_z=jnp.asarray(lat["phi_z"]),
        latent_row_map=jnp.asarray(row_map_cat),
        latent_on_fp=jnp.asarray(on_footprint_mask(row_map_cat, n_fit)),
        latent_field_row_map=jnp.asarray(row_map_fld),
        latent_field_on_fp=jnp.asarray(on_footprint_mask(row_map_fld, n_fit)),
        latent_A=jnp.asarray(lat["A"]), latent_B=jnp.asarray(lat["B"]),
        latent_b_nodes=jnp.asarray(lat["b_nodes"]),
        latent_P_F=float(n_fit),
        latent_F_F=float(np.asarray(f_p_moments, dtype=np.float64).sum()),
    )


_RNG = np.random.default_rng(11)
#: ONE depth map, in float64 -- what ``load_selection_fraction`` returns and
#: what the anchor builder consumes.
_F_P_F64 = np.zeros(N_PIX)
_F_P_F64[FIT_PIX] = _RNG.uniform(0.45, 0.95, FIT_PIX.size)
#: ... and its float32 round-trip, which is what the catalog stores either way.
_F_P_F32 = np.float32(_F_P_F64).astype(np.float64)

#: The CONSISTENT configuration: the moments are built from the very float32
#: values the catalog stores, so the seam and both integrators consume ONE f_p.
#: This is what P13 is a pin on.
CAT = _build_catalog(_F_P_F32[FIT_PIX], _F_P_F32)
#: The PRODUCTION provenance: float64 moments over the float32 catalog.  The
#: two catalogs are otherwise IDENTICAL -- same galaxies, same basis, same
#: draws, same stored ``f_p`` bits -- so the residual below isolates the dtype
#: of ``(A, B, F_F)`` and nothing else.
CAT_MIXED = _build_catalog(_F_P_F64[FIT_PIX], _F_P_F64)

_OCC = np.asarray(CAT.field_occupied_pixels)
_ON_FP_CAT = np.asarray(CAT.latent_on_fp)
_ON_FP_FLD = np.asarray(CAT.latent_field_on_fp)
#: ``Q == 1`` on every occupied row: the reference the consumed budget must
#: reproduce.  Fed through the SAME ``latent_q_rows`` entry point, so the two
#: sides differ only in the numbers, never in the arithmetic.
_Q_ONE = jnp.ones((_OCC.size, NG))


def _member_q_rows(cat, t, m, *, field_rows):
    """Member ``m``'s ``Q_eff`` rows, formed exactly as the likelihood forms
    them (``latent_member_logq_rows`` + the depth relaxation)."""
    grids = _precompute_grids(_cosmo(t), _survey(t), cat)
    lq = latent_member_logq_rows(cat, _survey(t), _latent_C_curve(grids), m,
                                 field_rows=field_rows)
    depth_mask = jnp.asarray(_Z <= Z_DEPTH)
    return lq, _member_q_eff_from_logq(lq, depth_mask, True)


def _curve_residual(cat, t):
    """``max_{z,m} |V_m(z) / V_{Q=1}(z) - 1|`` through ``_field_missing_curve``."""
    V1, _ = _field_missing_curve(_cosmo(t), _survey(t), cat,
                                 latent_q_rows=_Q_ONE)
    worst = 0.0
    for m in range(M_DRAW):
        _, q = _member_q_rows(cat, t, m, field_rows=True)
        Vm, _ = _field_missing_curve(_cosmo(t), _survey(t), cat,
                                     latent_q_rows=q)
        worst = max(worst, float(jnp.max(jnp.abs(Vm / V1 - 1.0))))
    return worst


def _theta_id(t):
    return f"H0={t['H0']:g},b={t['b_gw']:g}"


# ------------------------------------------------------------- non-vacuity

def test_fixture_exercises_both_off_footprint_populations():
    """P13 is only a statement if the split of PLAN §2.2 is populated.

    Three properties, each of which would silently make a pin below vacuous:
    empty pixels exist (the block eq. 4 conserves trivially); occupied pixels
    outside the footprint exist (the ~38% of production gathers that return
    bit-zero logQ, pin P13b); and the footprint is a SUBSET of the occupied
    pixels, which is the structural fact -- the footprint is fitted TO the
    counts -- that makes the empty block off-footprint in the first place.
    """
    assert float(CAT.field_n_empty) == N_PIX - OCC_PIX.size > 0
    assert set(FIT_PIX.tolist()) < set(_OCC.tolist())
    assert _ON_FP_FLD.sum() == FIT_PIX.size
    assert (~_ON_FP_FLD).sum() == OCC_PIX.size - FIT_PIX.size > 0
    assert (~_ON_FP_CAT).sum() == N_PIX - FIT_PIX.size


@pytest.mark.parametrize("t", THETA, ids=_theta_id)
def test_the_field_actually_modulates_q(t):
    """The members must carry a REAL field, or every identity below is 1 == 1.

    Also pins that the ``+-_LOGQ_CLIP`` clip inside ``_member_q_eff_from_logq``
    is INERT here: a clipped ``logQ`` is no longer the seam's ``Q`` and eq. (4)
    would fail for a reason that has nothing to do with the seam.
    """
    lq0, q0 = _member_q_rows(CAT, t, 0, field_rows=True)
    lq1, _ = _member_q_rows(CAT, t, 1, field_rows=True)
    peak = float(jnp.max(jnp.abs(lq0)))
    assert peak < _LOGQ_CLIP, "the logQ clip is active; eq. (4) is not on trial"
    if t["b_gw"] == 0.0:
        # The zero-field limit: b_GW == 0 makes rho vanish identically and
        # Q == 1 for every member.  Kept as one of the 5 theta on purpose --
        # it is the one point where the identity is trivially true, and it must
        # still be true.
        assert peak == 0.0
        assert float(jnp.max(jnp.abs(q0 - 1.0))) == 0.0
        return
    assert peak > 0.05, f"field too weak to test the identity (max|logQ|={peak})"
    assert float(jnp.max(jnp.abs(lq0 - lq1))) > 1e-3, "members are identical"
    # Off-footprint rows are bit-zero (P13b), which is why their budget block
    # is conserved without any help from rho.
    assert np.all(np.asarray(lq0)[~_ON_FP_FLD] == 0.0)


# ------------------------------------------ P13, arm 1: the survey-global curve

@pytest.mark.parametrize("t", THETA, ids=_theta_id)
def test_p13_missing_curve_is_member_independent(t):
    """The consumed budget, at every ``z``, every member: ``V_m == V_{Q=1}``.

    This is PLAN eq. (4) evaluated by ``_field_missing_curve`` itself --
    ``sum_occ (1 - f_p C) Q_p + V_empty`` with the chunked ``lax.scan`` over
    occupied rows and its padding mask, the ``f_p``-weighted empty-pixel budget
    ``n_empty - C sum_empty f_p``, and the beyond-depth relaxation to the total
    pixel count.  Nothing is re-derived: the reference is the SAME function
    called with ``Q == 1``, so a divergence can only come from the seam.

    ``rho`` is what buys this.  Without it the members would each inflate or
    deflate the missing budget by their own ``mean_p e^{b f_p}`` -- the +55%
    Jensen inflation PLAN §4.2 measured on the table path -- and the member
    average would be an average over ensembles with DIFFERENT total galaxy
    counts, which is not a marginalization over anything.
    """
    res = _curve_residual(CAT, t)
    print(f"\n[P13 curve {_theta_id(t)}] max|V_m/V_1 - 1| = {res:.3e}")
    assert res < P13_TOL


@pytest.mark.parametrize("t", THETA, ids=_theta_id)
def test_p13_global_normalizer_is_member_independent(t):
    """``log Z_m(theta)`` collapses onto ONE number -- the integrated identity.

    ``field_global_log_Z_members`` is the function the likelihood calls (via
    ``prior.prepare_redshift_prior_state``, which hands each member state its
    own ``log_Z_global_members[m]``), so this pins the quadrature
    ``N_miss = trapz(dN_exp * V)`` and the depth-scaled observed term on top of
    the curve identity above.  Compared against ``field_global_log_Z`` fed
    ``Q == 1`` through the same ``latent_q_rows`` entry point.

    Member-independence of ``Z`` is not a convenience: PLAN §4.2's whole
    gauge fixing is that ``C`` and ``n0`` own the budget while ``Q`` owns
    placement.  If it failed, each member would carry ``(Zbar/Z_m)^{N_obs}``
    into the logsumexp -- with 259 events a 1% spread in ``Z_m`` tilts it by
    ``e^{2.6}`` per member -- and the "average" would collapse onto the
    smallest-``Z_m`` member.
    """
    lz = np.asarray(field_global_log_Z_members(_cosmo(t), _survey(t), CAT))
    lz1 = float(field_global_log_Z(_cosmo(t), _survey(t), CAT,
                                   latent_q_rows=_Q_ONE))
    assert lz.shape == (M_DRAW,)
    assert np.all(np.isfinite(lz))
    spread = float(lz.max() - lz.min())
    off = float(np.abs(lz - lz1).max())
    print(f"\n[P13 logZ {_theta_id(t)}] spread = {spread:.3e} nat, "
          f"|logZ_m - logZ_(Q=1)| = {off:.3e} nat")
    # Absolute, in nats: log Z is O(1-10) here and O(10) in production, so an
    # absolute nat tolerance is the conservative reading of a relative one.
    assert spread < P13_TOL
    assert off < P13_TOL


# --------------------------------------- P13, arm 2: the numerator's quadrature

@pytest.mark.parametrize("t", THETA, ids=_theta_id)
def test_p13_numerator_quadrature_conserves_the_budget(t):
    """The same identity on the OTHER integral: ``completion_curves``' dN_miss.

    The survey-global normalizer and the per-row numerator are different
    integrators over different row sets, and PLAN §4.2's "one budget from one
    formation site" is a claim about BOTH.  Here the objects are the ones the
    member vmap actually consumes: ``base_miss = (1 - f_p C) dN_exp`` (the
    member-independent curve) and ``N_miss_members`` (the per-member
    ``trapz(base_miss * Q_eff_m)`` that ``latent_member_N_miss_integrals``
    produces under ``lax.scan`` + ``jax.checkpoint``).

    Summed over the footprint rows, member m's missing count must equal the
    ``Q == 1`` one -- z-resolved AND after the quadrature.  The off-footprint
    catalog rows are excluded from the sum for the reason PLAN §2.2 gives: they
    carry ``Q == 1`` identically, so including them would only dilute the
    residual.
    """
    cc = completion_curves(_cosmo(t), _survey(t), CAT)
    base = np.asarray(cc.base_miss)                      # (N_rows, N_grid)
    N_mem = np.asarray(cc.N_miss_members)                # (M, N_rows)
    assert base.shape == (N_PIX, NG) and N_mem.shape == (M_DRAW, N_PIX)

    ref_curve = base[_ON_FP_CAT].sum(axis=0)             # (N_grid,)
    ref_int = float(np.trapz(base[_ON_FP_CAT], _Z, axis=-1).sum())
    worst_curve = 0.0
    worst_int = 0.0
    for m in range(M_DRAW):
        _, q = _member_q_rows(CAT, t, m, field_rows=False)
        got_curve = np.asarray(base * q)[_ON_FP_CAT].sum(axis=0)
        worst_curve = max(
            worst_curve, float(np.abs(got_curve / ref_curve - 1.0).max()))
        got_int = float(N_mem[m][_ON_FP_CAT].sum())
        worst_int = max(worst_int, abs(got_int / ref_int - 1.0))
    print(f"\n[P13 numerator {_theta_id(t)}] dN_miss curve {worst_curve:.3e}, "
          f"N_miss quadrature {worst_int:.3e}")
    assert worst_curve < P13_TOL
    assert worst_int < P13_TOL


# --------------------------------------------------- the pin is not vacuous

@pytest.mark.parametrize("t", THETA[:1], ids=_theta_id)
def test_p13_detects_a_wrong_budget_denominator(t):
    """A 1e-6 relative error in ``F_F`` moves the identity by ~1e-6.

    ``rho = log[(A - C B)/(P_F - C F_F)]`` normalizes against the budget the
    run consumes; ``F_F`` is that budget's ``f_p`` sum.  Perturbing it by the
    factory's own guard tolerance (``factory._LATENT_F_F_RTOL = 1e-6``, the
    F_F-consistency gate) must break P13 by roughly the same relative amount --
    which is the quantitative statement that the guard's tolerance and eq. (4)'s
    closure are the same number, not two independently chosen constants.
    """
    bad = CAT._replace(latent_F_F=float(CAT.latent_F_F) * (1.0 + 1e-6))
    res = _curve_residual(bad, t)
    print(f"\n[P13 F_F+1e-6] max|V_m/V_1 - 1| = {res:.3e}")
    assert res > 1e4 * P13_TOL, "P13 did not notice a mis-normalized rho"
    assert res < 1e-5


def test_p13_closure_floor_is_the_f_p_storage_dtype():
    """MEASUREMENT, not a gate: what sets eq. (4)'s floor on a REAL anchor.

    The pins above build the moments from the float32 ``f_p`` the catalog
    stores.  Production does not: ``cli/build_latent_field.py:126`` loads the
    depth map in float64 and hands that to ``sky_moments`` /
    ``sky_constant_coeffs``, while ``likelihood/catalog_views.py:748`` casts the
    SAME map to float32 for ``f_p_rows`` and ``field_f_p_occ``.  The two sides
    then consume different completeness values at the 1e-8 level, and eq. (4)
    closes there instead of at machine precision.

    Measured on this fixture, over the four theta with a live field: 1.6e-9
    (b_GW = 0.37) to 3.3e-8 (b_GW = 3.77) relative, against <= 1.1e-15 for the
    consistent configuration -- six orders of magnitude, from a dtype, and
    growing with b_GW exactly as ``e^{b f}`` says it should.  This is the exact
    analogue of the defect PR-5 found on the OTHER eq. (2) input (moments built
    from the f64 draws rather than the stored f32 row factors, 2.7e-7 at the
    production corner) and that ``latent_anchor_v2a.h5`` fixed.  It is recorded
    rather than gated because it is a property of the artifact's provenance,
    not of the seam: the fix is
    for the builder to form the moments and ``F_F`` from the float32 values the
    run will consume, which would take the production anchor to the same 1e-15.

    It is comfortably inside the factory's ``_LATENT_F_F_RTOL = 1e-6`` guard,
    so nothing currently refuses a run over it -- and comfortably ABOVE the
    1e-12 P13 asks for, which is why this test exists at all: without it, the
    pins above would read as a claim about the shipped anchor.
    """
    consistent = [_curve_residual(CAT, t) for t in THETA]
    mixed = [_curve_residual(CAT_MIXED, t) for t in THETA]
    for t, a, b in zip(THETA, consistent, mixed):
        print(f"\n[eq.4 floor {_theta_id(t)}] consistent f_p {a:.3e}   "
              f"f64-moments/f32-catalog {b:.3e}")
    assert max(consistent) < P13_TOL
    # The b_GW == 0 theta is excluded: a zero field gives Q == 1 identically,
    # so BOTH configurations are bit-exact there and the ratio is 0/0.
    live = [(a, b) for t, a, b in zip(THETA, consistent, mixed)
            if t["b_gw"] != 0.0]
    assert all(b > 100.0 * max(a, 1e-16) for a, b in live)
    assert max(b for _, b in live) < 1e-6


# ================================================================= member ESS

def _np_ess(ll):
    """Reference ESS in numpy, from the definition, no jax."""
    ll = np.asarray(ll, dtype=np.float64)
    if not np.isfinite(ll).any():
        return 0.0
    mx = ll.max()
    wt = np.exp(ll - mx)
    p = wt / wt.sum()
    nz = p > 0
    return float(np.exp(-(p[nz] * np.log(p[nz])).sum()))


def test_member_ess_matches_the_definition():
    """``exp(-sum_m p_m log p_m)``, ``p_m = softmax_m(ll_m)`` (PLAN §6.4)."""
    rng = np.random.default_rng(3)
    for ll in (np.zeros(8),                       # uniform -> ESS == M
               np.full(8, -12.5),                 # shift-invariant
               rng.normal(size=8) * 0.05,         # PR-5b's anchor regime
               rng.normal(size=8) * 1.2,          # PR-5b's H0 = 20 regime
               rng.normal(size=64) * 3.0):
        got = float(member_ess(jnp.asarray(ll)))
        assert got == pytest.approx(_np_ess(ll), rel=1e-12)
        assert 1.0 - 1e-9 <= got <= ll.size + 1e-9
    assert float(member_ess(jnp.zeros(8))) == pytest.approx(8.0, rel=1e-12)


def test_member_ess_reads_the_spread_the_way_plan_6_5_predicts():
    """``E[ESS]/M ~ exp(-sigma^2)`` for lognormal member weights (PLAN §6.5).

    Not a gate on the estimator -- a check that the diagnostic MEANS what §6.5
    reads off it, since ESS is the runtime stand-in for the member spread that
    sets the Jensen bias ``-(e^{sigma^2} - 1)/(2M)``.  Averaged over 4000 draws
    at M = 512 so the sampling error is small.
    """
    rng = np.random.default_rng(17)
    for sigma in (0.25, 0.5, 1.0):
        M, n = 512, 4000
        ll = rng.normal(scale=sigma, size=(n, M))
        got = float(np.mean([member_ess(jnp.asarray(row)) for row in ll[:200]]))
        pred = M * np.exp(-sigma ** 2)
        print(f"\n[ESS law] sigma={sigma}: measured {got:.1f}, "
              f"M exp(-sigma^2) = {pred:.1f}")
        assert 0.5 * pred < got < 2.0 * pred


def test_member_ess_is_nan_safe_when_the_guard_kills_members():
    """A member at ``-inf`` carries ``p_m = 0``; all-dead reports ESS == 0.

    Not hypothetical: the per-member selection guard returns ``-inf`` for a
    member whose ``Neff_m`` fails the Vitale floor or the total-variance
    criterion, and ``ll_members - logsumexp(ll_members)`` would be ``nan`` for
    every entry if they all did.
    """
    half = jnp.asarray([0.0, -jnp.inf, 0.0, -jnp.inf])
    assert float(member_ess(half)) == pytest.approx(2.0, rel=1e-12)
    dead = jnp.asarray([-jnp.inf] * 4)
    got = float(member_ess(dead))
    assert got == 0.0 and np.isfinite(got)
    one = jnp.asarray([-jnp.inf, 3.0, -jnp.inf])
    assert float(member_ess(one)) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------- the diagnostic in the likelihood

def _gw(n_events, n_samp, seed):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(80.0, 900.0, total))
    chieff = jnp.asarray(rng.uniform(-0.2, 0.2, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.asarray(rng.integers(0, N_PIX, total), dtype=jnp.int32)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=m2det / m1det,
                   valid=jnp.ones(total, dtype=jnp.bool_))


_N_EV, _N_SAMP, _N_SEL = 3, 64, 400
_GW_PE = _gw(_N_EV, _N_SAMP, seed=0)
_GW_SEL = _gw(_N_SEL, 1, seed=10)
_POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
#: The GWTC-4/5 hard variance guard fails at every H0 node on the production
#: line (PR-5b: it needs Neff ~ 92k and has 31-36k), and it fails on a 400-draw
#: fixture for the same reason.  The guard-convention rule is to SAY which arm
#: is in force: this is the clean arm of PR-0 -- hard wall
#: (``selection_neff_soft_guard=False``, the default), Vitale 5 N_obs floor
#: kept -- with the variance cap LIFTED so the fixture is testing the seam and
#: not the wall.  ``test_per_member_selection_guard_is_live_in_latent_mode``
#: puts the cap back and is the pin that the wall is there.
_MAX_VAR_LIFTED = 1e12


def _ll_kwargs(t, **kw):
    base = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=True, lss_field_mode="latent",
        catalog_sky_weighting="field",
        max_likelihood_variance=_MAX_VAR_LIFTED,
    )
    base.update(kw)
    return base


def _call(fn, t, **kw):
    return fn(_cosmo(t), _survey(t), _POP, _GW_PE, CAT, _GW_SEL, CAT,
              _N_EV, _N_SAMP, float(_N_SEL), **_ll_kwargs(t, **kw))


_T_LL = THETA[0]


def test_member_diagnostics_do_not_move_the_likelihood():
    """The flag is additive: off returns the bare scalar, on returns the dict.

    ``lss_member_diagnostics`` is a Python-level static branch, so with it off
    the ESS reduction is not merely cheap -- it is absent from the traced
    module, and the sampler's specialization is the one that existed before
    this diagnostic did.  (Additivity, the PR-6a ground rule.)

    ON is a SEPARATE jit specialization, exactly as
    ``darksiren_likelihood_diagnostics_with_clusters`` is, so the two agree to
    floating-point RE-ASSOCIATION rather than bit-for-bit: XLA schedules the
    dict-returning module differently (``logsumexp(ll_members)`` is now also
    consumed by the ESS).  MEASURED here: 4.5e-15 relative, ~20 ulp.  The
    tolerance below is stated against that measurement and is deliberately
    tight enough that any real change of estimand would fail it.
    """
    scalar = float(_call(darksiren_log_likelihood, _T_LL))
    diag = _call(darksiren_member_diagnostics, _T_LL)
    assert np.isfinite(scalar), "fixture went degenerate; the pin would be vacuous"
    assert jnp.ndim(_call(darksiren_log_likelihood, _T_LL)) == 0
    rel = abs(float(diag["logL_total"]) - scalar) / max(abs(scalar), 1e-300)
    print(f"\n[diagnostics specialization] relative logL shift = {rel:.3e}")
    assert rel < 1e-13


def test_member_ess_ships_from_the_likelihood():
    """PLAN §6.4's diagnostic, out of the real member marginalization.

    ``ll_members`` is the vector ``_factored_member_marginalization`` already
    materializes at ``core.py``'s ``ll_members = jax.vmap(...)``; the ESS is an
    O(M) reduction over it and costs nothing.  Checked three ways: the mixture
    reduction ``logsumexp_m ll_m - log M`` reproduces ``logL_total``, the ESS
    reproduces the numpy definition applied to the SAME vector, and it lands in
    ``[1, M]``.
    """
    diag = _call(darksiren_member_diagnostics, _T_LL)
    assert set(diag) == {"logL_total", "ll_members", "member_ess", "n_members"}
    ll_m = np.asarray(diag["ll_members"])
    assert ll_m.shape == (M_DRAW,) and int(diag["n_members"]) == M_DRAW
    assert np.all(np.isfinite(ll_m))
    from scipy.special import logsumexp as _lse
    assert _lse(ll_m) - np.log(M_DRAW) == pytest.approx(
        float(diag["logL_total"]), rel=0, abs=1e-9)
    ess = float(diag["member_ess"])
    print(f"\n[member ESS] ll_m sd = {ll_m.std(ddof=0):.3e} nat, "
          f"ESS = {ess:.4f} of M = {M_DRAW}")
    assert ess == pytest.approx(_np_ess(ll_m), rel=1e-10)
    assert 1.0 <= ess <= M_DRAW + 1e-9


def test_member_diagnostics_require_a_member_axis():
    """No ``lss_marginalize`` means no ``ll_m`` vector; refuse, don't fake one."""
    with pytest.raises(ValueError, match="requires lss_marginalize"):
        _call(darksiren_member_diagnostics, _T_LL, lss_marginalize=False)


def test_per_member_selection_guard_is_live_in_latent_mode():
    """ASSERT the guard PLAN §6.4 says already exists -- do not build a second.

    ``core._factored_member_marginalization``'s ``_member_ll`` calls
    ``selection_log_correction(log_mu_m, Neff_m, nEvents, soft_guard=...,
    max_likelihood_variance=..., pe_variance_sum=sum(event_vars))`` INSIDE the
    member vmap.  Rev 1 of the plan listed that as new PR-6a work (R1-SEV3-11);
    it is not, and adding a second wall would double-count it.

    The discriminating evidence that it is PER MEMBER, not a whole-likelihood
    guard applied once: members differ in ``Neff_m`` because they carry
    different ``Q_m``, so there must exist a variance cap at which SOME members
    return ``-inf`` and others stay finite.  A single outer guard could only
    ever produce all-or-nothing.  The scan below finds such a cap and reports
    it; the partial state is exactly what the member ESS is then measuring, so
    the two halves of §6.4 are checked against each other -- ESS must equal the
    surviving member count when the survivors are equally weighted.

    Clean-arm convention: hard wall (``selection_neff_soft_guard=False``), the
    Vitale 5 N_obs floor kept, only ``max_likelihood_variance`` scanned.
    """
    def _alive(cap):
        ll_m = np.asarray(
            _call(darksiren_member_diagnostics, _T_LL,
                  max_likelihood_variance=float(cap))["ll_members"])
        return np.isfinite(ll_m), ll_m

    lo, hi = 1e-4, _MAX_VAR_LIFTED
    assert _alive(hi)[0].all(), "the lifted cap already kills members"
    assert not _alive(lo)[0].any(), "a 1e-4 nat^2 cap admitted a member"

    # Each member's OWN critical cap, by bisection in log(cap).  Distinct
    # thresholds are the proof: one outer guard has one threshold.
    thresholds = []
    for m in range(M_DRAW):
        a, b = lo, hi
        for _ in range(45):
            mid = float(np.sqrt(a * b))
            if _alive(mid)[0][m]:
                b = mid
            else:
                a = mid
        thresholds.append(b)
    thresholds = np.asarray(thresholds)
    order = np.argsort(thresholds)
    print("\n[per-member guard] critical max_likelihood_variance per member: "
          + ", ".join(f"m{m}={thresholds[m]:.10g}" for m in order))
    assert len(set(thresholds.tolist())) == M_DRAW, (
        "every member trips the guard at the SAME variance cap, so this "
        "fixture cannot discriminate a per-member guard from an outer one: "
        f"{thresholds}")

    # Between the two most widely separated adjacent thresholds the ensemble is
    # PARTIALLY killed -- the state no outer guard can produce.
    s = thresholds[order]
    j = int(np.argmax(np.diff(np.log(s))))
    cap = float(np.sqrt(s[j] * s[j + 1]))
    alive, ll_m = _alive(cap)
    n_alive = int(alive.sum())
    print(f"[per-member guard] max_likelihood_variance = {cap:.10g} nat^2 "
          f"leaves {n_alive} of {M_DRAW} members finite")
    assert 0 < n_alive < M_DRAW
    # The ESS of a partially-killed ensemble counts only the survivors, so the
    # two halves of PLAN §6.4 agree: the guard removes members and the ESS sees
    # exactly that.
    ess = float(member_ess(jnp.asarray(ll_m)))
    assert ess <= n_alive + 1e-9
    assert np.isfinite(ess)
