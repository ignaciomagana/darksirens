"""
likelihood.py
-------------
Hierarchical dark-siren likelihood factory.

Sentinel convention
-------------------
All log-probability floors are -jnp.inf, not finite magic numbers.

RAM note
--------
optimization_barrier MUST be applied before arrays enter any JIT closure
(i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from darksirens.redshift.catalog import attest_rows_sorted_for_windowing
from darksirens.redshift.completion import (
    bound_smoothing_operator,
    build_pixel_kde_cache,
    smoothing_operator as completion_smoothing_operator,
)
from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
from darksirens.likelihood.catalog_views import barrier, prepare_catalog_views
from darksirens.likelihood.events import pad_gw_event_to_multiple
from darksirens.likelihood.block_sizing import require_resolved_block_size
from darksirens.likelihood.latent_q import (
    footprint_row_map,
    load_latent_plan,
    on_footprint_mask,
)
from darksirens.likelihood.core import (
    darksiren_log_likelihood,
    redshift_prior_state_sharing,
    require_view_independent_mu_miss,
    WL_BACKEND_DISABLED,
    WL_BACKEND_LOGNORMAL,
    WL_BACKEND_TABULATED,
    WL_SELECTION_STANDARD,
    WL_SELECTION_LOGNORMAL,
)
from darksirens.inference.parameters import (
    build_parameter_decoder,
    complete_empty_pixel_policy_code,
)
from darksirens.core.types import EMCatalog, GWEvent
from darksirens.utils import cosmology

# Backward-compatible aliases for callers/tests that imported private helpers.
_barrier = barrier
_complete_empty_pixel_policy_code = complete_empty_pixel_policy_code


def _resolve_redshift_prior_materialization(opts) -> bool:
    """Resolve whether to keep the likelihood-internal redshift-prior barrier."""
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError(
            "redshift_prior_barrier must be one of {'auto', 'on', 'off'}, "
            f"got {mode!r}."
        )
    tinyns_cfg = getattr(opts, "tinyns_resolved_config", None) or {}
    is_tinyns_jax_rwalk = (
        getattr(opts, "sampler", None) == "tinyns"
        and tinyns_cfg.get("kernel") == "jax"
        and tinyns_cfg.get("sample") == "rwalk"
    )
    # NumPyro NUTS differentiates the likelihood w.r.t. theta; the redshift
    # prior STATE depends on theta and ``lax.optimization_barrier`` has no
    # differentiation rule, so the barrier must come off for gradient-based
    # sampling (otherwise dark/bright-siren NUTS dies with
    # NotImplementedError at the first gradient).
    is_numpyro = getattr(opts, "sampler", None) == "numpyro"
    return not (is_tinyns_jax_rwalk or is_numpyro)


def _redshift_prior_materialization_reason(opts, materialize: bool) -> str:
    mode = getattr(opts, "redshift_prior_barrier", "auto")
    if mode in {"on", "off"}:
        return f"forced {mode}"
    if materialize:
        return "auto -> on"
    if getattr(opts, "sampler", None) == "numpyro":
        return "auto -> off for NumPyro NUTS (optimization_barrier is not differentiable)"
    return "auto -> off for TinyNS JAX rwalk"


def _to_jax(data: dict, key: str) -> jnp.ndarray:
    val = data.get(key)
    return jnp.asarray(val) if val is not None else jnp.array([0.0])


# ----------------------------------------------------------- the latent seam
#
# ``lss_field_mode='latent'`` (field-level PR-5) replaces the resident
# ``(M_draw, N_rows, N_grid)`` log-Q table with the compact anchor artifact of
# PR-4: ``Q`` is GENERATED from ``row_fac`` / ``phi_z`` / the sky moments
# ``(A, B)`` by :mod:`darksirens.likelihood.latent_q`.  Everything below is
# HOST-SIDE and runs exactly once, at likelihood build: the leaves are resolved
# and barriered here (never inside ``body()`` -- see this module's docstring:
# under a trace the arrays are abstract and ``optimization_barrier`` is a
# no-op), and the guards fire BEFORE the first likelihood evaluation, which is
# the only place a mismatched artifact is still cheap to reject.
#
# Default is ``table``: with the flag off every function here returns empty
# dicts and the two ``EMCatalog(...)`` constructions below are textually and
# numerically the shipped ones.

#: The anchor builder floors the per-pixel selection fraction at ``1e-3``
#: before it forms ``F_F`` (``cli/build_latent_field.py``: ``np.maximum(f_p,
#: 1e-3)``).  The run's ``f_p`` map is NOT floored (``catalog_views`` gathers it
#: raw), so the consistency guard must apply the same floor or every footprint
#: pixel below it would read as a depth-map mismatch.
_LATENT_F_P_FLOOR = 1e-3

#: Relative tolerance of the ``F_F`` consistency guard.  MEASURED: the only
#: admissible difference between the two sums is that the run carries ``f_p``
#: as float32 (``catalog_views._fp_rows``) while the builder summed float64;
#: over 30,470 production footprint rows that round-trip moves the sum by
#: <= 4.5e-10 relative (50 random draws, both sides accumulated in float64,
#: which is why this guard casts before summing rather than trusting a float32
#: reduction -- that costs 1.0e-8 on its own).  1e-6 sits between that floor
#: (2200x above it, so representation alone can never fire the guard) and the
#: 2.7e-7 at which eq. (4)'s own budget identity closes at the production
#: corner (build_latent_field.py), i.e. the same order as the identity the
#: guard protects.  It is not a coarse gate: ``f_p <= 1`` bounds ``F_F <= P_F``
#: = 30,470, so ONE footprint pixel whose f_p moved by 0.1 shows up at
#: >= 3.3e-6 -- above the tolerance whatever the depth map looks like.
_LATENT_F_F_RTOL = 1e-6

#: Isotropy guard threshold (PLAN §4.4 successor 4): the anchor's angular and
#: radial correlation lengths must agree to within a factor of 1.5 in PHYSICAL
#: units at ``z_ref``.  Beyond that the field the artifact fits is a pancake --
#: a 4:1 anisotropy fits the radial modes to angular structure (and vice versa)
#: and the modulation it generates is no longer the clustering it claims to be.
_LATENT_ISOTROPY_LOG_TOL = float(np.log(1.5))


def _resolve_lss_field_mode(opts):
    """Validate ``--lss_field_mode`` and its pairing with the artifact path.

    Returns ``(mode, artifact_path)``.  A supplied artifact under ``table``
    mode is a HARD ERROR rather than a silent no-op: an input that is read,
    parsed and then ignored is exactly the class of failure this codebase
    refuses elsewhere (see the Q-table provenance checks), and the user who
    passed it believes the run is latent.
    """
    mode = getattr(opts, "lss_field_mode", "table") or "table"
    if mode not in ("table", "latent"):
        raise ValueError(
            f"lss_field_mode must be 'table' or 'latent', got {mode!r}. "
            "'table' is the shipped resident-log-Q path; 'latent' generates Q "
            "from the anchor artifact (--lss_field_artifact)."
        )
    path = getattr(opts, "lss_field_artifact", None)
    if mode == "table" and path is not None:
        raise ValueError(
            f"lss_field_artifact={path!r} was given but lss_field_mode is "
            "'table', which never reads it: the run would silently use the "
            "shipped table path and the artifact would have no effect on a "
            "single number. Pass --lss_field_mode latent to consume it, or "
            "drop the artifact."
        )
    if mode == "latent" and path is None:
        raise ValueError(
            "lss_field_mode='latent' requires --lss_field_artifact: in latent "
            "mode Q is GENERATED from the anchor artifact (row_fac, phi_z and "
            "the (A, B) sky moments), so there is no Q at all without it. "
            "Build one with darksirens_build_latent_field, or run with "
            "--lss_field_mode table."
        )
    return mode, path


def _latent_guard_exclusivity(opts) -> None:
    """PLAN §4.4 successor 6, the half that is visible at likelihood build.

    Latent mode is the ONE generator of ``Q``.  A loaded Q table
    (``--lss_completion``) would be a SECOND missing-galaxy modulation on the
    same rows, and ``--use_lss``'s per-pixel ``delta_g`` a third: the two
    multiply, so the budget eq. (4) conserves is not the budget the run
    consumes, and the doubled modulation lands directly on H0 through the
    missing-galaxy weight.  The ``c_mode`` half of guard 6 (per-pixel /
    stratified completeness does not factor through the sky moments) is
    enforced at the point of use, in ``redshift/completion.py``.
    """
    if getattr(opts, "lss_completion", None) is not None:
        raise ValueError(
            "lss_field_mode='latent' generates Q from the anchor artifact, so "
            f"a loaded Q_LSS table (--lss_completion "
            f"{getattr(opts, 'lss_completion')!r}) would apply a SECOND "
            "modulation to the same rows: the two multiply and the consumed "
            "missing budget stops being the one rho conserves (PLAN eq. 4). "
            "Drop --lss_completion, or run with --lss_field_mode table."
        )
    if bool(getattr(opts, "use_LSS", False)):
        raise ValueError(
            "lss_field_mode='latent' is incompatible with --use_lss: the "
            "per-pixel delta_g overdensity is a second missing-galaxy "
            "modulation on top of the generated Q, and b_miss is b_GW in "
            "latent mode (PLAN §4.3), so the field would be applied twice. "
            "Drop --use_lss, or run with --lss_field_mode table."
        )


def _latent_guard_ls_z_units(opts) -> None:
    """PLAN §4.4 successor 2: ``ls_z`` must be in ``zeta = log1p(z)`` units.

    The lognormal builders accept the radial correlation length in Mpc and map
    it to zeta at a reference redshift, ``ls_z = L / ((1 + z_ref) dchi/dz)``.
    That mapping scales like ``H0``: over the sampled prior ``H0 in [20, 140]``
    it varies by 7x, so an ASSUMED Mpc length becomes a standard ruler --- the
    field's own lengthscale would then inform H0, from 22.79M galaxies, without
    ever appearing in the likelihood as a parameter.  The anchor artifact is
    built in zeta directly (``--ls-z``), and there is no admissible Mpc-valued
    input in latent mode.
    """
    mpc = getattr(opts, "lss_corr_length_mpc", None)
    if mpc is not None:
        raise ValueError(
            f"lss_field_mode='latent' was given lss_corr_length_mpc={mpc!r}, "
            "an Mpc-valued radial correlation length. In latent mode ls_z is "
            "in zeta = log1p(z) units and is fixed by the artifact; converting "
            "an Mpc length to zeta needs ls_z = L/((1+z_ref) dchi/dz), which "
            "scales like H0 and varies 7x over H0 in [20, 140] -- an assumed "
            "length would act as a standard ruler against 22.79M galaxies. "
            "Drop lss_corr_length_mpc and set the anchor's --ls-z in zeta."
        )


def _latent_guard_resolution(plan) -> None:
    """PLAN §4.4 successor 3: the inducing grid must RESOLVE both lengthscales.

    A low-rank GP whose node spacing exceeds its kernel lengthscale collapses
    to the prior while still reporting convergence (Burt et al. 2019,
    arXiv:1903.03571); the shipped 50 Mpc fiducial was ~30x under-resolved and
    measured a fitted-vs-truth logQ slope of 0.04 on the closure experiment.

    ``cli/build_lognormal_completion._gp3d_resolution_guard`` encodes the same
    two inequalities but its SPHERE half only WARNS, and PLAN promotes both to
    HARD in latent mode; importing a CLI module from the likelihood factory to
    then re-implement half of it buys nothing, so the two inequalities are
    inlined here.  The anchor builder checks them too -- this is the consumer's
    copy, because the run does not necessarily own the artifact it was handed.
    """
    m_sph = int(plan.m_sph)
    m_z = int(plan.m_z)
    ls_sph = float(plan.meta["ls_sph"])
    ls_z = float(plan.meta["ls_z"])
    z_node_hi = float(plan.meta["z_node_hi"])

    d_sph = float(np.sqrt(4.0 * np.pi / max(m_sph, 1)))
    if d_sph > ls_sph:
        raise ValueError(
            f"latent artifact is under-resolved on the SPHERE: Fibonacci node "
            f"spacing sqrt(4pi/M_sph) = {d_sph:.4g} ({m_sph} nodes) exceeds "
            f"the chordal angular lengthscale ls_sph = {ls_sph:.4g}. A "
            "low-rank GP with spacing > lengthscale collapses to the prior "
            "(Burt et al. 2019) while still reporting convergence, so the "
            "generated Q would carry the prior's clustering, not the "
            f"catalog's. Rebuild the anchor with --m-sph >= "
            f"{int(np.ceil(4.0 * np.pi / ls_sph ** 2))}, or a larger --ls-sph."
        )
    if m_z < 2:
        raise ValueError(
            f"latent artifact has M_z = {m_z} radial inducing nodes; at least "
            "2 are needed to resolve any redshift structure. Rebuild the "
            "anchor with --m-z >= 2."
        )
    d_zeta = float(np.log1p(z_node_hi)) / (m_z - 1)
    if d_zeta > ls_z:
        raise ValueError(
            f"latent artifact is under-resolved in REDSHIFT: node spacing in "
            f"zeta = log1p(z) is {d_zeta:.4g} ({m_z} nodes up to z_node_hi = "
            f"{z_node_hi:.4g}) but the lengthscale is ls_z = {ls_z:.4g}. Same "
            "failure as the sphere side: the posterior collapses to the prior "
            "silently. Rebuild the anchor with --m-z >= "
            f"{int(np.ceil(np.log1p(z_node_hi) / ls_z)) + 1}, or a larger "
            "--ls-z."
        )


def _latent_guard_isotropy(plan) -> None:
    """PLAN §4.4 successor 4: refuse an anisotropic (pancake) anchor field.

        |log( (ls_sph chi(z_ref)) / (ls_z (1 + z_ref) dchi/dz) )| < log(1.5)

    ``ls_sph`` is chordal on the unit sphere, so ``ls_sph * chi`` is the
    TRANSVERSE physical correlation length at ``z_ref``; ``ls_z`` is in
    ``zeta = log1p(z)``, so ``dz = (1 + z) dzeta`` makes ``ls_z (1 + z_ref)
    dchi/dz`` the RADIAL one.  Their ratio is the field's aspect ratio: a 4:1
    pancake fits radial modes to angular structure and generates a modulation
    that is not the clustering it claims to be.

    ``z_ref`` is the MIDPOINT of the artifact's depth, ``0.5 * z_node_hi`` --
    the fitted volume's own scale, and the only z the artifact itself names.
    The ratio is H0-FREE by construction: ``chi = (c/H0) int dz/E`` and
    ``dchi/dz = (c/H0)/E`` carry the same ``c/H0``, so it cancels and only
    ``Om0`` survives (taken from the artifact's ``theta_ref`` when it has one,
    the Planck fiducial otherwise).  That is the same reason the ls_z-in-Mpc
    input is refused above: nothing here may depend on the sampled H0.
    """
    z_node_hi = float(plan.meta["z_node_hi"])
    z_ref = 0.5 * z_node_hi
    ls_sph = float(plan.meta["ls_sph"])
    ls_z = float(plan.meta["ls_z"])
    theta_ref = dict(plan.theta_ref or {})
    om0 = float(theta_ref.get("Om0", cosmology.Om0Planck))
    h0_cancels = 70.0  # any value: it cancels in the ratio below
    chi = float(cosmology.r_of_z(z_ref, h0_cancels, om0))
    dchi_dz = float(
        cosmology.speed_of_light / (h0_cancels * float(cosmology.E(z_ref, om0)))
    )
    transverse = ls_sph * chi
    radial = ls_z * (1.0 + z_ref) * dchi_dz
    aspect = transverse / radial
    if abs(np.log(aspect)) >= _LATENT_ISOTROPY_LOG_TOL:
        raise ValueError(
            f"latent artifact's correlation lengths are anisotropic at "
            f"z_ref = {z_ref:.4g} (the depth midpoint, Om0 = {om0:.4g}): "
            f"transverse ls_sph*chi = {transverse:.4g} Mpc vs radial "
            f"ls_z*(1+z)*dchi/dz = {radial:.4g} Mpc, an aspect ratio of "
            f"{aspect:.3g} (limit 1.5). A pancake field fits radial modes to "
            "angular structure and back, so the Q it generates is not the "
            "clustering it claims to be. Rebuild the anchor with --ls-sph and "
            "--ls-z matched in physical units at z_ref."
        )


def _latent_run_f_p_by_pixel(catalogs, n_rows_pe):
    """The run's per-pixel selection fraction, as ``(pixels, f_p, source)``.

    Prefers the FIELD rows (``field_f_p_occ`` on ``field_occupied_pixels``),
    which are the survey's occupied pixels -- exactly the set the anchor
    builder fits its footprint on -- and falls back to the PE catalog rows.
    ``(None, None, None)`` when the run carries no per-pixel completeness at
    all.
    """
    occ = getattr(catalogs, "field_occupied_pixels", None)
    fp_occ = getattr(catalogs, "field_f_p_occ", None)
    if occ is not None and fp_occ is not None:
        return (np.asarray(occ, dtype=np.int64),
                np.asarray(fp_occ, dtype=np.float64), "field_f_p_occ")
    fp_rows = getattr(catalogs, "f_p_rows_pe", None)
    if fp_rows is not None:
        up = getattr(catalogs, "unique_pixels_pe", None)
        pix = (np.arange(int(n_rows_pe), dtype=np.int64) if up is None
               else np.asarray(up, dtype=np.int64))
        return pix, np.asarray(fp_rows, dtype=np.float64), "f_p_rows_pe"
    return None, None, None


def _latent_guard_f_p_consistency(plan, catalogs, n_rows_pe) -> None:
    """PLAN §4.4 successor 1: the run's depth map must be the anchor's.

    ``F_F = sum_{p in F} f_p`` is the denominator constant of the budget
    normalizer ``rho = log[(A - C B)/(P_F - C F_F)]``, and ``B(z; b) =
    sum_p f_p e^{bf}`` carries the same ``f_p`` inside the artifact.  If the
    run's per-pixel completeness differs from the one the anchor was built
    against, ``rho`` normalizes against a budget nobody consumes and eq. (4)'s
    identity -- ``sum_p (1 - f_p C) Q_p == sum_p (1 - f_p C)`` at every z,
    member and theta -- silently stops holding.  Nothing downstream notices:
    the likelihood still evaluates, just against the wrong missing budget.

    So compare the two sums here, over the footprint rows the run will actually
    use.  Both sides accumulate in float64 and both apply the builder's
    ``1e-3`` floor; the tolerance is :data:`_LATENT_F_F_RTOL` (see its note for
    the measurement that fixes it).
    """
    fit_pixels = np.asarray(plan.meta["fit_pixels"], dtype=np.int64)
    if int(fit_pixels.size) != int(plan.n_fit):
        raise ValueError(
            f"latent artifact is internally inconsistent: {fit_pixels.size} "
            f"fit_pixels but row_fac carries {plan.n_fit} footprint rows "
            "(+1 zero pad). Rebuild the anchor."
        )
    if abs(float(plan.P_F) - float(plan.n_fit)) > 0.5:
        raise ValueError(
            f"latent artifact's P_F = {float(plan.P_F)} is not its footprint "
            f"size {plan.n_fit}; P_F is |F| by definition (PLAN eq. 2). "
            "Rebuild the anchor."
        )

    pix, f_p, source = _latent_run_f_p_by_pixel(catalogs, n_rows_pe)
    if pix is None:
        raise ValueError(
            "lss_field_mode='latent' requires the run's per-pixel selection "
            "fraction f_p (neither field_f_p_occ nor f_p_rows is present): "
            "the artifact's sky moment B(z; b) = sum_p f_p e^{bf} and its "
            "F_F = sum_p f_p are built from it, so without it the budget "
            "normalizer rho cannot be checked against the depth map the run "
            "actually consumes (PLAN §4.4 guard 1). Pass "
            "--per_pixel_completeness (the same map the anchor was built "
            "with)."
        )

    # Same searchsorted membership test ``footprint_row_map`` uses, run the
    # other way round: every footprint pixel must be present in the run's rows,
    # because those are the rows whose f_p the artifact's moments already sum.
    order = np.argsort(pix)
    pix_sorted = pix[order]
    pos = np.zeros(fit_pixels.shape, dtype=np.int64)
    if pix_sorted.size == 0:
        n_missing = int(fit_pixels.size)
    else:
        pos = np.clip(np.searchsorted(pix_sorted, fit_pixels),
                      0, pix_sorted.size - 1)
        hit = pix_sorted[pos] == fit_pixels
        n_missing = int((~hit).sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} of the latent artifact's {fit_pixels.size} footprint "
            f"pixels are absent from the run's {source} ({pix.size} pixels): "
            "the run's sky is not the sky the anchor was fit on, so the "
            "footprint row map would silently send those rows to the "
            "off-footprint pad (Q == 1) while the artifact's moments still "
            "count them. Rebuild the anchor against this run's catalog, or run "
            "with --lss_field_mode table."
        )
    f_p_fit = np.maximum(f_p[order][pos], _LATENT_F_P_FLOOR)
    f_f_run = float(f_p_fit.sum())
    f_f_art = float(plan.F_F)
    denom = max(abs(f_f_art), 1e-300)
    rel = abs(f_f_run - f_f_art) / denom
    if rel > _LATENT_F_F_RTOL:
        raise ValueError(
            f"latent artifact's F_F = {f_f_art:.10g} disagrees with the run's "
            f"sum of f_p over the same {fit_pixels.size} footprint pixels "
            f"({source}): {f_f_run:.10g}, a relative difference of {rel:.3g} "
            f"(tolerance {_LATENT_F_F_RTOL:g}, ~2200x the float32 storage "
            "round-trip and the same order as eq. 4's own closure floor). "
            "The two depth maps differ, so rho = log[(A - C B)/(P_F - C F_F)] "
            "would normalize against a budget this run never consumes and the "
            "eq. (4) identity would stop holding silently. Pass the SAME "
            "--per_pixel_completeness map the anchor was built with, or "
            "rebuild the anchor."
        )


def _resolve_latent_leaves(opts, catalogs, survey_z_depth, nside,
                           n_rows_pe, n_rows_sel):
    """Resolve the latent-seam EMCatalog leaves (``(mode, pe, sel)``).

    In ``table`` mode both dicts are EMPTY, so the ``EMCatalog(...)`` calls
    below are the shipped ones with a ``**{}`` splat and no leaf changes -- the
    flag-off path is bit-identical by construction, not by testing.

    In ``latent`` mode the artifact is loaded once (HOST-SIDE), the build-time
    guards fire, and every array is barriered HERE: PLAN §3.6 and this module's
    docstring both say the barrier must precede the JIT closure, because inside
    ``body()`` the leaves are already tracers and ``optimization_barrier`` has
    no effect.

    The theta-free blocks (``row_fac``, ``phi_z``, ``A``, ``B``, ``b_nodes``)
    are barriered ONCE and ALIASED into both dicts, and the row maps are
    aliased whenever the PE and selection views share their pixel array: the
    redshift-prior-state sharing verdict (``can_share_redshift_prior_state``)
    tests ``is`` identity over every EMCatalog field, so two separately
    barriered copies of the same array would collapse sharing to False and
    double the per-member prior-state precomputation.
    """
    mode, path = _resolve_lss_field_mode(opts)
    if mode == "table":
        return "table", {}, {}

    # Guards that need only the run configuration come first: they are free,
    # and the artifact is a 64 MB read at production rank.
    _latent_guard_exclusivity(opts)
    _latent_guard_ls_z_units(opts)

    plan = load_latent_plan(path, z_depth=survey_z_depth, expect_nside=nside)

    _latent_guard_resolution(plan)
    _latent_guard_isotropy(plan)
    _latent_guard_f_p_consistency(plan, catalogs, n_rows_pe)

    fit_pixels = np.asarray(plan.meta["fit_pixels"], dtype=np.int64)
    n_fit = int(plan.n_fit)

    def _row_leaves(unique_pixels, n_rows, *, where):
        """``(row_map, on_fp)`` for one row set, barriered.

        ``unique_pixels`` is the row -> global-HEALPix map of a COMPACT view;
        ``None`` is the legacy full-sky catalog, whose row index IS the pixel
        index.  Rows outside the fitted footprint are sent to the zero pad row
        ``n_fit`` and masked, which is what makes their logQ bit-zero (pin
        P13b) rather than merely small.
        """
        if unique_pixels is None:
            row_pixels = np.arange(int(n_rows), dtype=np.int64)
        else:
            row_pixels = np.asarray(unique_pixels, dtype=np.int64)
            if int(row_pixels.size) != int(n_rows):
                raise ValueError(
                    f"{where}: {row_pixels.size} unique pixels for "
                    f"{int(n_rows)} catalog rows; the latent row map indexes "
                    "catalog rows, so the two must agree."
                )
        row_map = footprint_row_map(row_pixels, fit_pixels, n_fit)
        on_fp = on_footprint_mask(row_map, n_fit)
        return (barrier(jnp.asarray(row_map, dtype=jnp.int32)),
                barrier(jnp.asarray(on_fp)))

    row_map_pe, on_fp_pe = _row_leaves(
        catalogs.unique_pixels_pe, n_rows_pe, where="latent PE row map")
    if catalogs.unique_pixels_pe is catalogs.unique_pixels_sel \
            and int(n_rows_pe) == int(n_rows_sel):
        row_map_sel, on_fp_sel = row_map_pe, on_fp_pe
    else:
        row_map_sel, on_fp_sel = _row_leaves(
            catalogs.unique_pixels_sel, n_rows_sel,
            where="latent selection row map")

    # FIELD rows: the survey-global normalizer's occupied rows
    # (``field_dN_obs_s``).  ``None`` under the conditional sky-weighting
    # convention, which never builds that normalizer -- and then the field
    # latent leaves are correctly None too.
    occ_pixels = getattr(catalogs, "field_occupied_pixels", None)
    if occ_pixels is None:
        field_row_map = field_on_fp = None
    else:
        occ_np = np.asarray(occ_pixels, dtype=np.int64)
        field_rows = footprint_row_map(occ_np, fit_pixels, n_fit)
        field_row_map = barrier(jnp.asarray(field_rows, dtype=jnp.int32))
        field_on_fp = barrier(jnp.asarray(
            on_footprint_mask(field_rows, n_fit)))

    shared = dict(
        latent_row_fac=barrier(jnp.asarray(plan.row_fac)),
        latent_phi_z=barrier(jnp.asarray(plan.phi_z)),
        latent_A=barrier(jnp.asarray(plan.A)),
        latent_B=barrier(jnp.asarray(plan.B)),
        latent_b_nodes=barrier(jnp.asarray(plan.b_nodes)),
        # P_F / F_F are the two eq. (2) SCALARS; they ride as plain Python
        # floats (as the seam's own fixtures do) so they land in the jit
        # signature as scalars rather than as two more device buffers.
        latent_P_F=float(plan.P_F),
        latent_F_F=float(plan.F_F),
        latent_field_row_map=field_row_map,
        latent_field_on_fp=field_on_fp,
    )
    leaves_pe = dict(shared, latent_row_map=row_map_pe, latent_on_fp=on_fp_pe)
    leaves_sel = dict(shared, latent_row_map=row_map_sel,
                      latent_on_fp=on_fp_sel)
    return "latent", leaves_pe, leaves_sel


def _jit_likelihood_body(body, operands):
    """Wrap a ``(coord, operands) -> logL`` body in ``jax.jit`` and bind ``operands``.

    Why jit the factory's own closure at all
    ----------------------------------------
    Only the inner :func:`darksiren_log_likelihood` was jitted, so everything the
    returned closure did around it ran EAGERLY — and ``dynesty`` calls that closure
    once per live point (``inference/sampling.py``: ``float(np.asarray(
    likelihood(jnp.asarray(theta))))``), i.e. ~1e5-1e6 times per run.  The eager
    work is ``parameter_decoder.decode(coord)`` (~30 individual device ops: one
    gather per sampled label, then ``jnp.array`` over the parameter sub-vectors)
    plus, before the hoists above, the GW container rebuild and selection padding.
    MEASURED on the H100 NVL: 7.7 ms of 30.1 ms for the spectral single pass, 13.8
    of 51.3 ms at the production auto plan, 7.8 of 17.9 ms for a dark-siren mock.
    ``tinyns`` and ``numpyro`` escape it because they trace the closure themselves;
    this is a dynesty tax, and dynesty is the sampler of most production runs.

    Why ``operands`` is an ARGUMENT and not a closure capture
    --------------------------------------------------------
    ``jax.jit`` lowers a closed-over concrete ``jax.Array`` to a ``dense<>`` HLO
    **constant**, not to a parameter — verified on jax 0.4.34: the lowered module
    text grows ~8 bytes per array element, so the ~1.07e6-row GW arrays alone would
    add tens of MB of HLO (and a dark-siren catalog's multi-GB tables far more),
    ballooning compile time and duplicating buffers that are already device
    resident.  Threading them through as an argument keeps them exactly what they
    are today for the inner jit: already-barriered, already-resident parameters.
    The barrier contract in this module's docstring is preserved — the barriers are
    applied eagerly at build time, before the arrays enter any jit.

    The cosmology table is an operand too
    -------------------------------------
    The same argument applies to the 106.8 MB comoving-distance table in
    ``utils.cosmology``, which every likelihood reaches through ``z_of_dL`` /
    ``dL_grid_bounds`` far below this frame.  It used to be a module global, i.e.
    a closure capture of THIS jit, and cost 427.5 MB of lowered module text for
    the production spectral likelihood (two ``dense<21x41x31x500xf64>`` literals
    at ~16 bytes of text per element).  It now travels as the third jit argument
    and is installed as the active table for the trace, so the ~45 cosmology call
    sites resolve it without carrying it in their signatures.

    Recompilation: ``coord`` is a 1-D float array of fixed length (the sampled
    dimension), ``operands`` is the same pytree of the same shapes/dtypes on
    every call, and ``distance_table`` is the same buffer, so the traced
    signature is constant and the body compiles once.
    """
    distance_table = cosmology.distance_table()
    # Same treatment for the (1000, 1000) expected-counts smoothing operator
    # (issue #305's residual): as a module-global capture it lowered as a
    # dense<1000x1000xf64> constant, ~16 MB of module text per specialization.
    smoothing_operator = completion_smoothing_operator()

    def _body_with_tables(coord: jnp.ndarray, operands, distance_table,
                          smoothing_operator):
        with cosmology.bound_distance_table(distance_table), \
                bound_smoothing_operator(smoothing_operator):
            return body(coord, operands)

    jitted = jax.jit(_body_with_tables)

    def likelihood(coord: jnp.ndarray) -> jnp.ndarray:
        return jitted(coord, operands, distance_table, smoothing_operator)

    # Host-side hook for tests/diagnostics that want to count compilations or reach
    # the un-bound body.
    likelihood.jitted_body = jitted
    likelihood.operands = operands
    likelihood.distance_table = distance_table
    likelihood.smoothing_operator = smoothing_operator
    return likelihood


def _make_mixture_likelihood(
    opts,
    data: dict,
    pop_params_fid,
    fixed_parameter_values: dict | None,
    n_catalogs: int,
):
    """Build the bundle-source likelihood callable (galaxy-aware models, K >= 1).

    Each catalog in ``data["catalogs"]`` carries its OWN nside/apix, compact
    PE/selection views, KDE cache, and Q_LSS table; the GW posterior and
    selection *physics* arrays (masses, distances, spins, sky vectors) are shared
    across catalogs.  The per-catalog compact pixel maps are stacked into ``(N,
    K)`` matrices; for K >= 2 the mixture weights come from ``decode_mixture``,
    while a single-bundle K = 1 run uses the plain decoder (same parameters as a
    flat-data K = 1 run).
    """
    bundles = data.get("catalogs")
    if bundles is None or len(bundles) != n_catalogs:
        raise ValueError(
            f"n_catalogs={n_catalogs} requires data['catalogs'] with "
            f"{n_catalogs} bundles; got "
            f"{None if bundles is None else len(bundles)}."
        )

    universe_model = opts.universe_model
    # Operands this path does NOT carry: the bundle EMCatalogs are built without
    # the counterpart plumbing and the stratified-selection inputs, and the body
    # forwards no weak-lensing operands.  For K >= 2 each of those combinations is
    # rejected elsewhere (core.darksiren_log_likelihood's mixture universe-model
    # gate, parameters.build_parameter_decoder's single-catalog strata rule), but
    # the K = 1-WITH-bundles route has no gate at all -- and a dropped counterpart
    # is SILENT (redshift/prior.py falls back to an arbitrary catalogued pixel as
    # "the counterpart"), while a dropped WL backend just evaluates with
    # magnification off.  Fail at BUILD time so the unsupported combinations are
    # unreachable by construction.
    dropped = [
        key
        for key in (
            "counterpart_pixel", "counterpart_pixels", "counterpart_zs",
            "counterpart_dzs", "wl_params", "pixel_stratum_map",
        )
        if data.get(key) is not None
    ]
    if dropped:
        raise NotImplementedError(
            "The bundle-source likelihood does not carry these operands: "
            f"{', '.join(dropped)}. Counterparts, weak lensing and stratified "
            "selection are single-catalog FLAT-data features (no "
            "data['catalogs'])."
        )
    if universe_model in ("bright_sirens", "spectral_sirens_wl"):
        raise NotImplementedError(
            f"universe_model={universe_model!r} is not supported by the "
            "bundle-source likelihood (its counterpart / weak-lensing operands "
            "are not carried); run it on the flat single-catalog path."
        )

    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    sel_batch_size = require_resolved_block_size(
        "sel_batch_size", getattr(opts, "sel_batch_size", None))
    pe_event_block = require_resolved_block_size(
        "pe_event_block", getattr(opts, "pe_event_block", None))
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
    _mnbc = getattr(opts, "mark_names_by_catalog", None)
    mark_names_all = (
        tuple(tuple(names or ()) for names in _mnbc) if _mnbc
        else (mark_names,) + ((),) * (n_catalogs - 1)
    )
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    max_likelihood_variance = float(getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

    def _check_bundle_marks(names_k, marks_k, ngals_k, field_k, *, where):
        """Reject mark tables that would saturate the ``log h`` clip.

        ``marks_k`` is keyed by EMCatalog field name (``mark_logmstar``, ...);
        the guard wants canonical mark names, so remap here.  Skipped entirely
        for the ``none`` model, which has no ``eta`` to keep alive.
        """
        if mark_model in (None, "none") or not names_k:
            return
        from darksirens.marks import (
            MARK_FIELDS as _MF,
            check_flat_marks_centred,
            check_marks_centred,
            get_mark_model,
        )

        model_k = get_mark_model(mark_model, names_k)
        check_marks_centred(
            model_k,
            {name: marks_k[_MF[name]] for name in names_k},
            ngals_k,
            where=where,
        )
        check_flat_marks_centred(
            model_k, field_k.get("field_mark_values"), where=where
        )

    def _bundle_field_inputs(bundle):
        """Per-bundle FIELD-convention normalization inputs (survey-global),
        precomputed by the loader (loaders.py) or supplied directly in tests.
        Shared by the bundle's PE and selection EMCatalogs so their global Z is
        the SAME value for the same theta (constants cancel structurally).
        Includes the budget-modulation rows (Q_LSS / delta_g) when present."""
        fobs = bundle.get("field_dN_obs_s")
        if fobs is None:
            return dict(
                field_dN_obs_s=None, field_n_empty=None, field_N_obs_total=None,
                field_occupied_pixels=None, field_lss_q=None,
                field_lss_q_empty_sum=None, field_delta_g=None,
                field_lss_q_members=None, field_lss_q_empty_sum_members=None,
                field_mark_z=None, field_mark_w=None, field_mark_values=None,
                field_depth_z=None, field_depth_dz=None, field_depth_c=None,
            )

        def _maybe(key, dtype=None):
            val = bundle.get(key)
            if val is None:
                return None
            arr = jnp.asarray(val) if dtype is None else jnp.asarray(val, dtype=dtype)
            return barrier(arr)

        return dict(
            field_dN_obs_s=barrier(jnp.asarray(fobs)),
            field_n_empty=jnp.asarray(bundle["field_n_empty"], dtype=jnp.float64),
            field_N_obs_total=jnp.asarray(
                bundle["field_N_obs_total"], dtype=jnp.float64
            ),
            field_occupied_pixels=_maybe("field_occupied_pixels", jnp.int32),
            field_lss_q=_maybe("field_lss_q"),
            field_lss_q_empty_sum=_maybe("field_lss_q_empty_sum"),
            field_delta_g=_maybe("field_delta_g"),
            field_lss_q_members=_maybe("field_lss_q_members"),
            field_lss_q_empty_sum_members=_maybe("field_lss_q_empty_sum_members"),
            field_mark_z=_maybe("field_mark_z"),
            field_mark_w=_maybe("field_mark_w"),
            field_mark_values=_maybe("field_mark_values"),
            # DEPTH-consistent global observed term (loaders build these only when
            # a survey depth is active; see build_field_depth_inputs).
            field_depth_z=_maybe("field_depth_z", jnp.float64),
            field_depth_dz=_maybe("field_depth_dz", jnp.float64),
            field_depth_c=_maybe("field_depth_c", jnp.float64),
        )

    # Shared (catalog-independent) GW / selection physics arrays.
    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    # Optional extra-spin block (DS-07): (N, d) columns beyond chieff, present
    # only for a non-chieff parameter space (basis negotiation, DS-09).  None
    # keeps the GWEvent pytree structure -- and every compiled likelihood --
    # identical to a build without the field.
    spin_pe = (barrier(jnp.asarray(data["spin_pe"], dtype=jnp.float64))
               if data.get("spin_pe") is not None else None)
    spin_sel = (barrier(jnp.asarray(data["spin_sel"], dtype=jnp.float64))
                if data.get("spin_sel") is not None else None)

    def _compact_lss_q_for(views, unique_pixels):
        # Per-catalog analogue of make_likelihood._compact_lss_q: slice each
        # catalog's own Q_LSS table to its union pixels so only the compact block
        # reaches the device.
        full = views.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(views.lss_completion_indexing or 0)
        if idx == 1:
            # Stamped compact: rows must already be union-pixel-aligned (same
            # pre-JIT validation as make_likelihood._compact_lss_q — a
            # mis-stamped global table would be consumed positionally).
            if unique_pixels is not None and int(full_j.shape[0]) != int(
                    jnp.asarray(unique_pixels).shape[0]):
                raise ValueError(
                    f"LSS completion table is stamped 'compact' but has "
                    f"{int(full_j.shape[0])} rows for "
                    f"{int(jnp.asarray(unique_pixels).shape[0])} union pixels; "
                    "a compact table must be row-aligned with this run's "
                    "union pixel set. The builders always emit 'global' "
                    "tables — rebuild the completion or fix the stamp."
                )
            return barrier(full_j), 1
        if unique_pixels is None:
            return barrier(full_j), idx
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    def _compact_lss_members_for(views, unique_pixels):
        # Per-catalog analogue of make_likelihood._compact_lss_members: slice
        # this catalog's (M, n_pix, n_grid) Q ensemble to the view's pixels.
        full = views.lss_completion_logq_members
        if full is None:
            return None
        full_j = jnp.asarray(full)
        idx = int(views.lss_completion_indexing or 0)
        if idx == 1:
            if unique_pixels is not None and int(full_j.shape[1]) != int(
                    jnp.asarray(unique_pixels).shape[0]):
                raise ValueError(
                    f"LSS completion ensemble is stamped 'compact' but has "
                    f"{int(full_j.shape[1])} pixel rows for "
                    f"{int(jnp.asarray(unique_pixels).shape[0])} union pixels; "
                    "a compact ensemble must be row-aligned with this run's "
                    "union pixel set. The builders always emit 'global' "
                    "tables — rebuild the completion or fix the stamp."
                )
            return barrier(full_j)
        if unique_pixels is None:
            return barrier(full_j)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[1]:
            raise ValueError(
                f"LSS completion ensemble has {full_j.shape[1]} pixels but a "
                f"catalog pixel index reaches {int(jnp.max(up))} (rebuild Q "
                "over the full nside)."
            )
        return barrier(full_j[:, up])

    em_catalogs_pe = []
    em_catalogs_sel = []
    pe_pixel_cols = []
    sel_pixel_cols = []
    for bundle_idx, bundle in enumerate(bundles):
        views = prepare_catalog_views(
            opts,
            bundle,
            universe_model,
            counterpart_pixel=None,
            cache_builder=build_pixel_kde_cache,
        )
        # Each catalog uses ITS OWN pixel area (its own nside): sharing a single
        # apix across catalogs of different resolutions would silently bias the
        # per-pixel galaxy densities that enter the completion.
        apix_k = bundle["apix"]
        # Bundle union (loaders.py compacts PE-union-selection once): the two
        # views share one galaxy table / cache / unique-pixel array, so slice the
        # Q table, Q ensemble and mark rows ONCE and alias them to both
        # EMCatalogs (IS-identical leaves).  Detected by identity on the aliased
        # view arrays; a non-union bundle (e.g. a hand-built test fixture with
        # distinct PE/sel rows) keeps the separate per-view compaction below.
        bundle_union = (
            views.zgals_pe_catalog is views.zgals_sel_catalog
            and views.unique_pixels_pe is views.unique_pixels_sel
        )
        lss_q_pe_k, lss_idx_pe_k = _compact_lss_q_for(views, views.unique_pixels_pe)
        lss_qm_pe_k = _compact_lss_members_for(views, views.unique_pixels_pe)
        field_k = _bundle_field_inputs(bundle)

        def _bundle_marks(unique_pixels):
            # Gather this bundle's full-sky z-centred mark tables (loaded by
            # load_multitracer_catalog_bundles) to the view's compact rows,
            # mirroring make_likelihood._compact_marks.
            from darksirens.marks import MARK_FIELDS as _MARK_FIELDS

            out = {}
            for field in _MARK_FIELDS.values():
                full = bundle.get(field)
                if full is None:
                    out[field] = None
                else:
                    full_j = jnp.asarray(full)
                    arr = (
                        full_j if unique_pixels is None
                        else full_j[jnp.asarray(unique_pixels)]
                    )
                    out[field] = barrier(arr)
            return out

        marks_pe_k = _bundle_marks(views.unique_pixels_pe)
        # Eager saturation guard: an uncentred mark table pins log h to the
        # clip rail across the whole eta prior, so the eta posterior would come
        # back flat with nothing anywhere reporting a problem.  Checked on the
        # COMPACT rows the model actually reads, per catalog (each carries its
        # own selected marks), before anything is traced.
        _check_bundle_marks(
            mark_names_all[bundle_idx] if bundle_idx < len(mark_names_all) else (),
            marks_pe_k, views.ngals_pe_catalog, field_k,
            where=f"catalog {bundle_idx + 1} PE view",
        )
        if bundle_union:
            lss_q_sel_k, lss_idx_sel_k = lss_q_pe_k, lss_idx_pe_k
            lss_qm_sel_k = lss_qm_pe_k
            marks_sel_k = marks_pe_k
        else:
            lss_q_sel_k, lss_idx_sel_k = _compact_lss_q_for(
                views, views.unique_pixels_sel
            )
            lss_qm_sel_k = _compact_lss_members_for(views, views.unique_pixels_sel)
            marks_sel_k = _bundle_marks(views.unique_pixels_sel)
            _check_bundle_marks(
                mark_names_all[bundle_idx] if bundle_idx < len(mark_names_all) else (),
                marks_sel_k, views.ngals_sel_catalog, field_k,
                where=f"catalog {bundle_idx + 1} selection view",
            )

        em_catalogs_pe.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_pe_catalog,
            dzgals=views.dzgals_pe_catalog,
            wgals=views.wgals_pe_catalog,
            ngals=views.ngals_pe_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_pe,
            pixel_to_cache_idx=views.pixel_to_cache_idx_pe,
            unique_pixels=views.unique_pixels_pe,
            sample_to_unique_idx=views.sample_to_unique_pe,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_pe_k,
            lss_completion_logq_members=lss_qm_pe_k,
            lss_completion_indexing=lss_idx_pe_k,
            mark_logmstar=marks_pe_k["mark_logmstar"],
            mark_logssfr=marks_pe_k["mark_logssfr"],
            mark_metallicity=marks_pe_k["mark_metallicity"],
            mark_color=marks_pe_k["mark_color"],
            **field_k,
        ))
        em_catalogs_sel.append(EMCatalog(
            apix=apix_k,
            zgals=views.zgals_sel_catalog,
            dzgals=views.dzgals_sel_catalog,
            wgals=views.wgals_sel_catalog,
            ngals=views.ngals_sel_catalog,
            delta_g_pix_z=views.delta_g_pix_z,
            dN_obs_kde=views.dN_obs_kde_sel,
            pixel_to_cache_idx=views.pixel_to_cache_idx_sel,
            unique_pixels=views.unique_pixels_sel,
            sample_to_unique_idx=views.sample_to_unique_sel,
            active_counterpart_index=0,
            bright_siren_sky_marginalized=False,
            lss_completion_logq=lss_q_sel_k,
            lss_completion_logq_members=lss_qm_sel_k,
            lss_completion_indexing=lss_idx_sel_k,
            mark_logmstar=marks_sel_k["mark_logmstar"],
            mark_logssfr=marks_sel_k["mark_logssfr"],
            mark_metallicity=marks_sel_k["mark_metallicity"],
            mark_color=marks_sel_k["mark_color"],
            **field_k,
        ))
        pe_pixel_cols.append(jnp.asarray(views.sample_to_unique_pe, dtype=jnp.int32))
        sel_pixel_cols.append(jnp.asarray(views.sample_to_unique_sel, dtype=jnp.int32))

    # Stack the per-catalog compact pixel maps into (N, K) matrices.
    pixels_pe = barrier(jnp.stack(pe_pixel_cols, axis=1))
    pixels_sel = barrier(jnp.stack(sel_pixel_cols, axis=1))

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=None,
    )

    gw_pe = GWEvent(
        m1det=m1det_pe,
        m2det=m2det_pe,
        dL=dL_pe,
        chieff=chieff_pe,
        prior_wt=p_pe,
        pixels=pixels_pe,
        q=q_pe,
        valid=jnp.ones_like(dL_pe, dtype=bool),
        nx=nx_pe,
        ny=ny_pe,
        nz=nz_pe,
        spin=spin_pe,
    )
    gw_sel = GWEvent(
        m1det=m1det_sel,
        m2det=m2det_sel,
        dL=dL_sel,
        chieff=chieff_sel,
        prior_wt=p_draw,
        pixels=pixels_sel,
        q=q_sel,
        valid=jnp.ones_like(dL_sel, dtype=bool),
        nx=nx_sel,
        ny=ny_sel,
        nz=nz_sel,
        spin=spin_sel,
    )
    if sel_batch_size is not None:
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

    em_catalog_pe_0 = em_catalogs_pe[0]
    em_catalog_sel_0 = em_catalogs_sel[0]
    mixture_em_catalogs_pe = tuple(em_catalogs_pe[1:])
    mixture_em_catalogs_sel = tuple(em_catalogs_sel[1:])

    # Decide PE/selection state sharing EAGERLY (concrete EMCatalogs, before jit
    # erases object identity): a union bundle's two views share every consumed
    # leaf, so darksiren_log_likelihood builds each catalog's redshift-prior
    # state once instead of twice.  Static (a tuple of bools).
    share_prior_state_by_catalog = redshift_prior_state_sharing(
        universe_model, em_catalogs_pe, em_catalogs_sel
    )
    # Same eager, pre-jit vantage point: refuse a marked-host model whose
    # view-level mu_miss(z|eta) would differ between the two seams.
    require_view_independent_mu_miss(
        mark_model, mark_names_all, catalog_sky_weighting,
        em_catalogs_pe, em_catalogs_sel,
    )

    # Device operands travel as jit ARGUMENTS (see :func:`_jit_likelihood_body` for
    # why closing over them would embed them as HLO constants instead).
    operands = (
        gw_pe, em_catalog_pe_0, gw_sel, em_catalog_sel_0,
        mixture_em_catalogs_pe, mixture_em_catalogs_sel,
    )

    # Arm the windowed catalog-KDE evaluator for the traced path: the catalogs
    # cross the jit boundary as ARGUMENTS, so the evaluator sees tracers and
    # cannot verify the z-sort invariant itself — verify every bound view HERE,
    # while the arrays are concrete (see catalog.attest_rows_sorted_for_windowing).
    attest_rows_sorted_for_windowing(
        em_catalog_pe_0, em_catalog_sel_0,
        *mixture_em_catalogs_pe, *mixture_em_catalogs_sel,
    )

    def _body(coord: jnp.ndarray, operands) -> jnp.ndarray:
        (
            gw_pe_, em_catalog_pe_0_, gw_sel_, em_catalog_sel_0_,
            mixture_em_catalogs_pe_, mixture_em_catalogs_sel_,
        ) = operands
        if n_catalogs >= 2:
            (
                cosmo,
                surveys,
                pop_params,
                sky_params,
                mark_params_all,
                log_w,
            ) = parameter_decoder.decode_mixture(coord)
            survey_0 = surveys[0]
            mixture_surveys = tuple(surveys[1:])
            mark_params = mark_params_all[0]
        else:
            # K = 1 bundle source: the plain decoder (no sticks, no per-catalog
            # blocks) -- the same parameters a flat-data K=1 run samples.
            cosmo, survey_0, pop_params, sky_params, mark_params = (
                parameter_decoder.decode(coord)
            )
            mixture_surveys = ()
            log_w = None
            mark_params_all = (mark_params,)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}."
            )
        return darksiren_log_likelihood(
            cosmo,
            survey_0,
            pop_params,
            gw_pe_,
            em_catalog_pe_0_,
            gw_sel_,
            em_catalog_sel_0_,
            nEvents,
            nsamp,
            Ndraw,
            pop_model,
            universe_model,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            sky_model=sky_model,
            sky_params=sky_params,
            mark_model=mark_model,
            mark_params=mark_params,
            mark_names=mark_names,
            mark_params_all=tuple(mark_params_all),
            mark_names_all=mark_names_all,
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            lss_marginalize=bool(getattr(opts, "lss_marginalize", False)),
            n_catalogs=n_catalogs,
            mixture_surveys=mixture_surveys,
            mixture_em_catalogs_pe=mixture_em_catalogs_pe_,
            mixture_em_catalogs_sel=mixture_em_catalogs_sel_,
            mixture_log_weights=log_w,
            catalog_sky_weighting=catalog_sky_weighting,
            share_prior_state_by_catalog=share_prior_state_by_catalog,
        )

    return _jit_likelihood_body(_body, operands)


def make_likelihood(opts, data: dict, pop_params_fid, fixed_parameter_values: dict | None = None):
    """
    Build and return the likelihood callable for the sampler.

    This wrapper prepares static catalog/GW views, decodes sampler coordinates,
    and delegates the pure JIT likelihood evaluation to
    :func:`darksirens.likelihood.core.darksiren_log_likelihood`.
    """
    # Flow-surrogate path: per-event normalizing flows replace the stored PE
    # samples (mutually exclusive with --gw_path; validated by the CLI).
    if data.get("flow_ensemble") is not None:
        from darksirens.likelihood.flow_events import make_flow_likelihood

        return make_flow_likelihood(
            opts, data, pop_params_fid, fixed_parameter_values
        )

    nEvents = data["nEvents"]
    nsamp = data["nsamp"]
    Ndraw = data["Ndraw"]
    apix = data["apix"]
    pop_model = opts.pop_model
    shared_beta = bool(getattr(opts, "shared_beta", True))
    shared_spin = bool(getattr(opts, "shared_spin", True))
    shared_gamma = bool(getattr(opts, "shared_gamma", True))
    universe_model = opts.universe_model
    sel_batch_size = require_resolved_block_size(
        "sel_batch_size", getattr(opts, "sel_batch_size", None))
    pe_event_block = require_resolved_block_size(
        "pe_event_block", getattr(opts, "pe_event_block", None))
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
    materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    selection_neff_soft_guard = bool(getattr(opts, "selection_neff_soft_guard", False))
    max_likelihood_variance = float(getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE))
    catalog_sky_weighting = getattr(opts, "catalog_sky_weighting", "conditional")

    # Weak-lensing magnification backend (resolved up front, before the heavy
    # catalog-view prep, so a missing WL config fails fast).  All values are
    # inert when wl_backend == WL_BACKEND_DISABLED, preserving behaviour for
    # every non-WL universe model.
    wl_backend = WL_BACKEND_DISABLED
    wl_a = jnp.asarray(0.0)
    wl_b = jnp.asarray(0.0)
    wl_z_grid = jnp.asarray([0.0, 1.0])
    wl_log_mu_grid = jnp.asarray([0.0, 1.0])
    wl_log_p_table = jnp.asarray([[0.0, 0.0], [0.0, 0.0]])
    wl_params = data.get("wl_params")
    if universe_model == "spectral_sirens_wl":
        if wl_params is None:
            raise ValueError(
                "universe_model='spectral_sirens_wl' requires data['wl_params'] "
                "to be present."
            )
        backend = int(wl_params.backend)
        if backend == WL_BACKEND_LOGNORMAL:
            wl_backend = WL_BACKEND_LOGNORMAL
            wl_a = jnp.asarray(wl_params.a)
            wl_b = jnp.asarray(wl_params.b)
        elif backend == WL_BACKEND_TABULATED:
            wl_backend = WL_BACKEND_TABULATED
            wl_z_grid = jnp.asarray(wl_params.z_grid)
            wl_log_mu_grid = jnp.asarray(wl_params.log_mu_grid)
            wl_log_p_table = jnp.asarray(wl_params.log_p_table)
        else:
            raise ValueError(
                "Unsupported weak-lensing backend in data['wl_params']: "
                f"{backend}. Expected {WL_BACKEND_LOGNORMAL} (LOGNORMAL) or "
                f"{WL_BACKEND_TABULATED} (TABULATED)."
            )
    # Selection-side WL marginalization (opt-in; lognormal backend only —
    # the core silently keeps the legacy selection path otherwise, mirroring
    # the cluster wrapper's semantics).
    wl_selection = (
        WL_SELECTION_LOGNORMAL
        if getattr(opts, "wl_selection", "standard") == "wl_lognormal"
        else WL_SELECTION_STANDARD
    )

    counterpart_pixel = data.get("counterpart_pixel")
    counterpart_pixels = (
        barrier(jnp.asarray(data["counterpart_pixels"], dtype=jnp.int32))
        if data.get("counterpart_pixels") is not None else None
    )
    counterpart_zs = (
        barrier(jnp.asarray(data["counterpart_zs"], dtype=float))
        if data.get("counterpart_zs") is not None else None
    )
    counterpart_dzs = (
        barrier(jnp.asarray(data["counterpart_dzs"], dtype=float))
        if data.get("counterpart_dzs") is not None else None
    )
    bright_siren_sky_marginalized = bool(
        data.get(
            "bright_siren_sky_marginalized",
            getattr(opts, "bright_siren_sky_marginalized", False),
        )
    )

    # Bundle operand source: per-catalog bundles in data["catalogs"] (built by
    # loaders.load_multitracer_catalog_bundles) for ANY K >= 1; required for
    # K >= 2, where the catalog-completed redshift prior becomes a per-catalog
    # mixture.  The flat-data path below stays for K = 1 callers without
    # bundles and is bit-identical to the pre-unification behaviour.
    n_catalogs = int(getattr(opts, "n_catalogs", 1))
    if data.get("catalogs") is not None or n_catalogs >= 2:
        # The latent seam is wired on the FLAT (single-catalog) path only: each
        # catalog of a K >= 2 mixture has its own footprint, depth map and
        # anchor, so one artifact cannot serve them and silently reusing it
        # would apply catalog 1's field to catalog 2's rows.  Refuse rather
        # than ignore the flag (PR-5 scope; the mixture wiring is its own PR).
        if _resolve_lss_field_mode(opts)[0] == "latent":
            raise NotImplementedError(
                "lss_field_mode='latent' is not wired on the K >= 2 mixture "
                "path: each catalog has its own footprint and depth map, so it "
                "needs its own anchor artifact, and reusing one across "
                "catalogs would evaluate one catalog's field on another's "
                "rows. Run the latent seam with a single catalog, or use "
                "--lss_field_mode table."
            )
        return _make_mixture_likelihood(
            opts, data, pop_params_fid, fixed_parameter_values, n_catalogs
        )

    catalogs = prepare_catalog_views(
        opts,
        data,
        universe_model,
        counterpart_pixel,
        cache_builder=build_pixel_kde_cache,
    )

    # The latent seam (field-level PR-5).  Resolved here, right after the views
    # exist and long before the first likelihood evaluation: the artifact is
    # read, its guards fire, and its leaves are barriered on the host.  Both
    # dicts are EMPTY under the default lss_field_mode='table', so the two
    # EMCatalog(...) constructions below are unchanged there.
    # The run's resolved per-catalog depth (cli.inference.resolve_survey_z_depth,
    # whose result is what parameters.py hands the SurveyParams).  This is the
    # flat K = 1 path, so element 0 is this catalog's; an unset list is the
    # legacy full-grid convention (no depth), and the artifact must have been
    # built the same way -- load_latent_plan checks exactly that.
    _resolved_depths = list(getattr(opts, "resolved_survey_z_depths", None) or ())
    lss_field_mode, latent_leaves_pe, latent_leaves_sel = _resolve_latent_leaves(
        opts,
        catalogs,
        _resolved_depths[0] if _resolved_depths else None,
        data.get("nside"),
        int(jnp.asarray(catalogs.zgals_pe_catalog).shape[0]),
        int(jnp.asarray(catalogs.zgals_sel_catalog).shape[0]),
    )

    # Slice the (global, host-side) Q_LSS table to each view's union pixels, so
    # only the compact (n_union, n_grid) block becomes a device/jit operand
    # rather than the full (n_pix, n_grid) table.  Returns (compact_logq, indexing).
    def _compact_lss_q(unique_pixels):
        full = catalogs.lss_completion_logq
        if full is None:
            return None, 0
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1:
            # Stamped compact: rows must already be union-pixel-aligned. A
            # mis-stamped GLOBAL table (all builders emit global) would have
            # its rows consumed positionally as union rows — wrong Q per
            # event when the shapes coincide, a traced-index crash when they
            # don't — so the row count is validated here, before JIT.
            if unique_pixels is not None and int(full_j.shape[0]) != int(
                    jnp.asarray(unique_pixels).shape[0]):
                raise ValueError(
                    f"LSS completion table is stamped 'compact' but has "
                    f"{int(full_j.shape[0])} rows for "
                    f"{int(jnp.asarray(unique_pixels).shape[0])} union pixels; "
                    "a compact table must be row-aligned with this run's "
                    "union pixel set. The builders always emit 'global' "
                    "tables — rebuild the completion or fix the stamp."
                )
            return barrier(full_j), 1
        if unique_pixels is None:
            # legacy full catalog (rows are global pixels)
            return barrier(full_j), idx
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[0]:
            raise ValueError(
                f"LSS completion table has {full_j.shape[0]} rows but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[up]), 1

    # Flat union path: prepare_catalog_views aliases the PE and selection views
    # onto ONE union galaxy table / unique-pixel array, so every per-view slice
    # below is a pure function of the SAME arguments.  Slice ONCE and alias, as
    # _make_mixture_likelihood's ``bundle_union`` does: the redshift-prior-state
    # sharing verdict tests ``is`` identity on exactly these leaves, so
    # re-slicing collapsed sharing to False -- and doubled the (M, N_rows,
    # N_grid) prior-state precomputation -- on every run carrying a Q table, a Q
    # ensemble or marks.  Detected by identity, never by value.
    union_views = (
        catalogs.unique_pixels_pe is catalogs.unique_pixels_sel
        and catalogs.zgals_pe_catalog is catalogs.zgals_sel_catalog
    )

    lss_q_pe, lss_idx_pe = _compact_lss_q(catalogs.unique_pixels_pe)
    lss_q_sel, lss_idx_sel = (
        (lss_q_pe, lss_idx_pe) if union_views
        else _compact_lss_q(catalogs.unique_pixels_sel)
    )

    # Slice the (optional) Q_LSS ENSEMBLE (M, n_pix, n_grid) to each view's union
    # pixels the same way, for the fully-Bayesian marginalisation (--lss_marginalize).
    def _compact_lss_members(unique_pixels):
        full = catalogs.lss_completion_logq_members
        if full is None:
            return None
        full_j = jnp.asarray(full)
        idx = int(catalogs.lss_completion_indexing or 0)
        if idx == 1:
            # Same pre-JIT row-alignment validation as _compact_lss_q, on the
            # ensemble's pixel axis.
            if unique_pixels is not None and int(full_j.shape[1]) != int(
                    jnp.asarray(unique_pixels).shape[0]):
                raise ValueError(
                    f"LSS completion ensemble is stamped 'compact' but has "
                    f"{int(full_j.shape[1])} pixel rows for "
                    f"{int(jnp.asarray(unique_pixels).shape[0])} union pixels; "
                    "a compact ensemble must be row-aligned with this run's "
                    "union pixel set. The builders always emit 'global' "
                    "tables — rebuild the completion or fix the stamp."
                )
            return barrier(full_j)
        if unique_pixels is None:
            return barrier(full_j)
        up = jnp.asarray(unique_pixels, dtype=jnp.int32)
        if int(jnp.max(up)) >= full_j.shape[1]:
            raise ValueError(
                f"LSS completion ensemble has {full_j.shape[1]} pixels but a catalog "
                f"pixel index reaches {int(jnp.max(up))} (rebuild Q over the full nside)."
            )
        return barrier(full_j[:, up])

    lss_qm_pe = _compact_lss_members(catalogs.unique_pixels_pe)
    lss_qm_sel = (
        lss_qm_pe if union_views
        else _compact_lss_members(catalogs.unique_pixels_sel)
    )
    lss_marginalize = bool(getattr(opts, "lss_marginalize", False))

    # Per-galaxy marks: gathered to the compact catalog rows using the SAME
    # unique-pixel map that compacts zgals, so they align row-for-row.  None
    # (mark absent) flows through to the legacy galaxy-count host model.
    from darksirens.marks import MARK_FIELDS as _MARK_FIELDS

    def _compact_marks(unique_pixels):
        out = {}
        for field in _MARK_FIELDS.values():
            full = data.get(field)
            if full is None:
                out[field] = None
            else:
                full = jnp.asarray(full)
                arr = full if unique_pixels is None else full[jnp.asarray(unique_pixels)]
                out[field] = barrier(arr)
        return out

    def _check_marks(marks, ngals_k, *, where):
        """Eager saturation guard — see the K>=2 twin (_check_bundle_marks).

        An uncentred mark table pins log h to the clip rail across the whole
        eta prior; the eta posterior then comes back flat with nothing
        downstream reporting a fault, so it has to be caught here at build time.
        """
        if mark_model in (None, "none") or not mark_names:
            return
        from darksirens.marks import (
            check_flat_marks_centred, check_marks_centred, get_mark_model,
        )

        model = get_mark_model(mark_model, mark_names)
        check_marks_centred(
            model,
            {name: marks[_MARK_FIELDS[name]] for name in mark_names},
            ngals_k,
            where=where,
        )
        check_flat_marks_centred(
            model, getattr(catalogs, "field_mark_values", None), where=where
        )

    marks_pe = _compact_marks(catalogs.unique_pixels_pe)
    _check_marks(marks_pe, catalogs.ngals_pe_catalog, where="PE catalog view")
    if union_views:
        marks_sel = marks_pe
    else:
        marks_sel = _compact_marks(catalogs.unique_pixels_sel)
        _check_marks(
            marks_sel, catalogs.ngals_sel_catalog, where="selection catalog view"
        )

    m1det_pe = barrier(_to_jax(data, "m1det"))
    m2det_pe = barrier(_to_jax(data, "m2det"))
    dL_pe = barrier(_to_jax(data, "dL"))
    chieff_pe = barrier(_to_jax(data, "chieff"))
    p_pe = barrier(_to_jax(data, "p_pe"))
    pixels_pe = catalogs.sample_to_unique_pe
    q_pe = barrier(m2det_pe / m1det_pe)
    nx_pe = barrier(_to_jax(data, "nx_pe"))
    ny_pe = barrier(_to_jax(data, "ny_pe"))
    nz_pe = barrier(_to_jax(data, "nz_pe"))

    m1det_sel = barrier(_to_jax(data, "m1detsels"))
    m2det_sel = barrier(_to_jax(data, "m2detsels"))
    dL_sel = barrier(_to_jax(data, "dLsels"))
    chieff_sel = barrier(_to_jax(data, "chieffsels"))
    p_draw = barrier(_to_jax(data, "p_draw"))
    pixels_sel = catalogs.sample_to_unique_sel
    q_sel = barrier(m2det_sel / m1det_sel)
    nx_sel = barrier(_to_jax(data, "nx_sel"))
    ny_sel = barrier(_to_jax(data, "ny_sel"))
    nz_sel = barrier(_to_jax(data, "nz_sel"))

    # Optional extra-spin block (DS-07): (N, d) columns beyond chieff, present
    # only for a non-chieff parameter space (basis negotiation, DS-09).  None
    # keeps the GWEvent pytree structure -- and every compiled likelihood --
    # identical to a build without the field.
    spin_pe = (barrier(jnp.asarray(data["spin_pe"], dtype=jnp.float64))
               if data.get("spin_pe") is not None else None)
    spin_sel = (barrier(jnp.asarray(data["spin_sel"], dtype=jnp.float64))
                if data.get("spin_sel") is not None else None)

    parameter_decoder = build_parameter_decoder(
        opts,
        pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
        wl_params=data.get("wl_params"),
    )

    # The PE / selection EMCatalogs are coord-INDEPENDENT (every field is a
    # static, already-barriered array), so build them ONCE here rather than on
    # every likelihood call.  Building them eagerly also resolves the
    # redshift-prior-state sharing verdict NOW, before jit erases the leaf object
    # identity it depends on: on the flat union path PE and selection share every
    # consumed leaf, so the state is built once instead of twice.
    em_catalog_pe = EMCatalog(
        apix=apix,
        zgals=catalogs.zgals_pe_catalog,
        dzgals=catalogs.dzgals_pe_catalog,
        wgals=catalogs.wgals_pe_catalog,
        ngals=catalogs.ngals_pe_catalog,
        delta_g_pix_z=catalogs.delta_g_pix_z,
        dN_obs_kde=catalogs.dN_obs_kde_pe,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_pe,
        unique_pixels=catalogs.unique_pixels_pe,
        sample_to_unique_idx=catalogs.sample_to_unique_pe,
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        active_counterpart_index=0,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        lss_completion_logq=lss_q_pe,
        lss_completion_logq_members=lss_qm_pe,
        lss_completion_indexing=lss_idx_pe,
        mark_logmstar=marks_pe["mark_logmstar"],
        mark_logssfr=marks_pe["mark_logssfr"],
        mark_metallicity=marks_pe["mark_metallicity"],
        mark_color=marks_pe["mark_color"],
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=getattr(catalogs, "field_lss_q", None),
        field_lss_q_empty_sum=getattr(catalogs, "field_lss_q_empty_sum", None),
        # PER-MEMBER survey-global Q rows: required by the field-convention
        # scope guard (redshift/prior.py) whenever the catalog carries a Q
        # ENSEMBLE.  prepare_catalog_views builds them; omitting them here made
        # every K=1 `--lss_marginalize` run under the DEFAULT field weighting
        # abort with "requires the per-member survey-global Q rows" before the
        # first likelihood evaluation.
        field_lss_q_members=getattr(catalogs, "field_lss_q_members", None),
        field_lss_q_empty_sum_members=getattr(
            catalogs, "field_lss_q_empty_sum_members", None
        ),
        field_delta_g=getattr(catalogs, "field_delta_g", None),
        field_mark_z=getattr(catalogs, "field_mark_z", None),
        field_mark_w=getattr(catalogs, "field_mark_w", None),
        field_mark_values=getattr(catalogs, "field_mark_values", None),
        field_depth_z=getattr(catalogs, "field_depth_z", None),
        field_depth_dz=getattr(catalogs, "field_depth_dz", None),
        field_depth_c=getattr(catalogs, "field_depth_c", None),
        pixel_stratum_map=getattr(catalogs, "pixel_stratum_map", None),
        empty_stratum_counts=getattr(catalogs, "empty_stratum_counts", None),
        field_lss_q_empty_sum_strata=getattr(
            catalogs, "field_lss_q_empty_sum_strata", None),
        field_lss_q_empty_sum_strata_members=getattr(
            catalogs, "field_lss_q_empty_sum_strata_members", None),
        f_p_rows=getattr(catalogs, "f_p_rows_pe", None),
        field_f_p_occ=getattr(catalogs, "field_f_p_occ", None),
        field_f_p_empty_sum=getattr(catalogs, "field_f_p_empty_sum", None),
        # Latent seam (PR-5): empty in the default table mode, so this
        # construction is textually the shipped one there.
        **latent_leaves_pe,
    )
    em_catalog_sel = EMCatalog(
        apix=apix,
        zgals=catalogs.zgals_sel_catalog,
        dzgals=catalogs.dzgals_sel_catalog,
        wgals=catalogs.wgals_sel_catalog,
        ngals=catalogs.ngals_sel_catalog,
        delta_g_pix_z=catalogs.delta_g_pix_z,
        dN_obs_kde=catalogs.dN_obs_kde_sel,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_sel,
        unique_pixels=catalogs.unique_pixels_sel,
        sample_to_unique_idx=catalogs.sample_to_unique_sel,
        counterpart_pixel=counterpart_pixel,
        counterpart_pixels=counterpart_pixels,
        counterpart_zs=counterpart_zs,
        counterpart_dzs=counterpart_dzs,
        active_counterpart_index=0,
        bright_siren_sky_marginalized=bright_siren_sky_marginalized,
        lss_completion_logq=lss_q_sel,
        lss_completion_logq_members=lss_qm_sel,
        lss_completion_indexing=lss_idx_sel,
        mark_logmstar=marks_sel["mark_logmstar"],
        mark_logssfr=marks_sel["mark_logssfr"],
        mark_metallicity=marks_sel["mark_metallicity"],
        mark_color=marks_sel["mark_color"],
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=getattr(catalogs, "field_lss_q", None),
        field_lss_q_empty_sum=getattr(catalogs, "field_lss_q_empty_sum", None),
        # PER-MEMBER survey-global Q rows: required by the field-convention
        # scope guard (redshift/prior.py) whenever the catalog carries a Q
        # ENSEMBLE.  prepare_catalog_views builds them; omitting them here made
        # every K=1 `--lss_marginalize` run under the DEFAULT field weighting
        # abort with "requires the per-member survey-global Q rows" before the
        # first likelihood evaluation.
        field_lss_q_members=getattr(catalogs, "field_lss_q_members", None),
        field_lss_q_empty_sum_members=getattr(
            catalogs, "field_lss_q_empty_sum_members", None
        ),
        field_delta_g=getattr(catalogs, "field_delta_g", None),
        field_mark_z=getattr(catalogs, "field_mark_z", None),
        field_mark_w=getattr(catalogs, "field_mark_w", None),
        field_mark_values=getattr(catalogs, "field_mark_values", None),
        field_depth_z=getattr(catalogs, "field_depth_z", None),
        field_depth_dz=getattr(catalogs, "field_depth_dz", None),
        field_depth_c=getattr(catalogs, "field_depth_c", None),
        pixel_stratum_map=getattr(catalogs, "pixel_stratum_map", None),
        empty_stratum_counts=getattr(catalogs, "empty_stratum_counts", None),
        field_lss_q_empty_sum_strata=getattr(
            catalogs, "field_lss_q_empty_sum_strata", None),
        field_lss_q_empty_sum_strata_members=getattr(
            catalogs, "field_lss_q_empty_sum_strata_members", None),
        f_p_rows=getattr(catalogs, "f_p_rows_sel", None),
        field_f_p_occ=getattr(catalogs, "field_f_p_occ", None),
        field_f_p_empty_sum=getattr(catalogs, "field_f_p_empty_sum", None),
        # Latent seam (PR-5): the theta-free blocks are the SAME objects the PE
        # catalog carries (aliased, not re-barriered), so the prior-state
        # sharing verdict still sees identical leaves.
        **latent_leaves_sel,
    )
    share_prior_state_by_catalog = redshift_prior_state_sharing(
        universe_model, (em_catalog_pe,), (em_catalog_sel,)
    )
    require_view_independent_mu_miss(
        mark_model, (mark_names,), catalog_sky_weighting,
        (em_catalog_pe,), (em_catalog_sel,),
    )

    # The GW containers are coord-INDEPENDENT too — every field is an
    # already-barriered data constant — so build them, and apply the selection
    # padding, ONCE here rather than on every likelihood call.  Building them
    # inside the (eager) closure re-ran two ``jnp.ones_like`` allocations plus, via
    # ``make_gw_event`` inside ``pad_gw_event_to_multiple``, 11 concatenates and 11
    # optimization barriers over the ~1.07e6-injection arrays on EVERY sampler
    # evaluation: 13.9 ms and 83 MB of identical fresh device buffers per call at
    # the production ``sel_batch_size=32768`` (1,067,946 % 32768 != 0, so the
    # padding never takes its early return).  ``_make_mixture_likelihood`` has
    # always hoisted them; this is the unpropagated twin.
    gw_pe = GWEvent(
        m1det=m1det_pe,
        m2det=m2det_pe,
        dL=dL_pe,
        chieff=chieff_pe,
        prior_wt=p_pe,
        pixels=pixels_pe,
        q=q_pe,
        valid=jnp.ones_like(dL_pe, dtype=bool),
        nx=nx_pe,
        ny=ny_pe,
        nz=nz_pe,
        spin=spin_pe,
    )
    gw_sel = GWEvent(
        m1det=m1det_sel,
        m2det=m2det_sel,
        dL=dL_sel,
        chieff=chieff_sel,
        prior_wt=p_draw,
        pixels=pixels_sel,
        q=q_sel,
        valid=jnp.ones_like(dL_sel, dtype=bool),
        nx=nx_sel,
        ny=ny_sel,
        nz=nz_sel,
        spin=spin_sel,
    )
    if sel_batch_size is not None:
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)

    # Device operands are passed as jit ARGUMENTS, never closed over — see
    # :func:`_jit_likelihood_body`.
    operands = (
        gw_pe, em_catalog_pe, gw_sel, em_catalog_sel,
        wl_a, wl_b, wl_z_grid, wl_log_mu_grid, wl_log_p_table,
    )

    # Same attestation as the mixture factory: window the traced catalogs only
    # after verifying the concrete arrays here at build time.
    attest_rows_sorted_for_windowing(em_catalog_pe, em_catalog_sel)

    def _body(coord: jnp.ndarray, operands) -> jnp.ndarray:
        (
            gw_pe_, em_catalog_pe_, gw_sel_, em_catalog_sel_,
            wl_a_, wl_b_, wl_z_grid_, wl_log_mu_grid_, wl_log_p_table_,
        ) = operands
        cosmo, survey, pop_params, sky_params, mark_params = parameter_decoder.decode(coord)
        if len(pop_params) != len(parameter_decoder.pop_labels):
            raise ValueError(
                "Population parameter length mismatch before likelihood "
                f"evaluation: decoded {len(pop_params)} values but pop_model "
                f"'{pop_model}' expects {len(parameter_decoder.pop_labels)}. "
                "Verify parameter-space construction for this population model."
            )

        if shared_beta and shared_spin and shared_gamma:
            return darksiren_log_likelihood(
                cosmo,
                survey,
                pop_params,
                gw_pe_,
                em_catalog_pe_,
                gw_sel_,
                em_catalog_sel_,
                nEvents,
                nsamp,
                Ndraw,
                pop_model,
                universe_model,
                sel_batch_size=sel_batch_size,
                pe_event_block=pe_event_block,
                sky_model=sky_model,
                sky_params=sky_params,
                mark_model=mark_model,
                mark_params=mark_params,
                mark_names=mark_names,
                wl_backend=wl_backend,
                wl_a=wl_a_,
                wl_b=wl_b_,
                wl_z_grid=wl_z_grid_,
                wl_log_mu_grid=wl_log_mu_grid_,
                wl_log_p_table=wl_log_p_table_,
                wl_selection=wl_selection,
                lss_marginalize=lss_marginalize,
                lss_field_mode=lss_field_mode,
                materialize_redshift_prior_state=materialize_redshift_prior_state,
                selection_neff_soft_guard=selection_neff_soft_guard,
                max_likelihood_variance=max_likelihood_variance,
                catalog_sky_weighting=catalog_sky_weighting,
                share_prior_state_by_catalog=share_prior_state_by_catalog,
            )
        return darksiren_log_likelihood(
            cosmo,
            survey,
            pop_params,
            gw_pe_,
            em_catalog_pe_,
            gw_sel_,
            em_catalog_sel_,
            nEvents,
            nsamp,
            Ndraw,
            pop_model,
            universe_model,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
            sel_batch_size=sel_batch_size,
            pe_event_block=pe_event_block,
            sky_model=sky_model,
            sky_params=sky_params,
            mark_model=mark_model,
            mark_params=mark_params,
            mark_names=mark_names,
            wl_backend=wl_backend,
            wl_a=wl_a_,
            wl_b=wl_b_,
            wl_z_grid=wl_z_grid_,
            wl_log_mu_grid=wl_log_mu_grid_,
            wl_log_p_table=wl_log_p_table_,
            wl_selection=wl_selection,
            lss_marginalize=lss_marginalize,
            lss_field_mode=lss_field_mode,
            materialize_redshift_prior_state=materialize_redshift_prior_state,
            selection_neff_soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            catalog_sky_weighting=catalog_sky_weighting,
            share_prior_state_by_catalog=share_prior_state_by_catalog,
        )

    return _jit_likelihood_body(_body, operands)
