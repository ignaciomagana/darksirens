"""Shared PR-5b harness: the production 259-event latent likelihood, per-member.

Everything PR-5b measures needs the SAME object -- the shipped production
likelihood of ``experiments/desi_full259`` with the latent leaves live
(``lss_field_mode='latent'``, ``lss_marginalize=True``) and nothing else
changed -- so it is built in one place here rather than copied into three
scripts.  Nothing under ``darksirens/`` is touched; PR-5b is a MEASUREMENT
rung (PLAN §7, "PR-5b -- member-spread measurement ... the deliverable is a
report plus two pins") and ships no production code.

Three things in this module are load-bearing and each is explained where it
is defined:

1. :func:`member_ll_patch` -- how the per-member ``ll_m`` vector is read out
   of a likelihood that reduces it internally.  PLAN §6.5 item 1 asks for
   "the ``ll_m`` vector at ``M_draw = 256``"; ``likelihood/core.py:1489``
   returns ``logsumexp(ll_members) - log M`` and never exposes the vector.
   The alternative -- 256 single-member evaluations per H0 node -- costs
   256 x 3.03 s = 13 min PER NODE against ~3.1 s for one M=256 call, i.e.
   7 h vs 2 min for the campaign, because the member-INDEPENDENT slice
   (population model, Jacobian, proposal reweighting, KDE) is 99.9% of the
   3027 ms baseline PR-0 measured and would be redone 256 times.

2. :func:`member_slice_patch` -- chunking the member axis.  Exact, because
   the members enter through ``jax.vmap``/``lax.scan`` with no cross-member
   coupling until the final reduction, which (1) removes.  It exists only to
   bound the transient memory of the member vmap, which scales 32x from the
   shipped ``M_draw = 8``.

3. :func:`clean_arm_opts` -- the variance-guard convention, which PR-0
   measured to be decisive on this line and which is stated in the report.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PR5B_DIR = Path(__file__).resolve().parent
PLAN_DIR = PR5B_DIR.parent
REPO_DIR = PLAN_DIR.parent.parent
FULL259 = REPO_DIR / "experiments" / "desi_full259"
if str(FULL259) not in sys.path:
    sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402  (pins DARKSIRENS_ZMAX=6.0; MUST precede darksirens)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

#: Candidate anchors, most-preferred first.  PR-5's ``latent_anchor_v2a.h5``
#: is the post-eq.(4)-f32-fix rebuild; ``pr4/latent_anchor_a.h5`` is the
#: PR-4 original.  The M=256 builds below are written next to this file.
ANCHOR_M8_CANDIDATES = (
    PLAN_DIR / "pr5" / "latent_anchor_v2a.h5",
    PLAN_DIR / "pr4" / "latent_anchor_a.h5",
)
ANCHOR_M256 = PR5B_DIR / "latent_anchor_m256.h5"
ANCHOR_MAP = PR5B_DIR / "latent_anchor_map_m1.h5"


def resolve_anchor_m8() -> Path:
    for p in ANCHOR_M8_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "no M_draw=8 anchor artifact found; looked for "
        + ", ".join(str(p) for p in ANCHOR_M8_CANDIDATES))


# --------------------------------------------------------------------- opts

def clean_arm_opts(artifact, *, soft_guard=False, max_var=1e6,
                   marginalize=True):
    """The production 259-event config, latent leaves live.

    Identical to ``experiments/desi_full259/run_h0_scans.py``'s ``sel`` arm
    (c_mode='selection', field sky weighting, plain injections, no cuts,
    theta at theta_hat) except for the three latent switches and the guard
    convention.

    **The guard convention is a decision, not a detail.**  PR-0 measured
    (REPORT.md item 3) that the hard GWTC-4/5 variance criterion fails at
    EVERY H0 node on this line -- ``pe_variance_sum = 0.2733`` inflates the
    required selection ``Neff`` to ~92k while the line delivers 31-36k -- so
    the hard guard returns ``-inf`` everywhere and the SOFT guard replaces
    the likelihood with a ~-1e6 nat wall term
    (``likelihood/selection.py``: ``-gate * (100 + 2 N softplus(-log mu))``).
    That wall is a function of ``Neff`` and ``log mu``, both of which are
    MEMBER-DEPENDENT, so under the soft guard the measured member spread
    would be the spread of the wall, not of the likelihood: the quantity
    PLAN §6.5 defines would not be what was measured.  PR-5b therefore
    quotes the ``*_nogv`` convention PR-0 introduced for exactly this reason
    -- soft guard OFF, variance cap lifted (``max_likelihood_variance =
    1e6``), Vitale ``5 N_obs`` floor retained -- and measures the soft-guard
    arm separately so the size of the difference is on the record.
    """
    from darksirens.redshift.selection import load_selection_fit_json

    sel = load_selection_fit_json(str(C.FIT_JSON))
    opts = SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None,
        lss_marginalize=bool(marginalize),
        # --- the latent switches (field-level PR-5) ---
        lss_field_mode="latent",
        lss_field_artifact=str(artifact),
        per_pixel_completeness=str(C.INGEST_DATA / "mth_map_nside128.h5"),
        # ----------------------------------------------
        c_mode="selection", catalog_sky_weighting="field",
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=bool(soft_guard),
        max_likelihood_variance=float(max_var),
    )
    opts.selection_fit = str(C.FIT_JSON)
    opts.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
    return opts


#: ``b_GW`` this campaign runs at.  PLAN §4.3's rule INVERSION makes
#: ``b_miss`` the GW bias in latent mode (``completion.latent_b_gw`` reads
#: ``survey.b_miss``), and the closed-form prediction was computed at
#: ``b_GW = 1``.
#:
#: It is NOT passed as a fixed value, and that is the second production gap
#: PR-5b found.  ``inference/parameters.build_parameter_decoder`` re-derives
#: the parameter space with ``use_lss=bool(opts.use_LSS)``
#: (``parameters.py:530``), so it applies the *table-mode* rule -- "b_miss is
#: inert with --use_lss off" -- and REFUSES a fixed ``b_miss`` with
#: ``ValueError: A fixed value was given for 'b_miss', but it is not a
#: sampled parameter of this configuration``.  The latent inversion
#: (``use_lss = opts.use_LSS or latent``) exists only in
#: ``cli/inference.py:3225``, i.e. it never reaches the factory, which is
#: what every non-CLI caller (this campaign, the scan scripts, the tests)
#: goes through.  So on the factory path ``b_GW`` is pinned at the
#: ``SurveyParams`` fiducial ``b_miss = 1.0``
#: (``core/constants.SURVEY_PARAMS_FID_BY_NAME``) and cannot be set to
#: anything else.  That fiducial happens to be exactly the value PR-5b wants,
#: so the campaign is unaffected -- but a ``b_GW != 1`` study is currently
#: CLI-only, and ``smoke_latent.py`` VERIFIES the decoded ``survey.b_miss``
#: rather than trusting the coincidence.
B_GW = 1.0


def fixed_values():
    """``theta_hat`` exactly as the shipped scans fix it.

    Deliberately does NOT contain ``b_miss``; see :data:`B_GW`.
    """
    from darksirens.redshift.selection import load_selection_fit_json

    sel = load_selection_fit_json(str(C.FIT_JSON))
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    return {
        "Om0": C.OM0, "sigma_kde": 0.003,
        "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
        "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
        "sigma_M": float(sel["sigma_M"]),
    }


# ------------------------------------------------------- the two monkeypatches

@contextlib.contextmanager
def member_ll_patch(expect_m: int):
    """Make the likelihood return the ``(M,)`` vector ``ll_m - log M``.

    ``_factored_member_marginalization`` ends at ``core.py:1489``

        return logsumexp(ll_members) - jnp.log(n_members)

    and the only thing between that and the caller is
    ``jnp.where(jnp.isfinite(ll), ll, -jnp.inf)`` (``core.py:1628``), which is
    elementwise.  Replacing the module-level ``logsumexp`` with the identity
    ON THAT CALL therefore turns the returned scalar into the member vector,
    with no other change to the computation: every ``ll_m`` is the number the
    shipped estimator would have summed.

    The interception is keyed on the call SIGNATURE, not on the value: the
    only other ``logsumexp`` in the module is the K-catalog mixture at
    ``core.py:770``, which always passes ``axis=0``.  A 1-D positional-only
    call of length ``M`` is the member reduction and nothing else.  The
    length check makes an accidental match loud rather than silent, and
    ``run_member_spread.py`` additionally GATES the patch by re-reducing the
    recovered vector and comparing against the unpatched scalar.
    """
    import darksirens.likelihood.core as core

    real = core.logsumexp
    seen = {"n": 0}

    def _patched(x, *a, **kw):
        if not a and not kw:
            arr = jnp.asarray(x)
            if arr.ndim == 1 and arr.shape[0] == int(expect_m):
                seen["n"] += 1
                return arr
            raise RuntimeError(
                "member_ll_patch intercepted a bare logsumexp of shape "
                f"{arr.shape}, expected ({expect_m},): the member reduction "
                "is not the only bare call on this path any more; re-read "
                "likelihood/core.py before trusting any PR-5b number.")
        return real(x, *a, **kw)

    core.logsumexp = _patched
    try:
        yield seen
    finally:
        core.logsumexp = real


@contextlib.contextmanager
def member_slice_patch(lo: int, hi: int):
    """Restrict the latent plan to members ``[lo, hi)`` at likelihood build.

    EXACT, not an approximation: the members enter the factored path through
    ``jax.vmap`` over ``_member_ll`` (``core.py:1486``) and ``lax.scan`` over
    ``latent_member_N_miss_integrals`` (``completion.py:1450``), neither of
    which couples members -- the ONLY cross-member operation is the final
    ``logsumexp``, which :func:`member_ll_patch` has already removed.  So
    ``ll_m`` computed in a chunk of 32 is bit-identical to ``ll_m`` computed
    in a chunk of 256, and ``run_member_spread.py`` gates exactly that.

    Chunking exists because the member vmap's transient memory scales
    linearly in ``M``: at the shipped ``M_draw = 8`` it is invisible, at 256
    it is 32x and the campaign runs on a 40 GB A100.  The alternative --
    writing 8 separate 32-member artifacts -- would break the common-random-
    numbers property this whole measurement rests on, because
    ``laplace_draws`` keys its normals on ``(n_draw // 2, M)`` and a
    different ``n_draw`` is a different draw set.  One 256-member artifact,
    sliced at load, keeps ONE draw set.
    """
    import darksirens.likelihood.factory as factory

    real = factory.load_latent_plan

    def _patched(path, **kw):
        plan = real(path, **kw)
        sl = slice(int(lo), int(hi))

        def _cut(x):
            return None if x is None else x[sl]

        return dataclasses.replace(
            plan, row_fac=_cut(plan.row_fac), A=_cut(plan.A),
            B=_cut(plan.B), dA=_cut(plan.dA), dB=_cut(plan.dB))

    factory.load_latent_plan = _patched
    try:
        yield
    finally:
        factory.load_latent_plan = real


# ------------------------------------------------------------------ building

def load_data(opts):
    """``load_all_data`` once.

    Split from :func:`build_likelihood` because the campaign rebuilds the
    LIKELIHOOD once per member chunk (the latent leaves are resolved inside
    ``make_likelihood``) but the DATA -- 259 events x 4096 samples, 1.07M
    injections, the nside-64 DESI union, the depth map -- does not depend on
    the artifact or on which members are sliced, and re-reading it eight
    times would dominate the campaign's wall.
    """
    from darksirens.inference.data import load_all_data

    with _ngals_key_shim():
        data = load_all_data(opts)
    return data


@contextlib.contextmanager
def _ngals_key_shim():
    """Work around a REAL production defect found by PR-5b; see the report.

    ``loaders.attach_selection_fraction_inputs`` (``loaders.py:1044``) reads

        ngals_full = np.asarray(data["ngals"])

    but ``inference/data.py`` never puts an ``"ngals"`` key in the K=1
    dark-siren ``data`` dict -- it stores the full-sky galaxy counts as
    ``"ngals_catalog"`` (``data.py:196``, alongside ``zgals``/``dzgals``/
    ``wgals`` which DO keep their bare names).  So on the real loader path
    ``--per_pixel_completeness`` raises ``KeyError: 'ngals'`` before it can
    attach ``f_p_map``.  The only coverage is
    ``tests/test_per_pixel_completeness.py:258``, which hand-builds
    ``data = dict(nside=2, ngals=np.zeros(48))`` and therefore cannot see it.

    This matters well beyond PR-5b: ``factory.py:398`` makes
    ``--per_pixel_completeness`` MANDATORY in latent mode (the artifact's
    ``B(z; b) = sum_p f_p e^{bf}`` and ``F_F = sum_p f_p`` are built from it),
    so as merged, ``--lss_field_mode latent`` cannot reach a single
    likelihood evaluation on any real catalog.

    PR-5b ships no production code, so the fix is NOT applied here -- the
    alias is installed only around ``load_all_data`` and removed afterwards,
    and the defect is reported for a one-line PR elsewhere.
    """
    from darksirens.inference import loaders

    real = loaders.attach_selection_fraction_inputs

    def _patched(opts, data):
        if "ngals" not in data and "ngals_catalog" in data:
            data = dict(data)
            data["ngals"] = data["ngals_catalog"]
        return real(opts, data)

    loaders.attach_selection_fraction_inputs = _patched
    try:
        yield
    finally:
        loaders.attach_selection_fraction_inputs = real


def build_likelihood(opts, data=None, *, fresh_trace=True):
    """``load_all_data`` + ``make_likelihood``, the shipped two-step.

    ``fresh_trace`` drops JAX's compilation caches first, and it is
    LOAD-BEARING for :func:`member_ll_patch`.  Both patches act at TRACE time,
    but ``darksiren_log_likelihood`` is jitted at module level and JAX keys its
    executable cache on the function object plus the argument avals -- neither
    of which changes when the module-level ``logsumexp`` is swapped.  So a
    patched build that follows an unpatched one with the same shapes gets a
    cache HIT, never re-traces, and silently returns the UNPATCHED scalar.

    That is not hypothetical: it is exactly what the first PR-5b smoke run
    did (``smoke_1136115``): ``intercepts = 0``, ``ll_m shape = ()``, and the
    "recovered member vector" was the scalar ``-766.7914443725264``, i.e. it
    reproduced the unpatched value to the last bit -- the gate S3 was written
    to catch precisely this and did.  Everything downstream would have been a
    34-node table of 1-member "ensembles" with ``sigma = 0``.
    """
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    if data is None:
        data = load_data(opts)
    if fresh_trace:
        jax.clear_caches()
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    pop_fid = get_fixed_population_params(opts.pop_model)
    return make_likelihood(opts, data, pop_fid,
                           fixed_parameter_values=fixed_values())


__all__ = [
    "C", "PR5B_DIR", "PLAN_DIR", "REPO_DIR", "ANCHOR_M256", "ANCHOR_MAP",
    "resolve_anchor_m8", "clean_arm_opts", "fixed_values", "member_ll_patch",
    "member_slice_patch", "build_likelihood", "load_data", "jnp", "np", "jax",
]
