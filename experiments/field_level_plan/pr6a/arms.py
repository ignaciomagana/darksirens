"""The three PR-6a arms on one nside-16 realization, and the H0 grid scan.

Arms (PLAN §6.2 Tier B; the fourth, PR-6b theta-coupled, is demoted by K9 and
out of scope):

  ``latent_off``  c_mode=selection, no LSS field at all -- the production
                  ``sel`` configuration of ``experiments/desi_full259``.
  ``table``       the same, plus a shipped radial lognormal Q table
                  (``--mode radial --c-mode selection --n-members 8``,
                  matching ``data/fits/q_radial.h5``'s own diagnostics), and
                  WITHOUT ``f_p`` -- the pairing the loader used to refuse.
  ``table_fp``    the Q table WITH ``f_p`` (S-3): the arm that differs from
                  ``latent`` only by where Q comes from, which is what a
                  latent-vs-table comparison actually wants.
  ``latent``      ``lss_field_mode='latent'`` + the nside-16 anchor +
                  ``lss_marginalize=True`` -- PR-6a.

Everything except the LSS treatment is held identical across the arms,
including ``--per_pixel_completeness``: PR-2's per-pixel ``C_p = f_p C(z)``
is available in every ``c_mode``, latent mode REQUIRES it
(``factory.py``'s guard 6), and letting only the latent arm have it would
confound the field with the per-pixel completeness.

**The guard convention.**  These runs quote PR-0's clean arm --
``selection_neff_soft_guard=False``, ``max_likelihood_variance=1e6``, Vitale's
``5 N_obs`` floor retained -- for the reason PR-5b's report gives: the soft
guard replaces the likelihood with a ``-gate (100 + 2 N softplus(-log mu))``
wall that is a function of ``Neff`` and ``log mu``, both member-dependent, so
under it a member ensemble's spread is the wall's spread and not the
likelihood's.  The hard criterion is EVALUATED and recorded per arm so the
choice is visible rather than implied.

The ``ngals`` shim is PR-5b's, unchanged and still needed: ``inference/
loaders.py:1044`` reads ``data["ngals"]`` while ``inference/data.py:196``
stores the K=1 dark-siren counts as ``ngals_catalog``, so
``--per_pixel_completeness`` raises ``KeyError: 'ngals'`` on the real loader
path.  PR-6a ships no production code either, so the alias is installed only
around ``load_all_data``.
"""
from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import numpy as np

import world16 as W16

import jax
import jax.numpy as jnp


@contextlib.contextmanager
def ngals_key_shim():
    """PR-5b's workaround for the ``ngals``/``ngals_catalog`` loader defect."""
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


def make_opts(paths, arm, *, soft_guard=False, max_var=1e6, marginalize=True,
              mth_override=None, fit_override=None):
    o = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        survey_path=str(paths["catalog"]), gw_path=str(paths["gw"]),
        gwselection_path=str(paths["selection"]), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None, lss_marginalize=False,
        lss_field_mode="table", lss_field_artifact=None,
        per_pixel_completeness=str(mth_override or paths["mth"]),
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
    fit = str(fit_override or paths["selection_fit"])
    o.selection_fit = fit
    from darksirens.redshift.selection import load_selection_fit_json
    sel = load_selection_fit_json(fit)
    o.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
    if arm in ("table", "latent_off_nofp"):
        # HISTORICAL, and kept so the shipped comparisons stay reproducible:
        # the loader used to REFUSE ``--per_pixel_completeness`` together with
        # a Q table (no f_p-weighted empty-pixel Q budget existed) while
        # ``factory.py``'s guard 6 REQUIRED it in latent mode, so no
        # configuration existed in which the table arm and the latent arm
        # differed only by the field.  That is why ``latent_off_nofp`` exists
        # -- the table arm's own control, so the latent-vs-table shift could be
        # split into a field part and an f_p part.  S-3 lifted the refusal;
        # ``table_fp`` below is the arm that pairing makes possible, and it is
        # the one to use for a latent-vs-table comparison from now on.
        o.per_pixel_completeness = None
        # S-3: these two arms are the unmasked controls, and the mock's sky is
        # footprint-limited (1,854 occupied pixels of 3,072), so the loader's
        # guard would otherwise refuse them.
        o.allow_unmasked_footprint = True
    if arm in ("table", "table_fp"):
        o.lss_completion = str(paths["q_table"])
    elif arm in ("latent", "latent_bgal"):
        # ``latent_bgal`` is ``latent`` pointed at an anchor whose member
        # ensemble was drawn from ``H^-1 + s_b^2 v v^T`` instead of ``H^-1``
        # (PLAN §3.4, S-2).  The arms are IDENTICAL on the inference side --
        # same opts, same guards, same data -- and differ only in which
        # artifact file is loaded, which is the only way the b_gal
        # propagation can be isolated: it lives entirely in the draws.
        o.lss_field_mode = "latent"
        o.lss_field_artifact = str(paths["anchor_bgal" if arm == "latent_bgal"
                                          else "anchor"])
        o.lss_marginalize = bool(marginalize)
    elif arm not in ("latent_off", "latent_off_nofp", "table_fp"):
        raise ValueError(f"unknown arm {arm!r}")
    return o, sel


def fixed_values(paths, sel):
    cal = json.load(open(paths["n0_calibration"]))
    return {"Om0": float(W16.OM0), "sigma_kde": 0.003,
            "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
            "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
            "sigma_M": float(sel["sigma_M"])}


def build(paths, arm, *, data_cache=None, **kw):
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.data import load_all_data
    from darksirens.likelihood.factory import make_likelihood

    opts, sel = make_opts(paths, arm, **kw)
    # The cache key must carry EVERY loader-visible switch: the Q table, the
    # marginalization flag and the depth map are all read inside
    # ``load_all_data``, so keying on the file triple alone silently serves the
    # previous arm's data (found the hard way -- the table arm reproduced the
    # latent_off arm to the last bit because its Q table was never loaded).
    # ``lss_field_artifact`` is deliberately NOT in the key: the anchor is read
    # by ``likelihood/factory.py:187`` inside ``make_likelihood``, never by
    # ``load_all_data`` (grepped), so ``latent`` and ``latent_bgal`` share one
    # data object by construction -- which is what makes them a paired
    # comparison rather than two runs that happen to agree.
    key = (opts.survey_path, opts.gw_path, opts.gwselection_path,
           opts.per_pixel_completeness, opts.lss_completion,
           bool(opts.lss_marginalize), opts.selection_fit)
    if data_cache is not None and key in data_cache:
        data = data_cache[key]
    else:
        with ngals_key_shim():
            data = load_all_data(opts)
        if data_cache is not None:
            data_cache[key] = data
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    jax.clear_caches()
    pop_fid = get_fixed_population_params(opts.pop_model)
    logl = make_likelihood(opts, data, pop_fid,
                           fixed_parameter_values=fixed_values(paths, sel))
    return logl, opts, data


def scan_h0(logl, h0):
    vals = np.empty(h0.size)
    for i, h in enumerate(h0):
        vals[i] = float(logl(jnp.asarray([h])))
    return vals


def summarize(h0, vals):
    """Posterior summary on a flat H0 prior over the scan range."""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return dict(finite=int(ok.sum()), median=None)
    pdf = np.where(ok, np.exp(v - np.nanmax(v[ok])), 0.0)
    norm = np.trapz(pdf, h0)
    pdf = pdf / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                           * np.diff(h0))])
    cdf /= cdf[-1]
    q = lambda a: float(np.interp(a, cdf, h0))  # noqa: E731
    # A "sigma" that is meaningful on a possibly-skewed 1-D posterior: half
    # the 68% credible width.  Quoted because Tier B's acceptance is stated
    # in sigma and the 90% interval is what the gate itself names.
    ci68 = [q(0.16), q(0.84)]
    ci90 = [q(0.05), q(0.95)]
    return dict(finite=int(ok.sum()), median=q(0.5), map=float(h0[np.argmax(pdf)]),
                ci68=ci68, ci90=ci90,
                sigma=0.5 * (ci68[1] - ci68[0]),
                width90=ci90[1] - ci90[0],
                cdf_at_truth=float(np.interp(W16.H0_TRUE, h0, cdf)),
                logl=v.tolist())


DEFAULT_GRID = np.arange(20.0, 140.5, 1.0)
