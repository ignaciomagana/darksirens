"""Independent re-verification of six cells of the PR-6a guard matrix.

The build phase reported a 35-row matrix pinned by ``tests/test_latent_guards.py``.
This file does not run those tests and does not read them: it re-derives six
cells -- three REFUSED, three PERMITTED -- from the PRODUCTION command line
(``experiments/desi_full259/sbatch_ns_joint_sel.sh``, quoted verbatim below)
by parsing real ``argv`` with the shipped ``build_parser()`` and then calling
the shipped guards in the ORDER ``cli/inference.main`` calls them:

    main:4082  _check_latent_field_mode(opts)              guard 6, pre-load
    main:4087  _stamp_latent_artifact_fingerprint(opts)    guard 1, pre-load
    ...:3475   _check_latent_selection_family(opts)        post-fits
    ...:3523   build_parameter_space(...)                  the real one
    ...:3560   _check_latent_budget_anchor(...)            guard 5, post-pspace

Nothing is mocked except the two things a guard check cannot afford: the
259-event data load (the pre-load guards fire before it by construction, and
the post-load ones read only ``opts`` fields that ``_resolve_selection_fits``
stamps, which is done here from the same JSON) and the likelihood build.

Refusals are asserted on the MESSAGE, not the exception type, and the
substrings checked here were chosen from PLAN §4.4's own vocabulary rather
than copied from the implementation's tests.
"""
from __future__ import annotations

import io
import json
import contextlib
import sys
import tempfile
from pathlib import Path

PR6A = Path(__file__).resolve().parent
PLAN_DIR = PR6A.parent
REPO = PLAN_DIR.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "desi_full259"))

import common as C  # noqa: E402  (pins DARKSIRENS_ZMAX=6.0)

from darksirens.cli import inference as I  # noqa: E402
from darksirens.inference.prior import build_parameter_space  # noqa: E402

ANCHOR = PLAN_DIR / "pr5" / "latent_anchor_v2a.h5"
MTH = C.INGEST_DATA / "mth_map_nside128.h5"
CAL = json.load(open(C.DATA_DIR / "n0_calibration.json"))
FIXED = json.dumps({"Om0": 0.3089, "sigma_kde": 0.003,
                    "log10n0": CAL["log10n0"], "delta": CAL["delta"]})
FIXED_NO_BUDGET = json.dumps({"Om0": 0.3089, "sigma_kde": 0.003})

#: sbatch_ns_joint_sel.sh, verbatim minus the sampler/checkpoint block.
BASE = [
    "--gw_path", str(C.GW_259),
    "--gwselection_path", str(C.INJ_PLAIN),
    "--survey_path", str(C.SURVEY_N64),
    "--universe_model", "dark_sirens",
    "--pop_model", "gwtc5_fiducial_bpl2peaks",
    "--c_mode", "selection",
    "--selection_fit", str(C.FIT_JSON),
    "--catalog_sky_weighting", "field",
    "--use_lss", "false",
    "--fix_population", "false",
    "--fix_cosmology", "false",
    "--fix_de", "true",
    "--fix_survey", "false",
    "--prior_overrides", '{"H0": [20.0, 140.0]}',
    "--sampler", "tinyns", "--nlive", "1000",
    "--tinyns_preset", "adaptive_gpu",
    "--selection_neff_guard", "soft",
]
LATENT = ["--lss_field_mode", "latent",
          "--lss_field_artifact", str(ANCHOR),
          "--per_pixel_completeness", str(MTH)]

RESULTS = []


def parse(extra):
    """Real argv -> real opts, through ``main``'s own first two steps.

    ``main`` calls ``_normalize_multitracer_paths`` (which resolves
    ``n_catalogs`` and the per-catalog path lists) immediately before guard 6,
    so it runs here too; everything after guard 1 is called explicitly at the
    point of use, in ``main``'s order.
    """
    o = I.build_parser().parse_args(BASE + list(extra))
    I._normalize_multitracer_paths(o)
    return o


def _pspace(opts):
    """``main``'s own ``build_parameter_space`` call, latent inversion included."""
    # ``main`` resolves the mark names at :3397 from the LOADED catalog's
    # datasets; with ``--mark_model none`` (the production setting) that
    # resolution is unconditionally the empty tuple (:3400 refuses a
    # non-none model with no marks), so nothing data-dependent is being
    # short-circuited here.
    if not hasattr(opts, "mark_names"):
        assert getattr(opts, "mark_model", "none") in (None, "none")
        opts.mark_names = ()
        opts.mark_names_by_catalog = ((),)
    prior_overrides, fixed_values = I._parse_structured_options(opts)
    # main:3470.  The empty ``data`` is safe and not a shortcut: the bundle is
    # touched only by the stratum-map attachment at the end of the resolver,
    # which returns early unless a MULTI-stratum fit was given -- and a
    # multi-stratum fit is itself refused in latent mode
    # (_check_latent_selection_strata).  This is what stamps
    # opts.selection_prior (M0hat/sigma_M -> ("normal", loc, scale)) and
    # opts.selection_family, both of which guard 5 and the family guard read.
    I._resolve_selection_fits(opts, {}, fixed_values)
    latent = I.latent_field_mode(opts)
    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology,
        opts.fix_survey, fix_de=opts.fix_de,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_values,
        universe_model=opts.universe_model,
        shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
        mark_model=opts.mark_model, mark_names=opts.mark_names,
        n_catalogs=opts.n_catalogs, lss_completion_active=[False],
        use_lss=bool(getattr(opts, "use_LSS", False)) or latent,
        c_mode=opts.c_mode, selection_prior=opts.selection_prior,
        selection_family=getattr(opts, "selection_family", None),
    )
    return res[0], res[11], fixed_values


def refused(name, cell, fn, must_say):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn()
    except BaseException as exc:                       # SystemExit included
        msg = buf.getvalue() + "\n" + str(exc)
        missing = [s for s in must_say if s.lower() not in msg.lower()]
        RESULTS.append({"case": name, "cell": cell, "expected": "REFUSED",
                        "raised": type(exc).__name__,
                        "message_head": " ".join(msg.split())[:400],
                        "required_substrings": list(must_say),
                        "missing_substrings": missing,
                        "verdict": "PASS" if not missing else "MESSAGE-FAIL"})
        return
    RESULTS.append({"case": name, "cell": cell, "expected": "REFUSED",
                    "raised": None, "stdout_head":
                    " ".join(buf.getvalue().split())[:400],
                    "verdict": "FAIL -- DID NOT REFUSE"})


def permitted(name, cell, fn, expect=None):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            got = fn()
    except BaseException as exc:
        RESULTS.append({"case": name, "cell": cell, "expected": "PERMITTED",
                        "raised": type(exc).__name__,
                        "message_head": " ".join(
                            (buf.getvalue() + " " + str(exc)).split())[:400],
                        "verdict": "FAIL -- REFUSED A PERMITTED CELL"})
        return
    rec = {"case": name, "cell": cell, "expected": "PERMITTED", "raised": None,
           "stdout_head": " ".join(buf.getvalue().split())[:400],
           "verdict": "PASS"}
    if expect is not None:
        rec["observed"] = got
        rec["expected_value"] = expect
        rec["verdict"] = "PASS" if got == expect else "FAIL -- WRONG VALUE"
    RESULTS.append(rec)


def main():
    # ---------------------------------------------------------- REFUSED 1
    # latent + --c_mode per_pixel (guard 6, pre-load).  PLAN 4.4: per-pixel
    # completion is a second, differently conditioned completion model.
    o = parse(LATENT + ["--c_mode", "per_pixel"])
    refused("R1 latent + --c_mode per_pixel", "guard 6 / pre-load",
            lambda: I._check_latent_field_mode(o),
            ("per_pixel", "latent"))

    # ---------------------------------------------------------- REFUSED 2
    # latent with NO artifact (guard 6, pre-load).
    o = parse(["--lss_field_mode", "latent",
               "--per_pixel_completeness", str(MTH)])
    refused("R2 latent without --lss_field_artifact", "guard 6 / pre-load",
            lambda: I._check_latent_field_mode(o),
            ("lss_field_artifact",))

    # ---------------------------------------------------------- REFUSED 3
    # guard 1: a 64-hex digest that is not this artifact's.  Uses the real
    # anchor file, so the refusal comes from the RECOMPUTED digest, not from
    # a malformed-input shortcut.
    bad = "0" * 63 + "1"
    o = parse(LATENT + ["--lss_field_sha256", bad])
    I._check_latent_field_mode(o)                       # must pass first
    refused("R3 --lss_field_sha256 mismatch", "guard 1 / build",
            lambda: I._stamp_latent_artifact_fingerprint(o),
            ("sha256",))

    # ---------------------------------------------------------- REFUSED 4
    # guard 5: the production configuration with the budget pins REMOVED.
    # log10n0 and delta then come from prior.py's flat survey kind_map and
    # the anchor's shell totals are being fit by a free budget.
    o = parse(LATENT + ["--fixed_parameter_values", FIXED_NO_BUDGET])
    I._check_latent_field_mode(o)
    I._stamp_latent_artifact_fingerprint(o)
    labels, kinds, fv = _pspace(o)
    sampled_budget = [l for l in labels
                      if l.split("_c")[0] in ("log10n0", "delta")]
    refused(f"R4 flat sampled budget {sampled_budget}", "guard 5 / post-pspace",
            lambda: I._check_latent_budget_anchor(o, labels, kinds, fv),
            ("ZERO information", "shell totals", "K10", "calibrate_n0.py"))

    # ---------------------------------------------------------- REFUSED 5
    # schechter selection family, post-fits.
    o = parse(LATENT)
    # The family is a property of the --selection_fit JSON and this line
    # stamps it the way ``_resolve_selection_fits`` would for a schechter
    # fit; the union catalog has no schechter fit on disk, so the stamp is
    # the honest way to reach the cell without fabricating a fit file.
    o.selection_family = "schechter"
    refused("R5 schechter selection family", "post-fits",
            lambda: I._check_latent_selection_family(o),
            ("schechter", "M_faint_offset", "m_faint_cut"))

    # ---------------------------------------------------------- REFUSED 6
    # table mode carrying an orphan --lss_field_sha256.
    o = parse(["--lss_field_sha256", "0" * 64])
    refused("R6 table mode + orphan --lss_field_sha256", "pre-load",
            lambda: I._check_latent_field_mode(o),
            ("lss_field_sha256", "latent"))

    # -------------------------------------------------------- PERMITTED 1
    # The shipped PR-6a configuration: c_mode selection + f_p + the real
    # artifact + its own stamped digest.  Guard 6 and guard 1 both pass and
    # guard 1 STAMPS the provenance opts settings.json will carry.
    import h5py
    with h5py.File(ANCHOR, "r") as f:
        stamped = f["latent_field"].attrs["sha256"]
    o = parse(LATENT + ["--lss_field_sha256", str(stamped)])

    def _p1():
        I._check_latent_field_mode(o)
        I._stamp_latent_artifact_fingerprint(o)
        return (getattr(o, "lss_field_stored_sha256", None),
                bool(getattr(o, "lss_field_content_sha256", None)),
                getattr(o, "lss_field_b_gal", None))

    permitted("P1 latent + selection + f_p + matching sha256",
              "guard 6 + guard 1 / pre-load", _p1,
              expect=(str(stamped), True, 1.0))

    # -------------------------------------------------------- PERMITTED 2
    # The production budget anchoring: log10n0/delta pinned from
    # calibrate_n0.py through --fixed_parameter_values.  Guard 5 silent.
    o = parse(LATENT + ["--fixed_parameter_values", FIXED])
    I._check_latent_field_mode(o)
    I._stamp_latent_artifact_fingerprint(o)
    labels2, kinds2, fv2 = _pspace(o)

    def _p2():
        I._check_latent_budget_anchor(o, labels2, kinds2, fv2)
        return sorted(l for l in labels2
                      if l.split("_c")[0] in ("log10n0", "delta"))

    permitted("P2 budget pinned via --fixed_parameter_values (production)",
              "guard 5 / post-pspace", _p2, expect=[])

    # -------------------------------------------------------- PERMITTED 3
    # --c_mode aggregate in latent mode: PERMITTED, and specifically it does
    # NOT require a Q table, which is the one place the aggregate guard is
    # mode-routed.  Table mode with the same c_mode and no table is refused
    # by the same function, so both halves are exercised.
    o_agg = parse(LATENT + ["--c_mode", "aggregate"])
    o_tab = parse(["--c_mode", "aggregate"])

    def _p3():
        I._check_latent_field_mode(o_agg)
        I._check_aggregate_requires_q(o_agg, [False])
        I._check_latent_selection_family(o_agg)
        return "aggregate-latent-ok"

    permitted("P3 latent + --c_mode aggregate, no Q table",
              "pre-load", _p3, expect="aggregate-latent-ok")
    refused("R7 (control) TABLE + --c_mode aggregate, no Q table", "pre-load",
            lambda: I._check_aggregate_requires_q(o_tab, [False]),
            ("aggregate", "lss_completion"))

    # -------------------------------------------------------- PERMITTED 4
    # The escape hatch: --allow_unanchored_budget must RUN, loudly.
    o = parse(LATENT + ["--fixed_parameter_values", FIXED_NO_BUDGET,
                        "--allow_unanchored_budget"])
    I._check_latent_field_mode(o)
    I._stamp_latent_artifact_fingerprint(o)
    labels3, kinds3, fv3 = _pspace(o)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        I._check_latent_budget_anchor(o, labels3, kinds3, fv3)
    txt = " ".join(buf.getvalue().split())
    RESULTS.append({
        "case": "P4 --allow_unanchored_budget escape hatch",
        "cell": "guard 5 / post-pspace", "expected": "PERMITTED WITH WARNING",
        "raised": None, "stdout_head": txt[:400],
        "verdict": ("PASS" if ("not a measurement" in txt.lower()
                               or "running anyway" in txt.lower())
                    else "FAIL -- ran silently")})

    out = {"what": "independent re-verification of 6+2 PR-6a guard cells",
           "base_argv": "experiments/desi_full259/sbatch_ns_joint_sel.sh",
           "anchor": str(ANCHOR), "results": RESULTS,
           "n_pass": sum(r["verdict"] == "PASS" for r in RESULTS),
           "n_total": len(RESULTS)}
    (PR6A / "verify_guards.json").write_text(json.dumps(out, indent=1))
    for r in RESULTS:
        print(f"{r['verdict']:<28} {r['case']}  [{r['cell']}]")
        if r["verdict"] != "PASS":
            print("      ", r.get("message_head") or r.get("stdout_head"))
    print(f"\n{out['n_pass']}/{out['n_total']} PASS  -> {PR6A}/verify_guards.json")
    return 0 if out["n_pass"] == out["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
