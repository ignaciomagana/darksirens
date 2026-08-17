"""THE latent-mode compatibility matrix, in one place (field-level PR-6a).

PLAN section 7's PR-6a gate list is a table, and until now it was scattered:
guard 6's exclusivity refusals live in ``cli/inference._check_latent_field_mode``,
the artifact guards in ``likelihood/factory``, the b_miss inversion in
``_build_and_report_parameter_space``, the strata refusal after the selection
fits are opened, the budget-anchor guard after the parameter space is built.
A reader who wants to answer "may I run latent mode with X?" had to find five
call sites and reconstruct the answer.  This module IS the answer: one row per
cell of the matrix, one driver, and every refusal asserted on its MESSAGE, not
merely on its exception type -- because a guard whose message does not explain
WHY sends the operator to ``--allow_...`` rather than to the fix.

The matrix (PLAN section 7, PR-6a):

    PERMITTED                    Om0, w0, wa, b_miss (= b_GW: PLAN 4.3 inverts
                                 the table-mode guard, so this is the one
                                 parameter whose PERMISSION is the change)
    PERMITTED ONLY UNDER THE     log10n0, delta -- pinned by
    CALIBRATION PRIOR            --fixed_parameter_values at the calibration
                                 fit, or carrying a normal prior.  Flat and
                                 sampled is REFUSED at every rung (guard 5),
                                 with --allow_unanchored_budget as the escape
                                 hatch.
    REFUSED                      --c_mode per_pixel; --lss_completion;
                                 --use_lss; the schechter selection family;
                                 --per_pixel_completeness=off

plus the two symmetric obligations the same gate list states: **the guards
fire in table mode**, and **the retired provenance checks are bypassed in
latent mode only**.

Why guard 5 is the subtle one.  At rung 0 -- which is exactly what PR-6a ships
-- the count channel is conditioned on the OBSERVED shell totals, so it carries
ZERO information about ``(log10n0, delta, theta_sel)`` by construction.  A flat
prior on the budget is therefore not marginalisation: the posterior in those
directions is the prior, and the missing-galaxy budget -- the only channel
through which the field reaches H0 -- is set by the declared prior volume.  The
resulting H0 shift is an artifact of the fiducial calibration, the same class of
defect as K10.  That is why the refusal is unconditional across rungs and why
the escape hatch has to say so in its own message.

Everything here is HOST-SIDE and cheap: the artifact rows write a ~10 KB
stand-in into ``tmp_path``, and no likelihood is ever evaluated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py

import darksirens.cli.inference as inference
from darksirens.cli.inference import (
    _check_aggregate_requires_q,
    _check_latent_budget_anchor,
    _check_latent_field_mode,
    _check_latent_selection_family,
    _check_latent_selection_strata,
    _check_selection_qtable_theta,
    _normalize_multitracer_paths,
    build_parser,
)
from darksirens.likelihood.factory import (
    _latent_guard_fingerprint,
    latent_artifact_fingerprint,
)


# ─────────────────────────────────────────────────────── artifact stand-ins ──

#: The guard-1 ingredient list is small; none of the ~64 MB blocks
#: (``row_fac``, the moment tables) is read by the fingerprint, so a compliant
#: stand-in is a few kilobytes.  ``load_latent_plan`` is NOT exercised here --
#: tests/test_latent_factory.py owns that -- so only the fingerprint fields are
#: written.
_BASIS_META = dict(jitter_mode="factored-v1", j_sph=1e-6, j_z=1e-6, amp=1.0,
                   ls_sph=0.8, ls_z=0.11, M_sph=24, M_z=6, z_node_hi=0.30)
_THETA_REF = dict(M0hat=-20.3, sigma_M=0.6, delta=0.0, Om0=0.315)


def _write_fp_artifact(path, *, drop=(), sha256="a" * 64,
                       format_version="darksirens-latent-field-1.0",
                       content_sha256=None, basis_meta=None, b_gal=1.0,
                       counts_scale=1.0):
    """A fingerprint-complete anchor stand-in; ``drop`` removes ingredients."""
    meta = dict(_BASIS_META if basis_meta is None else basis_meta)
    for k in drop:
        meta.pop(k, None)
    with h5py.File(path, "w") as f:
        g = f.create_group("latent_field")
        if "completeness" not in drop:
            g.create_dataset("completeness", data=np.linspace(0.55, 0.95, 8))
        if "shell_response" not in drop:
            g.create_dataset("shell_response", data=np.eye(3, 5))
        if "z_count_edges" not in drop:
            g.create_dataset("z_count_edges", data=np.linspace(0.0, 0.3, 4))
        if "counts" not in drop:
            g.create_dataset("counts",
                             data=np.full((3, 8), 7.0 * float(counts_scale)))
        g.attrs["basis_meta"] = json.dumps(meta)
        if "theta_ref" not in drop:
            g.attrs["theta_ref"] = json.dumps(_THETA_REF)
        if "b_gal" not in drop:
            g.attrs["b_gal"] = float(b_gal)
        if "nside" not in drop:
            g.attrs["nside"] = 4
        if sha256 is not None:
            g.attrs["sha256"] = sha256
        if format_version is not None:
            g.attrs["format_version"] = format_version
        if content_sha256 is not None:
            g.attrs["content_sha256"] = content_sha256
    return str(path)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    """One compliant artifact, reused by every row that only needs a file."""
    return _write_fp_artifact(
        tmp_path_factory.mktemp("latent") / "anchor.h5")


# ─────────────────────────────────────────────────────────────── the driver ──

def _opts(*extra, artifact=None, latent=True, **stamps):
    """A parsed namespace, exactly as ``main`` builds it before the guards."""
    argv = ["--sampler", "tinyns",
            "--universe_model", "dark_sirens",
            "--survey_path", "/nonexistent/survey.h5"]
    if latent:
        argv += ["--lss_field_mode", "latent"]
        if artifact:
            argv += ["--lss_field_artifact", str(artifact)]
    argv += [str(a) for a in extra]
    opts = build_parser().parse_args(argv)
    _normalize_multitracer_paths(opts)
    for k, v in stamps.items():
        setattr(opts, k, v)
    return opts


@dataclass(frozen=True)
class Row:
    """One cell of the compatibility matrix.

    ``stage`` is the phase of ``main`` the guard fires from, and naming it is
    what makes the matrix legible WITHOUT the call graph in hand:

    ``pre-load``     pure opts arithmetic (``_check_latent_field_mode``); the
                     operator pays a second, not a catalog load.
    ``post-fits``    needs the ``--selection_fit`` JSON open, which is the only
                     place the LF family and the stratum count exist.
    ``post-pspace``  needs the resolved labels AND their prior families -- the
                     first line at which "is the budget anchored?" has an
                     answer.
    ``build``        needs the artifact itself (guard 1 and the geometry
                     guards), i.e. likelihood-build time, where PLAN 4.4 places
                     every successor to the retired firewall.
    """

    id: str
    stage: str
    verdict: str                       # PERMITTED | REFUSED | ANCHORED-ONLY
    run: Callable[..., None]           # takes the fixtures it needs
    expect: tuple = ()                 # substrings REQUIRED in the message
    mode: str = "latent"               # latent | table


def _fatal_text(fn, capsys):
    """Run ``fn``; return the ``_fatal`` message it printed."""
    with pytest.raises(SystemExit) as exc:
        fn()
    assert exc.value.code == 1, "a guard must exit(1), not exit(%r)" % exc.value.code
    return capsys.readouterr().out


def _raise_text(fn):
    """Run ``fn``; return the ``ValueError`` text it raised."""
    with pytest.raises(ValueError) as exc:
        fn()
    return str(exc.value)


# ─────────────────────────────────────────────────────────────── the matrix ──
#
# Signature of every ``run``: ``(artifact, capsys) -> str | None``.  Returning
# a string means "this is the message"; returning None means "nothing was
# raised", which is what a PERMITTED row must do.

def _permitted_baseline(artifact, capsys):
    """The admitted PR-6a configuration itself: aggregate + f_p + no table."""
    _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5",
        artifact=artifact))
    return None


def _permitted_selection_c_mode(artifact, capsys):
    _check_latent_field_mode(_opts(
        "--c_mode", "selection", "--per_pixel_completeness", "/depth.h5",
        artifact=artifact))
    return None


def _permitted_cosmology_and_b_gw(artifact, capsys):
    """Om0/w0/wa/b_miss sample FLAT in latent mode and guard 5 ignores them.

    This cell has REAL content, because table mode refuses exactly these
    labels: ``inference/q_provenance._Q_CONDITIONED`` lists Om0/w0/wa (and
    b_miss) as parameters a Q table is CONDITIONED on, so sampling them against
    a table is a hard error -- the mismatch is absorbed into Q as spurious
    redshift structure.  Latent mode has no such conditioning: at rung 0 every
    latent array is theta-free, ``rho`` is re-formed in-likelihood at the
    sampled theta, and the cosmology is free.  The row below is the latent half
    (guard 5 must not mistake any of them for a budget direction); the
    ``table: sampling Om0 against a Q table is refused`` row is the other half.

    b_miss is the load-bearing one: PLAN 4.3 inverts the table-mode rule, so
    the parameter that table mode DROPS (inert against an all-zero delta_g) is
    the parameter latent mode samples, as b_GW.
    """
    labels = ["H0", "Om0", "w0", "wa", "b_miss"]
    _check_latent_budget_anchor(
        _opts(artifact=artifact), labels,
        [("uniform", None, None)] * len(labels), {})
    return None


def _anchored_by_pin(artifact, capsys):
    """log10n0/delta PINNED by --fixed_parameter_values: the production line."""
    labels = ["H0", "b_miss"]
    _check_latent_budget_anchor(
        _opts(artifact=artifact), labels, [("uniform", None, None)] * 2,
        {"log10n0": -2.31, "delta": 0.94})
    out = capsys.readouterr().out
    assert "log10n0" not in out, (
        "a pinned budget must not warn: this is the shipped production "
        "configuration (calibrate_n0.py -> --fixed_parameter_values)")
    return None


def _anchored_by_normal_prior(artifact, capsys):
    """log10n0/delta SAMPLED under a normal calibration prior (OWNER DEC. 6)."""
    labels = ["H0", "log10n0", "delta"]
    _check_latent_budget_anchor(
        _opts(artifact=artifact), labels,
        [("uniform", None, None), ("normal", -2.31, 0.05),
         ("normal", 0.94, 0.2)],
        {})
    return None


def _refused_flat_budget(artifact, capsys):
    labels = ["H0", "log10n0", "delta"]
    return _fatal_text(lambda: _check_latent_budget_anchor(
        _opts(artifact=artifact), labels,
        [("uniform", None, None)] * 3, {}), capsys)


def _refused_flat_budget_per_catalog(artifact, capsys):
    """The suffixed form: log10n0_c2 belongs to catalog 2 and is guarded too."""
    labels = ["H0", "log10n0_c2"]
    return _fatal_text(lambda: _check_latent_budget_anchor(
        _opts(artifact=artifact, n_catalogs=2), labels,
        [("uniform", None, None)] * 2, {"log10n0": -2.3, "delta": 0.9}),
        capsys)


def _escape_hatch_runs_and_says_so(artifact, capsys):
    """--allow_unanchored_budget: runs, but the run declares what it means."""
    labels = ["H0", "log10n0", "delta"]
    _check_latent_budget_anchor(
        _opts("--allow_unanchored_budget", artifact=artifact), labels,
        [("uniform", None, None)] * 3, {})
    return capsys.readouterr().out


def _refused_per_pixel(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "per_pixel", "--per_pixel_completeness", "/depth.h5",
        artifact=artifact)), capsys)


def _refused_per_pixel_completeness_off(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", artifact=artifact)), capsys)


def _refused_lss_completion(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5",
        "--lss_completion", "/q.h5", artifact=artifact)), capsys)


def _refused_use_lss(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5",
        "--use_lss", "true", artifact=artifact)), capsys)


def _refused_schechter(artifact, capsys):
    return _fatal_text(lambda: _check_latent_selection_family(
        _opts("--c_mode", "selection", "--per_pixel_completeness", "/depth.h5",
              artifact=artifact, selection_family="schechter")), capsys)


def _permitted_gaussian_family(artifact, capsys):
    _check_latent_selection_family(
        _opts("--c_mode", "selection", "--per_pixel_completeness", "/depth.h5",
              artifact=artifact, selection_family="gaussian"))
    return None


def _refused_stratum_map(artifact, capsys):
    # Stamped rather than passed on argv: ``_normalize_multitracer_paths``
    # refuses --stratum_map without --selection_fit first (a different, older
    # guard), and this row is about the LATENT refusal.
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "selection", "--per_pixel_completeness", "/depth.h5",
        artifact=artifact, stratum_map="/strata.h5")), capsys)


def _refused_multi_stratum_fit(artifact, capsys):
    return _fatal_text(lambda: _check_latent_selection_strata(
        _opts("--c_mode", "selection", "--per_pixel_completeness", "/depth.h5",
              artifact=artifact,
              selection_strata_by_catalog=[[{"j": 0}, {"j": 1}]])), capsys)


def _refused_non_galaxy_model(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--universe_model", "spectral_sirens", "--c_mode", "aggregate",
        "--per_pixel_completeness", "/depth.h5", artifact=artifact)), capsys)


def _refused_missing_artifact(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5")),
        capsys)


# ── guard 1: the fingerprint, the successor to every retired check ──────────

def _permitted_fingerprint(artifact, capsys):
    fp = _latent_guard_fingerprint(artifact, _opts(artifact=artifact))
    assert len(fp["content"]) == 64 and fp["content"] != fp["stored"]
    return None


def _permitted_fingerprint_pinned(artifact, capsys):
    """Both digests are accepted by the pin -- the stamped one and the content
    one.  The content digest is what survives a copy that dropped the attrs."""
    fp = latent_artifact_fingerprint(artifact)
    for pin in (fp["stored"], fp["content"], fp["content"].upper()):
        _latent_guard_fingerprint(
            artifact, _opts(artifact=artifact, lss_field_sha256=pin))
    return None


def _refused_fingerprint_mismatch(artifact, capsys):
    return _raise_text(lambda: _latent_guard_fingerprint(
        artifact, _opts(artifact=artifact, lss_field_sha256="b" * 64)))


def _refused_fingerprint_edited(artifact, capsys, tmp_path):
    """A file that declares a content digest it no longer hashes to."""
    fp = latent_artifact_fingerprint(artifact)
    p = _write_fp_artifact(tmp_path / "edited.h5",
                           content_sha256=fp["content"], counts_scale=2.0)
    return _raise_text(lambda: latent_artifact_fingerprint(p))


def _refused_fingerprint_missing_ingredient(artifact, capsys, tmp_path):
    p = _write_fp_artifact(tmp_path / "no_counts.h5", drop=("counts", "b_gal"))
    return _raise_text(lambda: _latent_guard_fingerprint(p, _opts(artifact=p)))


def _refused_fingerprint_no_stamp(artifact, capsys, tmp_path):
    p = _write_fp_artifact(tmp_path / "unstamped.h5", sha256="")
    return _raise_text(lambda: _latent_guard_fingerprint(p, _opts(artifact=p)))


def _refused_fingerprint_bad_format(artifact, capsys, tmp_path):
    p = _write_fp_artifact(tmp_path / "alien.h5", format_version="mystery-2")
    return _raise_text(lambda: _latent_guard_fingerprint(p, _opts(artifact=p)))


def _refused_pin_not_hex(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(_opts(
        "--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5",
        "--lss_field_sha256", "not-a-digest", artifact=artifact)), capsys)


# ── the symmetric half: guards fire in TABLE mode, latent-only flags do not ──

def _table_aggregate_requires_q(artifact, capsys):
    return _fatal_text(lambda: _check_aggregate_requires_q(
        _opts("--c_mode", "aggregate", latent=False), (False,)), capsys)


def _latent_aggregate_does_not_require_q(artifact, capsys):
    _check_aggregate_requires_q(
        _opts("--c_mode", "aggregate", "--per_pixel_completeness", "/depth.h5",
              artifact=artifact), (False,))
    return None


def _table_qtable_theta_firewall_fires(artifact, capsys):
    fid = {"path": "/q.h5", "selection_family": "schechter",
           "selection_Mstar_hat": -20.5}
    return _fatal_text(lambda: _check_selection_qtable_theta(
        [fid], _opts(latent=False, selection_fits=[None],
                     selection_family="gaussian")), capsys)


def _latent_qtable_theta_firewall_is_retired(artifact, capsys):
    """Bypassed because it has NO REFERENT: there is no table to stamp."""
    fid = {"path": "/q.h5", "selection_family": "schechter",
           "selection_Mstar_hat": -20.5}
    _check_selection_qtable_theta(
        [fid], _opts("--c_mode", "selection",
                     "--per_pixel_completeness", "/depth.h5",
                     artifact=artifact, selection_fits=[None],
                     selection_family="gaussian"))
    return None


def _table_cosmology_firewall_fires(artifact, capsys):
    """The other half of the Om0/w0/wa PERMITTED cell: table mode refuses them.

    ``check_lss_completion_provenance`` is the check PLAN 4.4 retires first by
    name.  Here it is, still firing on the table path -- which is what makes
    "permitted in latent mode" a statement rather than an absence.
    """
    from darksirens.inference.q_provenance import check_lss_completion_provenance

    fid = {"path": "/q.h5", "fiducial_Om0": 0.30, "fiducial_n0": 1e-2}
    return _raise_text(lambda: check_lss_completion_provenance(
        [fid], ["H0", "Om0", "log10n0"], {}))


def _table_budget_guard_is_a_no_op(artifact, capsys):
    """Guard 5 is a LATENT construction: table mode keeps its own firewall
    (q_provenance) and must not gain a new refusal from this PR."""
    labels = ["H0", "log10n0", "delta"]
    _check_latent_budget_anchor(
        _opts(latent=False), labels, [("uniform", None, None)] * 3, {})
    return None


def _table_refuses_orphan_pin(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(
        _opts(latent=False, lss_field_sha256="a" * 64)), capsys)


def _table_refuses_orphan_escape_hatch(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(
        _opts("--allow_unanchored_budget", latent=False)), capsys)


def _table_refuses_orphan_artifact(artifact, capsys):
    return _fatal_text(lambda: _check_latent_field_mode(
        _opts(latent=False, lss_field_artifact=str(artifact))), capsys)


MATRIX = (
    # ── PERMITTED ───────────────────────────────────────────────────────────
    Row("aggregate + per_pixel_completeness", "pre-load", "PERMITTED",
        _permitted_baseline),
    Row("selection + per_pixel_completeness", "pre-load", "PERMITTED",
        _permitted_selection_c_mode),
    Row("gaussian selection family", "post-fits", "PERMITTED",
        _permitted_gaussian_family),
    Row("Om0 / w0 / wa / b_miss(=b_GW) flat", "post-pspace", "PERMITTED",
        _permitted_cosmology_and_b_gw),
    Row("guard-1 fingerprint of a compliant artifact", "build", "PERMITTED",
        _permitted_fingerprint),
    Row("guard-1 pin matches either digest", "build", "PERMITTED",
        _permitted_fingerprint_pinned),

    # ── PERMITTED ONLY UNDER THE CALIBRATION PRIOR (guard 5) ────────────────
    Row("log10n0/delta pinned by --fixed_parameter_values",
        "post-pspace", "ANCHORED-ONLY", _anchored_by_pin),
    Row("log10n0/delta under a normal calibration prior",
        "post-pspace", "ANCHORED-ONLY", _anchored_by_normal_prior),
    Row("log10n0/delta FLAT and sampled", "post-pspace", "REFUSED",
        _refused_flat_budget,
        ("PLAN 4.4 guard 5", "ZERO information", "shell totals",
         "K10", "calibrate_n0.py", "--allow_unanchored_budget",
         "log10n0", "delta")),
    Row("log10n0_c2 FLAT and sampled", "post-pspace", "REFUSED",
        _refused_flat_budget_per_catalog,
        ("log10n0_c2", "guard 5")),
    Row("--allow_unanchored_budget (escape hatch)", "post-pspace",
        "ANCHORED-ONLY", _escape_hatch_runs_and_says_so,
        ("allow_unanchored_budget", "RUNNING ANYWAY", "ablation",
         "not a measurement")),

    # ── REFUSED ─────────────────────────────────────────────────────────────
    Row("--c_mode per_pixel", "pre-load", "REFUSED", _refused_per_pixel,
        ("per_pixel", "absorbs the observed angular", "double-count",
         "guard 6")),
    Row("--per_pixel_completeness off", "pre-load", "REFUSED",
        _refused_per_pixel_completeness_off,
        ("--per_pixel_completeness", "f_p", "PLAN eq. (4)", "guard 6")),
    Row("--lss_completion", "pre-load", "REFUSED", _refused_lss_completion,
        ("--lss_completion", "two different completion models", "guard 6")),
    Row("--use_lss", "pre-load", "REFUSED", _refused_use_lss,
        ("--use_lss", "delta_g", "b_GW", "guard 6")),
    Row("schechter selection family", "post-fits", "REFUSED",
        _refused_schechter,
        ("schechter", "M_faint_offset", "m_faint_cut", "PR-2", "gaussian")),
    Row("--stratum_map (stratified selection)", "pre-load", "REFUSED",
        _refused_stratum_map, ("--stratum_map", "rho_s", "guard 6")),
    Row("multi-stratum --selection_fit", "post-fits", "REFUSED",
        _refused_multi_stratum_fit,
        ("selection strata", "ONE", "(A, B)", "guard 6")),
    Row("non-galaxy-aware --universe_model", "pre-load", "REFUSED",
        _refused_non_galaxy_model,
        ("galaxy-aware", "missing-galaxy budget")),
    Row("latent without --lss_field_artifact", "pre-load", "REFUSED",
        _refused_missing_artifact,
        ("--lss_field_artifact", "darksirens_build_latent_field")),
    Row("--lss_field_sha256 mismatch", "build", "REFUSED",
        _refused_fingerprint_mismatch,
        ("MISMATCH", "PLAN", "4.4", "DIFFERENT field")),
    Row("artifact edited after build", "build", "REFUSED",
        _refused_fingerprint_edited,
        ("FAILED its own content fingerprint", "edited after it")),
    Row("artifact missing a guard-1 ingredient", "build", "REFUSED",
        _refused_fingerprint_missing_ingredient,
        ("guard-1 fingerprint ingredient", "counts", "b_gal",
         "cannot be fingerprinted")),
    Row("artifact with no sha256 stamp", "build", "REFUSED",
        _refused_fingerprint_no_stamp,
        ("no usable sha256 stamp", "--lss_field_sha256")),
    Row("artifact with an unknown format_version", "build", "REFUSED",
        _refused_fingerprint_bad_format,
        ("format_version", "ONLY provenance", "unknown producer")),
    Row("--lss_field_sha256 that is not a digest", "pre-load", "REFUSED",
        _refused_pin_not_hex, ("64-character hex", "guard 1")),

    # ── the symmetric obligation: table mode is untouched ───────────────────
    Row("table: aggregate still requires a Q table", "pre-load", "REFUSED",
        _table_aggregate_requires_q, ("--c_mode aggregate requires",),
        mode="table"),
    Row("latent: aggregate does NOT require a Q table", "pre-load",
        "PERMITTED", _latent_aggregate_does_not_require_q),
    Row("table: Q-table theta firewall fires", "post-fits", "REFUSED",
        _table_qtable_theta_firewall_fires, ("family",), mode="table"),
    Row("latent: Q-table theta firewall is retired", "post-fits", "PERMITTED",
        _latent_qtable_theta_firewall_is_retired),
    Row("table: sampling Om0 against a Q table is refused", "post-pspace",
        "REFUSED", _table_cosmology_firewall_fires,
        ("Q_LSS provenance mismatch", "Om0", "SAMPLED", "biases H0"),
        mode="table"),
    Row("table: guard 5 is a no-op", "post-pspace", "PERMITTED",
        _table_budget_guard_is_a_no_op, mode="table"),
    Row("table: --lss_field_sha256 is an orphan", "pre-load", "REFUSED",
        _table_refuses_orphan_pin,
        ("--lss_field_sha256", "never reads"), mode="table"),
    Row("table: --allow_unanchored_budget is an orphan", "pre-load", "REFUSED",
        _table_refuses_orphan_escape_hatch,
        ("--allow_unanchored_budget", "check_lss_completion_provenance"),
        mode="table"),
    Row("table: --lss_field_artifact is an orphan", "pre-load", "REFUSED",
        _table_refuses_orphan_artifact,
        ("--lss_field_artifact", "silently be a plain table run"),
        mode="table"),
)


@pytest.mark.parametrize("row", MATRIX, ids=[r.id for r in MATRIX])
def test_latent_compatibility_matrix(row, artifact, capsys, tmp_path):
    """One cell of PLAN section 7's PR-6a gate list.

    A REFUSED row must (a) refuse and (b) EXPLAIN: every substring in
    ``row.expect`` has to appear in the message.  Asserting on the type alone
    would pass for a message that says only "invalid configuration", and the
    operator's next move after an unexplained refusal is to look for a flag
    that turns the guard off.
    """
    kwargs = {}
    if "tmp_path" in row.run.__code__.co_varnames[:row.run.__code__.co_argcount]:
        kwargs["tmp_path"] = tmp_path
    message = row.run(artifact, capsys, **kwargs)

    if row.verdict == "REFUSED":
        assert message, f"{row.id}: expected a refusal, got none"
    for needle in row.expect:
        assert needle in message, (
            f"{row.id}: the guard's message does not explain itself -- "
            f"{needle!r} missing from:\n{message}")


def test_every_gate_list_cell_is_covered():
    """The matrix above must cover PLAN section 7's PR-6a list, verbatim.

    This is the test that keeps the table honest: if a future PR adds a cell to
    the plan's list, or renames one, the matrix has to grow with it rather than
    silently drift into being a sample of the guards instead of all of them.
    """
    ids = " | ".join(r.id for r in MATRIX)
    for required in ("Om0 / w0 / wa / b_miss", "log10n0/delta pinned",
                     "log10n0/delta under a normal", "log10n0/delta FLAT",
                     "--c_mode per_pixel", "--lss_completion", "--use_lss",
                     "schechter selection family",
                     "--per_pixel_completeness off"):
        assert required in ids, f"gate-list cell missing from the matrix: {required}"
    # And both directions of "guards fire in table mode, bypassed in latent".
    assert any(r.mode == "table" for r in MATRIX)
    assert any(r.mode == "latent" and "retired" in r.id for r in MATRIX)


# ────────────────────────────── the provenance rewiring, PLAN section 4.4 ───

def test_retired_checks_have_no_referent_in_latent_mode():
    """PLAN 4.4: five checks are RETIRED on the latent path, and the reason is
    uniform -- each is a statement about a Q TABLE's build-time stamps, and
    latent mode has no table to stamp.

    This pins the STRUCTURAL half of that claim, which is what makes the
    retirement honest rather than convenient:

    * ``check_lss_completion_provenance`` and ``_check_selection_qtable_theta``
      and ``_check_q_table_z_depth`` consume ``lss_completion_fiducials``, which
      only exists when ``maybe_load_lss_completion`` loaded a table;
    * the ``catalogs/lss.py`` float whitelist and the ``c_mode`` table-vs-run
      check run INSIDE that same loader;
    * ``realization_set_id`` / ``member_content_sha256`` matching runs only for
      a ``K>=2 --lss_marginalize`` mixture.

    Guard 6 refuses every route into the first five (``--lss_completion``
    pre-load, the in-catalog ``/lss_completion`` group post-load), and the
    factory refuses ``K>=2`` outright in latent mode, so all six are unreachable
    -- and the CLI asserts that unreachability rather than assuming it (see the
    ``_live`` check in ``_build_and_report_parameter_space``).
    """
    import inspect

    src = inspect.getsource(inference._build_and_report_parameter_space)
    # The bypass is an ASSERTION, not an early return: a table that reached the
    # retired checks in latent mode is a BUG and must be fatal, never silent.
    assert "if latent_field_mode(opts):" in src
    assert "produced" in src and "Q-table build fiducials" in src
    # ... and table mode still runs all three.
    assert "check_lss_completion_provenance(" in src
    assert "_check_selection_qtable_theta(_q_fiducials, opts)" in src
    assert "_check_q_table_z_depth(_q_fiducials, opts)" in src


def test_k_ge_2_is_refused_in_latent_mode_which_is_what_retires_realization_set_id():
    """``realization_set_id`` matching is a K>=2 construct; latent mode is K=1.

    PLAN 4.4 calls this "deleting the producer": the joint builder is the only
    source of a shared ``realization_set_id``, it is ``per_pixel``-only, and the
    only way to run a K>=2 shared-member marginalization today is
    ``--allow_unverified_shared_lss_members``, which marginalizes over an
    INDEPENDENT-fields product prior rather than the shared-field prior the
    estimator assumes.  One ``xi`` shared across tracers would make "member m of
    every catalog is the same realization" a theorem -- but that is PR-7.  At
    PR-6a the seam is K=1 only, so the matching has nothing to match.
    """
    src = __import__("inspect").getsource(
        __import__("darksirens.likelihood.factory", fromlist=["x"]))
    assert "not wired on the K >= 2 mixture" in src


def test_settings_json_carries_the_fingerprint(artifact):
    """The successor stamp has to land in the run's provenance record.

    ``settings.json`` is written BEFORE the likelihood is built, so the CLI runs
    guard 1 pre-load and stamps both digests onto ``opts``; ``save_settings_json``
    serializes ``vars(opts)`` wholesale, so this is what puts them on disk.  In
    latent mode they stand exactly where the table-mode ``Q_LSS build
    fiducials`` block stands.
    """
    opts = _opts("--c_mode", "aggregate", "--per_pixel_completeness",
                 "/depth.h5", artifact=artifact)
    inference._stamp_latent_artifact_fingerprint(opts)
    fp = latent_artifact_fingerprint(artifact)
    assert opts.lss_field_stored_sha256 == fp["stored"]
    assert opts.lss_field_content_sha256 == fp["content"]
    assert opts.lss_field_theta_ref == _THETA_REF
    assert opts.lss_field_b_gal == 1.0
    # Serializable, or settings.json would silently stringify it.
    json.dumps(vars(opts))


def test_stamping_is_a_no_op_in_table_mode():
    """The additive rule: a table run must not grow a single settings key."""
    opts = _opts(latent=False)
    before = set(vars(opts))
    inference._stamp_latent_artifact_fingerprint(opts)
    assert set(vars(opts)) == before


def test_the_two_new_flags_are_start_time_gates_not_semantic_ones():
    """Neither new flag may move a TABLE-mode run fingerprint.

    ``--lss_field_sha256`` pins an identity (the artifact's bytes are already
    content-hashed by the input-file scan) and ``--allow_unanchored_budget``
    waives a refusal (the prior it waives is already fingerprinted through the
    labels, bounds and prior families).  Both are therefore start-time gates in
    the sense ``run_fingerprint``'s own comment defines, like
    ``--allow_skymap_population`` -- and keeping them out is what keeps every
    table-mode digest byte-identical across this PR.
    """
    from darksirens.inference.run_fingerprint import _NON_SEMANTIC_KEYS

    assert "lss_field_sha256" in _NON_SEMANTIC_KEYS
    assert "allow_unanchored_budget" in _NON_SEMANTIC_KEYS
    # lss_field_mode itself stays SEMANTIC: it inverts the b_miss rule.
    assert "lss_field_mode" not in _NON_SEMANTIC_KEYS
    assert "lss_field_artifact" not in _NON_SEMANTIC_KEYS
