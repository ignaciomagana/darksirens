"""PR-7 pins for ``cli/build_latent_field.py`` at K = 1 and K >= 2.

The ladder's standing invariant is that K = 1 does not move, and for a BUILDER
"does not move" means the artifact is byte-identical, digest included.  That is
checked here two ways on a small synthetic world:

* a build with no ``--tracer-*`` argument at all, against a build that passes
  ``--tracer-labels`` pointing at an ALL-ZERO label array.  The second is
  ``K = 1`` reached through the PR-7 code path with the PR-7 switches on the
  command line, so if the tracer machinery leaked anything at all -- an extra
  dataset, an extra attribute, an extra key in the stamped config -- the two
  files would differ.  They must be byte-identical as FILES.
* the same build against a two-tracer one, where the artifact must gain a
  ``tracers`` subgroup, per-tracer moment tables, a ``bias_profile_cov``, and a
  DIFFERENT ``xi_hat`` (the stacked fit is a different fit), while the
  per-tracer counts still sum to the parent's exactly.

Two further pins cover what "K catalogs of selection against one field" has to
mean beyond ``f_p`` and ``b_k``: the SHELL RESPONSE is per tracer (a catalog's
own magnitude selection and photo-z kernel are what weight its galaxies within
a shell), and the ``(K, K)`` bias covariance is read at the PROFILE MAXIMUM
rather than at the anchor, with the residual gradient gated.

The world is deliberately tiny (nside 8, rank 48, two shells) but must still
clear the builder's own occupancy guard 7 (``>= 1e4`` galaxies and ``>= 500``
occupied pixels per shell), so the counts are generated to satisfy it rather
than shrunk until the guard is edited.

Reference-vs-PR-7 byte identity on the PRODUCTION-shaped world cannot live in a
test suite -- it needs the closure workstream's 65 MB catalog -- and was
measured instead by the PR-7 workstream:
``experiments/field_level_plan/pr7/REPORT.md`` records that a K = 1 rebuild of
``pr6a/data/rb`` at nside 16, rank 320, is byte-identical (``cmp``) to the same
build at commit ``03c5882``, sha256 ``662d1d7f...`` on both sides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hp = pytest.importorskip("healpy")

NSIDE = 8
NPIX = 12 * NSIDE ** 2          # 768
NMAX = 120
Z_DEPTH = 0.30
N_SHELL = 2


def _write_world(tmp_path, seed=4242, n_tracer=2):
    """A synthetic survey + depth map + fits that clears guard 7."""
    rng = np.random.default_rng(seed)
    # Every pixel occupied, so all 768 clear the ">= 500 occupied" half of the
    # guard in both shells; NMAX galaxies each gives 92160 total, ~46k a shell.
    ngals = np.full(NPIX, NMAX, dtype=np.int64)
    # A smooth angular modulation so the counts are not iid noise -- the solve
    # then has something to fit and ``xi_hat`` is not numerically degenerate.
    vec = np.column_stack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    tilt = 0.35 * vec[:, 2]
    zg = np.zeros((NPIX, NMAX))
    lab = np.zeros((NPIX, NMAX), dtype=np.int32)
    for p in range(NPIX):
        u = rng.uniform(size=NMAX) ** (1.0 / (1.0 + tilt[p] + 1.0))
        zg[p] = 0.02 + (Z_DEPTH - 0.04) * u
        lab[p] = rng.integers(0, n_tracer, size=NMAX)

    survey = tmp_path / "survey.h5"
    with h5py.File(survey, "w") as f:
        f.create_dataset("zgals", data=zg)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("tracer", data=lab)
        f.create_dataset("tracer_all_zero", data=np.zeros_like(lab))
        f.attrs["nside"] = NSIDE

    def _mth(path, masked):
        with h5py.File(path, "w") as f:
            f.create_dataset("counts", data=np.full(NPIX, 100.0))
            f.create_dataset("masked_frac", data=masked)
            f.attrs["nside"] = NSIDE
            f.attrs["ordering"] = "RING"

    mth1 = tmp_path / "mth1.h5"
    mth2 = tmp_path / "mth2.h5"
    _mth(mth1, rng.uniform(0.0, 0.3, NPIX))
    _mth(mth2, rng.uniform(0.2, 0.6, NPIX))

    def _sel(path, m_lim, M0hat, sigma_M):
        path.write_text(json.dumps(dict(
            format_version="darksirens-selection-fit-1.0",
            strata=[dict(family="gaussian", m_lim=m_lim, M0hat=M0hat,
                         sigma_M=sigma_M,
                         cov=[[1e-6, 0.0], [0.0, 1e-6]],
                         n_gal=int(ngals.sum()),
                         stratum="all", k_corr_coeffs=[],
                         meta=dict(Om0=0.3075, w0=-1.0, wa=0.0, H0_ref=100.0,
                                   nll=0.0))])))

    sel = tmp_path / "selection_fit.json"
    _sel(sel, 20.0, -20.2, 1.0)
    # A SECOND catalog's selection: a brighter limiting magnitude and a
    # narrower magnitude scatter, i.e. a genuinely different C(z) and so a
    # genuinely different within-shell weighting.
    sel2 = tmp_path / "selection_fit_2.json"
    _sel(sel2, 19.2, -21.0, 0.7)
    cal = tmp_path / "n0_calibration.json"
    cal.write_text(json.dumps(dict(log10n0=-4.3, delta=0.0)))
    return dict(survey=str(survey), mth1=str(mth1), mth2=str(mth2),
                sel=str(sel), sel2=str(sel2), cal=str(cal))


def _build(w, out, extra=()):
    cmd = [sys.executable, "-m", "darksirens.cli.build_latent_field",
           "--survey", w["survey"], "--selection-fit", w["sel"],
           "--n0-calibration", w["cal"],
           "--per-pixel-completeness", w["mth1"],
           "--out", str(out), "--om0", "0.3075",
           "--z-depth", str(Z_DEPTH), "--n-shells", str(N_SHELL),
           "--m-sph", "16", "--m-z", "3",
           "--ls-sph", "0.9", "--ls-z", "0.15",
           "--m-draw", "4", "--n-b-nodes", "5", "--b-max", "2.0"]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("DARKSIRENS_ZMAX", None)
    r = subprocess.run(cmd + list(extra), capture_output=True, text=True,
                       env=env,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.slow
def test_k1_artifact_byte_identical_through_the_pr7_path(tmp_path):
    """The PR-7 switches, exercised, must leave a K = 1 artifact untouched.

    ``--tracer-labels tracer_all_zero`` routes the build through every line
    PR-7 added -- the label load, the shape check, the ``n_tracer`` reduction,
    the per-tracer branch guards -- and lands on ``n_tracer == 1``.  The
    resulting file must be byte-identical to the build that never mentioned a
    tracer, which pins BOTH that the K = 1 arithmetic is unchanged and that the
    stamped config drops the four tracer keys (a key whose value is ``None``
    would change the digest while changing no array).
    """
    w = _write_world(tmp_path)
    a, b = tmp_path / "k1_plain.h5", tmp_path / "k1_labelled.h5"
    _build(w, a)
    _build(w, b, ["--tracer-labels", "tracer_all_zero"])
    assert a.read_bytes() == b.read_bytes()
    with h5py.File(a) as f:
        g = f["latent_field"]
        assert "tracers" not in g
        assert "n_tracer" not in g.attrs
        assert g.attrs["draw_covariance"].startswith("H^{-1} + s_b^2")


@pytest.mark.slow
def test_k2_artifact_carries_the_per_tracer_block(tmp_path):
    w = _write_world(tmp_path)
    a, b = tmp_path / "k1.h5", tmp_path / "k2.h5"
    _build(w, a)
    out = _build(w, b, ["--tracer-labels", "tracer",
                        "--tracer-completeness", w["mth1"],
                        "--tracer-completeness", w["mth2"],
                        "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0",
                        "--tracer-names", "gal,agn"])
    assert "disjointness verified=True" in out
    with h5py.File(a) as fa, h5py.File(b) as fb:
        ga, gb = fa["latent_field"], fb["latent_field"]
        assert int(gb.attrs["n_tracer"]) == 2
        assert json.loads(gb.attrs["tracer_names"]) == ["gal", "agn"]
        assert np.allclose(gb.attrs["tracer_bias"], [1.0, 2.0])
        assert "disjoint" in gb.attrs["tracer_overlap_policy"]
        assert gb.attrs["draw_covariance"].startswith("H^{-1} + V C_b V^T")
        # The stacked fit is a DIFFERENT fit -- two multinomials and one ridge.
        assert not np.array_equal(ga["xi_hat"][...], gb["xi_hat"][...])
        # ... on the same shared field: one xi, one H_chol, one member set.
        assert gb["xi_hat"].shape == ga["xi_hat"].shape
        assert gb["Xi_members"].shape == ga["Xi_members"].shape
        # K >= 2 ADDS COLUMNS (PLAN §3.4).  K = 1 carries
        # [M0hat, sigma_M, delta, Om0, b_gal]; K = 2 carries the two SHARED
        # columns (delta, Om0 -- one field, one cosmology) plus K columns for
        # each of the two per-catalog selection parameters and K bias columns.
        assert ga["sensitivity_S"].shape[1] == 5
        assert gb["sensitivity_S"].shape[1] == 2 + 2 * 2 + 2
        # The per-tracer block, and the partition property.
        t0, t1 = gb["tracers"]["0"], gb["tracers"]["1"]
        assert t0.attrs["name"] == "gal" and t1.attrs["name"] == "agn"
        assert np.array_equal(t0["counts"][...] + t1["counts"][...],
                              gb["counts"][...])
        # Per-tracer moments differ, which is why K of them are stored at all.
        assert not np.allclose(t0["B_moments"][...], t1["B_moments"][...])
        assert float(t0.attrs["F_F"]) != float(t1.attrs["F_F"])
        # The bias covariance is 2x2, positive definite, and has a NON-ZERO
        # off-diagonal -- the coupling through the shared field's H^{-1}.
        # Only the sign of that statement is pinned here, not its size: this
        # fixture's counts are not a draw from the model (the tracer labels are
        # assigned at random to a catalog with no clustering), so the counts
        # carry almost no information about the biases and the posterior is
        # nearly the stated prior, which is diagonal.  The SIZE of the coupling
        # -- corr = 0.99994 and a ratio 105x tighter than the decoupled fit --
        # is a Tier E measurement on a world drawn from the model, and lives in
        # experiments/field_level_plan/pr7/REPORT.md, not in a unit test.
        C = np.asarray(json.loads(gb.attrs["bias_profile_cov"]))
        assert C.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(C) > 0.0)
        assert C[0, 1] != 0.0
        assert abs(C[0, 1] - C[1, 0]) <= 1e-12 * abs(C[0, 1])
        assert float(gb.attrs["bias_prior_log_sd"]) == 0.05
        assert "RATIO" in gb.attrs["bias_cov_convention"]


@pytest.mark.slow
def test_shell_response_is_per_tracer(tmp_path):
    """Tracer k's WITHIN-SHELL weighting must be tracer k's own selection.

    ``W`` collapses the fine redshift grid onto shells weighted by
    ``C(z; m_lim, M0hat, sigma_M)`` convolved with the photo-z kernel, so it is
    a property of the CATALOG, not of the field.  PR-7 built one ``W`` at
    ``theta_ref`` and handed it to all K operators, which silently gave every
    tracer catalog 0's magnitude selection while its ``f_p`` and its ``b_k``
    were its own -- and the omission was invisible precisely because those two
    per-tracer inputs already existed.

    Two builds pin it: one where the K selections DIFFER (the responses must
    differ) and one where they are the shared default (the responses must be
    identical, so the per-tracer machinery cannot have perturbed the shared
    case).
    """
    w = _write_world(tmp_path)
    diff, same = tmp_path / "sel_diff.h5", tmp_path / "sel_same.h5"
    _build(w, diff, ["--tracer-labels", "tracer",
                     "--tracer-completeness", w["mth1"],
                     "--tracer-completeness", w["mth2"],
                     "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0",
                     "--tracer-names", "gal,agn",
                     "--tracer-selection-fit", w["sel"],
                     "--tracer-selection-fit", w["sel2"],
                     "--tracer-sigma-z", "0.023",
                     "--tracer-sigma-z", "0.05"])
    _build(w, same, ["--tracer-labels", "tracer",
                     "--tracer-completeness", w["mth1"],
                     "--tracer-completeness", w["mth2"],
                     "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0",
                     "--tracer-names", "gal,agn"])
    with h5py.File(diff) as fd, h5py.File(same) as fs:
        gd, gs = fd["latent_field"], fs["latent_field"]
        W0 = gd["tracers"]["0"]["shell_response"][...]
        W1 = gd["tracers"]["1"]["shell_response"][...]
        # Not a rounding difference: a different m_lim and a different photo-z
        # width move the within-shell weighting by tens of percent.
        assert np.max(np.abs(W1 - W0)) / np.max(np.abs(W0)) > 0.05
        # The top-level dataset stays tracer 0's, which is what it is at K = 1.
        assert np.array_equal(gd["shell_response"][...], W0)
        # The catalog's selection is recorded WITH its response, so the
        # artifact can say which selection weighted which counts.
        assert float(gd["tracers"]["0"].attrs["m_lim"]) == 20.0
        assert float(gd["tracers"]["1"].attrs["m_lim"]) == 19.2
        assert float(gd["tracers"]["0"].attrs["sigma_z"]) == 0.023
        assert float(gd["tracers"]["1"].attrs["sigma_z"]) == 0.05
        th1 = json.loads(gd["tracers"]["1"].attrs["theta_ref"])
        assert th1["M0hat"] == -21.0 and th1["sigma_M"] == 0.7
        # Shared inputs: one selection, so the K responses coincide exactly.
        assert np.array_equal(gs["tracers"]["1"]["shell_response"][...],
                              gs["tracers"]["0"]["shell_response"][...])

        # PLAN §3.4: "catalog k's own theta contributes its own column built
        # from its own operator".  The per-catalog selection parameters are K
        # columns each; delta and Om0 stay one apiece because the field and the
        # cosmology are shared.
        labels = json.loads(gd.attrs["sensitivity_labels"])
        assert labels == ["M0hat[gal]", "M0hat[agn]",
                          "sigma_M[gal]", "sigma_M[agn]",
                          "delta", "Om0", "b_gal[gal]", "b_gal[agn]"]
        S = gd["sensitivity_S"][...]
        c_gal, c_agn = S[:, 0], S[:, 1]
        assert np.linalg.norm(c_gal) > 0.0 and np.linalg.norm(c_agn) > 0.0
        # Two DIFFERENT columns, not one column stored twice: a shared W has
        # only one M0hat direction to give.
        assert (np.linalg.norm(c_agn - c_gal)
                > 0.01 * np.linalg.norm(c_gal))


@pytest.mark.slow
def test_bias_covariance_is_read_at_the_profile_maximum(tmp_path):
    """``inv(H_u)`` is a Laplace width, and a width needs a stationary point.

    The log-bias curvature is ``diag(b) H_b diag(b) + diag(b g_b)``; the second
    term is exactly the profile gradient and vanishes only at the maximum.  The
    anchor's own ``b_k`` is not the maximum -- the count channel decreases
    monotonically in the bias amplitude and the prior, centred on the anchor,
    contributes no gradient there -- so the builder profiles to the maximum
    first.  Both numbers are stamped, and the test pins that they are on
    opposite sides of the tolerance: the anchor residual is O(10) nats per nat
    on this world, the solved one is machine noise.
    """
    w = _write_world(tmp_path)
    out = tmp_path / "mode.h5"
    _build(w, out, ["--tracer-labels", "tracer",
                    "--tracer-completeness", w["mth1"],
                    "--tracer-completeness", w["mth2"],
                    "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0",
                    "--tracer-names", "gal,agn"])
    with h5py.File(out) as f:
        g = f["latent_field"]
        tol = float(g.attrs["bias_profile_grad_log_tol"])
        assert float(g.attrs["bias_profile_grad_log_inf"]) <= tol
        # The defect, on the record: what the anchor carried.
        assert float(g.attrs["bias_profile_grad_log_inf_at_anchor"]) > 1.0
        b_hat = np.asarray(g.attrs["tracer_bias_hat"])
        b_anchor = np.asarray(g.attrs["tracer_bias"])
        assert b_hat.shape == (2,)
        # The maximum is a DIFFERENT point from the anchor; if it were not,
        # the residual above could not have been large.
        assert np.max(np.abs(np.log(b_hat / b_anchor))) > 1e-3
        # The covariance is the inverse of the curvature that was gated.
        C_log = np.asarray(json.loads(g.attrs["bias_profile_cov_log"]))
        H_log = np.asarray(json.loads(g.attrs["bias_profile_curvature_log"]))
        assert np.allclose(C_log @ H_log, np.eye(2), atol=1e-10)
        assert np.all(np.asarray(
            g.attrs["bias_profile_curvature_log_eigenvalues"]) > 0.0)
        assert "PROFILE MAXIMUM" in g.attrs["bias_cov_convention"]

        # The K = 1 scalar names denote K = 1 objects and must not be filled
        # with repurposed K >= 2 quantities: there is no scalar s_b here, no
        # systematics floor, and the rank-1 along-v ratio has no K >= 2
        # reading.
        for gone in ("s_b", "s_b_profile", "s_b_floor", "s_b_floor_active",
                     "s_b_floor_frac", "b_gal_curvature_profile",
                     "b_gal_curvature_conditional",
                     "b_gal_spread_inflation_along_v"):
            assert gone not in g.attrs, gone
        infl = np.asarray(g.attrs["bias_spread_inflation_along_v_by_tracer"])
        assert infl.shape == (2,) and np.all(infl >= 1.0)


@pytest.mark.slow
def test_bias_profile_residual_gate_refuses_a_non_stationary_point(tmp_path):
    """``--bias-profile-outer 0`` leaves the profile AT the anchor.

    That is the pre-fix evaluation point exactly, so the gate must refuse it
    rather than write a covariance -- otherwise "evaluated at the maximum" is a
    claim the artifact makes without checking.
    """
    w = _write_world(tmp_path)
    cmd = [sys.executable, "-m", "darksirens.cli.build_latent_field",
           "--survey", w["survey"], "--selection-fit", w["sel"],
           "--n0-calibration", w["cal"],
           "--per-pixel-completeness", w["mth1"],
           "--out", str(tmp_path / "nonstationary.h5"), "--om0", "0.3075",
           "--z-depth", str(Z_DEPTH), "--n-shells", str(N_SHELL),
           "--m-sph", "16", "--m-z", "3", "--ls-sph", "0.9", "--ls-z", "0.15",
           "--m-draw", "4", "--n-b-nodes", "5", "--b-max", "2.0",
           "--tracer-labels", "tracer",
           "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0",
           "--bias-profile-outer", "0"]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("DARKSIRENS_ZMAX", None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    assert r.returncode != 0
    assert "did not reach a stationary point" in (r.stdout + r.stderr)
    assert not (tmp_path / "nonstationary.h5").exists()


@pytest.mark.slow
def test_k2_artifact_is_refused_by_the_single_catalog_seam(tmp_path):
    """A K >= 2 anchor must not silently load into the K = 1 seam.

    It would load: the top-level ``A``/``B`` are eq. (2) over tracer 0's
    completeness while ``xi_hat`` and ``row_fac`` are the STACKED fit, so the
    seam would normalize a K-tracer field against one tracer's budget with
    nothing to complain about.  The refusal lives where the artifact is opened.
    """
    from darksirens.likelihood.latent_q import load_latent_plan

    w = _write_world(tmp_path)
    out = tmp_path / "k2_seam.h5"
    _build(w, out, ["--tracer-labels", "tracer",
                    "--tracer-completeness", w["mth1"],
                    "--tracer-completeness", w["mth2"],
                    "--tracer-b-gal", "1.0", "--tracer-b-gal", "2.0"])
    with pytest.raises(ValueError, match="n_tracer=2"):
        load_latent_plan(str(out), z_depth=Z_DEPTH, expect_nside=NSIDE)


@pytest.mark.slow
def test_tracer_label_shape_is_checked(tmp_path):
    w = _write_world(tmp_path)
    with h5py.File(w["survey"], "a") as f:
        f.create_dataset("bad_labels", data=np.zeros((NPIX, NMAX - 1),
                                                     dtype=np.int32))
    cmd = [sys.executable, "-m", "darksirens.cli.build_latent_field",
           "--survey", w["survey"], "--selection-fit", w["sel"],
           "--n0-calibration", w["cal"],
           "--per-pixel-completeness", w["mth1"],
           "--out", str(tmp_path / "bad.h5"), "--om0", "0.3075",
           "--z-depth", str(Z_DEPTH), "--n-shells", str(N_SHELL),
           "--m-sph", "16", "--m-z", "3", "--ls-sph", "0.9", "--ls-z", "0.15",
           "--tracer-labels", "bad_labels"]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("DARKSIRENS_ZMAX", None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    assert r.returncode != 0
    assert "PER GALAXY" in (r.stdout + r.stderr)
