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

    sel = tmp_path / "selection_fit.json"
    sel.write_text(json.dumps(dict(
        format_version="darksirens-selection-fit-1.0",
        strata=[dict(family="gaussian", m_lim=20.0, M0hat=-20.2, sigma_M=1.0,
                     cov=[[1e-6, 0.0], [0.0, 1e-6]], n_gal=int(ngals.sum()),
                     stratum="all", k_corr_coeffs=[],
                     meta=dict(Om0=0.3075, w0=-1.0, wa=0.0, H0_ref=100.0,
                               nll=0.0))])))
    cal = tmp_path / "n0_calibration.json"
    cal.write_text(json.dumps(dict(log10n0=-4.3, delta=0.0)))
    return dict(survey=str(survey), mth1=str(mth1), mth2=str(mth2),
                sel=str(sel), cal=str(cal))


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
        # K >= 2 ADDS COLUMNS: 4 theta + K biases against 4 theta + 1.
        assert gb["sensitivity_S"].shape[1] == ga["sensitivity_S"].shape[1] + 1
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
