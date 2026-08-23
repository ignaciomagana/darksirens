"""``--q-support-depth``: Q owns PLACEMENT inside the catalog's support only.

The real DESI table is HARD-TRUNCATED at z = 0.3000 (measured: 0 galaxies
above).  The radial builder fits the full package zgrid, so above that
truncation every covered pixel hands the solver an ``N_obs = 0`` row against a
strictly positive model expectation ``f_p C(z) dN_exp``, and the MAP pushes
that pixel's Q down in proportion to its own model rate.  Under ``--depth-map``
that rate IS ``f_p``, so the build manufactures a coherent ``corr(Q, f_p)``
purely out of empty sky -- measured -0.86 at z ~ 0.65 on the v2 build, which
is 8.6x the mask-free tolerance and cost that table its ``f_p_aware`` stamp.

The cut says the missing-host budget above the support is ``C(z)``/``n0``'s job,
not Q's, and pins ``logQ`` to exactly 0 there (the same discipline the latent
seam applies outside its own support).

WHAT IS PINNED, AND WHY IT IS NOT "BIT-IDENTICAL BELOW THE CUT".  The cut is
applied to the SOLVE GRID, not to its output: ``chi_u`` is rebuilt over
``[chi(0), chi(z_top_fitted)]`` so ``N_obs``, ``C`` and the homogeneous
expectation never contain a bin above the support.  Each row's MAP is an
``N_grid``-dimensional field under a CIRCULANT (FFT) prior
(``gaussian_correlation_spectrum``), which couples every node to every other,
so shortening the domain necessarily moves the solution at the fitted nodes
too -- measured max|dlogQ| = 0.117 below a z <= 0.5 cut on this fixture.  Bit
identity below the cut would therefore be evidence of the WRONG
implementation: the one that masks the output while the fit still sees the
empty high-z bins.  What IS pinned here:

* exactly-zero logQ above the cut (bit-zero, every pixel, MAP and members);
* a cut at/above the grid top is BIT-IDENTICAL to no flag at all (the machinery
  is a pure restriction and adds nothing of its own);
* the fit really moved -- below-support values differ from the uncut build;
* the per-z budget renormalization runs per z-SLICE over the FITTED nodes only.
"""
import json

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("h5py")

from test_completion_depth_map_build import (            # noqa: E402
    NPIX,
    _write_depth_map,
    _write_selection_fit,
    _write_stratum_map,
    _write_survey,
)

Z_CUT = 0.5


def _n_fitted(z_cut=Z_CUT):
    from darksirens.redshift import zgrid

    return int(np.count_nonzero(np.asarray(zgrid, dtype=float) <= float(z_cut)))


def _covered_f_p():
    """Occupied 0-3 unmasked, empty-but-covered 4-7, off-footprint 8-11."""
    return np.array([1.0] * 4 + [0.9, 0.9, 0.4, 0.4] + [0.0] * 4)


# ---------------------------------------------------------------------------
# above the support: bit-zero logQ, and the stamp that says so
# ---------------------------------------------------------------------------

def test_above_support_logq_is_exactly_zero_and_the_cut_is_stamped(tmp_path):
    """Q = 1 EXACTLY above the cut -- not clipped, not small, bit-zero.

    Every pixel: occupied, empty-but-covered, and off footprint alike.  A
    "nearly zero" table would still hand the missing-host budget above the
    support to Q, which is precisely the division of labour the flag exists to
    restore.
    """
    from darksirens.cli.build_lognormal_completion import main
    from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5

    cat = _write_survey(tmp_path / "survey.h5")
    dmap = _write_depth_map(tmp_path / "depth.h5", _covered_f_p())
    out = tmp_path / "q.h5"
    # --allow-unconverged: the 200-iteration cap keeps the fixture fast; this
    # test is about WHERE logQ is zero, not about the solve's fixed point.
    main(["--catalog", cat, "--out", str(out), "--n-members", "2",
          "--maxiter", "200", "--mode", "radial", "--c-mode", "aggregate",
          "--allow-unconverged",
          "--depth-map", dmap, "--q-support-depth", str(Z_CUT)])

    n = _n_fitted()
    data = load_lss_completion_hdf5(str(out))
    lq, mem = data["logq_map"], data["logq_members"]
    assert lq.shape == (NPIX, len(data["zgrid"]))
    assert np.all(lq[:, n:] == 0.0)
    assert np.all(mem[:, :, n:] == 0.0)
    # ... and the fitted block is not trivially zero as well (the cut is a
    # truncation, not a wipe).
    assert float(np.abs(lq[:, :n]).max()) > 1e-3

    assert data["q_support_depth"] == pytest.approx(Z_CUT)
    diag = data["diagnostics"]
    assert diag["q_support_depth"] == pytest.approx(Z_CUT)
    assert diag["n_z_nodes_fitted"] == n
    assert diag["n_z_nodes_pinned"] == len(data["zgrid"]) - n


def test_an_unused_flag_writes_no_support_attr(tmp_path):
    """Absent, not False-y: a table with no cut must not claim one."""
    import h5py

    from darksirens.cli.build_lognormal_completion import main
    from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5

    cat = _write_survey(tmp_path / "survey.h5")
    out = tmp_path / "q.h5"
    main(["--catalog", cat, "--out", str(out), "--n-members", "0",
          "--maxiter", "200", "--mode", "radial", "--c-mode", "aggregate",
          "--allow-unconverged"])
    with h5py.File(str(out), "r") as f:
        assert "q_support_depth" not in f["lss_completion"].attrs
    assert load_lss_completion_hdf5(str(out))["q_support_depth"] is None


# ---------------------------------------------------------------------------
# the cut is a restriction of the FIT, not a mask on its output
# ---------------------------------------------------------------------------

def test_a_cut_at_the_grid_top_is_bit_identical_to_no_flag(tmp_path):
    """Cutting nothing must change nothing -- to the last bit.

    The flag rebuilds the solve grid, the power spectrum and the budget-weight
    block; if any of that were subtly different from the uncut recipe, every
    shipped table would silently move the day the flag was added.  With
    Z = zgrid[-1] the rebuild must collapse onto the original arrays exactly.
    """
    from darksirens.cli.build_lognormal_completion import build_completion
    from darksirens.redshift import zgrid

    cat = _write_survey(tmp_path / "survey.h5")
    dmap = _write_depth_map(tmp_path / "depth.h5", _covered_f_p())
    kw = dict(mode="radial", n_members=2, maxiter=200, workers=1,
              c_mode="aggregate", depth_map=dmap)
    lq_ref, mem_ref, diag_ref = build_completion(cat, **kw)
    lq_top, mem_top, diag_top = build_completion(
        cat, q_support_depth=float(np.asarray(zgrid)[-1]), **kw)

    np.testing.assert_array_equal(lq_top, lq_ref)
    np.testing.assert_array_equal(mem_top, mem_ref)
    np.testing.assert_array_equal(diag_top["budget_monopole_logq"],
                                  diag_ref["budget_monopole_logq"])
    assert diag_top["n_z_nodes_pinned"] == 0


def test_the_cut_moves_the_solve_not_only_the_output(tmp_path):
    """Below-support values MUST differ from the uncut build.

    Each row's MAP is a field under a circulant FFT prior coupling every radial
    node, so a genuinely shorter domain moves the fitted nodes too (measured
    max|dlogQ| 0.117 here).  An implementation that masked the output while the
    solver still saw the empty high-z bins would leave this difference at
    exactly 0 -- and would leave the manufactured corr(Q, f_p) in the fit.
    """
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    dmap = _write_depth_map(tmp_path / "depth.h5", _covered_f_p())
    kw = dict(mode="radial", n_members=0, maxiter=200, workers=1,
              c_mode="aggregate", depth_map=dmap)
    lq_full, _m, _d = build_completion(cat, **kw)
    lq_cut, _m, diag = build_completion(cat, q_support_depth=Z_CUT, **kw)

    n = _n_fitted()
    assert float(np.abs(lq_cut[:, :n] - lq_full[:, :n]).max()) > 1e-3
    # the solve grid itself shrank with the domain
    assert diag["n_z_nodes_fitted"] == n


def test_the_flag_is_independent_of_the_depth_map(tmp_path):
    """No --depth-map: the cut still applies (its use with f_p is a motive, not
    a requirement)."""
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    lq, _m, diag = build_completion(
        cat, mode="radial", n_members=0, maxiter=200, workers=1,
        c_mode="per_pixel", q_support_depth=Z_CUT)
    n = _n_fitted()
    assert np.all(lq[:, n:] == 0.0)
    assert diag["q_support_depth"] == pytest.approx(Z_CUT)


# ---------------------------------------------------------------------------
# the budget renormalization: per z-slice, fitted nodes only
# ---------------------------------------------------------------------------

def test_budget_renormalization_runs_per_z_slice_over_fitted_nodes_only(tmp_path):
    """renormalize_q_mean_one removes mono[z] = sum_p w_p Q_p / sum_p w_p.

    That is a PER-Z-BIN shift, identical across rows and mixing no two z bins,
    so restricting it to the fitted column block is well defined.  Pinned nodes
    must be handed no shift at all: a renormalization run over the full grid
    would give the pinned columns a monopole computed from Q = 1 rows and drag
    them off bit-zero the moment any fitted row's weights were not uniform.

    Measured against a ``--no-budget-renorm`` build of the same fixture: the
    difference on each fitted column is one constant, equal to
    -budget_monopole_logq[z], and is exactly 0 on every pinned column.
    """
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    dmap = _write_depth_map(tmp_path / "depth.h5", _covered_f_p())
    kw = dict(mode="radial", n_members=0, maxiter=200, workers=1,
              c_mode="aggregate", depth_map=dmap, q_support_depth=Z_CUT)
    lq_raw, _m, _d = build_completion(cat, budget_renorm=False, **kw)
    lq_ren, _m, diag = build_completion(cat, budget_renorm=True, **kw)

    n = _n_fitted()
    fit_rows = np.nonzero(_covered_f_p() > 0.0)[0]
    shift = lq_ren[fit_rows] - lq_raw[fit_rows]                # (n_fit, n_grid)
    mono = np.asarray(diag["budget_monopole_logq"], dtype=float)

    # one shift per z bin, shared by every fitted row (no cross-z mixing)
    np.testing.assert_allclose(
        shift[:, :n], np.broadcast_to(-mono[:n], shift[:, :n].shape),
        atol=1e-12)
    assert float(np.ptp(shift[:, :n], axis=0).max()) < 1e-12
    # the renormalization is a NO-OP above the cut, bit-exactly
    assert np.all(shift[:, n:] == 0.0)
    assert np.all(mono[n:] == 0.0)
    assert np.all(lq_ren[:, n:] == 0.0)
    # it did real work below the cut (otherwise the check above is vacuous)
    assert float(np.abs(mono[:n]).max()) > 0.0


# ---------------------------------------------------------------------------
# paths that cannot honour the cut must refuse it, not drop it
# ---------------------------------------------------------------------------

def test_q_support_depth_with_gp3d_is_refused_before_any_compute(tmp_path):
    """gp3d fits one field on its own zeta inducing grid; there is no radial
    zgrid cut to apply, so the flag would be silently dropped while the table
    still carried the stamp.

    The catalog here does not exist: the refusal must land before ANY I/O.
    """
    from darksirens.cli.build_lognormal_completion import main

    out = tmp_path / "q.h5"
    with pytest.raises(ValueError, match="not honoured by --mode 'gp3d'"):
        main(["--catalog", str(tmp_path / "missing.h5"), "--out", str(out),
              "--n-members", "0", "--mode", "gp3d", "--c-mode", "aggregate",
              "--q-support-depth", str(Z_CUT),
              "--gp3d-nz-nodes", "200", "--gp3d-nsph-nodes", "8"])
    assert not out.exists()


def test_q_support_depth_with_stratified_selection_is_refused(tmp_path, monkeypatch):
    """The per-stratum C_sel base is a separate branch the cut is not validated
    on; the pair is refused rather than stamped on an untested path."""
    import darksirens.cli.build_lognormal_completion as mod

    cat = _write_survey(tmp_path / "survey.h5")
    fit = _write_selection_fit(tmp_path / "fit2.json",
                               [(21.0, -20.8, 0.8), (20.5, -20.5, 0.9)])
    smap = _write_stratum_map(tmp_path / "strata.h5", np.arange(NPIX) % 2)
    out = tmp_path / "q.h5"

    def _no_compute(*a, **k):                       # pragma: no cover - guard
        raise AssertionError("the builder ran before the refusal")

    monkeypatch.setattr(mod, "_build_completion_radial", _no_compute)
    with pytest.raises(ValueError, match="STRATIFIED"):
        mod.main(["--catalog", cat, "--out", str(out), "--n-members", "0",
                  "--mode", "radial", "--c-mode", "selection",
                  "--selection-fit", fit, "--stratum-map", smap,
                  "--q-support-depth", str(Z_CUT)])
    assert not out.exists()


def test_a_cut_below_the_second_zgrid_node_is_refused(tmp_path):
    """Fewer than two fitted nodes is not a short fit, it is no field at all."""
    from darksirens.cli.build_lognormal_completion import build_completion
    from darksirens.redshift import zgrid

    cat = _write_survey(tmp_path / "survey.h5")
    tiny = 0.5 * float(np.asarray(zgrid)[1])
    with pytest.raises(ValueError, match="no radial field to fit"):
        build_completion(cat, mode="radial", n_members=0, maxiter=50,
                         c_mode="aggregate", q_support_depth=tiny)


@pytest.mark.parametrize("bad", [0.0, -0.3, float("nan")])
def test_a_nonpositive_or_nonfinite_cut_is_refused(tmp_path, bad):
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    with pytest.raises(ValueError, match="finite redshift > 0"):
        build_completion(cat, mode="radial", n_members=0, maxiter=50,
                         c_mode="aggregate", q_support_depth=bad)


def test_the_stamp_survives_a_round_trip_through_the_save_path(tmp_path):
    """The attr is written by save_lss_completion_hdf5's own kwarg, the same
    pass-through f_p_aware uses -- not by a caller poking the file."""
    from darksirens.redshift.lognormal_completion import (
        load_lss_completion_hdf5,
        save_lss_completion_hdf5,
    )

    path = str(tmp_path / "q.h5")
    save_lss_completion_hdf5(
        path, logq_map=np.zeros((4, 6)), zgrid=np.linspace(0, 1, 6),
        indexing="global", metadata={"note": json.dumps({"k": 1})},
        q_support_depth=0.30)
    assert load_lss_completion_hdf5(path)["q_support_depth"] == pytest.approx(0.30)
