"""Per-catalog --validate_completion for a K>=2 mixture run
(cli/inference.py run_completion_validation_multitracer).

The K>=2 dry run reuses the single-catalog diagnostic verbatim per catalog:
full survey rows re-read host-side, paired with the per-sample pixel indices
the bundle loader stashes when --validate_completion is set, and one
``completion_validation_c{k}__<ts>.json`` written per catalog."""
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from darksirens.cli.inference import (
    run_completion_validation,
    run_completion_validation_multitracer,
)

NSIDE = 1
NPIX = 12
MAXGALS = 4


def _write_survey(path, seed, weight=1.0):
    rng = np.random.default_rng(seed)
    zgals = np.zeros((NPIX, MAXGALS))
    dzgals = np.full((NPIX, MAXGALS), 0.02)
    wgals = np.zeros((NPIX, MAXGALS))
    ngals = np.zeros(NPIX, dtype=np.int32)
    for pix in range(0, NPIX, 2):  # populate every other pixel
        n = int(rng.integers(2, MAXGALS + 1))
        zgals[pix, :n] = rng.uniform(0.05, 0.4, size=n)
        wgals[pix, :n] = weight
        ngals[pix] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
        f.create_dataset("ngals", data=ngals)


def _bundle(pixels_pe, pixels_sel):
    import healpy as hp

    return dict(
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
        pixels_pe_full=np.asarray(pixels_pe, dtype=np.int32),
        pixels_sel_full=np.asarray(pixels_sel, dtype=np.int32),
    )


@pytest.fixture
def two_catalog_setup(tmp_path):
    # The two catalogs' GW samples land in DIFFERENT pixels (per-catalog
    # ang2pix at each catalog's own nside does this in production): the
    # diagnostics' per-pixel entries are the catalog-dependent payload.
    paths = []
    for k, seed in enumerate((101, 202), start=1):
        p = tmp_path / f"survey_c{k}.h5"
        _write_survey(p, seed)
        paths.append(str(p))
    opts = SimpleNamespace(
        survey_paths=paths,
        n_catalogs=2,
        save_path=str(tmp_path / "out"),
        completion_validation_pixels=8,
    )
    data = dict(catalogs=[
        _bundle([0, 2, 4, 6, 0, 2], [0, 2, 4, 8, 10]),
        _bundle([1, 3, 5, 7, 1, 3], [1, 3, 5, 9, 11]),
    ])
    return opts, data


def test_writes_one_json_per_catalog(two_catalog_setup):
    opts, data = two_catalog_setup
    paths = run_completion_validation_multitracer(opts, data, {}, {})
    assert len(paths) == 2
    for k, p in enumerate(paths, start=1):
        assert os.path.exists(p)
        assert f"completion_validation_c{k}__" in os.path.basename(p)
        with open(p) as f:
            diag = json.load(f)
        assert "survey_values" in diag and "cosmology_values" in diag


def test_catalogs_get_independent_diagnostics(two_catalog_setup):
    """Each catalog is validated on its OWN per-sample pixel indices: the
    per-pixel diagnostic entries must reflect each bundle's pixel set, not a
    shared/broadcast one."""
    opts, data = two_catalog_setup
    p1, p2 = run_completion_validation_multitracer(opts, data, {}, {})
    with open(p1) as f:
        d1 = json.load(f)
    with open(p2) as f:
        d2 = json.load(f)
    pix1 = [e["global_pixel"] for e in d1["per_pixel"]]
    pix2 = [e["global_pixel"] for e in d2["per_pixel"]]
    assert pix1 == sorted(set([0, 2, 4, 6, 8, 10]))[:8]
    assert pix2 == sorted(set([1, 3, 5, 7, 9, 11]))[:8]


def test_missing_stashed_pixels_is_fatal(two_catalog_setup):
    opts, data = two_catalog_setup
    del data["catalogs"][1]["pixels_pe_full"]
    with pytest.raises(SystemExit):
        run_completion_validation_multitracer(opts, data, {}, {})


def test_single_catalog_json_name_unchanged(tmp_path):
    """The K=1 filename contract (no suffix) is untouched."""
    p = tmp_path / "survey.h5"
    _write_survey(p, 303)
    import healpy as hp
    from darksirens.inference.loaders import load_survey

    _, ngals, zgals, dzgals, wgals, _ = load_survey(str(p), to_device=False)
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = dict(
        zgals_catalog=zgals, dzgals_catalog=dzgals, wgals_catalog=wgals,
        ngals_catalog=ngals,
        pixels_pe=np.array([0, 2, 4], dtype=np.int32),
        pixels_sel=np.array([0, 2, 6], dtype=np.int32),
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
    )
    path = run_completion_validation(opts, data, {}, {})
    assert os.path.basename(path).startswith("completion_validation__")


# ---------------------------------------------------------------------------
# Q_LSS awareness: the dry run must diagnose the factor the likelihood will use
# ---------------------------------------------------------------------------

def _saturating_logq(n_pix=NPIX):
    """Global (n_pix, N_grid) log-Q that rails the +-7 clip over much of the grid."""
    from darksirens.redshift import zgrid

    z = np.asarray(zgrid)
    return 9.0 * np.sin(6.0 * z)[None, :] * np.ones((n_pix, 1))


def _k1_validation_data(tmp_path, seed=303):
    import healpy as hp
    from darksirens.inference.loaders import load_survey

    p = tmp_path / "survey.h5"
    _write_survey(p, seed)
    _, ngals, zgals, dzgals, wgals, _ = load_survey(str(p), to_device=False)
    return dict(
        zgals_catalog=zgals, dzgals_catalog=dzgals, wgals_catalog=wgals,
        ngals_catalog=ngals,
        pixels_pe=np.array([0, 2, 4], dtype=np.int32),
        pixels_sel=np.array([0, 2, 6], dtype=np.int32),
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
    )


def test_validation_diagnoses_the_attached_Q_LSS_table(tmp_path):
    """Regression: the dry-run catalog was built WITHOUT the loaded Q_LSS table,
    so completion_clip_diagnostics silently took the legacy delta_g branch and
    reported a 0.0 clip fraction -- an operator checking a Q table before a
    campaign saw "clean" while the Q was pinned at the +-7 logQ bound over much of
    the redshift grid.  With the table attached the diagnostic must name Q_LSS as
    the source and report a NON-ZERO clip fraction."""
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = _k1_validation_data(tmp_path)

    with open(run_completion_validation(opts, dict(data), {}, {})) as f:
        blind = json.load(f)
    assert blind["lss_source"] == "legacy_delta_g"
    assert blind["lss_completion_attached"] == "none"
    assert blind["mean_rho_miss_eff_clipped_fraction"] == 0.0

    data_q = dict(data, lss_completion_logq=_saturating_logq(),
                  lss_completion_indexing=2)
    with open(run_completion_validation(opts, data_q, {}, {})) as f:
        seeing = json.load(f)
    assert seeing["lss_source"] == "Q_LSS"
    assert seeing["lss_completion_attached"] == "global_table_sliced"
    assert seeing["mean_rho_miss_eff_clipped_fraction"] > 0.1
    # C_eff clips too, where Q > 1 pushes 1 - (1 - C) Q negative.
    assert seeing["mean_C_eff_clipped_fraction"] > 0.0


def test_validation_skips_and_reports_a_compact_Q_table(tmp_path):
    """A COMPACT per-view Q block has no global pixel key, so it CANNOT be aligned
    to the validation pixels; the JSON must say so rather than imply a clean Q."""
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = dict(
        _k1_validation_data(tmp_path),
        lss_completion_logq=_saturating_logq(3), lss_completion_indexing=1,
    )
    with open(run_completion_validation(opts, data, {}, {})) as f:
        diag = json.load(f)
    assert diag["lss_completion_attached"] == "compact_table_skipped"
    assert diag["lss_source"] == "legacy_delta_g"


def test_per_catalog_validation_uses_each_bundles_own_Q(two_catalog_setup):
    """The K>=2 twin was worse: ``data_k`` omitted the Q table AND delta_g, so a
    --use_lss / --lss_completion mixture run got the (1, N_grid) dummy for every
    catalog.  Catalog 1 here carries a saturating Q, catalog 2 none."""
    opts, data = two_catalog_setup
    data["catalogs"][0]["lss_completion_logq"] = _saturating_logq()
    data["catalogs"][0]["lss_completion_indexing"] = 2
    p1, p2 = run_completion_validation_multitracer(opts, data, {}, {})
    with open(p1) as f:
        d1 = json.load(f)
    with open(p2) as f:
        d2 = json.load(f)
    assert d1["lss_source"] == "Q_LSS"
    assert d1["mean_rho_miss_eff_clipped_fraction"] > 0.1
    assert d2["lss_source"] == "legacy_delta_g"
    assert d2["lss_completion_attached"] == "none"


# ---------------------------------------------------------------------------
# c_mode awareness (deferred review MINOR): the dry run must diagnose the
# SAME completeness estimator the likelihood will form
# ---------------------------------------------------------------------------

def _k1_data(tmp_path, seed=404):
    import healpy as hp
    from darksirens.inference.loaders import load_survey

    p = tmp_path / "survey.h5"
    _write_survey(p, seed)
    _, ngals, zgals, dzgals, wgals, _ = load_survey(str(p), to_device=False)
    return dict(
        zgals_catalog=zgals, dzgals_catalog=dzgals, wgals_catalog=wgals,
        ngals_catalog=ngals,
        pixels_pe=np.array([0, 2, 4], dtype=np.int32),
        pixels_sel=np.array([0, 2, 6], dtype=np.int32),
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
    )


def _spy_survey(monkeypatch):
    """Capture the SurveyParams the dry run hands to the diagnostics."""
    from darksirens.cli import inference as cli

    seen = []
    real = cli.completion_clip_diagnostics

    def spy(*, cosmo, survey, em_catalog, max_pixels):
        seen.append(survey)
        return real(cosmo=cosmo, survey=survey, em_catalog=em_catalog,
                    max_pixels=max_pixels)

    monkeypatch.setattr(cli, "completion_clip_diagnostics", spy)
    return seen


def test_validation_survey_carries_the_runs_c_mode(tmp_path, monkeypatch):
    """per_pixel (legacy default) keeps c_mode=None; aggregate and selection
    runs hand the diagnostics a survey carrying THEIR estimator's structure
    -- previously every mode was silently validated as per_pixel."""
    import json as _json

    from darksirens.core.types import (
        C_MODE_AGGREGATE_STRUCT, C_MODE_SELECTION_STRUCT,
    )

    seen = _spy_survey(monkeypatch)
    base = dict(save_path=str(tmp_path / "out"),
                completion_validation_pixels=8)

    opts = SimpleNamespace(**base)                      # legacy: no c_mode
    run_completion_validation(opts, _k1_data(tmp_path, 404), {}, {})
    assert seen[-1].c_mode is None

    opts = SimpleNamespace(**base, c_mode="aggregate")
    p_agg = run_completion_validation(opts, _k1_data(tmp_path, 405), {}, {})
    assert seen[-1].c_mode is C_MODE_AGGREGATE_STRUCT
    assert _json.load(open(p_agg))["c_mode"] == "aggregate"

    # Wide-open selection (no fit): theta at the registry fiducials, m_lim
    # overridable through the fixed values exactly as in the likelihood.
    opts = SimpleNamespace(**base, c_mode="selection", n_catalogs=1,
                           selection_fit_paths=[None], stratum_map=None)
    p_sel = run_completion_validation(
        opts, _k1_data(tmp_path, 406), {}, {"m_lim": 19.5})
    sv = seen[-1]
    assert sv.c_mode is C_MODE_SELECTION_STRUCT
    assert float(sv.m_lim) == 19.5
    d = _json.load(open(p_sel))
    assert d["c_mode"] == "selection"
    assert d["selection_theta_used"]["m_lim"] == 19.5
    assert d["selection_family"] == "gaussian"


def test_validation_selection_theta_comes_from_the_fit(tmp_path, monkeypatch):
    """With a --selection_fit, the dry run resolves the SAME per-catalog fit
    records the likelihood uses: theta at the fit's effective values, the
    K(z) template threaded, m_lim pinned from the fit's datum."""
    import json as _json

    from darksirens.redshift.selection import SelectionFit

    fit = SelectionFit(family="gaussian", m_lim=20.25, M0hat=-20.4,
                       sigma_M=0.85, cov=np.eye(2) * 4e-4, n_gal=1000,
                       k_corr_coeffs=(0.7,))
    fp = tmp_path / "fit.json"
    fp.write_text(_json.dumps({
        "format_version": "darksirens-selection-fit-1.0",
        "strata": [fit.to_jsonable()]}))

    seen = _spy_survey(monkeypatch)
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8,
        c_mode="selection", n_catalogs=1,
        selection_fit_paths=[str(fp)], stratum_map=None)
    fixed = {}
    run_completion_validation(opts, _k1_data(tmp_path, 407), {}, fixed)
    sv = seen[-1]
    assert float(sv.m_lim) == 20.25          # pinned from the fit's datum
    assert float(sv.M0hat) == -20.4
    assert float(sv.sigma_M) == 0.85
    assert sv.k_corr_coeffs == (0.7,)
    assert fixed["m_lim"] == 20.25           # the same pin the run gets


# ---------------------------------------------------------------------------
# Provenance of the attached Q leaves, and the representative cosmology
# ---------------------------------------------------------------------------

def test_mixed_q_leaf_provenance_is_reported_per_key(tmp_path):
    """The single provenance scalar was last-write-wins over the four Q leaves,
    so a run whose GLOBAL deterministic table WAS attached while a
    wrongly-shaped members block was skipped reported 'compact_table_skipped'
    (members come last in _LSS_TABLE_ROW_AXIS order) -- the reader could not
    tell which leaf the clip fraction came from."""
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = dict(
        _k1_validation_data(tmp_path),
        lss_completion_logq=_saturating_logq(),          # global: attached
        # 3 pixel rows for a 12-pixel sky: no global pixel key -> skipped.
        lss_completion_logq_members=np.stack([_saturating_logq(3)] * 2),
        lss_completion_indexing=2,
    )
    with open(run_completion_validation(opts, data, {}, {})) as f:
        diag = json.load(f)
    assert diag["lss_completion_attached"] == "partial"
    by_key = diag["lss_completion_attached_by_key"]
    assert by_key["lss_completion_logq"] == "global_table_sliced"
    assert by_key["lss_completion_logq_members"] == "compact_table_skipped"
    # The attached deterministic table still drove the diagnostic.
    assert diag["lss_source"] == "Q_LSS"


def test_uniform_q_leaf_provenance_keeps_the_scalar_and_maps_every_leaf(tmp_path):
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = dict(_k1_validation_data(tmp_path),
                lss_completion_logq=_saturating_logq(),
                lss_completion_indexing=2)
    with open(run_completion_validation(opts, data, {}, {})) as f:
        diag = json.load(f)
    assert diag["lss_completion_attached"] == "global_table_sliced"
    assert diag["lss_completion_attached_by_key"] == {
        "lss_completion_logq": "global_table_sliced"}


def test_dry_run_cosmology_follows_prior_overrides(tmp_path):
    """The clip fractions are cosmology-dependent (dN_exp carries the comoving
    volume element), so the representative cosmology must follow
    --prior_overrides exactly as the survey block does -- pinning Planck15 under
    an H0 override certified a budget the run never forms."""
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    with open(run_completion_validation(
            opts, _k1_data(tmp_path, 501),
            {"H0": [40.0, 55.0], "log10n0": [-3.0, -1.0]}, {})) as f:
        overridden = json.load(f)
    assert overridden["cosmology_values"]["H0"] == 47.5
    assert overridden["survey_values"]["log10n0"] == -2.0

    # An explicit fixed value still wins over the override midpoint.
    with open(run_completion_validation(
            opts, _k1_data(tmp_path, 502),
            {"H0": [40.0, 55.0]}, {"H0": 70.0})) as f:
        fixed = json.load(f)
    assert fixed["cosmology_values"]["H0"] == 70.0

    # No override, no fixed value: the Planck15 fiducial, unchanged.
    from darksirens.core.constants import H0_FID, W0_FID, WA_FID

    with open(run_completion_validation(
            opts, _k1_data(tmp_path, 503), {}, {})) as f:
        default = json.load(f)
    assert default["cosmology_values"]["H0"] == float(H0_FID)
    assert default["cosmology_values"]["w0"] == float(W0_FID)
    assert default["cosmology_values"]["wa"] == float(WA_FID)
