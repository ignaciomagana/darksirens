"""Ensemble-provenance stamping (save_lss_completion_hdf5) and the K>=2
--lss_marginalize matched-realizations guard in
load_multitracer_catalog_bundles.

The estimator marginalizes a K-catalog mixture over ONE shared member index
(member m of every catalog must sample the SAME LSS realization).  The Q files
now carry a /lss_completion realization_set_id so the loader can verify that
per-catalog ensembles come from one matched realization set before it pairs
their members; independently built ensembles (distinct uuids) or legacy files
(no provenance) abort unless --allow_unverified_shared_lss_members is set.
"""
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from darksirens.redshift.grid import zgrid
from darksirens.redshift.lognormal_completion import (
    load_lss_completion_hdf5,
    save_lss_completion_hdf5,
)
from darksirens.inference.loaders import load_multitracer_catalog_bundles
from darksirens.inference.validation import validate_multitracer_run

NSIDE = 1
NPIX = 12
MAXGALS = 4
NGRID = len(zgrid)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_survey(path, seed):
    """Tiny nside=1 survey (mirrors test_completion_validation_multitracer)."""
    rng = np.random.default_rng(seed)
    zgals = np.zeros((NPIX, MAXGALS))
    dzgals = np.full((NPIX, MAXGALS), 0.02)
    wgals = np.zeros((NPIX, MAXGALS))
    ngals = np.zeros(NPIX, dtype=np.int32)
    for pix in range(0, NPIX, 2):
        n = int(rng.integers(2, MAXGALS + 1))
        zgals[pix, :n] = rng.uniform(0.05, 0.4, size=n)
        wgals[pix, :n] = 1.0
        ngals[pix] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
        f.create_dataset("ngals", data=ngals)
    return str(path)


def _members(seed, M=3, rows=NPIX):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(M, rows, NGRID)).astype(float)


def _map(rows=NPIX):
    return np.zeros((rows, NGRID), dtype=float)


def _save_q(path, *, seed, M=3, realization_set_id=None, with_members=True):
    save_lss_completion_hdf5(
        str(path),
        logq_map=_map(),
        logq_members=_members(seed, M=M) if with_members else None,
        zgrid=np.asarray(zgrid),
        realization_set_id=realization_set_id,
    )
    return str(path)


def _write_legacy_q(path, *, seed, with_members=True):
    """Hand-written HDF5 with NO provenance attrs (pre-stamping layout)."""
    with h5py.File(str(path), "w") as f:
        grp = f.create_group("lss_completion")
        grp.create_dataset("logq_map", data=_map())
        if with_members:
            grp.create_dataset("logq_members", data=_members(seed))
        grp.create_dataset("zgrid", data=np.asarray(zgrid))
        grp.attrs["indexing"] = "compact"
    return str(path)


def _gw_inputs(n=4):
    rng = np.random.default_rng(0)
    ra = rng.uniform(0.0, 2 * np.pi, size=n)
    dec = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    rasels = rng.uniform(0.0, 2 * np.pi, size=n)
    decsels = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    return {"ra": ra, "dec": dec, "rasels": rasels, "decsels": decsels}


def _opts(survey_paths, lss_completions, **overrides):
    base = dict(
        survey_paths=list(survey_paths),
        lss_completions=list(lss_completions),
        universe_model="dark_sirens",
        lss_marginalize=True,
        use_LSS=False,
        mark_model="none",
        mark_names_by_catalog=None,
        catalog_sky_weighting="conditional",
        validate_completion=False,
        allow_unverified_shared_lss_members=False,
        n_catalogs=len(survey_paths),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# writer / loader roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_members_stamps_id_hash_and_count(tmp_path):
    members = _members(7, M=5)
    save_lss_completion_hdf5(
        str(tmp_path / "q.h5"), logq_map=_map(), logq_members=members,
        zgrid=np.asarray(zgrid),
    )
    loaded = load_lss_completion_hdf5(str(tmp_path / "q.h5"))
    rid = loaded["realization_set_id"]
    assert isinstance(rid, str) and len(rid) == 32
    int(rid, 16)  # 32-hex uuid
    expected = hashlib.sha256(
        np.ascontiguousarray(np.asarray(members, dtype=float)).tobytes()
    ).hexdigest()
    assert loaded["member_content_sha256"] == expected
    assert loaded["n_members"] == 5


def test_roundtrip_map_only_has_id_but_no_member_attrs(tmp_path):
    save_lss_completion_hdf5(
        str(tmp_path / "q.h5"), logq_map=_map(), zgrid=np.asarray(zgrid),
    )
    loaded = load_lss_completion_hdf5(str(tmp_path / "q.h5"))
    assert isinstance(loaded["realization_set_id"], str)
    assert len(loaded["realization_set_id"]) == 32
    assert loaded["member_content_sha256"] is None
    assert loaded["n_members"] is None


def test_explicit_realization_set_id_is_honored(tmp_path):
    """A future JOINT builder stamps the SAME id across several files."""
    shared = "0123456789abcdef0123456789abcdef"
    for name in ("qA.h5", "qB.h5"):
        save_lss_completion_hdf5(
            str(tmp_path / name), logq_map=_map(), logq_members=_members(1),
            zgrid=np.asarray(zgrid), realization_set_id=shared,
        )
    a = load_lss_completion_hdf5(str(tmp_path / "qA.h5"))
    b = load_lss_completion_hdf5(str(tmp_path / "qB.h5"))
    assert a["realization_set_id"] == shared == b["realization_set_id"]


def test_distinct_saves_get_distinct_ids(tmp_path):
    a = load_lss_completion_hdf5(_save_q(tmp_path / "a.h5", seed=1))
    b = load_lss_completion_hdf5(_save_q(tmp_path / "b.h5", seed=2))
    assert a["realization_set_id"] != b["realization_set_id"]


def test_legacy_file_returns_none_provenance(tmp_path):
    loaded = load_lss_completion_hdf5(_write_legacy_q(tmp_path / "legacy.h5", seed=3))
    assert loaded["realization_set_id"] is None
    assert loaded["member_content_sha256"] is None
    assert loaded["n_members"] is None
    # legacy members still load (only provenance is absent)
    assert loaded["logq_members"] is not None


# ---------------------------------------------------------------------------
# K>=2 loader guard (real load_multitracer_catalog_bundles path)
# ---------------------------------------------------------------------------

def test_k2_marginalize_same_file_passes(tmp_path):
    """The SAME Q file loaded for both catalogs shares one realization set."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    q = _save_q(tmp_path / "q.h5", seed=100)
    opts = _opts([s1, s2], [q, q])
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 2
    ids = {b["lss_completion_provenance"]["realization_set_id"] for b in bundles}
    assert len(ids) == 1 and next(iter(ids)) is not None


def test_k2_marginalize_distinct_ids_raises(tmp_path):
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    qb = _save_q(tmp_path / "qb.h5", seed=200)
    opts = _opts([s1, s2], [qa, qb])
    with pytest.raises(ValueError) as exc:
        load_multitracer_catalog_bundles(opts, _gw_inputs())
    msg = str(exc.value)
    assert qa in msg and qb in msg
    assert "realization_set_id" in msg
    assert "SHARED member index" in msg


def test_k2_marginalize_allow_flag_warns_and_proceeds(tmp_path, capsys):
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    qb = _save_q(tmp_path / "qb.h5", seed=200)
    opts = _opts([s1, s2], [qa, qb], allow_unverified_shared_lss_members=True)
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 2
    out = capsys.readouterr().out
    assert "allow_unverified_shared_lss_members" in out
    assert "INDEPENDENT-fields" in out
    assert "product prior across catalogs" in out
    # both distinct realization ids are echoed in the loud warning
    assert "realization_set_id=" in out


def test_k2_marginalize_legacy_raises_mentioning_legacy(tmp_path):
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _write_legacy_q(tmp_path / "qa.h5", seed=100)
    qb = _write_legacy_q(tmp_path / "qb.h5", seed=200)
    opts = _opts([s1, s2], [qa, qb])
    with pytest.raises(ValueError) as exc:
        load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert "LEGACY (no provenance)" in str(exc.value)


def test_k2_marginalize_legacy_allow_flag_proceeds(tmp_path, capsys):
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _write_legacy_q(tmp_path / "qa.h5", seed=100)
    qb = _write_legacy_q(tmp_path / "qb.h5", seed=200)
    opts = _opts([s1, s2], [qa, qb], allow_unverified_shared_lss_members=True)
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 2
    assert "LEGACY (no provenance)" in capsys.readouterr().out


def test_k2_deterministic_skips_provenance_check(tmp_path):
    """K=2 WITHOUT --lss_marginalize never pairs members: no provenance check
    even for independently built (distinct-id) ensembles."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    qb = _save_q(tmp_path / "qb.h5", seed=200)
    opts = _opts([s1, s2], [qa, qb], lss_marginalize=False)
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 2


def test_k1_ensemble_skips_provenance_check(tmp_path):
    """A single catalog (len(bundles) < 2) is never subject to the shared-member
    guard, even with a legacy ensemble."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    qa = _write_legacy_q(tmp_path / "qa.h5", seed=100)
    opts = _opts([s1], [qa])
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 1


def test_k2_marginalize_missing_members_on_one_catalog_raises(tmp_path):
    """A catalog with a map-only (no-members) Q file cannot supply a matched
    member set: the per-catalog loader already refuses it by name."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    qb = _save_q(tmp_path / "qb.h5", seed=200, with_members=False)
    opts = _opts([s1, s2], [qa, qb])
    with pytest.raises(ValueError, match="requires an LSS-completion ENSEMBLE"):
        load_multitracer_catalog_bundles(opts, _gw_inputs())


def test_k2_marginalize_catalog_without_any_q_names_the_missing_ensemble(tmp_path):
    """A catalog with NO completion at all (the "" placeholder) used to be
    reported as 'LEGACY (no provenance)' with an instruction to rebuild the
    ensembles jointly -- impossible for a catalog that has no LSS field.  The
    abort must name the MISSING ENSEMBLE instead (the estimator needs one on
    every catalog: likelihood/core.py refuses a missing base_miss on either
    seam)."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    opts = _opts([s1, s2], [qa, ""])
    with pytest.raises(ValueError) as exc:
        load_multitracer_catalog_bundles(opts, _gw_inputs())
    msg = str(exc.value)
    assert "EVERY catalog needs its own Q_LSS ENSEMBLE" in msg
    assert s2 in msg and s1 not in msg
    assert "LEGACY (no provenance)" not in msg


def test_k2_marginalize_no_ensemble_is_not_bypassed_by_allow_flag(tmp_path):
    """--allow_unverified_shared_lss_members accepts an UNMATCHED realization
    set, not a MISSING one: a Q-less catalog must still abort (the likelihood
    would otherwise raise mid-trace, after the whole data load)."""
    s1 = _write_survey(tmp_path / "s1.h5", 11)
    s2 = _write_survey(tmp_path / "s2.h5", 22)
    qa = _save_q(tmp_path / "qa.h5", seed=100)
    opts = _opts([s1, s2], [qa, ""],
                 allow_unverified_shared_lss_members=True)
    with pytest.raises(ValueError, match="EVERY catalog needs its own"):
        load_multitracer_catalog_bundles(opts, _gw_inputs())


# ---------------------------------------------------------------------------
# validate_multitracer_run
# ---------------------------------------------------------------------------

def test_validate_wrong_length_z_depths_raises():
    opts = SimpleNamespace(n_catalogs=2, resolved_survey_z_depths=[0.5])
    with pytest.raises(ValueError) as exc:
        validate_multitracer_run(opts, {"catalogs": [{}, {}]})
    assert "resolved_survey_z_depths" in str(exc.value)


def test_validate_k2_with_one_bundle_raises():
    opts = SimpleNamespace(n_catalogs=2, resolved_survey_z_depths=[0.5, 0.5])
    with pytest.raises(ValueError) as exc:
        validate_multitracer_run(opts, {"catalogs": [{}]})
    assert "catalog bundle" in str(exc.value)


def test_validate_wrong_length_marks_raises():
    opts = SimpleNamespace(
        n_catalogs=2, resolved_survey_z_depths=[0.5, 0.5],
        mark_names_by_catalog=(("logmstar",),),
    )
    with pytest.raises(ValueError) as exc:
        validate_multitracer_run(opts, {"catalogs": [{}, {}]})
    assert "mark_names_by_catalog" in str(exc.value)


def test_validate_wrong_length_lss_completions_raises():
    opts = SimpleNamespace(n_catalogs=2, lss_completions=["a.h5"])
    with pytest.raises(ValueError) as exc:
        validate_multitracer_run(opts, {"catalogs": [{}, {}]})
    assert "lss_completions" in str(exc.value)


def test_validate_valid_config_passes():
    opts = SimpleNamespace(
        n_catalogs=2,
        resolved_survey_z_depths=[0.5, 0.7],
        lss_completions=["a.h5", "b.h5"],
        mark_names_by_catalog=(("logmstar",), ()),
    )
    validate_multitracer_run(opts, {"catalogs": [{}, {}]})


def test_validate_valid_k1_config_passes():
    opts = SimpleNamespace(
        n_catalogs=1, resolved_survey_z_depths=[0.5], lss_completions=[],
    )
    validate_multitracer_run(opts, {})


def test_validate_bare_opts_passes():
    """Tests build minimal SimpleNamespace opts; getattr defaults tolerate them."""
    validate_multitracer_run(SimpleNamespace(), {})


def test_validate_survey_free_empty_z_depths_passes():
    """A survey-free model resolves an EMPTY z-depth list; the truthiness guard
    skips the length check (n_catalogs defaults to 1)."""
    opts = SimpleNamespace(n_catalogs=1, resolved_survey_z_depths=[])
    validate_multitracer_run(opts, {})


# ---------------------------------------------------------------------------
# Q auto-discovery must fail loud, never degrade to a silent non-LSS run
# (review finding F-040)
# ---------------------------------------------------------------------------

def _single_opts(survey_path, lss_completion, universe_model="dark_sirens"):
    return SimpleNamespace(
        survey_path=survey_path, lss_completion=lss_completion,
        universe_model=universe_model, lss_marginalize=False,
    )


def test_unreadable_survey_probe_raises_instead_of_dropping_q(tmp_path):
    """An unreadable survey must not be reported as "no embedded Q table".

    Q REPLACES the (1 + alpha_miss*b_miss*delta_g) factor and supplies the
    missing-galaxy budget, so swallowing an HDF5 read failure (Lustre lock
    contention, a stale handle, a concurrent writer) silently changes the
    completeness model: lss_completion_fiducials stays None, so the whole
    Q_LSS provenance guard early-returns, and the run directory carries no
    diagnostic at all.
    """
    from darksirens.catalogs.lss import maybe_load_lss_completion

    broken = tmp_path / "not_hdf5.h5"
    broken.write_bytes(b"this is not an HDF5 file")
    with pytest.raises(OSError):
        maybe_load_lss_completion(_single_opts(str(broken), None), zgrid=zgrid)


def test_survey_without_the_group_is_still_a_clean_non_lss_run(tmp_path):
    """The one benign case: a readable survey that genuinely carries no
    /lss_completion group loads with no Q and no error."""
    from darksirens.catalogs.lss import maybe_load_lss_completion

    survey = _write_survey(tmp_path / "survey.h5", seed=3)
    out = maybe_load_lss_completion(_single_opts(survey, None), zgrid=zgrid)
    assert out["lss_completion_logq"] is None
    assert out["lss_completion_fiducials"] is None


def test_explicit_q_with_a_non_galaxy_aware_model_raises(tmp_path):
    """--lss_completion with a model that has no galaxy catalog silently did
    nothing, so the run did not target the completeness model it was given."""
    from darksirens.catalogs.lss import maybe_load_lss_completion

    q = _save_q(tmp_path / "q.h5", seed=1, with_members=False)
    with pytest.raises(ValueError, match="no galaxy catalog"):
        maybe_load_lss_completion(
            _single_opts(None, q, universe_model="spectral_sirens"), zgrid=zgrid
        )
