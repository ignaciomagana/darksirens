"""
Joint multi-survey Q_LSS builder
(darksirens/cli/build_joint_lognormal_completion.py).

One shared latent LSS field is inferred from K survey catalogs jointly and K
per-survey Q ensembles are emitted whose member ``m`` is the SAME field
realization, stamped with one shared ``realization_set_id`` -- making a K>=2
``--lss_marginalize`` run legitimate WITHOUT
``--allow_unverified_shared_lss_members``.

Gates (per the design doc):
  a. K=1 parity -- joint([A]) is bit-identical to the single-survey gp3d build.
  b. Shared-field borrowing -- an empty survey borrows another survey's structure.
  c. Matched members -- the per-survey member series share ONE xi draw set.
  d. Loader acceptance -- the K files pass the K>=2 shared-realization provenance
     gate WITHOUT the allow flag; two separate single builds still abort.
  e. End-to-end -- K=2 make_likelihood + --lss_marginalize -> finite.
  f. Heterogeneity + limits -- arity broadcast/error, empty joint, radial reject,
     outs/catalogs mismatch.
  g. CLI smoke via main([...]).
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("healpy")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healpy as hp

from darksirens.redshift import zgrid
from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
from darksirens.cli.build_lognormal_completion import build_completion
from darksirens.cli import build_joint_lognormal_completion as J
from darksirens.cli.build_joint_lognormal_completion import (
    build_joint_completion,
    main as joint_main,
)

NG = int(zgrid.size)
_Z = np.asarray(zgrid)

# The synthetic completeness base tracks N_obs for a self-consistent catalog, so
# the reconstructed field is ~0 at a realistic n0.  A very low expected density
# (huge comoving volume => n0 must be tiny for observed > expected) SATURATES the
# completeness at 1 and produces a clear overdensity to exercise borrowing --
# a legitimate builder knob (--log10n0), used here purely to make the field big.
_OVERDENSE_LOG10N0 = -10.0


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_survey(path, nside, pix_specs, seed=0):
    """A load_survey-schema catalog. ``pix_specs`` maps pixel -> (ngal, z0):
    ``ngal`` galaxies scattered about ``z0`` in that RING pixel."""
    npix = hp.nside2npix(int(nside))
    maxg = max([n for n, _ in pix_specs.values()] + [1])
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for p, (n, z0) in pix_specs.items():
        zgals[p, :n] = z0 + 0.03 * rng.standard_normal(n)
        dzgals[p, :n] = 0.01
        wgals[p, :n] = 1.0
        ngals[p] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return npix


def _farthest_pix(nside, d_vec):
    v = np.asarray(hp.pix2vec(int(nside), np.arange(hp.nside2npix(int(nside))))).T
    return int(np.argmax(((v - d_vec) ** 2).sum(axis=1)))


# ===========================================================================
# a. K=1 parity (hard gate)
# ===========================================================================

def test_k1_parity_bit_identical_to_single(tmp_path):
    """joint([A], seed, defaults, bias=[1.0]) == build_completion(A, 'gp3d', seed)
    bit-for-bit (logq_map, logq_members, zgrid) in the SAME process/backend."""
    cat = str(tmp_path / "A.h5")
    _write_survey(cat, nside=2, pix_specs={0: (3, 0.5), 1: (3, 0.5)})
    kw = dict(seed=7, gp3d_nz_solve=12, gp3d_pix_chunk=8)

    lm_s, le_s, _ = build_completion(cat, mode="gp3d", n_members=5, **kw)
    res = build_joint_completion([cat], [str(tmp_path / "A_out.h5")],
                                 n_members=5, biases=[1.0], **kw)
    lm_j, le_j = res[0]["logq_map"], res[0]["logq_members"]

    assert np.array_equal(lm_s, lm_j), "K=1 logq_map not bit-identical"
    assert np.array_equal(le_s, le_j), "K=1 logq_members not bit-identical"

    d = load_lss_completion_hdf5(str(tmp_path / "A_out.h5"))
    assert np.array_equal(np.asarray(d["logq_map"]), lm_s)
    assert np.array_equal(np.asarray(d["logq_members"]), le_s)
    assert np.array_equal(np.asarray(d["zgrid"]), _Z)
    assert d["indexing"] == "global"
    assert d["diagnostics"]["mode"] == "gp3d_joint"
    assert d["diagnostics"]["joint_n_surveys"] == 1


# ===========================================================================
# b. Shared-field borrowing (the whole point)
# ===========================================================================

def test_shared_field_borrowing_cross_survey(tmp_path):
    """Survey A clustered at direction d; survey B occupied FAR from d and EMPTY
    near d.  Joint Q_B at the pixel containing d clearly exceeds 1 (borrows A's
    field); the single-survey build of B alone reads ~1 there; pixels far from
    all data read ~1 in both.  Cross-resolution: B at a coarser nside than A."""
    d_vec = np.asarray(hp.pix2vec(2, 0), dtype=float)      # direction of A's pixel
    far2 = _farthest_pix(2, d_vec)
    # a pixel far from BOTH A (pix 0) and B (far2)
    v2 = np.asarray(hp.pix2vec(2, np.arange(48))).T
    d2 = np.minimum(((v2 - v2[0]) ** 2).sum(1), ((v2 - v2[far2]) ** 2).sum(1))
    midfar = int(np.argmax(d2))

    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=2, pix_specs={0: (8, 0.5)}, seed=1)
    _write_survey(catB, nside=2, pix_specs={far2: (8, 0.5)}, seed=2)

    res = build_joint_completion(
        [catA, catB], [str(tmp_path / "Ao.h5"), str(tmp_path / "Bo.h5")],
        n_members=0, gp3d_nz_solve=16, gp3d_pix_chunk=8,
        log10n0s=[_OVERDENSE_LOG10N0])   # 1-value broadcast to both
    qA, qB = res[0]["logq_map"], res[1]["logq_map"]
    jz = int(np.argmax(qA[0]))            # A's field-peak redshift bin

    assert qA[0, jz] > 0.1               # A is genuinely overdense at d
    assert qB[0, jz] > 0.1               # B BORROWS A's field at the co-located d
    assert abs(qB[midfar, jz]) < 1e-2    # far from all data -> Q ~ 1

    lmB_single, _, _ = build_completion(
        catB, mode="gp3d", n_members=0, gp3d_nz_solve=16, gp3d_pix_chunk=8,
        log10n0=_OVERDENSE_LOG10N0)
    assert abs(lmB_single[0, jz]) < 1e-3     # B alone: no data near d -> Q ~ 1
    assert abs(lmB_single[midfar, jz]) < 1e-3

    # Cross-resolution: B at nside=1 (coarser than A's nside=2).
    v1 = np.asarray(hp.pix2vec(1, np.arange(12))).T
    pd1 = int(hp.vec2pix(1, *d_vec))                 # nside=1 pixel containing d
    far1 = int(np.argmax(((v1 - d_vec) ** 2).sum(1)))
    catB1 = str(tmp_path / "B1.h5")
    _write_survey(catB1, nside=1, pix_specs={far1: (8, 0.5)}, seed=3)
    res2 = build_joint_completion(
        [catA, catB1], [str(tmp_path / "Ao2.h5"), str(tmp_path / "B1o.h5")],
        n_members=0, gp3d_nz_solve=16, gp3d_pix_chunk=8,
        log10n0s=[_OVERDENSE_LOG10N0])
    qB1 = res2[1]["logq_map"]
    assert qB1.shape == (12, NG)                     # B kept its own nside
    jz2 = int(np.argmax(res2[0]["logq_map"][0]))
    assert qB1[pd1, jz2] > 0.05                      # coarse B borrows fine A


# ===========================================================================
# c. Matched members
# ===========================================================================

def test_matched_members_shared_realization(tmp_path):
    """The two surveys' member series at the co-located structure pixel are
    strongly positively correlated across m (ONE xi draw set);
    joint_member_xi_sha256 / realization_set_id / n_members agree across files."""
    d_vec = np.asarray(hp.pix2vec(2, 0), dtype=float)
    far2 = _farthest_pix(2, d_vec)
    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=2, pix_specs={0: (8, 0.5)}, seed=1)
    _write_survey(catB, nside=2, pix_specs={far2: (8, 0.5)}, seed=2)

    outA, outB = str(tmp_path / "Ao.h5"), str(tmp_path / "Bo.h5")
    res = build_joint_completion(
        [catA, catB], [outA, outB], n_members=8, seed=123,
        gp3d_nz_solve=16, gp3d_pix_chunk=8, log10n0s=[_OVERDENSE_LOG10N0])
    memA, memB = res[0]["logq_members"], res[1]["logq_members"]
    jz = int(np.argmax(res[0]["logq_map"][0]))

    sA = memA[:, 0, jz]      # member series at A's structure pixel
    sB = memB[:, 0, jz]      # member series at the co-located B pixel (borrowed)
    assert sA.std() > 1e-6 and sB.std() > 1e-6      # members genuinely vary
    assert float(np.corrcoef(sA, sB)[0, 1]) > 0.9   # same underlying xi draws

    dA = load_lss_completion_hdf5(outA)
    dB = load_lss_completion_hdf5(outB)
    shaA = dA["diagnostics"]["joint_member_xi_sha256"]
    shaB = dB["diagnostics"]["joint_member_xi_sha256"]
    assert isinstance(shaA, str) and shaA == shaB
    assert dA["realization_set_id"] == dB["realization_set_id"] is not None
    assert dA["n_members"] == dB["n_members"] == 8
    # (member_content_sha256 -- the per-survey Q bytes -- is equal ONLY because A
    # and B share nside and bias here, so the shared field is evaluated at the
    # SAME directions; it differs whenever nside/mask/bias differ. The matched
    # SHARED-FIELD provenance is the xi-level joint_member_xi_sha256 above.)


# ===========================================================================
# d. Loader acceptance (real load_multitracer_catalog_bundles path)
# ===========================================================================

def _gw_inputs(n_pe=2, n_sel=8):
    rng = np.random.default_rng(0)
    return {
        "ra": rng.uniform(0.0, 2 * np.pi, size=n_pe),
        "dec": rng.uniform(-np.pi / 2, np.pi / 2, size=n_pe),
        "rasels": rng.uniform(0.0, 2 * np.pi, size=n_sel),
        "decsels": rng.uniform(-np.pi / 2, np.pi / 2, size=n_sel),
    }


def _loader_opts(survey_paths, lss_completions, **overrides):
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


def _two_joint_surveys(tmp_path, n_members=4):
    """Two nside=1 surveys + their JOINTLY built (shared-realization) Q files."""
    s1 = str(tmp_path / "s1.h5")
    s2 = str(tmp_path / "s2.h5")
    _write_survey(s1, nside=1, pix_specs={0: (3, 0.3), 4: (2, 0.4)}, seed=11)
    _write_survey(s2, nside=1, pix_specs={2: (3, 0.35), 8: (2, 0.5)}, seed=22)
    q1 = str(tmp_path / "q1.h5")
    q2 = str(tmp_path / "q2.h5")
    build_joint_completion([s1, s2], [q1, q2], n_members=n_members, seed=5,
                           gp3d_nz_solve=10, gp3d_pix_chunk=8)
    return s1, s2, q1, q2


def test_loader_accepts_joint_files_without_allow_flag(tmp_path):
    from darksirens.inference.loaders import load_multitracer_catalog_bundles
    s1, s2, q1, q2 = _two_joint_surveys(tmp_path)
    opts = _loader_opts([s1, s2], [q1, q2])   # allow flag ABSENT (False)
    bundles = load_multitracer_catalog_bundles(opts, _gw_inputs())
    assert len(bundles) == 2
    ids = {b["lss_completion_provenance"]["realization_set_id"] for b in bundles}
    assert len(ids) == 1 and next(iter(ids)) is not None


def test_loader_control_separate_single_builds_still_raise(tmp_path):
    """Control: independently built (single-survey) ensembles carry DISTINCT
    realization ids, so the K>=2 shared-member gate must still abort."""
    from darksirens.inference.loaders import load_multitracer_catalog_bundles
    s1 = str(tmp_path / "s1.h5")
    s2 = str(tmp_path / "s2.h5")
    _write_survey(s1, nside=1, pix_specs={0: (3, 0.3)}, seed=11)
    _write_survey(s2, nside=1, pix_specs={2: (3, 0.35)}, seed=22)
    q1, q2 = str(tmp_path / "q1.h5"), str(tmp_path / "q2.h5")
    for cat, out in ((s1, q1), (s2, q2)):
        lm, le, dg = build_completion(cat, mode="gp3d", n_members=4, seed=1,
                                      gp3d_nz_solve=10, gp3d_pix_chunk=8)
        from darksirens.redshift.lognormal_completion import save_lss_completion_hdf5
        save_lss_completion_hdf5(out, logq_map=lm, logq_members=le,
                                 zgrid=_Z, indexing="global", metadata=dg)
    opts = _loader_opts([s1, s2], [q1, q2])
    with pytest.raises(ValueError, match="SHARED member index"):
        load_multitracer_catalog_bundles(opts, _gw_inputs())


# ===========================================================================
# e. End-to-end: K=2 make_likelihood + --lss_marginalize -> finite
# ===========================================================================

def _pop_bits():
    from darksirens.gw.populations import pop_model_prior_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels)
             if lbl != sampled}
    return pop_lower, pop_upper, pop_fid, sampled, fixed


def _shared_physics(nsamp=2, n_sel=8):
    return dict(
        nEvents=1, nsamp=nsamp, Ndraw=float(n_sel),
        m1det=jnp.array([36.0, 38.0]), m2det=jnp.array([28.8, 30.4]),
        dL=jnp.array([460.0, 500.0]), chieff=jnp.array([0.0, 0.02]),
        p_pe=jnp.ones(nsamp),
        m1detsels=jnp.linspace(34.0, 40.0, n_sel),
        m2detsels=0.8 * jnp.linspace(34.0, 40.0, n_sel),
        dLsels=jnp.linspace(430.0, 530.0, n_sel), chieffsels=jnp.zeros(n_sel),
        p_draw=jnp.ones(n_sel),
    )


def test_e2e_k2_make_likelihood_lss_marginalize_finite(tmp_path):
    """The full circle: two JOINTLY built Q files -> load_multitracer bundles
    (provenance verified, allow flag ABSENT) -> make_likelihood with
    --lss_marginalize -> a finite value."""
    from darksirens.inference.loaders import load_multitracer_catalog_bundles
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space

    s1, s2, q1, q2 = _two_joint_surveys(tmp_path, n_members=4)
    pop_lower, pop_upper, _pop_fid, sampled, fixed = _pop_bits()

    loader_opts = _loader_opts([s1, s2], [q1, q2])
    bundles = load_multitracer_catalog_bundles(loader_opts, _gw_inputs())

    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, fix_de=False,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False, n_catalogs=2,
        lss_marginalize=True, catalog_sky_weighting="conditional",
        redshift_prior_barrier="off",
    )
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = bundles

    pop_fid = get_fixed_population_params("powerlaw+peak")
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

    labels, lower, upper = build_parameter_space(
        "powerlaw+peak", False, True, True,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, universe_model="dark_sirens", n_catalogs=2,
    )[:3]
    coord = jnp.asarray([0.5 * (float(lo) + float(hi))
                         for lo, hi in zip(lower, upper)])
    val = float(ll(coord))
    assert np.isfinite(val)


# ===========================================================================
# f. Heterogeneity + limits
# ===========================================================================

def test_arity_broadcast_and_error(tmp_path):
    """1-or-K arities broadcast; anything else is a clear error."""
    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=1, pix_specs={0: (3, 0.3)}, seed=1)
    _write_survey(catB, nside=1, pix_specs={2: (3, 0.35)}, seed=2)
    outs = [str(tmp_path / "Ao.h5"), str(tmp_path / "Bo.h5")]

    # per-survey biases/n0/delta of length K
    res = build_joint_completion(
        [catA, catB], outs, n_members=0, gp3d_nz_solve=8, gp3d_pix_chunk=8,
        biases=[1.0, 2.0], log10n0s=[-2.0, -3.0], deltas=[0.0, 0.5])
    assert res[0]["diagnostics"]["bias_b_miss"] == 1.0
    assert res[1]["diagnostics"]["bias_b_miss"] == 2.0
    assert res[0]["diagnostics"]["fiducial_delta"] == 0.0
    assert res[1]["diagnostics"]["fiducial_delta"] == 0.5
    assert res[0]["diagnostics"]["joint_biases"] == [1.0, 2.0]

    with pytest.raises(ValueError, match=r"--bias takes 1 or K=2"):
        build_joint_completion([catA, catB], outs, n_members=0,
                               gp3d_nz_solve=8, biases=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"--outs .* must have the same length"):
        build_joint_completion([catA, catB], outs[:1], n_members=0,
                               gp3d_nz_solve=8)


def test_empty_joint_writes_unity_with_shared_id(tmp_path):
    """No occupied pixels in ANY catalog -> Q = 1 everywhere in every file, all
    sharing one realization_set_id (mirrors the single-survey empty shortcut)."""
    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=1, pix_specs={}, seed=1)   # no galaxies
    _write_survey(catB, nside=1, pix_specs={}, seed=2)
    outA, outB = str(tmp_path / "Ao.h5"), str(tmp_path / "Bo.h5")
    res = build_joint_completion([catA, catB], [outA, outB], n_members=3,
                                 gp3d_nz_solve=8, gp3d_pix_chunk=8)
    for r in res:
        assert float(np.max(np.abs(r["logq_map"]))) == 0.0
        assert float(np.max(np.abs(r["logq_members"]))) == 0.0
        assert r["n_occupied"] == 0
    dA = load_lss_completion_hdf5(outA)
    dB = load_lss_completion_hdf5(outB)
    assert dA["realization_set_id"] == dB["realization_set_id"] is not None
    assert float(np.max(np.abs(dA["logq_map"]))) == 0.0
    assert dA["diagnostics"]["mode"] == "gp3d_joint"


def test_radial_mode_rejected(tmp_path):
    """--mode radial is rejected with an explanatory message (no shared field)."""
    catA = str(tmp_path / "A.h5")
    _write_survey(catA, nside=1, pix_specs={0: (3, 0.3)}, seed=1)
    with pytest.raises(SystemExit):
        joint_main(["--catalogs", catA, "--outs", str(tmp_path / "Ao.h5"),
                    "--mode", "radial", "--n-members", "0"])


def test_outs_catalogs_mismatch_cli_error(tmp_path):
    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=1, pix_specs={0: (3, 0.3)}, seed=1)
    _write_survey(catB, nside=1, pix_specs={2: (3, 0.3)}, seed=2)
    with pytest.raises(SystemExit):
        joint_main(["--catalogs", catA, catB, "--outs", str(tmp_path / "Ao.h5"),
                    "--n-members", "0"])


# ===========================================================================
# g. CLI smoke via main([...])
# ===========================================================================

def test_cli_main_smoke_writes_matched_files(tmp_path):
    catA = str(tmp_path / "A.h5")
    catB = str(tmp_path / "B.h5")
    _write_survey(catA, nside=1, pix_specs={0: (3, 0.3), 4: (2, 0.4)}, seed=1)
    _write_survey(catB, nside=1, pix_specs={2: (3, 0.35)}, seed=2)
    outA, outB = str(tmp_path / "qA.h5"), str(tmp_path / "qB.h5")
    rid = "0123456789abcdef0123456789abcdef"
    joint_main([
        "--catalogs", catA, catB, "--outs", outA, outB,
        "--n-members", "3", "--seed", "9", "--gp3d-nz-solve", "10",
        "--gp3d-pix-chunk", "8", "--bias", "1.0", "--realization-set-id", rid,
    ])
    dA = load_lss_completion_hdf5(outA)
    dB = load_lss_completion_hdf5(outB)
    assert dA["logq_map"].shape == (12, NG)
    assert dB["logq_map"].shape == (12, NG)
    assert dA["realization_set_id"] == rid == dB["realization_set_id"]
    assert dA["n_members"] == dB["n_members"] == 3
    assert dA["diagnostics"]["joint_survey_index"] == 0
    assert dB["diagnostics"]["joint_survey_index"] == 1
    assert dA["diagnostics"]["joint_catalogs"] == [catA, catB]
    assert (dA["diagnostics"]["joint_member_xi_sha256"]
            == dB["diagnostics"]["joint_member_xi_sha256"])
