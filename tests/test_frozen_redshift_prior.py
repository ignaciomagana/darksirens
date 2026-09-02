"""Build-time frozen redshift prior for population-only runs (perf PR-3).

When no cosmology, survey or mark label is sampled, the per-sample
``log p(z | pix)`` of every PE sample and injection is a run constant; the
factory evaluates it once and the likelihood body skips the per-proposal
kernel state, completion curves and windowed catalog KDE.  Contracts pinned
here: the gate reads labels only; the frozen likelihood reproduces the
per-proposal one to floating-point re-association; the in-graph probe turns a
violated premise into -inf; the operand is aligned with the pixel-sorted,
padded injections.
"""
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import darksirens.likelihood.factory as F
from darksirens.core.types import CosmoParams, SurveyParams
from darksirens.likelihood.core import (
    FrozenRedshiftPrior,
    frozen_prior_probe_vector,
)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _dec(sampled, pop=("alpha", "m_min", "gamma"), sky=("d_x",)):
    return SimpleNamespace(sampled_labels=tuple(sampled), pop_labels=pop, sky_labels=sky)


@pytest.mark.parametrize("model", ["dark_sirens", "dark_sirens_complete"])
def test_gate_admits_population_sky_and_stick_labels(model):
    assert F.frozen_prior_admissible(_dec(("alpha", "gamma")), model, "none", False)
    assert F.frozen_prior_admissible(_dec(("alpha", "d_x", "fcat_2")), model, None, False)
    assert F.frozen_prior_admissible(_dec(()), model, "none", False)


@pytest.mark.parametrize("blocked", [
    "H0", "Om0", "w0", "wa", "log10n0", "delta", "b_miss", "sigma_kde",
    "M0hat", "sigma_M", "log10n0_c2", "sigma_kde_c3", "eta_logmstar",
])
def test_gate_refuses_every_cosmology_survey_and_mark_label(blocked):
    assert not F.frozen_prior_admissible(
        _dec(("alpha", blocked)), "dark_sirens", "none", False)


def test_gate_refuses_marks_members_and_non_catalog_models():
    assert not F.frozen_prior_admissible(_dec(("alpha",)), "dark_sirens", "logmstar", False)
    assert not F.frozen_prior_admissible(_dec(("alpha",)), "dark_sirens", "none", True)
    for model in ("spectral_sirens", "bright_sirens", "spectral_sirens_wl"):
        assert not F.frozen_prior_admissible(_dec(("alpha",)), model, "none", False)


def test_probe_vector_reads_the_fixed_scalars_only():
    cosmo = CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    s = SurveyParams(n0=1e-2, z50=0.5, w=0.1, delta=0.0, b_miss=1.0, alpha_miss=0.5,
                     sigma_kde=0.01)
    v = frozen_prior_probe_vector(cosmo, (s,))
    assert v.shape == (4 + 10,)
    assert float(v[0]) == 67.74 and float(v[4]) == 1e-2
    v2 = frozen_prior_probe_vector(cosmo, (s._replace(sigma_kde=0.02),))
    assert not bool(jnp.all(v == v2))


# ---------------------------------------------------------------------------
# End to end through the factory
# ---------------------------------------------------------------------------

def _fixture(seed=11, n_sel=96, n_gal=10, nsamp=6):
    """Flat single-catalog nside=1 run with the cosmology and survey FIXED and
    the whole powerlaw+peak population sampled: the frozen premise."""
    import healpy as hp

    from darksirens.gw.populations import pop_model_prior_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.redshift import zgrid

    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(1)
    zg = np.sort(rng.uniform(0.02, 0.5, (npix, n_gal)), axis=1)
    dz = rng.uniform(0.003, 0.02, (npix, n_gal))
    wg = np.ones((npix, n_gal))
    ng = np.full(npix, n_gal, dtype=np.int32)
    ng[5] = 0                                   # one empty pixel
    pixels_pe = rng.integers(0, npix, nsamp).astype(np.int32)
    pixels_sel = rng.integers(0, npix, n_sel).astype(np.int32)

    def _dirs(n, phase):
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
        return (jnp.asarray(np.cos(ang) * np.sqrt(0.75)),
                jnp.asarray(np.sin(ang) * np.sqrt(0.75)), jnp.full(n, 0.5))

    nx_pe, ny_pe, nz_pe = _dirs(nsamp, 0.1)
    nx_sel, ny_sel, nz_sel = _dirs(n_sel, 0.7)
    data = {
        "nEvents": 1, "nsamp": nsamp, "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(1), "nside": 1, "n_pix_catalog": npix,
        "zgals": zg, "dzgals": dz, "wgals": wg, "ngals_catalog": ng,
        "zgals_catalog": zg, "dzgals_catalog": dz, "wgals_catalog": wg,
        "delta_g_pix_z": jnp.zeros((npix, len(zgrid))),
        "m1det": jnp.asarray(rng.uniform(30.0, 45.0, nsamp)),
        "m2det": jnp.asarray(rng.uniform(20.0, 28.0, nsamp)),
        "dL": jnp.asarray(rng.uniform(400.0, 2000.0, nsamp)),
        "chieff": jnp.asarray(rng.uniform(-0.05, 0.05, nsamp)),
        "p_pe": jnp.ones(nsamp), "pixels_pe": jnp.asarray(pixels_pe),
        "nx_pe": nx_pe, "ny_pe": ny_pe, "nz_pe": nz_pe,
        "m1detsels": jnp.asarray(rng.uniform(30.0, 45.0, n_sel)),
        "m2detsels": jnp.asarray(rng.uniform(20.0, 28.0, n_sel)),
        "dLsels": jnp.asarray(rng.uniform(300.0, 2500.0, n_sel)),
        "chieffsels": jnp.zeros(n_sel), "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.asarray(pixels_sel),
        "nx_sel": nx_sel, "ny_sel": ny_sel, "nz_sel": nz_sel,
    }
    _, _, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = np.asarray(get_fixed_population_params("powerlaw+peak"))
    opts = SimpleNamespace(
        universe_model="dark_sirens", pop_model="powerlaw+peak", sampler="dynesty",
        fix_cosmology=True, fix_de=False, fix_population=False, fix_survey=True,
        fixed_cosmology=True, fixed_de=False,
        prior_overrides=None, fixed_parameter_values=None,
        use_LSS=False, counterpart=None, lss_completions=None,
        sel_batch_size=None, pe_event_block=None, sky_model="isotropic",
        mark_model="none", mark_names=(), catalog_sky_weighting="conditional",
        n_catalogs=1, kde_window=None, drop_full_catalog=False,
        freeze_redshift_prior=True,
    )
    return opts, data, jnp.asarray(pop_fid), pop_labels


def _coords(pop_fid, n=3, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = np.asarray(pop_fid, dtype=float).copy()
        c = c * (1.0 + 0.03 * rng.standard_normal(c.shape))
        out.append(jnp.asarray(c))
    return out


def test_frozen_likelihood_matches_per_proposal_evaluation():
    opts, data, pop_fid, pop_labels = _fixture()
    frozen_like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=None)
    assert frozen_like.frozen_redshift_prior is True
    fp = frozen_like.operands[-1]
    assert isinstance(fp, FrozenRedshiftPrior)
    gw_sel = frozen_like.operands[2]
    assert fp.log_prior_pe.shape == (int(data["nEvents"]) * int(data["nsamp"]),)
    assert fp.log_prior_sel.shape == (int(gw_sel.dL.shape[0]),)
    # Empty pixel -> -inf catalog prior only through the missing branch: the
    # dark_sirens prior stays finite there (dN_miss owns it); no NaN anywhere.
    assert not np.any(np.isnan(np.asarray(fp.log_prior_pe)))
    assert not np.any(np.isnan(np.asarray(fp.log_prior_sel)))

    opts.freeze_redshift_prior = False
    live_like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=None)
    assert live_like.frozen_redshift_prior is False
    for c in _coords(pop_fid):
        a = float(frozen_like(c))
        b = float(live_like(c))
        assert np.isfinite(a) and np.isfinite(b), (a, b)
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (a, b)


def test_frozen_operand_is_aligned_with_pixel_sorted_padded_injections():
    opts, data, pop_fid, _ = _fixture()
    opts.sel_batch_size = 32          # 96 injections -> 3 batches, no padding
    like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=None)
    fp = like.operands[-1]
    gw_sel = like.operands[2]
    p = np.asarray(gw_sel.pixels)
    assert np.all(p[1:] >= p[:-1])
    assert fp.log_prior_sel.shape[0] == gw_sel.dL.shape[0] == 96
    opts.sel_batch_size = 40          # -> padded to 120
    like2 = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=None)
    fp2 = like2.operands[-1]
    assert fp2.log_prior_sel.shape[0] == like2.operands[2].dL.shape[0] == 120
    # The batched (padded) and single-pass values agree to re-association.
    for c in _coords(pop_fid, n=2):
        assert abs(float(like(c)) - float(like2(c))) <= 1e-12 * max(1.0, abs(float(like(c))))


def test_gate_is_off_when_a_cosmology_label_is_sampled():
    opts, data, pop_fid, _ = _fixture()
    opts.fix_cosmology = False
    opts.fixed_cosmology = False
    opts.fix_de = True
    opts.fixed_de = True
    opts.fixed_parameter_values = {"Om0": 0.3075}
    like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values={"Om0": 0.3075})
    assert like.frozen_redshift_prior is False
    assert like.operands[-1] is None


def test_probe_poisons_a_violated_premise():
    """A frozen prior built at one survey block, served a proposal whose fixed
    scalars differ (the sampled-label premise broken behind the graph's back),
    must return -inf rather than a plausible number."""
    opts, data, pop_fid, _ = _fixture()
    like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=None)
    fp = like.operands[-1]
    c = _coords(pop_fid, n=1)[0]
    ok = float(like(c))
    assert np.isfinite(ok)
    # Same arrays, a probe vector that says "built at a different sigma_kde".
    bad = fp._replace(probe_ref=fp.probe_ref.at[10].add(1e-3))
    operands = (*like.operands[:-1], bad)
    poisoned = float(like.jitted_body(c, operands, like.distance_table,
                                      like.smoothing_operator))
    assert poisoned == -np.inf
