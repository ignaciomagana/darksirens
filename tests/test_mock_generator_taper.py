"""Consistency checks for the tapered ``powerlaw+peak`` mock generator.

The mock generator's primary-mass / mass-ratio densities are meant to match the
inference ``powerlaw+peak`` model exactly (logistic inner-edge tapers via
``sfilter_low``/``sfilter_high``, a Gaussian peak), so the fitted model contains
the injected truth and there is no hard-edge-vs-tapered mismatch.  These tests
verify (a) the numpy tapers equal the jax originals, (b) the component densities
are normalised, and (c) the samplers draw from exactly the densities used to
build the stored ``pdraw``.

The generator imports ``healpy``/``jax`` at module load, so the whole file skips
where those are unavailable.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("jax")

# darksirens runs JAX in double precision (GW distances/redshifts need it); enable
# x64 so the jax sfilter is computed in float64 and matches the numpy mirror.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "scripts" / "mock_dark_sirens" / "generate_mock_data.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("mock_dark_gen_under_test", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the frozen-annotations dataclass can resolve its
    # module (mirrors scripts/mock_bright_sirens/generate_mock_bright_sirens.py).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


def test_numpy_tapers_match_inference_sfilters(gen):
    """The numpy ``_sfilter_low/high`` mirror the jax inference originals."""
    try:
        from darksirens.gw.populations.utils import sfilter_high, sfilter_low
    except Exception:  # pragma: no cover - package not importable
        pytest.skip("darksirens.gw.populations.utils not importable")

    m = np.linspace(0.0, 120.0, 700)
    for m_min, dm in [(5.0, 3.0), (2.0, 1.0), (10.0, 0.5)]:
        np.testing.assert_allclose(
            gen._sfilter_low(m, m_min, dm), np.asarray(sfilter_low(m, m_min, dm)), atol=1e-9
        )
    for m_max, dm in [(85.0, 10.0), (50.0, 5.0), (100.0, 2.0)]:
        np.testing.assert_allclose(
            gen._sfilter_high(m, m_max, dm), np.asarray(sfilter_high(m, m_max, dm)), atol=1e-9
        )


def test_component_pdfs_normalised(gen):
    """Power-law, peak, and q|m1 densities integrate to one."""
    pop = gen.PopulationConfig()
    grid = gen._MASS_NORM_GRID
    pl = gen._powerlaw_pdf(grid, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max)
    pk = gen._peak_pdf(grid, pop.peak_mu, pop.peak_sigma)
    assert np.isclose(gen._trapz(pl, grid), 1.0, atol=1e-3)
    assert np.isclose(gen._trapz(pk, grid), 1.0, atol=1e-3)

    qg = np.linspace(1e-4, 1.0, 4000)
    for m1 in (20.0, 40.0, 70.0):
        pq = gen._q_pdf(qg, np.full_like(qg, m1), pop)
        assert np.isclose(gen._trapz(pq, qg), 1.0, atol=2e-2)


def test_primary_mass_samples_follow_pdf(gen):
    """Drawn primary masses match the analytic marginal and stay in support."""
    pop = gen.PopulationConfig()
    rng = np.random.default_rng(0)
    m1 = gen._sample_powerlaw_peak_m1(rng, 200_000, pop)

    assert m1.min() >= gen._MASS_NORM_GRID[0]
    assert m1.max() <= gen._MASS_NORM_GRID[-1]
    # Tapered support is essentially [mmin, mmax]; the untruncated peak leaks a
    # negligible amount past the edges.
    assert np.mean((m1 < pop.mmin) | (m1 > pop.mmax)) < 1e-3

    edges = np.linspace(pop.mmin - 1.0, pop.mmax + 1.0, 70)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens, _ = np.histogram(m1, bins=edges, density=True)
    pl = gen._powerlaw_pdf(centers, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max)
    pk = gen._peak_pdf(centers, pop.peak_mu, pop.peak_sigma)
    pdf = (1.0 - pop.peak_fraction) * pl + pop.peak_fraction * pk

    mask = pdf > 0.05 * pdf.max()
    rel = np.abs(dens[mask] - pdf[mask]) / pdf[mask]
    assert np.median(rel) < 0.10
    assert np.max(rel) < 0.40


def test_mass_ratio_samples_follow_pdf(gen):
    """Drawn mass ratios match the tapered q|m1 density at fixed m1."""
    pop = gen.PopulationConfig()
    rng = np.random.default_rng(1)
    m1 = np.full(200_000, 45.0)
    q = gen._sample_q(rng, m1, pop)

    assert q.min() > 0.0 and q.max() <= 1.0
    edges = np.linspace(pop.mmin / 45.0 - 0.02, 1.0, 60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens, _ = np.histogram(q, bins=edges, density=True)
    pdf = gen._q_pdf(centers, np.full_like(centers, 45.0), pop)

    mask = pdf > 0.05 * pdf.max()
    rel = np.abs(dens[mask] - pdf[mask]) / pdf[mask]
    assert np.median(rel) < 0.10


def test_mass_spin_pdf_positive_and_finite(gen):
    """``_mass_spin_pdf`` (the source of stored pdraw) is finite and positive on draws."""
    pop = gen.PopulationConfig()
    rng = np.random.default_rng(2)
    m1 = gen._sample_powerlaw_peak_m1(rng, 5000, pop)
    q = gen._sample_q(rng, m1, pop)
    chi = gen._sample_chieff(rng, 5000, pop)
    p = gen._mass_spin_pdf(m1, q, chi, pop)
    assert np.all(np.isfinite(p))
    assert np.all(p > 0.0)


def _selection_is_neff(gen, b, grids, th):
    """Effective sample size of the selection integral mu(th) = <p_pop/pdraw>.

    Reconstructs the importance weight ``p_target(th)/pdraw`` over the *detected*
    injections in the canonical (m1det, q, chi, dL, sky) basis the generator draws
    in -- the same estimator the inference uses.  The q / chi / sky factors of the
    proposal are flat, so they cancel against the target except for the mass-spin
    density and the (1+z) detector-frame Jacobian, plus the redshift ratio
    ``p_z(z|th) / p_z(z|proposal)`` (proposal z ~ dVc/dz).
    """
    z = np.interp(np.asarray(b["dL"]), grids["dl"], grids["z"])
    m1src = np.asarray(b["m1src"])
    q = np.asarray(b["m2src"]) / m1src
    chi = np.asarray(b["chieff"])

    # mass-spin target (source frame) with the m1det->m1src Jacobian, over the flat
    # (uniform m1det in [2,200]) x (uniform q) x (uniform chi) proposal mass factor.
    p_ms = np.asarray(gen._mass_spin_pdf(m1src, q, chi, th)) / (1.0 + z)
    prop_mass = (1.0 / (200.0 - 2.0)) * 1.0 * 0.5

    dvc, zg = grids["dvc_dz"], grids["z"]
    pz_t = (1.0 + zg) ** (th.gamma - 1.0) * dvc
    pz_t = np.interp(z, zg, pz_t / np.trapezoid(pz_t, zg))
    pz_p = np.interp(z, zg, dvc / np.trapezoid(dvc, zg))

    w = (p_ms / prop_mass) * (pz_t / pz_p)
    w = w[np.isfinite(w) & (w > 0.0)]
    return float(w.sum() ** 2 / np.sum(w ** 2)), w.size


def test_selection_proposal_is_population_independent(gen):
    """The exact regression guard for the spectral-siren bug (gamma railing).

    The bug was that ``pdraw`` was the *population* density at the truth
    (``_mass_spin_pdf(..., pop)``), so the selection proposal was a narrow envelope
    tied to theta.  The importance-sampling estimate of mu(theta) was then only
    well-conditioned near the truth and the recovered population (esp. gamma, via
    the mass-z degeneracy) was biased.  The fix draws from a broad, *population-
    parameter-independent* proposal.  This pins exactly that: with the same seed,
    a wildly different population must give byte-identical proposal draws and pdraw.
    """
    import dataclasses

    grids = gen._cosmology_grids(gen._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=0.3)
    pop = gen.PopulationConfig()
    steep = dataclasses.replace(pop, alpha=5.0, beta=3.0, mmin=8.0, gamma=6.0, peak_fraction=0.4)

    b_truth = gen._draw_selection_batch(np.random.default_rng(0), 40000, grids, pop, snr_threshold=8.0, proposal="uniform")
    b_steep = gen._draw_selection_batch(np.random.default_rng(0), 40000, grids, steep, snr_threshold=8.0, proposal="uniform")

    # Population-independent proposal => same RNG stream => identical detections + pdraw.
    np.testing.assert_array_equal(b_truth["m1det"], b_steep["m1det"])
    np.testing.assert_array_equal(b_truth["chieff"], b_steep["chieff"])
    np.testing.assert_array_equal(np.asarray(b_truth["pdraw"]), np.asarray(b_steep["pdraw"]))


def test_selection_injections_broad_and_well_conditioned(gen):
    """The proposal is broad and the selection integral stays conditioned over the prior.

    (1) Broad: detected injections reach well past the population peak (~35) and
        mass cap (~85); a population-drawn proposal would not.
    (2) Non-collapsing: the IS effective sample size of mu(theta) at a stress point
        (steep slope, heavy pairing) stays a healthy fraction of its value at the
        truth.  The buggy population proposal gave Neff ~= Ndet at the truth but a
        catastrophic collapse off it; the fix keeps the ratio order-unity.
    """
    import dataclasses

    rng = np.random.default_rng(0)
    grids = gen._cosmology_grids(gen._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=0.3)
    pop = gen.PopulationConfig()
    b = gen._draw_selection_batch(rng, 200_000, grids, pop, snr_threshold=8.0, proposal="uniform")
    m1det = np.asarray(b["m1det"])
    assert m1det.size > 1000 and np.all(np.asarray(b["pdraw"]) > 0.0)

    # (1) Broad proposal.
    assert np.percentile(m1det, 99) > 120.0, "selection proposal looks population-narrow"

    # (2) IS conditioning does not collapse across the prior.
    neff_truth, _ = _selection_is_neff(gen, b, grids, pop)
    neff_stress, _ = _selection_is_neff(gen, b, grids, dataclasses.replace(pop, alpha=5.0, beta=3.0))
    assert neff_truth > 20.0, f"selection Neff at truth is degenerate: {neff_truth:.0f}"
    assert neff_stress > 0.2 * neff_truth, (
        f"selection IS Neff collapsed off the truth: {neff_stress:.0f} vs {neff_truth:.0f} "
        "(signature of a population-coupled proposal)"
    )


def test_population_proposal_is_population_coupled(gen):
    """The opt-in ``population`` proposal draws masses/spins FROM the population, so
    its draws and pdraw DO depend on the population parameters (the inference-matched
    proposal); it is narrow (no draws past the mass cap) and pdraw is the mass-spin
    density.  This is the complement of the ``uniform`` independence guard above."""
    import dataclasses

    grids = gen._cosmology_grids(gen._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=0.3)
    pop = gen.PopulationConfig()
    steep = dataclasses.replace(pop, alpha=5.0, mmax=60.0)

    b_truth = gen._draw_selection_batch(np.random.default_rng(0), 40000, grids, pop, snr_threshold=8.0, proposal="population")
    b_steep = gen._draw_selection_batch(np.random.default_rng(0), 40000, grids, steep, snr_threshold=8.0, proposal="population")

    # Population-coupled proposal => draws and pdraw move with the population.
    assert not np.array_equal(b_truth["m1det"], b_steep["m1det"])
    # Narrow proposal: detections stay near the population support (well under the
    # uniform proposal's [2, 200] m1det cap).
    assert np.percentile(np.asarray(b_truth["m1det"]), 99) < 120.0


# --- Per-component pairing pdraw + defensive population+uniform proposal --------
# The following tests pin the PR-1 fix: the mock's mass-ratio-spin density
# (source of the stored selection ``pdraw``) must pair EACH mass component
# separately, exactly like the inference ``powerlaw+peak`` mixture
# (MixtureModel.component_densities): the power law with its (m_min, dm_min)
# secondary-mass taper, the Gaussian peak with the fallback (M_LO=1, dm=0.01).
# The old single-tapered pairing mis-tapered the peak's secondaries at m_min=5,
# driving the in-likelihood weight p_inference/pdraw to e^31 for injections with
# m2src just above 5 (selection Neff -> 1, logL -> -inf at Ndraw >~ 1e6).


def _inference_model():
    """Build the inference ``powerlaw+peak`` model + fiducial theta, or skip."""
    reg = pytest.importorskip("darksirens.gw.populations.registry")
    model = reg.get_model("powerlaw+peak")
    theta = reg.get_fixed_population_params("powerlaw+peak")
    return model, theta


def test_pairing_matches_inference_component_densities(gen):
    """The generator's ``_mass_spin_pdf`` equals the inference mixture density.

    At z=0 (and gamma=0 fiducial, so the ``(1+z)**(gamma-1)`` factor is 1)
    ``exp(model.log_p_pop(m1, q, 0, chi, theta)) == mixture(m1, q, chi)``, which
    is precisely the per-component pairing the generator must reproduce.  The
    mock normalises its mass and pairing densities on the SAME linspaces and
    trapezoid the inference uses (_MASS_NORM_GRID == get_mass_grid(),
    _Q_NORM_GRID == get_q_grid()), so agreement is at machine precision -- far
    tighter than the 2% the fix requires away from the hard zeros."""
    import jax.numpy as jnp

    model, theta = _inference_model()
    pop = gen.PopulationConfig()
    chi = 0.05
    worst = 0.0
    for m1 in (5.2, 6.0, 10.0, 25.0, 35.0, 45.0, 70.0):
        for q in (0.3, 0.6, 0.9, 0.95, 0.99):
            g = float(np.asarray(gen._mass_spin_pdf(np.array([m1]), np.array([q]), np.array([chi]), pop))[0])
            lp = float(np.asarray(model.log_p_pop(
                jnp.array([float(m1)]), jnp.array([float(q)]), jnp.array([0.0]), jnp.array([chi]), theta))[0])
            inf = float(np.exp(lp))
            if inf <= 1e-300 and g <= 1e-300:   # both in a hard zero: nothing to compare
                continue
            rel = abs(g - inf) / abs(inf)
            worst = max(worst, rel)
    # Grids match the inference exactly, so the residual is pure float64 round-off
    # (measured worst ~2e-15); assert a tight bound well inside the 1e-6 target.
    assert worst < 1e-9, f"per-component pairing disagreement {worst:.3e} exceeds 1e-9"


def test_importance_weight_bounded_population_proposal(gen):
    """The in-likelihood weight p_inference/pdraw has no e^31 tail.

    Draw ~2e5 injections from the ``population`` proposal (threshold 0 keeps every
    draw), build the inference target density in the same canonical
    ``(m1det, q, dL)`` basis as the stored pdraw, and check the log importance
    weight is bounded.  Because the stored pdraw now uses the SAME per-component
    pairing as the inference, log w reduces to ``(gamma-1) log(1+z)`` (a mild,
    z-only spread) instead of the unbounded mass-ratio-tail blow-up of the bug."""
    import jax.numpy as jnp

    model, theta = _inference_model()
    grids = gen._cosmology_grids(gen._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=0.3)
    pop = gen.PopulationConfig()

    b = gen._draw_selection_batch(np.random.default_rng(0), 200_000, grids, pop,
                                  snr_threshold=0.0, proposal="population")
    z = np.interp(np.asarray(b["dL"]), grids["dl"], grids["z"])
    m1src = np.asarray(b["m1src"])
    q = np.asarray(b["m2src"]) / m1src
    chi = np.asarray(b["chieff"])

    pz = np.interp(z, grids["z"], grids["dvc_dz"]) / gen._trapz(grids["dvc_dz"], grids["z"])
    ddldz = np.interp(z, grids["z"], np.gradient(grids["dl"], grids["z"]))
    jac = ddldz * (1.0 + z)
    lp = np.asarray(model.log_p_pop(jnp.asarray(m1src), jnp.asarray(q), jnp.asarray(z), jnp.asarray(chi), theta))
    # Inference canonical target density (matches _selection_pdraw's factorisation).
    p_inf = np.exp(lp) * pz / np.maximum(jac, 1e-300) / (4.0 * np.pi)
    pdraw = np.asarray(b["pdraw"])

    logw = np.log(p_inf) - np.log(pdraw)
    assert np.all(np.isfinite(logw))
    assert np.max(np.abs(logw - np.median(logw))) < 5.0, (
        f"population-proposal importance weight has a heavy tail: "
        f"max|logw-median|={np.max(np.abs(logw - np.median(logw))):.2f}"
    )


def test_defensive_mixture_pdraw_and_neff(gen):
    """The ``population+uniform`` proposal: exact mixture pdraw + non-collapsing Neff.

    (1) pdraw is EXACTLY ``0.9 p_population + 0.1 p_uniform`` for every row,
        reconstructed from the two single-branch ``_selection_pdraw`` formulas.
    (2) The 0.1 uniform floor keeps the flat-weight (1/pdraw) selection Neff a
        healthy fraction of the detections even under a stress population, where
        the pure ``population`` proposal's Neff collapses."""
    import dataclasses

    grids = gen._cosmology_grids(gen._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=0.3)
    pop = gen.PopulationConfig()

    # (1) Exact mixture pdraw.
    b = gen._draw_selection_batch(np.random.default_rng(1), 120_000, grids, pop,
                                  snr_threshold=0.0, proposal="population+uniform")
    z = np.interp(np.asarray(b["dL"]), grids["dl"], grids["z"])
    m1src = np.asarray(b["m1src"])
    q = np.asarray(b["m2src"]) / m1src
    chi = np.asarray(b["chieff"])
    p_mix = np.asarray(gen._selection_pdraw("population+uniform", m1src, q, chi, z, grids, pop))
    p_pop = np.asarray(gen._selection_pdraw("population", m1src, q, chi, z, grids, pop))
    p_unif = np.asarray(gen._selection_pdraw("uniform", m1src, q, chi, z, grids, pop))
    np.testing.assert_allclose(p_mix, 0.9 * p_pop + 0.1 * p_unif, rtol=1e-9)

    # (2) Flat-weight selection Neff under a stress population.
    stress = dataclasses.replace(pop, alpha=5.0, beta=3.0)

    def _flat_neff(proposal):
        bb = gen._draw_selection_batch(np.random.default_rng(2), 200_000, grids, stress,
                                       snr_threshold=8.0, proposal=proposal)
        inv = 1.0 / np.asarray(bb["pdraw"])
        return float(inv.sum() ** 2 / np.square(inv).sum())

    neff_mix = _flat_neff("population+uniform")
    neff_pop = _flat_neff("population")
    assert neff_mix > 300.0, f"defensive-mixture flat-weight Neff is unhealthy: {neff_mix:.1f}"
    assert neff_mix > 10.0 * neff_pop, (
        f"defensive mixture did not rescue Neff under stress: "
        f"mixture={neff_mix:.1f} vs pure-population={neff_pop:.1f}"
    )


def test_sample_q_component_split(gen):
    """``_sample_q`` draws from the per-lane pairing selected by ``use_peak``.

    A forced all-peak mask must sample ``_pair_pdf(.; 1.0, 0.01)`` (secondaries
    down to m2=1); a forced all-power-law mask must sample the tapered
    ``_pair_pdf(.; m_min, dm_min)`` (secondaries floored at m2=m_min)."""
    pop = gen.PopulationConfig()
    m1v = 35.0
    m1 = np.full(300_000, m1v)
    rng = np.random.default_rng(5)

    q_peak = gen._sample_q(rng, m1, pop, use_peak=np.ones(len(m1), dtype=bool))
    q_pl = gen._sample_q(rng, m1, pop, use_peak=np.zeros(len(m1), dtype=bool))

    # Peak lanes reach below the power-law floor 5/m1 (secondaries m2 in [1, 5]);
    # power-law lanes stay at/above it.
    assert q_peak.min() < pop.mmin / m1v
    assert q_pl.min() > 0.9 * (pop.mmin / m1v)

    for qs, m_min, dm in ((q_peak, gen._PAIR_M_LO, gen._PAIR_DM),
                          (q_pl, pop.mmin, pop.dm_min)):
        edges = np.linspace(m_min / m1v - 0.01, 1.0, 60)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens, _ = np.histogram(qs, bins=edges, density=True)
        pdf = gen._pair_pdf(centers, np.full_like(centers, m1v), m_min, dm, pop.beta)
        mask = pdf > 0.05 * pdf.max()
        rel = np.abs(dens[mask] - pdf[mask]) / pdf[mask]
        assert np.median(rel) < 0.1, f"sampled q (m_min={m_min}) off its pair pdf: median rel {np.median(rel):.3f}"
