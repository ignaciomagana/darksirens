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
            gen._sfilter_low(m, m_min, dm), np.asarray(sfilter_low(m, m_min, dm)), atol=1e-6
        )
    for m_max, dm in [(85.0, 10.0), (50.0, 5.0), (100.0, 2.0)]:
        np.testing.assert_allclose(
            gen._sfilter_high(m, m_max, dm), np.asarray(sfilter_high(m, m_max, dm)), atol=1e-6
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
