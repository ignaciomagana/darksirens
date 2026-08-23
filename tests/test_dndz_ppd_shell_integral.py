"""The dN/dz PPD shells must be integrated over the shells (standing debt).

``scripts/dndz_ppd_check.py`` predicted each shell's galaxy count with

    np.trapz(np.where((zf >= lo) & (zf < hi), dens, 0.0), zf)

over the whole fine grid.  Masking and then trapezoidal-integrating across the
boundary cells is algebraically a RECTANGLE sum over the selected nodes, so the
effective width is ``(#nodes inside) * dz`` rather than ``hi - lo``.  The fine
grid is not aligned to the shell edges, so that node count wobbles (50, 50, ...,
49 on the default 600-node / 12-shell setup) and the predicted counts wobble
with it — measured up to 1.89% per shell on a synthetic integrand.

That matters because the script's whole output is per-shell Poisson z-scores:
a 2% shift on a shell holding ~4000 galaxies is ~80 counts against sigma ~63,
a z-score of order one manufactured by the quadrature and then read as
clustering.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# numpy 1 / numpy 2 both — fast_subset.txt forbids a bare np.trapz/np.trapezoid.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dndz_ppd_check.py"


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT.is_file():  # pragma: no cover
        pytest.skip("scripts/dndz_ppd_check.py not in this checkout")
    spec = importlib.util.spec_from_file_location("_dndz_ppd_check", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # top-level imports only numpy/argparse/json
    return m


# The script's own default geometry: --z-depth 0.3, --n-shells 12, 600 nodes.
Z_DEPTH, N_SHELLS, N_FINE = 0.3, 12, 600


def _grid():
    zf = np.linspace(1e-4, Z_DEPTH, N_FINE)
    edges = np.linspace(0.0, Z_DEPTH, N_SHELLS + 1)
    # Same character as the real integrand C(z) n0 (1+z)^delta dV/dz: smooth,
    # positive, steeply rising.
    dens = (1.0 + zf) ** 3 * zf ** 2 * 1e6
    return zf, edges, dens


def _reference(zf, dens, lo, hi, n=200_001):
    """Converged trapezoid on a dense sub-grid spanning exactly [lo, hi]."""
    g = np.linspace(max(lo, zf[0]), hi, n)
    return float(_trapezoid(np.interp(g, zf, dens), g))


def test_every_shell_matches_a_converged_reference(mod):
    zf, edges, dens = _grid()
    errs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        got = mod.shell_integral(zf, dens, lo, hi)
        ref = _reference(zf, dens, lo, hi)
        errs.append(abs(got / ref - 1.0))
    worst = max(errs)
    assert worst < 1e-5, (
        f"worst per-shell relative error {worst:.3%}; the masked-trapz form "
        "this replaces reached 1.89% on this same case."
    )


def test_it_beats_the_masked_trapz_form_it_replaces(mod):
    """Pin the improvement itself, not just an absolute tolerance."""
    zf, edges, dens = _grid()

    def masked_trapz(lo, hi):
        return float(_trapezoid(np.where((zf >= lo) & (zf < hi), dens, 0.0), zf))

    old_worst = max(abs(masked_trapz(lo, hi) / _reference(zf, dens, lo, hi) - 1.0)
                    for lo, hi in zip(edges[:-1], edges[1:]))
    new_worst = max(abs(mod.shell_integral(zf, dens, lo, hi) / _reference(zf, dens, lo, hi) - 1.0)
                    for lo, hi in zip(edges[:-1], edges[1:]))
    # The defect really is at the ~2% level on the script's own geometry ...
    assert old_worst > 0.01, f"old form only off by {old_worst:.3%}"
    # ... and the fix is orders of magnitude better, not marginally.
    assert new_worst < old_worst / 100.0


def test_shells_tile_the_interval_without_gap_or_overlap(mod):
    """Sum of the shells == one integral over the whole range.

    The masked form cannot satisfy this: its shells are node-counted, so they
    over- or under-cover the boundary cells and the pieces do not add up.
    """
    zf, edges, dens = _grid()
    parts = sum(mod.shell_integral(zf, dens, lo, hi)
                for lo, hi in zip(edges[:-1], edges[1:]))
    whole = mod.shell_integral(zf, dens, edges[0], edges[-1])
    assert parts == pytest.approx(whole, rel=1e-12)


def test_exact_on_a_linear_integrand_for_any_edges(mod):
    """Trapezoid is exact for a linear function; the edge splice must not
    break that, including for edges that fall between grid nodes."""
    zf = np.linspace(0.0, 1.0, 37)
    dens = 3.0 + 5.0 * zf
    for lo, hi in [(0.0, 1.0), (0.013, 0.4), (0.4, 0.9137), (0.5, 0.5 + 1e-3)]:
        got = mod.shell_integral(zf, dens, lo, hi)
        exact = 3.0 * (hi - lo) + 2.5 * (hi ** 2 - lo ** 2)
        assert got == pytest.approx(exact, rel=1e-12, abs=1e-14), (lo, hi)


def test_a_shell_narrower_than_one_grid_cell_is_still_integrated(mod):
    """The masked form returns exactly 0.0 when no node lands inside — a whole
    shell silently predicted empty."""
    zf = np.linspace(0.0, 1.0, 11)          # dz = 0.1
    dens = np.ones_like(zf) * 7.0
    lo, hi = 0.41, 0.49                     # no node inside
    assert _trapezoid(np.where((zf >= lo) & (zf < hi), dens, 0.0), zf) == 0.0
    assert mod.shell_integral(zf, dens, lo, hi) == pytest.approx(7.0 * (hi - lo))
