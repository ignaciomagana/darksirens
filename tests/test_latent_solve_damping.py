"""S-1 pins: ``count_map_solve``'s Armijo-damped Fisher/Newton iteration.

The defect (``experiments/field_level_plan/pr6a/CLOSURE.md`` S-1): the solve
used to take a fixed trip count of UNDAMPED Fisher steps.  For the shell
multinomial the link is canonical, so the Fisher information equals the
observed Hessian (PLAN §3.4) and the iteration is EXACT Newton on a convex
objective — locally convergent, and globally convergent only with a line
search.  Measured at nside 16 on 8 realizations drawn from the model's own
prior, **5 of 8 diverged** (P6 pass rate 0.375, ``grad_inf`` up to 4.9e5);
the first failing one leaves ``xi = 0`` with a step of length 41.1 against
``||xi_true|| = 16.9``, ``J`` rises 8.275e6 → 8.506e6, and the iteration
settles into a period-2 limit cycle at ``grad_inf = 4.41e5``.

What is pinned here, in the order it matters:

1. **The converged case does not move.**  This is the hard requirement: the
   production anchor (``latent_anchor_v2a.h5``) converged undamped at
   ``grad_inf = 1.09e-10``, so on every problem where the full step is
   accepted at every trip the damped solve must be BIT-IDENTICAL to the
   pre-fix one — not "agrees to 1e-12", identical.  Two independent pins:
   an explicit transcription of the pre-fix iteration
   (:func:`_undamped_solve`), and ``max_backtrack = 0``, which disables the
   line search and must therefore reproduce the pre-fix path even on a
   problem that diverges.
2. **The diverging problems converge.**  Both the self-contained fixture
   below and, when the closure world's inputs are present, the five actual
   nside-16 realizations.
3. **The backtrack is bounded and the objective descends.**

The fast fixture is a scaled-down copy of the closure world's shape (200
pixels x 8 shells, rank 48, 2e5 galaxies, ``xi_true ~ N(0, I)``) and it
reproduces the failure at ``seed = 6``: undamped ``grad_inf = 9.15e4``.  It
is here so that the pin survives without ``experiments/`` data and runs in
seconds; :func:`test_nside16_closure_realizations` is the real-world
version, and it is the one whose numbers the CLOSURE.md finding quotes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from darksirens.redshift.latent_counts import (
    TracerCounts,
    _damped_newton_step,
    count_map_solve,
    gradient,
    hessian_separable,
    make_count_operator,
    objective,
)
from darksirens.redshift.latent_field import build_latent_basis, shell_response

REPO = Path(__file__).resolve().parents[1]

M_SPH, M_Z, G_S, N_FIT = 16, 3, 8, 200
Z_HI = 0.3
#: The fixture seed whose undamped solve diverges (grad_inf 9.15e4), and a
#: set that converges undamped and must therefore not move at all.
SEED_DIVERGES = 6
SEEDS_CONVERGE = (0, 1, 2, 3, 4, 5)


def _undamped_solve(op, *, n_iter=13, xi0=None):
    """The PRE-FIX iteration, transcribed verbatim from the shipped
    ``count_map_solve`` before the Armijo backtrack was added.

    It lives in the test rather than in the module because it is not an
    alternative the library offers — it is the reference the fix is required
    to reproduce exactly on the problems where it worked, and required to
    beat on the problems where it did not.
    """
    xi = jnp.zeros(op.rank) if xi0 is None else jnp.asarray(xi0)

    def _step(xi, _):
        g = gradient(xi, op)
        H = hessian_separable(xi, op)
        L = jnp.linalg.cholesky(H)
        dx = jax.scipy.linalg.cho_solve((L, True), g)
        return xi - dx, None

    xi_hat, _ = lax.scan(_step, xi, None, length=n_iter)
    H = hessian_separable(xi_hat, op)
    L = jnp.linalg.cholesky(H)
    g = gradient(xi_hat, op)
    return dict(xi_hat=xi_hat, H_chol=L, grad_inf=jnp.max(jnp.abs(g)),
                J=objective(xi_hat, op), n_iter=n_iter)


def _fixture(seed, *, n_gal=2.0e5):
    """A count operator with the closure world's SHAPE at 1/6 its rank.

    The truth is drawn from the model's own prior and the counts are Poisson
    on the fit operator's own ``phi_shell``, which is the benign case: the
    divergence is not a misspecification artefact, it is what exact Newton
    does when ``||xi_true||`` puts the optimum outside the quadratic's basin
    of trust from ``xi = 0``.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_FIT, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z_fine = np.linspace(1e-3, Z_HI, 120)
    basis = build_latent_basis(
        v, np.log1p(z_fine), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_HI, ls_sph=0.9, ls_z=0.15, zeta_fine=np.log1p(z_fine))
    edges = np.linspace(0.02, Z_HI, G_S + 1)
    W = shell_response(edges, z_fine, lambda z: 0.02 * np.ones_like(z),
                       lambda z: z ** 2 + 1e-6)
    f_p = rng.uniform(0.5, 1.0, size=N_FIT)
    xi_true = rng.standard_normal(M_SPH * M_Z)
    phi_shell = np.asarray(W) @ np.asarray(basis.phi_z_fine)
    eta = (np.asarray(basis.phi_sph) @ xi_true.reshape(M_SPH, M_Z)
           @ phi_shell.T)
    mu = f_p[:, None] * np.exp(eta)
    mu = mu * (float(n_gal) / mu.sum())
    counts = rng.poisson(mu).astype(float)
    tracer = TracerCounts(pix=np.arange(N_FIT), counts=counts,
                          completeness=f_p, bias=1.0)
    op = make_count_operator(basis.phi_sph, basis.phi_z_fine, W, tracer)
    return op, xi_true


# ----------------------------------------------------------------- 1. no move

@pytest.mark.parametrize("seed", SEEDS_CONVERGE)
def test_converged_case_is_bit_identical(seed):
    """Where the undamped path converged, the damped path IS the undamped
    path — bit for bit, in ``xi_hat``, ``H_chol``, ``grad_inf`` and ``J``.

    ``atol = rtol = 0`` is the whole point.  The production anchor converged
    undamped at ``grad_inf = 1.09e-10``; a fix that shifts it by one ulp is a
    fix that invalidates a shipped artifact, so this is an equality test and
    not a tolerance test.
    """
    op, _ = _fixture(seed)
    ref = _undamped_solve(op)
    assert float(ref["grad_inf"]) < 1e-8, "fixture premise: this seed converges"
    got = count_map_solve(op)
    for key in ("xi_hat", "H_chol", "grad_inf", "J"):
        np.testing.assert_array_equal(np.asarray(got[key]),
                                      np.asarray(ref[key]))
    # ... and it got there by taking the full Newton step every trip.
    np.testing.assert_array_equal(np.asarray(got["alpha"]),
                                  np.ones(got["n_iter"]))
    assert int(np.sum(np.asarray(got["n_backtrack"]))) == 0


def test_zero_backtrack_reproduces_the_undamped_path_exactly():
    """``max_backtrack = 0`` switches the line search off, and must then
    reproduce the pre-fix iteration EVEN ON THE DIVERGING PROBLEM.

    This is the structural half of the bit-identity argument: the previous
    test could in principle pass because both paths landed on the same
    optimum from different iterates, while this one pins the arithmetic
    itself on a trajectory that never comes near an optimum.  It also
    documents that the damping is the ONLY behavioural change — same
    direction, same objective, same Hessian, same trip count.
    """
    op, _ = _fixture(SEED_DIVERGES)
    ref = _undamped_solve(op)
    got = count_map_solve(op, max_backtrack=0)
    np.testing.assert_array_equal(np.asarray(got["xi_hat"]),
                                  np.asarray(ref["xi_hat"]))
    assert float(ref["grad_inf"]) > 1e3     # the failure, still present


# ------------------------------------------------------------ 2. it converges

def test_diverging_fixture_now_converges():
    """The pin the fix exists for: undamped 9.15e4, damped < 1e-8, at the
    SAME fixed trip count."""
    op, xi_true = _fixture(SEED_DIVERGES)
    ref = _undamped_solve(op)
    assert float(ref["grad_inf"]) > 1e3
    got = count_map_solve(op)
    assert float(got["grad_inf"]) < 1e-8
    # the damped solve found a genuinely lower objective, not just a smaller
    # gradient at a runaway point
    assert float(got["J"]) < float(ref["J"])
    # and it recovers the truth to the accuracy the Hessian predicts
    err = float(np.linalg.norm(np.asarray(got["xi_hat"]) - xi_true))
    assert err < 0.5 * float(np.linalg.norm(xi_true))


@pytest.mark.parametrize("seed", (SEED_DIVERGES,) + SEEDS_CONVERGE)
def test_every_fixture_seed_meets_the_p6_gate(seed):
    op, _ = _fixture(seed)
    assert float(count_map_solve(op)["grad_inf"]) < 1e-8


# --------------------------------------------------- 3. bounded and monotone

def test_backtrack_is_bounded_and_alpha_is_a_power_of_two():
    """The trace is (n_iter,)-shaped, the halvings are capped, and
    ``alpha = shrink^{n_backtrack}`` exactly — i.e. the reported trace is the
    step that was actually taken, not a summary of it."""
    op, _ = _fixture(SEED_DIVERGES)
    sol = count_map_solve(op, n_iter=13, max_backtrack=4)
    alpha = np.asarray(sol["alpha"])
    n_bt = np.asarray(sol["n_backtrack"])
    assert alpha.shape == (13,) and n_bt.shape == (13,)
    assert np.all(n_bt >= 0) and np.all(n_bt <= 4)
    assert np.all(alpha > 0.0) and np.all(alpha <= 1.0)
    np.testing.assert_array_equal(alpha, 0.5 ** n_bt.astype(float))


def test_diverging_fixture_damps_exactly_one_trip():
    """The cure is small because the disease is: ONE halving, on ONE trip,
    and ``alpha = 1`` everywhere else.

    Measured on this fixture the halving lands on trip 3
    (``alpha = [1, 1, 1, 0.5, 1, ...]``); at nside 16 it lands on trip 0
    instead.  Which trip overshoots is a property of the problem, so what is
    pinned is the count — a line search that starts halving on many trips is
    crawling rather than converging, and that is the regression this catches.
    """
    op, _ = _fixture(SEED_DIVERGES)
    sol = count_map_solve(op)
    alpha = np.asarray(sol["alpha"])
    n_bt = np.asarray(sol["n_backtrack"])
    assert int(n_bt.sum()) == 1
    assert int((alpha < 1.0).sum()) == 1
    assert alpha[alpha < 1.0][0] == 0.5


@pytest.mark.parametrize("seed", (SEED_DIVERGES, 0))
def test_objective_decreases_monotonically(seed):
    """``J`` never rises across a trip, up to the Armijo slack.

    The slack is not slop: with ``J ~ 1e5``-``1e7`` the converged trips move
    ``J`` by less than its own f64 resolution, so the honest statement is
    "no increase beyond ``eps |J|``" and that is what is asserted.  The
    undamped path violates this on trip 1 of the diverging problem
    (8.275e6 → 8.506e6 at nside 16), which is the signature the test exists
    to exclude.
    """
    op, _ = _fixture(seed)
    xi = jnp.zeros(op.rank)
    Js = [float(objective(xi, op))]
    for _ in range(13):
        xi, _alpha, _nbt = _damped_newton_step(
            xi, op, c1=1e-4, shrink=0.5, max_backtrack=30, slack_rel=1e-12)
        Js.append(float(objective(xi, op)))
    Js = np.array(Js)
    rise = np.diff(Js)
    tol = 1e-12 * np.abs(Js[:-1])
    assert np.all(rise <= tol), f"J rose by {rise.max():.3e}"
    assert Js[-1] < Js[0]


def test_armijo_slack_is_load_bearing():
    """Without the ``1e-12 |J|`` absolute slack the line search stalls.

    The closure workstream recorded this as a trap and it is worth a pin:
    near the optimum the true decrease is below ``eps |J|``, so a STRICT
    sufficient-decrease test can find no acceptable ``alpha``, backtracks
    hard, and freezes the iterate while the gradient is still finite.

    Measured on this fixture (``seed = 6``, ``J ~ 1.6e5``): with
    ``slack_rel = 0`` the trailing trips halve 2, 11, 21, 19 and 13 times —
    i.e. they take steps of order ``2^-21`` — and the solve ends at
    ``grad_inf = 1.49e-4``, a P6 FAILURE for a rounding reason rather than a
    convergence one.  With the ``1e-12 |J|`` slack the same trips take full
    steps and it ends at 5.18e-12.  If a future refactor drops the slack
    "because it looks like a fudge", this test says what it costs.
    """
    op, _ = _fixture(SEED_DIVERGES)
    strict = count_map_solve(op, slack_rel=0.0)
    slack = count_map_solve(op)
    assert int(np.max(np.asarray(strict["n_backtrack"]))) >= 10
    assert float(strict["grad_inf"]) > 1e-8       # P6 fails, on rounding
    assert int(np.max(np.asarray(slack["n_backtrack"]))) == 1
    assert float(slack["grad_inf"]) < 1e-8


def test_no_host_control_flow_the_solve_still_traces():
    """The module's fixed-trip design exists so the solve can be traced
    (PLAN §10).  A Python ``while`` over the backtrack would break that; the
    :func:`jax.lax.while_loop` inside the :func:`jax.lax.scan` body does not,
    and ``jax.jit`` is the proof."""
    op, _ = _fixture(SEED_DIVERGES)
    jitted = jax.jit(lambda: count_map_solve(op))
    sol = jitted()
    assert float(sol["grad_inf"]) < 1e-8
    np.testing.assert_array_equal(
        np.asarray(sol["xi_hat"]), np.asarray(count_map_solve(op)["xi_hat"]))


# ------------------------------------------- 4. the actual closure world

PR6A = REPO / "experiments" / "field_level_plan" / "pr6a"
MTH_MAP = REPO / "experiments" / "desi_ingest" / "data" / "mth_map_nside128.h5"

#: The nside-16 sub-run, in a SUBPROCESS on purpose.  ``world16`` pins
#: ``DARKSIRENS_ZMAX = 1.5`` and refuses to import under any other value, while
#: the rest of this suite runs at whatever the caller set (the P12 goldens
#: require the DEFAULT zMax, the field-level tests are usually run at 1.0).
#: The z grid is frozen at ``darksirens.redshift`` import time, so the two
#: cannot share a process — and the honest fix is a second process, not a
#: monkeypatched grid.
_NSIDE16_DRIVER = textwrap.dedent("""
    import json, sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    sys.path.insert(0, sys.argv[2])
    sys.path.insert(0, sys.argv[3])
    import world16 as W16
    from darksirens.redshift.latent_counts import count_map_solve
    from test_latent_solve_damping import _undamped_solve

    world = W16.build_world()
    rows = []
    for k in range(8):
        rng = np.random.default_rng(6100 + 977 * k)
        xi_true = W16.draw_xi_true(world, rng)
        counts, _ = W16.draw_counts(world, xi_true, rng, n_target=1.1e6)
        op = W16.fit_operator(world, counts)
        sol = count_map_solve(op)
        ref = _undamped_solve(op)
        rows.append(dict(k=k, grad_inf=float(sol["grad_inf"]),
                         grad_inf_undamped=float(ref["grad_inf"]),
                         alpha=[float(a) for a in np.asarray(sol["alpha"])],
                         n_bt=int(np.sum(np.asarray(sol["n_backtrack"]))),
                         bit_identical=bool(np.array_equal(
                             np.asarray(sol["xi_hat"]),
                             np.asarray(ref["xi_hat"]))
                             and np.array_equal(np.asarray(sol["H_chol"]),
                                                np.asarray(ref["H_chol"])))))
    print("RESULT " + json.dumps(rows))
""")


@pytest.mark.skipif(not (PR6A / "world16.py").exists()
                    or not MTH_MAP.exists(),
                    reason="the PR-6a closure world (world16 + the DESI depth "
                           "map) is not present in this checkout")
def test_nside16_closure_realizations(tmp_path):
    """The five realizations that actually failed, at the geometry they
    failed on: nside 16, 1854 x 12 voxels, rank 320, 1.1e6 galaxies.

    Undamped, ``seed = 6100 + 977 k`` gave P6 pass rate 3/8 with ``grad_inf``
    of 4.41e5, 4.86e5, 4.36e5, 4.00e5, 4.07e5 on ``k = 1, 4, 5, 6, 7``
    (reproduced from ``tier_a.json`` to the digit).  Damped, all 8 must pass,
    and the five that failed must each show exactly one halving on trip 0.

    The three that already converged (``k = 0, 2, 3``) carry the bit-identity
    pin at PRODUCTION SCALE — rank 320 over 1854 x 12 voxels, the same shape
    and conditioning as the anchor the builder solves, rather than the
    48-mode fixture above.  ``xi_hat`` AND ``H_chol`` must match the undamped
    path exactly; ``H_chol`` matters as much as ``xi_hat`` because it is what
    the Laplace draws and the sensitivity solves are built against.
    """
    try:
        import healpy  # noqa: F401
    except ImportError:                                   # pragma: no cover
        pytest.skip("healpy not available")
    script = tmp_path / "nside16_driver.py"
    script.write_text(_NSIDE16_DRIVER)
    env = dict(os.environ)
    env["DARKSIRENS_ZMAX"] = "1.5"          # world16's own pin
    env.setdefault("JAX_PLATFORMS", "cpu")
    out = subprocess.run(
        [sys.executable, str(script), str(REPO), str(PR6A),
         str(Path(__file__).resolve().parent)],
        capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, out.stdout[-4000:] + out.stderr[-4000:]
    import json
    rows = json.loads(line[-1][len("RESULT "):])
    assert len(rows) == 8
    bad = [r for r in rows if r["grad_inf"] >= 1e-8]
    assert not bad, f"P6 gate still fails on {bad}"
    # the five that used to diverge damp once, on trip 0, and nowhere else;
    # the three that already converged are untouched, bit for bit
    for r in rows:
        if r["k"] in (1, 4, 5, 6, 7):
            assert r["grad_inf_undamped"] > 1e3, r      # the failure, pinned
            assert r["n_bt"] == 1, r
            assert r["alpha"][0] == 0.5, r
            assert all(a == 1.0 for a in r["alpha"][1:]), r
        else:
            assert r["grad_inf_undamped"] < 1e-8, r
            assert r["n_bt"] == 0, r
            assert r["bit_identical"], r
