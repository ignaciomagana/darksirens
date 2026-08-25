"""``make_prior_transform`` must be called through the CHEAPEST convention its
branch supports -- without moving a single sampled value (perf review rank 6).

dynesty calls the prior transform once per proposal, as often as the
likelihood.  Two of the three branches were paying for that dispatch:

* all-uniform with no joint constraint is a per-dimension affine map, yet
  ``dynesty_ptform``'s ``jnp.asarray`` sent it to the device and back --
  measured 681 us/call, against 1.0 us/call for the identical numpy
  expression;
* the non-uniform branch is a ~40-op graph (two truncated-normal PPFs, the
  Beta(1, b) PPF, three selects) dispatched op-by-op -- 26 ms/call, 0.64 ms
  jitted.

So ``make_prior_transform`` now advertises which convention its closure wants
(``host_native`` / ``prefer_jit``) and ``run_sampler`` dispatches on it -- the
jit only after checking, on the live backend, that the compiled transform
reproduces the eager one bit for bit.  The all-uniform branches are never
jitted: fusing ``u * span + lower`` into an FMA moves a sampled parameter by an
ulp, which the repo's bit-identity pin forbids.  Every test here pins EXACT
equality, since the whole change is only admissible if it is invisible.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from darksirens.inference.prior import (
    build_parameter_space,
    make_prior_transform,
    resolve_joint_prior_constraints,
)

NDIM = 6
LOWER = np.linspace(-2.0, 0.5, NDIM)
UPPER = np.linspace(1.5, 4.0, NDIM)

NON_UNIFORM_KINDS = [
    ("uniform", None, None),
    ("normal", 0.0, 1.0),
    ("lognormal", 0.0, 0.5),
    ("beta", 1.0, 3.0),
    ("uniform", None, None),
    ("normal", 1.0, 2.0),
]
NON_UNIFORM_LOWER = np.array([-2.0, -3.0, 1e-3, 0.0, 0.0, -3.0])
NON_UNIFORM_UPPER = np.array([2.0, 3.0, 10.0, 1.0, 1.0, 5.0])


def _cube(n=None, ndim=NDIM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(size=ndim if n is None else (n, ndim))


def _strip_dispatch(transform):
    """The same closure with its flags removed: the pre-review calling path."""
    for flag in ("host_native", "prefer_jit"):
        if hasattr(transform, flag):
            delattr(transform, flag)
    return transform


# --- which branch advertises what ------------------------------------------

def test_uniform_branch_without_joint_constraints_is_host_native():
    t = make_prior_transform(LOWER, UPPER)
    assert getattr(t, "host_native", False) is True
    assert not getattr(t, "prefer_jit", False)
    u = _cube()
    theta = t(u)
    # numpy in -> numpy out: dynesty never touches the device on this path.
    assert isinstance(theta, np.ndarray)
    # ... and bit-identical to the JAX expression it replaces.
    legacy = np.asarray(jnp.asarray(u) * (jnp.asarray(UPPER) - jnp.asarray(LOWER))
                        + jnp.asarray(LOWER))
    np.testing.assert_array_equal(theta, legacy)


def test_host_native_closure_is_still_traceable():
    """tinyns jits the transform and the preflight probe hands it a JAX array,
    so the host branch must keep lowering to EXACTLY the graph it used to.

    Note the two conventions are not interchangeable and never were: XLA
    contracts ``u * span + lower`` into an FMA, so the jitted answer already
    differs from the eager one by an ulp on this backend.  That is precisely
    why the dynesty call site jits the non-uniform branch only -- what must
    hold is that each convention returns what it returned before.
    """
    t = make_prior_transform(LOWER, UPPER)

    def legacy(u):                      # the pre-review closure, verbatim
        return u * (UPPER - LOWER) + LOWER

    u = _cube(seed=1)
    uj = jnp.asarray(u)
    # dynesty's new host convention == the old eager-JAX one, bit for bit.
    np.testing.assert_array_equal(np.asarray(t(u)), np.asarray(legacy(uj)))
    # tinyns' jitted convention: unchanged graph, unchanged values.
    np.testing.assert_array_equal(
        np.asarray(jax.jit(t)(uj)), np.asarray(jax.jit(legacy)(uj)))
    # ... and the eager-JAX call (the preflight probe) too.
    np.testing.assert_array_equal(np.asarray(t(uj)), np.asarray(legacy(uj)))


def test_uniform_branch_with_joint_constraints_is_not_flagged():
    """``_apply_joint`` is a JAX cube map, so there is no host path -- and
    jitting an all-uniform mul+add is refused (one-ulp FMA risk)."""
    labels, lower, upper, kinds = _gwtc5_space()
    jc = resolve_joint_prior_constraints(
        "gwtc5_fiducial_bpl2peaks", labels, lower, upper, kinds)
    assert jc, "fixture lost its joint constraints"
    t = make_prior_transform(lower, upper, kinds, joint_constraints=jc)
    assert not getattr(t, "host_native", False)
    assert not getattr(t, "prefer_jit", False)


def test_non_uniform_branch_is_flagged_for_jit():
    t = make_prior_transform(
        NON_UNIFORM_LOWER, NON_UNIFORM_UPPER, NON_UNIFORM_KINDS)
    assert getattr(t, "prefer_jit", False) is True
    assert not getattr(t, "host_native", False)


# --- the dispatcher itself -------------------------------------------------

def test_dispatcher_returns_the_eager_values_on_every_branch():
    """Whatever convention is picked, the theta dynesty sees is the theta the
    eager transform produced -- that is the entire numerics contract."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    for lower, upper, kinds in (
        (LOWER, UPPER, None),
        (NON_UNIFORM_LOWER, NON_UNIFORM_UPPER, NON_UNIFORM_KINDS),
    ):
        t = make_prior_transform(lower, upper, kinds)
        ptform = _make_dynesty_ptform(t, NDIM)
        for seed in (5, 6, 7):
            u = _cube(seed=seed)
            eager = np.asarray(
                _strip_dispatch(make_prior_transform(lower, upper, kinds))(
                    jnp.asarray(u)))
            np.testing.assert_array_equal(ptform(u), eager)


def test_dispatcher_jits_when_the_compiled_transform_is_exact():
    """``prefer_jit`` really does compile: the closure's Python body stops
    running once dynesty is calling it."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    body_runs = []

    def transform(u):
        body_runs.append(1)          # trace time only, once compiled
        return u * 2.0               # exact in binary: jit cannot move it
    transform.prefer_jit = True

    ptform = _make_dynesty_ptform(transform, 3)
    before = len(body_runs)
    for _ in range(5):
        np.testing.assert_array_equal(ptform(np.full(3, 0.25)), np.full(3, 0.5))
    assert len(body_runs) == before, "transform body re-ran: it was not jitted"


def test_dispatcher_falls_back_when_the_compiled_transform_moves_values():
    """The escape hatch: XLA contracts the truncated-normal PPF's polynomial
    into FMAs (measured ~30 ulps on the CPU backend, and 3-6% of draws moved on
    the H100; the Beta stick is exact on both).  Sampled values are pinned, so
    a configuration that fuses must keep the eager path."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    def transform(u):
        # Stands in for a lowering that does not reproduce the eager answer.
        if isinstance(u, jax.core.Tracer):
            return u * 3.0
        return u * 2.0
    transform.prefer_jit = True

    ptform = _make_dynesty_ptform(transform, 3)
    np.testing.assert_array_equal(ptform(np.full(3, 0.25)), np.full(3, 0.5))


# --- the decision is visible, and refusable -------------------------------

def test_every_dispatch_decision_is_announced_and_labelled(capsys):
    """Which convention produced the samples must never be silent -- in a repo
    that pins bit identity, an ACCEPTED jit needs a record as much as a
    rejected one does."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    def exact(u):
        return u * 2.0                 # exact in binary: the jit is accepted
    exact.prefer_jit = True

    def fusing(u):
        return u * 3.0 if isinstance(u, jax.core.Tracer) else u * 2.0
    fusing.prefer_jit = True

    labels, lower, upper, kinds = _gwtc5_space()
    jc = resolve_joint_prior_constraints(
        "gwtc5_fiducial_bpl2peaks", labels, lower, upper, kinds)

    cases = [
        (make_prior_transform(LOWER, UPPER), NDIM, "auto", "host"),
        (exact, 3, "auto", "jit"),
        (fusing, 3, "auto", "eager-not-bit-identical"),
        (make_prior_transform(lower, upper, kinds, joint_constraints=jc),
         len(labels), "auto", "eager"),
        (make_prior_transform(LOWER, UPPER), NDIM, "eager", "eager-forced"),
    ]
    for transform, ndim, mode, expected in cases:
        capsys.readouterr()
        ptform = _make_dynesty_ptform(transform, ndim, mode=mode)
        out = capsys.readouterr().out
        assert ptform.dispatch == expected
        assert expected in out, f"{expected} decision was not printed: {out!r}"


def test_eager_mode_restores_the_pre_review_call_path():
    """``--prior_transform_dispatch eager`` must reproduce the OLD wrapper
    exactly, on every branch, so a run predating the dispatch can be rerun."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    for lower, upper, kinds in (
        (LOWER, UPPER, None),
        (NON_UNIFORM_LOWER, NON_UNIFORM_UPPER, NON_UNIFORM_KINDS),
    ):
        forced = _make_dynesty_ptform(
            make_prior_transform(lower, upper, kinds), NDIM, mode="eager")
        legacy = _strip_dispatch(make_prior_transform(lower, upper, kinds))
        for seed in (8, 9):
            u = _cube(seed=seed)
            np.testing.assert_array_equal(
                forced(u), np.asarray(legacy(jnp.asarray(u))))


def test_a_flagged_transform_that_will_not_compile_falls_back(capsys):
    """A ``prefer_jit`` closure that cannot be traced is a speed problem, not a
    correctness one -- the run must continue on the eager path."""
    from darksirens.inference.sampling import _make_dynesty_ptform

    def untraceable(u):
        if isinstance(u, jax.core.Tracer):
            raise TypeError("cannot trace this")
        return np.asarray(u) * 2.0
    untraceable.prefer_jit = True

    ptform = _make_dynesty_ptform(untraceable, 3)
    assert ptform.dispatch == "eager-jit-unavailable"
    assert "could not be compiled or probed" in capsys.readouterr().out
    np.testing.assert_array_equal(ptform(np.full(3, 0.25)), np.full(3, 0.5))


def test_unknown_dispatch_mode_is_refused():
    from darksirens.inference.sampling import _make_dynesty_ptform

    with pytest.raises(ValueError, match="prior_transform_dispatch"):
        _make_dynesty_ptform(make_prior_transform(LOWER, UPPER), NDIM,
                             mode="jit")


# --- batching (what makes the test suite's cube draws affordable) ----------

def _gwtc5_space():
    res = build_parameter_space(
        "gwtc5_fiducial_bpl2peaks", False, True, True, fix_de=True,
        universe_model="spectral_sirens",
    )
    return list(res[0]), np.asarray(res[1], float), np.asarray(res[2], float), res[11]


@pytest.mark.parametrize("with_joint", [False, True])
def test_batched_cube_equals_the_per_row_loop(with_joint):
    """The transforms index with ``u[..., i]``, so a whole ``(n, ndim)`` cube
    goes through in one call.  Tests that used to loop row by row rely on that
    being EXACTLY the looped answer (measured 372x faster, maxdiff 0)."""
    labels, lower, upper, kinds = _gwtc5_space()
    jc = (resolve_joint_prior_constraints(
        "gwtc5_fiducial_bpl2peaks", labels, lower, upper, kinds)
        if with_joint else [])
    t = make_prior_transform(lower, upper, kinds, joint_constraints=jc)
    u = jnp.asarray(_cube(64, len(labels), seed=3))
    looped = np.asarray(jnp.stack([t(u[k]) for k in range(u.shape[0])]))
    np.testing.assert_array_equal(np.asarray(t(u)), looped)


def test_batched_cube_equals_the_per_row_loop_non_uniform():
    t = make_prior_transform(
        NON_UNIFORM_LOWER, NON_UNIFORM_UPPER, NON_UNIFORM_KINDS)
    u = jnp.asarray(_cube(64, NDIM, seed=4))
    looped = np.asarray(jnp.stack([t(u[k]) for k in range(u.shape[0])]))
    np.testing.assert_array_equal(np.asarray(t(u)), looped)


# --- the dynesty call site: same run, whichever convention it picks ---------

@pytest.mark.parametrize("kinds", [None, "non_uniform"])
def test_dynesty_run_is_unchanged_by_the_dispatch(kinds, tmp_path):
    """A/B escape hatch: strip the flags and dynesty falls back to the old
    eager path.  Same seed, so the two runs must agree bit for bit -- if the
    fast conventions perturbed a single transform output, the nested run would
    diverge immediately."""
    pytest.importorskip("dynesty")
    from darksirens.inference.sampling import run_sampler

    if kinds is None:
        lower, upper, prior_kinds = LOWER, UPPER, None
    else:
        lower, upper, prior_kinds = (
            NON_UNIFORM_LOWER, NON_UNIFORM_UPPER, NON_UNIFORM_KINDS)
    labels = [f"p{i}" for i in range(NDIM)]

    def loglike(theta):
        return -0.5 * jnp.sum(jnp.asarray(theta) ** 2)

    def _opts(dispatch="auto"):
        return SimpleNamespace(
            seed=11, show_progress=False, nlive=30, dlogz=10.0,
            max_samples=0, dynesty_diagnostics=False, save_path=str(tmp_path),
            sampler_preflight="off", prior_transform_dispatch=dispatch,
        )

    fast = run_sampler(
        "dynesty", loglike, make_prior_transform(lower, upper, prior_kinds),
        labels, lower, upper, _opts(),
    )
    slow = run_sampler(
        "dynesty", loglike,
        _strip_dispatch(make_prior_transform(lower, upper, prior_kinds)),
        labels, lower, upper, _opts(),
    )
    # ... and the same again through the CLI escape hatch, which must not need
    # the flags stripped in Python to get the old path back.
    forced = run_sampler(
        "dynesty", loglike, make_prior_transform(lower, upper, prior_kinds),
        labels, lower, upper, _opts("eager"),
    )
    for other in (slow, forced):
        np.testing.assert_array_equal(fast["samples"], other["samples"])
        assert fast["logZ"] == other["logZ"]
        assert fast["logZerr"] == other["logZerr"]
    # The convention that produced the samples is recorded for provenance.
    assert fast["prior_transform_dispatch"] in (
        "host", "jit", "eager", "eager-not-bit-identical",
        "eager-jit-unavailable")
    assert forced["prior_transform_dispatch"] == "eager-forced"
