import numpy as np
import pytest

from darksirens.cli.inference_lensing import _diagnostics_at_guard_clear_point


LOWER = np.zeros(3)
UPPER = np.ones(3)
MID = 0.5 * (LOWER + UPPER)


def _guard_error():
    return RuntimeError(
        "componentwise exact factorization: selection correction is "
        "NON-FINITE at the evaluation point (delta nan vs nan)"
    )


def test_midpoint_clear_uses_midpoint():
    calls = []

    def fn(point):
        calls.append(np.asarray(point))
        return {"logL_total": 1.0}

    diag, point = _diagnostics_at_guard_clear_point(fn, MID, LOWER, UPPER, seed=7)
    assert diag == {"logL_total": 1.0}
    assert np.allclose(point, MID)
    assert len(calls) == 1


def test_guarded_midpoint_falls_back_to_prior_draw():
    calls = []

    def fn(point):
        calls.append(np.asarray(point))
        if len(calls) == 1:
            raise _guard_error()
        return {"logL_total": 2.0}

    diag, point = _diagnostics_at_guard_clear_point(fn, MID, LOWER, UPPER, seed=7)
    assert diag == {"logL_total": 2.0}
    assert not np.allclose(point, MID)
    assert np.all((point >= LOWER) & (point <= UPPER))
    assert len(calls) == 2


def test_fallback_draws_are_seeded_deterministic():
    def fn_fail_first(point):
        if np.allclose(point, MID):
            raise _guard_error()
        return {"point": np.asarray(point).tolist()}

    d1, p1 = _diagnostics_at_guard_clear_point(fn_fail_first, MID, LOWER, UPPER, seed=11)
    d2, p2 = _diagnostics_at_guard_clear_point(fn_fail_first, MID, LOWER, UPPER, seed=11)
    assert np.allclose(p1, p2)


def test_all_points_guarded_raises_with_context():
    def fn(point):
        raise _guard_error()

    with pytest.raises(RuntimeError, match="no guard-clear evaluation point"):
        _diagnostics_at_guard_clear_point(fn, MID, LOWER, UPPER, seed=7, n_draws=4)


def test_unrelated_runtime_error_propagates_immediately():
    calls = []

    def fn(point):
        calls.append(1)
        raise RuntimeError("componentwise exact factorization failed: not count-only")

    with pytest.raises(RuntimeError, match="not count-only"):
        _diagnostics_at_guard_clear_point(fn, MID, LOWER, UPPER, seed=7)
    assert len(calls) == 1


def test_fiducial_candidate_tried_before_prior_draws():
    calls = []

    def fn(point):
        calls.append(np.asarray(point))
        if len(calls) == 1:
            raise _guard_error()
        return {"logL_total": 3.0}

    fid = np.array([0.25, 0.5, 0.75])
    diag, point = _diagnostics_at_guard_clear_point(
        fn, MID, LOWER, UPPER, seed=7, fiducial=fid
    )
    assert np.allclose(point, fid)
    assert len(calls) == 2


def test_fiducial_candidate_point_maps_labels_and_clips():
    from types import SimpleNamespace
    from darksirens.cli.inference_lensing import _fiducial_candidate_point
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.prior import pop_model_prior_parser

    pop_model = "powerlaw+peak@md"
    fid_vec = np.asarray(get_fixed_population_params(pop_model), dtype=float)
    _, _, pop_labels, _, _ = pop_model_prior_parser(pop_model)
    labels = list(pop_labels) + ["log10_tau_A"]
    n = len(labels)
    lower = np.full(n, -100.0)
    upper = np.full(n, 100.0)
    mid = 0.5 * (lower + upper)
    opts = SimpleNamespace(pop_model=pop_model, sl_tau_A=5e-4, sl_tau_n=3.0)
    point = _fiducial_candidate_point(labels, mid, lower, upper, opts, fid_vec)
    assert np.allclose(point[:-1], fid_vec)
    assert np.isclose(point[-1], np.log10(5e-4))
    # clipping: a fiducial outside the box lands strictly inside
    tight_upper = np.full(n, 1.0)
    tight_lower = np.full(n, -1.0)
    clipped = _fiducial_candidate_point(
        labels, 0.0 * mid, tight_lower, tight_upper, opts, fid_vec
    )
    assert np.all(clipped > tight_lower) and np.all(clipped < tight_upper)
