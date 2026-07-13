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
