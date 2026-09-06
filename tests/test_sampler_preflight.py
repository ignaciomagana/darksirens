"""Nested-sampler preflight + always-on selection-guard summary.

The preflight (``darksirens.inference.sampling._nested_sampler_preflight``,
called by ``run_sampler`` for dynesty/tinyns) probes a handful of prior draws
before the sampler tries to seed its live points.  When the likelihood is
``-inf`` everywhere — almost always the selection variance guard firing — it
fails fast with an actionable error instead of letting dynesty reject-sample
forever.  These tests pin:

* fail-fast: an all-``-inf`` likelihood through ``run_sampler`` with dynesty
  raises a ``RuntimeError`` naming ``--selection_neff_guard`` and
  ``--max_likelihood_variance`` BEFORE dynesty is ever imported/iterated;
* pass-through: a finite likelihood prints the ``preflight: k/N`` line and does
  not raise;
* ``--sampler_preflight off`` skips the probe (no print, no raise);
* low finite fraction (0 < k <= 3) prints a loud, non-fatal warning;
* the always-printed guard summary is produced by a small pure function
  (``format_selection_guard_summary``) with the resolved mode + why, cap, and
  criterion.

Kept dependency-light: numpy + jax only; the fail-fast path never reaches the
dynesty import, so ``dynesty`` need not be installed for that test.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.inference.sampling import run_sampler, _nested_sampler_preflight
from darksirens.cli.inference import format_selection_guard_summary


def _neg_inf_likelihood(theta):
    return jnp.asarray(-jnp.inf)


def _finite_likelihood(theta):
    # Trivial finite logL everywhere (sum of squares); the preflight only cares
    # that float(logL) is finite.
    return -0.5 * jnp.sum(jnp.asarray(theta) ** 2)


_IDENTITY_PTFORM = lambda u: jnp.asarray(u)


# ---------------------------------------------------------------------------
# Fail-fast
# ---------------------------------------------------------------------------

def test_preflight_fail_fast_raises_before_dynesty():
    """All-(-inf) likelihood + dynesty: run_sampler must raise from the
    preflight (which runs before the dynesty dispatch/import), and the message
    must name both escape-hatch flags and the diagnostic script."""
    opts = SimpleNamespace(seed=3, nlive=50)  # sampler_preflight defaults "on"
    with pytest.raises(RuntimeError) as excinfo:
        run_sampler(
            "dynesty",
            _neg_inf_likelihood,
            _IDENTITY_PTFORM,
            labels=["x", "y"],
            lower_bound=np.zeros(2),
            upper_bound=np.ones(2),
            opts=opts,
        )
    msg = str(excinfo.value)
    assert "preflight" in msg.lower()
    assert "--selection_neff_guard" in msg
    assert "--max_likelihood_variance" in msg
    assert "scripts/diagnose_selection_guard.py" in msg


def test_preflight_fail_fast_via_helper_reports_zero():
    opts = SimpleNamespace(seed=1)
    with pytest.raises(RuntimeError, match="ALL 32"):
        _nested_sampler_preflight(_neg_inf_likelihood, _IDENTITY_PTFORM, 2, opts)


# ---------------------------------------------------------------------------
# Pass-through when finite
# ---------------------------------------------------------------------------

def test_preflight_finite_passes_through(capsys):
    opts = SimpleNamespace(seed=7, nlive=40)
    # Must not raise.
    _nested_sampler_preflight(_finite_likelihood, _IDENTITY_PTFORM, 3, opts)
    out = capsys.readouterr().out
    assert "preflight: 4/4 prior draws have finite logL" in out
    assert "finite logL in [" in out
    # The elapsed-seconds cost is reported.
    assert " s]" in out
    # A fully finite probe emits no warning line.
    assert "preflight WARNING" not in out


# ---------------------------------------------------------------------------
# --sampler_preflight off
# ---------------------------------------------------------------------------

def test_preflight_off_skips(capsys):
    opts = SimpleNamespace(seed=7, nlive=40, sampler_preflight="off")
    # Even an all-(-inf) likelihood must NOT raise when the probe is off.
    _nested_sampler_preflight(_neg_inf_likelihood, _IDENTITY_PTFORM, 3, opts)
    out = capsys.readouterr().out
    assert "preflight:" not in out


# ---------------------------------------------------------------------------
# Resumed runs: the probe guards a FRESH initial-live-point search only
# ---------------------------------------------------------------------------

def test_preflight_skipped_on_a_resume(capsys):
    """A checkpoint already carries nlive finite-logL live points, and a tight
    posterior makes k == 0 out of 32 PRIOR draws the expected outcome: the
    fail-fast must not kill a requeued job (whose only escapes would all
    invalidate the resume fingerprint)."""
    opts = SimpleNamespace(
        seed=3, nlive=50,
        checkpoint_interval_seconds=0.0, checkpoint_file_resolved=None,
        resume_from_resolved="/nonexistent/checkpoint.save",
    )
    with pytest.raises(Exception) as excinfo:
        run_sampler(
            "dynesty",
            _neg_inf_likelihood,
            _IDENTITY_PTFORM,
            labels=["x", "y"],
            lower_bound=np.zeros(2),
            upper_bound=np.ones(2),
            opts=opts,
        )
    # It died restoring the (nonexistent) checkpoint, NOT in the preflight.
    assert "preflight" not in str(excinfo.value).lower()
    out = capsys.readouterr().out
    assert "preflight skipped: resuming" in out
    assert "prior draws have finite logL" not in out


def test_preflight_still_runs_for_a_fresh_checkpointed_run(capsys):
    """Checkpointing ON but nothing to resume from is a FRESH run: the probe
    must still fire (it is the only thing standing between an all--inf
    likelihood and dynesty spinning forever)."""
    opts = SimpleNamespace(
        seed=3, nlive=50,
        checkpoint_interval_seconds=1800.0,
        checkpoint_file_resolved="/tmp/nonexistent/checkpoint.save",
        resume_from_resolved=None,
    )
    with pytest.raises(RuntimeError, match="preflight"):
        run_sampler(
            "dynesty",
            _neg_inf_likelihood,
            _IDENTITY_PTFORM,
            labels=["x", "y"],
            lower_bound=np.zeros(2),
            upper_bound=np.ones(2),
            opts=opts,
        )


# ---------------------------------------------------------------------------
# Low finite fraction -> loud, non-fatal warning
# ---------------------------------------------------------------------------

def test_preflight_low_finite_fraction_warns(capsys):
    """A likelihood finite on only a thin sliver of the prior (here: exactly 2
    of the 32 probes) triggers the 0 < k <= 3 warning branch — loud but not
    fatal, with a slow-init estimate and the remedies."""
    n_finite = 2
    state = {"calls": 0}

    def corner_likelihood(theta):
        i = state["calls"]
        state["calls"] += 1
        return jnp.asarray(0.0 if i < n_finite else -jnp.inf)

    opts = SimpleNamespace(seed=5, nlive=100)
    # Non-fatal: returns normally.
    _nested_sampler_preflight(corner_likelihood, _IDENTITY_PTFORM, 2, opts)
    out = capsys.readouterr().out
    assert "preflight: 2/32 prior draws have finite logL" in out
    assert "preflight WARNING" in out
    # The slow-init estimate is nlive / (k/32) = 100 / (2/32) = 1600 draws.
    assert "1,600" in out
    assert "--selection_neff_guard soft" in out


# ---------------------------------------------------------------------------
# Independent RNG: the probe must not disturb the sampler's RNG stream
# ---------------------------------------------------------------------------

def test_preflight_uses_independent_rng(capsys):
    """The probe draws from np.random.default_rng(seed ^ 0xC0FFEE), a private
    Generator, so it must leave the global numpy RNG untouched (dynesty seeds
    its own default_rng(seed); see test_dynesty_seed)."""
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    opts = SimpleNamespace(seed=7, nlive=40)
    _nested_sampler_preflight(_finite_likelihood, _IDENTITY_PTFORM, 3, opts)
    after = np.random.random()
    assert before == after


# ---------------------------------------------------------------------------
# Always-on guard summary (pure function)
# ---------------------------------------------------------------------------

def test_guard_summary_dynesty_hard():
    s = format_selection_guard_summary("auto", "dynesty", soft_guard=False,
                                        max_likelihood_variance=1.0)
    assert s.startswith("selection guard: HARD (auto: dynesty)")
    assert "-inf when sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel > 1.0" in s
    assert "(--max_likelihood_variance)" in s
    assert "Vitale floor Neff > 5*N_obs always applies" in s


def test_guard_summary_numpyro_soft():
    s = format_selection_guard_summary("auto", "numpyro", soft_guard=True,
                                        max_likelihood_variance=1.0)
    assert s.startswith("selection guard: SOFT (auto: numpyro)")
    assert "smooth penalty wall" in s
    assert "verify the posterior clears it post hoc" in s
    assert "Vitale floor Neff > 5*N_obs always applies" in s


def test_guard_summary_explicit_modes_and_cap():
    hard = format_selection_guard_summary("hard", "tinyns", soft_guard=False,
                                          max_likelihood_variance=4.0)
    assert "HARD (explicit)" in hard
    assert "> 4.0 (--max_likelihood_variance)" in hard
    soft = format_selection_guard_summary("soft", "dynesty", soft_guard=True,
                                          max_likelihood_variance=1.0)
    assert "SOFT (explicit)" in soft


def test_preflight_stops_after_the_fourth_finite_draw(capsys):
    """The probe is an early exit: 4 finite draws settle both verdicts."""
    calls = []

    def counting_likelihood(theta):
        calls.append(theta)
        return -1.0

    opts = SimpleNamespace(seed=7, nlive=40)
    _nested_sampler_preflight(counting_likelihood, _IDENTITY_PTFORM, 3, opts)
    assert len(calls) == 4
    out = capsys.readouterr().out
    assert "preflight: 4/4 prior draws have finite logL" in out


def test_preflight_probes_all_draws_when_few_are_finite(capsys):
    """0 < k <= 3 cannot break early, so the warning still sees all 32."""
    calls = []

    def rarely_finite(theta):
        calls.append(theta)
        return -1.0 if len(calls) in (1, 5) else -np.inf

    opts = SimpleNamespace(seed=7, nlive=40)
    _nested_sampler_preflight(rarely_finite, _IDENTITY_PTFORM, 3, opts)
    assert len(calls) == 32
    out = capsys.readouterr().out
    assert "2/32 prior draws are finite" in out
