"""The gate scripts' inlined posterior summary must equal ``arms.summarize``.

`gate_complete.py`, `gate_specz.py` and `vollim/run_h0.py` each reimplement the
pdf/cdf/quantile reduction instead of calling `arms.summarize`, and their outputs
are compared DIRECTLY against Tier C's, which does call it.  If the two ever
diverge -- a different trapezoid convention, a different `-inf` policy, a
different quantile interpolation -- every one of those comparisons silently
becomes a comparison of two estimators rather than of two configurations.

Measured today: identical to the last bit over 400 randomised curves, including
the `-inf` leading nodes that real low-`H0` scans produce.  So this is a pin on a
duplication that is currently harmless, not a bug report.  It exists because this
campaign has already spent a day on two quantities that were computed different
ways and compared as if they were the same.

Run: pytest experiments/field_level_plan/pr6a/test_summary_consistency.py
"""
from __future__ import annotations

import numpy as np


def _gate_inline(h0, vals):
    """Verbatim reduction from gate_complete.py / gate_specz.py."""
    ok = np.isfinite(vals)
    v = np.where(ok, vals, -np.inf)
    pdf = np.exp(v - v.max())
    pdf = np.where(ok, pdf, 0.0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    pdf = pdf / trapz(pdf, h0)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                         * np.diff(h0))])
    cdf /= cdf[-1]
    q = lambda t: float(np.interp(t, cdf, h0))  # noqa: E731
    return q(0.5), 0.5 * (q(0.84) - q(0.16))


def test_gate_summary_matches_arms_summarize():
    import arms as A
    import world16 as W16

    rng = np.random.default_rng(0)
    h0 = np.arange(20.0, 140.0 + 0.5 * 2.5, 2.5)
    worst_med = worst_sig = worst_cdf = 0.0
    for t in range(400):
        c = rng.uniform(40.0, 110.0)
        w = rng.uniform(3.0, 25.0)
        vals = (-0.5 * ((h0 - c) / w) ** 2 * rng.uniform(0.8, 1.2)
                + rng.normal(0.0, 0.02, h0.size))
        if t % 7 == 0:
            # real low-H0 scans return -inf at the first few nodes
            vals[: rng.integers(1, 4)] = -np.inf
        a = A.summarize(h0, vals)
        med, sig = _gate_inline(h0, vals)
        worst_med = max(worst_med, abs(a["median"] - med))
        worst_sig = max(worst_sig, abs(a["sigma"] - sig))
        # cdf_at_truth is the calibration statistic every tier is judged on
        ok = np.isfinite(vals)
        v = np.where(ok, vals, -np.inf)
        pdf = np.exp(v - v.max())
        pdf = np.where(ok, pdf, 0.0)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        pdf = pdf / trapz(pdf, h0)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                             * np.diff(h0))])
        cdf /= cdf[-1]
        worst_cdf = max(worst_cdf, abs(
            a["cdf_at_truth"] - float(np.interp(W16.H0_TRUE, h0, cdf))))
    assert worst_med == 0.0, worst_med
    assert worst_sig == 0.0, worst_sig
    assert worst_cdf == 0.0, worst_cdf
