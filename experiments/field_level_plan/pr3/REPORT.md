# PR-3 — the count channel and the K9 promotion decision (2026-08-13)

Module: `darksirens/redshift/latent_counts.py` (unit pins P4/P5/P6 in
`tests/test_latent_counts.py`, all passing; theta is an argument — the
anchor is the `theta_ref` special case, verified by re-solving at a
different within-shell base with the same machinery).

## The campaign (job 1127052, 31 s on one H100)

Real DESI union counts: 30,470 occupied nside-64 pixels, 22,787,566
galaxies, 12 shells over [0, 0.30] (**all pass occupancy guard 7**:
>= 1e4 galaxies, >= 500 occupied pixels per shell). OD3 kernel
(ls_sph = 0.2, ls_z = 0.039, M = 315 x 8 = 2520), PR-2 `f_p` from the real
depth map, `b_gal = 1` fixed. Anchor solve: `grad_inf = 1.1e-10`,
per-mode field amplitude `||xi_hat||/sqrt(M) = 2.46` (the count channel is
strongly data-constrained — clustering is unambiguous). 20 theta drawn
around the anchor (sd: M0hat 0.05, sigma_M 0.05, delta 0.10, Om0 0.01),
each re-solved with the theta-dependent base inside `W(theta)`.

| gate | value | threshold | verdict |
|---|---|---|---|
| P7 (tau, misspecification) | 9.7e-2 | reported | at the 0.1 misspecification scale |
| **P7b (linear-response residual)** | **6.3e-3** | < 0.1 | linear response valid |
| **P7c (GW-side, osc a·Δxi)** | **5.4e-4 nat** | < 0.1 | **benign — 300x under the gate** |
| P7d (log-det drift) | 1.47 nat | < 0.1 | exceeds — evidence would need the tr(H^-1 dH/dtheta) term |
| P7e (galaxy-side evidence osc) | 59.9 nat | reported | strongly theta-dependent |

## K9 verdict

**The benign branch fires: `osc_theta [a . (xi_hat_theta - xi_hat_ref)] =
5.4e-4` nat means the GW-side field shift under theta is negligible —
PR-6b (theta coupling) is a no-op on the GW channel, and PR-6a
(frozen-anchor ensemble) is the honest deliverable.** The plan's own
instruction for this branch: report the bound and say so. The quoted
systematic for the paper: the anchor-frozen field differs from the
theta-consistent field by < 1e-3 nat of GW-likelihood content across the
selection/budget prior.

Consequences downstream:
- PR-6b is demoted from deliverable to the P18 inertness flag (the
  `dtheta = 0` branch ships anyway since it is the same code path, but no
  claim rests on it); the ladder's deliverable is **PR-6a**.
- P7e (60 nats) is the dN/dz-shape information the conditioned channel
  retains about `(delta, theta_sel)` against 22.8M galaxies. Per OWNER
  DECISION 13(a) it never enters the headline posterior; per guard 5, if a
  future rung admits the galaxy evidence, the coupling set must be
  restricted to `(Om0, w0, wa)` or the `delta` prior widened — and P7d
  (1.47 nat) says such a rung must also carry the log-det linear-response
  term.
- P7b = 6.3e-3 means linear response is available and accurate if a
  future dataset (deeper catalog, more events) moves P7c above the gate.

Combined with PR-0's `sigma_H = 6.0e-4` (member spread): on THIS
catalog/event pairing, both the theta-dependence and the posterior
uncertainty of the completion field are negligible in the GW likelihood;
what the field contributes is its MAP/mean placement (PR-0's 10.85-nat
Q-on/Q-off oscillation). The ensemble and coupling machinery are
correctness insurance, cheap by PR-0's cost table, not sources of signal.

Also shipped: `scripts/dndz_ppd_check.py` — the standalone dN/dz
posterior-predictive budget check (selection-channel-followups' queued
gate), OD13(a)-compliant (diagnostic only).
