# PR-4 — the anchor artifact (2026-08-13)

`darksirens/cli/build_latent_field.py` builds the first guard-compliant
gp3d-family latent artifact from the real DESI union:

- **41 s per build** on one H100 (the plan's "minutes" gate; rev-1's 24 h
  estimate was withdrawn and stays withdrawn — the solve is ~13 Fisher trips
  over a 2520 x 2520 system, and the moment tables dominate the wall).
- 64 MB artifact: `xi_hat`, `H_chol` (2520^2 f64), `sensitivity_S`
  (5 columns: M0hat, sigma_M, delta, Om0, b_gal — the stacked-K interface
  shaped from day one), 8 antithetic members, `row_fac`
  (8 x 30470 x 8 f32 — the seam's static leaf), `(A, B)` moment tables on a
  33-node Chebyshev–Lobatto b_GW grid over the consumption-grid nodes
  `z <= 0.30` (+ projected `(dA, dB)` theta-derivatives for the P18 flag),
  the frozen `W`, counts, `f_p`, `theta_ref` as a first-class attr, and the
  content sha256 (guard-1 fingerprint: kernel, jitter convention, catalog
  content hashes, theta_ref, b_gal).
- Hard guards at build time: resolution (both sides), occupancy guard 7
  (all 12 shells pass on the real catalog), P6 convergence (grad_inf
  1.1e-10 <= 1e-8 or refuse).
- **Reproducibility gate**: two same-seed builds are bit-identical in every
  array (verified dataset-by-dataset); the sha256 covers content +
  configuration (the output path is excluded — first run caught exactly
  that, the arrays were already identical).

Pins P9 (rho from the `(A, B)` tables, exact at nodes, 1e-6 under
barycentric b-interpolation over 200 random `(b, c)`) and P10 (Cholesky
log-det identity, 1e-8) in `tests/test_latent_anchor.py`.

With K9's benign branch (PR-3), the artifact's `sensitivity_S` and
`(dA, dB)` blocks are carried for the P18 inertness flag and future-data
insurance, not for the deliverable path: PR-6a consumes `row_fac`,
`(A, B)`, `P_F/F_F`, and the members.
