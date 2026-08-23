# H0 scan: completeness methods on the clustered mock

End-to-end H0 likelihood scans of 300 DAG-consistent GW dark sirens whose
hosts are drawn from the clustered closure-mock catalog, comparing every
completeness configuration on the SAME events, injections, and grid.

## Construction (Essick & Fishbach DAG rules)

* Hosts drawn uniformly over the COMPLETE catalog rows (hosts ∝ galaxies —
  the count-odds weighting the assembled prior assumes); everything downstream
  reuses `scripts/mock_dark_sirens/generate_mock_data.py` unchanged: the cut
  acts on the OBSERVED SNR, the PE conditions on the same recorded noise
  realization, and the 2M selection injections share the statistic+threshold
  (`population+uniform` proposal, reweightable across trial H0).
* GW horizon placed inside the catalog depth (`snr_ref=10`): detected events
  have z in [0.07, 0.47], median 0.21.
* `DARKSIRENS_ZMAX=0.52`: the model universe ends where the generative one
  does (0.5 + photo-z spill to 0.509). Q tables rebuilt on this grid
  (`fits_z052_*`; 12 z-nodes to 0.5, 64 sphere nodes — resolution guard
  satisfied). Fixing this exposed and fixed a real package bug (branch commit
  4566bdc): under any non-default DARKSIRENS_ZMAX where jnp's and numpy's
  expm1∘log differ by one ulp, zgrid's endpoint sat above the numpy-built
  distance table → NaN chi(zMax) → all-NaN Q tables.
* Selection variance guard: with 300 events at nsamp=4096 the summed
  per-event MC variances (~1 nat) exceed the 1-nat GWTC-style budget with
  nothing left for selection; `max_likelihood_variance=25` for the scan
  (method separations are O(10²) nats; see scan_h0.py comment). Neff > 12k
  across the whole grid (threshold ~3.8k) — injection-coverage gate passed.

## Machinery closure (gates before reading results)

* A fully canonical generator mock through the same scan driver:
  peak H0 = 67.0 (truth 67.74), interval [66, 70]. PASS.
* Complete-catalog gate on OUR events (true z, all 277k galaxies):
  69.54 ± 1.36 → +1.3σ from truth. PASS (single realization).

## Results (fine grid, Gaussian fit to the peak; truth H0 = 67.74)

| config | per-pixel C | aggregate C̄ |
|---|---|---|
| homogeneous | 69.07 ± 1.38 (+1.0σ) | 67.87 ± 1.32 (+0.1σ) |
| legacy 1+bδ_g | 68.07 ± 1.35 (+0.2σ) | 66.80 ± 1.39 (−0.7σ) |
| Q radial | 69.11 ± 1.38 (+1.0σ) | 66.91 ± 1.37 (−0.6σ) |
| Q gp3d | 68.99 ± 1.37 (+0.9σ) | 67.00 ± 1.36 (−0.5σ) |

2-D (H0, log10 n0) grids: profiling over n0 moves no peak by more than
~1 unit (homog_pp 69→67, homog_agg/qgp3d_agg stay ~67) — the budget monopole
is absorbed by n0 as designed, and H0 is not degenerate with it here.

## Reading

* EVERY configuration recovers truth within ~1σ at n=300: at this event
  count and depth, H0 is statistics-dominated (σ ≈ 1.4) and robust to the
  completeness method — consistent with the "How Low Can You Go" literature.
* The per-pixel family (except δ_g) shares a common +0.9–1.0σ offset while
  the aggregate family clusters within ±0.7σ of truth; the families differ by
  ~1.2 units. On ONE realization this is suggestive, NOT a bias detection —
  certifying a method bias at the ~1 km/s/Mpc level needs coverage over many
  disjoint event realizations (DAG gate 7d) or ~10× the events.
* Within each family the three clustering-aware methods are nearly
  degenerate in H0; their differences live in the prior SHAPES (see the
  closure experiment), which matter more at higher event counts and for
  per-event posteriors than for this ensemble scan.

## Files

* `generate_events.py` — events + injections + complete-catalog gate file
* `scan_h0.py` — grid scans (resume-friendly; selection diagnostics)
* `plot_h0_scan.py` — comparison figures + Gaussian peak fits
* `debug_event_term.py` — per-event/selection dissection tool
* `results_fine/` — 0.25-unit grid, all 9 configs (headline numbers)
* `results_final/` — 1.0-unit grid + 2-D (H0, log10n0) profiles
* `canonical/`, `results_canonical/` — generator-native closure check
