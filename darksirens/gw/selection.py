"""
selection.py
------------
Detection probability models for gravitational-wave sources.

Scope
-----
This module is for *injection-level* detection models: functions that
return the probability that a GW signal with given source parameters
would be detected by a network, given a specific noise realisation or
a population-averaged noise curve.

This is DISTINCT from the hierarchical selection integral in
``darksirens.likelihood.selection``, which computes

    μ = (1/N_draw) Σ_{det inj}  p_pop / p_draw

using a pre-existing injection set.  What belongs here instead:

  - Analytic SNR-threshold detection probability P_det(m1, m2, z, sky)
  - Sensitivity-curve-based P_det via matched-filter SNR grids
  - Semi-analytic approximations (e.g. Finn & Chernoff 1993 formalism)
  - Network sensitivity interpolators (O3 / O4 / ET / CE sensitivity curves)

These are useful when you want to replace the Monte Carlo selection
integral with a faster semi-analytic model, or when computing
selection-corrected rate estimates without a full injection campaign.

Status
------
Not yet implemented.  Populate this module before moving away from the
injection-based selection integral in ``darksirens.likelihood.selection``.

References
----------
- Finn & Chernoff (1993). PhysRevD 47, 2198
- Chen et al. (2021). arXiv:1612.00036
- Vitale et al. (2022). arXiv:2007.05579, Appendix B
"""