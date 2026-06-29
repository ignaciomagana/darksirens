"""Central constants shared across darksirens modules."""

from astropy.cosmology import Planck15

H0_FID = float(Planck15.H0.value)
OM0_FID = float(Planck15.Om0)
W0_FID = -1.0
WA_FID = 0.0
# (log10n0->n0, z50, w, delta, b_miss, alpha_miss, sigma_kde).  z50/w are
# inactive under the ratio-only dark-siren completeness; alpha_miss defaults
# to 1 because it enters only through the exact product alpha_miss*b_miss.
SURVEY_PARAMS_FID = (-2.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0)

COMPLETE_EMPTY_PIXEL_POLICIES = {"zero": 0, "volume": 1}
