"""Central constants shared across darksirens modules."""

from astropy.cosmology import Planck15

H0_FID = float(Planck15.H0.value)
OM0_FID = float(Planck15.Om0)
W0_FID = -1.0
WA_FID = 0.0
#: Prior label -> fiducial value for every :class:`~darksirens.core.types.SurveyParams`
#: field the parameter machinery addresses by name, in SurveyParams field order.
#: This is the single source of the survey fiducials: the decoder fills any field
#: that is not sampled from here, and the CLI's block-fixed parameter table prints
#: it verbatim.  Which of these labels are *sampleable* (and under which universe
#: model) is declared separately, in ``darksirens.inference.prior``.
#:
#: ``log10n0`` is log10 of the comoving density (SurveyParams carries ``n0``).
#: ``z50``/``w`` are generative-truth fields of the mock generator only, and
#: ``alpha_miss`` is pinned at 1 because it enters solely through the exact
#: product ``alpha_miss * b_miss``; none of the three is ever sampled.
SURVEY_PARAMS_FID_BY_NAME = {
    "log10n0": -2.0,
    "z50": 1.0,
    "w": 0.5,
    "delta": 0.0,
    "b_miss": 1.0,
    "alpha_miss": 1.0,
    "sigma_kde": 0.0,
}

#: Legacy positional spelling of :data:`SURVEY_PARAMS_FID_BY_NAME` (same order).
SURVEY_PARAMS_FID = tuple(SURVEY_PARAMS_FID_BY_NAME.values())

COMPLETE_EMPTY_PIXEL_POLICIES = {"zero": 0, "volume": 1}

#: Completeness estimator mode stored on ``SurveyParams.c_mode`` as an int enum
#: (string -> code, decoded eagerly pre-jit like the empty-pixel policy).
#: ``per_pixel`` (0, legacy default) is the per-pixel matched-kernel ratio;
#: ``aggregate`` (1) is ONE sky-aggregate curve
#: ``Cbar(z) = clip(Sum_p dN_obs_s(z|p) / (N_pix_total dN_exp_smooth(z)), 0, 1)``
#: so the angular clustering of the observed galaxies stays out of the budget
#: (see :mod:`darksirens.redshift.completion`).
C_MODES = {"per_pixel": 0, "aggregate": 1}
