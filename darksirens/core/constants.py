"""Central constants shared across darksirens modules."""

#: Planck-2015 fiducials, pinned to the exact ``float`` values Astropy's
#: ``Planck15`` carries (``float(Planck15.H0.value) == 67.74`` and
#: ``float(Planck15.Om0) == 0.3075`` are repr-exact, so this is bit-identical).
#: They are literals rather than an ``astropy.cosmology`` import because this
#: module sits at the bottom of nearly every import chain in the package --
#: including the argparse-only CLI path -- and ``astropy.cosmology`` drags
#: ``astropy.modeling`` -> ``nddata`` -> ``dask`` behind it for ~1 s per
#: process.  ``tests/test_cli_light_import.py`` pins the values against
#: Astropy itself.
H0_FID = 67.74
OM0_FID = 0.3075
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
    # Parametric selection-function block (active only under
    # c_mode="selection").  M0hat is the h-SCALED absolute-magnitude center
    # M0 - 5 log10 h -- the convention that makes C_sel exactly H0-invariant
    # (darksirens/redshift/selection.py); m_lim is a truncation DATUM of the
    # selection protocol, never sampled (only m_lim - M0hat is identified).
    "m_lim": 24.0,
    "M0hat": -20.2,
    "sigma_M": 1.0,
    # Schechter block of the same selection channel (active only under
    # c_mode="selection" + selection_family="schechter").  Mstar_hat is
    # h-SCALED like M0hat; alpha is the faint-end slope (kept above -2, the
    # edge of the one recurrence step the curve's upper-gamma ratio takes).
    # M_faint_offset is a PROTOCOL constant, never sampled: it must equal
    # darksirens.redshift.selection.M_FAINT_OFFSET_DEFAULT (pinned by a test).
    "Mstar_hat": -20.5,
    "alpha": -0.7,
    "M_faint_offset": 5.0,
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
#: ``selection`` (2) is the PARAMETRIC magnitude-limited completeness
#: ``C_sel(z) = Phi((m_lim - M0 - DM(z)) / sigma_M)`` evaluated from sampled
#: h-scaled LF parameters -- no counts enter the budget at all (the
#: clustering-safe channel; see darksirens/redshift/selection.py).
C_MODES = {"per_pixel": 0, "aggregate": 1, "selection": 2}

#: Luminosity-function families of the ``c_mode="selection"`` curve, in CLI
#: ``choices=`` order.  Lightweight spelling of
#: ``darksirens.redshift.selection.SELECTION_FAMILIES`` (which owns the
#: per-family theta tables) for callers that must not import the JAX module;
#: a test pins the two agree.
SELECTION_FAMILIES = ("gaussian", "schechter")

#: Single source of truth for the selection guard's default variance budget
#: (the GWTC-4.0/5.0 threshold): argparse defaults and getattr fallbacks all
#: reference this, so CLI and programmatic callers cannot silently diverge.
#: Lives here rather than in ``darksirens.likelihood.selection`` (which
#: re-exports it) so JAX-free tooling -- ``darksirens.io.results`` and anything
#: that only reads/writes HDF5 -- can use it without importing JAX.
DEFAULT_MAX_LIKELIHOOD_VARIANCE = 1.0

#: Version tags of the population fiducial vector a fixed-population run pins,
#: stamped into settings so a run records WHICH vector it fixed at.  Same reason
#: as above: they are the ONLY thing the CLIs' ``--population_fiducials`` option
#: needs, and ``darksirens.gw.populations.registry`` -- which owns the vectors
#: themselves and re-exports these names (as does the ``populations`` package,
#: the canonical spelling for every consumer) -- costs ~4 s of JAX/model imports
#: to reach.
FIDUCIAL_SET_LEGACY = "legacy"
FIDUCIAL_SET_IN_PRIOR = "in_prior_v2"
FIDUCIAL_SETS = (FIDUCIAL_SET_LEGACY, FIDUCIAL_SET_IN_PRIOR)

#: SIS time-delay scaling constant in seconds -- the ``--sl_T0_sec`` default.
#: ``darksirens.lensing.slmarks`` owns the physics (and re-exports this name,
#: which stays the canonical spelling); the value lives here because the lensing
#: CLI's argparse default is otherwise the ONLY reason it imports the
#: ``darksirens.lensing`` package, whose eager ``__init__`` costs ~2.6 s.
DEFAULT_T0_SECONDS = 5.36e6

#: ``max(sigma_delta_t)/T0`` below which the SIS time mark resolves to the
#: delta-collapse implementation.  Same reason as :data:`DEFAULT_T0_SECONDS`:
#: the lensing CLI mirrors it into a module constant its tests read directly,
#: and ``darksirens.lensing.marginal_diagnostics`` (which re-exports it and
#: remains the canonical spelling) costs the whole lensing import tree.
SIS_TIME_MARK_SHARPNESS = 0.02

#: Mock pair-tag selection kinds -- the ``--pair_tag_model`` choices.  Same
#: reason as :data:`DEFAULT_T0_SECONDS`; ``darksirens.lensing.pair_tag_selection``
#: re-exports it and remains the canonical spelling.
PAIR_TAG_SELECTION_MODEL_KINDS = (
    "constant",
    "snr_only",
    "snr_sky",
    "snr_time",
    "snr_time_sky",
    "file",
)
