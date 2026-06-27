import numpy as np
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.sky import sky_model_prior_parser
from darksirens.marks import mark_model_prior_parser
from darksirens.utils.cosmology import (
    Om0PriorLower,
    Om0PriorUpper,
    w0PriorLower,
    w0PriorUpper,
    waPriorLower,
    waPriorUpper,
)

#: Survey parameters that actually enter each universe model's likelihood.
#: Parameters outside a model's set are not sampled (they would be flat nuisance
#: dimensions); the decoder fills them from fiducial defaults.  Models absent
#: from this map sample the full survey block.
#:
#: ``dark_sirens_complete`` assumes a 100%-complete catalog, so the completion /
#: missing-galaxy parameters never enter its prior — only ``sigma_kde`` does.
#:
#: ``dark_sirens`` uses the ratio-only completeness estimator: there is no
#: parametric roll-off, so ``z50`` and ``w`` do not enter the likelihood.
#: ``alpha_miss`` and ``b_miss`` enter only through the exact product
#: ``alpha_miss * b_miss`` (a perfect degeneracy), so only ``b_miss`` is
#: sampled and ``alpha_miss`` stays at its fiducial of 1.
_ACTIVE_SURVEY_PARAMS = {
    "dark_sirens": ("log10n0", "delta", "b_miss", "sigma_kde"),
    "dark_sirens_complete": ("sigma_kde",),
    "spectral_sirens": (),
    "bright_sirens": (),
}


def apply_block_prior_overrides(block_name, labels, lower, upper, overrides):
    """Apply flat per-parameter prior overrides to a parameter block.

    Supported format:
        {"param_name": [low, high], ...}
    """
    if overrides is None:
        return list(lower), list(upper)

    if not isinstance(overrides, dict):
        raise TypeError(
            f"Prior overrides for block '{block_name}' must be a dict, got {type(overrides).__name__}."
        )

    lower_out = list(lower)
    upper_out = list(upper)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for label, bounds in overrides.items():
        if label not in label_to_index:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Override for '{label}' in block '{block_name}' must be [lower, upper]."
            )
        idx = label_to_index[label]
        lower_out[idx] = bounds[0]
        upper_out[idx] = bounds[1]

    return lower_out, upper_out


def validate_fixed_parameter_overrides(all_bounds, prior_overrides, fixed_parameter_values):
    """Validate and annotate labels that are both fixed and prior-overridden."""
    statuses = {}
    for label in fixed_parameter_values.keys() & prior_overrides.keys():
        lower, upper = all_bounds[label]
        fixed_value = float(fixed_parameter_values[label])
        if fixed_value < lower or fixed_value > upper:
            raise ValueError(
                f"Fixed value for '{label}' ({fixed_value}) is outside the "
                f"overridden prior bounds [{lower}, {upper}]."
            )
        statuses[label] = "fixed; override ignored"
    return statuses


def filter_fixed_parameters(labels, lower, upper, fixed_values):
    """Remove individually fixed labels from a sampled parameter block."""
    fixed_values = fixed_values or {}
    sampled = [
        (label, lo, hi)
        for label, lo, hi in zip(labels, lower, upper)
        if label not in fixed_values
    ]
    if not sampled:
        return [], [], []
    sampled_labels, sampled_lower, sampled_upper = zip(*sampled)
    return list(sampled_labels), list(sampled_lower), list(sampled_upper)


def resolve_parameter_values(sampled_coordinates, sampled_labels, fixed_parameter_values=None):
    """Map sampled coordinates plus fixed values to a label -> value dictionary.

    ``fixed_parameter_values`` consistently means a parameter is absent from the
    sampled coordinate vector.  This helper is the shared inverse operation: it
    validates that the coordinate length matches ``sampled_labels`` and then
    merges those sampled coordinates with the fixed-value map.
    """
    fixed_parameter_values = fixed_parameter_values or {}
    if len(sampled_coordinates) != len(sampled_labels):
        raise ValueError(
            f"Coordinate mismatch: expected {len(sampled_labels)} sampled "
            f"coordinates, got {len(sampled_coordinates)}."
        )

    values = {
        label: sampled_coordinates[idx]
        for idx, label in enumerate(sampled_labels)
    }
    values.update({label: float(value) for label, value in fixed_parameter_values.items()})
    return values


def build_parameter_space(
    pop_model,
    fix_population,
    fix_cosmology,
    fix_survey,
    prior_overrides=None,
    fixed_parameter_values=None,
    fix_de=False,
    universe_model=None,
    shared_beta=True,
    shared_spin=True,
    shared_gamma=True,
    sky_model="isotropic",
    mark_model="none",
    mark_names=(),
):
    """Construct labels and prior bounds for cosmological, population, survey, and sky parameters.

    Parameters
    ----------
    fix_cosmology
        When true, remove the full cosmology block (``H0``, ``Om0``, ``w0``,
        and ``wa``) from sampling.
    fix_de
        When true, remove only the dark-energy cosmology labels (``w0`` and
        ``wa``) from sampling. ``fix_cosmology`` supersedes this flag.
    """
    if prior_overrides is None:
        prior_overrides = {}
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    # --- Cosmology ---
    cosmo_labels = ["H0", "Om0", "w0", "wa"]
    cosmo_lower = [20.0, Om0PriorLower, w0PriorLower, waPriorLower]
    cosmo_upper = [120.0, Om0PriorUpper, w0PriorUpper, waPriorUpper]

    # --- Population ---
    pop_lower, pop_upper, pop_labels, pop_kinds, model_name = pop_model_prior_parser(
        pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )

    # --- Survey ---
    # ``log10n0`` is log10 of the comoving galaxy density in Mpc^-3,
    # matching dV_of_z [Mpc^3 sr^-1 dz^-1] times the HEALPix pixel area.
    # The redshift grid used by the completion model spans 0 <= z <= 5;
    # these defaults keep the survey rolloff inside that domain while avoiding
    # the formerly ultra-broad density/evolution fits that could force heavy
    # clipping throughout the completion grid.
    survey_labels = ["log10n0", "z50", "w", "delta", "b_miss", "alpha_miss", "sigma_kde"]
    survey_lower = [-4.0, 0.05, 0.02, -3.0, 0.0, 0.0, 0.0]
    survey_upper = [-1.0, 4.5, 1.5, 3.0, 3.0, 1.0, 0.05]

    # --- Sky (angular source distribution) ---
    # Appended after the survey block; ``isotropic`` contributes no parameters.
    sky_lower, sky_upper, sky_labels, sky_kinds, _sky_latex = sky_model_prior_parser(sky_model)
    sky_lower, sky_upper = list(sky_lower), list(sky_upper)

    # --- Marks (BBH-host efficiency eta block) ---
    # Appended after the sky block; ``none`` contributes no parameters.  The
    # eta coefficients are one per available mark (``mark_names``).
    mark_lower, mark_upper, mark_labels, mark_kinds, _mark_latex = mark_model_prior_parser(
        mark_model, mark_names
    )
    mark_lower, mark_upper = list(mark_lower), list(mark_upper)

    # Make sure all prior override keys are valid parameter labels
    known_labels = (
        set(cosmo_labels) | set(pop_labels) | set(survey_labels)
        | set(sky_labels) | set(mark_labels)
    )
    unknown = [k for k in prior_overrides.keys() if k not in known_labels]
    if unknown:
        raise KeyError(
            f"Unknown prior override labels: {unknown}. Valid labels for pop_model='{pop_model}': "
            f"{sorted(known_labels)}"
        )

    unknown_fixed = [k for k in fixed_parameter_values.keys() if k not in known_labels]
    if unknown_fixed:
        raise KeyError(
            f"Unknown fixed parameter labels: {unknown_fixed}. Valid labels for pop_model='{pop_model}': "
            f"{sorted(known_labels)}"
        )

    # Apply block overrides
    cosmo_lower, cosmo_upper = apply_block_prior_overrides(
        "cosmology", cosmo_labels, cosmo_lower, cosmo_upper, prior_overrides
    )
    pop_lower, pop_upper = apply_block_prior_overrides(
        "population", pop_labels, pop_lower, pop_upper, prior_overrides
    )
    survey_lower, survey_upper = apply_block_prior_overrides(
        "survey", survey_labels, survey_lower, survey_upper, prior_overrides
    )
    sky_lower, sky_upper = apply_block_prior_overrides(
        "sky", sky_labels, sky_lower, sky_upper, prior_overrides
    )

    all_bounds = {
        label: (float(lo), float(hi))
        for label, lo, hi in (
            list(zip(cosmo_labels, cosmo_lower, cosmo_upper))
            + list(zip(pop_labels, pop_lower, pop_upper))
            + list(zip(survey_labels, survey_lower, survey_upper))
            + list(zip(sky_labels, sky_lower, sky_upper))
        )
    }
    fixed_parameter_statuses = validate_fixed_parameter_overrides(
        all_bounds, prior_overrides, fixed_parameter_values
    )

    sampled_cosmo_labels, sampled_cosmo_lower, sampled_cosmo_upper = filter_fixed_parameters(
        cosmo_labels, cosmo_lower, cosmo_upper, fixed_parameter_values
    )
    if fix_de and not fix_cosmology:
        dark_energy_labels = {"w0", "wa"}
        sampled_cosmo = [
            (label, lo, hi)
            for label, lo, hi in zip(
                sampled_cosmo_labels, sampled_cosmo_lower, sampled_cosmo_upper
            )
            if label not in dark_energy_labels
        ]
        if sampled_cosmo:
            sampled_cosmo_labels, sampled_cosmo_lower, sampled_cosmo_upper = map(
                list, zip(*sampled_cosmo)
            )
        else:
            sampled_cosmo_labels, sampled_cosmo_lower, sampled_cosmo_upper = [], [], []
    sampled_pop_labels, sampled_pop_lower, sampled_pop_upper = filter_fixed_parameters(
        pop_labels, pop_lower, pop_upper, fixed_parameter_values
    )
    sampled_survey_labels, sampled_survey_lower, sampled_survey_upper = filter_fixed_parameters(
        survey_labels, survey_lower, survey_upper, fixed_parameter_values
    )
    sampled_sky_labels, sampled_sky_lower, sampled_sky_upper = filter_fixed_parameters(
        sky_labels, sky_lower, sky_upper, fixed_parameter_values
    )
    sampled_mark_labels, sampled_mark_lower, sampled_mark_upper = filter_fixed_parameters(
        mark_labels, mark_lower, mark_upper, fixed_parameter_values
    )

    # Drop survey parameters that do not enter this universe model's likelihood
    # (e.g. completion parameters under the complete-catalog model). They stay
    # in ``survey_labels`` so the decoder still fills SurveyParams from fiducials.
    active_survey = _ACTIVE_SURVEY_PARAMS.get(universe_model)
    if active_survey is not None:
        kept = [
            (label, lo, hi)
            for label, lo, hi in zip(
                sampled_survey_labels, sampled_survey_lower, sampled_survey_upper
            )
            if label in active_survey
        ]
        if kept:
            sampled_survey_labels, sampled_survey_lower, sampled_survey_upper = map(
                list, zip(*kept)
            )
        else:
            sampled_survey_labels, sampled_survey_lower, sampled_survey_upper = [], [], []

    labels = []
    lower = []
    upper = []

    if not fix_cosmology:
        labels += sampled_cosmo_labels
        lower += sampled_cosmo_lower
        upper += sampled_cosmo_upper
        n_cosmo_eff = len(sampled_cosmo_labels)
    else:
        n_cosmo_eff = 0

    if not fix_population:
        labels += sampled_pop_labels
        lower += sampled_pop_lower
        upper += sampled_pop_upper
        n_pop_eff = len(sampled_pop_labels)
    else:
        n_pop_eff = 0

    if not fix_survey:
        labels += sampled_survey_labels
        lower += sampled_survey_lower
        upper += sampled_survey_upper
        n_survey_eff = len(sampled_survey_labels)
    else:
        n_survey_eff = 0

    # Sky block: appended last so existing cosmo/pop/survey indices are stable.
    # The chosen ``sky_model`` decides the parameters (none for ``isotropic``);
    # individually-fixed sky labels were already removed above.
    labels += sampled_sky_labels
    lower += sampled_sky_lower
    upper += sampled_sky_upper
    n_sky_eff = len(sampled_sky_labels)

    # Mark block: appended after sky so all earlier indices stay stable.
    labels += sampled_mark_labels
    lower += sampled_mark_lower
    upper += sampled_mark_upper
    n_mark_eff = len(sampled_mark_labels)

    # Per-parameter prior family aligned to the final ``labels`` ordering.
    # Cosmology and survey blocks are uniform; the population block carries the
    # kinds reported by the parser (keyed by label, matching the codebase's
    # global-label-uniqueness assumption used for bounds/overrides).  Fixed
    # parameters are filtered out automatically because they are absent from
    # ``labels``.
    kind_map = {lbl: ("uniform", None, None) for lbl in cosmo_labels + survey_labels}
    for lbl, knd in zip(pop_labels, pop_kinds):
        kind_map[lbl] = knd
    for lbl, knd in zip(sky_labels, sky_kinds):
        kind_map[lbl] = knd
    for lbl, knd in zip(mark_labels, mark_kinds):
        kind_map[lbl] = knd
    prior_kinds = [kind_map.get(lbl, ("uniform", None, None)) for lbl in labels]

    return (
        labels,
        np.array(lower),
        np.array(upper),
        n_pop_eff,
        pop_labels,
        survey_labels,
        cosmo_labels,
        n_cosmo_eff,
        n_survey_eff,
        model_name,
        fixed_parameter_statuses,
        prior_kinds,
        sky_labels,
        mark_labels,
    )

def make_prior_transform(lower, upper, prior_kinds=None):
    """Unit-cube -> parameter inverse-CDF transform, per-parameter prior-aware.

    ``prior_kinds`` is an optional list aligned to ``lower``/``upper`` of
    ``(kind, loc, scale)`` triples (kind in ``{"uniform", "normal",
    "lognormal"}``).  ``None`` reproduces the legacy all-uniform affine map.
    Used by the nested samplers (dynesty/tinyns); numpyro builds its own prior
    via ``run_sampler``.  ``low``/``high`` always act as truncation bounds, so
    every kind maps the cube to ``[low, high]`` and the measure matches the
    corresponding numpyro distribution.

    Implementation note: dynesty wraps this with ``np.asarray(transform(
    jnp.asarray(u)))``, i.e. the transform always receives a JAX array, so
    ``jax.scipy.special`` is safe across all nested backends.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if prior_kinds is None or all(k[0] == "uniform" for k in prior_kinds):
        def prior_transform(u):
            return u * (upper - lower) + lower
        return prior_transform

    import jax.numpy as jnp
    from jax.scipy.special import ndtr, ndtri

    kinds = [k[0] for k in prior_kinds]
    loc = jnp.asarray([0.0 if k[1] is None else float(k[1]) for k in prior_kinds])
    scale = jnp.asarray([1.0 if k[2] is None else float(k[2]) for k in prior_kinds])
    lo_j, hi_j = jnp.asarray(lower), jnp.asarray(upper)
    is_normal = jnp.asarray([k == "normal" for k in kinds])
    is_lognorm = jnp.asarray([k == "lognormal" for k in kinds])

    def _trunc_normal_ppf(u, a, b, mu, sg):
        # inverse CDF of N(mu, sg^2) truncated to [a, b]
        za = (a - mu) / sg
        zb = (b - mu) / sg
        Phi_a, Phi_b = ndtr(za), ndtr(zb)
        x = ndtri(jnp.clip(Phi_a + u * (Phi_b - Phi_a), 1e-12, 1.0 - 1e-12))
        return mu + sg * x

    def prior_transform(u):
        u = jnp.asarray(u)
        uniform = u * (hi_j - lo_j) + lo_j
        # normal: truncated to [lo, hi]
        normal = _trunc_normal_ppf(u, lo_j, hi_j, loc, scale)
        # lognormal: exp of a normal in log-space truncated to [log lo, log hi]
        log_lo = jnp.log(jnp.clip(lo_j, 1e-300, None))
        log_hi = jnp.log(jnp.clip(hi_j, 1e-300, None))
        lognormal = jnp.exp(_trunc_normal_ppf(u, log_lo, log_hi, loc, scale))
        out = jnp.where(is_normal, normal, uniform)
        out = jnp.where(is_lognorm, lognormal, out)
        return out

    return prior_transform
