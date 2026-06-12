import numpy as np
from darksirens.gw.populations import pop_model_prior_parser
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
):
    """Construct labels and prior bounds for cosmological, population, and survey parameters.

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
    pop_lower, pop_upper, pop_labels, model_name = pop_model_prior_parser(pop_model)

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

    # Make sure all prior override keys are valid parameter labels
    known_labels = set(cosmo_labels) | set(pop_labels) | set(survey_labels)
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

    all_bounds = {
        label: (float(lo), float(hi))
        for label, lo, hi in (
            list(zip(cosmo_labels, cosmo_lower, cosmo_upper))
            + list(zip(pop_labels, pop_lower, pop_upper))
            + list(zip(survey_labels, survey_lower, survey_upper))
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
    )

def make_prior_transform(lower, upper):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    def prior_transform(u):
        return u * (upper - lower) + lower
    return prior_transform
