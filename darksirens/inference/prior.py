import re
import warnings
from collections.abc import Sequence
from typing import Callable, NamedTuple

import numpy as np
from darksirens.core.constants import SURVEY_PARAMS_FID_BY_NAME
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

# ---------------------------------------------------------------------------
# Survey-parameter registry
# ---------------------------------------------------------------------------
# ONE declarative source of truth for the sampled survey block: an ordered
# tuple of (label, bounds, activity rule).  ``build_parameter_space`` derives
# the base block AND every per-catalog ``_c{k}`` block from it alone, and the
# same rules produce the error messages for inert user config -- so a guard can
# never disagree with the gating it is guarding (the failure mode of the old
# build-the-full-block-then-filter design, which spawned four CLI-vs-decoder
# drift bugs and one wrong-reason error message).
#
# The block is the SAMPLEABLE labels only.  ``z50``/``w``/``alpha_miss`` are
# SurveyParams fields (see core/constants.SURVEY_PARAMS_FID_BY_NAME) that no
# universe model ever samples; they live in ``_PINNED_SURVEY_PARAMS`` below and
# never reach the prior at all.


class _Inert(NamedTuple):
    """Why a survey label is not sampled for the resolved configuration.

    ``reason`` is a sentence fragment naming the true cause and ``remedy`` an
    optional second way out; both are spliced into the guard messages verbatim.
    ``fatal_when_fixed`` marks the gates for which even a
    ``--fixed_parameter_values`` entry is an error rather than a warning: a
    fixed value is normally harmless (the decoder pins the field at it), but a
    b_miss pinned against an active Q_LSS table or a dummy delta_g asserts a
    modulation the likelihood provably does not apply.
    """

    reason: str
    fatal_when_fixed: bool = False
    remedy: str = ""
    #: The fixed-value path is the ENDORSED way to set this label (e.g. the
    #: m_lim truncation datum), so a fixed value draws no warning -- the CLI
    #: itself inserts one under --selection_fit.
    fixed_ok: bool = False


class _SurveyParam(NamedTuple):
    """One sampleable survey label: its prior bounds and its activity rule.

    ``inactive_reason(universe_model, use_lss, q_active, catalog)`` returns
    ``None`` when the label IS sampled for that configuration, else the
    :class:`_Inert` record explaining why it is not.  ``catalog`` is the 1-based
    catalog number the label belongs to (1 for the unsuffixed block).
    """

    label: str
    lower: float
    upper: float
    inactive_reason: Callable[[str | None, bool, bool, int], "_Inert | None"]


#: Universe models the registry knows about.  Anything else (including ``None``)
#: is unknown to the registry and samples the WHOLE block: the historical
#: fallback for direct/library callers that never name a universe model.
_REGISTERED_UNIVERSE_MODELS = frozenset({
    "dark_sirens",
    "dark_sirens_complete",
    "spectral_sirens",
    "bright_sirens",
    "spectral_sirens_wl",
})

#: Models with no galaxy catalog in the likelihood: they sample NO survey
#: parameter.  Omitting ``spectral_sirens_wl`` from the old allow-list made the
#: block filter skip entirely, so it sampled SEVEN phantom flat nuisance
#: dimensions (12 -> 19), slowing sampling and offsetting logZ by the extra
#: prior volumes -- invalidating Bayes-factor comparisons against
#: ``spectral_sirens`` (library review, CLI finding 1).
_CATALOG_FREE_MODELS = frozenset({
    "spectral_sirens", "bright_sirens", "spectral_sirens_wl",
})


def _catalog_free(universe_model):
    return _Inert(
        f"universe_model '{universe_model}' is catalog-free, so it samples no "
        f"survey parameters at all (no galaxy catalog enters its likelihood; a "
        f"survey label would be a phantom flat dimension that inflates the "
        f"prior volume and offsets logZ)",
        remedy="or run a universe_model that uses a galaxy catalog",
    )


_COMPLETE_CATALOG = _Inert(
    "universe_model 'dark_sirens_complete' assumes a 100%-complete catalog, so "
    "the completion / missing-galaxy parameters never enter its likelihood "
    "(only sigma_kde does)",
    remedy="or run universe_model 'dark_sirens', which models the missing galaxies",
)


def _completion_param_rule(universe_model, use_lss, q_active, catalog,
                           c_mode="per_pixel"):
    """Activity of the completion parameters (``log10n0``, ``delta``)."""
    if universe_model in _CATALOG_FREE_MODELS:
        return _catalog_free(universe_model)
    if universe_model == "dark_sirens_complete":
        return _COMPLETE_CATALOG
    return None


def _selection_rule(universe_model, use_lss, q_active, catalog,
                    c_mode="per_pixel"):
    """Activity of the parametric-selection parameters (``M0hat``, ``sigma_M``).

    They enter the likelihood ONLY through the parametric completeness curve
    ``C_sel(z) = Phi((m_lim - M0 - DM(z))/sigma_M)`` that
    ``c_mode="selection"`` forms in place of the counts-based estimators
    (redshift/completion.py), so under any other c_mode they are provably
    inert -- and a fixed value there asserts a selection model the likelihood
    does not apply.
    """
    model_rule = _completion_param_rule(
        universe_model, use_lss, q_active, catalog, c_mode=c_mode)
    if model_rule is not None:
        return model_rule
    if c_mode != "selection":
        return _Inert(
            f"catalog {catalog} runs c_mode='{c_mode}', whose completeness "
            f"estimator is counts-based and never reads the parametric "
            f"selection parameters (M0hat, sigma_M); they enter only the "
            f"C_sel(z) curve that c_mode='selection' forms",
            fatal_when_fixed=True,
            remedy="or run with --c_mode selection",
        )
    return None


def _b_miss_rule(universe_model, use_lss, q_active, catalog,
                 c_mode="per_pixel"):
    """Activity of ``b_miss``: the completion rule plus the two LSS gates.

    ``b_miss`` enters the likelihood ONLY through the local-overdensity factor
    ``max(1 + alpha_miss*b_miss*delta_g, 0)`` (redshift/completion.py), so it is
    additionally inert whenever that factor is not a function of it.
    """
    model_rule = _completion_param_rule(
        universe_model, use_lss, q_active, catalog, c_mode=c_mode)
    if model_rule is not None:
        return model_rule
    if q_active:
        # A Q_LSS table REPLACES the overdensity factor for this catalog.
        return _Inert(
            f"catalog {catalog} has an active Q_LSS completion table, which "
            f"replaces the local (1 + alpha_miss*b_miss*delta_g) factor for "
            f"that catalog, so b_miss does not enter its likelihood",
            fatal_when_fixed=True,
            remedy=f"or drop catalog {catalog}'s --lss_completion table",
        )
    if not use_lss:
        # delta_g is the all-zero (1, N_grid) dummy built by inference/loaders.py,
        # so the factor is identically 1 for ANY b_eff.  Missing this case made
        # the CLI-default dark_sirens run sample b_miss with exactly zero effect.
        return _Inert(
            "b_miss is inert with --use_lss off: delta_g is the all-zero "
            "(1, N_grid) dummy built by inference/loaders.py, so the "
            "(1 + alpha_miss*b_miss*delta_g) factor is identically 1 for any "
            "b_miss",
            fatal_when_fixed=True,
            remedy="or pass --use_lss true so delta_g carries a real overdensity field",
        )
    return None


def _kde_rule(universe_model, use_lss, q_active, catalog,
              c_mode="per_pixel"):
    """Activity of ``sigma_kde``: every catalog-based model broadens its kernels."""
    if universe_model in _CATALOG_FREE_MODELS:
        return _catalog_free(universe_model)
    return None


#: THE registry.  Order IS the sampled-block order.  ``log10n0`` is log10 of the
#: comoving galaxy density in Mpc^-3, matching dV_of_z [Mpc^3 sr^-1 dz^-1] times
#: the HEALPix pixel area.  The completion model's redshift grid spans
#: 0 <= z <= 5; these defaults keep the survey rolloff inside that domain while
#: avoiding the formerly ultra-broad density/evolution fits that could force
#: heavy clipping throughout the completion grid.
_SURVEY_BLOCK = (
    _SurveyParam("log10n0", -4.0, -1.0, _completion_param_rule),
    _SurveyParam("delta", -3.0, 3.0, _completion_param_rule),
    _SurveyParam("b_miss", 0.0, 3.0, _b_miss_rule),
    _SurveyParam("sigma_kde", 0.0, 0.05, _kde_rule),
    # Parametric-selection block, appended LAST so pre-existing coordinate
    # indices never move.  Bounds are truncation only: the information comes
    # from the magnitude-fit Gaussian prior (--selection_fit), which flips
    # these labels' prior kind to a truncated normal at build time; without a
    # fit they sample flat within bounds (the deliberate wide-open ablation).
    # sigma_M's floor keeps Phi from degenerating to a step (PPF/gradients
    # misbehave at sigma -> 0).  M0hat is h-SCALED (M0 - 5 log10 h).
    _SurveyParam("M0hat", -23.0, -18.0, _selection_rule),
    _SurveyParam("sigma_M", 0.05, 3.0, _selection_rule),
)

#: SurveyParams fields that are recognised as parameter labels but are NEVER
#: sampled by any universe model -- they carry no prior, no bounds and no block
#: slot.  Recognised (rather than simply unknown) so that naming one in
#: ``--prior_overrides`` / ``--fixed_parameter_values`` gets the true reason
#: instead of a bare "unknown label" or, worse, silence.
_PINNED_SURVEY_PARAMS = {
    "z50": _Inert(
        "'z50' is a generative-truth field of the mock generator "
        "(scripts/mock_dark_sirens/generate_mock_data.py), not a likelihood "
        "parameter: the dark-siren completeness is the data-driven kernel "
        "ratio, so no likelihood reads SurveyParams.z50 and it is never "
        "sampled for any universe model"
    ),
    "w": _Inert(
        "'w' is a generative-truth field of the mock generator "
        "(scripts/mock_dark_sirens/generate_mock_data.py), not a likelihood "
        "parameter: the dark-siren completeness is the data-driven kernel "
        "ratio, so no likelihood reads SurveyParams.w and it is never sampled "
        "for any universe model"
    ),
    "alpha_miss": _Inert(
        "'alpha_miss' enters only through the exact product alpha_miss*b_miss "
        "(a perfect degeneracy), so it is pinned at its fiducial of 1 and "
        "b_miss alone carries the modulation; it is never sampled for any "
        "universe model"
    ),
    "m_lim": _Inert(
        "'m_lim' is the truncation DATUM of the selection protocol, not an "
        "uncertain parameter: inside C_sel(z) = "
        "Phi((m_lim - M0hat - 5log10(h) - DM(z))/sigma_M) only the "
        "combination m_lim - M0hat is identified, so sampling both would be "
        "an exact flat direction; set it per run via "
        "--fixed_parameter_values (the decoder pins the field) and let "
        "M0hat carry the sampled freedom",
        fixed_ok=True,
    ),
}

#: Every survey label the parameter machinery recognises, sampleable or pinned.
_SURVEY_PARAM_LABELS = tuple(p.label for p in _SURVEY_BLOCK) + tuple(
    _PINNED_SURVEY_PARAMS
)

# The registry and the SurveyParams fiducial table must name exactly the same
# fields: the builder decides what is sampled, the decoder fills the rest from
# the fiducials, and a name present in only one of them is precisely the drift
# this registry exists to prevent.
if set(_SURVEY_PARAM_LABELS) != set(SURVEY_PARAMS_FID_BY_NAME):
    raise RuntimeError(
        "Survey registry / fiducial table mismatch: "
        f"{sorted(set(_SURVEY_PARAM_LABELS) ^ set(SURVEY_PARAMS_FID_BY_NAME))} "
        "appears in only one of darksirens.inference.prior._SURVEY_BLOCK + "
        "_PINNED_SURVEY_PARAMS and core.constants.SURVEY_PARAMS_FID_BY_NAME."
    )


def _survey_param_inactive_reason(spec, universe_model, use_lss, q_active,
                                  catalog, c_mode="per_pixel"):
    """``spec``'s :class:`_Inert` for this configuration, or ``None`` if sampled.

    A universe model the registry does not know about samples the whole
    LEGACY block (bounds and order as declared, no gating) -- the historical
    fallback for callers that never name a universe model.  The parametric
    selection parameters are the exception: they are gated on ``c_mode``
    regardless of the universe model (an unknown model without
    c_mode="selection" would otherwise sample two phantom flat dimensions --
    exactly the prior-volume inflation this registry exists to prevent).
    """
    if universe_model not in _REGISTERED_UNIVERSE_MODELS:
        if spec.inactive_reason is _selection_rule:
            return spec.inactive_reason(None, use_lss, q_active, catalog,
                                        c_mode=c_mode)
        return None
    return spec.inactive_reason(universe_model, use_lss, q_active, catalog,
                                c_mode=c_mode)


#: Matches the per-catalog ``_c{k}`` suffix appended to survey labels for the
#: K-catalog mixture (e.g. ``log10n0_c2``).  ``k`` is any positive integer.
_SURVEY_CATALOG_SUFFIX = re.compile(r"_c\d+$")


def _survey_base_name(label: str) -> str:
    """Strip a trailing ``_c{k}`` catalog suffix from a survey label.

    Used so per-catalog suffixed survey labels (``log10n0_c2`` etc.) resolve
    against the registry exactly like their base labels.  For a base label (no
    suffix) this is the identity, so the single-catalog path is unchanged.
    """
    return _SURVEY_CATALOG_SUFFIX.sub("", label)


def _survey_catalog_number(label: str) -> int:
    """1-based catalog number encoded in a survey label's ``_c{k}`` suffix."""
    match = _SURVEY_CATALOG_SUFFIX.search(label)
    return 1 if match is None else int(match.group(0)[2:])


def _validate_cosmology_within_interpolation_grid(
    labels, lower, upper, fixed_parameter_values=None
):
    """Refuse cosmology bounds the tabulated distance grid cannot represent.

    ``r_of_z`` interpolates a precomputed ``(Om0, w0, wa, z)`` table and returns
    NaN outside it rather than extrapolating; the likelihood turns that into
    ``logL = -inf``.  A prior bound outside the grid is therefore a SILENT prior
    truncation, not an error: the sampler explores the intersection of the
    requested prior with the grid while the run's settings.json and startup
    table report the range the user asked for.  Two concrete consequences --
    ``logZ`` is normalised against the wrong prior volume (so Bayes factors
    against a differently-prior'd run are offset), and the marginal shows a hard
    wall at the grid edge that reads as a data constraint.

    Both the sampled bounds and any ``--fixed_parameter_values`` entry are
    checked; the fixed case was previously unreachable by any validation
    (``validate_fixed_parameter_overrides`` only inspects labels that are BOTH
    fixed and overridden).  ``H0`` is exempt: it enters as an analytic rescaling
    of the table, not as an interpolation axis.
    """
    from darksirens.utils.cosmology import Om0grid, w0grid, wagrid

    grids = {"Om0": Om0grid, "w0": w0grid, "wa": wagrid}
    fixed = {str(k): float(v) for k, v in (fixed_parameter_values or {}).items()}
    problems = []

    for label, lo, hi in zip(labels, lower, upper):
        grid = grids.get(label)
        if grid is None:
            continue
        g_lo, g_hi = float(grid[0]), float(grid[-1])
        if label in fixed:
            value = fixed[label]
            if not (g_lo <= value <= g_hi):
                problems.append(
                    f"  - {label} fixed at {value:.6g}, outside the tabulated "
                    f"grid [{g_lo:.6g}, {g_hi:.6g}] -> every sample would be -inf"
                )
            continue
        if float(lo) < g_lo or float(hi) > g_hi:
            problems.append(
                f"  - {label} prior [{float(lo):.6g}, {float(hi):.6g}] extends "
                f"outside the tabulated grid [{g_lo:.6g}, {g_hi:.6g}]"
            )

    if problems:
        raise ValueError(
            "Cosmology prior outside the distance interpolation grid.\n"
            "darksirens.utils.cosmology tabulates comoving distance on a fixed "
            "(Om0, w0, wa, z) grid and returns NaN outside it, which the "
            "likelihood turns into -inf. The run would silently sample a "
            "TRUNCATED prior while reporting the one you asked for, biasing logZ "
            "and putting a spurious hard wall in the marginal.\n\n"
            + "\n".join(problems)
            + "\n\nFix by keeping the prior inside the tabulated range, or "
            "rebuild the grid to cover it (the guard bands are set by "
            "_OM0_GRID_PAD / _W0_GRID_PAD / _WA_GRID_PAD in utils/cosmology.py)."
        )


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
    n_catalogs: int = 1,
    lss_completion_active: bool | Sequence[bool] = False,
    use_lss: bool = True,
    mark_names_by_catalog=None,
    c_mode: "str | Sequence[str]" = "per_pixel",
    selection_prior=None,
):
    """Construct labels and prior bounds for cosmological, population, survey, and sky parameters.

    The survey block -- its labels, their order, their bounds and whether each
    one is sampled for the resolved ``(universe_model, use_lss,
    lss_completion_active)`` configuration -- is derived ENTIRELY from the
    ``_SURVEY_BLOCK`` registry at the top of this module, for catalog 1 and for
    every ``_c{k}`` catalog alike.  Naming a survey label in ``prior_overrides``
    that the registry does not sample here is an error (it would silently do
    nothing), and the error quotes the registry's own reason; ``z50``, ``w`` and
    ``alpha_miss`` are :class:`~darksirens.core.types.SurveyParams` fields that
    no model ever samples, so they carry no prior at all.

    Parameters
    ----------
    fix_cosmology
        When true, remove the full cosmology block (``H0``, ``Om0``, ``w0``,
        and ``wa``) from sampling.
    fix_de
        When true, remove only the dark-energy cosmology labels (``w0`` and
        ``wa``) from sampling. ``fix_cosmology`` supersedes this flag.
    n_catalogs
        Number of EM catalogs in the redshift-prior mixture.  ``1`` (default)
        is the single-catalog path and is left bit-for-bit unchanged.  For
        ``K >= 2`` the space gains per-catalog suffixed survey blocks
        (``log10n0_c2`` ... for catalogs ``2..K``) and the mixture-weight sticks
        ``fcat_2 .. fcat_K`` (Beta(1, K-m+1) priors), appended before the
        sky/mark blocks.  The 14-element return tuple arity is unchanged.
    lss_completion_active
        Whether a loaded Q_LSS completion table is active PER CATALOG (a Q table
        replaces the local ``(1 + alpha_miss*b_miss*delta_g)`` factor, so that
        catalog's ``b_miss`` no longer enters the likelihood and is dropped from
        its sampled survey block).  A scalar ``bool`` broadcasts to every catalog
        (the legacy semantics); a sequence must have length exactly
        ``n_catalogs`` and gates each catalog independently -- catalog ``k``'s
        ``b_miss_c{k}`` is dropped iff element ``k-1`` is true.  This is
        per-catalog because the likelihood chooses Q-vs-``b_miss`` per catalog
        (``completion.field_lss_q is not None``), so a mixed config (Q on some
        catalogs only) must keep ``b_miss_c{k}`` for the Q-free catalogs.
    selection_prior
        Magnitude-fit Gaussian priors ``{label: (loc, scale)}`` that flip the
        SAMPLED selection labels from flat to a truncated normal.  Labels are
        per catalog: ``M0hat``/``sigma_M`` for catalog 1 and
        ``M0hat_c{k}``/``sigma_M_c{k}`` for catalogs ``2..K``, each anchored by
        THAT catalog's own offline fit.  All-or-nothing: a run that anchors any
        selection label must anchor every sampled one (a partially anchored
        mixture leaves an unconstrained completeness nuisance); ``None`` leaves
        every catalog flat within bounds (the wide-open ablation).
    """
    if prior_overrides is None:
        prior_overrides = {}
    if fixed_parameter_values is None:
        fixed_parameter_values = {}

    # Normalize per-catalog Q_LSS activity to a length-n_catalogs tuple.  A
    # scalar bool broadcasts to every catalog (legacy semantics, bit-identical
    # to the pre-per-catalog path); a sequence gates each catalog independently
    # and MUST have exactly n_catalogs entries.
    if isinstance(lss_completion_active, Sequence) and not isinstance(
        lss_completion_active, (str, bytes)
    ):
        lss_active_by_cat = tuple(bool(x) for x in lss_completion_active)
        if len(lss_active_by_cat) != n_catalogs:
            raise ValueError(
                f"lss_completion_active given as a sequence of length "
                f"{len(lss_active_by_cat)}, but n_catalogs={n_catalogs}; pass "
                f"exactly one activity flag per catalog (or a single bool to "
                f"broadcast)."
            )
    else:
        lss_active_by_cat = (bool(lss_completion_active),) * n_catalogs

    # Normalize per-catalog c_mode the same way (a scalar string broadcasts).
    if isinstance(c_mode, Sequence) and not isinstance(c_mode, (str, bytes)):
        c_mode_by_cat = tuple(str(x) for x in c_mode)
        if len(c_mode_by_cat) != n_catalogs:
            raise ValueError(
                f"c_mode given as a sequence of length {len(c_mode_by_cat)}, "
                f"but n_catalogs={n_catalogs}; pass exactly one mode per "
                f"catalog (or a single string to broadcast).")
    else:
        c_mode_by_cat = (str(c_mode),) * n_catalogs

    #: 1-based catalog numbers running the parametric selection completeness.
    #: The magnitude-fit anchoring rule below is stated against THIS tuple (not
    #: against n_catalogs), so a future per-catalog c_mode needs no change here.
    selection_catalogs = tuple(
        k for k in range(1, n_catalogs + 1)
        if c_mode_by_cat[k - 1] == "selection"
    )

    def _survey_inactive_reason(label):
        """:class:`_Inert` for a (possibly ``_c{k}``-suffixed) survey label.

        ``None`` means the label IS sampled for this configuration.  Returns
        ``None`` for anything that is not a survey label at all -- those are
        handled by the unknown-label check.  This is the SAME rule the block
        builder uses, so a guard message can never contradict the gating.
        """
        base = _survey_base_name(label)
        catalog = _survey_catalog_number(label)
        if not (1 <= catalog <= n_catalogs):
            return None
        pinned = _PINNED_SURVEY_PARAMS.get(base)
        if pinned is not None:
            return pinned
        for spec in _SURVEY_BLOCK:
            if spec.label == base:
                return _survey_param_inactive_reason(
                    spec,
                    universe_model,
                    use_lss,
                    lss_active_by_cat[catalog - 1],
                    catalog,
                    c_mode=c_mode_by_cat[catalog - 1],
                )
        return None

    def _survey_block_for(catalog):
        """The sampled survey block for catalog ``catalog`` (1-based).

        Registry order and bounds, ``_c{k}``-suffixed for catalogs >= 2, gated
        by that catalog's OWN Q_LSS activity (the likelihood chooses
        Q-vs-b_miss per catalog), with ``prior_overrides`` applied under the
        catalog's own label spelling -- so a base-label override never leaks
        into a suffixed catalog's bounds.
        """
        suffix = "" if catalog == 1 else f"_c{catalog}"
        block = [
            (spec.label + suffix, spec.lower, spec.upper)
            for spec in _SURVEY_BLOCK
            if _survey_inactive_reason(spec.label + suffix) is None
        ]
        labels_c = [item[0] for item in block]
        lower_c, upper_c = apply_block_prior_overrides(
            f"survey_c{catalog}" if suffix else "survey",
            labels_c,
            [item[1] for item in block],
            [item[2] for item in block],
            prior_overrides,
        )
        return labels_c, lower_c, upper_c

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
    # Labels, order, bounds and per-configuration activity all come from the
    # ``_SURVEY_BLOCK`` registry at the top of this module; nothing about the
    # survey block is spelled out a second time here.  ``survey_labels`` is the
    # SAMPLEABLE block definition (the returned value), while the actually
    # sampled per-catalog blocks are built by ``_survey_block_for`` below.
    survey_labels = [spec.label for spec in _SURVEY_BLOCK]

    # --- Sky (angular source distribution) ---
    # Appended after the survey block; ``isotropic`` contributes no parameters.
    sky_lower, sky_upper, sky_labels, sky_kinds, _sky_latex = sky_model_prior_parser(sky_model)
    sky_lower, sky_upper = list(sky_lower), list(sky_upper)

    # --- Marks (BBH-host efficiency eta block) ---
    # Appended after the sky block; ``none`` contributes no parameters.  The
    # eta coefficients are one per available mark, PER CATALOG:
    # ``mark_names_by_catalog`` orders catalog 1 first (unsuffixed ``eta_<m>``
    # labels, identical to the historical K=1 block) then catalogs 2..K
    # (suffixed ``eta_<m>_c{k}`` blocks appended at the very end so every
    # pre-existing label index is untouched).  The flat ``mark_names`` remains
    # the K=1 / catalog-1 spelling.
    if mark_names_by_catalog is None:
        mark_names_by_catalog = (tuple(mark_names),) + ((),) * (n_catalogs - 1)
    mark_names_by_catalog = tuple(
        tuple(names) for names in mark_names_by_catalog
    )
    if len(mark_names_by_catalog) < n_catalogs:
        mark_names_by_catalog = mark_names_by_catalog + ((),) * (
            n_catalogs - len(mark_names_by_catalog)
        )
    def _mark_parser_for(names):
        # A catalog with no marks contributes no eta parameters: dispatch to
        # the null model rather than instantiating LogLinearMarks(()) (which
        # rejects an empty mark list by design).
        return mark_model_prior_parser(
            mark_model if names else "none", names
        )

    mark_lower, mark_upper, mark_labels, mark_kinds, _mark_latex = (
        _mark_parser_for(mark_names_by_catalog[0])
    )
    mark_lower, mark_upper = list(mark_lower), list(mark_upper)

    # Per-catalog suffixed eta blocks (catalogs 2..K).
    mark_blocks_ck = []
    for _k in range(2, n_catalogs + 1):
        _lo, _hi, _lbls, _knds, _ = _mark_parser_for(
            mark_names_by_catalog[_k - 1]
        )
        mark_blocks_ck.append((
            [f"{lbl}_c{_k}" for lbl in _lbls], list(_lo), list(_hi), list(_knds)
        ))

    # Make sure all prior override keys are valid parameter labels.  The survey
    # side uses every RECOGNISED survey label -- sampleable or pinned -- so that
    # naming a pinned label (z50/w/alpha_miss) reaches the survey guard below
    # and gets the true reason, instead of a bare "unknown label".
    known_labels = (
        set(cosmo_labels) | set(pop_labels) | set(_SURVEY_PARAM_LABELS)
        | set(sky_labels) | set(mark_labels)
    )
    # Multitracer: the per-catalog suffixed survey labels and the mixture-weight
    # sticks are also valid override / fixed-value keys for K >= 2.  No-op for K=1.
    if n_catalogs >= 2:
        for k in range(2, n_catalogs + 1):
            known_labels |= {f"{lbl}_c{k}" for lbl in _SURVEY_PARAM_LABELS}
        known_labels |= {f"fcat_{m}" for m in range(2, n_catalogs + 1)}
        for _lbls, _lo, _hi, _knds in mark_blocks_ck:
            known_labels |= set(_lbls)
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

    # Fail early on per-label config naming a survey parameter that is INERT for
    # the resolved configuration.  Such an entry is silently ignored otherwise (a
    # phantom config): the label is not sampled, so an override changes nothing
    # at all.  The reason is taken from the registry rule that did the gating, so
    # it is always the TRUE one -- the old guard reported "catalog 1 has an
    # active Q_LSS completion table" even when the real cause was --use_lss off.
    #
    # Block-level ``fix_survey`` is deliberately NOT a reason: it fixes the whole
    # block rather than a named label, and pinning survey parameters to a Q
    # table's build values under --fix_survey is the documented Q_LSS workflow.
    for _source_name, _source in (
        ("prior override", prior_overrides),
        ("fixed value", fixed_parameter_values),
    ):
        for _key in _source:
            _inert = _survey_inactive_reason(_key)
            if _inert is None:
                continue
            if _source is fixed_parameter_values and _inert.fixed_ok:
                # Endorsed fixed-value path (e.g. the m_lim truncation datum,
                # which --selection_fit itself pins): no warning.
                continue
            _message = (
                f"A {_source_name} was given for '{_key}', but it is not a "
                f"sampled parameter of this configuration: {_inert.reason}.\n"
                f"Remove the {_source_name} for '{_key}'"
                + (f", {_inert.remedy}" if _inert.remedy else "")
            )
            if _source is prior_overrides or _inert.fatal_when_fixed:
                # Overrides are ALWAYS an error: prior bounds on a parameter
                # that is not sampled can only mislead.  Fixed values are an
                # error only for the LSS gates (see _Inert.fatal_when_fixed).
                raise ValueError(_message + ".")
            # A fixed value on a never-sampled label is legitimate (it pins the
            # SurveyParams field the decoder builds), so warn rather than break
            # archived settings.json / mock scripts that pin generative truth.
            warnings.warn(
                _message + ", or leave it at the fiducial (the decoder pins "
                "the SurveyParams field at the value either way).",
                stacklevel=2,
            )

    # Apply block overrides.  EVERY block that carries sampled bounds is
    # resolved here -- including the base mark block, the per-catalog suffixed
    # mark/survey blocks, and the mixture sticks below -- BEFORE the canonical
    # bounds map is built.  Previously mark/suffixed overrides were validated
    # as known labels but never actually applied to their bounds (silently
    # ignored), and fixed+override overlap on those labels raised a bare
    # KeyError instead of the intended ValueError.
    cosmo_lower, cosmo_upper = apply_block_prior_overrides(
        "cosmology", cosmo_labels, cosmo_lower, cosmo_upper, prior_overrides
    )
    _validate_cosmology_within_interpolation_grid(
        cosmo_labels, cosmo_lower, cosmo_upper, fixed_parameter_values
    )
    pop_lower, pop_upper = apply_block_prior_overrides(
        "population", pop_labels, pop_lower, pop_upper, prior_overrides
    )
    sky_lower, sky_upper = apply_block_prior_overrides(
        "sky", sky_labels, sky_lower, sky_upper, prior_overrides
    )
    mark_lower, mark_upper = apply_block_prior_overrides(
        "mark", mark_labels, mark_lower, mark_upper, prior_overrides
    )
    # Per-catalog suffixed eta blocks (catalogs 2..K): re-resolve each block's
    # bounds against prior_overrides, labels/kinds untouched.
    _mark_blocks_ck_overridden = []
    for _k, (_lbls, _lo, _hi, _knds) in zip(range(2, n_catalogs + 1), mark_blocks_ck):
        _lo, _hi = apply_block_prior_overrides(
            f"mark_c{_k}", _lbls, _lo, _hi, prior_overrides
        )
        _mark_blocks_ck_overridden.append((_lbls, _lo, _hi, _knds))
    mark_blocks_ck = _mark_blocks_ck_overridden

    # Survey blocks, one per catalog, straight from the registry: catalog 1 is
    # the unsuffixed block and catalogs 2..K carry the ``_c{k}`` suffix (empty
    # for K = 1, matching the bit-identical single-catalog path).  Each block is
    # already gated by its own catalog's Q_LSS activity and has its own bounds
    # overrides applied, so the canonical bounds map below sees final bounds and
    # only fixed-parameter filtering remains.
    survey_labels_1, survey_lower, survey_upper = _survey_block_for(1)
    survey_blocks_ck = [_survey_block_for(k) for k in range(2, n_catalogs + 1)]

    # Mixture-weight sticks fcat_2..fcat_K, resolved here for the same reason.
    # Empty for K = 1.  Bounds act as TRUNCATION of the Beta(1, b) stick prior
    # (make_prior_transform's truncated closed-form PPF); a fixed stick outside
    # [0, 1] would silently become a -inf log weight, so both are rejected here.
    fcat_labels = [f"fcat_{m}" for m in range(2, n_catalogs + 1)]
    fcat_lower = [0.0] * len(fcat_labels)
    fcat_upper = [1.0] * len(fcat_labels)
    fcat_lower, fcat_upper = apply_block_prior_overrides(
        "mixture", fcat_labels, fcat_lower, fcat_upper, prior_overrides
    )
    for _lbl, _lo, _hi in zip(fcat_labels, fcat_lower, fcat_upper):
        if not (0.0 <= _lo < _hi <= 1.0):
            raise ValueError(
                f"{_lbl} prior bounds must satisfy 0 <= lo < hi <= 1 "
                f"(a mixture stick); got [{_lo}, {_hi}]."
            )
        fixed_val = (fixed_parameter_values or {}).get(_lbl)
        if fixed_val is not None and not (0.0 <= float(fixed_val) <= 1.0):
            raise ValueError(
                f"Fixed value for {_lbl} must lie in [0, 1]; got {fixed_val}."
            )

    # Canonical label -> bounds map covering every label the prior actually
    # accepts: cosmology, population, survey (base + suffixed), sky, marks
    # (base + suffixed), and the mixture sticks.  Used below so fixed+override
    # overlap validation works for ANY accepted label, not just the
    # cosmo/pop/survey/sky base set.
    all_bounds = {
        label: (float(lo), float(hi))
        for label, lo, hi in (
            list(zip(cosmo_labels, cosmo_lower, cosmo_upper))
            + list(zip(pop_labels, pop_lower, pop_upper))
            + list(zip(survey_labels_1, survey_lower, survey_upper))
            + list(zip(sky_labels, sky_lower, sky_upper))
            + list(zip(mark_labels, mark_lower, mark_upper))
            + [
                item
                for suffixed_labels, c_lower, c_upper in survey_blocks_ck
                for item in zip(suffixed_labels, c_lower, c_upper)
            ]
            + [
                item
                for _lbls, _lo, _hi, _knds in mark_blocks_ck
                for item in zip(_lbls, _lo, _hi)
            ]
            + list(zip(fcat_labels, fcat_lower, fcat_upper))
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
        survey_labels_1, survey_lower, survey_upper, fixed_parameter_values
    )
    sampled_sky_labels, sampled_sky_lower, sampled_sky_upper = filter_fixed_parameters(
        sky_labels, sky_lower, sky_upper, fixed_parameter_values
    )
    sampled_mark_labels, sampled_mark_lower, sampled_mark_upper = filter_fixed_parameters(
        mark_labels, mark_lower, mark_upper, fixed_parameter_values
    )

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

    # Multitracer: per-catalog suffixed survey blocks (catalogs 2..K) followed by
    # the mixture-weight sticks fcat_2..fcat_K, all appended AFTER the base survey
    # block and BEFORE the sky/mark blocks so the single-catalog indices above are
    # untouched.  The whole block is skipped for K = 1 (bit-identical path).
    # Each block came out of ``_survey_block_for`` already gated by its OWN
    # catalog's Q_LSS activity (a Q-on-catalog-1-only config keeps b_miss_c2)
    # with its bounds overrides applied; only fixed-parameter filtering remains.
    if n_catalogs >= 2:
        for suffixed_labels, c_lower, c_upper in survey_blocks_ck:
            (
                c_sampled_labels,
                c_sampled_lower,
                c_sampled_upper,
            ) = filter_fixed_parameters(
                suffixed_labels, c_lower, c_upper, fixed_parameter_values
            )
            if not fix_survey:
                labels += c_sampled_labels
                lower += c_sampled_lower
                upper += c_sampled_upper

        # Mixture-weight sticks fcat_2..fcat_K ([0,1] bounds; the Beta(1,b)
        # prior family is carried in prior_kinds below).  Sampled regardless of
        # fix_survey; individually fixable via fixed_parameter_values.  Bounds
        # were already resolved against prior_overrides and range-validated
        # above (before the canonical bounds map was built).
        fcat_labels, fcat_lower, fcat_upper = filter_fixed_parameters(
            fcat_labels, fcat_lower, fcat_upper, fixed_parameter_values
        )
        labels += fcat_labels
        lower += fcat_lower
        upper += fcat_upper

    # Sky block: appended last so existing cosmo/pop/survey indices are stable.
    # The chosen ``sky_model`` decides the parameters (none for ``isotropic``);
    # individually-fixed sky labels were already removed above.
    labels += sampled_sky_labels
    lower += sampled_sky_lower
    upper += sampled_sky_upper

    # Mark block: appended after sky so all earlier indices stay stable.
    labels += sampled_mark_labels
    lower += sampled_mark_lower
    upper += sampled_mark_upper
    n_mark_eff = len(sampled_mark_labels)

    # Per-catalog suffixed eta blocks (catalogs 2..K), appended LAST so the
    # K=1 label layout (and every earlier index) is untouched.
    for _lbls, _lo, _hi, _knds in mark_blocks_ck:
        _s_lbls, _s_lo, _s_hi = filter_fixed_parameters(
            _lbls, _lo, _hi, fixed_parameter_values
        )
        labels += _s_lbls
        lower += _s_lo
        upper += _s_hi
        n_mark_eff += len(_s_lbls)

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
    for _lbls, _lo, _hi, _knds in mark_blocks_ck:
        for lbl, knd in zip(_lbls, _knds):
            kind_map[lbl] = knd
    # Multitracer: mixture-weight sticks carry a Beta(1, K-m+1) prior; the
    # suffixed survey labels fall through to the default uniform kind via .get().
    if n_catalogs >= 2:
        for m in range(2, n_catalogs + 1):
            kind_map[f"fcat_{m}"] = ("beta", 1.0, float(n_catalogs - m + 1))
    # Selection-fit Gaussian priors: ``selection_prior = {label: (loc, scale)}``
    # (labels may carry the _c{k} suffix) flips those SAMPLED selection labels
    # from the flat default to a truncated normal -- the offline magnitude
    # fit's marginal Laplace sds.  Without a fit the labels stay flat within
    # bounds (the deliberate wide-open ablation).  Fail-closed: a prior for a
    # label that is not sampled in this configuration is a config error, not
    # a silent no-op.
    for lbl, (loc, scale) in (selection_prior or {}).items():
        if _survey_base_name(lbl) not in ("M0hat", "sigma_M"):
            raise ValueError(
                f"selection_prior names '{lbl}', but only the sampled "
                f"selection parameters M0hat / sigma_M (optionally "
                f"_c{{k}}-suffixed) accept a magnitude-fit Gaussian prior.")
        if lbl not in labels:
            reason = _survey_inactive_reason(lbl)
            why = f": {reason.reason}" if reason is not None else (
                " (is it fixed via --fixed_parameter_values, dropped by "
                "fix_survey=True -- which fixes the whole block including "
                "theta, incompatible with a selection-fit prior -- or "
                "suffixed for a catalog this run does not have?)")
            raise ValueError(
                f"selection_prior names '{lbl}', which is not sampled in "
                f"this configuration{why}.")
        if not (np.isfinite(loc) and np.isfinite(scale) and scale > 0):
            raise ValueError(
                f"selection_prior for '{lbl}' needs finite loc and positive "
                f"scale, got ({loc}, {scale}).")
        # The Gaussian is TRUNCATED to this label's bounds, and a fit center
        # outside them freezes the parameter instead of refusing: the PPF
        # evaluates ndtr at |z| >~ 40 on both edges, so Phi_a == Phi_b == 1
        # and every unit-cube value maps to the same near-edge constant.  The
        # >5-sd reach warning downstream cannot catch it (it fires on every
        # legal anchored run, where the box is ~5 mag and the fit sd ~1e-2).
        # Reachable per catalog now that heterogeneous tracers -- a bright
        # cluster sample and a deep photo-z catalog -- share one registry box.
        _i = labels.index(lbl)
        if not (lower[_i] < loc < upper[_i]):
            raise ValueError(
                f"selection_prior centers '{lbl}' at {loc}, outside its "
                f"sampling bounds [{lower[_i]}, {upper[_i]}]: the truncated "
                f"Gaussian would collapse to a single constant at the nearer "
                f"bound, silently freezing this catalog's completeness base "
                f"instead of sampling the fit.\nWiden the box to bracket the "
                f"fit, e.g. --prior_overrides '{{\"{lbl}\": "
                f"[{min(lower[_i], loc - 5.0 * scale):g}, "
                f"{max(upper[_i], loc + 5.0 * scale):g}]}}', or check that "
                f"this catalog's darksirens_fit_selection JSON is the right "
                f"one for it."
            )
        kind_map[lbl] = ("normal", float(loc), float(scale))

    # Fail-closed anchoring: with ANY magnitude fit in play, EVERY sampled
    # selection label must carry its own fit-anchored prior.  A partially
    # anchored mixture is the silent failure this replaces the old K>=2
    # refusal with: catalog k's M0hat_c{k}/sigma_M_c{k} would sample FLAT
    # across the 5-mag truncation box while catalog 1 sits inside ~1e-2 mag
    # of its fit, so the mixture's whole missing-galaxy budget is set by an
    # unconstrained nuisance that no data in the run constrains.  Supplying
    # NO fit at all stays legal for every K (the deliberate wide-open
    # ablation, identical to the single-catalog behaviour).
    if selection_prior:
        unanchored = [
            lbl
            for k in selection_catalogs
            for lbl in (
                f"M0hat{'' if k == 1 else f'_c{k}'}",
                f"sigma_M{'' if k == 1 else f'_c{k}'}",
            )
            if lbl in labels and lbl not in selection_prior
        ]
        if unanchored:
            raise ValueError(
                f"selection_prior anchors only part of this run's "
                f"c_mode='selection' block: {unanchored} are SAMPLED but "
                f"carry no magnitude-fit prior, so they would sample flat "
                f"across their truncation bounds while the anchored labels "
                f"sit within the fit's ~1e-2 mag uncertainty -- the mixture's "
                f"missing-galaxy budget would be set by an unconstrained "
                f"nuisance.\nPass one darksirens_fit_selection JSON PER "
                f"CATALOG (--selection_fit fitA.json,fitB.json, in "
                f"--survey_path order), or drop --selection_fit entirely for "
                f"the wide-open ablation on every catalog."
            )
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

def resolve_joint_prior_constraints(
    pop_model, labels, lower, upper, prior_kinds=None, *,
    shared_beta=True, shared_spin=True, shared_gamma=True,
    sky_model=None,
):
    """Index-resolved joint prior constraints for ``make_prior_transform``.

    A population model may declare ``constraint_groups`` — joint conditions
    its ``log_p_pop`` enforces by rejection (returning -inf), e.g. the
    GWTC-5 fiducial model's ``lambda0 + lambda1 <= 1`` and
    ``m2_low <= m1_low``.  Rejection alone leaves the invalid corner inside
    the sampled prior: nested proposals waste those draws and the evidence
    is shifted by the log prior fraction (log 1/4 for the two conditions
    above).  This resolver turns the declarations into cube-space map specs
    ``(kind, (i, j))`` over the FINAL sampled-label vector, applied by
    ``make_prior_transform`` before the per-dimension maps.

    A group is skipped (falling back to the rejection backstop, with a
    warning) when any member label is not sampled, or when the members'
    bounds/prior kinds break the map's measure-preservation requirements:
    ``ordered_le`` needs identical uniform bounds on both members (sorting
    then commutes with the shared monotone map), ``simplex`` needs exact
    uniform [0, 1] bounds (the cube fold IS the parameter-space fold only
    there).
    """
    from darksirens.gw.populations import get_model

    groups = []
    try:
        model = get_model(
            pop_model, shared_beta=shared_beta, shared_spin=shared_spin,
            shared_gamma=shared_gamma,
        )
        groups.extend(getattr(model, "constraint_groups", None) or ())
    except Exception:
        pass
    if sky_model is not None:
        from darksirens.sky import get_sky_model

        try:
            groups.extend(
                getattr(get_sky_model(sky_model), "constraint_groups", None)
                or ()
            )
        except Exception:
            pass
    if not groups:
        return []

    labels = [str(lab) for lab in labels]
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    resolved = []
    for kind, group_labels in groups:
        try:
            idx = tuple(labels.index(str(lab)) for lab in group_labels)
        except ValueError:
            continue  # a member is fixed/absent; rejection handles the rest
        uniform = prior_kinds is None or all(
            prior_kinds[i][0] == "uniform" for i in idx
        )
        if kind == "ordered_le" and len(idx) == 2:
            ok = uniform and (
                lower[idx[0]] == lower[idx[1]] and upper[idx[0]] == upper[idx[1]]
            )
        elif kind == "simplex" and len(idx) == 2:
            ok = uniform and all(
                lower[i] == 0.0 and upper[i] == 1.0 for i in idx
            )
        elif kind == "ball3" and len(idx) == 3:
            # The polar map targets the UNIT ball in parameter space, so the
            # component boxes must be exactly [-1, 1].
            ok = uniform and all(
                lower[i] == -1.0 and upper[i] == 1.0 for i in idx
            )
        else:
            ok = False
        if not ok:
            warnings.warn(
                f"joint prior constraint {kind}{tuple(group_labels)} cannot "
                "be applied as a constraint-preserving cube map with these "
                "bounds/prior kinds; falling back to rejection (the invalid "
                "region keeps zero likelihood, proposals there are wasted, "
                "and logZ carries the log prior-fraction offset).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved.append((str(kind), idx))
    return resolved


def make_prior_transform(lower, upper, prior_kinds=None, joint_constraints=None):
    """Unit-cube -> parameter inverse-CDF transform, per-parameter prior-aware.

    ``prior_kinds`` is an optional list aligned to ``lower``/``upper`` of
    ``(kind, loc, scale)`` triples (kind in ``{"uniform", "normal",
    "lognormal", "beta"}``; ``"beta"`` is Beta(1, scale), the mixture-weight
    stick).  ``None`` reproduces the legacy all-uniform affine map.
    Used by the nested samplers (dynesty/tinyns); numpyro builds its own prior
    via ``run_sampler``.  ``low``/``high`` always act as truncation bounds, so
    every kind maps the cube to ``[low, high]`` and the measure matches the
    corresponding numpyro distribution.

    ``joint_constraints`` (from :func:`resolve_joint_prior_constraints`) are
    measure-preserving cube maps applied BEFORE the per-dimension transforms,
    mapping the cube onto a model's jointly-constrained prior region instead
    of leaving it to likelihood-side rejection:

    * ``("ordered_le", (i, j))`` — sort, so parameter i <= parameter j.  Each
      ordered pair has exactly two preimages, so a uniform cube maps to the
      NORMALIZED uniform density on the ordered triangle.
    * ``("simplex", (i, j))`` — the fold ``(u_i, u_j) -> (1-u_i, 1-u_j)``
      when ``u_i + u_j > 1``; two preimages per point of the simplex, again
      the normalized uniform density.
    * ``("ball3", (i, j, k))`` — polar map onto the uniform unit ball
      (``r = u^(1/3)``), for parameters whose joint support is
      ``|v| <= 1`` in a ``[-1, 1]^3`` box (the dipole vector).

    Implementation note: dynesty wraps this with ``np.asarray(transform(
    jnp.asarray(u)))``, i.e. the transform always receives a JAX array, so
    ``jax.scipy.special`` is safe across all nested backends.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    joint_constraints = list(joint_constraints or [])

    if joint_constraints:
        import jax.numpy as _jnp

        def _apply_joint(u):
            u = _jnp.asarray(u)
            for kind, idx in joint_constraints:
                if kind == "ball3":
                    # Cube -> uniform unit ball (radius u^(1/3), polar angles
                    # from the other two coords) -> back to the [-1, 1] cube
                    # coordinates the per-dimension affine map expects.
                    i, j, k = idx
                    r = u[..., i] ** (1.0 / 3.0)
                    cth = 2.0 * u[..., j] - 1.0
                    sth = _jnp.sqrt(_jnp.maximum(1.0 - cth * cth, 0.0))
                    phi = 2.0 * _jnp.pi * u[..., k]
                    u = (
                        u.at[..., i].set(0.5 * (r * sth * _jnp.cos(phi) + 1.0))
                         .at[..., j].set(0.5 * (r * sth * _jnp.sin(phi) + 1.0))
                         .at[..., k].set(0.5 * (r * cth + 1.0))
                    )
                    continue
                i, j = idx
                ui, uj = u[..., i], u[..., j]
                if kind == "ordered_le":
                    new_i, new_j = _jnp.minimum(ui, uj), _jnp.maximum(ui, uj)
                else:  # simplex
                    over = (ui + uj) > 1.0
                    new_i = _jnp.where(over, 1.0 - ui, ui)
                    new_j = _jnp.where(over, 1.0 - uj, uj)
                u = u.at[..., i].set(new_i).at[..., j].set(new_j)
            return u
    else:
        def _apply_joint(u):
            return u

    if prior_kinds is None or all(k[0] == "uniform" for k in prior_kinds):
        def prior_transform(u):
            return _apply_joint(u) * (upper - lower) + lower
        return prior_transform

    import jax.numpy as jnp
    from jax.scipy.special import ndtr, ndtri

    kinds = [k[0] for k in prior_kinds]
    loc = jnp.asarray([0.0 if k[1] is None else float(k[1]) for k in prior_kinds])
    scale = jnp.asarray([1.0 if k[2] is None else float(k[2]) for k in prior_kinds])
    lo_j, hi_j = jnp.asarray(lower), jnp.asarray(upper)
    is_normal = jnp.asarray([k == "normal" for k in kinds])
    is_lognorm = jnp.asarray([k == "lognormal" for k in kinds])
    # Mixture-weight sticks: Beta(1, b) with the closed-form inverse CDF
    # x = 1 - (1 - u)^(1/b), b in the scale slot.  Bounds act as truncation
    # (like every other kind): with CDF F(x) = 1 - (1-x)^b the truncated PPF is
    # x = F^{-1}(F(lo) + u (F(hi) - F(lo))), still closed form.  For the
    # default [0, 1] bounds this reduces exactly to the untruncated map.
    is_beta = jnp.asarray([k == "beta" for k in kinds])

    def _trunc_normal_ppf(u, a, b, mu, sg):
        # inverse CDF of N(mu, sg^2) truncated to [a, b]
        za = (a - mu) / sg
        zb = (b - mu) / sg
        Phi_a, Phi_b = ndtr(za), ndtr(zb)
        x = ndtri(jnp.clip(Phi_a + u * (Phi_b - Phi_a), 1e-12, 1.0 - 1e-12))
        return mu + sg * x

    def prior_transform(u):
        u = _apply_joint(jnp.asarray(u))
        uniform = u * (hi_j - lo_j) + lo_j
        # normal: truncated to [lo, hi]
        normal = _trunc_normal_ppf(u, lo_j, hi_j, loc, scale)
        # lognormal: exp of a normal in log-space truncated to [log lo, log hi]
        log_lo = jnp.log(jnp.clip(lo_j, 1e-300, None))
        log_hi = jnp.log(jnp.clip(hi_j, 1e-300, None))
        lognormal = jnp.exp(_trunc_normal_ppf(u, log_lo, log_hi, loc, scale))
        # Beta(1, b) closed-form PPF truncated to [lo, hi]; scale defaults to 1
        # for non-beta params so this expression is finite everywhere.  The
        # bounds are clipped into [0, 1) so masked-out (non-beta) lanes with
        # arbitrary physical bounds cannot produce NaNs.
        b_lo = jnp.clip(lo_j, 0.0, 1.0 - 1e-12)
        b_hi = jnp.clip(hi_j, 0.0, 1.0)
        F_lo = 1.0 - (1.0 - b_lo) ** scale
        F_hi = 1.0 - (1.0 - b_hi) ** scale
        u_beta = F_lo + u * (F_hi - F_lo)
        beta = 1.0 - (1.0 - u_beta) ** (1.0 / scale)
        out = jnp.where(is_normal, normal, uniform)
        out = jnp.where(is_lognorm, lognormal, out)
        out = jnp.where(is_beta, beta, out)
        return out

    return prior_transform
