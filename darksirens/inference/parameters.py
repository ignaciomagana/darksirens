"""Parameter decoding helpers shared by inference likelihood builders."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from astropy.cosmology import Planck15

from darksirens.inference.prior import build_parameter_space, resolve_parameter_values
from darksirens.utils.containers import CosmoParams, SurveyParams

H0_FID = float(Planck15.H0.value)
OM0_FID = float(Planck15.Om0)
W0_FID = -1.0
WA_FID = 0.0
# (log10n0->n0, z50, w, delta, b_miss, alpha_miss, sigma_kde).  z50/w are
# inactive under the ratio-only dark-siren completeness; alpha_miss defaults
# to 1 because it enters only through the exact product alpha_miss*b_miss.
SURVEY_PARAMS_FID = (-2.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0)

COMPLETE_EMPTY_PIXEL_POLICIES = {"zero": 0, "volume": 1}


def complete_empty_pixel_policy_code(policy: str | int) -> int:
    """Return the integer code stored on ``SurveyParams`` for empty-pixel policy."""
    if isinstance(policy, str):
        return COMPLETE_EMPTY_PIXEL_POLICIES[policy]
    return int(policy)


@dataclass(frozen=True)
class ParameterDecoder:
    """Decode sampler coordinates into typed cosmology, survey, and population params."""

    sampled_labels: tuple[str, ...]
    fixed_parameter_values: dict[str, float]
    pop_labels: tuple[str, ...]
    survey_labels: tuple[str, ...]
    pop_params_fid: tuple[float, ...]
    complete_empty_pixel_policy: int

    def decode(self, coord: jnp.ndarray):
        """Return ``(cosmo, survey, pop_params)`` for sampler coordinate ``coord``."""
        coord = jnp.asarray(coord)
        values = resolve_parameter_values(
            coord, self.sampled_labels, self.fixed_parameter_values
        )

        def _get(label, default):
            return values[label] if label in values else default

        H0 = _get("H0", H0_FID)
        Om0 = _get("Om0", OM0_FID)
        w0 = _get("w0", W0_FID)
        wa = _get("wa", WA_FID)

        pop_params = jnp.array([
            _get(label, self.pop_params_fid[i])
            for i, label in enumerate(self.pop_labels)
        ])

        sp = jnp.array([
            _get(label, float(SURVEY_PARAMS_FID[i]))
            for i, label in enumerate(self.survey_labels)
        ])

        cosmo = CosmoParams(H0=H0, Om0=Om0, w0=w0, wa=wa)
        survey = SurveyParams(
            n0=10.0 ** sp[0],
            z50=sp[1],
            w=sp[2],
            delta=sp[3],
            b_miss=sp[4],
            alpha_miss=sp[5],
            sigma_kde=sp[6],
            complete_empty_pixel_policy=self.complete_empty_pixel_policy,
        )
        return cosmo, survey, pop_params


def build_parameter_decoder(
    opts,
    pop_params_fid,
    fixed_parameter_values: dict | None = None,
) -> ParameterDecoder:
    """Build the coordinate decoder using ``build_parameter_space`` ordering."""
    if fixed_parameter_values is None:
        fixed_parameter_values = {}
    fixed_parameter_values = {
        label: float(value) for label, value in fixed_parameter_values.items()
    }
    (
        sampled_labels,
        _lower,
        _upper,
        _n_pop_eff,
        pop_labels,
        survey_labels,
        _cosmo_labels,
        _n_cosmo_eff,
        _n_survey_eff,
        _model_name,
        _fixed_parameter_statuses,
    ) = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        getattr(opts, "fix_cosmology", getattr(opts, "fixed_cosmology", False)),
        opts.fix_survey,
        fix_de=getattr(opts, "fix_de", getattr(opts, "fixed_de", False)),
        prior_overrides=getattr(opts, "prior_overrides", None),
        fixed_parameter_values=fixed_parameter_values,
        universe_model=getattr(opts, "universe_model", None),
        shared_beta=getattr(opts, "shared_beta", True),
        shared_spin=getattr(opts, "shared_spin", True),
        shared_gamma=getattr(opts, "shared_gamma", True),
    )

    return ParameterDecoder(
        sampled_labels=tuple(sampled_labels),
        fixed_parameter_values=fixed_parameter_values,
        pop_labels=tuple(pop_labels),
        survey_labels=tuple(survey_labels),
        pop_params_fid=tuple(float(v) for v in pop_params_fid),
        complete_empty_pixel_policy=complete_empty_pixel_policy_code(
            getattr(opts, "complete_empty_pixel_policy", "zero")
        ),
    )
