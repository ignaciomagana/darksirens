"""Parameter decoding helpers shared by inference likelihood builders."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from darksirens.core.constants import (
    COMPLETE_EMPTY_PIXEL_POLICIES,
    H0_FID,
    OM0_FID,
    SURVEY_PARAMS_FID,
    W0_FID,
    WA_FID,
)
from darksirens.core.types import CosmoParams, SurveyParams
from darksirens.inference.prior import build_parameter_space, resolve_parameter_values
from darksirens.sky import get_fixed_sky_params
from darksirens.marks import mark_fiducial


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
    sky_labels: tuple[str, ...] = ()
    sky_params_fid: tuple[float, ...] = ()
    mark_labels: tuple[str, ...] = ()
    mark_params_fid: tuple[float, ...] = ()
    # Weak-lensing magnification model carried on SurveyParams (NOT a sampled
    # parameter); None for every non-WL universe model.
    wl_params: object | None = None

    def decode(self, coord: jnp.ndarray):
        """Return ``(cosmo, survey, pop_params, sky_params, mark_params)`` for ``coord``."""
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

        # Sky parameter sub-vector (empty for the isotropic model).
        sky_params = jnp.array([
            _get(label, self.sky_params_fid[i])
            for i, label in enumerate(self.sky_labels)
        ])

        # Mark (eta) sub-vector (empty for mark_model="none").
        mark_params = jnp.array([
            _get(label, self.mark_params_fid[i])
            for i, label in enumerate(self.mark_labels)
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
            wl_params=self.wl_params,
        )
        return cosmo, survey, pop_params, sky_params, mark_params


def build_parameter_decoder(
    opts,
    pop_params_fid,
    fixed_parameter_values: dict | None = None,
    wl_params=None,
) -> ParameterDecoder:
    """Build the coordinate decoder using ``build_parameter_space`` ordering."""
    if fixed_parameter_values is None:
        fixed_parameter_values = {}
    fixed_parameter_values = {
        label: float(value) for label, value in fixed_parameter_values.items()
    }
    sky_model = getattr(opts, "sky_model", "isotropic")
    mark_model = getattr(opts, "mark_model", "none")
    mark_names = tuple(getattr(opts, "mark_names", ()) or ())
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
        _prior_kinds,
        sky_labels,
        mark_labels,
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
        sky_model=sky_model,
        mark_model=mark_model,
        mark_names=mark_names,
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
        sky_labels=tuple(sky_labels),
        sky_params_fid=tuple(float(v) for v in get_fixed_sky_params(sky_model)),
        mark_labels=tuple(mark_labels),
        mark_params_fid=tuple(float(v) for v in mark_fiducial(mark_model, mark_names)),
        wl_params=(
            wl_params
            if getattr(opts, "universe_model", None) == "spectral_sirens_wl"
            else None
        ),
    )
