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


def _sticks_to_log_weights(v: jnp.ndarray) -> jnp.ndarray:
    """Stick-breaking sticks ``v = (fcat_2, ..., fcat_K)`` -> log mixture weights.

    Implements the Dirichlet(1, ..., 1) uniform-simplex construction where the
    sampled labels are the sticks ``v_1 .. v_{K-1}`` (``fcat_m`` is ``v_{m-1}``):

        log w_m = log(v_{m-1}) + sum_{i < m-1} log(1 - v_i)   for m = 2..K
        log w_1 = sum_j log(1 - v_j)                          (the remainder)

    Returns a ``(K,)`` array ordered ``[w_1, w_2, ..., w_K]``.  Stable via
    ``jnp.log`` / ``jnp.log1p``; boundary sticks (0 or 1) yield ``-inf`` log
    weights (a catalog of exactly zero weight), which drop cleanly out of the
    downstream ``logsumexp`` mixture.
    """
    v = jnp.asarray(v)
    log_v = jnp.log(v)          # log(v_j)
    log1m = jnp.log1p(-v)       # log(1 - v_j)
    # Exclusive prefix sum of log(1 - v): excl[s] = sum_{i < s} log1m[i].
    # Computed as a right-shifted inclusive cumsum rather than
    # ``cumsum - log1m`` so a boundary stick (v = 1 => log1m = -inf) does not
    # produce a -inf - (-inf) = NaN weight.
    incl = jnp.cumsum(log1m)
    excl = jnp.concatenate([jnp.zeros((1,), dtype=incl.dtype), incl[:-1]])
    log_w_tail = log_v + excl              # catalogs 2..K, shape (K-1,)
    log_w_head = jnp.sum(log1m)            # catalog 1 (the simplex remainder)
    return jnp.concatenate([jnp.reshape(log_w_head, (1,)), log_w_tail])


@dataclass(frozen=True)
class ParameterDecoder:
    """Decode sampler coordinates into typed cosmology, survey, and population params."""

    sampled_labels: tuple[str, ...]
    fixed_parameter_values: dict[str, float]
    pop_labels: tuple[str, ...]
    survey_labels: tuple[str, ...]
    pop_params_fid: tuple[float, ...]
    complete_empty_pixel_policy: int
    # Resolved per-catalog survey redshift depth (CLI override > per-catalog
    # file attr > None), positionally aligned with the catalog order (catalog
    # 1 first, then 2..K for a mixture). An empty tuple (the default, e.g. for
    # bare test opts that never set ``resolved_survey_z_depths``) means every
    # catalog gets ``z_depth=None`` -- the legacy full-grid missing-galaxy
    # budget, bit-identical to the pre-existing behaviour.
    z_depths: tuple[float | None, ...] = ()
    sky_labels: tuple[str, ...] = ()
    sky_params_fid: tuple[float, ...] = ()
    mark_labels: tuple[str, ...] = ()
    mark_params_fid: tuple[float, ...] = ()
    # Per-catalog mark (eta) blocks for the K-catalog mixture: BASE label names
    # per catalog (catalog 1 first == mark_labels; catalogs 2..K are sampled
    # under the ``_c{k}`` suffix) and the matching fiducials.
    mark_labels_by_catalog: tuple[tuple[str, ...], ...] = ()
    mark_fid_by_catalog: tuple[tuple[float, ...], ...] = ()
    # Weak-lensing magnification model carried on SurveyParams (NOT a sampled
    # parameter); None for every non-WL universe model.
    wl_params: object | None = None
    # Number of EM catalogs in the redshift-prior mixture.  1 (default) is the
    # single-catalog path; ``decode_mixture`` is only meaningful for K >= 2.
    n_catalogs: int = 1

    def decode(self, coord: jnp.ndarray):
        """Return ``(cosmo, survey, pop_params, sky_params, mark_params)`` for ``coord``."""
        coord = jnp.asarray(coord)
        values = resolve_parameter_values(
            coord, self.sampled_labels, self.fixed_parameter_values
        )
        return self._decode_from_values(values)

    def _decode_from_values(self, values: dict):
        """Decode an already-resolved label dict into the single-catalog params.

        Split out of :meth:`decode` so :meth:`decode_mixture` can resolve the
        label dictionary exactly once (via ``resolve_parameter_values``) and
        reuse it for both the catalog-1 component and the suffixed ``_c{k}``
        blocks.  :meth:`decode`'s public behaviour is unchanged.
        """

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
            z_depth=self.z_depths[0] if len(self.z_depths) >= 1 else None,
            wl_params=self.wl_params,
        )
        return cosmo, survey, pop_params, sky_params, mark_params

    def decode_mixture(self, coord: jnp.ndarray):
        """Decode ``coord`` into the K-catalog mixture components.

        Returns ``(cosmo, surveys, pop_params, sky_params, mark_params_all,
        log_w)`` where ``surveys`` is a length-``n_catalogs`` tuple of
        :class:`SurveyParams` (catalog 1 identical to :meth:`decode`; catalogs
        ``2..K`` built from the ``_c{k}`` suffixed survey labels, falling back
        to the shared fiducials, with ``wl_params=None``),
        ``mark_params_all`` is the length-K tuple of per-catalog eta vectors
        (element 0 == :meth:`decode`'s ``mark_params``; catalogs 2..K decoded
        from the ``eta_<mark>_c{k}`` labels), and ``log_w`` is the ``(K,)``
        array of log mixture weights from the sampled sticks ``fcat_2..fcat_K``.
        """
        # Resolve the label dictionary EXACTLY ONCE and reuse it for both the
        # catalog-1 component (via _decode_from_values, the shared body of
        # decode()) and the suffixed _c{k} blocks below.  Catalog 1 is thus
        # bit-identical to the single-catalog decode().
        coord = jnp.asarray(coord)
        values = resolve_parameter_values(
            coord, self.sampled_labels, self.fixed_parameter_values
        )
        cosmo, survey1, pop_params, sky_params, mark_params = (
            self._decode_from_values(values)
        )

        def _get(label, default):
            return values[label] if label in values else default

        surveys = [survey1]
        for k in range(2, self.n_catalogs + 1):
            sp_k = jnp.array([
                _get(f"{label}_c{k}", float(SURVEY_PARAMS_FID[i]))
                for i, label in enumerate(self.survey_labels)
            ])
            surveys.append(SurveyParams(
                n0=10.0 ** sp_k[0],
                z50=sp_k[1],
                w=sp_k[2],
                delta=sp_k[3],
                b_miss=sp_k[4],
                alpha_miss=sp_k[5],
                sigma_kde=sp_k[6],
                complete_empty_pixel_policy=self.complete_empty_pixel_policy,
                z_depth=self.z_depths[k - 1] if len(self.z_depths) >= k else None,
                wl_params=None,
            ))

        # Per-catalog eta blocks: catalog 1 is decode()'s vector verbatim;
        # catalogs 2..K read the suffixed labels (fiducial eta = 0 fallback).
        mark_params_all = [mark_params]
        for k in range(2, self.n_catalogs + 1):
            lbls = (
                self.mark_labels_by_catalog[k - 1]
                if len(self.mark_labels_by_catalog) >= k else ()
            )
            fids = (
                self.mark_fid_by_catalog[k - 1]
                if len(self.mark_fid_by_catalog) >= k else ()
            )
            mark_params_all.append(jnp.array([
                _get(f"{lbl}_c{k}", fids[i] if i < len(fids) else 0.0)
                for i, lbl in enumerate(lbls)
            ]))

        # Sticks fcat_2..fcat_K are always present in the coordinate/fixed values
        # for K >= 2 (build_parameter_space appends them); the default is a
        # defensive fallback only.
        sticks = jnp.array([
            _get(f"fcat_{m}", 0.0) for m in range(2, self.n_catalogs + 1)
        ])
        log_w = _sticks_to_log_weights(sticks)
        return (
            cosmo, tuple(surveys), pop_params, sky_params,
            tuple(mark_params_all), log_w,
        )


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
    n_catalogs = int(getattr(opts, "n_catalogs", 1))
    mark_names_by_catalog = getattr(opts, "mark_names_by_catalog", None)
    if mark_names_by_catalog is None:
        mark_names_by_catalog = (mark_names,) + ((),) * (n_catalogs - 1)
    mark_names_by_catalog = tuple(
        tuple(names or ()) for names in mark_names_by_catalog
    )
    # Per-catalog Q_LSS activity (drops b_miss_c{k} from the catalogs carrying a
    # Q table).  Prefer the CLI-resolved per-catalog tuple; fall back to the
    # legacy scalar bool (broadcast to all catalogs) for older opts / callers
    # that never set it -- this fallback is the P0.1 net so a decoder built from
    # a scalar-only opts still matches the sampler labels for the K=1 Q case.
    lss_completion_active = getattr(opts, "lss_completion_active_by_catalog", None)
    if lss_completion_active is None:
        lss_completion_active = getattr(opts, "lss_completion_active", False)
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
        n_catalogs=n_catalogs,
        lss_completion_active=lss_completion_active,
        mark_names_by_catalog=mark_names_by_catalog,
    )

    # Fail-fast net for CLI/decoder flag drift (the P0.1 bug class): the CLI
    # records the exact sampler labels on opts; if the decoder re-derives a
    # different set the two would silently disagree and resolve_parameter_values
    # would crash at the first likelihood call.  Surface it here with both
    # sequences instead.
    expected = getattr(opts, "expected_sampled_labels", None)
    if expected is not None and tuple(expected) != tuple(sampled_labels):
        raise ValueError(
            "Decoder sampled labels diverge from the CLI-resolved sampler "
            "labels (a flag resolved differently between build_parameter_space "
            "at the CLI and build_parameter_decoder -- e.g. per-catalog Q_LSS "
            "activity or a fix_* flag).\n"
            f"  CLI-resolved (sampler): {list(expected)}\n"
            f"  decoder-resolved:       {list(sampled_labels)}"
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
        # Resolved per-catalog survey z_depth (CLI --survey_z_depth override >
        # per-catalog file attr > None), computed host-side in the CLI before
        # the likelihood is built. Absent on bare/legacy ``opts`` (e.g. tests
        # that never set it) -> empty tuple -> every catalog's z_depth is None
        # (the legacy full-grid missing-galaxy budget).
        z_depths=tuple(getattr(opts, "resolved_survey_z_depths", None) or ()),
        sky_labels=tuple(sky_labels),
        sky_params_fid=tuple(float(v) for v in get_fixed_sky_params(sky_model)),
        mark_labels=tuple(mark_labels),
        mark_params_fid=tuple(
            float(v)
            for v in mark_fiducial(
                mark_model if mark_names_by_catalog[0] else "none",
                mark_names_by_catalog[0],
            )
        ),
        mark_labels_by_catalog=tuple(
            tuple(f"eta_{name}" for name in names)
            for names in mark_names_by_catalog
        ),
        mark_fid_by_catalog=tuple(
            tuple(
                float(v)
                for v in mark_fiducial(mark_model if names else "none", names)
            )
            for names in mark_names_by_catalog
        ),
        wl_params=(
            wl_params
            if getattr(opts, "universe_model", None) == "spectral_sirens_wl"
            else None
        ),
        n_catalogs=n_catalogs,
    )
