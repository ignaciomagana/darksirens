"""
pop_extractor.py
----------------
make_pop_extractor(settings) — single source of truth for extracting the
population parameter sub-vector from the flat sampled coordinate vector theta.

Multitracer post-processing
---------------------------
``catalog_sticks_to_weights`` / ``fcat_to_component_fraction`` turn the sampled
catalog-mixture sticks ``fcat_2 .. fcat_K`` into the interpretable per-catalog
mixture weights ``w_1 .. w_K`` (host fractions under the FIELD convention),
exactly reproducing the decoder's own ordering so a posterior chain of sticks
maps to the same weights the likelihood saw.
"""

from __future__ import annotations
import jax.numpy as jnp
from darksirens.gw.populations import pop_model_prior_parser, get_fixed_population_params
from darksirens.inference.parameters import _sticks_to_log_weights


def catalog_sticks_to_weights(v: jnp.ndarray) -> jnp.ndarray:
    """Catalog-mixture sticks ``fcat_2..fcat_K`` -> weights ``[w_1, .., w_K]``.

    Row-wise ``exp`` of :func:`darksirens.inference.parameters._sticks_to_log_weights`
    — the SAME stick-breaking construction the likelihood decoder
    (``ParameterDecoder.decode_mixture``, ``parameters.py``) applies to the sampled
    sticks — so a posterior chain of sticks maps to exactly the weights the
    likelihood mixture used.

    Ordering (REMAINDER-FIRST): the returned weights are ``[w_1, w_2, ..., w_K]``
    with ``w_1`` the stick-breaking *remainder* ``prod_j (1 - v_j)`` and
    ``w_m = v_{m-1} * prod_{i<m-1}(1 - v_i)`` for ``m = 2..K``, i.e. ``fcat_m`` is
    the stick ``v_{m-1}``.  This matches ``decode_mixture`` /
    ``_sticks_to_log_weights`` and is DELIBERATELY NOT the population-grammar
    stick helper (``darksirens.gw.populations.base._stick_breaking_weights``),
    which orders the remainder LAST — a different convention that would permute
    the catalogs.

    K = 2 special case: there is a single stick ``fcat_2`` and the weights are
    ``[1 - fcat_2, fcat_2]``, so ``w_2`` (the second catalog's host fraction)
    IS ``fcat_2`` exactly.

    Parameters
    ----------
    v : array, shape ``(K-1,)`` or ``(n_samples, K-1)``
        The sampled catalog sticks ``fcat_2..fcat_K`` (each in ``[0, 1]``).  A 1-D
        input is treated as a single sample; a 2-D input is mapped row-wise.

    Returns
    -------
    weights : array, shape ``(K,)`` or ``(n_samples, K)``
        The per-catalog mixture weights ``[w_1, .., w_K]`` (non-negative, each row
        summing to 1).  Boundary sticks (exactly 0 or 1) yield a weight of exactly
        0 for the emptied catalog, consistent with the ``-inf`` log-weight the
        decoder feeds the mixture ``logsumexp``.
    """
    v = jnp.asarray(v)
    if v.ndim == 0:
        v = v[None]
    if v.ndim == 1:
        return jnp.exp(_sticks_to_log_weights(v))
    if v.ndim == 2:
        from jax import vmap
        return jnp.exp(vmap(_sticks_to_log_weights)(v))
    raise ValueError(
        f"catalog_sticks_to_weights expects a (K-1,) or (n_samples, K-1) array; "
        f"got ndim={v.ndim}."
    )


def fcat_to_component_fraction(weights: jnp.ndarray, component_index: int) -> jnp.ndarray:
    """Select one catalog's mixture weight column: the population host fraction.

    A trivial selector on the output of :func:`catalog_sticks_to_weights`, named
    for its physical meaning: under the FIELD-convention catalog sky weighting the
    mixture weight ``w_k`` is the fraction of GW HOSTS drawn from catalog ``k``'s
    tracer population.  For the canonical AGN-vs-galaxy K=2 analysis
    ``fcat_to_component_fraction(weights, 1)`` returns ``w_2`` — the AGN host
    fraction ``f_agn`` (a.k.a. ``alpha_AGN``), the estimand of the gws-agn campaign.

    Parameters
    ----------
    weights : array, shape ``(K,)`` or ``(n_samples, K)``
        Per-catalog weights from :func:`catalog_sticks_to_weights`.
    component_index : int
        The 0-based catalog column to select (``0`` -> ``w_1``, ``1`` -> ``w_2``,
        ...).  Negative indices count from the end, NumPy-style.

    Returns
    -------
    fraction : array, shape ``()`` or ``(n_samples,)``
        The selected catalog's host fraction, one value per sample.
    """
    weights = jnp.asarray(weights)
    return weights[..., component_index]


def make_pop_extractor(settings: dict):
    """
    Return a JAX-compatible function  pop_theta = extractor(theta).

    Parameters
    ----------
    settings : dict  (loaded from the run's settings.json)
        Required key: "pop_model"
        Optional keys: "shared_beta", "shared_spin", "shared_gamma",
                       "fix_population", "fix_cosmology", "fix_de",
                       "fixed_de", "fix_survey", "fixed_parameter_values"

    Returns
    -------
    extractor : callable
        theta (1-D sampled coordinate vector) → pop_params (1-D array).
        Safe inside jax.jit and jax.vmap.
    """
    pop_model_name         = settings["pop_model"]
    shared_beta            = bool(settings.get("shared_beta", True))
    shared_spin            = bool(settings.get("shared_spin", True))
    shared_gamma           = bool(settings.get("shared_gamma", True))
    fix_population         = bool(settings.get("fix_population", False))
    fix_cosmology          = bool(settings.get("fix_cosmology",  False))
    fix_survey             = bool(settings.get("fix_survey",     False))
    fixed_parameter_values = settings.get("fixed_parameter_values") or {}

    # Fast path: entire population block is fixed.  Individual fixed values
    # still override the fiducial fixed-population vector.
    if fix_population:
        fixed_array = get_fixed_population_params(
            pop_model_name,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
        )
        _pop_lower, _pop_upper, pop_labels, _pop_kinds, _model_name = pop_model_prior_parser(
            pop_model_name,
            shared_beta=shared_beta,
            shared_spin=shared_spin,
            shared_gamma=shared_gamma,
        )
        overrides = [
            float(fixed_parameter_values[label])
            if label in fixed_parameter_values else fixed_array[idx]
            for idx, label in enumerate(pop_labels)
        ]
        fixed_array = jnp.array(overrides, dtype=fixed_array.dtype)
        def extractor_fixed(theta):
            return fixed_array
        return extractor_fixed

    # General path: use build_parameter_space as the ordering oracle.
    # This is the same function make_likelihood calls, so the two are
    # guaranteed to agree on parameter ordering.
    from darksirens.inference.prior import build_parameter_space

    # Per-catalog Q_LSS activity: prefer the authoritative per-catalog tuple the
    # CLI records (a JSON round-trip yields a list -> tuple of bool); fall back
    # to the legacy scalar bool for older settings.json without the new key.
    # Threading the wrong flag would drop (or keep) b_miss inconsistently vs the
    # sampled space and misalign the pop-coordinate indices.
    _lss_by_cat = settings.get("lss_completion_active_by_catalog")
    if _lss_by_cat is not None:
        lss_completion_active = tuple(bool(x) for x in _lss_by_cat)
    else:
        lss_completion_active = bool(settings.get("lss_completion_active", False))

    (
        labels,          # ordered list of sampled labels
        _lower,
        _upper,
        _n_pop_eff,
        pop_labels,      # population labels in model order
        _survey_labels,
        _cosmo_labels,
        _n_cosmo_eff,
        _n_survey_eff,
        _model_name,
        _fixed_parameter_statuses,
        _prior_kinds,
        _sky_labels,
        _mark_labels,
    ) = build_parameter_space(
        pop_model              = pop_model_name,
        fix_population         = fix_population,
        fix_cosmology          = fix_cosmology,
        fix_survey             = fix_survey,
        fix_de                 = bool(
            settings.get("fix_de", settings.get("fixed_de", False))
        ),
        fixed_parameter_values = fixed_parameter_values,
        universe_model         = settings.get("universe_model"),
        shared_beta            = shared_beta,
        shared_spin            = shared_spin,
        shared_gamma           = shared_gamma,
        sky_model              = settings.get("sky_model", "isotropic"),
        mark_model             = settings.get("mark_model", "none"),
        mark_names             = tuple(settings.get("mark_names", ()) or ()),
        # K>=2 runs insert per-catalog `_c{k}` survey blocks and `fcat_*`
        # sticks, and a Q-active run drops b_miss from the survey block;
        # omitting these made build_parameter_space reject such labels in
        # fixed_parameter_values (KeyError) for any K>=2 chain.
        n_catalogs             = int(settings.get("n_catalogs", 1) or 1),
        lss_completion_active  = lss_completion_active,
        # Mirror the CLI/decoder use_lss gate: with --use_lss off, b_miss is
        # dropped from the sampled survey block and must not shift labels here.
        # Default True reproduces pre-gate archived runs, which sampled b_miss
        # regardless of the flag (see issue #288).
        use_lss                = bool(settings.get("use_LSS", True)),
        mark_names_by_catalog  = (
            tuple(tuple(n) for n in settings["mark_names_by_catalog"])
            if settings.get("mark_names_by_catalog") else None
        ),
        # Selection mode gates M0hat/sigma_M into the sampled survey block;
        # rebuilding without it would silently misalign any label AFTER the
        # survey block for archived selection chains.
        c_mode                 = settings.get("c_mode", "per_pixel") or "per_pixel",
        # The selection FAMILY gates which theta labels the survey block carries
        # (M0hat/sigma_M vs Mstar_hat/alpha); rebuilding without it would
        # misalign every label after the survey block for archived schechter
        # chains.
        selection_family       = settings.get("selection_family", "gaussian") or "gaussian",
    )

    label_to_coord_idx = {label: idx for idx, label in enumerate(labels)}

    pop_coord_indices = []
    pop_fixed_mask    = []
    pop_fixed_values  = []

    for label in pop_labels:
        if label in fixed_parameter_values:
            pop_coord_indices.append(0)          # dummy; masked below
            pop_fixed_mask.append(True)
            pop_fixed_values.append(float(fixed_parameter_values[label]))
        else:
            if label not in label_to_coord_idx:
                raise KeyError(
                    f"Population label '{label}' not found in sampled coordinate "
                    f"labels — check fixed_parameter_values / fix_* flags."
                )
            pop_coord_indices.append(label_to_coord_idx[label])
            pop_fixed_mask.append(False)
            pop_fixed_values.append(0.0)

    idx_jnp   = jnp.array(pop_coord_indices, dtype=jnp.int32)
    mask_jnp  = jnp.array(pop_fixed_mask,    dtype=bool)
    fixed_jnp = jnp.array(pop_fixed_values,  dtype=jnp.float64)

    def extractor(theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mask_jnp, fixed_jnp, theta[idx_jnp])

    return extractor