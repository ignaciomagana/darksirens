"""Q_LSS provenance enforcement.

A prebuilt LSS completion table ``Q`` is not a free function of the inference
parameters: it is a *fit*, conditioned at build time on a specific cosmology,
comoving density ``n0``, density evolution ``delta`` and field bias ``b_miss``
(``cli/build_lognormal_completion.py`` stamps all of them into the file's
diagnostics group).  The builder is explicit about the failure mode:

    a mis-set n0 cannot be separated from completeness and is absorbed into Q
    with spurious redshift structure (which then biases any downstream H0
    inference)

Loading Q used to only *warn* about this while the inference happily kept
sampling ``log10n0``/``delta``, so a mismatch degraded H0 silently.  This module
turns that into a hard error.  It runs after the parameter space is built,
because that is the first point where both the Q fiducials and the final
sampled-label set are known — the loader itself runs too early to see them.

**H0 is deliberately exempt.**  ``fiducial_H0`` is stamped like the others, but
H0 is the quantity being measured and is sampled in every dark-siren run, so
treating it as fatal would make prebuilt Q tables unusable rather than safe.
Q enters as a dimensionless density ratio on a fixed comoving grid, which is the
construction that makes an H0 offset a second-order effect on it, whereas an n0
offset is absorbed at first order.  H0 keeps the loud warning.  If that
approximation is ever shown to matter at the precision of a given analysis, the
right fix is to interpolate Q over H0, not to relax the check on the others.
"""

from __future__ import annotations

#: Q-conditioning parameters that MUST NOT vary against a prebuilt table, mapped
#: to the diagnostics key holding the build-time value.  ``None`` means the
#: builder does not stamp a value, so the parameter may only be *absent* from
#: the sampled set, not checked against a number.
_Q_CONDITIONED = {
    "log10n0": "fiducial_n0",     # stamped as n0; compared in log10
    "delta": "fiducial_delta",
    "b_miss": "bias_b_miss",
    "Om0": "fiducial_Om0",
    "w0": "fiducial_w0",
    "wa": "fiducial_wa",
}

#: Stamped but exempt — see the module docstring.
_Q_EXEMPT = ("H0",)

#: Fractional tolerance when comparing a fixed value against the build fiducial.
_RTOL = 1e-6
_ATOL = 1e-9


#: Cosmology is GLOBAL: one value per run, shared by every catalog.  Each
#: catalog's Q table must therefore agree with it independently.
_GLOBAL_PARAMS = ("Om0", "w0", "wa")

#: Survey parameters are PER CATALOG (``log10n0_c2`` belongs to catalog 2 and
#: must be checked against catalog 2's own table, never catalog 1's).
_PER_CATALOG_PARAMS = ("log10n0", "delta", "b_miss")


def _base_name(label):
    """Strip a per-catalog ``_c{k}`` suffix so ``log10n0_c2`` gates like ``log10n0``."""
    from darksirens.inference.prior import _survey_base_name

    return _survey_base_name(str(label))


def _catalog_index(label):
    """0-based catalog index encoded in a label's ``_c{k}`` suffix.

    Catalog 1 is unsuffixed; catalogs 2..K carry ``_c2``..``_cK``, so
    ``log10n0`` -> 0 and ``log10n0_c3`` -> 2.
    """
    from darksirens.inference.prior import _SURVEY_CATALOG_SUFFIX

    m = _SURVEY_CATALOG_SUFFIX.search(str(label))
    if m is None:
        return 0
    return int(m.group(0)[2:]) - 1


def _normalise_fiducials(fiducials):
    """Accept a single dict (K=1) or a per-catalog sequence; return a list."""
    if fiducials is None:
        return []
    if isinstance(fiducials, dict):
        return [fiducials]
    return list(fiducials)


def _build_value(fiducials, param):
    """Build-time value of ``param`` in the units the sampler uses, or ``None``."""
    import math

    key = _Q_CONDITIONED[param]
    if key not in fiducials:
        return None
    value = float(fiducials[key])
    if param == "log10n0":
        # The builder stamps the linear density; the sampler works in log10.
        if not (value > 0.0):
            return None
        return math.log10(value)
    return value


def check_lss_completion_provenance(
    fiducials,
    sampled_labels,
    fixed_parameter_values=None,
):
    """Raise unless every Q-conditioning parameter is pinned to its build value.

    Parameters
    ----------
    fiducials : dict, sequence of (dict or None), or None
        The ``lss_completion_fiducials`` entries produced by
        :func:`darksirens.catalogs.lss.maybe_load_lss_completion`.  For a
        K-catalog mixture pass ONE ENTRY PER CATALOG in catalog order, using
        ``None`` for a catalog that carries no Q table; a bare dict is treated
        as the single-catalog case.  ``None`` (or an all-``None`` sequence)
        means no Q table is loaded and the check is a no-op.

        Passing a single table for a K>=2 run would validate catalog 2..K's
        independent survey parameters against catalog 1's build values, which
        both rejects the supported "Q on one catalog only" configuration and
        leaves the other catalogs' tables unverified.
    sampled_labels : sequence of str
        The final sampled parameter labels (``build_parameter_space`` output).
    fixed_parameter_values : dict, optional
        Label -> value for parameters held fixed.

    Raises
    ------
    ValueError
        If a conditioning parameter is sampled, or is fixed to a value that
        differs from the build-time fiducial, or if the table predates the
        fiducial stamping and so cannot be verified at all.
    """
    per_catalog = _normalise_fiducials(fiducials)
    if not any(f for f in per_catalog):
        return

    fixed = {str(k): float(v) for k, v in (fixed_parameter_values or {}).items()}

    # Group the sampled labels by (base name, owning catalog).
    sampled_by = {}
    for label in sampled_labels:
        key = (_base_name(str(label)), _catalog_index(label))
        sampled_by.setdefault(key, []).append(str(label))

    problems = []
    for cat_idx, fid in enumerate(per_catalog):
        if not fid:
            continue
        path = fid.get("path", "<unknown>")
        tag = "" if len(per_catalog) == 1 else f" (catalog {cat_idx + 1})"

        stamped = [k for k in _Q_CONDITIONED.values() if k in fid]
        if not stamped:
            raise ValueError(
                f"LSS completion table '{path}'{tag} carries no build-time fiducials, "
                "so the parameters it is conditioned on (n0, delta, bias, cosmology) "
                "cannot be verified against this run's parameter space. Q is a fit "
                "conditioned on those values and a mismatch is absorbed into Q as "
                "spurious redshift structure, biasing H0. Rebuild the table with a "
                "current darksirens_build_lognormal_completion, which stamps them."
            )

        for param in _Q_CONDITIONED:
            build_value = _build_value(fid, param)
            # Cosmology is global (always catalog slot 0's label namespace);
            # survey parameters belong to THIS catalog.
            owner = 0 if param in _GLOBAL_PARAMS else cat_idx
            offenders = sampled_by.get((param, owner), [])

            if offenders:
                shown = "not stamped" if build_value is None else f"{build_value:.6g}"
                problems.append(
                    f"  - {', '.join(sorted(offenders))}: SAMPLED, but Q "
                    f"'{path}'{tag} was built at {param} = {shown}"
                )
                continue

            if build_value is None:
                continue

            # Not sampled: held at a fixed value (explicit or fiducial default).
            # Only an explicit override can disagree with the build value; an
            # absent entry means the decoder uses the package fiducial, which is
            # what the builder used too.
            for label, value in fixed.items():
                if _base_name(label) != param or _catalog_index(label) != owner:
                    continue
                if abs(value - build_value) > (_ATOL + _RTOL * abs(build_value)):
                    problems.append(
                        f"  - {label}: fixed at {value:.6g}, but Q '{path}'{tag} "
                        f"was built at {param} = {build_value:.6g}"
                    )

    if problems:
        raise ValueError(
            "Q_LSS provenance mismatch.\n"
            "An LSS completion table is a fit conditioned on fixed build-time "
            "values. Sampling or re-fixing those parameters does not propagate "
            "into Q: the mismatch is absorbed into the completion field as "
            "spurious redshift structure and biases H0 (see "
            "cli/build_lognormal_completion.py).\n\n"
            + "\n".join(problems)
            + "\n\nFix by either:\n"
            "  * fixing these parameters to the build values with "
            "--fixed_parameter_values, or\n"
            "  * rebuilding Q at the values you intend to use "
            "(darksirens_build_lognormal_completion --log10n0 ... --delta ...).\n"
            f"H0 is exempt from this check ({', '.join(_Q_EXEMPT)}); see "
            "darksirens/inference/q_provenance.py for why."
        )
