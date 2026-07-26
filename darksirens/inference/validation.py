"""Canonical post-load validation for a multitracer inference run.

A single place, called once by the CLI right after data loading, that asserts
the per-catalog config sequences resolved on ``opts`` all agree with
``opts.n_catalogs`` (and, for a K>=2 mixture, that exactly ``n_catalogs``
compact catalog bundles were loaded).  These are cheap host-side length checks
whose only job is to turn a silently-misaligned config into an actionable
error before the parameter space / likelihood are built.

The checks are deliberately tolerant of bare/legacy ``opts`` (tests build
minimal ``SimpleNamespace`` opts): every per-catalog sequence is validated only
``if set`` via ``getattr`` defaults, so a valid or minimal config passes
untouched and no behaviour changes.
"""


def validate_multitracer_run(opts, data) -> None:
    """Assert per-catalog config lengths against ``opts.n_catalogs``.

    Raises ``ValueError`` naming the offending length and the expected
    ``n_catalogs`` when a resolved per-catalog sequence is misaligned.
    """
    n_catalogs = int(getattr(opts, "n_catalogs", 1))

    # Resolved per-catalog survey z-depths: one per catalog when any survey was
    # loaded (empty for survey-free models -- skipped by the truthiness guard).
    z_depths = getattr(opts, "resolved_survey_z_depths", None)
    if z_depths:
        if len(z_depths) != n_catalogs:
            raise ValueError(
                "resolved_survey_z_depths has "
                f"{len(z_depths)} entries but n_catalogs={n_catalogs}; "
                "each catalog must resolve exactly one survey z_depth."
            )

    # Positionally-aligned external completion paths: 0 or exactly n_catalogs
    # (already CLI-checked; re-asserted cheaply as the canonical invariant).
    lss_completions = getattr(opts, "lss_completions", None)
    if lss_completions is not None:
        if len(lss_completions) not in (0, n_catalogs):
            raise ValueError(
                "lss_completions has "
                f"{len(lss_completions)} entries but n_catalogs={n_catalogs}; "
                "pass 0 or exactly n_catalogs completion paths."
            )

    # Per-catalog mark selections (resolved pre-load for K>=2; None for a bare
    # or K=1 opts before mark resolution -- skipped by the None guard).
    mark_names_by_catalog = getattr(opts, "mark_names_by_catalog", None)
    if mark_names_by_catalog is not None:
        if len(mark_names_by_catalog) != n_catalogs:
            raise ValueError(
                "mark_names_by_catalog has "
                f"{len(mark_names_by_catalog)} entries but "
                f"n_catalogs={n_catalogs}; each catalog needs its own mark "
                "list (possibly empty)."
            )

    # K>=2 mixture: exactly n_catalogs compact catalog bundles must be loaded.
    if n_catalogs >= 2:
        catalogs = data.get("catalogs") if hasattr(data, "get") else None
        n_bundles = len(catalogs) if catalogs is not None else 0
        if n_bundles != n_catalogs:
            raise ValueError(
                f"loaded {n_bundles} catalog bundle(s) but n_catalogs="
                f"{n_catalogs}; the K-catalog mixture needs one compact bundle "
                "per survey path."
            )

        # FIELD-convention mixtures require a COMMON pixelisation.
        #
        # Under catalog_sky_weighting='field' each catalog's redshift prior is a
        # per-PIXEL probability MASS -- the numerator is normalised by the
        # survey-global Z(theta) and never divided by the pixel solid angle
        # (grep apix in redshift/prior.py: no hits).  The mixture combines those
        # masses as logsumexp_k [log w_k + log p_k(z | pix_k)], so for one sky
        # direction a catalog with larger pixels contributes proportionally more
        # simply because its pixel subtends more solid angle.  The effective
        # weight is w_k * Omega_k, not w_k, and the sampled host fraction
        # f_cat,k absorbs the pixel-area ratio -- a silent bias with no
        # diagnostic, since every individual catalog is self-consistent.
        #
        # Nothing else prevents this: each catalog loads its own nside from its
        # own file and the factory deliberately uses per-catalog apix for the
        # completion densities.  Rejecting the mixed-resolution case keeps the
        # 'field' estimand exactly as documented (f_cat,k is the catalog's GW
        # host fraction) and leaves every equal-nside run bit-identical.
        weighting = str(getattr(opts, "catalog_sky_weighting", "") or "")
        if weighting != "conditional":
            nsides = [b.get("nside") for b in catalogs]
            distinct = sorted({int(n) for n in nsides if n is not None})
            if len(distinct) > 1:
                raise ValueError(
                    "K>=2 mixtures under 'field' sky weighting require all "
                    "catalogs to share one HEALPix nside, but the loaded "
                    f"bundles have nside={nsides}.\n"
                    "The field estimand's per-pixel prior is a probability MASS, "
                    "so mixing pixelisations weights each catalog by its pixel "
                    "area: the effective mixture weight becomes w_k * Omega_k "
                    "and the inferred host fraction f_cat is biased by the "
                    "pixel-area ratio.\n"
                    "Re-run darksirens_pixelate on the offending survey(s) at a "
                    "common --nside."
                )
