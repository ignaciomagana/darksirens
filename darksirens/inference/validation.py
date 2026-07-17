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
