"""
models.py
---------
Marked-host efficiency models: the factor ``h(m | eta)`` that reweights each
catalog galaxy by its BBH-host efficiency given its marks
``m = {logM*, sSFR, Z, colour, ...}``.

Each model exposes the same duck type the redshift-prior builder consumes
(mirroring :mod:`darksirens.sky.models`):

* ``param_specs`` / ``prior_bounds()`` — the sampled ``eta`` block.
* ``log_h(em_catalog, eta) -> (N_rows, N_max_gals)`` — per-galaxy log host
  efficiency, read from the (z-centred) mark fields the model uses.

Marks are z-centred at load (``m_tilde = m - E[m|z]``), so
``log h = Σ_k eta_k m_tilde_k`` measures host preference **at fixed redshift**
and does not mimic ``R(z)/H0/gamma``.  ``h ≡ 1`` (``eta = 0``) recovers the
galaxy-count host model.
"""
from __future__ import annotations

import jax.numpy as jnp

from darksirens.gw.populations.base import ParamSpec, pack_specs

#: Canonical mark name -> EMCatalog attribute holding the (z-centred) array.
MARK_FIELDS = {
    "logmstar": "mark_logmstar",
    "logssfr": "mark_logssfr",
    "metallicity": "mark_metallicity",
    "color": "mark_color",
}

#: LaTeX for each mark's coefficient (for plots / labels).
MARK_LATEX = {
    "logmstar": r"$\eta_{\log M_\star}$",
    "logssfr": r"$\eta_{\log\,\mathrm{sSFR}}$",
    "metallicity": r"$\eta_{Z}$",
    "color": r"$\eta_{g-r}$",
}


def available_marks(em_catalog) -> tuple:
    """Canonical names of marks present (non-``None``) on ``em_catalog``,
    in canonical order."""
    return tuple(
        name for name, field in MARK_FIELDS.items()
        if getattr(em_catalog, field, None) is not None
    )


#: Fraction of real galaxies allowed to sit on the ``log h`` clip rail at the
#: ``eta`` prior edge before the mark table is rejected.  Past this the clip —
#: not the data — is what shapes the ``eta`` posterior: ``log h`` is locally
#: constant in ``eta`` for every pinned galaxy, so the posterior flattens to the
#: model's designed null and a run "converges" on a prior with no information in
#: it.  Half is deliberately loose; a properly z-centred table saturates only
#: its outlier tail (percent level), while an UNcentred one (raw ``logM* ~ 10.5``
#: against ``eta_bound = 5`` and a rail at 7) pins essentially every galaxy.
MARK_SATURATION_MAX_FRACTION: float = 0.5


def _log_h_clip() -> float:
    """The ``+-clip`` applied to ``log h`` by the redshift-prior builder.

    Imported lazily: :mod:`darksirens.redshift.prior` reaches into
    :mod:`darksirens.marks` for the mark-model parsers, so a module-level import
    here would close the cycle.
    """
    from darksirens.redshift.prior import _LOG_H_CLIP

    return float(_LOG_H_CLIP)


def _saturated_fraction(abs_sum, real, eta_bound, clip):
    """Fraction of ``real`` entries whose worst-case ``|log h|`` hits ``clip``.

    ``abs_sum`` is ``Sum_k |m_k|`` per galaxy.  Since ``log h = Sum_k eta_k m_k``
    and each ``eta_k`` ranges independently over ``[-eta_bound, eta_bound]``, the
    largest ``|log h|`` reachable inside the prior is ``eta_bound * Sum_k |m_k|``
    (take each ``eta_k``'s sign to match its mark's).  A galaxy is counted as
    saturating when that worst case reaches the rail.
    """
    n_real = jnp.sum(real)
    sat = (eta_bound * abs_sum >= clip) & real
    return jnp.where(n_real > 0, jnp.sum(sat) / jnp.maximum(n_real, 1), 0.0)


def check_marks_centred(model, mark_arrays, ngals, *, where):
    """Raise if ``mark_arrays`` would pin ``log h`` to the clip rail.

    Marks must reach the marked-host model z-centred (``m_tilde = m - E[m|z]``,
    applied at load by
    :func:`darksirens.catalogs.marks.load_and_center_survey_marks`).  A table
    that skipped that step carries its raw zero point — ``logM*`` of order 10.5,
    say — and ``eta * m`` then exceeds the ``+-`` clip over the whole prior, so
    the clipped ``log h`` is constant in ``eta`` and the ``eta`` posterior goes
    flat.  Nothing downstream is wrong when that happens, which is the problem:
    the run looks healthy and measures nothing.  Check it here, eagerly, at
    build time, rather than reading a null posterior later.

    ``mark_arrays`` maps canonical mark name -> ``(N_rows, N_max_gals)`` array
    aligned to ``model.mark_names``; ``ngals`` is the ``(N_rows,)`` real-galaxy
    count that masks the padded slots.  ``where`` names the caller for the error.
    """
    mark_names = tuple(getattr(model, "mark_names", ()) or ())
    if not mark_names:
        return  # NoMarks: h == 1 by construction, no eta to keep alive.
    eta_bound = float(getattr(model, "eta_bound", 0.0))
    if eta_bound <= 0.0:
        return

    ngals = jnp.asarray(ngals)
    abs_sum = None
    per_mark = {}
    for name in mark_names:
        arr = mark_arrays.get(name)
        if arr is None:
            # A selected mark with no array at all is a DATA error, reported
            # downstream by _gather_marks with the field name and catalog.  Say
            # nothing here rather than preempt it with a saturation complaint
            # about a table that does not exist.
            return
        arr = jnp.asarray(arr)
        abs_arr = jnp.abs(arr)
        abs_sum = abs_arr if abs_sum is None else abs_sum + abs_arr
        per_mark[name] = arr
    real = jnp.arange(abs_sum.shape[1])[None, :] < ngals[:, None]

    clip = _log_h_clip()
    frac = float(_saturated_fraction(abs_sum, real, eta_bound, clip))
    if frac <= MARK_SATURATION_MAX_FRACTION:
        return

    detail = []
    for name, arr in per_mark.items():
        masked = jnp.where(real, arr, jnp.nan)
        detail.append(
            f"      {name}: mean={float(jnp.nanmean(masked)):+.3g}, "
            f"max|m|={float(jnp.nanmax(jnp.abs(masked))):.3g}"
        )
    raise ValueError(
        f"{where}: the marked-host model would be dead on these marks. "
        f"{100.0 * frac:.1f}% of real galaxies reach the |log h| <= {clip:g} "
        f"clip somewhere inside the eta prior (|eta_k| <= {eta_bound:g}), so "
        "the clipped log h is constant in eta over most of the catalog and the "
        "eta posterior would flatten to the h == 1 null -- a run that looks "
        "converged and measures nothing.\n"
        "    Per-mark statistics over real galaxies:\n"
        + "\n".join(detail)
        + "\n    Marks must be z-centred (m - E[m|z]) before they reach the "
        "model; darksirens.catalogs.marks.load_and_center_survey_marks does "
        "this at load. A mean far from 0 means the raw zero point is still "
        "there (e.g. logM* ~ 10.5 dex instead of ~0)."
    )


def check_flat_marks_centred(model, values, *, where):
    """``check_marks_centred`` for the flat full-sky ``(N_gal, n_marks)`` table.

    These rows feed ``mu_miss`` (the survey-level host efficiency of the missing
    galaxies) through the same clip, so an uncentred flat table kills eta in the
    missing branch exactly as the per-pixel one does in the observed branch.
    Columns are ordered by ``model.mark_names``; every row is a real galaxy.
    """
    mark_names = tuple(getattr(model, "mark_names", ()) or ())
    if not mark_names or values is None:
        return
    eta_bound = float(getattr(model, "eta_bound", 0.0))
    if eta_bound <= 0.0:
        return

    values = jnp.asarray(values)
    abs_sum = jnp.sum(jnp.abs(values), axis=1)
    real = jnp.ones_like(abs_sum, dtype=bool)
    clip = _log_h_clip()
    frac = float(_saturated_fraction(abs_sum, real, eta_bound, clip))
    if frac <= MARK_SATURATION_MAX_FRACTION:
        return

    detail = "\n".join(
        f"      {name}: mean={float(jnp.mean(values[:, k])):+.3g}, "
        f"max|m|={float(jnp.max(jnp.abs(values[:, k]))):.3g}"
        for k, name in enumerate(mark_names)
    )
    raise ValueError(
        f"{where}: the full-sky flat mark table would kill mu_miss. "
        f"{100.0 * frac:.1f}% of galaxies reach the |log h| <= {clip:g} clip "
        f"inside the eta prior (|eta_k| <= {eta_bound:g}), so E_obs[h|z] is "
        "constant in eta and the missing branch carries no host preference.\n"
        "    Per-mark statistics:\n" + detail + "\n"
        "    Build these rows from the SAME z-centred marks as the per-pixel "
        "tables (darksirens.redshift.completion.build_field_mark_inputs)."
    )


def _gather_marks(em_catalog, mark_names):
    arrs = []
    for name in mark_names:
        field = MARK_FIELDS[name]
        arr = getattr(em_catalog, field, None)
        if arr is None:
            raise ValueError(
                f"mark '{name}' requested but EMCatalog.{field} is None "
                "(this mark was not provided in the catalog)."
            )
        arrs.append(jnp.asarray(arr))
    return arrs  # list of (N_rows, N_max_gals)


class NoMarks:
    """``h = 1`` — no mark reweighting (the galaxy-count host model)."""

    param_specs: list = []

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_h(self, em_catalog, eta):
        return jnp.zeros_like(jnp.asarray(em_catalog.zgals))

    def log_h_flat(self, values, eta):
        """``h = 1`` on a flat (N_gal, n_marks) value matrix."""
        return jnp.zeros(jnp.asarray(values).shape[0])


class LogLinearMarks:
    """``h = exp(Σ_k eta_k m_tilde_k)`` over the supplied (z-centred) marks.

    ``mark_names`` is the ordered subset of :data:`MARK_FIELDS` to use; the
    sampled block is one coefficient ``eta_<name>`` per mark (uniform on
    ``[-eta_bound, eta_bound]``).
    """

    def __init__(self, mark_names, eta_bound: float = 5.0):
        self.mark_names = tuple(mark_names)
        if not self.mark_names:
            raise ValueError(
                "LogLinearMarks needs at least one mark name; none were provided "
                "(is the catalog missing mark fields, or --marks empty?)."
            )
        unknown = [n for n in self.mark_names if n not in MARK_FIELDS]
        if unknown:
            raise ValueError(f"unknown mark(s) {unknown}; known: {tuple(MARK_FIELDS)}.")
        self._b = float(eta_bound)

    @property
    def eta_bound(self) -> float:
        """Half-width of the uniform ``eta_k`` prior (the saturation check's
        worst case runs to this edge)."""
        return self._b

    @property
    def param_specs(self):
        b = self._b
        # ASCII label (== name) so it is typeable in --prior_overrides and safe
        # as a sampler site name; MARK_LATEX is available for display/plots.
        return [
            ParamSpec(f"eta_{name}", -b, b, name=f"eta_{name}")
            for name in self.mark_names
        ]

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_h(self, em_catalog, eta):
        arrs = _gather_marks(em_catalog, self.mark_names)  # list of (N_rows, N_max)
        eta = jnp.asarray(eta)
        log_h = jnp.zeros_like(arrs[0])
        for k, arr in enumerate(arrs):
            log_h = log_h + eta[k] * arr
        return log_h  # (N_rows, N_max_gals)

    def log_h_flat(self, values, eta):
        """Same log-linear form on a flat (N_gal, n_marks) value matrix.

        ``values`` columns are ordered by ``self.mark_names`` (the field-
        normalizer flat full-sky marks); returns ``(N_gal,)``.
        """
        values = jnp.asarray(values)
        eta = jnp.asarray(eta, dtype=values.dtype)
        return values @ eta
