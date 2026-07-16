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
