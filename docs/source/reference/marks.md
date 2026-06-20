# Marked-host model (`darksirens.marks`)

The `marks` subpackage answers *which* galaxies host compact-binary mergers. If
the catalog carries per-galaxy marks $m_g$ (stellar mass, sSFR, metallicity,
colour), a sampled BBH-host efficiency $h(m_g\mid\eta)$ reweights each galaxy's
contribution to the dark-siren redshift prior — a marked Cox process layered on
top of the LSS completion. The model and its identifiability are derived on the
[Theory & methods](../theory.md) page.

## `darksirens.marks`

The package `__init__` exposes the registry helpers used by the parameter space
and the likelihood threading.

```{automodule} darksirens.marks
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.marks.models`

The mark-model classes. `NoMarks` is the $\eta$-empty default (a bit-for-bit
no-op). `LogLinearMarks` implements

$$
h(m_g \mid \eta) = \exp\!\Big(\textstyle\sum_k \eta_k\, \tilde m_{k,g}\Big),
$$

over the available marks, where $\tilde m_g = m_g - \mathbb{E}[m\mid z_g]$ are
**redshift-centred** so $\eta$ measures host preference at fixed $z$ and does not
mimic $R(z)/H_0/\gamma$. `log_h(em_catalog, eta)` returns the per-galaxy
log-efficiency used by the marked catalog kernel ([`em.catalog`](em.md)) and the
missing-branch efficiency $\mu_{\rm miss}(z\mid\eta)=\mathbb{E}_{\rm obs}[h\mid z]$.

```{automodule} darksirens.marks.models
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.marks.registry`

Maps `--mark_model` names (`none`, `loglinear`) to model factories and exposes
`mark_model_parser`, `mark_model_prior_parser(mark_model, mark_names)` (one
$\eta_k$ per used mark), and `mark_fiducial` ($\eta=0$). The set of marks is
resolved at load time as the present mark fields intersected with the optional
`--marks` selection.

```{automodule} darksirens.marks.registry
:members:
:undoc-members:
:show-inheritance:
```
