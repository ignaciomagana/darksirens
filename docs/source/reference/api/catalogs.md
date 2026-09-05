# catalogs (`darksirens.catalogs`)

Galaxy-catalog handling on the host side: loading a pixelated survey and its
per-galaxy marks, compacting it to the pixels the samples touch, and attaching
depth maps and LSS completion tables. The file layouts are described in
[Inputs](../../getting-started/inputs.md).

## `darksirens.catalogs.compact`

Compact catalog views used by inference data loading: reduce the padded
per-pixel arrays to the unique pixels the PE and selection samples occupy, and
validate the loaded shapes.

```{eval-rst}
.. automodule:: darksirens.catalogs.compact
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.catalogs.counterparts`

Builds the synthetic counterpart catalog for bright-siren runs from the
`--counterpart` RA/DEC/Z triplets.

```{eval-rst}
.. automodule:: darksirens.catalogs.counterparts
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.catalogs.depth_map`

The per-pixel selection fraction `f_p = 1 - masked_frac` read from a depth-map
artifact and degraded to the catalog nside by equal-area averaging.

```{eval-rst}
.. automodule:: darksirens.catalogs.depth_map
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.catalogs.io`

Loads pixelated survey HDF5 files and the optional per-galaxy mark and
property datasets, sorting rows by redshift for the windowed KDE.

```{eval-rst}
.. automodule:: darksirens.catalogs.io
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.catalogs.lss`

Loads and validates LSS completion tables, including an in-catalog
`/lss_completion` group and the per-catalog ensemble provenance a K>=2
marginalised run needs.

```{eval-rst}
.. automodule:: darksirens.catalogs.lss
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.catalogs.marks`

Loads per-galaxy marks and centres them in redshift, the form the loglinear
marked-host model expects.

```{eval-rst}
.. automodule:: darksirens.catalogs.marks
   :members:
   :undoc-members:
   :show-inheritance:
```
