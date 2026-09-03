# io (`darksirens.io`)

Reading and writing the run directory's artifacts. Both modules stay free of
JAX at import time so external readers and the checkpoint completion probe can
use them.

## `darksirens.io.results`

Writes `results.hdf5` atomically, together with the dead-point datasets and the
TinyNS metadata and diagnostics, and reports whether a results file is
complete.

```{eval-rst}
.. automodule:: darksirens.io.results
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.io.settings`

Writes `settings.json`: the resolved run options plus the code-identity
(package versions, git commit) and environment blocks.

```{eval-rst}
.. automodule:: darksirens.io.settings
   :members:
   :undoc-members:
   :show-inheritance:
```
