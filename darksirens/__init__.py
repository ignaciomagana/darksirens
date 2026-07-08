"""darksirens: hierarchical gravitational-wave dark-siren cosmology with
incomplete galaxy catalogs (JAX).

This root ``__init__`` exists so classic ``setuptools.find_packages`` can
discover the package: without it, ``pip install .`` resolved darksirens as a
PEP-420 namespace package and installed ZERO modules (library review, sky/
architecture finding 2) — the source tree only worked through the editable
path, alongside whatever stale copy happened to sit on the namespace path.
Keep it import-side-effect free: submodules configure JAX explicitly via
``darksirens.core.jax_config.configure_jax_runtime()``.
"""

__version__ = "0.0.1"
