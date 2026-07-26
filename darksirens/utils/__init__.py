"""Shared numerical utilities: cosmology grids, 2-D interpolation, plotting.

This ``__init__`` exists for the same reason as the root one: without it
``setuptools.find_packages(include=["darksirens", "darksirens.*"])`` never
returns ``darksirens.utils``, so ``pip install .`` (sdist and wheel alike)
ships ZERO of the modules in this directory and every import of
``darksirens.utils.cosmology`` — reached from the likelihood core, the priors,
the redshift layer and five CLIs — fails with ``ModuleNotFoundError``.  The
source tree hid the defect because ``darksirens.utils`` also resolves as a
PEP-420 namespace portion under the checkout root that ``pip install -e .``
puts on ``sys.path``.  ``tests/test_packaging_contract.py`` guards it for the
whole package tree.

Deliberately empty of re-exports: ``cosmology`` builds interpolation grids at
import time, so importing ``darksirens.utils`` must stay side-effect free.
Import the submodules explicitly (``from darksirens.utils.cosmology import
z_of_dL``).
"""
