"""Sphinx configuration for the darksirens documentation."""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "darksirens"
author = "Ignacio Magana Hernandez"
copyright = f"{datetime.now().year}, {author}"
release = "0.0.1"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "darksirens documentation"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# The scientific runtime stack is heavy (JAX/GPU wheels, survey data) and is not
# present on hosted docs builders.  We mock ONLY the dependencies that are not
# importable in the current environment: on a full ``darksirens-dev`` install all
# deps are present, so autodoc imports the real modules and renders complete
# signatures; on a hosted/minimal build the missing packages are mocked so the
# API pages still generate.  (A blanket mock breaks module-level code such as
# ``float(Planck15.Om0)`` in ``utils.cosmology``, which a Mock cannot satisfy.)
_OPTIONAL_RUNTIME_DEPS = [
    "arviz",
    "astropy",
    "corner",
    "dynesty",
    "emcee",
    "h5py",
    "gwdistributions",
    "healpy",
    "jax",
    "jaxlib",
    "jaxns",
    "matplotlib",
    "numpyro",
    "seaborn",
    "scipy",
    "tinygp",
    "tqdm",
]
autodoc_mock_imports = [
    name for name in _OPTIONAL_RUNTIME_DEPS
    if importlib.util.find_spec(name) is None
]

# ``dollarmath`` enables ``$...$`` / ``$$...$$`` LaTeX math; ``amsmath`` enables
# AMS environments (``align``, ``cases``, ...).  Math is rendered client-side by
# MathJax (Sphinx default) for the HTML build.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
# Generate anchors for headings (depth <= 3) so pages can cross-reference
# sections like concepts.md#population-models.
myst_heading_anchors = 3

# Keep examples readable in both light and dark themes.
pygments_style = "sphinx"
pygments_dark_style = "monokai"
