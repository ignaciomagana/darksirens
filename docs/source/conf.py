"""Sphinx configuration for the darksirens documentation."""

from __future__ import annotations

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
html_title = "darksirens documentation"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# The scientific runtime stack is intentionally mocked during hosted docs builds
# so API pages can be generated without GPU/JAX-specific wheels or survey data.
autodoc_mock_imports = [
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

myst_enable_extensions = ["colon_fence", "deflist"]
# Generate anchors for headings (depth <= 3) so pages can cross-reference
# sections like concepts.md#population-models.
myst_heading_anchors = 3

# Keep examples readable in both light and dark themes.
pygments_style = "sphinx"
pygments_dark_style = "monokai"
