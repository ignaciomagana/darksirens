"""Sphinx configuration for the darksirens documentation.

Builds locally (``make docs-html``) and on Read the Docs from the same file.
Runtime dependencies that are not importable in the build environment are
mocked so the API pages render everywhere; on a full install autodoc imports
the real modules and shows complete signatures.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "darksirens"
author = "Ignacio Magaña Hernandez"
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
    "sphinx_design",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# ---------------------------------------------------------------- HTML theme
html_theme = "furo"
html_title = "darksirens"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "source_repository": "https://github.com/ignaciomagana/darksirens/",
    "source_branch": "master",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#1f5f8b",
        "color-brand-content": "#1f5f8b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb3d5",
        "color-brand-content": "#7fb3d5",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ignaciomagana/darksirens",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 "
                "0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
                "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z\"></path></svg>"
            ),
            "class": "",
        },
    ],
}
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# ---------------------------------------------------------------- MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------- autodoc
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# NamedTuple fields are documented once, as ``:ivar:`` entries, instead of as
# attribute directives that duplicate the members autodoc already emits.
napoleon_use_ivar = True
autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"member-order": "bysource"}
# Private helpers and the jit-wrapped closures are not part of the public API
# surface; underscore names are skipped by default (no :private-members:).
autodoc_inherit_docstrings = True
# Module-level ``__all__`` is respected where defined.

# Heavy or optional runtime dependencies.  Only the ones that are NOT
# importable here are mocked, so a full install documents real signatures.
_OPTIONAL_RUNTIME_DEPS = [
    "arviz",
    "astropy",
    "corner",
    "dynesty",
    "equinox",
    "flowjax",
    "gwcat",
    "gwdistributions",
    "h5py",
    "healpy",
    "jax",
    "jaxlib",
    "matplotlib",
    "numpyro",
    "paramax",
    "scipy",
    "seaborn",
    "tinygp",
    "tinyns",
    "tqdm",
]
# ``DARKSIRENS_DOCS_FORCE_MOCK=gwcat,tinyns`` mocks packages that ARE installed,
# to rehearse a hosted build (CI and Read the Docs do not carry the git-pinned
# gwcat/tinyns) on a full development machine.
_FORCED = {
    name.strip()
    for name in __import__("os").environ.get("DARKSIRENS_DOCS_FORCE_MOCK", "").split(",")
    if name.strip()
}
autodoc_mock_imports = [
    name
    for name in _OPTIONAL_RUNTIME_DEPS
    if name in _FORCED or importlib.util.find_spec(name) is None
]


# Docstrings in this code base are prose-heavy and were not written for
# Sphinx; autodoc reports their formatting slips as warnings.  Those do not
# indicate broken documentation, so they are not promoted to errors, while
# broken cross-references, missing toctree entries and MyST errors still are.
suppress_warnings = [
    "docutils",
    "autodoc",
    "autodoc.import_object",
    "ref.python",
]
nitpicky = False

_ADORNMENT = set("-=~^\"'`+*#")


def _demote_docstring_titles(app, what, name, obj, options, lines):
    """Turn RST section titles inside module docstrings into rubrics.

    Many modules open with ``selection.py`` over a dashed line.  Rendered by
    automodule, every such title becomes a section label, and two modules
    with the same file name (``registry.py`` in ``marks`` and ``sky``) then
    collide as duplicate labels.  A rubric renders the same heading without
    registering a label.
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        stripped = nxt.strip()
        if (
            line.strip()
            and len(stripped) >= 3
            and set(stripped) <= _ADORNMENT
            and len(set(stripped)) == 1
            and len(stripped) >= len(line.strip()) - 1
            and not line.startswith(" ")
        ):
            out.extend([f".. rubric:: {line.strip()}", ""])
            i += 2
            continue
        out.append(line)
        i += 1
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _demote_docstring_titles)

copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
