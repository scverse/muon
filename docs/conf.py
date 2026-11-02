# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/page/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("muon")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "scverse_doc",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "scverse_misc.sphinx_ext",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "tutorials/index.rst",
    "tutorials/*/index.rst",
    "tutorials/**/README.md",
    "tutorials/README.md",
]

intersphinx_mapping = intersphinx() | {
    "leidenalg": ("https://leidenalg.readthedocs.io/en/stable/", None),
    "louvain": ("https://louvain-igraph.readthedocs.io/en/latest/", None),
}


# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_logo = "_static/img/muon_logo.png"
html_css_files = ["css/custom.css"]

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
    # Placeholder muon defines when louvain-igraph is unavailable, so it has no documentation to link to.
    ("py:class", "louvain.VertexPartition.LouvainMutableVertexPartition")
]
