from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.registry import intersphinx

info = metadata("muon")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]

extensions = [
    "scverse_doc",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "scverse_misc.sphinx_ext",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

intersphinx_mapping = intersphinx() | {
    "leidenalg": ("https://leidenalg.readthedocs.io/en/stable/", None),
    "louvain": ("https://louvain-igraph.readthedocs.io/en/latest/", None),
}

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
