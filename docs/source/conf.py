# Configuration file for the Sphinx documentation builder.

# -- Path setup
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("../../py_champ"))  # To include sub-modules

# -- Project information

project = "PyCHAMP"
copyright = "2024, Chung-Yi Lin and Sameer Dhakal"
author = "Chung-Yi Lin and Sameer Dhakal"

release = "1.0.0"
version = "1.0.0"

# -- General configuration

autodoc_member_order = "bysource"

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx-prompt",
    # to include docstrings from modules
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # to link to other Sphinx docs
    "sphinx.ext.intersphinx",
    # enables Sphinx to parse both NumPy and Google style docstrings
    "sphinx.ext.napoleon",  
    # to use markdown
    "myst_parser",  
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

numfig = True

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
