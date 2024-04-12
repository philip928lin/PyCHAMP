# Configuration file for the Sphinx documentation builder.

# -- Path setup 
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('../../py_champ')) # To include sub-modules

# -- Project information

project = 'PyCHAMP'
copyright = '2024, Chung-Yi Lin and Sameer Dhakal'
author = 'Chung-Yi Lin and Sameer Dhakal'

release = '1.0.0'
version = '1.0.0'

# -- General configuration

autodoc_member_order = 'bysource'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx-prompt',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon', # enables Sphinx to parse both NumPy and Google style docstrings
    "myst_parser",   # to use markdown
    #"autoapi.extension"
]

#autoapi_type = 'python'
#autoapi_dirs = ['../../your/source/code']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

numfig = True

# -- Options for HTML output
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
