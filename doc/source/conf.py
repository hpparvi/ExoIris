# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__name__), '..'))

import easyts

project = 'EasyTS'
copyright = '2024, Hannu Parviainen'
author = 'Hannu Parviainen'

version = ".".join(easyts.__version__.split('.')[:2])
release = easyts.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'numpydoc'
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ["**/.ipynb_checkpoints"]
numpydoc_show_class_members = False

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = "EasyTS"
html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/hpparvi/EasyTS"
}
html_static_path = ['_static']
#html_css_files = [
#    'css/custom.css',
#]

default_role = 'py:obj'

intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'astropy': ('https://docs.astropy.org/en/stable/', None)}