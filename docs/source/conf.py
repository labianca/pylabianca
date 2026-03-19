# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'pylabianca'
copyright = '2025, pylabianca developers'
author = 'pylabianca developers'

version = '0.4'
release = '0.4.dev0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

# Mock heavy optional dependencies so autodoc can run without them
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'scikit-learn',
    'sklearn',
    'borsar',
    'h5io',
    'h5py',
    'xlrd',
    'xarray',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
