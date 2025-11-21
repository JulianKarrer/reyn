
import sys
import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'REYN'
copyright = '2025, Julian Karrer'
author = 'Julian Karrer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

extensions = [
    # for interfacing with docygen output
    "breathe",
    # for latex support
    "sphinx.ext.mathjax",
    # auto generate API pages
    "exhale",
    # allow markdown
    "myst_parser",
    # custom favicon
    "sphinx_favicon",
]

# myst settings
myst_enable_extensions = [
    # enable $ and $$
    "dollarmath",
    # enable \[\] and \begin{equation}
    "amsmath",
]

# point breathe to doxygen output
doxygen_xml_dir = os.path.join(
    os.path.dirname(__file__), "doxygen-output", "xml")
breathe_projects = {"REYN": doxygen_xml_dir}
breathe_default_project = "REYN"

# exhale args
exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ Documentation",
    "doxygenStripFromPath": ROOT,

    "createTreeView": True,
    # "treeViewIsBootstrap": True,

    # do not generate table of contents, furo does this
    "contentsDirectives": False,

    # use CUDA lexer
    "lexerMapping": {
        r".*\.cuh": "cuda",
        r".*\.cu": "cuda",
    },

    # 'define', 'enum', 'function', 'class', 'struct', 'typedef', 'union', 'variable'
    "unabridgedOrphanKinds": {
        'define',
        'enum',
        'function',
        # 'class',
        # 'struct',
        'typedef',
        'union',
        'variable',
        'namespace',
        # 'file',
        'dir'
    }
}

# favicon settings
favicons = [
    "icon.png",
]

# read the docs theme
html_theme = "furo"
html_static_path = ['_staticc']
html_logo = "_staticc/icon.png"
html_theme_options = {
    "sidebar_hide_name": True,
}


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
