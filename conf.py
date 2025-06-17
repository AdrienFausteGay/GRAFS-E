# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # adapte selon ton arborescence

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GRAFS-E"
copyright = "2025, Adrien Fauste-Gay"
author = "Adrien Fauste-Gay"
release = "01/02/25"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # Handle Google/NumPy style docstrings
    # "sphinxcontrib.bibtex",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.viewcode",
]

html_theme_options = {
    "show_version_warning_banner": True,
    "navbar_align": "left",
    "show_prev_next": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "show_toc_level": 4,
}

# *** AJOUTEZ OU MODIFIEZ CETTE LIGNE ***
# Important : assurez-vous que c'est l'URL complète de votre documentation,
# y compris le nom du dépôt, et se terminant par un slash.
# Remplacez 'AdrienFausteGay' par votre nom d'utilisateur GitHub
# et 'grafs-e-docs' par le nom de votre dépôt de documentation.
# html_baseurl = "https://adrienfaustegay.github.io/grafs-e-docs/"

html_context = {
    "github_url": "https://github.com/AdrienFausteGay/grafs-e-docs",
    "github_repo": "grafs-e-docs",
    "github_user": "AdrienFausteGay",
    "github_version": "main",  # ou gh-pages si c'est votre branche de déploiement
    "doc_path": ".docs",  # si votre conf.py est dans le dossier 'docs' de votre dépôt principal
}


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

toc_object_entries_show_parents = "hide"

napoleon_google_docstring = False
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
