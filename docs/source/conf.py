project, author, release = "GRAFS-E", "Adrien Fauste-Gay", "1.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
]
autosummary_generate = True

myst_enable_extensions = [
    "dollarmath",              # $x^2$ et $$ ... $$
    "amsmath",                 # \begin{align} ... \end{align}, etc.
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {"use_edit_page_button": False, "icon_links": []}
html_show_sourcelink = False
html_baseurl = ""                  # pas de canonical public
html_meta = {"robots": "noindex, nofollow"}  # doc locale/privée

# Fichier(s) .bib
bibtex_bibfiles = ["refs.bib"]
# (facultatif) style & tri
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"  # ou "label", "super"