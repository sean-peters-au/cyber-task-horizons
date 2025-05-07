# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Cybersecurity Dataset Analysis'
copyright = '2024, Your Name/Org' # TODO: Update copyright
author = 'Your Name/Org' # TODO: Update author
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # For Google/NumPy style docstrings if you have them in code
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages', # Helps with GitHub Pages deployment
]

# templates_path = ['_templates'] # If you have custom templates
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster' # A popular default theme, others include 'sphinx_rtd_theme'
# html_static_path = ['_static'] # If you have custom static files (css, images not tied to a doc)

# If using sphinx_rtd_theme, you might want to add it to extensions and set it here:
# extensions = ['sphinx_rtd_theme']
# html_theme = "sphinx_rtd_theme"

# -- Options for csv-table directive (built-in) ----------------------------
# No specific global options needed here for basic usage, but options can be set per table. 