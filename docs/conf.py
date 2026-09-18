# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Sphinx configuration for the sdfio documentation."""

from __future__ import annotations

import sdfio

project = "sdfio"
copyright = "2026, Thomas Ascher"
author = "Thomas Ascher"
release = sdfio.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "imported-members": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
