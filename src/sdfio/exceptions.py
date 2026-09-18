# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Exceptions raised by :mod:`sdfio`."""

from __future__ import annotations

__all__ = ["SdfError", "SdfFormatError", "SdfVersionError"]


class SdfError(Exception):
    """Base class for all sdfio errors."""


class SdfFormatError(SdfError):
    """Raised when file content does not conform to the SDF format."""


class SdfVersionError(SdfFormatError):
    """Raised when an SDF version or version/feature combination is unsupported."""
