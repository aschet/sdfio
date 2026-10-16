# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Exceptions raised by :mod:`sdfio`."""

from __future__ import annotations

__all__ = ["SdfError", "SdfFormatError", "SdfVersionError"]


class SdfError(Exception):
    """Base class for all sdfio errors."""


class SdfFormatError(SdfError, ValueError):
    """Raised when file content or a value given for a file is not valid in the SDF format.

    It is a :class:`ValueError`, like :class:`json.JSONDecodeError`, so code that catches that
    also catches a bad value passed to sdfio.
    """


class SdfVersionError(SdfFormatError):
    """Raised when an SDF revision string is not recognized."""
