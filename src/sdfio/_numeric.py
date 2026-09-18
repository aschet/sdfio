# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Shared numeric formatting helpers for the SDF ASCII representation."""

from __future__ import annotations

__all__ = ["format_scientific"]


def format_scientific(value: float, precision: int, exponent_width: int) -> str:
    """Format ``value`` in scientific notation with a fixed-width exponent.

    Produces a normalized signed floating point number in scientific
    notation with a fixed number of digits after the decimal point and a
    zero-padded, signed exponent of fixed width, e.g. ``"3.141593e+00"``.

    :param value: The number to format.
    :param precision: Digits after the decimal point (6 for ``binary32``,
        14 for ``binary64``).
    :param exponent_width: Digits used for the exponent (2 for ``binary32``,
        3 for ``binary64``).

    The standard mandates 7 significant digits for ``binary32`` and 15 for
    ``binary64``. ``binary64`` (IEEE 754 double) needs 17 to round-trip
    exactly, so formatting a ``binary64`` value this way and parsing it back
    can differ by up to ~1 ULP at the 15th significant digit.
    """
    # value + 0.0 normalizes -0.0 to +0.0 (IEEE 754: -0.0 + 0.0 == 0.0), since
    # the standard's zero format ("0.000000e+00") has no sign to preserve.
    mantissa, exponent = f"{value + 0.0:.{precision}e}".split("e")
    exponent_value = int(exponent)
    sign = "+" if exponent_value >= 0 else "-"
    return f"{mantissa}e{sign}{abs(exponent_value):0{exponent_width}d}"
