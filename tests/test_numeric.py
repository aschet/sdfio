# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from sdfio._numeric import format_scientific


@pytest.mark.parametrize(
    ("value", "precision", "exponent_width", "expected"),
    [
        (0.0, 6, 2, "0.000000e+00"),
        (0.0, 14, 3, "0.00000000000000e+000"),
        (-0.0, 6, 2, "0.000000e+00"),
        (-0.0, 14, 3, "0.00000000000000e+000"),
        (3.14159265358979, 6, 2, "3.141593e+00"),
        (1e-6, 14, 3, "1.00000000000000e-006"),
        (-6.463130e-01, 6, 2, "-6.463130e-01"),
    ],
)
def test_format_scientific(
    value: float, precision: int, exponent_width: int, expected: str
) -> None:
    assert format_scientific(value, precision, exponent_width) == expected


def test_format_scientific_roundtrips_through_float() -> None:
    text = format_scientific(1.234567891234e123, 14, 3)
    assert float(text) == pytest.approx(1.234567891234e123, rel=1e-13)
