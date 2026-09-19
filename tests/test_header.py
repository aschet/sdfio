# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from sdfio.exceptions import SdfFormatError, SdfVersionError
from sdfio.header import SdfDialect, SdfHeader, format_sdf_datetime, parse_sdf_datetime


def test_datetime_roundtrip() -> None:
    value = datetime(2024, 9, 3, 18, 16)
    text = format_sdf_datetime(value)
    assert text == "030920241816"
    parsed = parse_sdf_datetime(text)
    assert parsed is not None
    assert parsed.replace(tzinfo=None) == value


def test_datetime_tagged_as_utc_for_iso_2_0() -> None:
    parsed = parse_sdf_datetime("030920241816", SdfDialect.ISO_2_0)
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed == datetime(2024, 9, 3, 18, 16, tzinfo=UTC)


def test_datetime_tagged_as_local_for_iso_1_0() -> None:
    # The standard only specifies UTC for version 2.0, so version 1.0 is
    # treated as local time.
    parsed = parse_sdf_datetime("030920241816", SdfDialect.ISO_1_0)
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed == datetime(2024, 9, 3, 18, 16).astimezone()


def test_format_datetime_converts_aware_value_to_local_for_iso_1_0() -> None:
    aware = datetime(2024, 9, 3, 18, 16, tzinfo=UTC)
    text = format_sdf_datetime(aware, SdfDialect.ISO_1_0)
    assert text == aware.astimezone().strftime("%d%m%Y%H%M")


def test_header_rejects_unsupported_dialect() -> None:
    with pytest.raises(SdfVersionError):
        SdfHeader(dialect="9.9")  # type: ignore[arg-type]


def test_header_magic() -> None:
    assert SdfHeader(dialect=SdfDialect.ISO_2_0, binary=True).magic == "bISO-2.0"
    assert SdfHeader(dialect=SdfDialect.ISO_1_0, binary=False).magic == "aISO-1.0"


def test_header_magic_bcr() -> None:
    header = SdfHeader(dialect=SdfDialect.BCR_1_0, binary=True)
    assert header.magic == "bBCR-1.0"


def test_header_shape() -> None:
    header = SdfHeader(num_points=5, num_profiles=3)
    assert header.shape == (3, 5)


def test_header_defaults_are_valid_iso_dialect() -> None:
    header = SdfHeader()
    assert header.dialect == SdfDialect.ISO_2_0
    assert header.binary is True


def test_header_rejects_naive_datetime_for_iso_2_0() -> None:
    with pytest.raises(SdfFormatError, match="timezone-aware UTC"):
        SdfHeader(dialect=SdfDialect.ISO_2_0, create_date=datetime(2024, 9, 3))


def test_header_rejects_non_utc_datetime_for_iso_2_0() -> None:
    non_utc = datetime(2024, 9, 3, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(SdfFormatError, match="timezone-aware UTC"):
        SdfHeader(dialect=SdfDialect.ISO_2_0, mod_date=non_utc)


def test_header_allows_naive_datetime_for_iso_1_0() -> None:
    SdfHeader(
        dialect=SdfDialect.ISO_1_0, create_date=datetime(2024, 9, 3), mod_date=datetime(2024, 9, 3)
    )


def test_datetime_all_zero_placeholder_parses_as_none() -> None:
    # Undocumented in either spec, but observed in real files (e.g. from a
    # MATLAB-based exporter) as a "date not recorded" placeholder.
    assert parse_sdf_datetime("000000000000") is None


def test_datetime_none_formats_as_all_zero_placeholder() -> None:
    assert format_sdf_datetime(None) == "000000000000"


def test_header_allows_none_datetime_for_iso_2_0() -> None:
    # A None date has no timezone to validate, so it bypasses the UTC check
    # rather than being rejected.
    SdfHeader(dialect=SdfDialect.ISO_2_0, create_date=None, mod_date=None)
