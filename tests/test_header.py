# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from sdfio.exceptions import SdfFormatError, SdfVersionError
from sdfio.header import SdfDialect, SdfHeader, SdfVersion, format_sdf_datetime, parse_sdf_datetime


def test_datetime_roundtrip() -> None:
    value = datetime(2024, 9, 3, 18, 16)
    text = format_sdf_datetime(value)
    assert text == "030920241816"
    assert parse_sdf_datetime(text) == value


def test_datetime_tagged_as_utc_for_version_2_0() -> None:
    parsed = parse_sdf_datetime("030920241816", SdfVersion.V2_0)
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed == datetime(2024, 9, 3, 18, 16, tzinfo=UTC)


def test_datetime_left_naive_for_version_1_0() -> None:
    parsed = parse_sdf_datetime("030920241816", SdfVersion.V1_0)
    assert parsed is not None
    assert parsed.tzinfo is None


def test_header_rejects_unsupported_version() -> None:
    with pytest.raises(SdfVersionError):
        SdfHeader(version="9.9")  # type: ignore[arg-type]


def test_header_magic() -> None:
    assert SdfHeader(version=SdfVersion.V2_0, binary=True).magic == "bISO-2.0"
    assert SdfHeader(version=SdfVersion.V1_0, binary=False).magic == "aISO-1.0"


def test_header_magic_bcr() -> None:
    header = SdfHeader(version=SdfVersion.V1_0, dialect=SdfDialect.BCR, binary=True)
    assert header.magic == "bBCR-1.0"


def test_header_rejects_bcr_with_version_2_0() -> None:
    with pytest.raises(SdfFormatError, match="does not support version"):
        SdfHeader(version=SdfVersion.V2_0, dialect=SdfDialect.BCR)


def test_header_shape() -> None:
    header = SdfHeader(num_points=5, num_profiles=3)
    assert header.shape == (3, 5)


def test_header_defaults_are_valid_iso_version() -> None:
    header = SdfHeader()
    assert header.version == SdfVersion.V2_0
    assert header.binary is True


def test_header_rejects_naive_datetime_for_version_2_0() -> None:
    with pytest.raises(SdfFormatError, match="timezone-aware UTC"):
        SdfHeader(version=SdfVersion.V2_0, create_date=datetime(2024, 9, 3))


def test_header_rejects_non_utc_datetime_for_version_2_0() -> None:
    non_utc = datetime(2024, 9, 3, tzinfo=timezone(timedelta(hours=2)))
    with pytest.raises(SdfFormatError, match="timezone-aware UTC"):
        SdfHeader(version=SdfVersion.V2_0, mod_date=non_utc)


def test_header_allows_naive_datetime_for_version_1_0() -> None:
    SdfHeader(
        version=SdfVersion.V1_0, create_date=datetime(2024, 9, 3), mod_date=datetime(2024, 9, 3)
    )


def test_datetime_all_zero_placeholder_parses_as_none() -> None:
    # Undocumented in either spec, but observed in real files (e.g. from a
    # MATLAB-based exporter) as a "date not recorded" placeholder.
    assert parse_sdf_datetime("000000000000") is None


def test_datetime_none_formats_as_all_zero_placeholder() -> None:
    assert format_sdf_datetime(None) == "000000000000"


def test_header_allows_none_datetime_for_version_2_0() -> None:
    # A None date has no timezone to validate, so it bypasses the UTC check
    # rather than being rejected.
    SdfHeader(version=SdfVersion.V2_0, create_date=None, mod_date=None)
