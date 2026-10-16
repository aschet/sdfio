# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

import sdfio
from sdfio.exceptions import SdfVersionError
from sdfio.header import (
    MAGICS,
    FileFormat,
    Header,
    Metadata,
    Revision,
    format_sdf_datetime,
    parse_sdf_datetime,
    sanitize_tagged_fields,
)


def test_datetime_roundtrip() -> None:
    value = datetime(2024, 9, 3, 18, 16)
    text = format_sdf_datetime(value)
    assert text == "030920241816"
    parsed = parse_sdf_datetime(text)
    assert parsed is not None
    assert parsed.tzinfo == UTC  # held as UTC; the text is local time
    assert parsed.astimezone().replace(tzinfo=None) == value


def test_datetime_tagged_as_utc_for_iso_2_0() -> None:
    parsed = parse_sdf_datetime("030920241816", Revision.ISO_2_0)
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed == datetime(2024, 9, 3, 18, 16, tzinfo=UTC)


def test_datetime_of_iso_1_0_is_read_as_local_time_and_held_as_utc() -> None:
    # The standard only specifies UTC for version 2.0, so version 1.0 is
    # local time, which is converted to UTC when it is read.
    parsed = parse_sdf_datetime("030920241816", Revision.ISO_1_0)
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed == datetime(2024, 9, 3, 18, 16).astimezone()


def test_format_datetime_converts_aware_value_to_local_for_iso_1_0() -> None:
    aware = datetime(2024, 9, 3, 18, 16, tzinfo=UTC)
    text = format_sdf_datetime(aware, Revision.ISO_1_0)
    assert text == aware.astimezone().strftime("%d%m%Y%H%M")


def test_header_rejects_unsupported_revision() -> None:
    with pytest.raises(SdfVersionError):
        Header(revision="9.9")  # type: ignore[arg-type]


def test_header_magic() -> None:
    assert Header(revision=Revision.ISO_2_0, format=FileFormat.BINARY).magic == "bISO-2.0"
    assert Header(revision=Revision.ISO_1_0, format=FileFormat.ASCII).magic == "aISO-1.0"


def test_header_magic_bcr() -> None:
    header = Header(revision=Revision.BCR_1_0, format=FileFormat.BINARY)
    assert header.magic == "bBCR-1.0"


def test_sanitize_tagged_fields_drops_non_matching_lines() -> None:
    text = "OperatorName = Jane Doe\r\nsome free text without equals\r\nPartName = X"

    assert sanitize_tagged_fields(text) == "OperatorName = Jane Doe\r\nPartName = X\r\n"


def test_sanitize_tagged_fields_never_raises_on_fully_malformed_text() -> None:
    assert sanitize_tagged_fields("just some prose\nwith no tags at all") == ""


def test_magics_covers_every_revision_in_both_formats_and_prefix_cases() -> None:
    assert MAGICS == (
        b"aISO-1.0",
        b"aISO-2.0",
        b"aBCR-1.0",
        b"bISO-1.0",
        b"bISO-2.0",
        b"bBCR-1.0",
        b"AISO-1.0",
        b"AISO-2.0",
        b"ABCR-1.0",
        b"BISO-1.0",
        b"BISO-2.0",
        b"BBCR-1.0",
    )


def test_reverses_profile_order_is_bcr_only() -> None:
    assert Revision.BCR_1_0.reverses_profile_order is True
    assert Revision.ISO_1_0.reverses_profile_order is False
    assert Revision.ISO_2_0.reverses_profile_order is False


def test_header_defaults_are_valid_iso_revision() -> None:
    header = Header()
    assert header.revision == Revision.ISO_2_0
    assert header.format is FileFormat.BINARY


@pytest.mark.parametrize("revision", list(Revision))
def test_header_holds_dates_as_utc_for_every_revision(revision: Revision) -> None:
    plus_two = datetime(2024, 9, 3, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    header = Header(revision=revision, create_date=plus_two, mod_date=datetime(2024, 9, 3, 14, 0))

    assert header.create_date == datetime(2024, 9, 3, 12, 0, tzinfo=UTC)
    assert header.create_date is not None
    assert header.create_date.utcoffset() == timedelta(0)
    # A naive date is local time.
    assert header.mod_date == datetime(2024, 9, 3, 14, 0).astimezone(UTC)


def test_header_converts_a_date_that_is_assigned_later() -> None:
    header = Header(revision=Revision.ISO_2_0)
    header.create_date = datetime(2024, 9, 3, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    header.mod_date = None

    assert header.create_date == datetime(2024, 9, 3, 12, 0, tzinfo=UTC)
    assert header.create_date.tzinfo == UTC
    assert header.mod_date is None


def test_datetime_all_zero_placeholder_parses_as_none() -> None:
    # Undocumented in either spec, but observed in real files (e.g. from a
    # MATLAB-based exporter) as a "date not recorded" placeholder.
    assert parse_sdf_datetime("000000000000") is None


def test_datetime_none_formats_as_all_zero_placeholder() -> None:
    assert format_sdf_datetime(None) == "000000000000"


def test_header_allows_none_datetime_for_iso_2_0() -> None:
    # A None date ("not recorded") has no timezone to convert.
    Header(revision=Revision.ISO_2_0, create_date=None, mod_date=None)


def test_dates_are_written_as_utc_for_iso_2_0_and_local_time_for_the_older_revisions() -> None:
    instant = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    local = instant.astimezone().strftime("%d%m%Y%H%M")  # the wall clock of this machine

    assert format_sdf_datetime(instant, Revision.ISO_2_0) == "051020261200"
    assert format_sdf_datetime(instant, Revision.ISO_1_0) == local
    assert format_sdf_datetime(instant, Revision.BCR_1_0) == local


@pytest.mark.parametrize("revision", list(Revision))
@pytest.mark.parametrize("binary", [True, False])
def test_the_instant_survives_a_file_in_every_revision(revision: Revision, binary: bool) -> None:
    plus_two = datetime(2026, 10, 5, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    header = Header(
        revision=revision,
        format=FileFormat.BINARY if binary else FileFormat.ASCII,
        create_date=plus_two,
        mod_date=plus_two,
        data_type=sdfio.DataType.INT16,
    )
    loaded = sdfio.SdfFile.loads(sdfio.SdfFile(header=header, data=np.zeros((1, 1))).dumps())

    assert loaded.header.create_date == datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    assert loaded.header.create_date is not None
    assert loaded.header.create_date.tzinfo == UTC
    assert loaded.header.mod_date == loaded.header.create_date


@pytest.mark.parametrize("revision", [Revision.ISO_1_0, Revision.BCR_1_0])
def test_the_older_revisions_store_local_time_in_the_file(revision: Revision) -> None:
    instant = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    header = Header(
        revision=revision,
        format=FileFormat.ASCII,
        create_date=instant,
        mod_date=instant,
        data_type=sdfio.DataType.INT16,
    )
    text = sdfio.SdfFile(header=header, data=np.zeros((1, 1))).dumps().decode("ascii")

    assert f"CreateDate = {instant.astimezone().strftime('%d%m%Y%H%M')}" in text


def test_metadata_holds_dates_as_utc_like_the_header() -> None:
    plus_two = datetime(2024, 9, 3, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    metadata = Metadata(create_date=plus_two, mod_date=datetime(2024, 9, 3, 14, 0))

    assert metadata.create_date == datetime(2024, 9, 3, 12, 0, tzinfo=UTC)
    assert metadata.create_date is not None
    assert metadata.create_date.tzinfo == UTC
    assert metadata.mod_date == datetime(2024, 9, 3, 14, 0).astimezone(UTC)
    assert Metadata().create_date is None  # the default is the time of the write


def test_dates_have_the_resolution_of_the_field_so_a_file_returns_what_was_set() -> None:
    seconds = datetime(2026, 10, 16, 9, 44, 24, 123456, tzinfo=UTC)
    header = Header(revision=Revision.ISO_2_0, create_date=seconds, mod_date=seconds)

    assert header.create_date == datetime(2026, 10, 16, 9, 44, tzinfo=UTC)
    assert header.mod_date == header.create_date
    header.mod_date = datetime(2026, 10, 16, 9, 45, 59, tzinfo=UTC)
    assert header.mod_date == datetime(2026, 10, 16, 9, 45, tzinfo=UTC)
    assert Metadata(create_date=seconds).create_date == header.create_date


@pytest.mark.parametrize("revision", list(Revision))
def test_the_date_read_from_a_file_equals_the_one_that_was_set(revision: Revision) -> None:
    header = Header(
        revision=revision,
        create_date=datetime.now(UTC),
        data_type=sdfio.DataType.INT16,
    )
    loaded = sdfio.SdfFile.loads(sdfio.SdfFile(header=header, data=np.zeros((1, 1))).dumps())

    assert loaded.header.create_date == header.create_date
    assert loaded.header.mod_date == header.mod_date


@pytest.mark.parametrize(
    "text",
    [
        "",
        " " * 12,
        "abcdefghijkl",
        "\x00" * 12,
        "321320261200",  # day 32, month 13
        "051320261200",
        "052026252500",
        "290220251200",  # 29 February 2025 does not exist
        "051000001200",  # year 0
        "0510",
        "05102026120",  # one digit short
        "0510202612000",  # one digit too many
        "05102026\ufffd\ufffd00",
    ],
)
@pytest.mark.parametrize("revision", [None, *Revision])
def test_a_date_field_that_is_no_date_is_not_recorded(text: str, revision: Revision | None) -> None:
    assert parse_sdf_datetime(text, revision) is None


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("revision", list(Revision))
def test_a_file_with_a_bad_date_field_is_still_read(revision: Revision, binary: bool) -> None:
    header = Header(
        revision=revision,
        format=FileFormat.BINARY if binary else FileFormat.ASCII,
        z_scale=1e-9,
        data_type=sdfio.DataType.INT16,
    )
    blob = sdfio.SdfFile(header=header, data=np.zeros((1, 2))).dumps()
    good = format_sdf_datetime(header.create_date, revision).encode("ascii")
    assert good in blob
    broken = blob.replace(good, b"32132026xxxx", 1)  # the CreateDate field

    loaded = sdfio.SdfFile.loads(broken)

    assert loaded.header.create_date is None
    assert loaded.header.mod_date == header.mod_date  # only the broken field is not recorded
    assert loaded.data.shape == (1, 2)


def test_the_header_carries_the_file_format_as_an_enum() -> None:
    assert Header().format is FileFormat.BINARY
    assert Header(format=FileFormat.ASCII).format is FileFormat.ASCII
    assert Header(format=FileFormat.ASCII).magic == "aISO-2.0"


@pytest.mark.parametrize("value", [True, False, 1, "binary", None])
def test_the_header_rejects_a_file_format_that_is_not_an_enum_member(value: object) -> None:
    with pytest.raises(TypeError, match="FileFormat"):
        Header(format=value)  # type: ignore[arg-type]


def test_the_former_binary_flag_is_gone() -> None:
    with pytest.raises(TypeError):
        Header(binary=True)  # type: ignore[call-arg]
    assert not hasattr(Header(), "binary")
