# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io

import numpy as np
import pytest

from sdfio import _ascii
from sdfio.exceptions import SdfFormatError
from sdfio.header import SdfDialect, SdfHeader

# Adapted from the standard's own worked example, using a small, fully
# specified 2x3 grid (the original elides most values with "......").
ISO_ANNEX_A_EXAMPLE = (
    "aISO-2.0\r\n"
    "ManufacID = ISOTC213\r\n"
    "CreateDate = 040120220853\r\n"
    "ModDate = 050320231353\r\n"
    "NumPoints = 3\r\n"
    "NumProfiles = 2\r\n"
    "Xscale = 1.00000000000000e-006\r\n"
    "Yscale = 1.00000000000000e-006\r\n"
    "Zscale = 1.00000000000000e-006\r\n"
    "Zresolution = 1.00000000000000e-009\r\n"
    "Compression = 0\r\n"
    "DataType = 3\r\n"
    "CheckType = 0\r\n"
    "*\r\n"
    "-1.000000e+00 6.948286e-01 3.170994e-01\r\n"
    " 4.897644e-01 BAD -6.463130e-01\r\n"
    "*\r\n"
    "<OperatorName> WG 16 </OperatorName>\r\n"
    "*\r\n"
)


def test_loads_parses_header() -> None:
    header, _data, _trailer = _ascii.loads(ISO_ANNEX_A_EXAMPLE)
    assert header.dialect == SdfDialect.ISO_2_0
    assert header.binary is False
    assert header.manufacturer_id == "ISOTC213"
    assert header.num_points == 3
    assert header.num_profiles == 2
    assert header.data_type == 3
    assert header.z_scale == pytest.approx(1e-6)


def test_loads_scales_data_and_marks_invalid_points() -> None:
    header, data, _trailer = _ascii.loads(ISO_ANNEX_A_EXAMPLE)
    assert data.shape == (2, 3)
    assert data[0, 0] == pytest.approx(-1.0 * header.z_scale)
    assert data[1, 0] == pytest.approx(4.897644e-01 * header.z_scale)
    assert np.isnan(data[1, 1])


def test_loads_captures_trailer() -> None:
    _header, _data, trailer = _ascii.loads(ISO_ANNEX_A_EXAMPLE)
    assert trailer == "<OperatorName> WG 16 </OperatorName>"


def test_loads_rejoins_a_trailer_containing_an_embedded_terminator_line() -> None:
    # A trailer line that is itself just "*" is indistinguishable from a
    # record terminator; loads() should reconstruct it losslessly rather
    # than silently dropping content or corrupting adjacent parts.
    text = ISO_ANNEX_A_EXAMPLE.replace(
        "<OperatorName> WG 16 </OperatorName>\r\n*\r\n",
        "<Note>part one</Note>\r\n*\r\n<Note>part two</Note>",
    )
    _header, _data, trailer = _ascii.loads(text)
    assert trailer == "<Note>part one</Note>*<Note>part two</Note>"


def test_loads_treats_adjacent_terminators_as_empty_trailer() -> None:
    # An empty trailer explicitly represented by a "*" immediately after the
    # data section's own "*" (as opposed to omitting the trailer entirely,
    # which dumps() itself does) -- observed in real NPL SoftGauges files.
    text = ISO_ANNEX_A_EXAMPLE.replace(
        "*\r\n<OperatorName> WG 16 </OperatorName>\r\n*\r\n",
        "*\r\n*\r\n",
    )
    _header, _data, trailer = _ascii.loads(text)
    assert trailer == ""


def test_loads_rejects_binary_magic() -> None:
    with pytest.raises(SdfFormatError, match="Binary magic"):
        _ascii.loads("bISO-2.0\r\n*\r\n*\r\n*\r\n")


def test_loads_rejects_wrong_value_count() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("NumPoints = 3", "NumPoints = 4")
    with pytest.raises(SdfFormatError, match="Expected 8 data values"):
        _ascii.loads(bad)


def test_loads_rejects_invalid_integer_header_field() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("NumPoints = 3", "NumPoints = three")
    with pytest.raises(SdfFormatError, match=r"Invalid integer value.*NumPoints"):
        _ascii.loads(bad)


def test_loads_rejects_invalid_float_header_field() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("Xscale = 1.00000000000000e-006", "Xscale = not-a-number")
    with pytest.raises(SdfFormatError, match=r"Invalid float value.*Xscale"):
        _ascii.loads(bad)


def test_loads_rejects_invalid_data_value() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("6.948286e-01", "not-a-number")
    with pytest.raises(SdfFormatError, match="Invalid data value"):
        _ascii.loads(bad)


def test_loads_rejects_unsupported_data_type_for_dialect() -> None:
    # binary32 (code 3) is only valid for ISO-2.0/BCR-1.0.
    bad = ISO_ANNEX_A_EXAMPLE.replace("aISO-2.0", "aISO-1.0")
    with pytest.raises(SdfFormatError, match="not valid for SDF dialect"):
        _ascii.loads(bad)


def test_loads_rejects_missing_newline() -> None:
    with pytest.raises(SdfFormatError, match="missing the header and data records"):
        _ascii.loads("aISO-2.0")


def test_loads_rejects_malformed_header_line() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("ManufacID = ISOTC213", "ManufacID ISOTC213")
    with pytest.raises(SdfFormatError, match="Malformed SDF tagged field line"):
        _ascii.loads(bad)


def test_loads_rejects_missing_header_field() -> None:
    bad = "\n".join(
        line for line in ISO_ANNEX_A_EXAMPLE.splitlines() if not line.startswith("ManufacID")
    )
    with pytest.raises(SdfFormatError, match="Missing required SDF header field"):
        _ascii.loads(bad)


def test_loads_rejects_missing_records() -> None:
    with pytest.raises(SdfFormatError, match="header, data and trailer records"):
        _ascii.loads("aISO-2.0\r\nonly one record\r\n")


def test_loads_rejects_compressed_data() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("Compression = 0", "Compression = 1")
    with pytest.raises(SdfFormatError, match="Compressed SDF data areas"):
        _ascii.loads(bad)


def test_loads_rejects_checksummed_data() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("CheckType = 0", "CheckType = 1")
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _ascii.loads(bad)


def test_load_reads_from_file_object() -> None:
    header, data, _trailer = _ascii.load(io.StringIO(ISO_ANNEX_A_EXAMPLE))
    assert header.manufacturer_id == "ISOTC213"
    assert data.shape == (2, 3)


def test_dumps_rejects_out_of_range_integer_value() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, z_scale=1.0, data_type=4)  # int8
    with pytest.raises(SdfFormatError, match="out of range"):
        _ascii.dumps(header, np.array([[1000.0]]))


def test_dumps_rejects_sentinel_collision() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, z_scale=1.0, data_type=4)  # int8
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        _ascii.dumps(header, np.array([[-128.0]]))


def test_dumps_rejects_malformed_v2_trailer() -> None:
    header = SdfHeader(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1, data_type=7)
    with pytest.raises(SdfFormatError, match="tagged 'Name = Value' format"):
        _ascii.dumps(header, np.zeros((1, 1)), trailer="not tagged fields")


def test_dumps_allows_tagged_v2_trailer() -> None:
    header = SdfHeader(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1, data_type=7)
    _ascii.dumps(header, np.zeros((1, 1)), trailer="Note = hello")


def test_dumps_allows_empty_v2_trailer() -> None:
    header = SdfHeader(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1, data_type=7)
    _ascii.dumps(header, np.zeros((1, 1)), trailer="")


def test_dumps_loads_roundtrips_empty_trailer_as_empty_string() -> None:
    # dumps() always terminates the trailer record, even when empty -- this
    # produces two adjacent "*" markers with nothing between them.
    header = SdfHeader(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1, data_type=7)
    text = _ascii.dumps(header, np.zeros((1, 1)), trailer="")
    assert text.endswith("*\r\n*\r\n")
    _header, _data, trailer = _ascii.loads(text)
    assert trailer == ""


def test_dumps_v1_trailer_need_not_be_xml() -> None:
    header = SdfHeader(dialect=SdfDialect.ISO_1_0, num_points=1, num_profiles=1, data_type=7)
    _ascii.dumps(header, np.zeros((1, 1)), trailer="Operator = Jane Doe")


def test_dumps_rejects_non_ascii_trailer() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7)
    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _ascii.dumps(header, np.zeros((1, 1)), trailer="Müller café")


def test_dumps_rejects_non_ascii_manufacturer_id() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7, manufacturer_id="Müller")
    with pytest.raises(SdfFormatError, match="manufacturer_id must be 7-bit ASCII"):
        _ascii.dumps(header, np.zeros((1, 1)))


def test_loads_returns_non_ascii_trailer_as_is() -> None:
    """A non-compliant trailer must not prevent reading the (already-parsed) depth data."""
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7)
    # dumps() itself always terminates the (here, empty) trailer; replace
    # that terminator to simulate a non-compliant trailer added after the fact.
    text = _ascii.dumps(header, np.zeros((1, 1)))
    assert text.endswith("*\r\n")
    text = text[: -len("*\r\n")] + "Müller café\r\n*\r\n"

    _, data, trailer = _ascii.loads(text)

    np.testing.assert_array_equal(data, np.zeros((1, 1)))
    assert trailer == "Müller café"


def test_dumps_roundtrips_loads() -> None:
    data = np.array([[1e-6, np.nan, -2.5e-6], [0.0, 3.14159e-6, 1.0e-6]])
    header = SdfHeader(
        dialect=SdfDialect.ISO_2_0,
        num_points=3,
        num_profiles=2,
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale=1e-6,
        data_type=7,
    )
    text = _ascii.dumps(header, data)
    round_tripped_header, round_tripped_data, _trailer = _ascii.loads(text)

    assert round_tripped_header.num_points == header.num_points
    assert round_tripped_header.num_profiles == header.num_profiles
    np.testing.assert_allclose(round_tripped_data, data, rtol=1e-10, equal_nan=True)


def test_dumps_roundtrips_integer_data_type() -> None:
    data = np.array([[1e-6, np.nan], [-2e-6, 0.0]])
    header = SdfHeader(num_points=2, num_profiles=2, z_scale=1e-6, data_type=6)
    text = _ascii.dumps(header, data)
    _header, round_tripped_data, _trailer = _ascii.loads(text)
    np.testing.assert_allclose(round_tripped_data, data, rtol=1e-9, equal_nan=True)


def test_dump_writes_to_file_object() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7)
    fp = io.StringIO()
    _ascii.dump(header, np.zeros((1, 1)), fp)
    assert fp.getvalue().startswith("aISO-2.0")


def test_dumps_rejects_shape_mismatch() -> None:
    header = SdfHeader(num_points=2, num_profiles=2)
    with pytest.raises(SdfFormatError, match="does not match header shape"):
        _ascii.dumps(header, np.zeros((3, 3)))


def test_dumps_loads_roundtrip_bcr_dialect() -> None:
    header = SdfHeader(
        dialect=SdfDialect.BCR_1_0,
        num_points=2,
        num_profiles=1,
        z_scale=1e-6,
        data_type=3,
    )
    data = np.array([[1e-6, -2e-6]])
    text = _ascii.dumps(header, data)
    assert text.startswith("aBCR-1.0")

    round_tripped_header, round_tripped_data, _trailer = _ascii.loads(text)
    assert round_tripped_header.dialect == SdfDialect.BCR_1_0
    np.testing.assert_allclose(round_tripped_data, data, rtol=1e-6)


def test_dumps_reverses_profile_order_for_bcr_on_disk() -> None:
    # Each row has a distinct value so the on-disk row order is directly
    # observable, independent of the dialect-aware unflipping loads() does.
    data = np.array([[0.0], [1.0], [2.0], [3.0]])
    iso_header = SdfHeader(
        dialect=SdfDialect.ISO_1_0, num_points=1, num_profiles=4, z_scale=1.0, data_type=7
    )
    bcr_header = SdfHeader(
        dialect=SdfDialect.BCR_1_0, num_points=1, num_profiles=4, z_scale=1.0, data_type=7
    )

    iso_text = _ascii.dumps(iso_header, data)
    bcr_text = _ascii.dumps(bcr_header, data)

    def data_values(text: str) -> list[float]:
        return [float(line) for line in text.splitlines()[14:18]]

    assert data_values(iso_text) == [0.0, 1.0, 2.0, 3.0]
    assert data_values(bcr_text) == [3.0, 2.0, 1.0, 0.0]

    # But reading either back gives the same logical (row 0 = y=0) array.
    _iso_header, iso_read_data, _t = _ascii.loads(iso_text)
    _bcr_header, bcr_read_data, _t = _ascii.loads(bcr_text)
    np.testing.assert_array_equal(iso_read_data, data)
    np.testing.assert_array_equal(bcr_read_data, data)


def test_dumps_loads_roundtrip_none_dates() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7, create_date=None, mod_date=None)
    text = _ascii.dumps(header, np.zeros((1, 1)))
    assert "CreateDate = 000000000000" in text

    round_tripped_header, _data, _trailer = _ascii.loads(text)
    assert round_tripped_header.create_date is None
    assert round_tripped_header.mod_date is None


def test_loads_rejects_unknown_dialect() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("aISO-2.0", "aXYZ-2.0")
    with pytest.raises(SdfFormatError, match="Unknown or unsupported SDF dialect"):
        _ascii.loads(bad)


def test_loads_rejects_bcr_checksum_too() -> None:
    # BCR's IntegerTrace checksum (CheckType=1) is a value inline in the
    # data area, not a header value.
    text = (
        "aBCR-1.0\r\n"
        "ManufacID = sdfio\r\n"
        "CreateDate = 010120240000\r\n"
        "ModDate = 010120240000\r\n"
        "NumPoints = 2\r\n"
        "NumProfiles = 2\r\n"
        "Xscale = 1.0e-6\r\n"
        "Yscale = 1.0e-6\r\n"
        "Zscale = 1.0e-6\r\n"
        "Zresolution = -1\r\n"
        "Compression = 0\r\n"
        "DataType = 7\r\n"
        "CheckType = 1\r\n"
        "*\r\n"
        "1 2 123\r\n"
        "3 4 456\r\n"
        "*\r\n"
    )
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _ascii.loads(text)
