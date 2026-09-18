# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io

import numpy as np
import pytest

from sdfio import _ascii
from sdfio.exceptions import SdfFormatError
from sdfio.header import SdfHeader, SdfVersion

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
    assert header.version == SdfVersion.V2_0
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


def test_loads_rejects_unsupported_data_type_for_version() -> None:
    # binary32 (code 3) is only valid for version 2.0.
    bad = ISO_ANNEX_A_EXAMPLE.replace("aISO-2.0", "aISO-1.0")
    with pytest.raises(SdfFormatError, match="not valid for SDF version"):
        _ascii.loads(bad)


def test_loads_rejects_missing_newline() -> None:
    with pytest.raises(SdfFormatError, match="missing the header and data records"):
        _ascii.loads("aISO-2.0")


def test_loads_rejects_malformed_header_line() -> None:
    bad = ISO_ANNEX_A_EXAMPLE.replace("ManufacID = ISOTC213", "ManufacID ISOTC213")
    with pytest.raises(SdfFormatError, match="Malformed SDF header line"):
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


def test_dumps_rejects_compression() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, compression=1)
    with pytest.raises(SdfFormatError, match="Compressed SDF data areas"):
        _ascii.dumps(header, np.zeros((1, 1)))


def test_dumps_rejects_check_type() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, check_type=1)
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _ascii.dumps(header, np.zeros((1, 1)))


def test_dumps_rejects_malformed_v2_trailer() -> None:
    header = SdfHeader(version=SdfVersion.V2_0, num_points=1, num_profiles=1, data_type=7)
    with pytest.raises(SdfFormatError, match="must be well-formed XML"):
        _ascii.dumps(header, np.zeros((1, 1)), trailer="not xml")


def test_dumps_allows_empty_v2_trailer() -> None:
    header = SdfHeader(version=SdfVersion.V2_0, num_points=1, num_profiles=1, data_type=7)
    _ascii.dumps(header, np.zeros((1, 1)), trailer="")


def test_dumps_loads_roundtrips_empty_trailer_as_empty_string() -> None:
    # A blank trailer must not be confused with a trailer whose literal
    # content is the record terminator "*" (see the adjacent terminator
    # markers this produces with no content between them).
    header = SdfHeader(version=SdfVersion.V2_0, num_points=1, num_profiles=1, data_type=7)
    text = _ascii.dumps(header, np.zeros((1, 1)), trailer="")
    _header, _data, trailer = _ascii.loads(text)
    assert trailer == ""


def test_dumps_v1_trailer_need_not_be_xml() -> None:
    header = SdfHeader(version=SdfVersion.V1_0, num_points=1, num_profiles=1, data_type=7)
    _ascii.dumps(header, np.zeros((1, 1)), trailer="Operator = Jane Doe")


def test_dumps_rejects_non_ascii_trailer() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7)
    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _ascii.dumps(header, np.zeros((1, 1)), trailer="Müller café")


def test_dumps_rejects_non_ascii_manufacturer_id() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7, manufacturer_id="Müller")
    with pytest.raises(SdfFormatError, match="manufacturer_id must be 7-bit ASCII"):
        _ascii.dumps(header, np.zeros((1, 1)))


def test_loads_rejects_non_ascii_trailer() -> None:
    header = SdfHeader(num_points=1, num_profiles=1, data_type=7)
    text = _ascii.dumps(header, np.zeros((1, 1))) + "Müller café\r\n*\r\n"  # non-compliant trailer

    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _ascii.loads(text)


def test_dumps_roundtrips_loads() -> None:
    data = np.array([[1e-6, np.nan, -2.5e-6], [0.0, 3.14159e-6, 1.0e-6]])
    header = SdfHeader(
        version=SdfVersion.V2_0,
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
