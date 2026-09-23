# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import struct
from typing import Any

import numpy as np
import pytest

from sdfio import _binary
from sdfio.exceptions import SdfFormatError
from sdfio.header import SdfDialect, SdfHeader


def _make_header(**overrides: Any) -> SdfHeader:
    defaults: dict[str, Any] = {
        "dialect": SdfDialect.ISO_2_0,
        "binary": True,
        "manufacturer_id": "sdfio",
        "num_points": 3,
        "num_profiles": 2,
        "x_scale": 1e-6,
        "y_scale": 1e-6,
        "z_scale": 1e-6,
        "z_resolution": -1.0,
        "data_type": 7,
    }
    defaults.update(overrides)
    return SdfHeader(**defaults)


def test_header_size_matches_iso_table_2() -> None:
    assert _binary.HEADER_SIZE[SdfDialect.ISO_1_0] == 81
    assert _binary.HEADER_SIZE[SdfDialect.ISO_2_0] == 85


@pytest.mark.parametrize("dialect", [SdfDialect.ISO_1_0, SdfDialect.ISO_2_0])
@pytest.mark.parametrize("data_type", [5, 6, 7])
def test_roundtrip(dialect: SdfDialect, data_type: int) -> None:
    header = _make_header(dialect=dialect, data_type=data_type)
    data = np.array([[1e-6, np.nan, -2e-6], [0.0, 3e-6, 1e-6]])

    buffer = io.BytesIO()
    _binary.dump(header, data, buffer, trailer=b"Note = hello")
    buffer.seek(0)
    read_header, read_data, trailer = _binary.load(buffer)

    assert read_header.dialect == dialect
    assert read_header.data_type == data_type
    assert read_header.num_points == 3
    assert read_header.num_profiles == 2
    assert read_header.manufacturer_id == "sdfio"
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9, equal_nan=True)
    assert trailer == b"Note = hello"


@pytest.mark.parametrize("data_type", [3, 4])
def test_roundtrip_iso_2_0_only_data_types(data_type: int) -> None:
    header = _make_header(
        dialect=SdfDialect.ISO_2_0, data_type=data_type, num_points=2, num_profiles=1
    )
    data = np.array([[1e-6, -1e-6]])
    buffer = io.BytesIO()
    _binary.dump(header, data, buffer)
    buffer.seek(0)
    _read_header, read_data, _trailer = _binary.load(buffer)
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9)


def test_load_rejects_ascii_magic() -> None:
    with pytest.raises(SdfFormatError, match="ASCII magic"):
        _binary.load(io.BytesIO(b"aISO-2.0" + b" " * 80))


def test_dump_rejects_out_of_range_integer_value() -> None:
    header = _make_header(data_type=4, num_points=1, num_profiles=1)  # int8
    with pytest.raises(SdfFormatError, match="out of range"):
        _binary.dump(header, np.array([[1000.0]]), io.BytesIO())


def test_dump_rejects_sentinel_collision() -> None:
    header = _make_header(data_type=4, num_points=1, num_profiles=1, z_scale=1.0)  # int8
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        _binary.dump(header, np.array([[-128.0]]), io.BytesIO())


def test_load_rejects_truncated_file() -> None:
    with pytest.raises(SdfFormatError, match="too short"):
        _binary.load(io.BytesIO(b"bISO"))


def test_load_rejects_unsupported_dialect() -> None:
    with pytest.raises(SdfFormatError, match="Unknown or unsupported SDF dialect"):
        _binary.load(io.BytesIO(b"bISO-3.0" + b" " * 80))


def test_load_rejects_compressed_data() -> None:
    header = _make_header()
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((2, 3)), buffer)
    corrupted = bytearray(buffer.getvalue())
    compression_offset = 8 + 10 + 12 + 12 + 4 + 4 + 8 * 4
    corrupted[compression_offset] = 1
    with pytest.raises(SdfFormatError, match="Compressed SDF data areas"):
        _binary.load(io.BytesIO(bytes(corrupted)))


def test_load_rejects_checksummed_data() -> None:
    header = _make_header()
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((2, 3)), buffer)
    corrupted = bytearray(buffer.getvalue())
    check_type_offset = 8 + 10 + 12 + 12 + 4 + 4 + 8 * 4 + 2
    corrupted[check_type_offset] = 1
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _binary.load(io.BytesIO(bytes(corrupted)))


def test_load_rejects_unsupported_data_type_for_dialect() -> None:
    # Hand-assemble an ISO-1.0 header (uint16 counts) with DataType=3
    # (binary32), which is only valid for ISO-2.0/BCR-1.0.
    payload = struct.pack(
        "<8s10s12s12sHHddddBBB",
        b"bISO-1.0",
        b"sdfio     ",
        b"010120240000",
        b"010120240000",
        1,
        1,
        1e-6,
        1e-6,
        1e-6,
        -1.0,
        0,
        3,
        0,
    )
    with pytest.raises(SdfFormatError, match="not valid for SDF dialect"):
        _binary.load(io.BytesIO(payload))


def test_load_rejects_truncated_data_area() -> None:
    header = _make_header(num_points=4, num_profiles=4)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((4, 4)), buffer)
    truncated = buffer.getvalue()[:-4]
    with pytest.raises(SdfFormatError, match="Unexpected end of file"):
        _binary.load(io.BytesIO(truncated))


def test_dump_rejects_shape_mismatch() -> None:
    header = _make_header(num_points=2, num_profiles=2)
    with pytest.raises(SdfFormatError, match="does not match header shape"):
        _binary.dump(header, np.zeros((3, 3)), io.BytesIO())


def test_dump_rejects_unsupported_dialect() -> None:
    header = _make_header(num_points=1, num_profiles=1)
    # Bypass SdfHeader.__post_init__ validation to test dump()'s own check.
    header.dialect = "3.0"  # type: ignore[assignment]
    with pytest.raises(SdfFormatError, match="Unsupported SDF dialect"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_dump_rejects_malformed_v2_trailer() -> None:
    header = _make_header(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1)
    with pytest.raises(SdfFormatError, match="tagged 'Name = Value' format"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"not tagged fields")


def test_dump_allows_tagged_v2_trailer() -> None:
    header = _make_header(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1)
    _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"Note = hello")


def test_dump_allows_empty_v2_trailer() -> None:
    header = _make_header(dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1)
    _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"")


def test_dump_rejects_non_ascii_trailer() -> None:
    header = _make_header(num_points=1, num_profiles=1)
    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer="Müller café".encode())


def test_dump_rejects_non_ascii_manufacturer_id() -> None:
    header = _make_header(num_points=1, num_profiles=1, manufacturer_id="Müller")
    with pytest.raises(SdfFormatError, match="manufacturer_id must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_load_returns_non_ascii_trailer_as_is() -> None:
    """A non-compliant trailer must not prevent reading the (already-decoded) depth data."""
    header = _make_header(num_points=1, num_profiles=1)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    buffer.write("Müller café".encode())  # append a non-compliant trailer after the fact
    buffer.seek(0)

    _, data, trailer = _binary.load(buffer)

    np.testing.assert_array_equal(data, np.zeros((1, 1)))
    assert trailer == "Müller café".encode()


def test_dump_rejects_data_type_invalid_for_dialect() -> None:
    header = _make_header(dialect=SdfDialect.ISO_1_0, data_type=3, num_points=2, num_profiles=1)
    with pytest.raises(SdfFormatError, match="not valid for SDF dialect"):
        _binary.dump(header, np.zeros((1, 2)), io.BytesIO())


def test_dump_rejects_num_points_exceeding_dialect_width() -> None:
    header = _make_header(dialect=SdfDialect.ISO_1_0, data_type=7, num_points=70000, num_profiles=1)
    with pytest.raises(SdfFormatError, match="NumPoints/NumProfiles must be between"):
        _binary.dump(header, np.zeros((1, 70000)), io.BytesIO())


def test_dumps_loads_roundtrip() -> None:
    header = _make_header(num_points=2, num_profiles=1)
    data = np.array([[1e-6, np.nan]])
    blob = _binary.dumps(header, data, trailer=b"Note = a")
    assert isinstance(blob, bytes)

    round_tripped_header, round_tripped_data, trailer = _binary.loads(blob)

    assert round_tripped_header.num_points == 2
    assert trailer == b"Note = a"
    np.testing.assert_allclose(round_tripped_data, data, equal_nan=True)


def test_roundtrip_bcr_dialect() -> None:
    header = _make_header(dialect=SdfDialect.BCR_1_0, data_type=3, num_points=2, num_profiles=1)
    data = np.array([[1e-6, -2e-6]])
    buffer = io.BytesIO()
    _binary.dump(header, data, buffer)
    buffer.seek(0)
    read_header, read_data, _trailer = _binary.load(buffer)

    assert read_header.dialect == SdfDialect.BCR_1_0
    assert read_header.magic == "bBCR-1.0"
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9)


def test_dump_reverses_profile_order_for_bcr_on_disk() -> None:
    # Each row has a distinct value so the on-disk row order is directly
    # observable, independent of the dialect-aware unflipping load() does.
    data = np.array([[0.0], [1.0], [2.0], [3.0]])
    iso_header = _make_header(
        dialect=SdfDialect.ISO_1_0, data_type=7, num_points=1, num_profiles=4, z_scale=1.0
    )
    bcr_header = _make_header(
        dialect=SdfDialect.BCR_1_0, data_type=7, num_points=1, num_profiles=4, z_scale=1.0
    )

    iso_blob = _binary.dumps(iso_header, data)
    bcr_blob = _binary.dumps(bcr_header, data)

    iso_on_disk = np.frombuffer(
        iso_blob[_binary.HEADER_SIZE[SdfDialect.ISO_1_0] :], dtype="<f8", count=4
    )
    bcr_on_disk = np.frombuffer(
        bcr_blob[_binary.HEADER_SIZE[SdfDialect.BCR_1_0] :], dtype="<f8", count=4
    )
    np.testing.assert_array_equal(iso_on_disk, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bcr_on_disk, [3.0, 2.0, 1.0, 0.0])

    # But reading either back gives the same logical (row 0 = y=0) array.
    _iso_header, iso_read_data, _t = _binary.loads(iso_blob)
    _bcr_header, bcr_read_data, _t = _binary.loads(bcr_blob)
    np.testing.assert_array_equal(iso_read_data, data)
    np.testing.assert_array_equal(bcr_read_data, data)


def test_roundtrip_none_dates() -> None:
    header = _make_header(num_points=1, num_profiles=1, create_date=None, mod_date=None)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    buffer.seek(0)
    read_header, _data, _trailer = _binary.load(buffer)
    assert read_header.create_date is None
    assert read_header.mod_date is None


def test_load_rejects_unknown_dialect() -> None:
    with pytest.raises(SdfFormatError, match="Unknown or unsupported SDF dialect"):
        _binary.load(io.BytesIO(b"bXYZ-1.0" + b" " * 80))


def test_load_rejects_bcr_checksum_too() -> None:
    # BCR's IntegerTrace checksum (check_type=1) is a value inline in the
    # data area, not a header value.
    header_bytes = struct.pack(
        "<8s10s12s12sHHddddBBB",
        b"bBCR-1.0",
        b"sdfio     ",
        b"010120240000",
        b"010120240000",
        2,
        2,
        1e-6,
        1e-6,
        1e-6,
        -1.0,
        0,
        7,  # binary64
        1,  # check_type: IntegerTrace
    )
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _binary.load(io.BytesIO(header_bytes))
