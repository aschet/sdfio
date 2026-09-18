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
from sdfio.header import SdfHeader, SdfVersion


def _make_header(**overrides: Any) -> SdfHeader:
    defaults: dict[str, Any] = {
        "version": SdfVersion.V2_0,
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
    assert _binary.HEADER_SIZE[SdfVersion.V1_0] == 81
    assert _binary.HEADER_SIZE[SdfVersion.V2_0] == 85


@pytest.mark.parametrize("version", [SdfVersion.V1_0, SdfVersion.V2_0])
@pytest.mark.parametrize("data_type", [5, 6, 7])
def test_roundtrip(version: SdfVersion, data_type: int) -> None:
    header = _make_header(version=version, data_type=data_type)
    data = np.array([[1e-6, np.nan, -2e-6], [0.0, 3e-6, 1e-6]])

    buffer = io.BytesIO()
    _binary.dump(header, data, buffer, trailer=b"<Note>hello</Note>")
    buffer.seek(0)
    read_header, read_data, trailer = _binary.load(buffer)

    assert read_header.version == version
    assert read_header.data_type == data_type
    assert read_header.num_points == 3
    assert read_header.num_profiles == 2
    assert read_header.manufacturer_id == "sdfio"
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9, equal_nan=True)
    assert trailer == b"<Note>hello</Note>"


@pytest.mark.parametrize("data_type", [3, 4])
def test_roundtrip_version_2_only_data_types(data_type: int) -> None:
    header = _make_header(
        version=SdfVersion.V2_0, data_type=data_type, num_points=2, num_profiles=1
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


def test_load_rejects_unsupported_version() -> None:
    with pytest.raises(SdfFormatError, match="Unsupported SDF version"):
        _binary.load(io.BytesIO(b"bISO-3.0" + b" " * 80))


def test_load_rejects_compressed_data() -> None:
    header = _make_header(compression=0)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((2, 3)), buffer)
    corrupted = bytearray(buffer.getvalue())
    compression_offset = 8 + 10 + 12 + 12 + 4 + 4 + 8 * 4
    corrupted[compression_offset] = 1
    with pytest.raises(SdfFormatError, match="Compressed SDF data areas"):
        _binary.load(io.BytesIO(bytes(corrupted)))


def test_load_rejects_checksummed_data() -> None:
    header = _make_header(check_type=0)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((2, 3)), buffer)
    corrupted = bytearray(buffer.getvalue())
    check_type_offset = 8 + 10 + 12 + 12 + 4 + 4 + 8 * 4 + 2
    corrupted[check_type_offset] = 1
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _binary.load(io.BytesIO(bytes(corrupted)))


def test_load_rejects_unsupported_data_type_for_version() -> None:
    # Hand-assemble a version "1.0" header (uint16 counts) with DataType=3
    # (binary32), which is only valid for version 2.0.
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
    with pytest.raises(SdfFormatError, match="not valid for SDF version"):
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


def test_dump_rejects_unsupported_version() -> None:
    header = _make_header(num_points=1, num_profiles=1)
    # Bypass SdfHeader.__post_init__ validation to test dump()'s own check.
    header.version = "3.0"  # type: ignore[assignment]
    with pytest.raises(SdfFormatError, match="Unsupported SDF version"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_dump_rejects_compression() -> None:
    header = _make_header(num_points=1, num_profiles=1, compression=1)
    with pytest.raises(SdfFormatError, match="Compressed SDF data areas"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_dump_rejects_check_type() -> None:
    header = _make_header(num_points=1, num_profiles=1, check_type=1)
    with pytest.raises(SdfFormatError, match="Checksummed SDF data areas"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_dump_rejects_malformed_v2_trailer() -> None:
    header = _make_header(version=SdfVersion.V2_0, num_points=1, num_profiles=1)
    with pytest.raises(SdfFormatError, match="must be well-formed XML"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"not xml")


def test_dump_allows_empty_v2_trailer() -> None:
    header = _make_header(version=SdfVersion.V2_0, num_points=1, num_profiles=1)
    _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"")


def test_dump_rejects_non_ascii_trailer() -> None:
    header = _make_header(num_points=1, num_profiles=1)
    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer="Müller café".encode())


def test_dump_rejects_non_ascii_manufacturer_id() -> None:
    header = _make_header(num_points=1, num_profiles=1, manufacturer_id="Müller")
    with pytest.raises(SdfFormatError, match="manufacturer_id must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_load_rejects_non_ascii_trailer() -> None:
    header = _make_header(num_points=1, num_profiles=1)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    buffer.write("Müller café".encode())  # append a non-compliant trailer after the fact
    buffer.seek(0)

    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _binary.load(buffer)


def test_dump_rejects_data_type_invalid_for_version() -> None:
    header = _make_header(version=SdfVersion.V1_0, data_type=3, num_points=2, num_profiles=1)
    with pytest.raises(SdfFormatError, match="not valid for SDF version"):
        _binary.dump(header, np.zeros((1, 2)), io.BytesIO())


def test_dump_rejects_num_points_exceeding_version_width() -> None:
    header = _make_header(version=SdfVersion.V1_0, data_type=7, num_points=70000, num_profiles=1)
    with pytest.raises(SdfFormatError, match="NumPoints/NumProfiles must be between"):
        _binary.dump(header, np.zeros((1, 70000)), io.BytesIO())


def test_dumps_loads_roundtrip() -> None:
    header = _make_header(num_points=2, num_profiles=1)
    data = np.array([[1e-6, np.nan]])
    blob = _binary.dumps(header, data, trailer=b"<a/>")
    assert isinstance(blob, bytes)

    round_tripped_header, round_tripped_data, trailer = _binary.loads(blob)

    assert round_tripped_header.num_points == 2
    assert trailer == b"<a/>"
    np.testing.assert_allclose(round_tripped_data, data, equal_nan=True)
