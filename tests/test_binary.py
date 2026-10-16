# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import struct
import warnings
from typing import Any

import numpy as np
import pytest

import sdfio
from sdfio import _binary
from sdfio.datatypes import DataType
from sdfio.exceptions import SdfFormatError
from sdfio.file import SdfFile
from sdfio.header import MAGIC_SIZE, FileFormat, Header, Revision


def _make_header(**overrides: Any) -> Header:
    defaults: dict[str, Any] = {
        "revision": Revision.ISO_2_0,
        "format": FileFormat.BINARY,
        "manufacturer_id": "sdfio",
        "x_scale": 1e-6,
        "y_scale": 1e-6,
        "z_scale": 1e-6,
        "z_resolution": -1.0,
        "data_type": 7,
    }
    defaults.update(overrides)
    return Header(**defaults)


def test_header_size_matches_iso_table_2() -> None:
    assert _binary.HEADER_SIZE[Revision.ISO_1_0] == 81
    assert _binary.HEADER_SIZE[Revision.ISO_2_0] == 85


@pytest.mark.parametrize("revision", [Revision.ISO_1_0, Revision.ISO_2_0])
@pytest.mark.parametrize("data_type", [5, 6, 7])
def test_roundtrip(revision: Revision, data_type: int) -> None:
    header = _make_header(revision=revision, data_type=data_type)
    data = np.array([[1e-6, np.nan, -2e-6], [0.0, 3e-6, 1e-6]])

    buffer = io.BytesIO()
    _binary.dump(header, data, buffer, trailer=b"Note = hello")
    buffer.seek(0)
    read_header, read_data, trailer = _binary.load(buffer)

    assert read_header.revision == revision
    assert read_header.data_type == data_type
    assert read_data.shape == (2, 3)
    assert read_header.manufacturer_id == "sdfio"
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9, equal_nan=True)
    assert trailer == b"Note = hello"


@pytest.mark.parametrize("data_type", [3, 4])
def test_roundtrip_iso_2_0_only_data_types(data_type: int) -> None:
    header = _make_header(revision=Revision.ISO_2_0, data_type=data_type)
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
    header = _make_header(data_type=4)  # int8
    with pytest.raises(SdfFormatError, match="out of range"):
        _binary.dump(header, np.array([[1000.0]]), io.BytesIO())


def test_dump_rejects_sentinel_collision() -> None:
    header = _make_header(data_type=4, z_scale=1.0)  # int8
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        _binary.dump(header, np.array([[-128.0]]), io.BytesIO())


def test_load_rejects_truncated_file() -> None:
    with pytest.raises(SdfFormatError, match="too short"):
        _binary.load(io.BytesIO(b"bISO"))


def test_load_rejects_unsupported_revision() -> None:
    with pytest.raises(SdfFormatError, match="Unknown or unsupported SDF revision"):
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


def test_load_rejects_unsupported_data_type_for_revision() -> None:
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
    with pytest.raises(SdfFormatError, match="not valid for SDF revision"):
        _binary.load(io.BytesIO(payload))


def test_load_rejects_truncated_data_area() -> None:
    header = _make_header()
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((4, 4)), buffer)
    truncated = buffer.getvalue()[:-4]
    with pytest.raises(SdfFormatError, match="Unexpected end of file"):
        _binary.load(io.BytesIO(truncated))


def test_dump_rejects_unsupported_revision() -> None:
    header = _make_header()
    # Bypass Header.__post_init__ validation to test dump()'s own check.
    header.revision = "3.0"  # type: ignore[assignment]
    with pytest.raises(SdfFormatError, match="Unsupported SDF revision"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_dump_rejects_malformed_v2_trailer() -> None:
    header = _make_header(revision=Revision.ISO_2_0)
    with pytest.raises(SdfFormatError, match="tagged 'Name = Value' format"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"not tagged fields")


def test_dump_allows_tagged_v2_trailer() -> None:
    header = _make_header(revision=Revision.ISO_2_0)
    _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"Note = hello")


def test_dump_allows_empty_v2_trailer() -> None:
    header = _make_header(revision=Revision.ISO_2_0)
    _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer=b"")


def test_dump_rejects_non_ascii_trailer() -> None:
    header = _make_header()
    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO(), trailer="Müller café".encode())


def test_dump_rejects_non_ascii_manufacturer_id() -> None:
    header = _make_header(manufacturer_id="Müller")
    with pytest.raises(SdfFormatError, match="manufacturer_id must be 7-bit ASCII"):
        _binary.dump(header, np.zeros((1, 1)), io.BytesIO())


def test_load_truncates_manufacturer_id_at_first_nul() -> None:
    """BCR allows a NUL-terminated ManufacID (some ISO writers do too); strip past the NUL."""
    header = _make_header()
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    raw = bytearray(buffer.getvalue())
    manufacid_offset = raw.index(b"sdfio")
    raw[manufacid_offset + len(b"sdfio") : manufacid_offset + 10] = b"\x00\x07\x00\x00\x02"

    read_header, _data, _trailer = _binary.load(io.BytesIO(bytes(raw)))

    assert read_header.manufacturer_id == "sdfio"


def test_load_returns_non_ascii_trailer_as_is() -> None:
    """A non-compliant trailer must not prevent reading the (already-decoded) depth data."""
    header = _make_header()
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    buffer.write("Müller café".encode())  # append a non-compliant trailer after the fact
    buffer.seek(0)

    _, data, trailer = _binary.load(buffer)

    np.testing.assert_array_equal(data, np.zeros((1, 1)))
    assert trailer == "Müller café".encode()


def test_dump_rejects_data_type_invalid_for_revision() -> None:
    header = _make_header(revision=Revision.ISO_1_0, data_type=3)
    with pytest.raises(SdfFormatError, match="not valid for SDF revision"):
        _binary.dump(header, np.zeros((1, 2)), io.BytesIO())


def test_dump_rejects_num_points_exceeding_revision_width() -> None:
    header = _make_header(revision=Revision.ISO_1_0, data_type=7)
    with pytest.raises(SdfFormatError, match="NumPoints/NumProfiles must be between"):
        _binary.dump(header, np.zeros((1, 70000)), io.BytesIO())


def test_dumps_loads_roundtrip() -> None:
    header = _make_header()
    data = np.array([[1e-6, np.nan]])
    blob = _binary.dumps(header, data, trailer=b"Note = a")
    assert isinstance(blob, bytes)

    _header, round_tripped_data, trailer = _binary.loads(blob)

    assert round_tripped_data.shape == (1, 2)
    assert trailer == b"Note = a"
    np.testing.assert_allclose(round_tripped_data, data, equal_nan=True)


def test_roundtrip_bcr_revision() -> None:
    header = _make_header(revision=Revision.BCR_1_0, data_type=3)
    data = np.array([[1e-6, -2e-6]])
    buffer = io.BytesIO()
    _binary.dump(header, data, buffer)
    buffer.seek(0)
    read_header, read_data, _trailer = _binary.load(buffer)

    assert read_header.revision == Revision.BCR_1_0
    assert read_header.magic == "bBCR-1.0"
    np.testing.assert_allclose(read_data, data, rtol=1e-6, atol=1e-9)


def test_dump_reverses_profile_order_for_bcr_on_disk() -> None:
    # Each row has a distinct value so the on-disk row order is directly
    # observable, independent of the revision-aware unflipping load() does.
    data = np.array([[0.0], [1.0], [2.0], [3.0]])
    iso_header = _make_header(revision=Revision.ISO_1_0, data_type=7, z_scale=1.0)
    bcr_header = _make_header(revision=Revision.BCR_1_0, data_type=7, z_scale=1.0)

    iso_blob = _binary.dumps(iso_header, data)
    bcr_blob = _binary.dumps(bcr_header, data)

    iso_on_disk = np.frombuffer(
        iso_blob[_binary.HEADER_SIZE[Revision.ISO_1_0] :], dtype="<f8", count=4
    )
    bcr_on_disk = np.frombuffer(
        bcr_blob[_binary.HEADER_SIZE[Revision.BCR_1_0] :], dtype="<f8", count=4
    )
    np.testing.assert_array_equal(iso_on_disk, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bcr_on_disk, [3.0, 2.0, 1.0, 0.0])

    # But reading either back gives the same logical (row 0 = y=0) array.
    _iso_header, iso_read_data, _t = _binary.loads(iso_blob)
    _bcr_header, bcr_read_data, _t = _binary.loads(bcr_blob)
    np.testing.assert_array_equal(iso_read_data, data)
    np.testing.assert_array_equal(bcr_read_data, data)


def test_roundtrip_none_dates() -> None:
    header = _make_header(create_date=None, mod_date=None)
    buffer = io.BytesIO()
    _binary.dump(header, np.zeros((1, 1)), buffer)
    buffer.seek(0)
    read_header, _data, _trailer = _binary.load(buffer)
    assert read_header.create_date is None
    assert read_header.mod_date is None


def test_load_rejects_unknown_revision() -> None:
    with pytest.raises(SdfFormatError, match="Unknown or unsupported SDF revision"):
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


def test_loads_accepts_upper_case_binary_prefix() -> None:
    # BCR-1.0 allows "b" or "B" as the first magic character.
    header = _make_header(revision=Revision.BCR_1_0, data_type=7)
    blob = _binary.dumps(header, np.zeros((1, 1)))
    read_header, _data, _trailer = _binary.loads(b"B" + blob[1:])
    assert read_header.revision == Revision.BCR_1_0


@pytest.mark.parametrize("revision", list(Revision))
def test_max_binary_count_is_what_the_binary_header_can_hold(revision: Revision) -> None:
    field_size = _binary._COUNTS_STRUCT[revision].size // 2  # NumPoints and NumProfiles
    assert revision.rules.max_binary_count == 2 ** (8 * field_size) - 1


@pytest.mark.parametrize("revision", [Revision.ISO_1_0, Revision.BCR_1_0])
def test_binary_files_of_the_older_revisions_hold_at_most_the_max_count(
    revision: Revision,
) -> None:
    limit = revision.rules.max_binary_count
    data_type = DataType.INT16

    def write(points: int, format: FileFormat) -> bytes:
        header = Header(
            revision=revision,
            format=format,
            z_scale=1e-9,
            data_type=data_type,
        )
        return SdfFile(header=header, data=np.zeros((1, points))).dumps()

    assert sdfio.SdfFile.loads(write(limit, format=FileFormat.BINARY)).data.shape == (1, limit)
    with pytest.raises(SdfFormatError, match=str(limit)):
        write(limit + 1, format=FileFormat.BINARY)
    assert sdfio.SdfFile.loads(write(limit + 1, format=FileFormat.ASCII)).data.shape == (
        1,
        limit + 1,
    )


def _binary_file(revision: Revision) -> bytes:
    buffer = io.BytesIO()
    sdfio.write(
        buffer,
        np.arange(6, dtype=np.float64).reshape(2, 3) * 1e-6,
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale=1e-9,
        metadata=sdfio.Metadata(revision=revision, data_type=DataType.INT16),
        trailer="a = b" if revision is Revision.ISO_2_0 else "free",
    )
    return buffer.getvalue()


@pytest.mark.parametrize("revision", list(Revision))
def test_a_binary_file_that_ends_in_the_header_is_a_format_error(revision: Revision) -> None:
    blob = _binary_file(revision)

    for length in range(_binary.HEADER_SIZE[revision]):
        # Shorter than the magic, it is not recognized as an SDF file at all.
        message = "too short" if length >= MAGIC_SIZE else "unexpected magic"
        with pytest.raises(SdfFormatError, match=message):
            sdfio.SdfFile.loads(blob[:length])


@pytest.mark.parametrize("revision", list(Revision))
def test_a_binary_file_cut_anywhere_never_raises_anything_but_a_format_error(
    revision: Revision,
) -> None:
    blob = _binary_file(revision)
    header_size = _binary.HEADER_SIZE[revision]
    data_end = header_size + 6 * 2  # the data area: 6 int16 values

    for length in range(len(blob)):
        try:
            sdfio.SdfFile.loads(blob[:length])
        except SdfFormatError:
            assert length < data_end  # a shorter trailer is still a valid file
        else:
            assert length >= data_end


@pytest.mark.parametrize("revision", list(Revision))
def test_a_header_that_names_more_data_than_any_file_holds_is_a_format_error(
    revision: Revision,
) -> None:
    blob = bytearray(_binary_file(revision))
    counts = _binary._COUNTS_STRUCT[revision]
    offset = 8 + 10 + 24  # the magic, the manufacturer's ID and the two dates
    largest = revision.rules.max_binary_count
    blob[offset : offset + counts.size] = counts.pack(largest, largest)

    with pytest.raises(SdfFormatError, match="Unexpected end of file"):
        sdfio.SdfFile.loads(bytes(blob))


@pytest.mark.parametrize("z_scale", [float("inf"), float("-inf"), float("nan"), 1e308, 0.0, -1.0])
@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.BINARY64])
def test_a_z_scale_without_a_sensible_product_is_read_without_a_warning(
    z_scale: float, data_type: DataType
) -> None:
    # The writer refuses such a scale, so a file that holds one is made by hand.
    header = Header(z_scale=1e-9, data_type=data_type)
    valid = sdfio.SdfFile(header=header, data=np.zeros((2, 3))).dumps()
    scale_at = 8 + 10 + 24 + 8 + 8 + 8  # up to Zscale in a version 2.0 header
    blob = valid[:scale_at] + struct.pack("<d", z_scale) + valid[scale_at + 8 :]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = sdfio.SdfFile.loads(blob)

    assert loaded.data.shape == (2, 3)
