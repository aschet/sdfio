# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Binary (fixed-width) representation of the SDF format."""

from __future__ import annotations

import io
import struct
from typing import IO

import numpy as np

from .datatypes import decode_raw, encode_raw, require_supported_data_type
from .exceptions import SdfFormatError
from .header import (
    Header,
    format_sdf_datetime,
    parse_sdf_datetime,
    validate_check_type,
    validate_compression,
    validate_for_write,
)
from .revision import ASCII_PREFIX, BINARY_PREFIX, MAGIC_SIZE, FileFormat, Revision

__all__ = ["HEADER_SIZE", "dump", "dumps", "load", "loads"]

# Field widths from the standard's record 1 header field table.
_FAMILY_SIZE = len("ISO")  # == len("BCR")
_MAGIC_PREFIX_SIZE = 1 + _FAMILY_SIZE + 1  # a/b + family + '-'
_MANUFACTURER_ID_SIZE = 10
_DATE_SIZE = 12

_STRINGS_STRUCT = struct.Struct(f"<{_MANUFACTURER_ID_SIZE}s{_DATE_SIZE}s{_DATE_SIZE}s")
_TAIL_STRUCT = struct.Struct("<ddddBBB")
_COUNTS_STRUCT = {
    Revision.ISO_1_0: struct.Struct("<HH"),
    Revision.ISO_2_0: struct.Struct("<II"),
    Revision.BCR_1_0: struct.Struct("<HH"),
}

#: Total header size in bytes for each SDF revision.
HEADER_SIZE = {
    revision: MAGIC_SIZE + _STRINGS_STRUCT.size + counts_struct.size + _TAIL_STRUCT.size
    for revision, counts_struct in _COUNTS_STRUCT.items()
}


def _pad(value: str, length: int) -> bytes:
    # Oversized values are silently truncated (not rejected) to fit the
    # fixed-width field; only ManufacID can realistically be long enough to
    # hit this, since the date fields are always exactly formatted per the
    # standard's header field table. Callers already validate ASCII-ness
    # before this point.
    return value.encode("ascii")[:length].ljust(length, b" ")


def _read_header_bytes(fp: IO[bytes], size: int) -> bytes:
    """Read exactly ``size`` bytes of the header.

    :raises SdfFormatError: If the file ends before.
    """
    data = fp.read(size)
    if len(data) != size:
        raise SdfFormatError("File is too short to contain an SDF header")
    return data


def load(fp: IO[bytes]) -> tuple[Header, np.ndarray, bytes]:
    """Parse a binary SDF file object opened in binary mode.

    :param fp: File object to read from, positioned at the start of the file.
    :returns: A ``(header, data, trailer)`` tuple. ``data`` is a
        ``(num_profiles, num_points)`` array of height values in metres,
        with ``NaN`` marking non-measured or spurious points.
    :raises SdfFormatError: If the file is malformed, e.g. a bad magic,
        unsupported revision or data type, or it ends within the header or the data area. The
        trailer is not validated on read; it is returned as-is even if not 7-bit
        ASCII. ``ManufacID`` is truncated at the first NUL byte: BCR-1.0 lets
        a string shorter than its field be NUL-terminated, and some ISO
        writers do the same instead of space-padding it as ISO requires.
    """
    magic = _read_header_bytes(fp, MAGIC_SIZE)
    magic_text = magic.decode("ascii", errors="replace")
    if (
        len(magic_text) != MAGIC_SIZE
        or magic_text[0].lower() not in (ASCII_PREFIX, BINARY_PREFIX)
        or magic_text[4] != "-"
    ):
        raise SdfFormatError(f"Not an SDF file, unexpected magic: {magic_text!r}")
    family_text = magic_text[1:4]
    version_text = magic_text[_MAGIC_PREFIX_SIZE:]
    revision = Revision.resolve(family_text, version_text, magic_text)
    if magic_text[0].lower() != BINARY_PREFIX:
        raise SdfFormatError("ASCII magic found while parsing a binary SDF file")

    manufacturer_raw, create_raw, mod_raw = _STRINGS_STRUCT.unpack(
        _read_header_bytes(fp, _STRINGS_STRUCT.size)
    )
    counts_struct = _COUNTS_STRUCT[revision]
    num_points, num_profiles = counts_struct.unpack(_read_header_bytes(fp, counts_struct.size))
    (
        x_scale,
        y_scale,
        z_scale,
        z_resolution,
        compression,
        data_type_code,
        check_type,
    ) = _TAIL_STRUCT.unpack(_read_header_bytes(fp, _TAIL_STRUCT.size))
    validate_compression(compression)
    validate_check_type(check_type)

    data_type = require_supported_data_type(data_type_code, revision)

    # errors="replace" here is a deliberate read-side leniency (unlike the
    # trailer, ManufacID/dates are not re-validated as ASCII on read) so a
    # file that's merely non-compliant here can still be opened and
    # inspected; only *writing* one is rejected (validate_manufacturer_id_ascii).
    # ManufacID is truncated at the first NUL byte before decoding: BCR-1.0
    # lets a string shorter than its field be NUL-terminated (what follows is
    # undefined), while ISO requires space-padding. Some ISO writers follow the
    # BCR rule; truncating handles both and keeps the undefined bytes out of
    # the decoded value.
    header = Header._read(
        revision=revision,
        format=FileFormat.BINARY,
        manufacturer_id=manufacturer_raw.split(b"\x00", 1)[0]
        .decode("ascii", errors="replace")
        .rstrip(),
        create_date=parse_sdf_datetime(create_raw.decode("ascii", errors="replace"), revision),
        mod_date=parse_sdf_datetime(mod_raw.decode("ascii", errors="replace"), revision),
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        z_resolution=z_resolution,
        data_type=data_type.type,
    )

    # The data area and the trailer are read together and split, instead of asking for the size
    # that the header claims: a damaged header can name a size that no file holds, which is not
    # something to pass to read().
    count = num_points * num_profiles
    byte_count = count * data_type.dtype.itemsize
    rest = fp.read()
    if len(rest) < byte_count:
        raise SdfFormatError("Unexpected end of file while reading the SDF data area")
    raw = np.frombuffer(rest, dtype=data_type.dtype, count=count)
    data = decode_raw(raw, data_type, header.z_scale, revision).reshape(num_profiles, num_points)
    data = revision.rules.order_rows(data)

    # Not validated on read, unlike on write: the trailer is secondary to
    # the header/data area, and a non-compliant trailer in an otherwise
    # valid file shouldn't prevent reading the (already successfully
    # decoded) depth data.
    trailer = rest[byte_count:]
    return header, data, trailer


def dump(header: Header, data: np.ndarray, fp: IO[bytes], trailer: bytes = b"") -> None:
    """Serialize a header/data pair to the binary SDF representation.

    :param header: Header describing ``data``; ``header.format`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional raw record 3 trailer content.
    """
    if header.revision not in _COUNTS_STRUCT:
        raise SdfFormatError(f"Unsupported SDF revision {header.revision!r}")
    data_type = validate_for_write(header, trailer)

    fp.write(f"{BINARY_PREFIX}{header.revision}".encode("ascii"))
    fp.write(_pad(header.manufacturer_id, _MANUFACTURER_ID_SIZE))
    fp.write(_pad(format_sdf_datetime(header.create_date, header.revision), _DATE_SIZE))
    fp.write(_pad(format_sdf_datetime(header.mod_date, header.revision), _DATE_SIZE))
    counts_struct = _COUNTS_STRUCT[header.revision]
    max_count = header.revision.rules.max_binary_count
    num_profiles, num_points = data.shape
    if max(data.shape) > max_count:
        raise SdfFormatError(
            f"NumPoints/NumProfiles must be between 0 and {max_count} for SDF "
            f"revision {header.revision} (binary format)"
        )
    fp.write(counts_struct.pack(num_points, num_profiles))
    fp.write(
        _TAIL_STRUCT.pack(
            header.x_scale,
            header.y_scale,
            header.z_scale,
            header.z_resolution,
            0,  # Compression: never supported, always written as 0.
            header.data_type,
            0,  # CheckType: never written, always 0 (see Header's docstring).
        )
    )

    write_data = header.revision.rules.order_rows(data)
    raw = encode_raw(write_data, data_type, header.z_scale, header.revision)
    fp.write(raw.tobytes())

    if trailer:
        fp.write(trailer)


def loads(data: bytes) -> tuple[Header, np.ndarray, bytes]:
    """Parse an in-memory binary SDF blob.

    Equivalent to :func:`load`, reading from an in-memory ``data`` buffer.

    :returns: A ``(header, data, trailer)`` tuple. ``data`` is a
        ``(num_profiles, num_points)`` array of height values in metres,
        with ``NaN`` marking non-measured or spurious points.
    """
    return load(io.BytesIO(data))


def dumps(header: Header, data: np.ndarray, trailer: bytes = b"") -> bytes:
    """Serialize a header/data pair to an in-memory binary SDF blob.

    :param header: Header describing ``data``; ``header.format`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional raw record 3 trailer content.
    """
    fp = io.BytesIO()
    dump(header, data, fp, trailer)
    return fp.getvalue()
