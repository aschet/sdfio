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
    ASCII_PREFIX,
    BINARY_PREFIX,
    SdfDialect,
    SdfHeader,
    SdfVersion,
    format_sdf_datetime,
    parse_sdf_datetime,
    validate_check_type,
    validate_compression,
    validate_manufacturer_id_ascii,
    validate_trailer_ascii,
    validate_trailer_xml,
    validate_z_scale,
)

__all__ = ["HEADER_SIZE", "dump", "dumps", "load", "loads"]

# Field widths from the standard's record 1 header field table.
_VERSION_SIZE = 8
_DIALECT_SIZE = len("ISO")  # == len("BCR")
_MAGIC_PREFIX_SIZE = 1 + _DIALECT_SIZE + 1  # a/b + dialect + '-'
_MANUFACTURER_ID_SIZE = 10
_DATE_SIZE = 12

_STRINGS_STRUCT = struct.Struct(f"<{_MANUFACTURER_ID_SIZE}s{_DATE_SIZE}s{_DATE_SIZE}s")
_TAIL_STRUCT = struct.Struct("<ddddBBB")
_COUNTS_STRUCT = {SdfVersion.V1_0: struct.Struct("<HH"), SdfVersion.V2_0: struct.Struct("<II")}

#: Total header size in bytes for each SDF version.
HEADER_SIZE = {
    version: _VERSION_SIZE + _STRINGS_STRUCT.size + counts_struct.size + _TAIL_STRUCT.size
    for version, counts_struct in _COUNTS_STRUCT.items()
}


def _pad(value: str, length: int) -> bytes:
    # Oversized values are silently truncated (not rejected) to fit the
    # fixed-width field; only ManufacID can realistically be long enough to
    # hit this, since the date fields are always exactly formatted per the
    # standard's header field table. Callers already validate ASCII-ness
    # before this point.
    return value.encode("ascii")[:length].ljust(length, b" ")


def load(fp: IO[bytes]) -> tuple[SdfHeader, np.ndarray, bytes]:
    """Parse a binary SDF file object opened in binary mode.

    :param fp: File object to read from, positioned at the start of the file.
    :returns: A ``(header, data, trailer)`` tuple. ``data`` is a
        ``(num_profiles, num_points)`` array of height values in metres,
        with ``NaN`` marking non-measured or spurious points.
    :raises SdfFormatError: If the file is malformed, e.g. a bad magic,
        unsupported version or data type, wrong data length, or a trailer
        that isn't 7-bit ASCII.
    """
    magic = fp.read(_VERSION_SIZE)
    if len(magic) != _VERSION_SIZE:
        raise SdfFormatError("File is too short to contain an SDF header")
    magic_text = magic.decode("ascii", errors="replace")
    if (
        len(magic_text) != _VERSION_SIZE
        or magic_text[0] not in (ASCII_PREFIX, BINARY_PREFIX)
        or magic_text[4] != "-"
    ):
        raise SdfFormatError(f"Not an SDF file, unexpected magic: {magic_text!r}")
    dialect_text = magic_text[1:4]
    try:
        dialect = SdfDialect(dialect_text)
    except ValueError:
        raise SdfFormatError(
            f"Unknown SDF dialect {dialect_text!r} in magic {magic_text!r}"
        ) from None
    if magic_text[0] != BINARY_PREFIX:
        raise SdfFormatError("ASCII magic found while parsing a binary SDF file")
    version_text = magic_text[_MAGIC_PREFIX_SIZE:]
    if version_text not in _COUNTS_STRUCT:
        raise SdfFormatError(f"Unsupported SDF version {version_text!r}")
    version = SdfVersion(version_text)

    manufacturer_raw, create_raw, mod_raw = _STRINGS_STRUCT.unpack(fp.read(_STRINGS_STRUCT.size))
    counts_struct = _COUNTS_STRUCT[version]
    num_points, num_profiles = counts_struct.unpack(fp.read(counts_struct.size))
    (
        x_scale,
        y_scale,
        z_scale,
        z_resolution,
        compression,
        data_type_code,
        check_type,
    ) = _TAIL_STRUCT.unpack(fp.read(_TAIL_STRUCT.size))
    validate_compression(compression)
    validate_check_type(check_type)

    data_type = require_supported_data_type(data_type_code, version, dialect)

    # errors="replace" here is a deliberate read-side leniency (unlike the
    # trailer, ManufacID/dates are not re-validated as ASCII on read) so a
    # file that's merely non-compliant here can still be opened and
    # inspected; only *writing* one is rejected (validate_manufacturer_id_ascii).
    header = SdfHeader(
        version=version,
        dialect=dialect,
        binary=True,
        manufacturer_id=manufacturer_raw.decode("ascii", errors="replace").rstrip(),
        create_date=parse_sdf_datetime(create_raw.decode("ascii", errors="replace"), version),
        mod_date=parse_sdf_datetime(mod_raw.decode("ascii", errors="replace"), version),
        num_points=num_points,
        num_profiles=num_profiles,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        z_resolution=z_resolution,
        data_type=data_type_code,
    )

    count = header.num_points * header.num_profiles
    byte_count = count * data_type.dtype.itemsize
    raw_bytes = fp.read(byte_count)
    if len(raw_bytes) != byte_count:
        raise SdfFormatError("Unexpected end of file while reading the SDF data area")
    raw = np.frombuffer(raw_bytes, dtype=data_type.dtype)
    data = decode_raw(raw, data_type, header.z_scale, dialect).reshape(
        header.num_profiles, header.num_points
    )

    trailer = fp.read()
    validate_trailer_ascii(trailer)
    return header, data, trailer


def dump(header: SdfHeader, data: np.ndarray, fp: IO[bytes], trailer: bytes = b"") -> None:
    """Serialize a header/data pair to the binary SDF representation.

    :param header: Header describing ``data``; ``header.binary`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional raw record 3 trailer content.
    """
    if data.shape != header.shape:
        raise SdfFormatError(f"Data shape {data.shape} does not match header shape {header.shape}")
    if header.version not in _COUNTS_STRUCT:
        raise SdfFormatError(f"Unsupported SDF version {header.version!r}")
    validate_z_scale(header.z_scale)
    validate_manufacturer_id_ascii(header.manufacturer_id)
    validate_trailer_ascii(trailer)
    validate_trailer_xml(header.version, trailer)
    data_type = require_supported_data_type(header.data_type, header.version, header.dialect)

    fp.write(f"{BINARY_PREFIX}{header.dialect}-{header.version}".encode("ascii"))
    fp.write(_pad(header.manufacturer_id, _MANUFACTURER_ID_SIZE))
    fp.write(_pad(format_sdf_datetime(header.create_date, header.version), _DATE_SIZE))
    fp.write(_pad(format_sdf_datetime(header.mod_date, header.version), _DATE_SIZE))
    counts_struct = _COUNTS_STRUCT[header.version]
    # counts_struct.size is NumPoints+NumProfiles combined (e.g. 4 bytes for
    # v1.0's two uint16 fields); halving it gives one field's byte width, and
    # from that its max unsigned value (65535 for uint16, 2**32-1 for uint32).
    max_count = 2 ** (8 * (counts_struct.size // 2)) - 1
    if not (0 <= header.num_points <= max_count and 0 <= header.num_profiles <= max_count):
        raise SdfFormatError(
            f"NumPoints/NumProfiles must be between 0 and {max_count} for SDF "
            f"version {header.version} (binary format)"
        )
    fp.write(counts_struct.pack(header.num_points, header.num_profiles))
    fp.write(
        _TAIL_STRUCT.pack(
            header.x_scale,
            header.y_scale,
            header.z_scale,
            header.z_resolution,
            0,  # Compression: never supported, always written as 0.
            header.data_type,
            0,  # CheckType: never written, always 0 (see SdfHeader's docstring).
        )
    )

    raw = encode_raw(data, data_type, header.z_scale, header.dialect)
    fp.write(raw.tobytes())

    if trailer:
        fp.write(trailer)


def loads(data: bytes) -> tuple[SdfHeader, np.ndarray, bytes]:
    """Parse an in-memory binary SDF blob.

    Equivalent to :func:`load`, reading from an in-memory ``data`` buffer.

    :returns: A ``(header, data, trailer)`` tuple. ``data`` is a
        ``(num_profiles, num_points)`` array of height values in metres,
        with ``NaN`` marking non-measured or spurious points.
    """
    return load(io.BytesIO(data))


def dumps(header: SdfHeader, data: np.ndarray, trailer: bytes = b"") -> bytes:
    """Serialize a header/data pair to an in-memory binary SDF blob.

    :param header: Header describing ``data``; ``header.binary`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional raw record 3 trailer content.
    """
    fp = io.BytesIO()
    dump(header, data, fp, trailer)
    return fp.getvalue()
