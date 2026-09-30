# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""ASCII (text) representation of the SDF format."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import IO

import numpy as np

from ._numeric import format_scientific
from .datatypes import DataType, SdfDataType, require_supported_data_type, validate_data_range
from .exceptions import SdfFormatError
from .header import (
    ASCII_PREFIX,
    BINARY_PREFIX,
    SdfDialect,
    SdfHeader,
    format_sdf_datetime,
    parse_sdf_datetime,
    parse_tagged_fields,
    validate_check_type,
    validate_compression,
    validate_manufacturer_id_ascii,
    validate_trailer_ascii,
    validate_trailer_tagged,
    validate_z_scale,
)

__all__ = ["dump", "dumps", "load", "loads"]

_MAGIC_RE = re.compile(
    rf"^(?P<prefix>[{ASCII_PREFIX}{BINARY_PREFIX}])(?P<dialect>[A-Z]{{3}})-(?P<version>\d\.\d)$"
)
_INVALID_MARKER = "BAD"
# The standard mandates <CRLF> line endings; dumps() always writes them.
# Reading also accepts a bare LF or CR, which BCR-1.0's specification allows.
_LINE_SPLIT_RE = re.compile(r"\r\n|\n|\r")
_TERMINATOR_RE = re.compile(r"^[ \t]*\*[ \t]*$")
_COMMENT_MARKER = ";"

# BCR-1.0 lets an ASCII header spell these fields as names instead of codes,
# matched case-insensitively. For the data type these are the specification's
# suggested abbreviations, not the full type names.
_BCR_DATA_TYPE_NAMES = {
    "UCHAR": DataType.UINT8,
    "UINTEGER": DataType.UINT16,
    "ULONGINT": DataType.UINT32,
    "FLOAT": DataType.BINARY32,
    "CHAR": DataType.INT8,
    "INTEGER": DataType.INT16,
    "LONGINT": DataType.INT32,
    "DOUBLE": DataType.BINARY64,
}
# "NULL" is what the specification's own ASCII example uses for "none".
_BCR_COMPRESSION_NAMES = {"NULL": 0, "NONE": 0, "RLL": 1}
_BCR_CHECK_TYPE_NAMES = {"NULL": 0, "NONE": 0, "UINTTRACE": 1}


def _format_scale_field(value: float) -> str:
    # Xscale/Yscale/Zscale/Zresolution are always written at binary64
    # precision, regardless of the data area's own DataType.
    return format_scientific(value, 14, 3)


def _strip_comment(line: str) -> str:
    return line.split(_COMMENT_MARKER, 1)[0]


def _split_records(remainder: str, dialect: SdfDialect) -> list[str]:
    # Splitting by matching the "*" delimiter together with its surrounding
    # newlines (as one combined pattern) can't tell two delimiters directly
    # adjacent to each other (an explicit empty record, e.g. an empty
    # trailer written as "*<CRLF>*<CRLF>") apart from a single delimiter --
    # the newline between them would need to be consumed by both matches at
    # once. Splitting into lines first and testing each one for being a bare
    # "*" avoids that ambiguity.
    # With comments enabled, a "*" followed by a comment still terminates its
    # record, and the comment is dropped from the header and data records. The
    # trailer (third record) is free text, so it is kept verbatim.
    comments = dialect.allows_ascii_comments
    records = []
    current: list[str] = []
    for line in _LINE_SPLIT_RE.split(remainder):
        bare = _strip_comment(line) if comments else line
        if _TERMINATOR_RE.match(bare):
            records.append("\n".join(current))
            current = []
        else:
            current.append(bare if comments and len(records) < 2 else line)
    records.append("\n".join(current))
    return records


def _field(fields: Mapping[str, str], name: str) -> str:
    for key, value in fields.items():
        if key.lower() == name.lower():
            return value
    raise SdfFormatError(f"Missing required SDF header field {name!r}")


def _parse_int(fields: Mapping[str, str], name: str) -> int:
    value = _field(fields, name)
    try:
        return int(value)
    except ValueError:
        raise SdfFormatError(
            f"Invalid integer value for SDF header field {name!r}: {value!r}"
        ) from None


def _parse_coded(
    fields: Mapping[str, str], name: str, dialect: SdfDialect, symbols: Mapping[str, int]
) -> int:
    value = _field(fields, name)
    try:
        return int(value)
    except ValueError:
        if dialect.allows_symbolic_header_values and value.upper() in symbols:
            return int(symbols[value.upper()])
        raise SdfFormatError(
            f"Invalid integer value for SDF header field {name!r}: {value!r}"
        ) from None


def _parse_float(fields: Mapping[str, str], name: str) -> float:
    value = _field(fields, name)
    try:
        return float(value)
    except ValueError:
        raise SdfFormatError(
            f"Invalid float value for SDF header field {name!r}: {value!r}"
        ) from None


def loads(text: str) -> tuple[SdfHeader, np.ndarray, str]:
    """Parse an in-memory ASCII SDF document.

    :param text: Full contents of an ASCII-encoded ``.sdf`` file.
    :returns: A ``(header, data, trailer)`` tuple. ``data`` is a
        ``(num_profiles, num_points)`` array of height values in metres,
        with ``NaN`` marking non-measured or spurious points.
    :raises SdfFormatError: If the file is malformed, e.g. a bad magic,
        unsupported dialect or data type, or missing records. The trailer is
        not validated on read; it is returned as-is even if not 7-bit ASCII
        (any non-ASCII byte was already replaced during decoding, per
        ``errors="replace"`` in :meth:`SdfFile.loads`).
    """
    text = text.lstrip("\ufeff")
    first_newline = _LINE_SPLIT_RE.search(text)
    if first_newline is None:
        raise SdfFormatError("SDF file is missing the header and data records")
    magic_line = text[: first_newline.start()].strip()
    remainder = text[first_newline.end() :]

    magic_match = _MAGIC_RE.match(_strip_comment(magic_line).strip())
    if not magic_match:
        raise SdfFormatError(f"Not an ASCII SDF file, unexpected magic: {magic_line!r}")
    if magic_match.group("prefix") != ASCII_PREFIX:
        raise SdfFormatError("Binary magic found while parsing an ASCII SDF file")
    dialect_text = magic_match.group("dialect")
    version_text = magic_match.group("version")
    dialect = SdfDialect.resolve(dialect_text, version_text, magic_line)
    if _COMMENT_MARKER in magic_line and not dialect.allows_ascii_comments:
        raise SdfFormatError(f"Not an ASCII SDF file, unexpected magic: {magic_line!r}")

    records = _split_records(remainder, dialect)
    if len(records) < 3:
        raise SdfFormatError("SDF file must contain header, data and trailer records")
    # The trailer is itself terminated by its own "*" record; drop
    # a resulting trailing empty segment instead of re-joining it back in.
    header_text, data_text, *trailer_parts = records
    if trailer_parts and trailer_parts[-1] == "":
        trailer_parts = trailer_parts[:-1]
    trailer_text = "*".join(trailer_parts)

    fields = parse_tagged_fields(header_text)
    data_type_code = _parse_coded(fields, "DataType", dialect, _BCR_DATA_TYPE_NAMES)
    data_type = require_supported_data_type(data_type_code, dialect)
    validate_compression(_parse_coded(fields, "Compression", dialect, _BCR_COMPRESSION_NAMES))
    validate_check_type(_parse_coded(fields, "CheckType", dialect, _BCR_CHECK_TYPE_NAMES))

    header = SdfHeader(
        dialect=dialect,
        binary=False,
        manufacturer_id=_field(fields, "ManufacID").strip(),
        create_date=parse_sdf_datetime(_field(fields, "CreateDate"), dialect),
        mod_date=parse_sdf_datetime(_field(fields, "ModDate"), dialect),
        num_points=_parse_int(fields, "NumPoints"),
        num_profiles=_parse_int(fields, "NumProfiles"),
        x_scale=_parse_float(fields, "Xscale"),
        y_scale=_parse_float(fields, "Yscale"),
        z_scale=_parse_float(fields, "Zscale"),
        z_resolution=_parse_float(fields, "Zresolution"),
        data_type=data_type_code,
    )

    data = _parse_data(data_text, header, data_type)
    # Not validated on read, unlike on write: the trailer is secondary to
    # the header/data area, and a non-compliant trailer in an otherwise
    # valid file shouldn't prevent reading the (already successfully
    # parsed) depth data.
    trailer = trailer_text.strip()
    return header, data, trailer


def _parse_data(data_text: str, header: SdfHeader, data_type: SdfDataType) -> np.ndarray:
    tokens = data_text.split()
    expected = header.num_points * header.num_profiles
    if len(tokens) != expected:
        raise SdfFormatError(f"Expected {expected} data values, found {len(tokens)}")

    is_float = data_type.type in (DataType.BINARY32, DataType.BINARY64)
    values = np.empty(header.num_points * header.num_profiles, dtype=np.float64)
    for index, token in enumerate(tokens):
        # The standard only specifies the literal "BAD"; matching
        # case-insensitively is a deliberate leniency, not mandated.
        if token.upper() == _INVALID_MARKER:
            values[index] = np.nan
        else:
            try:
                # int(token) first for non-float types so a fractional token
                # (e.g. "1.5" for an int16 field) is rejected outright,
                # rather than silently truncated by float()'s wider parsing.
                values[index] = float(token) if is_float else float(int(token))
            except ValueError:
                raise SdfFormatError(f"Invalid data value at position {index}: {token!r}") from None

    invalid_mask = np.isnan(values)
    scaled = values * header.z_scale
    scaled[invalid_mask] = np.nan
    data = scaled.reshape(header.num_profiles, header.num_points)
    return data[::-1] if header.dialect.reverses_profile_order else data


def load(fp: IO[str]) -> tuple[SdfHeader, np.ndarray, str]:
    """Parse an ASCII SDF file object opened in text mode.

    Equivalent to :func:`loads`, reading the text from ``fp`` first.
    """
    return loads(fp.read())


def _format_value(value: float, data_type: SdfDataType) -> str:
    if np.isnan(value):
        return _INVALID_MARKER
    if data_type.type == DataType.BINARY32:
        return format_scientific(value, 6, 2)
    if data_type.type == DataType.BINARY64:
        return format_scientific(value, 14, 3)
    return str(round(value))


def dumps(header: SdfHeader, data: np.ndarray, trailer: str = "") -> str:
    """Serialize a header/data pair to the ASCII SDF text representation.

    :param header: Header describing ``data``; ``header.binary`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional record 3 trailer content.

    For :attr:`~sdfio.DataType.BINARY64`, this is lossy at the ~1e-15
    relative level: the standard's mandated 15 significant digits are one
    short of the 17 an IEEE 754 double needs to round-trip exactly (see
    :func:`sdfio._numeric.format_scientific`). Use the binary format to
    avoid this.
    """
    if data.shape != header.shape:
        raise SdfFormatError(f"Data shape {data.shape} does not match header shape {header.shape}")
    validate_z_scale(header.z_scale)
    validate_manufacturer_id_ascii(header.manufacturer_id)
    validate_trailer_ascii(trailer)
    validate_trailer_tagged(header.dialect, trailer)
    data_type = require_supported_data_type(header.data_type, header.dialect)

    lines = [f"{ASCII_PREFIX}{header.dialect}"]
    fields = (
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", format_sdf_datetime(header.create_date, header.dialect)),
        ("ModDate", format_sdf_datetime(header.mod_date, header.dialect)),
        ("NumPoints", str(header.num_points)),
        ("NumProfiles", str(header.num_profiles)),
        ("Xscale", _format_scale_field(header.x_scale)),
        ("Yscale", _format_scale_field(header.y_scale)),
        ("Zscale", _format_scale_field(header.z_scale)),
        ("Zresolution", _format_scale_field(header.z_resolution)),
        ("Compression", "0"),  # Never supported, see SdfHeader's docstring.
        ("DataType", str(header.data_type)),
        ("CheckType", "0"),  # Never written, see SdfHeader's docstring.
    )
    for name, value in fields:
        lines.append(f"{name} = {value}")
    lines.append("*")

    write_data = data[::-1] if header.dialect.reverses_profile_order else data
    raw = write_data / header.z_scale
    validate_data_range(raw, np.isnan(write_data), data_type, header.dialect)
    for row in raw:
        lines.append(" ".join(_format_value(value, data_type) for value in row))
    lines.append("*")

    # The standard doesn't unambiguously specify how a missing/empty
    # trailer should be represented on disk; this always terminates the
    # trailer record, even when empty. loads() tells that apart from an
    # actual trailer whose content happens to be empty by record position,
    # not by counting "*" markers.
    if trailer:
        # trailer may already end in its own line terminator (e.g. built via
        # format_tagged_fields(), which CRLF-terminates every field). Strip
        # it so the "\r\n".join() below doesn't add a second one, producing
        # a blank line before the closing "*".
        lines.append(trailer.removesuffix("\r\n").removesuffix("\n"))
    lines.append("*")
    return "\r\n".join(lines) + "\r\n"


def dump(header: SdfHeader, data: np.ndarray, fp: IO[str], trailer: str = "") -> None:
    """Serialize a header/data pair as ASCII SDF text to a file object.

    Equivalent to :func:`dumps`, writing the result to ``fp``.
    """
    fp.write(dumps(header, data, trailer))
