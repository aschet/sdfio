# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""ASCII (text) representation of the SDF format."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import IO

import numpy as np
from numpy.typing import NDArray

from ._datatypes import DataTypeInfo, require_supported_data_type, validate_data_range
from ._dates import format_sdf_datetime, parse_sdf_datetime
from ._numeric import format_scientific
from ._validation import validate_check_type, validate_compression, validate_for_write
from .exceptions import SdfFormatError
from .header import Header
from .revision import ASCII_PREFIX, BINARY_PREFIX, DataType, FileFormat, Revision
from .trailer import parse_tagged_fields

__all__ = ["dump", "dumps", "load", "loads"]

_MAGIC_RE = re.compile(
    rf"^(?P<prefix>[{ASCII_PREFIX}{BINARY_PREFIX}{ASCII_PREFIX.upper()}{BINARY_PREFIX.upper()}])"
    rf"(?P<family>[A-Z]{{3}})-(?P<version>\d\.\d)$"
)
#: The literal that marks a point that is not measured, which is what is written.
_INVALID_MARKER = "BAD"
#: What is read as such a point: the literal, and the text ``NaN``, which some writers write for
#: it instead.
_INVALID_MARKERS = (_INVALID_MARKER, "NAN")
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


def _split_records(remainder: str, revision: Revision) -> list[str]:
    # Lines are split first and each is tested for being a bare "*", because splitting on the
    # delimiter with its newlines cannot tell an empty record ("*<CRLF>*<CRLF>") from one
    # delimiter. With comments, a "*" followed by a comment still ends its record, and the
    # comment is dropped from the first two records; the trailer is free text and kept as it is.
    comments = revision.rules.ascii_files.comments
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
    fields: Mapping[str, str], name: str, revision: Revision, symbols: Mapping[str, int]
) -> int:
    value = _field(fields, name)
    try:
        return int(value)
    except ValueError:
        if revision.rules.ascii_files.symbolic_header_values and value.upper() in symbols:
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


def loads(text: str) -> tuple[Header, NDArray[np.float64], str]:
    """Parse the text of an ASCII SDF file.

    :returns: The header, the data and the trailer, which is not validated on read.
    :raises SdfFormatError: If the file is malformed, such as a bad magic, an unsupported
        revision or data type, or a missing record.
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
    if magic_match.group("prefix").lower() != ASCII_PREFIX:
        raise SdfFormatError("Binary magic found while parsing an ASCII SDF file")
    family_text = magic_match.group("family")
    version_text = magic_match.group("version")
    revision = Revision.resolve(family_text, version_text, magic_line)
    if _COMMENT_MARKER in magic_line and not revision.rules.ascii_files.comments:
        raise SdfFormatError(f"Not an ASCII SDF file, unexpected magic: {magic_line!r}")

    records = _split_records(remainder, revision)
    if len(records) < 3:
        raise SdfFormatError("SDF file must contain header, data and trailer records")
    # The trailer is itself terminated by its own "*" record; drop
    # a resulting trailing empty segment instead of re-joining it back in.
    header_text, data_text, *trailer_parts = records
    if trailer_parts and trailer_parts[-1] == "":
        trailer_parts = trailer_parts[:-1]
    trailer_text = "*".join(trailer_parts)

    fields = parse_tagged_fields(header_text)
    data_type_code = _parse_coded(fields, "DataType", revision, _BCR_DATA_TYPE_NAMES)
    data_type = require_supported_data_type(data_type_code, revision)
    validate_compression(_parse_coded(fields, "Compression", revision, _BCR_COMPRESSION_NAMES))
    validate_check_type(_parse_coded(fields, "CheckType", revision, _BCR_CHECK_TYPE_NAMES))

    header = Header._read(
        revision=revision,
        format=FileFormat.ASCII,
        manufacturer_id=_field(fields, "ManufacID").strip(),
        create_date=parse_sdf_datetime(_field(fields, "CreateDate"), revision),
        mod_date=parse_sdf_datetime(_field(fields, "ModDate"), revision),
        x_scale=_parse_float(fields, "Xscale"),
        y_scale=_parse_float(fields, "Yscale"),
        z_scale=_parse_float(fields, "Zscale"),
        z_resolution=_parse_float(fields, "Zresolution"),
        data_type=data_type.type,
    )

    shape = (_parse_int(fields, "NumProfiles"), _parse_int(fields, "NumPoints"))
    data = _parse_data(data_text, header, shape, data_type)
    # Not validated on read: a bad trailer must not prevent reading the heights.
    trailer = trailer_text.strip()
    return header, data, trailer


def _parse_data(
    data_text: str, header: Header, shape: tuple[int, int], data_type: DataTypeInfo
) -> NDArray[np.float64]:
    tokens = data_text.split()
    expected = shape[0] * shape[1]
    if len(tokens) != expected:
        raise SdfFormatError(f"Expected {expected} data values, found {len(tokens)}")

    is_float = data_type.type in (DataType.BINARY32, DataType.BINARY64)
    values = np.empty(expected, dtype=np.float64)
    for index, token in enumerate(tokens):
        # The standard only specifies the literal "BAD"; matching it case-insensitively, and
        # reading "NaN" as the same marker, are deliberate leniencies, not mandated. "NaN" is
        # not left to float(), so that it means the same for every data type.
        if token.upper() in _INVALID_MARKERS:
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
    if not header.revision.rules.ascii_files.invalid_token:
        # BCR marks a bad point with the type's maximum value, as a number.
        invalid_mask |= data_type.is_invalid(values, header.revision)
    # NaN before scaling: a huge marker value could overflow when scaled.
    values[invalid_mask] = np.nan
    with np.errstate(invalid="ignore", over="ignore"):  # see decode_raw
        data = (values * header.z_scale).reshape(shape)
    return header.revision.rules.order_rows(data)


def load(fp: IO[str]) -> tuple[Header, NDArray[np.float64], str]:
    """Parse an ASCII SDF file object opened in text mode.

    Equivalent to :func:`loads`, reading the text from ``fp`` first.
    """
    return loads(fp.read())


def _format_value(value: float, data_type: DataTypeInfo, revision: Revision) -> str:
    if np.isnan(value):
        if revision.rules.ascii_files.invalid_token:
            return _INVALID_MARKER
        value = data_type.ascii_invalid_value(revision)
    if data_type.type == DataType.BINARY32:
        return format_scientific(value, 6, 2)
    if data_type.type == DataType.BINARY64:
        return format_scientific(value, 14, 3)
    return str(round(value))


def dumps(header: Header, data: NDArray[np.float64], trailer: str = "") -> str:
    """Serialize a header/data pair to the ASCII SDF text representation.

    :param header: Header describing ``data``; ``header.format`` is ignored.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param trailer: Optional record 3 trailer content.

    For :attr:`~sdfio.DataType.BINARY64`, this is lossy at the ~1e-15
    relative level: the standard's mandated 15 significant digits are one
    short of the 17 an IEEE 754 double needs to round-trip exactly (see
    :func:`sdfio._numeric.format_scientific`). Use the binary format to
    avoid this.
    """
    data_type = validate_for_write(header, trailer)

    lines = [f"{ASCII_PREFIX}{header.revision}"]
    fields = (
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", format_sdf_datetime(header.create_date, header.revision)),
        ("ModDate", format_sdf_datetime(header.mod_date, header.revision)),
        ("NumPoints", str(data.shape[1])),
        ("NumProfiles", str(data.shape[0])),
        ("Xscale", _format_scale_field(header.x_scale)),
        ("Yscale", _format_scale_field(header.y_scale)),
        ("Zscale", _format_scale_field(header.z_scale)),
        ("Zresolution", _format_scale_field(header.z_resolution)),
        ("Compression", "0"),  # Never supported, see Header's docstring.
        ("DataType", str(header.data_type)),
        ("CheckType", "0"),  # Never written, see Header's docstring.
    )
    for name, value in fields:
        lines.append(f"{name} = {value}")
    lines.append("*")

    write_data = header.revision.rules.order_rows(data)
    raw = write_data / header.z_scale
    validate_data_range(raw, np.isnan(write_data), data_type, header.revision)
    for row in raw:
        lines.append(" ".join(_format_value(value, data_type, header.revision) for value in row))
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


def dump(header: Header, data: NDArray[np.float64], fp: IO[str], trailer: str = "") -> None:
    """Serialize a header/data pair as ASCII SDF text to a file object.

    Equivalent to :func:`dumps`, writing the result to ``fp``.
    """
    fp.write(dumps(header, data, trailer))
