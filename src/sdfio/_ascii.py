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
    SdfHeader,
    SdfVersion,
    format_sdf_datetime,
    parse_sdf_datetime,
    validate_manufacturer_id_ascii,
    validate_trailer_ascii,
    validate_trailer_xml,
    validate_uncompressed,
    validate_z_scale,
)

__all__ = ["dump", "dumps", "load", "loads"]

_MAGIC_RE = re.compile(r"^[ab]ISO-(?P<version>\d\.\d)$")
_FIELD_RE = re.compile(r"^(?P<name>\w+)\s*=\s*(?P<value>.*)$")
_INVALID_MARKER = "BAD"
# The standard mandates <CRLF> line endings; dumps() always writes them.
# \r? here is a read-side leniency to also accept bare LF.
_RECORD_SPLIT_RE = re.compile(r"[ \t]*\r?\n[ \t]*\*[ \t]*(?:\r?\n|$)")


def _read_fields(header_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _FIELD_RE.match(line)
        if not match:
            raise SdfFormatError(f"Malformed SDF header line: {line!r}")
        fields[match.group("name")] = match.group("value").strip()
    return fields


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
        unsupported version or data type, missing records, or a trailer
        that isn't 7-bit ASCII.
    """
    text = text.lstrip("\ufeff")
    first_newline = re.search(r"\r?\n", text)
    if first_newline is None:
        raise SdfFormatError("SDF file is missing the header and data records")
    magic_line = text[: first_newline.start()].strip()
    remainder = text[first_newline.end() :]

    magic_match = _MAGIC_RE.match(magic_line)
    if not magic_match:
        raise SdfFormatError(f"Not an ASCII SDF file, unexpected magic: {magic_line!r}")
    if magic_line[0] != "a":
        raise SdfFormatError("Binary magic found while parsing an ASCII SDF file")
    version_text = magic_match.group("version")
    try:
        version = SdfVersion(version_text)
    except ValueError:
        raise SdfFormatError(f"Unsupported SDF version {version_text!r}") from None

    records = _RECORD_SPLIT_RE.split(remainder)
    if len(records) < 3:
        raise SdfFormatError("SDF file must contain header, data and trailer records")
    # The trailer is itself terminated by its own "*" record; drop
    # a resulting trailing empty segment instead of re-joining it back in.
    header_text, data_text, *trailer_parts = records
    if trailer_parts and trailer_parts[-1] == "":
        trailer_parts = trailer_parts[:-1]
    trailer_text = "*".join(trailer_parts)

    fields = _read_fields(header_text)
    data_type_code = _parse_int(fields, "DataType")
    data_type = require_supported_data_type(data_type_code, version)

    header = SdfHeader(
        version=version,
        binary=False,
        manufacturer_id=_field(fields, "ManufacID").strip(),
        create_date=parse_sdf_datetime(_field(fields, "CreateDate"), version),
        mod_date=parse_sdf_datetime(_field(fields, "ModDate"), version),
        num_points=_parse_int(fields, "NumPoints"),
        num_profiles=_parse_int(fields, "NumProfiles"),
        x_scale=_parse_float(fields, "Xscale"),
        y_scale=_parse_float(fields, "Yscale"),
        z_scale=_parse_float(fields, "Zscale"),
        z_resolution=_parse_float(fields, "Zresolution"),
        compression=_parse_int(fields, "Compression"),
        data_type=data_type_code,
        check_type=_parse_int(fields, "CheckType"),
    )
    validate_uncompressed(header.compression, header.check_type)

    data = _parse_data(data_text, header, data_type)
    trailer = trailer_text.strip()
    validate_trailer_ascii(trailer)
    return header, data, trailer


def _parse_data(data_text: str, header: SdfHeader, data_type: SdfDataType) -> np.ndarray:
    tokens = data_text.split()
    expected = header.num_points * header.num_profiles
    if len(tokens) != expected:
        raise SdfFormatError(f"Expected {expected} data values, found {len(tokens)}")

    is_float = data_type.type in (DataType.BINARY32, DataType.BINARY64)
    values = np.empty(expected, dtype=np.float64)
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
    return scaled.reshape(header.num_profiles, header.num_points)


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
    validate_uncompressed(header.compression, header.check_type)
    validate_trailer_ascii(trailer)
    validate_trailer_xml(header.version, trailer)
    data_type = require_supported_data_type(header.data_type, header.version)

    lines = [f"aISO-{header.version}"]
    fields = (
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", format_sdf_datetime(header.create_date)),
        ("ModDate", format_sdf_datetime(header.mod_date)),
        ("NumPoints", str(header.num_points)),
        ("NumProfiles", str(header.num_profiles)),
        ("Xscale", format_scientific(header.x_scale, 14, 3)),
        ("Yscale", format_scientific(header.y_scale, 14, 3)),
        ("Zscale", format_scientific(header.z_scale, 14, 3)),
        ("Zresolution", format_scientific(header.z_resolution, 14, 3)),
        ("Compression", str(header.compression)),
        ("DataType", str(header.data_type)),
        ("CheckType", str(header.check_type)),
    )
    for name, value in fields:
        lines.append(f"{name} = {value}")
    lines.append("*")

    raw = data / header.z_scale
    validate_data_range(raw, np.isnan(data), data_type)
    for row in raw:
        lines.append(" ".join(_format_value(value, data_type) for value in row))
    lines.append("*")

    # An empty trailer is omitted entirely, rather than emitting a second "*"
    # terminator immediately after the data's, which loads() cannot tell
    # apart from an actual trailer whose content happens to be empty.
    if trailer:
        lines.append(trailer)
        lines.append("*")
    return "\r\n".join(lines) + "\r\n"


def dump(header: SdfHeader, data: np.ndarray, fp: IO[str], trailer: str = "") -> None:
    """Serialize a header/data pair as ASCII SDF text to a file object.

    Equivalent to :func:`dumps`, writing the result to ``fp``.
    """
    fp.write(dumps(header, data, trailer))
