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

__all__ = ["dump", "dumps", "load", "loads"]

_MAGIC_RE = re.compile(
    rf"^(?P<prefix>[{ASCII_PREFIX}{BINARY_PREFIX}])(?P<dialect>[A-Z]{{3}})-(?P<version>\d\.\d)$"
)
_FIELD_RE = re.compile(r"^(?P<name>\w+)\s*=\s*(?P<value>.*)$")
_INVALID_MARKER = "BAD"
# The standard mandates <CRLF> line endings; dumps() always writes them.
# \r? here is a read-side leniency to also accept bare LF.
_LINE_SPLIT_RE = re.compile(r"\r?\n")
_TERMINATOR_RE = re.compile(r"^[ \t]*\*[ \t]*$")


def _split_records(remainder: str) -> list[str]:
    # Splitting by matching the "*" delimiter together with its surrounding
    # newlines (as one combined pattern) can't tell two delimiters directly
    # adjacent to each other (an explicit empty record, e.g. an empty
    # trailer written as "*<CRLF>*<CRLF>") apart from a single delimiter --
    # the newline between them would need to be consumed by both matches at
    # once. Splitting into lines first and testing each one for being a bare
    # "*" avoids that ambiguity.
    records = []
    current: list[str] = []
    for line in _LINE_SPLIT_RE.split(remainder):
        if _TERMINATOR_RE.match(line):
            records.append("\n".join(current))
            current = []
        else:
            current.append(line)
    records.append("\n".join(current))
    return records


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
    if magic_match.group("prefix") != ASCII_PREFIX:
        raise SdfFormatError("Binary magic found while parsing an ASCII SDF file")
    dialect_text = magic_match.group("dialect")
    try:
        dialect = SdfDialect(dialect_text)
    except ValueError:
        raise SdfFormatError(
            f"Unknown SDF dialect {dialect_text!r} in magic {magic_line!r}"
        ) from None
    version_text = magic_match.group("version")
    try:
        version = SdfVersion(version_text)
    except ValueError:
        raise SdfFormatError(f"Unsupported SDF version {version_text!r}") from None

    records = _split_records(remainder)
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
    data_type = require_supported_data_type(data_type_code, version, dialect)
    validate_compression(_parse_int(fields, "Compression"))
    validate_check_type(_parse_int(fields, "CheckType"))

    header = SdfHeader(
        version=version,
        dialect=dialect,
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
        data_type=data_type_code,
    )

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
    validate_trailer_ascii(trailer)
    validate_trailer_xml(header.version, trailer)
    data_type = require_supported_data_type(header.data_type, header.version, header.dialect)

    lines = [f"{ASCII_PREFIX}{header.dialect}-{header.version}"]
    fields = (
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", format_sdf_datetime(header.create_date, header.version)),
        ("ModDate", format_sdf_datetime(header.mod_date, header.version)),
        ("NumPoints", str(header.num_points)),
        ("NumProfiles", str(header.num_profiles)),
        ("Xscale", format_scientific(header.x_scale, 14, 3)),
        ("Yscale", format_scientific(header.y_scale, 14, 3)),
        ("Zscale", format_scientific(header.z_scale, 14, 3)),
        ("Zresolution", format_scientific(header.z_resolution, 14, 3)),
        ("Compression", "0"),  # Never supported, see SdfHeader's docstring.
        ("DataType", str(header.data_type)),
        ("CheckType", "0"),  # Never written, see SdfHeader's docstring.
    )
    for name, value in fields:
        lines.append(f"{name} = {value}")
    lines.append("*")

    raw = data / header.z_scale
    validate_data_range(raw, np.isnan(data), data_type, header.dialect)
    for row in raw:
        lines.append(" ".join(_format_value(value, data_type) for value in row))
    lines.append("*")

    # The standard doesn't unambiguously specify how a missing/empty
    # trailer should be represented on disk; this always terminates the
    # trailer record, even when empty. loads() tells that apart from an
    # actual trailer whose content happens to be empty by record position,
    # not by counting "*" markers.
    if trailer:
        lines.append(trailer)
    lines.append("*")
    return "\r\n".join(lines) + "\r\n"


def dump(header: SdfHeader, data: np.ndarray, fp: IO[str], trailer: str = "") -> None:
    """Serialize a header/data pair as ASCII SDF text to a file object.

    Equivalent to :func:`dumps`, writing the result to ``fp``.
    """
    fp.write(dumps(header, data, trailer))
