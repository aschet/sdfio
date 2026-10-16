# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The trailer, the third record of an SDF file, and the tagged fields it may hold."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .exceptions import SdfFormatError
from .revision import Revision

__all__ = [
    "format_tagged_fields",
    "parse_tagged_fields",
    "sanitize_tagged_fields",
    "validate_trailer_ascii",
    "validate_trailer_tagged",
]


#: Matches one "Name = Value" line of the tagged field format the header uses.
#: Field names are not case-sensitive; the value is everything after "=",
#: trimmed by the caller.
_TAGGED_FIELD_RE = re.compile(r"^(?P<name>\w+)\s*=\s*(?P<value>.*)$")


def _iter_tagged_fields(text: str, *, strict: bool) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TAGGED_FIELD_RE.match(line)
        if not match:
            if strict:
                raise SdfFormatError(f"Malformed SDF tagged field line: {line!r}")
            continue
        fields[match.group("name")] = match.group("value").strip()
    return fields


def parse_tagged_fields(text: str) -> dict[str, str]:
    """Parse "Name = Value" lines into a ``{name: value}`` dict.

    That is the form of the header, and the form that ISO-2.0 requires of the trailer on write.
    Blank lines are skipped. Names are kept as written; compare them case-insensitively if
    needed.

    :raises SdfFormatError: If a line is not "Name = Value".
    """
    return _iter_tagged_fields(text, strict=True)


def format_tagged_fields(fields: Mapping[str, str]) -> str:
    """Format a ``{name: value}`` mapping as ``<CRLF>``-terminated "Name = Value" lines."""
    return "".join(f"{name} = {value}\r\n" for name, value in fields.items())


def sanitize_tagged_fields(text: str) -> str:
    """Reformat ``text``, keeping the lines that are "Name = Value" and leaving out the rest.

    Unlike :func:`parse_tagged_fields` this never raises, which salvages a trailer that is mostly
    tagged, such as when converting a file to ISO-2.0.
    """
    return format_tagged_fields(_iter_tagged_fields(text, strict=False))


def validate_trailer_tagged(revision: Revision, trailer: str | bytes) -> None:
    """Reject a trailer that is not tagged, for a revision that requires it.

    Only a write is checked; a file with another trailer is read. An empty trailer is always
    accepted.

    :raises SdfFormatError: If :attr:`RevisionRules.requires_tagged_trailer` is ``True`` and the
        trailer is not "Name = Value" lines, see :func:`parse_tagged_fields`.
    """
    if not revision.rules.requires_tagged_trailer or not trailer:
        return
    text = trailer.decode("ascii", errors="replace") if isinstance(trailer, bytes) else trailer
    try:
        parse_tagged_fields(text)
    except SdfFormatError as error:
        raise SdfFormatError(
            f"SDF {revision} trailer must use the tagged 'Name = Value' format: {error}"
        ) from error


def validate_trailer_ascii(trailer: str | bytes) -> None:
    """Reject a trailer that is not 7-bit ASCII.

    The standard defines record 3 as "a sequence of ASCII values" for every revision and
    encoding. Only a write is checked; a file with another trailer is read.

    :raises SdfFormatError: If ``trailer`` has a non-ASCII byte or character.
    """
    if not trailer.isascii():
        raise SdfFormatError("SDF trailer must be 7-bit ASCII")
