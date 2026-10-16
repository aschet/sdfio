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

    This is the tagged field format the SDF header uses, and that a revision
    for which :attr:`RevisionRules.requires_tagged_trailer` is ``True`` also
    requires for the trailer on write (see :func:`validate_trailer_tagged`).
    Blank lines are skipped; field names are kept exactly as written
    (callers needing a case-insensitive lookup should match keys
    case-insensitively themselves).

    :raises SdfFormatError: If a non-blank line doesn't match "Name = Value".
    """
    return _iter_tagged_fields(text, strict=True)


def format_tagged_fields(fields: Mapping[str, str]) -> str:
    """Format a ``{name: value}`` mapping as ``<CRLF>``-terminated "Name = Value" lines."""
    return "".join(f"{name} = {value}\r\n" for name, value in fields.items())


def sanitize_tagged_fields(text: str) -> str:
    """Reformat ``text``, keeping only lines that already match "Name = Value".

    Unlike :func:`parse_tagged_fields`, this never raises: a non-blank line
    that doesn't match "Name = Value" is silently dropped instead of
    rejected. Useful for salvaging a trailer that's mostly, but not fully,
    in the tagged format (e.g. converting a source file to a revision that
    requires it) rather than discarding the whole thing.
    """
    return format_tagged_fields(_iter_tagged_fields(text, strict=False))


def validate_trailer_tagged(revision: Revision, trailer: str | bytes) -> None:
    """Reject a non-empty, non-tagged trailer for a revision that requires the tagged format.

    Enforced on write only (see :func:`validate_trailer_ascii` for the same
    read/write split). An empty ``trailer`` is always accepted (the trailer
    is optional), and a revision for which
    :attr:`RevisionRules.requires_tagged_trailer` is ``False`` has no format
    requirement for the trailer.

    :raises SdfFormatError: If ``revision.rules.requires_tagged_trailer``, ``trailer``
        is non-empty, and it is not valid "Name = Value" lines (see
        :func:`parse_tagged_fields`).
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

    Record 3 (the trailer) is defined as "a sequence of ASCII values",
    regardless of SDF revision or ASCII/binary file representation. Only
    enforced on write -- a file that already has a non-ASCII trailer can
    still be read; the trailer is secondary to the header/data area, which
    is unaffected by it.

    :raises SdfFormatError: If ``trailer`` contains any non-ASCII byte or
        character.
    """
    if not trailer.isascii():
        raise SdfFormatError("SDF trailer must be 7-bit ASCII")
