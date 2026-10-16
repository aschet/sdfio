# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF record 1 header."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum, IntEnum, StrEnum, auto

import numpy as np

from .exceptions import SdfFormatError, SdfVersionError

__all__ = [
    "ASCII_PREFIX",
    "BINARY_PREFIX",
    "MAGICS",
    "MAGIC_SIZE",
    "DataType",
    "FileFormat",
    "Header",
    "Metadata",
    "Revision",
    "RevisionRules",
    "format_sdf_datetime",
    "format_tagged_fields",
    "parse_sdf_datetime",
    "parse_tagged_fields",
    "sanitize_tagged_fields",
    "to_sdf_datetime",
    "validate_check_type",
    "validate_compression",
    "validate_manufacturer_id_ascii",
    "validate_trailer_ascii",
    "validate_trailer_tagged",
    "validate_z_scale",
]


#: First character of the file magic: selects the ASCII representation.
ASCII_PREFIX = "a"
#: First character of the file magic: selects the binary representation.
BINARY_PREFIX = "b"


class FileFormat(Enum):
    """ASCII or binary SDF file representation."""

    ASCII = auto()
    BINARY = auto()


class Revision(StrEnum):
    """Revision of the SDF format, as embedded verbatim in the file magic (e.g. ``aISO-2.0``).

    ``BCR`` is the pre-standardization proposal this format is based on;
    ``ISO`` is its standardized successor. BCR never had more than one
    version. Members are named ``<FAMILY>_<VERSION>``; their string value
    is the exact ``<FAMILY>-<VERSION>`` text written into the file magic.
    """

    ISO_1_0 = "ISO-1.0"
    ISO_2_0 = "ISO-2.0"
    BCR_1_0 = "BCR-1.0"

    @classmethod
    def resolve(cls, family_text: str, version_text: str, magic: str) -> Revision:
        """Look up the revision for a magic's separately-parsed family and version text.

        A small shared helper for the ASCII/binary readers, which each parse
        the magic's structure differently but need the same lookup and error
        message once they have the two pieces.

        :param family_text: The magic's 3-letter family abbreviation (e.g. ``"ISO"``).
        :param version_text: The magic's version text (e.g. ``"2.0"``).
        :param magic: The full original magic text, used only for the error message.
        :raises SdfFormatError: If ``family_text``/``version_text`` don't form
            a known, valid revision combination.
        """
        try:
            return cls(f"{family_text}-{version_text}")
        except ValueError:
            raise SdfFormatError(
                f"Unknown or unsupported SDF revision {family_text}-{version_text!r} "
                f"in magic {magic!r}"
            ) from None

    @property
    def rules(self) -> RevisionRules:
        """How this revision differs from the others, see :class:`RevisionRules`."""
        return _RULES[self]


@dataclass(frozen=True)
class RevisionRules:
    """What a revision of the SDF format decides, one instance per :class:`Revision`.

    :param requires_utc: Whether the file stores ``create_date``/``mod_date`` as UTC. The
        standard mandates UTC from version 2.0; the other revisions store local time. In memory
        the dates are UTC for every revision, see :class:`Header`.
    :param max_binary_count: Largest number of points per profile, and of profiles, of the
        binary encoding: 65535 where the binary header holds both in 2 bytes, and 4294967295 for
        4 bytes (ISO 25178-71, Table 2, where the data types of these fields are listed for the
        binary format only). The standard sets no limit for the ASCII encoding.
    :param requires_tagged_trailer: Whether a non-empty trailer written to disk must use the
        tagged ``Name = Value`` format of the header. Enforced on write only: a file with a
        non-conforming trailer can still be read, see :func:`validate_trailer_tagged`.
    :param uses_max_sentinel: Whether the invalid-point sentinel is the maximum of a data type
        instead of its minimum. The 1993 BCR proposal used the maximum, since it still had
        unsigned integer types whose minimum (0) is a valid low bound. ISO reserves the
        minimum, see :meth:`sdfio.DataTypeInfo.invalid_value`.
    :param allows_ascii_comments: Whether an ASCII file may carry ``;`` comments in its header
        and data records: BCR says everything after a ``;`` on a line is ignored. ISO 25178-71
        has no such rule. The trailer is free text, so a ``;`` there is never a comment.
    :param allows_symbolic_header_values: Whether an ASCII header may spell
        ``DataType``/``Compression``/``CheckType`` as names, such as ``DataType = INTEGER``,
        instead of the numeric code. Only BCR does.
    :param has_ascii_invalid_token: Whether an ASCII file marks a non-measured point with the
        literal ``BAD``. BCR has no such token and writes the maximum of the type as a number,
        as binary files do; ``BAD`` is still accepted when reading it.
    :param reverses_profile_order: Whether the first profile stored on disk is at the maximum y,
        not at y=0. BCR never fixed a direction; ISO 25178-71 fixed row 1 at y=0. This reflects
        observed behaviour, not a requirement: MountainsMap renders BCR-1.0 with the first
        profile at the maximum y, and sdfio matches that on read and write, so
        :attr:`SdfFile.y_axis` increases with the row for every revision.

    >>> Revision.ISO_1_0.rules.max_binary_count
    65535
    >>> Revision.ISO_2_0.rules.max_binary_count
    4294967295
    """

    requires_utc: bool
    max_binary_count: int
    requires_tagged_trailer: bool
    uses_max_sentinel: bool
    allows_ascii_comments: bool
    allows_symbolic_header_values: bool
    has_ascii_invalid_token: bool
    reverses_profile_order: bool

    def order_rows(self, data: np.ndarray) -> np.ndarray:
        """Return ``data`` with the rows in the order of the file, or back in increasing y.

        The same operation converts in both directions: a reversal for a revision for which
        :attr:`reverses_profile_order` is ``True``, and ``data`` as it is otherwise.
        """
        return data[::-1] if self.reverses_profile_order else data


_ISO_1_0_RULES = RevisionRules(
    requires_utc=False,
    max_binary_count=2**16 - 1,
    requires_tagged_trailer=False,
    uses_max_sentinel=False,
    allows_ascii_comments=False,
    allows_symbolic_header_values=False,
    has_ascii_invalid_token=True,
    reverses_profile_order=False,
)
_RULES = {
    Revision.ISO_1_0: _ISO_1_0_RULES,
    Revision.ISO_2_0: replace(
        _ISO_1_0_RULES,
        requires_utc=True,
        max_binary_count=2**32 - 1,
        requires_tagged_trailer=True,
    ),
    Revision.BCR_1_0: replace(
        _ISO_1_0_RULES,
        uses_max_sentinel=True,
        allows_ascii_comments=True,
        allows_symbolic_header_values=True,
        has_ascii_invalid_token=False,
        reverses_profile_order=True,
    ),
}


#: Total length in bytes/characters of the file magic (e.g. ``aISO-2.0``):
#: the a/b prefix plus a revision value, which are all the same length.
MAGIC_SIZE = len(ASCII_PREFIX) + len(Revision.ISO_2_0)

#: Every file magic sdfio can read, as bytes, e.g. for magic-sniffing file type detection.
#: BCR-1.0 also allows an upper case ``A``/``B`` prefix, accepted for every revision.
MAGICS = tuple(
    f"{prefix}{revision}".encode("ascii")
    for prefix in (ASCII_PREFIX, BINARY_PREFIX, ASCII_PREFIX.upper(), BINARY_PREFIX.upper())
    for revision in Revision
)


class DataType(IntEnum):
    """SDF data area storage types, by their ``DataType`` header field code.

    Defined here (not in :mod:`sdfio.datatypes`, which owns the heavier
    dtype/encoding logic) so :class:`Header`/:class:`Metadata` can
    reference it directly for their ``data_type`` default -- :mod:`.datatypes`
    itself imports :class:`Revision` from this module, so the reverse
    import would be circular. :mod:`.datatypes` re-exports this name.

    Codes 0-2 are unsigned integer types from the pre-standard BCR-1.0
    proposal; ISO dropped them (see :mod:`.datatypes`'s ``DATA_TYPES`` for
    which revisions accept which code).
    """

    UINT8 = 0
    UINT16 = 1
    UINT32 = 2
    BINARY32 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    BINARY64 = 7


#: Fixed-width ``datetime`` field format, DDMMYYYYHHMM.
_DATETIME_FORMAT = "%d%m%Y%H%M"
# Neither spec defines an "unknown date" convention (unlike Zresolution,
# which explicitly allows a negative value for that) -- this all-zero value
# is undocumented, vendor-specific practice observed in real BCR files
# (day/month 00 isn't a valid calendar date under DDMMYYYYHHMM either way),
# treated here as "not recorded" rather than rejected outright.
_UNSET_DATETIME = "0" * len("DDMMYYYYHHMM")


def to_sdf_datetime(value: datetime | None) -> datetime | None:
    """Return ``value`` as the timezone-aware UTC datetime that sdfio holds.

    ``None`` stays ``None``. The ``datetime`` field of the standard has the resolution of a
    minute, so the seconds and microseconds are dropped. A naive ``value`` is taken as local
    time, like :meth:`datetime.astimezone` does.

    >>> from datetime import timedelta, timezone
    >>> to_sdf_datetime(datetime(2024, 1, 1, 14, 0, 24, tzinfo=timezone(timedelta(hours=2))))
    datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if value is None:
        return None
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def format_sdf_datetime(value: datetime | None, revision: Revision | None = None) -> str:
    """Format a datetime as the fixed-width SDF ``datetime`` field.

    :param value: ``None`` is formatted as the all-zero "not recorded"
        placeholder (see :func:`parse_sdf_datetime`). A naive value is taken
        as local time; the seconds are not written, as the field has the
        resolution of a minute.
    :param revision: If :attr:`RevisionRules.requires_utc`, the field holds the
        UTC wall clock. Otherwise, or if omitted, the revision stores local
        time, so the field holds the wall clock of the system timezone (see
        the limitation noted in :func:`parse_sdf_datetime`).
    """
    utc = to_sdf_datetime(value)
    if utc is None:
        return _UNSET_DATETIME
    if revision is None or not revision.rules.requires_utc:
        utc = utc.astimezone()
    return utc.strftime(_DATETIME_FORMAT)


def parse_sdf_datetime(value: str, revision: Revision | None = None) -> datetime | None:
    """Parse the fixed-width SDF ``datetime`` field; the result is always UTC.

    :param revision: If :attr:`RevisionRules.requires_utc`, the field is UTC.
        Otherwise, or if omitted, the revision stores local time, which is
        taken as the wall clock of the system timezone and converted to UTC.

        This is only correct if the file is read on a system in the same
        timezone it was written in -- neither format records the writer's
        actual timezone, so there is no way to recover it otherwise. A file
        moved to a different timezone before being read ends up with an
        incorrect value; this is a limitation of the format, not something
        sdfio can detect or correct.
    :returns: ``None`` if ``value`` is the all-zero placeholder some files
        use for a date that was never recorded, and also if it is not a date
        in the form ``DDMMYYYYhhmm`` that the standard gives, such as a blank
        field or a day 32. The date is secondary to the heights, so a file with
        such a field is read, like one with a non-compliant trailer, and the
        date is reported as not recorded.
    """
    stripped = value.strip()
    if (
        stripped == _UNSET_DATETIME
        or len(stripped) != len(_UNSET_DATETIME)
        or not stripped.isdigit()
    ):
        return None
    try:
        parsed = datetime.strptime(stripped, _DATETIME_FORMAT)
    except ValueError:  # a number that is no date, such as day 32 or month 13
        return None
    if revision is not None and revision.rules.requires_utc:
        return parsed.replace(tzinfo=UTC)
    return to_sdf_datetime(parsed)


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


def validate_manufacturer_id_ascii(manufacturer_id: str) -> None:
    """Reject a manufacturer_id that is not 7-bit ASCII.

    ManufacID's data type is an ASCII character string per the standard,
    regardless of SDF revision or ASCII/binary file representation. Oversized
    values are silently truncated elsewhere (the binary field is
    fixed-width); this only rejects characters that can't be represented as
    ASCII at all.

    :raises SdfFormatError: If ``manufacturer_id`` contains any non-ASCII
        character.
    """
    if not manufacturer_id.isascii():
        raise SdfFormatError("manufacturer_id must be 7-bit ASCII")


def validate_z_scale(z_scale: float) -> None:
    """Reject a non-positive Z-scale factor.

    The standard specifies the Z-scale factor as "a non-zero positive number
    to scale the coded height information".

    :raises SdfFormatError: If ``z_scale`` is not a positive number.
    """
    if z_scale <= 0:
        raise SdfFormatError(f"z_scale must be a positive number, got {z_scale!r}")


def validate_compression(compression: int) -> None:
    """Reject a compressed data area.

    No compression scheme (e.g. BCR's proposed RLL) is implemented for
    any revision.

    :raises SdfFormatError: If ``compression`` is nonzero.
    """
    if compression != 0:
        raise SdfFormatError("Compressed SDF data areas are not supported")


def validate_check_type(check_type: int) -> None:
    """Reject a checksummed data area.

    No checksum scheme is implemented for any revision. BCR's
    ``IntegerTrace`` checksum is an inline value trailing each profile in
    the data area itself, not a value carried by the header.

    :raises SdfFormatError: If ``check_type`` is nonzero.
    """
    if check_type != 0:
        raise SdfFormatError("Checksummed SDF data areas are not supported")


@dataclass
class Header:
    """SDF record 1 header fields.

    ``Compression`` and ``CheckType`` are not modelled as fields here:
    neither compression nor a checksummed data area is supported, for
    any revision, on read or write (rejected outright, always written as
    ``0``) -- see :func:`validate_compression`/:func:`validate_check_type`.
    ``NumPoints`` and ``NumProfiles`` are not fields either: they are the shape of the data
    (:attr:`SdfFile.data`), which is written with them.

    :param revision: Revision of the SDF format (see :class:`Revision`).
        Accepts a raw ``"ISO-2.0"``-shaped string as a convenience, which is
        coerced to :class:`Revision`.
    :param format: Whether the file uses the binary or the ASCII representation, see
        :class:`FileFormat`.
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Original creation date and time, or ``None`` if not
        recorded in the file (see :func:`parse_sdf_datetime`). It is always
        held as a timezone-aware UTC datetime with the resolution of a minute,
        as the field has, whatever the revision: any timezone-aware value is
        converted when it is set, and a naive one is taken as local time. A
        revision for which :attr:`RevisionRules.requires_utc` is ``False`` writes
        it as local time (see :func:`format_sdf_datetime`).
    :param mod_date: Last modification date and time, or ``None`` if not
        recorded in the file; held like ``create_date``.
    :param x_scale: Sampling interval in metres along x.
    :param y_scale: Sampling interval in metres along y.
    :param z_scale: Scale factor applied to the coded height values.
    :param z_resolution: The measurement instrument's original quantization
        step in metres, or a negative value if unknown. Distinct from
        ``z_scale``: data can be rescaled or reprocessed (e.g. datum removal)
        after acquisition, changing ``z_scale``, while ``z_resolution``
        stays as a record of the instrument's original base resolution.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`.
    :raises SdfVersionError: If ``revision`` is not a valid :class:`Revision`
        member.
    :raises TypeError: If ``format`` is not a :class:`FileFormat` member.
    """

    revision: Revision = Revision.ISO_2_0
    format: FileFormat = FileFormat.BINARY
    manufacturer_id: str = "sdfio"
    create_date: datetime | None = field(default_factory=lambda: datetime.now(UTC))
    mod_date: datetime | None = field(default_factory=lambda: datetime.now(UTC))
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    z_resolution: float = -1.0
    data_type: int = DataType.BINARY64

    def __post_init__(self) -> None:
        """Coerce and validate the SDF revision, and validate the file format."""
        try:
            self.revision = Revision(self.revision)
        except ValueError:
            raise SdfVersionError(
                f"Unsupported SDF revision {self.revision!r}, expected one of {tuple(Revision)}"
            ) from None
        if not isinstance(self.format, FileFormat):
            raise TypeError(f"format must be a FileFormat member, got {self.format!r}")

    def __setattr__(self, name: str, value: object) -> None:
        """Hold ``create_date`` and ``mod_date`` as UTC to the minute, whenever they are set."""
        if name in ("create_date", "mod_date"):
            value = to_sdf_datetime(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

    @property
    def magic(self) -> str:
        """8-character file magic (family and version) written at the start of the file."""
        prefix = BINARY_PREFIX if self.format is FileFormat.BINARY else ASCII_PREFIX
        return f"{prefix}{self.revision}"


@dataclass(frozen=True)
class Metadata:
    """Reusable :func:`sdfio.write` header template.

    Holds exactly the :class:`Header` fields that are meaningful to reuse
    across many :func:`sdfio.write` calls -- as opposed to ``x_scale``/
    ``y_scale``/``z_scale``/``format``, which differ on every call and are
    write()'s own parameters.
    Every field here is used as given, with no overriding.

    :param revision: Revision of the SDF format (see :class:`Revision`).
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Creation timestamp; ``None`` (the default) means the
        current UTC time at the time :func:`sdfio.write` is called. Held like
        :attr:`Header.create_date`: as UTC to the minute.
    :param mod_date: Modification timestamp, held and defaulting like
        ``create_date``.
    :param z_resolution: Quantization step in metres, or a negative value if
        unknown.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`; must be
        valid for ``revision``.
    """

    revision: Revision = Revision.ISO_2_0
    manufacturer_id: str = "sdfio"
    create_date: datetime | None = None
    mod_date: datetime | None = None
    z_resolution: float = -1.0
    data_type: int = DataType.BINARY64

    def __post_init__(self) -> None:
        """Hold ``create_date`` and ``mod_date`` as UTC to the minute."""
        object.__setattr__(self, "create_date", to_sdf_datetime(self.create_date))
        object.__setattr__(self, "mod_date", to_sdf_datetime(self.mod_date))
