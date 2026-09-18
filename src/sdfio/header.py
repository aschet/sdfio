# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF record 1 header."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Final

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

from .exceptions import SdfFormatError, SdfVersionError

__all__ = [
    "ASCII_PREFIX",
    "BINARY_PREFIX",
    "SUPPORTED_VERSIONS",
    "DataType",
    "SdfDialect",
    "SdfHeader",
    "SdfMetadata",
    "SdfVersion",
    "format_sdf_datetime",
    "parse_sdf_datetime",
    "validate_check_type",
    "validate_compression",
    "validate_manufacturer_id_ascii",
    "validate_trailer_ascii",
    "validate_trailer_xml",
    "validate_z_scale",
]


#: First character of the file magic: selects the ASCII representation.
ASCII_PREFIX = "a"
#: First character of the file magic: selects the binary representation.
BINARY_PREFIX = "b"


class SdfVersion(StrEnum):
    """SDF format version, as embedded verbatim in the file magic (e.g. ``aISO-2.0``)."""

    V1_0 = "1.0"
    V2_0 = "2.0"


class SdfDialect(StrEnum):
    """SDF dialect, as embedded verbatim in the file magic (e.g. ``aISO-2.0``).

    ``BCR`` is the pre-standardization proposal this format is based on;
    ``ISO`` is its standardized successor. The two dialects share the same
    version-1.0 record layout, but differ in which versions are valid
    (BCR never had more than one) and in the bad-data sentinel convention
    (see :meth:`sdfio.SdfDataType.invalid_value`).
    """

    ISO = "ISO"
    BCR = "BCR"


class DataType(IntEnum):
    """SDF data area storage types, by their ``DataType`` header field code.

    Defined here (not in :mod:`sdfio.datatypes`, which owns the heavier
    dtype/encoding logic) so :class:`SdfHeader`/:class:`SdfMetadata` can
    reference it directly for their ``data_type`` default -- :mod:`.datatypes`
    itself imports :class:`SdfVersion` from this module, so the reverse
    import would be circular. :mod:`.datatypes` re-exports this name.

    Codes start at 3, not 0: pre-standard/vendor variants of this format
    used codes 0-2 for unsigned integer types the standard doesn't support;
    those codes were dropped rather than reused for something else.
    """

    BINARY32 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    BINARY64 = 7


SUPPORTED_VERSIONS: Final[tuple[SdfVersion, ...]] = tuple(SdfVersion)

#: Fixed-width ``datetime`` field format, DDMMYYYYHHMM.
_DATETIME_FORMAT = "%d%m%Y%H%M"
# Neither spec defines an "unknown date" convention (unlike Zresolution,
# which explicitly allows a negative value for that) -- this all-zero value
# is undocumented, vendor-specific practice observed in real BCR files
# (day/month 00 isn't a valid calendar date under DDMMYYYYHHMM either way),
# treated here as "not recorded" rather than rejected outright.
_UNSET_DATETIME = "0" * len("DDMMYYYYHHMM")


def format_sdf_datetime(value: datetime | None, version: SdfVersion | None = None) -> str:
    """Format a datetime as the fixed-width SDF ``datetime`` field.

    :param value: ``None`` is formatted as the all-zero "not recorded"
        placeholder (see :func:`parse_sdf_datetime`).
    :param version: If :attr:`SdfVersion.V2_0`, ``value`` is assumed to
        already be UTC (as required by :class:`SdfHeader`) and its wall-clock
        components are used as-is. Otherwise, or if omitted, version 1.0 and
        BCR are treated as local time (the standard only specifies UTC for
        version 2.0): a timezone-aware ``value`` is converted to the system
        timezone first (see the limitation noted in
        :func:`parse_sdf_datetime`); a naive ``value`` is assumed to already
        represent local time and is used as-is.
    """
    if value is None:
        return _UNSET_DATETIME
    if version != SdfVersion.V2_0 and value.tzinfo is not None:
        value = value.astimezone()
    return value.strftime(_DATETIME_FORMAT)


def parse_sdf_datetime(value: str, version: SdfVersion | None = None) -> datetime | None:
    """Parse the fixed-width SDF ``datetime`` field.

    :param version: If :attr:`SdfVersion.V2_0`, the result is tagged with UTC
        ``tzinfo`` (version 2.0 timestamps are defined to be UTC). Otherwise,
        or if omitted, version 1.0 and BCR are treated as local time (the
        standard only specifies UTC for version 2.0), and the result is
        tagged with the system's local timezone.

        This is only correct if the file is read on a system in the same
        timezone it was written in -- neither format records the writer's
        actual timezone, so there is no way to recover it otherwise. A file
        moved to a different timezone before being read, or converted to
        version 2.0 there, ends up with an incorrect UTC value; this is a
        limitation of the format, not something sdfio can detect or correct.
    :returns: ``None`` if ``value`` is the all-zero placeholder some files
        use for a date that was never recorded.
    """
    stripped = value.strip()
    if stripped == _UNSET_DATETIME:
        return None
    parsed = datetime.strptime(stripped, _DATETIME_FORMAT)
    return parsed.replace(tzinfo=UTC) if version == SdfVersion.V2_0 else parsed.astimezone()


def validate_trailer_xml(version: SdfVersion, trailer: str | bytes) -> None:
    """Reject a non-empty version 2.0 trailer that isn't well-formed XML.

    An empty ``trailer`` is always accepted (the trailer is optional), and
    version 1.0 has no format requirement for the trailer.

    Parsing is hardened against entity-expansion attacks (``defusedxml``),
    since this runs on every version 2.0 file read/converted and callers
    have no way to substitute their own parser for it.

    :raises SdfFormatError: If ``version`` is :attr:`SdfVersion.V2_0`,
        ``trailer`` is non-empty, and it is not well-formed XML (or uses
        DTDs/entities, which are rejected outright rather than expanded).
    """
    if version != SdfVersion.V2_0 or not trailer:
        return
    try:
        fromstring(trailer)
    except (ParseError, DefusedXmlException) as error:
        raise SdfFormatError(f"SDF version 2.0 trailer must be well-formed XML: {error}") from error


def validate_trailer_ascii(trailer: str | bytes) -> None:
    """Reject a trailer that is not 7-bit ASCII.

    Record 3 (the trailer) is defined as "a sequence of ASCII values",
    regardless of SDF version or ASCII/binary file representation.

    :raises SdfFormatError: If ``trailer`` contains any non-ASCII byte or
        character.
    """
    if not trailer.isascii():
        raise SdfFormatError("SDF trailer must be 7-bit ASCII")


def validate_manufacturer_id_ascii(manufacturer_id: str) -> None:
    """Reject a manufacturer_id that is not 7-bit ASCII.

    ManufacID's data type is an ASCII character string per the standard,
    regardless of SDF version or ASCII/binary file representation. Oversized
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
    either dialect.

    :raises SdfFormatError: If ``compression`` is nonzero.
    """
    if compression != 0:
        raise SdfFormatError("Compressed SDF data areas are not supported")


def validate_check_type(check_type: int) -> None:
    """Reject a checksummed data area.

    No checksum scheme is implemented for either dialect. BCR's
    ``IntegerTrace`` checksum is an inline value trailing each profile in
    the data area itself, not a value carried by the header.

    :raises SdfFormatError: If ``check_type`` is nonzero.
    """
    if check_type != 0:
        raise SdfFormatError("Checksummed SDF data areas are not supported")


@dataclass
class SdfHeader:
    """SDF record 1 header fields.

    ``Compression`` and ``CheckType`` are not modelled as fields here:
    neither compression nor a checksummed data area is supported, for
    either dialect, on read or write (rejected outright, always written as
    ``0``) -- see :func:`validate_compression`/:func:`validate_check_type`.

    :param version: SDF format version. Accepts a raw ``"1.0"``/``"2.0"``
        string as a convenience, which is coerced to :class:`SdfVersion`.
    :param binary: Whether the file uses the binary (``True``) or ASCII
        (``False``) representation.
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Original creation date and time, or ``None`` if not
        recorded in the file (see :func:`parse_sdf_datetime`); version 2.0
        requires UTC when set.
    :param mod_date: Last modification date and time, or ``None`` if not
        recorded in the file; version 2.0 requires UTC when set.
    :param num_points: Number of columns *N* in the data matrix (x-direction).
    :param num_profiles: Number of rows *M* in the data matrix (y-direction).
        The standard uses ``1`` here for a profile (as opposed to an areal
        surface) -- in that case ``y_scale`` is meaningless and should be
        ignored.
    :param x_scale: Sampling interval in metres along x.
    :param y_scale: Sampling interval in metres along y.
    :param z_scale: Scale factor applied to the coded height values.
    :param z_resolution: The measurement instrument's original quantization
        step in metres, or a negative value if unknown. Distinct from
        ``z_scale``: data can be rescaled or reprocessed (e.g. datum removal)
        after acquisition, changing ``z_scale``, while ``z_resolution``
        stays as a record of the instrument's original base resolution.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`.
    :param dialect: SDF dialect (see :class:`SdfDialect`).
    :raises SdfVersionError: If ``version`` is not one of
        :data:`SUPPORTED_VERSIONS`.
    :raises SdfFormatError: If ``version`` is :attr:`SdfVersion.V2_0` and
        ``create_date`` or ``mod_date`` is not a timezone-aware UTC datetime,
        or if ``dialect`` is :attr:`SdfDialect.BCR` and ``version`` is not
        :attr:`SdfVersion.V1_0` (BCR never had another version).
    """

    version: SdfVersion = SdfVersion.V2_0
    dialect: SdfDialect = SdfDialect.ISO
    binary: bool = True
    manufacturer_id: str = "sdfio"
    create_date: datetime | None = field(default_factory=lambda: datetime.now(UTC))
    mod_date: datetime | None = field(default_factory=lambda: datetime.now(UTC))
    num_points: int = 0
    num_profiles: int = 0
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    z_resolution: float = -1.0
    data_type: int = DataType.BINARY64

    def __post_init__(self) -> None:
        """Coerce/validate the SDF version and, for version 2.0, timestamp timezones."""
        try:
            self.version = SdfVersion(self.version)
        except ValueError:
            raise SdfVersionError(
                f"Unsupported SDF version {self.version!r}, expected one of {SUPPORTED_VERSIONS}"
            ) from None
        if self.dialect == SdfDialect.BCR and self.version != SdfVersion.V1_0:
            raise SdfFormatError(
                f"SDF dialect {self.dialect} does not support version {self.version} "
                "(BCR only ever had version 1.0)"
            )
        if self.version == SdfVersion.V2_0:
            for name, value in (("create_date", self.create_date), ("mod_date", self.mod_date)):
                # A None date ("not recorded") has no timezone to validate,
                # and is left as-is rather than rejected.
                if value is None:
                    continue
                # utcoffset() (not "tzinfo is UTC") accepts any tzinfo that's
                # equivalent to UTC, and doubles as the naive-datetime check
                # (utcoffset() is None for those, which also != timedelta(0)).
                if value.utcoffset() != timedelta(0):
                    raise SdfFormatError(
                        f"{name} must be a timezone-aware UTC datetime for SDF version 2.0 "
                        f"(e.g. datetime.now(UTC)), got {value!r}"
                    )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(num_profiles, num_points)`` of the data matrix."""
        return (self.num_profiles, self.num_points)

    @property
    def magic(self) -> str:
        """8-character version identifier written at the start of the file."""
        prefix = BINARY_PREFIX if self.binary else ASCII_PREFIX
        return f"{prefix}{self.dialect}-{self.version}"


@dataclass(frozen=True)
class SdfMetadata:
    """Reusable :func:`sdfio.write` header template.

    Holds exactly the :class:`SdfHeader` fields that are meaningful to reuse
    across many :func:`sdfio.write` calls -- as opposed to ``x_scale``/
    ``y_scale``/``z_scale``/``format``, which differ on every call and are
    write()'s own parameters, and ``num_points``/``num_profiles``/``binary``,
    which write() always derives itself.
    Every field here is used as given, with no overriding.

    :param version: SDF format version.
    :param dialect: SDF dialect (see :class:`SdfDialect`); must be
        :attr:`SdfDialect.ISO` for ``version`` :attr:`SdfVersion.V2_0`, since
        BCR never had a version other than 1.0.
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Creation timestamp; ``None`` (the default) means the
        current UTC time at the time :func:`sdfio.write` is called.
    :param mod_date: Modification timestamp; ``None`` (the default) means the
        current UTC time at the time :func:`sdfio.write` is called.
    :param z_resolution: Quantization step in metres, or a negative value if
        unknown.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`; must be
        valid for ``version`` and ``dialect``.
    """

    version: SdfVersion = SdfVersion.V2_0
    dialect: SdfDialect = SdfDialect.ISO
    manufacturer_id: str = "sdfio"
    create_date: datetime | None = None
    mod_date: datetime | None = None
    z_resolution: float = -1.0
    data_type: int = DataType.BINARY64
