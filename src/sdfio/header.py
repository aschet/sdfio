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
    "SUPPORTED_VERSIONS",
    "DataType",
    "SdfHeader",
    "SdfMetadata",
    "SdfVersion",
    "format_sdf_datetime",
    "parse_sdf_datetime",
    "validate_manufacturer_id_ascii",
    "validate_trailer_ascii",
    "validate_trailer_xml",
    "validate_uncompressed",
    "validate_z_scale",
]


class SdfVersion(StrEnum):
    """SDF format version, as embedded verbatim in the file magic (e.g. ``aISO-2.0``)."""

    V1_0 = "1.0"
    V2_0 = "2.0"


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


def format_sdf_datetime(value: datetime) -> str:
    """Format a datetime as the fixed-width SDF ``datetime`` field.

    Only ``value``'s wall-clock components are used; any ``tzinfo`` is
    ignored rather than converted. The caller is responsible for ensuring
    those components already represent the correct timezone (UTC, for
    version 2.0).
    """
    return value.strftime(_DATETIME_FORMAT)


def parse_sdf_datetime(value: str, version: SdfVersion | None = None) -> datetime:
    """Parse the fixed-width SDF ``datetime`` field.

    :param version: If :attr:`SdfVersion.V2_0`, the result is tagged with UTC
        ``tzinfo`` (version 2.0 timestamps are defined to be UTC); otherwise,
        or if omitted, a naive datetime is returned, since no timezone is
        defined.
    """
    parsed = datetime.strptime(value.strip(), _DATETIME_FORMAT)
    return parsed.replace(tzinfo=UTC) if version == SdfVersion.V2_0 else parsed


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


def validate_uncompressed(compression: int, check_type: int) -> None:
    """Reject a compressed or checksummed data area.

    No compression or checksum is supported for SDF versions 1.0 or 2.0.

    :raises SdfFormatError: If ``compression`` or ``check_type`` is nonzero.
    """
    if compression != 0:
        raise SdfFormatError("Compressed SDF data areas are not supported")
    if check_type != 0:
        raise SdfFormatError("Checksummed SDF data areas are not supported")


@dataclass
class SdfHeader:
    """SDF record 1 header fields.

    :param version: SDF format version. Accepts a raw ``"1.0"``/``"2.0"``
        string as a convenience, which is coerced to :class:`SdfVersion`.
    :param binary: Whether the file uses the binary (``True``) or ASCII
        (``False``) representation.
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Original creation date and time; version 2.0 requires
        UTC.
    :param mod_date: Last modification date and time; version 2.0 requires
        UTC.
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
    :param compression: Compression type; only ``0`` (no compression) is
        supported.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`.
    :param check_type: Checksum type; only ``0`` (no checksum) is supported.
    :raises SdfVersionError: If ``version`` is not one of
        :data:`SUPPORTED_VERSIONS`.
    :raises SdfFormatError: If ``version`` is :attr:`SdfVersion.V2_0` and
        ``create_date`` or ``mod_date`` is not a timezone-aware UTC datetime.
    """

    version: SdfVersion = SdfVersion.V2_0
    binary: bool = True
    manufacturer_id: str = "sdfio"
    create_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    mod_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    num_points: int = 0
    num_profiles: int = 0
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    z_resolution: float = -1.0
    compression: int = 0
    data_type: int = DataType.BINARY64
    check_type: int = 0

    def __post_init__(self) -> None:
        """Coerce/validate the SDF version and, for version 2.0, timestamp timezones."""
        try:
            self.version = SdfVersion(self.version)
        except ValueError:
            raise SdfVersionError(
                f"Unsupported SDF version {self.version!r}, expected one of {SUPPORTED_VERSIONS}"
            ) from None
        if self.version == SdfVersion.V2_0:
            for name, value in (("create_date", self.create_date), ("mod_date", self.mod_date)):
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
        prefix = "b" if self.binary else "a"
        return f"{prefix}ISO-{self.version}"


@dataclass(frozen=True)
class SdfMetadata:
    """Reusable :func:`sdfio.write` header template.

    Holds exactly the :class:`SdfHeader` fields that are meaningful to reuse
    across many :func:`sdfio.write` calls -- as opposed to ``x_scale``/
    ``y_scale``/``z_scale``/``format``, which differ on every call and are
    write()'s own parameters, and ``num_points``/``num_profiles``/``binary``/
    ``compression``/``check_type``, which write() always derives itself.
    Every field here is used as given, with no overriding.

    :param version: SDF format version.
    :param manufacturer_id: Measurement instrument manufacturer's identifier;
        may include the source of the data, hardware and software identifiers.
    :param create_date: Creation timestamp; ``None`` (the default) means the
        current UTC time at the time :func:`sdfio.write` is called.
    :param mod_date: Modification timestamp; ``None`` (the default) means the
        current UTC time at the time :func:`sdfio.write` is called.
    :param z_resolution: Quantization step in metres, or a negative value if
        unknown.
    :param data_type: ``DataType`` code, see :mod:`sdfio.datatypes`; must be
        valid for ``version``.
    """

    version: SdfVersion = SdfVersion.V2_0
    manufacturer_id: str = "sdfio"
    create_date: datetime | None = None
    mod_date: datetime | None = None
    z_resolution: float = -1.0
    data_type: int = DataType.BINARY64
