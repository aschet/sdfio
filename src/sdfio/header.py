# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF record 1 header."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .datatypes import DataType, DataTypeInfo, get_data_type, require_supported_data_type
from .exceptions import SdfFormatError, SdfVersionError
from .revision import ASCII_PREFIX, BINARY_PREFIX, FileFormat, Revision
from .trailer import validate_trailer_ascii, validate_trailer_tagged

__all__ = [
    "Header",
    "Metadata",
    "format_sdf_datetime",
    "parse_sdf_datetime",
    "to_sdf_datetime",
    "validate_check_type",
    "validate_compression",
    "validate_manufacturer_id_ascii",
    "validate_z_scale",
]


#: Fixed-width ``datetime`` field format, DDMMYYYYHHMM.
_DATETIME_FORMAT = "%d%m%Y%H%M"
# Neither revision defines a date that is unknown. Files in practice, BCR ones among them, use all
# zeros, which is no calendar date; it is read as "not recorded".
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
    """Format a datetime as the ``datetime`` field of the file.

    :param value: ``None`` becomes the all-zero "not recorded" placeholder. A naive value is
        local time. The seconds are dropped, as the field has the resolution of a minute.
    :param revision: A revision with :attr:`RevisionRules.requires_utc` gets the UTC clock.
        Another one, or none, gets the clock of the system timezone, see
        :func:`parse_sdf_datetime`.
    """
    utc = to_sdf_datetime(value)
    if utc is None:
        return _UNSET_DATETIME
    if revision is None or not revision.rules.requires_utc:
        utc = utc.astimezone()
    return utc.strftime(_DATETIME_FORMAT)


def parse_sdf_datetime(value: str, revision: Revision | None = None) -> datetime | None:
    """Parse the ``datetime`` field of a file; the result is UTC.

    :param revision: A revision with :attr:`RevisionRules.requires_utc` holds UTC. Another one,
        or none, holds the clock of the system timezone, which is converted. That is only right
        if the file is read in the timezone it was written in, because neither format records
        it. This is a limit of the format.
    :returns: ``None`` for the all-zero placeholder, and for a field that is no date of the form
        ``DDMMYYYYhhmm``, such as a blank one or a day 32. The date matters less than the
        heights, so such a file is read and the date is reported as not recorded.
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


def validate_manufacturer_id_ascii(manufacturer_id: str) -> None:
    """Reject a manufacturer ID that is not 7-bit ASCII, as the standard has it.

    :raises SdfFormatError: If ``manufacturer_id`` has a non-ASCII character.
    """
    if not manufacturer_id.isascii():
        raise SdfFormatError("manufacturer_id must be 7-bit ASCII")


def validate_z_scale(z_scale: float) -> None:
    """Reject a Z-scale that is not positive.

    The standard calls it "a non-zero positive number to scale the coded height information".

    :raises SdfFormatError: If ``z_scale`` is not positive.
    """
    if z_scale <= 0:
        raise SdfFormatError(f"z_scale must be a positive number, got {z_scale!r}")


def validate_compression(compression: int) -> None:
    """Reject a compressed data area, which no revision is read or written with here.

    :raises SdfFormatError: If ``compression`` is not 0.
    """
    if compression != 0:
        raise SdfFormatError("Compressed SDF data areas are not supported")


def validate_check_type(check_type: int) -> None:
    """Reject a checksummed data area, which no revision is read or written with here.

    The ``IntegerTrace`` checksum of BCR is a value after each profile in the data area, not a
    value of the header.

    :raises SdfFormatError: If ``check_type`` is not 0.
    """
    if check_type != 0:
        raise SdfFormatError("Checksummed SDF data areas are not supported")


def validate_for_write(header: Header, trailer: str | bytes) -> DataTypeInfo:
    """Reject what the file of ``header`` and ``trailer`` could not hold, for either encoding.

    :returns: The data type of the header, which is supported by its revision.
    :raises SdfFormatError: If the Z-scale is not positive, the manufacturer ID or the trailer is
        not 7-bit ASCII, the trailer is not tagged where the revision requires it, or the data
        type is unknown or not supported by the revision.
    """
    validate_z_scale(header.z_scale)
    validate_manufacturer_id_ascii(header.manufacturer_id)
    validate_trailer_ascii(trailer)
    validate_trailer_tagged(header.revision, trailer)
    return require_supported_data_type(header.data_type, header.revision)


def _as_revision(value: str) -> Revision:
    try:
        return Revision(value)
    except ValueError:
        raise SdfVersionError(
            f"Unsupported SDF revision {value!r}, expected one of {tuple(Revision)}"
        ) from None


@dataclass
class Header:
    """SDF record 1 header fields.

    ``Compression`` and ``CheckType`` are not modelled as fields here:
    neither compression nor a checksummed data area is supported, for
    any revision, on read or write (rejected outright, always written as
    ``0``) -- see :func:`validate_compression`/:func:`validate_check_type`.
    ``NumPoints`` and ``NumProfiles`` are not fields either: they are the shape of the data
    (:attr:`SdfFile.data`), which is written with them.

    A header is checked when it is made and when a field is set: the revision, the file format,
    the data type, which the revision must have, a positive ``z_scale`` and a 7-bit ASCII
    ``manufacturer_id``. To change the revision together with a data type that the old revision
    does not have, make a new header, see :meth:`SdfFile.with_revision`.
    A header that a reader returns is not, because the readers accept files with such values, so
    that they can be inspected and fixed; the write rejects them.

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
    data_type: DataType = DataType.BINARY64

    @classmethod
    def _read(cls, **fields: object) -> Header:
        """Return a header of what a file holds, which the readers accept as it is.

        The readers tolerate what the standard forbids and a writer would reject, such as a
        Z-scale that is not positive or a manufacturer ID that is not 7-bit ASCII, so that such a
        file can be inspected and fixed. Nothing is checked here; :func:`validate_for_write` does
        before a file is written, and every later assignment is checked as for any header.
        """
        header = cls.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(header, name, value)
        return header

    def __setattr__(self, name: str, value: object) -> None:
        """Check a field when it is set, and hold the dates as UTC to the minute."""
        if name == "revision":
            value = _as_revision(value)  # type: ignore[arg-type]
            if "data_type" in self.__dict__:
                require_supported_data_type(self.data_type, value)
        elif name == "format":
            if not isinstance(value, FileFormat):
                raise TypeError(f"format must be a FileFormat member, got {value!r}")
        elif name == "data_type":
            value = get_data_type(value).type  # type: ignore[arg-type]
            if "revision" in self.__dict__:
                require_supported_data_type(value, self.revision)
        elif name == "z_scale":
            validate_z_scale(value)  # type: ignore[arg-type]
        elif name == "manufacturer_id":
            validate_manufacturer_id_ascii(value)  # type: ignore[arg-type]
        elif name in ("create_date", "mod_date"):
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
    data_type: DataType = DataType.BINARY64

    def __post_init__(self) -> None:
        """Hold the dates as UTC to the minute, and check the data type against the revision."""
        object.__setattr__(
            self, "data_type", require_supported_data_type(self.data_type, self.revision).type
        )
        object.__setattr__(self, "create_date", to_sdf_datetime(self.create_date))
        object.__setattr__(self, "mod_date", to_sdf_datetime(self.mod_date))
