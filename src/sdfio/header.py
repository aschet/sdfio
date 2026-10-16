# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF record 1 header."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ._dates import to_sdf_datetime
from ._validation import validate_manufacturer_id_ascii, validate_z_scale
from .datatypes import DataType, get_data_type, require_supported_data_type
from .exceptions import SdfVersionError
from .revision import ASCII_PREFIX, BINARY_PREFIX, FileFormat, Revision

__all__ = ["Header", "Metadata"]


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
    ``0``).
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
        recorded in the file. It is always
        held as a timezone-aware UTC datetime with the resolution of a minute,
        as the field has, whatever the revision: any timezone-aware value is
        converted when it is set, and a naive one is taken as local time. A
        revision for which :attr:`RevisionRules.requires_utc` is ``False`` writes
        it as local time.
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
        file can be inspected and fixed. Nothing is checked here; the write does,
        and every later assignment is checked as for any header.
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
