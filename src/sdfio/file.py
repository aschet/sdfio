# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""High-level, NumPy-oriented interface to SDF surface data files."""

from __future__ import annotations

import os

# Only Element/tostring are used from here, not parsing (see trailer_xml).
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import IO, Literal

import numpy as np
from defusedxml.ElementTree import fromstring

from . import _ascii, _binary
from .datatypes import DataType, encode_raw, get_data_type, suggest_z_scale
from .exceptions import SdfFormatError
from .header import SdfHeader, SdfMetadata, SdfVersion, validate_trailer_xml

__all__ = ["FileFormat", "SdfFile", "read", "write"]

PathLike = str | os.PathLike[str]
PathOrStream = PathLike | IO[bytes]


class FileFormat(Enum):
    """ASCII or binary SDF file representation."""

    ASCII = auto()
    BINARY = auto()


# eq=False: dataclass's auto __eq__ would do `data == other.data`, which
# raises on a numpy array's ambiguous truth value; __eq__ below replaces it
# with a NaN-aware, array-safe comparison instead.
@dataclass(eq=False)
class SdfFile:
    """In-memory representation of an SDF surface data file.

    :param header: Parsed record 1 header fields.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres, with ``NaN`` marking non-measured or spurious points.
    :param trailer: Raw record 3 trailer content; text for ASCII files,
        bytes for binary files.
    """

    header: SdfHeader
    data: np.ndarray
    trailer: str | bytes = ""

    def __eq__(self, other: object) -> bool:
        """Compare header, data (NaN-aware) and trailer for equality."""
        if not isinstance(other, SdfFile):
            return NotImplemented
        return (
            self.header == other.header
            and np.array_equal(self.data, other.data, equal_nan=True)
            and self.trailer == other.trailer
        )

    @property
    def x_axis(self) -> np.ndarray:
        """X-axis coordinates in metres: ``x_j = (j - 1) * x_scale``, ``j = 1..num_points``.

        ``num_points`` (not ``num_profiles``) is the correct length here, per
        the standard's own coordinate formula and its illustrating figure --
        a separate sentence elsewhere in the standard states the opposite
        pairing, but the formula and figure agree with each other and
        contradict that sentence, which is the more likely drafting error.
        """
        return np.arange(self.header.num_points) * self.header.x_scale

    @property
    def y_axis(self) -> np.ndarray:
        """Y-axis coordinates in metres: ``y_i = (i - 1) * y_scale``, ``i = 1..num_profiles``.

        See :attr:`x_axis` for why ``num_profiles`` (not ``num_points``) is
        the correct length here.

        For a profile (``num_profiles == 1``), the standard defines
        ``y_scale`` as meaningless -- this still returns ``[0.0]``, but that
        single value shouldn't be treated as physically significant.
        """
        return np.arange(self.header.num_profiles) * self.header.y_scale

    @property
    def data_type(self) -> DataType:
        """The data area storage type (see :mod:`sdfio.datatypes`).

        A convenience for ``header.data_type``, which stores the same value
        as a plain ``int`` (the raw on-disk ``DataType`` code).
        """
        return get_data_type(self.header.data_type).type

    @data_type.setter
    def data_type(self, value: DataType | int) -> None:
        self.header.data_type = get_data_type(value).type

    def with_version(
        self,
        version: SdfVersion,
        *,
        data_type: DataType | int | None = None,
        assume_utc: bool = False,
        trailer: str | bytes | None = None,
    ) -> SdfFile:
        """Return a copy of this file retargeted to a different SDF version.

        Converting between versions can fail for three reasons, each
        resolved by an explicit, opt-in parameter rather than silently
        guessing:

        - ``data_type``: some data types are version-restricted
          (``binary32``/``int8`` are version 2.0 only). Pass a compatible
          replacement if the current type isn't valid for ``version``.
        - Timestamps: version 2.0 requires ``create_date``/``mod_date`` to
          be timezone-aware UTC datetimes. If this file's dates are naive
          (e.g. read from a version 1.0 file) and ``version`` is
          :attr:`SdfVersion.V2_0`, pass ``assume_utc=True`` to reinterpret
          their existing wall-clock values as already being UTC (no time
          shift), or construct new dates yourself beforehand.
        - Trailer: version 2.0 requires a well-formed XML trailer, or none
          at all; version 1.0 has no such requirement. If the current
          trailer wouldn't be valid for ``version``, pass a replacement
          (e.g. ``""`` to drop it).

        :raises SdfFormatError: If ``data_type`` isn't valid for
            ``version``, the trailer isn't valid for ``version``, or (for
            ``version=SdfVersion.V2_0``) the timestamps aren't UTC-aware and
            ``assume_utc`` wasn't set.

        Converting to a different version::

            >>> import io
            >>> import numpy as np
            >>> import sdfio
            >>> buf = io.BytesIO()
            >>> sdfio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6,
            ...              metadata=sdfio.SdfMetadata(version=sdfio.SdfVersion.V1_0))
            >>> sdf = sdfio.SdfFile.open(io.BytesIO(buf.getvalue()))
            >>> converted = sdf.with_version(sdfio.SdfVersion.V2_0, assume_utc=True)
            >>> str(converted.header.version)
            '2.0'

        The data type can be changed on its own too, by passing the
        file's current version back in unchanged::

            >>> retyped = sdf.with_version(sdf.header.version, data_type=sdfio.DataType.INT16)
            >>> retyped.data_type.name
            'INT16'
        """
        resolved_data_type = get_data_type(
            self.header.data_type if data_type is None else data_type
        )
        if not resolved_data_type.is_supported(version):
            raise SdfFormatError(
                f"Data type {resolved_data_type.type.name} is not valid for SDF version "
                f"{version}; pass a compatible data_type"
            )

        create_date = self.header.create_date
        mod_date = self.header.mod_date
        if version == SdfVersion.V2_0 and assume_utc:
            if create_date.tzinfo is None:
                create_date = create_date.replace(tzinfo=UTC)
            if mod_date.tzinfo is None:
                mod_date = mod_date.replace(tzinfo=UTC)

        new_trailer = self.trailer if trailer is None else trailer
        validate_trailer_xml(version, new_trailer)

        new_header = replace(
            self.header,
            version=version,
            data_type=resolved_data_type.type,
            create_date=create_date,
            mod_date=mod_date,
        )
        return SdfFile(header=new_header, data=self.data, trailer=new_trailer)

    def raw_data(self) -> np.ndarray:
        """Coded height values in the file's native on-disk dtype.

        This is what is actually written to (or read from) the data area:
        ``data / z_scale``, rounded and cast to ``header.data_type``'s dtype,
        with non-measured/spurious points set to that type's sentinel value
        instead of ``NaN``. Recomputed on each call from the
        current ``data``/``header``, so it always reflects in-place edits to
        either.

        :raises SdfFormatError: If a value is out of range for
            ``header.data_type``, or collides with its invalid-point sentinel.
        """
        data_type = get_data_type(self.header.data_type)
        return encode_raw(self.data, data_type, self.header.z_scale)

    @property
    def trailer_xml(self) -> ET.Element:
        """Trailer content parsed as a well-formed XML document.

        The XML format is mandated for the version 2.0 trailer (no element
        schema is specified); it is not required for version 1.0, but is
        nonetheless commonly used there too. Parsing is hardened against
        entity-expansion attacks (``defusedxml``).

        :raises xml.etree.ElementTree.ParseError: If ``trailer`` is not
            well-formed XML.
        :raises defusedxml.common.DefusedXmlException: If ``trailer`` uses
            DTDs or entities, which are rejected outright rather than
            expanded.

        >>> import xml.etree.ElementTree as ET
        >>> import numpy as np
        >>> from sdfio import SdfFile, SdfHeader
        >>> sdf = SdfFile(header=SdfHeader(), data=np.zeros((1, 1)))
        >>> sdf.trailer_xml = ET.Element("note", attrib={"author": "sdfio"})
        >>> sdf.trailer_xml.tag
        'note'
        >>> sdf.trailer_xml.get("author")
        'sdfio'
        """
        return fromstring(_as_text(self.trailer))

    @trailer_xml.setter
    def trailer_xml(self, root: ET.Element) -> None:
        body = ET.tostring(root, encoding="unicode")
        self.trailer = f'<?xml version="1.0" encoding="UTF-8"?>\r\n{body}'

    @classmethod
    def loads(cls, data: bytes) -> SdfFile:
        """Parse an in-memory SDF blob, detecting the ASCII/binary format from its magic.

        :param data: Full contents of a ``.sdf`` file, in either format.
        :raises SdfFormatError: If ``data`` does not start with a recognized
            SDF magic, or its contents are otherwise malformed.
        """
        if len(data) < 8 or data[:1] not in (b"a", b"b"):
            raise SdfFormatError("Not an SDF file, unexpected magic")

        trailer: str | bytes
        if data[:1] == b"b":
            header, values, trailer = _binary.loads(data)
        else:
            header, values, trailer = _ascii.loads(data.decode("ascii", errors="replace"))
        return cls(header=header, data=values, trailer=trailer)

    @classmethod
    def open(cls, path: PathOrStream) -> SdfFile:
        """Read an SDF file, detecting the ASCII/binary format from its magic.

        :param path: Path to a ``.sdf`` file, or an open binary file object,
            in either format.
        :raises SdfFormatError: If the file does not start with a recognized
            SDF magic, or its contents are otherwise malformed.
        """
        return cls.loads(_read_bytes(path))

    def dumps(self, *, format: FileFormat | None = None) -> bytes:
        """Serialize this file to an in-memory blob, overriding the ASCII/binary format if given.

        :param format: Defaults to the format the data was read with (or
            ``header.binary`` if constructed directly).
        :returns: The serialized ``.sdf`` file contents, always as ``bytes``
            (the ASCII format is 7-bit ASCII text, so it is returned
            encoded rather than as ``str``).
        :raises SdfFormatError: If ``data``'s shape does not match
            ``header``, ``trailer`` isn't 7-bit ASCII, or any other header
            field is invalid (see :func:`write`).
        :raises TypeError: If ``format`` is given and is not a
            :class:`FileFormat` member.
        """
        if format is not None and not isinstance(format, FileFormat):
            raise TypeError(f"format must be a FileFormat member, got {format!r}")
        binary = self.header.binary if format is None else format == FileFormat.BINARY
        header = replace(self.header, binary=binary)
        if binary:
            return _binary.dumps(header, self.data, _as_bytes(self.trailer))
        return _ascii.dumps(header, self.data, _as_text(self.trailer)).encode("ascii")

    def save(self, path: PathOrStream, *, format: FileFormat | None = None) -> None:
        """Write this file to disk or a binary stream, optionally overriding the format.

        :param path: Destination path, or an open binary file object.
        :param format: Defaults to the format the data was read with (or
            ``header.binary`` if constructed directly).
        :raises SdfFormatError: See :meth:`dumps`.
        """
        _write_bytes(path, self.dumps(format=format))


def _read_bytes(source: PathOrStream) -> bytes:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    return source.read()


def _write_bytes(destination: PathOrStream, data: bytes) -> None:
    if isinstance(destination, (str, os.PathLike)):
        Path(destination).write_bytes(data)
    else:
        destination.write(data)


def _as_bytes(trailer: str | bytes) -> bytes:
    # A lossless str<->bytes bridge only -- used for both writing (where
    # validate_trailer_ascii() in _ascii.py/_binary.py enforces the SDF
    # trailer's ASCII-only requirement) and reading back an already-loaded
    # trailer (trailer_xml getter), which stays lenient by design.
    return trailer if isinstance(trailer, bytes) else trailer.encode("utf-8")


def _as_text(trailer: str | bytes) -> str:
    return trailer if isinstance(trailer, str) else trailer.decode("utf-8", errors="replace")


def read(path: PathOrStream) -> SdfFile:
    """Read an SDF surface data file into memory.

    :param path: Path to a ``.sdf`` file, or an open binary file object, in
        either ASCII or binary format.
    :returns: The parsed header, data matrix (in metres, ``NaN`` for
        non-measured or spurious points) and raw trailer.
    :raises SdfFormatError: If the file does not start with a recognized
        SDF magic, or its contents are otherwise malformed.
    """
    return SdfFile.open(path)


def write(
    path: PathOrStream,
    data: np.ndarray,
    *,
    x_scale: float,
    y_scale: float,
    z_scale: float | Literal["auto"] = 1e-6,
    format: FileFormat = FileFormat.BINARY,
    metadata: SdfMetadata | None = None,
    trailer: str | bytes = "",
) -> None:
    """Write a NumPy array to an SDF surface data file.

    :param path: Destination path, or an open binary file object.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres. ``NaN`` marks non-measured or spurious points.
    :param x_scale: Sampling interval along x, in metres.
    :param y_scale: Sampling interval along y, in metres.
    :param z_scale: Scale factor applied to the coded height values, or
        ``"auto"`` to pick the smallest (highest-resolution) scale that
        fits ``data`` into ``metadata.data_type`` without overflow (see
        :func:`sdfio.datatypes.suggest_z_scale`).
    :param format: :attr:`FileFormat.ASCII` or :attr:`FileFormat.BINARY`.
    :param metadata: Reusable header template (``version``,
        ``manufacturer_id``, ``create_date``, ``mod_date``, ``z_resolution``,
        ``data_type``); defaults to :class:`SdfMetadata`'s own defaults (SDF
        version 2.0, ``binary64``, current UTC timestamps). Every field on it
        is used as given -- there is no overlap with ``x_scale``/``y_scale``/
        ``z_scale``/``format`` above, or with ``num_points``/``num_profiles``,
        which are always taken from ``data``'s shape. Pass a shared template
        to write several files with the same metadata, e.g.
        ``metadata=SdfMetadata(manufacturer_id="acme")``.
        ``manufacturer_id`` must be 7-bit ASCII; values longer than 10 bytes
        are silently truncated.
    :param trailer: Optional record 3 trailer content; must be 7-bit ASCII.
    :raises SdfFormatError: If ``data`` is not 2-D, ``z_scale`` is not
        positive, ``metadata.manufacturer_id`` isn't 7-bit ASCII,
        ``metadata.data_type`` is invalid or unsupported for
        ``metadata.version``, ``metadata``'s timestamps aren't UTC-aware for
        SDF version 2.0, ``trailer`` isn't 7-bit ASCII, a non-empty
        ``trailer`` isn't well-formed XML for version 2.0, or ``data`` does
        not fit ``metadata.data_type`` without overflow.

    ``path`` also accepts an open binary stream, e.g. to avoid touching disk::

        >>> import io
        >>> import numpy as np
        >>> import sdfio
        >>> buf = io.BytesIO()
        >>> sdfio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> sdf = sdfio.read(io.BytesIO(buf.getvalue()))
        >>> sdf.data.shape
        (2, 3)
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise SdfFormatError("SDF data must be a 2-D (num_profiles, num_points) array")
    meta = metadata if metadata is not None else SdfMetadata()
    resolved_data_type = get_data_type(meta.data_type)
    resolved_z_scale = suggest_z_scale(array, resolved_data_type) if z_scale == "auto" else z_scale
    # Version 2.0 timestamps are UTC.
    now = datetime.now(UTC)
    header = SdfHeader(
        version=meta.version,
        binary=format == FileFormat.BINARY,
        manufacturer_id=meta.manufacturer_id,
        create_date=meta.create_date or now,
        mod_date=meta.mod_date or now,
        num_points=array.shape[1],
        num_profiles=array.shape[0],
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=resolved_z_scale,
        z_resolution=meta.z_resolution,
        compression=0,
        data_type=resolved_data_type.type,
        check_type=0,
    )
    SdfFile(header=header, data=array, trailer=trailer).save(path, format=format)
