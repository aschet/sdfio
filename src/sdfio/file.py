# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""High-level, NumPy-oriented interface to SDF surface data files."""

from __future__ import annotations

import codecs
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import IO, Literal

import numpy as np

from . import _ascii, _binary
from .datatypes import DataType, encode_raw, get_data_type, suggest_z_scale
from .exceptions import SdfFormatError
from .header import (
    ASCII_PREFIX,
    BINARY_PREFIX,
    MAGIC_SIZE,
    FileFormat,
    Header,
    Metadata,
    Revision,
    format_tagged_fields,
    parse_tagged_fields,
    sanitize_tagged_fields,
    validate_trailer_tagged,
)

__all__ = ["FileFormat", "SdfFile", "read", "write"]

PathLike = str | os.PathLike[str]
PathOrStream = PathLike | IO[bytes]


# eq=False: dataclass's auto __eq__ would do `data == other.data`, which
# raises on a numpy array's ambiguous truth value; __eq__ below replaces it
# with a NaN-aware, array-safe comparison instead.
@dataclass(eq=False)
class SdfFile:
    """In-memory representation of an SDF surface data file.

    :param header: Parsed record 1 header fields.
    :param data: ``(num_profiles, num_points)`` array of height values in
        metres, with ``NaN`` marking non-measured or spurious points. ``y``
        increases with row index, in a right-handed coordinate system --
        see :attr:`x_axis`/:attr:`y_axis`.
    :param trailer: Raw record 3 trailer content; text for ASCII files,
        bytes for binary files.
    """

    header: Header
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
        the standard's own coordinate formula and its illustrating figure.
        The current edition has a separate sentence elsewhere that states the
        opposite pairing, contradicting the formula and figure; a later
        edition fixes that sentence to agree with the formula instead.
        """
        return np.arange(self.data.shape[1]) * self.header.x_scale

    @property
    def y_axis(self) -> np.ndarray:
        """Y-axis coordinates in metres: ``y_i = (i - 1) * y_scale``, ``i = 1..num_profiles``.

        ``y`` increases with row index, per the standard's right-handed
        coordinate system. :attr:`Revision.reverses_profile_order` is what
        keeps this true for BCR-1.0 too, despite its different on-disk row
        order.

        See :attr:`x_axis` for why ``num_profiles`` (not ``num_points``) is
        the correct length here.

        For a profile (:attr:`is_profile`), the standard calls ``y_scale``
        meaningless; this returns ``[0.0]`` regardless of its value.
        """
        return np.arange(self.data.shape[0]) * self.header.y_scale

    def points(self, *, drop_invalid: bool = False) -> np.ndarray:
        """Return the points of the grid as an ``(num_profiles, num_points, 3)`` array in metres.

        The last axis holds ``x``, ``y`` and ``z``: ``x`` from :attr:`x_axis` along the columns,
        ``y`` from :attr:`y_axis` along the rows, and ``z`` from :attr:`data`. The array is
        new, so changing it does not change the file. A point that is not measured has ``NaN``
        in ``z`` and keeps its ``x`` and ``y``. A profile has one row, with ``y`` equal to 0.

        ``reshape(-1, 3)`` gives the ``(N, 3)`` array of points that point cloud libraries take.

        :param drop_invalid: Leave out the points that are not measured. The result is then the
            ``(N, 3)`` array of the measured points, row by row, and the shape of the grid is lost.

        >>> import numpy as np
        >>> from sdfio import SdfFile, Header
        >>> header = Header(x_scale=1e-6, y_scale=2e-6)
        >>> sdf = SdfFile(header=header, data=np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]]))
        >>> points = sdf.points()
        >>> points.shape
        (2, 3, 3)
        >>> points[1, 2].tolist()
        [2e-06, 2e-06, 5.0]
        >>> sdf.points(drop_invalid=True).shape
        (5, 3)
        """
        grid = np.empty((*self.data.shape, 3), dtype=np.float64)
        grid[..., 0] = self.x_axis
        grid[..., 1] = self.y_axis[:, np.newaxis]
        grid[..., 2] = self.data
        if not drop_invalid:
            return grid
        flat = grid.reshape(-1, 3)
        return flat[np.isfinite(flat[:, 2])]

    @property
    def is_profile(self) -> bool:
        """Whether the file holds a profile and not an areal surface.

        The standard defines a profile as a file with one row ("in case of a profile, the value
        of 'Number of rows in the data matrix' is 1") and calls ``y_scale`` meaningless for it.
        A surface with a single row is therefore a profile.

        >>> import numpy as np
        >>> from sdfio import SdfFile, Header
        >>> profile = SdfFile(header=Header(), data=np.zeros((1, 3)))
        >>> surface = SdfFile(header=Header(), data=np.zeros((2, 3)))
        >>> profile.is_profile, surface.is_profile
        (True, False)
        """
        return bool(self.data.shape[0] == 1)

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

    def with_revision(
        self,
        revision: Revision,
        *,
        data_type: DataType | int | None = None,
        trailer: str | bytes | None = None,
        refit_z_scale: bool = False,
    ) -> SdfFile:
        """Return a copy of this file retargeted to a different SDF revision.

        Converting between revisions can fail for three reasons, each resolved
        by an explicit, opt-in parameter rather than silently guessing:

        - ``data_type``: some data types are revision-restricted (e.g.
          ``binary32``/``int8`` are ISO-2.0 only; BCR has no such
          restriction). Pass a compatible replacement if the current type
          isn't valid for ``revision``.
        - Trailer: a revision for which :attr:`Revision.requires_tagged_trailer`
          is ``True`` requires a tagged "Name = Value" trailer, or none at
          all; others have no such requirement. If the current trailer
          wouldn't be valid for ``revision``, pass a replacement (e.g. ``""``
          to drop it).
        - ``refit_z_scale``: the data may not fit the new data type with the
          current ``z_scale``, for example when ``binary64`` becomes ``int16``,
          which the file would only report when it is written. With
          ``refit_z_scale=True`` the ``z_scale`` is kept if the data fits the new
          data type in ``revision``, and otherwise replaced by the smallest one that
          does, see :func:`sdfio.datatypes.suggest_z_scale`. The data is not changed.

        Timestamps need no such parameter: ``create_date``/``mod_date`` are
        held as UTC for every revision and are converted to the convention of
        ``revision`` (UTC, or the local time of the system for the older
        revisions) when the file is written, see
        :func:`sdfio.header.format_sdf_datetime`.

        :raises SdfFormatError: If ``data_type`` isn't valid for ``revision``,
            or the trailer isn't valid for ``revision``.

        Converting to a different revision::

            >>> import io
            >>> import numpy as np
            >>> import sdfio
            >>> buf = io.BytesIO()
            >>> sdfio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6,
            ...              metadata=sdfio.Metadata(revision=sdfio.Revision.ISO_1_0))
            >>> sdf = sdfio.SdfFile.open(io.BytesIO(buf.getvalue()))
            >>> converted = sdf.with_revision(sdfio.Revision.ISO_2_0)
            >>> str(converted.header.revision)
            'ISO-2.0'

        The data type can be changed on its own too, by passing the
        file's current revision back in unchanged::

            >>> retyped = sdf.with_revision(sdf.header.revision, data_type=sdfio.DataType.INT16)
            >>> retyped.data_type.name
            'INT16'

        Heights that do not fit the new data type at the old ``z_scale`` are
        scaled so that they do, with ``refit_z_scale``::

            >>> tall = sdfio.SdfFile.from_array(
            ...     np.array([[0.0, 0.5]]), x_scale=1e-6, y_scale=1e-6, z_scale=1e-9
            ... )
            >>> tall.with_revision(
            ...     sdfio.Revision.ISO_2_0, data_type=sdfio.DataType.INT16, refit_z_scale=True
            ... ).header.z_scale > 1e-9
            True
        """
        resolved_data_type = get_data_type(
            self.header.data_type if data_type is None else data_type
        )
        if not resolved_data_type.is_supported(revision):
            raise SdfFormatError(
                f"Data type {resolved_data_type.type.name} is not valid for SDF revision "
                f"{revision}; pass a compatible data_type"
            )

        new_trailer = self.trailer if trailer is None else trailer
        validate_trailer_tagged(revision, new_trailer)

        z_scale = self.header.z_scale
        if refit_z_scale:
            try:
                encode_raw(self.data, resolved_data_type, z_scale, revision)
            except SdfFormatError:  # the data does not fit the new type at this scale
                z_scale = suggest_z_scale(self.data, resolved_data_type, revision)

        new_header = replace(
            self.header,
            revision=revision,
            data_type=resolved_data_type.type,
            z_scale=z_scale,
        )
        return SdfFile(header=new_header, data=self.data, trailer=new_trailer)

    def with_data(
        self,
        data: np.ndarray,
        *,
        x_scale: float | None = None,
        y_scale: float | None = None,
    ) -> SdfFile:
        """Return a copy with other heights, for example a part of this file.

        The rest of the header and the trailer are kept. Like :meth:`from_array`, this only
        checks the shape of ``data``; whether it fits the data type and the ``z_scale`` is checked
        when the file is written.

        :param data: ``(num_profiles, num_points)`` array of height values in metres.
        :param x_scale: New sampling interval along x, in metres, if it is not the one of the file.
        :param y_scale: New sampling interval along y, in metres, if it is not the one of the file.
        :raises SdfFormatError: If ``data`` is not 2-D.

        >>> import numpy as np
        >>> sdf = SdfFile.from_array(np.zeros((4, 6)), x_scale=1e-6, y_scale=1e-6)
        >>> part = sdf.with_data(sdf.data[1:3, ::2], x_scale=2e-6)
        >>> part.data.shape, part.header.x_scale
        ((2, 3), 2e-06)
        """
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise SdfFormatError("SDF data must be a 2-D (num_profiles, num_points) array")
        header = replace(
            self.header,
            x_scale=self.header.x_scale if x_scale is None else x_scale,
            y_scale=self.header.y_scale if y_scale is None else y_scale,
        )
        return SdfFile(header=header, data=array, trailer=self.trailer)

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
        return encode_raw(self.data, data_type, self.header.z_scale, self.header.revision)

    @property
    def trailer_fields(self) -> Mapping[str, str]:
        """Trailer content as "Name = Value" fields, the tagged format the header uses.

        Required on write for a revision whose
        :attr:`Revision.requires_tagged_trailer` is ``True`` (ISO-2.0);
        optional elsewhere.

        Never raises: a line that isn't "Name = Value" is silently dropped
        (see :func:`sdfio.header.sanitize_tagged_fields`). Use
        :func:`sdfio.header.parse_tagged_fields` directly to detect a
        malformed trailer instead.

        A read-only, freshly-parsed snapshot, not a live view -- mutating it
        in place raises ``TypeError``; assign the whole property to change
        the trailer (``sdf.trailer_fields = {**sdf.trailer_fields, "X": "Y"}``).

        >>> import numpy as np
        >>> from sdfio import SdfFile, Header
        >>> sdf = SdfFile(header=Header(), data=np.zeros((1, 1)))
        >>> sdf.trailer_fields = {"OperatorName": "Jane Doe"}
        >>> sdf.trailer_fields
        mappingproxy({'OperatorName': 'Jane Doe'})
        """
        # sanitize_tagged_fields() already guarantees clean "Name = Value"
        # text, so parse_tagged_fields() here can never raise.
        sanitized = sanitize_tagged_fields(_as_text(self.trailer))
        return MappingProxyType(parse_tagged_fields(sanitized))

    @trailer_fields.setter
    def trailer_fields(self, fields: Mapping[str, str]) -> None:
        self.trailer = format_tagged_fields(fields)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        x_scale: float,
        y_scale: float,
        z_scale: float | Literal["auto"] = 1e-6,
        format: FileFormat = FileFormat.BINARY,
        metadata: Metadata | None = None,
        trailer: str | bytes = "",
    ) -> SdfFile:
        """Build an in-memory SDF file from a NumPy array, without writing it.

        The arguments are those of :func:`sdfio.write`, which builds the file this way and
        saves it. Only the shape of ``data`` and the data type code are checked here. Everything
        else, such as whether ``data`` fits the data type, whether the data type exists in the
        revision, the ``z_scale``, the manufacturer ID and the trailer, is checked when the file is
        written, see :meth:`save` and :meth:`dumps`.

        :param data: ``(num_profiles, num_points)`` array of height values in
            metres. ``NaN`` marks non-measured or spurious points. An array that already holds
            ``float64`` values is used as it is, not copied.
        :param x_scale: Sampling interval along x, in metres.
        :param y_scale: Sampling interval along y, in metres.
        :param z_scale: Scale factor applied to the coded height values, or
            ``"auto"`` to pick the smallest (highest-resolution) scale that
            fits ``data`` into ``metadata.data_type`` without overflow (see
            :func:`sdfio.datatypes.suggest_z_scale`).
        :param format: :attr:`FileFormat.ASCII` or :attr:`FileFormat.BINARY`, which is what
            :attr:`Header.format` is set to.
        :param metadata: Reusable header template (``revision``,
            ``manufacturer_id``, ``create_date``, ``mod_date``, ``z_resolution``,
            ``data_type``); defaults to :class:`Metadata`'s own defaults
            (ISO-2.0, ``binary64``, the current time). Every field on it is used as given --
            there is no overlap with ``x_scale``/``y_scale``/``z_scale``/``format`` above.
            Pass a shared template to build several files with the same metadata, e.g.
            ``metadata=Metadata(manufacturer_id="acme")``. ``manufacturer_id`` must be 7-bit
            ASCII; values longer than 10 bytes are silently truncated. The dates are held as UTC
            to the minute, see :class:`Header`.
        :param trailer: Optional record 3 trailer content.
        :raises SdfFormatError: If ``data`` is not 2-D or ``metadata.data_type`` is not a known
            data type code.

        >>> import numpy as np
        >>> sdf = SdfFile.from_array(np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> sdf.data.shape, sdf.header.revision.value, sdf.header.format.name
        ((2, 3), 'ISO-2.0', 'BINARY')
        """
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise SdfFormatError("SDF data must be a 2-D (num_profiles, num_points) array")
        meta = metadata if metadata is not None else Metadata()
        resolved_data_type = get_data_type(meta.data_type)
        resolved_z_scale = (
            suggest_z_scale(array, resolved_data_type, meta.revision)
            if z_scale == "auto"
            else z_scale
        )
        now = datetime.now(UTC)
        header = Header(
            revision=meta.revision,
            format=format,
            manufacturer_id=meta.manufacturer_id,
            create_date=meta.create_date or now,
            mod_date=meta.mod_date or now,
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=resolved_z_scale,
            z_resolution=meta.z_resolution,
            data_type=resolved_data_type.type,
        )
        return cls(header=header, data=array, trailer=trailer)

    @classmethod
    def loads(cls, data: bytes) -> SdfFile:
        """Parse an in-memory SDF blob, detecting the ASCII/binary format from its magic.

        :param data: Full contents of a ``.sdf`` file, in either format. A UTF-8 byte order mark
            in front of the magic, which some tools write, is ignored.
        :raises SdfFormatError: If ``data`` does not start with a recognized
            SDF magic, or its contents are otherwise malformed.
        """
        data = data.removeprefix(codecs.BOM_UTF8)
        ascii_prefix = ASCII_PREFIX.encode("ascii")
        binary_prefix = BINARY_PREFIX.encode("ascii")
        prefix = data[:1].lower()
        if len(data) < MAGIC_SIZE or prefix not in (ascii_prefix, binary_prefix):
            raise SdfFormatError("Not an SDF file, unexpected magic")

        trailer: str | bytes
        if prefix == binary_prefix:
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
            ``header.format`` if constructed directly).
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
        header = replace(self.header, format=self.header.format if format is None else format)
        if header.format is FileFormat.BINARY:
            return _binary.dumps(header, self.data, _as_bytes(self.trailer))
        return _ascii.dumps(header, self.data, _as_text(self.trailer)).encode("ascii")

    def save(self, path: PathOrStream, *, format: FileFormat | None = None) -> None:
        """Write this file to disk or a binary stream, optionally overriding the format.

        :param path: Destination path, or an open binary file object.
        :param format: Defaults to the format the data was read with (or
            ``header.format`` if constructed directly).
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
    # A lossless str<->bytes bridge only -- validate_trailer_ascii() in
    # _ascii.py/_binary.py enforces the SDF trailer's ASCII-only requirement
    # when writing; this itself stays lenient by design.
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
    metadata: Metadata | None = None,
    trailer: str | bytes = "",
) -> None:
    """Write a NumPy array to an SDF surface data file.

    The file is built by :meth:`SdfFile.from_array`, which takes the same arguments and returns
    it without writing, and saved with :meth:`SdfFile.save`.

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
    :param metadata: Reusable header template (``revision``,
        ``manufacturer_id``, ``create_date``, ``mod_date``, ``z_resolution``,
        ``data_type``); defaults to :class:`Metadata`'s own defaults
        (ISO-2.0, ``binary64``, current UTC timestamps). Every field on it is
        used as given -- there is no overlap with
        ``x_scale``/``y_scale``/``z_scale``/``format`` above. Pass a shared template to write
        several files with the same metadata, e.g. ``metadata=Metadata(manufacturer_id="acme")``.
        ``manufacturer_id`` must be 7-bit ASCII; values longer than 10 bytes
        are silently truncated.
    :param trailer: Optional record 3 trailer content; must be 7-bit ASCII.
    :raises SdfFormatError: If ``data`` is not 2-D, ``z_scale`` is not
        positive, ``metadata.manufacturer_id`` isn't 7-bit ASCII,
        ``metadata.data_type`` is invalid or unsupported for
        ``metadata.revision``, ``trailer`` isn't 7-bit ASCII, a
        non-empty ``trailer`` isn't in the tagged "Name = Value" format for a
        revision that requires it, or ``data`` does not fit
        ``metadata.data_type`` without overflow.

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
    SdfFile.from_array(
        data,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        format=format,
        metadata=metadata,
        trailer=trailer,
    ).save(path)
