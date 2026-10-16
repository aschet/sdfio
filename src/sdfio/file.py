# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""High-level, NumPy-oriented interface to SDF surface data files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

import numpy as np

from . import _io
from .datatypes import (
    DataType,
    encode_raw,
    get_data_type,
    require_supported_data_type,
    suggest_z_scale,
)
from .exceptions import SdfFormatError
from .header import Header, Metadata
from .revision import FileFormat, Revision
from .trailer import (
    format_tagged_fields,
    parse_tagged_fields,
    sanitize_tagged_fields,
    validate_trailer_tagged,
)

__all__ = ["SdfFile", "read", "write"]

PathOrStream = _io.PathOrStream


# eq=False: the generated __eq__ compares arrays with ==, which is ambiguous for NumPy.
@dataclass(eq=False)
class SdfFile:
    """An SDF surface data file in memory.

    :param header: Record 1.
    :param data: ``(num_profiles, num_points)`` array of height values in metres, with ``NaN``
        for a point that was not measured. It is held as a 2-D ``float64`` array whenever it is
        set, and its shape is the number of profiles and points of the file. ``y`` increases with
        the row, see :attr:`y_axis`.
    :param trailer: Record 3: text from an ASCII file, bytes from a binary file.
    """

    header: Header
    data: np.ndarray
    trailer: str | bytes = ""

    def __setattr__(self, name: str, value: object) -> None:
        """Hold ``data`` as a 2-D ``float64`` array, whenever it is set."""
        if name == "data":
            value = _as_grid(value)
        super().__setattr__(name, value)

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
        """Return the x coordinates of the columns in metres: ``x_j = (j - 1) * x_scale``.

        There is one per point of a row. One sentence of the 2026 edition of the standard pairs
        the numbers the other way round, against its formula and figure; a later edition fixes it.
        """
        return np.arange(self.data.shape[1]) * self.header.x_scale

    @property
    def y_axis(self) -> np.ndarray:
        """Return the y coordinates of the rows in metres: ``y_i = (i - 1) * y_scale``.

        ``y`` increases with the row, in the right-handed coordinate system of the standard, for
        every revision, see :attr:`RevisionRules.reverses_profile_order`. For a profile the
        standard calls ``y_scale`` meaningless, and the result is ``[0.0]``.
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

    def with_revision(
        self,
        revision: Revision,
        *,
        data_type: DataType | int | None = None,
        trailer: str | bytes | None = None,
        refit_z_scale: bool = False,
    ) -> SdfFile:
        """Return a copy of this file retargeted to another revision.

        What the new revision does not take is not guessed; each is a parameter:

        - ``data_type``: some data types exist in some revisions only, such as ``binary32`` and
          ``int8`` in ISO-2.0 and BCR-1.0. Pass one the new revision has.
        - ``trailer``: ISO-2.0 requires a tagged "Name = Value" trailer, or none. Pass a
          replacement, such as ``""``, if the current one doesn't fit.
        - ``refit_z_scale``: the data may not fit the new data type at the current ``z_scale``,
          which the write would report. With ``True`` the ``z_scale`` is kept if the data fits,
          and otherwise replaced by the smallest that does, see
          :func:`sdfio.datatypes.suggest_z_scale`. The data is not changed.

        The dates need no parameter. They are UTC in every revision and converted to the
        convention of the revision when the file is written.

        :raises SdfFormatError: If ``data_type`` or the trailer doesn't fit ``revision``.

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

        To change only the data type, pass the current revision::

            >>> retyped = sdf.with_revision(sdf.header.revision, data_type=sdfio.DataType.INT16)
            >>> retyped.header.data_type.name
            'INT16'

        ``refit_z_scale`` takes a scale that fits::

            >>> tall = sdfio.SdfFile.from_array(
            ...     np.array([[0.0, 0.5]]), x_scale=1e-6, y_scale=1e-6, z_scale=1e-9
            ... )
            >>> tall.with_revision(
            ...     sdfio.Revision.ISO_2_0, data_type=sdfio.DataType.INT16, refit_z_scale=True
            ... ).header.z_scale > 1e-9
            True
        """
        resolved_data_type = require_supported_data_type(
            self.header.data_type if data_type is None else data_type, revision
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
        array = _as_grid(data)
        header = replace(
            self.header,
            x_scale=self.header.x_scale if x_scale is None else x_scale,
            y_scale=self.header.y_scale if y_scale is None else y_scale,
        )
        return SdfFile(header=header, data=array, trailer=self.trailer)

    def raw_data(self) -> np.ndarray:
        """Return the coded heights in the on-disk dtype of the data type.

        That is ``data / z_scale``, rounded, with the marker of the revision instead of ``NaN``
        for a point that was not measured. It is computed on each call, so it follows changes
        of ``data`` and ``header``.

        :raises SdfFormatError: If a value is out of range for
            ``header.data_type``, or collides with its invalid-point sentinel.
        """
        data_type = get_data_type(self.header.data_type)
        return encode_raw(self.data, data_type, self.header.z_scale, self.header.revision)

    @property
    def trailer_fields(self) -> Mapping[str, str]:
        """Return the trailer as "Name = Value" fields, the tagged format of the header.

        ISO-2.0 requires this form on write. A line of another form is left out, see
        :func:`sdfio.trailer.sanitize_tagged_fields`; use
        :func:`sdfio.trailer.parse_tagged_fields` to detect it instead. The result is a read-only
        snapshot. Assign the whole property to change the trailer:
        ``sdf.trailer_fields = {**sdf.trailer_fields, "X": "Y"}``.

        >>> import numpy as np
        >>> from sdfio import SdfFile, Header
        >>> sdf = SdfFile(header=Header(), data=np.zeros((1, 1)))
        >>> sdf.trailer_fields = {"OperatorName": "Jane Doe"}
        >>> sdf.trailer_fields
        mappingproxy({'OperatorName': 'Jane Doe'})
        """
        sanitized = sanitize_tagged_fields(_io.as_text(self.trailer))
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

        The arguments are those of :func:`sdfio.write`, which builds the file this way and saves
        it. Only the shape of ``data`` is checked here, and ``metadata`` has checked its data type
        against its revision. Whether ``data`` fits the data type, the manufacturer ID and the
        trailer are checked when the file is written, see :meth:`save`.

        :param data: ``(num_profiles, num_points)`` array of height values in metres. ``NaN``
            marks points that were not measured. A ``float64`` array is used as it is, not copied.
        :param x_scale: Sampling interval along x, in metres.
        :param y_scale: Sampling interval along y, in metres.
        :param z_scale: Scale factor of the coded heights, or ``"auto"`` for the smallest that
            fits ``data`` into ``metadata.data_type``, see
            :func:`sdfio.datatypes.suggest_z_scale`.
        :param format: The encoding, which becomes :attr:`Header.format`.
        :param metadata: Header template, shared by files that have the same metadata, such as
            ``Metadata(manufacturer_id="acme")``. The defaults are ISO-2.0, ``binary64`` and the
            current time. A binary file holds 10 characters of the manufacturer ID.
        :param trailer: Record 3, the trailer.
        :raises SdfFormatError: If ``data`` is not 2-D.

        >>> import numpy as np
        >>> sdf = SdfFile.from_array(np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> sdf.data.shape, sdf.header.revision.value, sdf.header.format.name
        ((2, 3), 'ISO-2.0', 'BINARY')
        """
        array = _as_grid(data)
        meta = metadata if metadata is not None else Metadata()
        resolved_z_scale = (
            suggest_z_scale(array, get_data_type(meta.data_type), meta.revision)
            if z_scale == "auto"
            else z_scale
        )
        now = datetime.now(UTC)
        template = {field.name: getattr(meta, field.name) for field in fields(Metadata)}
        template["create_date"] = meta.create_date or now
        template["mod_date"] = meta.mod_date or now
        header = Header(
            **template, format=format, x_scale=x_scale, y_scale=y_scale, z_scale=resolved_z_scale
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
        header, values, trailer = _io.decode(data)
        return cls(header=header, data=values, trailer=trailer)

    @classmethod
    def open(cls, path: PathOrStream) -> SdfFile:
        """Read an SDF file, detecting the ASCII/binary format from its magic.

        :param path: Path to a ``.sdf`` file, or an open binary file object,
            in either format.
        :raises SdfFormatError: If the file does not start with a recognized
            SDF magic, or its contents are otherwise malformed.
        """
        return cls.loads(_io.read_bytes(path))

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
        return _io.encode(header, self.data, self.trailer)

    def save(self, path: PathOrStream, *, format: FileFormat | None = None) -> None:
        """Write this file to disk or a binary stream, optionally overriding the format.

        :param path: Destination path, or an open binary file object.
        :param format: Defaults to the format the data was read with (or
            ``header.format`` if constructed directly).
        :raises SdfFormatError: See :meth:`dumps`.
        """
        _io.write_bytes(path, self.dumps(format=format))


def _as_grid(data: object) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise SdfFormatError("SDF data must be a 2-D (num_profiles, num_points) array")
    return array


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
    :param data: ``(num_profiles, num_points)`` array of height values in metres. ``NaN`` marks
        points that were not measured.
    :param x_scale: Sampling interval along x, in metres.
    :param y_scale: Sampling interval along y, in metres.
    :param z_scale: Scale factor of the coded heights, or ``"auto"`` for the smallest that fits
        ``data`` into ``metadata.data_type``, see :func:`sdfio.datatypes.suggest_z_scale`.
    :param format: The encoding, ASCII or binary.
    :param metadata: Header template, shared by files that have the same metadata, such as
        ``Metadata(manufacturer_id="acme")``. The defaults are ISO-2.0, ``binary64`` and the
        current time. A binary file holds 10 characters of the manufacturer ID.
    :param trailer: Record 3, the trailer, as 7-bit ASCII.
    :raises SdfFormatError: If ``data`` is not 2-D, ``z_scale`` is not positive, the manufacturer
        ID or the trailer is not 7-bit ASCII, a trailer is not tagged "Name = Value" where the
        revision requires it, or ``data`` does not fit ``metadata.data_type``.

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
