# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""sdfio: read and write ISO 25178-71 SDF surface data files.

The Surface Data File (SDF) format is defined by ISO 25178-71, "Geometrical
product specifications (GPS) -- Surface texture: Areal -- Part 71: Surface
data file (SDF) file format". It stores areal (or profile) surface topography
measurements as a rectangular grid of height values, in either a
human-readable ASCII or a compact binary format.

Basic usage::

    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> import sdfio
    >>> path = os.path.join(tempfile.mkdtemp(), "example.sdf")
    >>> sdfio.write(path, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
    >>> sdf = sdfio.read(path)
    >>> sdf.data.shape
    (2, 3)
    >>> sdf.header.z_scale
    1e-06

The examples below write to memory, so that they leave no files behind.

A surface is a ``(rows, columns)`` array of heights in metres. ``y`` grows with the row and ``x``
with the column. Points that are not measured are ``NaN``::

    >>> import io
    >>> heights = np.array([[0.0, 1e-6, 2e-6], [3e-6, np.nan, 5e-6]])
    >>> buffer = io.BytesIO()
    >>> sdfio.write(buffer, heights, x_scale=2e-6, y_scale=3e-6)
    >>> surface = sdfio.read(io.BytesIO(buffer.getvalue()))
    >>> surface.data.shape, bool(np.isnan(surface.data[1, 1]))
    ((2, 3), True)
    >>> surface.x_axis.tolist(), surface.y_axis.tolist()
    ([0.0, 2e-06, 4e-06], [0.0, 3e-06])

A profile is a surface with one row::

    >>> profile = np.sin(np.linspace(0.0, 6.0, 100)).reshape(1, -1) * 1e-6
    >>> buffer = io.BytesIO()
    >>> sdfio.write(buffer, profile, x_scale=1e-6, y_scale=1e-6)
    >>> sdfio.read(io.BytesIO(buffer.getvalue())).data.shape
    (1, 100)

The file is described by the header. The dialect, the data type, the manufacturer, the dates and
the quantization step come from a template, and the representation from ``format``. The heights
are stored as integers of ``data_type`` times ``z_scale``, which ``"auto"`` chooses as the
smallest scale that fits the data::

    >>> metadata = sdfio.SdfMetadata(
    ...     dialect=sdfio.SdfDialect.ISO_2_0,
    ...     manufacturer_id="acme",
    ...     data_type=sdfio.DataType.INT16,
    ... )
    >>> buffer = io.BytesIO()
    >>> sdfio.write(
    ...     buffer, heights, x_scale=2e-6, y_scale=3e-6, z_scale="auto",
    ...     format=sdfio.FileFormat.ASCII, metadata=metadata,
    ... )
    >>> text = buffer.getvalue().decode("ascii")
    >>> text.splitlines()[0], "DataType = 5" in text, "BAD" in text
    ('aISO-2.0', True, True)

The trailer holds free text, or "Name = Value" fields, which ISO-2.0 requires::

    >>> sdf = sdfio.read(io.BytesIO(buffer.getvalue()))
    >>> sdf.trailer_fields = {"OperatorName": "Jane Doe"}
    >>> sdfio.read(io.BytesIO(sdf.dumps())).trailer_fields["OperatorName"]
    'Jane Doe'

A file is converted to another dialect, data type or representation with
:meth:`SdfFile.with_dialect` and :meth:`SdfFile.dumps`::

    >>> converted = sdf.with_dialect(sdfio.SdfDialect.ISO_1_0, trailer="")
    >>> str(converted.header.dialect)
    'ISO-1.0'
    >>> binary = sdfio.read(io.BytesIO(converted.dumps(format=sdfio.FileFormat.BINARY)))
    >>> binary.header.binary
    True

:meth:`SdfFile.points` gives the points of the grid with their ``x`` and ``y`` coordinates, as the
``(rows, columns, 3)`` array that plotting libraries take. With ``drop_invalid`` it is the
``(N, 3)`` list of the measured points that point cloud libraries take::

    >>> points = surface.points()
    >>> points.shape
    (2, 3, 3)
    >>> points[1, 2].tolist()
    [4e-06, 3e-06, 5e-06]
    >>> surface.points(drop_invalid=True).shape
    (5, 3)
"""

from __future__ import annotations

from .datatypes import (
    DATA_TYPES,
    DataType,
    SdfDataType,
    decode_raw,
    encode_raw,
    get_data_type,
    suggest_z_scale,
)
from .exceptions import SdfError, SdfFormatError, SdfVersionError
from .file import FileFormat, SdfFile, read, write
from .header import (
    MAGICS,
    SdfDialect,
    SdfHeader,
    SdfMetadata,
    format_tagged_fields,
    parse_tagged_fields,
    sanitize_tagged_fields,
    validate_trailer_ascii,
    validate_trailer_tagged,
)

__all__ = [
    "DATA_TYPES",
    "MAGICS",
    "DataType",
    "FileFormat",
    "SdfDataType",
    "SdfDialect",
    "SdfError",
    "SdfFile",
    "SdfFormatError",
    "SdfHeader",
    "SdfMetadata",
    "SdfVersionError",
    "__version__",
    "decode_raw",
    "encode_raw",
    "format_tagged_fields",
    "get_data_type",
    "parse_tagged_fields",
    "read",
    "sanitize_tagged_fields",
    "suggest_z_scale",
    "validate_trailer_ascii",
    "validate_trailer_tagged",
    "write",
]

__version__ = "1.3.0"
