# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""sdfio: read and write ISO 25178-71 SDF surface data files.

The Surface Data File (SDF) format is defined by ISO 25178-71, "Geometrical
product specifications (GPS) -- Surface texture: Areal -- Part 71: Software
measurement standards". It stores areal (or profile) surface topography
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
from .header import MAGICS, SdfDialect, SdfHeader, SdfMetadata

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
    "get_data_type",
    "read",
    "suggest_z_scale",
    "write",
]

__version__ = "1.1.0"
