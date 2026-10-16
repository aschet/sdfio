# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Revisions, data types and representations, and converting between them."""

import io

import numpy as np

import sdfio

heights = 1e-6 * np.random.default_rng(3).normal(size=(100, 120))

# Each revision defines the data types that it allows. The heights of an integer type are stored as
# multiples of the scale, so that the range of the type and the resolution are traded off.
print(
    "data types of ISO-1.0:",
    [
        data_type.name
        for data_type in sdfio.DataType
        if sdfio.get_data_type(data_type).is_supported(sdfio.Revision.ISO_1_0)
    ],
)

sizes = {}
for revision in sdfio.Revision:
    for data_type in sdfio.DataType:
        if not sdfio.get_data_type(data_type).is_supported(revision):
            continue
        if data_type in (sdfio.DataType.UINT8, sdfio.DataType.UINT16, sdfio.DataType.UINT32):
            continue  # unsigned types cannot hold the negative heights
        for file_format in sdfio.FileFormat:
            metadata = sdfio.Metadata(revision=revision, data_type=data_type)
            buffer = io.BytesIO()
            sdfio.write(
                buffer,
                heights,
                x_scale=1e-6,
                y_scale=1e-6,
                z_scale="auto",
                format=file_format,
                metadata=metadata,
            )
            sizes[revision, data_type.name, file_format.name] = len(buffer.getvalue())
print("ISO-2.0 int16, binary:", sizes[sdfio.Revision.ISO_2_0, "INT16", "BINARY"], "bytes")
print("ISO-2.0 int16, ASCII:", sizes[sdfio.Revision.ISO_2_0, "INT16", "ASCII"], "bytes")

# The smallest scale that fits the data into a type.
print("scale for int16:", sdfio.suggest_z_scale(heights, sdfio.get_data_type(sdfio.DataType.INT16)))

# A file is converted to another revision or data type. The scale and the trailer have to suit it.
buffer = io.BytesIO()
sdfio.write(
    buffer,
    heights,
    x_scale=1e-6,
    y_scale=1e-6,
    z_scale="auto",
    metadata=sdfio.Metadata(revision=sdfio.Revision.ISO_2_0, data_type=sdfio.DataType.INT16),
)
sdf = sdfio.read(io.BytesIO(buffer.getvalue()))
older = sdf.with_revision(sdfio.Revision.ISO_1_0)
print("converted to", older.header.revision, "with", older.data_type.name)

# A data type of ISO-2.0 only is refused by ISO-1.0.
try:
    sdf.with_revision(sdfio.Revision.ISO_1_0, data_type=sdfio.DataType.INT8)
except sdfio.SdfFormatError as error:
    print("refused:", error)

# The representation is chosen when the file is saved.
ascii_bytes = older.dumps(format=sdfio.FileFormat.ASCII)
print("as ASCII text:", len(ascii_bytes), "bytes, begins", ascii_bytes[:8])
