# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The header and the trailer of a file."""

import io
from datetime import UTC, datetime

import numpy as np

import sdfio

# The metadata is a template for the header: the revision, the data type, who made the data, when
# and with which quantization step. Dates are in UTC for ISO-2.0.
metadata = sdfio.Metadata(
    revision=sdfio.Revision.ISO_2_0,
    manufacturer_id="Acme",
    create_date=datetime(2026, 3, 1, 9, 30, tzinfo=UTC),
    mod_date=datetime(2026, 3, 2, 14, 0, tzinfo=UTC),
    z_resolution=1e-9,
    data_type=sdfio.DataType.INT32,
)
heights = 1e-6 * np.random.default_rng(4).normal(size=(50, 80))

buffer = io.BytesIO()
sdfio.write(
    buffer,
    heights,
    x_scale=1e-6,
    y_scale=1e-6,
    z_scale=1e-12,
    metadata=metadata,
    trailer=sdfio.format_tagged_fields({"OperatorName": "Jane Doe", "PartName": "Gauge 7"}),
)
sdf = sdfio.read(io.BytesIO(buffer.getvalue()))

header = sdf.header
print(header.revision, header.manufacturer_id, header.create_date, "..", header.mod_date)
print("quantization step:", header.z_resolution, "m, z scale:", header.z_scale)

# The trailer holds free text that belongs to the measurement. As a list of Name = Value fields,
# which ISO-2.0 requires, it is a mapping.
print("trailer fields:", dict(sdf.trailer_fields))
sdf.trailer_fields = {**sdf.trailer_fields, "Shift": "2"}
print("changed:", dict(sdfio.read(io.BytesIO(sdf.dumps())).trailer_fields))

# ISO-2.0 refuses a trailer of another form, and ISO-1.0 takes any 7-bit ASCII text.
try:
    sdfio.write(
        io.BytesIO(), heights, x_scale=1e-6, y_scale=1e-6, metadata=metadata, trailer="free text"
    )
except sdfio.SdfFormatError as error:
    print("refused:", error)
older = sdfio.Metadata(revision=sdfio.Revision.ISO_1_0)
sdfio.write(io.BytesIO(), heights, x_scale=1e-6, y_scale=1e-6, metadata=older, trailer="free text")
print("ISO-1.0 accepts free text")
