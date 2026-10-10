# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Points that were not measured."""

import io

import numpy as np

import sdfio

# NaN marks a point that was not measured, or is spurious. The file has a value that no height
# can have in its place: the minimum of the data type in ISO, the maximum in BCR, and in the ASCII
# representation of ISO the text BAD.
heights = 1e-6 * np.random.default_rng(2).normal(size=(40, 60))
heights[10:15, 20:30] = np.nan

for dialect in (sdfio.SdfDialect.ISO_2_0, sdfio.SdfDialect.BCR_1_0):
    metadata = sdfio.SdfMetadata(dialect=dialect, data_type=sdfio.DataType.INT16)
    buffer = io.BytesIO()
    sdfio.write(buffer, heights, x_scale=1e-6, y_scale=1e-6, z_scale="auto", metadata=metadata)
    sdf = sdfio.read(io.BytesIO(buffer.getvalue()))
    raw = sdf.raw_data()
    bad = np.unique(raw[np.isnan(sdf.data)])
    print(dialect, "stores the bad points as", bad.tolist(), "and reads them as NaN again")
    print("  missing points:", int(np.isnan(sdf.data).sum()))

# In the ASCII representation of ISO the bad points are written as BAD.
buffer = io.BytesIO()
sdfio.write(buffer, heights, x_scale=1e-6, y_scale=1e-6, format=sdfio.FileFormat.ASCII)
print("BAD in the text:", buffer.getvalue().count(b"BAD"))

# The measured points, as the list of points that other libraries take.
measured = sdf.points(drop_invalid=True)
print("measured points:", len(measured), "of", sdf.data.size)
