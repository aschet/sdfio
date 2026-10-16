# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Change the values of a file, and make files of parts of it."""

import dataclasses
import io

import numpy as np

import sdfio

heights = 1e-6 * (np.random.default_rng(5).normal(size=(200, 300)) + 5.0)
buffer = io.BytesIO()
sdfio.write(buffer, heights, x_scale=1e-6, y_scale=1e-6)
sdf = sdfio.read(io.BytesIO(buffer.getvalue()))

# The data is a NumPy array that belongs to the file, so changing it changes the file.
print("mean height before:", float(sdf.data.mean()))
sdf.data -= sdf.data.mean()
print(f"mean height after: {sdf.data.mean():.1e}")

# A crop is a new file: the header of the new size, with the same scales.
crop = sdf.data[50:150, 100:250].copy()
header = dataclasses.replace(sdf.header, num_profiles=crop.shape[0], num_points=crop.shape[1])
part = sdfio.SdfFile(header=header, data=crop, trailer=sdf.trailer)
print("crop:", part.data.shape, "x runs to", part.x_axis[-1])

# Every second point, with the scale that goes with it.
header = dataclasses.replace(
    sdf.header,
    num_points=sdf.data[:, ::2].shape[1],
    x_scale=2 * sdf.header.x_scale,
)
coarse = sdfio.SdfFile(header=header, data=sdf.data[:, ::2].copy(), trailer=sdf.trailer)
print("coarse:", coarse.data.shape, "step:", coarse.header.x_scale)

# The data type is changed with ``with_revision``, which is given the revision of the file itself.
# The scale has to fit the new type.
retyped = coarse.with_revision(coarse.header.revision, data_type=sdfio.DataType.INT32)
retyped.header.z_scale = sdfio.suggest_z_scale(
    coarse.data, sdfio.get_data_type(sdfio.DataType.INT32)
)
print("saved as", retyped.data_type.name, len(retyped.dumps()), "bytes")
