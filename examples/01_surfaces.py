# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read a surface."""

import tempfile
from pathlib import Path

import numpy as np

import sdfio

# A surface is an array of heights in metres: the rows are along y, the columns along x, and
# NaN marks a point that was not measured. The sampling intervals are given in metres as well.
heights = 1e-6 * np.random.default_rng(0).normal(size=(480, 640))
heights[100:110, 200:210] = np.nan

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "surface.sdf"
    sdfio.write(path, heights, x_scale=1e-6, y_scale=1e-6)

    sdf = sdfio.read(path)
    print("rows and columns:", sdf.data.shape, "measured points:", int(np.isfinite(sdf.data).sum()))
    print("the file has", path.stat().st_size, "bytes")

# y grows with the row and x with the column; the axes are in metres.
width, height = sdf.x_axis[-1] * 1e3, sdf.y_axis[-1] * 1e3
print(f"the surface is {width:.3f} mm wide and {height:.3f} mm high")

# Without a template a file is written as ISO-2.0 in the binary representation, with the heights
# as 64-bit floating point numbers. The header holds what describes the file.
print(sdf.header.dialect, "binary:", sdf.header.binary, sdf.data_type.name)
print("sampling intervals:", sdf.header.x_scale, sdf.header.y_scale)

# The points of the grid with their coordinates, as an array of x, y and z.
points = sdf.points()
print("points:", points.shape, "the point at row 2, column 3:", points[2, 3])
