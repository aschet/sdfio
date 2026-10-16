# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read a profile, and take one from a surface."""

import io

import numpy as np

import sdfio

# A profile is a surface with one row. The standard calls the sampling interval along y of a
# profile meaningless, and a file needs a value for it anyway.
x = np.arange(500) * 2e-6
profile = 1e-6 * (np.sin(2.0 * np.pi * x / 100e-6) + 0.2 * np.sin(2.0 * np.pi * x / 23e-6))

buffer = io.BytesIO()
sdfio.write(buffer, profile.reshape(1, -1), x_scale=2e-6, y_scale=2e-6)

sdf = sdfio.read(io.BytesIO(buffer.getvalue()))
print("rows and columns:", sdf.data.shape, "a profile:", sdf.is_profile)
print("x runs to", sdf.x_axis[-1] * 1e3, "mm, y is", sdf.y_axis.tolist())

# The points of a profile have one row, with y equal to 0.
points = sdf.points()
print("points:", points.shape, "first and last:", points[0, 0], points[0, -1])

# A row of a surface is a profile. Taking it as a slice keeps the row axis.
surface = 1e-6 * np.random.default_rng(1).normal(size=(100, 200))
row = surface[40:41]
buffer = io.BytesIO()
sdfio.write(buffer, row, x_scale=1e-6, y_scale=1e-6)
print("a row of the surface as a profile:", sdfio.read(io.BytesIO(buffer.getvalue())).data.shape)
