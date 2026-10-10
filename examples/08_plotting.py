# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Plot the data of SDF files with matplotlib: python examples/08_plotting.py [figure.png]."""

import io
import sys
from pathlib import Path

import sdfio

# The sample files of the repository; use the path of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples" / "iso25178-71-2026"
surface = sdfio.read(SAMPLES / "iso2_surface_binary.sdf")
profile = sdfio.read(SAMPLES / "iso2_profile_binary.sdf")

# The heights are a NumPy array in metres with one row per profile, and the axes give the
# coordinates of the columns and the rows. Points that were not measured are NaN.
print("size:", surface.data.shape, "step:", surface.header.x_scale, surface.header.y_scale)

# ``points()`` joins the axes and the heights into an array of shape (rows, columns, 3).
points = surface.points()

# matplotlib is not a dependency of sdfio; the plots are made if it is installed. The values are
# in metres, so they are scaled for the labels. NaN stays blank.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed, no plot")
else:
    figure = plt.figure(figsize=(14, 4.5), layout="constrained")

    # A surface as an image. ``y`` grows with the row, so the origin is in the lower left corner.
    x, y = surface.x_axis * 1e6, surface.y_axis * 1e6
    step_x, step_y = surface.header.x_scale * 1e6, surface.header.y_scale * 1e6
    extent = (x[0] - step_x / 2, x[-1] + step_x / 2, y[0] - step_y / 2, y[-1] + step_y / 2)
    axes = figure.add_subplot(1, 3, 1)
    image = axes.imshow(surface.data * 1e6, origin="lower", extent=extent)
    figure.colorbar(image, ax=axes, label="z / um")
    axes.set(xlabel="x / um", ylabel="y / um", title="Surface: Height Map")

    # The same surface in 3-D from the points; the missing region is a hole.
    scaled = points * 1e6
    axes = figure.add_subplot(1, 3, 2, projection="3d")
    axes.plot_surface(
        scaled[..., 0], scaled[..., 1], scaled[..., 2], rstride=1, cstride=1, cmap="viridis"
    )
    axes.set(xlabel="x / um", ylabel="y / um", title="Surface: 3-D View")
    axes.set_zlabel("z / um", labelpad=8)
    axes.set_box_aspect((3, 2, 1))  # the heights are smaller than the lateral size

    # A profile is a single row; the missing points break the line.
    axes = figure.add_subplot(1, 3, 3)
    axes.plot(profile.x_axis * 1e6, profile.data[0] * 1e6)
    axes.set(xlabel="x / um", ylabel="z / um", title="Profile")

    if len(sys.argv) > 1:
        figure.savefig(sys.argv[1])
        print("figure written:", sys.argv[1])
    else:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        print("plots written:", len(buffer.getvalue()) > 0)
    plt.close(figure)
