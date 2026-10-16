"""Read a file and look at its heights, header and trailer: python examples/read.py."""

import io
from pathlib import Path

import sdfio

# The sample files of the repository; use the paths of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples" / "iso25178-71-2026"
path = SAMPLES / "iso2_surface_binary.sdf"

sdf = sdfio.read(path)
print(sdf.header.revision, sdf.header.format.name.lower(), sdf.data_type.name, sdf.data.shape)

# Heights and axes are in metres; points that were not measured are NaN.
print("x runs to", sdf.x_axis[-1], "m, y to", sdf.y_axis[-1], "m")

# Joins the axes and the heights; without the missing points it is an (N, 3) point cloud.
points = sdf.points(drop_invalid=True)
print("measured points:", len(points), "of", sdf.data.size)

header = sdf.header
print(header.manufacturer_id, header.create_date, "z scale:", header.z_scale)

# ``trailer`` is the raw text; ``trailer_fields`` parses Name = Value lines, as ISO-2.0 requires.
print("trailer:", dict(sdf.trailer_fields))

profile = sdfio.read(SAMPLES / "iso2_profile_binary.sdf")
print("profile:", profile.is_profile, profile.data.shape)

# Streams and bytes work as well as paths.
with open(path, "rb") as stream:
    from_stream = sdfio.read(stream)
from_bytes = sdfio.SdfFile.loads(path.read_bytes())
print("same data from a stream and from bytes:", from_stream == from_bytes == sdf)

# All errors of sdfio are SdfError.
try:
    sdfio.read(io.BytesIO(b"not an SDF file at all"))
except sdfio.SdfError as error:
    print("not read:", error)
