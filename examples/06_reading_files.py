# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Read files of unknown kind."""

import io
from pathlib import Path

import sdfio

# The sample files of the repository, of each revision and in both representations; use the paths
# of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples"
paths = [
    SAMPLES / "iso25178-71-2026" / "iso1_surface_ascii.sdf",
    SAMPLES / "iso25178-71-2026" / "iso2_surface_binary.sdf",
    SAMPLES / "iso25178-71-2026" / "iso2_profile_binary.sdf",
    SAMPLES / "bcr-1.0" / "bcr1_surface_uint16_ascii.sdf",
]

# ``read`` takes a path or an open file, and finds the revision and the representation by itself.
for path in paths:
    sdf = sdfio.read(path)
    kind = "profile" if sdf.header.num_profiles == 1 else "surface"
    print(
        f"{path.name}: {kind}, {sdf.header.revision}, "
        f"{sdf.header.format.name.lower()}, {sdf.data_type.name}, {sdf.data.shape}"
    )

with open(paths[1], "rb") as stream:
    print("from a stream:", sdfio.read(stream).data.shape)
print("from bytes:", sdfio.SdfFile.loads(paths[0].read_bytes()).data.shape)

# The beginning of a file says what it is: the first character is a for ASCII and b for binary.
print([magic.decode() for magic in sdfio.MAGICS][:3], "...")

# Anything else is an error.
try:
    sdfio.read(io.BytesIO(b"not an SDF file at all"))
except sdfio.SdfFormatError as error:
    print("not read:", error)
