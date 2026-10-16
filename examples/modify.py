"""Change a file, make files of parts of it, and convert it: python examples/modify.py."""

import io

import numpy as np

import sdfio

# ``SdfFile.from_array`` takes the arguments of ``write`` and gives the file without writing it.
heights = 1e-6 * (np.random.default_rng(5).normal(size=(200, 300)) + 5.0)
sdf = sdfio.SdfFile.from_array(heights, x_scale=1e-6, y_scale=1e-6)

# The data is the file's own NumPy array.
sdf.data -= sdf.data.mean()

# The part keeps the header and trailer, and shares the heights with sdf.
part = sdf.with_data(sdf.data[50:150, 100:250])

# Every second point, with the matching scale.
coarse = sdf.with_data(sdf.data[:, ::2], x_scale=2 * sdf.header.x_scale)

# A slice keeps the row axis, so a row is a profile.
row = sdf.with_data(sdf.data[40:41])
print("crop:", part.data.shape, "coarse:", coarse.data.shape, "row is a profile:", row.is_profile)

# A z scale that does not fit the new type is kept by default and the file can't be written;
# ``refit_z_scale`` keeps it when the data fits and otherwise takes the smallest that fits.
older = coarse.with_revision(
    sdfio.Revision.ISO_1_0, data_type=sdfio.DataType.INT32, refit_z_scale=True
)
print("converted to", older.header.revision, older.header.data_type.name)

for file_format in sdfio.FileFormat:
    buffer = io.BytesIO()
    older.save(buffer, format=file_format)
    print(f"{file_format.name.lower():>6}: {len(buffer.getvalue())} bytes")

# A data type that only ISO-2.0 has is refused by ISO-1.0.
try:
    coarse.with_revision(sdfio.Revision.ISO_1_0, data_type=sdfio.DataType.INT8)
except sdfio.SdfFormatError as error:
    print("refused:", error)
