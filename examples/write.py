"""Write surfaces and profiles: python examples/write.py [folder]."""

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import sdfio

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
folder.mkdir(parents=True, exist_ok=True)

# Heights and intervals are in metres; NaN marks a point that was not measured.
surface = 1e-6 * np.random.default_rng(0).normal(size=(480, 640))
surface[100:110, 200:210] = np.nan

# A profile is a surface with one row; its y interval is meaningless but still needs a value.
x = np.arange(500) * 2e-6
profile = 1e-6 * np.sin(2.0 * np.pi * x / 100e-6)

# Defaults: ISO-2.0, binary, 64-bit floats.
sdfio.write(folder / "surface.sdf", surface, x_scale=1e-6, y_scale=1e-6)
sdfio.write(folder / "profile.sdf", profile.reshape(1, -1), x_scale=2e-6, y_scale=2e-6)

# The metadata is a template for the header. Dates are UTC. ISO-2.0 wants the trailer as
# Name = Value fields; the older revisions take any 7-bit text.
metadata = sdfio.Metadata(
    revision=sdfio.Revision.ISO_2_0,
    manufacturer_id="Acme",
    create_date=datetime(2026, 3, 1, 9, 30, tzinfo=UTC),
    z_resolution=1e-9,
    data_type=sdfio.DataType.INT16,
)
trailer = sdfio.format_tagged_fields({"OperatorName": "Jane Doe", "PartName": "Gauge 7"})

# Integer types store multiples of z_scale; "auto" takes the smallest scale that fits the data.
sdfio.write(
    folder / "surface_int16.sdf",
    surface,
    x_scale=1e-6,
    y_scale=1e-6,
    z_scale="auto",
    metadata=metadata,
    trailer=trailer,
)

# Other revisions, data types and representations; each revision allows only some data types.
variants = [
    ("surface_iso1_ascii", sdfio.Revision.ISO_1_0, sdfio.DataType.INT32, sdfio.FileFormat.ASCII),
    ("surface_bcr_int16", sdfio.Revision.BCR_1_0, sdfio.DataType.INT16, sdfio.FileFormat.BINARY),
    ("surface_float32", sdfio.Revision.ISO_2_0, sdfio.DataType.BINARY32, sdfio.FileFormat.BINARY),
]
for name, revision, data_type, file_format in variants:
    sdfio.write(
        folder / f"{name}.sdf",
        surface,
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale="auto",
        format=file_format,
        metadata=sdfio.Metadata(revision=revision, data_type=data_type),
        trailer=trailer,
    )

print("written to", folder)
for path in sorted(folder.glob("*.sdf")):
    print(f"{path.name:22s}{path.stat().st_size:>10} bytes")
