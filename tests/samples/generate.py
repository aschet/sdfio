# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Make the sample files of this folder: ``python tests/samples/generate.py``.

Profiles and surfaces are made in each dialect of the format, in the ASCII and in the binary
representation, with the data types that the dialect defines. The data is synthetic and does not
depend on the version of sdfio, so the files only change when this script does.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import sdfio
from sdfio import DataType, FileFormat, SdfDialect, SdfMetadata

FOLDER = Path(__file__).parent
#: The folder of each dialect, named after the document that defines it: ISO 25178-71:2026
#: specifies the versions 1.0 and 2.0.
FOLDERS = {
    SdfDialect.ISO_1_0: FOLDER / "iso25178-71-2026",
    SdfDialect.ISO_2_0: FOLDER / "iso25178-71-2026",
    SdfDialect.BCR_1_0: FOLDER / "bcr-1.0",
}
#: The start of the name of the files of each dialect, as the folder holds several.
PREFIXES = {SdfDialect.ISO_1_0: "iso1", SdfDialect.ISO_2_0: "iso2", SdfDialect.BCR_1_0: "bcr1"}
FORMATS = {"ascii": FileFormat.ASCII, "binary": FileFormat.BINARY}
#: A time that does not depend on when the files are made: UTC for the dialect that requires it,
#: and a time without a zone, which is written as it is, for the others.
_DATE = datetime(2026, 1, 1, 12, 0)

X_SCALE = 2e-6
Y_SCALE = 3e-6
#: The scale factor of the coded heights for each data type: the quantization step of an integer
#: type, which is chosen so that the heights of the samples fit, and micrometres for the others.
Z_SCALES = {
    DataType.UINT8: 2e-8,
    DataType.UINT16: 1e-10,
    DataType.UINT32: 1e-13,
    DataType.BINARY32: 1e-6,
    DataType.INT8: 2e-8,
    DataType.INT16: 1e-10,
    DataType.INT32: 1e-13,
    DataType.BINARY64: 1e-6,
}
#: The size of the surfaces, as rows and columns, and the number of points of the profiles.
ROWS, COLUMNS = 32, 48
POINTS = 128


@dataclass(frozen=True)
class Sample:
    """The content of a sample file, without its representation."""

    description: str
    data: np.ndarray
    data_type: DataType
    trailer: str = ""


def _surface() -> np.ndarray:
    """Return a smooth, bumpy surface in metres, shaped ``(ROWS, COLUMNS)``."""
    x = np.arange(COLUMNS) * X_SCALE
    y = (np.arange(ROWS) * Y_SCALE)[:, np.newaxis]
    surface: np.ndarray = 1e-6 * (
        np.cos(2.0 * np.pi * x / 40e-6) * np.cos(2.0 * np.pi * y / 36e-6)
        + 0.3 * np.sin(2.0 * np.pi * x / 14e-6)
    )
    return surface


def _profile() -> np.ndarray:
    """Return a profile in metres, shaped ``(1, POINTS)``."""
    x = np.arange(POINTS) * X_SCALE
    return 1e-6 * (np.sin(2.0 * np.pi * x / 80e-6) + 0.2 * np.sin(2.0 * np.pi * x / 17e-6))[None]


def _with_missing_points(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    if data.shape[0] == 1:
        data[0, 60:64] = np.nan
    else:
        data[10:14, 20:30] = np.nan
    return data


def trailer() -> str:
    """Return the trailer of the sample that has one, as ``Name = Value`` fields in all dialects."""
    return sdfio.format_tagged_fields({"OperatorName": "Jane Doe", "PartName": "Synthetic bumps"})


def build(dialect: SdfDialect) -> dict[str, Sample]:
    """Return the samples of a dialect by name, without the suffix of the representation."""
    surface = _surface()
    missing = _with_missing_points(surface)
    samples = {
        "profile": Sample(
            "A profile of 128 points with missing points, binary64",
            _with_missing_points(_profile()),
            DataType.BINARY64,
        ),
        "surface": Sample(
            "A surface of 48 by 32 points with a region of missing points and a trailer, int16",
            missing,
            DataType.INT16,
            trailer(),
        ),
        "surface_int32": Sample("The same surface, int32", missing, DataType.INT32),
        "surface_binary64": Sample("The same surface, binary64", missing, DataType.BINARY64),
    }
    if dialect is SdfDialect.ISO_2_0:
        # The data types that version 2.0 adds.
        samples["surface_binary32"] = Sample(
            "The same surface, binary32", missing, DataType.BINARY32
        )
        samples["surface_int8"] = Sample("The same surface, int8", missing, DataType.INT8)
    if dialect is SdfDialect.BCR_1_0:
        # An unsigned type cannot hold negative heights, so the surface lies above zero.
        samples["surface_uint16"] = Sample(
            "The same surface above zero, uint16",
            _with_missing_points(surface - np.nanmin(surface)),
            DataType.UINT16,
        )
    return samples


def dumps(dialect: SdfDialect, sample: Sample, file_format: FileFormat) -> bytes:
    """Return the content of a sample file in a dialect and representation."""
    created = _DATE.replace(tzinfo=UTC) if dialect.requires_utc else _DATE
    z_scale = Z_SCALES[sample.data_type]
    is_integer = sample.data_type not in (DataType.BINARY32, DataType.BINARY64)
    buffer = io.BytesIO()
    sdfio.write(
        buffer,
        sample.data,
        x_scale=X_SCALE,
        y_scale=Y_SCALE,
        z_scale=z_scale,
        format=file_format,
        metadata=SdfMetadata(
            dialect=dialect,
            manufacturer_id="sdfio",
            create_date=created,
            mod_date=created,
            z_resolution=z_scale if is_integer else -1.0,
            data_type=sample.data_type,
        ),
        trailer=sample.trailer,
    )
    return buffer.getvalue()


def path(dialect: SdfDialect, name: str, suffix: str) -> Path:
    """Return the path of a sample file, by dialect, name and suffix of the representation."""
    return FOLDERS[dialect] / f"{PREFIXES[dialect]}_{name}_{suffix}.sdf"


def main() -> None:
    """Write all sample files, replacing the ones that exist."""
    for folder in set(FOLDERS.values()):
        folder.mkdir(exist_ok=True)
        for old in folder.glob("*.sdf"):
            old.unlink()
    for dialect in FOLDERS:
        for name, sample in build(dialect).items():
            for suffix, file_format in FORMATS.items():
                path(dialect, name, suffix).write_bytes(dumps(dialect, sample, file_format))


if __name__ == "__main__":
    main()
