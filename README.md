# sdfio

[![CI](https://github.com/aschet/sdfio/actions/workflows/ci.yml/badge.svg)](https://github.com/aschet/sdfio/actions/workflows/ci.yml)
[![Docs](https://github.com/aschet/sdfio/actions/workflows/docs.yml/badge.svg)](https://aschet.github.io/sdfio/)
[![PyPI](https://img.shields.io/pypi/v/sdfio.svg)](https://pypi.org/project/sdfio/)

<!-- docs-include-start -->

Read and write ISO 25178-71 Surface Data Files (`.sdf`) in Python.

SDF stores areal (or profile) surface topography measurements as a
rectangular grid of height values, in either a human-readable ASCII or a
compact binary format. It is used as a software measurement standard
(a "softgauge") to verify the software of surface texture measuring
instruments, and is also used more generally as a surface topography
interchange format.

Supported revisions: ISO-1.0, ISO-2.0, and BCR-1.0 (the
pre-standardization proposal this format is based on), without
compression or checksummed data areas. Vendor-specific extensions are not
supported.

## Installation

```bash
pip install sdfio
```

## Usage

```python
import numpy as np
import sdfio

data = np.zeros((480, 640))

sdfio.write("surface.sdf", data, x_scale=1e-6, y_scale=1e-6, z_scale=1e-9)

sdf = sdfio.read("surface.sdf")
print(sdf.data.shape)
print(sdf.header)
```

`sdf.data` is a `(num_profiles, num_points)` array of height values in
metres, in the standard's right-handed coordinate system: y increases with
row index and x with column index (`sdfio.SdfFile.x_axis`/`y_axis` return
the coordinates, also in metres).

`sdf.points()` combines the coordinates and the heights in a
`(num_profiles, num_points, 3)` array of x, y and z in metres.
`sdf.points(drop_invalid=True)` leaves out the points that are not measured and
returns the `(N, 3)` array of points that point cloud libraries take.

`sdfio.SdfFile.from_array` takes the arguments of `sdfio.write` and returns the file in memory
without writing it, to change it first or to `save()` it later.

`sdf.with_data(heights)` returns a copy with other heights, such as a crop, and
`sdf.with_revision(revision, data_type=..., refit_z_scale=True)` a copy in another revision. The
number of points and profiles of a file are the shape of `sdf.data`. The copies share the heights
with the original, so a crop is a view; use `.copy()` on the heights to edit them independently.
`examples/modify.py` shows both.

The header checks what it is given: the data type has to exist in the revision, the Z-scale has to
be positive and the manufacturer ID 7-bit ASCII. Reading is more tolerant, see below.

Non-measured or spurious points (the `BAD` marker in ASCII files, or the
data type's reserved sentinel value in binary files) are represented as
`NaN` in `sdf.data`, and `sdfio.write` writes `NaN` values back the same way.

The trailer (record 3) holds free-form historical information, such as an
operator name; ISO-2.0 requires it to use the same tagged `Name = Value`
format as the header, which other revisions don't. `sdf.trailer_fields` reads
and writes it as a dict in that format either way:

```python
sdf = sdfio.read("surface.sdf")
sdf.trailer_fields = {"OperatorName": "Jane Doe"}
print(sdf.trailer_fields)
```

## Non-Compliant Files

Reading tolerates known real-world deviations from the standard, and behaviour that the
standard does not define:

- A `datetime` field that is all zeros, blank or not a valid date parses as `None` ("not
  recorded"); the rest of the file is read.
- A binary `ManufacID` that's NUL-terminated instead of space-padded is
  truncated at the NUL byte (allowed by BCR-1.0).
- Bytes that are not 7-bit ASCII in `ManufacID` are decoded with replacement characters, not
  rejected.
- ASCII records ending in LF or CR instead of CRLF are read normally (allowed
  by BCR-1.0).
- A UTF-8 byte order mark in front of the magic is ignored.
- An upper case `A` or `B` as the first magic character is accepted (allowed
  by BCR-1.0).
- The marker `BAD` for a point that is not measured is accepted in BCR-1.0 ASCII files, which
  otherwise use the maximum value of the data type.
- In ASCII files the text `NaN` is read as the marker `BAD`, for every data type. In binary
  files a `binary32`/`binary64` value written as `NaN` instead of the sentinel still decodes as a
  non-measured point.
- A trailer isn't validated on read: one that isn't 7-bit ASCII, or isn't in the tagged form
  that ISO-2.0 requires, is returned as-is, not rejected.
- `Xscale`, `Yscale` and `Zscale` are read as they are, also when they are zero, negative or not
  finite; the heights are then as meaningless as the scale, but never an error or a warning.
  Writing rejects such a `Zscale`.
- A file with no columns or no rows is read as empty data.

## Documentation

- [SDF Format](https://aschet.github.io/sdfio/format.html): what an SDF file contains.
- [Examples](https://aschet.github.io/sdfio/examples.html): a short script for each topic, in
  the `examples/` folder.

## Command Line

```bash
sdfio info surface.sdf
sdfio convert -f ascii surface.sdf surface_ascii.sdf
sdfio convert -d ISO-2.0 -x surface_v1.sdf surface_v2.sdf
sdfio convert -t int16 surface.sdf surface_int16.sdf
```

`convert` can change the ASCII/binary format (`-f`/`--format`), the SDF
revision (`-d`/`--revision`, e.g. `ISO-2.0` or `BCR-1.0`), and the
data area storage type (`-t`/`--type`) independently; any not given are kept
from the source file. Converting to a revision whose trailer requirements the
source file doesn't meet fails unless `-x`/`--fix-trailer` is passed to clean
it up instead. Run `sdfio convert --help` for details. The CLI is also
runnable as `python -m sdfio`.

## Development

```bash
python3 -m venv .venv  # Windows: python -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e . --group dev
pre-commit install
pytest
pre-commit run --all-files
pip-audit
```

## References

- ISO 25178-71:2026, [*Geometrical product specifications (GPS) — Surface
  texture: Areal — Part 71: Surface data file (SDF) file
  format*](https://www.iso.org/standard/88787.html)
- K. J. Stout et al., [*The Development of Methods for the Characterisation of
  Roughness in Three Dimensions*](https://op.europa.eu/publication-detail/-/publication/f912e210-d0cc-4016-8726-690a576ed3e2),
  EUR 15178 EN, EC Brussels, 1993
