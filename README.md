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

Supported dialects: ISO-1.0, ISO-2.0, and BCR-1.0 (the
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

Non-measured or spurious points (the `BAD` marker in ASCII files, or the
data type's reserved sentinel value in binary files) are represented as
`NaN` in `sdf.data`, and `sdfio.write` writes `NaN` values back the same way.

## Command Line

```bash
sdfio info surface.sdf
sdfio convert -f ascii surface.sdf surface_ascii.sdf
sdfio convert -d ISO-1.0 -r surface_v2.sdf surface_v1.sdf
sdfio convert -t int16 surface.sdf surface_int16.sdf
```

`convert` can change the ASCII/binary format (`-f`/`--format`), the SDF
dialect and version (`-d`/`--dialect`, e.g. `ISO-2.0` or `BCR-1.0`), and the
data area storage type (`-t`/`--type`) independently; any not given are kept
from the source file. Converting to a dialect whose trailer requirements the
source file doesn't meet fails unless `-r`/`--remove-trailer` is passed to
discard it. Run `sdfio convert --help` for details. The CLI is also runnable
as `python -m sdfio`.

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

- ISO 25178-71:2026, *Geometrical product specifications (GPS) — Surface
  texture: Areal — Part 71: SDF file format*
- K. J. Stout et al., *The Development of Methods for the Characterisation of
  Roughness in Three Dimensions*, EUR 15178 EN, EC Brussels, 1993
