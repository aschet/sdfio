# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Regression tests against real, third-party-exported sample files.

See tests/interop_samples/README.md for provenance and known quirks. Unlike
every other test in this suite, which exercises sdfio's reader against
sdfio's own writer, these catch a case where sdfio's interpretation of the
standard diverges from how a real tool actually produces files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sdfio
from sdfio.header import SdfDialect

SAMPLES_DIR = Path(__file__).parent / "interop_samples"
NUM_PROFILES, NUM_POINTS = 20, 30
MM11_FILES = sorted(SAMPLES_DIR.glob("mm11_*.sdf"))


def _expected_data() -> np.ndarray:
    data: np.ndarray = np.fromfunction(
        lambda i, j: (i * NUM_POINTS + j) * 1e-9, (NUM_PROFILES, NUM_POINTS)
    )
    data[0, NUM_POINTS - 1] = np.nan
    return data


def _expected_dialect(path: Path) -> SdfDialect:
    return SdfDialect.ISO_1_0 if "_iso1_" in path.name else SdfDialect.ISO_2_0


@pytest.mark.parametrize("path", MM11_FILES, ids=lambda p: p.name)
def test_mm11_export_matches_input_data(path: Path) -> None:
    """Check the re-export is still the same asymmetric ramp and landmark NaN sdfio wrote.

    Regardless of dialect, format, or data type -- catching an orientation
    or transform bug a sdfio-generated-only fixture structurally can't.
    """
    sdf = sdfio.read(path)

    assert sdf.data.shape == (NUM_PROFILES, NUM_POINTS)
    expected = _expected_data()
    assert np.array_equal(np.isnan(sdf.data), np.isnan(expected))
    # atol, not rtol: values span down to 0, and the known ASCII binary64
    # precision limit (15 significant digits vs. the 17 a double needs) is
    # an absolute few femto-units here, not a fraction of each value.
    np.testing.assert_allclose(sdf.data, expected, atol=1e-13, equal_nan=True)
    assert sdf.header.dialect == _expected_dialect(path)
    assert not sdf.trailer  # this exporter doesn't write one, see the README


@pytest.mark.parametrize(
    "path", [p for p in MM11_FILES if "_binary_" in p.name], ids=lambda p: p.name
)
def test_mm11_binary_export_manufacturer_id_is_recovered_cleanly(path: Path) -> None:
    """Check sdfio recovers ManufacID cleanly from a real file, not just the synthetic case.

    This exporter NUL-terminates ManufacID instead of space-padding it,
    leaving trailing garbage -- see tests/interop_samples/README.md and
    test_binary.py's test_load_truncates_manufacturer_id_at_first_nul.
    """
    assert sdfio.read(path).header.manufacturer_id == "test"


def test_input_sample_matches_expected_data() -> None:
    """Sanity-check the input file itself, so a future regenerate can't silently drift."""
    sdf = sdfio.read(SAMPLES_DIR / "input_iso2_ascii.sdf")
    expected = _expected_data()
    assert np.array_equal(np.isnan(sdf.data), np.isnan(expected))
    np.testing.assert_allclose(sdf.data, expected, atol=1e-13, equal_nan=True)
