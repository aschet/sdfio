# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import sdfio
from sdfio.datatypes import DataType
from sdfio.exceptions import SdfFormatError
from sdfio.file import FileFormat
from sdfio.header import SdfDialect


def _sample_data() -> np.ndarray:
    return np.array(
        [
            [1.0e-6, 2.0e-6, np.nan],
            [-1.0e-6, 0.0, 3.5e-6],
        ]
    )


def test_write_read_roundtrip_via_path(tmp_path: Path) -> None:
    path = tmp_path / "surface.sdf"
    data = _sample_data()
    sdfio.write(path, data, x_scale=1e-6, y_scale=1e-6)

    sdf = sdfio.read(path)

    np.testing.assert_allclose(sdf.data, data, rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("file_format", [FileFormat.BINARY, FileFormat.ASCII])
@pytest.mark.parametrize("dialect", [SdfDialect.ISO_1_0, SdfDialect.ISO_2_0])
def test_write_read_roundtrip(file_format: FileFormat, dialect: SdfDialect) -> None:
    buffer = io.BytesIO()
    data = _sample_data()
    sdfio.write(
        buffer,
        data,
        x_scale=1e-6,
        y_scale=2e-6,
        z_scale=1e-7,
        format=file_format,
        metadata=sdfio.SdfMetadata(dialect=dialect, data_type=DataType.BINARY64),
    )
    buffer.seek(0)

    sdf = sdfio.read(buffer)

    assert sdf.header.dialect == dialect
    assert sdf.header.binary == (file_format == FileFormat.BINARY)
    assert sdf.header.num_points == 3
    assert sdf.header.num_profiles == 2
    np.testing.assert_allclose(sdf.data, data, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(sdf.x_axis, np.arange(3) * 1e-6)
    np.testing.assert_allclose(sdf.y_axis, np.arange(2) * 2e-6)
    create_date = sdf.header.create_date
    mod_date = sdf.header.mod_date
    assert create_date is not None
    assert mod_date is not None
    if dialect == SdfDialect.ISO_2_0:
        assert create_date.tzinfo == UTC
        assert mod_date.tzinfo == UTC
    else:
        # The standard only specifies UTC for version 2.0, so version 1.0 is
        # treated as local time; a freshly-read date is tagged with the
        # system timezone, not naive.
        assert create_date.tzinfo is not None
        assert mod_date.tzinfo is not None


def test_write_defaults_to_binary_iso_2_0() -> None:
    buffer = io.BytesIO()
    sdfio.write(buffer, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6)
    buffer.seek(0)
    sdf = sdfio.read(buffer)
    assert sdf.header.binary is True
    assert sdf.header.dialect == SdfDialect.ISO_2_0


@pytest.mark.parametrize("file_format", [FileFormat.BINARY, FileFormat.ASCII])
@pytest.mark.parametrize("dialect", [SdfDialect.ISO_1_0, SdfDialect.ISO_2_0])
def test_trailer_roundtrip(file_format: FileFormat, dialect: SdfDialect) -> None:
    # Neither dialect imposes a format requirement on the trailer's content,
    # so arbitrary ASCII text round-trips as-is.
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=dialect, binary=file_format == FileFormat.BINARY, num_points=1, num_profiles=1
        ),
        data=np.zeros((1, 1)),
        trailer="OperatorName = WG 16",
    )

    reloaded = sdfio.SdfFile.loads(sdf.dumps())

    expected = (
        b"OperatorName = WG 16" if file_format == FileFormat.BINARY else "OperatorName = WG 16"
    )
    assert reloaded.trailer == expected


@pytest.mark.parametrize("binary", [True, False])
def test_dumps_rejects_non_ascii_trailer(binary: bool) -> None:
    """Record 3 (the trailer) is defined as a sequence of ASCII values, for both formats."""
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(binary=binary, num_points=1, num_profiles=1),
        data=np.zeros((1, 1)),
        trailer="Operator = Müller café",
    )

    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        sdf.dumps()


@pytest.mark.parametrize("binary", [True, False])
def test_loads_reads_data_despite_corrupt_trailer(binary: bool) -> None:
    """A corrupt trailer must not prevent reading the (already-decoded) depth data."""
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(binary=binary, num_points=2, num_profiles=2),
        data=np.zeros((2, 2)),
        trailer=b"Operator = X" if binary else "Operator = X",
    )
    raw = bytearray(sdf.dumps())
    raw[-1] = 0xE9  # corrupt the last trailer byte to a non-ASCII value

    reloaded = sdfio.SdfFile.loads(bytes(raw))

    np.testing.assert_array_equal(reloaded.data, np.zeros((2, 2)))


def test_trailer_fields_roundtrip() -> None:
    sdf = sdfio.SdfFile(header=sdfio.SdfHeader(num_points=1, num_profiles=1), data=np.zeros((1, 1)))

    sdf.trailer_fields = {"OperatorName": "WG 16", "PartName": "Example"}

    assert sdf.trailer == "OperatorName = WG 16\r\nPartName = Example\r\n"
    reloaded = sdfio.SdfFile.loads(sdf.dumps())
    assert reloaded.trailer_fields == {"OperatorName": "WG 16", "PartName": "Example"}


def test_trailer_fields_getter_returns_read_only_view() -> None:
    """Mutating the returned mapping in place must not silently vanish; it must fail loudly."""
    sdf = sdfio.SdfFile(header=sdfio.SdfHeader(num_points=1, num_profiles=1), data=np.zeros((1, 1)))
    sdf.trailer_fields = {"OperatorName": "WG 16"}

    with pytest.raises(TypeError):
        sdf.trailer_fields["PartName"] = "Example"

    assert sdf.trailer_fields == {"OperatorName": "WG 16"}


def test_trailer_fields_getter_drops_malformed_lines_instead_of_raising() -> None:
    """Inspecting the trailer must not fail just because it's malformed."""
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(num_points=1, num_profiles=1),
        data=np.zeros((1, 1)),
        trailer="OperatorName = WG 16\r\nnot tagged fields\r\nPartName = X",
    )

    assert sdf.trailer_fields == {"OperatorName": "WG 16", "PartName": "X"}


def test_trailer_fields_getter_stays_lenient_on_non_ascii_bytes() -> None:
    """Inspecting an already non-ASCII trailer (e.g. from a non-compliant file) must not raise."""
    trailer = "Operator = Müller café".encode()
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(num_points=1, num_profiles=1), data=np.zeros((1, 1)), trailer=trailer
    )

    assert sdf.trailer_fields == {"Operator": "Müller café"}


def test_write_auto_z_scale_maximizes_resolution() -> None:
    buffer = io.BytesIO()
    data = np.array([[2.0e-3, -1.0e-3], [0.5e-3, np.nan]])
    sdfio.write(
        buffer,
        data,
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale="auto",
        metadata=sdfio.SdfMetadata(data_type=DataType.INT16),
    )
    buffer.seek(0)

    sdf = sdfio.read(buffer)

    assert sdf.header.z_scale != 1e-6  # picked to fit int16, not the float default
    np.testing.assert_allclose(sdf.data, data, rtol=1e-4, equal_nan=True)
    assert np.abs(sdf.raw_data()[np.isfinite(data)]).max() > 30000  # uses most of int16's range


@pytest.mark.parametrize("file_format", [FileFormat.BINARY, FileFormat.ASCII])
@pytest.mark.parametrize("data_type", list(DataType))
def test_write_read_roundtrip_all_data_types(file_format: FileFormat, data_type: DataType) -> None:
    data = np.array([[1.0, np.nan, -2.0], [0.0, 3.0, 4.0]])
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            binary=file_format == FileFormat.BINARY,
            num_points=3,
            num_profiles=2,
            x_scale=1e-6,
            y_scale=1e-6,
            z_scale=1.0,
            data_type=data_type,
        ),
        data=data,
    )

    reloaded = sdfio.SdfFile.loads(sdf.dumps())

    assert reloaded.header.data_type == data_type
    np.testing.assert_allclose(reloaded.data, data, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_raw_data_exposes_native_dtype() -> None:
    data = np.array([[1.0e-6, np.nan], [-2.0e-6, 0.0]])
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            num_points=2,
            num_profiles=2,
            x_scale=1e-6,
            y_scale=1e-6,
            z_scale=1e-6,
            data_type=DataType.INT16,
        ),
        data=data,
    )

    assert sdf.raw_data().dtype == np.dtype("<i2")
    np.testing.assert_array_equal(sdf.raw_data(), [[1, -32768], [-2, 0]])


def test_modifying_data_and_saving_retains_original_data_type() -> None:
    data = np.array([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6]])
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            num_points=2,
            num_profiles=2,
            x_scale=1e-6,
            y_scale=1e-6,
            z_scale=1e-6,
            data_type=DataType.INT16,
        ),
        data=data,
    )

    sdf.data = sdf.data * 2.0
    reloaded = sdfio.SdfFile.loads(sdf.dumps())

    assert reloaded.header.data_type == sdf.header.data_type == DataType.INT16
    np.testing.assert_allclose(reloaded.data, data * 2.0, rtol=1e-9)


def test_changing_data_type_after_read_reencodes_on_save() -> None:
    data = np.array([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6]])
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            num_points=2,
            num_profiles=2,
            x_scale=1e-6,
            y_scale=1e-6,
            z_scale=1e-6,
            data_type=DataType.INT16,
        ),
        data=data,
    )

    sdf.data_type = DataType.BINARY64
    reloaded = sdfio.SdfFile.loads(sdf.dumps())

    assert reloaded.header.data_type == DataType.BINARY64
    np.testing.assert_allclose(reloaded.data, data, rtol=1e-9)


def test_data_type_roundtrips_and_rejects_unknown_code() -> None:
    sdf = sdfio.SdfFile(header=sdfio.SdfHeader(num_points=1, num_profiles=1), data=np.zeros((1, 1)))
    assert sdf.data_type == DataType.BINARY64

    sdf.data_type = DataType.BINARY32
    assert sdf.data_type == DataType.BINARY32
    assert sdf.header.data_type == DataType.BINARY32

    with pytest.raises(SdfFormatError, match="Unknown SDF data type code"):
        sdf.data_type = 99


def test_changing_data_type_incompatible_with_dialect_raises() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=SdfDialect.ISO_1_0, num_points=1, num_profiles=1, data_type=5
        ),
        data=np.array([[1.0]]),
    )
    sdf.data_type = DataType.BINARY32  # ISO-2.0/BCR-1.0 only
    with pytest.raises(SdfFormatError, match="not valid for SDF dialect"):
        sdf.dumps()


def test_with_dialect_upgrades_iso_1_0_to_2_0() -> None:
    # The standard only specifies UTC for version 2.0, so a naive date is
    # presumed to already be local time; upgrading to ISO-2.0 then converts
    # it to UTC, shifting the wall clock accordingly.
    naive_date = datetime(2024, 1, 1, 12, 0)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=SdfDialect.ISO_1_0,
            num_points=1,
            num_profiles=1,
            create_date=naive_date,
            mod_date=naive_date,
        ),
        data=np.array([[1.0]]),
    )
    assert sdf.header.create_date is not None
    assert sdf.header.create_date.tzinfo is None

    upgraded = sdf.with_dialect(SdfDialect.ISO_2_0)

    assert upgraded.header.dialect == SdfDialect.ISO_2_0
    upgraded_create_date = upgraded.header.create_date
    upgraded_mod_date = upgraded.header.mod_date
    assert upgraded_create_date is not None
    assert upgraded_mod_date is not None
    assert upgraded_create_date.tzinfo == UTC
    assert upgraded_mod_date.tzinfo == UTC
    assert upgraded_create_date == naive_date.astimezone(UTC)


def test_with_dialect_downgrades_iso_2_0_to_1_0() -> None:
    # The standard only specifies UTC for version 2.0, so a UTC-aware date
    # is converted to the system's local timezone for version 1.0, rather
    # than being kept as UTC.
    utc_date = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=SdfDialect.ISO_2_0,
            num_points=1,
            num_profiles=1,
            create_date=utc_date,
            mod_date=utc_date,
        ),
        data=np.array([[1.0]]),
    )

    downgraded = sdf.with_dialect(SdfDialect.ISO_1_0)

    assert downgraded.header.dialect == SdfDialect.ISO_1_0
    assert downgraded.header.create_date == utc_date
    assert downgraded.header.mod_date == utc_date


def test_with_dialect_rejects_incompatible_data_type_unless_replaced() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=SdfDialect.ISO_2_0, num_points=1, num_profiles=1, data_type=DataType.BINARY32
        ),
        data=np.array([[1.0]]),
    )
    with pytest.raises(SdfFormatError, match="not valid for SDF dialect"):
        sdf.with_dialect(SdfDialect.ISO_1_0)

    downgraded = sdf.with_dialect(SdfDialect.ISO_1_0, data_type=DataType.INT16)
    assert downgraded.header.data_type == DataType.INT16


def test_with_dialect_rejects_non_tagged_trailer_unless_replaced() -> None:
    naive_date = datetime(2024, 1, 1, 12, 0)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            dialect=SdfDialect.ISO_1_0,
            num_points=1,
            num_profiles=1,
            create_date=naive_date,
            mod_date=naive_date,
        ),
        data=np.array([[1.0]]),
        trailer="some freeform notes, not tagged fields",
    )
    with pytest.raises(SdfFormatError, match="tagged 'Name = Value' format"):
        sdf.with_dialect(SdfDialect.ISO_2_0)

    upgraded = sdf.with_dialect(SdfDialect.ISO_2_0, trailer="")
    assert upgraded.trailer == ""


def test_save_can_convert_format() -> None:
    data = _sample_data()
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            binary=True, num_points=3, num_profiles=2, x_scale=1e-6, y_scale=1e-6
        ),
        data=data,
    )

    converted = sdfio.SdfFile.loads(sdf.dumps(format=FileFormat.ASCII))

    assert converted.header.binary is False
    np.testing.assert_allclose(converted.data, data, rtol=1e-9, equal_nan=True)


def test_write_rejects_non_2d_array() -> None:
    with pytest.raises(SdfFormatError, match="2-D"):
        sdfio.write("unused.sdf", np.zeros(4), x_scale=1e-6, y_scale=1e-6)


@pytest.mark.parametrize("z_scale", [0.0, -1e-6])
def test_write_rejects_non_positive_z_scale(z_scale: float) -> None:
    with pytest.raises(SdfFormatError, match="z_scale must be a positive number"):
        sdfio.write("unused.sdf", np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6, z_scale=z_scale)


@pytest.mark.parametrize("z_scale", [0.0, -1e-6])
def test_dumps_rejects_non_positive_z_scale(z_scale: float) -> None:
    """The check must hold for direct SdfHeader/SdfFile construction too, not just write()."""
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(num_points=1, num_profiles=1, z_scale=z_scale), data=np.zeros((1, 1))
    )

    with pytest.raises(SdfFormatError, match="z_scale must be a positive number"):
        sdf.dumps()


@pytest.mark.parametrize("file_format", [FileFormat.BINARY, FileFormat.ASCII])
def test_sdffile_loads_dumps_roundtrip(file_format: FileFormat) -> None:
    data = _sample_data()
    header = sdfio.SdfHeader(
        dialect=SdfDialect.ISO_2_0,
        binary=file_format == FileFormat.BINARY,
        num_points=3,
        num_profiles=2,
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale=1e-7,
        data_type=7,
    )
    sdf = sdfio.SdfFile(header=header, data=data)

    blob = sdf.dumps()
    assert isinstance(blob, bytes)

    round_tripped = sdfio.SdfFile.loads(blob)
    assert round_tripped.header.binary == (file_format == FileFormat.BINARY)
    np.testing.assert_allclose(round_tripped.data, data, rtol=1e-9, equal_nan=True)


def test_sdffile_open_save_via_binary_stream() -> None:
    data = _sample_data()
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(num_points=3, num_profiles=2, x_scale=1e-6, y_scale=1e-6),
        data=data,
    )

    buffer = io.BytesIO()
    sdf.save(buffer)
    buffer.seek(0)
    reloaded = sdfio.SdfFile.open(buffer)

    np.testing.assert_allclose(reloaded.data, data, rtol=1e-9, equal_nan=True)


def test_sdffile_loads_rejects_unknown_magic() -> None:
    with pytest.raises(SdfFormatError, match="Not an SDF file"):
        sdfio.SdfFile.loads(b"not-an-sdf-blob")


def test_sdffile_eq_compares_nan_aware_data() -> None:
    header = sdfio.SdfHeader(num_points=2, num_profiles=1)
    data = np.array([[1.0, np.nan]])

    a = sdfio.SdfFile(header=header, data=data.copy())
    b = sdfio.SdfFile(header=header, data=data.copy())
    c = sdfio.SdfFile(header=header, data=np.array([[1.0, 2.0]]))

    assert a == b
    assert a != c
    assert a != object()
