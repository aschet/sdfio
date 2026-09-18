# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import sdfio
from sdfio.datatypes import DataType
from sdfio.exceptions import SdfFormatError
from sdfio.file import FileFormat
from sdfio.header import SdfVersion


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
@pytest.mark.parametrize("version", [SdfVersion.V1_0, SdfVersion.V2_0])
def test_write_read_roundtrip(file_format: FileFormat, version: SdfVersion) -> None:
    buffer = io.BytesIO()
    data = _sample_data()
    sdfio.write(
        buffer,
        data,
        x_scale=1e-6,
        y_scale=2e-6,
        z_scale=1e-7,
        format=file_format,
        metadata=sdfio.SdfMetadata(version=version, data_type=DataType.BINARY64),
    )
    buffer.seek(0)

    sdf = sdfio.read(buffer)

    assert sdf.header.version == version
    assert sdf.header.binary == (file_format == FileFormat.BINARY)
    assert sdf.header.num_points == 3
    assert sdf.header.num_profiles == 2
    np.testing.assert_allclose(sdf.data, data, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(sdf.x_axis, np.arange(3) * 1e-6)
    np.testing.assert_allclose(sdf.y_axis, np.arange(2) * 2e-6)
    if version == SdfVersion.V2_0:
        assert sdf.header.create_date.tzinfo == UTC
        assert sdf.header.mod_date.tzinfo == UTC
    else:
        assert sdf.header.create_date.tzinfo is None
        assert sdf.header.mod_date.tzinfo is None


def test_write_defaults_to_binary_version_2_0() -> None:
    buffer = io.BytesIO()
    sdfio.write(buffer, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6)
    buffer.seek(0)
    sdf = sdfio.read(buffer)
    assert sdf.header.binary is True
    assert sdf.header.version == SdfVersion.V2_0


@pytest.mark.parametrize("file_format", [FileFormat.BINARY, FileFormat.ASCII])
@pytest.mark.parametrize("version", [SdfVersion.V1_0, SdfVersion.V2_0])
def test_trailer_xml_roundtrip(file_format: FileFormat, version: SdfVersion) -> None:
    # A single well-formed XML document with one root element, as required
    # for version 2.0 (and commonly used for version 1.0 too).
    root = ET.Element("Metadata")
    ET.SubElement(root, "FILENAME").text = "surface.sdf"

    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            version=version, binary=file_format == FileFormat.BINARY, num_points=1, num_profiles=1
        ),
        data=np.zeros((1, 1)),
    )
    sdf.trailer_xml = root

    reloaded = sdfio.SdfFile.loads(sdf.dumps())
    parsed = reloaded.trailer_xml
    assert parsed.tag == "Metadata"
    filename = parsed.find("FILENAME")
    assert filename is not None
    assert filename.text == "surface.sdf"


@pytest.mark.parametrize("binary", [True, False])
def test_dumps_rejects_non_ascii_trailer(binary: bool) -> None:
    """Record 3 (the trailer) is defined as a sequence of ASCII values, for both formats."""
    root = ET.Element("Metadata")
    ET.SubElement(root, "Operator").text = "Müller café"
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(binary=binary, num_points=1, num_profiles=1),
        data=np.zeros((1, 1)),
    )
    sdf.trailer_xml = root

    with pytest.raises(SdfFormatError, match="must be 7-bit ASCII"):
        sdf.dumps()


def test_trailer_xml_getter_stays_lenient_on_non_ascii_bytes() -> None:
    """Inspecting an already non-ASCII trailer (e.g. from a non-compliant file) must not raise."""
    trailer = "<Metadata><Operator>Müller café</Operator></Metadata>".encode()
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(num_points=1, num_profiles=1), data=np.zeros((1, 1)), trailer=trailer
    )

    operator = sdf.trailer_xml.find("Operator")

    assert operator is not None
    assert operator.text == "Müller café"


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


def test_changing_data_type_incompatible_with_version_raises() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(version=SdfVersion.V1_0, num_points=1, num_profiles=1, data_type=5),
        data=np.array([[1.0]]),
    )
    sdf.data_type = DataType.BINARY32  # version 2.0 only
    with pytest.raises(SdfFormatError, match="not valid for SDF version"):
        sdf.dumps()


def test_with_version_upgrades_1_0_to_2_0() -> None:
    naive_date = datetime(2024, 1, 1, 12, 0)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            version=SdfVersion.V1_0,
            num_points=1,
            num_profiles=1,
            create_date=naive_date,
            mod_date=naive_date,
        ),
        data=np.array([[1.0]]),
    )
    assert sdf.header.create_date.tzinfo is None

    upgraded = sdf.with_version(SdfVersion.V2_0, assume_utc=True)

    assert upgraded.header.version == SdfVersion.V2_0
    assert upgraded.header.create_date.tzinfo == UTC
    assert upgraded.header.mod_date.tzinfo == UTC
    # assume_utc only reattaches tzinfo; it must not shift the wall clock.
    assert upgraded.header.create_date.replace(tzinfo=None) == sdf.header.create_date


def test_with_version_rejects_naive_dates_without_assume_utc() -> None:
    naive_date = datetime(2024, 1, 1, 12, 0)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            version=SdfVersion.V1_0,
            num_points=1,
            num_profiles=1,
            create_date=naive_date,
            mod_date=naive_date,
        ),
        data=np.array([[1.0]]),
    )
    with pytest.raises(SdfFormatError, match="timezone-aware UTC"):
        sdf.with_version(SdfVersion.V2_0)


def test_with_version_rejects_incompatible_data_type_unless_replaced() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            version=SdfVersion.V2_0, num_points=1, num_profiles=1, data_type=DataType.BINARY32
        ),
        data=np.array([[1.0]]),
    )
    with pytest.raises(SdfFormatError, match="not valid for SDF version"):
        sdf.with_version(SdfVersion.V1_0)

    downgraded = sdf.with_version(SdfVersion.V1_0, data_type=DataType.INT16)
    assert downgraded.header.data_type == DataType.INT16


def test_with_version_rejects_non_xml_trailer_unless_replaced() -> None:
    naive_date = datetime(2024, 1, 1, 12, 0)
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(
            version=SdfVersion.V1_0,
            num_points=1,
            num_profiles=1,
            create_date=naive_date,
            mod_date=naive_date,
        ),
        data=np.array([[1.0]]),
        trailer="Operator = Jane Doe",
    )
    with pytest.raises(SdfFormatError, match="must be well-formed XML"):
        sdf.with_version(SdfVersion.V2_0, assume_utc=True)

    upgraded = sdf.with_version(SdfVersion.V2_0, assume_utc=True, trailer="")
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
        version=SdfVersion.V2_0,
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
