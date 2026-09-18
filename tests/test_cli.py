# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sdfio
from sdfio.cli import main


def test_cli_info_prints_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((2, 3)), x_scale=1e-6, y_scale=2e-6)

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "NumPoints   = 3" in output
    assert "NumProfiles = 2" in output


def test_cli_info_omits_empty_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6)

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Trailer " not in output


def test_cli_info_prints_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6, trailer="<Note>hi</Note>")

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Trailer " in output
    assert "<Note>hi</Note>" in output


def test_cli_info_reports_missing_file_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", str(tmp_path / "missing.sdf")]) == 1

    err = capsys.readouterr().err
    assert err.startswith("sdfio: error:")
    assert "Traceback" not in err


def test_cli_convert(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "surface_ascii.sdf"
    sdfio.write(source, np.ones((2, 2)), x_scale=1e-6, y_scale=1e-6, format=sdfio.FileFormat.BINARY)

    assert main(["convert", str(source), str(destination), "--format", "ascii"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.binary is False
    np.testing.assert_allclose(converted.data, np.ones((2, 2)), rtol=1e-9)


def test_cli_convert_changes_data_type(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        z_scale=1.0,
        metadata=sdfio.SdfMetadata(data_type=sdfio.DataType.INT16),
    )

    args = [
        "convert",
        str(source),
        str(destination),
        "--format",
        "binary",
        "--type",
        "binary64",
    ]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.header.data_type == sdfio.DataType.BINARY64
    np.testing.assert_allclose(converted.data, np.ones((2, 2)), rtol=1e-9)


def test_cli_convert_without_format_keeps_source_format(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(source, np.ones((2, 2)), x_scale=1e-6, y_scale=1e-6, format=sdfio.FileFormat.ASCII)

    assert main(["convert", str(source), str(destination), "--type", "binary64"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.binary is False


def test_cli_convert_changes_version(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(version=sdfio.SdfVersion.V1_0),
    )

    assert main(["convert", str(source), str(destination), "--number", "2.0"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.version == sdfio.SdfVersion.V2_0
    assert converted.header.create_date.tzinfo is not None


def test_cli_convert_version_rejects_incompatible_data_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(
            version=sdfio.SdfVersion.V2_0, data_type=sdfio.DataType.BINARY32
        ),
    )

    assert main(["convert", str(source), str(destination), "--number", "1.0"]) == 1
    assert "not valid for SDF version" in capsys.readouterr().err


def test_cli_convert_version_rejects_incompatible_trailer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(version=sdfio.SdfVersion.V1_0),
        trailer="Operator = Jane Doe",
    )

    assert main(["convert", str(source), str(destination), "--number", "2.0"]) == 1
    assert "must be well-formed XML" in capsys.readouterr().err


def test_cli_convert_drop_trailer_allows_incompatible_trailer(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(version=sdfio.SdfVersion.V1_0),
        trailer="Operator = Jane Doe",
    )

    args = ["convert", str(source), str(destination), "--number", "2.0", "--drop-trailer"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.header.version == sdfio.SdfVersion.V2_0
    assert not converted.trailer


def test_cli_convert_drop_trailer_keeps_already_valid_trailer(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        format=sdfio.FileFormat.ASCII,
        metadata=sdfio.SdfMetadata(version=sdfio.SdfVersion.V1_0),
        trailer="<Note>keep me</Note>",
    )

    args = ["convert", str(source), str(destination), "--number", "2.0", "-d"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.trailer == "<Note>keep me</Note>"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "sdfio" in output


def test_python_dash_m_sdfio_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sdfio", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "sdfio" in result.stdout
