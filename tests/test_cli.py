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


def test_cli_info_prints_dialect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((2, 3)), x_scale=1e-6, y_scale=2e-6)

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Dialect     = ISO-2.0" in output


def test_cli_info_prints_not_recorded_for_none_dates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "surface.sdf"
    header = sdfio.SdfHeader(num_points=1, num_profiles=1, create_date=None, mod_date=None)
    sdfio.SdfFile(header=header, data=np.zeros((1, 1))).save(path)

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "CreateDate  = (not recorded)" in output
    assert "ModDate     = (not recorded)" in output


def test_cli_info_omits_empty_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6)

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Trailer " not in output


def test_cli_info_prints_trailer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "surface.sdf"
    sdfio.write(path, np.zeros((1, 1)), x_scale=1e-6, y_scale=1e-6, trailer="Note = hi")

    assert main(["info", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Trailer " in output
    assert "Note = hi" in output


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


def test_cli_convert_changes_dialect(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
    )

    assert main(["convert", str(source), str(destination), "--dialect", "ISO-2.0"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.dialect == sdfio.SdfDialect.ISO_2_0
    assert converted.header.create_date is not None
    assert converted.header.create_date.tzinfo is not None


def test_cli_convert_dialect_rejects_incompatible_data_type(
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
            dialect=sdfio.SdfDialect.ISO_2_0, data_type=sdfio.DataType.BINARY32
        ),
    )

    assert main(["convert", str(source), str(destination), "--dialect", "ISO-1.0"]) == 1
    assert "not valid for SDF dialect" in capsys.readouterr().err


def test_cli_convert_dialect_keeps_already_tagged_trailer_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
        trailer="Operator = Jane Doe",
    )

    assert main(["convert", str(source), str(destination), "--dialect", "ISO-2.0"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.dialect == sdfio.SdfDialect.ISO_2_0
    assert converted.trailer == b"Operator = Jane Doe"


def test_cli_convert_dialect_rejects_incompatible_trailer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
        trailer="not tagged fields",
    )

    assert main(["convert", str(source), str(destination), "--dialect", "ISO-2.0"]) == 1
    assert "tagged 'Name = Value' format" in capsys.readouterr().err


def test_cli_convert_fix_trailer_drops_entirely_incompatible_trailer(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
        trailer="not tagged fields",
    )

    args = ["convert", str(source), str(destination), "--dialect", "ISO-2.0", "--fix-trailer"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.header.dialect == sdfio.SdfDialect.ISO_2_0
    assert not converted.trailer


def test_cli_convert_fix_trailer_replaces_non_ascii_char_for_non_tagged_dialect(
    tmp_path: Path,
) -> None:
    """-x must also catch the universal ASCII rule, not just the dialect-specific tagged format."""
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdf = sdfio.SdfFile(
        header=sdfio.SdfHeader(dialect=sdfio.SdfDialect.ISO_1_0, num_points=1, num_profiles=1),
        data=np.zeros((1, 1)),
        trailer=b"Note = X\nAnother line",
    )
    raw = bytearray(sdf.dumps())
    raw[raw.index(b"X")] = 0xE9  # corrupt just the 'X', leaving the rest untouched
    source.write_bytes(bytes(raw))

    # BCR-1.0 has no tagged-format requirement, so freeform content (like
    # "Another line", with no "=") must survive; only the ASCII rule applies,
    # and the bad byte is replaced (not the whole line dropped).
    args = ["convert", str(source), str(destination), "--dialect", "BCR-1.0", "--fix-trailer"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.trailer == b"Note = ?\r\nAnother line\r\n"


def test_cli_convert_fix_trailer_keeps_only_tagged_lines(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        format=sdfio.FileFormat.ASCII,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
        trailer="OperatorName = WG 16\r\nsome free text without equals\r\nPartName = X",
    )

    args = ["convert", str(source), str(destination), "--dialect", "ISO-2.0", "--fix-trailer"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.header.dialect == sdfio.SdfDialect.ISO_2_0
    assert converted.trailer_fields == {"OperatorName": "WG 16", "PartName": "X"}


def test_cli_convert_fix_trailer_keeps_already_valid_trailer(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        format=sdfio.FileFormat.ASCII,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.ISO_1_0),
        trailer="Note = keep me",
    )

    args = ["convert", str(source), str(destination), "--dialect", "ISO-2.0", "-x"]
    assert main(args) == 0

    converted = sdfio.read(destination)
    assert converted.trailer == "Note = keep me"


def test_cli_convert_changes_from_bcr_to_iso_2_0(tmp_path: Path) -> None:
    source = tmp_path / "surface.sdf"
    destination = tmp_path / "converted.sdf"
    sdfio.write(
        source,
        np.ones((2, 2)),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=sdfio.SdfMetadata(dialect=sdfio.SdfDialect.BCR_1_0),
    )

    assert main(["convert", str(source), str(destination), "--dialect", "ISO-2.0"]) == 0

    converted = sdfio.read(destination)
    assert converted.header.dialect == sdfio.SdfDialect.ISO_2_0


@pytest.mark.parametrize("target", ["XYZ-9.9", "BCR-2.0"])
def test_cli_convert_rejects_unrecognized_target(
    target: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Neither a wholly unknown dialect string nor a recognized-but-invalid
    # combination (BCR never had version 2.0, so "BCR-2.0" isn't a valid
    # SdfDialect member) is constructible -- both are rejected by argument
    # parsing itself, before a source file is even read.
    with pytest.raises(SystemExit) as exc_info:
        main(["convert", "in.sdf", "out.sdf", "--dialect", target])
    assert exc_info.value.code == 2
    assert "invalid dialect" in capsys.readouterr().err


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
