# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The synthetic sample files in ``samples/``: they are current, valid and read as intended."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import sdfio
from sdfio import DataType, FileFormat, Revision, SdfFile
from sdfio.cli import main

SAMPLES = Path(__file__).parent / "samples"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate", SAMPLES / "generate.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate"] = module
    spec.loader.exec_module(module)
    return module


generate = _generator()

#: The data types of the samples of each revision.
DATA_TYPES = {
    Revision.ISO_1_0: {DataType.INT16, DataType.INT32, DataType.BINARY64},
    Revision.ISO_2_0: {
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.BINARY32,
        DataType.BINARY64,
    },
    Revision.BCR_1_0: {DataType.UINT16, DataType.INT16, DataType.INT32, DataType.BINARY64},
}
FLOAT_TYPES = (DataType.BINARY32, DataType.BINARY64)
CASES = [
    (revision, name, suffix)
    for revision in generate.FOLDERS
    for name in generate.build(revision)
    for suffix in generate.FORMATS
]


def _path(revision: Revision, name: str, suffix: str) -> Path:
    return Path(generate.path(revision, name, suffix))


def _id(case: tuple[Revision, str, str]) -> str:
    revision, name, suffix = case
    return f"{revision}/{name}_{suffix}"


def test_there_are_exactly_the_files_that_the_generator_makes() -> None:
    expected = {_path(*case) for case in CASES}
    found = {path for folder in generate.FOLDERS.values() for path in folder.glob("*.sdf")}
    assert found == expected
    folders = {folder for folder in SAMPLES.iterdir() if folder.is_dir()}
    assert folders - {SAMPLES / "__pycache__"} == set(generate.FOLDERS.values())


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_sample_is_what_the_generator_makes(case: tuple[Revision, str, str]) -> None:
    revision, name, suffix = case
    sample = sdfio.read(_path(*case))
    made = SdfFile.loads(
        generate.dumps(revision, generate.build(revision)[name], generate.FORMATS[suffix])
    )

    # The bytes are not compared: the last digit of a sine depends on the platform, which can
    # change a float in the last place and an integer by one step in a rounding that is close.
    assert sample.header == made.header
    assert sample.trailer == made.trailer
    assert np.array_equal(np.isnan(sample.data), np.isnan(made.data))
    step = sample.header.z_scale if sample.header.data_type not in FLOAT_TYPES else 0.0
    np.testing.assert_allclose(
        sample.data, made.data, rtol=1e-9, atol=max(step * (1 + 1e-9), 1e-15), equal_nan=True
    )


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_sample_reads_as_it_was_meant(case: tuple[Revision, str, str]) -> None:
    revision, name, suffix = case
    expected = generate.build(revision)[name]

    sdf = sdfio.read(_path(*case))

    assert sdf.header.revision is revision
    assert sdf.header.format is (FileFormat.BINARY if suffix == "binary" else FileFormat.ASCII)
    assert sdf.header.data_type is expected.data_type
    assert sdf.data.shape == expected.data.shape
    assert (sdf.header.x_scale, sdf.header.y_scale) == (generate.X_SCALE, generate.Y_SCALE)
    assert sdf.header.z_scale == generate.Z_SCALES[expected.data_type]
    # The points that are not measured are those of the data, at the same rows and columns.
    assert np.array_equal(np.isnan(sdf.data), np.isnan(expected.data))
    # The heights are the data to the precision of the type: the quantization step of an
    # integer type and the digits of the text of a float.
    if expected.data_type in FLOAT_TYPES:
        np.testing.assert_allclose(sdf.data, expected.data, rtol=1e-6, atol=1e-12, equal_nan=True)
    else:
        step = sdf.header.z_scale
        np.testing.assert_allclose(
            sdf.data, expected.data, rtol=0, atol=step / 2 * (1 + 1e-9), equal_nan=True
        )


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_sample_is_written_back_as_it_was_read(case: tuple[Revision, str, str]) -> None:
    revision, name, _ = case
    sdf = sdfio.read(_path(*case))

    assert SdfFile.loads(sdf.dumps()) == sdf
    if not generate.build(revision)[name].trailer:
        # The line ends of a trailer are not kept in the ASCII representation.
        assert sdf.dumps() == _path(*case).read_bytes()


@pytest.mark.parametrize(
    ("revision", "name"),
    [(revision, name) for revision in generate.FOLDERS for name in generate.build(revision)],
    ids=lambda value: str(value),
)
def test_the_ascii_and_the_binary_file_hold_the_same_data(revision: Revision, name: str) -> None:
    ascii_file = sdfio.read(_path(revision, name, "ascii"))
    binary_file = sdfio.read(_path(revision, name, "binary"))

    assert ascii_file.header.format is FileFormat.ASCII
    assert binary_file.header.format is FileFormat.BINARY
    assert ascii_file.header.data_type == binary_file.header.data_type
    assert np.array_equal(np.isnan(ascii_file.data), np.isnan(binary_file.data))
    # The text of a binary32 has 7 digits, the text of a binary64 has 15.
    rtol = 1e-6 if ascii_file.header.data_type is DataType.BINARY32 else 1e-9
    np.testing.assert_allclose(
        ascii_file.data, binary_file.data, rtol=rtol, atol=1e-15, equal_nan=True
    )
    assert ascii_file.trailer_fields == binary_file.trailer_fields


@pytest.mark.parametrize("revision", list(generate.FOLDERS))
def test_each_revision_has_samples_of_data_types_that_it_defines(revision: Revision) -> None:
    used = {sample.data_type for sample in generate.build(revision).values()}
    assert used == DATA_TYPES[revision]
    assert used <= set(revision.data_types)


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_the_points_of_a_sample_have_the_right_coordinates(
    case: tuple[Revision, str, str],
) -> None:
    revision, name, _ = case
    expected = generate.build(revision)[name]
    sdf = sdfio.read(_path(*case))

    points = sdf.points()

    rows, columns = expected.data.shape
    assert points.shape == (rows, columns, 3)
    # x grows along the columns and y along the rows, in every revision.
    np.testing.assert_allclose(points[0, :, 0], np.arange(columns) * generate.X_SCALE)
    np.testing.assert_allclose(points[:, 0, 1], np.arange(rows) * generate.Y_SCALE)
    assert np.array_equal(np.isnan(points[..., 2]), np.isnan(expected.data))


@pytest.mark.parametrize("revision", list(generate.FOLDERS))
def test_the_missing_region_is_at_the_same_rows_in_every_revision(revision: Revision) -> None:
    # BCR-1.0 stores the first profile at the maximum y, so a wrong order would show up here.
    sdf = sdfio.read(_path(revision, "surface", "binary"))

    missing = np.isnan(sdf.data)

    assert np.flatnonzero(missing.any(axis=1)).tolist() == [10, 11, 12, 13]
    assert np.flatnonzero(missing.any(axis=0)).tolist() == list(range(20, 30))
    assert int(missing.sum()) == 40


@pytest.mark.parametrize("revision", list(generate.FOLDERS))
def test_the_surface_has_a_trailer_and_the_profile_has_none(revision: Revision) -> None:
    for suffix in generate.FORMATS:
        sdf = sdfio.read(_path(revision, "surface", suffix))
        text = sdf.trailer if isinstance(sdf.trailer, str) else sdf.trailer.decode("ascii")
        assert "Jane Doe" in text
        assert dict(sdf.trailer_fields) == {
            "OperatorName": "Jane Doe",
            "PartName": "Synthetic bumps",
        }
    assert sdfio.read(_path(revision, "profile", "binary")).trailer in ("", b"")


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_the_command_line_describes_a_sample(
    case: tuple[Revision, str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    revision, _, suffix = case

    assert main(["info", str(_path(*case))]) == 0

    output = capsys.readouterr().out
    assert f"Revision    = {revision}" in output
    assert f"Format      = {'ASCII' if suffix == 'ascii' else 'binary'}" in output


@pytest.mark.parametrize("name", ["surface", "surface_int32", "surface_binary64"])
def test_a_sample_converts_between_the_revisions_and_back(name: str, tmp_path: Path) -> None:
    source = sdfio.read(_path(Revision.ISO_1_0, name, "binary"))
    for target in (Revision.ISO_2_0, Revision.BCR_1_0):
        converted = tmp_path / f"{target}.sdf"
        assert (
            main(
                [
                    "convert",
                    "-d",
                    str(target),
                    "-x",
                    str(_path(Revision.ISO_1_0, name, "binary")),
                    str(converted),
                ]
            )
            == 0
        )
        again = sdfio.read(converted)
        assert again.header.revision is target
        assert np.array_equal(np.isnan(again.data), np.isnan(source.data))
        np.testing.assert_allclose(again.data, source.data, rtol=0, atol=1e-12, equal_nan=True)
        # The first profile is at y = 0 whatever the order on disk is.
        assert np.flatnonzero(np.isnan(again.data).any(axis=1)).tolist() == [10, 11, 12, 13]


def test_the_readme_lists_every_sample() -> None:
    text = (SAMPLES / "README.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"`([a-z0-9_]+)`", text))
    names = {name for revision in generate.FOLDERS for name in generate.build(revision)}
    assert names <= listed
