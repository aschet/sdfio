# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The points of a file: the coordinates of the grid with the heights."""

from __future__ import annotations

import io

import numpy as np

import sdfio
from sdfio.file import FileFormat
from sdfio.revision import Revision


def _sample_data() -> np.ndarray:
    return np.array(
        [
            [1.0e-6, 2.0e-6, np.nan],
            [-1.0e-6, 0.0, 3.5e-6],
        ]
    )


def test_points_have_the_coordinates_of_the_axes_and_the_heights() -> None:
    buffer = io.BytesIO()
    sdfio.write(buffer, _sample_data(), x_scale=1e-6, y_scale=2e-6)
    sdf = sdfio.read(io.BytesIO(buffer.getvalue()))

    points = sdf.points()

    assert points.shape == (2, 3, 3)
    np.testing.assert_allclose(points[..., 0], np.tile(sdf.x_axis, (2, 1)))
    np.testing.assert_allclose(points[..., 1], np.tile(sdf.y_axis[:, None], (1, 3)))
    np.testing.assert_array_equal(points[..., 2], sdf.data)
    assert points[0, 2, :2].tolist() == [2e-6, 0.0]
    assert np.isnan(points[0, 2, 2])  # a point that is not measured keeps x and y


def test_points_follow_the_right_handed_system_of_every_revision() -> None:
    data = np.arange(6.0).reshape(2, 3)
    for revision in Revision:
        buffer = io.BytesIO()
        sdfio.write(
            buffer,
            data,
            x_scale=1.0,
            y_scale=10.0,
            metadata=sdfio.Metadata(revision=revision),
            format=FileFormat.ASCII,
        )
        points = sdfio.read(io.BytesIO(buffer.getvalue())).points()
        np.testing.assert_allclose(points[..., 2], data, atol=1e-6)
        assert points[1, 0, 1] == 10.0  # y grows with the row, also for BCR-1.0
        assert points[0, 2, 0] == 2.0  # x grows with the column


def test_points_of_a_profile_have_one_row_with_y_zero() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.Header(x_scale=1e-6, y_scale=5.0),
        data=np.arange(4.0).reshape(1, 4),
    )

    points = sdf.points()

    assert points.shape == (1, 4, 3)
    assert points[..., 1].tolist() == [[0.0] * 4]


def test_points_are_a_copy_and_can_be_reshaped_for_point_cloud_libraries() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.Header(x_scale=1.0, y_scale=1.0),
        data=_sample_data(),
    )

    points = sdf.points()
    points[..., 2] = 0.0
    measured = sdf.points().reshape(-1, 3)
    measured = measured[np.isfinite(measured[:, 2])]

    assert np.isnan(sdf.data[0, 2])  # the file is not changed
    assert measured.shape == (5, 3)


def test_points_can_leave_out_the_points_that_are_not_measured() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.Header(x_scale=1.0, y_scale=10.0),
        data=_sample_data(),
    )

    measured = sdf.points(drop_invalid=True)

    assert measured.shape == (5, 3)
    assert np.isfinite(measured).all()
    np.testing.assert_array_equal(measured, sdf.points().reshape(-1, 3)[[0, 1, 3, 4, 5]])
    assert sdf.points(drop_invalid=False).shape == (2, 3, 3)


def test_points_of_a_profile_without_missing_points_keep_all_of_them() -> None:
    sdf = sdfio.SdfFile(
        header=sdfio.Header(x_scale=2.0, y_scale=7.0),
        data=np.array([[1.0, 2.0, 3.0]]),
    )

    measured = sdf.points(drop_invalid=True)

    assert measured.tolist() == [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0], [4.0, 0.0, 3.0]]
