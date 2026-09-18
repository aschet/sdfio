# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from sdfio.datatypes import (
    DataType,
    decode_raw,
    encode_raw,
    get_data_type,
    suggest_z_scale,
    validate_data_range,
)
from sdfio.exceptions import SdfFormatError
from sdfio.header import SdfDialect, SdfVersion


def test_get_data_type_by_code() -> None:
    dt = get_data_type(7)
    assert dt.type == DataType.BINARY64
    assert dt.dtype == np.dtype("<f8")


def test_get_data_type_accepts_data_type_member() -> None:
    assert get_data_type(DataType.INT16).type == DataType.INT16


def test_get_data_type_unknown_code() -> None:
    with pytest.raises(SdfFormatError):
        get_data_type(99)


@pytest.mark.parametrize(
    ("data_type", "version", "expected"),
    [
        (DataType.BINARY32, SdfVersion.V2_0, True),
        (DataType.BINARY32, SdfVersion.V1_0, False),
        (DataType.INT8, SdfVersion.V2_0, True),
        (DataType.INT8, SdfVersion.V1_0, False),
        (DataType.INT16, SdfVersion.V1_0, True),
        (DataType.INT16, SdfVersion.V2_0, True),
        (DataType.INT32, SdfVersion.V1_0, True),
        (DataType.BINARY64, SdfVersion.V1_0, True),
        (DataType.BINARY64, SdfVersion.V2_0, True),
    ],
)
def test_version_support(data_type: DataType, version: SdfVersion, expected: bool) -> None:
    assert get_data_type(data_type).is_supported(version, SdfDialect.ISO) is expected


@pytest.mark.parametrize("data_type", [DataType.BINARY32, DataType.INT8, DataType.INT16])
def test_version_support_bcr_ignores_iso_version_restriction(data_type: DataType) -> None:
    assert get_data_type(data_type).is_supported(SdfVersion.V1_0, SdfDialect.BCR) is True


def test_invalid_value_iso_is_minimum() -> None:
    data_type = get_data_type(DataType.INT16)
    assert data_type.invalid_value(SdfDialect.ISO) == np.iinfo(data_type.dtype).min


def test_invalid_value_bcr_is_maximum() -> None:
    # BCR still had unsigned types where 0 (the minimum for a signed type)
    # is a valid low bound, so it used the maximum as the sentinel instead.
    data_type = get_data_type(DataType.INT16)
    assert data_type.invalid_value(SdfDialect.BCR) == np.iinfo(data_type.dtype).max


def test_validate_data_range_ignores_invalid_entries() -> None:
    data_type = get_data_type(DataType.INT8)
    raw = np.array([1.0, -128.0])  # -128 is the sentinel, but marked invalid
    validate_data_range(raw, invalid_mask=np.array([False, True]), data_type=data_type)


def test_validate_data_range_rejects_out_of_range_integer() -> None:
    data_type = get_data_type(DataType.INT8)
    raw = np.array([1000.0])
    with pytest.raises(SdfFormatError, match="out of range"):
        validate_data_range(raw, invalid_mask=np.array([False]), data_type=data_type)


def test_validate_data_range_rejects_out_of_range_float() -> None:
    data_type = get_data_type(DataType.BINARY32)
    raw = np.array([1e40])
    with pytest.raises(SdfFormatError, match="out of range"):
        validate_data_range(raw, invalid_mask=np.array([False]), data_type=data_type)


def test_validate_data_range_rejects_sentinel_collision() -> None:
    data_type = get_data_type(DataType.INT16)
    raw = np.array([-32768.0])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(raw, invalid_mask=np.array([False]), data_type=data_type)


def test_validate_data_range_rejects_float_sentinel_collision() -> None:
    data_type = get_data_type(DataType.BINARY64)
    raw = np.array([data_type.invalid_value(SdfDialect.ISO)])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(raw, invalid_mask=np.array([False]), data_type=data_type)


def test_validate_data_range_rejects_bcr_sentinel_collision() -> None:
    data_type = get_data_type(DataType.INT16)
    raw = np.array([data_type.invalid_value(SdfDialect.BCR)])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(
            raw, invalid_mask=np.array([False]), data_type=data_type, dialect=SdfDialect.BCR
        )


@pytest.mark.parametrize("data_type_code", list(DataType))
def test_encode_decode_raw_roundtrip(data_type_code: DataType) -> None:
    data_type = get_data_type(data_type_code)
    data = np.array([1.0, np.nan, -2.0, 0.0])
    z_scale = 1.0

    raw = encode_raw(data, data_type, z_scale)

    assert raw.dtype == data_type.dtype
    assert raw[1] == data_type.dtype.type(data_type.invalid_value(SdfDialect.ISO))
    np.testing.assert_allclose(decode_raw(raw, data_type, z_scale), data, equal_nan=True)


@pytest.mark.parametrize("data_type_code", [DataType.INT16, DataType.INT32, DataType.BINARY64])
def test_encode_decode_raw_roundtrip_bcr(data_type_code: DataType) -> None:
    data_type = get_data_type(data_type_code)
    data = np.array([1.0, np.nan, -2.0, 0.0])
    z_scale = 1.0

    raw = encode_raw(data, data_type, z_scale, SdfDialect.BCR)

    assert raw[1] == data_type.dtype.type(data_type.invalid_value(SdfDialect.BCR))
    np.testing.assert_allclose(
        decode_raw(raw, data_type, z_scale, SdfDialect.BCR), data, equal_nan=True
    )


def test_encode_raw_rejects_out_of_range_value() -> None:
    data_type = get_data_type(DataType.INT8)
    with pytest.raises(SdfFormatError, match="out of range"):
        encode_raw(np.array([1000.0]), data_type, z_scale=1.0)


def test_decode_raw_marks_sentinel_as_nan() -> None:
    data_type = get_data_type(DataType.INT16)
    raw = np.array([1, -32768, -1], dtype=data_type.dtype)
    decoded = decode_raw(raw, data_type, z_scale=1e-6)
    assert np.isnan(decoded[1])
    np.testing.assert_allclose(decoded[[0, 2]], [1e-6, -1e-6])


def test_suggest_z_scale_maximizes_resolution() -> None:
    data_type = get_data_type(DataType.INT16)  # range -32767..32767 (excl. sentinel)
    data = np.array([1.0e-3, -0.5e-3, np.nan])

    z_scale = suggest_z_scale(data, data_type)
    raw = encode_raw(data, data_type, z_scale)

    assert raw[np.argmax(data[:2])] == 32767  # uses the full positive range
    np.testing.assert_allclose(decode_raw(raw, data_type, z_scale)[:2], data[:2], rtol=1e-4)


def test_suggest_z_scale_handles_negative_dominant_range() -> None:
    data_type = get_data_type(DataType.INT8)
    data = np.array([-100.0, 1.0])
    z_scale = suggest_z_scale(data, data_type)
    # Must not raise: the suggested scale keeps both extremes in range.
    encode_raw(data, data_type, z_scale)


def test_suggest_z_scale_returns_one_for_all_nan_or_zero() -> None:
    data_type = get_data_type(DataType.INT16)
    assert suggest_z_scale(np.array([np.nan, np.nan]), data_type) == 1.0
    assert suggest_z_scale(np.array([0.0, 0.0]), data_type) == 1.0


def test_suggest_z_scale_is_a_noop_for_float_types() -> None:
    data_type = get_data_type(DataType.BINARY64)
    assert suggest_z_scale(np.array([1.0e10, -1.0e10]), data_type) == 1.0


def test_suggest_z_scale_maximizes_resolution_bcr() -> None:
    # Mirrors test_suggest_z_scale_maximizes_resolution, but BCR reserves the
    # *maximum* as its sentinel, so the positive-dominant case is the one
    # that must avoid it (the ISO test above already covers the negative one).
    data_type = get_data_type(DataType.INT16)
    data = np.array([1.0e-3, -0.5e-3, np.nan])

    z_scale = suggest_z_scale(data, data_type, SdfDialect.BCR)
    raw = encode_raw(data, data_type, z_scale, SdfDialect.BCR)

    assert raw[np.argmax(data[:2])] == 32766  # avoids the reserved maximum, 32767
    np.testing.assert_allclose(
        decode_raw(raw, data_type, z_scale, SdfDialect.BCR)[:2], data[:2], rtol=1e-4
    )
