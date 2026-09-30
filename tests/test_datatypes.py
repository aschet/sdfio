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
from sdfio.header import SdfDialect


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
    ("data_type", "dialect", "expected"),
    [
        (DataType.BINARY32, SdfDialect.ISO_2_0, True),
        (DataType.BINARY32, SdfDialect.ISO_1_0, False),
        (DataType.BINARY32, SdfDialect.BCR_1_0, True),
        (DataType.INT8, SdfDialect.ISO_2_0, True),
        (DataType.INT8, SdfDialect.ISO_1_0, False),
        (DataType.INT8, SdfDialect.BCR_1_0, True),
        (DataType.INT16, SdfDialect.ISO_1_0, True),
        (DataType.INT16, SdfDialect.ISO_2_0, True),
        (DataType.INT16, SdfDialect.BCR_1_0, True),
        (DataType.INT32, SdfDialect.ISO_1_0, True),
        (DataType.BINARY64, SdfDialect.ISO_1_0, True),
        (DataType.BINARY64, SdfDialect.ISO_2_0, True),
    ],
)
def test_dialect_support(data_type: DataType, dialect: SdfDialect, expected: bool) -> None:
    assert get_data_type(data_type).is_supported(dialect) is expected


def test_invalid_value_iso_is_minimum() -> None:
    data_type = get_data_type(DataType.INT16)
    assert data_type.invalid_value(SdfDialect.ISO_2_0) == np.iinfo(data_type.dtype).min


def test_invalid_value_bcr_is_maximum() -> None:
    # BCR still had unsigned types where 0 (the minimum for a signed type)
    # is a valid low bound, so it used the maximum as the sentinel instead.
    data_type = get_data_type(DataType.INT16)
    assert data_type.invalid_value(SdfDialect.BCR_1_0) == np.iinfo(data_type.dtype).max


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
    raw = np.array([data_type.invalid_value(SdfDialect.ISO_2_0)])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(raw, invalid_mask=np.array([False]), data_type=data_type)


def test_validate_data_range_rejects_sentinel_collision_after_rounding() -> None:
    # 32766.999999999996 != 32767.0, but it rounds to the BCR int16 sentinel on encode.
    data_type = get_data_type(DataType.INT16)
    raw = np.array([32766.999999999996])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(
            raw, invalid_mask=np.array([False]), data_type=data_type, dialect=SdfDialect.BCR_1_0
        )


def test_validate_data_range_rejects_bcr_sentinel_collision() -> None:
    data_type = get_data_type(DataType.INT16)
    raw = np.array([data_type.invalid_value(SdfDialect.BCR_1_0)])
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(
            raw, invalid_mask=np.array([False]), data_type=data_type, dialect=SdfDialect.BCR_1_0
        )


@pytest.mark.parametrize(
    "data_type_code",
    [DataType.BINARY32, DataType.INT8, DataType.INT16, DataType.INT32, DataType.BINARY64],
)
def test_encode_decode_raw_roundtrip(data_type_code: DataType) -> None:
    data_type = get_data_type(data_type_code)
    data = np.array([1.0, np.nan, -2.0, 0.0])
    z_scale = 1.0

    raw = encode_raw(data, data_type, z_scale)

    assert raw.dtype == data_type.dtype
    assert raw[1] == data_type.dtype.type(data_type.invalid_value(SdfDialect.ISO_2_0))
    np.testing.assert_allclose(decode_raw(raw, data_type, z_scale), data, equal_nan=True)


@pytest.mark.parametrize("data_type_code", [DataType.INT16, DataType.INT32, DataType.BINARY64])
def test_encode_decode_raw_roundtrip_bcr(data_type_code: DataType) -> None:
    data_type = get_data_type(data_type_code)
    data = np.array([1.0, np.nan, -2.0, 0.0])
    z_scale = 1.0

    raw = encode_raw(data, data_type, z_scale, SdfDialect.BCR_1_0)

    assert raw[1] == data_type.dtype.type(data_type.invalid_value(SdfDialect.BCR_1_0))
    np.testing.assert_allclose(
        decode_raw(raw, data_type, z_scale, SdfDialect.BCR_1_0), data, equal_nan=True
    )


@pytest.mark.parametrize("data_type_code", [DataType.UINT8, DataType.UINT16, DataType.UINT32])
def test_encode_decode_raw_roundtrip_bcr_unsigned(data_type_code: DataType) -> None:
    # Unsigned types can't represent a negative value, unlike the signed BCR types above.
    data_type = get_data_type(data_type_code)
    data = np.array([1.0, np.nan, 2.0, 0.0])
    z_scale = 1.0

    raw = encode_raw(data, data_type, z_scale, SdfDialect.BCR_1_0)

    assert raw[1] == data_type.dtype.type(data_type.invalid_value(SdfDialect.BCR_1_0))
    np.testing.assert_allclose(
        decode_raw(raw, data_type, z_scale, SdfDialect.BCR_1_0), data, equal_nan=True
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

    z_scale = suggest_z_scale(data, data_type, SdfDialect.BCR_1_0)
    raw = encode_raw(data, data_type, z_scale, SdfDialect.BCR_1_0)

    assert raw[np.argmax(data[:2])] == 32766  # avoids the reserved maximum, 32767
    np.testing.assert_allclose(
        decode_raw(raw, data_type, z_scale, SdfDialect.BCR_1_0)[:2], data[:2], rtol=1e-4
    )


def test_decode_raw_bcr_float_treats_values_at_or_above_nominal_maximum_as_invalid() -> None:
    binary32 = get_data_type(DataType.BINARY32)
    raw32 = np.array([1.0, 3.4e38, np.finfo(np.float32).max, np.inf, 3.3e38], dtype=binary32.dtype)
    decoded32 = decode_raw(raw32, binary32, 1.0, SdfDialect.BCR_1_0)
    np.testing.assert_array_equal(np.isnan(decoded32), [False, True, True, True, False])

    binary64 = get_data_type(DataType.BINARY64)
    raw64 = np.array([1.0, 1.7e308, np.finfo(np.float64).max, 1.6e308], dtype=binary64.dtype)
    decoded64 = decode_raw(raw64, binary64, 1.0, SdfDialect.BCR_1_0)
    np.testing.assert_array_equal(np.isnan(decoded64), [False, True, True, False])


def test_decode_raw_iso_float_value_near_maximum_is_data() -> None:
    # ISO's sentinel is the minimum, so a huge positive value is ordinary data.
    binary32 = get_data_type(DataType.BINARY32)
    raw = np.array([3.4e38], dtype=binary32.dtype)
    assert not np.isnan(decode_raw(raw, binary32, 1.0, SdfDialect.ISO_2_0)).any()


def test_validate_data_range_rejects_bcr_float_value_at_nominal_maximum() -> None:
    data_type = get_data_type(DataType.BINARY32)
    with pytest.raises(SdfFormatError, match="invalid-point sentinel"):
        validate_data_range(
            np.array([3.4e38]),
            invalid_mask=np.array([False]),
            data_type=data_type,
            dialect=SdfDialect.BCR_1_0,
        )
    validate_data_range(
        np.array([3.3e38]),
        invalid_mask=np.array([False]),
        data_type=data_type,
        dialect=SdfDialect.BCR_1_0,
    )
