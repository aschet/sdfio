# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF data area storage types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .exceptions import SdfFormatError
from .header import DataType, SdfVersion

# DataType lives in .header (see its docstring for why); re-exported here
# since callers reasonably expect it alongside the rest of the data-type API.
__all__ = [
    "DATA_TYPES",
    "DataType",
    "SdfDataType",
    "decode_raw",
    "encode_raw",
    "get_data_type",
    "require_supported_data_type",
    "suggest_z_scale",
    "validate_data_range",
]


@dataclass(frozen=True)
class SdfDataType:
    """Description of one SDF data area storage type.

    :param type: ``DataType`` header field value.
    :param dtype: Little-endian NumPy dtype used to store the raw values.
    :param invalid_value: Sentinel raw value marking a non-measured or
        spurious point.
    :param versions: SDF versions for which this data type is valid.
    """

    type: DataType
    dtype: np.dtype
    invalid_value: float
    versions: frozenset[SdfVersion]

    def is_supported(self, version: SdfVersion) -> bool:
        """Return whether this data type is valid for the given SDF version."""
        return version in self.versions


#: All supported SDF data types, keyed by their ``DataType`` code.
DATA_TYPES: Final[dict[DataType, SdfDataType]] = {
    DataType.BINARY32: SdfDataType(
        DataType.BINARY32, np.dtype("<f4"), -3.402823466e38, frozenset({SdfVersion.V2_0})
    ),
    DataType.INT8: SdfDataType(
        DataType.INT8, np.dtype("<i1"), -128.0, frozenset({SdfVersion.V2_0})
    ),
    DataType.INT16: SdfDataType(
        DataType.INT16, np.dtype("<i2"), -32768.0, frozenset({SdfVersion.V1_0, SdfVersion.V2_0})
    ),
    DataType.INT32: SdfDataType(
        DataType.INT32,
        np.dtype("<i4"),
        -2147483648.0,
        frozenset({SdfVersion.V1_0, SdfVersion.V2_0}),
    ),
    DataType.BINARY64: SdfDataType(
        DataType.BINARY64,
        np.dtype("<f8"),
        -1.7976931348623158e308,
        frozenset({SdfVersion.V1_0, SdfVersion.V2_0}),
    ),
}


def get_data_type(identifier: int) -> SdfDataType:
    """Look up a supported SDF data type by its numeric ``DataType`` code.

    :param identifier: ``DataType`` header field value (see :class:`DataType`).
    :raises SdfFormatError: If no matching data type is known.
    """
    try:
        return DATA_TYPES[DataType(identifier)]
    except ValueError:
        raise SdfFormatError(f"Unknown SDF data type code {identifier!r}") from None


def require_supported_data_type(identifier: int, version: SdfVersion) -> SdfDataType:
    """Look up a data type by code and ensure it is valid for ``version``.

    :param identifier: ``DataType`` header field value (see :class:`DataType`).
    :param version: SDF format version the data type must be valid for.
    :raises SdfFormatError: If no matching data type is known, or it is not
        valid for ``version``.
    """
    data_type = get_data_type(identifier)
    if not data_type.is_supported(version):
        raise SdfFormatError(
            f"Data type {data_type.type.name} is not valid for SDF version {version}"
        )
    return data_type


def validate_data_range(raw: np.ndarray, invalid_mask: np.ndarray, data_type: SdfDataType) -> None:
    """Reject values that would silently overflow or corrupt on encoding.

    :param raw: Coded height values (``data / z_scale``), same shape as
        ``invalid_mask``; entries where ``invalid_mask`` is set are ignored.
    :param invalid_mask: ``True`` where the corresponding point is
        non-measured/spurious (``NaN`` in the physical data).
    :param data_type: Target SDF data area storage type.
    :raises SdfFormatError: If a valid value is out of range for
        ``data_type``, or exactly equals its invalid-point sentinel, which
        would make it unrepresentable or misread as
        invalid on the next read.
    """
    valid = raw[~invalid_mask]
    if valid.size == 0:
        return
    if np.issubdtype(data_type.dtype, np.integer):
        # Round before range-checking: a value like 32767.4 (int16, max
        # 32767) is in range once rounded and must not be rejected for
        # exceeding the limit before rounding is even applied.
        rounded = np.rint(valid)
        info = np.iinfo(data_type.dtype)
        if rounded.min() < info.min or rounded.max() > info.max:
            raise SdfFormatError(
                f"Data value out of range for SDF data type {data_type.type.name} "
                f"({info.min}..{info.max})"
            )
    elif valid.min() < np.finfo(data_type.dtype).min or valid.max() > np.finfo(data_type.dtype).max:
        raise SdfFormatError(f"Data value out of range for SDF data type {data_type.type.name}")

    if np.any(valid == data_type.invalid_value):
        raise SdfFormatError(
            f"A data value equals the {data_type.type.name} invalid-point sentinel "
            f"({data_type.invalid_value!r}); rescale the data to avoid this value"
        )


def encode_raw(data: np.ndarray, data_type: SdfDataType, z_scale: float) -> np.ndarray:
    """Convert physical height values (metres) to the raw on-disk coded values.

    :param data: Height values in metres; ``NaN`` marks non-measured or
        spurious points.
    :param data_type: Target SDF data area storage type.
    :param z_scale: Scale factor; ``raw = data / z_scale``.
    :returns: Array of dtype ``data_type.dtype``, with non-measured or
        spurious points set to the type's sentinel.
    :raises SdfFormatError: See :func:`validate_data_range`.
    """
    invalid_mask = np.isnan(data)
    # 0.0 is just a safe placeholder for invalid points -- it's never used
    # (overwritten by the sentinel below); it only exists so NaN doesn't
    # reach np.rint()/astype(), whose behaviour on NaN is undefined for
    # integer dtypes.
    scaled = np.where(invalid_mask, 0.0, data / z_scale)
    validate_data_range(scaled, invalid_mask, data_type)
    if np.issubdtype(data_type.dtype, np.integer):
        raw = np.rint(scaled).astype(data_type.dtype)
    else:
        raw = scaled.astype(data_type.dtype)
    raw[invalid_mask] = data_type.dtype.type(data_type.invalid_value)
    return raw


def decode_raw(raw: np.ndarray, data_type: SdfDataType, z_scale: float) -> np.ndarray:
    """Convert raw on-disk coded values back to physical height values (metres).

    :param raw: Array of dtype ``data_type.dtype``.
    :param data_type: SDF data area storage type of ``raw``.
    :param z_scale: Scale factor; ``data = raw * z_scale``.
    :returns: ``float64`` array in metres, with the sentinel
        replaced by ``NaN``.
    """
    invalid_mask = raw == data_type.invalid_value
    scaled = raw.astype(np.float64) * z_scale
    scaled[invalid_mask] = np.nan
    return scaled


def suggest_z_scale(data: np.ndarray, data_type: SdfDataType) -> float:
    """Suggest the smallest ``z_scale`` that encodes ``data`` without overflow.

    For integer data types, ``z_scale`` directly trades off range against
    resolution: this picks the smallest value (i.e. highest resolution)
    such that ``data / z_scale`` stays within the type's representable
    range and avoids its invalid-point sentinel. Floating point data types
    already cover an enormous dynamic range at fixed relative precision, so
    ``1.0`` (no rescaling) is suggested for those.

    :param data: Height values in metres; ``NaN`` marks non-measured or
        spurious points.
    :param data_type: Target SDF data area storage type.
    :returns: A positive ``z_scale`` value, or ``1.0`` if ``data`` has no
        finite values or is not an integer data type.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0 or not np.issubdtype(data_type.dtype, np.integer):
        return 1.0

    info = np.iinfo(data_type.dtype)
    high, low = float(finite.max()), float(finite.min())
    candidates = [high / info.max] if high > 0 else []
    if low < 0:
        # info.min itself is reserved as the invalid-point sentinel (the
        # standard sets it to each integer type's minimum), so info.min + 1
        # is the smallest value actually usable for real data.
        candidates.append(low / (info.min + 1))
    z_scale = max(candidates) if candidates else 1.0
    return z_scale if z_scale > 0 else 1.0
