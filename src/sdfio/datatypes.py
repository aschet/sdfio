# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF data area storage types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .exceptions import SdfFormatError
from .header import DataType, SdfDialect

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
    :param dialects: SDF dialects for which this data type is valid, see
        :meth:`is_supported`.
    """

    type: DataType
    dtype: np.dtype
    dialects: frozenset[SdfDialect]

    def is_supported(self, dialect: SdfDialect) -> bool:
        """Return whether this data type is valid for the given SDF dialect."""
        return dialect in self.dialects

    def invalid_value(self, dialect: SdfDialect) -> float:
        """Sentinel raw value marking a non-measured/spurious point, for ``dialect``.

        ISO reserves each type's minimum representable value; BCR (the
        pre-standard proposal this format descends from) used the maximum
        instead, since it still had unsigned integer types whose minimum
        (0) is a valid low bound rather than an obviously-invalid one.
        """
        bounds = (
            np.iinfo(self.dtype) if np.issubdtype(self.dtype, np.integer) else np.finfo(self.dtype)
        )
        return float(bounds.max if dialect.uses_max_sentinel else bounds.min)


#: All supported SDF data types, keyed by their ``DataType`` code.
DATA_TYPES: Final[dict[DataType, SdfDataType]] = {
    DataType.BINARY32: SdfDataType(
        DataType.BINARY32,
        np.dtype("<f4"),
        frozenset({SdfDialect.ISO_2_0, SdfDialect.BCR_1_0}),
    ),
    DataType.INT8: SdfDataType(
        DataType.INT8, np.dtype("<i1"), frozenset({SdfDialect.ISO_2_0, SdfDialect.BCR_1_0})
    ),
    DataType.INT16: SdfDataType(
        DataType.INT16,
        np.dtype("<i2"),
        frozenset({SdfDialect.ISO_1_0, SdfDialect.ISO_2_0, SdfDialect.BCR_1_0}),
    ),
    DataType.INT32: SdfDataType(
        DataType.INT32,
        np.dtype("<i4"),
        frozenset({SdfDialect.ISO_1_0, SdfDialect.ISO_2_0, SdfDialect.BCR_1_0}),
    ),
    DataType.BINARY64: SdfDataType(
        DataType.BINARY64,
        np.dtype("<f8"),
        frozenset({SdfDialect.ISO_1_0, SdfDialect.ISO_2_0, SdfDialect.BCR_1_0}),
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


def require_supported_data_type(identifier: int, dialect: SdfDialect) -> SdfDataType:
    """Look up a data type by code and ensure it is valid for ``dialect``.

    :param identifier: ``DataType`` header field value (see :class:`DataType`).
    :param dialect: SDF dialect the data type must be valid for.
    :raises SdfFormatError: If no matching data type is known, or it is not
        valid for ``dialect``.
    """
    data_type = get_data_type(identifier)
    if not data_type.is_supported(dialect):
        raise SdfFormatError(
            f"Data type {data_type.type.name} is not valid for SDF dialect {dialect}"
        )
    return data_type


def validate_data_range(
    raw: np.ndarray,
    invalid_mask: np.ndarray,
    data_type: SdfDataType,
    dialect: SdfDialect = SdfDialect.ISO_2_0,
) -> None:
    """Reject values that would silently overflow or corrupt on encoding.

    :param raw: Coded height values (``data / z_scale``), same shape as
        ``invalid_mask``; entries where ``invalid_mask`` is set are ignored.
    :param invalid_mask: ``True`` where the corresponding point is
        non-measured/spurious (``NaN`` in the physical data).
    :param data_type: Target SDF data area storage type.
    :param dialect: SDF dialect, see :meth:`SdfDataType.invalid_value`.
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

    invalid_value = data_type.invalid_value(dialect)
    if np.any(valid == invalid_value):
        raise SdfFormatError(
            f"A data value equals the {data_type.type.name} invalid-point sentinel "
            f"({invalid_value!r}); rescale the data to avoid this value"
        )


def encode_raw(
    data: np.ndarray,
    data_type: SdfDataType,
    z_scale: float,
    dialect: SdfDialect = SdfDialect.ISO_2_0,
) -> np.ndarray:
    """Convert physical height values (metres) to the raw on-disk coded values.

    :param data: Height values in metres; ``NaN`` marks non-measured or
        spurious points.
    :param data_type: Target SDF data area storage type.
    :param z_scale: Scale factor; ``raw = data / z_scale``.
    :param dialect: SDF dialect, see :meth:`SdfDataType.invalid_value`.
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
    validate_data_range(scaled, invalid_mask, data_type, dialect)
    if np.issubdtype(data_type.dtype, np.integer):
        raw = np.rint(scaled).astype(data_type.dtype)
    else:
        raw = scaled.astype(data_type.dtype)
    raw[invalid_mask] = data_type.dtype.type(data_type.invalid_value(dialect))
    return raw


def decode_raw(
    raw: np.ndarray,
    data_type: SdfDataType,
    z_scale: float,
    dialect: SdfDialect = SdfDialect.ISO_2_0,
) -> np.ndarray:
    """Convert raw on-disk coded values back to physical height values (metres).

    :param raw: Array of dtype ``data_type.dtype``.
    :param data_type: SDF data area storage type of ``raw``.
    :param z_scale: Scale factor; ``data = raw * z_scale``.
    :param dialect: SDF dialect, see :meth:`SdfDataType.invalid_value`.
    :returns: ``float64`` array in metres, with the sentinel
        replaced by ``NaN``.
    """
    invalid_mask = raw == data_type.invalid_value(dialect)
    scaled = raw.astype(np.float64) * z_scale
    scaled[invalid_mask] = np.nan
    return scaled


def suggest_z_scale(
    data: np.ndarray, data_type: SdfDataType, dialect: SdfDialect = SdfDialect.ISO_2_0
) -> float:
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
    :param dialect: SDF dialect, see :meth:`SdfDataType.invalid_value`.
    :returns: A positive ``z_scale`` value, or ``1.0`` if ``data`` has no
        finite values or is not an integer data type.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0 or not np.issubdtype(data_type.dtype, np.integer):
        return 1.0

    info = np.iinfo(data_type.dtype)
    # Whichever bound `dialect` reserves as the invalid-point sentinel is
    # nudged one step towards zero, since that exact value isn't usable for
    # real data.
    usable_max = info.max - 1 if dialect.uses_max_sentinel else info.max
    usable_min = info.min if dialect.uses_max_sentinel else info.min + 1
    high, low = float(finite.max()), float(finite.min())
    candidates = [high / usable_max] if high > 0 else []
    if low < 0:
        candidates.append(low / usable_min)
    z_scale = max(candidates) if candidates else 1.0
    return z_scale if z_scale > 0 else 1.0
