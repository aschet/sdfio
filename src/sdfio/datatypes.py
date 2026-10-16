# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""SDF data area storage types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from .exceptions import SdfFormatError
from .revision import Revision

__all__ = [
    "DATA_TYPES",
    "DataType",
    "DataTypeInfo",
    "decode_raw",
    "encode_raw",
    "get_data_type",
    "require_supported_data_type",
    "suggest_z_scale",
    "validate_data_range",
]


class DataType(IntEnum):
    """SDF data area storage types, by their ``DataType`` header field code.

    Codes 0-2 are unsigned integer types from the pre-standard BCR-1.0
    proposal; ISO dropped them (see :mod:`.datatypes`'s ``DATA_TYPES`` for
    which revisions accept which code).
    """

    UINT8 = 0
    UINT16 = 1
    UINT32 = 2
    BINARY32 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    BINARY64 = 7


@dataclass(frozen=True)
class DataTypeInfo:
    """One storage type of the data area.

    :param type: The ``DataType`` code of the header.
    :param dtype: The little-endian NumPy dtype of the raw values.
    :param revisions: The revisions that have this type, see :meth:`is_supported`.
    """

    type: DataType
    dtype: np.dtype
    revisions: frozenset[Revision]

    def is_supported(self, revision: Revision) -> bool:
        """Return whether ``revision`` has this data type."""
        return revision in self.revisions

    def invalid_value(self, revision: Revision) -> float:
        """Return the raw value that marks a point that was not measured, in ``revision``."""
        return revision.rules.sentinel.invalid_value(self.dtype)

    def ascii_invalid_value(self, revision: Revision) -> float:
        """Return the number that an ASCII file without ``BAD`` writes for such a point."""
        return revision.rules.sentinel.ascii_invalid_value(self.dtype)

    def is_invalid(self, raw: np.ndarray, revision: Revision) -> np.ndarray:
        """Return the mask of the raw values that mark a point that was not measured."""
        return revision.rules.sentinel.is_invalid(raw, self.dtype)


#: All supported SDF data types, keyed by their ``DataType`` code.
DATA_TYPES: Final[dict[DataType, DataTypeInfo]] = {
    DataType.UINT8: DataTypeInfo(DataType.UINT8, np.dtype("<u1"), frozenset({Revision.BCR_1_0})),
    DataType.UINT16: DataTypeInfo(DataType.UINT16, np.dtype("<u2"), frozenset({Revision.BCR_1_0})),
    DataType.UINT32: DataTypeInfo(DataType.UINT32, np.dtype("<u4"), frozenset({Revision.BCR_1_0})),
    DataType.BINARY32: DataTypeInfo(
        DataType.BINARY32,
        np.dtype("<f4"),
        frozenset({Revision.ISO_2_0, Revision.BCR_1_0}),
    ),
    DataType.INT8: DataTypeInfo(
        DataType.INT8, np.dtype("<i1"), frozenset({Revision.ISO_2_0, Revision.BCR_1_0})
    ),
    DataType.INT16: DataTypeInfo(
        DataType.INT16,
        np.dtype("<i2"),
        frozenset({Revision.ISO_1_0, Revision.ISO_2_0, Revision.BCR_1_0}),
    ),
    DataType.INT32: DataTypeInfo(
        DataType.INT32,
        np.dtype("<i4"),
        frozenset({Revision.ISO_1_0, Revision.ISO_2_0, Revision.BCR_1_0}),
    ),
    DataType.BINARY64: DataTypeInfo(
        DataType.BINARY64,
        np.dtype("<f8"),
        frozenset({Revision.ISO_1_0, Revision.ISO_2_0, Revision.BCR_1_0}),
    ),
}


def get_data_type(identifier: int) -> DataTypeInfo:
    """Look up a supported SDF data type by its numeric ``DataType`` code.

    :param identifier: ``DataType`` header field value (see :class:`DataType`).
    :raises SdfFormatError: If no matching data type is known.
    """
    try:
        return DATA_TYPES[DataType(identifier)]
    except ValueError:
        raise SdfFormatError(f"Unknown SDF data type code {identifier!r}") from None


def require_supported_data_type(identifier: int, revision: Revision) -> DataTypeInfo:
    """Look up a data type by code and ensure it is valid for ``revision``.

    :param identifier: ``DataType`` header field value (see :class:`DataType`).
    :param revision: SDF revision the data type must be valid for.
    :raises SdfFormatError: If no matching data type is known, or it is not
        valid for ``revision``.
    """
    data_type = get_data_type(identifier)
    if not data_type.is_supported(revision):
        raise SdfFormatError(
            f"Data type {data_type.type.name} is not valid for SDF revision {revision}"
        )
    return data_type


def validate_data_range(
    raw: np.ndarray,
    invalid_mask: np.ndarray,
    data_type: DataTypeInfo,
    revision: Revision = Revision.ISO_2_0,
) -> None:
    """Reject values that would silently overflow or corrupt on encoding.

    :param raw: Coded height values (``data / z_scale``), same shape as
        ``invalid_mask``; entries where ``invalid_mask`` is set are ignored.
    :param invalid_mask: ``True`` where the corresponding point is
        non-measured/spurious (``NaN`` in the physical data).
    :param data_type: Target SDF data area storage type.
    :param revision: SDF revision, see :meth:`DataTypeInfo.invalid_value`.
    :raises SdfFormatError: If a valid value is out of range for
        ``data_type``, or exactly equals its invalid-point sentinel, which
        would make it unrepresentable or misread as
        invalid on the next read.
    """
    valid = raw[~invalid_mask]
    if valid.size == 0:
        return
    if np.issubdtype(data_type.dtype, np.integer):
        # A value like 32767.4 is in range for int16 once rounded.
        valid = np.rint(valid)
        info = np.iinfo(data_type.dtype)
        if valid.min() < info.min or valid.max() > info.max:
            raise SdfFormatError(
                f"Data value out of range for SDF data type {data_type.type.name} "
                f"({info.min}..{info.max})"
            )
    elif valid.min() < np.finfo(data_type.dtype).min or valid.max() > np.finfo(data_type.dtype).max:
        raise SdfFormatError(f"Data value out of range for SDF data type {data_type.type.name}")

    # A float is compared as stored: a value that rounds to the marker is one.
    candidates = valid.astype(data_type.dtype) if data_type.dtype.kind == "f" else valid
    if np.any(data_type.is_invalid(candidates, revision)):
        raise SdfFormatError(
            f"A data value collides with the {data_type.type.name} invalid-point sentinel "
            f"({data_type.invalid_value(revision)!r}); rescale the data to avoid this value"
        )


def encode_raw(
    data: np.ndarray,
    data_type: DataTypeInfo,
    z_scale: float,
    revision: Revision = Revision.ISO_2_0,
) -> np.ndarray:
    """Convert physical height values (metres) to the raw on-disk coded values.

    :param data: Height values in metres; ``NaN`` marks non-measured or
        spurious points.
    :param data_type: Target SDF data area storage type.
    :param z_scale: Scale factor; ``raw = data / z_scale``.
    :param revision: SDF revision, see :meth:`DataTypeInfo.invalid_value`.
    :returns: Array of dtype ``data_type.dtype``, with non-measured or
        spurious points set to the type's sentinel.
    :raises SdfFormatError: See :func:`validate_data_range`.
    """
    invalid_mask = np.isnan(data)
    # NaN must not reach rint() or astype(), which are undefined for it with integer types; the
    # placeholder is overwritten with the marker below.
    scaled = np.where(invalid_mask, 0.0, data / z_scale)
    validate_data_range(scaled, invalid_mask, data_type, revision)
    if np.issubdtype(data_type.dtype, np.integer):
        raw = np.rint(scaled).astype(data_type.dtype)
    else:
        raw = scaled.astype(data_type.dtype)
    raw[invalid_mask] = data_type.dtype.type(data_type.invalid_value(revision))
    return raw


def decode_raw(
    raw: np.ndarray,
    data_type: DataTypeInfo,
    z_scale: float,
    revision: Revision = Revision.ISO_2_0,
) -> np.ndarray:
    """Convert raw on-disk coded values back to physical height values (metres).

    :param raw: Array of dtype ``data_type.dtype``.
    :param data_type: SDF data area storage type of ``raw``.
    :param z_scale: Scale factor; ``data = raw * z_scale``.
    :param revision: SDF revision, see :meth:`DataTypeInfo.invalid_value`.
    :returns: ``float64`` array in metres, with ``NaN`` for the marker. A writer that wrote ``NaN``
        itself into ``binary32``/``binary64`` data, which some do, is read the same way.
    """
    invalid_mask = data_type.is_invalid(raw, revision)
    # NaN before scaling: a huge marker value could overflow when scaled.
    scaled = raw.astype(np.float64)
    scaled[invalid_mask] = np.nan
    # A file can hold any ``z_scale``, such as an infinite one, whose product is not a number:
    # that is the file's content, not something to warn about while reading it.
    with np.errstate(invalid="ignore", over="ignore"):
        return scaled * z_scale


def suggest_z_scale(
    data: np.ndarray, data_type: DataType, revision: Revision = Revision.ISO_2_0
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
    :param revision: SDF revision, see :meth:`DataTypeInfo.invalid_value`.
    :returns: A positive ``z_scale`` value, or ``1.0`` if ``data`` has no
        finite values or is not an integer data type.
    """
    dtype = get_data_type(data_type).dtype
    finite = data[np.isfinite(data)]
    if finite.size == 0 or not np.issubdtype(dtype, np.integer):
        return 1.0

    usable_min, usable_max = revision.rules.sentinel.usable_range(dtype)
    high, low = float(finite.max()), float(finite.min())
    candidates = [high / usable_max] if high > 0 else []
    # usable_min is 0 for an unsigned dtype, which can't represent a negative value at any
    # z_scale; skip the candidate rather than dividing by zero and let encode_raw's own range
    # check reject the actual value with a clear error if this is ever attempted.
    if low < 0 and usable_min != 0:
        candidates.append(low / usable_min)
    z_scale = max(candidates) if candidates else 1.0
    return z_scale if z_scale > 0 else 1.0
