# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""What a header, a trailer and a data type must be for a file to be written."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .datatypes import DataTypeInfo, require_supported_data_type
from .exceptions import SdfFormatError
from .trailer import validate_trailer_ascii, validate_trailer_tagged

if TYPE_CHECKING:
    from .header import Header

__all__ = [
    "validate_check_type",
    "validate_compression",
    "validate_for_write",
    "validate_manufacturer_id_ascii",
    "validate_z_scale",
]


def validate_manufacturer_id_ascii(manufacturer_id: str) -> None:
    """Reject a manufacturer ID that is not 7-bit ASCII, as the standard has it.

    :raises SdfFormatError: If ``manufacturer_id`` has a non-ASCII character.
    """
    if not manufacturer_id.isascii():
        raise SdfFormatError("manufacturer_id must be 7-bit ASCII")


def validate_z_scale(z_scale: float) -> None:
    """Reject a Z-scale that is not positive.

    The standard calls it "a non-zero positive number to scale the coded height information".

    :raises SdfFormatError: If ``z_scale`` is not positive.
    """
    if z_scale <= 0:
        raise SdfFormatError(f"z_scale must be a positive number, got {z_scale!r}")


def validate_compression(compression: int) -> None:
    """Reject a compressed data area, which no revision is read or written with here.

    :raises SdfFormatError: If ``compression`` is not 0.
    """
    if compression != 0:
        raise SdfFormatError("Compressed SDF data areas are not supported")


def validate_check_type(check_type: int) -> None:
    """Reject a checksummed data area, which no revision is read or written with here.

    The ``IntegerTrace`` checksum of BCR is a value after each profile in the data area, not a
    value of the header.

    :raises SdfFormatError: If ``check_type`` is not 0.
    """
    if check_type != 0:
        raise SdfFormatError("Checksummed SDF data areas are not supported")


def validate_for_write(header: Header, trailer: str | bytes) -> DataTypeInfo:
    """Reject what the file of ``header`` and ``trailer`` could not hold, for either encoding.

    :returns: The data type of the header, which is supported by its revision.
    :raises SdfFormatError: If the Z-scale is not positive, the manufacturer ID or the trailer is
        not 7-bit ASCII, the trailer is not tagged where the revision requires it, or the data
        type is unknown or not supported by the revision.
    """
    validate_z_scale(header.z_scale)
    validate_manufacturer_id_ascii(header.manufacturer_id)
    validate_trailer_ascii(trailer)
    validate_trailer_tagged(header.revision, trailer)
    return require_supported_data_type(header.data_type, header.revision)
