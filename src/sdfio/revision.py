# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Revisions, representations and data type codes of the SDF format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, StrEnum, auto
from typing import Final

import numpy as np

from .exceptions import SdfFormatError

__all__ = [
    "ASCII_PREFIX",
    "BINARY_PREFIX",
    "MAGICS",
    "MAGIC_SIZE",
    "AsciiRules",
    "DataType",
    "FileFormat",
    "Revision",
    "RevisionRules",
    "Sentinel",
]


#: First character of the file magic: selects the ASCII representation.
ASCII_PREFIX = "a"
#: First character of the file magic: selects the binary representation.
BINARY_PREFIX = "b"


class FileFormat(Enum):
    """ASCII or binary SDF file representation."""

    ASCII = auto()
    BINARY = auto()


class Revision(StrEnum):
    """A revision of the SDF format, as written in the magic of a file, such as ``aISO-2.0``.

    ``BCR`` is the proposal from before the standard, which ISO succeeded; it has one version.
    The value of a member is the text of the magic.
    """

    ISO_1_0 = "ISO-1.0"
    ISO_2_0 = "ISO-2.0"
    BCR_1_0 = "BCR-1.0"

    @classmethod
    def resolve(cls, family_text: str, version_text: str, magic: str) -> Revision:
        """Return the revision of the family and version text of a magic.

        :param magic: The whole magic, for the error message.
        :raises SdfFormatError: If the two do not name a revision.
        """
        try:
            return cls(f"{family_text}-{version_text}")
        except ValueError:
            raise SdfFormatError(
                f"Unknown or unsupported SDF revision {family_text}-{version_text!r} "
                f"in magic {magic!r}"
            ) from None

    @property
    def rules(self) -> RevisionRules:
        """How this revision differs from the others, see :class:`RevisionRules`."""
        return _RULES[self]


class DataType(IntEnum):
    """Storage types of the data area, by their ``DataType`` code in the header.

    Codes 0 to 2 are the unsigned integer types of BCR-1.0, which ISO dropped. Which revision has
    which type is :attr:`RevisionRules.data_types`.
    """

    UINT8 = 0
    UINT16 = 1
    UINT32 = 2
    BINARY32 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    BINARY64 = 7


class Sentinel(ABC):
    """How a revision marks a point that was not measured, in the raw values of the data area."""

    @abstractmethod
    def invalid_value(self, dtype: np.dtype) -> float:
        """Return the raw value that marks a point that was not measured, for ``dtype``."""

    @abstractmethod
    def usable_range(self, dtype: np.dtype) -> tuple[int, int]:
        """Return the smallest and largest raw value of an integer ``dtype`` that is data."""

    def ascii_invalid_value(self, dtype: np.dtype) -> float:
        """Return the number an ASCII file without a ``BAD`` token writes for a bad point."""
        return self.invalid_value(dtype)

    def is_invalid(self, raw: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Return the mask of the raw values that mark a point that was not measured."""
        return np.asarray(raw == self.invalid_value(dtype))


def _limits(dtype: np.dtype) -> np.iinfo | np.finfo:
    return np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)


class MinimumSentinel(Sentinel):
    """ISO reserves the minimum of each data type."""

    def invalid_value(self, dtype: np.dtype) -> float:
        """Return the minimum of ``dtype``."""
        return float(_limits(dtype).min)

    def usable_range(self, dtype: np.dtype) -> tuple[int, int]:
        """Return the range of ``dtype`` without its minimum."""
        info = np.iinfo(dtype)
        return info.min + 1, info.max


# BCR-1.0 marks a bad floating point value with any value at or above the nominal
# maximum of its type, which is slightly below the largest representable value.
_BCR_FLOAT_BAD_THRESHOLD: Final[dict[int, float]] = {4: 3.4e38, 8: 1.7e308}


class MaximumSentinel(Sentinel):
    """BCR reserves the maximum, since it still had unsigned types whose minimum (0) is data."""

    def invalid_value(self, dtype: np.dtype) -> float:
        """Return the maximum of ``dtype``."""
        return float(_limits(dtype).max)

    def usable_range(self, dtype: np.dtype) -> tuple[int, int]:
        """Return the range of ``dtype`` without its maximum."""
        info = np.iinfo(dtype)
        return info.min, info.max - 1

    def ascii_invalid_value(self, dtype: np.dtype) -> float:
        """Return the nominal maximum for a floating point type.

        It lies within the range of the type and reads back as bad, unlike the largest
        representable value, which has no exact decimal form that stays in range.
        """
        if dtype.kind == "f":
            return _BCR_FLOAT_BAD_THRESHOLD[dtype.itemsize]
        return self.invalid_value(dtype)

    def is_invalid(self, raw: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Return the mask of raw values at or above the nominal maximum of a float type.

        That includes ``inf``: 3.4E+38 for ``binary32`` and 1.7E+308 for ``binary64``.
        """
        if dtype.kind == "f":
            return np.asarray(raw >= dtype.type(_BCR_FLOAT_BAD_THRESHOLD[dtype.itemsize]))
        return super().is_invalid(raw, dtype)


@dataclass(frozen=True)
class AsciiRules:
    """What the ASCII encoding of a revision allows, beyond ISO-1.0.

    :param comments: Whether an ASCII file may carry ``;`` comments in its header and data
        records: BCR says everything after a ``;`` on a line is ignored. ISO 25178-71 has no such
        rule. The trailer is free text, so a ``;`` there is never a comment.
    :param symbolic_header_values: Whether an ASCII header may spell
        ``DataType``/``Compression``/``CheckType`` as names, such as ``DataType = INTEGER``,
        instead of the numeric code.
    :param invalid_token: Whether a non-measured point is marked with the literal ``BAD``. BCR has
        no such token and writes the maximum of the type as a number, as binary files do; ``BAD``
        is still accepted when reading it.
    """

    comments: bool
    symbolic_header_values: bool
    invalid_token: bool


@dataclass(frozen=True)
class RevisionRules:
    """What a revision of the SDF format decides, one instance per :class:`Revision`.

    :param requires_utc: Whether the file stores ``create_date``/``mod_date`` as UTC. The
        standard mandates UTC from version 2.0; the other revisions store local time. In memory
        the dates are UTC for every revision, see :class:`Header`.
    :param max_binary_count: Largest number of points per profile, and of profiles, of the
        binary encoding: 65535 where the binary header holds both in 2 bytes, and 4294967295 for
        4 bytes (ISO 25178-71, Table 2, where the data types of these fields are listed for the
        binary format only). The standard sets no limit for the ASCII encoding.
    :param requires_tagged_trailer: Whether a non-empty trailer written to disk must use the
        tagged ``Name = Value`` format of the header. Enforced on write only: a file with a
        non-conforming trailer can still be read, see :func:`validate_trailer_tagged`.
    :param sentinel: How a point that was not measured is marked in the raw values: ISO reserves
        the minimum of a data type, and the 1993 BCR proposal the maximum, since it still had
        unsigned integer types whose minimum (0) is a valid low bound.
    :param data_types: The data types the revision has, by their codes.
    :param ascii: What the ASCII encoding allows, see :class:`AsciiRules`.
    :param reverses_profile_order: Whether the first profile stored on disk is at the maximum y,
        not at y=0. BCR never fixed a direction; ISO 25178-71 fixed row 1 at y=0. This reflects
        observed behaviour, not a requirement: MountainsMap renders BCR-1.0 with the first
        profile at the maximum y, and sdfio matches that on read and write, so
        :attr:`SdfFile.y_axis` increases with the row for every revision.

    >>> [data_type.name for data_type in Revision.ISO_1_0.rules.data_types]
    ['INT16', 'INT32', 'BINARY64']
    >>> Revision.ISO_1_0.rules.max_binary_count
    65535
    >>> Revision.ISO_2_0.rules.max_binary_count
    4294967295
    """

    requires_utc: bool
    max_binary_count: int
    requires_tagged_trailer: bool
    sentinel: Sentinel
    data_types: tuple[DataType, ...]
    ascii: AsciiRules
    reverses_profile_order: bool

    def order_rows(self, data: np.ndarray) -> np.ndarray:
        """Return ``data`` with the rows in the order of the file, or back in increasing y.

        The same operation converts in both directions: a reversal for a revision for which
        :attr:`reverses_profile_order` is ``True``, and ``data`` as it is otherwise.
        """
        return data[::-1] if self.reverses_profile_order else data


_ISO_1_0_RULES = RevisionRules(
    requires_utc=False,
    max_binary_count=2**16 - 1,
    requires_tagged_trailer=False,
    sentinel=MinimumSentinel(),
    data_types=(DataType.INT16, DataType.INT32, DataType.BINARY64),
    ascii=AsciiRules(comments=False, symbolic_header_values=False, invalid_token=True),
    reverses_profile_order=False,
)
_RULES = {
    Revision.ISO_1_0: _ISO_1_0_RULES,
    Revision.ISO_2_0: replace(
        _ISO_1_0_RULES,
        requires_utc=True,
        max_binary_count=2**32 - 1,
        requires_tagged_trailer=True,
        data_types=tuple(DataType)[3:],
    ),
    Revision.BCR_1_0: replace(
        _ISO_1_0_RULES,
        sentinel=MaximumSentinel(),
        ascii=AsciiRules(comments=True, symbolic_header_values=True, invalid_token=False),
        reverses_profile_order=True,
        data_types=tuple(DataType),
    ),
}


#: Total length in bytes/characters of the file magic (e.g. ``aISO-2.0``):
#: the a/b prefix plus a revision value, which are all the same length.
MAGIC_SIZE = len(ASCII_PREFIX) + len(Revision.ISO_2_0)

#: Every file magic sdfio can read, as bytes, e.g. for magic-sniffing file type detection.
#: BCR-1.0 also allows an upper case ``A``/``B`` prefix, accepted for every revision.
MAGICS = tuple(
    f"{prefix}{revision}".encode("ascii")
    for prefix in (ASCII_PREFIX, BINARY_PREFIX, ASCII_PREFIX.upper(), BINARY_PREFIX.upper())
    for revision in Revision
)
