# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Bytes, paths and streams: choosing the codec of a file and moving its bytes."""

from __future__ import annotations

import codecs
import os
from pathlib import Path
from typing import IO

import numpy as np

from . import _ascii, _binary
from .exceptions import SdfFormatError
from .header import Header
from .revision import ASCII_PREFIX, BINARY_PREFIX, MAGIC_SIZE, FileFormat

__all__ = [
    "PathLike",
    "PathOrStream",
    "as_text",
    "decode",
    "encode",
    "read_bytes",
    "write_bytes",
]

PathLike = str | os.PathLike[str]
PathOrStream = PathLike | IO[bytes]


def decode(data: bytes) -> tuple[Header, np.ndarray, str | bytes]:
    """Parse the contents of a ``.sdf`` file, detecting the encoding from its magic.

    A UTF-8 byte order mark in front of the magic, which some tools write, is ignored.

    :returns: The header, the data and the trailer; the trailer is text for an ASCII file.
    :raises SdfFormatError: If ``data`` does not start with a recognized SDF magic, or its
        contents are otherwise malformed.
    """
    data = data.removeprefix(codecs.BOM_UTF8)
    ascii_prefix = ASCII_PREFIX.encode("ascii")
    binary_prefix = BINARY_PREFIX.encode("ascii")
    prefix = data[:1].lower()
    if len(data) < MAGIC_SIZE or prefix not in (ascii_prefix, binary_prefix):
        raise SdfFormatError("Not an SDF file, unexpected magic")
    if prefix == binary_prefix:
        return _binary.loads(data)
    return _ascii.loads(data.decode("ascii", errors="replace"))


def encode(header: Header, data: np.ndarray, trailer: str | bytes) -> bytes:
    """Serialize a file in the encoding of ``header.format``.

    :raises SdfFormatError: If the file can't be written, see :func:`sdfio.write`.
    """
    if header.format is FileFormat.BINARY:
        return _binary.dumps(header, data, _as_bytes(trailer))
    return _ascii.dumps(header, data, as_text(trailer)).encode("ascii")


def read_bytes(source: PathOrStream) -> bytes:
    """Return the bytes of a path or of an open binary stream."""
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    return source.read()


def write_bytes(destination: PathOrStream, data: bytes) -> None:
    """Write ``data`` to a path or an open binary stream."""
    if isinstance(destination, (str, os.PathLike)):
        Path(destination).write_bytes(data)
    else:
        destination.write(data)


def _as_bytes(trailer: str | bytes) -> bytes:
    # A lossless str<->bytes bridge only -- validate_trailer_ascii() enforces the SDF trailer's
    # ASCII-only requirement when writing; this itself stays lenient by design.
    return trailer if isinstance(trailer, bytes) else trailer.encode("utf-8")


def as_text(trailer: str | bytes) -> str:
    """Return a trailer as text, whichever way it is held."""
    return trailer if isinstance(trailer, str) else trailer.decode("utf-8", errors="replace")
