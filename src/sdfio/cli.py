# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Command line interface for :mod:`sdfio`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .datatypes import DataType
from .exceptions import SdfError, SdfFormatError
from .file import FileFormat, read
from .header import (
    SdfDialect,
    format_tagged_fields,
    parse_tagged_fields,
    sanitize_tagged_fields,
    validate_trailer_ascii,
    validate_trailer_tagged,
)

__all__ = ["main"]


def _parse_dialect(value: str) -> SdfDialect:
    try:
        return SdfDialect(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid dialect {value!r}, expected one of {', '.join(str(d) for d in SdfDialect)}"
        ) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdfio",
        description="Inspect and convert ISO 25178-71 SDF surface data files.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="print SDF header information")
    info_parser.add_argument("path")

    convert_parser = subparsers.add_parser(
        "convert", help="convert SDF format, dialect and/or data type"
    )
    convert_parser.add_argument("source")
    convert_parser.add_argument("destination")
    convert_parser.add_argument(
        "-f",
        "--format",
        choices=[file_format.name.lower() for file_format in FileFormat],
        help="target file format",
    )
    convert_parser.add_argument(
        "-d",
        "--dialect",
        type=_parse_dialect,
        metavar="{" + ",".join(str(d) for d in SdfDialect) + "}",
        help="convert to a different SDF dialect and version, e.g. ISO-2.0 or BCR-1.0",
    )
    convert_parser.add_argument(
        "-t",
        "--type",
        dest="data_type",
        choices=[data_type.name.lower() for data_type in DataType],
        help="convert the data area storage type",
    )
    convert_parser.add_argument(
        "-x",
        "--fix-trailer",
        action="store_true",
        help="if the trailer isn't valid for --dialect, fix it instead of failing",
    )

    return parser


def _ascii_safe(text: str) -> str:
    # "?" mirrors str.encode(..., errors="replace")'s own substitute for an
    # unencodable character, the encode-side counterpart to the "�" a
    # lenient decode (errors="replace") already used to read this text.
    return "".join(char if char.isascii() else "?" for char in text)


def _fix_trailer(text: str, dialect: SdfDialect) -> str:
    # ISO-2.0 requires the tagged "Name = Value" shape, so salvage by shape
    # first, then make any surviving field 7-bit ASCII. Other dialects have
    # no shape requirement at all -- imposing one would destroy legitimate
    # freeform content, so only ASCII-ness is fixed, line by line.
    if dialect.requires_tagged_trailer:
        fields = {
            _ascii_safe(name): _ascii_safe(value)
            for name, value in parse_tagged_fields(sanitize_tagged_fields(text)).items()
        }
        return format_tagged_fields(fields)
    return "".join(f"{_ascii_safe(line)}\r\n" for line in text.splitlines())


def _print_info(path: str) -> None:
    sdf = read(path)
    header = sdf.header
    fields = (
        ("Format", "binary" if header.binary else "ASCII"),
        ("Dialect", header.dialect),
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", header.create_date if header.create_date is not None else "(not recorded)"),
        ("ModDate", header.mod_date if header.mod_date is not None else "(not recorded)"),
        ("NumPoints", header.num_points),
        ("NumProfiles", header.num_profiles),
        ("Xscale", f"{header.x_scale:g}"),
        ("Yscale", f"{header.y_scale:g}"),
        ("Zscale", f"{header.z_scale:g}"),
        ("Zresolution", f"{header.z_resolution:g}"),
        ("DataType", sdf.data_type.name.lower()),
    )
    width = max(len(name) for name, _ in fields)
    for name, value in fields:
        print(f"{name:<{width}} = {value}")
    if sdf.trailer:
        trailer = sdf.trailer
        # The trailer isn't validated on read (see validate_trailer_ascii),
        # so a non-compliant file's trailer might not decode cleanly here.
        text = trailer if isinstance(trailer, str) else trailer.decode("ascii", errors="replace")
        print(f"{'Trailer':<{width}} =")
        print(text)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    :param argv: Arguments to parse; defaults to :data:`sys.argv`.
    :returns: The process exit status.
    """
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "info":
            _print_info(args.path)
            return 0
        if args.command == "convert":
            sdf = read(args.source)
            target_dialect = args.dialect if args.dialect is not None else sdf.header.dialect
            data_type = DataType[args.data_type.upper()] if args.data_type is not None else None
            if target_dialect != sdf.header.dialect or data_type is not None:
                trailer = None
                if args.fix_trailer:
                    try:
                        validate_trailer_ascii(sdf.trailer)
                        validate_trailer_tagged(target_dialect, sdf.trailer)
                    except SdfFormatError:
                        source_trailer = sdf.trailer
                        text = (
                            source_trailer
                            if isinstance(source_trailer, str)
                            else source_trailer.decode("ascii", errors="replace")
                        )
                        trailer = _fix_trailer(text, target_dialect)
                sdf = sdf.with_dialect(target_dialect, data_type=data_type, trailer=trailer)
            file_format = FileFormat[args.format.upper()] if args.format is not None else None
            sdf.save(args.destination, format=file_format)
            return 0
    # Deliberately narrow: only known, expected failure modes (malformed SDF
    # content, filesystem errors) get the friendly one-line message. Anything
    # else is a real bug and should surface as an actual traceback.
    except (SdfError, OSError) as error:
        print(f"sdfio: error: {error}", file=sys.stderr)
        return 1
    return 1
