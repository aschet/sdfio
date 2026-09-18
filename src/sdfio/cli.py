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
from .header import SUPPORTED_VERSIONS, SdfVersion, validate_trailer_xml

__all__ = ["main"]


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
        "convert", help="convert SDF format, version and/or data type"
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
        "-n",
        "--number",
        type=SdfVersion,
        choices=SUPPORTED_VERSIONS,
        help="convert to a different SDF version number",
    )
    convert_parser.add_argument(
        "-t",
        "--type",
        dest="data_type",
        choices=[data_type.name.lower() for data_type in DataType],
        help="convert the data area storage type",
    )
    convert_parser.add_argument(
        "-d",
        "--drop-trailer",
        action="store_true",
        help="drop the trailer if it isn't valid for --number",
    )

    return parser


def _print_info(path: str) -> None:
    sdf = read(path)
    header = sdf.header
    fields = (
        ("Format", "binary" if header.binary else "ASCII"),
        ("Version", header.version),
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", header.create_date),
        ("ModDate", header.mod_date),
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
        text = trailer if isinstance(trailer, str) else trailer.decode("ascii")
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
            target_version = args.number or sdf.header.version
            data_type = DataType[args.data_type.upper()] if args.data_type is not None else None
            if target_version != sdf.header.version or data_type is not None:
                trailer = None
                if args.drop_trailer:
                    try:
                        validate_trailer_xml(target_version, sdf.trailer)
                    except SdfFormatError:
                        trailer = ""
                # assume_utc: the operator explicitly requested this conversion.
                sdf = sdf.with_version(
                    target_version, data_type=data_type, assume_utc=True, trailer=trailer
                )
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
