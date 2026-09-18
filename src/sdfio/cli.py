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
from .header import SdfDialect, SdfVersion, validate_trailer_xml

__all__ = ["main"]

#: Every valid dialect/version combination, e.g. "ISO-2.0", "BCR-1.0" --
#: mirrors the file magic's own <dialect>-<version> shape (minus the a/b
#: prefix). BCR only ever had version 1.0, so it contributes just one.
_TARGETS = [
    f"{dialect}-{version}"
    for dialect in SdfDialect
    for version in SdfVersion
    if not (dialect == SdfDialect.BCR and version != SdfVersion.V1_0)
]


def _parse_target(value: str) -> tuple[SdfDialect, SdfVersion]:
    dialect_text, _, version_text = value.partition("-")
    try:
        return SdfDialect(dialect_text), SdfVersion(version_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid target {value!r}, expected one of {', '.join(_TARGETS)}"
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
        type=_parse_target,
        metavar="{" + ",".join(_TARGETS) + "}",
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
        ("Dialect", header.dialect),
        ("Version", header.version),
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
            target_dialect, target_version = args.number or (sdf.header.dialect, sdf.header.version)
            data_type = DataType[args.data_type.upper()] if args.data_type is not None else None
            if (target_dialect, target_version) != (sdf.header.dialect, sdf.header.version) or (
                data_type is not None
            ):
                trailer = None
                if args.drop_trailer:
                    try:
                        validate_trailer_xml(target_version, sdf.trailer)
                    except SdfFormatError:
                        trailer = ""
                # assume_utc: the operator explicitly requested this conversion.
                sdf = sdf.with_version(
                    target_version,
                    dialect=target_dialect,
                    data_type=data_type,
                    assume_utc=True,
                    trailer=trailer,
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
